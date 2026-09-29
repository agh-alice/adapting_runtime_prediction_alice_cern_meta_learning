# Auto-generated from model_trans_emb_research.ipynb
# Run with: python model_trans_emb_research.py


# %%

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.model_runner import AliceModelRunner
from alice_jobs_package.training.metrics import *



# %%

import argparse
import math
import torch
from torch import nn
import torch.nn.functional as F

from copy import deepcopy
from typing import cast, List, Optional
from collections import defaultdict, deque
from tqdm import tqdm

from alice_jobs_package.utils import logging
from alice_jobs_package.utils.project_config import *
from alice_jobs_package.models.base_alice_model import BaseAliceModel

logger = logging.get_logger(__name__)

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument(
    "--training_args_mode", type=str, default="FILE", choices=["FILE", "CMD_LINE"]
)
parser.add_argument("--training_args_path", type=str)

# ---------------------------
# Online adaptation hyperparams — edit here before each run
# ---------------------------
ONLINE_MAX_SUPPORT = 100
ONLINE_MIN_SUPPORT = 4
ONLINE_INNER_STEPS = 2
ONLINE_INNER_LR = 0.02

def pred_log_to_raw(pred_log: torch.Tensor) -> torch.Tensor:
    return torch.expm1(pred_log).clamp(min=0.0)


def denormalize_column_value(value: float, col_name: str, training_config: TrainingConfig) -> float:
    cfg = training_config.num_config[col_name]
    if "mean" in cfg and "std" in cfg:
        return value * cfg["std"] + cfg["mean"]
    if "min" in cfg and "max" in cfg:
        return value * (cfg["max"] - cfg["min"]) + cfg["min"]
    raise ValueError(f"Cannot denormalize {col_name}. Config: {cfg}")


def raw_jobtypeid_from_column(x: torch.Tensor, col: int, mean: float, std: float) -> torch.Tensor:
    """Recover the raw (un-normalized) lpmjobtypeid from its z-score-standardized
    column value, rounding to the nearest integer, instead of truncating the raw
    standardized float directly with .long(). lpmjobtypeid is a NUMERICAL column
    in this pipeline (not embedding-encoded); truncating the standardized value
    collapses many distinct real production IDs into the same integer bucket
    whenever std is large relative to typical ID-to-ID gaps (observed: ~163
    distinct IDs collapsed into ~2 buckets on the Aliprod validation range)."""
    return torch.round(x[:, col] * std + mean).long()


# ---------------------------
# Expert MLP with LayerNorm + Dropout (meta-friendly; no BatchNorm)
# ---------------------------
class MLPExpert(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_dropouts=(0.10, 0.10, 0.05, 0.05),
            max_target: float = 24.0,
    ):
        super().__init__()
        self.max_target = max_target

        self.fc1 = nn.Linear(input_size, 512)
        self.ln1 = nn.LayerNorm(512)
        self.do1 = nn.Dropout(hidden_dropouts[0])

        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.do2 = nn.Dropout(hidden_dropouts[1])

        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.do3 = nn.Dropout(hidden_dropouts[2])

        self.fc4 = nn.Linear(128, 64)
        self.ln4 = nn.LayerNorm(64)
        self.do4 = nn.Dropout(hidden_dropouts[3])

        self.out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.ln1(self.fc1(x)))
        x = self.do1(x)

        x = F.silu(self.ln2(self.fc2(x)))
        x = self.do2(x)

        x = F.silu(self.ln3(self.fc3(x)))
        x = self.do3(x)

        x = F.silu(self.ln4(self.fc4(x)))
        x = self.do4(x)

        return torch.clamp(F.softplus(self.out(x)), max=math.log1p(self.max_target))


# ---------------------------
# Meta-learner (Reptile-style) + online few-shot per lpmjobtypeid
# ---------------------------
class MLPEmbedded512_MAML(BaseAliceModel):
    """
    Reptile-style meta-learning + log-space Huber objective (+ small relative term).

    Additionally:
    - optional online few-shot adaptation in eval forward:
      * the model buffers (x, y) per lpmjobtypeid (job type),
      * once enough support samples are available, it creates a temporary
        adapted expert for that job type and uses it for predictions of that job type.
    """

    def __init__(self, training_config: TrainingConfig):
        super().__init__(training_config)
        self.training_config = training_config

        # --- Columns & configs ---
        self.num_col_numbers = training_config.num_col_numbers
        self.cat_col_numbers = training_config.cat_col_numbers
        self.lpmjobtypeid_column = training_config.column_names.index("lpmjobtypeid")

        self.num_config = training_config.num_config
        self._lpmjobtypeid_mean = float(self.num_config["lpmjobtypeid"]["mean"])
        self._lpmjobtypeid_std = float(self.num_config["lpmjobtypeid"]["std"])
        self.cat_config = training_config.cat_config
        self.numerical_dim = len(self.num_config)
        self.categories_dim = len(self.cat_config)

        # --- Hyperparams / defaults ---
        args = getattr(training_config, "args", {})
        self.embeding_reduction_const = getattr(args, "embeding_reduction_const", 10)
        self.inner_lr = float(getattr(args, "inner_lr", 0.02))
        self.inner_steps = int(getattr(args, "inner_steps", 4))
        self.meta_step_size = float(getattr(args, "meta_step_size", 0.005))
        self.weight_decay = float(getattr(args, "weight_decay", 1e-5))
        self.embedding_dropout_p = float(getattr(args, "embedding_dropout", 0.05))
        self.hidden_dropouts = tuple(
            getattr(args, "hidden_dropouts", (0.10, 0.10, 0.05, 0.05))
        )
        self.huber_delta_log = float(getattr(args, "huber_delta_log", 0.5))
        self.rel_lambda = float(getattr(args, "rel_lambda", 0.1))
        self.max_target = float(getattr(args, "max_target", 24.0))
        self.eps = 1e-6

        # --- Online adaptation hyperparams ---
        self.online_adaptation_enabled = bool(
            getattr(args, "online_adaptation_enabled", True)
        )

        self.online_inner_steps = ONLINE_INNER_STEPS
        self.online_min_support = ONLINE_MIN_SUPPORT
        self.online_max_support = ONLINE_MAX_SUPPORT
        self.online_inner_lr = ONLINE_INNER_LR

        tqdm.write(
            f"[MAML config] args type={type(args).__name__} | "
            f"online_inner_steps={self.online_inner_steps} | "
            f"online_min_support={self.online_min_support} | "
            f"online_max_support={self.online_max_support} | "
            f"online_inner_lr={self.online_inner_lr:.4f} | "
            f"inner_lr={self.inner_lr:.4f} | "
            f"inner_steps={self.inner_steps}"
        )

        # jobtypeid -> deque[(x_cpu, y_cpu)]
        self._online_buffers = defaultdict(
            lambda: deque(maxlen=self.online_max_support)
        )

        # --- Build modules ---
        self._build_common_layers()
        self.expert = MLPExpert(
            self.input_size,
            hidden_dropouts=self.hidden_dropouts,
            max_target=self.max_target,
        ).to(self.device)

        # Optional: small Adam step to apply weight decay / smooth drifts (no gradient set)
        self.optimizer = torch.optim.Adam(
            self.expert.parameters(), lr=1e-3, weight_decay=self.weight_decay
        )

    # ---------------------------
    # Embeddings & preprocessing
    # ---------------------------
    def _build_common_layers(self):
        self.embeddings = nn.ModuleDict()
        self._cat_num_embeddings = []
        for num, key in enumerate(self.cat_config.keys()):
            card = len(self.cat_config[key])
            emb_dim = max(min(10, card), (card // self.embeding_reduction_const) + 1)
            emb = nn.Embedding(num_embeddings=card + 1, embedding_dim=emb_dim)
            self.embeddings[str(num)] = emb
            self._cat_num_embeddings.append(card + 1)

        self.embed_dropout = nn.Dropout(self.embedding_dropout_p)

        self.embeded_cat_dim = sum(
            emb.embedding_dim for emb in self.embeddings.values()
        )
        self.input_size = self.embeded_cat_dim + self.numerical_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess_x(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.cat_col_numbers) > 0:
            cat_x = x[:, self.cat_col_numbers].long()
            cat_embeds: List[torch.Tensor] = []
            for i in range(cat_x.shape[1]):
                vocab = self._cat_num_embeddings[i]
                idx = torch.clamp(cat_x[:, i], 0, vocab - 1)
                emb = self.embeddings[str(i)](idx)
                cat_embeds.append(emb)
            cat_embed = torch.cat(cat_embeds, dim=1)
            cat_embed = self.embed_dropout(cat_embed)
        else:
            cat_embed = torch.empty((x.size(0), 0), device=x.device)

        num_x = x[:, self.num_col_numbers].float()
        return torch.cat([cat_embed, num_x], dim=1)

    # ---------------------------
    # Losses (log-space Huber + small relative term)
    # ---------------------------
    def _log_huber(self, y_pred_log: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_true_log = torch.log1p(torch.clamp(y_true, min=self.eps))

        return F.huber_loss(
            y_pred_log,
            y_true_log,
            delta=self.huber_delta_log,
            reduction="mean",
        )


    def _relative_mae(self, y_pred_log: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_raw = pred_log_to_raw(y_pred_log).clamp(min=self.eps)

        denom = torch.clamp(y_true.abs(), min=self.eps)

        return (y_pred_raw - y_true).abs().div(denom).mean()


    def _task_loss(self, y_pred_log: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return (
                self._log_huber(y_pred_log, y_true)
                + self.rel_lambda * self._relative_mae(y_pred_log, y_true)
        )

    # ---------------------------
    # Task building
    # ---------------------------
    def _split_tasks(self, x: torch.Tensor, y: torch.Tensor, support_ratio: float = 0.5) -> list:
        jobtype_ids = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std).unique()
        tasks = []

        for jobtypeid in jobtype_ids:
            mask = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std) == jobtypeid
            x_task = x[mask]
            y_task = y[mask]

            if x_task.numel() == 0:
                continue

            perm = torch.randperm(len(x_task))
            x_task = x_task[perm]
            y_task = y_task[perm]

            split = max(1, min(int(len(x_task) * support_ratio), len(x_task) - 1))
            support_x, query_x = x_task[:split], x_task[split:]
            support_y, query_y = y_task[:split], y_task[split:]

            sx = self._preprocess_x(support_x).detach().to(self.device)
            sy = support_y.detach().to(self.device)

            if len(query_x) > 0:
                qx = self._preprocess_x(query_x).detach().to(self.device)
                qy = query_y.detach().to(self.device)
            else:
                qx = torch.empty(0, self.input_size, device=self.device)
                qy = torch.empty(0, device=self.device)

            tasks.append((sx, sy, qx, qy))

        return tasks

    # ---------------------------
    # Inner-loop adaptation (SGD) with grad clipping
    # ---------------------------
    def _adapt_once(self, model: nn.Module, sx: torch.Tensor, sy: torch.Tensor, lr: Optional[float] = None) -> None:
        if lr is None:
            lr = self.inner_lr
        inner_optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0)
        y_pred = model(sx)
        loss = self._task_loss(y_pred, sy)
        inner_optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        inner_optim.step()

    def _task_adapt(self, sx: torch.Tensor, sy: torch.Tensor) -> nn.Module:
        adapted = deepcopy(self.expert)
        adapted.train()
        for _ in range(self.inner_steps):
            self._adapt_once(adapted, sx, sy)
        return adapted

    # ---------------------------
    # Meta step: pure Reptile
    # ---------------------------
    def maml_train_step(self, x: torch.Tensor, y: torch.Tensor, support_ratio: float = 0.5) -> dict:
        self.expert.train()

        tasks = self._split_tasks(x, y, support_ratio=support_ratio)
        if not tasks:
            return {
                "loss": 0.0,
                "huber": 0.0,
                "r2": 0.0,
            }

        with torch.no_grad():
            accum_delta = [torch.zeros_like(p) for p in self.expert.parameters()]

        query_losses = []
        query_preds = []
        query_targets = []

        for sx, sy, qx, qy in tasks:
            adapted = self._task_adapt(sx, sy)

            if len(qx) > 0:
                with torch.no_grad():
                    q_pred = adapted(qx)

                    q_loss = self._task_loss(q_pred, qy).item()
                    query_losses.append(q_loss)

                    query_preds.append(q_pred.detach().view(-1).cpu())
                    query_targets.append(qy.detach().view(-1).cpu())

            with torch.no_grad():
                for acc, p_adapted, p_init in zip(
                        accum_delta, adapted.parameters(), self.expert.parameters()
                ):
                    acc.add_(p_adapted.data - p_init.data)

        scale = self.meta_step_size / float(len(tasks))

        with torch.no_grad():
            for p, d in zip(self.expert.parameters(), accum_delta):
                p.add_(scale * d)

        for param in self.expert.parameters():
            param.grad = None

        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.step()

        if len(query_losses) == 0:
            return {
                "loss": 0.0,
                "huber": 0.0,
                "r2": 0.0,
            }

        # q_pred was produced in log-space, so convert to raw scale for metrics
        y_pred_log_all = torch.cat(query_preds)
        y_true_all = torch.cat(query_targets)

        y_pred_all = pred_log_to_raw(y_pred_log_all)

        # Raw-scale Huber metric
        huber = torch.nn.functional.huber_loss(
            y_pred_all,
            y_true_all,
            delta=1.0
        ).item()

        # Raw-scale R2
        r2 = r2_score(y_pred_all, y_true_all)

        return {
            "loss": sum(query_losses) / len(query_losses),  # training loss, probably log-space
            "huber": huber,                                # raw-space metric
            "r2": r2,                                      # raw-space metric
        }

    # ---------------------------
    # Online few-shot utils
    # ---------------------------
    def _push_online_support(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if self.training:
            return
        jobtype_ids = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std).cpu()
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        for i in range(x_cpu.size(0)):
            jid = int(jobtype_ids[i].item())
            self._online_buffers[jid].append((x_cpu[i], y_cpu[i]))

    _debug_adapt_logged: int = 0

    def _build_online_adapted_expert(self, jobtype_id: int, persist: bool = False) -> Optional[nn.Module]:
        buf = self._online_buffers.get(jobtype_id, None)
        if not buf or len(buf) < self.online_min_support:
            return None

        xs = torch.stack([t[0] for t in buf], dim=0).to(self.device)
        ys = torch.stack([t[1] for t in buf], dim=0).to(self.device)

        sx = self._preprocess_x(xs)
        sy = ys

        log_this = self._debug_adapt_logged < 3
        if log_this:
            self._debug_adapt_logged += 1
            step_losses = []

        # persist=True: adapt the shared expert in place (this experiment).
        # persist=False: adapt a throwaway copy, shared expert stays frozen (original ephemeral design).
        target = self.expert if persist else deepcopy(self.expert)
        with torch.enable_grad():
            target.train()
            for step in range(self.online_inner_steps):
                inner_optim = torch.optim.SGD(target.parameters(), lr=self.online_inner_lr, weight_decay=0.0)
                y_pred = target(sx)
                loss = self._task_loss(y_pred, sy)
                if log_this:
                    step_losses.append(loss.item())
                inner_optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(target.parameters(), max_norm=1.0)
                inner_optim.step()

        if log_this:
            losses_str = " | ".join(f"s{i}={v:.4f}" for i, v in enumerate(step_losses))
            tqdm.write(
                f"[adapt debug] jid={jobtype_id} buf={len(buf)} persist={persist} "
                f"steps={self.online_inner_steps} lr={self.online_inner_lr:.4f} | "
                f"{losses_str}"
            )

        target.eval()
        return target

    # ---------------------------
    # Inference + optional online few-shot adaptation
    # ---------------------------
    def forward(
            self,
            x: torch.Tensor,
            y_true: Optional[torch.Tensor] = None,
            online_adapt: bool = True,
            online_persist: bool = False,
    ) -> torch.Tensor:
        x = x.to(self.device)

        if self.training:
            self.expert.train()
            x_processed = self._preprocess_x(x)
            return self.expert(x_processed)

        if y_true is not None:
            y_true = y_true.to(self.device)
            self._push_online_support(x, y_true)

        if not (self.online_adaptation_enabled and online_adapt):
            self.expert.eval()
            x_processed = self._preprocess_x(x)
            with torch.no_grad():
                return self.expert(x_processed)

        self.expert.eval()
        jobtype_ids = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std)
        unique_ids = jobtype_ids.unique()

        preds = torch.empty(x.size(0), 1, device=self.device)

        for jid in unique_ids:
            jid_int = int(jid.item())
            mask = jobtype_ids == jid
            x_group = x[mask]

            adapted = self._build_online_adapted_expert(jid_int, persist=online_persist)
            model_for_group = adapted if adapted is not None else self.expert

            with torch.no_grad():
                x_proc_group = self._preprocess_x(x_group)
                preds_group = model_for_group(x_proc_group)

            preds[mask] = preds_group

        return preds

    # ---------------------------
    # Config passthrough
    # ---------------------------
    def _get_training_config(self) -> TrainingConfig:
        if hasattr(self, "training_config"):
            return self.training_config
        elif hasattr(self, "module") and hasattr(self.module, "training_config"):
            return self.module.training_config
        else:
            raise AttributeError(
                "Could not find `training_config` in self or self.module"
            )

    # ---------------------------
    # >>> ADDED: Export for deploy_app <<<
    # ---------------------------
    def save_for_deploy(self, output_path: str) -> None:
        """
        Export self-contained checkpoint for deploy_app.

        Packs model weights + all preprocessor metadata (cat_config, num_config,
        column_names, col_numbers) into a single .pt file.

        Usage on Athena after training:
            model.save_for_deploy("maml_embedded512_float32.pt")
        """
        tc = self._get_training_config()
        args = getattr(tc, "args", None)

        hyperparams = {
            "embeding_reduction_const": self.embeding_reduction_const,
            "inner_lr": self.inner_lr,
            "inner_steps": self.inner_steps,
            "meta_step_size": self.meta_step_size,
            "max_target": self.max_target,
            "hidden_dropouts": list(self.hidden_dropouts),
            "embedding_dropout": self.embedding_dropout_p,
            "huber_delta_log": self.huber_delta_log,
            "rel_lambda": self.rel_lambda,
            "online_adaptation_enabled": self.online_adaptation_enabled,
            "max_support": self.online_max_support,
            "min_support": self.online_min_support,
            "online_inner_steps": self.online_inner_steps,
            "online_inner_lr": self.online_inner_lr,
        }

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "cat_config": tc.cat_config,
            "num_config": tc.num_config,
            "column_names": tc.column_names,
            "cat_col_numbers": tc.cat_col_numbers,
            "num_col_numbers": tc.num_col_numbers,
            "hyperparams": hyperparams,
        }

        torch.save(checkpoint, output_path)
        # logger.info(f"Saved deploy checkpoint to {output_path}")
        # print(f"Saved deploy checkpoint to {output_path}")
        # print(f"  cat_config keys: {list(tc.cat_config.keys())}")
        # print(f"  num_config keys: {list(tc.num_config.keys())}")
        # print(f"  column_names ({len(tc.column_names)}): {tc.column_names[:5]}...")
        # print(f"  cat_col_numbers: {tc.cat_col_numbers}")
        # print(f"  num_col_numbers: {tc.num_col_numbers}")

    @classmethod
    def load_for_deploy(
            cls,
            checkpoint_path: str,
            training_config: TrainingConfig,
            map_location: Optional[str | torch.device] = None,
    ) -> "MLPEmbedded512_MAML":
        """
        Load model exported with save_for_deploy().

        Usage:
            training_config = TrainingConfig(...)
            model = MLPEmbedded512_MAML.load_for_deploy(
                "MAML_model.pt",
                training_config,
            )
            model.eval()
        """
        if map_location is None:
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        model = cls(training_config)

        state_dict = checkpoint["model_state_dict"]

        # In case checkpoint was saved from DDP with "module." prefixes
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {
                k.replace("module.", "", 1): v
                for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict, strict=True)
        model.to(model.device)
        model.eval()

        logger.info(f"Loaded deploy checkpoint from {checkpoint_path}")
        print(f"Loaded deploy checkpoint from {checkpoint_path}")

        return model




# %%

training_config = TrainingConfig(args_mode = ArgsMode.FILE, training_args_path = './training_config.json')
X_, y_ = AliceModelRunner.load_numpy_data(training_config)
train_dataloader, valid_dataloader = AliceModelRunner.prepare_dataloaders(training_config, X_, y_)



# %%

percent = 1

max_batches = max(1, len(train_dataloader) * percent)
val_max_batches = max(1, len(valid_dataloader) * percent)



# %%

model = MLPEmbedded512_MAML(training_config).to(training_config.device)



# %%

import torch
import torch.nn.functional as F

def regression_metrics(y_true, y_pred, eps=1e-8):
    y_true = torch.cat(y_true).detach().float().cpu().view(-1)
    y_pred = torch.cat(y_pred).detach().float().cpu().view(-1)

    _mae = mae(y_pred, y_true)
    _mse = mse(y_pred, y_true)
    _rmse = rmse(y_pred, y_true)
    _mape = mape(y_pred, y_true)
    _smape = smape(y_pred, y_true)
    _uep = uep(y_pred, y_true)
    _r2 = r2_score(y_pred, y_true)
    _huber = huber(y_pred, y_true)
    _loss = huber(y_pred, y_true, delta=1)

    return {
        "c_loss": _loss.item(),
        "huber": _huber.item(),
        "mae": _mae.item(),
        "mse": _mse.item(),
        "rmse": _rmse.item(),
        "mape": _mape.item(),
        "smape": _smape.item(),
        "uep": _uep.item(),
        "r2": _r2.item(),
    }



# %%

from tqdm import tqdm
import torch


best_valid_loss = float("inf")

history_preds = {
    "train": {},
    "valid": {},
}

epochs = training_config.args.epochs

for epoch in range(epochs):

    # =====================
    # TRAIN
    # =====================
    model.train()

    pbar = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch+1}/{epochs}",
        leave=True,
    )

    train_losses = []
    train_hubers = []
    train_r2s = []
    train_y_true = []
    train_y_pred = []

    for i, (x, y) in enumerate(pbar):
        if i >= max_batches:
            break

        x = x.to(model.device)
        y = y.to(model.device)

        metrics = model.maml_train_step(
            x=x,
            y=y,
            support_ratio=0.5,
        )

        train_losses.append(metrics["loss"])
        train_hubers.append(metrics["huber"])
        train_r2s.append(metrics["r2"])

        if epoch + 1 == epochs:
            with torch.no_grad():
                model.eval()
                pred = model(x, online_adapt=False)
                pred_raw = pred_log_to_raw(pred)
                model.train()

            train_y_true.append(y.detach().cpu().view(-1))
            train_y_pred.append(pred_raw.detach().cpu().view(-1))

        pbar.set_postfix(
            loss=f"{sum(train_losses)/len(train_losses):.4f}",
            huber=f"{sum(train_hubers)/len(train_hubers):.4f}",
            r2=f"{sum(train_r2s)/len(train_r2s):.4f}",
        )

    print(
        f"Epoch {epoch+1:03d} | "
        f"loss={sum(train_losses)/len(train_losses):.4f} | "
        f"huber={sum(train_hubers)/len(train_hubers):.4f} | "
        f"r2={sum(train_r2s)/len(train_r2s):.4f}"
    )

    if epoch + 1 == epochs:
        history_preds["train"][epochs] = {
            "y_true": torch.cat(train_y_true),
            "y_pred": torch.cat(train_y_pred),
        }

    train_loss = sum(train_losses) / max(len(train_losses), 1)

    # =====================
    # VALIDATION
    # =====================
    model.eval()

    valid_y_true = []
    valid_y_pred = []

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(valid_dataloader, desc="Validation", leave=False)):
            if i >= val_max_batches:
                break

            x = x.to(model.device)
            y = y.to(model.device)

            preds = model(
                x,
                online_adapt=False,
            )
            valid_y_true.append(y.detach())
            valid_y_pred.append(pred_log_to_raw(preds).detach())

    valid_metrics = regression_metrics(valid_y_true, valid_y_pred)

    print(
        f"Validation after epoch {epoch+1:03d} | "
        f"huber={valid_metrics['huber']:.6f} | "
        f"mae={valid_metrics['mae']:.4f} | "
        f"mse={valid_metrics['mse']:.4f} | "
        f"rmse={valid_metrics['rmse']:.4f} | "
        f"mape={valid_metrics['mape']:.2f}% | "
        f"smape={valid_metrics['smape']:.2f}% | "
        f"uep={valid_metrics['uep']:.2f}% | "
        f"r2={valid_metrics['r2']:.4f}"
    )

    # =====================
    # SAVE BEST
    # =====================
    if valid_metrics["huber"] < best_valid_loss:
        best_valid_loss = valid_metrics["huber"]

        model.save_for_deploy("best_maml_adapt_shared.pt")

        print(
            f"✓ New best model | "
            f"huber={valid_metrics['huber']:.6f} | "
            f"r2={valid_metrics['r2']:.4f}"
        )

# final export
model.save_for_deploy("MAML_model_adapt_shared.pt")



# %%

import heapq
import torch
from tqdm import tqdm

started_col = training_config.column_names.index("startedtimestamp")

model._online_buffers.clear()
model._debug_adapt_logged = 0
model.eval()

tqdm.write(f"[debug] lpmjobtypeid_column index={model.lpmjobtypeid_column}")
tqdm.write(f"[debug] column_names={training_config.column_names}")

samples = []

for i, (x_batch, y_batch) in enumerate(tqdm(valid_dataloader, desc="Collecting validation data")):
    if i >= val_max_batches:
        break

    x_batch = x_batch.detach().cpu()
    y_batch = y_batch.detach().cpu()

    for x, y in zip(x_batch, y_batch):
        started_time_raw = denormalize_column_value(
            float(x[started_col].item()), "startedtimestamp", training_config
        )
        samples.append({
            "x": x,
            "y": y,
            "started_time": started_time_raw,
        })

samples.sort(key=lambda s: s["started_time"])

tqdm.write(f"[time debug] started_time_raw samples (first 5): {[s['started_time'] for s in samples[:5]]}")
tqdm.write(f"[time debug] walltime_hours samples (first 5): {[float(s['y'].view(-1)[0].item()) for s in samples[:5]]}")

pending_finished_jobs = []
counter = 0

all_preds = []
all_targets = []

for sample in tqdm(samples, desc="Online deployment-style validation (adapt-shared)"):
    current_time = sample["started_time"]

    # A job only becomes available as training support once it has actually
    # FINISHED (its true walltime is known) -- not merely once it started.
    # Pop and learn from every previously-seen job whose finish time has
    # elapsed by now, in finish-time order, before predicting the new sample.
    # online_persist=True: SGD updates from the online few-shot adaptation are written
    # back into the shared self.expert (not a throwaway copy) — this is the "adapt
    # shared" variant, to be compared against the original ephemeral online-validation run.
    while pending_finished_jobs and pending_finished_jobs[0][0] <= current_time:
        _, _, finished_x, finished_y = heapq.heappop(pending_finished_jobs)

        with torch.no_grad():
            _ = model(
                finished_x.unsqueeze(0).to(model.device),
                y_true=finished_y.unsqueeze(0).to(model.device),
                online_adapt=True,
                online_persist=True,
            )

    x = sample["x"]
    y = sample["y"]

    with torch.no_grad():
        pred = model(
            x.unsqueeze(0).to(model.device),
            online_adapt=True,
            online_persist=True,
        )
    all_preds.append(pred_log_to_raw(pred).detach().cpu().view(-1))
    all_targets.append(y.detach().cpu().view(-1))

    # y is in hours, current_time in ms -> convert: 1h = 3_600_000 ms
    finish_time = current_time + float(y.view(-1)[0].item()) * 3_600_000.0

    heapq.heappush(
        pending_finished_jobs,
        (finish_time, counter, x, y),
    )
    counter += 1

buf_sizes = sorted([len(buf) for buf in model._online_buffers.values()], reverse=True)
adapted_count = sum(1 for s in buf_sizes if s >= model.online_min_support)
total_types = len(buf_sizes)
tqdm.write(f"[buffer stats] job types total={total_types} | adapted={adapted_count} ({100*adapted_count/max(total_types,1):.1f}%)")
if buf_sizes:
    tqdm.write(
        f"[buffer stats] buf sizes: max={buf_sizes[0]} | "
        f"p75={buf_sizes[int(len(buf_sizes)*0.25)]} | "
        f"median={buf_sizes[len(buf_sizes)//2]} | "
        f"min={buf_sizes[-1]}"
    )

online_valid_metrics = regression_metrics(
    all_targets,
    all_preds,
)

print(
    f"Online validation (adapt-shared) | "
    f"huber={online_valid_metrics['huber']:.6f} | "
    f"mae={online_valid_metrics['mae']:.4f} | "
    f"mse={online_valid_metrics['mse']:.4f} | "
    f"rmse={online_valid_metrics['rmse']:.4f} | "
    f"mape={online_valid_metrics['mape']:.2f}% | "
    f"smape={online_valid_metrics['smape']:.2f}% | "
    f"uep={online_valid_metrics['uep']:.2f}% | "
    f"r2={online_valid_metrics['r2']:.4f}"
)

history_preds["valid"][epochs] = {
    "y_true": torch.cat(all_targets),
    "y_pred": torch.cat(all_preds),
}



# %%

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import torch


def ensure_png_path(filepath):
    filepath = Path(filepath)
    return filepath if filepath.suffix == ".png" else filepath.with_suffix(".png")


def plot_true_vs_pred(
        y_true,
        y_pred,
        y_uncertainty=None,
        filepath=None,
        show=False,
        title="True vs Predicted",
):
    fig, ax = plt.subplots(figsize=(8, 8))

    y_true_ = torch.tensor(y_true).detach().cpu().float()
    y_pred_ = torch.tensor(y_pred).detach().cpu().float()

    y_true_ = torch.squeeze(y_true_)
    y_pred_ = torch.squeeze(y_pred_)

    if y_uncertainty is None:
        y_uncertainty_ = torch.zeros_like(y_pred_)
    else:
        y_uncertainty_ = torch.tensor(y_uncertainty).detach().cpu().float()
        y_uncertainty_ = torch.squeeze(y_uncertainty_)

    errors = torch.abs(y_pred_ - y_true_)

    thresholds = {
        "q50 ~ ": torch.quantile(errors, 0.5).item(),
        "q75 ~ ": torch.quantile(errors, 0.75).item(),
        "q95 ~ ": torch.quantile(errors, 0.95).item(),
        "q99 ~ ": torch.quantile(errors, 0.99).item(),
    }

    y_uncertainty_clipped = torch.clamp(torch.abs(y_uncertainty_), min=1e-6)

    sc = ax.scatter(
        y_true_,
        y_pred_,
        alpha=0.25,
        c=y_uncertainty_clipped,
        s=1,
        label="Predictions",
        cmap="viridis",
        norm=LogNorm(),
    )

    x_min, x_max = 0, 24
    ax.plot([x_min, x_max], [x_min, x_max], "r--", label="Ideal Fit")

    mean_std = y_uncertainty_clipped.mean().item()
    for delta, style in [(mean_std, "r--")]:
        x_start = max(x_min, 0 - delta)
        x_end = min(x_max, 24 - delta)
        ax.plot(
            [x_start, x_end],
            [x_start + delta, x_end + delta],
            style,
            linewidth=1,
            label=f"Mean Std ±{delta:.2f}",
        )

        x_start = max(x_min, 0 + delta)
        x_end = min(x_max, 24 + delta)
        ax.plot(
            [x_start, x_end],
            [x_start - delta, x_end - delta],
            style,
            linewidth=1,
        )

    colors = ["orange", "blue", "green", "purple"]
    for (label, delta), color in zip(thresholds.items(), colors):
        x_start = max(x_min, 0 - delta)
        x_end = min(x_max, 24 - delta)
        ax.plot(
            [x_start, x_end],
            [x_start + delta, x_end + delta],
            color=color,
            linewidth=1,
            label=label + f"{delta:.2f}h",
        )

        x_start = max(x_min, 0 + delta)
        x_end = min(x_max, 24 + delta)
        ax.plot(
            [x_start, x_end],
            [x_start - delta, x_end - delta],
            color=color,
            linewidth=1,
        )

    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Predictive Std")

    plt.tight_layout()

    if show:
        plt.show()
    else:
        filepath = ensure_png_path(filepath)
        plt.savefig(filepath, dpi=300, format="png", bbox_inches="tight")

    plt.close()



# %%

plot_true_vs_pred(
    history_preds["train"][epochs]["y_true"],
    history_preds["train"][epochs]["y_pred"],
    filepath="maml_train_last_epoch_adapt_shared.png",
    title=f"MAML Train - Last Epoch {epochs}",
)

plot_true_vs_pred(
    history_preds["valid"][epochs]["y_true"],
    history_preds["valid"][epochs]["y_pred"],
    filepath="maml_valid_online_adapt_shared_last_epoch.png",
    title=f"MAML Validation Online Adapt-Shared - Last Epoch {epochs}",
)
