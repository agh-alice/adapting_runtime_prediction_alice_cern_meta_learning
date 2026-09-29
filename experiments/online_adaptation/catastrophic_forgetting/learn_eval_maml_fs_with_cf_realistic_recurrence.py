# Auto-generated from model_trans_emb_research.ipynb
# Run with: python model_trans_emb_research.py
#
# REALISTIC-RECURRENCE CF ANALYSIS -- companion to learn_eval_maml_fs_with_cf.py.
#
# The original CF analysis measures backward transfer (BWT) as if EVERY job
# type seen in an earlier quarter Q_j will still need to be predicted after the
# model has continued adapting through Q_{j+1}..Q_T -- i.e. it retrospectively
# rescores ALL of Q_j's samples against the final checkpoint theta_T. This is a
# worst-case measurement: per ALICE Grid operations (C. Grigoras, personal
# communication), the vast majority of production campaigns complete within
# 2-3 weeks, well inside a single one of our quarters, and are never queried
# again by a later, drifted checkpoint. Forgetting is only operationally
# relevant for the job types that actually DO recur into a later quarter.
#
# This script reuses the identical training + quarter-checkpointing + causal
# replay machinery, but ADDITIONALLY:
#   1. Measures, directly from the data, what fraction of each quarter's job
#      types (and job VOLUME) actually recur in a strictly later quarter --
#      instead of assuming a recurrence rate.
#   2. Recomputes BWT restricted to only that recurring subset ("realistic"
#      BWT), alongside the original all-samples ("worst-case") BWT.
#   3. Combines the two into a volume-weighted "practical expected" BWT --
#      the realistic BWT scaled by how often a job actually belongs to a
#      recurring job type in the first place (since forgetting never
#      manifests for job types that never recur).


# %%

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.model_runner import AliceModelRunner
from alice_jobs_package.training.metrics import *



# %%

import argparse
import csv
import math
import torch
from torch import nn
import torch.nn.functional as F

from copy import deepcopy
from pathlib import Path
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

        # persist=False (default): ephemeral adaptation — SGD runs on a throwaway copy,
        #   self.expert is left untouched (current production design).
        # persist=True: continual adaptation — SGD runs directly on self.expert, so the
        #   update survives across job types/quarters. Used to reproduce catastrophic
        #   forgetting for the ablation, never used in the ephemeral/deployed path.
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
        tc = self._get_training_config()

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


# %%

training_config = TrainingConfig(args_mode=ArgsMode.FILE, training_args_path='./training_config.json')
X_, y_ = AliceModelRunner.load_numpy_data(training_config)
train_dataloader, valid_dataloader = AliceModelRunner.prepare_dataloaders(training_config, X_, y_)


# %%

percent = 1

max_batches = max(1, len(train_dataloader) * percent)
val_max_batches = max(1, len(valid_dataloader) * percent)


# %%

model = MLPEmbedded512_MAML(training_config).to(training_config.device)


# %%

def regression_metrics(y_true, y_pred, eps=1e-8):
    y_true = torch.cat(y_true).detach().float().cpu().view(-1)
    y_pred = torch.cat(y_pred).detach().float().cpu().view(-1)

    return {
        "c_loss": huber(y_pred, y_true, delta=1).item(),
        "huber": huber(y_pred, y_true).item(),
        "mae": mae(y_pred, y_true).item(),
        "mse": mse(y_pred, y_true).item(),
        "rmse": rmse(y_pred, y_true).item(),
        "mape": mape(y_pred, y_true).item(),
        "smape": smape(y_pred, y_true).item(),
        "uep": uep(y_pred, y_true).item(),
        "r2": r2_score(y_pred, y_true).item(),
    }


def regression_metrics_masked(y_true, y_pred, mask):
    """Same as regression_metrics, but restricted to samples where mask[i] is
    True. y_true/y_pred are lists of per-sample tensors (as returned by
    causal_replay), mask is a parallel list/sequence of booleans. Returns None
    if the mask selects zero samples (can happen for a quarter with very few
    recurring job types)."""
    selected = [i for i, m in enumerate(mask) if m]
    if not selected:
        return None
    y_true_sel = [y_true[i] for i in selected]
    y_pred_sel = [y_pred[i] for i in selected]
    return regression_metrics(y_true_sel, y_pred_sel)


# %%
# ------------------------------------------------------------
# Resumability: Athena's wall-time cap (48h) is not always enough for the full
# training + forward + retrospective pipeline below, especially for Aliprod's
# larger validation range. Every expensive stage is cached to disk right after
# it completes and skipped on a re-run if its cache file already exists, so a
# resubmission after a timeout resumes from the last completed stage instead
# of starting over from scratch.
# ------------------------------------------------------------

BASE_CKPT_PATH = "cf_base_meta_trained.pt"

if Path(BASE_CKPT_PATH).exists():
    tqdm.write(f"[resume] found cached meta-trained checkpoint at {BASE_CKPT_PATH}; skipping training")
    model.load_state_dict(torch.load(BASE_CKPT_PATH, map_location=model.device))
else:
    best_valid_loss = float("inf")

    epochs = training_config.args.epochs

    for epoch in range(epochs):
        model.train()

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        train_losses, train_hubers, train_r2s = [], [], []

        for i, (x, y) in enumerate(pbar):
            if i >= max_batches:
                break

            x = x.to(model.device)
            y = y.to(model.device)

            metrics = model.maml_train_step(x=x, y=y, support_ratio=0.5)

            train_losses.append(metrics["loss"])
            train_hubers.append(metrics["huber"])
            train_r2s.append(metrics["r2"])

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

        model.eval()
        valid_y_true, valid_y_pred = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(valid_dataloader, desc="Validation", leave=False)):
                if i >= val_max_batches:
                    break
                x = x.to(model.device)
                y = y.to(model.device)
                preds = model(x, online_adapt=False)
                valid_y_true.append(y.detach())
                valid_y_pred.append(pred_log_to_raw(preds).detach())

        valid_metrics = regression_metrics(valid_y_true, valid_y_pred)
        print(
            f"Validation after epoch {epoch+1:03d} | "
            f"huber={valid_metrics['huber']:.6f} | mae={valid_metrics['mae']:.4f} | "
            f"rmse={valid_metrics['rmse']:.4f} | smape={valid_metrics['smape']:.2f}% | "
            f"uep={valid_metrics['uep']:.2f}% | r2={valid_metrics['r2']:.4f}"
        )

        if valid_metrics["huber"] < best_valid_loss:
            best_valid_loss = valid_metrics["huber"]
            model.save_for_deploy("best_maml_cf_realistic.pt")
            print(f"✓ New best model | huber={valid_metrics['huber']:.6f} | r2={valid_metrics['r2']:.4f}")

    model.save_for_deploy("MAML_model_cf_realistic.pt")
    torch.save(model.state_dict(), BASE_CKPT_PATH)
    tqdm.write(f"[resume] cached meta-trained checkpoint to {BASE_CKPT_PATH}")


# %%
# ============================================================
# Catastrophic-forgetting analysis: sequential quarter adaptation,
# WORST-CASE (all samples) vs REALISTIC (recurring-job-types-only).
#
# IMPORTANT prerequisite: training_config.json must have `sub_valid_splits: null`
# (or absent) so that `valid_dataloader` covers the FULL validation range —
# quarters are carved out here explicitly, not via sub_valid_splits.
#
# Buffers are reset at the start of every quarter (in both the forward pass and
# the retrospective rescoring pass below), by design: this isolates drift in the
# shared self.expert weights from drift in per-jobtype buffer contents.
# ============================================================

import copy
import heapq

N_QUARTERS = 4
started_col = training_config.column_names.index("startedtimestamp")

tqdm.write(f"[debug] lpmjobtypeid_column index={model.lpmjobtypeid_column}")

# lpmjobtypeid is a NUMERICAL (z-score standardized) column in this pipeline,
# not an embedded categorical one -- see training_config.num_config. Truncating
# the standardized float directly to int (as the model's own internal grouping
# does, via `.long()`, for buffer-key purposes) can collapse many distinct raw
# production IDs into the same integer bucket whenever `std` is large relative
# to typical ID-to-ID gaps. For THIS script's recurrence measurement (a
# read-only diagnostic, separate from the model's internal buffering, which is
# left untouched), we instead denormalize back to the raw ID before rounding,
# using the existing denormalize_column_value() helper.
_lpmjobtypeid_stats = training_config.num_config.get("lpmjobtypeid")
tqdm.write(f"[debug] lpmjobtypeid numerical stats (for denormalization): {_lpmjobtypeid_stats}")


def raw_jobtypeid_from_tensor(x: torch.Tensor) -> int:
    denorm = denormalize_column_value(
        float(x[model.lpmjobtypeid_column].item()), "lpmjobtypeid", training_config
    )
    return int(round(denorm))


def collect_sorted_samples(dataloader, max_batches):
    samples = []
    truncated_ids_seen = set()   # old (buggy) approach: truncate the standardized float directly
    denormalized_ids_seen = set()  # new (fixed) approach: denormalize, then round
    for i, (x_batch, y_batch) in enumerate(tqdm(dataloader, desc="Collecting validation data")):
        if i >= max_batches:
            break
        x_batch = x_batch.detach().cpu()
        y_batch = y_batch.detach().cpu()
        for x, y in zip(x_batch, y_batch):
            started_time_raw = denormalize_column_value(
                float(x[started_col].item()), "startedtimestamp", training_config
            )
            jobtypeid = raw_jobtypeid_from_tensor(x)
            truncated_ids_seen.add(int(x[model.lpmjobtypeid_column].item()))
            denormalized_ids_seen.add(jobtypeid)
            samples.append({"x": x, "y": y, "started_time": started_time_raw, "jobtypeid": jobtypeid})
    samples.sort(key=lambda s: s["started_time"])
    tqdm.write(
        f"[debug] distinct lpmjobtypeid over the whole validation range: "
        f"truncated-standardized-float (old, likely wrong)={len(truncated_ids_seen)} | "
        f"denormalized-and-rounded (new, fixed)={len(denormalized_ids_seen)}"
    )
    return samples


def split_into_quarters(samples, n_quarters=N_QUARTERS):
    """Contiguous, equal-count chronological chunks — mirrors the sub_valid_splits scheme."""
    quarters = []
    left = len(samples)
    remaining = n_quarters
    start = 0
    for _ in range(n_quarters):
        size = left // remaining if remaining > 1 else left
        quarters.append(samples[start:start + size])
        start += size
        left -= size
        remaining -= 1
    return quarters


def causal_replay(model, quarter_samples, persist: bool):
    """Causal, chronological replay over quarter_samples using the existing
    per-jobtype online few-shot mechanism. Does not touch buffer state on its own —
    caller is responsible for clearing buffers before the call if a fresh pass is wanted.

    A job only becomes available as training support once it has actually
    FINISHED (its true walltime is known), not merely once it started -- the
    pending-finished-jobs heap is local to this call, so it starts empty on
    every fresh quarter-level replay, consistent with the buffer being reset
    per quarter by the caller."""
    pending_finished_jobs = []
    counter = 0
    preds, targets = [], []
    for sample in tqdm(quarter_samples, desc=f"replay (persist={persist})", leave=False):
        current_time = sample["started_time"]

        while pending_finished_jobs and pending_finished_jobs[0][0] <= current_time:
            _, _, finished_x, finished_y = heapq.heappop(pending_finished_jobs)
            with torch.no_grad():
                _ = model(
                    finished_x.unsqueeze(0).to(model.device),
                    y_true=finished_y.unsqueeze(0).to(model.device),
                    online_adapt=True,
                    online_persist=persist,
                )

        x, y = sample["x"], sample["y"]
        with torch.no_grad():
            pred = model(
                x.unsqueeze(0).to(model.device),
                online_adapt=True,
                online_persist=persist,
            )
        preds.append(pred_log_to_raw(pred).detach().cpu().view(-1))
        targets.append(y.detach().cpu().view(-1))

        finish_time = current_time + float(y.view(-1)[0].item()) * 3_600_000.0
        heapq.heappush(pending_finished_jobs, (finish_time, counter, x, y))
        counter += 1
    return preds, targets


# --- Collect + partition validation data once ---
all_samples = collect_sorted_samples(valid_dataloader, val_max_batches)
quarters = split_into_quarters(all_samples, N_QUARTERS)
for qi, q in enumerate(quarters, start=1):
    tqdm.write(
        f"[quarter split] Q{qi}: n={len(q)} "
        f"range=({q[0]['started_time']:.0f} .. {q[-1]['started_time']:.0f})"
    )

# --- Recurrence analysis: which job types in quarter j are ever seen again in
# a strictly later quarter, and what fraction of Q_j's job VOLUME (not just
# distinct job types) do they represent? Measured directly from the data. ---
job_types_per_quarter = [set(int(s["jobtypeid"]) for s in q) for q in quarters]

recurring_job_types = {}          # qj -> set of job types from Q_qj seen again later
recurrence_mask_per_quarter = {}  # qj -> list[bool], parallel to quarters[qj-1]
recurrence_rate_volume = {}       # qj -> fraction of Q_qj's samples that recur

for qj in range(1, N_QUARTERS):  # last quarter has no strictly-later quarter to recur into
    later_job_types = set()
    for qk in range(qj + 1, N_QUARTERS + 1):
        later_job_types |= job_types_per_quarter[qk - 1]

    recurring = job_types_per_quarter[qj - 1] & later_job_types
    recurring_job_types[qj] = recurring
    recurrence_mask_per_quarter[qj] = [s["jobtypeid"] in recurring for s in quarters[qj - 1]]

    n_types_total = len(job_types_per_quarter[qj - 1])
    n_types_recurring = len(recurring)
    n_samples_total = len(quarters[qj - 1])
    n_samples_recurring = sum(recurrence_mask_per_quarter[qj])
    recurrence_rate_volume[qj] = n_samples_recurring / max(n_samples_total, 1)

    tqdm.write(
        f"[recurrence] Q{qj}: {n_types_recurring}/{n_types_total} job types "
        f"({100.0 * n_types_recurring / max(n_types_total, 1):.1f}%) recur in a later quarter | "
        f"{n_samples_recurring}/{n_samples_total} samples "
        f"({100.0 * recurrence_rate_volume[qj]:.1f}% of job volume) belong to recurring job types"
    )

# --- Snapshot the freshly meta-trained model, before any quarter is seen ---
base_state_dict = copy.deepcopy(model.state_dict())

forgetting_results = {}

for mode_name, persist in [("ephemeral", False), ("continual", True)]:
    tqdm.write(f"\n===== Mode: {mode_name} (persist={persist}) =====")

    model.load_state_dict(base_state_dict)
    model._online_buffers.clear()
    model._debug_adapt_logged = 0
    model.eval()

    diagonal_metrics = {}            # Q_qj scored right after being adapted on (worst-case, all samples)
    diagonal_metrics_recurring = {}  # same, restricted to the recurring-job-type subset
    checkpoints = {}

    # --- forward pass: Q1 -> Q2 -> Q3 -> Q4, persisting model state as we go.
    # Cached as a whole (all 4 quarters + metrics) so a resubmission after a
    # timeout during the (much more expensive) retrospective pass below does
    # not have to redo training + all 4 forward passes. ---
    FORWARD_CKPT_PATH = f"cf_forward_{mode_name}.pt"

    if Path(FORWARD_CKPT_PATH).exists():
        tqdm.write(f"[resume] found cached forward pass at {FORWARD_CKPT_PATH}; skipping forward pass for {mode_name}")
        _fwd_cache = torch.load(FORWARD_CKPT_PATH, map_location=model.device)
        diagonal_metrics = _fwd_cache["diagonal_metrics"]
        diagonal_metrics_recurring = _fwd_cache["diagonal_metrics_recurring"]
        checkpoints = _fwd_cache["checkpoints"]
    else:
        for qi, quarter_samples in enumerate(quarters, start=1):
            model._online_buffers.clear()
            preds, targets = causal_replay(model, quarter_samples, persist=persist)
            diagonal_metrics[qi] = regression_metrics(targets, preds)
            if qi in recurrence_mask_per_quarter:
                diagonal_metrics_recurring[qi] = regression_metrics_masked(
                    targets, preds, recurrence_mask_per_quarter[qi]
                )
            checkpoints[qi] = copy.deepcopy(model.state_dict())

            tqdm.write(
                f"[{mode_name}] Q{qi} (forward, n={len(quarter_samples)}) | "
                f"huber={diagonal_metrics[qi]['huber']:.6f} | "
                f"mae={diagonal_metrics[qi]['mae']:.4f} | "
                f"rmse={diagonal_metrics[qi]['rmse']:.4f} | "
                f"smape={diagonal_metrics[qi]['smape']:.2f}%"
            )

        torch.save(
            {
                "diagonal_metrics": diagonal_metrics,
                "diagonal_metrics_recurring": diagonal_metrics_recurring,
                "checkpoints": checkpoints,
            },
            FORWARD_CKPT_PATH,
        )
        tqdm.write(f"[resume] cached forward pass to {FORWARD_CKPT_PATH}")

    # --- retrospective pass: re-score selected (checkpoint, quarter) pairs from
    #     a FRESH buffer. persist=False always here -- retro scoring must never
    #     itself mutate the checkpoint being evaluated.
    #
    #     We compute exactly two slices of the full O(T^2) grid, not the whole
    #     thing (which would be 10 pairs for N_QUARTERS=4) -- this is what keeps
    #     the pipeline inside Athena's 48h wall-time cap:
    #       (a) the FINAL-checkpoint row (N_QUARTERS, qj) for qj < N_QUARTERS --
    #           needed for the backward-transfer table below.
    #       (b) the Q1 COLUMN (ci, 1) for ci = 1..N_QUARTERS-1 -- needed for the
    #           multi-checkpoint decay-curve figure (recurring-only variant of
    #           aliprod_catastrophic_forgetting_all_metrics.png), tracking how
    #           Q1's retrospective error evolves as the model continues
    #           adapting through Q2, Q3, Q4.
    #     (N_QUARTERS, 1) is shared by both and computed only once. Any other
    #     (ci, qj) pair from the original full grid (e.g. (2,2), (3,2), (3,3))
    #     is not needed by anything downstream and is skipped.
    #
    #     Cached incrementally (one entry per pair) so a timeout mid-retrospective
    #     resumes only the missing pairs instead of restarting this pass. ---
    RETRO_CKPT_PATH = f"cf_retro_{mode_name}.pt"

    retro_metrics = {}            # retro_metrics[(ci, qj)] -- worst-case, all samples in Q_qj
    retro_metrics_recurring = {}  # same, restricted to the recurring-job-type subset of Q_qj

    if Path(RETRO_CKPT_PATH).exists():
        _retro_cache = torch.load(RETRO_CKPT_PATH, map_location=model.device)
        retro_metrics = _retro_cache["retro_metrics"]
        retro_metrics_recurring = _retro_cache["retro_metrics_recurring"]
        tqdm.write(
            f"[resume] found cached retrospective results at {RETRO_CKPT_PATH}; "
            f"{len(retro_metrics)} pair(s) already done for {mode_name}"
        )

    retro_pairs_needed = sorted(set(
        [(N_QUARTERS, qj) for qj in range(1, N_QUARTERS)]
        + [(ci, 1) for ci in range(1, N_QUARTERS)]
    ))
    ci_values_needed = sorted(set(ci for ci, _ in retro_pairs_needed))

    for ci in ci_values_needed:
        pairs_for_this_ci = [qj for (pair_ci, qj) in retro_pairs_needed if pair_ci == ci]
        if all((ci, qj) in retro_metrics for qj in pairs_for_this_ci):
            tqdm.write(f"[resume] skipping checkpoint ci={ci} for {mode_name}, all its pairs already cached")
            continue

        model.load_state_dict(checkpoints[ci])
        for qj in pairs_for_this_ci:
            if (ci, qj) in retro_metrics:
                tqdm.write(f"[resume] skipping retrospective (ci={ci}, Q{qj}) for {mode_name}, already cached")
                continue

            model._online_buffers.clear()
            model._debug_adapt_logged = 999
            model.eval()
            preds, targets = causal_replay(model, quarters[qj - 1], persist=False)
            retro_metrics[(ci, qj)] = regression_metrics(targets, preds)
            if qj in recurrence_mask_per_quarter:
                retro_metrics_recurring[(ci, qj)] = regression_metrics_masked(
                    targets, preds, recurrence_mask_per_quarter[qj]
                )

            torch.save(
                {"retro_metrics": retro_metrics, "retro_metrics_recurring": retro_metrics_recurring},
                RETRO_CKPT_PATH,
            )
            tqdm.write(f"[resume] cached retrospective (ci={ci}, Q{qj}) for {mode_name} to {RETRO_CKPT_PATH}")

    # --- backward transfer: worst-case (all samples), realistic (recurring
    #     job types only), and practical-expected (realistic scaled by how
    #     often a job actually belongs to a recurring job type in the first
    #     place, since forgetting never manifests for job types that never
    #     recur). Computed for every standard metric reported elsewhere in
    #     this work (Loss/Huber, MAE, RMSE, SMAPE). ---
    BWT_METRICS = [("huber", "Loss", "{:.4f}"), ("mae", "MAE", "{:.4f}"),
                   ("rmse", "RMSE", "{:.4f}"), ("smape", "SMAPE", "{:.2f}%")]

    bwt_worst_case = {metric_key: {} for metric_key, _, _ in BWT_METRICS}
    bwt_realistic = {metric_key: {} for metric_key, _, _ in BWT_METRICS}
    bwt_practical_expected = {metric_key: {} for metric_key, _, _ in BWT_METRICS}

    for metric_key, _, _ in BWT_METRICS:
        for qj in range(1, N_QUARTERS):
            now = diagonal_metrics[qj][metric_key]
            later = retro_metrics[(N_QUARTERS, qj)][metric_key]
            bwt_worst_case[metric_key][qj] = later - now

            now_r = diagonal_metrics_recurring.get(qj)
            later_r = retro_metrics_recurring.get((N_QUARTERS, qj))
            if now_r is None or later_r is None:
                bwt_realistic[metric_key][qj] = None
                bwt_practical_expected[metric_key][qj] = None
            else:
                delta_r = later_r[metric_key] - now_r[metric_key]
                bwt_realistic[metric_key][qj] = delta_r
                bwt_practical_expected[metric_key][qj] = recurrence_rate_volume[qj] * delta_r

    tqdm.write(
        f"\n[{mode_name}] Backward transfer -- worst-case (all samples) vs. realistic "
        f"(recurring job types only) vs. practical-expected (realistic x recurrence rate):"
    )
    for metric_key, metric_label, fmt in BWT_METRICS:
        tqdm.write(f"  -- {metric_label} --")
        for qj in range(1, N_QUARTERS):
            wc = bwt_worst_case[metric_key][qj]
            r = bwt_realistic[metric_key][qj]
            pe = bwt_practical_expected[metric_key][qj]
            r_str = f"{r:+.4f}" if r is not None else "n/a"
            pe_str = f"{pe:+.4f}" if pe is not None else "n/a"
            tqdm.write(
                f"    Q{qj}: worst-case={wc:+.4f} | realistic(recurring-only)={r_str} | "
                f"practical-expected={pe_str} (recurrence_rate={100*recurrence_rate_volume[qj]:.1f}%)"
            )

    avg_bwt_worst_case = {
        metric_key: sum(bwt_worst_case[metric_key].values()) / len(bwt_worst_case[metric_key])
        for metric_key, _, _ in BWT_METRICS
    }
    avg_bwt_realistic = {
        metric_key: (
            sum(v for v in bwt_realistic[metric_key].values() if v is not None)
            / max(sum(1 for v in bwt_realistic[metric_key].values() if v is not None), 1)
        )
        for metric_key, _, _ in BWT_METRICS
    }
    avg_bwt_practical_expected = {
        metric_key: (
            sum(v for v in bwt_practical_expected[metric_key].values() if v is not None)
            / max(sum(1 for v in bwt_practical_expected[metric_key].values() if v is not None), 1)
        )
        for metric_key, _, _ in BWT_METRICS
    }

    for metric_key, metric_label, _ in BWT_METRICS:
        tqdm.write(
            f"[{mode_name}] Average BWT ({metric_label}): "
            f"worst-case={avg_bwt_worst_case[metric_key]:+.4f} | "
            f"realistic={avg_bwt_realistic[metric_key]:+.4f} | "
            f"practical-expected={avg_bwt_practical_expected[metric_key]:+.4f}"
        )

    forgetting_results[mode_name] = {
        "diagonal": diagonal_metrics,
        "diagonal_recurring": diagonal_metrics_recurring,
        "retro": retro_metrics,
        "retro_recurring": retro_metrics_recurring,
        "bwt_worst_case": bwt_worst_case,
        "bwt_realistic": bwt_realistic,
        "bwt_practical_expected": bwt_practical_expected,
        "avg_bwt_worst_case": avg_bwt_worst_case,
        "avg_bwt_realistic": avg_bwt_realistic,
        "avg_bwt_practical_expected": avg_bwt_practical_expected,
    }

# --- Final ephemeral-vs-continual comparison, across all four tracked metrics ---
tqdm.write("\n===== Ephemeral vs Continual: forgetting summary (worst-case / realistic / practical-expected) =====")
for metric_key, metric_label, _ in BWT_METRICS:
    tqdm.write(f"-- {metric_label} --")
    eph = forgetting_results["ephemeral"]
    con = forgetting_results["continual"]
    tqdm.write(
        f"  ephemeral avg: worst-case={eph['avg_bwt_worst_case'][metric_key]:+.4f} | "
        f"realistic={eph['avg_bwt_realistic'][metric_key]:+.4f} | "
        f"practical-expected={eph['avg_bwt_practical_expected'][metric_key]:+.4f}"
    )
    tqdm.write(
        f"  continual avg: worst-case={con['avg_bwt_worst_case'][metric_key]:+.4f} | "
        f"realistic={con['avg_bwt_realistic'][metric_key]:+.4f} | "
        f"practical-expected={con['avg_bwt_practical_expected'][metric_key]:+.4f}"
    )


# %%
# ------------------------------------------------------------
# Q1 decay-curve figure, RECURRING-ONLY variant of
# aliprod_catastrophic_forgetting_all_metrics.png: retrospective error on Q1's
# recurring-job-type subset, tracked across all four metrics, as the model
# continues to adapt through Q2, Q3, Q4 -- ephemeral vs. continual.
#
# Uses retro_metrics_recurring[(ci, 1)] for ci=1..N_QUARTERS (the Q1 column
# computed above), restricted to the recurring-job-type subset of Q1 instead
# of all of Q1's samples, per the realistic-recurrence framing of this script.
# ------------------------------------------------------------

import matplotlib.pyplot as plt


def plot_forgetting_decay_recurring(forgetting_results, n_quarters, filepath):
    checkpoints_x = list(range(1, n_quarters + 1))  # ci = 1..N_QUARTERS

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, (metric_key, metric_label, _) in zip(axes, BWT_METRICS):
        for mode_name, style in [("ephemeral", "o--"), ("continual", "s-")]:
            retro_recurring = forgetting_results[mode_name]["retro_recurring"]
            ys = [retro_recurring[(ci, 1)][metric_key] for ci in checkpoints_x]
            ax.plot(checkpoints_x, ys, style, label=mode_name, linewidth=2, markersize=6)

        ax.set_xticks(checkpoints_x)
        ax.set_xticklabels([f"Q{ci}" for ci in checkpoints_x])
        ax.set_xlabel("Checkpoint (after adapting through)")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Q1 retrospective {metric_label} (recurring job types only)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()

    fig.suptitle("Retrospective error on Q1's recurring-job-type subset, Aliprod (ephemeral vs. continual)")
    plt.tight_layout()
    plt.savefig(ensure_png_path(filepath), dpi=300, format="png", bbox_inches="tight")
    plt.close()


def ensure_png_path(filepath):
    from pathlib import Path as _Path
    filepath = _Path(filepath)
    return filepath if filepath.suffix == ".png" else filepath.with_suffix(".png")


plot_forgetting_decay_recurring(
    forgetting_results,
    N_QUARTERS,
    filepath="catastrophic_forgetting_q1_decay_recurring.png",
)
tqdm.write("\n[plot] wrote catastrophic_forgetting_q1_decay_recurring.png")


# %%
# ------------------------------------------------------------
# CSV export for downstream reporting (one row per mode x quarter x metric).
# ------------------------------------------------------------

csv_path = "cf_realistic_recurrence_results.csv"
fieldnames = [
    "mode", "metric", "quarter", "recurrence_rate_volume_pct",
    "bwt_worst_case", "bwt_realistic", "bwt_practical_expected",
]

rows = []
for mode_name in ["ephemeral", "continual"]:
    res = forgetting_results[mode_name]
    for metric_key, _, _ in BWT_METRICS:
        for qj in range(1, N_QUARTERS):
            rows.append({
                "mode": mode_name,
                "metric": metric_key,
                "quarter": qj,
                "recurrence_rate_volume_pct": 100.0 * recurrence_rate_volume[qj],
                "bwt_worst_case": res["bwt_worst_case"][metric_key][qj],
                "bwt_realistic": res["bwt_realistic"][metric_key][qj],
                "bwt_practical_expected": res["bwt_practical_expected"][metric_key][qj],
            })

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

tqdm.write(f"\nFull results written to {csv_path}")
