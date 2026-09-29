# Auto-generated companion to ablation_06b_learn_eval_maml_min_support_crossover.py
# Run with: python ablation_06c_learn_eval_maml_min_support_crossover_bootstrap.py
#
# MIN-SUPPORT CROSSOVER ANALYSIS, WITH BOOTSTRAP UNCERTAINTY -- follow-up to
# ablation_06b. The simulation mechanics below (model, meta-training, the
# single causal online-validation pass with the REAL adapt-shared background
# dynamics) are IDENTICAL to ablation_06b; only the analysis performed on the
# collected errors has changed. See ablation_06b's header for the full
# rationale behind the simulation itself.
#
# Why this follow-up exists: ablation_06b's original conclusion picked the
# first buffer occupancy k at which the one-off adapted prediction was more
# accurate than the shared global expert AND stayed more accurate for every
# larger k observed ("k=34"), and recommended raising online_min_support from
# 4 to that point. That criterion is wrong on two counts:
#
#   1. It only asks where adaptation is NEVER harmful in this one sample; it
#      ignores how often each occupancy actually occurs, so it does not
#      measure how much benefit would be forgone by raising the gate.
#      Weighting each k by how many jobs were actually observed at it and
#      summing n_k * (mae_global_k - mae_adapted_k) shows that the range
#      k=4..33 -- which would lose adaptation entirely if the gate moved to
#      34 -- is net POSITIVE in aggregate (helpful occupancies outweigh
#      harmful ones there). Raising the gate to 34 would therefore forfeit
#      more benefit than it avoids in harm, on this run's numbers.
#   2. That aggregate is itself computed from the same low-occupancy bins
#      already known to be sparse and noisy (k=4..33 covers only ~1-2% of
#      all validation samples in a typical run, with individual bins in the
#      tens to low thousands), so the point estimate alone cannot be trusted
#      without knowing its uncertainty.
#
# This script fixes both issues without changing the underlying simulation:
#   (a) it retains every per-sample (k, abs_err_global, abs_err_adapted)
#       triple instead of only per-k sums, and
#   (b) at the end, it bootstraps the weighted-benefit sum (in sample-hours)
#       for the operationally meaningful k-ranges -- below the current gate
#       (k<4), the range that a gate=34 change would sacrifice (4<=k<=33),
#       the plateau (k>=34), and the total -- reporting a 95% CI and the
#       empirical fraction of resamples with benefit <= 0, instead of a
#       single "first-always-positive-k" number.
#
# What this script does NOT fix: bootstrapping only re-estimates the
# uncertainty already present in the samples this run happened to collect.
# Low-occupancy events are rare because each Production ID only passes
# through k=0..33 once, near its own launch, before settling at k=40 for the
# rest of its life -- so their count scales with the number of distinct
# Production ID startups captured in the validation window, not with the
# total number of jobs. If the resulting CI is still too wide to act on,
# the fix is to widen the validation window (more historical time -> more
# distinct production startups), not to re-run this same window again. That
# is a change to the underlying data window in `training_config.json` /
# the training-data pipeline, outside what this script controls, and is not
# made here.


# %%

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.model_runner import AliceModelRunner
from alice_jobs_package.training.metrics import *


# %%

import argparse
import csv
import heapq
import math
import numpy as np
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
# Online adaptation hyperparams -- edit here before each run.
# These are the REAL deployed values; the crossover probe overrides
# min_support down to 1 only for its own non-destructive measurement calls
# (see `min_support_override` below), never for the background dynamics.
# ---------------------------
ONLINE_MAX_SUPPORT = 40
ONLINE_MIN_SUPPORT = 4
ONLINE_INNER_STEPS = 2
ONLINE_INNER_LR = 0.02

# How large a buffer occupancy we bother tracking per-k accuracy for. Prior
# results (Table~tab:buffer_sensitivity_combined, optimum at n=20-40) suggest
# the crossover point is well under 40, but we track up to ONLINE_MAX_SUPPORT
# so the whole deployed range is covered.
MIN_SUPPORT_CROSSOVER_MAX_K = ONLINE_MAX_SUPPORT

# k-ranges we report a weighted, bootstrapped benefit estimate for. Edit if
# a different production decision (e.g. a different candidate min_support)
# needs to be evaluated.
REPORT_RANGES = [
    ("below current gate (k<4)", 0, 3),
    ("would be lost if gate->34 (4<=k<=33)", 4, 33),
    ("plateau (k>=34)", 34, MIN_SUPPORT_CROSSOVER_MAX_K),
    ("all occupancies (k=0..%d)" % MIN_SUPPORT_CROSSOVER_MAX_K, 0, MIN_SUPPORT_CROSSOVER_MAX_K),
]

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42


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
    whenever std is large relative to typical ID-to-ID gaps."""
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
# Meta-learner (Reptile-style) + persisted "adapt-shared" online few-shot,
# same mechanism as ablation_06_learn_eval_maml_adapt_shared.py, with one
# addition: `min_support_override` on `_build_online_adapted_expert` so the
# crossover probe below can measure k < the real online_min_support without
# changing what the background simulation itself uses.
# ---------------------------
class MLPEmbedded512_MAML(BaseAliceModel):
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

        # --- Online adaptation hyperparams (REAL deployed values) ---
        self.online_adaptation_enabled = bool(
            getattr(args, "online_adaptation_enabled", True)
        )
        self.online_inner_steps = ONLINE_INNER_STEPS
        self.online_min_support = ONLINE_MIN_SUPPORT
        self.online_max_support = ONLINE_MAX_SUPPORT
        self.online_inner_lr = ONLINE_INNER_LR

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
            y_pred_log, y_true_log, delta=self.huber_delta_log, reduction="mean",
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
    # Task building (meta-training only)
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

    def maml_train_step(self, x: torch.Tensor, y: torch.Tensor, support_ratio: float = 0.5) -> dict:
        self.expert.train()

        tasks = self._split_tasks(x, y, support_ratio=support_ratio)
        if not tasks:
            return {"loss": 0.0, "huber": 0.0, "r2": 0.0}

        with torch.no_grad():
            accum_delta = [torch.zeros_like(p) for p in self.expert.parameters()]

        query_losses, query_preds, query_targets = [], [], []

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
            return {"loss": 0.0, "huber": 0.0, "r2": 0.0}

        y_pred_log_all = torch.cat(query_preds)
        y_true_all = torch.cat(query_targets)
        y_pred_all = pred_log_to_raw(y_pred_log_all)

        huber = torch.nn.functional.huber_loss(y_pred_all, y_true_all, delta=1.0).item()
        r2 = r2_score(y_pred_all, y_true_all)

        return {
            "loss": sum(query_losses) / len(query_losses),
            "huber": huber,
            "r2": r2,
        }

    # ---------------------------
    # Online few-shot utils (same mechanics as ablation_06's adapt-shared
    # variant, plus `min_support_override` for non-destructive probing)
    # ---------------------------
    def _push_online_support(self, x: torch.Tensor, y: torch.Tensor) -> None:
        jobtype_ids = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std).cpu()
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        for i in range(x_cpu.size(0)):
            jid = int(jobtype_ids[i].item())
            self._online_buffers[jid].append((x_cpu[i], y_cpu[i]))

    def _build_online_adapted_expert(
            self,
            jobtype_id: int,
            persist: bool = False,
            min_support_override: Optional[int] = None,
    ) -> Optional[nn.Module]:
        """Identical to ablation_06's method (persist=True mutates the shared
        `self.expert` in place -- the deployed behavior; persist=False adapts
        a throwaway copy, used only by the non-destructive crossover probe),
        with one addition: `min_support_override` lets the probe require as
        few as 1 buffered sample without changing `self.online_min_support`,
        which continues to gate the REAL background adapt-shared dynamics."""
        buf = self._online_buffers.get(jobtype_id, None)
        min_support = self.online_min_support if min_support_override is None else min_support_override
        if not buf or len(buf) < min_support:
            return None

        xs = torch.stack([t[0] for t in buf], dim=0).to(self.device)
        ys = torch.stack([t[1] for t in buf], dim=0).to(self.device)

        sx = self._preprocess_x(xs).detach()
        sy = ys

        # persist=True: adapt the shared expert in place (deployed behavior).
        # persist=False: adapt a throwaway copy, shared expert stays frozen.
        target = self.expert if persist else deepcopy(self.expert)
        with torch.enable_grad():
            target.train()
            for _ in range(self.online_inner_steps):
                inner_optim = torch.optim.SGD(target.parameters(), lr=self.online_inner_lr, weight_decay=0.0)
                y_pred = target(sx)
                loss = self._task_loss(y_pred, sy)
                inner_optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(target.parameters(), max_norm=1.0)
                inner_optim.step()

        target.eval()
        return target

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
            raise AttributeError("Could not find `training_config` in self or self.module")

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


# %%
# ------------------------------------------------------------
# Meta-train ONCE (identical training loop to ablation_06b -- this follow-up
# only changes what happens with the errors collected during online
# validation, not the simulation that produces them).
# ------------------------------------------------------------

model = MLPEmbedded512_MAML(training_config).to(training_config.device)

epochs = training_config.args.epochs
best_valid_loss = float("inf")

for epoch in range(epochs):
    model.train()
    for i, (x, y) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
        if i >= max_batches:
            break
        x = x.to(model.device)
        y = y.to(model.device)
        model.maml_train_step(x=x, y=y, support_ratio=0.5)

    model.eval()
    valid_y_true, valid_y_pred = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_dataloader):
            if i >= val_max_batches:
                break
            x = x.to(model.device)
            y = y.to(model.device)
            preds = model.expert(model._preprocess_x(x))
            valid_y_true.append(y.detach())
            valid_y_pred.append(pred_log_to_raw(preds).detach())

    valid_metrics = regression_metrics(valid_y_true, valid_y_pred)
    tqdm.write(
        f"Validation after epoch {epoch+1:03d} | "
        f"huber={valid_metrics['huber']:.6f} | mae={valid_metrics['mae']:.4f} | "
        f"rmse={valid_metrics['rmse']:.4f} | smape={valid_metrics['smape']:.2f}% | "
        f"uep={valid_metrics['uep']:.2f}% | r2={valid_metrics['r2']:.4f}"
    )

    if valid_metrics["huber"] < best_valid_loss:
        best_valid_loss = valid_metrics["huber"]
        model.save_for_deploy("best_maml_min_support_crossover_bootstrap.pt")

model.save_for_deploy("MAML_model_min_support_crossover_bootstrap.pt")


# %%
# ------------------------------------------------------------
# Collect + chronologically sort validation samples ONCE.
# ------------------------------------------------------------

started_col = training_config.column_names.index("startedtimestamp")

tqdm.write("[crossover] collecting validation samples once")

_samples = []
for i, (x_batch, y_batch) in enumerate(tqdm(valid_dataloader, desc="Collecting validation data")):
    if i >= val_max_batches:
        break
    x_batch = x_batch.detach().cpu()
    y_batch = y_batch.detach().cpu()
    for x, y in zip(x_batch, y_batch):
        started_time_raw = denormalize_column_value(
            float(x[started_col].item()), "startedtimestamp", training_config
        )
        _samples.append({"x": x, "y": y, "started_time": started_time_raw})

_samples.sort(key=lambda s: s["started_time"])
tqdm.write(f"[crossover] collected {len(_samples)} validation samples")


# %%
# ------------------------------------------------------------
# Single causal pass. The REAL adapt-shared background dynamics run exactly
# as in ablation_06b (persist=True, real online_min_support=4). Layered
# non-destructively on top, for every sample we snapshot the current shared
# expert ("global") and a throwaway-copy few-shot adaptation forced down to
# min_support_override=1 ("adapted").
#
# Unlike ablation_06b, we keep every PER-SAMPLE (k, abs_err_global,
# abs_err_adapted) triple -- not just the per-k running sums -- so the
# aggregation step below can bootstrap a confidence interval instead of
# reporting only a point estimate.
# ------------------------------------------------------------

model._online_buffers.clear()
model.eval()

pending_finished_jobs = []
counter = 0

# per-sample records, parallel lists (memory-light: 3 numbers per sample).
sample_k: List[int] = []
sample_err_global: List[float] = []
sample_err_adapted: List[float] = []

for sample in tqdm(_samples, desc="[crossover] Online validation (adapt-shared background)", leave=False):
    current_time = sample["started_time"]

    # REAL background dynamics: a job only becomes support once it has
    # actually finished. This call mutates `self.expert` for real, exactly
    # like production, gated by the REAL online_min_support (not overridden).
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
    y_true_raw = float(y.view(-1)[0].item())

    jid = int(raw_jobtypeid_from_column(
        x.unsqueeze(0), model.lpmjobtypeid_column, model._lpmjobtypeid_mean, model._lpmjobtypeid_std
    ).item())
    buf = model._online_buffers.get(jid)
    k = len(buf) if buf else 0
    k_bucket = min(k, MIN_SUPPORT_CROSSOVER_MAX_K)

    x_proc = model._preprocess_x(x.unsqueeze(0).to(model.device))

    # --- non-destructive measurement (does not touch the real self.expert) ---
    with torch.no_grad():
        global_pred_log = model.expert(x_proc)
    global_pred_raw = float(pred_log_to_raw(global_pred_log).view(-1)[0].item())

    adapted = model._build_online_adapted_expert(jid, persist=False, min_support_override=1)
    if adapted is None:
        adapted_pred_raw = global_pred_raw
    else:
        with torch.no_grad():
            adapted_pred_raw = float(pred_log_to_raw(adapted(x_proc)).view(-1)[0].item())

    sample_k.append(k_bucket)
    sample_err_global.append(abs(global_pred_raw - y_true_raw))
    sample_err_adapted.append(abs(adapted_pred_raw - y_true_raw))

    # --- REAL production call: this is what determines the trajectory going
    # forward (mutates self.expert for real iff the REAL online_min_support
    # is met). Its own prediction is not separately scored here -- when
    # k >= online_min_support it is numerically equivalent to the "adapted"
    # measurement above (same starting weights, same buffer, same SGD steps).
    with torch.no_grad():
        _ = model(
            x.unsqueeze(0).to(model.device),
            online_adapt=True,
            online_persist=True,
        )

    finish_time = current_time + y_true_raw * 3_600_000.0
    heapq.heappush(pending_finished_jobs, (finish_time, counter, x, y))
    counter += 1


# %%
# ------------------------------------------------------------
# Persist the per-sample records (this is the piece ablation_06b never
# saved, and the reason its conclusion could not be re-checked for
# significance without a brand new run). Written incrementally-safe as a
# flat CSV; at ~1-2M rows x 3 columns this is a few tens of MB, not a
# concern to keep in memory or on disk.
# ------------------------------------------------------------

per_sample_csv_path = "min_support_crossover_per_sample.csv"
with open(per_sample_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k", "abs_err_global", "abs_err_adapted"])
    writer.writerows(zip(sample_k, sample_err_global, sample_err_adapted))
tqdm.write(f"[crossover] wrote {len(sample_k)} per-sample rows to {per_sample_csv_path}")


# %%
# ------------------------------------------------------------
# Aggregate per-k MAE (same summary as ablation_06b, kept for continuity /
# comparison / the raw MAE-vs-k plot).
# ------------------------------------------------------------

per_k = defaultdict(lambda: [0, 0.0, 0.0])
for k, eg, ea in zip(sample_k, sample_err_global, sample_err_adapted):
    acc = per_k[k]
    acc[0] += 1
    acc[1] += eg
    acc[2] += ea

csv_path = "min_support_crossover_results.csv"
fieldnames = ["k", "n", "mae_global", "mae_adapted", "adapted_better"]

rows = []
for k in sorted(per_k.keys()):
    n, sum_global, sum_adapted = per_k[k]
    if n == 0:
        continue
    mae_global = sum_global / n
    mae_adapted = sum_adapted / n
    rows.append(
        {
            "k": k,
            "n": n,
            "mae_global": mae_global,
            "mae_adapted": mae_adapted,
            "adapted_better": mae_adapted < mae_global,
        }
    )

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

tqdm.write("\n===================== MIN-SUPPORT CROSSOVER SUMMARY (adapt-shared background) =====================")
tqdm.write(f"{'k':>4} | {'n':>8} | {'MAE global':>11} | {'MAE adapted':>12} | {'adapted better?':>15}")
for row in rows:
    tqdm.write(
        f"{row['k']:>4} | {row['n']:>8} | {row['mae_global']:>11.4f} | "
        f"{row['mae_adapted']:>12.4f} | {str(row['adapted_better']):>15}"
    )
tqdm.write(f"\nPer-k results written to {csv_path}")


# %%
# ------------------------------------------------------------
# Weighted-benefit bootstrap for each range in REPORT_RANGES. This replaces
# ablation_06b's "first k that is always better from here on" criterion with
# the criterion that actually matters for a production decision: how much
# aggregate error (in sample-hours) does adaptation save or cost over a
# given k-range, and how confident are we in that number given the samples
# actually collected in that range?
# ------------------------------------------------------------

def bootstrap_weighted_benefit(errs_global, errs_adapted, n_boot: int, seed: int):
    """Resample-with-replacement bootstrap of sum(err_global - err_adapted)
    over the given (paired) per-sample errors. Returns (point_estimate,
    ci_lo_95, ci_hi_95, frac_leq_zero) where frac_leq_zero is the empirical
    fraction of bootstrap resamples with total benefit <= 0 (an empirical,
    one-sided estimate of how often this much data would fail to show a
    positive aggregate benefit at all, i.e. an informal p-value against
    "no benefit").

    Vectorized with numpy: for the largest range (all occupancies, k=0..40)
    n is on the order of 1-2M, so a pure-Python nested loop over n_boot x n
    draws would be several billion Python-level operations -- infeasible.
    Resampling is done as one (n_boot, n) integer-index draw plus a single
    matrix-vector product against `diffs`, which numpy executes in compiled
    code."""
    n = len(errs_global)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0

    diffs = np.asarray(errs_global, dtype=np.float64) - np.asarray(errs_adapted, dtype=np.float64)
    point = float(diffs.sum())

    rng = np.random.default_rng(seed)
    # Resample in chunks to bound peak memory (chunk_size x n indices at a time)
    # rather than materializing all n_boot x n draws in memory at once.
    max_elems_per_chunk = 20_000_000  # ~160MB of int64 indices per chunk
    chunk_size = max(1, min(n_boot, max_elems_per_chunk // max(n, 1)))
    boot_sums = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        this_chunk = min(chunk_size, n_boot - done)
        idx = rng.integers(0, n, size=(this_chunk, n))
        boot_sums[done:done + this_chunk] = diffs[idx].sum(axis=1)
        done += this_chunk

    boot_sums.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = int(0.975 * n_boot) - 1
    ci_lo = float(boot_sums[max(0, lo_idx)])
    ci_hi = float(boot_sums[min(n_boot - 1, hi_idx)])
    frac_leq_zero = float(np.mean(boot_sums <= 0.0))

    return point, ci_lo, ci_hi, frac_leq_zero


tqdm.write("\n===================== WEIGHTED BENEFIT, BOOTSTRAPPED (%d resamples) =====================" % N_BOOTSTRAP)
tqdm.write(
    "Positive = adaptation saves error in aggregate over this range; "
    "negative = adaptation costs error in aggregate. sample-hours = sum over\n"
    "all samples in the range of (abs_err_global - abs_err_adapted), in hours."
)

benefit_report_rows = []
for label, lo, hi in REPORT_RANGES:
    idx_global = [errg for k, errg in zip(sample_k, sample_err_global) if lo <= k <= hi]
    idx_adapted = [erra for k, erra in zip(sample_k, sample_err_adapted) if lo <= k <= hi]
    n_range = len(idx_global)

    point, ci_lo, ci_hi, frac_leq_zero = bootstrap_weighted_benefit(
        idx_global, idx_adapted, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )

    verdict = (
        "net POSITIVE, CI excludes 0" if ci_lo > 0 else
        "net NEGATIVE, CI excludes 0" if ci_hi < 0 else
        "NOT distinguishable from 0 at 95% CI"
    )

    tqdm.write(
        f"\n{label} (n={n_range}):\n"
        f"    point estimate      = {point:+.1f} sample-hours\n"
        f"    95% bootstrap CI    = [{ci_lo:+.1f}, {ci_hi:+.1f}] sample-hours\n"
        f"    P(benefit <= 0)     = {frac_leq_zero:.3f} (empirical, over resamples of THIS run's data only)\n"
        f"    verdict             = {verdict}"
    )

    benefit_report_rows.append({
        "range": label, "k_lo": lo, "k_hi": hi, "n": n_range,
        "point_sample_hours": point, "ci_lo_95": ci_lo, "ci_hi_95": ci_hi,
        "p_benefit_leq_0": frac_leq_zero, "verdict": verdict,
    })

benefit_csv_path = "min_support_crossover_weighted_benefit.csv"
with open(benefit_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(benefit_report_rows[0].keys()))
    writer.writeheader()
    for row in benefit_report_rows:
        writer.writerow(row)
tqdm.write(f"\nWeighted-benefit summary written to {benefit_csv_path}")

tqdm.write(
    "\n[reminder] A CI that is still wide relative to the point estimate does NOT mean 'collect more\n"
    "bootstrap resamples' (already at %d here) -- it means this run did not observe enough distinct\n"
    "low-occupancy events. Each Production ID passes through low k only once, near its own launch, so\n"
    "that count scales with the number of distinct production startups in the validation window, not\n"
    "with N_BOOTSTRAP or with the total sample count. Narrowing it requires a wider validation window."
    % N_BOOTSTRAP
)


# %%
# ------------------------------------------------------------
# Plots: (1) the raw MAE-vs-k curves (as in ablation_06b -- dominated by
# bin-to-bin difficulty noise, kept for continuity), and (2) the relative
# improvement per k, which is what actually isolates the effect of interest.
# ------------------------------------------------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 6))
ks = [r["k"] for r in rows]
ax.plot(ks, [r["mae_global"] for r in rows], label="Current shared expert (adapt-shared background, no extra step)", color="gray", linewidth=2)
ax.plot(ks, [r["mae_adapted"] for r in rows], label="One-off few-shot adaptation on top", color="teal", linewidth=2)
ax.axvline(4, color="orange", linestyle=":", label="Current deployed min_support=4")
ax.set_xlabel("Buffer occupancy k (support samples seen so far for this job type)")
ax.set_ylabel("MAE (hours)")
ax.set_title("Few-shot adaptation vs. the real adapt-shared expert as the buffer fills")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("min_support_crossover.png", dpi=300, format="png", bbox_inches="tight")
plt.close()
tqdm.write("Raw MAE-vs-k plot written to min_support_crossover.png")

fig, ax = plt.subplots(figsize=(11, 6))
diff_pct = [
    100.0 * (r["mae_global"] - r["mae_adapted"]) / r["mae_global"] if r["mae_global"] > 0 else 0.0
    for r in rows
]
colors = ["#2ca02c" if d > 0 else "#d62728" for d in diff_pct]
ax.bar(ks, diff_pct, color=colors, width=0.8)
ax.axhline(0, color="black", linewidth=1)
ax.axvline(4, color="orange", linestyle=":", linewidth=2, label="Current deployed min_support=4")
ax.set_xlabel("Buffer occupancy k (support samples seen so far for this job type)")
ax.set_ylabel("Relative MAE improvement from adaptation (%)\n(mae_global - mae_adapted) / mae_global")
ax.set_title("When does one-off few-shot adaptation actually help?\nRelative MAE improvement over the shared expert, by buffer occupancy")
ax.grid(axis="y", alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor="#2ca02c", label="Adaptation helps (positive)"),
    Patch(facecolor="#d62728", label="Adaptation hurts (negative)"),
    plt.Line2D([0], [0], color="orange", linestyle=":", linewidth=2, label="Current deployed min_support=4"),
], loc="upper right")
plt.tight_layout()
plt.savefig("min_support_crossover_diff.png", dpi=300, format="png", bbox_inches="tight")
plt.close()
tqdm.write("Relative-improvement plot written to min_support_crossover_diff.png")
