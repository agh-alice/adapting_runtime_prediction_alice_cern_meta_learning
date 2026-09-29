import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional
from collections import defaultdict, deque

from alice_jobs_package.models.base_alice_model import BaseAliceModel
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.utils import logging

logger = logging.get_logger(__name__)


def raw_jobtypeid_from_column(x, col, mean, std):
    """Recover the raw (un-normalized) lpmjobtypeid from its z-score-standardized
    column value, rounding to the nearest integer, instead of truncating the raw
    standardized float directly with .long(). lpmjobtypeid is a NUMERICAL column
    in this pipeline (not embedding-encoded); truncating the standardized value
    collapses many distinct real production IDs into the same integer bucket
    whenever std is large relative to typical ID-to-ID gaps (observed: ~163
    distinct IDs collapsed into ~2 buckets on the Aliprod validation range)."""
    return torch.round(x[:, col] * std + mean).long()


# ---------------------------
# Utility: positive head
# ---------------------------
def positive_head(z: torch.Tensor, max_val: float = 24.0) -> torch.Tensor:
    # Softplus avoids zero-gradient near 0, clamp prevents exploding tails
    return torch.clamp(F.softplus(z), max=max_val)


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
        # SiLU tends to work well on tabular MLPs
        x = F.silu(self.ln1(self.fc1(x)))
        x = self.do1(x)

        x = F.silu(self.ln2(self.fc2(x)))
        x = self.do2(x)

        x = F.silu(self.ln3(self.fc3(x)))
        x = self.do3(x)

        x = F.silu(self.ln4(self.fc4(x)))
        x = self.do4(x)

        return positive_head(self.out(x), self.max_target)


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
        self.online_max_support = int(getattr(args, "online_max_support", 40))
        self.online_min_support = int(getattr(args, "online_min_support", 4))
        self.online_inner_steps = int(getattr(args, "online_inner_steps", 2))
        self.online_inner_lr = float(
            getattr(args, "online_inner_lr", self.inner_lr)
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
        # Reserve an UNK bucket (+1) for each categorical to be safe for OOV values at inference
        self.embeddings = nn.ModuleDict()
        self._cat_num_embeddings = []  # track sizes for clamping
        for num, key in enumerate(self.cat_config.keys()):
            card = len(self.cat_config[key])
            emb_dim = max(min(10, card), (card // self.embeding_reduction_const) + 1)
            # +1 for UNK/OOV bucket
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
        # Categorical
        if len(self.cat_col_numbers) > 0:
            cat_x = x[:, self.cat_col_numbers].long()
            # Clamp to [0, num_embeddings-1] to map OOV to UNK index
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

        # Numerical
        num_x = x[:, self.num_col_numbers].float()

        return torch.cat([cat_embed, num_x], dim=1)

    # ---------------------------
    # Losses (log-space Huber + small relative term)
    # ---------------------------
    def _log_huber(
        self, y_pred_pos: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        # Train in log1p space (robust and aligns better with relative error)
        log_p = torch.log1p(y_pred_pos)
        log_t = torch.log1p(torch.clamp(y_true, min=self.eps))
        return F.huber_loss(
            log_p, log_t, delta=self.huber_delta_log, reduction="mean"
        )

    def _relative_mae(
        self, y_pred_pos: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        # Small stabilizer to help MAPE without over-optimizing it
        denom = torch.clamp(y_true.abs(), min=self.eps)
        return (y_pred_pos - y_true).abs().div(denom).mean()

    def _task_loss(
        self, y_pred_pos: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return self._log_huber(y_pred_pos, y_true) + self.rel_lambda * self._relative_mae(
            y_pred_pos, y_true
        )

    # ---------------------------
    # Task building (no threshold, always build a task)
    # ---------------------------
    def _split_tasks(
        self, x: torch.Tensor, y: torch.Tensor, support_ratio: float = 0.5
    ) -> list:
        """
        Returns:
          tasks: list of (sx, sy, qx, qy) for every group (may have empty query if group size == 1)
        """
        jobtype_ids = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std).unique()
        tasks = []

        for jobtypeid in jobtype_ids:
            mask = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std) == jobtypeid
            x_task = x[mask]
            y_task = y[mask]

            if x_task.numel() == 0:
                continue

            # Shuffle within-group
            perm = torch.randperm(len(x_task))
            x_task = x_task[perm]
            y_task = y_task[perm]

            # Split (support non-empty; query may be empty if len==1)
            split = max(1, min(int(len(x_task) * support_ratio), len(x_task) - 1))
            support_x, query_x = x_task[:split], x_task[split:]
            support_y, query_y = y_task[:split], y_task[split:]

            sx = self._preprocess_x(support_x).to(self.device)
            sy = support_y.to(self.device)

            if len(query_x) > 0:
                qx = self._preprocess_x(query_x).to(self.device)
                qy = query_y.to(self.device)
            else:
                # Create empty tensors for query path (used only for logging)
                qx = torch.empty(0, self.input_size, device=self.device)
                qy = torch.empty(0, device=self.device)

            tasks.append((sx, sy, qx, qy))

        return tasks

    # ---------------------------
    # Inner-loop adaptation (SGD) with grad clipping
    # ---------------------------
    def _adapt_once(
        self,
        model: nn.Module,
        sx: torch.Tensor,
        sy: torch.Tensor,
        lr: Optional[float] = None,
    ) -> None:
        """Single inner-loop step (SGD) on given model."""
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
    # Meta step: pure Reptile (no fallback ERM)
    # ---------------------------
    def maml_train_step(
        self, x: torch.Tensor, y: torch.Tensor, support_ratio: float = 0.5
    ) -> float:
        """
        Reptile update:
          For each task: w' = Adapt(w; support), accumulate delta = (w' - w).
          Outer update:  w <- w + meta_step_size * mean_delta
        Returns:
          mean query loss over tasks that had a non-empty query set (for logging only).
        """
        self.expert.train()

        tasks = self._split_tasks(x, y, support_ratio=support_ratio)
        if not tasks:
            return 0.0

        # Accumulate parameter deltas and query losses for monitoring
        with torch.no_grad():
            accum_delta = [torch.zeros_like(p) for p in self.expert.parameters()]

        query_losses = []

        for sx, sy, qx, qy in tasks:
            adapted = self._task_adapt(sx, sy)

            # Optional monitoring on query set if it exists
            if len(qx) > 0:
                with torch.no_grad():
                    q_pred = adapted(qx)
                    q_loss = self._task_loss(q_pred, qy).item()
                    query_losses.append(q_loss)

            # Accumulate delta (w' - w)
            with torch.no_grad():
                for acc, p_adapted, p_init in zip(
                    accum_delta, adapted.parameters(), self.expert.parameters()
                ):
                    acc.add_(p_adapted.data - p_init.data)

        # Apply averaged delta (outer/meta step)
        scale = self.meta_step_size / float(len(tasks))
        with torch.no_grad():
            for p, d in zip(self.expert.parameters(), accum_delta):
                p.add_(scale * d)

        # Optional: small Adam step to apply weight decay / smooth drifts (no gradient set)
        for param in self.expert.parameters():
            param.grad = None
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.step()  # weight_decay shrinkage only

        # Return mean query loss for logging
        if len(query_losses) == 0:
            return 0.0
        return sum(query_losses) / len(query_losses)

    # ---------------------------
    # Online few-shot utils
    # ---------------------------
    def _push_online_support(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Store (x, y) in per-jobtype buffers.
        We keep data on CPU to avoid unnecessary VRAM usage.
        """
        # Do not mix training data into online buffers
        if self.training:
            return

        jobtype_ids = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std).cpu()
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()

        for i in range(x_cpu.size(0)):
            jid = int(jobtype_ids[i].item())
            self._online_buffers[jid].append((x_cpu[i], y_cpu[i]))

    def _build_online_adapted_expert(
        self, jobtype_id: int
    ) -> Optional[nn.Module]:
        """
        Build a temporary adapted expert for a given jobtype_id
        from the buffered support set. Returns None if we have
        too few samples.
        """
        buf = self._online_buffers.get(jobtype_id, None)
        if not buf or len(buf) < self.online_min_support:
            return None

        xs = torch.stack([t[0] for t in buf], dim=0).to(self.device)
        ys = torch.stack([t[1] for t in buf], dim=0).to(self.device)

        sx = self._preprocess_x(xs)
        sy = ys

        adapted = deepcopy(self.expert)

        # Force gradients on (in case someone calls in a no_grad context),
        # but we will only use this in eval/inference anyway.
        with torch.enable_grad():
            adapted.train()
            for _ in range(self.online_inner_steps):
                self._adapt_once(adapted, sx, sy, lr=self.online_inner_lr)

        adapted.eval()
        return adapted

    # ---------------------------
    # Inference + optional online few-shot adaptation
    # ---------------------------
    def forward(
        self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        online_adapt: bool = True,
    ) -> torch.Tensor:
        """
        Training mode (self.training == True):
          - standard differentiable forward pass (no online adaptation).
        Eval / inference (self.training == False):
          - if y_true is provided, we push (x, y_true) into per-jobtype buffers,
          - if online_adapt is enabled, we attempt to build temporary adapted
            experts for each lpmjobtypeid and use them if enough support data
            is available; otherwise we fall back to the global expert.
        """
        x = x.to(self.device)

        # -----------------------------
        # 1) TRAINING MODE: no online adaptation, keep gradients
        # -----------------------------
        if self.training:
            self.expert.train()
            x_processed = self._preprocess_x(x)
            return self.expert(x_processed)

        # -----------------------------
        # 2) EVAL / INFERENCE MODE
        # -----------------------------

        # If we have ground truth (e.g. callback after job finishes), log to buffers.
        if y_true is not None:
            y_true = y_true.to(self.device)
            self._push_online_support(x, y_true)

        # If online adaptation is disabled or explicitly turned off -> plain expert inference.
        if not (self.online_adaptation_enabled and online_adapt):
            self.expert.eval()
            x_processed = self._preprocess_x(x)
            with torch.no_grad():
                return self.expert(x_processed)

        # -----------------------------
        # 3) EVAL + ONLINE FEW-SHOT ADAPTATION
        # -----------------------------
        self.expert.eval()
        jobtype_ids = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std)
        unique_ids = jobtype_ids.unique()

        preds = torch.empty(x.size(0), 1, device=self.device)

        # For each jobtype, use either adapted expert (if available) or global expert
        for jid in unique_ids:
            jid_int = int(jid.item())
            mask = jobtype_ids == jid
            x_group = x[mask]

            adapted = self._build_online_adapted_expert(jid_int)
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


# Backward-compatible alias
MLPEmbedded512 = MLPEmbedded512_MAML
