# Ablation 0: Baseline MAML (paper baseline, Table 5 / Tab. maml-vs-reptile)
# Architecture: LayerNorm + SiLU (same as published model)
# Meta-update:  Second-order MAML (backprop through inner loop, create_graph=True)
# Loss:         Standard Huber on raw hours (NOT log-space, NOT relative MAE)
# Buffer:       None
#
# This corresponds to the "MAML" column in Table 5 of the paper.
# The key difference from Reptile Improved is the meta-update rule and the loss.

from alice_jobs_package.utils.project_config import *
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.model_runner import AliceModelRunner
from alice_jobs_package.training.metrics import *

import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Optional
from collections import defaultdict, deque
from tqdm import tqdm

from alice_jobs_package.utils import logging
from alice_jobs_package.utils.project_config import *
from alice_jobs_package.models.base_alice_model import BaseAliceModel

logger = logging.get_logger(__name__)

ABLATION_NAME = "ablation_00_baseline_maml"


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
# LayerNorm + SiLU expert — output in raw hours
# Same architecture as published model; loss and meta-update differ.
# ---------------------------
class MLPExpert(nn.Module):
    def __init__(self, input_size: int, hidden_dropouts=(0.10, 0.10, 0.05, 0.05), max_target: float = 24.0):
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
        x = F.silu(self.ln1(self.fc1(x))); x = self.do1(x)
        x = F.silu(self.ln2(self.fc2(x))); x = self.do2(x)
        x = F.silu(self.ln3(self.fc3(x))); x = self.do3(x)
        x = F.silu(self.ln4(self.fc4(x))); x = self.do4(x)
        return torch.clamp(F.softplus(self.out(x)), max=self.max_target)


class MLPEmbedded512_MAML(BaseAliceModel):
    def __init__(self, training_config: TrainingConfig):
        super().__init__(training_config)
        self.training_config = training_config

        self.num_col_numbers = training_config.num_col_numbers
        self.cat_col_numbers = training_config.cat_col_numbers
        self.lpmjobtypeid_column = training_config.column_names.index("lpmjobtypeid")

        self.num_config = training_config.num_config
        self._lpmjobtypeid_mean = float(self.num_config["lpmjobtypeid"]["mean"])
        self._lpmjobtypeid_std = float(self.num_config["lpmjobtypeid"]["std"])
        self.cat_config = training_config.cat_config
        self.numerical_dim = len(self.num_config)
        self.categories_dim = len(self.cat_config)

        args = getattr(training_config, "args", {})
        self.embeding_reduction_const = getattr(args, "embeding_reduction_const", 10)
        self.inner_lr = float(getattr(args, "inner_lr", 0.02))
        self.inner_steps = int(getattr(args, "inner_steps", 4))
        self.meta_step_size = float(getattr(args, "meta_step_size", 0.005))
        self.weight_decay = float(getattr(args, "weight_decay", 1e-5))
        self.embedding_dropout_p = float(getattr(args, "embedding_dropout", 0.05))
        self.hidden_dropouts = tuple(getattr(args, "hidden_dropouts", (0.10, 0.10, 0.05, 0.05)))
        self.max_target = float(getattr(args, "max_target", 24.0))
        self.eps = 1e-6

        tqdm.write(
            f"[{ABLATION_NAME}] inner_lr={self.inner_lr:.4f} | inner_steps={self.inner_steps} | "
            f"meta_step_size={self.meta_step_size} | second_order=True"
        )

        self._build_common_layers()
        self.expert = MLPExpert(
            self.input_size,
            hidden_dropouts=self.hidden_dropouts,
            max_target=self.max_target,
        ).to(self.device)

        # Meta-optimizer: Adam with meta_step_size as lr (outer loop for MAML)
        self.optimizer = torch.optim.Adam(
            self.expert.parameters(), lr=self.meta_step_size, weight_decay=self.weight_decay
        )

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
        self.embeded_cat_dim = sum(emb.embedding_dim for emb in self.embeddings.values())
        self.input_size = self.embeded_cat_dim + self.numerical_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess_x(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.cat_col_numbers) > 0:
            cat_x = x[:, self.cat_col_numbers].long()
            cat_embeds: List[torch.Tensor] = []
            for i in range(cat_x.shape[1]):
                vocab = self._cat_num_embeddings[i]
                idx = torch.clamp(cat_x[:, i], 0, vocab - 1)
                cat_embeds.append(self.embeddings[str(i)](idx))
            cat_embed = self.embed_dropout(torch.cat(cat_embeds, dim=1))
        else:
            cat_embed = torch.empty((x.size(0), 0), device=x.device)
        return torch.cat([cat_embed, x[:, self.num_col_numbers].float()], dim=1)

    # Functional forward with explicit fast-weight list — required for second-order MAML
    def _expert_forward_with_params(self, x: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
        it = iter(params)
        w_fc1, b_fc1 = next(it), next(it)
        w_ln1, b_ln1 = next(it), next(it)
        w_fc2, b_fc2 = next(it), next(it)
        w_ln2, b_ln2 = next(it), next(it)
        w_fc3, b_fc3 = next(it), next(it)
        w_ln3, b_ln3 = next(it), next(it)
        w_fc4, b_fc4 = next(it), next(it)
        w_ln4, b_ln4 = next(it), next(it)
        w_out, b_out = next(it), next(it)

        x = F.linear(x, w_fc1, b_fc1)
        x = F.layer_norm(x, (x.size(-1),), w_ln1, b_ln1, eps=self.expert.ln1.eps)
        x = F.silu(x); x = F.dropout(x, p=self.expert.do1.p, training=True)

        x = F.linear(x, w_fc2, b_fc2)
        x = F.layer_norm(x, (x.size(-1),), w_ln2, b_ln2, eps=self.expert.ln2.eps)
        x = F.silu(x); x = F.dropout(x, p=self.expert.do2.p, training=True)

        x = F.linear(x, w_fc3, b_fc3)
        x = F.layer_norm(x, (x.size(-1),), w_ln3, b_ln3, eps=self.expert.ln3.eps)
        x = F.silu(x); x = F.dropout(x, p=self.expert.do3.p, training=True)

        x = F.linear(x, w_fc4, b_fc4)
        x = F.layer_norm(x, (x.size(-1),), w_ln4, b_ln4, eps=self.expert.ln4.eps)
        x = F.silu(x); x = F.dropout(x, p=self.expert.do4.p, training=True)

        x = F.linear(x, w_out, b_out)
        return torch.clamp(F.softplus(x), max=self.max_target)

    # Standard Huber on raw hours — matches paper's MAML baseline loss
    def _task_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(y_pred, y_true, delta=1.0, reduction="mean")

    def _split_tasks(self, x: torch.Tensor, y: torch.Tensor, support_ratio: float = 0.5) -> list:
        jobtype_ids = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std).unique()
        tasks = []
        for jobtypeid in jobtype_ids:
            mask = raw_jobtypeid_from_column(x, self.lpmjobtypeid_column, self._lpmjobtypeid_mean, self._lpmjobtypeid_std) == jobtypeid
            x_task = x[mask][torch.randperm(mask.sum())]
            y_task = y[mask][torch.randperm(mask.sum())]
            if x_task.numel() == 0:
                continue
            split = max(1, min(int(len(x_task) * support_ratio), len(x_task) - 1))
            sx = self._preprocess_x(x_task[:split]).detach().to(self.device)
            sy = y_task[:split].detach().to(self.device)
            qx = self._preprocess_x(x_task[split:]).detach().to(self.device) if len(x_task) > split else torch.empty(0, self.input_size, device=self.device)
            qy = y_task[split:].detach().to(self.device) if len(x_task) > split else torch.empty(0, device=self.device)
            tasks.append((sx, sy, qx, qy))
        return tasks

    def maml_train_step(self, x: torch.Tensor, y: torch.Tensor, support_ratio: float = 0.5) -> dict:
        self.expert.train()
        tasks = self._split_tasks(x, y, support_ratio=support_ratio)
        if not tasks:
            return {"loss": 0.0, "huber": 0.0, "r2": 0.0}

        self.optimizer.zero_grad(set_to_none=True)

        meta_loss_val = 0.0
        loss_tasks = 0
        query_preds, query_targets = [], []

        for sx, sy, qx, qy in tasks:
            # Fast weights start from current parameters (no clone — grads flow through)
            fast_params = list(self.expert.parameters())

            # Inner loop: gradient steps with create_graph=True for second-order MAML
            for _ in range(self.inner_steps):
                y_pred_support = self._expert_forward_with_params(sx, fast_params)
                inner_loss = self._task_loss(y_pred_support, sy)
                grads = torch.autograd.grad(inner_loss, fast_params, create_graph=True)
                fast_params = [w - self.inner_lr * g for w, g in zip(fast_params, grads)]

            if qx.numel() == 0:
                continue

            # Outer loss: backprop through inner loop (second-order)
            y_pred_query = self._expert_forward_with_params(qx, fast_params)
            query_loss = self._task_loss(y_pred_query, qy)
            (query_loss / len(tasks)).backward()

            with torch.no_grad():
                meta_loss_val += query_loss.item()
                query_preds.append(y_pred_query.detach().view(-1).cpu())
                query_targets.append(qy.detach().view(-1).cpu())
            loss_tasks += 1

        if loss_tasks == 0:
            return {"loss": 0.0, "huber": 0.0, "r2": 0.0}

        torch.nn.utils.clip_grad_norm_(self.expert.parameters(), max_norm=1.0)
        self.optimizer.step()

        y_pred_all = torch.cat(query_preds)
        y_true_all = torch.cat(query_targets)
        huber = F.huber_loss(y_pred_all, y_true_all, delta=1.0).item()
        return {
            "loss": meta_loss_val / float(loss_tasks),
            "huber": huber,
            "r2": r2_score(y_pred_all, y_true_all),
        }

    def forward(self, x: torch.Tensor, y_true: Optional[torch.Tensor] = None, online_adapt: bool = False) -> torch.Tensor:
        x = x.to(self.device)
        if self.training:
            self.expert.train()
            return self.expert(self._preprocess_x(x))
        self.expert.eval()
        with torch.no_grad():
            return self.expert(self._preprocess_x(x))

    def _get_training_config(self) -> TrainingConfig:
        if hasattr(self, "training_config"):
            return self.training_config
        raise AttributeError("Could not find `training_config`")

    def save_for_deploy(self, output_path: str) -> None:
        tc = self._get_training_config()
        torch.save({
            "model_state_dict": self.state_dict(),
            "cat_config": tc.cat_config,
            "num_config": tc.num_config,
            "column_names": tc.column_names,
            "cat_col_numbers": tc.cat_col_numbers,
            "num_col_numbers": tc.num_col_numbers,
            "ablation": ABLATION_NAME,
        }, output_path)


# %%

training_config = TrainingConfig(args_mode=ArgsMode.FILE, training_args_path='./training_config.json')
X_, y_ = AliceModelRunner.load_numpy_data(training_config)
train_dataloader, valid_dataloader = AliceModelRunner.prepare_dataloaders(training_config, X_, y_)

percent = 1
max_batches = max(1, len(train_dataloader) * percent)
val_max_batches = max(1, len(valid_dataloader) * percent)

model = MLPEmbedded512_MAML(training_config).to(training_config.device)


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

best_valid_loss = float("inf")
history_preds = {"train": {}, "valid": {}}
epochs = training_config.args.epochs

for epoch in range(epochs):
    model.train()
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
    train_losses, train_hubers, train_r2s = [], [], []
    train_y_true, train_y_pred = [], []

    for i, (x, y) in enumerate(pbar):
        if i >= max_batches:
            break
        x = x.to(model.device); y = y.to(model.device)
        metrics = model.maml_train_step(x=x, y=y, support_ratio=0.5)
        train_losses.append(metrics["loss"])
        train_hubers.append(metrics["huber"])
        train_r2s.append(metrics["r2"])

        if epoch + 1 == epochs:
            with torch.no_grad():
                model.eval(); pred = model(x, online_adapt=False); model.train()
            train_y_true.append(y.detach().cpu().view(-1))
            train_y_pred.append(pred.detach().cpu().view(-1))

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

    model.eval()
    valid_y_true, valid_y_pred = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(valid_dataloader, desc="Validation", leave=False)):
            if i >= val_max_batches:
                break
            x = x.to(model.device); y = y.to(model.device)
            preds = model(x, online_adapt=False)
            valid_y_true.append(y.detach())
            valid_y_pred.append(preds.detach())

    valid_metrics = regression_metrics(valid_y_true, valid_y_pred)
    print(
        f"Validation after epoch {epoch+1:03d} | "
        f"huber={valid_metrics['huber']:.6f} | "
        f"mae={valid_metrics['mae']:.4f} | "
        f"rmse={valid_metrics['rmse']:.4f} | "
        f"smape={valid_metrics['smape']:.2f}% | "
        f"r2={valid_metrics['r2']:.4f}"
    )

    if valid_metrics["huber"] < best_valid_loss:
        best_valid_loss = valid_metrics["huber"]
        model.save_for_deploy(f"best_{ABLATION_NAME}.pt")
        print(f"✓ New best model | huber={valid_metrics['huber']:.6f} | r2={valid_metrics['r2']:.4f}")

model.save_for_deploy(f"{ABLATION_NAME}_final.pt")

# %%  Final validation — pure offline inference, no buffer

model.eval()
final_y_true, final_y_pred = [], []
with torch.no_grad():
    for i, (x, y) in enumerate(tqdm(valid_dataloader, desc=f"[{ABLATION_NAME}] Final validation")):
        if i >= val_max_batches:
            break
        x = x.to(model.device); y = y.to(model.device)
        preds = model(x, online_adapt=False)
        final_y_true.append(y.detach())
        final_y_pred.append(preds.detach())

final_metrics = regression_metrics(final_y_true, final_y_pred)
history_preds["valid"][epochs] = {
    "y_true": torch.cat([t.cpu().view(-1) for t in final_y_true]),
    "y_pred": torch.cat([t.cpu().view(-1) for t in final_y_pred]),
}
print(
    f"[{ABLATION_NAME}] Final | "
    f"huber={final_metrics['huber']:.6f} | "
    f"mae={final_metrics['mae']:.4f} | "
    f"rmse={final_metrics['rmse']:.4f} | "
    f"smape={final_metrics['smape']:.2f}% | "
    f"r2={final_metrics['r2']:.4f}"
)
