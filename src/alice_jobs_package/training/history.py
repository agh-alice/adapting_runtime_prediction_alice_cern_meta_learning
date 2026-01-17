import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from abc import ABC, abstractmethod
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

from alice_jobs_package.utils import logging, tools

logger = logging.get_logger(__name__)

class History(ABC):
    def __init__(self, metrics : dict):
        self.metrics = list(metrics.keys())
    
    def save_history_to_file(self, filepath : Path): 
        data = self.dump_history_to_json()
        
        filepath = tools.parse_to_pathlib(filepath)
        filepath = tools.ensure_extension(filepath, '.json')

        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)

    def load_history_from_file(self, filepath : Path):
        filepath = tools.parse_to_pathlib(filepath)
        filepath = tools.ensure_extension(filepath, ".json")

        with open(filepath, "r") as file:
            data = json.load(file)
            self.load_history_from_json(data)

    @abstractmethod
    def dump_history_to_json(self) -> dict:
        pass
    
    @abstractmethod
    def load_history_from_json(self, data : dict):
        pass

class TrainingHistory(History):
    def __init__(self, metrics : dict = {}):
        super().__init__(metrics)
        self.lr = []

        self.loss_train = []
        self.loss_valid = []
        self.uncertainty_valid = []

        for metric in metrics:
            setattr(self, f'{metric}_train', [])
            setattr(self, f'{metric}_valid', [])

    def dump_history_to_json(self) -> dict:
        data = {
            "lr": self.lr,
            "loss_train": self.loss_train,
            "loss_valid": self.loss_valid,
            "uncertainty_valid": self.uncertainty_valid
        }

        for metric in self.metrics:
            data[f"{metric}_train"] = getattr(self, f"{metric}_train")
            data[f"{metric}_valid"] = getattr(self, f"{metric}_valid")
        
        data["metrics"] = self.metrics
        return data
    
    def load_history_from_json(self, data):
        self.metrics = data["metrics"]

        self.lr = data["lr"]
        self.loss_train = data["loss_train"]
        self.loss_valid = data["loss_valid"]
        self.uncertainty_valid = data["uncertainty_valid"]

        for metric in self.metrics:
            setattr(self, f"{metric}_train", data[f"{metric}_train"])
            setattr(self, f"{metric}_valid", data[f"{metric}_valid"])

    def plot_history(self, filepath : Path = None, show : bool = False, plot_train : bool = True, plot_valid : bool = True, evaluation_frequency : int = 1): 
        assert plot_train or plot_valid

        add_plots = 3

        plt.figure(figsize=(8, 16))
        plot_num = add_plots + len(self.metrics)

        plt.subplot(plot_num, 1, 1)
        if plot_train:
            plt.plot(range(1, len(self.lr) + 1), self.lr, label=f'Training Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.title('LR Curve')
        plt.legend()
        plt.grid()

        plt.subplot(plot_num, 1, 2)
        
        if plot_train:
            for i, loss_component in enumerate(np.array(self.loss_train).T):
                plt.plot(range(1, len(loss_component) + 1), loss_component, label=f'Training Loss {i}')

        if plot_valid:
            for i, loss_component in enumerate(np.array(self.loss_valid).T):
                validation_indices = range(evaluation_frequency, len(self.loss_train) + 1, evaluation_frequency)
                plt.plot(validation_indices, loss_component, label=f'Valid Epoch Loss_{i}')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid()

        plt.subplot(plot_num, 1, 3)
        if plot_valid:
            validation_indices = range(evaluation_frequency, len(self.uncertainty_valid) + 1, evaluation_frequency)
            plt.plot(validation_indices, self.uncertainty_valid, label=f'Valid Epoch Uncertainty')
        plt.xlabel('Epoch')
        plt.ylabel('Uncertainty')
        plt.title('Uncertainty Curve')
        plt.legend()
        plt.grid()

        for i, metric in enumerate(self.metrics, add_plots + 1):
            plt.subplot(plot_num, 1, i)
            if plot_train:
                metric_data = getattr(self, f'{metric}_train')
                plt.plot(range(1, len(metric_data) + 1), metric_data, label=f'Training Epoch {metric}')
            if plot_valid:
                metric_data_val = getattr(self, f'{metric}_valid')
                validation_indices = range(evaluation_frequency, len(metric_data) + 1, evaluation_frequency)
                plt.plot(validation_indices, metric_data_val, label=f'Valid Epoch {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(f'{metric}')
            plt.title(f'{metric} Curve')
            plt.legend()
            plt.grid()

        plt.tight_layout()
        if show:
            plt.show()
        else:
            filepath = tools.ensure_extension(filepath, '.png')
            plt.savefig(filepath, dpi=300, format='png', bbox_inches='tight')
        plt.close()

class EvalHistory(History):
    def __init__(self, metrics : dict = {}):
        super().__init__(metrics)
        self.loss_eval = []
        self.uncertainty_eval = []

        self.y_pred = []
        self.y_uncertainty = []
        self.y_true = []

        for metric in metrics:
            setattr(self, f'{metric}_eval', [])

    def dump_history_to_json(self) -> dict:
        data = {
            "loss_eval": self.loss_eval,
            "y_pred": self.y_pred,
            "y_uncertainty": self.y_uncertainty,
            "y_true": self.y_true
        }

        for metric in self.metrics:
            data[f"{metric}_eval"] = getattr(self, f"{metric}_eval")
        
        data["metrics"] = self.metrics
        return data
    
    def load_history_from_json(self, data):
        self.metrics = data["metrics"]

        self.loss_eval = data["loss_eval"]
        self.y_pred = data["y_pred"]
        self.y_uncertainty = data["y_uncertainty"]
        self.y_true = data["y_true"]

        for metric in self.metrics:
            setattr(self, f"{metric}_eval", data[f"{metric}_eval"])

    def plot_history(self, filepath : Path = None, show : bool = False):
        fig, ax = plt.subplots(figsize=(8, 8))  # Square figure

        y_true_ = torch.tensor(self.y_true)
        y_uncertainty_ = torch.tensor(self.y_uncertainty)
        y_pred_ = torch.tensor(self.y_pred)

        if len(y_true_.shape) > 2:
            y_true_ = torch.squeeze(y_true_, dim=-1)
            y_uncertainty_ = torch.squeeze(y_uncertainty_, dim=-1)
            y_pred_ = torch.squeeze(y_pred_, dim=-1)

        # Calculate absolute errors
        errors = torch.abs(y_pred_ - y_true_)

        # Compute thresholds for percentiles including 99%
        thresholds = {
            "q50 ~ ": torch.quantile(errors, 0.5).item(),
            "q75 ~ ": torch.quantile(errors, 0.75).item(),
            "q95 ~ ": torch.quantile(errors, 0.95).item(),
            "q99 ~ ": torch.quantile(errors, 0.99).item()
        }

        # Scatter plot with log-scaled colors
        y_uncertainty_clipped = torch.clamp(torch.abs(y_uncertainty_), min=1e-6)
        sc = ax.scatter(y_true_, y_pred_, alpha=0.7, c=y_uncertainty_clipped, s=1, label="Predictions", cmap='viridis', norm=LogNorm())

        # Identity line (perfect prediction)
        x_min, x_max = 0, 24
        ax.plot([x_min, x_max], [x_min, x_max], 'r--', label="Ideal Fit")

        # Mean std parallel dashed lines (± mean std)
        mean_std = y_uncertainty_clipped.mean().item()
        for delta, style in [(mean_std, 'r--')]:
            # upper line y = x + delta
            x_start = max(x_min, 0 - delta)
            x_end = min(x_max, 24 - delta)
            y_start = x_start + delta
            y_end = x_end + delta
            ax.plot([x_start, x_end], [y_start, y_end], style, linewidth=1, label=f'Mean Std ±{delta:.2f}')

            # lower line y = x - delta
            x_start = max(x_min, 0 + delta)
            x_end = min(x_max, 24 + delta)
            y_start = x_start - delta
            y_end = x_end - delta
            ax.plot([x_start, x_end], [y_start, y_end], style, linewidth=1)

        # Add percentile bounds with lines clipped to [0, 24] on x-axis
        colors = ['orange', 'blue', 'green', 'purple']
        for (label, delta), color in zip(thresholds.items(), colors):
            # Upper bound: y = x + delta
            x_start = max(x_min, 0 - delta)
            x_end = min(x_max, 24 - delta)
            y_start = x_start + delta
            y_end = x_end + delta
            ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=1, label=label + f'{delta:.2f}h')

            # Lower bound: y = x - delta
            x_start = max(x_min, 0 + delta)
            x_end = min(x_max, 24 + delta)
            y_start = x_start - delta
            y_end = x_end - delta
            ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=1)

        # Axis settings
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("True vs Predicted")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left')

        # Add narrower colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label('Predictive Std')

        plt.tight_layout()
        if show:
            plt.show()
        else:
            filepath = tools.ensure_extension(filepath, '.png')
            plt.savefig(filepath, dpi=300, format='png', bbox_inches='tight')
        plt.close()