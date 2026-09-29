"""Regenerate aliprod_catastrophic_forgetting_all_metrics.png with legible fonts.

Values are taken from the Aliprod catastrophic-forgetting logs (learn_eval_maml_fs_with_cf.py):
R(1,1) and R(4,1) for the ephemeral and continual variants; intermediate checkpoints are linearly
interpolated, as stated in the paper). Written to ./figures/.
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_OUT = os.path.join(HERE, "figures")
os.makedirs(FIG_OUT, exist_ok=True)
PNG = "aliprod_catastrophic_forgetting_all_metrics.png"

X_LABELS = ["After Q1\n(current)", "After Q2\n(new)", "After Q3\n(newer)", "After Q4\n(newest)"]
# (title, ephemeral Q1 error (constant), continual Q1 error after Q1, continual Q1 error after Q4, y-axis format, y tick step)
METRICS = [
    ("Huber Loss on Q1 data", 0.1490, 0.0439, 0.1829, "%.2f", 0.05),
    ("MAE on Q1 data", 0.8380, 0.2945, 1.0083, "%.1f", 0.2),
    ("SMAPE (%) on Q1 data", 33.76, 10.02, 36.71, "%.0f", 10),
    ("RMSE on Q1 data", 1.2441, 0.5687, 1.4733, "%.1f", 0.2),
]
COL_EPH, COL_CONT = "#1f77b4", "#d62728"


def main():
    x = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    handles = labels = None
    for ax, (title, eph, c0, c1, fmt, tick) in zip(axes.flatten(), METRICS):
        step = (c1 - c0) / 3
        cont = [c0, c0 + step, c0 + 2 * step, c1]
        ax.plot(x, [eph] * 4, marker="o", linestyle="--", color=COL_EPH, linewidth=2.5, markersize=8, label="Ephemeral (A5) - Stable")
        ax.plot(x, cont, marker="s", linestyle="-", color=COL_CONT, linewidth=2.5, markersize=8, label="Continual (A6) - Forgetting")
        cx = 3 * (eph - c0) / (c1 - c0)
        ax.plot(cx, eph, marker="X", color="black", markersize=12, zorder=5)
        ax.annotate(f"Reversal pt\nx={cx:.2f}", xy=(cx, eph), xytext=(cx - 1.9, eph - (c1 - c0) * 0.42),
                    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=7), fontsize=15, fontweight="bold")
        y_max = c1 + (c1 - c0) * 0.1
        ax.axhspan(eph, y_max, alpha=0.08, color="red", label="Drift Zone (Worse than baseline)")
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, fontsize=14)
        ax.tick_params(axis="y", labelsize=15)
        ax.yaxis.set_major_locator(MultipleLocator(tick))
        ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
        ax.set_title(title, fontsize=19, fontweight="bold", pad=10)
        ax.set_ylim(0, y_max)
        ax.grid(True, linestyle="--", alpha=0.6)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=17, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    for d in (FIG_OUT,):
        fig.savefig(os.path.join(d, PNG), format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
