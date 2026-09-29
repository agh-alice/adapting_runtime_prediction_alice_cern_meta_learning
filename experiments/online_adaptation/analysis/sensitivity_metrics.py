"""Regenerate sensitivity_metrics_aliprod.pdf and sensitivity_metrics_alidaq.pdf consistently.
Data = Table `tab:buffer_sensitivity_combined` (values from the buffer-size sweep logs). Dashed line = best value per metric; no shading, no suptitle, SMAPE label."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.figsize": (8.5, 6), "figure.dpi": 300})
buffers = ['10', '20', '40', '60', '80', '100', '250', '500']; x = np.arange(len(buffers))
DATA = {
 "aliprod": {'Loss': [0.128178, 0.125862, 0.125856, 0.126248, 0.126585, 0.126898, 0.128200, 0.129500],
             'MAE':  [0.7320, 0.7202, 0.7201, 0.7221, 0.7238, 0.7254, 0.7321, 0.7385],
             'RMSE': [1.0870, 1.0700, 1.0678, 1.0694, 1.0711, 1.0729, 1.0808, 1.0889],
             'SMAPE':[30.05, 29.35, 29.17, 29.18, 29.21, 29.26, 29.51, 29.81]},
 "alidaq":  {'Loss': [0.0788, 0.0784, 0.0787, 0.0790, 0.0793, 0.0794, 0.0801, 0.0817],
             'MAE':  [0.4706, 0.4687, 0.4703, 0.4718, 0.4733, 0.4741, 0.4775, 0.4859],
             'RMSE': [1.1546, 1.1516, 1.1578, 1.1648, 1.1711, 1.1760, 1.1958, 1.2183],
             'SMAPE':[38.49, 38.52, 38.63, 38.70, 38.77, 38.84, 39.17, 39.62]},
}
TITLES = {'Loss': ('Loss Function', 'Loss'), 'MAE': ('Mean Absolute Error', 'MAE'),
          'RMSE': ('Root Mean Square Error', 'RMSE'), 'SMAPE': ('Symmetric MAPE', 'SMAPE [%]')}
def panel(ax, v, title, ylab):
    ax.plot(x, v, marker='o', color='#d62728', linewidth=2, markersize=5, zorder=3)
    r = max(v) - min(v) or 0.1; ax.set_ylim(min(v) - 0.1*r, max(v) + 0.1*r)
    ax.axvline(x=int(np.argmin(v)), color='gray', linestyle='--', linewidth=1, zorder=1)
    ax.set_title(title, pad=10); ax.set_ylabel(ylab); ax.grid(True, linestyle=':', alpha=0.7, zorder=0)
for wl, data in DATA.items():
    fig, axs = plt.subplots(2, 2, sharex=True); fig.subplots_adjust(hspace=0.15, wspace=0.25)
    for ax, key in zip(axs.flat, ['Loss', 'MAE', 'RMSE', 'SMAPE']):
        panel(ax, data[key], *TITLES[key])
    for ax in axs[1, :]: ax.set_xticks(x); ax.set_xticklabels(buffers); ax.set_xlabel('FIFO Buffer Size ($n$)')
    for ax in axs[0, :]: ax.tick_params(labelbottom=False)
    fig.savefig(os.path.join(OUT, f"sensitivity_metrics_{wl}.pdf"), format='pdf', bbox_inches='tight'); plt.close(fig)
    print("written", wl)
