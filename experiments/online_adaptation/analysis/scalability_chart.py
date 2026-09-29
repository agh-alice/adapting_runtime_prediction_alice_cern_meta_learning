"""Regenerate maml_scalability_chart.pdf for the /predict endpoint only (Gatling 3.13.5 reports rebuilt from
alice_research/gatling-results/maml{20,40,60}reqs/simulation.log, request 'Get Walltime Prediction')."""
import os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
rates = [20, 40, 60]
series = {'50th pct': [14, 14, 6635], '75th pct': [15, 15, 23085], '95th pct': [15, 15, 32376],
          '99th pct': [21, 181, 34518], 'Mean': [14, 18, 12731]}
markers = {'50th pct': 'o', '75th pct': 's', '95th pct': '^', '99th pct': 'D', 'Mean': 'v'}
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10})
fig, ax = plt.subplots(figsize=(8, 5.2))
for name, vals in series.items():
    ax.plot(rates, vals, marker=markers[name], linewidth=2, markersize=7, label=name)
ax.set_yscale('log'); ax.set_xticks(rates)
ax.set_xlabel('Request Rate (req/s)', fontweight='bold'); ax.set_ylabel('Response Time (ms) - Log Scale', fontweight='bold')
ax.axhline(40, color='gray', linestyle=':', linewidth=1); ax.text(20.3, 44, '40 ms limit', color='gray', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5); ax.legend(title='Metrics (/predict)')
fig.tight_layout(); fig.savefig(os.path.join(OUT, "maml_scalability_chart.pdf"), format='pdf', bbox_inches='tight')
print("written maml_scalability_chart.pdf")
