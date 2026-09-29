"""Regenerate the two Section 3.3 sample-collection figures from the 2026 data snapshot.

Outputs (same file names as in the paper sources, written to ./figures/):
  - accumulative collection samples chart_new.png  (stacked bars, sorted by time to collect 100 samples)
  - Time to collect N samples_large_new.png        (median / P75 time to collect N samples, N = 1..100)
Also prints the statistics quoted in Section 3.3.

Mirrors the data pipeline of adaptation_40/compute_adaptation_40.py and the plotting cells of
the paper's data-analysis notebook (Section 3.3 figures).
"""
import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = os.environ.get("ALICE_DATA_ROOT", "./data/")  # folder with job_info_*.csv, trace_*.csv, mon_jdls_parsed_*_min.csv.gz, combined_site_sonar.json
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_OUT = os.path.join(HERE, "figures")
os.makedirs(FIG_OUT, exist_ok=True)
BAR_PNG = "accumulative collection samples chart_new.png"
CURVE_PNG = "Time to collect N samples_large_new.png"


def load_sharded_csv(pattern, usecols=None):
    files = sorted(glob.glob(os.path.join(DATA_ROOT, pattern)))
    if not files:
        raise SystemExit(f"no files for {pattern}")
    return pd.concat([pd.read_csv(f, usecols=usecols) for f in files], ignore_index=True)


def load_valid_jobs():
    mon_jdls = load_sharded_csv("mon_jdls_parsed_*_min.csv.gz")
    job_info = load_sharded_csv("job_info_*.csv", usecols=["job_id", "status"])
    trace = load_sharded_csv("trace_*.csv", usecols=["job_id", "host", "requestedcpus", "startedtimestamp", "walltime"])
    hosts = pd.read_json(os.path.join(DATA_ROOT, "combined_site_sonar.json"), lines=True)[["hostname"]].drop_duplicates()

    trace["walltime"] = trace["walltime"] / trace["requestedcpus"]
    trace = trace[trace["walltime"] >= 0]
    jobs = (job_info.where(job_info.status == "DONE").set_index("job_id")
            .join(trace.set_index("job_id"), how="inner")
            .join(mon_jdls.set_index("job_id"), how="inner").reset_index())
    jobs = jobs[jobs["startedtimestamp"] > 0]
    jobs = jobs.set_index("host").join(hosts.set_index("hostname"), how="inner").reset_index()
    valid = jobs[jobs["startedtimestamp"].notna() & jobs["walltime"].notna() & jobs["lpmjobtypeid"].notna()].copy()
    valid["start_time"] = pd.to_datetime(valid["startedtimestamp"], unit="ms")
    return valid.sort_values(["lpmjobtypeid", "start_time"])


def hours_to_nth(group, n):
    return (group["start_time"].iloc[n - 1] - group["start_time"].iloc[0]).total_seconds() / 3600


def main():
    valid = load_valid_jobs()
    groups = [g for _, g in valid.groupby("lpmjobtypeid")]
    print(f"valid jobs: {len(valid)} productions: {len(groups)}")

    # --- Figure 1: time to collect 100 and 1000 samples, sorted by the 100-sample time
    rows = []
    for g in groups:
        rows.append({
            "lpmjobtypeid": g["lpmjobtypeid"].iloc[0], "n_jobs": len(g),
            "hours_to_collect_100": hours_to_nth(g, 100) if len(g) >= 100 else np.nan,
            "hours_to_collect_1000": hours_to_nth(g, 1000) if len(g) >= 1000 else np.nan,
        })
    coll = pd.DataFrame(rows)
    coll.to_csv(os.path.join(HERE, "sample_collection_per_production.csv"), index=False)

    df_plot = coll[(coll.hours_to_collect_100 > 0) & (coll.hours_to_collect_1000 > 0)].copy()
    df_plot["hours_100_to_1000"] = df_plot.hours_to_collect_1000 - df_plot.hours_to_collect_100
    df_plot = df_plot.sort_values("hours_to_collect_100").reset_index(drop=True)
    x = np.arange(len(df_plot))
    plt.figure(figsize=(14, 7))
    plt.bar(x, df_plot.hours_to_collect_100, width=0.6, label="First 100 samples")
    plt.bar(x, df_plot.hours_100_to_1000, width=0.6, bottom=df_plot.hours_to_collect_100, label="Samples 100 → 1000")
    plt.yscale("log")
    plt.xlabel("Production IDs (sorted by time to collect 100 samples)", fontsize=22)
    plt.ylabel("Time to Collect Samples (hours)", fontsize=22)
    plt.xticks([])
    plt.tick_params(axis="x", length=0)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, loc="upper left")
    plt.tight_layout()
    for d in (FIG_OUT,):
        plt.savefig(os.path.join(d, BAR_PNG), dpi=300, bbox_inches="tight")
    plt.close()

    # --- Figure 2: median / P75 time to collect N samples (population at N: productions with >= N jobs)
    durations = {n: [] for n in range(1, 101)}
    for g in groups:
        for n in range(1, min(101, len(g) + 1)):
            durations[n].append(hours_to_nth(g, n))
    curve = pd.DataFrame({
        "samples_count": list(durations),
        "median_time_hours": [np.median(v) for v in durations.values()],
        "p75_time_hours": [np.percentile(v, 75) for v in durations.values()],
        "productions": [len(v) for v in durations.values()],
    })
    curve.to_csv(os.path.join(HERE, "time_to_collect_n_samples.csv"), index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(curve.samples_count, curve.median_time_hours, label="Median", linewidth=2)
    plt.plot(curve.samples_count, curve.p75_time_hours, label="P75", linewidth=2, linestyle="--")
    plt.xlabel("Number of Samples", fontsize=22)
    plt.ylabel("Time to Collect (hours)", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    for d in (FIG_OUT,):
        plt.savefig(os.path.join(d, CURVE_PNG), dpi=400, bbox_inches="tight")
    plt.close()

    # --- Statistics quoted in Section 3.3
    h100 = coll.hours_to_collect_100.dropna()
    h1000 = coll.hours_to_collect_1000.dropna()
    print(f"productions >=100: {len(h100)}  >=1000: {len(h1000)}  bars in figure 1: {len(df_plot)}")
    print("hours_to_collect_100 percentiles 50/75/90/99:", np.round(np.percentile(h100, [50, 75, 90, 99]), 2))
    print(f"100 samples: >24h {int((h100 > 24).sum())}, <=24h {int((h100 <= 24).sum())}")
    print(f"1000 samples: >24h {int((h1000 > 24).sum())}, <=24h {int((h1000 <= 24).sum())}")
    print(f"slowest 1% of h100 exceeds {np.percentile(h100, 99) / 24:.0f} days")
    c = curve.set_index("samples_count")
    for n in (10, 16, 20, 30, 36, 40, 60, 80, 100):
        print(f"N={n}: median={c.loc[n, 'median_time_hours']:.2f} h  P75={c.loc[n, 'p75_time_hours']:.1f} h  (productions={c.loc[n, 'productions']})")
    jumps = c.p75_time_hours.diff()
    print("largest P75 steps (N: +hours):", {int(k): round(v, 1) for k, v in jumps.nlargest(6).items()})
    print("median max over N<=100:", round(c.median_time_hours.max(), 2))


if __name__ == "__main__":
    main()
