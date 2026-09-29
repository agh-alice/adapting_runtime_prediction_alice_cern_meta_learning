"""Regenerate user_production_time_distribution_new.png with legible axis fonts.

Box plots of
job duration (<= 24 h, per-CPU walltime) for the 20 most frequent productions of the alitrain and
aliprod users. The current data extract (mon_jdls_parsed_*_min.csv.gz) carries no user column, so the
user of each production is taken from the labels of the previous version of the figure.
Written to ./figures/.
"""
import glob
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = os.environ.get("ALICE_DATA_ROOT", "./data/")  # folder with job_info_*.csv, trace_*.csv, mon_jdls_parsed_*_min.csv.gz, combined_site_sonar.json
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_OUT = os.path.join(HERE, "figures")
os.makedirs(FIG_OUT, exist_ok=True)
PNG = "user_production_time_distribution_new.png"

USER_OF_PRODUCTION = {
    **{p: "aliprod" for p in (31996, 32282, 32470, 32666, 32667, 32668, 32669, 32670, 32778, 32779,
                              32781, 32787, 32846, 32847, 32927, 32928, 32984)},
    **{p: "alitrain" for p in (1929, 5780, 20117)},
}


def load_sharded_csv(pattern, usecols=None):
    files = sorted(glob.glob(os.path.join(DATA_ROOT, pattern)))
    return pd.concat([pd.read_csv(f, usecols=usecols) for f in files], ignore_index=True)


def main():
    mon_jdls = load_sharded_csv("mon_jdls_parsed_*_min.csv.gz")
    job_info = load_sharded_csv("job_info_*.csv", usecols=["job_id", "status"])
    trace = load_sharded_csv("trace_*.csv", usecols=["job_id", "requestedcpus", "startedtimestamp", "walltime"])
    trace["walltime"] = trace["walltime"] / trace["requestedcpus"]
    trace = trace[trace["walltime"] >= 0]
    jobs = (job_info.where(job_info.status == "DONE").set_index("job_id")
            .join(trace.set_index("job_id"), how="inner")
            .join(mon_jdls.set_index("job_id"), how="inner").reset_index())
    jobs = jobs[jobs["startedtimestamp"] > 0]

    counts = jobs["lpmjobtypeid"].value_counts()
    print("top 20 productions by job count in the current data:", list(counts.head(20).index.astype(int)))
    print("productions in the figure:", sorted(USER_OF_PRODUCTION))

    d = jobs[jobs["lpmjobtypeid"].isin(USER_OF_PRODUCTION)].copy()
    d["lpmjobtypeid"] = d["lpmjobtypeid"].astype(int)
    d["user"] = d["lpmjobtypeid"].map(USER_OF_PRODUCTION)
    d["duration_hours"] = d["walltime"] / 3600
    d = d[d["duration_hours"] <= 24]
    print("jobs plotted:", len(d))

    fig, ax = plt.subplots(figsize=(14, 7))
    d.boxplot(by=["user", "lpmjobtypeid"], column=["duration_hours"], ax=ax)
    plt.suptitle("")
    ax.set_title("")
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 4))
    ax.set_xlabel("[User, Production ID]", fontsize=26)
    ax.set_ylabel("Duration [hours]", fontsize=26)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=18)
    ax.tick_params(axis="y", labelsize=22)
    plt.tight_layout()
    for out_dir in (FIG_OUT,):
        plt.savefig(os.path.join(out_dir, PNG), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
