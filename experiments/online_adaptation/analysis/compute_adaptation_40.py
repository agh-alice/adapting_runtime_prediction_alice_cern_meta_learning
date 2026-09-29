"""Recompute adaptation-latency statistics and figures for the 40-sample threshold.

Reproduces the adaptation-efficiency analysis of the paper (Section 6.3)
and then (a) prints describe() for the _40 and _100 metrics on productions with >= 100 jobs
and (b) saves the three figures used in Section 6.3 for N=40,
in a single-panel style.

Run with an environment that has pandas + seaborn:
  ALICE_DATA_ROOT=/path/to/data python compute_adaptation_40.py
Requires mon_jdls_parsed_*_min.csv.gz, job_info_*.csv, trace_*.csv and combined_site_sonar.json in DATA_ROOT.
"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

DATA_ROOT = os.environ.get("ALICE_DATA_ROOT", "./data/")  # folder with job_info_*.csv, trace_*.csv, mon_jdls_parsed_*_min.csv.gz, combined_site_sonar.json
OUT = os.path.dirname(os.path.abspath(__file__))  # CSV goes here
FIG_OUT = os.path.join(OUT, "figures")
os.makedirs(FIG_OUT, exist_ok=True)
N_SEL = 40          # threshold analysed
MIN_JOBS = 100      # population filter used in the paper ("productions with >= 100 jobs")

plt.rc('font', size=22); plt.rc('axes', labelsize=26, titlesize=24)
plt.rc('xtick', labelsize=22); plt.rc('ytick', labelsize=22)

def load_sharded_csv(pattern, usecols=None):
    files = sorted(glob.glob(os.path.join(DATA_ROOT, pattern)))
    if not files: sys.exit(f"no files for {pattern} in {DATA_ROOT}")
    print(f"loading {len(files)} files for {pattern}", flush=True)
    return pd.concat([pd.read_csv(f, usecols=usecols) for f in files], ignore_index=True)

# two-column extract (job_id, lpmjobtypeid) of mon_jdls_parsed_*.csv, produced on Athena (rows without LPMJobTypeID dropped)
mon_jdls = load_sharded_csv("mon_jdls_parsed_*_min.csv.gz")
job_info = load_sharded_csv("job_info_*.csv", usecols=['job_id', 'status'])
trace = load_sharded_csv("trace_*.csv", usecols=['job_id', 'host', 'requestedcpus', 'startedtimestamp', 'walltime'])
hosts = pd.read_json(os.path.join(DATA_ROOT, "combined_site_sonar.json"), lines=True)
hosts_parsed = hosts[['hostname']].drop_duplicates()

trace['walltime'] = trace['walltime'] / trace['requestedcpus']
trace = trace[trace['walltime'] >= 0]

j = (job_info.where(job_info.status == 'DONE').set_index("job_id")
     .join(trace.set_index('job_id'), how='inner')
     .join(mon_jdls.set_index('job_id'), how='inner').reset_index())
j = j[j['startedtimestamp'] > 0]
j = j.set_index("host", drop=False).join(hosts_parsed.set_index('hostname'), how='inner').reset_index()

df_valid = j[j['startedtimestamp'].notna() & j['walltime'].notna() & j['lpmjobtypeid'].notna()].copy()
df_valid['start_time'] = pd.to_datetime(df_valid['startedtimestamp'], unit='ms')
median_wall = df_valid['walltime'].median()
df_valid['walltime_td'] = pd.to_timedelta(df_valid['walltime'], unit='ms') if median_wall > 1e6 else pd.to_timedelta(df_valid['walltime'], unit='s')
df_valid['end_time'] = df_valid['start_time'] + df_valid['walltime_td']
df_valid = df_valid.sort_values(['lpmjobtypeid', 'start_time'])
print("valid jobs:", len(df_valid), "productions:", df_valid['lpmjobtypeid'].nunique(), flush=True)

rows = []
for jid, g in df_valid.groupby('lpmjobtypeid'):
    g = g.sort_values('start_time'); n_total = len(g)
    t0, tend = g['start_time'].min(), g['end_time'].max(); Tprod = tend - t0
    row = {'lpmjobtypeid': jid, 'n_jobs_total': n_total, 't_0': t0, 't_end': tend, 'Tprod_hours': Tprod.total_seconds()/3600}
    for n in (40, 100, 1000):
        if n_total >= n and Tprod.total_seconds() > 0:
            t_adapt = g.iloc[n-1]['start_time']
            row[f'relative_adaptation_time_{n}'] = (t_adapt - t0) / Tprod
            row[f'relative_job_fraction_{n}'] = n / n_total
            row[f'hours_to_collect_{n}'] = (t_adapt - t0).total_seconds()/3600
        else:
            row[f'relative_adaptation_time_{n}'] = np.nan; row[f'relative_job_fraction_{n}'] = np.nan; row[f'hours_to_collect_{n}'] = np.nan
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "adaptation_metrics_per_production.csv"), index=False)

sub = df[df['n_jobs_total'] >= MIN_JOBS]
print(f"\n=== productions with >= {MIN_JOBS} jobs: {len(sub)} ===")
cols = [f'relative_adaptation_time_{N_SEL}', f'relative_job_fraction_{N_SEL}', 'relative_adaptation_time_100', 'relative_job_fraction_100']
print(sub[cols].describe(percentiles=[.25,.5,.75,.9,.95]).to_string())
t = sub[f'relative_adaptation_time_{N_SEL}']
print(f"\nshare with relative time <= 0.15: {(t<=0.15).mean():.3f}; <= 0.25: {(t<=0.25).mean():.3f}; >= 0.95: {(t>=0.95).sum()} productions ({(t>=0.95).mean():.3f})")
f = sub[f'relative_job_fraction_{N_SEL}']
print(f"share with job fraction <= 0.15: {(f<=0.15).mean():.3f}; <= 0.40: {(f<=0.40).mean():.3f}")
print(f"\n=== all productions with >= {N_SEL} jobs: {df[f'relative_adaptation_time_{N_SEL}'].notna().sum()} ===")
print(df[[f'relative_adaptation_time_{N_SEL}', f'relative_job_fraction_{N_SEL}']].describe(percentiles=[.25,.5,.75,.95]).to_string())

def hist(col, xlabel, fname):
    plt.figure(figsize=(10, 7))
    sns.histplot(sub[col].dropna(), bins=20, kde=True)
    plt.xlabel(xlabel); plt.ylabel('Count'); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_OUT, fname), dpi=300, bbox_inches='tight'); plt.close()
hist(f'relative_adaptation_time_{N_SEL}', r'$(t_{\mathrm{adapt}} - t_0)/T_{\mathrm{prod}}$', 'adaptation_time_distributions_large_100_new.png')
hist(f'relative_job_fraction_{N_SEL}', r'$n_{\mathrm{jobs,adapt}}/n_{\mathrm{jobs,total}}$', 'adaptation_job_fraction_distributions_large_100_new.png')
plt.figure(figsize=(10, 8.5))
sns.scatterplot(data=sub, x=f'relative_job_fraction_{N_SEL}', y=f'relative_adaptation_time_{N_SEL}', alpha=0.7, s=80)
plt.xlabel('Relative job fraction'); plt.ylabel('Relative adaptation time'); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_OUT, 'adaptation_efficiency_large_new.png'), dpi=300, bbox_inches='tight'); plt.close()
print("\nfigures written to", OUT)
