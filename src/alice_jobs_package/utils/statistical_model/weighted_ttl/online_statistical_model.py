import pandas as pd
import numpy as np


class OnlineStatisticalModel:
    def __init__(self):
        self.knowledge = {}
        self.weight_factor = 50

    def process_row(
        self, row: pd.Series, train: bool = True
    ) -> tuple[float | None, float | None]:
        site = row["site"]
        cpu_model = row["cpu_info.cpu_model_name"]
        production_id = row["lpmjobtypeid"]
        req_ttl = row["ttl"] / 60 / 60
        stat_key = (production_id, site, cpu_model)

        if (
            req_ttl < 0
        ):  # Rest cases could be more/less valid, but ttl < 0 is highly invalid.
            return (None, None)

        # --- Prediction ---
        if stat_key in self.knowledge:
            data = self.knowledge[stat_key]
            jobs = data["jobs"]
            estimated_ttl = data["estimatedTTL"]
            weight = self.weight_factor / (self.weight_factor + jobs)
            weighted_ttl = weight * req_ttl + (1 - weight) * estimated_ttl
            result = (min(weighted_ttl, req_ttl), data["stddev"])
        else:
            result = (req_ttl, 0)  # fallback

        # --- Optional Training ---
        if train:
            real_time = row["walltime_hours"]

            if stat_key not in self.knowledge:
                self.knowledge[stat_key] = {
                    "welfordM": 0,
                    "mean": real_time,
                    "jobs": 1,
                    "stddev": 0,
                    "maxTime": real_time,
                    "estimatedTTL": real_time,
                }
            else:
                data = self.knowledge[stat_key]
                jobs = data["jobs"]
                mean = data["mean"]
                welfordM = data["welfordM"]
                max_time = data["maxTime"]

                new_jobs = jobs + 1
                new_mean = (jobs * mean + real_time) / new_jobs
                new_welfordM = welfordM + (real_time - mean) * (real_time - new_mean)
                new_stddev = np.sqrt(new_welfordM / new_jobs) if new_jobs > 1 else 0

                if new_stddev == 0 or (real_time - new_mean) / new_stddev > 3:
                    new_max_time = max_time
                else:
                    new_max_time = max(max_time, real_time)

                new_estimated_ttl = new_max_time + 2 * new_stddev

                self.knowledge[stat_key] = {
                    "welfordM": new_welfordM,
                    "mean": new_mean,
                    "jobs": new_jobs,
                    "stddev": new_stddev,
                    "maxTime": new_max_time,
                    "estimatedTTL": new_estimated_ttl,
                }

        return result
