import pandas as pd
import numpy as np
from alice_jobs_package.training.config import TrainingConfig
import torch

class StatisticalModelRecreationPandas(torch.nn.Module):
    def __init__(self):
        ''' This alorithm constructs keys: (productionID, site, cpuModelName) and for each key calculates the mean and stddev using welfords algoritm. '''
        # self.training_config = training_config
        # self.column_names = training_config.column_names
        # self.variable_name_to_index_dict = training_config.variable_name_to_index_dict
        self.knowledge = {}
        self.training = True
        self.weight_factor = 50
    
    def forward(self, X: pd.DataFrame, y: pd.DataFrame):
        if self.training:
            self.train(X, y)
        else:
            return self.predict(X)

    # def get_cpu_model_name(row):
    #     sample_cols = [col for col in row.columns if row.startswith('cpu_info.cpu_model_name')]
    #     return row[sample_cols].idxmax(axis=1)

    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        print("X.shape", X.shape)

        for index, row in X.iterrows():

            walltime = y.loc[index]['walltime_hours'] # TODO: Does this make sense?

            site = row["site"]

            # if row["requestedcpus"] > 0:
            #     real_time = walltime / row["requestedcpus"]
            # else:
            #     continue
            real_time = walltime

            # cpuModelName = self.get_cpu_model_name(row)
            cpuModelName = row['cpu_info.cpu_model_name']
            productionID = row["lpmjobtypeid"]
            stat_key = (productionID, site, cpuModelName)

            if stat_key not in self.knowledge:
                self.knowledge[stat_key] = {
                    "welfordM": 0,
                    "mean": real_time,
                    "jobs": 1,
                    "stddev": 0,
                    "maxTime": real_time,
                    "estimatedTTL": real_time
                }
            else:
                data = self.knowledge[stat_key]
                jobs = data["jobs"]
                mean = data["mean"]
                welfordM = data["welfordM"]
                maxTime = data["maxTime"]

                new_jobs = jobs + 1
                new_mean = (jobs * mean + real_time) / new_jobs
                new_welfordM = welfordM + (real_time - mean) * (real_time - new_mean)
                new_stddev = np.sqrt(new_welfordM / new_jobs) if new_jobs > 1 else 0

                # Update maxTime based on standard deviation threshold
                if new_stddev == 0:
                    new_maxTime = maxTime
                elif (real_time - new_mean) / new_stddev > 3:
                    new_maxTime = maxTime
                else:
                    new_maxTime = max(maxTime, real_time)

                # Update estimatedTTL
                new_estimatedTTL = int(new_maxTime + 2 * new_stddev)

                self.knowledge[stat_key] = {
                    "welfordM": new_welfordM,
                    "mean": new_mean,
                    "jobs": new_jobs,
                    "stddev": new_stddev,
                    "maxTime": new_maxTime,
                    "estimatedTTL": new_estimatedTTL
                }

    def predict(self, X: pd.DataFrame):
        
        print("X.shape", X.shape)
        y = pd.DataFrame([0] * X.shape[0], columns=['weighted_ttl']) # TODO: Does this make sense

        for index, row in X.iterrows():

            site = row["site"]
            # cpuModelName = self.get_cpu_model_name(row)
            cpuModelName = row['cpu_info.cpu_model_name']
            productionID = int(row["lpmjobtypeid"])
            stat_key = (productionID, site, cpuModelName)

            reqTTL = row["requestedttl"] / 60 / 60 # Map to hours

            if stat_key in self.knowledge:
                data = self.knowledge[stat_key]
                jobs = data["jobs"]
                estimatedTTL = data["estimatedTTL"]
                weight = self.weight_factor / (self.weight_factor + jobs)
                weightedTTL = weight * reqTTL + (1 - weight) * estimatedTTL
                if weightedTTL < reqTTL:
                    y.at[index, 'weighted_ttl'] = weightedTTL
                else:
                    y.at[index, 'weighted_ttl'] = reqTTL

        return y
    
