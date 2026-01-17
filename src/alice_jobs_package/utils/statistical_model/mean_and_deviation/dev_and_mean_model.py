import torch
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.utils.statistical_model.mean_and_deviation.statistical_model_config import StatisticalModelConfig

class StatisticalModelDevAndMean(torch.nn.Module):
    def __init__(self, training_config : StatisticalModelConfig):
        ''' This alorithm constructs keys: (productionID, site, cpuModelName) and for each key calculates the mean and stddev using welfords algoritm. '''
        self.training_config = training_config
        self.column_names = training_config.column_names
        self.variable_name_to_index_dict = training_config.variable_name_to_index_dict
        self.knowledge = {}
        self.training = True
    
    def forward(self, X: torch.Tensor, y: torch.Tensor):
        if self.training:
            self.train(X, y)
        else:
            return self.predict(X)

    def train(self, X: torch.Tensor, y: torch.Tensor):
        # Handles (batch, features) and (batch, seq, features)
        # y = walltime
        if X.ndim == 2:
            X = X.unsqueeze(1)
            y = y.unsqueeze(1)

        batch_size, seq_len, _ = X.shape

        for i in range(batch_size):
            for t in range(seq_len):
                row = X[i][t]
                walltime = y[i][t][0]

                def r(variable_name):
                    column_index = self.variable_name_to_index_dict[variable_name]
                    return row[column_index]

                site = r("site")

                if int(r("cpu_cores")) != 0:
                    real_time = walltime / int(r("cpu_cores"))
                else:
                    continue

                cpuModelName = r("cpu_model")
                productionID = r("production_id")
                stat_key = (float(productionID), float(site), float(cpuModelName))

                print("key", stat_key, "wt", walltime)
    
                if stat_key not in self.knowledge:
                    self.knowledge[stat_key] = {
                        "welfordM": 0,
                        "mean": real_time,
                        "jobs": 1,
                        "stddev": 0,
                    }
                else:
                    data = self.knowledge[stat_key]
                    jobs = data["jobs"]
                    mean = data["mean"]
                    welfordM = data["welfordM"]
    
                    new_jobs = jobs + 1
                    new_mean = (jobs * mean + real_time) / new_jobs
                    new_welfordM = welfordM + (real_time - mean) * (real_time - new_mean)
                    new_stddev = torch.sqrt(new_welfordM / new_jobs) if new_jobs > 1 else 0
    
                    self.knowledge[stat_key] = {
                        "welfordM": new_welfordM,
                        "mean": new_mean,
                        "jobs": new_jobs,
                        "stddev": new_stddev,
                    }

    def predict(self, X: torch.Tensor):
        # Handles (batch, features) and (batch, seq, features)
        # Returns np.array([mean, stddev]) for each row of features. Will return [0, 0] if either some values are missing or key was not encountered before.
        # Returns (batch, 2) if input was (batch, features), otherwise returns (batch, seq, 2)
        if X.ndim == 2:
            X = X.unsqueeze(1)

        batch_size, seq_len, _ = X.shape
        y = torch.zeros(batch_size, seq_len, 2, device=X.device)

        for i in range(batch_size):
            for t in range(seq_len):
                row = X[i][t]

                def r(variable_name):
                    column_index = self.variable_name_to_index_dict[variable_name]
                    return row[column_index]

                site = r("site")

                cpuModelName = r("cpu_model")
                productionID = int(r("production_id"))
                stat_key = (float(productionID), float(site), float(cpuModelName))
    
                if stat_key in self.knowledge:
                    data = self.knowledge[stat_key]
                    y[i][t][0] = data["mean"]
                    y[i][t][1] = data["stddev"]

        return y if seq_len > 1 else y[:, 0, :]
    