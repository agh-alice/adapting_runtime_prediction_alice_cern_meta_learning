from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.utils import project_config

class StatisticalModelConfig(TrainingConfig):
    def __init__(self, args_mode: project_config.ArgsMode, training_args_path=None, alice_preprocessor: AliceDataPreprocessor = None):
        super().__init__(args_mode, training_args_path, alice_preprocessor)
        # In case of changes in column names the values in this dictionary can be changed to account for that.
        self.variable_name_to_column_name_dict = {"production_id": "lpmjobtypeid", "cpu_cores": "requestedcpus", "cpu_model": "cpu_info.cpu_model_name", "site": "site", "ce_name": "ce_name"}
        self.findIndecies()

    def findIndecies(self):
        self.variable_name_to_index_dict = {}
        for variable_name, column_name in self.variable_name_to_column_name_dict.items():
            if not column_name in self.column_names:
                raise Exception(f'Field \'{column_name}\' missing from data for StatisticalModelConfig')
            else:
                self.variable_name_to_index_dict[variable_name] = self.column_names.index(column_name)
