import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from alice_jobs_package.training.config import TrainingConfig

class BaseAliceModel(nn.Module, ABC):
    def __init__(self, training_config : TrainingConfig):
        super(BaseAliceModel, self).__init__()
        self.num_config : dict = training_config.num_config
        self.cat_config : dict = training_config.cat_config

        self.num_col_numbers : int = training_config.num_col_numbers
        self.cat_col_numbers : int = training_config.cat_col_numbers

        self.numerical_dim : int = len(self.num_config)
        self.categories_dim : int = len(self.cat_config)
    
    def set_dropout_to_train(self):
        """Turn on dropout layers only, keep rest of model in eval mode."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @abstractmethod
    def forward(self, x):
        pass