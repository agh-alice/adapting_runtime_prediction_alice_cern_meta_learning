import torch
from torch import nn

from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.models.base_alice_model import BaseAliceModel

class BaseMLP(BaseAliceModel):
        def __init__(self, training_config : TrainingConfig):
            super().__init__(training_config)
            
            self.input_size = len(self.num_config) + sum(len(v) for v in self.cat_config.values())
            self._build_common_layers()
        
        def _build_common_layers(self):
            # Define the layers
            self.fc1 = nn.Linear(self.input_size, 1024)
            self.dropout1 = nn.Dropout(0.4)
            self.fc2 = nn.Linear(1024, 512)
            self.dropout2 = nn.Dropout(0.4)
            self.fc3 = nn.Linear(512, 256)
            self.dropout3 = nn.Dropout(0.3)
            self.fc4 = nn.Linear(256, 128)
            self.dropout4 = nn.Dropout(0.2)
            self.fc5 = nn.Linear(128, 64)
            self.dropout5 = nn.Dropout(0.1)
            self.fc6 = nn.Linear(64, 1)

            self.batch_norm1 = nn.BatchNorm1d(1024)
            self.batch_norm2 = nn.BatchNorm1d(512)
            self.batch_norm3 = nn.BatchNorm1d(256)
            self.batch_norm4 = nn.BatchNorm1d(128)
            self.batch_norm5 = nn.BatchNorm1d(64)

            self.relu = nn.ReLU()
            self.softplus = nn.Softplus()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.batch_norm1(x)
            x = self.dropout1(x)

            x = self.fc2(x)
            x = self.relu(x)
            x = self.batch_norm2(x)
            x = self.dropout2(x)

            x = self.fc3(x)
            x = self.relu(x)
            x = self.batch_norm3(x)
            x = self.dropout3(x)

            x = self.fc4(x)
            x = self.relu(x)
            x = self.batch_norm4(x)
            x = self.dropout4(x)

            x = self.fc5(x)
            x = self.relu(x)
            x = self.batch_norm5(x)
            x = self.dropout5(x)

            x = self.fc6(x)
            x = torch.clamp(self.softplus(x), max=24)

            return x