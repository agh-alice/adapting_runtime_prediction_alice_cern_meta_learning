import torch
from torch import nn

from alice_jobs_package.utils import logging
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.models.base_alice_model import BaseAliceModel

logger = logging.get_logger(__name__)

class MLPEmbedded512(BaseAliceModel):
    def __init__(self, training_config : TrainingConfig):
        super(MLPEmbedded512, self).__init__(training_config)

        self.num_col_numbers = training_config.num_col_numbers
        self.cat_col_numbers = training_config.cat_col_numbers

        self.numerical_dim = len(self.num_config)
        self.categories_dim = len(self.cat_config)

        self.embeding_reduction_const = training_config.args.embeding_reduction_const

        self._build_common_layers()

    def _build_common_layers(self):
        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleDict({
            f'{num}': nn.Embedding(
                num_embeddings=len(self.cat_config[key]), 
                embedding_dim=max(min(10, len(self.cat_config[key])), (len(self.cat_config[key]) // self.embeding_reduction_const) + 1)
            )
            for num, key in enumerate(self.cat_config.keys())
        })

        logger.info(f"Embedings dims: {
            { key : self.embeddings[f'{num}'].embedding_dim 
             for num, key in enumerate(self.cat_config.keys()) }
            }" 
        )

        # Compute total input size
        self.embeded_cat_dim = sum(emb.embedding_dim for emb in self.embeddings.values())
        self.input_size = self.embeded_cat_dim + self.numerical_dim

        # MLP layers
        self.fc1 = nn.Linear(self.input_size, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.batch_norm2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.batch_norm3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.1)
        self.batch_norm4 = nn.BatchNorm1d(64)

        self.fc5 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        numerical_inputs = x[:, self.num_col_numbers]
        categorical_inputs = x[:, self.cat_col_numbers].to(dtype=torch.int32)

        embedded_features = [self.embeddings[f'{col}'](categorical_inputs[:, col]) for col in range(categorical_inputs.shape[1])]
        embedded_features = torch.cat(embedded_features, dim=1)

        x = torch.cat([embedded_features, numerical_inputs], dim=1)

        # MLP forward pass
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
        x = torch.clamp(self.softplus(x), max=24)

        return x