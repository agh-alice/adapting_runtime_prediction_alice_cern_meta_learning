import torch
import torch.nn as nn

from alice_jobs_package.models.base_alice_model import BaseAliceModel
from alice_jobs_package.models.base_mlp_emb import BaseMLPEmbedded
from alice_jobs_package.training.config import TrainingConfig

class MLPEmbedded512(BaseAliceModel):
    def __init__(self, training_config : TrainingConfig):
        super().__init__(training_config)

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

class MoeMLPEmbedded_2x512(BaseAliceModel):
    def __init__(self, training_config: TrainingConfig):
        super().__init__(training_config)

        self.routing_column = 'user' #Pass to config
        self.expert_value_map = {'aliprod': 0, 'alitrain':1}
        self.user_cat_config = training_config.cat_config[self.routing_column]
        print(self.user_cat_config)
        self.routing_column_num = training_config.column_names.index(self.routing_column)
        self.routing_map = self.setup_routing_map()
        print(self.routing_map)
        self.distinct_experts = sorted(set(self.routing_map.values()))
        
        self.experts = nn.ModuleList(
            [MLPEmbedded512(training_config) for _ in range(len(self.routing_map))]
            )
    
    def setup_routing_map(self):
        # Create the final mapping
        final_mapping = {
            self.user_cat_config[key]: value
            for key, value in self.expert_value_map.items()
            if key in self.user_cat_config
        }

        return final_mapping

    def forward(self, x):
        # Check input shape
        if x.dim() == 2:
            # x: (batch, feature)
            batch_size = x.size(0)
            user_ids = x[:, self.routing_column_num].long()
            is_3d = False
        elif x.dim() == 3:
            # x: (batch, seq, feature)
            batch_size = x.size(0)
            user_ids = x[:, 0, self.routing_column_num].long()  # Assume routing value is in first time step
            is_3d = True
        else:
            raise ValueError(f"Expected input of shape (B, F) or (B, S, F), got {x.shape}")

        # Map user_ids to expert_ids
        mapped_ids = torch.tensor(
            [self.routing_map.get(uid.item(), self.routing_map.get('any_other', 0)) for uid in user_ids],
            device=x.device
        )

        # Prepare output tensor
        if is_3d:
            seq_len = x.size(1)
            outputs = torch.zeros((batch_size, seq_len, 1), device=x.device)
        else:
            outputs = torch.zeros((batch_size, 1), device=x.device)

        #Run experts
        for expert_id in range(len(self.experts)):
            mask = mapped_ids == expert_id  # (batch,)
            if mask.any():
                # Slice input for this expert
                x_expert = x[mask]  # Shape: (masked_batch, ...) — either 2D or 3D

                y_expert = self.experts[expert_id](x_expert)

                outputs[mask] = y_expert

        return outputs