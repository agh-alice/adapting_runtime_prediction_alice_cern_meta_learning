import torch
import torch.nn as nn

from alice_jobs_package.models.base_alice_model import BaseAliceModel
from alice_jobs_package.models.base_mlp_emb import BaseMLPEmbedded
from alice_jobs_package.training.config import TrainingConfig

class MoeMLPEmbedded(BaseAliceModel):
    def __init__(self, training_config: TrainingConfig):
        super().__init__(training_config)

        self.routing_column = 'user'
        self.expert_value_map = {'aliprod': 0, 'alitrain':1, 'alihyperloop': 2, 'alidaq':3, 'any_other': 4}
        self.user_cat_config = training_config.cat_config[self.routing_column]
        self.routing_column_num = training_config.column_names.index(self.routing_column)
        self.routing_map = self.setup_routing_map()
        
        self.experts = nn.ModuleList(
            [BaseMLPEmbedded(training_config) for _ in range(len(self.routing_map))]
            )
    
    def setup_routing_map(self):
        # Create the final mapping
        final_mapping = {
            self.user_cat_config[key]: value
            for key, value in self.expert_value_map.items()
            if key in self.user_cat_config
        }

        # Assign the last expert id for all remaining items
        all_rest_id = max(self.user_cat_config.values()) + 1
        final_mapping[all_rest_id] = self.expert_value_map['any_other']

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