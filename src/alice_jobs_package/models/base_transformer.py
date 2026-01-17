import torch
from torch import nn

from alice_jobs_package.utils import logging
from alice_jobs_package.training.config import TrainingConfig
from alice_jobs_package.models.base_alice_model import BaseAliceModel

logger = logging.get_logger(__name__)

class BaseTransforemr(BaseAliceModel):
    def __init__(self, training_config : TrainingConfig):
        super().__init__(training_config)

        #Network parameters
        self.positional_encoding = training_config.args.positional_encoding
        self.output_dim = training_config.args.output_dim
        self.teacher_forcing = training_config.args.teacher_forcing

        #Separation layer, it is created to separate data from trainig examples that should be processed in separate way
        self.separate = SeparationLayer(training_config)

        #Embedding layer
        self.embeddings = Embedding(training_config)

        # Additional constants
        self.d_model = self.embeddings.d_model
        self.ffn_hidden = self.d_model * 2

        #Possitional encoding
        if self.positional_encoding == "sincos":
            self.positional_encoding = PositionalEncoding(training_config, self.d_model)

        #Encoder layer * encoder_block_num
        self.encoder_blocks = EncoderBlocks(training_config, self.d_model, self.ffn_hidden)

        #Last feed forward with casting to output_dim (in case of this project 1 neuron)
        self.output = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.output_dim),
            nn.Softplus()
        )

    def forward(self, x):

        categorical_inputs, numerical_inputs = self.separate(x)

        x = self.embeddings(categorical_inputs, numerical_inputs)

        if self.positional_encoding:
            x = self.positional_encoding(x)

        x = self.encoder_blocks(x)
        x = self.output(x)
        x = torch.clamp(x, max=24)

        # Select only the last job's representation
        if not self.teacher_forcing:
            x = x[:, -1, :] # (batch_size, sequence, d_model) -> (batch_size, d_model)
        
        return x

class SeparationLayer(nn.Module):
    def __init__(self, training_config : TrainingConfig):
        super().__init__()

        self.cat_col_numbers = training_config.cat_col_numbers
        self.num_col_numbers = training_config.num_col_numbers.copy()

    def forward(self, x):
        # Extract numerical & categorical features
        numerical_inputs = x[:, :, self.num_col_numbers]
        categorical_inputs = x[:, :, self.cat_col_numbers].to(dtype=torch.int32)
        
        return categorical_inputs, numerical_inputs

class Embedding(nn.Module):
    def __init__(self, training_config : TrainingConfig):
        super().__init__()

        self.numerical_dim = len(training_config.num_config)
        self.cat_config = training_config.cat_config
        self.embeding_reduction_const = training_config.args.embeding_reduction_const

        self.embeddings = nn.ModuleDict({
            f'{num}': nn.Embedding(
                num_embeddings=len(self.cat_config[key]),
                embedding_dim=self._adjust_embedding_dim(len(self.cat_config[key]), training_config.args.attn_heads)
            )
            for num, key in enumerate(self.cat_config.keys())
        }).to(training_config.device)

        logger.info(f"Embedings dims: {
            { key : self.embeddings[f'{num}'].embedding_dim 
             for num, key in enumerate(self.cat_config.keys()) }
            }" 
        )

        self.embeded_cat_dim = sum(emb.embedding_dim for emb in self.embeddings.values())
        self.d_model = self.embeded_cat_dim + self.numerical_dim
    
    def _adjust_embedding_dim(self, original_dim, num_heads):
        adjusted_dim = max(min(num_heads, original_dim), (original_dim // self.embeding_reduction_const) + 1)
        return ((adjusted_dim // num_heads) + 1) * num_heads

    def forward(self, categorical_inputs, numerical_inputs):
        # Apply embeddings for each categorical column
        embedded_features = [self.embeddings[f'{col}'](categorical_inputs[:, :, col]) for col in range(categorical_inputs.shape[-1])]
        embedded_features = torch.cat(embedded_features, dim=-1)

        x = torch.cat([embedded_features, numerical_inputs], dim=-1)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, training_config : TrainingConfig, d_model : int):
        super().__init__()

        self.seq_len = training_config.args.sequence_length
        self.d_model = d_model

        pe = torch.zeros(self.seq_len, self.d_model)  # (seq_len, d_model)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        self.positional_encoding = pe.unsqueeze(0).to(training_config.device)  # (1, seq_len, d_model) for batch broadcasting
        
    def forward(self, x):
        """
        x shape: (batch, seq, d_model) 
        """
        return x + self.positional_encoding[:, :x.shape[1], :]

class EncoderBlocks(nn.Module):
    def __init__(self, training_config : TrainingConfig, d_model : int, ffn_hidden : int):
        super().__init__()

        self.encoder_block_num = training_config.args.encoder_block_num
        self.encoder_blocks = nn.Sequential(
            *[EncoderBlock(id, training_config, d_model, ffn_hidden) for id in range(self.encoder_block_num)]
            )

    def forward(self, x):
        """
        x shape: (batch, seq, d_model) 
        """
        return self.encoder_blocks(x)

class EncoderBlock(nn.Module):
    def __init__(self, layer_id : int, training_config : TrainingConfig, d_model : int, ffn_hidden : int):
        super().__init__()

        self.layer_id = layer_id

        self.attn = MultiHeadAttention(training_config, d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(training_config, d_model, ffn_hidden)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, training_config : TrainingConfig, d_model : int):
        super().__init__()

        if training_config.args.teacher_forcing:
            self.mask = self.generate_reverse_causal_mask(training_config.args.sequence_length).to(training_config.device)
        else:
            self.mask = None

        self.attn = nn.MultiheadAttention(embed_dim=d_model, 
                                          num_heads=training_config.args.attn_heads, 
                                          dropout=training_config.args.dropout_rate, 
                                          batch_first=True)
        
        self.dropout = nn.Dropout(training_config.args.dropout_rate)

    def generate_reverse_causal_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)  # Lower triangular mask
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Replace 0s with -inf (mask past tokens)
        return mask  # Shape: (seq_len, seq_len)                                

    def forward(self, x):
        x, _ = self.attn(x, x, x, attn_mask = self.mask)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, training_config : TrainingConfig, d_model : int, ffn_hidden : int):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(training_config.args.dropout_rate),
            nn.Linear(ffn_hidden, d_model)
        )

    def forward(self, x):
        x = self.ffn(x)
        return x
