"""
Transformer utilities for LatentVLA
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def get_sinusoidal_encoding(position: int, d_model: int) -> torch.Tensor:
    """
    Generate sinusoidal positional encoding as described in the paper.
    
    Args:
        position: Position index
        d_model: Embedding dimension
        
    Returns:
        Positional encoding tensor of shape [d_model]
    """
    encoding = torch.zeros(d_model)
    
    for j in range(d_model):
        if j % 2 == 0:
            encoding[j] = math.sin(position / (10000 ** (j / d_model)))
        else:
            encoding[j] = math.cos(position / (10000 ** ((j - 1) / d_model)))
            
    return encoding


def create_positional_encoding_matrix(max_len: int, d_model: int) -> torch.Tensor:
    """
    Create a matrix of sinusoidal positional encodings.
    
    Args:
        max_len: Maximum sequence length
        d_model: Embedding dimension
        
    Returns:
        Positional encoding matrix of shape [max_len, d_model]
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


def make_mlp(
    input_dim: int, 
    hidden_dims: list, 
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False
) -> nn.Module:
    """
    Create a multi-layer perceptron.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function name
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
        
    Returns:
        MLP module
    """
    layers = []
    prev_dim = input_dim
    
    activation_fn = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh
    }.get(activation, nn.ReLU)
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    
    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))
    
    return nn.Sequential(*layers)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing temporal sequences.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.d_model = d_model
        
    def forward(
        self, 
        src: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, d_model]
            mask: [seq_len, seq_len]
            src_key_padding_mask: [batch_size, seq_len]
            
        Returns:
            Encoded sequence: [batch_size, seq_len, d_model]
        """
        return self.transformer(
            src, 
            mask=mask, 
            src_key_padding_mask=src_key_padding_mask
        )