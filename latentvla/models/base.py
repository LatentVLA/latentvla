"""
Base model classes for LatentVLA
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple
from omegaconf import DictConfig


@dataclass
class LatentVLAOutput:
    """Output dataclass for LatentVLA models"""
    latent_action: torch.Tensor
    reconstructed_obs: Optional[torch.Tensor] = None
    decoded_action: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    temporal_encoding: Optional[torch.Tensor] = None
    
    # Loss components for debugging
    reconstruction_loss: Optional[torch.Tensor] = None
    action_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None


@dataclass
class IDMOutput:
    """Output dataclass for IDM"""
    latent_action: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    temporal_encoding: Optional[torch.Tensor] = None
    transformer_hidden: Optional[torch.Tensor] = None


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all LatentVLA models.
    Follows the design pattern from CLAM.
    """
    
    def __init__(self, cfg: DictConfig, input_dim: Union[int, Tuple]):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.name = "BaseModel"
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass implementation"""
        pass
        
    def get_parameters_count(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def freeze_parameters(self, module_names: Optional[list] = None):
        """Freeze specified modules or all parameters"""
        if module_names is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name in module_names:
                if hasattr(self, name):
                    module = getattr(self, name)
                    for param in module.parameters():
                        param.requires_grad = False
                        
    def unfreeze_parameters(self, module_names: Optional[list] = None):
        """Unfreeze specified modules or all parameters"""
        if module_names is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name in module_names:
                if hasattr(self, name):
                    module = getattr(self, name)
                    for param in module.parameters():
                        param.requires_grad = True


def log(message: str, color: str = "blue"):
    """Simple logging function with color support"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m", 
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    
    color_code = colors.get(color, colors["blue"])
    reset_code = colors["reset"]
    print(f"{color_code}{message}{reset_code}")