"""
Data utilities for LatentVLA
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from omegaconf import DictConfig
import cv2
from PIL import Image
import torchvision.transforms as transforms


class LatentVLADataset(Dataset):
    """
    Dataset for LatentVLA training supporting multiple data sources.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        dataset_config: Dict[str, Any],
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ):
        self.cfg = cfg
        self.dataset_config = dataset_config
        self.split = split
        self.transform = transform
        
        # Data properties
        self.history_length = cfg.model.history_length
        self.image_size = cfg.data.image_size
        self.use_reconstruction_loss = dataset_config.get("use_reconstruction_loss", True)
        self.use_action_loss = dataset_config.get("use_action_loss", False)
        
        # Load data paths
        self.data_paths = self._load_data_paths()
        
    def _load_data_paths(self) -> List[str]:
        """Load data file paths"""
        # This is a placeholder - implement based on your data structure
        data_dir = self.dataset_config["path"]
        # Return list of data file paths
        return []
        
    def _load_sequence(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single sequence of data.
        
        Returns:
            Dictionary containing:
            - observations: [seq_len, num_views, C, H, W]
            - proprioception: [seq_len, proprioception_dim]
            - language_instruction: str
            - absolute_timesteps: [seq_len]
            - robot_action: [robot_action_dim] (if available)
        """
        # Placeholder implementation
        seq_len = self.history_length + 1
        num_views = self.cfg.model.idm.num_views
        C, H, W = self.cfg.model.input_dim
        
        # Generate dummy data for demonstration
        observations = torch.randn(seq_len, num_views, C, H, W)
        proprioception = torch.randn(seq_len, self.cfg.model.proprioception_dim)
        language_instruction = "Pick up the red cube"
        absolute_timesteps = torch.arange(seq_len)
        
        data = {
            "observations": observations,
            "proprioception": proprioception, 
            "language_instruction": language_instruction,
            "absolute_timesteps": absolute_timesteps
        }
        
        # Add action if available
        if self.use_action_loss:
            robot_action = torch.randn(self.cfg.model.robot_action_dim)
            data["robot_action"] = robot_action
            
        return data
        
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.data_paths) if self.data_paths else 1000  # Dummy size
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data sample"""
        sequence_data = self._load_sequence(idx)
        
        # Apply transforms if specified
        if self.transform is not None:
            observations = sequence_data["observations"]
            seq_len, num_views = observations.shape[:2]
            
            # Apply transform to each image
            transformed_obs = []
            for t in range(seq_len):
                view_obs = []
                for v in range(num_views):
                    img = observations[t, v]  # [C, H, W]
                    if img.max() <= 1.0:  # Normalize to [0, 255] if needed
                        img = (img * 255).byte()
                    img_pil = transforms.ToPILImage()(img)
                    img_transformed = self.transform(img_pil)
                    view_obs.append(img_transformed)
                transformed_obs.append(torch.stack(view_obs))
            sequence_data["observations"] = torch.stack(transformed_obs)
        
        return sequence_data


def create_data_transforms(cfg: DictConfig, split: str = "train") -> transforms.Compose:
    """
    Create data transforms for training/validation.
    
    Args:
        cfg: Configuration object
        split: Data split ('train' or 'val')
        
    Returns:
        Composed transforms
    """
    image_size = cfg.data.image_size
    normalize = transforms.Normalize(
        mean=cfg.data.normalization.mean,
        std=cfg.data.normalization.std
    )
    
    if split == "train" and cfg.data.get("image_augmentation", False):
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1, 
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            normalize
        ]
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ]
    
    return transforms.Compose(transform_list)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for LatentVLA data.
    
    Args:
        batch: List of data samples
        
    Returns:
        Batched data dictionary
    """
    # Stack observations, proprioception, and timesteps
    observations = torch.stack([item["observations"] for item in batch])
    proprioception = torch.stack([item["proprioception"] for item in batch])
    absolute_timesteps = torch.stack([item["absolute_timesteps"] for item in batch])
    
    # Collect language instructions
    language_instructions = [item["language_instruction"] for item in batch]
    
    batched_data = {
        "observations": observations,
        "proprioception": proprioception,
        "language_instruction": language_instructions,
        "absolute_timesteps": absolute_timesteps
    }
    
    # Add robot actions if available
    if "robot_action" in batch[0]:
        robot_actions = torch.stack([item["robot_action"] for item in batch])
        batched_data["robot_action"] = robot_actions
    
    return batched_data


def create_data_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create transforms
    train_transform = create_data_transforms(cfg, "train")
    val_transform = create_data_transforms(cfg, "val")
    
    # Create datasets
    train_datasets = []
    val_datasets = []
    
    for dataset_config in cfg.data.pretrain_datasets:
        # Training dataset
        train_dataset = LatentVLADataset(
            cfg=cfg,
            dataset_config=dataset_config,
            split="train",
            transform=train_transform
        )
        train_datasets.append(train_dataset)
        
        # Validation dataset  
        val_dataset = LatentVLADataset(
            cfg=cfg,
            dataset_config=dataset_config,
            split="val",
            transform=val_transform
        )
        val_datasets.append(val_dataset)
    
    # Combine datasets if multiple sources
    if len(train_datasets) > 1:
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader


def preprocess_sequence(
    observations: torch.Tensor,
    proprioception: torch.Tensor, 
    history_length: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocess sequence data for training.
    
    Args:
        observations: [seq_len, num_views, C, H, W]
        proprioception: [seq_len, proprioception_dim]
        history_length: Length of history window
        
    Returns:
        Tuple of (input_obs, input_proprio, target_obs)
    """
    seq_len = observations.shape[0]
    
    if seq_len < history_length + 2:
        raise ValueError(f"Sequence length {seq_len} too short for history {history_length}")
    
    # Input sequence (history + current)
    input_obs = observations[:history_length+1]  # [history_len+1, num_views, C, H, W]
    input_proprio = proprioception[:history_length+1]  # [history_len+1, proprioception_dim]
    
    # Target is next observation
    target_obs = observations[history_length+1]  # [num_views, C, H, W]
    
    return input_obs, input_proprio, target_obs