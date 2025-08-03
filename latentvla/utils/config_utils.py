"""
Configuration utilities for LatentVLA
"""

import os
import yaml
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, Optional
import logging


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file using OmegaConf.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML file
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to OmegaConf
    cfg = OmegaConf.create(config_dict)
    
    # Set structured config mode for better type checking
    OmegaConf.set_struct(cfg, True)
    
    return cfg


def save_config(cfg: DictConfig, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        cfg: OmegaConf configuration object
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        OmegaConf.save(cfg, f)


def merge_configs(base_cfg: DictConfig, override_cfg: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.
    
    Args:
        base_cfg: Base configuration
        override_cfg: Override configuration
        
    Returns:
        Merged configuration
    """
    return OmegaConf.merge(base_cfg, override_cfg)


def validate_config(cfg: DictConfig) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        cfg: Configuration to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    required_fields = [
        "model.latent_action_dim",
        "model.input_dim", 
        "model.proprioception_dim",
        "model.robot_action_dim",
        "training.learning_rate",
        "training.batch_size"
    ]
    
    for field in required_fields:
        if not OmegaConf.select(cfg, field):
            raise ValueError(f"Required configuration field missing: {field}")
    
    # Validate dimensions
    if cfg.model.latent_action_dim <= 0:
        raise ValueError("latent_action_dim must be positive")
        
    if len(cfg.model.input_dim) != 3:
        raise ValueError("input_dim must be [C, H, W] format")
        
    if cfg.training.batch_size <= 0:
        raise ValueError("batch_size must be positive")
        
    if cfg.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    return True


def setup_logging(cfg: DictConfig) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('latentvla.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('LatentVLA')
    logger.info("Logging setup completed")
    
    return logger


def create_experiment_dir(cfg: DictConfig, base_dir: str = "experiments") -> str:
    """
    Create experiment directory with timestamp.
    
    Args:
        cfg: Configuration object
        base_dir: Base directory for experiments
        
    Returns:
        Path to created experiment directory
    """
    import time
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = cfg.model.name.lower()
    exp_name = f"{model_name}_{timestamp}"
    
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration in experiment directory
    save_config(cfg, os.path.join(exp_dir, "config.yaml"))
    
    return exp_dir