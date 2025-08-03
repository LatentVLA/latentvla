#!/usr/bin/env python3
"""
Training script for TA-LAM (Stage 1 of LatentVLA)

This script implements the training loop for the Temporal-Attentive Latent Action Model,
following the methodology described in the LatentVLA paper.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from latentvla.models import TA_LAM
from latentvla.utils.config_utils import (
    load_config, 
    validate_config, 
    setup_logging,
    create_experiment_dir
)
from latentvla.utils.data_utils import create_data_loaders


class TA_LAMTrainer:
    """
    Trainer for Temporal-Attentive Latent Action Model.
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.hardware.device)
        
        # Setup logging
        self.logger = setup_logging(cfg)
        self.logger.info("Initializing TA-LAM Trainer")
        
        # Create experiment directory
        self.exp_dir = create_experiment_dir(cfg)
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        
        # Initialize model
        self.model = self._build_model()
        self.logger.info(f"Model initialized with {self.model.get_parameters_count()/1e6:.1f}M parameters")
        
        # Setup optimization
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler() if cfg.hardware.mixed_precision else None
        
        # Data loaders
        self.train_loader, self.val_loader = create_data_loaders(cfg)
        self.logger.info(f"Data loaders created: {len(self.train_loader)} train, {len(self.val_loader)} val batches")
        
        # Initialize wandb
        if cfg.get("logging", {}).get("wandb", {}).get("project"):
            self._init_wandb()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _build_model(self) -> TA_LAM:
        """Build TA-LAM model"""
        model = TA_LAM(
            cfg=self.cfg.model,
            input_dim=tuple(self.cfg.model.input_dim),
            proprioception_dim=self.cfg.model.proprioception_dim,
            robot_action_dim=self.cfg.model.robot_action_dim,
            latent_action_dim=self.cfg.model.latent_action_dim
        )
        
        model = model.to(self.device)
        
        # Compile model if specified (PyTorch 2.0)
        if self.cfg.hardware.get("compile_model", False):
            model = torch.compile(model)
            self.logger.info("Model compiled with PyTorch 2.0")
        
        return model
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay
        )
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if self.cfg.training.scheduler.type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.num_epochs,
                eta_min=1e-6
            )
        elif self.cfg.training.scheduler.type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = None
            
        return scheduler
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb_cfg = self.cfg.logging.wandb
        wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            config=dict(self.cfg),
            dir=self.exp_dir
        )
        wandb.watch(self.model)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0, 
            "action_loss": 0.0
        }
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            self.global_step += 1
            
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Prepare inputs
            observations = batch["observations"]
            proprioception = batch["proprioception"]
            language_instruction = batch["language_instruction"]
            absolute_timesteps = batch["absolute_timesteps"]
            
            # Target data for losses
            next_observation = observations[:, -1, 0]  # Use first view as target
            robot_action = batch.get("robot_action")
            
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    output = self.model(
                        observations=observations[:, :-1],  # Remove last timestep
                        proprioception=proprioception[:, :-1],
                        language_instruction=language_instruction,
                        absolute_timesteps=absolute_timesteps[:, :-1],
                        next_observation=next_observation,
                        robot_action=robot_action
                    )
                    loss = output.total_loss
            else:
                output = self.model(
                    observations=observations[:, :-1],
                    proprioception=proprioception[:, :-1], 
                    language_instruction=language_instruction,
                    absolute_timesteps=absolute_timesteps[:, :-1],
                    next_observation=next_observation,
                    robot_action=robot_action
                )
                loss = output.total_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Accumulate losses
            epoch_losses["total_loss"] += loss.item()
            if output.reconstruction_loss is not None:
                epoch_losses["reconstruction_loss"] += output.reconstruction_loss.item()
            if output.action_loss is not None:
                epoch_losses["action_loss"] += output.action_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "lr": self.optimizer.param_groups[0]["lr"]
            })
            
            # Log to wandb
            if self.global_step % self.cfg.logging.log_interval == 0:
                log_dict = {
                    "train/total_loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train/epoch": self.epoch,
                    "train/step": self.global_step
                }
                if output.reconstruction_loss is not None:
                    log_dict["train/reconstruction_loss"] = output.reconstruction_loss.item()
                if output.action_loss is not None:
                    log_dict["train/action_loss"] = output.action_loss.item()
                
                if wandb.run is not None:
                    wandb.log(log_dict, step=self.global_step)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_losses = {
            "total_loss": 0.0,
            "reconstruction_loss": 0.0,
            "action_loss": 0.0
        }
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Prepare inputs
                observations = batch["observations"]
                proprioception = batch["proprioception"]
                language_instruction = batch["language_instruction"]
                absolute_timesteps = batch["absolute_timesteps"]
                
                # Target data
                next_observation = observations[:, -1, 0]
                robot_action = batch.get("robot_action")
                
                # Forward pass
                output = self.model(
                    observations=observations[:, :-1],
                    proprioception=proprioception[:, :-1],
                    language_instruction=language_instruction,
                    absolute_timesteps=absolute_timesteps[:, :-1],
                    next_observation=next_observation,
                    robot_action=robot_action
                )
                
                # Accumulate losses
                val_losses["total_loss"] += output.total_loss.item()
                if output.reconstruction_loss is not None:
                    val_losses["reconstruction_loss"] += output.reconstruction_loss.item()
                if output.action_loss is not None:
                    val_losses["action_loss"] += output.action_loss.item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
            
        return val_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": dict(self.cfg)
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        ckpt_path = os.path.join(self.exp_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, ckpt_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.exp_dir, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best checkpoint saved: {best_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training")
        
        for epoch in range(self.cfg.training.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            if epoch % self.cfg.logging.get("eval_interval", 1) == 0:
                val_losses = self.validate()
                
                # Check if best model
                is_best = val_losses["total_loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses["total_loss"]
                
                # Log validation results
                self.logger.info(f"Epoch {epoch}: Train Loss={train_losses['total_loss']:.4f}, "
                               f"Val Loss={val_losses['total_loss']:.4f}")
                
                if wandb.run is not None:
                    wandb.log({
                        "val/total_loss": val_losses["total_loss"],
                        "val/reconstruction_loss": val_losses["reconstruction_loss"],
                        "val/action_loss": val_losses["action_loss"],
                        "val/epoch": epoch
                    }, step=self.global_step)
                
                # Save checkpoint
                if epoch % self.cfg.logging.get("save_interval", 5) == 0:
                    self.save_checkpoint(is_best)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.logger.info("Training completed")


def main():
    parser = argparse.ArgumentParser(description="Train TA-LAM model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint for resuming training")
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    validate_config(cfg)
    
    # Create trainer and start training
    trainer = TA_LAMTrainer(cfg)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.epoch = checkpoint["epoch"]
        trainer.global_step = checkpoint["global_step"]
        trainer.best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed training from epoch {trainer.epoch}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()