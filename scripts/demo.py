#!/usr/bin/env python3
"""
LatentVLA Demo Script

Simple examples demonstrating how to use TA-LAM model for inference and training.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from omegaconf import OmegaConf
from latentvla.models import TA_LAM
from latentvla.utils.config_utils import load_config


def create_demo_config():
    """Create demo configuration"""
    cfg = OmegaConf.create({
        "model": {
            "name": "TA-LAM",
            "latent_action_dim": 512,
            "history_length": 4,
            "input_dim": [3, 224, 224],
            "proprioception_dim": 14,
            "robot_action_dim": 14,
            "lambda_action": 1.0,
            
            "idm": {
                "d_model": 768,
                "nhead": 8,
                "num_layers": 6,
                "dim_feedforward": 2048,
                "dropout": 0.1,
                "siglip_model": "google/siglip-base-patch16-224",
                "freeze_siglip": True,
                "attention_temperature": 1.0,
                "num_views": 2,
                "d_gamma": 128,
                "d_pe": 128,
                "max_episode_length": 1000
            },
            
            "fdm": {
                "hidden_dims": [1024, 512, 256],
                "activation": "relu",
                "dropout": 0.1,
                "batch_norm": True
            },
            
            "action_decoder": {
                "hidden_dims": [256, 128],
                "activation": "relu",
                "dropout": 0.0,
                "batch_norm": False
            }
        }
    })
    
    return cfg


def create_dummy_data(batch_size=2, history_length=4, num_views=2):
    """Create dummy data for demonstration"""
    # Observation sequence: [batch_size, history_length+1, num_views, C, H, W]
    observations = torch.randn(batch_size, history_length+1, num_views, 3, 224, 224)
    
    # Proprioceptive states: [batch_size, history_length+1, proprioception_dim]
    proprioception = torch.randn(batch_size, history_length+1, 14)
    
    # Language instructions
    language_instructions = [
        "Pick up the red cube and place it in the box",
        "Grasp the blue bottle with left hand"
    ]
    
    # Absolute timesteps: [batch_size, history_length+1]
    absolute_timesteps = torch.arange(history_length+1).unsqueeze(0).expand(batch_size, -1)
    
    # Target observation (for reconstruction loss): [batch_size, C, H, W]
    next_observation = torch.randn(batch_size, 3, 224, 224)
    
    # Robot action (for action loss): [batch_size, robot_action_dim]
    robot_action = torch.randn(batch_size, 14)
    
    return {
        "observations": observations,
        "proprioception": proprioception,
        "language_instruction": language_instructions,
        "absolute_timesteps": absolute_timesteps,
        "next_observation": next_observation,
        "robot_action": robot_action
    }


def demo_model_forward():
    """Demonstrate model forward pass"""
    print("=== LatentVLA TA-LAM Demo ===\n")
    
    # Create configuration
    cfg = create_demo_config()
    print("✓ Configuration created")
    
    # Create model
    model = TA_LAM(
        cfg=cfg.model,
        input_dim=tuple(cfg.model.input_dim),
        proprioception_dim=cfg.model.proprioception_dim,
        robot_action_dim=cfg.model.robot_action_dim,
        latent_action_dim=cfg.model.latent_action_dim
    )
    
    print(f"✓ Model created with {model.get_parameters_count()/1e6:.1f}M parameters")
    
    # Create dummy data
    data = create_dummy_data()
    print("✓ Dummy data created")
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        try:
            output = model(
                observations=data["observations"][:, :-1],  # Remove last timestep
                proprioception=data["proprioception"][:, :-1],
                language_instruction=data["language_instruction"],
                absolute_timesteps=data["absolute_timesteps"][:, :-1],
                next_observation=data["next_observation"],
                robot_action=data["robot_action"]
            )
            
            print("\n=== Output Results ===")
            print(f"Latent action shape: {output.latent_action.shape}")
            print(f"Reconstructed obs shape: {output.reconstructed_obs.shape}")
            print(f"Decoded action shape: {output.decoded_action.shape}")
            print(f"Attention weights shape: {output.attention_weights.shape}")
            
            if output.reconstruction_loss is not None:
                print(f"Reconstruction loss: {output.reconstruction_loss.item():.4f}")
            if output.action_loss is not None:
                print(f"Action loss: {output.action_loss.item():.4f}")
            if output.total_loss is not None:
                print(f"Total loss: {output.total_loss.item():.4f}")
                
            print("\n✓ Forward pass completed successfully!")
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            raise


def demo_attention_visualization():
    """Demonstrate attention weights visualization"""
    print("\n=== Attention Visualization Demo ===")
    
    # Create simple attention weights (dummy data)
    batch_size, seq_len, num_views = 2, 5, 2
    num_patches = 14 * 14  # Assumed SigLIP patch count
    
    attention_weights = torch.softmax(torch.randn(batch_size, seq_len, num_views, num_patches), dim=-1)
    
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights sum: {attention_weights.sum(dim=-1)[0, 0, 0]:.4f} (should be close to 1.0)")
    
    # Calculate average attention
    avg_attention = attention_weights.mean(dim=(0, 1))  # [num_views, num_patches]
    print(f"Average attention max position: {avg_attention.argmax(dim=-1)}")


def demo_training_step():
    """Demonstrate training step"""
    print("\n=== Training Step Demo ===")
    
    # Create configuration and model
    cfg = create_demo_config()
    model = TA_LAM(
        cfg=cfg.model,
        input_dim=tuple(cfg.model.input_dim),
        proprioception_dim=cfg.model.proprioception_dim,
        robot_action_dim=cfg.model.robot_action_dim,
        latent_action_dim=cfg.model.latent_action_dim
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Set model to training mode
    model.train()
    
    # Create training data
    data = create_dummy_data()
    
    # Training step
    optimizer.zero_grad()
    
    output = model(
        observations=data["observations"][:, :-1],
        proprioception=data["proprioception"][:, :-1],
        language_instruction=data["language_instruction"],
        absolute_timesteps=data["absolute_timesteps"][:, :-1],
        next_observation=data["next_observation"],
        robot_action=data["robot_action"]
    )
    
    loss = output.total_loss
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step completed, loss: {loss.item():.4f}")
    print(f"✓ Gradient update successful")


def main():
    """Main demo function"""
    try:
        # Demonstrate model forward pass
        demo_model_forward()
        
        # Demonstrate attention visualization
        demo_attention_visualization()
        
        # Demonstrate training step
        demo_training_step()
        
        print("\n🎉 All demos completed successfully!")
        print("\n📋 Next steps:")
        print("1. Prepare your dataset")
        print("2. Modify configuration file configs/ta_lam_config.yaml")
        print("3. Run training script: python scripts/train_ta_lam.py --config configs/ta_lam_config.yaml")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()