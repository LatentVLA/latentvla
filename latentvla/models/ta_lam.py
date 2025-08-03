"""
Temporal-Attentive Latent Action Model (TA-LAM) for LatentVLA

This module implements the first stage of LatentVLA as described in the paper:
- Attention-Driven Inverse Dynamics Model (IDM)
- Latent Forward Dynamics Model (FDM)  
- Action Decoder
- Joint optimization objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from omegaconf import DictConfig

from .base import BaseModel, LatentVLAOutput, IDMOutput, log
from .utils.transformer_utils import (
    get_sinusoidal_encoding, 
    create_positional_encoding_matrix,
    make_mlp,
    TransformerEncoder
)
from .utils.attention_utils import LanguageGuidedAttention
from .utils.siglip_utils import create_siglip_model, preprocess_text


class TemporalAttentiveIDM(BaseModel):
    """
    Attention-Driven Inverse Dynamics Model with:
    1. Language-Guided Spatial Attention using SigLIP
    2. Temporal Disentanglement with absolute and relative encodings
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Tuple[int, int, int],  # (C, H, W) for images
        proprioception_dim: int = 14,
        latent_action_dim: int = 512,
        history_length: int = 4
    ):
        super().__init__(cfg, input_dim)
        self.name = "TemporalAttentiveIDM"
        
        log(f"Initializing {self.name}", "blue")
        
        # Configuration
        self.latent_action_dim = latent_action_dim
        self.history_length = history_length
        self.proprioception_dim = proprioception_dim
        self.d_model = cfg.get("d_model", 768)
        self.nhead = cfg.get("nhead", 8)
        self.num_layers = cfg.get("num_layers", 6)
        
        # Language-guided spatial attention
        self.language_attention = LanguageGuidedAttention(
            siglip_model_name=cfg.get("siglip_model", "google/siglip-base-patch16-224"),
            temperature=cfg.get("attention_temperature", 1.0),
            freeze_siglip=cfg.get("freeze_siglip", True)
        )
        
        # SigLIP for text encoding
        self.siglip = create_siglip_model(
            model_name=cfg.get("siglip_model", "google/siglip-base-patch16-224"),
            freeze_backbone=cfg.get("freeze_siglip", True)
        )
        
        # Compute feature dimensions
        self.visual_feature_dim = self.siglip.vision_embed_dim * cfg.get("num_views", 1)
        self.text_feature_dim = self.siglip.text_embed_dim
        
        # Temporal encoding dimensions
        self.d_gamma = cfg.get("d_gamma", 128)  # Absolute temporal encoding
        self.d_pe = cfg.get("d_pe", 128)       # Relative positional encoding
        
        # Input projection to transformer dimension
        context_input_dim = (
            self.visual_feature_dim + 
            self.text_feature_dim + 
            self.proprioception_dim + 
            self.d_gamma +
            self.d_pe
        )
        
        self.input_projection = nn.Linear(context_input_dim, self.d_model)
        
        # Transformer encoder for temporal processing
        self.transformer_encoder = TransformerEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=cfg.get("dim_feedforward", 2048),
            dropout=cfg.get("dropout", 0.1)
        )
        
        # Latent action head
        self.latent_head = nn.Linear(self.d_model, self.latent_action_dim)
        
        # Precompute positional encoding matrices
        max_episode_length = cfg.get("max_episode_length", 1000)
        max_context_length = history_length + 1
        
        self.register_buffer(
            "absolute_pe_matrix",
            create_positional_encoding_matrix(max_episode_length, self.d_gamma)
        )
        self.register_buffer(
            "relative_pe_matrix", 
            create_positional_encoding_matrix(max_context_length, self.d_pe)
        )
        
        log(f"IDM initialized with latent_action_dim={latent_action_dim}, "
            f"d_model={self.d_model}, visual_dim={self.visual_feature_dim}", "green")
    
    def _get_absolute_temporal_encoding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get absolute temporal encoding γ(t) for given timesteps.
        
        Args:
            timesteps: [batch_size, seq_len] absolute timesteps
            
        Returns:
            Absolute temporal encodings: [batch_size, seq_len, d_gamma]
        """
        batch_size, seq_len = timesteps.shape
        device = timesteps.device
        
        # Ensure timesteps are within bounds
        timesteps = torch.clamp(timesteps, 0, self.absolute_pe_matrix.shape[0] - 1)
        
        # Lookup encodings
        encodings = self.absolute_pe_matrix[timesteps.long()]  # [batch_size, seq_len, d_gamma]
        
        return encodings
    
    def _get_relative_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Get relative positional encoding PE(k) for sequence positions.
        
        Args:
            seq_len: Sequence length
            device: Device to place tensor on
            
        Returns:
            Relative positional encodings: [seq_len, d_pe]
        """
        positions = torch.arange(seq_len, device=device)
        return self.relative_pe_matrix[positions]  # [seq_len, d_pe]
    
    def forward(
        self,
        observations: torch.Tensor,      # [batch_size, history_len+1, num_views, C, H, W]
        proprioception: torch.Tensor,   # [batch_size, history_len+1, proprioception_dim]
        language_instruction: List[str], # List of text instructions
        absolute_timesteps: torch.Tensor # [batch_size, history_len+1] absolute episode timesteps
    ) -> IDMOutput:
        """
        Forward pass of the IDM.
        
        Args:
            observations: Multi-view image observations
            proprioception: Proprioceptive states  
            language_instruction: Text instructions
            absolute_timesteps: Absolute timesteps in episode
            
        Returns:
            IDMOutput with latent action and attention weights
        """
        batch_size, seq_len, num_views = observations.shape[:3]
        device = observations.device
        
        # Process text instruction
        text_tokens = preprocess_text(language_instruction, self.siglip.processor)
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
        text_embedding = self.siglip.encode_text(text_tokens["input_ids"])  # [batch_size, text_embed_dim]
        
        # Process each timestep
        context_sequence = []
        attention_weights_list = []
        
        for t in range(seq_len):
            # Extract current timestep data
            obs_t = observations[:, t]  # [batch_size, num_views, C, H, W]
            proprio_t = proprioception[:, t]  # [batch_size, proprioception_dim]
            timestep_t = absolute_timesteps[:, t]  # [batch_size]
            
            # Apply language-guided spatial attention
            visual_features, attn_weights = self.language_attention(obs_t, text_embedding)
            attention_weights_list.append(attn_weights)
            
            # Get temporal encodings
            absolute_encoding = self._get_absolute_temporal_encoding(
                timestep_t.unsqueeze(1)
            ).squeeze(1)  # [batch_size, d_gamma]
            
            relative_encoding = self._get_relative_positional_encoding(
                1, device
            ).squeeze(0)  # [d_pe]
            relative_encoding = relative_encoding.unsqueeze(0).expand(batch_size, -1)
            
            # Create context token for this timestep
            context_token = torch.cat([
                visual_features,      # [batch_size, visual_feature_dim]
                text_embedding,       # [batch_size, text_feature_dim] 
                proprio_t,           # [batch_size, proprioception_dim]
                absolute_encoding,   # [batch_size, d_gamma]
                relative_encoding    # [batch_size, d_pe]
            ], dim=-1)  # [batch_size, context_input_dim]
            
            context_sequence.append(context_token)
        
        # Stack context sequence and project to transformer dimension
        context_sequence = torch.stack(context_sequence, dim=1)  # [batch_size, seq_len, context_input_dim]
        context_sequence = self.input_projection(context_sequence)  # [batch_size, seq_len, d_model]
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(context_sequence)  # [batch_size, seq_len, d_model]
        
        # Extract latent action from final timestep
        final_hidden = transformer_output[:, -1]  # [batch_size, d_model]
        latent_action = self.latent_head(final_hidden)  # [batch_size, latent_action_dim]
        
        # Stack attention weights
        attention_weights = torch.stack(attention_weights_list, dim=1)  # [batch_size, seq_len, num_views, num_patches]
        
        return IDMOutput(
            latent_action=latent_action,
            attention_weights=attention_weights,
            temporal_encoding=context_sequence,
            transformer_hidden=final_hidden
        )


class LatentForwardDynamicsModel(BaseModel):
    """
    Forward Dynamics Model that predicts future observations from latent actions.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Tuple[int, int, int],  # (C, H, W)
        latent_action_dim: int = 512,
        transformer_hidden_dim: int = 768
    ):
        super().__init__(cfg, input_dim)
        self.name = "LatentForwardDynamicsModel"
        
        log(f"Initializing {self.name}", "blue")
        
        self.latent_action_dim = latent_action_dim
        self.transformer_hidden_dim = transformer_hidden_dim
        self.output_dim = input_dim[0] * input_dim[1] * input_dim[2]  # Flattened image
        
        # Decoder network
        hidden_dims = cfg.get("hidden_dims", [1024, 512, 256])
        input_decoder_dim = latent_action_dim + transformer_hidden_dim
        
        self.decoder = make_mlp(
            input_dim=input_decoder_dim,
            hidden_dims=hidden_dims,
            output_dim=self.output_dim,
            activation=cfg.get("activation", "relu"),
            dropout=cfg.get("dropout", 0.1),
            batch_norm=cfg.get("batch_norm", True)
        )
        
        log(f"FDM initialized with input_dim={input_decoder_dim}, "
            f"output_dim={self.output_dim}", "green")
    
    def forward(
        self,
        latent_action: torch.Tensor,     # [batch_size, latent_action_dim]
        transformer_hidden: torch.Tensor # [batch_size, transformer_hidden_dim]
    ) -> torch.Tensor:
        """
        Predict next observation from latent action and context.
        
        Args:
            latent_action: Latent action representation
            transformer_hidden: Hidden state from IDM transformer
            
        Returns:
            Predicted next observation: [batch_size, C, H, W]
        """
        # Concatenate inputs
        decoder_input = torch.cat([latent_action, transformer_hidden], dim=-1)
        
        # Decode to observation
        flat_obs = self.decoder(decoder_input)  # [batch_size, C*H*W]
        
        # Reshape to image format
        predicted_obs = flat_obs.view(-1, *self.input_dim)  # [batch_size, C, H, W]
        
        return predicted_obs


class ActionDecoder(BaseModel):
    """
    Action Decoder that maps latent actions to robot commands.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        latent_action_dim: int = 512,
        robot_action_dim: int = 14
    ):
        super().__init__(cfg, latent_action_dim)
        self.name = "ActionDecoder"
        
        log(f"Initializing {self.name}", "blue")
        
        self.latent_action_dim = latent_action_dim
        self.robot_action_dim = robot_action_dim
        
        # Simple MLP decoder
        hidden_dims = cfg.get("hidden_dims", [256, 128])
        
        self.decoder = make_mlp(
            input_dim=latent_action_dim,
            hidden_dims=hidden_dims,
            output_dim=robot_action_dim,
            activation=cfg.get("activation", "relu"),
            dropout=cfg.get("dropout", 0.0),
            batch_norm=cfg.get("batch_norm", False)
        )
        
        log(f"Action Decoder initialized: {latent_action_dim} -> {robot_action_dim}", "green")
    
    def forward(self, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Decode latent action to robot action.
        
        Args:
            latent_action: [batch_size, latent_action_dim]
            
        Returns:
            Robot action: [batch_size, robot_action_dim]
        """
        return self.decoder(latent_action)


class TA_LAM(BaseModel):
    """
    Temporal-Attentive Latent Action Model - Main class integrating IDM, FDM, and Action Decoder.
    Implements Stage 1 of LatentVLA framework.
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Tuple[int, int, int],  # (C, H, W)
        proprioception_dim: int = 14,
        robot_action_dim: int = 14,
        latent_action_dim: int = 512
    ):
        super().__init__(cfg, input_dim)
        self.name = "TA-LAM"
        
        log("=== Initializing Temporal-Attentive Latent Action Model ===", "purple")
        
        self.latent_action_dim = latent_action_dim
        self.robot_action_dim = robot_action_dim
        self.proprioception_dim = proprioception_dim
        
        # Initialize three core modules
        self.idm = TemporalAttentiveIDM(
            cfg.idm,
            input_dim=input_dim,
            proprioception_dim=proprioception_dim,
            latent_action_dim=latent_action_dim,
            history_length=cfg.get("history_length", 4)
        )
        
        self.fdm = LatentForwardDynamicsModel(
            cfg.fdm,
            input_dim=input_dim,
            latent_action_dim=latent_action_dim,
            transformer_hidden_dim=cfg.idm.get("d_model", 768)
        )
        
        self.action_decoder = ActionDecoder(
            cfg.action_decoder,
            latent_action_dim=latent_action_dim,
            robot_action_dim=robot_action_dim
        )
        
        # Loss weights
        self.lambda_action = cfg.get("lambda_action", 1.0)
        
        log(f"TA-LAM initialized with {self.get_parameters_count()/1e6:.1f}M parameters", "green")
    
    def forward(
        self,
        observations: torch.Tensor,        # [batch_size, history_len+1, num_views, C, H, W]
        proprioception: torch.Tensor,     # [batch_size, history_len+1, proprioception_dim]  
        language_instruction: List[str],   # List of text instructions
        absolute_timesteps: torch.Tensor, # [batch_size, history_len+1]
        next_observation: Optional[torch.Tensor] = None,  # [batch_size, C, H, W] for reconstruction loss
        robot_action: Optional[torch.Tensor] = None       # [batch_size, robot_action_dim] for action loss
    ) -> LatentVLAOutput:
        """
        Forward pass of TA-LAM.
        
        Args:
            observations: Multi-view image sequence
            proprioception: Proprioceptive state sequence
            language_instruction: Text instructions
            absolute_timesteps: Absolute timesteps in episode
            next_observation: Ground truth next observation (for training)
            robot_action: Ground truth robot action (for training)
            
        Returns:
            LatentVLAOutput with predictions and losses
        """
        # Forward through IDM
        idm_output = self.idm(
            observations, 
            proprioception, 
            language_instruction, 
            absolute_timesteps
        )
        
        latent_action = idm_output.latent_action
        
        # Forward through FDM for observation reconstruction
        predicted_obs = self.fdm(latent_action, idm_output.transformer_hidden)
        
        # Forward through Action Decoder for robot action prediction
        predicted_action = self.action_decoder(latent_action)
        
        # Compute losses if ground truth is provided
        reconstruction_loss = None
        action_loss = None
        total_loss = None
        
        if next_observation is not None:
            reconstruction_loss = F.mse_loss(predicted_obs, next_observation)
            
        if robot_action is not None:
            action_loss = F.mse_loss(predicted_action, robot_action)
            
        if reconstruction_loss is not None:
            total_loss = reconstruction_loss
            if action_loss is not None:
                total_loss = total_loss + self.lambda_action * action_loss
        elif action_loss is not None:
            total_loss = action_loss
        
        return LatentVLAOutput(
            latent_action=latent_action,
            reconstructed_obs=predicted_obs,
            decoded_action=predicted_action,
            attention_weights=idm_output.attention_weights,
            temporal_encoding=idm_output.temporal_encoding,
            reconstruction_loss=reconstruction_loss,
            action_loss=action_loss,
            total_loss=total_loss
        )