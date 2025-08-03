"""
Attention utilities for language-guided spatial attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_language_guided_attention(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute language-guided spatial attention weights.
    
    Args:
        image_features: [batch_size, num_patches, feature_dim]
        text_features: [batch_size, feature_dim] 
        temperature: Attention temperature for softmax
        
    Returns:
        Attention weights: [batch_size, num_patches]
    """
    # Compute dot-product similarity
    # [batch_size, num_patches, feature_dim] @ [batch_size, feature_dim, 1]
    # -> [batch_size, num_patches, 1] -> [batch_size, num_patches]
    similarities = torch.bmm(
        image_features, 
        text_features.unsqueeze(-1)
    ).squeeze(-1)
    
    # Apply temperature and softmax
    attention_weights = F.softmax(similarities / temperature, dim=-1)
    
    return attention_weights


def apply_language_guided_attention(
    images: torch.Tensor,
    text_embedding: torch.Tensor,
    siglip_model,
    upsample_method: str = "bilinear"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply language-guided spatial attention to images as described in the paper.
    
    Args:
        images: [batch_size, num_views, channels, height, width]
        text_embedding: [batch_size, text_embed_dim]
        siglip_model: SigLIP model wrapper
        upsample_method: Method for upsampling attention maps
        
    Returns:
        attention_masked_features: Final masked visual features
        attention_weights: Attention weights for visualization
    """
    batch_size, num_views = images.shape[:2]
    device = images.device
    
    # Reshape for processing: [batch_size * num_views, C, H, W]
    images_flat = images.view(-1, *images.shape[2:])
    
    # Expand text embedding for all views
    text_expanded = text_embedding.unsqueeze(1).expand(-1, num_views, -1)
    text_flat = text_expanded.reshape(-1, text_embedding.shape[-1])
    
    # Compute attention weights for each view
    image_features_for_attn = siglip_model.encode_image_for_attention(images_flat)
    attention_weights = compute_language_guided_attention(
        image_features_for_attn, 
        text_flat
    )
    
    # Reshape attention weights to spatial dimensions
    # Assuming SigLIP uses 16x16 patches for 224x224 images -> 14x14 patches
    patch_size = 16
    img_size = images.shape[-1]  # Assume square images
    num_patches_per_side = img_size // patch_size
    
    attention_spatial = attention_weights.view(
        -1, num_patches_per_side, num_patches_per_side
    )
    
    # Upsample attention maps to original image size
    attention_upsampled = F.interpolate(
        attention_spatial.unsqueeze(1),  # Add channel dim
        size=(img_size, img_size),
        mode=upsample_method,
        align_corners=False
    ).squeeze(1)  # Remove channel dim
    
    # Apply attention mask to images
    masked_images = images_flat * attention_upsampled.unsqueeze(1)
    
    # Extract final visual features from masked images
    visual_features = siglip_model.encode_image_for_features(masked_images)
    
    # Reshape back to original batch structure
    visual_features = visual_features.view(batch_size, num_views, -1)
    attention_weights = attention_weights.view(batch_size, num_views, -1)
    
    # Concatenate features across views
    final_features = visual_features.flatten(1)  # [batch_size, num_views * feature_dim]
    
    return final_features, attention_weights


class LanguageGuidedAttention(nn.Module):
    """
    Module for language-guided spatial attention.
    """
    
    def __init__(
        self,
        siglip_model_name: str = "google/siglip-base-patch16-224",
        temperature: float = 1.0,
        freeze_siglip: bool = True
    ):
        super().__init__()
        
        from .siglip_utils import create_siglip_model
        
        self.siglip = create_siglip_model(siglip_model_name, freeze_siglip)
        self.temperature = temperature
        
    def forward(
        self, 
        images: torch.Tensor, 
        text_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply language-guided attention.
        
        Args:
            images: [batch_size, num_views, C, H, W]
            text_embedding: [batch_size, text_embed_dim]
            
        Returns:
            attended_features: [batch_size, num_views * feature_dim]
            attention_weights: [batch_size, num_views, num_patches]
        """
        return apply_language_guided_attention(
            images, 
            text_embedding, 
            self.siglip,
            upsample_method="bilinear"
        )