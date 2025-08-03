"""
SigLIP utilities for language-guided spatial attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModel, AutoProcessor


class SigLIPWrapper(nn.Module):
    """
    Wrapper for SigLIP model to handle vision-language tasks.
    Provides separate access to text encoder, image encoder, and vision encoder.
    """
    
    def __init__(
        self, 
        model_name: str = "google/siglip-base-patch16-224",
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # Load pre-trained SigLIP model
        self.siglip_model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Extract components
        self.text_encoder = self.siglip_model.text_model
        self.vision_model = self.siglip_model.vision_model
        
        # Feature dimensions
        self.text_embed_dim = self.text_encoder.config.hidden_size
        self.vision_embed_dim = self.vision_model.config.hidden_size
        
        # Freeze backbone if specified
        if freeze_backbone:
            self.freeze_backbone()
            
    def freeze_backbone(self):
        """Freeze the pre-trained SigLIP backbone"""
        for param in self.siglip_model.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze the pre-trained SigLIP backbone"""
        for param in self.siglip_model.parameters():
            param.requires_grad = True
    
    def encode_text(self, text_input: torch.Tensor) -> torch.Tensor:
        """
        Encode text using SigLIP text encoder.
        
        Args:
            text_input: Tokenized text input
            
        Returns:
            Text embeddings: [batch_size, text_embed_dim]
        """
        outputs = self.text_encoder(text_input)
        # Use pooled output (CLS token equivalent)
        return outputs.pooler_output
        
    def encode_image_for_attention(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images for attention computation.
        
        Args:
            images: [batch_size, channels, height, width]
            
        Returns:
            Image features for attention: [batch_size, num_patches, vision_embed_dim]
        """
        outputs = self.vision_model(images)
        # Return patch embeddings (not pooled)
        return outputs.last_hidden_state
        
    def encode_image_for_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images for final feature extraction.
        
        Args:
            images: [batch_size, channels, height, width]
            
        Returns:
            Image features: [batch_size, vision_embed_dim]
        """
        outputs = self.vision_model(images)
        # Use pooled output
        return outputs.pooler_output


def create_siglip_model(
    model_name: str = "google/siglip-base-patch16-224",
    freeze_backbone: bool = True
) -> SigLIPWrapper:
    """
    Factory function to create SigLIP model wrapper.
    
    Args:
        model_name: HuggingFace model name
        freeze_backbone: Whether to freeze the backbone
        
    Returns:
        SigLIP model wrapper
    """
    return SigLIPWrapper(model_name, freeze_backbone)


def preprocess_text(texts: list, processor) -> torch.Tensor:
    """
    Preprocess text inputs for SigLIP.
    
    Args:
        texts: List of text strings
        processor: SigLIP processor
        
    Returns:
        Tokenized text tensor
    """
    return processor(text=texts, return_tensors="pt", padding=True, truncation=True)


def preprocess_images(images: torch.Tensor, processor) -> torch.Tensor:
    """
    Preprocess image inputs for SigLIP.
    
    Args:
        images: Image tensor [batch_size, channels, height, width]
        processor: SigLIP processor
        
    Returns:
        Preprocessed image tensor
    """
    # Convert to PIL images if needed, then back to tensor
    # This ensures proper preprocessing
    if images.dim() == 4:
        batch_size = images.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            # Convert single image
            img = images[i]  # [C, H, W]
            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            
            processed_images.append(img)
        
        return torch.stack(processed_images)
    
    return images