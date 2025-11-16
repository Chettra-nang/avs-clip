"""
DINOv2 vision encoder for highway multi-agent RL.

DINOv2 advantages over CLIP:
- Better semantic understanding (self-supervised on 142M images)
- Superior object detection (focuses on foreground objects like vehicles)
- No fine-tuning needed (works great out-of-the-box)
- Proven better for autonomous driving tasks
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np
from typing import Union


class DINOv2VisionEncoder(nn.Module):
    """
    DINOv2-based vision encoder for highway driving scenes.
    
    Args:
        model_name: DINOv2 model variant
            - "facebook/dinov2-small" (384-dim, fastest)
            - "facebook/dinov2-base" (768-dim, recommended) ⭐
            - "facebook/dinov2-large" (1024-dim, most accurate)
        freeze_backbone: Whether to freeze DINOv2 weights during training
        output_dim: Final embedding dimension (default: same as model)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        freeze_backbone: bool = True,
        output_dim: int = None
    ):
        super().__init__()
        
        print(f"Loading DINOv2 model: {model_name}")
        
        # Load DINOv2 model and processor
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Get DINOv2 output dimension
        self.dinov2_dim = self.model.config.hidden_size
        
        # Set output dimension
        if output_dim is None:
            output_dim = self.dinov2_dim
        self.output_dim = output_dim
        
        # Projection layer if output_dim differs from dinov2_dim
        if output_dim != self.dinov2_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.dinov2_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            )
        else:
            self.projection = nn.Identity()
        
        # Optionally freeze DINOv2 backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            print("✓ DINOv2 backbone frozen (fine-tuning disabled)")
        else:
            print("✓ DINOv2 backbone trainable (fine-tuning enabled)")
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv2.
        
        Args:
            images: (batch_size, 3, H, W) - RGB images from environment
            
        Returns:
            embeddings: (batch_size, output_dim) - Scene embeddings
        """
        # DINOv2 forward pass
        with torch.set_grad_enabled(self.model.training and not self._is_frozen()):
            outputs = self.model(pixel_values=images)
            
            # Use CLS token (first token) as scene representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, dinov2_dim)
        
        # Project to desired output dimension
        projected = self.projection(cls_embedding)  # (batch, output_dim)
        
        return projected
    
    def _is_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return not next(self.model.parameters()).requires_grad
    
    def process_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for DINOv2.
        
        Args:
            image: (H, W, 3) numpy array in [0, 255] range
            
        Returns:
            processed: (1, 3, 224, 224) tensor ready for DINOv2
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        # Process with DINOv2 processor (handles resizing, normalization)
        inputs = self.processor(images=image, return_tensors="pt")
        
        return inputs['pixel_values']
    
    def extract_features(self, images: list) -> torch.Tensor:
        """
        Extract DINOv2 features from a batch of images.
        
        Args:
            images: List of numpy arrays or PIL Images
            
        Returns:
            features: (batch_size, output_dim) embeddings
        """
        # Process all images
        processed_images = torch.cat([
            self.process_image(img) for img in images
        ], dim=0)
        
        # Move to model device
        processed_images = processed_images.to(next(self.parameters()).device)
        
        # Extract features
        with torch.no_grad():
            features = self.forward(processed_images)
        
        return features


def create_dinov2_encoder(variant: str = "base", **kwargs):
    """
    Factory function to create DINOv2 encoder.
    
    Args:
        variant: Model size ("small", "base", "large")
        **kwargs: Additional arguments for DINOv2VisionEncoder
        
    Returns:
        encoder: DINOv2VisionEncoder instance
    """
    model_map = {
        "small": "facebook/dinov2-small",   # 384-dim, 22M params
        "base": "facebook/dinov2-base",     # 768-dim, 86M params ⭐
        "large": "facebook/dinov2-large",   # 1024-dim, 300M params
    }
    
    if variant not in model_map:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(model_map.keys())}")
    
    return DINOv2VisionEncoder(model_name=model_map[variant], **kwargs)
