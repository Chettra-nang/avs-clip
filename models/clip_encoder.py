"""
CLIP encoder wrapper for vision-language embeddings.
"""
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
from typing import List, Union


class CLIPBackbone(nn.Module):
    """
    Wrapper around HuggingFace CLIP model for extracting image and text embeddings.
    
    Args:
        model_id: HuggingFace model identifier (default: openai/clip-vit-base-patch32)
        trainable: If True, enables gradients for fine-tuning. If False, freezes weights.
    """
    
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", trainable: bool = False):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        # Freeze or unfreeze weights based on trainable parameter
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
    
    @torch.no_grad()
    def img_embed(self, pil_or_np: Union[Image.Image, np.ndarray], device: torch.device) -> torch.Tensor:
        """
        Extract image embedding from PIL Image or numpy array.
        
        Args:
            pil_or_np: PIL Image or numpy array [H, W, 3]
            device: torch device to place tensor on
            
        Returns:
            L2-normalized image embedding [1, 512]
        """
        # Convert numpy to PIL if needed
        if isinstance(pil_or_np, np.ndarray):
            pil_or_np = Image.fromarray(pil_or_np.astype(np.uint8))
        
        # Process image
        inputs = self.processor(images=pil_or_np, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract image features
        image_features = self.model.get_image_features(**inputs)
        
        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features  # [1, 512]
    
    def txt_embed(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Extract text embeddings from list of strings.
        
        Args:
            texts: List of text strings
            device: torch device to place tensor on
            
        Returns:
            L2-normalized text embeddings [N, 512]
        """
        # Process text
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract text features
        text_features = self.model.get_text_features(**inputs)
        
        # L2 normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features  # [N, 512]
