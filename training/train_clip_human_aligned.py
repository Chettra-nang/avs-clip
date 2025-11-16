"""
CLIP fine-tuning for human alignment.

This approach focuses on aligning CLIP with human-readable driving instructions
while preserving the strong visual features of the pre-trained model.

Key differences from standard fine-tuning:
1. Lower learning rate (5e-6 instead of 1e-5)
2. Freeze vision encoder, only train text encoder
3. Use contrastive loss with temperature scaling
4. Early stopping based on validation similarity
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.logger_tb import TBLogger


class DrivingPairsDataset(Dataset):
    """Dataset of (image, text) pairs for CLIP human alignment."""
    
    def __init__(self, data_dirs: List[str]):
        self.pairs = []
        
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            jsonl_path = data_path / "dataset.jsonl"
            
            if not jsonl_path.exists():
                print(f"Warning: {jsonl_path} not found, skipping")
                continue
            
            with open(jsonl_path, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    image_path = data_path / record['image']
                    
                    # Use human-aligned instructions
                    for agent in record['agents']:
                        instruction = agent.get('instruction', '')
                        if image_path.exists() and instruction:
                            self.pairs.append((str(image_path), instruction))
        
        print(f"Loaded {len(self.pairs)} image-text pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]


def train_human_aligned_clip(
    data_dirs: List[str],
    model_id: str = "openai/clip-vit-base-patch32",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-6,
    freeze_vision: bool = True,
    val_split: float = 0.1,
    device: str = "cuda",
    log_dir: str = "runs/clip_human_aligned",
    save_dir: str = "models/clip_human_aligned"
):
    """
    Fine-tune CLIP for human alignment.
    
    Args:
        data_dirs: List of data directories
        model_id: Pre-trained CLIP model
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate (lower than standard fine-tuning!)
        freeze_vision: Whether to freeze vision encoder (recommended!)
        val_split: Validation split ratio
        device: Device to use
        log_dir: TensorBoard log directory
        save_dir: Model save directory
    """
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading {model_id}...")
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.to(device)
    
    # Freeze vision encoder (preserve visual features!)
    if freeze_vision:
        print("✓ Freezing vision encoder (preserving visual features)")
        for param in model.vision_model.parameters():
            param.requires_grad = False
        # Only train text encoder
        for param in model.text_model.parameters():
            param.requires_grad = True
    else:
        print("⚠️  Training both vision and text encoders")
    
    # Create dataset
    print("Creating dataset...")
    full_dataset = DrivingPairsDataset(data_dirs)
    
    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Collate function
    def collate_fn(batch):
        images = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        return images, texts
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn
    )
    
    # Optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    
    # Logger
    logger = TBLogger(log_dir, "clip_human_aligned")
    
    # Training loop
    global_step = 0
    best_val_similarity = 0.0
    
    print(f"\nStarting human alignment training for {epochs} epochs...")
    print(f"Learning rate: {lr} (lower to preserve features)")
    
    for epoch in range(epochs):
        # Training
        model.train()
        if freeze_vision:
            model.vision_model.eval()  # Keep vision in eval mode
        
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, texts) in enumerate(pbar):
            # Load images
            pil_images = [Image.open(img_path).convert('RGB') for img_path in images]
            
            # Process inputs
            inputs = processor(
                text=texts,
                images=pil_images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log every 100 steps
            if global_step % 100 == 0:
                logger.scalar("loss/train", loss.item(), global_step)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_similarities = []
        
        with torch.no_grad():
            for images, texts in tqdm(val_loader, desc="Validation"):
                pil_images = [Image.open(img_path).convert('RGB') for img_path in images]
                
                inputs = processor(
                    text=texts,
                    images=pil_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs, return_loss=True)
                val_loss += outputs.loss.item()
                
                # Compute similarities
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Normalize
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Similarity
                similarities = (image_embeds * text_embeds).sum(dim=-1)
                val_similarities.extend(similarities.cpu().numpy())
        
        # Compute metrics
        avg_train_loss = epoch_loss / num_batches
        avg_val_loss = val_loss / len(val_loader)
        avg_val_similarity = np.mean(val_similarities)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Similarity: {avg_val_similarity:.4f}")
        
        # Log epoch metrics
        logger.scalar("loss/train_epoch", avg_train_loss, epoch)
        logger.scalar("loss/val_epoch", avg_val_loss, epoch)
        logger.scalar("similarity/val", avg_val_similarity, epoch)
        
        # Save best model
        if avg_val_similarity > best_val_similarity:
            best_val_similarity = avg_val_similarity
            print(f"  ✓ New best validation similarity: {best_val_similarity:.4f}")
            
            # Save model
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"  ✓ Model saved to: {save_dir}")
    
    print(f"\nTraining complete!")
    print(f"Best validation similarity: {best_val_similarity:.4f}")
    
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP for human alignment")
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to data directories"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Pre-trained CLIP model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs (fewer than standard fine-tuning)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate (lower than standard fine-tuning!)"
    )
    parser.add_argument(
        "--freeze_vision",
        action="store_true",
        default=True,
        help="Freeze vision encoder (recommended!)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/clip_human_aligned",
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/clip_human_aligned",
        help="Model save directory"
    )
    
    args = parser.parse_args()
    
    train_human_aligned_clip(
        data_dirs=args.data_dirs,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_vision=args.freeze_vision,
        val_split=args.val_split,
        device=args.device,
        log_dir=args.log_dir,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
