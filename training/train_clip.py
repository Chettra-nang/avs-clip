"""
CLIP fine-tuning on driving image-instruction pairs.
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.logger_tb import TBLogger


class DrivingPairs(Dataset):
    """
    Dataset of (image, text) pairs for CLIP fine-tuning.
    
    Loads data from multiple scenario directories, where each directory contains:
    - dataset.jsonl: JSON lines file with frame metadata
    - images/: Directory with PNG images
    
    Each agent in each frame creates one (image, text) pair where:
    - image: 224x224 RGB driving scene
    - text: "{action_name}: {scene_dict}"
    """
    
    def __init__(self, data_dirs: List[str]):
        """
        Args:
            data_dirs: List of paths to scenario data directories
        """
        self.pairs = []
        
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            jsonl_path = data_path / "dataset.jsonl"
            
            if not jsonl_path.exists():
                print(f"Warning: {jsonl_path} not found, skipping")
                continue
            
            # Parse dataset.jsonl
            with open(jsonl_path, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    image_path = data_path / record['image']
                    
                    # Create one pair per agent
                    for agent in record['agents']:
                        text = self._format_text(agent)
                        self.pairs.append((str(image_path), text))
        
        print(f"Loaded {len(self.pairs)} image-text pairs from {len(data_dirs)} scenarios")
    
    def _format_text(self, agent: dict) -> str:
        """
        Format agent data as text instruction with role-aware prefixes.
        
        For heterogeneous mode:
        - Ambulance: "AMBULANCE {action_name}: {scene_dict}"
        - Normal: "YIELD {action_name}: {scene_dict}"
        
        For homogeneous mode (backward compatibility):
        - "{action_name}: {scene_dict}"
        
        Examples:
            Ambulance: "AMBULANCE FASTER: {'front_dist': 35.2, 'front_ttc': 8.1, 'risk': 0.25}"
            Normal (no ambulance): "YIELD IDLE: {'front_dist': 15.2, 'front_ttc': 3.1, 'risk': 0.68}"
            Normal (ambulance nearby): "YIELD LANE_RIGHT: {'front_dist': 20.0, 'ambulance_dist': 25.5, 'risk': 0.45}"
            Homogeneous: "LANE_LEFT: {'front_dist': 15.2, 'front_ttc': 3.1, 'risk': 0.68}"
        """
        action_name = agent['action_name']
        scene = agent['scene']
        
        # Check if agent has role field (heterogeneous mode)
        if 'role' in agent:
            agent_role = agent['role']
            
            # Format with role prefix
            if agent_role == 'ambulance':
                return f"AMBULANCE {action_name}: {scene}"
            else:  # normal agent
                return f"YIELD {action_name}: {scene}"
        else:
            # Backward compatibility: homogeneous mode without role prefix
            return f"{action_name}: {scene}"
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        """
        Returns:
            image: PIL Image
            text: Text instruction string
        """
        image_path, text = self.pairs[idx]
        image = Image.open(image_path).convert('RGB')
        return image, text



def train_clip(
    data_dirs: List[str],
    model_id: str = "openai/clip-vit-base-patch32",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-5,
    num_workers: int = 4,
    log_dir: str = "runs/clip",
    save_dir: str = "models/clip"
):
    """
    Fine-tune CLIP on driving image-instruction pairs.
    
    Args:
        data_dirs: List of paths to scenario data directories
        model_id: HuggingFace CLIP model identifier
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate for AdamW optimizer
        num_workers: Number of DataLoader workers
        log_dir: TensorBoard log directory
        save_dir: Directory to save fine-tuned model
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and processor
    print(f"Loading {model_id}...")
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    # Modify processor to resize instead of center crop (preserves all agents in wide frame)
    processor.image_processor.do_center_crop = False
    processor.image_processor.do_resize = True
    processor.image_processor.size = {"height": 224, "width": 224}
    
    model = model.to(device)
    model.train()
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = DrivingPairs(data_dirs)
    
    # Custom collate function to handle PIL images
    def collate_fn(batch):
        images = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        return images, texts
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Setup logger
    logger = TBLogger(log_dir, "clip_finetune")
    
    # Training loop
    global_step = 0
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, texts) in enumerate(pbar):
            # Process inputs
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass with contrastive loss
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
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to TensorBoard every 100 steps
            if global_step % 100 == 0:
                logger.scalar("loss/train", loss.item(), global_step)
                
                # Log sample images (first 4 in batch)
                for i in range(min(4, len(images))):
                    img_array = torch.tensor(images[i]).permute(2, 0, 1).numpy()
                    logger.image(f"samples/image_{i}", img_array, global_step, dataformats='CHW')
                
                # Log sample texts
                sample_texts = "\n".join([f"{i}: {texts[i]}" for i in range(min(4, len(texts)))])
                logger.text("samples/texts", sample_texts, global_step)
        
        # Log epoch metrics
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        logger.scalar("loss/epoch", avg_loss, epoch)
    
    # Save fine-tuned model
    print(f"\nSaving fine-tuned model to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print("Training complete!")
    
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on driving data")
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to scenario data directories (e.g., data/highway data/merge data/intersection)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model identifier"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/clip",
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/clip",
        help="Directory to save fine-tuned model"
    )
    
    args = parser.parse_args()
    
    train_clip(
        data_dirs=args.data_dirs,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        log_dir=args.log_dir,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
