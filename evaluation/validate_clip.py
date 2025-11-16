"""
Validate fine-tuned CLIP model against pre-trained baseline.

This script compares:
1. Pre-trained CLIP (openai/clip-vit-base-patch32)
2. Fine-tuned CLIP (your trained model)

Metrics:
- Image-text similarity scores
- Retrieval accuracy (image → text, text → image)
- Semantic understanding of driving scenarios
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_clip_model(model_path: str, device: str = 'cuda'):
    """Load CLIP model (pre-trained or fine-tuned)."""
    print(f"Loading CLIP from: {model_path}")
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor


def load_test_data(data_dir: str, max_samples: int = 100):
    """Load test samples from dataset."""
    data_path = Path(data_dir)
    jsonl_path = data_path / "dataset.jsonl"
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"dataset.jsonl not found in {data_dir}")
    
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            image_path = data_path / record['image']
            
            # Get first agent's instruction
            if record['agents']:
                agent = record['agents'][0]
                text = agent.get('instruction', '')
                
                if image_path.exists() and text:
                    samples.append({
                        'image_path': str(image_path),
                        'text': text,
                        'action': agent.get('action_name', ''),
                        'role': agent.get('role', 'unknown')
                    })
            
            if len(samples) >= max_samples:
                break
    
    print(f"Loaded {len(samples)} test samples from {data_dir}")
    return samples


def compute_similarity(model, processor, image_path: str, text: str, device: str):
    """Compute CLIP similarity between image and text."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Process inputs
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get features
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (image_embeds @ text_embeds.T).item()
    
    return similarity


def evaluate_retrieval(model, processor, samples: List[Dict], device: str):
    """
    Evaluate retrieval accuracy.
    
    For each image, rank all texts by similarity.
    Compute Recall@K (how often correct text is in top K).
    """
    print("\nEvaluating retrieval accuracy...")
    
    # Extract all images and texts
    images = [Image.open(s['image_path']).convert('RGB') for s in samples]
    texts = [s['text'] for s in samples]
    
    # Encode all images
    print("Encoding images...")
    image_embeds_list = []
    for img in tqdm(images, desc="Images"):
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embeds = model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            image_embeds_list.append(embeds)
    
    image_embeds = torch.cat(image_embeds_list, dim=0)  # [N, 512]
    
    # Encode all texts
    print("Encoding texts...")
    text_embeds_list = []
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Texts"):
        batch_texts = texts[i:i+batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            embeds = model.get_text_features(**inputs)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            text_embeds_list.append(embeds)
    
    text_embeds = torch.cat(text_embeds_list, dim=0)  # [N, 512]
    
    # Compute similarity matrix
    similarity_matrix = image_embeds @ text_embeds.T  # [N, N]
    
    # Image-to-Text Retrieval
    print("\nImage-to-Text Retrieval:")
    i2t_ranks = []
    for i in range(len(samples)):
        # Get similarities for this image
        sims = similarity_matrix[i]
        # Rank texts by similarity
        ranking = torch.argsort(sims, descending=True)
        # Find position of correct text
        correct_idx = i
        rank = (ranking == correct_idx).nonzero(as_tuple=True)[0].item()
        i2t_ranks.append(rank)
    
    i2t_recall_1 = np.mean([r == 0 for r in i2t_ranks]) * 100
    i2t_recall_5 = np.mean([r < 5 for r in i2t_ranks]) * 100
    i2t_recall_10 = np.mean([r < 10 for r in i2t_ranks]) * 100
    
    print(f"  Recall@1:  {i2t_recall_1:.2f}%")
    print(f"  Recall@5:  {i2t_recall_5:.2f}%")
    print(f"  Recall@10: {i2t_recall_10:.2f}%")
    
    # Text-to-Image Retrieval
    print("\nText-to-Image Retrieval:")
    t2i_ranks = []
    for i in range(len(samples)):
        # Get similarities for this text
        sims = similarity_matrix[:, i]
        # Rank images by similarity
        ranking = torch.argsort(sims, descending=True)
        # Find position of correct image
        correct_idx = i
        rank = (ranking == correct_idx).nonzero(as_tuple=True)[0].item()
        t2i_ranks.append(rank)
    
    t2i_recall_1 = np.mean([r == 0 for r in t2i_ranks]) * 100
    t2i_recall_5 = np.mean([r < 5 for r in t2i_ranks]) * 100
    t2i_recall_10 = np.mean([r < 10 for r in t2i_ranks]) * 100
    
    print(f"  Recall@1:  {t2i_recall_1:.2f}%")
    print(f"  Recall@5:  {t2i_recall_5:.2f}%")
    print(f"  Recall@10: {t2i_recall_10:.2f}%")
    
    return {
        'i2t_recall_1': i2t_recall_1,
        'i2t_recall_5': i2t_recall_5,
        'i2t_recall_10': i2t_recall_10,
        't2i_recall_1': t2i_recall_1,
        't2i_recall_5': t2i_recall_5,
        't2i_recall_10': t2i_recall_10,
    }


def evaluate_similarity(model, processor, samples: List[Dict], device: str):
    """Evaluate average similarity scores."""
    print("\nEvaluating similarity scores...")
    
    similarities = []
    for sample in tqdm(samples, desc="Computing similarities"):
        sim = compute_similarity(
            model, processor, 
            sample['image_path'], 
            sample['text'], 
            device
        )
        similarities.append(sim)
    
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    print(f"  Mean similarity: {mean_sim:.4f} ± {std_sim:.4f}")
    print(f"  Min similarity:  {np.min(similarities):.4f}")
    print(f"  Max similarity:  {np.max(similarities):.4f}")
    
    return {
        'mean': mean_sim,
        'std': std_sim,
        'min': np.min(similarities),
        'max': np.max(similarities),
        'scores': similarities
    }


def compare_models(pretrained_path: str, finetuned_path: str, 
                  data_dir: str, max_samples: int = 100, device: str = 'cuda'):
    """Compare pre-trained and fine-tuned CLIP models."""
    
    print("=" * 80)
    print("CLIP MODEL VALIDATION")
    print("=" * 80)
    
    # Load test data
    samples = load_test_data(data_dir, max_samples)
    
    # Evaluate pre-trained CLIP
    print("\n" + "=" * 80)
    print("EVALUATING PRE-TRAINED CLIP")
    print("=" * 80)
    pretrained_model, pretrained_processor = load_clip_model(pretrained_path, device)
    
    pretrained_sim = evaluate_similarity(pretrained_model, pretrained_processor, samples, device)
    pretrained_retrieval = evaluate_retrieval(pretrained_model, pretrained_processor, samples, device)
    
    # Evaluate fine-tuned CLIP
    print("\n" + "=" * 80)
    print("EVALUATING FINE-TUNED CLIP")
    print("=" * 80)
    finetuned_model, finetuned_processor = load_clip_model(finetuned_path, device)
    
    finetuned_sim = evaluate_similarity(finetuned_model, finetuned_processor, samples, device)
    finetuned_retrieval = evaluate_retrieval(finetuned_model, finetuned_processor, samples, device)
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print("\nSimilarity Scores:")
    print(f"  Pre-trained:  {pretrained_sim['mean']:.4f} ± {pretrained_sim['std']:.4f}")
    print(f"  Fine-tuned:   {finetuned_sim['mean']:.4f} ± {finetuned_sim['std']:.4f}")
    improvement = ((finetuned_sim['mean'] - pretrained_sim['mean']) / pretrained_sim['mean']) * 100
    print(f"  Improvement:  {improvement:+.2f}%")
    
    print("\nImage-to-Text Retrieval:")
    print(f"  Recall@1:  {pretrained_retrieval['i2t_recall_1']:.2f}% → {finetuned_retrieval['i2t_recall_1']:.2f}% ({finetuned_retrieval['i2t_recall_1'] - pretrained_retrieval['i2t_recall_1']:+.2f}%)")
    print(f"  Recall@5:  {pretrained_retrieval['i2t_recall_5']:.2f}% → {finetuned_retrieval['i2t_recall_5']:.2f}% ({finetuned_retrieval['i2t_recall_5'] - pretrained_retrieval['i2t_recall_5']:+.2f}%)")
    print(f"  Recall@10: {pretrained_retrieval['i2t_recall_10']:.2f}% → {finetuned_retrieval['i2t_recall_10']:.2f}% ({finetuned_retrieval['i2t_recall_10'] - pretrained_retrieval['i2t_recall_10']:+.2f}%)")
    
    print("\nText-to-Image Retrieval:")
    print(f"  Recall@1:  {pretrained_retrieval['t2i_recall_1']:.2f}% → {finetuned_retrieval['t2i_recall_1']:.2f}% ({finetuned_retrieval['t2i_recall_1'] - pretrained_retrieval['t2i_recall_1']:+.2f}%)")
    print(f"  Recall@5:  {pretrained_retrieval['t2i_recall_5']:.2f}% → {finetuned_retrieval['t2i_recall_5']:.2f}% ({finetuned_retrieval['t2i_recall_5'] - pretrained_retrieval['t2i_recall_5']:+.2f}%)")
    print(f"  Recall@10: {pretrained_retrieval['t2i_recall_10']:.2f}% → {finetuned_retrieval['t2i_recall_10']:.2f}% ({finetuned_retrieval['t2i_recall_10'] - pretrained_retrieval['t2i_recall_10']:+.2f}%)")
    
    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    if improvement > 5:
        print("✅ Fine-tuning SUCCESSFUL! Significant improvement detected.")
        print("   Recommendation: Use fine-tuned CLIP for MAPPO training.")
    elif improvement > 0:
        print("⚠️  Fine-tuning shows MINOR improvement.")
        print("   Recommendation: Fine-tuned CLIP is slightly better, use it.")
    else:
        print("❌ Fine-tuning did NOT improve performance.")
        print("   Recommendation: Check training logs, may need more epochs or data.")
    
    # Save results
    results = {
        'pretrained': {
            'similarity': pretrained_sim,
            'retrieval': pretrained_retrieval
        },
        'finetuned': {
            'similarity': finetuned_sim,
            'retrieval': finetuned_retrieval
        },
        'improvement_percent': improvement
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate fine-tuned CLIP model")
    parser.add_argument(
        '--pretrained',
        type=str,
        default='openai/clip-vit-base-patch32',
        help='Path to pre-trained CLIP model'
    )
    parser.add_argument(
        '--finetuned',
        type=str,
        required=True,
        help='Path to fine-tuned CLIP model'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to test data directory (with dataset.jsonl)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=100,
        help='Maximum number of test samples'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='clip_validation_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_models(
        args.pretrained,
        args.finetuned,
        args.data_dir,
        args.max_samples,
        args.device
    )
    
    # Save results
    with open(args.output, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        results_json = convert(results)
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
