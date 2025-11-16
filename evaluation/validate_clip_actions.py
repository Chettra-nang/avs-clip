"""
Validate CLIP on 5-action classification task.

This script evaluates how well CLIP can classify driving scenes
into the 5 action categories: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 5 action categories
ACTIONS = ['LANE_LEFT', 'IDLE', 'LANE_RIGHT', 'FASTER', 'SLOWER']

# Action templates for zero-shot classification
ACTION_TEMPLATES = {
    'LANE_LEFT': [
        "A car changing lanes to the left",
        "Vehicle moving to the left lane",
        "Changing to left lane on highway"
    ],
    'IDLE': [
        "A car ma intaining current speed and lane",
        "Vehicle staying in current lane",
        "Maintaining speed on highway"
    ],
    'LANE_RIGHT': [
        "A car changing lanes to the right",
        "Vehicle moving to the right lane",
        "Changing to right lane on highway"
    ],
    'FASTER': [
        "A car accelerating on highway",
        "Vehicle speeding up",
        "Accelerating to higher speed"
    ],
    'SLOWER': [
        "A car slowing down on highway",
        "Vehicle decelerating",
        "Reducing speed on highway"
    ]
}


def load_clip_model(model_path: str, device: str = 'cuda'):
    """Load CLIP model."""
    print(f"Loading CLIP from: {model_path}")
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, processor


def load_test_data_by_action(data_dir: str, samples_per_action: int = 50):
    """Load balanced test samples for each action."""
    data_path = Path(data_dir)
    jsonl_path = data_path / "dataset.jsonl"
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"dataset.jsonl not found in {data_dir}")
    
    # Collect samples by action
    samples_by_action = defaultdict(list)
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            image_path = data_path / record['image']
            
            # Get first agent's action
            if record['agents'] and image_path.exists():
                agent = record['agents'][0]
                action = agent.get('action_name', '')
                
                if action in ACTIONS:
                    samples_by_action[action].append({
                        'image_path': str(image_path),
                        'action': action,
                        'instruction': agent.get('instruction', ''),
                        'role': agent.get('role', 'unknown')
                    })
    
    # Balance samples
    balanced_samples = []
    for action in ACTIONS:
        action_samples = samples_by_action[action]
        if len(action_samples) > samples_per_action:
            # Randomly sample
            indices = np.random.choice(len(action_samples), samples_per_action, replace=False)
            action_samples = [action_samples[i] for i in indices]
        balanced_samples.extend(action_samples)
    
    print(f"\nLoaded test samples:")
    for action in ACTIONS:
        count = len([s for s in balanced_samples if s['action'] == action])
        print(f"  {action}: {count} samples")
    print(f"  Total: {len(balanced_samples)} samples")
    
    return balanced_samples


def classify_image(model, processor, image_path: str, device: str):
    """
    Classify image into one of 5 actions using zero-shot CLIP.
    
    Returns:
        predicted_action: str
        probabilities: dict mapping action -> probability
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create text prompts for each action (use all templates)
    all_texts = []
    action_indices = []
    for action in ACTIONS:
        templates = ACTION_TEMPLATES[action]
        all_texts.extend(templates)
        action_indices.extend([action] * len(templates))
    
    # Process inputs
    inputs = processor(
        text=all_texts,
        images=image,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get features
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # [1, num_texts]
        probs = logits_per_image.softmax(dim=1)[0]  # [num_texts]
    
    # Average probabilities for each action
    action_probs = {}
    for action in ACTIONS:
        # Get indices for this action
        indices = [i for i, a in enumerate(action_indices) if a == action]
        # Average probabilities
        action_probs[action] = probs[indices].mean().item()
    
    # Normalize
    total = sum(action_probs.values())
    action_probs = {k: v/total for k, v in action_probs.items()}
    
    # Get prediction
    predicted_action = max(action_probs, key=action_probs.get)
    
    return predicted_action, action_probs


def evaluate_action_classification(model, processor, samples: List[Dict], device: str):
    """Evaluate 5-action classification accuracy."""
    print("\nEvaluating 5-action classification...")
    
    # Track predictions
    y_true = []
    y_pred = []
    
    # Confusion matrix
    confusion = {action: {a: 0 for a in ACTIONS} for action in ACTIONS}
    
    # Classify each sample
    for sample in tqdm(samples, desc="Classifying"):
        true_action = sample['action']
        pred_action, probs = classify_image(
            model, processor, 
            sample['image_path'], 
            device
        )
        
        y_true.append(true_action)
        y_pred.append(pred_action)
        confusion[true_action][pred_action] += 1
    
    # Compute metrics
    accuracy = np.mean([t == p for t, p in zip(y_true, y_pred)]) * 100
    
    # Per-action accuracy
    per_action_acc = {}
    for action in ACTIONS:
        action_samples = [i for i, a in enumerate(y_true) if a == action]
        if action_samples:
            correct = sum([y_true[i] == y_pred[i] for i in action_samples])
            per_action_acc[action] = (correct / len(action_samples)) * 100
        else:
            per_action_acc[action] = 0.0
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")
    
    print("\nPer-Action Accuracy:")
    for action in ACTIONS:
        print(f"  {action:12s}: {per_action_acc[action]:5.2f}%")
    
    print("\nConfusion Matrix:")
    header = "True \\ Pred"
    print(f"{header:<12s}", end="")
    for action in ACTIONS:
        print(f"{action[:8]:>8s}", end="")
    print()
    print("-" * 60)
    
    for true_action in ACTIONS:
        print(f"{true_action:<12s}", end="")
        for pred_action in ACTIONS:
            count = confusion[true_action][pred_action]
            print(f"{count:>8d}", end="")
        print()
    
    return {
        'accuracy': accuracy,
        'per_action_accuracy': per_action_acc,
        'confusion_matrix': confusion,
        'y_true': y_true,
        'y_pred': y_pred
    }


def compare_models_on_actions(pretrained_path: str, finetuned_path: str,
                              data_dir: str, samples_per_action: int = 50,
                              device: str = 'cuda'):
    """Compare pre-trained and fine-tuned CLIP on 5-action classification."""
    
    print("=" * 80)
    print("CLIP 5-ACTION CLASSIFICATION VALIDATION")
    print("=" * 80)
    
    # Load test data
    samples = load_test_data_by_action(data_dir, samples_per_action)
    
    # Evaluate pre-trained CLIP
    print("\n" + "=" * 80)
    print("EVALUATING PRE-TRAINED CLIP")
    print("=" * 80)
    pretrained_model, pretrained_processor = load_clip_model(pretrained_path, device)
    pretrained_results = evaluate_action_classification(
        pretrained_model, pretrained_processor, samples, device
    )
    
    # Evaluate fine-tuned CLIP
    print("\n" + "=" * 80)
    print("EVALUATING FINE-TUNED CLIP")
    print("=" * 80)
    finetuned_model, finetuned_processor = load_clip_model(finetuned_path, device)
    finetuned_results = evaluate_action_classification(
        finetuned_model, finetuned_processor, samples, device
    )
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\nOverall Accuracy:")
    print(f"  Pre-trained:  {pretrained_results['accuracy']:.2f}%")
    print(f"  Fine-tuned:   {finetuned_results['accuracy']:.2f}%")
    improvement = finetuned_results['accuracy'] - pretrained_results['accuracy']
    print(f"  Improvement:  {improvement:+.2f}%")
    
    print(f"\nPer-Action Accuracy Comparison:")
    print(f"{'Action':<12s} {'Pre-trained':>12s} {'Fine-tuned':>12s} {'Improvement':>12s}")
    print("-" * 60)
    for action in ACTIONS:
        pre_acc = pretrained_results['per_action_accuracy'][action]
        fine_acc = finetuned_results['per_action_accuracy'][action]
        diff = fine_acc - pre_acc
        print(f"{action:<12s} {pre_acc:>11.2f}% {fine_acc:>11.2f}% {diff:>11.2f}%")
    
    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    
    if improvement > 10:
        print("✅ Fine-tuning HIGHLY SUCCESSFUL!")
        print(f"   +{improvement:.2f}% improvement in action classification")
        print("   Recommendation: Use fine-tuned CLIP for MAPPO training.")
    elif improvement > 5:
        print("✅ Fine-tuning SUCCESSFUL!")
        print(f"   +{improvement:.2f}% improvement in action classification")
        print("   Recommendation: Use fine-tuned CLIP for MAPPO training.")
    elif improvement > 0:
        print("⚠️  Fine-tuning shows MINOR improvement.")
        print(f"   +{improvement:.2f}% improvement in action classification")
        print("   Recommendation: Fine-tuned CLIP is slightly better, use it.")
    else:
        print("❌ Fine-tuning did NOT improve action classification.")
        print(f"   {improvement:.2f}% change in accuracy")
        print("   Recommendation: Check training approach or use pre-trained CLIP.")
    
    return {
        'pretrained': pretrained_results,
        'finetuned': finetuned_results,
        'improvement': improvement
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate CLIP on 5-action classification"
    )
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
        '--samples_per_action',
        type=int,
        default=50,
        help='Number of samples per action (balanced)'
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
        default='clip_action_validation_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run comparison
    results = compare_models_on_actions(
        args.pretrained,
        args.finetuned,
        args.data_dir,
        args.samples_per_action,
        args.device
    )
    
    # Save results
    with open(args.output, 'w') as f:
        # Convert to JSON-serializable format
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
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
