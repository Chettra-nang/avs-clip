# How to Validate Fine-tuned CLIP

Complete guide to test if your CLIP fine-tuning improved the model.

## What Gets Validated

The validation compares:
1. **Pre-trained CLIP** (openai/clip-vit-base-patch32)
2. **Your fine-tuned CLIP** (models/clip_finetuned)

### Metrics

1. **Similarity Scores**
   - How well images match their text descriptions
   - Higher = better understanding

2. **Retrieval Accuracy**
   - Image → Text: Given image, find correct text
   - Text → Image: Given text, find correct image
   - Measured as Recall@1, @5, @10

3. **Overall Improvement**
   - Percentage improvement over baseline
   - Verdict: Success / Minor / Failed

---

## Quick Validation (Local)

### After Downloading from Kaggle

```bash
# Extract fine-tuned CLIP
unzip ~/Downloads/clip_model.zip -d models/clip_finetuned/

# Run validation
PYTHONPATH=. python evaluation/validate_clip.py \
    --finetuned models/clip_finetuned \
    --data_dir data/highway_heterogeneous \
    --max_samples 100 \
    --device cuda
```

### Expected Output

```
================================================================================
CLIP MODEL VALIDATION
================================================================================

Loaded 100 test samples from data/highway_heterogeneous

================================================================================
EVALUATING PRE-TRAINED CLIP
================================================================================

Evaluating similarity scores...
  Mean similarity: 0.2543 ± 0.0821
  Min similarity:  0.1234
  Max similarity:  0.4567

Evaluating retrieval accuracy...
Image-to-Text Retrieval:
  Recall@1:  15.00%
  Recall@5:  42.00%
  Recall@10: 58.00%

Text-to-Image Retrieval:
  Recall@1:  16.00%
  Recall@5:  45.00%
  Recall@10: 61.00%

================================================================================
EVALUATING FINE-TUNED CLIP
================================================================================

Evaluating similarity scores...
  Mean similarity: 0.3124 ± 0.0654
  Min similarity:  0.1876
  Max similarity:  0.5234

Evaluating retrieval accuracy...
Image-to-Text Retrieval:
  Recall@1:  28.00%
  Recall@5:  65.00%
  Recall@10: 78.00%

Text-to-Image Retrieval:
  Recall@1:  30.00%
  Recall@5:  68.00%
  Recall@10: 81.00%

================================================================================
COMPARISON SUMMARY
================================================================================

Similarity Scores:
  Pre-trained:  0.2543 ± 0.0821
  Fine-tuned:   0.3124 ± 0.0654
  Improvement:  +22.85%

Image-to-Text Retrieval:
  Recall@1:  15.00% → 28.00% (+13.00%)
  Recall@5:  42.00% → 65.00% (+23.00%)
  Recall@10: 58.00% → 78.00% (+20.00%)

Text-to-Image Retrieval:
  Recall@1:  16.00% → 30.00% (+14.00%)
  Recall@5:  45.00% → 68.00% (+23.00%)
  Recall@10: 61.00% → 81.00% (+20.00%)

================================================================================
VERDICT
================================================================================
✅ Fine-tuning SUCCESSFUL! Significant improvement detected.
   Recommendation: Use fine-tuned CLIP for MAPPO training.

✓ Results saved to: clip_validation_results.json
```

---

## Validation on Kaggle

Add this cell after CLIP training completes:

```python
# Validate fine-tuned CLIP on Kaggle
import os, sys

os.environ['PYTHONPATH'] = '/kaggle/working/code'
sys.path.insert(0, '/kaggle/working/code')

# Run validation
!python evaluation/validate_clip.py \
    --finetuned /kaggle/working/clip_finetuned \
    --data_dir /kaggle/input/avs-3-sen/highway_heterogeneous \
    --max_samples 100 \
    --device cuda \
    --output /kaggle/working/clip_validation.json

print("✓ Validation complete!")
```

---

## Interpreting Results

### Similarity Scores

**Good improvement:** +10% or more
```
Pre-trained:  0.25
Fine-tuned:   0.30  (+20%)  ← Excellent!
```

**Minor improvement:** +5% to +10%
```
Pre-trained:  0.25
Fine-tuned:   0.27  (+8%)   ← Okay
```

**No improvement:** < +5%
```
Pre-trained:  0.25
Fine-tuned:   0.26  (+4%)   ← May need more training
```

### Retrieval Accuracy

**Recall@1** (most important):
- Pre-trained: ~15-20%
- Fine-tuned target: >25%
- Excellent: >30%

**Recall@5**:
- Pre-trained: ~40-50%
- Fine-tuned target: >60%
- Excellent: >70%

### Overall Verdict

**✅ Success (>10% improvement)**
- Use fine-tuned CLIP for MAPPO
- Expect better RL performance

**⚠️ Minor (5-10% improvement)**
- Fine-tuned CLIP is slightly better
- May not significantly impact RL performance
- Still worth using

**❌ Failed (<5% improvement)**
- Fine-tuning didn't help
- Possible issues:
  - Not enough training epochs
  - Data quality issues
  - Learning rate too high/low
- Consider:
  - Training longer (20 epochs)
  - Collecting more data
  - Using pre-trained CLIP instead

---

## Advanced: Visual Inspection

Create a script to visualize examples:

```python
# File: evaluation/visualize_clip.py

import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor

def visualize_top_matches(model_path, data_dir, num_examples=5):
    """Show top image-text matches."""
    
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    model.eval()
    
    # Load samples
    # ... (load images and texts)
    
    # Compute similarities
    # ... (compute similarity matrix)
    
    # Plot top matches
    fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4*num_examples))
    
    for i in range(num_examples):
        # Show image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis('off')
        
        # Show top matching text
        top_text_idx = similarities[i].argmax()
        axes[i, 1].text(0.5, 0.5, texts[top_text_idx], 
                       ha='center', va='center', wrap=True)
        axes[i, 1].set_title(f"Top Match (sim={similarities[i, top_text_idx]:.3f})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('clip_matches.png')
    print("✓ Saved visualization to clip_matches.png")

# Run
visualize_top_matches('models/clip_finetuned', 'data/highway_heterogeneous')
```

---

## Troubleshooting

### Low Similarity Scores (<0.2)

**Possible causes:**
- CLIP not trained enough
- Data quality issues
- Text descriptions too generic

**Solutions:**
- Train for more epochs (20 instead of 10)
- Improve text descriptions
- Collect more diverse data

### Retrieval Accuracy Not Improving

**Possible causes:**
- Overfitting to training data
- Test data too different from training data
- Learning rate too high

**Solutions:**
- Use validation split during training
- Collect more varied data
- Lower learning rate (5e-6 instead of 1e-5)

### Fine-tuned Worse Than Pre-trained

**Possible causes:**
- Catastrophic forgetting
- Learning rate too high
- Trained too long

**Solutions:**
- Use lower learning rate
- Train for fewer epochs
- Use pre-trained CLIP instead

---

## Decision Tree

```
After validation:

Improvement > 15%?
├─ YES → ✅ Use fine-tuned CLIP, expect good RL results
└─ NO
   └─ Improvement > 5%?
      ├─ YES → ⚠️ Use fine-tuned CLIP, minor benefit
      └─ NO
         └─ Improvement > 0%?
            ├─ YES → ⚠️ Use fine-tuned CLIP or pre-trained (similar)
            └─ NO → ❌ Use pre-trained CLIP, fine-tuning failed
```

---

## Next Steps

### If Validation Successful (>10% improvement)

```bash
# 1. Use fine-tuned CLIP for MAPPO training
PYTHONPATH=. python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --clip_dir models/clip_finetuned \
    --steps 100000 \
    --device cuda

# 2. Compare with baseline (no fine-tuning)
PYTHONPATH=. python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --clip_dir openai/clip-vit-base-patch32 \
    --steps 100000 \
    --device cuda

# 3. Evaluate both models
python evaluation/eval_mappo.py --model models/vla_mappo_finetuned.pth
python evaluation/eval_mappo.py --model models/vla_mappo_pretrained.pth
```

### If Validation Failed (<5% improvement)

```bash
# Option 1: Train CLIP longer
python training/train_clip.py \
    --data_dirs data/highway_heterogeneous \
    --epochs 20 \
    --lr 5e-6 \
    --batch_size 4

# Option 2: Use pre-trained CLIP
python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --steps 100000 \
    --device cuda

# Option 3: Try DINOv2 instead
# (See ROADMAP_TO_DINOV2.md)
```

---

## Summary

**Validation checks:**
- ✅ Similarity scores improved
- ✅ Retrieval accuracy improved
- ✅ Overall improvement >10%

**If successful:**
- Use fine-tuned CLIP for MAPPO
- Expect better RL performance
- Document improvement in paper

**If failed:**
- Use pre-trained CLIP
- Consider more training or DINOv2
- Focus on RL algorithm tuning instead

---

**Quick command:**
```bash
PYTHONPATH=. python evaluation/validate_clip.py \
    --finetuned models/clip_finetuned \
    --data_dir data/highway_heterogeneous \
    --max_samples 100
```

**Expected time:** 2-5 minutes

**Output:** Detailed comparison + JSON results file
