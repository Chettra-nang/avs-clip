# Validate CLIP - Quick Reference

## One Command

```bash
PYTHONPATH=. python evaluation/validate_clip.py \
    --finetuned models/clip_finetuned \
    --data_dir data/highway_heterogeneous \
    --max_samples 100 \
    --device cuda
```

## What It Does

Compares pre-trained vs fine-tuned CLIP on:
1. **Similarity scores** (higher = better)
2. **Retrieval accuracy** (Recall@1, @5, @10)
3. **Overall improvement** (%)

## Expected Results

### ✅ Success (>10% improvement)
```
Similarity: 0.25 → 0.30 (+20%)
Recall@1:   15% → 28% (+13%)
Verdict: Use fine-tuned CLIP!
```

### ⚠️ Minor (5-10% improvement)
```
Similarity: 0.25 → 0.27 (+8%)
Recall@1:   15% → 20% (+5%)
Verdict: Slightly better, use it
```

### ❌ Failed (<5% improvement)
```
Similarity: 0.25 → 0.26 (+4%)
Recall@1:   15% → 17% (+2%)
Verdict: Use pre-trained instead
```

## On Kaggle

Add after CLIP training:

```python
!python evaluation/validate_clip.py \
    --finetuned /kaggle/working/clip_finetuned \
    --data_dir /kaggle/input/avs-3-sen/highway_heterogeneous \
    --max_samples 100 \
    --device cuda
```

## Time

2-5 minutes

## Full Guide

See `VALIDATE_CLIP.md` for:
- Detailed interpretation
- Troubleshooting
- Visual inspection
- Next steps
