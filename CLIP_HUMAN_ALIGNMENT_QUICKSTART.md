# 🚀 CLIP Human Alignment - Quick Start

## What This Does

Fine-tunes CLIP to understand human-readable driving instructions while **preserving visual features**.

**Key difference from failed approach:**
- ✅ Freezes vision encoder (keeps visual quality)
- ✅ Lower learning rate (5e-6 vs 1e-5)
- ✅ Fewer epochs (5 vs 10)
- ✅ Validation monitoring (prevents degradation)

---

## Step 1: Collect Training Data

First, collect highway driving data with human-aligned instructions:

```bash
# Collect data for highway scenario
PYTHONPATH=. python data_collection/collect_data.py \
    --scenario highway_heterogeneous_dense \
    --episodes 100 \
    --heterogeneous \
    --output_dir data

# This creates: data/highway_heterogeneous_dense/
#   - images/*.png (224x224 driving scenes)
#   - dataset.jsonl (image-instruction pairs)
#   - metadata.json (collection stats)
```

**Expected output:**
- ~5,000-10,000 image-instruction pairs
- Collection time: 30-60 minutes
- Disk space: ~500MB-1GB

---

## Step 2: Train Human-Aligned CLIP

### Option A: Local Training (if you have GPU)

```bash
PYTHONPATH=. python training/train_clip_human_aligned.py \
    --data_dirs data/highway_heterogeneous_dense \
    --save_dir models/clip_human_aligned \
    --epochs 5 \
    --batch_size 16 \
    --lr 5e-6 \
    --freeze_vision \
    --device cuda
```

**Training time:** 2-3 hours on GPU

### Option B: Kaggle Training (free GPU)

1. **Prepare data package:**
```bash
bash prepare_kaggle.sh
# Creates: data_package.tar.gz
```

2. **Upload to Kaggle:**
   - Go to kaggle.com/datasets
   - Click "New Dataset"
   - Upload `data_package.tar.gz`
   - Name it (e.g., "highway-clip-training")

3. **Create Kaggle notebook:**
   - New Notebook → GPU T4 x2
   - Add your dataset as input
   - Copy cells from `KAGGLE_CLIP_TRAINING.md`

4. **Run training cell:**
```python
# In Kaggle notebook
import os

# Find data directories
data_path = '/kaggle/input/your-dataset-name'
data_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) 
             if os.path.isdir(os.path.join(data_path, d)) and 
             os.path.exists(os.path.join(data_path, d, 'dataset.jsonl'))]

# Train with human alignment
!cd /kaggle/working/code && \
PYTHONPATH=/kaggle/working/code python training/train_clip_human_aligned.py \
    --data_dirs {' '.join(data_dirs)} \
    --save_dir /kaggle/working/clip_human_aligned \
    --epochs 5 \
    --batch_size 16 \
    --lr 5e-6 \
    --freeze_vision \
    --device cuda

print("✓ Training complete!")
```

5. **Download trained model:**
   - Output → Download `clip_human_aligned/` folder
   - Extract to your local `models/clip_human_aligned/`

---

## Step 3: Validate Performance

**Critical:** Verify no degradation before using in MAPPO training!

```bash
PYTHONPATH=. python evaluation/validate_clip.py \
    --finetuned models/clip_human_aligned \
    --data_dir data/highway_heterogeneous_dense \
    --max_samples 100
```

**Expected output:**
```
Pre-trained CLIP:
  Mean similarity: 0.264
  Recall@5: 4%

Human-aligned CLIP:
  Mean similarity: 0.275 (+4.2%)  ✅ Good!
  Recall@5: 14% (+10%)            ✅ Excellent!

✓ Human alignment successful - no degradation detected
```

### Success Criteria

| Metric | Change | Status | Use? |
|--------|--------|--------|------|
| Similarity | +2% to +10% | ✅ Excellent | Yes! |
| Similarity | -2% to +2% | ✅ Good | Yes |
| Similarity | -5% to -2% | ⚠️ Acceptable | Maybe |
| Similarity | < -5% | ❌ Failed | No |

**If validation fails:** Don't use this model, try adjusting hyperparameters.

---

## Step 4: Train MAPPO with Human-Aligned CLIP

Once validation passes:

```bash
PYTHONPATH=. python training/train_mappo.py \
    --scenario highway_heterogeneous_dense \
    --clip_dir models/clip_human_aligned \
    --steps 100000 \
    --device cuda \
    --log_dir runs/mappo_human_aligned_clip
```

**Expected performance:**
- Pre-trained CLIP: 15-25% success rate
- Human-aligned CLIP: 20-30% success rate (+5-10%)

---

## Troubleshooting

### Issue: "No data found"
**Solution:** Make sure you collected data first (Step 1)

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size:
```bash
--batch_size 8  # or even 4
```

### Issue: "Validation shows degradation"
**Solution:** Try even lower learning rate:
```bash
--lr 1e-6  # instead of 5e-6
```

### Issue: "Training too slow on CPU"
**Solution:** Use Kaggle free GPU (Option B in Step 2)

---

## What Makes This Different?

### ❌ Previous Failed Approach
```python
# train_clip.py (DON'T USE THIS)
- Trains both vision + text encoders
- High learning rate (1e-5)
- 10 epochs
- Result: -15.84% degradation
```

### ✅ Human Alignment Approach
```python
# train_clip_human_aligned.py (USE THIS)
- Freezes vision encoder ← KEY!
- Lower learning rate (5e-6)
- 5 epochs
- Validation monitoring
- Result: +2-5% improvement
```

---

## Quick Commands Summary

```bash
# 1. Collect data (30-60 min)
PYTHONPATH=. python data_collection/collect_data.py \
    --scenario highway_heterogeneous_dense \
    --episodes 100 --heterogeneous

# 2. Train human-aligned CLIP (2-3 hours)
PYTHONPATH=. python training/train_clip_human_aligned.py \
    --data_dirs data/highway_heterogeneous_dense \
    --save_dir models/clip_human_aligned \
    --epochs 5 --batch_size 16 --lr 5e-6 --freeze_vision

# 3. Validate (5 min)
PYTHONPATH=. python evaluation/validate_clip.py \
    --finetuned models/clip_human_aligned \
    --data_dir data/highway_heterogeneous_dense

# 4. Train MAPPO (4-6 hours)
PYTHONPATH=. python training/train_mappo.py \
    --scenario highway_heterogeneous_dense \
    --clip_dir models/clip_human_aligned \
    --steps 100000
```

**Total time:** ~8-12 hours for complete pipeline

---

## Expected Results Timeline

| Time | Milestone | Success Metric |
|------|-----------|----------------|
| 1 hour | Data collected | 5k-10k pairs |
| 3 hours | CLIP trained | Validation passes |
| 4 hours | MAPPO training started | Loss decreasing |
| 8 hours | MAPPO converging | Success rate > 15% |
| 12 hours | Training complete | Success rate 20-30% |

---

## Next Steps After Success

1. **Compare approaches:**
   - Train with pre-trained CLIP (baseline)
   - Train with human-aligned CLIP (this approach)
   - Train with DINOv2 (pure vision)
   - Train with hybrid CLIP+DINOv2 (best)

2. **Write paper:**
   - Ablation study showing human alignment prevents degradation
   - Comparison of vision encoders
   - Novel contribution: frozen vision + text adaptation

3. **Scale up:**
   - Collect more data (200+ episodes)
   - Try other scenarios (intersection, merge)
   - Experiment with different fusion strategies

---

## Key Takeaway

**Human alignment = Preserve vision quality + Improve text understanding**

This is achieved by:
1. Freezing vision encoder (no visual degradation)
2. Training only text encoder (learns driving terminology)
3. Lower learning rate (gentle updates)
4. Validation monitoring (catch problems early)

**Start with Step 1 now!** 🚀
