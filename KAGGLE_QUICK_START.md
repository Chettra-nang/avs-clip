# Kaggle GPU Training - Quick Start

Train your model on Kaggle's free GPU in 3 simple steps!

## Prerequisites

- GitHub account with your code pushed
- Kaggle account (free)

## Step 1: Push to GitHub (1 minute)

```bash
git add .
git commit -m "Ready for Kaggle training"
git push origin main
```

## Step 2: Create Kaggle Notebook (2 minutes)

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. **Settings → Accelerator → GPU T4 x2** ⚠️ IMPORTANT!
4. **Settings → Internet → ON** ⚠️ IMPORTANT!

## Step 3: Copy & Run (2-3 hours)

Copy these cells into your Kaggle notebook:

### 📦 Cell 1: Setup

```python
# Fix NumPy version compatibility
!pip uninstall -y numpy
!pip install -q "numpy<2.0"

# Install dependencies
!pip install -q gymnasium highway-env transformers torch pillow pyyaml tqdm tensorboard opencv-python

# Clone code
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /kaggle/working/code
%cd /kaggle/working/code

# Verify setup
import sys
sys.path.insert(0, '/kaggle/working/code')
import torch
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 🚀 Cell 2: Train

```python
!PYTHONPATH=/kaggle/working/code python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --steps 20000 \
    --device cuda \
    --logdir /kaggle/working/logs
```

### 💾 Cell 3: Download

```python
import shutil
shutil.make_archive('/kaggle/working/results', 'zip', '/kaggle/working/code/models')
print("✓ Download 'results.zip' from Output tab")
```

## That's It!

Training will take ~2-3 hours. Download results from the Output tab when done.

## Full Guide

See [KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md) for:
- Detailed instructions
- Troubleshooting
- Advanced options
- Resume from checkpoints
- Upload to Google Drive

## Quick Reference

| What | Time | GPU Hours |
|------|------|-----------|
| 20k steps | 2-3 hours | 3 hours |
| 100k steps | 10-12 hours | 12 hours |
| Free quota | Per week | 30 hours |

**Tip:** Train in batches (20k steps per session) to fit in free quota!
