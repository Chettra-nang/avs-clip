# Train CLIP on Kaggle - Quick Reference

Your data is at: `/kaggle/input/avs-3-sen`

## Setup (Copy these 3 cells)

### Cell 1: Install
```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "gymnasium", "highway-env", "transformers", "torch", 
                      "pillow", "pyyaml", "tqdm", "tensorboard", "opencv-python"])
print("✓ Installed")
```

### Cell 2: Clone Code
```python
import os, sys
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /kaggle/working/code
os.chdir('/kaggle/working/code')
sys.path.insert(0, '/kaggle/working/code')
print("✓ Code loaded")
```

### Cell 3: Train CLIP
```python
import os, sys, shutil

os.environ['PYTHONPATH'] = '/kaggle/working/code'
os.makedirs('/kaggle/working/clip_finetuned', exist_ok=True)

# Find your data
data_path = '/kaggle/input/avs-3-sen'
data_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) 
             if os.path.isdir(os.path.join(data_path, d)) and 
             os.path.exists(os.path.join(data_path, d, 'dataset.jsonl'))]

print(f"Found {len(data_dirs)} scenarios")

# Train
!python training/train_clip.py \
    --data_dirs {' '.join(data_dirs)} \
    --save_dir /kaggle/working/clip_finetuned \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-5 \
    --device cuda

# Archive
shutil.make_archive('/kaggle/working/clip_model', 'zip', '/kaggle/working/clip_finetuned')
print("✓ Download clip_model.zip from Output tab!")
```

## That's It!

Training takes ~2-3 hours. Download `clip_model.zip` when done.

## Full Guide

See `KAGGLE_CLIP_TRAINING.md` for:
- Detailed instructions
- CLIP + MAPPO in one session
- Troubleshooting
- How to use locally

## Replace GitHub URL

Don't forget to replace `YOUR_USERNAME/YOUR_REPO` in Cell 2!
