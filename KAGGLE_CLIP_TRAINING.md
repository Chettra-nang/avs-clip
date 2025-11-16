# Train CLIP on Kaggle with Your Data

Complete guide to fine-tune CLIP on Kaggle using your uploaded dataset.

## Your Dataset Location

```
/kaggle/input/avs-3-sen/
```

This should contain your collected data (images + dataset.jsonl files).

---

## Kaggle Notebook Cells

### Cell 1: Fix NumPy & Install Dependencies

```python
import subprocess
import sys

print("Fixing NumPy compatibility...")
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2.0"])

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "gymnasium", "highway-env", "transformers", "torch", 
                      "torchvision", "pillow", "pyyaml", "tqdm", 
                      "tensorboard", "opencv-python"])

print("✓ Dependencies installed")
```

---

### Cell 2: Clone Code from GitHub

```python
import os
import sys

# Clone your code repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /kaggle/working/code

# Change to code directory
os.chdir('/kaggle/working/code')
sys.path.insert(0, '/kaggle/working/code')

print("✓ Code loaded")
```

---

### Cell 3: Verify Data Location

```python
import os

# Check if data exists
data_path = '/kaggle/input/avs-3-sen'

print(f"Checking data at: {data_path}")
print(f"Data exists: {os.path.exists(data_path)}")

# List contents
if os.path.exists(data_path):
    print("\nData contents:")
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
            # Check for dataset.jsonl
            jsonl_path = os.path.join(item_path, 'dataset.jsonl')
            if os.path.exists(jsonl_path):
                print(f"     ✓ dataset.jsonl found")
        else:
            print(f"  📄 {item}")
else:
    print("⚠️  Data not found! Make sure dataset is added in Kaggle.")
```

---

### Cell 4: Create Output Directories

```python
import os

# Create directories for CLIP fine-tuning output
os.makedirs('/kaggle/working/clip_finetuned', exist_ok=True)
os.makedirs('/kaggle/working/logs_clip', exist_ok=True)

print("✓ Directories created")
```

---

### Cell 5: Train CLIP

```python
import os
import sys

# Set Python path
os.environ['PYTHONPATH'] = '/kaggle/working/code'
sys.path.insert(0, '/kaggle/working/code')

# Find data directories
data_path = '/kaggle/input/avs-3-sen'
data_dirs = []

# Collect all scenario directories that have dataset.jsonl
for item in os.listdir(data_path):
    item_path = os.path.join(data_path, item)
    if os.path.isdir(item_path):
        jsonl_path = os.path.join(item_path, 'dataset.jsonl')
        if os.path.exists(jsonl_path):
            data_dirs.append(item_path)
            print(f"Found data: {item_path}")

print(f"\nTraining CLIP on {len(data_dirs)} scenarios...")

# Run CLIP training
!python training/train_clip.py \
    --data_dirs {' '.join(data_dirs)} \
    --save_dir /kaggle/working/clip_finetuned \
    --log_dir /kaggle/working/logs_clip \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-5 \
    --device cuda \
    --num_workers 2

print("✓ CLIP fine-tuning complete!")
```

---

### Cell 6: Verify Fine-tuned CLIP

```python
import os

# Check if CLIP model was saved
clip_dir = '/kaggle/working/clip_finetuned'

print("Checking fine-tuned CLIP model...")
if os.path.exists(clip_dir):
    files = os.listdir(clip_dir)
    print(f"✓ CLIP model saved with {len(files)} files:")
    for f in files:
        print(f"  - {f}")
else:
    print("⚠️  CLIP model not found!")
```

---

### Cell 7: Archive CLIP Model

```python
import shutil

# Archive fine-tuned CLIP for download
shutil.make_archive('/kaggle/working/clip_finetuned_model', 'zip',
                    '/kaggle/working/clip_finetuned')

print("✓ CLIP model archived: clip_finetuned_model.zip")
print("\nDownload from Output tab and use for MAPPO training!")
```

---

## Alternative: Train CLIP and MAPPO in One Session

If you want to do both in one Kaggle session:

### Cell 8: Train MAPPO with Fine-tuned CLIP

```python
import os
import sys

os.environ['PYTHONPATH'] = '/kaggle/working/code'
sys.path.insert(0, '/kaggle/working/code')

# Create directories for MAPPO
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/logs_mappo', exist_ok=True)

# Train MAPPO using fine-tuned CLIP
!python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --clip_dir /kaggle/working/clip_finetuned \
    --steps 20000 \
    --horizon 256 \
    --lr 3e-4 \
    --device cuda \
    --logdir /kaggle/working/logs_mappo

print("✓ MAPPO training complete!")
```

---

### Cell 9: Archive All Results

```python
import shutil
import os

# Archive CLIP model
if os.path.exists('/kaggle/working/clip_finetuned'):
    shutil.make_archive('/kaggle/working/clip_model', 'zip',
                        '/kaggle/working/clip_finetuned')
    print("✓ CLIP archived: clip_model.zip")

# Archive MAPPO checkpoints
if os.path.exists('/kaggle/working/code/models'):
    shutil.make_archive('/kaggle/working/mappo_models', 'zip',
                        '/kaggle/working/code/models')
    print("✓ MAPPO archived: mappo_models.zip")

# Archive logs
if os.path.exists('/kaggle/working/logs_clip'):
    shutil.make_archive('/kaggle/working/logs_clip_archive', 'zip',
                        '/kaggle/working/logs_clip')
    print("✓ CLIP logs archived: logs_clip_archive.zip")

if os.path.exists('/kaggle/working/logs_mappo'):
    shutil.make_archive('/kaggle/working/logs_mappo_archive', 'zip',
                        '/kaggle/working/logs_mappo')
    print("✓ MAPPO logs archived: logs_mappo_archive.zip")

print("\n✓ All results archived!")
print("\nDownload from Output tab:")
print("  - clip_model.zip")
print("  - mappo_models.zip")
print("  - logs_clip_archive.zip")
print("  - logs_mappo_archive.zip")
```

---

## Expected Timeline

### CLIP Fine-tuning Only
- **Setup:** 5 minutes
- **CLIP training:** 2-3 hours (10 epochs)
- **Total:** ~3 hours

### CLIP + MAPPO (Full Pipeline)
- **Setup:** 5 minutes
- **CLIP training:** 2-3 hours
- **MAPPO training:** 2-3 hours (20k steps)
- **Total:** ~5-6 hours

---

## Troubleshooting

### Data Not Found

If data is not at `/kaggle/input/avs-3-sen`:

1. Check "Add Data" in right sidebar
2. Search for your dataset name
3. Click "Add"
4. Note the actual path (might be different)
5. Update `data_path` variable

### Out of Memory During CLIP Training

Reduce batch size:

```python
!python training/train_clip.py \
    --batch_size 2 \
    --epochs 10 \
    ...
```

### CLIP Training Too Slow

Reduce epochs or use smaller dataset:

```python
!python training/train_clip.py \
    --epochs 5 \
    --batch_size 4 \
    ...
```

---

## Download and Use Locally

After training completes:

### 1. Download from Kaggle

- Click "Output" tab
- Download `clip_model.zip`
- Download `mappo_models.zip` (if you trained MAPPO too)

### 2. Extract Locally

```bash
cd ~/Desktop/avs

# Extract CLIP model
unzip ~/Downloads/clip_model.zip -d models/clip_finetuned/

# Extract MAPPO models (if applicable)
unzip ~/Downloads/mappo_models.zip -d models/
```

### 3. Use Fine-tuned CLIP

```bash
# Train MAPPO locally with Kaggle-trained CLIP
PYTHONPATH=. python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --clip_dir models/clip_finetuned \
    --steps 100000 \
    --device cuda
```

---

## Summary

**Workflow:**

1. ✅ Upload data to Kaggle dataset (`/kaggle/input/avs-3-sen`)
2. ✅ Clone code from GitHub
3. ✅ Train CLIP on Kaggle (2-3 hours)
4. ✅ Download fine-tuned CLIP
5. ✅ Use for MAPPO training (Kaggle or local)

**Advantages:**

- ✅ Free GPU for CLIP fine-tuning
- ✅ More VRAM (16 GB vs 3.68 GB)
- ✅ Can train both CLIP and MAPPO in one session
- ✅ No local GPU wear

**Total time:** 3 hours (CLIP only) or 5-6 hours (CLIP + MAPPO)

---

## Quick Copy-Paste Version

For fastest setup, use this single training cell after setup:

```python
import os, sys, shutil

# Setup paths
os.environ['PYTHONPATH'] = '/kaggle/working/code'
sys.path.insert(0, '/kaggle/working/code')
os.makedirs('/kaggle/working/clip_finetuned', exist_ok=True)

# Find data directories
data_path = '/kaggle/input/avs-3-sen'
data_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) 
             if os.path.isdir(os.path.join(data_path, d)) and 
             os.path.exists(os.path.join(data_path, d, 'dataset.jsonl'))]

print(f"Training on {len(data_dirs)} scenarios: {data_dirs}")

# Train CLIP
!python training/train_clip.py \
    --data_dirs {' '.join(data_dirs)} \
    --save_dir /kaggle/working/clip_finetuned \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-5 \
    --device cuda

# Archive
shutil.make_archive('/kaggle/working/clip_model', 'zip', '/kaggle/working/clip_finetuned')
print("✓ Done! Download clip_model.zip from Output tab")
```

---

**Remember:** Make sure your dataset is added in Kaggle's "Add Data" section!
