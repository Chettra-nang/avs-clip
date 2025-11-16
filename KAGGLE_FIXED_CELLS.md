# Fixed Kaggle Cells (Copy & Paste)

## ⚠️ NumPy Compatibility Fix

Kaggle has NumPy 2.x but highway-env needs NumPy 1.x. Use these fixed cells:

---

## Cell 1: Fix NumPy & Install Dependencies

```python
# CRITICAL: Fix NumPy version first!
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

print("✓ All dependencies installed with NumPy 1.x")
```

---

## Cell 2: Clone Code from GitHub

```python
import os
import sys

# Clone your repository
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /kaggle/working/code

# Change directory
os.chdir('/kaggle/working/code')

# Add to Python path
sys.path.insert(0, '/kaggle/working/code')

print("✓ Code loaded from GitHub")
```

**⚠️ Replace `YOUR_USERNAME/YOUR_REPO` with your actual GitHub URL!**

---

## Cell 3: Verify Setup

```python
import torch
import numpy as np

# Check NumPy version
print(f"NumPy version: {np.__version__}")
assert np.__version__.startswith('1.'), "NumPy must be 1.x!"

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test imports
import gymnasium as gym
import highway_env
import custom_envs

print("✓ All imports successful")
```

---

## Cell 4: Create Directories

```python
import os

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/logs', exist_ok=True)

print("✓ Directories created")
```

---

## Cell 5: Run Training

```python
import os
import sys

# Set environment
os.environ['PYTHONPATH'] = '/kaggle/working/code'
sys.path.insert(0, '/kaggle/working/code')

# Run training
!python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --steps 20000 \
    --horizon 256 \
    --lr 3e-4 \
    --device cuda \
    --logdir /kaggle/working/logs

print("✓ Training complete!")
```

---

## Cell 6: Archive Results

```python
import shutil
import os

# Create archives
if os.path.exists('/kaggle/working/code/models'):
    shutil.make_archive('/kaggle/working/training_results', 'zip', 
                        '/kaggle/working/code/models')
    print("✓ Models archived: training_results.zip")

if os.path.exists('/kaggle/working/logs'):
    shutil.make_archive('/kaggle/working/training_logs', 'zip',
                        '/kaggle/working/logs')
    print("✓ Logs archived: training_logs.zip")

print("\n✓ Download from Output tab:")
print("  - training_results.zip")
print("  - training_logs.zip")
```

---

## Complete Notebook (All Cells)

Copy all cells above in order. Total time: ~2-3 hours.

### Expected Output

**Cell 1:**
```
Fixing NumPy compatibility...
Installing dependencies...
✓ All dependencies installed with NumPy 1.x
```

**Cell 2:**
```
Cloning into '/kaggle/working/code'...
✓ Code loaded from GitHub
```

**Cell 3:**
```
NumPy version: 1.26.4
GPU Available: True
GPU Name: Tesla T4
GPU Memory: 15.0 GB
✓ All imports successful
```

**Cell 4:**
```
✓ Directories created
```

**Cell 5:**
```
Using device: cuda
Loaded environment: highway_heterogeneous
...
[Training progress]
...
✓ Training complete!
```

**Cell 6:**
```
✓ Models archived: training_results.zip
✓ Logs archived: training_logs.zip

✓ Download from Output tab:
  - training_results.zip
  - training_logs.zip
```

---

## Troubleshooting

### If NumPy error persists:

```python
# Force reinstall NumPy 1.x
!pip install --force-reinstall "numpy==1.26.4"

# Restart kernel (Kernel → Restart)
# Then re-run all cells
```

### If highway-env import fails:

```python
# Reinstall highway-env
!pip uninstall -y highway-env
!pip install highway-env==1.8.2
```

### If CUDA out of memory:

```python
# Reduce batch size in training
# Edit train_mappo.py or use smaller horizon:
!python training/train_mappo.py --horizon 128 --steps 20000
```

---

## Quick Copy-Paste Version

For fastest setup, copy this single cell:

```python
# All-in-one setup cell
import subprocess, sys, os

# Fix NumPy
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2.0"])

# Install deps
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                      "gymnasium", "highway-env", "transformers", "torch", 
                      "pillow", "pyyaml", "tqdm", "tensorboard", "opencv-python"])

# Clone code (REPLACE WITH YOUR REPO!)
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /kaggle/working/code
os.chdir('/kaggle/working/code')
sys.path.insert(0, '/kaggle/working/code')

# Verify
import torch, numpy as np
print(f"✓ NumPy: {np.__version__}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Create dirs
os.makedirs('/kaggle/working/logs', exist_ok=True)

print("✓ Setup complete! Run training cell next.")
```

Then run training cell separately.

---

**Remember:** Replace `YOUR_USERNAME/YOUR_REPO` with your actual GitHub repository URL!
