# Kaggle Training Checklist

Complete checklist to train on Kaggle GPU. Follow in order!

## ✅ Pre-Training (Do Once)

### 1. Finish Local CLIP Fine-tuning
- [ ] Wait for CLIP training to complete (~3-4 hours)
- [ ] Verify `models/clip_finetuned/` exists
- [ ] Check model files are saved

### 2. Prepare GitHub Repository
```bash
# Add fine-tuned CLIP to git
git add models/clip_finetuned/
git add -A
git commit -m "Add fine-tuned CLIP model for Kaggle training"
git push origin main
```

- [ ] Code pushed to GitHub
- [ ] Repository is public (or Kaggle has access)
- [ ] Note your GitHub URL: `https://github.com/YOUR_USERNAME/YOUR_REPO`

### 3. Create Kaggle Account
- [ ] Sign up at https://www.kaggle.com (free)
- [ ] Verify email
- [ ] Complete profile

## 🚀 Kaggle Setup (5 minutes)

### 4. Create New Notebook
- [ ] Go to https://www.kaggle.com/code
- [ ] Click "New Notebook"
- [ ] Name it: "Ambulance Priority MAPPO Training"

### 5. Configure Notebook Settings
- [ ] Click Settings (gear icon, right sidebar)
- [ ] **Accelerator:** Select "GPU T4 x2" ⚠️ CRITICAL!
- [ ] **Internet:** Turn ON ⚠️ CRITICAL!
- [ ] **Persistence:** Turn ON (optional, keeps session alive)
- [ ] Click "Save"

### 6. Verify GPU
Add this cell and run:
```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

- [ ] Output shows "GPU: True"
- [ ] GPU name shows "Tesla T4" or similar

## 📝 Training Cells (Copy & Paste)

### Cell 1: Install Dependencies
```python
# Fix NumPy compatibility (CRITICAL!)
!pip uninstall -y numpy
!pip install -q "numpy<2.0"

# Install dependencies
!pip install -q gymnasium highway-env transformers torch torchvision
!pip install -q pillow pyyaml tqdm tensorboard opencv-python
print("✓ Dependencies installed")
```

- [ ] Cell runs without errors
- [ ] See "✓ Dependencies installed"
- [ ] NumPy downgraded to 1.x

### Cell 2: Clone Code
```python
import os, sys
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /kaggle/working/code
%cd /kaggle/working/code
sys.path.insert(0, '/kaggle/working/code')
print("✓ Code loaded")
```

- [ ] Replace `YOUR_USERNAME/YOUR_REPO` with your actual GitHub URL
- [ ] Cell runs without errors
- [ ] See "✓ Code loaded"

### Cell 3: Verify Setup
```python
import torch
import gymnasium as gym
import highway_env
import custom_envs

print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("✓ All imports successful")
```

- [ ] No import errors
- [ ] GPU shows ~16 GB VRAM

### Cell 4: Create Directories
```python
import os
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/logs', exist_ok=True)
print("✓ Directories created")
```

- [ ] Directories created

### Cell 5: Start Training
```python
!PYTHONPATH=/kaggle/working/code python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --clip_dir models/clip_finetuned \
    --steps 20000 \
    --horizon 256 \
    --lr 3e-4 \
    --device cuda \
    --logdir /kaggle/working/logs

print("✓ Training complete!")
```

- [ ] Training starts
- [ ] See progress updates
- [ ] No CUDA out of memory errors

## ⏱️ During Training (2-3 hours)

### Monitor Progress
- [ ] Check progress bar updates
- [ ] Loss decreases over time
- [ ] No errors in output

### Optional: Monitor Remotely
- [ ] Enable session persistence (Settings)
- [ ] Can close browser, training continues
- [ ] Check back in 2-3 hours

## 💾 After Training

### Cell 6: Archive Results
```python
import shutil
shutil.make_archive('/kaggle/working/training_results', 'zip', 
                    '/kaggle/working/code/models')
shutil.make_archive('/kaggle/working/training_logs', 'zip',
                    '/kaggle/working/logs')
print("✓ Results archived")
print("Download from Output tab:")
print("  - training_results.zip")
print("  - training_logs.zip")
```

- [ ] Archives created
- [ ] No errors

### Download Results
- [ ] Click "Output" tab (top right)
- [ ] Download `training_results.zip`
- [ ] Download `training_logs.zip`
- [ ] Save to local machine

## 🏠 Local Evaluation

### Extract Results
```bash
cd ~/Desktop/avs
unzip ~/Downloads/training_results.zip -d models/
unzip ~/Downloads/training_logs.zip -d runs/
```

- [ ] Files extracted
- [ ] Model checkpoint exists: `models/vla_mappo_heterogeneous_highway.pth`

### Evaluate Model
```bash
PYTHONPATH=. python evaluation/eval_mappo.py \
    --model models/vla_mappo_heterogeneous_highway.pth \
    --scenario highway_heterogeneous \
    --episodes 100 \
    --render
```

- [ ] Evaluation runs
- [ ] See success rate, collision rate
- [ ] Results look reasonable

### View TensorBoard Logs
```bash
tensorboard --logdir runs/
# Open http://localhost:6006
```

- [ ] TensorBoard opens
- [ ] See training curves
- [ ] Loss decreases over time

## 🎉 Success Criteria

Training is successful if:
- [ ] No CUDA out of memory errors
- [ ] Training completes all steps
- [ ] Model checkpoint saved
- [ ] Success rate > 10% (for 20k steps)
- [ ] Loss decreases from ~4.0 to ~1.5-2.0

## 🔧 Troubleshooting

### GPU Not Available
- [ ] Check Settings → Accelerator → GPU T4 x2
- [ ] Restart notebook
- [ ] Try different time (GPU quota may be exhausted)

### Out of Memory
- [ ] Reduce batch size in code
- [ ] Reduce horizon: `--horizon 128`
- [ ] Use smaller model

### Code Not Found
- [ ] Check GitHub URL is correct
- [ ] Repository is public
- [ ] Internet is enabled in Settings

### Training Too Slow
- [ ] Verify GPU is being used (check nvidia-smi)
- [ ] Check batch size isn't too small
- [ ] Ensure CUDA is enabled

## 📊 Expected Timeline

| Step | Time | Total |
|------|------|-------|
| Setup Kaggle | 5 min | 5 min |
| Install deps | 2 min | 7 min |
| Clone code | 1 min | 8 min |
| Training (20k) | 2-3 hrs | 3 hrs |
| Download | 2 min | 3 hrs |
| **Total** | | **~3 hours** |

## 🔄 For Multiple Runs

To train for 100k steps (split into sessions):

### Session 1: 0-20k
```bash
--steps 20000 --save_interval 5000
```

### Session 2: 20k-40k
```bash
--steps 40000 --resume_from checkpoints/checkpoint_20000.pt
```

### Session 3: 40k-60k
```bash
--steps 60000 --resume_from checkpoints/checkpoint_40000.pt
```

Continue until 100k steps reached.

## ✅ Final Checklist

Before closing Kaggle:
- [ ] Training completed successfully
- [ ] Results downloaded
- [ ] Logs downloaded
- [ ] Notebook saved (optional, for reference)
- [ ] GPU session ended (to save quota)

## 🎯 Next Steps

After successful Kaggle training:
1. [ ] Evaluate model locally
2. [ ] Compare with baseline (no vision)
3. [ ] Try DINOv2 if results are poor
4. [ ] Write paper with results
5. [ ] Celebrate! 🎉

---

**Estimated total time:** 3-4 hours (mostly waiting for training)

**GPU hours used:** 2-3 hours (out of 30 free hours/week)

**Cost:** $0 (completely free!)

Good luck with your training! 🚀
