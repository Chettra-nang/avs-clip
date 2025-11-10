# Quick Start Guide - Ubuntu + RTX 4090

## 🚀 5-Minute Setup

```bash
# 1. Clone repository
git clone <your-repo-url> vlm-marl-highway
cd vlm-marl-highway

# 2. Run automated setup
chmod +x setup.sh
./setup.sh

# 3. Activate environment
source venv/bin/activate
source gpu_config.sh

# 4. Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

## 📊 Run Full Pipeline (6-8 hours)

```bash
# Activate environment
source venv/bin/activate
source gpu_config.sh

# Run complete pipeline
./run.sh highway_heterogeneous --heterogeneous

# Monitor progress
tensorboard --logdir runs --port 6006
# Open: http://localhost:6006
```

## 🎯 Step-by-Step Execution

### 1. Data Collection (45 minutes)

```bash
source venv/bin/activate
export PYTHONPATH=.

# Normal spacing (70 episodes)
python data_collection/collect_data.py \
    --scenario highway_heterogeneous \
    --episodes 70 \
    --max-steps 1000 \
    --output-dir data/highway_heterogeneous_normal

# Dense spacing (30 episodes)
python data_collection/collect_data.py \
    --scenario highway_heterogeneous_dense \
    --episodes 30 \
    --max-steps 1000 \
    --output-dir data/highway_heterogeneous_dense

# Merge datasets
mkdir -p data/highway_heterogeneous/images
cat data/highway_heterogeneous_normal/dataset.jsonl \
    data/highway_heterogeneous_dense/dataset.jsonl \
    > data/highway_heterogeneous/dataset.jsonl
cp -r data/highway_heterogeneous_normal/images/* data/highway_heterogeneous/images/
cp -r data/highway_heterogeneous_dense/images/* data/highway_heterogeneous/images/

# Verify
python verify_agent_spacing.py data/highway_heterogeneous/dataset.jsonl
```

### 2. CLIP Fine-tuning (1-2 hours)

```bash
python training/train_clip.py \
    --data_dirs data/highway_heterogeneous \
    --model_id openai/clip-vit-base-patch32 \
    --epochs 5 \
    --batch_size 64 \
    --lr 1e-5 \
    --num_workers 4 \
    --log_dir runs/clip_hetero \
    --save_dir models/clip_hetero
```

### 3. MAPPO Training (3-4 hours)

```bash
python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --steps 100000 \
    --horizon 256 \
    --logdir runs/mappo_hetero \
    --clip_dir models/clip_hetero \
    --lr 3e-4
```

### 4. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir runs --port 6006 --bind_all

# Watch GPU usage
watch -n 1 nvidia-smi
```

## 🔧 RTX 4090 Optimizations

```bash
# Enable GPU optimizations
source gpu_config.sh

# Use larger batch sizes (RTX 4090 has 24GB VRAM)
python training/train_clip.py --batch_size 128  # Instead of 64
python training/train_mappo.py --horizon 512    # Instead of 256
```

## 📈 Expected Performance

| Phase | Time | GPU Usage | Memory |
|-------|------|-----------|--------|
| Data Collection | 45 min | 0% (CPU) | - |
| CLIP Training | 1-2 hours | 80-95% | 8-12GB |
| MAPPO Training | 3-4 hours | 70-85% | 4-6GB |

## ✅ Verification

```bash
# Check everything is working
python -c "
import torch
import gymnasium
import highway_env
import transformers
from PIL import Image

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
print('✓ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('✓ Gymnasium:', gymnasium.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ All dependencies OK!')
"
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python training/train_clip.py --batch_size 32
python training/train_mappo.py --horizon 128
```

### Import Errors
```bash
export PYTHONPATH=.
# Or add to ~/.bashrc:
echo 'export PYTHONPATH=.' >> ~/.bashrc
```

### Slow Training
```bash
# Check GPU usage
nvidia-smi
# Should show 70-95% utilization

# Enable optimizations
source gpu_config.sh
```

## 📁 Project Structure

```
vlm-marl-highway/
├── configs/envs/          # Environment configurations
├── data_collection/       # Data collection scripts
├── training/              # CLIP and MAPPO training
├── models/                # Model architectures
├── common/                # Utilities
├── data/                  # Collected datasets (created)
├── runs/                  # TensorBoard logs (created)
├── requirements.txt       # Python dependencies
├── setup.sh              # Automated setup script
└── run.sh                # Full pipeline script
```

## 🎓 Documentation

- **Full Setup**: See `SETUP_GUIDE.md`
- **README**: See `README.md`
- **Diversity Analysis**: See `DIVERSITY_IMPROVEMENT_SUMMARY.md`
- **Task Completion**: See `TASK_11_COMPLETION_SUMMARY.md`

## 🚀 Ready to Train!

Your RTX 4090 is perfect for this project. Expected results:
- **CLIP Accuracy**: 88-92%
- **MAPPO Success Rate**: 85-92%
- **Total Training Time**: 6-8 hours

Good luck! 🎉
