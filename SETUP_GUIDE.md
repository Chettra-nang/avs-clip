# VLM-MARL Highway Setup Guide - Ubuntu + RTX 4090

Complete setup instructions for running the VLM-MARL ambulance priority system on Ubuntu Linux with NVIDIA RTX 4090.

## System Requirements

- **OS**: Ubuntu 20.04+ (tested on 22.04)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free space
- **CUDA**: 11.8+ (will be installed)
- **Python**: 3.8-3.11

## Quick Start (5 minutes)

```bash
# 1. Clone the repository
git clone <your-repo-url> vlm-marl-highway
cd vlm-marl-highway

# 2. Run automated setup
chmod +x setup.sh
./setup.sh

# 3. Activate environment
source venv/bin/activate

# 4. Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 5. Run quick test
python data_collection/collect_data.py --scenario highway_heterogeneous --episodes 1 --max-steps 10 --output-dir test_output
```

## Detailed Setup Instructions

### Step 1: System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and build tools
sudo apt install -y python3.10 python3.10-venv python3-pip
sudo apt install -y build-essential git wget curl

# Install NVIDIA drivers (if not already installed)
sudo apt install -y nvidia-driver-535  # Or latest stable version
sudo reboot  # Reboot after driver installation

# Verify NVIDIA driver
nvidia-smi  # Should show RTX 4090 with CUDA 12.x
```

### Step 2: Clone Repository

```bash
# Clone the project
git clone <your-repo-url> vlm-marl-highway
cd vlm-marl-highway

# Check project structure
ls -la
# Should see: configs/, data_collection/, models/, training/, etc.
```

### Step 3: Python Environment Setup

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 (compatible with RTX 4090)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch: 2.1.0+cu118
CUDA: True
GPU: NVIDIA GeForce RTX 4090
```

### Step 4: Verify Installation

```bash
# Test data collection (quick)
python data_collection/collect_data.py \
    --scenario highway_heterogeneous \
    --episodes 1 \
    --max-steps 10 \
    --output-dir test_output \
    --seed 42

# Should complete in ~5 seconds and create test_output/highway_heterogeneous/
```

### Step 5: GPU Optimization for RTX 4090

```bash
# Create GPU config file
cat > gpu_config.sh << 'EOF'
#!/bin/bash
# GPU optimization for RTX 4090

# Enable TF32 for faster training (RTX 30/40 series)
export NVIDIA_TF32_OVERRIDE=1

# Set memory growth to avoid OOM
export TF_FORCE_GPU_ALLOW_GROWTH=true

# CUDA optimization
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090 architecture

# cuDNN optimization
export CUDNN_BENCHMARK=1

echo "GPU optimizations enabled for RTX 4090"
EOF

chmod +x gpu_config.sh

# Source before training
source gpu_config.sh
```

## Running the Full Pipeline

### Option 1: Automated Pipeline (Recommended)

```bash
# Activate environment
source venv/bin/activate
source gpu_config.sh

# Run full heterogeneous pipeline
./run.sh highway_heterogeneous --heterogeneous

# This will:
# 1. Collect 100 episodes (70 normal + 30 dense) - ~45 min
# 2. Fine-tune CLIP - ~1-2 hours
# 3. Train MAPPO - ~3-4 hours
# 4. Save models to models/
```

### Option 2: Step-by-Step Execution

```bash
# Activate environment
source venv/bin/activate
source gpu_config.sh
export PYTHONPATH=.

# Step 1: Data Collection (45 minutes)
echo "=== Step 1: Data Collection ==="

# Normal spacing (70 episodes)
python data_collection/collect_data.py \
    --scenario highway_heterogeneous \
    --episodes 70 \
    --max-steps 1000 \
    --output-dir data/highway_heterogeneous_normal \
    --log-dir runs/data_collection_normal \
    --seed 42

# Dense spacing (30 episodes)
python data_collection/collect_data.py \
    --scenario highway_heterogeneous_dense \
    --episodes 30 \
    --max-steps 1000 \
    --output-dir data/highway_heterogeneous_dense \
    --log-dir runs/data_collection_dense \
    --seed 43

# Merge datasets
mkdir -p data/highway_heterogeneous/images
cat data/highway_heterogeneous_normal/dataset.jsonl \
    data/highway_heterogeneous_dense/dataset.jsonl \
    > data/highway_heterogeneous/dataset.jsonl
cp -r data/highway_heterogeneous_normal/images/* data/highway_heterogeneous/images/
cp -r data/highway_heterogeneous_dense/images/* data/highway_heterogeneous/images/

# Verify diversity
python verify_agent_spacing.py data/highway_heterogeneous/dataset.jsonl

# Step 2: CLIP Fine-tuning (1-2 hours)
echo "=== Step 2: CLIP Fine-tuning ==="
python training/train_clip.py \
    --data_dirs data/highway_heterogeneous \
    --model_id openai/clip-vit-base-patch32 \
    --epochs 5 \
    --batch_size 64 \
    --lr 1e-5 \
    --num_workers 4 \
    --log_dir runs/clip_hetero \
    --save_dir models/clip_hetero

# Step 3: MAPPO Training (3-4 hours)
echo "=== Step 3: MAPPO Training ==="
python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --steps 100000 \
    --horizon 256 \
    --logdir runs/mappo_hetero \
    --clip_dir models/clip_hetero \
    --lr 3e-4

# Step 4: Monitor Training
tensorboard --logdir runs --port 6006
# Open browser: http://localhost:6006
```

## Performance Optimization for RTX 4090

### Batch Size Tuning

The RTX 4090 has 24GB VRAM. Recommended batch sizes:

```bash
# CLIP Training
--batch_size 128  # Can go up to 256 with RTX 4090

# MAPPO Training
--horizon 512     # Can increase from default 256
```

### Multi-GPU Setup (if you have multiple GPUs)

```bash
# Check available GPUs
nvidia-smi

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Or use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs
```

### Memory Optimization

```bash
# If you encounter OOM errors, reduce batch size:
python training/train_clip.py --batch_size 32  # Instead of 64
python training/train_mappo.py --horizon 128   # Instead of 256

# Enable gradient checkpointing (saves memory)
# Add to training scripts if needed
```

## Monitoring and Debugging

### Check GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvtop (more detailed)
sudo apt install nvtop
nvtop
```

### TensorBoard Monitoring

```bash
# Start TensorBoard
tensorboard --logdir runs --port 6006 --bind_all

# Access from browser
# Local: http://localhost:6006
# Remote: http://<your-ip>:6006
```

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

```bash
# Solution: Reduce batch size
python training/train_clip.py --batch_size 32
python training/train_mappo.py --horizon 128
```

#### Issue 2: Slow Data Collection

```bash
# Solution: Reduce max_steps or use fewer episodes for testing
python data_collection/collect_data.py --episodes 10 --max-steps 100
```

#### Issue 3: Import Errors

```bash
# Solution: Set PYTHONPATH
export PYTHONPATH=.
# Or add to ~/.bashrc:
echo 'export PYTHONPATH=.' >> ~/.bashrc
```

#### Issue 4: Highway-env Rendering Issues

```bash
# Solution: Install additional dependencies
sudo apt install -y python3-opengl freeglut3-dev
pip install pyglet==1.5.27
```

## Expected Training Times (RTX 4090)

| Phase | Episodes/Steps | Time | GPU Utilization |
|-------|---------------|------|-----------------|
| Data Collection | 100 episodes | 45 min | 0% (CPU only) |
| CLIP Fine-tuning | 5 epochs | 1-2 hours | 80-95% |
| MAPPO Training | 100k steps | 3-4 hours | 70-85% |
| **Total** | - | **5-7 hours** | - |

## Verification Checklist

After setup, verify everything works:

```bash
# 1. Python environment
python --version  # Should be 3.8-3.11

# 2. PyTorch + CUDA
python -c "import torch; assert torch.cuda.is_available(); print('✓ PyTorch CUDA OK')"

# 3. Dependencies
python -c "import gymnasium, highway_env, transformers, PIL; print('✓ Dependencies OK')"

# 4. Project structure
ls configs/ data_collection/ models/ training/  # Should all exist

# 5. Quick test
python data_collection/collect_data.py --scenario highway_heterogeneous --episodes 1 --max-steps 10 --output-dir test_output
# Should complete without errors

# 6. GPU test
python -c "import torch; x = torch.randn(1000, 1000).cuda(); print(f'✓ GPU test OK: {x.device}')"
```

All checks should pass ✓

## Project Structure

```
vlm-marl-highway/
├── configs/
│   └── envs/
│       ├── highway.yaml
│       ├── highway_heterogeneous.yaml
│       ├── highway_heterogeneous_dense.yaml
│       ├── merge.yaml
│       ├── merge_heterogeneous.yaml
│       ├── intersection.yaml
│       └── intersection_heterogeneous.yaml
├── data_collection/
│   ├── collect_data.py
│   ├── instruction_generator.py
│   └── README.md
├── training/
│   ├── train_clip.py
│   └── train_mappo.py
├── models/
│   ├── clip_encoder.py
│   ├── vla_mappo.py
│   └── heterogeneous_mappo.py
├── common/
│   ├── utils.py
│   ├── logger_tb.py
│   └── scenario.py
├── envs/
│   └── make_env.py
├── requirements.txt
├── run.sh
├── README.md
└── verify_agent_spacing.py
```

## Next Steps

After successful setup:

1. **Collect full dataset** (45 min)
2. **Train CLIP** (1-2 hours)
3. **Train MAPPO** (3-4 hours)
4. **Evaluate results** (30 min)
5. **Visualize in TensorBoard**

## Support

If you encounter issues:

1. Check GPU: `nvidia-smi`
2. Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check logs: `tail -f runs/*/events.out.tfevents.*`
4. Check TensorBoard: `tensorboard --logdir runs`

## Performance Benchmarks (RTX 4090)

Expected performance on RTX 4090:

- **Data Collection**: ~0.5 episodes/min (CPU-bound)
- **CLIP Training**: ~200 samples/sec
- **MAPPO Training**: ~1000 steps/min
- **GPU Memory Usage**: 8-12GB (CLIP), 4-6GB (MAPPO)
- **GPU Utilization**: 80-95% (CLIP), 70-85% (MAPPO)

Your RTX 4090 is perfect for this project! 🚀
