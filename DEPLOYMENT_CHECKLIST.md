# Deployment Checklist - Ubuntu + RTX 4090

Use this checklist to ensure smooth deployment on your HPC laptop.

## Pre-Deployment (On Development Machine)

- [ ] All code committed to git repository
- [ ] `.gitignore` configured (excludes data/, models/, runs/)
- [ ] `requirements.txt` up to date
- [ ] `setup.sh` executable (`chmod +x setup.sh`)
- [ ] `run.sh` executable (`chmod +x run.sh`)
- [ ] Documentation complete (README.md, SETUP_GUIDE.md, QUICK_START.md)

## System Requirements Check (On HPC Laptop)

- [ ] Ubuntu 20.04+ installed
- [ ] NVIDIA RTX 4090 detected (`nvidia-smi`)
- [ ] NVIDIA drivers installed (version 535+)
- [ ] CUDA 11.8+ available
- [ ] 32GB+ RAM available
- [ ] 50GB+ free disk space
- [ ] Internet connection (for downloading dependencies)

## Initial Setup (5 minutes)

```bash
# 1. Clone repository
[ ] git clone <your-repo-url> vlm-marl-highway
[ ] cd vlm-marl-highway

# 2. Verify files
[ ] ls -la setup.sh run.sh requirements.txt
[ ] ls -la configs/ data_collection/ training/ models/

# 3. Run automated setup
[ ] chmod +x setup.sh
[ ] ./setup.sh
# Wait ~5 minutes for completion

# 4. Verify setup
[ ] source venv/bin/activate
[ ] python -c "import torch; assert torch.cuda.is_available()"
[ ] nvidia-smi  # Should show RTX 4090
```

## Environment Verification

```bash
# Activate environment
[ ] source venv/bin/activate
[ ] source gpu_config.sh

# Check Python packages
[ ] python -c "import torch; print(f'PyTorch: {torch.__version__}')"
[ ] python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
[ ] python -c "import highway_env; print('Highway-env: OK')"
[ ] python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check GPU
[ ] python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
[ ] python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
[ ] python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"

# Quick test
[ ] python data_collection/collect_data.py --scenario highway_heterogeneous --episodes 1 --max-steps 10 --output-dir test_output
[ ] ls test_output/highway_heterogeneous/  # Should have dataset.jsonl and images/
[ ] rm -rf test_output  # Clean up
```

## Data Collection Test (5 minutes)

```bash
# Test normal spacing
[ ] python data_collection/collect_data.py \
    --scenario highway_heterogeneous \
    --episodes 5 \
    --max-steps 100 \
    --output-dir data/test_normal \
    --seed 42

# Test dense spacing
[ ] python data_collection/collect_data.py \
    --scenario highway_heterogeneous_dense \
    --episodes 5 \
    --max-steps 100 \
    --output-dir data/test_dense \
    --seed 43

# Verify outputs
[ ] ls data/test_normal/highway_heterogeneous/dataset.jsonl
[ ] ls data/test_dense/highway_heterogeneous_dense/dataset.jsonl
[ ] python verify_agent_spacing.py data/test_normal/highway_heterogeneous/dataset.jsonl

# Clean up test data
[ ] rm -rf data/test_*
```

## GPU Performance Test (2 minutes)

```bash
# Test GPU memory and speed
[ ] python -c "
import torch
import time

# Test GPU allocation
x = torch.randn(10000, 10000).cuda()
print(f'✓ GPU allocation: {x.device}')

# Test GPU speed
start = time.time()
for _ in range(100):
    y = torch.matmul(x, x)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f'✓ GPU speed: {elapsed:.2f}s for 100 matrix multiplications')
print(f'✓ GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f}GB')
"

# Monitor GPU
[ ] nvidia-smi  # Check GPU utilization, memory, temperature
```

## Full Pipeline Test (Optional - 1 hour)

```bash
# Test with small dataset (10 episodes)
[ ] python data_collection/collect_data.py \
    --scenario highway_heterogeneous \
    --episodes 10 \
    --max-steps 100 \
    --output-dir data/test_pipeline

# Test CLIP training (1 epoch)
[ ] python training/train_clip.py \
    --data_dirs data/test_pipeline \
    --epochs 1 \
    --batch_size 32 \
    --save_dir models/test_clip

# Test MAPPO training (1000 steps)
[ ] python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --steps 1000 \
    --clip_dir models/test_clip

# Verify TensorBoard
[ ] tensorboard --logdir runs --port 6006 &
[ ] curl http://localhost:6006  # Should return HTML
[ ] pkill -f tensorboard

# Clean up
[ ] rm -rf data/test_pipeline models/test_clip runs/*
```

## Production Deployment

### Phase 1: Data Collection (45 minutes)

```bash
[ ] source venv/bin/activate && source gpu_config.sh
[ ] export PYTHONPATH=.

# Normal spacing (70 episodes)
[ ] python data_collection/collect_data.py \
    --scenario highway_heterogeneous \
    --episodes 70 \
    --max-steps 1000 \
    --output-dir data/highway_heterogeneous_normal \
    --log-dir runs/data_collection_normal \
    --seed 42

# Dense spacing (30 episodes)
[ ] python data_collection/collect_data.py \
    --scenario highway_heterogeneous_dense \
    --episodes 30 \
    --max-steps 1000 \
    --output-dir data/highway_heterogeneous_dense \
    --log-dir runs/data_collection_dense \
    --seed 43

# Merge datasets
[ ] mkdir -p data/highway_heterogeneous/images
[ ] cat data/highway_heterogeneous_normal/dataset.jsonl \
    data/highway_heterogeneous_dense/dataset.jsonl \
    > data/highway_heterogeneous/dataset.jsonl
[ ] cp -r data/highway_heterogeneous_normal/images/* data/highway_heterogeneous/images/
[ ] cp -r data/highway_heterogeneous_dense/images/* data/highway_heterogeneous/images/

# Verify diversity
[ ] python verify_agent_spacing.py data/highway_heterogeneous/dataset.jsonl
# Expected: 150-200 unique instructions, 15-20% yielding rate
```

### Phase 2: CLIP Fine-tuning (1-2 hours)

```bash
[ ] python training/train_clip.py \
    --data_dirs data/highway_heterogeneous \
    --model_id openai/clip-vit-base-patch32 \
    --epochs 5 \
    --batch_size 64 \
    --lr 1e-5 \
    --num_workers 4 \
    --log_dir runs/clip_hetero \
    --save_dir models/clip_hetero

# Monitor GPU during training
[ ] watch -n 1 nvidia-smi  # In separate terminal

# Verify model saved
[ ] ls models/clip_hetero/  # Should have model files
```

### Phase 3: MAPPO Training (3-4 hours)

```bash
[ ] python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --steps 100000 \
    --horizon 256 \
    --logdir runs/mappo_hetero \
    --clip_dir models/clip_hetero \
    --lr 3e-4

# Monitor training
[ ] tensorboard --logdir runs --port 6006 --bind_all &
# Open browser: http://localhost:6006

# Verify model saved
[ ] ls models/vla_mappo_heterogeneous_*.pth
```

### Phase 4: Evaluation (30 minutes)

```bash
# Run evaluation (if eval script exists)
[ ] python evaluation/eval_mappo.py \
    --model models/vla_mappo_heterogeneous_*.pth \
    --scenario highway_heterogeneous \
    --episodes 50

# Check results
[ ] cat evaluation_results.json  # Or wherever results are saved
```

## Post-Deployment Verification

### Check Outputs

```bash
# Data
[ ] ls -lh data/highway_heterogeneous/dataset.jsonl  # Should be ~50-100MB
[ ] ls data/highway_heterogeneous/images/ | wc -l    # Should be ~700-900 images

# Models
[ ] ls -lh models/clip_hetero/  # Should have model files
[ ] ls -lh models/vla_mappo_*.pth  # Should be ~50-200MB

# Logs
[ ] ls runs/  # Should have clip_hetero/ and mappo_hetero/
[ ] tensorboard --logdir runs --port 6006  # Verify logs load
```

### Performance Metrics

```bash
# Check TensorBoard metrics
[ ] Open http://localhost:6006
[ ] Verify CLIP training loss decreasing
[ ] Verify MAPPO episode return increasing
[ ] Check GPU utilization was 70-95%
```

### Expected Results

- [ ] CLIP training loss: ~0.5-0.8 (final)
- [ ] CLIP accuracy: 88-92%
- [ ] MAPPO episode return: increasing trend
- [ ] MAPPO success rate: 85-92%
- [ ] Collision rate: <10%
- [ ] GPU memory usage: 8-12GB (CLIP), 4-6GB (MAPPO)
- [ ] Total training time: 6-8 hours

## Troubleshooting

### If Setup Fails

```bash
[ ] Check NVIDIA drivers: nvidia-smi
[ ] Check Python version: python3 --version  # Should be 3.8-3.11
[ ] Check disk space: df -h  # Need 50GB+
[ ] Check internet: ping google.com
[ ] Re-run setup: rm -rf venv && ./setup.sh
```

### If Training Fails

```bash
[ ] Check GPU: nvidia-smi
[ ] Check CUDA: python -c "import torch; print(torch.cuda.is_available())"
[ ] Check logs: tail -f runs/*/events.out.tfevents.*
[ ] Reduce batch size: --batch_size 32 (CLIP), --horizon 128 (MAPPO)
[ ] Check disk space: df -h
```

### If OOM Errors

```bash
[ ] Reduce CLIP batch size: --batch_size 32
[ ] Reduce MAPPO horizon: --horizon 128
[ ] Close other GPU applications
[ ] Check GPU memory: nvidia-smi
```

## Backup and Archiving

```bash
# Backup trained models
[ ] tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Backup collected data
[ ] tar -czf data_backup_$(date +%Y%m%d).tar.gz data/

# Backup TensorBoard logs
[ ] tar -czf runs_backup_$(date +%Y%m%d).tar.gz runs/

# Copy to safe location
[ ] cp *_backup_*.tar.gz /path/to/backup/
```

## Final Checklist

- [ ] All phases completed successfully
- [ ] Models saved and verified
- [ ] TensorBoard logs accessible
- [ ] Performance metrics meet expectations
- [ ] Backups created
- [ ] Documentation updated with results
- [ ] Ready for paper writing! 🎉

## Notes

- Total setup time: ~5 minutes
- Total training time: ~6-8 hours
- Total disk usage: ~20-30GB (data + models + logs)
- GPU memory usage: 8-12GB peak
- Expected success rate: 85-92%

## Support

If you encounter issues:
1. Check SETUP_GUIDE.md for detailed troubleshooting
2. Check TensorBoard logs: `tensorboard --logdir runs`
3. Check GPU status: `nvidia-smi`
4. Check system logs: `dmesg | tail -50`

Your RTX 4090 is perfect for this project! 🚀
