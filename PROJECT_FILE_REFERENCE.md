# 📁 VLM-MARL Highway Project - Complete File Reference

## 🏗️ Core Architecture

### Models (`models/`)

| File | Purpose | Key Features |
|------|---------|--------------|
| `heterogeneous_mappo.py` | Base MAPPO implementation for heterogeneous agents | Multi-agent PPO, shared/separate policies, ambulance priority |
| `vla_mappo.py` | Vision-Language-Action MAPPO with CLIP | CLIP ViT-B/32 encoder, text instruction processing, vision-language fusion |
| `clip_encoder.py` | CLIP vision encoder wrapper | Pre-trained/fine-tuned CLIP, feature extraction, frozen/trainable modes |
| `dinov2_encoder.py` | DINOv2 vision encoder | Self-supervised ViT, multiple variants (small/base/large), frozen backbone |
| `dinov2_mappo.py` | MAPPO with DINOv2 vision encoder | Superior visual features, no language alignment, pure vision approach |
| `hybrid_clip_dinov2.py` | Hybrid CLIP + DINOv2 fusion model | Combines vision-language + superior vision, multiple fusion strategies |

### Training Scripts (`training/`)

| File | Purpose | Key Features |
|------|---------|--------------|
| `train_mappo.py` | Main training script for all models | Supports CLIP/DINOv2/hybrid encoders, multi-agent coordination, TensorBoard logging |
| `train_clip.py` | CLIP fine-tuning on highway data | Vision-language alignment, instruction-image pairs, contrastive learning |
| `train_clip_human_aligned.py` | Improved CLIP fine-tuning | Freezes vision encoder, preserves visual features, prevents degradation |

### Environments (`envs/`, `custom_envs/`)

| File | Purpose | Key Features |
|------|---------|--------------|
| `envs/make_env.py` | Environment factory and wrapper | Creates highway-env instances, applies wrappers, handles multi-agent setup |
| `custom_envs/multi_agent_merge_env.py` | Fixed multi-agent merge environment | 4-vehicle support, proper controlled_vehicles handling, custom spawn logic |
| `custom_envs/README.md` | Custom environment documentation | Registration guide, configuration examples, usage instructions |

### Configuration (`configs/envs/`)

| File | Purpose | Key Features |
|------|---------|--------------|
| `highway.yaml` | Standard highway scenario | 4 lanes, high-speed driving, lane changes |
| `highway_heterogeneous_dense.yaml` | Dense highway with heterogeneous agents | Ambulance + regular vehicles, priority rewards, dense traffic |
| `intersection.yaml` | Intersection scenario | Traffic lights, turn coordination, collision avoidance |
| `intersection_heterogeneous.yaml` | Intersection with ambulance | Emergency vehicle priority, traffic yielding |
| `merge.yaml` | Official merge scenario | Single-agent merging task |
| `merge_multi_agent.yaml` | Custom 4-agent merge | Fixed multi-agent support, proper vehicle spawning |
| `merge_heterogeneous.yaml` | Merge with ambulance priority | Emergency vehicle in merge scenario |

### Common Utilities (`common/`)

| File | Purpose | Key Features |
|------|---------|--------------|
| `scenario.py` | Scenario definitions and rewards | Ambulance detection, priority rewards, collision penalties |
| `utils.py` | Utility functions | Observation processing, action handling, helper functions |
| `logger_tb.py` | TensorBoard logging | Training metrics, episode statistics, visualization |

### Data Collection (`data_collection/`)

| File | Purpose | Key Features |
|------|---------|--------------|
| `collect_data.py` | Collect vision-language training data | Captures highway images, generates human-aligned instructions, 90 diverse templates |
| `README.md` | Data collection documentation | Usage guide, template examples, dataset structure |

### Evaluation (`evaluation/`)

| File | Purpose | Key Features |
|------|---------|--------------|
| `validate_clip.py` | CLIP model validation and comparison | Pre-trained vs fine-tuned comparison, cosine similarity metrics, performance analysis |

---

## 📚 Documentation

### Quick Start Guides

| File | Purpose | Target Audience |
|------|---------|-----------------|
| `README.md` | Main project documentation | All users - comprehensive overview, setup, usage |
| `DINOV2_QUICK_START.md` | DINOv2 training guide | Users wanting superior vision encoder |
| `KAGGLE_QUICK_START.md` | Kaggle GPU setup | Users needing free GPU training |
| `VALIDATE_CLIP_QUICK.md` | Quick CLIP validation | Users checking CLIP performance |

### Training Guides

| File | Purpose | Key Information |
|------|---------|-----------------|
| `KAGGLE_TRAINING_GUIDE.md` | Complete Kaggle training setup | Notebook creation, GPU setup, data upload, training execution |
| `KAGGLE_CLIP_TRAINING.md` | CLIP fine-tuning on Kaggle | CLIP-specific training, dataset preparation |
| `TRAINING_OPTIONS_SUMMARY.md` | Overview of all training approaches | Comparison of CLIP/DINOv2/hybrid options |
| `TRAINING_WITH_OFFICIAL_MERGE.md` | Using official merge environment | Official vs custom merge scenarios |

### Analysis & Validation

| File | Purpose | Key Findings |
|------|---------|--------------|
| `CLIP_VALIDATION_ANALYSIS.md` | CLIP fine-tuning failure analysis | -15.84% degradation discovered, root cause analysis |
| `CLIP_HUMAN_ALIGNMENT.md` | Improved CLIP fine-tuning approach | Freezes vision encoder, preserves visual features |
| `HYBRID_CLIP_DINOV2.md` | Hybrid model documentation | Architecture, fusion strategies, expected performance |
| `VALIDATE_CLIP.md` | Detailed CLIP validation guide | Comprehensive validation methodology |

### Problem Solving

| File | Purpose | Solution Provided |
|------|---------|-------------------|
| `MERGE_SCENARIO_ISSUE.md` | Multi-agent merge environment bug | Documents single-agent limitation, custom solution |
| `CUSTOM_MERGE_SOLUTION.md` | Custom merge implementation | Technical details of MultiAgentMergeEnv fix |
| `KAGGLE_FIXED_CELLS.md` | Kaggle notebook cell fixes | NumPy 2.x compatibility solutions |
| `KAGGLE_CHECKLIST.md` | Pre-training checklist | Ensures all setup steps completed |

### Roadmaps & Planning

| File | Purpose | Content |
|------|---------|---------|
| `ROADMAP_TO_DINOV2.md` | DINOv2 implementation plan | Step-by-step integration roadmap |
| `ACTION_PLAN_NOW.md` | Current action items | Immediate next steps and priorities |

---

## 🔧 Setup & Execution Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `setup.sh` | Environment setup script | Installs dependencies, sets up virtual environment |
| `train.sh` | Training execution script | Runs training with default parameters |
| `run.sh` | General execution script | Flexible script for various tasks |
| `prepare_kaggle.sh` | Kaggle package preparation | Creates data package for Kaggle upload |
| `requirements.txt` | Python dependencies | All required packages and versions |

---

## 📊 Data & Outputs

### Data (`data/`)

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `data/highway_images/` | Collected highway screenshots | Training data for CLIP fine-tuning |
| `data/instructions.json` | Human-aligned text instructions | Vision-language training pairs |
| `data_package.tar.gz` | Compressed training data | Ready for Kaggle upload |

### Outputs

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `runs/` | TensorBoard logs | Training metrics and visualizations |
| `test_outputs/` | Test results and artifacts | Validation outputs, model checkpoints |
| `models/checkpoints/` | Saved model weights | Trained model parameters |

---

## 🎯 Specifications (`.kiro/specs/`)

### VLM-MARL Highway Spec

| File | Purpose | Content |
|------|---------|---------|
| `vlm-marl-highway/requirements.md` | Project requirements | Feature specifications, success criteria |
| `vlm-marl-highway/design.md` | System design | Architecture decisions, component design |
| `vlm-marl-highway/tasks.md` | Task breakdown | 11 tasks with completion status |

### Ambulance Priority Spec

| File | Purpose | Content |
|------|---------|---------|
| `ambulance-priority/requirements.md` | Ambulance feature requirements | Priority system specifications |
| `ambulance-priority/design.md` | Ambulance system design | Detection logic, reward structure |
| `ambulance-priority/tasks.md` | Implementation tasks | Task list with completion tracking |

---

## 🔍 File Count Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Model Files** | 6 | Core neural network architectures |
| **Training Scripts** | 3 | Training and fine-tuning scripts |
| **Environment Files** | 3 | Environment implementations and wrappers |
| **Config Files** | 7 | Environment configuration YAMLs |
| **Common Utilities** | 3 | Shared utility functions |
| **Documentation** | 20+ | Guides, analyses, and references |
| **Setup Scripts** | 5 | Installation and execution scripts |
| **Spec Files** | 6 | Requirements, design, and tasks |
| **Total Core Files** | 50+ | Excluding generated outputs |

---

## 🚀 Key File Relationships

### Training Pipeline Flow
```
train_mappo.py
    ↓
make_env.py → [highway.yaml, merge_multi_agent.yaml, etc.]
    ↓
heterogeneous_mappo.py → [clip_encoder.py, dinov2_encoder.py, hybrid_clip_dinov2.py]
    ↓
scenario.py (rewards, ambulance detection)
    ↓
logger_tb.py (TensorBoard logging)
```

### CLIP Training Flow
```
collect_data.py → data/highway_images/ + instructions.json
    ↓
train_clip.py OR train_clip_human_aligned.py
    ↓
clip_encoder.py (fine-tuned weights)
    ↓
validate_clip.py (performance validation)
```

### Environment Creation Flow
```
configs/envs/*.yaml
    ↓
make_env.py
    ↓
[highway-env OR custom_envs/multi_agent_merge_env.py]
    ↓
scenario.py (reward shaping)
```

---

## 🎓 Usage Recommendations

### For New Users
1. Start with `README.md` - comprehensive overview
2. Run `setup.sh` - install dependencies
3. Follow `KAGGLE_QUICK_START.md` - free GPU training
4. Check `TRAINING_OPTIONS_SUMMARY.md` - choose approach

### For Training
1. **CLIP**: Use `train_clip_human_aligned.py` (not `train_clip.py`)
2. **DINOv2**: Follow `DINOV2_QUICK_START.md`
3. **Hybrid**: Use `train_mappo.py --vision_encoder hybrid`
4. **Validation**: Run `validate_clip.py` before training

### For Development
1. **Models**: Extend `models/heterogeneous_mappo.py`
2. **Environments**: Add configs to `configs/envs/`
3. **Custom envs**: Follow `custom_envs/README.md`
4. **Rewards**: Modify `common/scenario.py`

### For Debugging
1. **CLIP issues**: Check `CLIP_VALIDATION_ANALYSIS.md`
2. **Merge env**: See `MERGE_SCENARIO_ISSUE.md`
3. **Kaggle errors**: Review `KAGGLE_FIXED_CELLS.md`
4. **Ambulance detection**: See `ambulance-priority/design.md`

---

## 📈 Expected Performance by File

| Model File | Expected Success Rate | Training Time |
|------------|----------------------|---------------|
| `heterogeneous_mappo.py` (baseline) | 10-15% | 2-4 hours |
| `vla_mappo.py` (pre-trained CLIP) | 15-25% | 3-5 hours |
| `vla_mappo.py` (fine-tuned CLIP) | 5-10% ❌ | 4-6 hours |
| `vla_mappo.py` (human-aligned CLIP) | 20-30% | 4-6 hours |
| `dinov2_mappo.py` | 25-35% | 3-5 hours |
| `hybrid_clip_dinov2.py` | **30-40%** ⭐ | 4-6 hours |

---

## 🏆 Most Important Files

### Critical for Training
1. `train_mappo.py` - Main training script
2. `models/hybrid_clip_dinov2.py` - Best model
3. `configs/envs/highway_heterogeneous_dense.yaml` - Best scenario
4. `common/scenario.py` - Reward logic

### Critical for Understanding
1. `README.md` - Project overview
2. `HYBRID_CLIP_DINOV2.md` - Best approach documentation
3. `CLIP_VALIDATION_ANALYSIS.md` - What NOT to do
4. `TRAINING_OPTIONS_SUMMARY.md` - All options compared

### Critical for Setup
1. `setup.sh` - Local setup
2. `KAGGLE_TRAINING_GUIDE.md` - GPU setup
3. `requirements.txt` - Dependencies
4. `prepare_kaggle.sh` - Data packaging

---

## 📝 File Naming Conventions

- **UPPERCASE.md**: Documentation and guides
- **lowercase.py**: Python source code
- **lowercase.yaml**: Configuration files
- **lowercase.sh**: Shell scripts
- **snake_case**: Python modules and scripts
- **kebab-case**: Spec directories

---

## 🎯 Next Steps

Based on this file structure, you should:

1. **Train hybrid model**: `python training/train_mappo.py --vision_encoder hybrid`
2. **Compare all approaches**: Train CLIP, DINOv2, and hybrid
3. **Analyze results**: Use TensorBoard logs in `runs/`
4. **Write paper**: Use documentation as reference

This project contains everything needed for a complete VLM-MARL research system! 🚀
