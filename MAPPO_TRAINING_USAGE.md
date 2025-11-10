# MAPPO Training Pipeline - Usage Guide

## Overview
The MAPPO training pipeline (`training/train_mappo.py`) implements Multi-Agent Proximal Policy Optimization with CLIP vision-language fusion for cooperative autonomous driving.

## Implementation Status
✅ **Task 7: MAPPO Training Pipeline - COMPLETED**

All sub-tasks implemented:
- ✅ 7.1: Observation flattening utilities
- ✅ 7.2: Rollout collection
- ✅ 7.3: GAE computation
- ✅ 7.4: PPO update
- ✅ 7.5: Training loop with TensorBoard logging

## Usage

### Basic Training Command
```bash
python training/train_mappo.py --scenario highway --steps 100000
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | `highway` | Scenario to train on: `highway`, `merge`, or `intersection` |
| `--steps` | `100000` | Maximum training steps |
| `--horizon` | `256` | Rollout horizon (steps per episode) |
| `--logdir` | `runs` | TensorBoard log directory |
| `--clip_dir` | `models/clip` | Path to fine-tuned CLIP model |
| `--lr` | `3e-4` | Learning rate |
| `--device` | `cuda`/`cpu` | Device to use (auto-detected) |

### Example Commands

**Train on highway scenario:**
```bash
python training/train_mappo.py --scenario highway --steps 100000 --horizon 256
```

**Train on merge scenario with custom learning rate:**
```bash
python training/train_mappo.py --scenario merge --steps 50000 --lr 1e-4
```

**Train on intersection with CPU:**
```bash
python training/train_mappo.py --scenario intersection --device cpu --steps 10000
```

## Hyperparameters

The implementation uses the following hyperparameters as specified in the design:

- **Rollout horizon**: 256 steps
- **PPO epochs**: 4
- **Clip epsilon**: 0.2
- **Value coefficient**: 0.5
- **Entropy coefficient**: 0.01
- **Learning rate**: 3e-4
- **Gradient clipping**: 0.5
- **GAE gamma**: 0.99
- **GAE lambda**: 0.95

## Output

### Model Checkpoint
Trained model saved to: `models/vla_mappo_{scenario}.pth`

### TensorBoard Logs
Logs saved to: `runs/mappo_{scenario}/`

View with:
```bash
tensorboard --logdir runs
```

### Logged Metrics
- `return/episode`: Episode return
- `return/avg_100`: Average return over last 100 episodes
- `loss/policy`: Policy loss
- `loss/value`: Value loss
- `policy/entropy`: Policy entropy
- `steps/total`: Total training steps

## Implementation Details

### Key Functions

1. **`flatten_obs_tuple(obs_tuple)`**
   - Converts tuple observations to per-agent and joint arrays
   - Returns: `(per_agent [N, obs_dim], joint [N*obs_dim])`

2. **`collect_rollout(env, net, device, horizon, text_prompt)`**
   - Collects rollout of experiences
   - Renders frames and encodes with CLIP
   - Samples actions using Categorical distribution
   - Returns: `(trajectories, episode_return)`

3. **`compute_gae(trajectories, gamma, lam)`**
   - Computes Generalized Advantage Estimation
   - Returns: `(advantages, returns)`

4. **`ppo_update(net, optimizer, trajectories, advantages, returns, device, ...)`**
   - Performs PPO update with clipped surrogate objective
   - Includes value loss and entropy bonus
   - Returns: `(policy_loss, value_loss, entropy)`

### Architecture
- **Decentralized Actor**: Takes per-agent kinematics + CLIP embeddings → action logits
- **Centralized Critic**: Takes joint kinematics + CLIP embeddings → state value
- **CLIP Encoder**: Frozen pretrained or fine-tuned model for vision-language embeddings

## Requirements Satisfied

### Requirement 4 (MAPPO Training)
- ✅ 4.1: Shared actor with per-agent kinematics + CLIP embeddings
- ✅ 4.2: Centralized critic with joint kinematics + CLIP embeddings
- ✅ 4.3: Rollout collection with RGB rendering and CLIP encoding
- ✅ 4.4: GAE computation with gamma=0.99, lambda=0.95
- ✅ 4.5: PPO updates with clipping, value loss, entropy
- ✅ 4.6: TensorBoard logging of all metrics
- ✅ 4.7: Model saving after training

### Requirement 5 (TensorBoard Logging)
- ✅ 5.2: Scalar metrics logged with tags and steps
- ✅ 5.5: Single tensorboard command for visualization

### Requirement 7 (Multi-Agent Handling)
- ✅ 7.1: Observation tuple flattening
- ✅ 7.2: Joint observation concatenation
- ✅ 7.3: Per-agent action sampling
- ✅ 7.4: Action tensor to tuple conversion
- ✅ 7.5: Reward tuple handling (mean computation)

## Next Steps

After completing this task, you can:
1. Run data collection (Task 4)
2. Fine-tune CLIP (Task 5)
3. Train MAPPO with this pipeline (Task 7 - this task)
4. Create orchestration scripts (Task 8)
5. Verify end-to-end functionality (Task 9)

## Notes

- The implementation handles both pretrained and fine-tuned CLIP models
- Automatically detects CUDA availability
- Supports all three scenarios: highway, merge, intersection
- Logs progress every 10 iterations
- Saves final model checkpoint automatically
