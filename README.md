# VLM-MARL Highway: Vision-Language Multi-Agent Reinforcement Learning

A research-grade implementation of Multi-Agent Proximal Policy Optimization (MAPPO) with CLIP vision-language encoders for cooperative autonomous driving in highway-env scenarios.

## Overview

This project implements Centralized Training with Decentralized Execution (CTDE) for multi-agent reinforcement learning, combining:
- **CLIP ViT-B/32** for vision-language scene understanding
- **MAPPO** for cooperative multi-agent policy learning
- **highway-env** for realistic driving simulation

The system trains 4 cooperative agents to navigate three challenging scenarios: highway cruising, ramp merging, and intersection navigation.

## Features

- **Multi-Agent Coordination**: 4 agents learn cooperative driving policies
- **Heterogeneous Agent Roles**: Support for mixed agent types (ambulance priority scenario)
- **Vision-Language Fusion**: CLIP embeddings provide semantic scene context
- **Expert Data Collection**: Heuristic-based data generation with quality filtering
- **Modular Architecture**: Clean separation of data collection, CLIP fine-tuning, and RL training
- **TensorBoard Integration**: Comprehensive metrics logging for all training phases

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository and navigate to the project directory:
```bash
cd vlm-marl-highway
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- `gymnasium` - RL environment interface
- `highway-env` - Driving simulation environments
- `torch` - Deep learning framework
- `transformers` - HuggingFace CLIP models
- `tensorboard` - Metrics visualization
- `pillow` - Image processing
- `pyyaml` - Configuration management

## Usage

### Quick Start

Run the complete training pipeline:
```bash
./run.sh
```

This will:
1. Collect data from all three scenarios (highway, merge, intersection)
2. Fine-tune CLIP on the collected data
3. Train MAPPO on the highway scenario

To train on a different scenario:
```bash
./run.sh merge        # Train on merge scenario
./run.sh intersection # Train on intersection scenario
```

### Heterogeneous Mode (Emergency Vehicle Priority)

The system supports heterogeneous agent roles for training emergency vehicle scenarios where one ambulance (Agent 0) must navigate through traffic while three normal vehicles (Agents 1-3) learn to yield and clear lanes.

#### Configuration

Enable heterogeneous mode by using a `*_heterogeneous.yaml` configuration file:

```yaml
heterogeneous_agents: true
agent_roles:
  - ambulance    # Agent 0: priority vehicle
  - normal       # Agent 1: yielding vehicle
  - normal       # Agent 2: yielding vehicle
  - normal       # Agent 3: yielding vehicle
agent_colors:
  ambulance: [255, 0, 0]  # Red
  normal: [0, 255, 0]     # Green
  npc: [0, 0, 255]        # Blue
```

#### Heterogeneous Data Collection

Collect data with role-specific expert heuristics:

```bash
python data_collection/collect_data.py \
    --scenario highway_heterogeneous \
    --episodes 100 \
    --max-steps 1000 \
    --output-dir data \
    --log-dir runs/data_collection_hetero \
    --seed 42
```

**Heterogeneous Expert Behavior:**
- **Ambulance (Agent 0)**: Aggressive driving, prioritizes speed, attempts passing maneuvers
- **Normal Agents (1-3)**: Detect ambulance within 50m, yield by moving right or slowing down

**Output Format:**
Dataset includes role information and ambulance distance for normal agents:
```json
{
  "agents": [
    {
      "role": "ambulance",
      "action_name": "FASTER",
      "scene": {"front_dist": 35.2, "risk": 0.25}
    },
    {
      "role": "normal",
      "action_name": "LANE_RIGHT",
      "scene": {"front_dist": 20.0, "ambulance_dist": 25.5}
    }
  ]
}
```

#### Heterogeneous MAPPO Training

Train with role-specific actor networks and asymmetric rewards:

```bash
python training/train_mappo.py \
    --scenario highway_heterogeneous \
    --steps 100000 \
    --horizon 256 \
    --logdir runs \
    --clip_dir models/clip \
    --lr 3e-4
```

**Heterogeneous Architecture:**
- **HeterogeneousActor**: Separate sub-networks for ambulance and normal agents
- **Centralized Critic**: Shared value function for cooperative credit assignment
- **Asymmetric Rewards**:
  - Ambulance: `0.7 × speed - 0.3 × collision`
  - Normal: `0.5 × lane_clearance - 0.5 × blocking_penalty`

**Heterogeneous Metrics:**
- `ambulance/average_speed`: Ambulance speed across episodes
- `normal/lane_clearance_rate`: % timesteps not blocking ambulance
- `heterogeneous/priority_passage_success`: Ambulance reaches goal without collision

**Model Output:**
- `models/vla_mappo_heterogeneous_{scenario}.pth`: Trained model with both sub-networks

#### Quick Start (Heterogeneous Pipeline)

Run the complete heterogeneous training pipeline:

```bash
./run.sh highway_heterogeneous --heterogeneous
```

Or use the step-by-step commands above for more control.

### Step-by-Step Usage

#### 1. Data Collection

Collect driving data with expert heuristic labels:

```bash
python data_collection/collect_data.py \
    --scenario highway \
    --episodes 100 \
    --max-steps 1000 \
    --output-dir data \
    --log-dir runs/data_collection \
    --seed 42
```

**Parameters:**
- `--scenario`: Scenario name
  - Standard: `highway`, `merge`, `intersection`
  - Heterogeneous (ambulance priority): `highway_heterogeneous`, `intersection_heterogeneous`, `merge_multi_agent`
  - Dense traffic: `highway_heterogeneous_dense`
  - ⚠️ Note: `merge_heterogeneous` not recommended (use `merge_multi_agent` instead)
- `--episodes`: Number of episodes to collect (default: 100)
- `--max-steps`: Maximum steps per episode (default: 1000)
- `--output-dir`: Output directory for dataset (default: `data`)
- `--log-dir`: TensorBoard log directory (default: `runs/data_collection`)
- `--seed`: Random seed for reproducibility (default: 42)

**Output:**
- `data/{scenario}/dataset.jsonl`: Frame metadata with agent states and actions
- `data/{scenario}/images/`: 224×224 RGB images
- `data/{scenario}/metadata.json`: Collection statistics

**Expert Heuristic:**
The data collector uses a rule-based policy that:
- Performs emergency maneuvers when TTC < 2.0s or distance < 10m
- Slows down when following vehicles at < 20m
- Explores lane changes with 12% probability
- Optimizes speed when safe

**Quality Filtering:**
Frames are saved if quality score > 0.5 OR with 25% random probability. Quality is computed as:
- 40% weight on time-to-collision (lower is better)
- 30% weight on minimum distance (closer is better)
- 30% weight on risk score (higher is better)

#### 2. CLIP Fine-tuning

Fine-tune CLIP on driving-specific image-instruction pairs:

```bash
python training/train_clip.py \
    --data_dirs data/highway data/merge data/intersection \
    --model_id openai/clip-vit-base-patch32 \
    --epochs 5 \
    --batch_size 64 \
    --lr 1e-5 \
    --num_workers 4 \
    --log_dir runs/clip \
    --save_dir models/clip
```

**Parameters:**
- `--data_dirs`: Paths to scenario data directories (space-separated)
- `--model_id`: HuggingFace CLIP model identifier (default: `openai/clip-vit-base-patch32`)
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size for training (default: 64)
- `--lr`: Learning rate for AdamW optimizer (default: 1e-5)
- `--num_workers`: Number of DataLoader workers (default: 4)
- `--log_dir`: TensorBoard log directory (default: `runs/clip`)
- `--save_dir`: Directory to save fine-tuned model (default: `models/clip`)

**Output:**
- `models/clip/`: Fine-tuned CLIP model and processor

**Training Details:**
- Uses contrastive loss to align images with text descriptions
- Text format: `"{action_name}: {scene_dict}"`
- Example: `"LANE_LEFT: {'front_dist': 15.2, 'front_ttc': 3.1, 'risk': 0.68}"`
- Logs loss, sample images, and sample texts every 100 steps

#### 3. MAPPO Training

Train multi-agent policies with MAPPO:

```bash
python training/train_mappo.py \
    --scenario highway \
    --steps 100000 \
    --horizon 256 \
    --logdir runs \
    --clip_dir models/clip \
    --lr 3e-4
```

**Parameters:**
- `--scenario`: Scenario to train on
  - Standard: `highway`, `merge`, `intersection`
  - Heterogeneous (ambulance priority): `highway_heterogeneous`, `intersection_heterogeneous`, `merge_multi_agent`
  - Dense traffic: `highway_heterogeneous_dense`
- `--steps`: Maximum training steps (default: 100000)
- `--horizon`: Rollout horizon in steps (default: 256)
- `--logdir`: TensorBoard log directory (default: `runs`)
- `--clip_dir`: Path to fine-tuned CLIP model (default: `models/clip`)
- `--lr`: Learning rate for Adam optimizer (default: 3e-4)
- `--device`: Device to use (`cuda` or `cpu`, auto-detected by default)

**Output:**
- `models/vla_mappo_{scenario}.pth`: Trained actor-critic model

**Hyperparameters:**
- **Rollout horizon**: 256 steps per episode
- **PPO epochs**: 4 update epochs per rollout
- **Clip epsilon**: 0.2 (policy clipping)
- **Value coefficient**: 0.5 (value loss weight)
- **Entropy coefficient**: 0.01 (exploration bonus)
- **Learning rate**: 3e-4
- **Gradient clipping**: 0.5 max norm
- **GAE gamma**: 0.99 (discount factor)
- **GAE lambda**: 0.95 (advantage smoothing)

**Training Details:**
- Centralized critic uses joint observations from all agents
- Decentralized actors use per-agent observations
- CLIP embeddings provide semantic scene context
- Shared actor weights across all agents for sample efficiency

#### 4. TensorBoard Visualization

Monitor training progress:

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

**Available Metrics:**

*Data Collection:*
- `quality/mean`: Average frame quality score
- `traffic/nearby_vehicles`: Number of nearby vehicles
- `traffic/min_ttc`: Minimum time-to-collision
- `actions/{action_name}`: Action distribution
- `episode/return`: Episode return
- `episode/saved_frames`: Number of saved frames

*CLIP Fine-tuning:*
- `loss/train`: Training loss per step
- `loss/epoch`: Average loss per epoch
- `samples/image_{i}`: Sample training images
- `samples/texts`: Sample text instructions

*MAPPO Training:*
- `return/episode`: Episode return
- `return/avg_100`: Average return over last 100 episodes
- `loss/policy`: Policy loss (clipped surrogate)
- `loss/value`: Value function loss (MSE)
- `policy/entropy`: Policy entropy (exploration)
- `steps/total`: Total environment steps

## Architecture

### System Components

```
vlm-marl/
├── configs/envs/          # YAML environment configurations
│   ├── highway.yaml       # Highway cruising scenario
│   ├── merge.yaml         # Ramp merging scenario
│   └── intersection.yaml  # Intersection navigation scenario
├── common/                # Shared utilities
│   ├── logger_tb.py      # TensorBoard wrapper
│   ├── utils.py          # Risk scoring, action mappings
│   └── scenario.py       # Environment factory
├── envs/
│   └── make_env.py       # Environment initialization
├── data_collection/
│   └── collect_data.py   # Data collection pipeline
├── models/
│   ├── clip_encoder.py   # CLIP wrapper
│   └── vla_mappo.py      # Actor-Critic networks
├── training/
│   ├── train_clip.py     # CLIP fine-tuning
│   └── train_mappo.py    # MAPPO trainer
└── run.sh                # Orchestration script
```

### Network Architecture

**Actor (Decentralized):**
```
Input: [agent_kinematics (obs_dim), clip_embedding (512)]
→ Linear(obs_dim+512, 512) + ReLU
→ Linear(512, 256) + ReLU
→ Linear(256, n_actions)
Output: action logits [n_actions]
```

**Critic (Centralized):**
```
Input: [joint_kinematics (n_agents*obs_dim), clip_embedding (512)]
→ Linear(n_agents*obs_dim+512, 512) + ReLU
→ Linear(512, 256) + ReLU
→ Linear(256, 1)
Output: state value [1]
```

### Data Flow

```
Environment Config (YAML)
    ↓
Highway-Env (MultiAgent)
    ↓
RGB Frames + Kinematics Tuples
    ↓
Data Collector → dataset.jsonl + images/
    ↓
CLIP Fine-tuning → models/clip/
    ↓
MAPPO Training → models/vla_mappo_{scenario}.pth
    ↓
TensorBoard Logs (runs/)
```

## Design Rationale

### Multi-Agent Observation and Action

The system uses highway-env's `MultiAgentObservation` and `MultiAgentAction` wrappers to control 4 agents simultaneously. Each agent observes 8 nearby vehicles with 7 features (presence, x, y, vx, vy, cos_h, sin_h) and selects from 5 discrete meta-actions (LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER).

**Reference:** [highway-env documentation](https://highway-env.farama.org/multi_agent/)

### MAPPO Algorithm

MAPPO (Multi-Agent PPO) extends PPO to multi-agent settings with centralized training and decentralized execution. The centralized critic uses global state information during training to improve credit assignment, while decentralized actors enable independent execution at test time.

**Reference:** Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", arXiv:2103.01955, 2021

### Vision-Language Integration

CLIP provides semantic scene understanding that complements geometric kinematics. Fine-tuning on driving-specific image-instruction pairs adapts the model to recognize driving-relevant concepts (e.g., "merge", "lane change", "congested traffic").

**Reference:** Sima et al., "DriveLM: Driving with Graph Visual Question Answering", arXiv:2312.14150, 2023

### TensorBoard Logging

All training components log exclusively to TensorBoard using PyTorch's built-in `torch.utils.tensorboard.SummaryWriter`. This provides unified experiment tracking without external dependencies.

**Reference:** [PyTorch TensorBoard Tutorial](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)

## Scenarios

### Highway (highway-v0)
- **Description**: Multi-lane highway cruising with dense traffic
- **Difficulty**: Easy
- **Key Challenges**: Lane keeping, speed optimization, collision avoidance
- **Heterogeneous Variant**: `highway_heterogeneous` - Emergency vehicle priority passage

### Merge (merge-v0)
- **Description**: Ramp merging with conflicting traffic flows
- **Difficulty**: Medium
- **Key Challenges**: Gap acceptance, speed matching, cooperative yielding
- **Heterogeneous Variant**: `merge_heterogeneous` - Ambulance merging priority

### Intersection (intersection-v1/v0)
- **Description**: Unsignalized intersection with crossing traffic
- **Difficulty**: Hard
- **Key Challenges**: Conflict resolution, turn negotiation, deadlock avoidance
- **Heterogeneous Variant**: `intersection_heterogeneous` - Emergency vehicle intersection priority

## Expected Results

After training for 100k steps on the highway scenario, you should observe:
- **Episode Return**: Increasing from ~0 to 20-30
- **Collision Rate**: Decreasing from ~50% to <20%
- **Policy Entropy**: Gradually decreasing as policy becomes more deterministic
- **Value Loss**: Stabilizing after initial fluctuations

Training time (approximate):
- Data collection: ~30 minutes per scenario (300 episodes total)
- CLIP fine-tuning: ~1 hour (5 epochs on combined dataset)
- MAPPO training: ~4-6 hours (100k steps on highway)

## Troubleshooting

### Environment Initialization Errors

If you encounter `gymnasium.error.NameNotFound` for intersection-v1:
- The system automatically falls back to intersection-v0
- Check `configs/envs/intersection.yaml` for `id_candidates` list

### CUDA Out of Memory

If training fails with OOM errors:
- Reduce batch size: `--batch_size 32` (CLIP) or `--horizon 128` (MAPPO)
- Use CPU: `--device cpu` (slower but works on any machine)

### Low Episode Returns

If MAPPO training shows no improvement:
- Verify CLIP fine-tuning completed successfully
- Check TensorBoard for policy entropy (should be >0.5 initially)
- Try increasing learning rate: `--lr 5e-4`
- Collect more diverse data with different seeds

### Data Collection Hangs

If data collection freezes:
- Check environment rendering: `env.render()` may fail on headless systems
- Set `render_mode="rgb_array"` in YAML configs (already default)
- Reduce `max_steps` if episodes are too long

## Citation

If you use this code in your research, please cite:

```bibtex
@software{vlm_marl_highway,
  title={VLM-MARL Highway: Vision-Language Multi-Agent Reinforcement Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/vlm-marl-highway}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **highway-env**: Edouard Leurent and contributors
- **CLIP**: OpenAI
- **MAPPO**: Yu et al. (2021)
- **PyTorch**: Facebook AI Research

## Contributing

Contributions are welcome! Please open an issue or pull request for:
- Bug fixes
- New scenarios or environments
- Performance improvements
- Documentation enhancements

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]
