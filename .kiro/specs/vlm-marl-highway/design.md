# Design Document

## Overview

The VLM-MARL Highway system implements a research-grade multi-agent reinforcement learning pipeline that combines vision-language models (CLIP ViT-B/32) with Multi-Agent Proximal Policy Optimization (MAPPO) for cooperative autonomous driving in highway-env scenarios. The architecture follows the Centralized Training with Decentralized Execution (CTDE) paradigm, where agents share a centralized critic during training but execute independently using decentralized actors.

The system consists of three main phases:
1. **Data Collection**: Gather high-quality driving frames with expert heuristic labels
2. **CLIP Fine-tuning**: Adapt the vision encoder to driving-specific semantics
3. **MAPPO Training**: Train cooperative multi-agent policies with vision-language context

All components log exclusively to TensorBoard for unified experiment tracking.

## Architecture

### System Components

```
vlm-marl/
├── configs/envs/          # YAML environment configurations
├── common/                # Shared utilities
│   ├── logger_tb.py      # TensorBoard wrapper
│   ├── utils.py          # Risk scoring, action mappings
│   └── scenario.py       # Environment factory
├── envs/
│   └── make_env.py       # Environment initialization
├── data_collection/
│   └── collect_data.py   # Specialist data collector
├── models/
│   ├── clip_encoder.py   # CLIP wrapper
│   └── vla_mappo.py      # Actor-Critic networks
└── training/
    ├── train_clip.py     # CLIP fine-tuning
    └── train_mappo.py    # MAPPO trainer
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

## Components and Interfaces

### 1. Environment Configuration (configs/envs/)

**Purpose**: Define scenario-specific parameters for highway-env in declarative YAML format.

**Interface**:
- Input: YAML file path
- Output: Configured Gymnasium environment

**Key Design Decisions**:
- Use `MultiAgentObservation` with `Kinematics` observation type (8 vehicles, 7 features: presence, x, y, vx, vy, cos_h, sin_h)
- Use `MultiAgentAction` with `DiscreteMetaAction` (5 actions: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER)
- Set `controlled_vehicles: 4` to control 4 agents simultaneously
- Set `flatten: true` for stable vector observations
- Configure scenario-specific spawn rates and traffic density

**YAML Schema**:
```yaml
id: <env-id>                    # highway-v0 | merge-v0 | intersection-v0/v1
render_mode: rgb_array
config:
  observation:
    type: MultiAgentObservation
    observation_config:
      type: Kinematics
      vehicles_count: 8
      features: [presence, x, y, vx, vy, cos_h, sin_h]
      absolute: true
      flatten: true
  action:
    type: MultiAgentAction
    action_config:
      type: DiscreteMetaAction
      lateral: true
      longitudinal: true
  controlled_vehicles: 4
  # ... scenario-specific params
```

### 2. TensorBoard Logger (common/logger_tb.py)

**Purpose**: Unified logging interface wrapping `torch.utils.tensorboard.SummaryWriter`.

**Interface**:
```python
class TBLogger:
    def __init__(self, log_dir: str, name: str)
    def scalar(self, tag: str, value: float, step: int)
    def image(self, tag: str, img: np.ndarray, step: int, dataformats: str)
    def text(self, tag: str, text: str, step: int)
    def close(self)
```

**Design Rationale**: Single abstraction layer allows easy switching of logging backends if needed while maintaining consistent API across all training components.

### 3. Utility Functions (common/utils.py)

**Purpose**: Reusable calculations for risk assessment and action mapping.

**Key Functions**:
- `ttc(ego_pos, ego_vel, other_pos, other_vel)`: Time-to-collision calculation
- `risk_score(ego, other)`: Composite risk metric (40% TTC, 30% distance, 30% relative velocity)
- `ACTIONS_ALL`: Dictionary mapping action IDs to names

**Design Rationale**: Risk scoring provides a principled way to filter high-quality training data. The weighted combination balances temporal urgency (TTC), spatial proximity (distance), and dynamic interaction (relative velocity).

### 4. Environment Factory (envs/make_env.py)

**Purpose**: Load and configure highway-env instances from YAML configs.

**Interface**:
```python
def make_env(scenario: str) -> gym.Env
```

**Implementation**:
- Read YAML config for scenario
- Handle `id_candidates` list for robust fallback (intersection v1→v0)
- Create Gymnasium environment with `render_mode="rgb_array"`
- Update environment config with YAML parameters
- Return configured environment

### 5. Data Collector (data_collection/collect_data.py)

**Purpose**: Generate high-quality training dataset with expert labels.

**Architecture**:
```
Episode Loop
  ↓
For each step:
  1. Render RGB frame (800×800)
  2. Extract controlled vehicles
  3. For each agent:
     - Compute ego state (position, velocity, heading)
     - Scan nearby vehicles (within 60m)
     - Calculate min TTC, min distance, max risk
     - Determine lane availability (left/right)
     - Apply expert heuristic → action label
  4. Compute quality score (avg across agents)
  5. If quality > 0.5 OR random < 0.25:
     - Center crop to 224×224
     - Save image as PNG
     - Write JSON record to dataset.jsonl
     - Log metrics to TensorBoard
```

**Expert Heuristic Logic**:
```
IF front_ttc < 2.0 OR front_dist < 10.0:
    IF can_left AND random < 0.5: LANE_LEFT
    ELIF can_right: LANE_RIGHT
    ELSE: SLOWER
ELIF front_dist < 20.0 AND front_ttc < 4.0:
    SLOWER
ELIF random < 0.12:
    LANE_LEFT/RIGHT if available, else IDLE
ELSE:
    FASTER (50%) or IDLE (50%)
```

**Output Format** (dataset.jsonl):
```json
{
  "index": 0,
  "image": "images/000000.png",
  "episode": 0,
  "step": 5,
  "quality": 0.73,
  "agents": [
    {
      "speed": 25.3,
      "position": [120.5, 8.0],
      "heading": 0.05,
      "scene": {
        "front_dist": 15.2,
        "front_ttc": 3.1,
        "risk": 0.68,
        "can_left": true,
        "can_right": false
      },
      "action_id": 0,
      "action_name": "LANE_LEFT"
    }
    // ... more agents
  ],
  "timestamp": "2025-11-09T..."
}
```

**Design Rationale**: 
- Quality filtering reduces dataset size by ~75% while retaining informative frames
- 224×224 matches CLIP ViT-B/32 input requirements
- Expert heuristic provides reasonable supervision without human labeling
- Per-agent labels enable multi-agent imitation learning if desired

### 6. CLIP Encoder (models/clip_encoder.py)

**Purpose**: Wrap HuggingFace CLIP for image/text embedding extraction.

**Interface**:
```python
class CLIPBackbone(nn.Module):
    def __init__(self, model_id: str, trainable: bool)
    def img_embed(self, pil_or_np, device) -> torch.Tensor  # [1, 512]
    def txt_embed(self, texts: List[str], device) -> torch.Tensor  # [N, 512]
```

**Design Decisions**:
- Use `openai/clip-vit-base-patch32` (512-dim embeddings)
- Freeze weights by default (`trainable=False`) for inference
- Enable gradients during fine-tuning phase
- Normalize embeddings with L2 norm for stable fusion

### 7. CLIP Fine-tuning (training/train_clip.py)

**Purpose**: Adapt CLIP to driving-specific image-instruction pairs.

**Dataset Construction**:
- Load all three scenario datasets (highway, merge, intersection)
- For each agent in each frame: pair (image, text)
- Text format: `"{action_name}: {scene_dict}"`
- Example: `"LANE_LEFT: {'front_dist': 15.2, 'front_ttc': 3.1, ...}"`

**Training Loop**:
```python
for epoch in range(epochs):
    for batch in dataloader:
        inputs = processor(text=texts, images=images, ...)
        outputs = model(**inputs, return_loss=True)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        # Log to TensorBoard every 100 steps
```

**Hyperparameters**:
- Epochs: 5
- Batch size: 64
- Learning rate: 1e-5 (AdamW)
- Num workers: 4

**Output**: Fine-tuned model saved to `models/clip/`

**Design Rationale**: Fine-tuning on domain-specific data improves semantic alignment between visual scenes and driving actions. The contrastive loss encourages the model to associate similar scenes with similar action descriptions.

### 8. VLA Actor-Critic (models/vla_mappo.py)

**Purpose**: Implement MAPPO networks with vision-language fusion.

**Architecture**:

```
Actor (Decentralized):
  Input: [agent_kinematics (obs_dim), clip_embedding (512)]
  → Linear(obs_dim+512, 512) + ReLU
  → Linear(512, 256) + ReLU
  → Linear(256, n_actions)
  Output: action logits [n_actions]

Critic (Centralized):
  Input: [joint_kinematics (n_agents*obs_dim), clip_embedding (512)]
  → Linear(n_agents*obs_dim+512, 512) + ReLU
  → Linear(512, 256) + ReLU
  → Linear(256, 1)
  Output: state value [1]
```

**Forward Pass**:
```python
def forward(self, obs_per_agent, joint_obs, clip_emb):
    # obs_per_agent: [n_agents, obs_dim]
    # joint_obs: [1, n_agents*obs_dim]
    # clip_emb: [1, 512]
    
    # Actor: per-agent logits
    clip_repeated = clip_emb.repeat(n_agents, 1)
    actor_input = torch.cat([obs_per_agent, clip_repeated], dim=-1)
    logits = self.actor(actor_input)  # [n_agents, n_actions]
    
    # Critic: centralized value
    critic_input = torch.cat([joint_obs, clip_emb], dim=-1)
    value = self.critic(critic_input)  # [1, 1]
    
    return logits, value
```

**CLIP Integration**:
- Extract image embedding from rendered frame each step
- Optional text prompt per scenario (e.g., "highway driving", "merge driving")
- Freeze CLIP weights during RL training
- Normalize embeddings before fusion

**Design Rationale**: 
- Shared actor weights across agents improve sample efficiency
- Centralized critic enables credit assignment in cooperative tasks
- CLIP embeddings provide semantic context (e.g., "congested", "clear road") that complements geometric kinematics
- Fusion via concatenation is simple and effective

### 9. MAPPO Trainer (training/train_mappo.py)

**Purpose**: Train cooperative multi-agent policies with PPO.

**Training Loop**:

```
Initialize environment, actor-critic network, optimizer
total_steps = 0

while total_steps < max_steps:
    # 1. Collect rollout
    batch, episode_return = collect_rollout(env, net, horizon=256)
    total_steps += len(batch)
    
    # 2. Compute advantages (GAE)
    advantages, returns = compute_gae(batch, gamma=0.99, lambda=0.95)
    
    # 3. PPO update (4 epochs)
    policy_loss, value_loss, entropy = ppo_update(net, batch, epochs=4)
    
    # 4. Log metrics
    logger.scalar("return/episode", episode_return, total_steps)
    logger.scalar("loss/policy", policy_loss, total_steps)
    logger.scalar("loss/value", value_loss, total_steps)
    logger.scalar("policy/entropy", entropy, total_steps)

# Save trained model
torch.save(net.state_dict(), f"models/vla_mappo_{scenario}.pth")
```

**Rollout Collection**:
```python
def collect_rollout(env, net, device, horizon):
    obs, info = env.reset()
    trajectories = []
    
    for t in range(horizon):
        # Render and encode
        frame = env.render()  # [H, W, 3]
        clip_emb = net._embed(frame, device)  # [1, 512]
        
        # Flatten observations
        per_agent, joint = flatten_obs_tuple(obs)  # [N, D], [N*D]
        
        # Forward pass
        logits, value = net(per_agent, joint, clip_emb)
        
        # Sample actions
        dist = Categorical(logits=logits)
        actions = dist.sample()  # [N]
        log_probs = dist.log_prob(actions)  # [N]
        
        # Environment step
        actions_tuple = tuple([int(a) for a in actions])
        next_obs, reward, done, trunc, info = env.step(actions_tuple)
        
        # Store transition
        trajectories.append({
            'obs': per_agent, 'joint': joint, 'clip': clip_emb,
            'action': actions, 'log_prob': log_probs,
            'reward': np.mean(reward), 'value': value, 'done': done
        })
        
        obs = next_obs
        if done: break
    
    return trajectories
```

**GAE Computation**:
```python
def compute_gae(trajectories, gamma=0.99, lam=0.95):
    values = [t['value'] for t in trajectories] + [0.0]
    rewards = [t['reward'] for t in trajectories]
    dones = [t['done'] for t in trajectories]
    
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns
```

**PPO Update**:
```python
def ppo_update(net, batch, epochs=4, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    for epoch in range(epochs):
        for t in range(len(batch)):
            # Forward pass
            logits, value = net(batch['obs'][t], batch['joint'][t], batch['clip'][t])
            dist = Categorical(logits=logits)
            
            # Policy loss (clipped surrogate)
            log_prob = dist.log_prob(batch['action'][t])
            ratio = torch.exp(log_prob - batch['old_log_prob'][t])
            surr1 = ratio * batch['advantage'][t]
            surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * batch['advantage'][t]
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (MSE)
            value_loss = F.mse_loss(value, batch['return'][t])
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
```

**Hyperparameters**:
- Horizon: 256 steps per rollout
- PPO epochs: 4
- Clip epsilon: 0.2
- Value coefficient: 0.5
- Entropy coefficient: 0.01
- Learning rate: 3e-4
- Gradient clipping: 0.5
- GAE gamma: 0.99
- GAE lambda: 0.95

**Design Rationale**:
- MAPPO is state-of-the-art for cooperative MARL
- Centralized critic enables better credit assignment than independent learners
- Decentralized actors allow independent execution at test time
- PPO's clipped objective provides stable, sample-efficient learning
- GAE reduces variance in advantage estimates
- Entropy bonus encourages exploration

## Data Models

### Environment Observation
```python
# MultiAgent tuple (one per controlled vehicle)
obs: Tuple[np.ndarray, ...]
# Each element shape: (vehicles_count * features,) = (8 * 7,) = (56,)
# Flattened: [presence, x, y, vx, vy, cos_h, sin_h] × 8 vehicles
```

### Environment Action
```python
# MultiAgent tuple (one per controlled vehicle)
actions: Tuple[int, ...]
# Each element: 0 (LANE_LEFT) | 1 (IDLE) | 2 (LANE_RIGHT) | 3 (FASTER) | 4 (SLOWER)
```

### Dataset Record
```python
{
  "index": int,              # Global frame index
  "image": str,              # Relative path to PNG
  "episode": int,            # Episode number
  "step": int,               # Step within episode
  "quality": float,          # Quality score [0, 1]
  "agents": [                # List of agent states
    {
      "speed": float,
      "position": [float, float],
      "heading": float,
      "scene": {
        "front_dist": float,
        "front_ttc": float,
        "risk": float,
        "can_left": bool,
        "can_right": bool
      },
      "action_id": int,
      "action_name": str
    }
  ],
  "timestamp": str           # ISO format
}
```

### MAPPO Batch
```python
{
  "obs": np.ndarray,         # [T, N, obs_dim]
  "joint": np.ndarray,       # [T, 1, N*obs_dim]
  "act": np.ndarray,         # [T, N]
  "logp": np.ndarray,        # [T, N]
  "adv": np.ndarray,         # [T]
  "ret": np.ndarray,         # [T]
  "clip": np.ndarray         # [T, 1, 512]
}
```

## Error Handling

### Environment Initialization
- **Issue**: Intersection environment version mismatch
- **Solution**: Try `intersection-v1` first, fallback to `intersection-v0`
- **Implementation**: `id_candidates` list in YAML config

### Data Collection
- **Issue**: No controlled vehicles in environment
- **Solution**: Check `controlled_vehicles` list, skip if empty
- **Implementation**: `if not isinstance(controlled, list): controlled=[controlled]`

### CLIP Embedding
- **Issue**: Image size mismatch
- **Solution**: CLIPProcessor handles resizing automatically
- **Implementation**: Use processor for all image inputs

### MAPPO Training
- **Issue**: Reward format varies (scalar vs tuple)
- **Solution**: Detect type and compute mean if tuple
- **Implementation**: 
```python
if isinstance(reward, (list, tuple, np.ndarray)):
    r = float(np.mean(reward))
else:
    r = float(reward)
```

### Gradient Explosion
- **Issue**: Large policy updates destabilize training
- **Solution**: Gradient clipping at 0.5 norm
- **Implementation**: `nn.utils.clip_grad_norm_(net.parameters(), 0.5)`

### Division by Zero
- **Issue**: TTC calculation with zero relative velocity
- **Solution**: Add epsilon (1e-6) to distance calculations
- **Implementation**: `dist = np.linalg.norm(rel) + eps`

## Testing Strategy

### Unit Tests

**Environment Configuration**:
- Test YAML loading for all three scenarios
- Verify `controlled_vehicles` count equals 4
- Verify observation space shape matches expected dimensions
- Verify action space is Discrete(5)

**Risk Scoring**:
- Test TTC calculation with known ego/other states
- Test risk_score with edge cases (zero distance, zero velocity)
- Verify risk score bounds [0, 1]

**Data Collector**:
- Test quality score computation
- Test expert heuristic with various scene configurations
- Verify image cropping produces 224×224 output
- Verify JSON serialization of dataset records

**CLIP Encoder**:
- Test image embedding shape [1, 512]
- Test text embedding shape [N, 512]
- Verify embeddings are L2-normalized

**VLA Actor-Critic**:
- Test forward pass with dummy inputs
- Verify actor output shape [n_agents, n_actions]
- Verify critic output shape [1, 1]
- Test gradient flow through networks

### Integration Tests

**Data Collection Pipeline**:
- Run 10 episodes of each scenario
- Verify dataset.jsonl is valid JSON
- Verify all referenced images exist
- Verify action distribution is reasonable (not all one action)

**CLIP Fine-tuning**:
- Train for 1 epoch on small dataset
- Verify loss decreases
- Verify model saves successfully
- Verify TensorBoard logs are created

**MAPPO Training**:
- Train for 1000 steps on highway scenario
- Verify episode returns are logged
- Verify model saves successfully
- Verify no NaN losses or gradients

### End-to-End Test

**Full Pipeline**:
1. Collect 50 episodes per scenario (150 total)
2. Fine-tune CLIP for 2 epochs
3. Train MAPPO for 10k steps on highway
4. Verify TensorBoard contains all expected metrics
5. Load trained model and run 10 evaluation episodes
6. Verify average return > random policy baseline

**Success Criteria**:
- No crashes or exceptions
- All output files created
- TensorBoard logs viewable
- Trained policy achieves positive average return
- Collision rate < 50% (better than random)

## Performance Considerations

### Data Collection
- **Bottleneck**: Environment rendering (800×800 RGB)
- **Optimization**: Reduce `screen_width/height` if collection is slow
- **Trade-off**: Smaller images may lose visual detail

### CLIP Fine-tuning
- **Bottleneck**: Image preprocessing and forward pass
- **Optimization**: Use DataLoader with `num_workers=4` for parallel loading
- **Optimization**: Use mixed precision training (torch.cuda.amp) if GPU available
- **Trade-off**: Larger batch sizes improve throughput but require more VRAM

### MAPPO Training
- **Bottleneck**: Environment rendering every step for CLIP encoding
- **Optimization**: Render at lower resolution (400×400) and upscale
- **Optimization**: Cache CLIP embeddings if same frame seen multiple times (unlikely)
- **Trade-off**: Lower resolution reduces visual fidelity

### Memory Usage
- **CLIP Model**: ~350MB (ViT-B/32)
- **Actor-Critic**: ~10MB (small MLPs)
- **Rollout Buffer**: ~50MB (256 steps × 4 agents × 512 CLIP dims)
- **Total**: ~500MB GPU memory (fits on most GPUs)

### Scalability
- **More Agents**: Linear increase in actor computation, quadratic in critic input size
- **Larger Scenarios**: May require larger `vehicles_count` in observations
- **Longer Horizons**: Linear increase in memory for rollout buffer

## Alternative Designs Considered

### 1. Image-based Observations vs Render-based
**Chosen**: Render RGB frames with `env.render()` and encode with CLIP
**Alternative**: Use `GrayscaleObservation` wrapper with `stack_size=4`
**Rationale**: Render-based approach is simpler to integrate with pretrained CLIP and doesn't require modifying observation space

### 2. MAPPO vs HAPPO/HATRPO
**Chosen**: MAPPO (PPO-based)
**Alternative**: HAPPO/HATRPO (trust-region methods)
**Rationale**: MAPPO is simpler to implement and provides good baseline performance. Trust-region methods can be added later if monotonic improvement is critical.

### 3. Shared vs Independent Actors
**Chosen**: Shared actor weights across all agents
**Alternative**: Independent actor networks per agent
**Rationale**: Shared weights improve sample efficiency and are standard in homogeneous multi-agent settings (all agents are cars with same action space)

### 4. TensorBoard vs Weights & Biases
**Chosen**: TensorBoard only
**Alternative**: W&B for cloud logging and experiment tracking
**Rationale**: User explicitly requested no W&B dependency. TensorBoard is sufficient for local experiment tracking and is built into PyTorch.

### 5. Expert Heuristic vs Human Demonstrations
**Chosen**: Rule-based expert heuristic (TTC, gap analysis)
**Alternative**: Collect human driving demonstrations
**Rationale**: Heuristic is faster to implement and provides reasonable supervision. Human demos would be higher quality but require significant data collection effort.

### 6. CLIP Fine-tuning vs Frozen CLIP
**Chosen**: Fine-tune CLIP on driving data
**Alternative**: Use frozen pretrained CLIP
**Rationale**: Fine-tuning adapts the model to driving-specific semantics (e.g., "merge", "lane change") which may not be well-represented in CLIP's original training data (web images).

## Future Extensions

### 1. Trust-Region Methods
Add HAPPO/HATRPO implementation for stronger convergence guarantees in safety-critical scenarios.

### 2. Attention Mechanisms
Replace concatenation fusion with cross-attention between CLIP embeddings and kinematics for more expressive feature interaction.

### 3. Recurrent Policies
Add LSTM/GRU layers to actor-critic for temporal reasoning (e.g., predicting other vehicles' future trajectories).

### 4. Curriculum Learning
Start training on highway (easiest), then merge, then intersection (hardest) to improve sample efficiency.

### 5. Multi-Task Learning
Train single policy across all three scenarios with task conditioning (e.g., scenario ID as input).

### 6. Sim-to-Real Transfer
Evaluate trained policies in more realistic simulators (CARLA, SUMO) or real-world datasets (nuScenes, Waymo).

### 7. Interpretability
Visualize CLIP attention maps to understand which visual features drive policy decisions.

### 8. Safety Constraints
Add constrained RL (e.g., CPO, PCPO) to enforce hard collision avoidance constraints during training.
