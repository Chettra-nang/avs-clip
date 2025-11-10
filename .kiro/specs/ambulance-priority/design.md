# Design Document

## Overview

The Heterogeneous Ambulance Priority system extends the existing VLM-MARL Highway architecture to support mixed agent roles with asymmetric objectives. The system implements a realistic emergency vehicle scenario where one ambulance must navigate through traffic while three cooperative normal vehicles learn to detect and yield to the priority vehicle. This design follows the HARL (Heterogeneous-Agent Reinforcement Learning) framework while maintaining compatibility with the existing MAPPO+CLIP pipeline.

**Key Design Principles:**
1. **Minimal Disruption**: Extend existing architecture rather than rewrite
2. **Role-Based Specialization**: Separate policies for ambulance vs normal agents
3. **Asymmetric Cooperation**: Different rewards but shared centralized critic
4. **Backward Compatibility**: Configuration flag to enable/disable heterogeneous mode

## Architecture

### System Components (Extensions to Existing)

```
vlm-marl/
├── configs/envs/
│   └── highway_heterogeneous.yaml  # NEW: Heterogeneous config
├── common/
│   ├── utils.py                    # EXTEND: Add role-aware reward functions
│   └── scenario.py                 # EXTEND: Support agent_roles config
├── models/
│   └── heterogeneous_mappo.py      # NEW: HeterogeneousActor class
├── data_collection/
│   └── collect_data.py             # EXTEND: Role-specific expert heuristics
└── training/
    └── train_mappo.py              # EXTEND: Detect and use heterogeneous mode
```

### Data Flow (Heterogeneous Mode)

```
Environment Config (heterogeneous_agents: true)
    ↓
Agent Roles: [ambulance, normal, normal, normal]
    ↓
Observations: [role, presence, x, y, vx, vy, cos_h, sin_h] × 8 vehicles
    ↓
HeterogeneousActor:
  ├─ ambulance_actor(obs[0]) → logits[0]
  └─ normal_actor(obs[1:4]) → logits[1:4]
    ↓
Actions: (ambulance_action, normal_action_1, normal_action_2, normal_action_3)
    ↓
Asymmetric Rewards:
  ├─ ambulance: 0.7*speed + 0.3*(-collision)
  └─ normal: 0.5*lane_clear + 0.5*(-blocking)
    ↓
Centralized Critic: V(joint_obs, clip_emb)
    ↓
MAPPO Update: Both sub-networks
```

## Components and Interfaces

### 1. Configuration Extension (configs/envs/highway_heterogeneous.yaml)

**Purpose**: Enable heterogeneous mode with role assignments.

**New Fields**:
```yaml
id: highway-v0
render_mode: rgb_array
heterogeneous_agents: true  # NEW FLAG
agent_roles:                # NEW FIELD
  - ambulance
  - normal
  - normal
  - normal
agent_colors:               # NEW FIELD
  ambulance: [255, 0, 0]    # Red
  normal: [0, 255, 0]       # Green
  npc: [0, 0, 255]          # Blue
config:
  observation:
    type: MultiAgentObservation
    observation_config:
      type: Kinematics
      vehicles_count: 8
      features: [role, presence, x, y, vx, vy, cos_h, sin_h]  # ADDED 'role'
      absolute: true
      flatten: true
  # ... rest of config unchanged
```

**Design Rationale**: Declarative configuration allows easy switching between homogeneous and heterogeneous modes without code changes.

### 2. Role-Aware Observation Space

**Current Observation** (per agent):
```python
# Shape: (56,) = 8 vehicles × 7 features
[presence, x, y, vx, vy, cos_h, sin_h] × 8
```

**New Observation** (per agent):
```python
# Shape: (64,) = 8 vehicles × 8 features
[role, presence, x, y, vx, vy, cos_h, sin_h] × 8

# role encoding:
# 0.0 = ambulance (Agent 0)
# 1.0 = normal agent (Agents 1-3)
# 2.0 = NPC vehicle
```

**Implementation**:
```python
def add_role_encoding(obs: np.ndarray, controlled_vehicles, all_vehicles) -> np.ndarray:
    """
    Add role encoding as first feature for each vehicle.
    
    Args:
        obs: Original observation [8, 7]
        controlled_vehicles: List of controlled vehicle objects
        all_vehicles: List of all observed vehicle objects
        
    Returns:
        Extended observation [8, 8] with role encoding
    """
    roles = []
    for vehicle in all_vehicles[:8]:
        if vehicle == controlled_vehicles[0]:
            roles.append(0.0)  # Ambulance
        elif vehicle in controlled_vehicles[1:4]:
            roles.append(1.0)  # Normal
        else:
            roles.append(2.0)  # NPC
    
    # Reshape obs from (56,) to (8, 7)
    obs_reshaped = obs.reshape(8, 7)
    
    # Prepend role column
    roles_col = np.array(roles).reshape(8, 1)
    obs_with_roles = np.concatenate([roles_col, obs_reshaped], axis=1)
    
    # Flatten back to (64,)
    return obs_with_roles.flatten()
```

**Design Rationale**: Role encoding enables agents to identify priority vehicles and adjust behavior accordingly. Prepending (rather than appending) makes it the first feature agents observe.

### 3. Heterogeneous Actor Architecture (models/heterogeneous_mappo.py)

**Class Definition**:
```python
import torch
import torch.nn as nn
from typing import Tuple

class HeterogeneousActor(nn.Module):
    """
    Heterogeneous actor with role-specific sub-networks.
    
    Architecture:
        ambulance_actor: MLP for Agent 0 (ambulance)
        normal_actor: MLP for Agents 1-3 (normal vehicles)
    
    Both sub-networks share the same architecture but have independent weights.
    """
    
    def __init__(self, obs_dim: int = 64, n_actions: int = 5):
        super().__init__()
        
        # Ambulance actor (Agent 0)
        self.ambulance_actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        # Normal actor (Agents 1-3)
        self.normal_actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, obs_per_agent: torch.Tensor, agent_roles: list) -> torch.Tensor:
        """
        Forward pass with role-based routing.
        
        Args:
            obs_per_agent: [n_agents, obs_dim] observations
            agent_roles: ['ambulance', 'normal', 'normal', 'normal']
            
        Returns:
            logits: [n_agents, n_actions] action logits
        """
        logits = []
        
        for i, (obs, role) in enumerate(zip(obs_per_agent, agent_roles)):
            if role == 'ambulance':
                logits.append(self.ambulance_actor(obs))
            else:  # role == 'normal'
                logits.append(self.normal_actor(obs))
        
        return torch.stack(logits)
```

**Integration with Existing VLAActorCritic**:
```python
class HeterogeneousVLAActorCritic(nn.Module):
    """
    Extends VLAActorCritic with heterogeneous actor.
    Critic remains centralized (shared across all agents).
    """
    
    def __init__(self, obs_dim: int, n_agents: int, n_actions: int, 
                 clip_model_path: str, agent_roles: list):
        super().__init__()
        
        # Load CLIP (frozen)
        self.clip = CLIPBackbone(clip_model_path, trainable=False)
        
        # Heterogeneous actor
        self.actor = HeterogeneousActor(obs_dim, n_actions)
        self.agent_roles = agent_roles
        
        # Centralized critic (unchanged)
        critic_input_dim = n_agents * obs_dim + 512  # joint obs + CLIP
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, obs_per_agent, joint_obs, clip_emb):
        """
        Forward pass with heterogeneous actor and centralized critic.
        
        Args:
            obs_per_agent: [n_agents, obs_dim]
            joint_obs: [1, n_agents*obs_dim]
            clip_emb: [1, 512]
            
        Returns:
            logits: [n_agents, n_actions]
            value: [1, 1]
        """
        # Actor: role-based routing
        logits = self.actor(obs_per_agent, self.agent_roles)
        
        # Critic: centralized (unchanged)
        critic_input = torch.cat([joint_obs, clip_emb], dim=-1)
        value = self.critic(critic_input)
        
        return logits, value
```

**Design Rationale**: 
- Separate actors enable role-specific policy learning
- Shared critic enables cooperative credit assignment
- Minimal changes to existing VLAActorCritic interface

### 4. Asymmetric Reward Functions (common/utils.py)

**Implementation**:
```python
def compute_heterogeneous_rewards(
    agents: list,
    ambulance_idx: int = 0,
    collision_occurred: bool = False,
    max_speed: float = 30.0
) -> list:
    """
    Compute role-specific rewards for heterogeneous agents.
    
    Args:
        agents: List of controlled vehicle objects
        ambulance_idx: Index of ambulance agent (default 0)
        collision_occurred: Whether any collision happened this step
        max_speed: Maximum velocity for normalization
        
    Returns:
        rewards: List of per-agent rewards [r0, r1, r2, r3]
    """
    rewards = []
    ambulance = agents[ambulance_idx]
    
    for i, agent in enumerate(agents):
        if i == ambulance_idx:
            # Ambulance reward: maximize speed, minimize collisions
            speed_reward = agent.speed / max_speed
            collision_penalty = 5.0 if collision_occurred else 0.0
            reward = 0.7 * speed_reward - 0.3 * collision_penalty
        else:
            # Normal agent reward: maximize lane clearance, minimize blocking
            lane_clearance_reward = compute_lane_clearance(agent, ambulance)
            blocking_penalty = compute_blocking_penalty(agent, ambulance)
            reward = 0.5 * lane_clearance_reward - 0.5 * blocking_penalty
        
        rewards.append(reward)
    
    return rewards


def compute_lane_clearance(agent, ambulance) -> float:
    """
    Reward for being in a different lane than ambulance.
    
    Returns:
        1.0 if in different lane, 0.0 if in same lane
    """
    same_lane = abs(agent.position[1] - ambulance.position[1]) < 2.0
    return 0.0 if same_lane else 1.0


def compute_blocking_penalty(agent, ambulance, threshold: float = 20.0) -> float:
    """
    Penalty for being ahead of ambulance in same lane.
    
    Args:
        agent: Normal vehicle
        ambulance: Ambulance vehicle
        threshold: Distance threshold (meters)
        
    Returns:
        2.0 if blocking, 0.0 otherwise
    """
    # Check if in same lane
    same_lane = abs(agent.position[1] - ambulance.position[1]) < 2.0
    if not same_lane:
        return 0.0
    
    # Check if ahead of ambulance
    ahead = agent.position[0] > ambulance.position[0]
    if not ahead:
        return 0.0
    
    # Check if within threshold distance
    distance = agent.position[0] - ambulance.position[0]
    if distance < threshold:
        return 2.0
    
    return 0.0
```

**Design Rationale**:
- Ambulance prioritizes speed (70%) over collision avoidance (30%)
- Normal agents balance lane clearance (50%) and not blocking (50%)
- Blocking penalty is high (2.0) to strongly discourage obstructing ambulance

### 5. Role-Specific Expert Heuristics (data_collection/collect_data.py)

**Ambulance Expert**:
```python
def _ambulance_expert_action(ego, nearby_vehicles, can_left, can_right, rng) -> int:
    """
    Aggressive expert heuristic for ambulance.
    
    Priority: Speed > Lane changes > Safety
    """
    # Find closest vehicle ahead
    front_vehicle = None
    front_dist = 999.0
    
    for vehicle in nearby_vehicles:
        if vehicle.position[0] > ego.position[0]:  # Ahead
            dist = vehicle.position[0] - ego.position[0]
            if dist < front_dist:
                front_dist = dist
                front_vehicle = vehicle
    
    # Aggressive logic: prioritize speed
    if front_dist > 30.0:
        return 3  # FASTER
    
    # Obstacle ahead: try to change lanes
    if front_dist < 30.0 and front_vehicle is not None:
        if can_left and rng.random() < 0.6:
            return 0  # LANE_LEFT (prefer left for passing)
        elif can_right:
            return 2  # LANE_RIGHT
        else:
            return 1  # IDLE (wait for gap)
    
    # Default: maintain speed
    return 3  # FASTER
```

**Normal Agent Expert**:
```python
def _normal_expert_action(ego, nearby_vehicles, ambulance, can_left, can_right, rng) -> int:
    """
    Yielding expert heuristic for normal agents.
    
    Priority: Detect ambulance > Clear lane > Safety > Normal driving
    """
    # Check if ambulance is behind in same lane
    ambulance_behind = False
    ambulance_distance = 999.0
    
    if ambulance is not None:
        same_lane = abs(ego.position[1] - ambulance.position[1]) < 2.0
        behind = ambulance.position[0] < ego.position[0]
        distance = ego.position[0] - ambulance.position[0]
        
        if same_lane and behind and distance < 50.0:
            ambulance_behind = True
            ambulance_distance = distance
    
    # Priority 1: Yield to ambulance
    if ambulance_behind:
        # Try to move right (clear leftmost lanes for ambulance)
        if can_right:
            return 2  # LANE_RIGHT
        elif can_left:
            return 0  # LANE_LEFT
        else:
            return 4  # SLOWER (create gap)
    
    # Priority 2: Normal safety-based driving (existing TTC logic)
    front_vehicle = None
    front_dist = 999.0
    front_ttc = 999.0
    
    for vehicle in nearby_vehicles:
        if vehicle.position[0] > ego.position[0]:
            dist = vehicle.position[0] - ego.position[0]
            if dist < front_dist:
                front_dist = dist
                front_vehicle = vehicle
                front_ttc = ttc(ego.position, ego.velocity, 
                               vehicle.position, vehicle.velocity)
    
    # Emergency logic (existing)
    if front_ttc < 2.0 or front_dist < 10.0:
        if can_left and rng.random() < 0.5:
            return 0  # LANE_LEFT
        elif can_right:
            return 2  # LANE_RIGHT
        else:
            return 4  # SLOWER
    
    # Caution logic (existing)
    if front_dist < 20.0 and front_ttc < 4.0:
        return 4  # SLOWER
    
    # Random exploration (existing)
    if rng.random() < 0.12:
        if can_left and can_right:
            return 0 if rng.random() < 0.5 else 2
        elif can_left:
            return 0
        elif can_right:
            return 2
    
    # Default: speed optimization (existing)
    return 3 if rng.random() < 0.5 else 1  # FASTER or IDLE
```

**Design Rationale**:
- Ambulance expert is aggressive (prioritizes speed, takes risks)
- Normal expert detects ambulance within 50m and yields
- Normal expert falls back to existing safety logic when no ambulance nearby

### 6. CLIP Text Format Extension

**Current Format**:
```python
text = f"{action_name}: {scene_dict}"
# Example: "FASTER: {'front_dist': 15.2, 'front_ttc': 3.1, 'risk': 0.68}"
```

**New Format (Heterogeneous)**:
```python
def format_clip_text(agent_role: str, action_name: str, scene: dict, 
                     ambulance_distance: float = None) -> str:
    """
    Format text label with role prefix for CLIP training.
    
    Args:
        agent_role: 'ambulance' or 'normal'
        action_name: Action taken (e.g., 'FASTER')
        scene: Scene context dict
        ambulance_distance: Distance to ambulance (for normal agents)
        
    Returns:
        Formatted text string
    """
    if agent_role == 'ambulance':
        return f"AMBULANCE {action_name}: {scene}"
    else:  # normal agent
        if ambulance_distance is not None and ambulance_distance < 50.0:
            scene['ambulance_dist'] = ambulance_distance
        return f"YIELD {action_name}: {scene}"

# Examples:
# Ambulance: "AMBULANCE FASTER: {'front_dist': 35.2, 'front_ttc': 8.1, 'risk': 0.25}"
# Normal (no ambulance): "YIELD IDLE: {'front_dist': 15.2, 'front_ttc': 3.1, 'risk': 0.68}"
# Normal (ambulance nearby): "YIELD LANE_RIGHT: {'front_dist': 20.0, 'ambulance_dist': 25.5, 'risk': 0.45}"
```

**Design Rationale**:
- Role prefix ("AMBULANCE" vs "YIELD") helps CLIP learn semantic associations
- Including ambulance_distance for normal agents provides context for yielding behavior
- Maintains backward compatibility (existing format still works for homogeneous mode)

## Data Models

### Extended Observation
```python
# Per-agent observation (heterogeneous mode)
obs: np.ndarray  # Shape: (64,) = 8 vehicles × 8 features

# Breakdown:
# Vehicle 0: [role, presence, x, y, vx, vy, cos_h, sin_h]
# Vehicle 1: [role, presence, x, y, vx, vy, cos_h, sin_h]
# ...
# Vehicle 7: [role, presence, x, y, vx, vy, cos_h, sin_h]

# role values:
# 0.0 = ambulance (Agent 0)
# 1.0 = normal (Agents 1-3)
# 2.0 = NPC
```

### Heterogeneous Reward
```python
# Per-agent rewards (heterogeneous mode)
rewards: list[float]  # Length: 4

# rewards[0]: Ambulance reward
#   = 0.7 * (speed / 30.0) - 0.3 * collision_penalty
#   Range: [-1.5, 0.7]

# rewards[1-3]: Normal agent rewards
#   = 0.5 * lane_clearance - 0.5 * blocking_penalty
#   Range: [-1.0, 0.5]
```

### Dataset Record (Extended)
```python
{
  "index": int,
  "image": str,
  "episode": int,
  "step": int,
  "quality": float,
  "agents": [
    {
      "role": str,              # NEW: "ambulance" or "normal"
      "speed": float,
      "position": [float, float],
      "heading": float,
      "scene": {
        "front_dist": float,
        "front_ttc": float,
        "risk": float,
        "can_left": bool,
        "can_right": bool,
        "ambulance_dist": float  # NEW: Only for normal agents
      },
      "action_id": int,
      "action_name": str
    }
    // ... 3 more agents
  ],
  "timestamp": str
}
```

## Error Handling

### Configuration Validation
- **Issue**: Invalid agent_roles configuration
- **Solution**: Validate that agent_roles list has exactly 4 elements and first is "ambulance"
- **Implementation**:
```python
def validate_heterogeneous_config(config):
    if config.get('heterogeneous_agents'):
        roles = config.get('agent_roles', [])
        if len(roles) != 4:
            raise ValueError("heterogeneous mode requires exactly 4 agent roles")
        if roles[0] != 'ambulance':
            raise ValueError("Agent 0 must be 'ambulance' in heterogeneous mode")
        if not all(r in ['ambulance', 'normal'] for r in roles):
            raise ValueError("agent_roles must be 'ambulance' or 'normal'")
```

### Observation Dimension Mismatch
- **Issue**: Model trained with obs_dim=56 loaded in heterogeneous mode (obs_dim=64)
- **Solution**: Check obs_dim in saved model and raise clear error
- **Implementation**:
```python
def load_heterogeneous_model(path, obs_dim):
    checkpoint = torch.load(path)
    model_obs_dim = checkpoint['config']['obs_dim']
    if model_obs_dim != obs_dim:
        raise ValueError(
            f"Model obs_dim ({model_obs_dim}) doesn't match config ({obs_dim}). "
            f"Heterogeneous models require obs_dim=64."
        )
```

### Role Detection Failure
- **Issue**: Cannot identify which vehicle is ambulance during data collection
- **Solution**: Use vehicle index (Agent 0 is always ambulance)
- **Implementation**:
```python
def get_agent_role(agent_idx: int) -> str:
    """Agent 0 is ambulance, others are normal."""
    return 'ambulance' if agent_idx == 0 else 'normal'
```

## Testing Strategy

### Unit Tests

**Role Encoding**:
- Test add_role_encoding() with known vehicle configurations
- Verify role values (0.0, 1.0, 2.0) are correctly assigned
- Verify output shape is (64,) instead of (56,)

**Heterogeneous Actor**:
- Test forward pass with 4 agents
- Verify Agent 0 routes to ambulance_actor
- Verify Agents 1-3 route to normal_actor
- Test gradient flow through both sub-networks

**Asymmetric Rewards**:
- Test compute_heterogeneous_rewards() with various scenarios
- Verify ambulance gets speed-based reward
- Verify normal agents get lane_clearance and blocking_penalty
- Test edge cases (same lane, different lanes, various distances)

**Expert Heuristics**:
- Test _ambulance_expert_action() prioritizes FASTER
- Test _normal_expert_action() yields when ambulance behind
- Test fallback to safety logic when no ambulance nearby

### Integration Tests

**Data Collection**:
- Collect 10 episodes in heterogeneous mode
- Verify dataset.jsonl has role field for each agent
- Verify Agent 0 has role="ambulance"
- Verify Agents 1-3 have role="normal"
- Verify ambulance_dist field present for normal agents when applicable

**CLIP Training**:
- Train CLIP for 1 epoch on heterogeneous data
- Verify text labels have "AMBULANCE" and "YIELD" prefixes
- Verify loss decreases
- Verify model saves successfully

**MAPPO Training**:
- Train MAPPO for 1000 steps in heterogeneous mode
- Verify HeterogeneousActor is instantiated
- Verify per-agent rewards are computed (not mean)
- Verify both sub-networks receive gradient updates
- Verify model saves with both ambulance_actor and normal_actor weights

### End-to-End Test

**Heterogeneous Pipeline**:
1. Collect 50 episodes per scenario in heterogeneous mode
2. Verify ambulance (red) and normal agents (green) are visually distinct
3. Fine-tune CLIP for 2 epochs on heterogeneous data
4. Train MAPPO for 10k steps on highway heterogeneous
5. Evaluate trained policy:
   - Ambulance average speed > 25 m/s
   - Lane clearance rate > 60%
   - Collision rate < 20%
   - Priority passage success > 70%

**Success Criteria**:
- Ambulance learns to drive faster than normal agents
- Normal agents learn to detect and yield to ambulance
- System maintains backward compatibility (homogeneous mode still works)

## Performance Considerations

### Memory Usage
- **Heterogeneous Actor**: ~20MB (2 sub-networks × 10MB each)
- **Observation Buffer**: +14% increase (64 vs 56 features)
- **Total**: ~550MB GPU memory (vs 500MB for homogeneous)

### Training Speed
- **Rollout Collection**: Same speed (observation extension is negligible)
- **PPO Update**: ~10% slower (gradient updates for 2 sub-networks)
- **Overall**: ~5% slower end-to-end training time

### Scalability
- **More Agent Types**: Easy to add (e.g., truck, motorcycle) by extending HeterogeneousActor
- **More Agents**: Linear scaling (each new agent adds one sub-network)
- **Larger Scenarios**: Role encoding overhead is constant (1 feature per vehicle)

## Alternative Designs Considered

### 1. Single Actor with Role Conditioning
**Considered**: Pass role as input to single shared actor
**Rejected**: Less expressive than separate networks; harder to learn specialized behaviors

### 2. Separate Critics per Role
**Considered**: Role-specific critics for better credit assignment
**Rejected**: Breaks cooperative learning; centralized critic is core to MAPPO

### 3. Hierarchical Policies
**Considered**: High-level policy selects role, low-level executes
**Rejected**: Over-engineered for 2 roles; adds unnecessary complexity

### 4. Reward Shaping with Potential Functions
**Considered**: Use potential-based reward shaping for smoother learning
**Rejected**: Asymmetric rewards are simpler and more interpretable

## Future Extensions

### 1. Dynamic Role Assignment
Allow agents to switch roles during episode (e.g., ambulance becomes normal after emergency)

### 2. Multi-Ambulance Scenarios
Support multiple priority vehicles with coordination

### 3. Communication Channels
Add explicit communication between agents (e.g., ambulance broadcasts siren signal)

### 4. Attention Mechanisms
Replace concatenation with attention for better role-aware feature fusion

### 5. Curriculum Learning
Start with 1 ambulance + 1 normal, gradually increase to 1 + 3

### 6. Real-World Transfer
Evaluate on nuScenes or Waymo datasets with real emergency vehicle scenarios
