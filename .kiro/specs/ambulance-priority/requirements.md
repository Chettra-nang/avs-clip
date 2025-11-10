# Requirements Document

## Introduction

This document specifies requirements for extending the VLM-MARL Highway system to support heterogeneous agent roles, specifically implementing an emergency vehicle priority scenario where one ambulance (Agent 0) must navigate through traffic while three cooperative normal vehicles (Agents 1-3) learn to yield and clear lanes. This builds upon the existing homogeneous multi-agent system by introducing role-based policies, asymmetric rewards, and priority-aware behaviors.

## Glossary

- **Heterogeneous_MARL_System**: Multi-agent system with agents having different roles, policies, and objectives
- **Ambulance_Agent**: Agent 0 with priority status that must maximize speed while minimizing collisions
- **Normal_Agent**: Agents 1-3 that must yield to the ambulance and clear lanes
- **Role_Encoding**: Additional observation feature indicating agent type (0=ambulance, 1=normal, 2=NPC)
- **HARL**: Heterogeneous-Agent Reinforcement Learning framework for mixed agent types
- **Priority_Passage**: Behavior where normal vehicles detect and yield to emergency vehicles
- **Lane_Clearance_Reward**: Reward signal for normal agents that successfully move out of ambulance's path
- **Blocking_Penalty**: Penalty for normal agents positioned ahead of ambulance in same lane
- **Asymmetric_Reward**: Different reward functions for different agent roles

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to train heterogeneous agents with different roles and priorities, so that I can study emergency vehicle coordination and priority passage behaviors.

#### Acceptance Criteria

1. THE Heterogeneous_MARL_System SHALL configure Agent 0 as Ambulance_Agent with priority status
2. THE Heterogeneous_MARL_System SHALL configure Agents 1-3 as Normal_Agent with yielding behavior
3. WHEN the environment initializes, THE Heterogeneous_MARL_System SHALL assign distinct visual markers (ambulance=red, normal=green) to controlled vehicles
4. THE Heterogeneous_MARL_System SHALL maintain the existing 4-agent MultiAgentObservation and MultiAgentAction configuration
5. THE Heterogeneous_MARL_System SHALL preserve backward compatibility with the homogeneous system through a configuration flag

### Requirement 2

**User Story:** As a researcher, I want agents to observe role information in their observations, so that they can learn role-aware policies.

#### Acceptance Criteria

1. THE Heterogeneous_MARL_System SHALL extend the kinematics observation from 7 features to 8 features per vehicle
2. THE Role_Encoding feature SHALL be the first element in each vehicle's observation vector
3. THE Role_Encoding SHALL use value 0.0 for Ambulance_Agent, 1.0 for Normal_Agent, and 2.0 for NPC vehicles
4. WHEN an agent observes nearby vehicles, THE Heterogeneous_MARL_System SHALL include role information for all 8 observed vehicles
5. THE observation dimension SHALL increase from 56 (8 vehicles × 7 features) to 64 (8 vehicles × 8 features)

### Requirement 3

**User Story:** As a researcher, I want separate actor networks for ambulance and normal agents, so that each role can learn specialized behaviors.

#### Acceptance Criteria

1. THE Heterogeneous_MARL_System SHALL implement a HeterogeneousActor class with two sub-networks
2. THE ambulance_actor sub-network SHALL process observations for Agent 0
3. THE normal_actor sub-network SHALL process observations for Agents 1-3
4. WHEN the HeterogeneousActor performs a forward pass, THE Heterogeneous_MARL_System SHALL route each agent's observation to its role-specific sub-network
5. THE centralized critic SHALL remain shared across all agents to enable cooperative credit assignment
6. THE HeterogeneousActor SHALL accept obs_dim=64 to accommodate role encoding

### Requirement 4

**User Story:** As a researcher, I want asymmetric reward functions for different agent roles, so that the ambulance maximizes speed while normal agents maximize lane clearance.

#### Acceptance Criteria

1. THE Ambulance_Agent SHALL receive reward computed as: 0.7 × speed_reward + 0.3 × (-collision_penalty)
2. THE Normal_Agent SHALL receive reward computed as: 0.5 × lane_clearance_reward + 0.5 × (-blocking_penalty)
3. THE lane_clearance_reward SHALL be 1.0 when the agent successfully moves to a lane different from the ambulance's lane
4. THE blocking_penalty SHALL be 2.0 when the agent is within 20 meters ahead of the ambulance in the same lane
5. THE speed_reward for Ambulance_Agent SHALL be normalized velocity divided by maximum velocity (30 m/s)
6. THE Heterogeneous_MARL_System SHALL compute per-agent rewards instead of mean reward across all agents

### Requirement 5

**User Story:** As a researcher, I want role-specific expert heuristics for data collection, so that the ambulance demonstrates aggressive driving and normal agents demonstrate yielding behavior.

#### Acceptance Criteria

1. THE Ambulance_Agent expert heuristic SHALL prioritize FASTER action when front_dist > 30.0 meters
2. THE Ambulance_Agent expert heuristic SHALL attempt lane changes when front_dist < 30.0 meters and a vehicle is ahead
3. THE Normal_Agent expert heuristic SHALL detect when the ambulance is within 50 meters behind in the same lane
4. WHEN a Normal_Agent detects the ambulance behind, THE expert heuristic SHALL prioritize LANE_RIGHT action to clear leftmost lanes
5. WHEN a Normal_Agent detects the ambulance behind and cannot change lanes, THE expert heuristic SHALL execute SLOWER action to create a gap
6. WHEN a Normal_Agent does not detect the ambulance behind, THE expert heuristic SHALL follow the existing TTC-based safety logic

### Requirement 6

**User Story:** As a researcher, I want CLIP training to learn role-specific semantic patterns, so that the vision encoder understands ambulance priority context.

#### Acceptance Criteria

1. THE Data_Collector SHALL format text labels for Ambulance_Agent as: "AMBULANCE {action_name}: {scene_dict}"
2. THE Data_Collector SHALL format text labels for Normal_Agent as: "YIELD {action_name}: {scene_dict}"
3. THE scene_dict for Normal_Agent SHALL include ambulance_distance field when ambulance is within 50 meters
4. THE CLIP_Encoder SHALL train on role-prefixed text labels to learn semantic associations
5. THE Heterogeneous_MARL_System SHALL preserve the existing CLIP fine-tuning pipeline with updated text format

### Requirement 7

**User Story:** As a researcher, I want to configure heterogeneous vs homogeneous mode via YAML, so that I can easily switch between research scenarios.

#### Acceptance Criteria

1. THE environment configuration YAML SHALL include a heterogeneous_agents boolean flag
2. WHEN heterogeneous_agents is true, THE Heterogeneous_MARL_System SHALL enable role-based policies and asymmetric rewards
3. WHEN heterogeneous_agents is false, THE Heterogeneous_MARL_System SHALL use the existing homogeneous shared-actor configuration
4. THE Heterogeneous_MARL_System SHALL validate that heterogeneous mode requires exactly 4 controlled vehicles
5. THE configuration SHALL specify agent_roles list: [ambulance, normal, normal, normal]

### Requirement 8

**User Story:** As a researcher, I want to visualize agent roles in rendered frames, so that I can verify ambulance and normal agent behaviors during data collection.

#### Acceptance Criteria

1. THE Heterogeneous_MARL_System SHALL render Ambulance_Agent with red color (RGB: 255, 0, 0)
2. THE Heterogeneous_MARL_System SHALL render Normal_Agent with green color (RGB: 0, 255, 0)
3. THE Heterogeneous_MARL_System SHALL render NPC vehicles with blue color (RGB: 0, 0, 255)
4. WHEN data collection saves frames, THE images SHALL clearly distinguish ambulance (red) from normal agents (green)
5. THE TensorBoard logger SHALL log sample frames showing color-coded agent roles

### Requirement 9

**User Story:** As a researcher, I want to measure priority passage success metrics, so that I can evaluate how well normal agents yield to the ambulance.

#### Acceptance Criteria

1. THE Heterogeneous_MARL_System SHALL track ambulance_average_speed metric across episodes
2. THE Heterogeneous_MARL_System SHALL track lane_clearance_rate metric (percentage of timesteps where normal agents are not blocking)
3. THE Heterogeneous_MARL_System SHALL track collision_rate metric separately for ambulance vs normal agents
4. THE Heterogeneous_MARL_System SHALL log priority_passage_success metric (1.0 if ambulance reaches destination without collisions)
5. THE TensorBoard_Logger SHALL create separate scalar plots for ambulance metrics vs normal agent metrics

### Requirement 10

**User Story:** As a researcher, I want the heterogeneous system to integrate seamlessly with existing MAPPO training, so that I can leverage the current VLM-MARL pipeline.

#### Acceptance Criteria

1. THE HeterogeneousActor SHALL be a drop-in replacement for the existing VLAActorCritic actor network
2. THE MAPPO_Trainer SHALL detect heterogeneous mode from configuration and instantiate HeterogeneousActor accordingly
3. THE rollout collection SHALL compute per-agent rewards instead of mean reward when in heterogeneous mode
4. THE GAE computation SHALL remain unchanged (centralized value function)
5. THE PPO update SHALL apply gradient updates to both ambulance_actor and normal_actor sub-networks
6. THE model saving SHALL serialize both sub-networks to models/vla_mappo_heterogeneous_{scenario}.pth
