# Requirements Document

## Introduction

This document specifies requirements for a Vision-Language Model (VLM) powered Multi-Agent Reinforcement Learning (MARL) system that trains cooperative autonomous driving agents in highway-env scenarios using MAPPO (Multi-Agent Proximal Policy Optimization) with CLIP (Contrastive Language-Image Pre-training) vision encoders. The system implements Centralized Training with Decentralized Execution (CTDE) and logs all training metrics to TensorBoard.

## Glossary

- **VLM_MARL_System**: The complete training pipeline including data collection, CLIP fine-tuning, and MAPPO training
- **CLIP_Encoder**: OpenAI CLIP ViT-B/32 model that encodes images and text into a shared embedding space
- **MAPPO_Trainer**: Multi-Agent PPO algorithm with centralized critic and decentralized actors
- **Highway_Env**: Gymnasium-based driving simulation environment supporting multi-agent scenarios
- **Data_Collector**: Specialist module that captures high-quality training frames with expert labels
- **TensorBoard_Logger**: PyTorch TensorBoard integration for metrics visualization
- **CTDE**: Centralized Training with Decentralized Execution paradigm
- **Controlled_Vehicle**: Agent vehicle directly controlled by the policy (not NPC)
- **Expert_Heuristic**: Rule-based policy using TTC and gap analysis for action labeling

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to train multiple cooperative agents in realistic highway scenarios, so that I can study emergent multi-agent driving behaviors.

#### Acceptance Criteria

1. WHEN the VLM_MARL_System initializes a scenario, THE Highway_Env SHALL configure exactly 4 Controlled_Vehicles with MultiAgentObservation and MultiAgentAction wrappers
2. THE Highway_Env SHALL support three distinct scenarios: highway-v0 (multilane cruising), merge-v0 (ramp merging), and intersection-v0/v1 (conflict zones)
3. WHEN the Highway_Env executes a step, THE Highway_Env SHALL return observation tuples with one kinematics vector per Controlled_Vehicle
4. THE Highway_Env SHALL accept action tuples with one discrete meta-action (5 choices: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER) per Controlled_Vehicle
5. WHERE intersection-v1 is unavailable, THE VLM_MARL_System SHALL fallback to intersection-v0 without failure

### Requirement 2

**User Story:** As a researcher, I want to collect only informative training data with expert labels, so that I can efficiently train vision-language models without wasting storage on redundant frames.

#### Acceptance Criteria

1. WHEN the Data_Collector evaluates a frame, THE Data_Collector SHALL compute a quality score based on minimum TTC (40% weight), minimum distance (30% weight), and maximum risk (30% weight)
2. THE Data_Collector SHALL save frames where quality score exceeds 0.5 or with 25% random probability
3. WHEN the Data_Collector saves a frame, THE Data_Collector SHALL crop the rendered image to 224×224 pixels centered on the scene
4. FOR each Controlled_Vehicle in a saved frame, THE Data_Collector SHALL compute an Expert_Heuristic action label using TTC, front distance, and lane availability
5. THE Data_Collector SHALL log frame quality, action distributions, and traffic metrics to TensorBoard_Logger during collection
6. THE Data_Collector SHALL output dataset.jsonl with frame metadata and a corresponding images/ directory per scenario

### Requirement 3

**User Story:** As a researcher, I want to fine-tune CLIP on driving-specific image-instruction pairs, so that the vision encoder provides semantically meaningful embeddings for driving decisions.

#### Acceptance Criteria

1. THE CLIP_Encoder SHALL load the pretrained openai/clip-vit-base-patch32 model with HuggingFace transformers
2. WHEN the CLIP_Encoder trains, THE CLIP_Encoder SHALL process image-text pairs where text describes the expert action and scene context
3. THE CLIP_Encoder SHALL train on combined datasets from all three scenarios (highway, merge, intersection)
4. WHEN the CLIP_Encoder completes an optimization step, THE CLIP_Encoder SHALL log training loss, sample images, and sample texts to TensorBoard_Logger
5. THE CLIP_Encoder SHALL save the fine-tuned model and processor to models/clip/ directory after training completes

### Requirement 4

**User Story:** As a researcher, I want to train decentralized actors with a centralized critic using MAPPO, so that agents learn cooperative policies while maintaining independent execution capability.

#### Acceptance Criteria

1. THE MAPPO_Trainer SHALL implement a shared actor network that takes per-agent kinematics concatenated with CLIP image embeddings as input
2. THE MAPPO_Trainer SHALL implement a centralized critic network that takes joint kinematics of all agents concatenated with global CLIP image embeddings as input
3. WHEN the MAPPO_Trainer collects a rollout, THE MAPPO_Trainer SHALL render the environment to obtain RGB frames and encode them with CLIP_Encoder
4. THE MAPPO_Trainer SHALL compute Generalized Advantage Estimation (GAE) with gamma=0.99 and lambda=0.95 over the centralized value function
5. THE MAPPO_Trainer SHALL perform PPO updates with policy clipping (epsilon=0.2), value function coefficient (0.5), and entropy coefficient (0.01)
6. WHEN the MAPPO_Trainer completes a rollout, THE MAPPO_Trainer SHALL log episode returns, policy loss, value loss, and entropy to TensorBoard_Logger
7. THE MAPPO_Trainer SHALL save the trained actor-critic model to models/vla_mappo_{scenario}.pth after training completes

### Requirement 5

**User Story:** As a researcher, I want all training metrics logged to TensorBoard, so that I can monitor and compare experiments without external dependencies.

#### Acceptance Criteria

1. THE TensorBoard_Logger SHALL create separate log directories for data collection, CLIP training, and MAPPO training
2. WHEN any training component logs a scalar metric, THE TensorBoard_Logger SHALL write it with the appropriate tag and global step
3. WHEN the Data_Collector logs sample data, THE TensorBoard_Logger SHALL write image previews with HWC format
4. THE TensorBoard_Logger SHALL NOT depend on Weights & Biases or any external logging service
5. THE VLM_MARL_System SHALL provide a single tensorboard --logdir runs command to visualize all logged metrics

### Requirement 6

**User Story:** As a researcher, I want a modular codebase with clear separation of concerns, so that I can easily modify individual components without breaking the pipeline.

#### Acceptance Criteria

1. THE VLM_MARL_System SHALL organize code into separate directories: configs/, common/, envs/, data_collection/, models/, and training/
2. THE VLM_MARL_System SHALL define environment configurations in YAML files with scenario-specific parameters
3. THE VLM_MARL_System SHALL provide a make_env() factory function that loads environments from YAML configurations
4. THE VLM_MARL_System SHALL implement reusable utilities (TensorBoard logger, risk scoring, action mappings) in common/ module
5. THE VLM_MARL_System SHALL provide a run.sh orchestration script that executes data collection, CLIP training, and MAPPO training sequentially

### Requirement 7

**User Story:** As a researcher, I want the system to handle multi-agent observations and actions correctly, so that all controlled vehicles receive proper policy outputs.

#### Acceptance Criteria

1. WHEN the VLM_MARL_System receives observation tuples from Highway_Env, THE VLM_MARL_System SHALL flatten each agent's kinematics vector and stack them into a per-agent array
2. THE VLM_MARL_System SHALL concatenate all agents' kinematics into a joint observation vector for the centralized critic
3. WHEN the MAPPO_Trainer samples actions, THE MAPPO_Trainer SHALL produce one discrete action per Controlled_Vehicle
4. THE MAPPO_Trainer SHALL convert sampled action tensors to a tuple of Python integers before passing to Highway_Env.step()
5. WHEN the Highway_Env returns rewards as a tuple, THE MAPPO_Trainer SHALL compute the mean cooperative return for logging

### Requirement 8

**User Story:** As a researcher, I want clear documentation of design choices with academic citations, so that I can understand the theoretical foundations and reproduce results.

#### Acceptance Criteria

1. THE VLM_MARL_System SHALL document the use of MultiAgentObservation and MultiAgentAction with references to highway-env documentation
2. THE VLM_MARL_System SHALL document the MAPPO algorithm with citation to the original arXiv paper
3. THE VLM_MARL_System SHALL document the CLIP integration approach with references to vision-language driving research (e.g., DriveLM)
4. THE VLM_MARL_System SHALL document the TensorBoard logging approach with references to PyTorch documentation
5. THE VLM_MARL_System SHALL provide a README.md with installation instructions, usage examples, and expected outcomes
