"""
MAPPO (Multi-Agent Proximal Policy Optimization) training pipeline.

Implements Centralized Training with Decentralized Execution (CTDE) with CLIP vision-language fusion.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, List, Dict
import highway_env  # Register highway-env environments

from common.scenario import make_hwy_env_from_yaml, apply_vehicle_colors
from common.logger_tb import TBLogger
from models.vla_mappo import VLAActorCritic
from models.heterogeneous_mappo import HeterogeneousVLAActorCritic
from common.utils import compute_heterogeneous_rewards


def flatten_obs_tuple(obs_tuple: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert tuple observations to per-agent and joint arrays.
    
    Args:
        obs_tuple: Tuple of observations, one per agent [agent_0_obs, agent_1_obs, ...]
                   Each element is a numpy array of shape (obs_dim,) or (vehicles, features)
    
    Returns:
        per_agent: Stacked per-agent observations [n_agents, obs_dim]
        joint: Concatenated joint observation [n_agents * obs_dim]
    """
    # Flatten each agent's observation if needed (e.g., from (8, 7) to (56,))
    flattened_obs = []
    for obs in obs_tuple:
        if obs.ndim > 1:
            obs = obs.flatten()
        flattened_obs.append(obs)
    
    # Stack observations into per-agent array
    per_agent = np.stack(flattened_obs, axis=0)  # [n_agents, obs_dim]
    
    # Flatten to joint observation
    joint = per_agent.flatten()  # [n_agents * obs_dim]
    
    return per_agent, joint


def collect_rollout(
    env,
    net,
    device: torch.device,
    horizon: int,
    text_prompt: str = None,
    is_heterogeneous: bool = False,
    heterogeneous_config: Dict = None,
    logger = None,
    global_step: int = 0
) -> Tuple[List[Dict], float, Dict]:
    """
    Collect a rollout of experiences from the environment.
    
    Args:
        env: Gymnasium environment
        net: VLAActorCritic or HeterogeneousVLAActorCritic network
        device: torch device
        horizon: Maximum steps per rollout
        text_prompt: Optional text prompt for CLIP encoding
        is_heterogeneous: Whether to use per-agent heterogeneous rewards
        heterogeneous_config: Configuration dict with agent_roles and agent_colors
        logger: Optional TensorBoard logger for frame logging
        global_step: Global step counter for logging
    
    Returns:
        trajectories: List of transition dictionaries
        episode_return: Total episode return
        metrics: Dictionary of episode metrics (for heterogeneous mode)
    """
    obs, info = env.reset()
    
    # Apply vehicle colors if in heterogeneous mode
    if heterogeneous_config is not None:
        apply_vehicle_colors(
            env,
            heterogeneous_config['agent_roles'],
            heterogeneous_config['agent_colors']
        )
    
    trajectories = []
    episode_return = 0.0
    
    # Initialize metrics tracking for heterogeneous mode
    metrics = {
        'ambulance_speeds': [],
        'lane_clearance_steps': 0,
        'total_steps': 0,
        'ambulance_collision': False,
        'normal_collisions': 0,
        'episode_success': False
    }
    
    for t in range(horizon):
        # Render RGB frame and encode with CLIP
        frame = env.render()  # [H, W, 3] numpy array
        clip_emb = net._embed(frame, text_prompt=text_prompt)  # [1, 512]
        
        # Log sample frames every 50 steps (for visual inspection during training)
        if logger is not None and t % 50 == 0:
            logger.image('training/sample_frame', frame, global_step + t, dataformats='HWC')
        
        # Flatten observations to per-agent and joint arrays
        per_agent, joint = flatten_obs_tuple(obs)  # [N, D], [N*D]
        
        # Convert to tensors
        per_agent_t = torch.from_numpy(per_agent).float().to(device)  # [N, obs_dim]
        joint_t = torch.from_numpy(joint).float().unsqueeze(0).to(device)  # [1, N*obs_dim]
        
        # Forward pass through actor-critic
        with torch.no_grad():
            logits, value = net(per_agent_t, joint_t, clip_emb)  # [N, n_actions], [1, 1]
        
        # Sample actions using Categorical distribution
        dist = Categorical(logits=logits)
        actions = dist.sample()  # [N]
        log_probs = dist.log_prob(actions)  # [N]
        
        # Convert action tensors to tuple of ints for environment step
        actions_tuple = tuple([int(a.item()) for a in actions])
        
        # Environment step
        next_obs, reward, done, trunc, info = env.step(actions_tuple)
        
        # Compute rewards based on mode
        if is_heterogeneous:
            # Get controlled vehicles from environment for heterogeneous reward computation
            controlled_vehicles = env.unwrapped.controlled_vehicles
            collision_occurred = info.get('crashed', False)
            
            # Compute per-agent heterogeneous rewards
            per_agent_rewards = compute_heterogeneous_rewards(
                agents=controlled_vehicles,
                ambulance_idx=0,
                collision_occurred=collision_occurred,
                max_speed=30.0
            )
            reward_scalar = float(np.mean(per_agent_rewards))  # Mean for logging
            reward_storage = per_agent_rewards  # Store per-agent rewards
            
            # Track heterogeneous metrics
            metrics['total_steps'] += 1
            
            # Track ambulance speed
            ambulance = controlled_vehicles[0]
            metrics['ambulance_speeds'].append(ambulance.speed)
            
            # Track lane clearance (count steps where normal agents are NOT blocking)
            from common.utils import compute_blocking_penalty
            blocking_count = 0
            for i in range(1, len(controlled_vehicles)):
                if compute_blocking_penalty(controlled_vehicles[i], ambulance) > 0:
                    blocking_count += 1
            
            # If no normal agents are blocking, increment clearance counter
            if blocking_count == 0:
                metrics['lane_clearance_steps'] += 1
            
            # Track collisions per agent type
            if collision_occurred:
                # Check which agent(s) collided
                # In highway-env, crashed flag applies to all agents, so we track by type
                metrics['ambulance_collision'] = True
                # Assume normal agents also involved if collision occurred
                metrics['normal_collisions'] += 1
        else:
            # Homogeneous mode: use mean reward
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward_scalar = float(np.mean(reward))
            else:
                reward_scalar = float(reward)
            reward_storage = reward_scalar
        
        # Store transition
        trajectories.append({
            'obs': per_agent,  # numpy [N, obs_dim]
            'joint': joint,  # numpy [N*obs_dim]
            'clip': clip_emb.cpu().numpy(),  # numpy [1, 512]
            'action': actions.cpu().numpy(),  # numpy [N]
            'log_prob': log_probs.cpu().numpy(),  # numpy [N]
            'reward': reward_storage,  # scalar or list depending on mode
            'value': value.item(),  # scalar
            'done': done or trunc
        })
        
        episode_return += reward_scalar
        obs = next_obs
        
        if done or trunc:
            # Check if episode was successful (ambulance reached goal without collision)
            if is_heterogeneous:
                # Success if episode ended without ambulance collision
                metrics['episode_success'] = not metrics['ambulance_collision']
            break
    
    return trajectories, episode_return, metrics


def compute_gae(
    trajectories: List[Dict],
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        trajectories: List of transition dictionaries
        gamma: Discount factor
        lam: GAE lambda parameter
    
    Returns:
        advantages: Array of advantages [T]
        returns: Array of returns [T]
    """
    # Extract values, rewards, and dones
    values = [t['value'] for t in trajectories] + [0.0]  # Append bootstrap value
    
    # Handle both scalar and per-agent rewards
    # Convert per-agent rewards to mean for GAE computation (centralized value function)
    rewards = []
    for t in trajectories:
        r = t['reward']
        if isinstance(r, (list, np.ndarray)):
            rewards.append(float(np.mean(r)))
        else:
            rewards.append(float(r))
    
    dones = [t['done'] for t in trajectories]
    
    # Compute TD deltas with gamma=0.99
    advantages = []
    gae = 0.0
    
    for t in reversed(range(len(rewards))):
        # TD delta: r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        
        # GAE: delta_t + gamma * lambda * (1 - done) * gae_{t+1}
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    advantages = np.array(advantages, dtype=np.float32)
    
    # Compute returns as advantages + values
    returns = advantages + np.array(values[:-1], dtype=np.float32)
    
    return advantages, returns


def ppo_update(
    net,
    optimizer: torch.optim.Optimizer,
    trajectories: List[Dict],
    advantages: np.ndarray,
    returns: np.ndarray,
    device: torch.device,
    epochs: int = 4,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5
) -> Tuple[float, float, float]:
    """
    Perform PPO update on the collected batch.
    
    Args:
        net: VLAActorCritic or HeterogeneousVLAActorCritic network
        optimizer: Adam optimizer (includes parameters from both sub-networks if heterogeneous)
        trajectories: List of transition dictionaries
        advantages: Computed advantages [T]
        returns: Computed returns [T]
        device: torch device
        epochs: Number of PPO epochs
        clip_eps: PPO clipping epsilon
        vf_coef: Value function coefficient
        ent_coef: Entropy coefficient
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        avg_policy_loss: Average policy loss
        avg_value_loss: Average value loss
        avg_entropy: Average entropy
    """
    # Normalize advantages (mean=0, std=1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    update_count = 0
    
    # Loop over PPO epochs (4 epochs)
    for epoch in range(epochs):
        # For each timestep in the batch
        for t in range(len(trajectories)):
            traj = trajectories[t]
            
            # Convert to tensors
            obs_t = torch.from_numpy(traj['obs']).float().to(device)  # [N, obs_dim]
            joint_t = torch.from_numpy(traj['joint']).float().unsqueeze(0).to(device)  # [1, N*obs_dim]
            clip_t = torch.from_numpy(traj['clip']).float().to(device)  # [1, 512]
            action_t = torch.from_numpy(traj['action']).long().to(device)  # [N]
            old_log_prob_t = torch.from_numpy(traj['log_prob']).float().to(device)  # [N]
            advantage_t = torch.tensor(advantages[t], dtype=torch.float32, device=device)  # scalar
            return_t = torch.tensor(returns[t], dtype=torch.float32, device=device)  # scalar
            
            # Forward pass
            logits, value = net(obs_t, joint_t, clip_t)  # [N, n_actions], [1, 1]
            dist = Categorical(logits=logits)
            
            # Compute policy loss (clipped surrogate)
            log_prob = dist.log_prob(action_t)  # [N]
            ratio = torch.exp(log_prob - old_log_prob_t)  # [N]
            
            # Average ratio across agents for cooperative setting
            ratio_mean = ratio.mean()
            
            surr1 = ratio_mean * advantage_t
            surr2 = torch.clamp(ratio_mean, 1 - clip_eps, 1 + clip_eps) * advantage_t
            policy_loss = -torch.min(surr1, surr2)
            
            # Compute value loss (MSE)
            value_loss = F.mse_loss(value.squeeze(), return_t)
            
            # Compute entropy bonus
            entropy = dist.entropy().mean()
            
            # Compute total loss: policy_loss + 0.5*value_loss - 0.01*entropy
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            
            # Backward pass with gradient clipping (max_norm=0.5)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            
            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            update_count += 1
    
    # Return average losses
    avg_policy_loss = total_policy_loss / update_count
    avg_value_loss = total_value_loss / update_count
    avg_entropy = total_entropy / update_count
    
    return avg_policy_loss, avg_value_loss, avg_entropy


def main():
    """
    Main training loop with TensorBoard logging.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train MAPPO with VLA")
    parser.add_argument("--scenario", type=str, default="highway", 
                        choices=["highway", "merge", "intersection",
                                "highway_heterogeneous", "merge_heterogeneous", 
                                "intersection_heterogeneous", "highway_heterogeneous_dense",
                                "merge_multi_agent"],
                        help="Scenario to train on (use *_heterogeneous for ambulance priority, merge_multi_agent for custom multi-agent merge)")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Maximum training steps")
    parser.add_argument("--horizon", type=int, default=256,
                        help="Rollout horizon (steps per episode)")
    parser.add_argument("--logdir", type=str, default="runs",
                        help="TensorBoard log directory")
    parser.add_argument("--clip_dir", type=str, default="models/clip",
                        help="Path to fine-tuned CLIP model")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize environment
    config_path = f"configs/envs/{args.scenario}.yaml"
    env, heterogeneous_config = make_hwy_env_from_yaml(config_path)
    print(f"Loaded environment: {args.scenario}")
    
    # Check if heterogeneous mode is enabled
    is_heterogeneous = heterogeneous_config is not None and heterogeneous_config.get('heterogeneous_agents', False)
    agent_roles = heterogeneous_config.get('agent_roles', []) if is_heterogeneous else None
    
    if is_heterogeneous:
        print(f"Heterogeneous mode enabled with roles: {agent_roles}")
    else:
        print("Homogeneous mode (standard MAPPO)")
    
    # Get observation dimensions
    obs, info = env.reset()
    n_agents = len(obs)
    # Flatten observation to get actual dimension
    if obs[0].ndim > 1:
        obs_dim = obs[0].flatten().shape[0]
    else:
        obs_dim = obs[0].shape[0]
    n_actions = env.action_space[0].n  # Assuming all agents have same action space
    
    print(f"Number of agents: {n_agents}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Number of actions: {n_actions}")
    
    # Create VLAActorCritic network (heterogeneous or homogeneous)
    # Check if fine-tuned CLIP exists, otherwise use pretrained
    if os.path.exists(args.clip_dir):
        clip_model_path = args.clip_dir
        print(f"Loading fine-tuned CLIP from: {clip_model_path}")
    else:
        clip_model_path = "openai/clip-vit-base-patch32"
        print(f"Using pretrained CLIP: {clip_model_path}")
    
    # Instantiate appropriate model based on mode
    if is_heterogeneous:
        net = HeterogeneousVLAActorCritic(
            obs_dim=obs_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            clip_model_path=clip_model_path,
            agent_roles=agent_roles,
            device=device
        )
        print("Created HeterogeneousVLAActorCritic network")
    else:
        net = VLAActorCritic(
            obs_dim=obs_dim,
            n_agents=n_agents,
            n_actions=n_actions,
            clip_model_path=clip_model_path,
            device=device
        )
        print("Created VLAActorCritic network")
    
    # Create optimizer with lr=3e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    # Create TensorBoard logger
    logger = TBLogger(args.logdir, f"mappo_{args.scenario}")
    print(f"Logging to: {logger.log_path}")
    
    # Training loop
    total_steps = 0
    iteration = 0
    episode_returns = []
    
    # Heterogeneous metrics accumulators
    if is_heterogeneous:
        heterogeneous_metrics = {
            'ambulance_speeds': [],
            'lane_clearance_rates': [],
            'ambulance_collisions': 0,
            'normal_collisions': 0,
            'priority_successes': 0,
            'total_episodes': 0
        }
    
    print(f"\nStarting training for {args.steps} steps...")
    
    # Loop until total_steps >= max_steps
    while total_steps < args.steps:
        iteration += 1
        
        # Collect rollout
        trajectories, episode_return, episode_metrics = collect_rollout(
            env, net, device, args.horizon, 
            text_prompt=f"{args.scenario} driving",
            is_heterogeneous=is_heterogeneous,
            heterogeneous_config=heterogeneous_config,
            logger=logger,
            global_step=total_steps
        )
        
        total_steps += len(trajectories)
        episode_returns.append(episode_return)
        
        # Accumulate heterogeneous metrics
        if is_heterogeneous:
            heterogeneous_metrics['total_episodes'] += 1
            
            # Ambulance average speed for this episode
            if len(episode_metrics['ambulance_speeds']) > 0:
                avg_speed = np.mean(episode_metrics['ambulance_speeds'])
                heterogeneous_metrics['ambulance_speeds'].append(avg_speed)
            
            # Lane clearance rate for this episode
            if episode_metrics['total_steps'] > 0:
                clearance_rate = episode_metrics['lane_clearance_steps'] / episode_metrics['total_steps']
                heterogeneous_metrics['lane_clearance_rates'].append(clearance_rate)
            
            # Collision tracking
            if episode_metrics['ambulance_collision']:
                heterogeneous_metrics['ambulance_collisions'] += 1
            heterogeneous_metrics['normal_collisions'] += episode_metrics['normal_collisions']
            
            # Priority passage success
            if episode_metrics['episode_success']:
                heterogeneous_metrics['priority_successes'] += 1
        
        # Compute GAE
        advantages, returns = compute_gae(trajectories, gamma=0.99, lam=0.95)
        
        # PPO update
        policy_loss, value_loss, entropy = ppo_update(
            net, optimizer, trajectories, advantages, returns, device,
            epochs=4, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5
        )
        
        # Log metrics to TensorBoard
        logger.scalar("return/episode", episode_return, total_steps)
        logger.scalar("loss/policy", policy_loss, total_steps)
        logger.scalar("loss/value", value_loss, total_steps)
        logger.scalar("policy/entropy", entropy, total_steps)
        logger.scalar("steps/total", total_steps, iteration)
        
        # Log heterogeneous metrics per episode
        if is_heterogeneous:
            # Ambulance metrics
            if len(episode_metrics['ambulance_speeds']) > 0:
                avg_speed = np.mean(episode_metrics['ambulance_speeds'])
                logger.scalar("ambulance/average_speed", avg_speed, total_steps)
            
            # Lane clearance rate
            if episode_metrics['total_steps'] > 0:
                clearance_rate = episode_metrics['lane_clearance_steps'] / episode_metrics['total_steps']
                logger.scalar("normal/lane_clearance_rate", clearance_rate, total_steps)
            
            # Collision indicators (1.0 if collision, 0.0 otherwise)
            logger.scalar("ambulance/collision", 1.0 if episode_metrics['ambulance_collision'] else 0.0, total_steps)
            logger.scalar("normal/collision", 1.0 if episode_metrics['normal_collisions'] > 0 else 0.0, total_steps)
            
            # Priority passage success (1.0 if success, 0.0 otherwise)
            logger.scalar("heterogeneous/priority_passage_success", 1.0 if episode_metrics['episode_success'] else 0.0, total_steps)
        
        # Log average return over last 100 episodes every 10 iterations
        if iteration % 10 == 0:
            recent_returns = episode_returns[-100:] if len(episode_returns) >= 100 else episode_returns
            avg_return = np.mean(recent_returns)
            logger.scalar("return/avg_100", avg_return, total_steps)
            
            # Log aggregated heterogeneous metrics every 10 iterations
            if is_heterogeneous and heterogeneous_metrics['total_episodes'] > 0:
                # Ambulance average speed (across all episodes)
                if len(heterogeneous_metrics['ambulance_speeds']) > 0:
                    overall_avg_speed = np.mean(heterogeneous_metrics['ambulance_speeds'])
                    logger.scalar("ambulance/overall_average_speed", overall_avg_speed, total_steps)
                
                # Lane clearance rate (across all episodes)
                if len(heterogeneous_metrics['lane_clearance_rates']) > 0:
                    overall_clearance_rate = np.mean(heterogeneous_metrics['lane_clearance_rates'])
                    logger.scalar("normal/overall_lane_clearance_rate", overall_clearance_rate, total_steps)
                
                # Collision rates
                ambulance_collision_rate = heterogeneous_metrics['ambulance_collisions'] / heterogeneous_metrics['total_episodes']
                normal_collision_rate = heterogeneous_metrics['normal_collisions'] / heterogeneous_metrics['total_episodes']
                logger.scalar("ambulance/collision_rate", ambulance_collision_rate, total_steps)
                logger.scalar("normal/collision_rate", normal_collision_rate, total_steps)
                
                # Priority passage success rate
                success_rate = heterogeneous_metrics['priority_successes'] / heterogeneous_metrics['total_episodes']
                logger.scalar("heterogeneous/priority_passage_success_rate", success_rate, total_steps)
            
            # Print statement
            if is_heterogeneous and heterogeneous_metrics['total_episodes'] > 0:
                # Include heterogeneous metrics in print
                overall_avg_speed = np.mean(heterogeneous_metrics['ambulance_speeds']) if len(heterogeneous_metrics['ambulance_speeds']) > 0 else 0.0
                overall_clearance_rate = np.mean(heterogeneous_metrics['lane_clearance_rates']) if len(heterogeneous_metrics['lane_clearance_rates']) > 0 else 0.0
                success_rate = heterogeneous_metrics['priority_successes'] / heterogeneous_metrics['total_episodes']
                
                print(f"Iter {iteration} | Steps {total_steps}/{args.steps} | "
                      f"Return {episode_return:.2f} | Avg100 {avg_return:.2f} | "
                      f"PolicyLoss {policy_loss:.4f} | ValueLoss {value_loss:.4f} | "
                      f"Entropy {entropy:.4f} | "
                      f"AmbSpeed {overall_avg_speed:.2f} | Clearance {overall_clearance_rate:.2%} | "
                      f"Success {success_rate:.2%}")
            else:
                print(f"Iter {iteration} | Steps {total_steps}/{args.steps} | "
                      f"Return {episode_return:.2f} | Avg100 {avg_return:.2f} | "
                      f"PolicyLoss {policy_loss:.4f} | ValueLoss {value_loss:.4f} | "
                      f"Entropy {entropy:.4f}")
    
    # Save trained model
    os.makedirs("models", exist_ok=True)
    
    if is_heterogeneous:
        # Save heterogeneous model with custom save method
        model_path = f"models/vla_mappo_heterogeneous_{args.scenario}.pth"
        net.save(model_path)
        print(f"\nTraining complete! Heterogeneous model saved to: {model_path}")
        print(f"Agent roles: {agent_roles}")
    else:
        # Save homogeneous model with standard state_dict
        model_path = f"models/vla_mappo_{args.scenario}.pth"
        torch.save(net.state_dict(), model_path)
        print(f"\nTraining complete! Model saved to: {model_path}")
    
    # Close logger
    logger.close()
    env.close()


if __name__ == "__main__":
    main()
