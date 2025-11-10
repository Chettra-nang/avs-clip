"""Utility functions for risk assessment and action mapping."""

import numpy as np


# Action mapping dictionary (highway-env DiscreteMetaAction)
ACTIONS_ALL = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER"
}


def ttc(ego_pos: np.ndarray, ego_vel: np.ndarray, 
        other_pos: np.ndarray, other_vel: np.ndarray, 
        eps: float = 1e-6) -> float:
    """
    Calculate time-to-collision (TTC) between ego vehicle and another vehicle.
    
    Args:
        ego_pos: Ego vehicle position [x, y]
        ego_vel: Ego vehicle velocity [vx, vy]
        other_pos: Other vehicle position [x, y]
        other_vel: Other vehicle velocity [vx, vy]
        eps: Small epsilon to prevent division by zero
    
    Returns:
        Time-to-collision in seconds (positive value, or large value if no collision)
    """
    # Relative position and velocity
    rel_pos = other_pos - ego_pos
    rel_vel = ego_vel - other_vel
    
    # Distance
    dist = np.linalg.norm(rel_pos) + eps
    
    # If vehicles are moving apart, TTC is infinite
    closing_speed = np.dot(rel_pos, rel_vel) / dist
    if closing_speed <= 0:
        return 999.0  # Large value indicating no collision
    
    # TTC = distance / closing_speed
    return dist / closing_speed


def risk_score(ego: dict, other: dict, eps: float = 1e-6) -> float:
    """
    Compute composite risk score between ego vehicle and another vehicle.
    
    Weighted combination:
    - 40% TTC (time-to-collision)
    - 30% distance
    - 30% relative velocity
    
    Args:
        ego: Dictionary with 'position' and 'velocity' keys (numpy arrays)
        other: Dictionary with 'position' and 'velocity' keys (numpy arrays)
        eps: Small epsilon for numerical stability
    
    Returns:
        Risk score in [0, 1] range (higher = more risky)
    """
    # Calculate TTC
    ttc_value = ttc(ego['position'], ego['velocity'], 
                    other['position'], other['velocity'], eps)
    
    # Calculate distance
    dist = np.linalg.norm(other['position'] - ego['position']) + eps
    
    # Calculate relative velocity magnitude
    rel_vel = np.linalg.norm(ego['velocity'] - other['velocity'])
    
    # Normalize components to [0, 1] range
    # TTC: lower is riskier (inverse relationship)
    ttc_risk = np.clip(1.0 / (ttc_value + 1.0), 0.0, 1.0)
    
    # Distance: closer is riskier (inverse relationship, normalize by 50m)
    dist_risk = np.clip(1.0 - (dist / 50.0), 0.0, 1.0)
    
    # Relative velocity: higher is riskier (normalize by 30 m/s)
    vel_risk = np.clip(rel_vel / 30.0, 0.0, 1.0)
    
    # Weighted combination
    risk = 0.4 * ttc_risk + 0.3 * dist_risk + 0.3 * vel_risk
    
    return float(risk)


def add_role_encoding(obs: np.ndarray, controlled_vehicles: list, 
                     all_vehicles: list) -> np.ndarray:
    """
    Add role encoding as first feature for each vehicle in observation.
    
    Extends observation from (56,) to (64,) by prepending role feature:
    - 0.0 = ambulance (Agent 0)
    - 1.0 = normal agent (Agents 1-3)
    - 2.0 = NPC vehicle
    
    Args:
        obs: Original observation array, shape (56,) = 8 vehicles × 7 features
        controlled_vehicles: List of controlled vehicle objects
        all_vehicles: List of all observed vehicle objects
        
    Returns:
        Extended observation array, shape (64,) = 8 vehicles × 8 features
    """
    # Reshape obs from (56,) to (8, 7)
    obs_reshaped = obs.reshape(8, 7)
    
    # Determine role for each of the 8 observed vehicles
    roles = []
    for i in range(8):
        if i < len(all_vehicles):
            vehicle = all_vehicles[i]
            
            # Check if this vehicle is Agent 0 (ambulance)
            if len(controlled_vehicles) > 0 and vehicle == controlled_vehicles[0]:
                roles.append(0.0)  # Ambulance
            # Check if this vehicle is Agents 1-3 (normal)
            elif vehicle in controlled_vehicles[1:4]:
                roles.append(1.0)  # Normal agent
            else:
                roles.append(2.0)  # NPC vehicle
        else:
            # No vehicle at this index (shouldn't happen with presence feature)
            roles.append(2.0)  # Default to NPC
    
    # Create role column: shape (8, 1)
    roles_col = np.array(roles, dtype=np.float32).reshape(8, 1)
    
    # Prepend role column to observation: (8, 1) + (8, 7) = (8, 8)
    obs_with_roles = np.concatenate([roles_col, obs_reshaped], axis=1)
    
    # Flatten back to (64,)
    return obs_with_roles.flatten()


def compute_lane_clearance(agent, ambulance) -> float:
    """
    Compute lane clearance reward for normal agents.
    
    Rewards agents for being in a different lane than the ambulance,
    encouraging them to clear the ambulance's path.
    
    Args:
        agent: Normal vehicle object with position attribute
        ambulance: Ambulance vehicle object with position attribute
        
    Returns:
        1.0 if in different lane, 0.0 if in same lane
    """
    # Check if in same lane (y-coordinate difference < 2.0 meters)
    same_lane = abs(agent.position[1] - ambulance.position[1]) < 2.0
    
    # Return 1.0 for different lane (reward), 0.0 for same lane
    return 0.0 if same_lane else 1.0


def compute_blocking_penalty(agent, ambulance, threshold: float = 20.0) -> float:
    """
    Compute blocking penalty for normal agents.
    
    Penalizes agents that are ahead of the ambulance in the same lane,
    especially when within a close distance threshold.
    
    Args:
        agent: Normal vehicle object with position attribute
        ambulance: Ambulance vehicle object with position attribute
        threshold: Distance threshold in meters (default: 20.0)
        
    Returns:
        2.0 if blocking (within threshold ahead in same lane), 0.0 otherwise
    """
    # Check if in same lane (y-coordinate difference < 2.0 meters)
    same_lane = abs(agent.position[1] - ambulance.position[1]) < 2.0
    if not same_lane:
        return 0.0
    
    # Check if ahead of ambulance (x-coordinate comparison)
    ahead = agent.position[0] > ambulance.position[0]
    if not ahead:
        return 0.0
    
    # Check if within threshold distance
    distance = agent.position[0] - ambulance.position[0]
    if distance < threshold:
        return 2.0
    
    return 0.0


def compute_heterogeneous_rewards(
    agents: list,
    ambulance_idx: int = 0,
    collision_occurred: bool = False,
    max_speed: float = 30.0
) -> list:
    """
    Compute role-specific rewards for heterogeneous agents.
    
    Implements asymmetric reward functions:
    - Ambulance (Agent 0): Prioritizes speed while avoiding collisions
    - Normal agents (Agents 1-3): Prioritize lane clearance and avoiding blocking
    
    Args:
        agents: List of controlled vehicle objects with position, speed attributes
        ambulance_idx: Index of ambulance agent (default: 0)
        collision_occurred: Whether any collision happened this step
        max_speed: Maximum velocity for normalization (default: 30.0 m/s)
        
    Returns:
        List of per-agent rewards [r0, r1, r2, r3]
    """
    rewards = []
    ambulance = agents[ambulance_idx]
    
    for i, agent in enumerate(agents):
        if i == ambulance_idx:
            # Ambulance reward: maximize speed, minimize collisions
            # Formula: 0.7 * speed_reward - 0.3 * collision_penalty
            speed_reward = agent.speed / max_speed
            collision_penalty = 5.0 if collision_occurred else 0.0
            reward = 0.7 * speed_reward - 0.3 * collision_penalty
        else:
            # Normal agent reward: maximize lane clearance, minimize blocking
            # Formula: 0.5 * lane_clearance - 0.5 * blocking_penalty
            lane_clearance_reward = compute_lane_clearance(agent, ambulance)
            blocking_penalty = compute_blocking_penalty(agent, ambulance)
            reward = 0.5 * lane_clearance_reward - 0.5 * blocking_penalty
        
        rewards.append(reward)
    
    return rewards
