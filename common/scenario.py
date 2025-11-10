"""Environment configuration loading utilities."""
import os
import yaml
import gymnasium as gym
from typing import Tuple, Optional, Dict, List


def apply_vehicle_colors(env: gym.Env, agent_roles: List[str], agent_colors: Dict[str, List[int]]) -> None:
    """
    Apply color customization to vehicles based on their roles.
    
    Args:
        env: Gymnasium environment with controlled_vehicles
        agent_roles: List of role strings for controlled vehicles
        agent_colors: Dictionary mapping role names to RGB colors
            Example: {'ambulance': [255, 0, 0], 'normal': [0, 255, 0], 'npc': [0, 0, 255]}
    """
    # Get controlled vehicles from environment
    controlled_vehicles = env.unwrapped.controlled_vehicles
    all_vehicles = env.unwrapped.road.vehicles
    
    # Apply colors to controlled vehicles based on their roles
    for i, (vehicle, role) in enumerate(zip(controlled_vehicles, agent_roles)):
        if role in agent_colors:
            # Convert list to tuple for pygame compatibility
            vehicle.color = tuple(agent_colors[role])
    
    # Apply NPC color to non-controlled vehicles
    if 'npc' in agent_colors:
        npc_color = tuple(agent_colors['npc'])
        for vehicle in all_vehicles:
            if vehicle not in controlled_vehicles:
                vehicle.color = npc_color


def validate_heterogeneous_config(config: dict) -> None:
    """
    Validate heterogeneous agent configuration.
    
    Args:
        config: Dictionary containing heterogeneous configuration
            - heterogeneous_agents: bool flag
            - agent_roles: list of role strings
            - agent_colors: dict mapping roles to RGB colors
            
    Raises:
        ValueError: If configuration is invalid
    """
    if not config.get('heterogeneous_agents'):
        return
    
    agent_roles = config.get('agent_roles', [])
    
    # Check agent_roles has exactly 4 elements
    if len(agent_roles) != 4:
        raise ValueError(
            f"Heterogeneous mode requires exactly 4 agent roles, got {len(agent_roles)}"
        )
    
    # Check first role is 'ambulance'
    if agent_roles[0] != 'ambulance':
        raise ValueError(
            f"Agent 0 must be 'ambulance' in heterogeneous mode, got '{agent_roles[0]}'"
        )
    
    # Check all roles are valid ('ambulance' or 'normal')
    valid_roles = {'ambulance', 'normal'}
    for i, role in enumerate(agent_roles):
        if role not in valid_roles:
            raise ValueError(
                f"Invalid role '{role}' at index {i}. Must be 'ambulance' or 'normal'"
            )


def make_hwy_env_from_yaml(yaml_path: str) -> Tuple[gym.Env, Optional[dict]]:
    """
    Load environment configuration from YAML and create Gymnasium environment.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Tuple of (configured Gymnasium environment, heterogeneous config dict or None)
        heterogeneous config contains:
            - heterogeneous_agents: bool flag
            - agent_roles: list of role strings
            - agent_colors: dict mapping roles to RGB colors
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If no valid environment ID found or invalid heterogeneous config
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse heterogeneous configuration
    heterogeneous_config = None
    if config.get('heterogeneous_agents', False):
        heterogeneous_config = {
            'heterogeneous_agents': True,
            'agent_roles': config.get('agent_roles', []),
            'agent_colors': config.get('agent_colors', {})
        }
        # Validate heterogeneous configuration
        validate_heterogeneous_config(heterogeneous_config)
    
    # Handle id_candidates list for robust fallback
    env_id = None
    if 'id_candidates' in config:
        # Try each candidate in order
        for candidate_id in config['id_candidates']:
            try:
                # Create environment with config
                env = gym.make(
                    candidate_id,
                    render_mode=config.get('render_mode', 'rgb_array'),
                    config=config.get('config', {})
                )
                env_id = candidate_id
                break
            except gym.error.Error:
                continue
        
        if env_id is None:
            raise ValueError(f"None of the id_candidates {config['id_candidates']} are available")
    else:
        # Single ID specified
        env_id = config['id']
        env = gym.make(
            env_id,
            render_mode=config.get('render_mode', 'rgb_array'),
            config=config.get('config', {})
        )
    
    return env, heterogeneous_config
