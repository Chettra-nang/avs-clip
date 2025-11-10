"""Environment factory for creating highway-env scenarios."""
import os
import gymnasium as gym
import highway_env  # Register highway-env environments
from typing import Tuple, Optional, Dict
from common.scenario import make_hwy_env_from_yaml


def make_env(scenario: str, heterogeneous: bool = False) -> Tuple[gym.Env, Optional[Dict]]:
    """
    Factory function to create highway-env environment from scenario name.
    
    Args:
        scenario: Scenario name ('highway', 'merge', 'intersection', or '*_heterogeneous' variants)
        heterogeneous: If True, load heterogeneous configuration (auto-detected from scenario name)
        
    Returns:
        Tuple of (configured Gymnasium environment, heterogeneous config dict or None)
        heterogeneous config contains:
            - heterogeneous_agents: bool flag
            - agent_roles: list of role strings
            - agent_colors: dict mapping roles to RGB colors
        
    Raises:
        ValueError: If scenario name is invalid
        FileNotFoundError: If config file doesn't exist
    """
    # Auto-detect heterogeneous mode from scenario name
    if '_heterogeneous' in scenario:
        heterogeneous = True
        base_scenario = scenario.replace('_heterogeneous', '').replace('_dense', '')
    else:
        base_scenario = scenario
    
    # Remove _dense suffix if present
    base_scenario = base_scenario.replace('_dense', '')
    
    valid_scenarios = ['highway', 'merge', 'intersection']
    
    if base_scenario not in valid_scenarios:
        raise ValueError(f"Invalid scenario '{scenario}'. Base scenario must be one of {valid_scenarios}")
    
    # Construct path to YAML config
    if heterogeneous:
        config_path = os.path.join('configs', 'envs', f'{base_scenario}_heterogeneous.yaml')
    else:
        config_path = os.path.join('configs', 'envs', f'{base_scenario}.yaml')
    
    # Load and create environment
    env, heterogeneous_config = make_hwy_env_from_yaml(config_path)
    
    return env, heterogeneous_config
