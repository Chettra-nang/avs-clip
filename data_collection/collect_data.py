"""Data collection pipeline for VLM-MARL highway driving."""

import os
import sys
import json
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.make_env import make_env
from common.utils import ACTIONS_ALL, ttc, risk_score, add_role_encoding
from common.logger_tb import TBLogger
from common.scenario import apply_vehicle_colors
from data_collection.instruction_generator import generate_human_instruction


def _vehicle_state(vehicle) -> Dict:
    """
    Extract vehicle state information.
    
    Args:
        vehicle: Highway-env vehicle object
        
    Returns:
        Dictionary with position, velocity, speed, heading
    """
    position = np.array([vehicle.position[0], vehicle.position[1]])
    velocity = np.array([vehicle.velocity[0], vehicle.velocity[1]])
    speed = float(np.linalg.norm(velocity))
    heading = float(vehicle.heading)
    
    return {
        'position': position,
        'velocity': velocity,
        'speed': speed,
        'heading': heading
    }


def _scan_scene(ego_vehicle, all_vehicles, radius: float = 60.0) -> Dict:
    """
    Scan scene to find nearby vehicles and compute risk metrics.
    
    Args:
        ego_vehicle: The controlled vehicle
        all_vehicles: List of all vehicles in the environment
        radius: Search radius in meters
        
    Returns:
        Dictionary with scene analysis results
    """
    ego_state = _vehicle_state(ego_vehicle)
    ego_pos = ego_state['position']
    ego_vel = ego_state['velocity']
    
    # Find nearby vehicles
    nearby_vehicles = []
    for other in all_vehicles:
        if other is ego_vehicle:
            continue
        
        other_pos = np.array([other.position[0], other.position[1]])
        dist = np.linalg.norm(other_pos - ego_pos)
        
        if dist <= radius:
            other_state = _vehicle_state(other)
            nearby_vehicles.append({
                'vehicle': other,
                'state': other_state,
                'distance': dist
            })
    
    # Compute min TTC, min distance, max risk
    min_ttc = 999.0
    min_distance = 999.0
    max_risk = 0.0
    front_vehicle = None
    front_dist = 999.0
    front_ttc = 999.0
    
    for nearby in nearby_vehicles:
        other_state = nearby['state']
        
        # Calculate TTC
        ttc_value = ttc(ego_pos, ego_vel, 
                       other_state['position'], other_state['velocity'])
        
        # Calculate risk
        risk = risk_score(ego_state, other_state)
        
        # Update minimums/maximums
        min_ttc = min(min_ttc, ttc_value)
        min_distance = min(min_distance, nearby['distance'])
        max_risk = max(max_risk, risk)
        
        # Check if vehicle is in front (positive x-direction relative to ego)
        rel_pos = other_state['position'] - ego_pos
        # Transform to ego's reference frame
        cos_h = np.cos(ego_state['heading'])
        sin_h = np.sin(ego_state['heading'])
        rel_x = rel_pos[0] * cos_h + rel_pos[1] * sin_h
        rel_y = -rel_pos[0] * sin_h + rel_pos[1] * cos_h
        
        # Vehicle is in front if rel_x > 0 and within same lane (|rel_y| < 4m)
        if rel_x > 0 and abs(rel_y) < 4.0:
            if nearby['distance'] < front_dist:
                front_vehicle = nearby['vehicle']
                front_dist = nearby['distance']
                front_ttc = ttc_value
    
    # Determine lane availability using road network
    can_left = False
    can_right = False
    
    try:
        road = ego_vehicle.road
        lane_index = ego_vehicle.lane_index
        
        # Check if left lane exists
        try:
            left_lane = road.network.get_lane((*lane_index[:-1], lane_index[-1] + 1))
            if left_lane is not None:
                can_left = True
        except (KeyError, AttributeError, IndexError):
            pass
        
        # Check if right lane exists
        try:
            right_lane = road.network.get_lane((*lane_index[:-1], lane_index[-1] - 1))
            if right_lane is not None:
                can_right = True
        except (KeyError, AttributeError, IndexError):
            pass
    except (AttributeError, TypeError):
        # Fallback: assume lanes available if not at boundaries
        can_left = True
        can_right = True
    
    return {
        'min_ttc': min_ttc,
        'min_distance': min_distance,
        'max_risk': max_risk,
        'front_dist': front_dist,
        'front_ttc': front_ttc,
        'can_left': can_left,
        'can_right': can_right,
        'nearby_count': len(nearby_vehicles)
    }



def _ambulance_expert_action(ego_vehicle, nearby_vehicles: list, 
                            can_left: bool, can_right: bool, 
                            rng: np.random.Generator) -> int:
    """
    Aggressive expert heuristic for ambulance (Agent 0).
    
    Priority: Speed > Lane changes > Safety
    
    Logic:
    1. If front clear (> 30m): FASTER
    2. If obstacle ahead (< 30m): Try lane change (prefer LANE_LEFT for passing)
    3. Default: FASTER (maintain speed)
    
    Args:
        ego_vehicle: Ambulance vehicle object
        nearby_vehicles: List of nearby vehicle objects
        can_left: Whether left lane change is available
        can_right: Whether right lane change is available
        rng: Random number generator
        
    Returns:
        Action ID (0-4)
    """
    # Find closest vehicle ahead
    front_vehicle = None
    front_dist = 999.0
    
    ego_pos = np.array([ego_vehicle.position[0], ego_vehicle.position[1]])
    
    for vehicle in nearby_vehicles:
        if vehicle is ego_vehicle:
            continue
        
        # Check if vehicle is ahead (positive x-direction)
        if vehicle.position[0] > ego_vehicle.position[0]:
            dist = vehicle.position[0] - ego_vehicle.position[0]
            
            # Check if in same lane (y-coordinate difference < 4m)
            if abs(vehicle.position[1] - ego_vehicle.position[1]) < 4.0:
                if dist < front_dist:
                    front_dist = dist
                    front_vehicle = vehicle
    
    # Aggressive logic: prioritize speed when front is clear
    if front_dist > 30.0:
        return 3  # FASTER
    
    # Obstacle ahead: try to change lanes
    if front_dist < 30.0 and front_vehicle is not None:
        # Prefer left lane for passing (60% probability)
        if can_left and rng.random() < 0.6:
            return 0  # LANE_LEFT
        elif can_right:
            return 2  # LANE_RIGHT
        elif can_left:
            return 0  # LANE_LEFT (fallback if right not available)
        else:
            return 1  # IDLE (wait for gap)
    
    # Default: maintain speed
    return 3  # FASTER


def _normal_expert_action(ego_vehicle, nearby_vehicles: list, 
                         ambulance, can_left: bool, can_right: bool, 
                         rng: np.random.Generator) -> int:
    """
    Yielding expert heuristic for normal agents (Agents 1-3).
    
    Priority: Detect ambulance > Clear lane > Safety > Normal driving
    
    Logic:
    1. If ambulance behind in same lane (< 50m): Yield (LANE_RIGHT preferred, or SLOWER)
    2. Otherwise: Follow existing TTC-based safety logic
    
    Args:
        ego_vehicle: Normal agent vehicle object
        nearby_vehicles: List of nearby vehicle objects
        ambulance: Ambulance vehicle object (Agent 0)
        can_left: Whether left lane change is available
        can_right: Whether right lane change is available
        rng: Random number generator
        
    Returns:
        Action ID (0-4)
    """
    # Check if ambulance is behind in same lane
    ambulance_behind = False
    ambulance_distance = 999.0
    
    if ambulance is not None:
        # Check if in same lane (y-coordinate difference < 2.0 meters)
        same_lane = abs(ego_vehicle.position[1] - ambulance.position[1]) < 2.0
        
        # Check if ambulance is behind (lower x-coordinate)
        behind = ambulance.position[0] < ego_vehicle.position[0]
        
        # Calculate distance
        distance = ego_vehicle.position[0] - ambulance.position[0]
        
        # Ambulance is behind in same lane and within 50m
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
            return 4  # SLOWER (create gap if cannot change lanes)
    
    # Priority 2: Normal safety-based driving (existing TTC logic)
    # Find closest vehicle ahead
    front_vehicle = None
    front_dist = 999.0
    front_ttc = 999.0
    
    ego_pos = np.array([ego_vehicle.position[0], ego_vehicle.position[1]])
    ego_vel = np.array([ego_vehicle.velocity[0], ego_vehicle.velocity[1]])
    
    for vehicle in nearby_vehicles:
        if vehicle is ego_vehicle:
            continue
        
        # Check if vehicle is ahead
        if vehicle.position[0] > ego_vehicle.position[0]:
            dist = vehicle.position[0] - ego_vehicle.position[0]
            
            # Check if in same lane
            if abs(vehicle.position[1] - ego_vehicle.position[1]) < 4.0:
                if dist < front_dist:
                    front_dist = dist
                    front_vehicle = vehicle
                    
                    # Calculate TTC
                    other_pos = np.array([vehicle.position[0], vehicle.position[1]])
                    other_vel = np.array([vehicle.velocity[0], vehicle.velocity[1]])
                    front_ttc = ttc(ego_pos, ego_vel, other_pos, other_vel)
    
    # Emergency logic: TTC < 2.0s OR front_dist < 10m
    if front_ttc < 2.0 or front_dist < 10.0:
        # Try to change lanes
        if can_left and rng.random() < 0.5:
            return 0  # LANE_LEFT
        elif can_right:
            return 2  # LANE_RIGHT
        else:
            return 4  # SLOWER
    
    # Caution logic: front_dist < 20m AND front_ttc < 4.0s
    if front_dist < 20.0 and front_ttc < 4.0:
        return 4  # SLOWER
    
    # Random exploration: 12% probability for lane changes
    if rng.random() < 0.12:
        if can_left and can_right:
            return 0 if rng.random() < 0.5 else 2  # LANE_LEFT or LANE_RIGHT
        elif can_left:
            return 0  # LANE_LEFT
        elif can_right:
            return 2  # LANE_RIGHT
    
    # Default: speed optimization (FASTER 50% or IDLE 50%)
    if rng.random() < 0.5:
        return 3  # FASTER
    else:
        return 1  # IDLE


def _expert_action(scene: Dict, rng: np.random.Generator) -> int:
    """
    Compute expert heuristic action based on scene analysis.
    
    Logic:
    1. Emergency: TTC < 2.0s OR front_dist < 10m → lane change or slow down
    2. Caution: front_dist < 20m AND front_ttc < 4.0s → slow down
    3. Exploration: 12% random lane change
    4. Speed optimization: FASTER (50%) or IDLE (50%)
    
    Args:
        scene: Scene analysis dictionary from _scan_scene
        rng: Random number generator
        
    Returns:
        Action ID (0-4)
    """
    front_ttc = scene['front_ttc']
    front_dist = scene['front_dist']
    can_left = scene['can_left']
    can_right = scene['can_right']
    
    # Emergency logic: TTC < 2.0s OR front_dist < 10m
    if front_ttc < 2.0 or front_dist < 10.0:
        # Try to change lanes
        if can_left and rng.random() < 0.5:
            return 0  # LANE_LEFT
        elif can_right:
            return 2  # LANE_RIGHT
        else:
            return 4  # SLOWER
    
    # Caution logic: front_dist < 20m AND front_ttc < 4.0s
    if front_dist < 20.0 and front_ttc < 4.0:
        return 4  # SLOWER
    
    # Random exploration: 12% probability for lane changes
    if rng.random() < 0.12:
        if can_left and can_right:
            return 0 if rng.random() < 0.5 else 2  # LANE_LEFT or LANE_RIGHT
        elif can_left:
            return 0  # LANE_LEFT
        elif can_right:
            return 2  # LANE_RIGHT
        else:
            return 1  # IDLE
    
    # Speed optimization: FASTER (50%) or IDLE (50%)
    if rng.random() < 0.5:
        return 3  # FASTER
    else:
        return 1  # IDLE



def format_clip_text(agent_role: str, action_name: str, scene: dict, 
                     ambulance_distance: Optional[float] = None) -> str:
    """
    Format text label with role prefix for CLIP training.
    
    This function creates role-aware text labels that help CLIP learn semantic
    associations between visual scenes and agent behaviors. Ambulance agents
    get "AMBULANCE" prefix to indicate priority behavior, while normal agents
    get "YIELD" prefix to indicate cooperative yielding behavior.
    
    Args:
        agent_role: 'ambulance' or 'normal'
        action_name: Action taken (e.g., 'FASTER', 'LANE_RIGHT')
        scene: Scene context dictionary with keys like 'front_dist', 'front_ttc', etc.
        ambulance_distance: Distance to ambulance (for normal agents when ambulance nearby)
        
    Returns:
        Formatted text string for CLIP training
        
    Examples:
        Ambulance: "AMBULANCE FASTER: {'front_dist': 35.2, 'front_ttc': 8.1, 'risk': 0.25}"
        Normal (no ambulance): "YIELD IDLE: {'front_dist': 15.2, 'front_ttc': 3.1, 'risk': 0.68}"
        Normal (ambulance nearby): "YIELD LANE_RIGHT: {'front_dist': 20.0, 'ambulance_dist': 25.5, 'risk': 0.45}"
    """
    # Create a copy of scene dict to avoid modifying original
    scene_copy = scene.copy()
    
    # Add ambulance_distance for normal agents when applicable
    if agent_role == 'normal' and ambulance_distance is not None and ambulance_distance < 50.0:
        scene_copy['ambulance_dist'] = ambulance_distance
    
    # Format with role prefix
    if agent_role == 'ambulance':
        return f"AMBULANCE {action_name}: {scene_copy}"
    else:  # normal agent
        return f"YIELD {action_name}: {scene_copy}"


def _compute_quality_score(scenes: List[Dict]) -> float:
    """
    Compute quality score as weighted average across agents.
    
    Quality is based on:
    - Lower TTC = higher quality (more interesting)
    - Closer distance = higher quality
    - Higher risk = higher quality
    
    Args:
        scenes: List of scene dictionaries for each agent
        
    Returns:
        Quality score in [0, 1] range
    """
    if not scenes:
        return 0.0
    
    quality_scores = []
    for scene in scenes:
        # Normalize TTC (lower is better, cap at 10s)
        ttc_quality = 1.0 - min(scene['min_ttc'] / 10.0, 1.0)
        
        # Normalize distance (closer is better, cap at 50m)
        dist_quality = 1.0 - min(scene['min_distance'] / 50.0, 1.0)
        
        # Risk is already in [0, 1]
        risk_quality = scene['max_risk']
        
        # Weighted combination (same as risk_score weights)
        quality = 0.4 * ttc_quality + 0.3 * dist_quality + 0.3 * risk_quality
        quality_scores.append(quality)
    
    # Average across all agents
    return float(np.mean(quality_scores))


def _center_crop_224(image: np.ndarray) -> Image.Image:
    """
    Resize full image to 224x224 pixels to preserve all agents.
    
    Args:
        image: Input image as numpy array [H, W, 3]
        
    Returns:
        PIL Image resized to 224x224
    """
    # Convert to PIL and resize full frame to 224x224
    # This preserves all agents in the scene (though aspect ratio changes)
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize((224, 224), Image.BILINEAR)
    
    return pil_img


def _save_frame_data(
    frame_rgb: np.ndarray,
    agents_data: List[Dict],
    quality: float,
    episode: int,
    step: int,
    frame_index: int,
    output_dir: str,
    rng: np.random.Generator
) -> Optional[Dict]:
    """
    Save frame data if quality threshold met or random save.
    
    Save logic: quality > 0.5 OR random < 0.25
    
    Args:
        frame_rgb: RGB frame from environment
        agents_data: List of agent data dictionaries
        quality: Quality score
        episode: Episode number
        step: Step within episode
        frame_index: Global frame index
        output_dir: Output directory path
        rng: Random number generator
        
    Returns:
        JSON record dictionary if saved, None otherwise
    """
    # Check save condition
    should_save = quality > 0.5 or rng.random() < 0.25
    
    if not should_save:
        return None
    
    # Create directories
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Center crop to 224x224
    cropped_img = _center_crop_224(frame_rgb)
    
    # Save image as PNG
    img_filename = f"{frame_index:06d}.png"
    img_path = os.path.join(images_dir, img_filename)
    cropped_img.save(img_path)
    
    # Create JSON record
    record = {
        'index': frame_index,
        'image': f"images/{img_filename}",
        'episode': episode,
        'step': step,
        'quality': quality,
        'agents': agents_data,
        'timestamp': datetime.now().isoformat()
    }
    
    return record



@dataclass
class CollectCfg:
    """Configuration for data collection."""
    scenario: str
    episodes: int
    max_steps: int = 1000
    output_dir: str = 'data'
    log_dir: str = 'runs/data_collection'
    seed: int = 42
    heterogeneous: bool = False


def collect_data(cfg: CollectCfg):
    """
    Main data collection loop.
    
    Args:
        cfg: Collection configuration
    """
    # Initialize random number generator
    rng = np.random.default_rng(cfg.seed)
    
    # Create output directory
    scenario_dir = os.path.join(cfg.output_dir, cfg.scenario)
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Initialize environment
    print(f"Initializing {cfg.scenario} environment...")
    env, heterogeneous_config = make_env(cfg.scenario, heterogeneous=cfg.heterogeneous)
    
    # Initialize TensorBoard logger
    logger = TBLogger(cfg.log_dir, cfg.scenario)
    
    # Statistics tracking
    frame_index = 0
    saved_frames = 0
    action_counts = {i: 0 for i in range(5)}
    quality_scores = []
    
    # Open dataset.jsonl for writing
    jsonl_path = os.path.join(scenario_dir, 'dataset.jsonl')
    
    print(f"Collecting data for {cfg.episodes} episodes...")
    
    with open(jsonl_path, 'w') as jsonl_file:
        for episode in tqdm(range(cfg.episodes), desc="Episodes"):
            obs, info = env.reset(seed=cfg.seed + episode)
            
            # Apply vehicle colors if in heterogeneous mode
            if heterogeneous_config is not None:
                apply_vehicle_colors(
                    env, 
                    heterogeneous_config['agent_roles'],
                    heterogeneous_config['agent_colors']
                )
            
            episode_return = 0.0
            episode_steps = 0
            
            for step in range(cfg.max_steps):
                # Render RGB frame
                frame_rgb = env.render()
                
                # Get controlled vehicles
                try:
                    controlled_vehicles = env.unwrapped.controlled_vehicles
                    if not isinstance(controlled_vehicles, list):
                        controlled_vehicles = [controlled_vehicles]
                except AttributeError:
                    print("Warning: Could not access controlled_vehicles")
                    break
                
                # Get all vehicles for scene analysis
                try:
                    all_vehicles = env.unwrapped.road.vehicles
                except AttributeError:
                    print("Warning: Could not access road vehicles")
                    break
                
                # Process each controlled vehicle
                agents_data = []
                scenes = []
                actions = []
                
                # Get ambulance reference (Agent 0) for normal agents
                ambulance = controlled_vehicles[0] if len(controlled_vehicles) > 0 else None
                
                for agent_idx, ego_vehicle in enumerate(controlled_vehicles):
                    # Add role encoding to observation
                    # Note: obs is per-agent observation from environment
                    # For heterogeneous mode, we extend it from (56,) to (64,)
                    if isinstance(obs, (list, tuple, np.ndarray)):
                        if len(obs) > agent_idx:
                            agent_obs = obs[agent_idx] if isinstance(obs, (list, tuple)) else obs
                            # Apply role encoding
                            agent_obs_with_role = add_role_encoding(
                                agent_obs, 
                                controlled_vehicles, 
                                all_vehicles[:8]  # Only first 8 vehicles are observed
                            )
                    
                    # Determine agent role
                    agent_role = 'ambulance' if agent_idx == 0 else 'normal'
                    
                    # Extract vehicle state
                    ego_state = _vehicle_state(ego_vehicle)
                    
                    # Scan scene
                    scene = _scan_scene(ego_vehicle, all_vehicles)
                    scenes.append(scene)
                    
                    # Calculate ambulance distance for normal agents
                    ambulance_distance = None
                    if agent_role == 'normal' and ambulance is not None:
                        # Detect ambulance from any direction within 50m radius
                        amb_pos = np.array([ambulance.position[0], ambulance.position[1]])
                        ego_pos = np.array([ego_vehicle.position[0], ego_vehicle.position[1]])
                        distance = np.linalg.norm(amb_pos - ego_pos)
                        
                        if distance < 50.0:  # Detection radius matching expert heuristic
                            ambulance_distance = distance
                    
                    # Compute expert action based on agent role
                    # Agent 0 = ambulance, Agents 1-3 = normal
                    if agent_idx == 0:
                        # Ambulance expert heuristic
                        action = _ambulance_expert_action(
                            ego_vehicle, 
                            all_vehicles,
                            scene['can_left'],
                            scene['can_right'],
                            rng
                        )
                    else:
                        # Normal agent expert heuristic
                        action = _normal_expert_action(
                            ego_vehicle,
                            all_vehicles,
                            ambulance,
                            scene['can_left'],
                            scene['can_right'],
                            rng
                        )
                    
                    actions.append(action)
                    action_counts[action] += 1
                    
                    # Create scene dict for storage
                    scene_dict = {
                        'front_dist': scene['front_dist'],
                        'front_ttc': scene['front_ttc'],
                        'risk': scene['max_risk'],
                        'can_left': scene['can_left'],
                        'can_right': scene['can_right']
                    }
                    
                    # Add ambulance_dist for normal agents when applicable
                    if ambulance_distance is not None:
                        scene_dict['ambulance_dist'] = ambulance_distance
                    
                    # Generate human-aligned instruction for CLIP training
                    instruction = generate_human_instruction(
                        agent_role=agent_role,
                        action_name=ACTIONS_ALL[action],
                        scene_dict=scene_dict,
                        ambulance_dist=ambulance_distance
                    )
                    
                    # Store agent data with role information
                    agent_data = {
                        'role': agent_role,
                        'speed': ego_state['speed'],
                        'position': ego_state['position'].tolist(),
                        'heading': ego_state['heading'],
                        'scene': scene_dict,
                        'action_id': action,
                        'action_name': ACTIONS_ALL[action],
                        'instruction': instruction
                    }
                    agents_data.append(agent_data)
                
                # Compute quality score
                quality = _compute_quality_score(scenes)
                quality_scores.append(quality)
                
                # Save frame data if quality threshold met
                record = _save_frame_data(
                    frame_rgb, agents_data, quality,
                    episode, step, frame_index,
                    scenario_dir, rng
                )
                
                if record is not None:
                    # Write to JSONL
                    jsonl_file.write(json.dumps(record) + '\n')
                    jsonl_file.flush()
                    saved_frames += 1
                
                # Log to TensorBoard every 10 steps
                if frame_index % 10 == 0:
                    logger.scalar('quality/mean', quality, frame_index)
                    logger.scalar('traffic/nearby_vehicles', 
                                 np.mean([s['nearby_count'] for s in scenes]), 
                                 frame_index)
                    logger.scalar('traffic/min_ttc', 
                                 np.mean([s['min_ttc'] for s in scenes]), 
                                 frame_index)
                    
                    # Log action distribution
                    total_actions = sum(action_counts.values())
                    if total_actions > 0:
                        for action_id, count in action_counts.items():
                            logger.scalar(f'actions/{ACTIONS_ALL[action_id]}', 
                                        count / total_actions, 
                                        frame_index)
                
                # Log sample frames every 100 steps (for visual inspection)
                if frame_index % 100 == 0:
                    logger.image('samples/frame', frame_rgb, frame_index, dataformats='HWC')
                
                # Step environment with expert actions
                actions_tuple = tuple(actions)
                obs, reward, done, truncated, info = env.step(actions_tuple)
                
                # Track episode return
                if isinstance(reward, (list, tuple, np.ndarray)):
                    episode_return += np.mean(reward)
                else:
                    episode_return += reward
                
                episode_steps += 1
                frame_index += 1
                
                if done or truncated:
                    break
            
            # Log episode metrics
            logger.scalar('episode/return', episode_return, episode)
            logger.scalar('episode/steps', episode_steps, episode)
            logger.scalar('episode/saved_frames', saved_frames, episode)
    
    # Save metadata
    metadata = {
        'scenario': cfg.scenario,
        'episodes': cfg.episodes,
        'total_frames': frame_index,
        'saved_frames': saved_frames,
        'save_rate': saved_frames / frame_index if frame_index > 0 else 0.0,
        'mean_quality': float(np.mean(quality_scores)) if quality_scores else 0.0,
        'action_distribution': {
            ACTIONS_ALL[i]: count / sum(action_counts.values()) 
            for i, count in action_counts.items()
        },
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(scenario_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCollection complete!")
    print(f"Total frames: {frame_index}")
    print(f"Saved frames: {saved_frames} ({saved_frames/frame_index*100:.1f}%)")
    print(f"Mean quality: {metadata['mean_quality']:.3f}")
    print(f"Output directory: {scenario_dir}")
    
    # Close logger and environment
    logger.close()
    env.close()


def main():
    """Command-line interface for data collection."""
    parser = argparse.ArgumentParser(description='Collect driving data with expert labels')
    parser.add_argument('--scenario', type=str, required=True,
                       choices=['highway', 'merge', 'intersection',
                               'highway_heterogeneous', 'merge_heterogeneous', 'intersection_heterogeneous',
                               'highway_heterogeneous_dense'],
                       help='Scenario name (use *_heterogeneous for emergency vehicle priority mode, *_dense for tighter spacing)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to collect')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for dataset')
    parser.add_argument('--log-dir', type=str, default='runs/data_collection',
                       help='TensorBoard log directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--heterogeneous', action='store_true',
                       help='Enable heterogeneous mode (ambulance priority) - auto-detected from scenario name')
    
    args = parser.parse_args()
    
    # Auto-detect heterogeneous mode from scenario name
    heterogeneous = args.heterogeneous or '_heterogeneous' in args.scenario
    
    cfg = CollectCfg(
        scenario=args.scenario,
        episodes=args.episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        heterogeneous=heterogeneous
    )
    
    collect_data(cfg)


if __name__ == '__main__':
    main()
