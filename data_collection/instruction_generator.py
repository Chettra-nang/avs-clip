"""
Human-aligned instruction generator for CLIP-RLDrive style training.

Based on Section 2.4 of the CLIP-RLDrive paper, this module generates
natural language instructions that describe driving actions in context.
"""

from typing import Optional


def generate_human_instruction(
    agent_role: str,
    action_name: str,
    scene_dict: dict,
    ambulance_dist: Optional[float] = None
) -> str:
    """
    Generate CLIP-RLDrive style human-aligned instructions.
    
    Creates natural language descriptions of driving actions based on:
    - Agent role (ambulance vs normal vehicle)
    - Action being taken (FASTER, SLOWER, LANE_LEFT, LANE_RIGHT, IDLE)
    - Scene context (distances, time-to-collision, traffic conditions)
    - Ambulance proximity (for cooperative yielding behavior)
    
    Args:
        agent_role: 'ambulance' or 'normal'
        action_name: Action being taken (e.g., 'FASTER', 'LANE_RIGHT')
        scene_dict: Scene context with keys like 'front_dist', 'front_ttc', 'risk'
        ambulance_dist: Distance to ambulance in meters (for normal agents)
        
    Returns:
        Human-readable instruction string describing the action in context
        
    Examples:
        >>> generate_human_instruction('ambulance', 'FASTER', {'front_dist': 150, 'front_ttc': 10})
        "Speed up since the road ahead is clear and the ambulance must reach its destination quickly"
        
        >>> generate_human_instruction('normal', 'LANE_RIGHT', {'front_dist': 30}, ambulance_dist=25.0)
        "Move to the right lane to yield and clear space for the emergency vehicle 25m away"
    """
    front_dist = scene_dict.get('front_dist', 999.0)
    front_ttc = scene_dict.get('front_ttc', 999.0)
    
    # === AMBULANCE INSTRUCTIONS ===
    if agent_role == 'ambulance':
        if action_name == 'FASTER':
            # Granular distance-based templates for better diversity
            if front_dist > 150:
                return "Speed up aggressively since the road is completely clear ahead for emergency response"
            elif front_dist > 100:
                return "Speed up since the road ahead is clear and the ambulance must reach its destination quickly"
            elif front_dist > 70:
                return "Accelerate as there is sufficient space ahead for safe high-speed emergency vehicle travel"
            elif front_dist > 50:
                return "Speed up cautiously while monitoring traffic at moderate distance ahead"
            elif front_dist > 30:
                return "Increase speed carefully as vehicle ahead is within visible range"
            else:
                return "Speed up with caution due to close proximity of traffic ahead"
        
        elif action_name == 'LANE_LEFT':
            if front_dist < 30:
                return f"Change to the left lane immediately to avoid vehicle at {front_dist:.0f}m and maintain emergency priority"
            elif front_dist < 60:
                return f"Move left to navigate around traffic at {front_dist:.0f}m and maintain ambulance priority"
            elif front_dist < 100:
                return f"Shift to the left lane to overtake slower vehicles at {front_dist:.0f}m ahead"
            else:
                return "Change to the left lane for optimal emergency vehicle positioning"
        
        elif action_name == 'LANE_RIGHT':
            if front_dist < 30:
                return f"Move to the right lane immediately to avoid vehicle at {front_dist:.0f}m"
            elif front_dist < 60:
                return f"Change right to overtake slower vehicles at {front_dist:.0f}m ahead"
            else:
                return "Shift to the right lane for better emergency vehicle passage"
        
        elif action_name == 'SLOWER':
            if front_ttc < 1.5:
                return f"Slow down immediately as there is critical collision risk with vehicle {front_ttc:.1f}s ahead"
            elif front_ttc < 2.5:
                return f"Reduce speed urgently due to high collision risk with vehicle {front_ttc:.1f}s ahead"
            elif front_ttc < 4.0:
                return f"Slow down as there is collision risk with vehicle {front_ttc:.1f}s ahead"
            else:
                return "Reduce speed temporarily to safely navigate through dense traffic"
        
        else:  # IDLE
            if front_dist > 100:
                return "Maintain current high speed while monitoring traffic conditions for emergency vehicle passage"
            elif front_dist > 50:
                return "Maintain current speed while monitoring traffic conditions for emergency vehicle passage"
            else:
                return "Maintain current speed while carefully monitoring nearby traffic"
    
    # === NORMAL AGENT INSTRUCTIONS ===
    else:
        # Priority 1: Ambulance nearby (cooperative behavior)
        if ambulance_dist is not None and ambulance_dist < 50:
            if action_name == 'LANE_RIGHT':
                if ambulance_dist < 20:
                    return f"Move to the right lane immediately to yield to the emergency vehicle {ambulance_dist:.0f}m away"
                elif ambulance_dist < 35:
                    return f"Change to the right lane to clear space for the approaching ambulance at {ambulance_dist:.0f}m"
                else:
                    return f"Shift right to prepare for the emergency vehicle {ambulance_dist:.0f}m behind"
            
            elif action_name == 'LANE_LEFT':
                if ambulance_dist < 20:
                    return f"Change to the left lane immediately to allow the ambulance at {ambulance_dist:.0f}m to pass"
                elif ambulance_dist < 35:
                    return f"Move left to clear the path for the emergency vehicle at {ambulance_dist:.0f}m"
                else:
                    return f"Shift to the left lane to allow the ambulance at {ambulance_dist:.0f}m to pass safely"
            
            elif action_name == 'SLOWER':
                if ambulance_dist < 20:
                    return f"Slow down urgently to create a safe gap for the emergency vehicle {ambulance_dist:.0f}m away"
                elif ambulance_dist < 35:
                    return f"Reduce speed to allow the ambulance at {ambulance_dist:.0f}m to pass safely"
                else:
                    return f"Slow down to create space for the approaching emergency vehicle at {ambulance_dist:.0f}m"
            
            elif action_name == 'IDLE':
                if ambulance_dist < 20:
                    return f"Maintain current position while the ambulance at {ambulance_dist:.0f}m passes urgently"
                elif ambulance_dist < 35:
                    return f"Maintain current speed and position while the ambulance at {ambulance_dist:.0f}m passes"
                else:
                    return f"Continue at current speed while monitoring the emergency vehicle at {ambulance_dist:.0f}m"
            
            else:  # FASTER
                if ambulance_dist < 20:
                    return f"Speed up quickly to clear the path before the ambulance at {ambulance_dist:.0f}m arrives"
                elif ambulance_dist < 35:
                    return f"Accelerate to clear the path before the ambulance at {ambulance_dist:.0f}m arrives"
                else:
                    return f"Speed up to move ahead of the emergency vehicle at {ambulance_dist:.0f}m"
        
        # Priority 2: Normal driving without ambulance
        else:
            if action_name == 'FASTER':
                if front_dist > 150:
                    return "Speed up since the chance of collision is extremely low with clear road ahead"
                elif front_dist > 100:
                    return "Speed up since the chance of collision with other vehicles is low and the road is clear"
                elif front_dist > 70:
                    return "Accelerate as there is sufficient space ahead for safe driving"
                elif front_dist > 50:
                    return "Speed up cautiously while maintaining safe following distance"
                else:
                    return "Increase speed carefully as traffic is within moderate distance"
            
            elif action_name == 'SLOWER':
                if front_ttc < 2.0:
                    return f"Slow down immediately as there is high collision risk with vehicle {front_ttc:.1f}s ahead"
                elif front_ttc < 3.5:
                    return f"Reduce speed as there is a chance of collision with vehicle {front_ttc:.1f}s ahead"
                elif front_ttc < 5.0:
                    return f"Slow down to maintain safe distance from vehicle {front_ttc:.1f}s ahead"
                else:
                    return "Reduce speed to maintain safe speed in current traffic conditions"
            
            elif action_name == 'LANE_LEFT':
                if front_dist < 30:
                    return f"Change to the left lane immediately to avoid the vehicle {front_dist:.0f}m ahead"
                elif front_dist < 60:
                    return f"Move left to navigate around the vehicle at {front_dist:.0f}m ahead"
                else:
                    return f"Shift to the left lane for better positioning with vehicle at {front_dist:.0f}m"
            
            elif action_name == 'LANE_RIGHT':
                if front_dist < 30:
                    return f"Move to the right lane immediately to avoid vehicle at {front_dist:.0f}m"
                elif front_dist < 60:
                    return f"Change right to navigate around the vehicle at {front_dist:.0f}m"
                else:
                    return f"Shift to the right lane for optimal positioning with traffic at {front_dist:.0f}m"
            
            else:  # IDLE
                if front_dist > 100:
                    return "Proceed at the same speed as traffic conditions are stable and road is clear"
                elif front_dist > 50:
                    return "Proceed at the same speed as traffic conditions are stable and safe"
                else:
                    return "Maintain current speed while monitoring nearby traffic"


def get_instruction_statistics(instructions: list) -> dict:
    """
    Analyze instruction diversity for dataset quality assessment.
    
    Args:
        instructions: List of instruction strings
        
    Returns:
        Dictionary with statistics:
        - unique_count: Number of unique instructions
        - total_count: Total number of instructions
        - diversity_ratio: unique/total ratio
        - top_10: Most common instructions with counts
    """
    from collections import Counter
    
    counter = Counter(instructions)
    unique_count = len(counter)
    total_count = len(instructions)
    diversity_ratio = unique_count / total_count if total_count > 0 else 0.0
    
    return {
        'unique_count': unique_count,
        'total_count': total_count,
        'diversity_ratio': diversity_ratio,
        'top_10': counter.most_common(10)
    }
