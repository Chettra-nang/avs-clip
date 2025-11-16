"""Custom multi-agent merge environment for ambulance priority scenarios."""
from highway_env.envs.merge_env import MergeEnv
from highway_env import utils
import numpy as np


class MultiAgentMergeEnv(MergeEnv):
    """
    Extended merge environment supporting multiple controlled vehicles.
    
    Key improvements over original MergeEnv:
    1. Respects 'controlled_vehicles' config parameter
    2. Creates multiple ego vehicles with staggered positions
    3. Reduces traffic density to prevent immediate collisions
    4. Updates termination logic for multi-agent scenarios
    """
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "controlled_vehicles": 4,
            "vehicles_count": 8,  # Reduced from default
            "initial_spacing": 2.5,
            "duration": 40,
            "collision_reward": -1,
            "normalize_reward": True,
            # Override observation and action to be multi-agent
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction"
                }
            }
        })
        return config
    
    def define_spaces(self) -> None:
        """Override to initialize controlled_vehicles before action/observation spaces."""
        # Pre-initialize controlled_vehicles list for MultiAgentAction
        # This is called twice: once during __init__ and once after _reset()
        # Only create placeholders if no real vehicles exist yet
        if not self.controlled_vehicles or (self.controlled_vehicles and not hasattr(self.controlled_vehicles[0], 'road')):
            # Create placeholder vehicles for space definition
            from highway_env.vehicle.kinematics import Vehicle
            self.controlled_vehicles = [
                Vehicle(None, [0, 0], 0, 0) 
                for _ in range(self.config["controlled_vehicles"])
            ]
        
        # Now define spaces with correct number of agents
        super().define_spaces()
    
    def _make_vehicles(self) -> None:
        """
        Override to create multiple controlled vehicles.
        
        Strategy:
        - Place controlled vehicles in staggered formation
        - Reduce number of AI vehicles to prevent congestion
        - Ensure sufficient spacing between all vehicles
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        
        # Clear any existing vehicles
        road.vehicles = []
        
        # Create controlled vehicles
        controlled_count = self.config["controlled_vehicles"]
        
        created_vehicles = []
        for i in range(controlled_count):
            # Stagger positions: 20m, 40m, 60m, 80m
            position = 20.0 + (i * 20.0)
            
            # Alternate lanes to avoid immediate rear-end collisions
            lane_id = i % 2  # Use lanes 0, 1
            
            # Vary speeds: 28-31 m/s
            speed = 28.0 + (i * 1.0)
            
            # Create controlled vehicle
            ego_vehicle = self.action_type.vehicle_class(
                road,
                road.network.get_lane(("a", "b", lane_id)).position(position, 0),
                speed=speed
            )
            road.vehicles.append(ego_vehicle)
            created_vehicles.append(ego_vehicle)
        
        # Store references to controlled vehicles
        self.controlled_vehicles = created_vehicles
        # DON'T set self.vehicle - it has a setter that overwrites controlled_vehicles!
        
        # Create fewer AI vehicles (only 2 vs original 3)
        # Position them further ahead to reduce congestion
        ai_positions = [(120, 29.0), (160, 30.0)]
        
        for position, speed in ai_positions:
            lane_id = self.np_random.integers(2)
            lane = road.network.get_lane(("a", "b", lane_id))
            vehicle = other_vehicles_type(
                road,
                lane.position(position + self.np_random.uniform(-5.0, 5.0), 0),
                speed=speed + self.np_random.uniform(-1.0, 1.0)
            )
            road.vehicles.append(vehicle)
        
        # Create merging vehicle
        merging_v = other_vehicles_type(
            road,
            road.network.get_lane(("j", "k", 0)).position(110.0, 0.0),
            speed=20.0
        )
        merging_v.target_speed = 30.0
        road.vehicles.append(merging_v)
    
    def _is_terminated(self) -> bool:
        """
        Multi-agent termination logic.
        
        Episode ends when:
        1. ANY controlled vehicle reaches the goal (x > 370m), OR
        2. ALL controlled vehicles crash
        """
        # Success: Any controlled vehicle reaches goal
        for vehicle in self.controlled_vehicles:
            if vehicle.position[0] > 370:
                return True
        
        # Failure: All controlled vehicles crashed
        if all(v.crashed for v in self.controlled_vehicles):
            return True
        
        return False
    
    def _is_truncated(self) -> bool:
        """Episode truncation (time limit)."""
        return self.time >= self.config["duration"]
    
    def _info(self, obs, action) -> dict:
        """Add per-agent info for debugging."""
        info = super()._info(obs, action)
        
        # Add per-agent crash status
        info.update({
            "agents_crashed": [v.crashed for v in self.controlled_vehicles],
            "agents_positions": [v.position[0] for v in self.controlled_vehicles],
            "num_active_agents": sum(1 for v in self.controlled_vehicles if not v.crashed),
        })
        
        return info
    
    def _reward(self, action) -> float:
        """
        Multi-agent reward.
        
        Strategy:
        - Team reward: All agents benefit from any agent's progress
        - Individual penalties: Each agent penalized for crashing
        """
        reward = 0.0
        
        # Team reward: Progress of leading agent
        max_position = max(v.position[0] for v in self.controlled_vehicles)
        reward += (max_position - 30.0) / 340.0  # Normalize to ~[0, 1]
        
        # Individual penalties
        for vehicle in self.controlled_vehicles:
            if vehicle.crashed:
                reward += self.config["collision_reward"]
        
        # Success bonus
        if any(v.position[0] > 370 for v in self.controlled_vehicles):
            reward += 2.0
        
        return reward
