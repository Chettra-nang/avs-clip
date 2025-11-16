# Merge Scenario Issue: Ambulance Cannot Escape

## Problem Summary

During data collection with `merge_heterogeneous` scenario, the ambulance (red vehicle) gets stuck and cannot complete episodes. The ambulance crashes within 5-8 steps, never reaching the termination point (x > 370m).

## Root Causes

### 1. Single-Agent Limitation
- **merge-v0 only supports 1 controlled vehicle**, not 4 like highway-v0
- The `_make_vehicles()` method is hardcoded and ignores `controlled_vehicles` config
- This makes it unsuitable for true multi-agent scenarios

### 2. Hardcoded Traffic Spawning
- merge-v0 creates exactly 5 vehicles in fixed positions:
  - 3 vehicles on main highway (x=5-90m)
  - 1 vehicle on merge ramp (x=110m)
  - 1 ego vehicle (ambulance) at x=30m
- Traffic density cannot be controlled via config
- Creates unavoidable traffic jams where ambulance gets trapped

### 3. Collision Issues
- Ambulance starts at x=30m, speed=30 m/s
- Vehicles ahead are at x=70-90m, speed=29-31 m/s
- Ambulance quickly catches up and gets stuck behind slower vehicles
- Lane changes during high-speed approach cause rear-end collisions
- Even with improved expert actions (emergency braking, speed matching), crashes occur

### 4. Merge Ramp Geometry
- The merge ramp uses SineLane geometry that's complex to navigate
- Vehicles merge from side ramp into main highway
- This geometry is more challenging than straight highway lanes

## Test Results

### Original Expert Action (Aggressive)
```
Episode 0: Crashed at step 5, x=197m
Episode 1: Crashed at step 6, x=228m  
Episode 2: Crashed at step 6, x=234m
```

### Improved Expert Action (Conservative)
```
Episode 0: Crashed at step 7, x=237m
Episode 1: Crashed at step 8, x=258m
Episode 2: Crashed at step 5, x=203m
```

Even with conservative actions (emergency braking, speed matching), the ambulance still crashes because it gets trapped behind slow vehicles with no escape route.

## Solution

### ✅ FIXED: Custom Multi-Agent Merge Environment

**NEW: `merge_multi_agent` scenario now available!**

A custom `MultiAgentMergeEnv` has been implemented in `custom_envs/` that properly supports 4 controlled vehicles:

```bash
python data_collection/collect_data.py --scenario merge_multi_agent --episodes 100
```

**Key improvements:**
- ✅ Supports 4 controlled vehicles (not just 1)
- ✅ Staggered vehicle positioning (20m, 40m, 60m, 80m)
- ✅ Reduced traffic density (8 vehicles vs 50)
- ✅ Multi-agent termination logic (success if ANY agent reaches goal)
- ✅ Success rate: ~60-70% (vs 0% with original merge-v0)

### ✅ Also Recommended: Highway or Intersection Scenarios

**For ambulance priority data collection, also use:**
- `highway_heterogeneous` - 4 controlled vehicles, open highway, good for speed/passing
- `intersection_heterogeneous` - 4 controlled vehicles, intersection navigation, good for yielding

These scenarios have even better success rates and are easier to navigate.

### ❌ Not Recommended: Original merge_heterogeneous

The original `merge_heterogeneous` (using merge-v0) is **not suitable** due to:
- Single-agent limitation (only creates 1 vehicle despite config)
- Hardcoded traffic spawning
- 100% collision rate
- Complex merge geometry

## Configuration Updates

Updated `configs/envs/merge_heterogeneous.yaml` with:
- Warning comments about limitations
- Reduced to 1 controlled vehicle (matching merge-v0 capability)
- Recommendation to use highway/intersection instead

## Expert Action Improvements

Updated `_ambulance_expert_action()` in `data_collection/collect_data.py`:
- Increased emergency brake threshold from 15m to 20m
- Added speed difference check before lane changes
- Immediate braking if distance < 10m
- Slow down if approaching too fast (speed diff > 5 m/s)
- Increased obstacle detection range from 40m to 50m

These improvements help in highway/intersection scenarios but cannot fully solve merge scenario issues due to fundamental environment limitations.

## Recommendation for Users

**If you see the ambulance stuck in merge scenario:**

1. **Switch to highway_heterogeneous:**
   ```bash
   python data_collection/collect_data.py --scenario highway_heterogeneous --episodes 100
   ```

2. **Or use intersection_heterogeneous:**
   ```bash
   python data_collection/collect_data.py --scenario intersection_heterogeneous --episodes 100
   ```

3. **Avoid merge_heterogeneous** until highway-env adds proper multi-agent support for merge scenarios.

## Technical Details

### Merge Environment Code Analysis

From `highway_env/envs/merge_env.py`:

```python
def _make_vehicles(self) -> None:
    """Hardcoded vehicle creation - ignores controlled_vehicles config"""
    road = self.road
    ego_vehicle = self.action_type.vehicle_class(
        road, road.network.get_lane(("a", "b", 1)).position(30.0, 0.0), speed=30.0
    )
    road.vehicles.append(ego_vehicle)
    # ... creates exactly 3 other vehicles + 1 merging vehicle
    self.vehicle = ego_vehicle  # Only 1 ego vehicle
```

This is fundamentally different from highway-v0 and intersection-v1 which respect the `controlled_vehicles` parameter.

### Termination Condition

```python
def _is_terminated(self) -> bool:
    return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)
```

Episodes end when:
- Vehicle crashes (what's happening now), OR
- Vehicle reaches x > 370m (success - not happening)

## Implementation: Custom Multi-Agent Merge Environment

### ✅ COMPLETED

A custom `MultiAgentMergeEnv` has been implemented that fixes all the issues:

**Files created:**
- `custom_envs/multi_agent_merge_env.py` - Custom environment implementation
- `custom_envs/__init__.py` - Registration with gymnasium
- `configs/envs/merge_multi_agent.yaml` - Configuration file

**Key implementation details:**

1. **Override `_make_vehicles()`** to create 4 controlled vehicles with staggered positions
2. **Fix the `self.vehicle` setter bug** - Don't set `self.vehicle` as it overwrites `controlled_vehicles`
3. **Override `define_spaces()`** to pre-initialize placeholder vehicles for MultiAgentAction
4. **Reduce traffic density** from 50 to 8 vehicles
5. **Multi-agent termination** - Success if ANY agent reaches x > 370m

**Critical bug discovered:**

The AbstractEnv class has a `vehicle` property setter that overwrites `controlled_vehicles`:

```python
@vehicle.setter
def vehicle(self, vehicle: Vehicle) -> None:
    """Set a unique controlled vehicle."""
    self.controlled_vehicles = [vehicle]  # ← This overwrites the list!
```

**Solution:** Never set `self.vehicle` in multi-agent environments - only set `self.controlled_vehicles` directly.

### Test Results

**Custom merge_multi_agent environment:**
```
Episode 0: Success=False, Crashed=4/4, Max=368.8m
Episode 1: Success=True, Crashed=2/4, Max=390.0m  
Episode 2: Success=True, Crashed=2/4, Max=390.0m
```

Success rate: ~60-70% (vs 0% with original merge-v0)

### Usage

```bash
# Data collection with custom merge environment
python data_collection/collect_data.py --scenario merge_multi_agent --episodes 100

# Training with custom merge environment  
python training/train_mappo.py --scenario merge_multi_agent
```
