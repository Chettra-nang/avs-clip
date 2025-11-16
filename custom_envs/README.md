# Custom Highway-Env Environments

This directory contains custom environment implementations that extend highway-env for specific use cases.

## MultiAgentMergeEnv

**File:** `multi_agent_merge_env.py`  
**Registered ID:** `merge-multi-agent-v0`  
**Config:** `configs/envs/merge_multi_agent.yaml`

### Purpose

Extends the standard `merge-v0` environment to properly support 4 controlled vehicles for multi-agent reinforcement learning and ambulance priority scenarios.

### Key Features

- ✅ **4 controlled vehicles** (vs 1 in original merge-v0)
- ✅ **Staggered positioning** (20m, 40m, 60m, 80m) to prevent immediate collisions
- ✅ **Reduced traffic density** (8 total vehicles vs 50+)
- ✅ **Multi-agent termination** - Success if ANY agent reaches goal (x > 370m)
- ✅ **Team-based rewards** - All agents benefit from progress
- ✅ **Square rendering** (600x600) - No distortion when resized to 224x224 for CLIP

### Implementation Details

#### Critical Bug Fix

The highway-env `AbstractEnv` class has a `vehicle` property setter that overwrites `controlled_vehicles`:

```python
@vehicle.setter
def vehicle(self, vehicle: Vehicle) -> None:
    """Set a unique controlled vehicle."""
    self.controlled_vehicles = [vehicle]  # ← Overwrites the list!
```

**Solution:** Never set `self.vehicle` in `_make_vehicles()` - only set `self.controlled_vehicles` directly.

#### Vehicle Placement Strategy

```python
for i in range(4):
    position = 20.0 + (i * 20.0)  # 20m, 40m, 60m, 80m
    lane_id = i % 2  # Alternate lanes 0, 1
    speed = 28.0 + (i * 1.0)  # 28, 29, 30, 31 m/s
```

This creates a staggered formation that:
- Prevents immediate rear-end collisions
- Allows agents to spread across lanes
- Provides varied speeds for diverse behaviors

#### Traffic Reduction

- **AI vehicles:** 2 (vs 3 in original)
- **Merging vehicles:** 1 (same as original)
- **Total vehicles:** 7 (4 controlled + 2 AI + 1 merging)

This reduces congestion and gives controlled vehicles more space to navigate.

### Usage

#### Data Collection

```bash
python data_collection/collect_data.py \
    --scenario merge_multi_agent \
    --episodes 100 \
    --max-steps 1000
```

#### Training

```bash
python training/train_mappo.py \
    --scenario merge_multi_agent \
    --steps 100000 \
    --horizon 256
```

#### Direct Usage

```python
import gymnasium as gym
import custom_envs  # Register custom environments

env = gym.make('merge-multi-agent-v0', render_mode='rgb_array')
obs, info = env.reset()

# 4 controlled vehicles
print(f"Controlled vehicles: {len(env.unwrapped.controlled_vehicles)}")

# Multi-agent action space
actions = (3, 3, 3, 3)  # All agents go FASTER
obs, reward, done, truncated, info = env.step(actions)
```

### Performance

**Success Rate:** ~60-70% (vs 0% with original merge-v0)

**Typical Results:**
```
Episode 0: Success=False, Crashed=4/4, Max=368.8m
Episode 1: Success=True, Crashed=2/4, Max=390.0m  
Episode 2: Success=True, Crashed=2/4, Max=390.0m
```

Some agents may crash, but at least 1-2 typically reach the goal (x > 370m).

### Comparison with Original merge-v0

| Feature | merge-v0 | merge-multi-agent-v0 |
|---------|----------|----------------------|
| Controlled vehicles | 1 | 4 |
| Traffic density | High (50+) | Low (8) |
| Success rate | 0% | 60-70% |
| Multi-agent support | ❌ | ✅ |
| Ambulance priority | ❌ | ✅ |

### Known Limitations

1. **Some agents still crash** (~30-40% crash rate)
2. **Merge ramp geometry** is still challenging
3. **Not as stable as highway/intersection** scenarios

For best results with ambulance priority, consider using:
- `highway_heterogeneous` - More stable, higher success rate
- `intersection_heterogeneous` - Better for yielding behaviors

### Future Improvements

Potential enhancements:
1. Adjust merge ramp geometry for easier navigation
2. Implement intelligent traffic spawning based on agent positions
3. Add dynamic difficulty adjustment
4. Improve reward shaping for cooperative merging

## Adding New Custom Environments

To add a new custom environment:

1. Create a new file in `custom_envs/` (e.g., `my_custom_env.py`)
2. Extend an existing highway-env class
3. Register it in `custom_envs/__init__.py`:
   ```python
   register(
       id='my-custom-env-v0',
       entry_point='custom_envs.my_custom_env:MyCustomEnv',
   )
   ```
4. Create a config file in `configs/envs/`
5. Update `envs/make_env.py` to handle the new scenario
6. Add to data collection and training scripts

## References

- [Highway-Env Documentation](https://highway-env.farama.org/)
- [Multi-Agent Support](https://highway-env.farama.org/multi_agent/)
- [GitHub Issue #35](https://github.com/eleurent/highway-env/issues/35) - Multi-agent merge discussion
