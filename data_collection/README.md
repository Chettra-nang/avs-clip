# Data Collection Module

This module implements the data collection pipeline for VLM-MARL highway driving.

## Features

- **Vehicle State Extraction**: Extracts position, velocity, speed, and heading from vehicles
- **Scene Analysis**: Scans nearby vehicles within 60m radius and computes risk metrics
- **Expert Heuristic**: Rule-based policy using TTC and gap analysis for action labeling
- **Quality Filtering**: Saves only high-quality frames (quality > 0.5 or 25% random)
- **Image Processing**: Center crops frames to 224×224 for CLIP compatibility
- **TensorBoard Logging**: Logs quality, action distributions, and traffic metrics

## Usage

### Basic Usage

```bash
python3 -m data_collection.collect_data \
    --scenario highway \
    --episodes 100 \
    --max-steps 1000
```

### All Options

```bash
python3 -m data_collection.collect_data \
    --scenario highway \           # highway, merge, or intersection
    --episodes 100 \                # Number of episodes
    --max-steps 1000 \              # Max steps per episode
    --output-dir data \             # Output directory
    --log-dir runs/data_collection \  # TensorBoard log directory
    --seed 42                       # Random seed
```

## Output Structure

```
data/
└── highway/
    ├── dataset.jsonl          # Frame metadata and labels
    ├── images/                # Cropped 224×224 PNG images
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    └── metadata.json          # Collection statistics
```

## Dataset Format

Each line in `dataset.jsonl` contains:

```json
{
  "index": 0,
  "image": "images/000000.png",
  "episode": 0,
  "step": 5,
  "quality": 0.73,
  "agents": [
    {
      "speed": 25.3,
      "position": [120.5, 8.0],
      "heading": 0.05,
      "scene": {
        "front_dist": 15.2,
        "front_ttc": 3.1,
        "risk": 0.68,
        "can_left": true,
        "can_right": false
      },
      "action_id": 0,
      "action_name": "LANE_LEFT"
    }
  ],
  "timestamp": "2025-11-09T..."
}
```

## Expert Heuristic Logic

1. **Emergency** (TTC < 2.0s OR front_dist < 10m): Lane change or slow down
2. **Caution** (front_dist < 20m AND front_ttc < 4.0s): Slow down
3. **Exploration** (12% probability): Random lane change
4. **Speed Optimization**: FASTER (50%) or IDLE (50%)

## Quality Score

Quality is computed as weighted average across agents:
- 40% TTC (lower is better)
- 30% distance (closer is better)
- 30% risk (higher is better)

Frames are saved if quality > 0.5 OR with 25% random probability.
