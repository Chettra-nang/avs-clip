#!/bin/bash
set -e

# VLM-MARL Highway Training Pipeline
# This script orchestrates data collection, CLIP fine-tuning, and MAPPO training
#
# Usage:
#   ./run.sh [scenario] [--heterogeneous]
#
# Examples:
#   ./run.sh                                    # Default: highway homogeneous
#   ./run.sh highway                            # Highway homogeneous
#   ./run.sh highway_heterogeneous              # Highway heterogeneous (emergency vehicle)
#   ./run.sh merge --heterogeneous              # Merge heterogeneous
#
# Heterogeneous Mode:
#   Enables emergency vehicle priority scenario where:
#   - Agent 0 (ambulance): Red vehicle, aggressive driving, prioritizes speed
#   - Agents 1-3 (normal): Green vehicles, yield to ambulance, clear lanes
#   - Asymmetric rewards and role-specific policies

# Set PYTHONPATH to current directory
export PYTHONPATH=.

echo "=== VLM-MARL Highway Training Pipeline ==="
echo ""

# Parse arguments
SCENARIO=${1:-highway}
HETEROGENEOUS=false

# Check for --heterogeneous flag
if [[ "$2" == "--heterogeneous" ]] || [[ "$SCENARIO" == *"_heterogeneous"* ]]; then
    HETEROGENEOUS=true
    # If scenario doesn't already have _heterogeneous suffix, add it
    if [[ "$SCENARIO" != *"_heterogeneous"* ]]; then
        SCENARIO="${SCENARIO}_heterogeneous"
    fi
fi

echo "Configuration:"
echo "  Scenario: $SCENARIO"
echo "  Heterogeneous Mode: $HETEROGENEOUS"
echo ""

# Step 1: Data Collection
if [ "$HETEROGENEOUS" = true ]; then
    echo "Step 1: Collecting heterogeneous data (emergency vehicle priority)..."
    echo "----------------------------------------------------------------------"
    echo "  Agent 0 (ambulance): Red vehicle, aggressive driving"
    echo "  Agents 1-3 (normal): Green vehicles, yield to ambulance"
    echo ""
else
    echo "Step 1: Collecting data from all three scenarios..."
    echo "---------------------------------------------------"
fi

if [ "$HETEROGENEOUS" = true ]; then
    # Heterogeneous mode: collect only for specified scenario
    python3 data_collection/collect_data.py \
        --scenario $SCENARIO \
        --episodes 100 \
        --max-steps 1000 \
        --output-dir data \
        --log-dir runs/data_collection_hetero \
        --seed 42
else
    # Homogeneous mode: collect for all three scenarios
    python3 data_collection/collect_data.py \
        --scenario highway \
        --episodes 100 \
        --max-steps 1000 \
        --output-dir data \
        --log-dir runs/data_collection \
        --seed 42

    python3 data_collection/collect_data.py \
        --scenario merge \
        --episodes 100 \
        --max-steps 1000 \
        --output-dir data \
        --log-dir runs/data_collection \
        --seed 43

    python3 data_collection/collect_data.py \
        --scenario intersection \
        --episodes 100 \
        --max-steps 1000 \
        --output-dir data \
        --log-dir runs/data_collection \
        --seed 44
fi

echo ""
echo "Data collection complete!"
echo ""

# Step 2: CLIP Fine-tuning
echo "Step 2: Fine-tuning CLIP on collected data..."
echo "----------------------------------------------"

if [ "$HETEROGENEOUS" = true ]; then
    # Heterogeneous mode: train on heterogeneous data only
    # Extract base scenario name (remove _heterogeneous suffix)
    BASE_SCENARIO=${SCENARIO/_heterogeneous/}
    
    python3 training/train_clip.py \
        --data_dirs data/$SCENARIO \
        --model_id openai/clip-vit-base-patch32 \
        --epochs 5 \
        --batch_size 64 \
        --lr 1e-5 \
        --num_workers 4 \
        --log_dir runs/clip_hetero \
        --save_dir models/clip_hetero
    
    echo ""
    echo "CLIP fine-tuning complete (heterogeneous mode)!"
    echo "  Text labels include role prefixes: 'AMBULANCE' and 'YIELD'"
    echo ""
else
    # Homogeneous mode: train on all scenarios
    python3 training/train_clip.py \
        --data_dirs data/highway data/merge data/intersection \
        --model_id openai/clip-vit-base-patch32 \
        --epochs 5 \
        --batch_size 64 \
        --lr 1e-5 \
        --num_workers 4 \
        --log_dir runs/clip \
        --save_dir models/clip
    
    echo ""
    echo "CLIP fine-tuning complete!"
    echo ""
fi

# Step 3: MAPPO Training
echo "Step 3: Training MAPPO with VLA..."
echo "-----------------------------------"

if [ "$HETEROGENEOUS" = true ]; then
    echo "Training heterogeneous MAPPO on: $SCENARIO"
    echo "  Architecture: HeterogeneousActor with role-specific sub-networks"
    echo "  Rewards: Asymmetric (ambulance: speed, normal: lane clearance)"
    echo ""
    
    python3 training/train_mappo.py \
        --scenario $SCENARIO \
        --steps 100000 \
        --horizon 256 \
        --logdir runs \
        --clip_dir models/clip_hetero \
        --lr 3e-4
    
    echo ""
    echo "Heterogeneous MAPPO training complete!"
    echo "  Model saved to: models/vla_mappo_heterogeneous_${SCENARIO}.pth"
    echo ""
else
    echo "Training MAPPO on: $SCENARIO"
    echo ""
    
    python3 training/train_mappo.py \
        --scenario $SCENARIO \
        --steps 100000 \
        --horizon 256 \
        --logdir runs \
        --clip_dir models/clip \
        --lr 3e-4
    
    echo ""
    echo "MAPPO training complete!"
    echo ""
fi

# Step 4: TensorBoard Visualization
echo "=== Training Pipeline Complete ==="
echo ""

if [ "$HETEROGENEOUS" = true ]; then
    echo "Heterogeneous training metrics available in TensorBoard:"
    echo "  - ambulance/average_speed: Ambulance speed performance"
    echo "  - normal/lane_clearance_rate: % timesteps not blocking"
    echo "  - heterogeneous/priority_passage_success: Goal completion rate"
    echo ""
fi

echo "To visualize training metrics, run:"
echo "  tensorboard --logdir runs"
echo ""
echo "Then open http://localhost:6006 in your browser"
