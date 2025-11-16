#!/bin/bash
# Training wrapper script with proper Python path

# Activate virtual environment
source avs_venv/bin/activate

# Create runs directory if it doesn't exist
mkdir -p runs

# Set Python path to include current directory
export PYTHONPATH=.

# Run training with all arguments passed through
python training/train_mappo.py "$@"
