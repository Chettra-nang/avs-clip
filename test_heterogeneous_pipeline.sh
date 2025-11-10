#!/bin/bash
# Test script for heterogeneous pipeline with ambulance detection fix

set -e

echo "=========================================="
echo "Heterogeneous Pipeline Test"
echo "=========================================="
echo ""

# Activate environment
source avs_venv/bin/activate
export PYTHONPATH=.

# Clean up previous test data
rm -rf data_test_pipeline
mkdir -p data_test_pipeline

echo "Step 1: Testing ambulance detection fix..."
echo "------------------------------------------"
python3 data_collection/collect_data.py \
    --scenario highway_heterogeneous \
    --episodes 5 \
    --max-steps 100 \
    --output-dir data_test_pipeline \
    --log-dir runs/test_pipeline \
    --seed 42

echo ""
echo "Step 2: Verifying data quality..."
echo "------------------------------------------"
python3 verify_agent_spacing.py data_test_pipeline/highway_heterogeneous/dataset.jsonl

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Expected Results:"
echo "  ✓ Ambulance detection rate: >80%"
echo "  ✓ Yielding behavior: >60% when detected"
echo "  ✓ Average quality: 0.6-0.7"
echo ""
echo "Next Steps:"
echo "  1. Collect full dataset (70 normal + 30 dense episodes)"
echo "  2. Run CLIP fine-tuning"
echo "  3. Run MAPPO training"
echo ""
