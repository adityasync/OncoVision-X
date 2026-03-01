#!/bin/bash

# Master script to run all experiments

echo "======================================"
echo "RUNNING ALL EXPERIMENTS"
echo "======================================"

# Array of all experiments
experiments=(
    "ablation_no_context"
    "ablation_no_attention"
    "ablation_no_curriculum"
    "ablation_no_uncertainty"
    "baseline_resnet3d18"
    "baseline_resnet2d18"
)

# Run each experiment
for exp in "${experiments[@]}"; do
    echo ""
    echo "======================================"
    echo "Starting: $exp"
    echo "======================================"
    
    # Train
    python scripts/train_experiment.py --experiment $exp
    
    # Evaluate
    python scripts/evaluate_experiment.py --experiment $exp --checkpoint best --split test
    
    echo "✓ Completed: $exp"
    echo ""
done

# Compare all results
echo ""
echo "======================================"
echo "GENERATING COMPARISON RESULTS"
echo "======================================"
python scripts/compare_all_experiments.py

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "======================================"
