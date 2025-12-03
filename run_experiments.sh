#!/bin/bash

# Comprehensive experiment runner for RL gain scheduling project
# Run all experiments with proper logging and organization

set -e  # Exit on any error

echo "====================================================="
echo "RL Gain Scheduling - Comprehensive Experiment Suite"
echo "====================================================="

# Create main experiment directory
EXPERIMENT_ROOT="./experiment_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_ROOT"
echo "All results will be saved to: $EXPERIMENT_ROOT"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Function to run experiment with logging
run_experiment() {
    local exp_name=$1
    local script_path=$2
    local log_file="$EXPERIMENT_ROOT/${exp_name}.log"
    
    echo ""
    echo "Running: $exp_name"
    echo "Script: $script_path"
    echo "Log: $log_file"
    echo "----------------------------------------"
    
    if python "$script_path" 2>&1 | tee "$log_file"; then
        echo "✓ $exp_name completed successfully"
        
        # Move generated files to experiment directory
        if [ -d "analysis_results" ]; then
            mv analysis_results "$EXPERIMENT_ROOT/${exp_name}_analysis_results"
        fi
        
        if [ -f "baseline_comparison.png" ]; then
            mv baseline_comparison.png "$EXPERIMENT_ROOT/"
        fi
        
        if [ -f "algorithm_comparison.png" ]; then
            mv algorithm_comparison.png "$EXPERIMENT_ROOT/"
        fi
        
        if [ -f "reward_weight_ablation.png" ]; then
            mv reward_weight_ablation.png "$EXPERIMENT_ROOT/"
        fi
        
        if [ -f "*.json" ]; then
            mv *.json "$EXPERIMENT_ROOT/"
        fi
        
    else
        echo "✗ $exp_name failed - check log file"
        return 1
    fi
}

# 1. Basic Training (if no model exists)
if [ ! -f "models/sac_planar_arm.zip" ]; then
    echo ""
    echo "No trained model found. Training basic SAC model first..."
    run_experiment "01_basic_training" "train_sac.py --timesteps 150000 --experiment_name basic_sac"
else
    echo "Found existing trained model, skipping basic training"
fi

# 2. Baseline Comparison
echo ""
echo "Starting baseline comparison experiments..."
run_experiment "02_baseline_comparison" "experiments/baseline_comparison.py"

# 3. Hyperparameter Tuning
echo ""
echo "Starting hyperparameter tuning experiments..."
run_experiment "03_hyperparameter_tuning" "experiments/hyperparameter_tuning.py"

# 4. Algorithm Comparison (shorter training for comparison)
echo ""
echo "Starting algorithm comparison experiments..."
run_experiment "04_algorithm_comparison" "experiments/algorithm_comparison.py"

# 5. Performance Analysis
echo ""
echo "Starting performance analysis..."
run_experiment "05_performance_analysis" "experiments/performance_analysis.py"

# 6. Generate comprehensive report
echo ""
echo "Generating comprehensive experiment report..."

cat > "$EXPERIMENT_ROOT/experiment_summary.md" << EOF
# RL Gain Scheduling Experiment Results

Generated on: $(date)

## Experiments Conducted

### 1. Baseline Comparison
- **Purpose**: Compare RL gain scheduling against fixed PD controllers
- **Files**: \`02_baseline_comparison.log\`, \`baseline_comparison.png\`
- **Key Metrics**: Mean return, success rate, positioning error

### 2. Hyperparameter Tuning
- **Purpose**: Ablation study on reward weights and other hyperparameters  
- **Files**: \`03_hyperparameter_tuning.log\`, \`reward_weight_ablation.png\`
- **Key Finding**: Impact of different reward weight combinations

### 3. Algorithm Comparison
- **Purpose**: Compare SAC vs PPO vs TD3 for this task
- **Files**: \`04_algorithm_comparison.log\`, \`algorithm_comparison.png\`
- **Key Metrics**: Sample efficiency, final performance, robustness

### 4. Performance Analysis
- **Purpose**: Detailed trajectory and gain scheduling analysis
- **Files**: \`05_performance_analysis.log\`, analysis results directory
- **Key Insights**: How RL agent adapts gains during different phases

## Key Results Summary

Check individual log files and generated plots for detailed results.

## Reproducibility

All experiments can be reproduced using:
\`\`\`bash
./run_experiments.sh
\`\`\`

Configuration files and random seeds are saved for each experiment.

## Next Steps

Based on these results, consider:
1. Fine-tuning the best-performing configuration
2. Testing on more complex scenarios (obstacles, tracking tasks)
3. Real robot validation
4. Comparison with classical adaptive control methods

EOF

# Create results overview
echo ""
echo "Creating results overview..."

python << EOF
import os
import json
import matplotlib.pyplot as plt

# Collect all JSON results
results_dir = "$EXPERIMENT_ROOT"
json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

print(f"Found {len(json_files)} result files:")
for f in json_files:
    print(f"  - {f}")

# You could add more sophisticated analysis here
EOF

echo ""
echo "====================================================="
echo "All experiments completed!"
echo "====================================================="
echo ""
echo "Results saved in: $EXPERIMENT_ROOT"
echo ""
echo "Quick overview:"
echo "  - Logs: *.log files"
echo "  - Plots: *.png files" 
echo "  - Data: *.json files"
echo "  - Summary: experiment_summary.md"
echo ""
echo "To view results:"
echo "  cd $EXPERIMENT_ROOT"
echo "  ls -la"
echo ""
echo "To start tensorboard (if training was run):"
echo "  tensorboard --logdir experiments/"