#!/bin/bash

# probe2.sh
# Run Experiment 2: [Vision encoder outputs] -> actions
# Linear regression probes for vision encoder patch features

set -e  # Exit on any error

# Configuration
DATA_PATH="/work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data"
OUTPUT_DIR="results/experiment_2"
DEBUG=true

# Activate environment if needed
# source activate probe

echo "============================================="
echo "Experiment 2: [Vision encoder outputs] -> actions"
echo "============================================="

echo "Configuration:"
echo "  Data path: $DATA_PATH"
echo "  Output dir: $OUTPUT_DIR" 
echo "  Debug: $DEBUG"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run Experiment 2
echo ""
echo "Running Experiment 2..."
python3 experiments/experiment_2_vision_to_actions.py \
    "$DATA_PATH" \
    "$OUTPUT_DIR" \
    --successful-only \
    --test-size 0.2 \
    --random-seed 42 \
    $([ "$DEBUG" = true ] && echo "--debug")

echo ""
echo "============================================="
echo "Experiment 2 completed!"
echo "============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"

# Display key results if available
if [ -f "$OUTPUT_DIR/experiment_2_complete_results.json" ]; then
    echo ""
    echo "Key Results:"
    python3 -c "
import json
with open('$OUTPUT_DIR/experiment_2_complete_results.json', 'r') as f:
    results = json.load(f)
    
if 'experiment_summary' in results and 'error' not in results['experiment_summary']:
    summary = results['experiment_summary']
    print(f'  Normal R2: {summary[\"normal_r2\"]:.4f}')
    print(f'  Random R2: {summary[\"random_r2\"]:.4f}')
    print(f'  Noise R2: {summary[\"noise_r2\"]:.4f}')
    print(f'  Linear separability: {summary[\"linear_separability_strength\"]:.4f}')
else:
    print('  Summary not available or error occurred')
"
fi

# Generate visualizations
echo ""
echo "============================================="
echo "Generating visualizations..."
echo "============================================="

if [ -f "$OUTPUT_DIR/experiment_2_complete_results.json" ]; then
    echo "Creating Experiment 2 visualizations..."
    
    # Check if Experiment 1 results exist for comparison
    EXP1_RESULTS="results/experiment_1/experiment_1_complete_results.json"
    if [ -f "$EXP1_RESULTS" ]; then
        echo "Found Experiment 1 results - creating comparison plots..."
        python3 visualize_experiment_2.py \
            "$OUTPUT_DIR/experiment_2_complete_results.json" \
            "$OUTPUT_DIR" \
            --exp1-results "$EXP1_RESULTS" \
            --dpi 300
    else
        echo "Experiment 1 results not found - creating Experiment 2 plots only..."
        python3 visualize_experiment_2.py \
            "$OUTPUT_DIR/experiment_2_complete_results.json" \
            "$OUTPUT_DIR" \
            --dpi 300
    fi
    
    echo ""
    echo "Visualizations saved to: $OUTPUT_DIR"
    echo "Generated plots:"
    echo "  - experiment_2_baseline_comparison.png"
    echo "  - experiment_2_action_dimensions.png (if per-dimension data available)"
    echo "  - experiment_2_separability_summary.png"
    if [ -f "$EXP1_RESULTS" ]; then
        echo "  - experiments_1_vs_2_comparison.png"
    fi
else
    echo "No results file found - skipping visualizations"
fi

echo ""
echo "============================================="
echo "Experiment 2 pipeline completed!"
echo "============================================="