#!/bin/bash
#
# probe1.sh
#
# Experiment 1: [Hidden state] -> actions
# Train linear regression probes for every layer's hidden state on every timestep.
# Evaluate with R2/MSE on held-out trajectories.
#
# Automatically runs all three baselines:
# 1. Normal: Original data
# 2. Randomized pairs: randomly shuffle hidden states and action sequences on trajectory basis  
# 3. Noise baseline: [Hidden state] -> gaussian noise with same dim as actions
#
# Usage: bash probe1.sh
#

# Hardcoded configuration for Experiment 1
DATA_PATH="/work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data"
EXPERIMENT=1
OUTPUT_DIR="./results/experiment_1"
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"  # All 25 layers (0-24)
GENERATION_STEPS="0"  # Full input processing step
TEST_SIZE=0.2
RANDOM_SEED=42
DEBUG="--debug"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_info "=== Experiment 1: [Hidden state] -> actions ==="
print_info "Linear regression probes testing what action information is linearly accessible"
print_info "in hidden states from each transformer layer."
print_info ""
print_info "This experiment automatically runs THREE baselines for comparison:"
print_info "  1. Normal: Original hidden states -> actions"
print_info "  2. Randomized: Shuffled hidden states -> actions (tests chance performance)"
print_info "  3. Noise: Hidden states -> Gaussian noise (tests overfitting)"
print_info ""
print_info "Configuration:"
print_info "  Data: $DATA_PATH"
print_info "  Output: $OUTPUT_DIR"
print_info "  Layers: All available (0-24, 25 total layers)"
print_info "  Generation steps: 0 (full input processing)"
print_info "  Test size: $TEST_SIZE"
print_info "  Random seed: $RANDOM_SEED"
print_info "  Debug: enabled"
print_info ""

# Ensure we're in the correct directory
SCRIPT_DIR="/u/xzhang42/Inspire/probing"
cd "$SCRIPT_DIR" || { print_error "Could not change to $SCRIPT_DIR"; exit 1; }

# Verify required files exist
if [[ ! -f "probe.py" ]]; then
    print_error "probe.py not found in $SCRIPT_DIR"
    exit 1
fi

if [[ ! -d "$DATA_PATH" ]]; then
    print_error "Data directory not found: $DATA_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build and run command - this will automatically run all 3 baselines
CMD="python probe.py"
CMD="$CMD --data-path \"$DATA_PATH\""
CMD="$CMD --experiment $EXPERIMENT"
CMD="$CMD --output-dir \"$OUTPUT_DIR\""
CMD="$CMD --layers $LAYERS"
CMD="$CMD --generation-steps $GENERATION_STEPS"
CMD="$CMD --test-size $TEST_SIZE"
CMD="$CMD --random-seed $RANDOM_SEED"
CMD="$CMD --successful-only"
CMD="$CMD $DEBUG"

print_info "Running Experiment 1 with all baselines..."
print_info "Command: $CMD"
print_info ""

# Execute experiment (this runs Normal, Randomized, and Noise baselines automatically)
eval $CMD
PROBE_EXIT_CODE=$?

if [[ $PROBE_EXIT_CODE -eq 0 ]]; then
    print_success "Experiment 1 completed successfully!"
    print_info ""
    print_info "All three baselines have been evaluated:"
    print_info "  ✓ Normal baseline (original data)"
    print_info "  ✓ Randomized baseline (shuffled pairs)"  
    print_info "  ✓ Noise baseline (Gaussian targets)"
    print_info ""
    
    # Generate visualizations
    print_info "Generating visualizations..."
    RESULTS_FILE="$OUTPUT_DIR/experiment_1_complete_results.json"
    
    if [[ -f "$RESULTS_FILE" ]]; then
        PLOTS_DIR="$OUTPUT_DIR/plots"
        mkdir -p "$PLOTS_DIR"
        
        VISUALIZE_CMD="python visualize_results.py \"$RESULTS_FILE\" --output-dir \"$PLOTS_DIR\" --experiment 1 --debug"
        print_info "Running: $VISUALIZE_CMD"
        print_info ""
        
        eval $VISUALIZE_CMD
        VISUALIZE_EXIT_CODE=$?
        
        if [[ $VISUALIZE_EXIT_CODE -eq 0 ]]; then
            print_success "Visualizations generated successfully!"
            print_info "Plots saved to: $PLOTS_DIR/"
            print_info ""
            
            # List generated plots
            if ls "$PLOTS_DIR"/*.png 1> /dev/null 2>&1; then
                print_info "Generated visualization plots:"
                for plot in "$PLOTS_DIR"/*.png; do
                    print_info "  - $(basename "$plot")"
                done
            fi
        else
            print_warning "Visualization generation failed (exit code: $VISUALIZE_EXIT_CODE)"
        fi
    else
        print_warning "Results file not found: $RESULTS_FILE"
    fi
    
    print_info ""
    print_success "=== Experiment 1 Complete ==="
    print_info "Results directory: $OUTPUT_DIR"
    print_info "Visualization plots: $OUTPUT_DIR/plots/"
    print_info ""
    print_info "Key output files:"
    print_info "  - experiment_1_complete_results.json (comprehensive results)"
    print_info "  - layer_*_gen_*_results.json (per-layer detailed results)"
    print_info "  - plots/experiment_1_*.png (visualization plots)"
    print_info ""
    print_info "The results compare linear separability across:"
    print_info "  • All transformer layers (0-24, 25 total layers)"
    print_info "  • All three baseline conditions"
    print_info "  • R2 scores show how well actions are linearly accessible from hidden states"
    
else
    print_error "Experiment 1 failed (exit code: $PROBE_EXIT_CODE)"
    print_error "Check the debug output above for details"
    exit $PROBE_EXIT_CODE
fi