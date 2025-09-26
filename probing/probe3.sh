#!/usr/bin/env bash
set -euo pipefail

# Experiment 3: Hidden -> Concepts (general_1)
# Usage: bash probing/probe3.sh

# Hardcoded defaults (no extra input required)
DATA_PATH="/work/nvme/bfbo/xzhang42/data/single_episode"
OUTPUT_DIR="./results/experiment_3"
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
GENERATION_STEPS="0"
TEST_SIZE=0.2
RANDOM_SEED=42

# Simple logging colors (consistent with probe1/2)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Ensure we run from the probing directory so relative paths resolve
SCRIPT_DIR="/u/xzhang42/Inspire/probing"
cd "$SCRIPT_DIR" || { print_error "Could not change to $SCRIPT_DIR"; exit 1; }

# Verify required files and data
if [[ ! -f "probe.py" ]]; then
  print_error "probe.py not found in $SCRIPT_DIR"
  exit 1
fi
if [[ ! -d "$DATA_PATH" ]]; then
  print_error "Data directory not found: $DATA_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

print_info "=== Experiment 3: [Hidden] -> Concepts (general_1) ==="
print_info "Data: $DATA_PATH"
print_info "Output: $OUTPUT_DIR"
print_info "Layers: $LAYERS"
print_info "Generation steps: $GENERATION_STEPS"

CMD="python -u probe.py"
CMD+=" --data-path \"$DATA_PATH\""
CMD+=" --experiment 3"
CMD+=" --output-dir \"$OUTPUT_DIR\""
CMD+=" --layers $LAYERS"
CMD+=" --generation-steps $GENERATION_STEPS"
CMD+=" --successful-only"
CMD+=" --test-size $TEST_SIZE"
CMD+=" --random-seed $RANDOM_SEED"
CMD+=" --task-category general_1"  # retain flag for future probing sets
CMD+=" --debug"

print_info "Command: $CMD"
eval $CMD
print_success "Experiment 3 completed. Results under: $OUTPUT_DIR"
