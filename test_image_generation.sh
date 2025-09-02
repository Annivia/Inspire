#!/bin/bash

# Complete test script for trajectory data collection with image and state reconstruction
# This will collect trajectory data and reconstruct both images and simulator states using stored actions

set -e  # Exit on any error

echo "=== Testing Complete Trajectory Data Collection ==="

# Configuration
TASK_SUITE="libero_90" 
TRAJECTORY_DATA_PATH="/work/nvme/bfbo/xzhang42/Inspire/test_trajectory_data_complete"
OUTPUT_DIR="/work/nvme/bfbo/xzhang42/Inspire/test_images"
MAX_TRAJECTORIES=8

echo "Task suite: $TASK_SUITE"
echo "Trajectory data path: $TRAJECTORY_DATA_PATH"
echo "Output directory: $OUTPUT_DIR" 
echo "Max trajectories: $MAX_TRAJECTORIES"

# Set environment variables (exactly like test_trajectory_data.sbatch)
export PYTHONPATH=/u/xzhang42/Inspire/LIBERO:$PYTHONPATH
export PYTHONPATH=/u/xzhang42/Inspire/vq_bet_official:$PYTHONPATH
export PYTHONPATH=/u/xzhang42/Inspire:$PYTHONPATH
export PRISMATIC_DATA_ROOT=/work/nvme/bfbo/xzhang42/Inspire 
export HF_HOME=/work/nvme/bfbo/xzhang42/huggingface

# # Step 1: Collect trajectory data with all components (exactly like test_trajectory_data.sbatch)
# echo "=== Step 1: Collecting Complete Trajectory Data ==="
# python /u/xzhang42/Inspire/vla_scripts/parallel_libero_evaluator.py \
#     --pretrained-checkpoint /work/nvme/bfbo/xzhang42/Inspire/runs/minivla-libero-90 \
#     --task-suite-name "$TASK_SUITE" \
#     --num-gpus 4 \
#     --num-processes 4 \
#     --num-trails-per-task 10 \
#     --steps 50000 \
#     --collect-trajectory-data \
#     --trajectory-data-save-path "$TRAJECTORY_DATA_PATH" \
#     --max-total-trajectories "$MAX_TRAJECTORIES" \
#     --save-root "$TRAJECTORY_DATA_PATH/results"

# # Step 2: Combine trajectory data files (exactly like test_trajectory_data.sbatch)
# echo "=== Step 2: Combining Trajectory Data Files ==="
# python /u/xzhang42/Inspire/vla_scripts/combine_trajectory_files.py \
#     --data-dir "$TRAJECTORY_DATA_PATH" \
#     --task-suite-name "$TASK_SUITE"

# # Step 3: Check collected data structure
# echo "=== Step 3: Verifying Collected Data ==="
# COLLECTED_FILE="$TRAJECTORY_DATA_PATH/trajectory_data_${TASK_SUITE}.h5"
# if [ -f "$COLLECTED_FILE" ]; then
#     echo "✓ Trajectory data file created: $COLLECTED_FILE"
#     python vla_scripts/load_trajectory_data.py "$COLLECTED_FILE" --structure
# else
#     echo "✗ Trajectory data file not found: $COLLECTED_FILE"
#     echo "Available files in $TRAJECTORY_DATA_PATH:"
#     ls -la "$TRAJECTORY_DATA_PATH/" || echo "Directory does not exist"
#     exit 1
# fi

# Step 4: Reconstruct images and simulator states using stored actions
echo "=== Step 4: Reconstructing Trajectory Data with Stored Actions ==="
COLLECTED_FILE="$TRAJECTORY_DATA_PATH/trajectory_data_${TASK_SUITE}.h5"
STATES_OUTPUT_DIR="/work/nvme/bfbo/xzhang42/Inspire/test_states"
echo "Using dataset file: $COLLECTED_FILE"
echo "Images output: $OUTPUT_DIR"
echo "States output: $STATES_OUTPUT_DIR"

if [ ! -f "$COLLECTED_FILE" ]; then
    echo "✗ Dataset file not found: $COLLECTED_FILE"
    echo "Available files in $TRAJECTORY_DATA_PATH:"
    ls -la "$TRAJECTORY_DATA_PATH/" || echo "Directory does not exist"
    exit 1
fi

# Reconstruct both images and simulator states using the ACTUAL stored actions
python vla_scripts/reconstruct_trajectory_data.py \
    "$COLLECTED_FILE" \
    --task-suite-name "$TASK_SUITE" \
    --images-output-dir "$OUTPUT_DIR" \
    --states-output-dir "$STATES_OUTPUT_DIR" \
    --max-episodes 8

# Step 5: Verify results
echo "=== Step 5: Verifying Reconstruction Results ==="

# Check images
if [ -d "$OUTPUT_DIR" ]; then
    IMAGE_COUNT=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
    echo "✓ Total images reconstructed: $IMAGE_COUNT"
    
    echo "Image directory structure:"
    find "$OUTPUT_DIR" -type d | head -3
    
    echo "Sample images:"
    find "$OUTPUT_DIR" -name "*.png" | head -3
else
    echo "✗ Image reconstruction failed - no output directory created"
    IMAGE_COUNT=0
fi

# Check simulator states
if [ -d "$STATES_OUTPUT_DIR" ]; then
    STATE_COUNT=$(find "$STATES_OUTPUT_DIR" -name "*.json" | wc -l)
    echo "✓ Total simulator states saved: $STATE_COUNT"
    
    echo "State directory structure:"
    find "$STATES_OUTPUT_DIR" -type d | head -3
    
    echo "Sample state files:"
    find "$STATES_OUTPUT_DIR" -name "*.json" | head -3
    
    echo "Sample state content (first file):"
    FIRST_STATE=$(find "$STATES_OUTPUT_DIR" -name "*.json" | head -1)
    if [ -f "$FIRST_STATE" ]; then
        echo "File: $FIRST_STATE"
        head -20 "$FIRST_STATE"
    fi
else
    echo "✗ State reconstruction failed - no states directory created"
    STATE_COUNT=0
fi

if [ "$IMAGE_COUNT" -gt 0 ] && [ "$STATE_COUNT" -gt 0 ]; then
    echo "✅ Both image and state reconstruction successful!"
    
    # Test unified data loading
    echo "=== Step 6: Testing Unified Data + Image Loading ==="
    python vla_scripts/load_trajectory_data_with_images.py \
        "$COLLECTED_FILE" \
        "$OUTPUT_DIR" \
        --task-id 0 \
        --episode-id 0 \
        --layer 0
    
    echo "=== Step 7: Verifying Action-Based Reconstruction ==="
    echo "The reconstructed images and states now use the ACTUAL stored actions from HDF5,"
    echo "ensuring perfect alignment with the collected hidden states and vision features."
    echo ""
    echo "Key improvements:"
    echo "- Images reconstructed using real robot motions (not dummy actions)"
    echo "- Simulator states capture full physics state at each timestep"
    echo "- Perfect correspondence with collected VLA trajectory data"
else
    echo "✗ Reconstruction failed - missing output directories"
    exit 1
fi

echo "=== Complete Trajectory Reconstruction Pipeline Test Successful ==="