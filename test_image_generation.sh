#!/bin/bash

# Complete test script for trajectory data collection with image reconstruction clues
# This will collect trajectory data with hidden states, actions, vision features, AND image clues

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

# Step 4: Generate images from collected data
echo "=== Step 4: Generating Images from Trajectory Data ==="
COLLECTED_FILE="$TRAJECTORY_DATA_PATH/trajectory_data_${TASK_SUITE}.h5"
echo "Using dataset file: $COLLECTED_FILE"

if [ ! -f "$COLLECTED_FILE" ]; then
    echo "✗ Dataset file not found: $COLLECTED_FILE"
    echo "Available files in $TRAJECTORY_DATA_PATH:"
    ls -la "$TRAJECTORY_DATA_PATH/" || echo "Directory does not exist"
    exit 1
fi

python vla_scripts/generate_images_from_trajectory_data.py \
    "$COLLECTED_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --task-suite-name "$TASK_SUITE"

# Step 5: Verify results
echo "=== Step 5: Verifying Results ==="
if [ -d "$OUTPUT_DIR" ]; then
    IMAGE_COUNT=$(find "$OUTPUT_DIR" -name "*.png" | wc -l)
    echo "✓ Total images generated: $IMAGE_COUNT"
    
    echo "Directory structure:"
    find "$OUTPUT_DIR" -type d | head -5
    
    echo "Sample images:"
    find "$OUTPUT_DIR" -name "*.png" | head -3
    
    # Test unified data loading
    echo "=== Step 6: Testing Unified Data + Image Loading ==="
    python vla_scripts/load_trajectory_data_with_images.py \
        "$COLLECTED_FILE" \
        "$OUTPUT_DIR" \
        --task-id 0 \
        --episode-id 0 \
        --layer 0
else
    echo "✗ Image generation failed - no output directory created"
    exit 1
fi

echo "=== Complete Pipeline Test Successful ==="