#!/bin/bash

# Test trajectory data collection script for pilot data (16 trajectories only)
# Quick collection for testing the optimized format before full dataset collection

set -e  # Exit on any error

# Default parameters - Pilot test collection
NUM_TASKS=2
SAVE_DIR=""
TASK_SUITE="libero_90"
NUM_TRIALS_PER_TASK=8  # 2 tasks * 8 trials = 16 trajectories total
NUM_GPUS=4
NUM_PROCESSES=16  # Reduced for pilot test
RECONSTRUCT_IMAGES=false
RECONSTRUCT_STATES=false

# Usage function
usage() {
    echo "Usage: $0 --save-dir DIR [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --save-dir DIR          Directory to save pilot trajectory data (should be in /work/nvme/...)"
    echo ""
    echo "Optional arguments:"
    echo "  --num-tasks N           Number of LIBERO tasks to collect (default: 2)"
    echo "  --num-trials TRIALS     Number of trials per task (default: 8)"
    echo "  --task-suite SUITE      LIBERO task suite (default: libero_90)"
    echo "  --num-gpus GPUS         Number of GPUs to use (default: 4)"
    echo "  --num-processes PROCS   Number of processes (default: 16)"
    echo "  --with-images           Enable image reconstruction"
    echo "  --with-states           Enable simulator state reconstruction"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --save-dir /work/nvme/bfbo/xzhang42/data/pilot_test"
    echo "  $0 --save-dir /work/nvme/bfbo/xzhang42/data/test --num-tasks 1 --num-trials 16"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --task-suite)
            TASK_SUITE="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS_PER_TASK="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --num-processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        --with-images)
            RECONSTRUCT_IMAGES=true
            shift
            ;;
        --with-states)
            RECONSTRUCT_STATES=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SAVE_DIR" ]]; then
    echo "ERROR: --save-dir is required"
    usage
fi

if [[ "$NUM_TASKS" -lt 1 || "$NUM_TASKS" -gt 90 ]]; then
    echo "ERROR: --num-tasks must be between 1 and 90"
    exit 1
fi

# Validate save directory is in fast storage
if [[ "$SAVE_DIR" != *"/work/nvme/"* ]]; then
    echo "WARNING: Save directory should be in /work/nvme/ for fast storage"
    echo "Current save directory: $SAVE_DIR"
fi

# Calculate expected trajectories
EXPECTED_TRAJECTORIES=$((NUM_TASKS * NUM_TRIALS_PER_TASK))

echo "=== Pilot Trajectory Data Collection Test ==="
echo "Number of tasks: $NUM_TASKS"
echo "Task suite: $TASK_SUITE"
echo "Save directory: $SAVE_DIR"
echo "Trials per task: $NUM_TRIALS_PER_TASK"
echo "Expected total trajectories: $EXPECTED_TRAJECTORIES"
echo "GPUs: $NUM_GPUS, Processes: $NUM_PROCESSES"
echo "Reconstruct images: $RECONSTRUCT_IMAGES"
echo "Reconstruct states: $RECONSTRUCT_STATES"
echo ""

if [[ "$EXPECTED_TRAJECTORIES" -gt 50 ]]; then
    echo "WARNING: This will collect $EXPECTED_TRAJECTORIES trajectories."
    echo "For pilot testing, consider using fewer tasks/trials."
    echo "Recommended: --num-tasks 2 --num-trials 8 (16 trajectories)"
    echo "Auto-continuing for batch processing..."
    echo ""
fi

# Set environment variables
export PYTHONPATH=/u/xzhang42/Inspire/LIBERO:$PYTHONPATH
export PYTHONPATH=/u/xzhang42/Inspire/vq_bet_official:$PYTHONPATH
export PYTHONPATH=/u/xzhang42/Inspire:$PYTHONPATH
export PRISMATIC_DATA_ROOT=/work/nvme/bfbo/xzhang42/Inspire
export HF_HOME=/work/nvme/bfbo/xzhang42/huggingface

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p "$SAVE_DIR/results"
mkdir -p "$SAVE_DIR/trajectory_data"

echo "=== Step 1: Collecting Pilot Trajectory Data ==="
echo "Starting data collection for $NUM_TASKS tasks (pilot test)..."
echo "NOTE: Testing VLM embeddings collection alongside vision features"

# Run trajectory data collection with early stopping
python /u/xzhang42/Inspire/vla_scripts/parallel_libero_evaluator.py \
    --pretrained-checkpoint /work/nvme/bfbo/xzhang42/Inspire/runs/minivla-libero-90 \
    --task-suite-name "$TASK_SUITE" \
    --num-gpus "$NUM_GPUS" \
    --num-processes "$NUM_PROCESSES" \
    --num-trails-per-task "$NUM_TRIALS_PER_TASK" \
    --max-total-trajectories "$EXPECTED_TRAJECTORIES" \
    --steps 50000 \
    --collect-trajectory-data \
    --trajectory-data-save-path "$SAVE_DIR/trajectory_data" \
    --save-root "$SAVE_DIR/results" || {
    echo "ERROR: Pilot data collection failed"
    exit 1
}

echo "Pilot data collection completed successfully"

echo "=== Step 2: Combining Optimized Pilot Data Files ==="
echo "Combining chunk files from pilot run into optimized format..."
echo "NOTE: Including VLM embeddings in optimized format"

# Combine trajectory data files into optimized multi-file format
python /u/xzhang42/Inspire/vla_scripts/combine_optimized_trajectory_files.py \
    --temp-dir "$SAVE_DIR/trajectory_data/temp_trajectory_processing" \
    --output-dir "$SAVE_DIR/optimized_trajectory_data" \
    --task-suite "$TASK_SUITE" \
    --cleanup-temp || {
    echo "ERROR: Optimized file combination failed"
    exit 1
}

echo "Optimized pilot data combination completed successfully"

# Verify optimized format was created
OPTIMIZED_DIR="$SAVE_DIR/optimized_trajectory_data"
if [[ ! -d "$OPTIMIZED_DIR" ]]; then
    echo "ERROR: Optimized trajectory directory not found: $OPTIMIZED_DIR"
    exit 1
fi

echo "✓ Optimized pilot trajectory data created: $OPTIMIZED_DIR"

# Check optimized format structure
echo "=== Step 3: Verifying Optimized Pilot Data ==="
OPTIMIZED_DIR="$SAVE_DIR/optimized_trajectory_data"
python -c "
import sys
sys.path.append('/u/xzhang42/Inspire')
from vla_scripts.load_optimized_trajectory_data import OptimizedTrajectoryLoader
import json
from pathlib import Path

optimized_dir = Path('$OPTIMIZED_DIR')
if optimized_dir.exists():
    try:
        loader = OptimizedTrajectoryLoader(optimized_dir)
        info = loader.get_dataset_info()
        
        print(f'✓ Pilot trajectory data verification completed')
        print(f'Pilot dataset info:')
        print(json.dumps(info, indent=2))
        
        # Calculate total file size
        total_size = sum(f.stat().st_size for f in optimized_dir.rglob('*.h5'))
        print(f'Total pilot data size: {total_size / (1024*1024):.2f} MB')
        
        # Quick performance test - load one layer
        if info['summary']['layer_indices']:
            test_layer = info['summary']['layer_indices'][0]
            print(f'\\n=== Performance Test: Loading Layer {test_layer} ===')
            import time
            start_time = time.time()
            hidden_states, episodes = loader.load_layer_data(test_layer)
            load_time = time.time() - start_time
            print(f'✓ Loaded layer {test_layer}: {hidden_states.shape} in {load_time:.3f}s')
            print(f'✓ Performance test passed - optimized loading works!')
        
        # Test VLM embeddings loading
        print(f'\\n=== VLM Embeddings Test ===')
        start_time = time.time()
        vlm_embeddings, episodes = loader.load_vlm_embeddings_data()
        load_time = time.time() - start_time
        if vlm_embeddings.size > 0:
            print(f'✓ Loaded VLM embeddings: {vlm_embeddings.shape} in {load_time:.3f}s')
            print(f'✓ VLM embeddings collection successful!')
        else:
            print(f'⚠ VLM embeddings array is empty - may indicate collection issue')
        
    except Exception as e:
        print(f'ERROR: Failed to verify pilot data: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
else:
    print('ERROR: Optimized directory not found')
    exit(1)
" || {
    echo "ERROR: Pilot data verification failed"
    exit 1
}

# Reconstruction phase (optional for pilot)
if [[ "$RECONSTRUCT_IMAGES" == "true" || "$RECONSTRUCT_STATES" == "true" ]]; then
    echo "=== Step 4: Reconstructing Pilot Images/States ==="
    echo "Note: Reconstruction uses episode_index.h5 for metadata"
    
    # TODO: Update reconstruction script for optimized format
    echo "WARNING: Reconstruction not yet implemented for optimized format"
    echo "Skipping reconstruction phase for pilot test"
else
    echo "=== Step 4: Skipping Reconstruction (disabled) ==="
fi

echo "=== Final Pilot Test Summary ==="
echo "✅ Pilot trajectory data collection completed successfully!"
echo ""
echo "Pilot data location: $SAVE_DIR"
echo "├── optimized_trajectory_data/            # Optimized multi-file format"
echo "│   ├── hidden_states/                    # Individual layer files"
echo "│   ├── actions.h5                        # Action data"
echo "│   ├── vision_features.h5                # Raw vision encoder features"
echo "│   ├── vlm_embeddings.h5                 # VLM-transformed visual embeddings"
echo "│   └── episode_index.h5                  # Episode metadata & indexing"
echo "├── results/                              # Evaluation results"
echo ""
echo "🚀 Ready to test linear probing with optimized format!"
echo ""
echo "Next steps:"
echo "1. Test probing experiment: cd probing && bash probe1.sh"
echo "2. Update data path in probe1.sh to point to: $OPTIMIZED_DIR"
echo "3. If pilot test works, run full collection with collect_trajectory_data.sh"

# Quick data summary
python -c "
import sys
sys.path.append('/u/xzhang42/Inspire')
from vla_scripts.load_optimized_trajectory_data import OptimizedTrajectoryLoader

loader = OptimizedTrajectoryLoader('$OPTIMIZED_DIR')
info = loader.get_dataset_info()
ep_stats = info['episode_stats']

print(f'📊 Pilot Collection Statistics:')
print(f'   Total episodes: {ep_stats[\"total_episodes\"]}')
print(f'   Successful episodes: {ep_stats[\"successful_episodes\"]} ({ep_stats[\"successful_episodes\"]/ep_stats[\"total_episodes\"]*100:.1f}%)')
print(f'   Unique tasks: {ep_stats[\"unique_tasks\"]}')
print(f'   Available layers: {len(info[\"summary\"][\"layer_indices\"])}')
"

echo ""
echo "=== Pilot Test Complete ==="