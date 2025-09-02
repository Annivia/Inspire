#!/bin/bash

# Production trajectory data collection script for LIBERO tasks
# Collects trajectory data for n tasks and runs complete processing pipeline

set -e  # Exit on any error

# Default parameters
NUM_TASKS=10
SAVE_DIR=""
TASK_SUITE="libero_90"
NUM_TRIALS_PER_TASK=10
NUM_GPUS=4
NUM_PROCESSES=32
RECONSTRUCT_IMAGES=false
RECONSTRUCT_STATES=false

# Usage function
usage() {
    echo "Usage: $0 --num-tasks N --save-dir DIR [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --num-tasks N           Number of LIBERO tasks to collect (0-90)"
    echo "  --save-dir DIR          Directory to save trajectory data (should be in /work/nvme/...)"
    echo ""
    echo "Optional arguments:"
    echo "  --task-suite SUITE      LIBERO task suite (default: libero_90)"
    echo "  --num-trials TRIALS     Number of trials per task (default: 10)"
    echo "  --num-gpus GPUS         Number of GPUs to use (default: 4)"
    echo "  --num-processes PROCS   Number of processes (default: 4)"
    echo "  --no-images             Skip image reconstruction"
    echo "  --no-states             Skip simulator state reconstruction"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --num-tasks 10 --save-dir /work/nvme/bfbo/xzhang42/data/pilot"
    echo "  $0 --num-tasks 90 --save-dir /work/nvme/bfbo/xzhang42/data/full --num-trials 5"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-tasks)
            NUM_TASKS="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
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
        --no-images)
            RECONSTRUCT_IMAGES=false
            shift
            ;;
        --no-states)
            RECONSTRUCT_STATES=false
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

if [[ "$NUM_TASKS" -lt 0 || "$NUM_TASKS" -gt 90 ]]; then
    echo "ERROR: --num-tasks must be between 0 and 90"
    exit 1
fi

if [[ "$NUM_TASKS" -eq 0 ]]; then
    echo "WARNING: --num-tasks is 0, no data will be collected"
    exit 0
fi

# Validate save directory is in fast storage
if [[ "/work/nvme/" != *"$SAVE_DIR"* ]]; then
    echo "WARNING: Save directory should be in /work/nvme/ for fast storage"
    echo "Current save directory: $SAVE_DIR"
fi

echo "=== Production Trajectory Data Collection ==="
echo "Number of tasks: $NUM_TASKS"
echo "Task suite: $TASK_SUITE"
echo "Save directory: $SAVE_DIR"
echo "Trials per task: $NUM_TRIALS_PER_TASK"
echo "GPUs: $NUM_GPUS, Processes: $NUM_PROCESSES"
echo "Reconstruct images: $RECONSTRUCT_IMAGES"
echo "Reconstruct states: $RECONSTRUCT_STATES"
echo ""

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

echo "=== Step 1: Collecting Trajectory Data ==="
echo "Starting data collection for $NUM_TASKS tasks..."

# Calculate max trajectories based on tasks and trials
MAX_TRAJECTORIES=$((NUM_TASKS * NUM_TRIALS_PER_TASK))
echo "Expected total trajectories: $MAX_TRAJECTORIES"

# Run trajectory data collection
python /u/xzhang42/Inspire/vla_scripts/parallel_libero_evaluator.py \
    --pretrained-checkpoint /work/nvme/bfbo/xzhang42/Inspire/runs/minivla-libero-90 \
    --task-suite-name "$TASK_SUITE" \
    --num-gpus "$NUM_GPUS" \
    --num-processes "$NUM_PROCESSES" \
    --num-trails-per-task "$NUM_TRIALS_PER_TASK" \
    --steps 50000 \
    --collect-trajectory-data \
    --trajectory-data-save-path "$SAVE_DIR/trajectory_data" \
    --save-root "$SAVE_DIR/results" || {
    echo "ERROR: Data collection failed"
    exit 1
}

echo "Data collection completed successfully"

echo "=== Step 2: Combining Trajectory Data Files ==="
echo "Combining data files from multiple processes..."

# Combine trajectory data files from multiple processes
python /u/xzhang42/Inspire/vla_scripts/combine_trajectory_files.py \
    --data-dir "$SAVE_DIR/trajectory_data" \
    --task-suite-name "$TASK_SUITE" || {
    echo "ERROR: File combination failed"
    exit 1
}

echo "File combination completed successfully"

# Verify combined file was created
COMBINED_FILE="$SAVE_DIR/trajectory_data/trajectory_data_${TASK_SUITE}.h5"
if [[ ! -f "$COMBINED_FILE" ]]; then
    echo "ERROR: Combined trajectory file not found: $COMBINED_FILE"
    exit 1
fi

echo "✓ Combined trajectory file created: $COMBINED_FILE"

# Check file size and structure
echo "=== Step 3: Verifying Trajectory Data ==="
python -c "
import h5py
from pathlib import Path

file_path = Path('$COMBINED_FILE')
if file_path.exists():
    print(f'File size: {file_path.stat().st_size / (1024*1024*1024):.2f} GB')
    
    with h5py.File(file_path, 'r') as f:
        print(f'Task groups: {len(f.keys())}')
        total_episodes = sum(len(f[task_key].keys()) for task_key in f.keys() if task_key.startswith('task_'))
        print(f'Total episodes: {total_episodes}')
        
        # Sample first episode metadata
        if len(f.keys()) > 0:
            first_task = list(f.keys())[0]
            if len(f[first_task].keys()) > 0:
                first_episode = list(f[first_task].keys())[0]
                metadata_path = f'{first_task}/{first_episode}/metadata'
                if metadata_path in f:
                    metadata = f[metadata_path]
                    print(f'Sample episode - Task: {metadata.attrs.get(\"task_description\", \"N/A\")}')
                    print(f'Sample episode - Success: {metadata.attrs.get(\"success\", \"N/A\")}')
                    print(f'Sample episode - Timesteps: {metadata.attrs.get(\"num_timesteps\", \"N/A\")}')
        print('✓ Trajectory data verification completed')
else:
    print('ERROR: Combined file not found')
    exit(1)
" || {
    echo "ERROR: Data verification failed"
    exit 1
}

# Reconstruction phase
if [[ "$RECONSTRUCT_IMAGES" == "true" || "$RECONSTRUCT_STATES" == "true" ]]; then
    echo "=== Step 4: Reconstructing Images and Simulator States ==="
    
    # Set reconstruction arguments
    RECON_ARGS=("$COMBINED_FILE" "--task-suite-name" "$TASK_SUITE")
    
    if [[ "$RECONSTRUCT_IMAGES" == "true" ]]; then
        IMAGES_DIR="$SAVE_DIR/reconstructed_images"
        mkdir -p "$IMAGES_DIR"
        RECON_ARGS+=("--images-output-dir" "$IMAGES_DIR")
        echo "Images will be saved to: $IMAGES_DIR"
    fi
    
    if [[ "$RECONSTRUCT_STATES" == "true" ]]; then
        STATES_DIR="$SAVE_DIR/reconstructed_states"
        mkdir -p "$STATES_DIR"
        RECON_ARGS+=("--states-output-dir" "$STATES_DIR")
        echo "States will be saved to: $STATES_DIR"
    fi
    
    echo "Starting reconstruction with stored actions..."
    python /u/xzhang42/Inspire/vla_scripts/reconstruct_trajectory_data.py "${RECON_ARGS[@]}" || {
        echo "ERROR: Reconstruction failed"
        exit 1
    }
    
    echo "Reconstruction completed successfully"
    
    # Verify reconstruction results
    if [[ "$RECONSTRUCT_IMAGES" == "true" ]]; then
        IMAGE_COUNT=$(find "$IMAGES_DIR" -name "*.png" | wc -l)
        echo "✓ Images reconstructed: $IMAGE_COUNT"
    fi
    
    if [[ "$RECONSTRUCT_STATES" == "true" ]]; then
        STATE_COUNT=$(find "$STATES_DIR" -name "*.json" | wc -l)
        echo "✓ Simulator states saved: $STATE_COUNT"
    fi
else
    echo "=== Step 4: Skipping Reconstruction (disabled by user) ==="
fi

echo "=== Step 5: Final Summary ==="
echo "✅ Trajectory data collection pipeline completed successfully!"
echo ""
echo "Data location: $SAVE_DIR"
echo "├── trajectory_data/"
echo "│   └── trajectory_data_${TASK_SUITE}.h5  # Main HDF5 dataset"
echo "├── results/                              # Evaluation results"

if [[ "$RECONSTRUCT_IMAGES" == "true" ]]; then
    echo "├── reconstructed_images/                 # Reconstructed images using stored actions"
fi

if [[ "$RECONSTRUCT_STATES" == "true" ]]; then
    echo "└── reconstructed_states/                 # Simulator states (JSON files)"
fi

echo ""
echo "Ready for linear probe training!"

# Final data summary
python -c "
import h5py
from pathlib import Path

file_path = Path('$COMBINED_FILE')
with h5py.File(file_path, 'r') as f:
    total_episodes = sum(len(f[task_key].keys()) for task_key in f.keys() if task_key.startswith('task_'))
    successful_episodes = 0
    total_timesteps = 0
    
    for task_key in f.keys():
        if task_key.startswith('task_'):
            for episode_key in f[task_key].keys():
                if episode_key.startswith('episode_'):
                    metadata_path = f'{task_key}/{episode_key}/metadata'
                    if metadata_path in f:
                        metadata = f[metadata_path]
                        if metadata.attrs.get('success', False):
                            successful_episodes += 1
                        total_timesteps += metadata.attrs.get('num_timesteps', 0)
    
    print(f'📊 Final Statistics:')
    print(f'   Total episodes: {total_episodes}')
    print(f'   Successful episodes: {successful_episodes} ({successful_episodes/total_episodes*100:.1f}%)')
    print(f'   Total timesteps: {total_timesteps}')
    print(f'   Average timesteps per episode: {total_timesteps/total_episodes:.1f}')
"

echo "=== Pipeline Complete ==="