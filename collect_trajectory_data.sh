#!/bin/bash
set -e

# Full dataset collection settings (edit here; no CLI args required)
NUM_TASKS=90
SAVE_DIR="/work/nvme/bfbo/xzhang42/data/full_run"
TASK_SUITE="libero_90"
NUM_TRIALS_PER_TASK=10
NUM_GPUS=4
NUM_PROCESSES=32
RECONSTRUCT_IMAGES=false
RECONSTRUCT_STATES=false

# Env
export PYTHONPATH=/u/xzhang42/Inspire/LIBERO:$PYTHONPATH
export PYTHONPATH=/u/xzhang42/Inspire/vq_bet_official:$PYTHONPATH
export PYTHONPATH=/u/xzhang42/Inspire:$PYTHONPATH
export PRISMATIC_DATA_ROOT=/work/nvme/bfbo/xzhang42/Inspire
export HF_HOME=/work/nvme/bfbo/xzhang42/huggingface

mkdir -p "$SAVE_DIR" "$SAVE_DIR/results" "$SAVE_DIR/trajectory_data"

echo "=== Step 1 ==="
python /u/xzhang42/Inspire/vla_scripts/parallel_libero_evaluator.py \
  --pretrained-checkpoint /work/nvme/bfbo/xzhang42/Inspire/runs/minivla-libero-90 \
  --task-suite-name "$TASK_SUITE" \
  --num-gpus "$NUM_GPUS" \
  --num-processes "$NUM_PROCESSES" \
  --num-trails-per-task "$NUM_TRIALS_PER_TASK" \
  --steps 50000 \
  --collect-trajectory-data \
  --trajectory-data-save-path "$SAVE_DIR/trajectory_data" \
  --save-root "$SAVE_DIR/results"

echo "Concept CSVs saved under: $SAVE_DIR/trajectory_data/concepts/"

echo "=== Step 2 ==="
python /u/xzhang42/Inspire/vla_scripts/combine_optimized_trajectory_files.py \
  --temp-dir "$SAVE_DIR/trajectory_data/temp_trajectory_processing" \
  --output-dir "$SAVE_DIR/optimized_trajectory_data" \
  --task-suite "$TASK_SUITE" \
  --cleanup-temp

echo "=== Step 3 ==="
python -c "
import sys, json
from pathlib import Path
sys.path.append('/u/xzhang42/Inspire')
from vla_scripts.load_optimized_trajectory_data import OptimizedTrajectoryLoader
p = Path('$SAVE_DIR/optimized_trajectory_data')
loader = OptimizedTrajectoryLoader(p)
print(json.dumps(loader.get_dataset_info(), indent=2))
total = sum(f.stat().st_size for f in p.rglob('*.h5'))
print(f'Total size GB: {total/1024/1024/1024:.2f}')
"

echo "=== Step 4 ==="
if [[ "$RECONSTRUCT_IMAGES" == "true" || "$RECONSTRUCT_STATES" == "true" ]]; then
  RECON_ARGS=("$COMBINED_FILE" "--task-suite-name" "$TASK_SUITE")
  if [[ "$RECONSTRUCT_IMAGES" == "true" ]]; then
    IMAGES_DIR="$SAVE_DIR/reconstructed_images"; mkdir -p "$IMAGES_DIR"
    RECON_ARGS+=("--images-output-dir" "$IMAGES_DIR")
  fi
  if [[ "$RECONSTRUCT_STATES" == "true" ]]; then
    STATES_DIR="$SAVE_DIR/reconstructed_states"; mkdir -p "$STATES_DIR"
    RECON_ARGS+=("--states-output-dir" "$STATES_DIR")
  fi
  python /u/xzhang42/Inspire/vla_scripts/reconstruct_trajectory_data.py "${RECON_ARGS[@]}"
else
  echo "reconstruction skipped"
fi
