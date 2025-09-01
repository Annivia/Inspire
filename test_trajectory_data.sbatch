#!/bin/bash
#SBATCH --job-name=TEST_TRAJ_DATA
#SBATCH --output=results/test-trajectory-data-%J.out
#SBATCH --error=results/test-trajectory-data-%J.err
#SBATCH --cpus-per-task=12
#SBATCH --time=0:30:00  # 30 minutes should be enough for 8 trajectories
#SBATCH --account=bfbo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1

module load cuda/12.6.1
source ~/.bashrc
conda activate inspire-safe

NUM_GPUS=$SLURM_GPUS_ON_NODE
echo "Number of GPUs allocated: $NUM_GPUS"

python - <<'EOF'
import torch
print("Is CUDA available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
EOF

export PYTHONPATH=/u/xzhang42/Inspire/LIBERO:$PYTHONPATH
export PYTHONPATH=/u/xzhang42/Inspire/vq_bet_official:$PYTHONPATH
export PYTHONPATH=/u/xzhang42/Inspire:$PYTHONPATH

echo "Transferring data..."
rsync -av /projects/bfbo/xzhang42/Inspire /work/nvme/bfbo/xzhang42/
echo "Data Transfer Completed..."

export PRISMATIC_DATA_ROOT=/work/nvme/bfbo/xzhang42/Inspire 
export HF_HOME=/work/nvme/bfbo/xzhang42/huggingface
echo "[HF_CACHE] Hugging Face cache directory is set to: $HF_HOME"

echo "=== TESTING TRAJECTORY DATA COLLECTION ==="
echo "Starting trajectory data collection test with early stopping..."

# Run trajectory data collection test
python /u/xzhang42/Inspire/vla_scripts/parallel_libero_evaluator.py \
    --pretrained-checkpoint /work/nvme/bfbo/xzhang42/Inspire/runs/minivla-libero-90 \
    --task-suite-name libero_90 \
    --num-gpus 4 \
    --num-processes 4 \
    --num-trails-per-task 10 \
    --steps 50000 \
    --collect-trajectory-data \
    --trajectory-data-save-path /work/nvme/bfbo/xzhang42/Inspire/test_trajectory_data \
    --max-total-trajectories 8 \
    --save-root /work/nvme/bfbo/xzhang42/Inspire/test_results

echo "Evaluation completed. Combining trajectory data files..."

# Combine trajectory data files from multiple processes
python /u/xzhang42/Inspire/vla_scripts/combine_trajectory_files.py \
    --data-dir /work/nvme/bfbo/xzhang42/Inspire/test_trajectory_data \
    --task-suite-name libero_90

echo "Checking combined trajectory data file..."

# Check if trajectory data file was created and show its structure
python - <<'EOF'
import os
import h5py
from pathlib import Path

trajectory_file = Path("/work/nvme/bfbo/xzhang42/Inspire/test_trajectory_data/trajectory_data_libero_90.h5")

if trajectory_file.exists():
    print(f"✅ SUCCESS: Trajectory data file created: {trajectory_file}")
    print(f"File size: {trajectory_file.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        with h5py.File(trajectory_file, 'r') as f:
            print(f"\n📁 HDF5 File Structure:")
            
            def print_structure(name, obj, indent=0):
                spaces = "  " * indent
                if hasattr(obj, 'shape'):
                    print(f"{spaces}{name}: dataset {obj.shape} {obj.dtype}")
                elif hasattr(obj, 'keys'):
                    print(f"{spaces}{name}: group with {len(obj.keys())} items")
                    if indent < 3:  # Limit depth to avoid too much output
                        for key in list(obj.keys())[:3]:  # Show first 3 items
                            print_structure(key, obj[key], indent + 1)
                        if len(obj.keys()) > 3:
                            print(f"{spaces}  ... and {len(obj.keys())-3} more items")
                else:
                    print(f"{spaces}{name}: {type(obj)}")
            
            print(f"Root level: {len(f.keys())} task groups")
            for key in list(f.keys())[:2]:  # Show first 2 tasks
                print_structure(key, f[key])
            if len(f.keys()) > 2:
                print(f"  ... and {len(f.keys())-2} more task groups")
                
            # Print some metadata from first episode
            first_task = list(f.keys())[0]
            first_episode = list(f[first_task].keys())[0]
            metadata_path = f"{first_task}/{first_episode}/metadata"
            if metadata_path in f:
                metadata = f[metadata_path]
                print(f"\n📋 Sample Metadata from {first_task}/{first_episode}:")
                for attr_name in metadata.attrs.keys():
                    print(f"  {attr_name}: {metadata.attrs[attr_name]}")
                    
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"❌ FAILURE: No trajectory data file found at {trajectory_file}")
    
    # Check if directory exists
    data_dir = trajectory_file.parent
    if data_dir.exists():
        print(f"Data directory exists: {data_dir}")
        print(f"Contents: {list(data_dir.glob('*'))}")
    else:
        print(f"Data directory does not exist: {data_dir}")
EOF

echo "Transferring results and trajectory data back..."
rsync -av /work/nvme/bfbo/xzhang42/Inspire/test_results /projects/bfbo/xzhang42/Inspire/
rsync -av /work/nvme/bfbo/xzhang42/Inspire/test_trajectory_data /projects/bfbo/xzhang42/Inspire/
echo "Transfer completed."

echo "=== TEST COMPLETED ==="