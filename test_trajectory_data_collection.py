#!/usr/bin/env python3
"""
Test script for trajectory data collection with early stopping.
Only runs on 8 trajectories to verify hidden states are correctly stored.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("=" * 60)
    print("TESTING TRAJECTORY DATA COLLECTION")
    print("=" * 60)
    
    # Configuration
    pretrained_checkpoint = "/work/nvme/bfbo/xzhang42/Inspire/runs/minivla-libero-90"
    task_suite_name = "libero_90"
    max_trajectories = 8
    trajectory_data_path = "./test_trajectory_data"
    
    print(f"Pretrained checkpoint: {pretrained_checkpoint}")
    print(f"Task suite: {task_suite_name}")
    print(f"Max trajectories: {max_trajectories}")
    print(f"Data save path: {trajectory_data_path}")
    print()
    
    # Check if checkpoint exists
    if not os.path.exists(pretrained_checkpoint):
        print(f"ERROR: Checkpoint path does not exist: {pretrained_checkpoint}")
        return 1
    
    # Create command
    cmd = [
        "python", "/u/xzhang42/Inspire/vla_scripts/parallel_libero_evaluator.py",
        "--pretrained-checkpoint", pretrained_checkpoint,
        "--task-suite-name", task_suite_name,
        "--num-gpus", "1",  # Use only 1 GPU for testing
        "--num-processes", "1",  # Use only 1 process for testing
        "--num-trails-per-task", "10",  # Will be limited by max_total_trajectories
        "--steps", "50000",
        "--collect-trajectory-data",  # Enable data collection
        "--trajectory-data-save-path", trajectory_data_path,
        "--max-total-trajectories", str(max_trajectories),  # Early stopping
        "--save-root", "./test_results"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run the evaluation
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        # Check if trajectory data was created
        trajectory_file = Path(trajectory_data_path) / f"trajectory_data_{task_suite_name}.h5"
        if trajectory_file.exists():
            print(f"\n✅ SUCCESS: Trajectory data file created: {trajectory_file}")
            
            # Try to read the file and show structure
            try:
                import h5py
                with h5py.File(trajectory_file, 'r') as f:
                    print(f"\n📁 HDF5 File Structure:")
                    def print_structure(name, obj, indent=0):
                        spaces = "  " * indent
                        if hasattr(obj, 'shape'):
                            print(f"{spaces}{name}: dataset {obj.shape} {obj.dtype}")
                        elif hasattr(obj, 'keys'):
                            print(f"{spaces}{name}: group")
                            for key in obj.keys():
                                print_structure(key, obj[key], indent + 1)
                        else:
                            print(f"{spaces}{name}: {type(obj)}")
                    
                    for key in f.keys():
                        print_structure(key, f[key])
            except ImportError:
                print("h5py not available, cannot read file structure")
            except Exception as e:
                print(f"Error reading HDF5 file: {e}")
        else:
            print(f"\n❌ FAILURE: No trajectory data file found at {trajectory_file}")
            
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("❌ FAILURE: Command timed out after 30 minutes")
        return 1
    except Exception as e:
        print(f"❌ FAILURE: Error running command: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)