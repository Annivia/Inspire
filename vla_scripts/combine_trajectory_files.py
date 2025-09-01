#!/usr/bin/env python3
"""
Combine separate trajectory data files from multiple processes into a single HDF5 file.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse


def combine_trajectory_files(data_dir: str, task_suite_name: str, output_file: str = None):
    """
    Combine trajectory data files from multiple processes into a single file.
    
    Args:
        data_dir: Directory containing trajectory data files
        task_suite_name: Task suite name (e.g., 'libero_90')
        output_file: Optional output file name
    """
    data_path = Path(data_dir)
    
    # Find all process files
    process_files = list(data_path.glob(f"trajectory_data_{task_suite_name}_proc_*.h5"))
    
    if not process_files:
        print(f"No trajectory data files found in {data_path}")
        return
    
    print(f"Found {len(process_files)} trajectory data files to combine")
    for f in process_files:
        print(f"  - {f}")
    
    # Output file
    if output_file is None:
        output_file = data_path / f"trajectory_data_{task_suite_name}.h5"
    else:
        output_file = Path(output_file)
    
    print(f"Combining into: {output_file}")
    
    # Combine files
    with h5py.File(output_file, 'w') as output_h5:
        # Copy attributes from first file
        with h5py.File(process_files[0], 'r') as first_file:
            for attr_name in first_file.attrs.keys():
                output_h5.attrs[attr_name] = first_file.attrs[attr_name]
        
        # Combine data from all files
        for proc_file in process_files:
            print(f"Processing {proc_file.name}...")
            
            with h5py.File(proc_file, 'r') as proc_h5:
                def copy_item(name, obj):
                    if name not in output_h5:
                        if hasattr(obj, 'keys'):  # Group
                            output_h5.create_group(name)
                            # Copy attributes
                            for attr_name in obj.attrs.keys():
                                output_h5[name].attrs[attr_name] = obj.attrs[attr_name]
                        else:  # Dataset
                            output_h5.create_dataset(name, data=obj[:])
                            # Copy attributes
                            for attr_name in obj.attrs.keys():
                                output_h5[name].attrs[attr_name] = obj.attrs[attr_name]
                    else:
                        print(f"  Warning: {name} already exists, skipping")
                
                proc_h5.visititems(copy_item)
    
    print(f"Successfully combined {len(process_files)} files into {output_file}")
    
    # Show final structure
    with h5py.File(output_file, 'r') as f:
        print(f"\nFinal structure:")
        print(f"Root level: {len(f.keys())} task groups")
        
        total_episodes = 0
        for task_key in f.keys():
            task_group = f[task_key]
            num_episodes = len(task_group.keys())
            total_episodes += num_episodes
            print(f"  {task_key}: {num_episodes} episodes")
        
        print(f"Total episodes: {total_episodes}")
        print(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Optionally remove individual process files
    print(f"\nKeeping individual process files for debugging")
    # for proc_file in process_files:
    #     proc_file.unlink()
    #     print(f"Removed {proc_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine trajectory data files from multiple processes")
    parser.add_argument("--data-dir", required=True, help="Directory containing trajectory data files")
    parser.add_argument("--task-suite-name", required=True, help="Task suite name (e.g., libero_90)")
    parser.add_argument("--output-file", help="Optional output file name")
    
    args = parser.parse_args()
    
    combine_trajectory_files(args.data_dir, args.task_suite_name, args.output_file)