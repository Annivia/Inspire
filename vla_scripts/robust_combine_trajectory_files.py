#!/usr/bin/env python3
"""
robust_combine_trajectory_files.py

Robust trajectory data file combiner that filters out corrupted files
and only merges valid HDF5 files. Designed to handle incomplete data
collection runs and file corruption issues.

Usage: python robust_combine_trajectory_files.py
(All parameters are hardcoded for ease of use)
"""

import h5py
import numpy as np
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple


# HARDCODED CONFIGURATION
DATA_DIR = "/work/hdd/bfbo/trajectory_data"
TASK_SUITE_NAME = "libero_90"
OUTPUT_FILE = "/work/hdd/bfbo/trajectory_data/trajectory_data_libero_90_robust.h5"
DEBUG = True

# Color codes for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def print_info(msg): print(f"{BLUE}[INFO]{NC} {msg}")
def print_success(msg): print(f"{GREEN}[SUCCESS]{NC} {msg}")
def print_warning(msg): print(f"{YELLOW}[WARNING]{NC} {msg}")
def print_error(msg): print(f"{RED}[ERROR]{NC} {msg}")
def print_debug(msg): 
    if DEBUG: print(f"{BLUE}[DEBUG]{NC} {msg}")


def check_hdf5_file_validity(file_path: Path) -> Tuple[bool, str, Dict]:
    """
    Check if an HDF5 file is valid and get basic info about it.
    
    Args:
        file_path: Path to HDF5 file to check
        
    Returns:
        Tuple of (is_valid, error_message, file_info)
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Try to access basic structure
            root_keys = list(f.keys())
            
            # Count tasks and episodes
            task_count = 0
            episode_count = 0
            
            for key in root_keys:
                if key.startswith('task_'):
                    task_count += 1
                    task_group = f[key]
                    for ep_key in task_group.keys():
                        if ep_key.startswith('episode_'):
                            episode_count += 1
            
            file_info = {
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'root_keys': len(root_keys),
                'task_count': task_count,
                'episode_count': episode_count,
                'created_at': f.attrs.get('created_at', 'unknown'),
                'task_suite': f.attrs.get('task_suite', 'unknown')
            }
            
            return True, "", file_info
            
    except Exception as e:
        return False, str(e), {}


def extract_data_from_valid_file(file_path: Path, output_h5: h5py.File) -> Tuple[int, int]:
    """
    Extract data from a valid HDF5 file and add it to the output file.
    
    Args:
        file_path: Path to valid input file
        output_h5: Output HDF5 file handle
        
    Returns:
        Tuple of (tasks_added, episodes_added)
    """
    tasks_added = 0
    episodes_added = 0
    
    try:
        with h5py.File(file_path, 'r') as input_h5:
            print_debug(f"Extracting data from {file_path.name}...")
            
            # Copy attributes from first file if output is empty
            if not output_h5.attrs:
                for attr_name in input_h5.attrs.keys():
                    output_h5.attrs[attr_name] = input_h5.attrs[attr_name]
                print_debug(f"Copied global attributes from {file_path.name}")
            
            # Process each task
            for task_key in input_h5.keys():
                if not task_key.startswith('task_'):
                    continue
                    
                task_group = input_h5[task_key]
                
                # Create task group in output if it doesn't exist
                if task_key not in output_h5:
                    output_task_group = output_h5.create_group(task_key)
                    tasks_added += 1
                    print_debug(f"Created task group: {task_key}")
                else:
                    output_task_group = output_h5[task_key]
                
                # Process each episode in this task
                for episode_key in task_group.keys():
                    if not episode_key.startswith('episode_'):
                        continue
                    
                    full_episode_path = f"{task_key}/{episode_key}"
                    
                    # Skip if episode already exists
                    if episode_key in output_task_group:
                        print_debug(f"Episode {full_episode_path} already exists, skipping")
                        continue
                    
                    try:
                        # Copy entire episode group
                        input_h5.copy(full_episode_path, output_task_group)
                        episodes_added += 1
                        print_debug(f"Copied episode: {full_episode_path}")
                        
                    except Exception as e:
                        print_error(f"Failed to copy episode {full_episode_path}: {e}")
                        continue
                        
    except Exception as e:
        print_error(f"Error processing file {file_path}: {e}")
        return 0, 0
    
    return tasks_added, episodes_added


def main():
    print_info("=== Robust Trajectory Data File Combiner ===")
    print_info(f"Data directory: {DATA_DIR}")
    print_info(f"Task suite: {TASK_SUITE_NAME}")
    print_info(f"Output file: {OUTPUT_FILE}")
    print_info("")
    
    start_time = time.time()
    data_path = Path(DATA_DIR)
    
    if not data_path.exists():
        print_error(f"Data directory not found: {DATA_DIR}")
        return 1
    
    # Find all process files
    pattern = f"trajectory_data_{TASK_SUITE_NAME}_proc_*.h5"
    process_files = list(data_path.glob(pattern))
    
    if not process_files:
        print_error(f"No trajectory data files found matching pattern: {pattern}")
        return 1
    
    print_info(f"Found {len(process_files)} potential trajectory data files")
    
    # Check validity of each file
    valid_files = []
    invalid_files = []
    file_stats = []
    
    print_info("Checking file validity...")
    for file_path in process_files:
        print_debug(f"Checking {file_path.name}...")
        
        is_valid, error_msg, file_info = check_hdf5_file_validity(file_path)
        
        if is_valid:
            valid_files.append(file_path)
            file_stats.append(file_info)
            print_debug(f"✓ {file_path.name}: {file_info['episode_count']} episodes, {file_info['size_mb']:.1f} MB")
        else:
            invalid_files.append((file_path, error_msg))
            print_warning(f"✗ {file_path.name}: {error_msg}")
    
    print_info(f"File validation complete:")
    print_success(f"  ✓ Valid files: {len(valid_files)}")
    if invalid_files:
        print_warning(f"  ✗ Invalid files: {len(invalid_files)}")
        for file_path, error in invalid_files:
            print_warning(f"    - {file_path.name}: {error}")
    
    if not valid_files:
        print_error("No valid files found to combine!")
        return 1
    
    # Create output directory
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine valid files
    print_info(f"Combining {len(valid_files)} valid files into {OUTPUT_FILE}")
    
    total_tasks_added = 0
    total_episodes_added = 0
    
    try:
        with h5py.File(OUTPUT_FILE, 'w') as output_h5:
            for i, file_path in enumerate(valid_files):
                print_info(f"Processing file {i+1}/{len(valid_files)}: {file_path.name}")
                
                tasks_added, episodes_added = extract_data_from_valid_file(file_path, output_h5)
                total_tasks_added += tasks_added
                total_episodes_added += episodes_added
                
                print_debug(f"  Added {episodes_added} episodes from {file_path.name}")
    
    except Exception as e:
        print_error(f"Error creating combined file: {e}")
        return 1
    
    # Verify the output file
    print_info("Verifying output file...")
    is_valid, error_msg, final_info = check_hdf5_file_validity(Path(OUTPUT_FILE))
    
    if not is_valid:
        print_error(f"Output file verification failed: {error_msg}")
        return 1
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print_info("")
    print_success("=== Combining Complete ===")
    print_info(f"Processing time: {elapsed_time:.1f} seconds")
    print_info(f"Valid input files: {len(valid_files)}")
    print_info(f"Invalid input files: {len(invalid_files)}")
    print_info(f"Total episodes combined: {total_episodes_added}")
    print_info(f"Output file: {OUTPUT_FILE}")
    print_info(f"Output file size: {final_info['size_mb']:.1f} MB")
    print_info(f"Final structure: {final_info['task_count']} tasks, {final_info['episode_count']} episodes")
    
    # Save processing log
    log_file = output_path.parent / "combine_log.json"
    log_data = {
        'timestamp': time.time(),
        'processing_time': elapsed_time,
        'input_files_found': len(process_files),
        'valid_files': len(valid_files),
        'invalid_files': len(invalid_files),
        'invalid_file_details': [(str(f), err) for f, err in invalid_files],
        'episodes_combined': total_episodes_added,
        'output_file': str(OUTPUT_FILE),
        'output_file_info': final_info
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)
    
    print_info(f"Processing log saved to: {log_file}")
    
    # Test loading the combined file
    print_info("Testing final file with trajectory data loader...")
    try:
        # Add path for imports
        import sys
        sys.path.append('/u/xzhang42/Inspire/vla_scripts')
        
        from vla_scripts.legacy.load_trajectory_data import load_trajectory_dataset
        
        dataset = load_trajectory_dataset(OUTPUT_FILE)
        print_success(f"✓ File loads successfully with trajectory loader!")
        print_info(f"  Loaded episodes: {dataset['summary']['loaded_episodes']}")
        print_info(f"  Available layers: {dataset['summary']['layers'][:5]}... ({len(dataset['summary']['layers'])} total)")
        
    except Exception as e:
        print_warning(f"Trajectory loader test failed: {e}")
        print_warning("File was created but may have compatibility issues")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)