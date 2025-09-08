#!/usr/bin/env python3
"""
combine_optimized_trajectory_files.py

Combine temporary chunk files from parallel processing into final optimized multi-file format.
This script should be run after all parallel processes have completed data collection.
"""

import argparse
import sys
from pathlib import Path
import json
import time

# Add project root to path
sys.path.append('/u/xzhang42/Inspire')

from vla_scripts.trajectory_data_collector_optimized import combine_chunks_to_optimized_format


def main():
    parser = argparse.ArgumentParser(description='Combine trajectory chunk files to optimized format')
    parser.add_argument('--temp-dir', required=True, 
                       help='Directory containing gpu_process_* temporary chunk directories')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for final optimized multi-file format')
    parser.add_argument('--task-suite', default='libero_90',
                       help='Task suite name (default: libero_90)')
    parser.add_argument('--cleanup-temp', action='store_true',
                       help='Remove temporary files after successful combination')
    
    args = parser.parse_args()
    
    temp_dir = Path(args.temp_dir)
    output_dir = Path(args.output_dir)
    
    print(f"=== Trajectory Data Combination ===")
    print(f"Temp directory: {temp_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Task suite: {args.task_suite}")
    print(f"Cleanup temp files: {args.cleanup_temp}")
    
    # Validate temp directory exists
    if not temp_dir.exists():
        print(f"ERROR: Temp directory not found: {temp_dir}")
        return 1
    
    # Find process directories
    process_dirs = [d for d in temp_dir.iterdir() if d.is_dir() and d.name.startswith('gpu_process_')]
    print(f"Found {len(process_dirs)} process directories")
    
    if len(process_dirs) == 0:
        print(f"ERROR: No gpu_process_* directories found in {temp_dir}")
        return 1
    
    # Check that all processes have completed (manifest files exist)
    incomplete_processes = []
    for process_dir in process_dirs:
        manifest_path = process_dir / "chunk_manifest.json"
        if not manifest_path.exists():
            incomplete_processes.append(process_dir.name)
    
    if incomplete_processes:
        print(f"WARNING: The following processes appear incomplete (no manifest file):")
        for process_name in incomplete_processes:
            print(f"  - {process_name}")
        print("Continuing anyway (auto-continue enabled)...")
    
    try:
        # Run the combination
        start_time = time.time()
        summary = combine_chunks_to_optimized_format(
            temp_processing_dir=str(temp_dir),
            output_dir=str(output_dir),
            task_suite_name=args.task_suite
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n=== Combination Complete ===")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Successful episodes: {summary['successful_episodes']}")
        print(f"Layer indices: {summary['layer_indices']}")
        print(f"Output directory: {output_dir}")
        
        # List generated files
        print(f"\nGenerated files:")
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {file_path.relative_to(output_dir)}: {size_mb:.1f} MB")
        
        # Cleanup temp files if requested
        if args.cleanup_temp:
            print(f"\nCleaning up temporary files...")
            import shutil
            
            for process_dir in process_dirs:
                if process_dir.exists():
                    shutil.rmtree(process_dir)
                    print(f"  Removed: {process_dir}")
            
            # Remove empty temp directory if it only contained process dirs
            try:
                temp_dir.rmdir()
                print(f"  Removed empty temp directory: {temp_dir}")
            except OSError:
                print(f"  Temp directory not empty, keeping: {temp_dir}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR during combination: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())