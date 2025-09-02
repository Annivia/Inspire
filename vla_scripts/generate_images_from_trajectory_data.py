#!/usr/bin/env python3
"""
Generate images from trajectory data using minimal reconstruction clues.

This script reads a combined trajectory dataset HDF5 file and regenerates all images
using LIBERO environment reconstruction. Images are saved to the fast memory location.
"""

import os
os.environ["MUJOCO_GL"] = "egl" 
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

import argparse
import h5py
import numpy as np
from pathlib import Path
from PIL import Image
import sys
sys.path.append('/u/xzhang42/Inspire')
sys.path.append('/u/xzhang42/Inspire/LIBERO')
sys.path.append('/u/xzhang42/Inspire/vq_bet_official')

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action, get_libero_image


def generate_images_from_dataset(dataset_path: str, output_dir: str, task_suite_name: str = "libero_90"):
    """
    Generate all images from trajectory dataset using minimal reconstruction clues.
    
    Args:
        dataset_path: Path to combined trajectory data HDF5 file
        output_dir: Directory to save generated images (should be in /work/nvme/...)  
        task_suite_name: LIBERO task suite name
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    
    print(f"[debug-image] Loading dataset from: {dataset_path}")
    print(f"[debug-image] Output directory: {output_dir}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LIBERO benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    print(f"[debug-image] Initialized LIBERO task suite: {task_suite_name}")
    
    total_images_generated = 0
    
    with h5py.File(dataset_path, 'r') as f:
        print(f"[debug-image] Dataset contains {len(f.keys())} task groups")
        
        for task_group_name in f.keys():
            if not task_group_name.startswith('task_'):
                continue
                
            print(f"[debug-image] Processing {task_group_name}...")
            task_group = f[task_group_name]
            
            for episode_group_name in task_group.keys():
                if not episode_group_name.startswith('episode_'):
                    continue
                    
                print(f"[debug-image] Processing {task_group_name}/{episode_group_name}...")
                episode_group = task_group[episode_group_name]
                
                # Extract image reconstruction clues from metadata
                metadata = episode_group['metadata']
                
                if 'img_task_id' not in metadata.attrs:
                    print(f"[debug-image] WARNING: No image reconstruction clues in {task_group_name}/{episode_group_name}")
                    continue
                
                img_task_id = int(metadata.attrs['img_task_id'])
                img_episode_id = int(metadata.attrs['img_episode_id']) 
                img_env_seed = int(metadata.attrs['img_env_seed'])
                num_timesteps = int(metadata.attrs['num_timesteps'])
                
                print(f"[debug-image] Image reconstruction clues - task:{img_task_id}, episode:{img_episode_id}, seed:{img_env_seed}")
                print(f"[debug-image] Need to generate {num_timesteps} images")
                
                try:
                    # Initialize LIBERO environment with reconstruction clues
                    task = task_suite.get_task(img_task_id)
                    env, task_description = get_libero_env(task, "prismatic", resolution=224)
                    env.seed(img_env_seed)
                    env.reset()
                    
                    # Set initial state if not libero_object
                    if task_suite_name != 'libero_object':
                        initial_states = task_suite.get_task_init_states(img_task_id)
                        env.set_init_state(initial_states[img_episode_id])
                    
                    print(f"[debug-image] Environment initialized for task: {task_description}")
                    
                    # Create episode output directory
                    episode_output_dir = output_dir / task_group_name / episode_group_name
                    episode_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate images for each timestep
                    for timestep in range(num_timesteps):
                        if timestep > 0:
                            # Step environment forward (use dummy action)
                            obs, reward, done, info = env.step(get_libero_dummy_action("prismatic"))
                        else:
                            # First timestep - get initial observation from reset
                            obs = env.reset()
                        
                        # Extract and process image
                        img_array = get_libero_image(obs, 224, key="agentview_image")
                        img = Image.fromarray(img_array.astype(np.uint8))
                        
                        # Save image
                        img_path = episode_output_dir / f"timestep_{timestep:04d}.png"
                        img.save(img_path)
                        
                        if timestep % 10 == 0:
                            print(f"[debug-image] Generated image {timestep}/{num_timesteps}: {img_path}")
                    
                    env.close()
                    total_images_generated += num_timesteps
                    print(f"[debug-image] Generated {num_timesteps} images for {task_group_name}/{episode_group_name}")
                    
                except Exception as e:
                    print(f"[debug-image] ERROR generating images for {task_group_name}/{episode_group_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    if 'env' in locals():
                        env.close()
                    continue
    
    print(f"[debug-image] ===== IMAGE GENERATION COMPLETE =====")
    print(f"[debug-image] Total images generated: {total_images_generated}")
    print(f"[debug-image] Images saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate images from trajectory dataset')
    parser.add_argument('dataset_path', help='Path to combined trajectory data HDF5 file')
    parser.add_argument('--output-dir', required=True, help='Output directory for images (should be in /work/nvme/...)')
    parser.add_argument('--task-suite-name', default='libero_90', help='LIBERO task suite name')
    
    args = parser.parse_args()
    
    # Validate that output is in fast memory
    if '/work/nvme/' not in str(args.output_dir):
        print(f"[debug-image] WARNING: Output directory should be in /work/nvme/ for fast storage")
        print(f"[debug-image] Current output: {args.output_dir}")
    
    generate_images_from_dataset(args.dataset_path, args.output_dir, args.task_suite_name)


if __name__ == "__main__":
    main()