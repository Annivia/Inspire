#!/usr/bin/env python3
"""
Reconstruct trajectory data using stored actions from HDF5 files.

This script replays VLA trajectories using the exact actions that were stored during evaluation,
allowing perfect reconstruction of both images and simulator states that correspond to the
collected hidden states and vision features.
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
import json
sys.path.append('/u/xzhang42/Inspire')
sys.path.append('/u/xzhang42/Inspire/LIBERO')
sys.path.append('/u/xzhang42/Inspire/vq_bet_official')

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image


def extract_simulator_state(env):
    """
    Extract comprehensive simulator state information.
    
    Returns:
        Dict containing robot state, object positions, contact info, etc.
    """
    sim_state = {}
    
    try:
        # Get MuJoCo simulation data
        sim = env.sim
        
        # Robot joint positions and velocities
        sim_state['robot_joint_pos'] = sim.data.qpos[:7].copy()
        sim_state['robot_joint_vel'] = sim.data.qvel[:7].copy() 
        
        # End-effector position and orientation
        ee_pos = sim.data.site_xpos[sim.model.site_name2id('gripper0_grip_site')].copy()
        ee_quat = sim.data.get_body_xquat('gripper0_eef').copy()  
        sim_state['ee_pos'] = ee_pos
        sim_state['ee_quat'] = ee_quat
        
        # All object positions and orientations
        sim_state['object_positions'] = {}
        sim_state['object_orientations'] = {}
        
        for body_id in range(sim.model.nbody):
            body_name = sim.model.body_id2name(body_id)
            if body_name and not body_name.startswith('robot0'):  # Skip robot bodies
                pos = sim.data.body_xpos[body_id].copy()
                quat = sim.data.body_xquat[body_id].copy()
                sim_state['object_positions'][body_name] = pos
                sim_state['object_orientations'][body_name] = quat
        
        # Contact information
        contacts = []
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            contact_info = {
                'geom1': contact.geom1,
                'geom2': contact.geom2, 
                'pos': contact.pos.copy(),
                'frame': contact.frame.copy(),
                'dist': contact.dist
            }
            contacts.append(contact_info)
        sim_state['contacts'] = contacts
        
        # Time and physics info
        sim_state['time'] = sim.data.time
        sim_state['energy'] = sim.data.energy.copy() if hasattr(sim.data, 'energy') else None
        
    except Exception as e:
        print(f"[debug-state] WARNING: Could not extract full simulator state: {e}")
        # Fallback to basic state
        sim_state['error'] = str(e)
        sim_state['time'] = getattr(env.sim.data, 'time', 0.0)
        
    return sim_state


def reconstruct_trajectory_episode(
    dataset_path: str,
    task_id: int, 
    episode_id: int,
    task_suite_name: str = "libero_90",
    images_output_dir: str = None,
    states_output_dir: str = None
):
    """
    Reconstruct a single episode trajectory using stored actions.
    
    Args:
        dataset_path: Path to HDF5 trajectory data file
        task_id: Task ID to reconstruct
        episode_id: Episode ID to reconstruct  
        task_suite_name: LIBERO task suite name
        images_output_dir: Directory to save reconstructed images (if None, skip images)
        states_output_dir: Directory to save simulator states (if None, skip states)
    """
    print(f"[debug-recon] Reconstructing task_{task_id}/episode_{episode_id}")
    
    # Load trajectory data
    with h5py.File(dataset_path, 'r') as f:
        episode_path = f'task_{task_id}/episode_{episode_id}'
        
        if episode_path not in f:
            raise ValueError(f"Episode not found: {episode_path}")
            
        episode_group = f[episode_path]
        metadata = episode_group['metadata']
        
        # Extract metadata and reconstruction clues
        img_task_id = int(metadata.attrs['img_task_id'])
        img_episode_id = int(metadata.attrs['img_episode_id']) 
        img_env_seed = int(metadata.attrs['img_env_seed'])
        num_timesteps = int(metadata.attrs['num_timesteps'])
        task_description = metadata.attrs['task_description']
        
        # Load stored actions - this is the key fix!
        stored_actions = episode_group['timesteps/actions'][:]  # Shape: [timesteps, 7, 7] or [timesteps, 7]
        print(f"[debug-recon] Loaded stored actions: {stored_actions.shape}")
        
        # Handle action shape - take first action if multiple per timestep
        if len(stored_actions.shape) == 3:
            stored_actions = stored_actions[:, 0, :]  # Take first action per timestep
        elif len(stored_actions.shape) == 2 and stored_actions.shape[1] == 7:
            pass  # Already correct shape [timesteps, 7]
        else:
            raise ValueError(f"Unexpected action shape: {stored_actions.shape}")
            
        print(f"[debug-recon] Using actions shape: {stored_actions.shape}")
    
    # Initialize LIBERO environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(img_task_id)
    env, _ = get_libero_env(task, "prismatic", resolution=224)
    
    try:
        # Set up environment exactly as during data collection
        env.seed(img_env_seed)
        env.reset()
        
        # Set initial state if not libero_object
        if task_suite_name != 'libero_object':
            initial_states = task_suite.get_task_init_states(img_task_id)
            env.set_init_state(initial_states[img_episode_id])
        
        print(f"[debug-recon] Environment initialized - Task: {task_description}")
        print(f"[debug-recon] Reconstruction clues - task:{img_task_id}, episode:{img_episode_id}, seed:{img_env_seed}")
        
        # Create output directories
        if images_output_dir:
            episode_img_dir = Path(images_output_dir) / f"task_{task_id}" / f"episode_{episode_id}"
            episode_img_dir.mkdir(parents=True, exist_ok=True)
            
        if states_output_dir:
            episode_state_dir = Path(states_output_dir) / f"task_{task_id}" / f"episode_{episode_id}" 
            episode_state_dir.mkdir(parents=True, exist_ok=True)
        
        # Replay trajectory using stored actions
        all_states = []
        images_saved = 0
        states_saved = 0
        
        for timestep in range(num_timesteps):
            if timestep > 0:
                # Use the ACTUAL stored action from HDF5 file!
                action = stored_actions[timestep-1]  # Actions are offset by 1
                obs, reward, done, info = env.step(action.tolist())
            else:
                # First timestep - get initial observation
                obs = env.reset()
                if task_suite_name != 'libero_object':
                    initial_states = task_suite.get_task_init_states(img_task_id)
                    env.set_init_state(initial_states[img_episode_id])
            
            # Save reconstructed image
            if images_output_dir:
                img_array = get_libero_image(obs, 224, key="agentview_image")
                img = Image.fromarray(img_array.astype(np.uint8))
                img_path = episode_img_dir / f"timestep_{timestep:04d}.png"
                img.save(img_path)
                images_saved += 1
            
            # Save simulator state
            if states_output_dir:
                sim_state = extract_simulator_state(env)
                sim_state['timestep'] = timestep
                sim_state['stored_action'] = stored_actions[timestep-1].tolist() if timestep > 0 else None
                
                state_path = episode_state_dir / f"timestep_{timestep:04d}.json"
                with open(state_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization (recursive)
                    def convert_for_json(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {k: convert_for_json(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_for_json(item) for item in obj]
                        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                            return int(obj)
                        elif isinstance(obj, (np.float64, np.float32, np.float16)):
                            return float(obj)
                        else:
                            return obj
                    
                    json_state = convert_for_json(sim_state)
                    json.dump(json_state, f, indent=2)
                all_states.append(sim_state)
                states_saved += 1
                
            if timestep % 5 == 0:
                print(f"[debug-recon] Processed timestep {timestep}/{num_timesteps}")
        
        print(f"[debug-recon] Successfully reconstructed {task_id}/{episode_id}")
        print(f"[debug-recon] Images saved: {images_saved}, States saved: {states_saved}")
        
    finally:
        env.close()
    
    return {
        'images_saved': images_saved,
        'states_saved': states_saved,
        'task_description': task_description
    }


def reconstruct_dataset(
    dataset_path: str,
    images_output_dir: str = None,
    states_output_dir: str = None, 
    task_suite_name: str = "libero_90",
    max_episodes: int = None
):
    """
    Reconstruct all episodes in a trajectory dataset.
    
    Args:
        dataset_path: Path to trajectory data HDF5 file
        images_output_dir: Directory to save images (None to skip)
        states_output_dir: Directory to save simulator states (None to skip)
        task_suite_name: LIBERO task suite name
        max_episodes: Maximum episodes to process (None for all)
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"[debug-recon] Loading dataset: {dataset_path}")
    print(f"[debug-recon] Images output: {images_output_dir}")
    print(f"[debug-recon] States output: {states_output_dir}")
    
    if images_output_dir:
        Path(images_output_dir).mkdir(parents=True, exist_ok=True)
    if states_output_dir:
        Path(states_output_dir).mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    total_states = 0
    episodes_processed = 0
    
    with h5py.File(dataset_path, 'r') as f:
        print(f"[debug-recon] Dataset contains {len(f.keys())} task groups")
        
        for task_group_name in f.keys():
            if not task_group_name.startswith('task_'):
                continue
                
            task_id = int(task_group_name.split('_')[1])
            task_group = f[task_group_name]
            
            print(f"[debug-recon] Processing {task_group_name}...")
            
            for episode_group_name in task_group.keys():
                if not episode_group_name.startswith('episode_'):
                    continue
                    
                if max_episodes and episodes_processed >= max_episodes:
                    break
                    
                episode_id = int(episode_group_name.split('_')[1])
                
                try:
                    result = reconstruct_trajectory_episode(
                        dataset_path=dataset_path,
                        task_id=task_id,
                        episode_id=episode_id,
                        task_suite_name=task_suite_name,
                        images_output_dir=images_output_dir,
                        states_output_dir=states_output_dir
                    )
                    
                    total_images += result['images_saved']
                    total_states += result['states_saved'] 
                    episodes_processed += 1
                    
                except Exception as e:
                    print(f"[debug-recon] ERROR processing {task_group_name}/{episode_group_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if max_episodes and episodes_processed >= max_episodes:
                break
    
    print(f"[debug-recon] ===== RECONSTRUCTION COMPLETE =====")
    print(f"[debug-recon] Episodes processed: {episodes_processed}")
    print(f"[debug-recon] Total images saved: {total_images}")
    print(f"[debug-recon] Total states saved: {total_states}")


def main():
    parser = argparse.ArgumentParser(description='Reconstruct trajectory data using stored actions')
    parser.add_argument('dataset_path', help='Path to trajectory data HDF5 file')
    parser.add_argument('--task-suite-name', default='libero_90', help='LIBERO task suite name')
    parser.add_argument('--images-output-dir', help='Output directory for reconstructed images')
    parser.add_argument('--states-output-dir', help='Output directory for simulator states')
    parser.add_argument('--max-episodes', type=int, help='Maximum episodes to process')
    parser.add_argument('--task-id', type=int, help='Specific task ID to reconstruct')
    parser.add_argument('--episode-id', type=int, help='Specific episode ID to reconstruct')
    
    args = parser.parse_args()
    
    if not args.images_output_dir and not args.states_output_dir:
        print("ERROR: Must specify at least one of --images-output-dir or --states-output-dir")
        return
    
    # Validate output directories are in fast storage
    for output_dir in [args.images_output_dir, args.states_output_dir]:
        if output_dir and '/work/nvme/' not in str(output_dir):
            print(f"[debug-recon] WARNING: {output_dir} should be in /work/nvme/ for fast storage")
    
    if args.task_id is not None and args.episode_id is not None:
        # Reconstruct single episode
        result = reconstruct_trajectory_episode(
            dataset_path=args.dataset_path,
            task_id=args.task_id,
            episode_id=args.episode_id,
            task_suite_name=args.task_suite_name,
            images_output_dir=args.images_output_dir,
            states_output_dir=args.states_output_dir
        )
        print(f"Single episode reconstruction complete: {result}")
    else:
        # Reconstruct entire dataset
        reconstruct_dataset(
            dataset_path=args.dataset_path,
            images_output_dir=args.images_output_dir,
            states_output_dir=args.states_output_dir,
            task_suite_name=args.task_suite_name,
            max_episodes=args.max_episodes
        )


if __name__ == "__main__":
    main()