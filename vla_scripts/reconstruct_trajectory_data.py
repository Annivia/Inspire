#!/usr/bin/env python3
"""
Reconstruct trajectory data using stored actions from optimized trajectory dataset.

This script replays VLA trajectories using the exact actions stored in the optimized format,
allowing perfect reconstruction of both images and simulator states that correspond to the
collected hidden states and vision features for probes 3 and 4.

Supports smart metadata-only loading for efficient episode selection.
"""

import os
os.environ["MUJOCO_GL"] = "egl" 
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

import argparse
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import sys
import json
from typing import Dict, List, Optional, Tuple
sys.path.append('/u/xzhang42/Inspire')
sys.path.append('/u/xzhang42/Inspire/LIBERO')
sys.path.append('/u/xzhang42/Inspire/vq_bet_official')

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image
import threading
import time
from collections import defaultdict


def extract_simulator_state(env):
    """
    Extract comprehensive simulator state as structured numpy arrays (tensor format).
    
    Returns:
        Dict containing tensors for robot state, object positions, contact info, etc.
    """
    sim_state = {}
    
    try:
        # Get MuJoCo simulation data
        sim = env.sim
        
        # Robot joint positions and velocities (fixed size tensors)
        sim_state['robot_joint_pos'] = sim.data.qpos[:7].copy().astype(np.float32)
        sim_state['robot_joint_vel'] = sim.data.qvel[:7].copy().astype(np.float32)
        
        # End-effector position and orientation (fixed size tensors)
        ee_pos = sim.data.site_xpos[sim.model.site_name2id('gripper0_grip_site')].copy()
        ee_quat = sim.data.get_body_xquat('gripper0_eef').copy()
        sim_state['ee_pos'] = ee_pos.astype(np.float32)  # [3]
        sim_state['ee_quat'] = ee_quat.astype(np.float32)  # [4]
        
        # All object positions and orientations as structured tensors
        object_positions = []
        object_orientations = []
        object_names = []
        
        for body_id in range(sim.model.nbody):
            body_name = sim.model.body_id2name(body_id)
            if body_name and not body_name.startswith('robot0'):  # Skip robot bodies
                pos = sim.data.body_xpos[body_id].copy()
                quat = sim.data.body_xquat[body_id].copy()
                object_positions.append(pos)
                object_orientations.append(quat)
                object_names.append(body_name)
        
        # Convert to numpy arrays for efficient storage
        if object_positions:
            sim_state['object_positions'] = np.stack(object_positions, axis=0).astype(np.float32)  # [N_objects, 3]
            sim_state['object_orientations'] = np.stack(object_orientations, axis=0).astype(np.float32)  # [N_objects, 4]
            # Store names separately for indexing
            sim_state['object_names'] = np.array([name.encode('utf-8') for name in object_names], dtype='S64')
        else:
            sim_state['object_positions'] = np.zeros((0, 3), dtype=np.float32)
            sim_state['object_orientations'] = np.zeros((0, 4), dtype=np.float32)
            sim_state['object_names'] = np.array([], dtype='S64')
        
        # Contact information as structured arrays
        if sim.data.ncon > 0:
            contact_geom1 = np.array([sim.data.contact[i].geom1 for i in range(sim.data.ncon)], dtype=np.int32)
            contact_geom2 = np.array([sim.data.contact[i].geom2 for i in range(sim.data.ncon)], dtype=np.int32)
            contact_pos = np.array([sim.data.contact[i].pos.copy() for i in range(sim.data.ncon)], dtype=np.float32)
            contact_dist = np.array([sim.data.contact[i].dist for i in range(sim.data.ncon)], dtype=np.float32)
            
            sim_state['contact_geom1'] = contact_geom1
            sim_state['contact_geom2'] = contact_geom2  
            sim_state['contact_pos'] = contact_pos  # [N_contacts, 3]
            sim_state['contact_dist'] = contact_dist  # [N_contacts]
        else:
            sim_state['contact_geom1'] = np.array([], dtype=np.int32)
            sim_state['contact_geom2'] = np.array([], dtype=np.int32)
            sim_state['contact_pos'] = np.zeros((0, 3), dtype=np.float32)
            sim_state['contact_dist'] = np.array([], dtype=np.float32)
        
        # Time and physics info as scalars
        sim_state['time'] = np.float32(sim.data.time)
        
    except Exception as e:
        print(f"[debug-state] WARNING: Could not extract full simulator state: {e}")
        # Fallback to empty tensors
        sim_state = {
            'robot_joint_pos': np.zeros(7, dtype=np.float32),
            'robot_joint_vel': np.zeros(7, dtype=np.float32),
            'ee_pos': np.zeros(3, dtype=np.float32),
            'ee_quat': np.array([0, 0, 0, 1], dtype=np.float32),
            'object_positions': np.zeros((0, 3), dtype=np.float32),
            'object_orientations': np.zeros((0, 4), dtype=np.float32),
            'object_names': np.array([], dtype='S64'),
            'contact_geom1': np.array([], dtype=np.int32),
            'contact_geom2': np.array([], dtype=np.int32),
            'contact_pos': np.zeros((0, 3), dtype=np.float32),
            'contact_dist': np.array([], dtype=np.float32),
            'time': np.float32(0.0),
            'error': str(e)
        }
        
    return sim_state


def load_episode_metadata(dataset_dir: str) -> pd.DataFrame:
    """
    Smart metadata-only loading from optimized trajectory dataset.
    
    Args:
        dataset_dir: Path to optimized trajectory data directory
        
    Returns:
        DataFrame with episode metadata including reconstruction clues
    """
    dataset_dir = Path(dataset_dir)
    episode_index_path = dataset_dir / "episode_index.h5"
    
    if not episode_index_path.exists():
        raise FileNotFoundError(f"Episode index not found: {episode_index_path}")
    
    print(f"[debug-metadata] Loading episode metadata from {episode_index_path}")
    
    # Load episode metadata efficiently
    episode_data = {}
    with h5py.File(episode_index_path, 'r') as f:
        for key in f.keys():
            dataset = f[key]
            if dataset.dtype.kind == 'S':  # String data
                episode_data[key] = [item.decode('utf-8') if hasattr(item, 'decode') else str(item) 
                                   for item in dataset[:]]
            else:
                episode_data[key] = dataset[:].tolist()
    
    episode_df = pd.DataFrame(episode_data)
    print(f"[debug-metadata] Loaded {len(episode_df)} episodes")
    
    return episode_df


class ParallelReconstructionCollector:
    """
    Parallel reconstruction collector for efficient state reconstruction.
    Follows same pattern as OptimizedTrajectoryDataCollector.
    """
    
    def __init__(self, 
                 save_dir: str,
                 process_id: int = 0,
                 temp_dir: Optional[str] = None):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.process_id = process_id
        
        # Setup temporary processing directory
        if temp_dir:
            self.temp_dir = Path(temp_dir) / f"reconstruction_process_{process_id}"
        else:
            self.temp_dir = self.save_dir / "temp_reconstruction_processing" / f"reconstruction_process_{process_id}"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory data accumulation for reconstruction states
        self.accumulated_data = {
            'states': defaultdict(list),  # state_field -> list of samples
            'images': [],  # For when rendering is enabled
            'episodes': []  # Episode metadata with indexing info
        }
        
        self.current_sample_count = 0
        self.lock = threading.Lock()
        
        print(f"[PARALLEL_RECONSTRUCTION] Process {process_id} initialized")
        print(f"[PARALLEL_RECONSTRUCTION] Save directory: {self.save_dir}")
        print(f"[PARALLEL_RECONSTRUCTION] Temp directory: {self.temp_dir}")
    
    def save_episode_reconstruction(self,
                                  episode_idx: int,
                                  task_id: int,
                                  episode_id: int,
                                  all_states: List[Dict],
                                  all_images: List[np.ndarray] = None,
                                  task_description: str = "",
                                  success: bool = True):
        """
        Accumulate episode reconstruction data in memory for later batch writing.
        """
        print(f"[PARALLEL_RECONSTRUCTION] Accumulating episode {episode_idx}: task_{task_id}/episode_{episode_id}")
        print(f"[PARALLEL_RECONSTRUCTION] States: {len(all_states)}, Images: {len(all_images) if all_images else 0}")
        
        if len(all_states) == 0:
            print(f"[PARALLEL_RECONSTRUCTION] WARNING: No state data to save!")
            return
        
        with self.lock:
            episode_start_idx = self.current_sample_count
            
            # Process each timestep's state data
            for timestep_idx, state_data in enumerate(all_states):
                
                # Accumulate state fields
                for state_field, state_value in state_data.items():
                    if state_field not in ['error']:  # Skip error strings
                        self.accumulated_data['states'][state_field].append(state_value)
                
                # Accumulate images if provided
                if all_images and timestep_idx < len(all_images):
                    self.accumulated_data['images'].append(all_images[timestep_idx])
                
                self.current_sample_count += 1
            
            episode_end_idx = self.current_sample_count - 1
            
            # Store episode metadata with indexing
            episode_metadata = {
                'episode_idx': episode_idx,
                'task_id': task_id,
                'episode_id': episode_id,
                'success': success,
                'task_description': task_description,
                'num_timesteps': len(all_states),
                'start_idx': episode_start_idx,
                'end_idx': episode_end_idx
            }
            
            self.accumulated_data['episodes'].append(episode_metadata)
            
            print(f"[PARALLEL_RECONSTRUCTION] Accumulated {len(all_states)} samples "
                  f"(total: {self.current_sample_count})")
    
    def save_chunk_to_temp(self):
        """
        Save accumulated reconstruction data to temporary chunk files with HDF5 compression.
        """
        print(f"[PARALLEL_RECONSTRUCTION] Saving accumulated data to temp files...")
        print(f"[PARALLEL_RECONSTRUCTION] Total samples: {self.current_sample_count}")
        
        if self.current_sample_count == 0:
            print(f"[PARALLEL_RECONSTRUCTION] No data to save!")
            return
        
        with self.lock:
            # HDF5 compression settings
            compression_kwargs = {
                'compression': 'gzip',
                'compression_opts': 6,
                'shuffle': True
            }
            
            # Save reconstructed states
            if self.accumulated_data['states']:
                states_path = self.temp_dir / "states_chunk.h5"
                with h5py.File(states_path, 'w') as f:
                    # Save each state field as stacked arrays
                    for state_field, field_data in self.accumulated_data['states'].items():
                        if field_data:
                            try:
                                # Stack to create [samples, ...] arrays
                                if state_field in ['object_names']:  # String arrays
                                    stacked_array = np.array(field_data, dtype='S64')
                                else:
                                    stacked_array = np.stack(field_data, axis=0)
                                
                                f.create_dataset(state_field, data=stacked_array, **compression_kwargs)
                                print(f"[PARALLEL_RECONSTRUCTION] Saved {state_field}: {stacked_array.shape}")
                            except Exception as e:
                                print(f"[RECONSTRUCTION_ERROR] Failed to stack {state_field}: {e}")
            
            # Save images if available
            if self.accumulated_data['images']:
                images_path = self.temp_dir / "images_chunk.h5"
                images_array = np.stack(self.accumulated_data['images'], axis=0)
                with h5py.File(images_path, 'w') as f:
                    f.create_dataset('images', data=images_array, **compression_kwargs)
                print(f"[PARALLEL_RECONSTRUCTION] Saved images: {images_array.shape}")
            
            # Save episode metadata
            episodes_path = self.temp_dir / "episodes_chunk.json"
            with open(episodes_path, 'w') as f:
                json.dump(self.accumulated_data['episodes'], f, indent=2)
            
            # Create processing manifest
            manifest = {
                'process_id': self.process_id,
                'total_samples': self.current_sample_count,
                'total_episodes': len(self.accumulated_data['episodes']),
                'state_fields': list(self.accumulated_data['states'].keys()),
                'has_images': len(self.accumulated_data['images']) > 0,
                'timestamp': time.time()
            }
            
            manifest_path = self.temp_dir / "reconstruction_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"[PARALLEL_RECONSTRUCTION] Chunk saving complete!")
            print(f"[PARALLEL_RECONSTRUCTION] Temp directory: {self.temp_dir}")


def reconstruct_trajectory_episode(
    dataset_dir: str,
    episode_idx: int,
    task_suite_name: str = "libero_90",
    images_output_dir: str = None,
    states_output_dir: str = None,
    episode_metadata: pd.DataFrame = None,
    enable_rendering: bool = True
):
    """
    Reconstruct a single episode trajectory using stored actions from optimized format.
    
    Args:
        dataset_dir: Path to optimized trajectory data directory
        episode_idx: Episode index in the dataset (0-based)
        task_suite_name: LIBERO task suite name
        images_output_dir: Directory to save reconstructed images (if None, skip images)
        states_output_dir: Directory to save simulator states (if None, skip states)
        episode_metadata: Pre-loaded episode metadata (for efficiency)
        enable_rendering: Whether to enable rendering for image reconstruction (disable for scaling)
    """
    dataset_dir = Path(dataset_dir)
    
    # Load episode metadata if not provided
    if episode_metadata is None:
        episode_metadata = load_episode_metadata(dataset_dir)
    
    if episode_idx >= len(episode_metadata):
        raise ValueError(f"Episode index {episode_idx} out of range (max: {len(episode_metadata)-1})")
    
    episode_info = episode_metadata.iloc[episode_idx]
    
    # Extract episode information
    task_id = int(episode_info['task_id'])
    episode_id = int(episode_info['episode_id'])
    img_task_id = int(episode_info['img_task_id'])
    img_episode_id = int(episode_info['img_episode_id'])
    img_env_seed = int(episode_info['img_env_seed'])
    num_timesteps = int(episode_info['num_timesteps'])
    start_idx = int(episode_info['start_idx'])
    end_idx = int(episode_info['end_idx'])
    task_description = episode_info['task_description']
    
    print(f"[debug-recon] Reconstructing episode {episode_idx}: task_{task_id}/episode_{episode_id}")
    print(f"[debug-recon] Data range: samples {start_idx}-{end_idx} ({num_timesteps} timesteps)")
    
    # Load stored actions for this episode
    actions_path = dataset_dir / "actions.h5"
    with h5py.File(actions_path, 'r') as f:
        # Extract actions for this episode using index range
        stored_actions = f['actions'][start_idx:end_idx+1]  # Include end_idx
        print(f"[debug-recon] Loaded stored actions: {stored_actions.shape}")
        
        # Fix: VQ-BET returns action horizons with shape (N, horizon, action_dim)
        # We only need the current action (first horizon element)
        if len(stored_actions.shape) == 3 and stored_actions.shape[1] > 1:
            print(f"[debug-recon] Action shape before horizon fix: {stored_actions.shape}")
            stored_actions = stored_actions[:, 0, :]  # Take first horizon element: (N, horizon, action_dim) -> (N, action_dim)
            print(f"[debug-recon] Action shape after horizon fix: {stored_actions.shape}")
        
        # Verify we have the right number of actions
        print(f"[debug-recon] Final actions shape: {stored_actions.shape}")
        print(f"[debug-recon] Expected timesteps: {num_timesteps}")
        print(f"[debug-recon] First few actions: {stored_actions[:3] if len(stored_actions) > 0 else 'None'}")
        
        if stored_actions.shape[0] != num_timesteps:
            print(f"[debug-recon] WARNING: Action count mismatch - expected {num_timesteps}, got {stored_actions.shape[0]}")
            # Adjust to match expected timesteps
            if stored_actions.shape[0] > num_timesteps:
                stored_actions = stored_actions[:num_timesteps]
            else:
                raise ValueError(f"Not enough actions for episode: expected {num_timesteps}, got {stored_actions.shape[0]}")
    
    # Initialize LIBERO environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(img_task_id)
    env, _ = get_libero_env(task, "prismatic", resolution=224)
    
    try:
        # CRITICAL: Set up environment exactly as during data collection
        # The order matters: seed AFTER getting env, then reset, then set initial state
        print(f"[debug-recon] Setting environment seed to {img_env_seed}")
        env.seed(img_env_seed)
        
        # Reset environment after seeding
        obs = env.reset()
        print(f"[debug-recon] Environment reset complete")
        
        # Set initial state if not libero_object (AFTER reset)
        if task_suite_name != 'libero_object':
            initial_states = task_suite.get_task_init_states(img_task_id)
            obs = env.set_init_state(initial_states[img_episode_id])
            print(f"[debug-recon] Set initial state for episode {img_episode_id}")
        else:
            print(f"[debug-recon] Using default initial state for libero_object")
        
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
        all_images = []  # Store images for GIF generation
        images_saved = 0
        states_saved = 0
        
        for timestep in range(num_timesteps):
            if timestep == 0:
                # First timestep - environment is already reset and initialized above
                print(f"[debug-recon] Starting trajectory replay from timestep 0")
            else:
                # Use the ACTUAL stored action from HDF5 file!
                action_idx = timestep - 1  # Actions are offset by 1 from timesteps
                if action_idx < len(stored_actions):
                    action = stored_actions[action_idx]
                    print(f"[debug-recon] Timestep {timestep}: action_idx={action_idx}, action={action}")
                    obs, reward, done, info = env.step(action.tolist())
                    if timestep <= 3:  # Debug first few steps
                        print(f"[debug-recon] Step result: reward={reward:.3f}, done={done}")
                else:
                    print(f"[debug-recon] ERROR: action_idx {action_idx} >= len(stored_actions) {len(stored_actions)}")
                    break
            
            # Save reconstructed image (only if rendering is enabled)
            if images_output_dir and enable_rendering:
                try:
                    img_array = get_libero_image(obs, 224, key="agentview_image")
                    img = Image.fromarray(img_array.astype(np.uint8))
                    
                    # Store image for GIF generation
                    all_images.append(img_array.astype(np.uint8))
                    
                    # Also save individual PNG if needed
                    img_path = episode_img_dir / f"timestep_{timestep:04d}.png"
                    img.save(img_path)
                    images_saved += 1
                except Exception as e:
                    print(f"[debug-recon] WARNING: Could not render image for timestep {timestep}: {e}")
                    if timestep == 0:
                        print(f"[debug-recon] Note: Rendering disabled for efficiency. Set enable_rendering=True if needed.")
            elif images_output_dir and not enable_rendering:
                # Skip rendering but count what would have been saved
                images_saved += 1
            
            # Accumulate simulator state (will save to HDF5 later)
            if states_output_dir:
                sim_state = extract_simulator_state(env)
                sim_state['timestep'] = timestep
                sim_state['stored_action'] = stored_actions[timestep-1] if timestep > 0 else np.zeros(7)
                all_states.append(sim_state)
                states_saved += 1
                
            if timestep % 5 == 0:
                print(f"[debug-recon] Processed timestep {timestep}/{num_timesteps}")
        
        # Generate GIF from collected images
        if images_output_dir and enable_rendering and all_images:
            try:
                gif_path = episode_img_dir / "trajectory.gif"
                
                # Convert numpy arrays to PIL Images
                pil_images = [Image.fromarray(img_array) for img_array in all_images]
                
                # Create GIF with reasonable duration (100ms per frame = 10 FPS)
                pil_images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=100,  # 100ms per frame
                    loop=0  # Loop forever
                )
                
                print(f"[debug-recon] Generated trajectory GIF: {gif_path}")
                print(f"[debug-recon] GIF contains {len(all_images)} frames at 10 FPS")
                
            except Exception as e:
                print(f"[debug-recon] WARNING: Could not generate GIF: {e}")
        
        # Save states to HDF5 after collecting all timesteps (using same pattern as data collection)
        if states_output_dir and all_states:
            states_h5_path = episode_state_dir / "states.h5"
            
            # HDF5 compression settings (same as data collection)
            compression_kwargs = {
                'compression': 'gzip',
                'compression_opts': 6,
                'shuffle': True
            }
            
            with h5py.File(states_h5_path, 'w') as f:
                # Stack each state field across timesteps (same pattern as trajectory collector)
                for key in all_states[0].keys():
                    if key == 'error':  # Skip error strings
                        continue
                    
                    # Stack tensors across timesteps
                    stacked_data = []
                    for state in all_states:
                        if key in state:
                            stacked_data.append(state[key])
                    
                    if stacked_data:
                        try:
                            # Stack to create [timesteps, ...] arrays
                            if key in ['object_names']:  # String arrays need special handling
                                stacked_array = np.array(stacked_data, dtype='S64')
                            else:
                                stacked_array = np.stack(stacked_data, axis=0)
                            
                            f.create_dataset(key, data=stacked_array, **compression_kwargs)
                        except Exception as e:
                            print(f"[debug-recon] WARNING: Could not stack {key}: {e}")
            
            print(f"[debug-recon] Saved {len(all_states)} states to HDF5: {states_h5_path}")
        
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
    dataset_dir: str,
    images_output_dir: str = None,
    states_output_dir: str = None, 
    task_suite_name: str = "libero_90",
    max_episodes: int = None,
    episode_filter: Dict = None,
    enable_rendering: bool = True
):
    """
    Reconstruct all episodes in an optimized trajectory dataset.
    
    Args:
        dataset_dir: Path to optimized trajectory data directory
        images_output_dir: Directory to save images (None to skip)
        states_output_dir: Directory to save simulator states (None to skip)
        task_suite_name: LIBERO task suite name
        max_episodes: Maximum episodes to process (None for all)
        episode_filter: Dict with filtering criteria (e.g., {'success': True, 'task_id': [1,2,3]})
        enable_rendering: Whether to enable rendering for image reconstruction (disable for scaling)
    """
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    print(f"[debug-recon] Loading optimized dataset: {dataset_dir}")
    print(f"[debug-recon] Images output: {images_output_dir}")
    print(f"[debug-recon] States output: {states_output_dir}")
    
    if images_output_dir:
        Path(images_output_dir).mkdir(parents=True, exist_ok=True)
    if states_output_dir:
        Path(states_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load episode metadata once
    episode_metadata = load_episode_metadata(dataset_dir)
    
    # Apply filtering
    if episode_filter:
        print(f"[debug-recon] Applying episode filter: {episode_filter}")
        for key, value in episode_filter.items():
            if key in episode_metadata.columns:
                if isinstance(value, list):
                    episode_metadata = episode_metadata[episode_metadata[key].isin(value)]
                else:
                    episode_metadata = episode_metadata[episode_metadata[key] == value]
        print(f"[debug-recon] Episodes after filtering: {len(episode_metadata)}")
    
    # Apply max_episodes limit
    if max_episodes:
        episode_metadata = episode_metadata.head(max_episodes)
        print(f"[debug-recon] Limited to {len(episode_metadata)} episodes")
    
    total_images = 0
    total_states = 0
    episodes_processed = 0
    
    for idx in range(len(episode_metadata)):
        episode_info = episode_metadata.iloc[idx]
        task_id = episode_info['task_id']
        episode_id = episode_info['episode_id']
        
        try:
            result = reconstruct_trajectory_episode(
                dataset_dir=dataset_dir,
                episode_idx=idx,
                task_suite_name=task_suite_name,
                images_output_dir=images_output_dir,
                states_output_dir=states_output_dir,
                episode_metadata=episode_metadata,
                enable_rendering=enable_rendering
            )
            
            total_images += result['images_saved']
            total_states += result['states_saved'] 
            episodes_processed += 1
            
        except Exception as e:
            print(f"[debug-recon] ERROR processing episode {idx} (task_{task_id}/episode_{episode_id}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[debug-recon] ===== RECONSTRUCTION COMPLETE =====")
    print(f"[debug-recon] Episodes processed: {episodes_processed}")
    print(f"[debug-recon] Total images saved: {total_images}")
    print(f"[debug-recon] Total states saved: {total_states}")


def combine_reconstruction_chunks_to_optimized_format(
    temp_processing_dir: str,
    output_dir: str
):
    """
    Combine temporary reconstruction chunk files into final optimized format.
    Follows same pattern as combine_chunks_to_optimized_format for trajectory data.
    
    Args:
        temp_processing_dir: Directory containing reconstruction_process_* subdirectories
        output_dir: Final output directory for reconstructed data
    """
    temp_dir = Path(temp_processing_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[RECONSTRUCTION_COMBINER] Combining reconstruction chunks...")
    print(f"[RECONSTRUCTION_COMBINER] Temp dir: {temp_dir}")
    print(f"[RECONSTRUCTION_COMBINER] Output dir: {output_dir}")
    
    # Find all process directories
    process_dirs = [d for d in temp_dir.iterdir() if d.is_dir() and d.name.startswith('reconstruction_process_')]
    print(f"[RECONSTRUCTION_COMBINER] Found {len(process_dirs)} process directories")
    
    if not process_dirs:
        print(f"[RECONSTRUCTION_COMBINER] No process directories found!")
        return
    
    # Load all manifests to understand data structure
    manifests = []
    total_samples = 0
    all_state_fields = set()
    has_images = False
    
    for process_dir in process_dirs:
        manifest_path = process_dir / "reconstruction_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                manifests.append({**manifest, 'process_dir': process_dir})
                total_samples += manifest['total_samples']
                all_state_fields.update(manifest['state_fields'])
                has_images = has_images or manifest['has_images']
    
    print(f"[RECONSTRUCTION_COMBINER] Total samples across all processes: {total_samples}")
    print(f"[RECONSTRUCTION_COMBINER] State fields: {sorted(all_state_fields)}")
    print(f"[RECONSTRUCTION_COMBINER] Has images: {has_images}")
    
    # HDF5 compression settings
    compression_kwargs = {
        'compression': 'gzip',
        'compression_opts': 6,
        'shuffle': True
    }
    
    # Combine states by field
    print(f"[RECONSTRUCTION_COMBINER] Combining state fields...")
    states_output_path = output_dir / "reconstructed_states.h5"
    
    with h5py.File(states_output_path, 'w') as output_f:
        for state_field in sorted(all_state_fields):
            field_chunks = []
            
            for manifest in manifests:
                states_path = manifest['process_dir'] / "states_chunk.h5"
                if states_path.exists():
                    with h5py.File(states_path, 'r') as f:
                        if state_field in f:
                            field_chunks.append(f[state_field][:])
            
            if field_chunks:
                combined_field = np.concatenate(field_chunks, axis=0)
                
                # Optimize chunking for sequential access
                chunk_size = min(10000, combined_field.shape[0])
                if len(combined_field.shape) == 1:
                    chunks = (chunk_size,)
                elif len(combined_field.shape) == 2:
                    chunks = (chunk_size, combined_field.shape[1])
                elif len(combined_field.shape) == 3:
                    chunks = (chunk_size, combined_field.shape[1], combined_field.shape[2])
                else:
                    chunks = True
                
                output_f.create_dataset(state_field,
                                      data=combined_field,
                                      chunks=chunks,
                                      **compression_kwargs)
                print(f"[RECONSTRUCTION_COMBINER] Combined {state_field}: {combined_field.shape}")
    
    # Combine images if available
    if has_images:
        print(f"[RECONSTRUCTION_COMBINER] Combining images...")
        image_chunks = []
        for manifest in manifests:
            images_path = manifest['process_dir'] / "images_chunk.h5"
            if images_path.exists():
                with h5py.File(images_path, 'r') as f:
                    image_chunks.append(f['images'][:])
        
        if image_chunks:
            combined_images = np.concatenate(image_chunks, axis=0)
            images_output_path = output_dir / "reconstructed_images.h5"
            with h5py.File(images_output_path, 'w') as f:
                f.create_dataset('images', data=combined_images, **compression_kwargs)
            print(f"[RECONSTRUCTION_COMBINER] Combined images: {combined_images.shape}")
    
    # Combine episode metadata and create index
    print(f"[RECONSTRUCTION_COMBINER] Creating episode index...")
    all_episodes = []
    sample_offset = 0
    
    for manifest in manifests:
        episodes_path = manifest['process_dir'] / "episodes_chunk.json"
        if episodes_path.exists():
            with open(episodes_path, 'r') as f:
                process_episodes = json.load(f)
                
                # Adjust start_idx and end_idx to account for global indexing
                for episode in process_episodes:
                    episode['start_idx'] += sample_offset
                    episode['end_idx'] += sample_offset
                
                all_episodes.extend(process_episodes)
                sample_offset += manifest['total_samples']
    
    # Create episode index DataFrame and save
    episode_df = pd.DataFrame(all_episodes)
    episode_index_path = output_dir / "reconstruction_episode_index.h5"
    
    with h5py.File(episode_index_path, 'w') as f:
        # Save each column separately for efficient access
        for col in episode_df.columns:
            if episode_df[col].dtype == 'object':
                # String columns need special handling
                f.create_dataset(col, data=episode_df[col].astype('S'))
            else:
                f.create_dataset(col, data=episode_df[col].values, **compression_kwargs)
    
    print(f"[RECONSTRUCTION_COMBINER] Saved episode index: {len(episode_df)} episodes")
    
    # Create summary metadata
    summary_path = output_dir / "reconstruction_summary.json"
    summary = {
        'total_samples': int(total_samples),
        'total_episodes': len(all_episodes),
        'state_fields': sorted(list(all_state_fields)),
        'has_images': has_images,
        'successful_episodes': int(episode_df['success'].sum() if 'success' in episode_df.columns else 0),
        'created_at': time.time(),
        'format_version': '1.0_reconstructed_states'
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[RECONSTRUCTION_COMBINER] Combination complete!")
    print(f"[RECONSTRUCTION_COMBINER] Output directory: {output_dir}")
    print(f"[RECONSTRUCTION_COMBINER] Summary: {summary}")
    
    return summary


def get_reconstruction_paths(dataset_dir: str) -> Dict[str, str]:
    """
    Derive reconstruction paths from dataset directory following mother folder structure.
    
    Args:
        dataset_dir: Path to optimized trajectory data (e.g., /path/to/optimized_trajectory_data)
        
    Returns:
        Dict with reconstruction paths in same mother folder
        
    Example:
        Input:  /work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data
        Output: {
            'reconstructed_data_dir': /work/nvme/bfbo/xzhang42/data/pilot_test/reconstructed_trajectory_data,
            'temp_processing_dir': /work/nvme/bfbo/xzhang42/data/pilot_test/temp_reconstruction_processing
        }
    """
    dataset_path = Path(dataset_dir)
    mother_dir = dataset_path.parent  # e.g., /work/nvme/bfbo/xzhang42/data/pilot_test/
    
    return {
        'reconstructed_data_dir': str(mother_dir / 'reconstructed_trajectory_data'),
        'temp_processing_dir': str(mother_dir / 'temp_reconstruction_processing')
    }


def main():
    parser = argparse.ArgumentParser(description='Reconstruct trajectory data from optimized dataset')
    parser.add_argument('dataset_dir', help='Path to optimized trajectory data directory')
    parser.add_argument('--task-suite-name', default='libero_90', help='LIBERO task suite name')
    parser.add_argument('--images-output-dir', help='Output directory for reconstructed images (auto-derived if not specified)')
    parser.add_argument('--states-output-dir', help='Output directory for simulator states (auto-derived if not specified)')
    parser.add_argument('--max-episodes', type=int, help='Maximum episodes to process')
    parser.add_argument('--episode-idx', type=int, help='Specific episode index to reconstruct (0-based)')
    parser.add_argument('--filter-success', action='store_true', help='Only process successful episodes')
    parser.add_argument('--filter-task-ids', nargs='+', type=int, help='Only process specific task IDs')
    parser.add_argument('--metadata-only', action='store_true', help='Only load and display metadata')
    parser.add_argument('--disable-rendering', action='store_true', help='Disable rendering for efficient state-only reconstruction')
    parser.add_argument('--auto-paths', action='store_true', help='Automatically derive output paths from dataset directory')
    
    args = parser.parse_args()
    
    # Handle metadata-only mode
    if args.metadata_only:
        episode_metadata = load_episode_metadata(args.dataset_dir)
        print("\n===== EPISODE METADATA =====")
        print(episode_metadata.to_string())
        return
    
    # Auto-derive paths if requested or if no output dirs specified
    if args.auto_paths or (not args.images_output_dir and not args.states_output_dir):
        reconstruction_paths = get_reconstruction_paths(args.dataset_dir)
        
        if not args.states_output_dir:
            args.states_output_dir = reconstruction_paths['reconstructed_data_dir']
            print(f"[auto-paths] Using auto-derived states output: {args.states_output_dir}")
        
        if not args.images_output_dir and not args.disable_rendering:
            args.images_output_dir = reconstruction_paths['reconstructed_data_dir'] + "/images"
            print(f"[auto-paths] Using auto-derived images output: {args.images_output_dir}")
    
    if not args.images_output_dir and not args.states_output_dir:
        print("ERROR: Must specify at least one of --images-output-dir or --states-output-dir (or use --metadata-only or --auto-paths)")
        return
    
    # Validate output directories are in fast storage
    for output_dir in [args.images_output_dir, args.states_output_dir]:
        if output_dir and '/work/nvme/' not in str(output_dir):
            print(f"[debug-recon] WARNING: {output_dir} should be in /work/nvme/ for fast storage")
    
    # Build episode filter
    episode_filter = {}
    if args.filter_success:
        episode_filter['success'] = True
    if args.filter_task_ids:
        episode_filter['task_id'] = args.filter_task_ids
    
    if args.episode_idx is not None:
        # Reconstruct single episode by index
        result = reconstruct_trajectory_episode(
            dataset_dir=args.dataset_dir,
            episode_idx=args.episode_idx,
            task_suite_name=args.task_suite_name,
            images_output_dir=args.images_output_dir,
            states_output_dir=args.states_output_dir,
            enable_rendering=not args.disable_rendering
        )
        print(f"Single episode reconstruction complete: {result}")
    else:
        # Reconstruct dataset
        reconstruct_dataset(
            dataset_dir=args.dataset_dir,
            images_output_dir=args.images_output_dir,
            states_output_dir=args.states_output_dir,
            task_suite_name=args.task_suite_name,
            max_episodes=args.max_episodes,
            episode_filter=episode_filter if episode_filter else None,
            enable_rendering=not args.disable_rendering
        )


if __name__ == "__main__":
    main()