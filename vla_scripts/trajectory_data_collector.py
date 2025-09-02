"""
trajectory_data_collector.py

Basic trajectory data collector for storing hidden states during VLA evaluation.
Start with minimal implementation focusing only on hidden states storage.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import threading
import time


class TrajectoryDataCollector:
    def __init__(self, save_path: str, task_suite_name: str, process_id: int = 0):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        # Use separate file per process to avoid locking conflicts
        self.data_file = self.save_path / f"trajectory_data_{task_suite_name}_proc_{process_id}.h5"
        self.lock = threading.Lock()
        self.process_id = process_id
        
        print(f"[DATA_COLLECTOR] Initialized trajectory data collector")
        print(f"[DATA_COLLECTOR] Save path: {self.save_path}")
        print(f"[DATA_COLLECTOR] Data file: {self.data_file}")
        
        # Create the HDF5 file if it doesn't exist
        with h5py.File(self.data_file, 'a') as f:
            f.attrs['created_at'] = time.time()
            f.attrs['task_suite'] = task_suite_name
            print(f"[DATA_COLLECTOR] HDF5 file initialized: {self.data_file}")
    
    def save_episode_hidden_states(
        self, 
        task_id: int, 
        episode: int, 
        hidden_states_data: List[Dict],
        task_description: str,
        success: bool,
        image_reconstruction_clues: Dict = None
    ):
        """
        Save hidden states and actions for a single episode trajectory.
        
        Args:
            task_id: LIBERO task ID
            episode: Episode number
            hidden_states_data: List of dicts, one per timestep, containing hidden states and actions
            task_description: Task description string
            success: Whether the episode was successful
        """
        print(f"[DATA_COLLECTOR] Saving episode data: task_{task_id}/episode_{episode}")
        print(f"[DATA_COLLECTOR] Number of timesteps: {len(hidden_states_data)}")
        print(f"[DATA_COLLECTOR] Task description: {task_description}")
        print(f"[DATA_COLLECTOR] Success: {success}")
        
        if len(hidden_states_data) == 0:
            print(f"[DATA_COLLECTOR] WARNING: No hidden states data to save!")
            return
        
        # Debug: Check if actions and vision features are present in the data
        first_step = hidden_states_data[0]
        print(f"[debug-action] First timestep keys: {list(first_step.keys())}")
        
        if 'actions' in first_step:
            print(f"[debug-action] Actions found - shape: {np.array(first_step['actions']).shape}")
        else:
            print(f"[debug-action] WARNING: No 'actions' key found in timestep data!")
            
        if 'vision_features' in first_step:
            print(f"[debug-visual] Vision features found - shape: {np.array(first_step['vision_features']).shape}")
        else:
            print(f"[debug-visual] WARNING: No 'vision_features' key found in timestep data!")
        
        with self.lock:
            try:
                with h5py.File(self.data_file, 'a') as f:
                    # Create episode group
                    episode_path = f'task_{task_id}/episode_{episode}'
                    if episode_path in f:
                        print(f"[DATA_COLLECTOR] WARNING: Episode {episode_path} already exists, skipping")
                        return
                        
                    episode_group = f.create_group(episode_path)
                    print(f"[DATA_COLLECTOR] Created group: {episode_path}")
                    
                    # Save metadata
                    meta_group = episode_group.create_group('metadata')
                    meta_group.attrs['task_description'] = task_description
                    meta_group.attrs['success'] = success
                    meta_group.attrs['num_timesteps'] = len(hidden_states_data)
                    meta_group.attrs['libero_task_id'] = task_id
                    meta_group.attrs['libero_episode_id'] = episode
                    
                    # Save image reconstruction clues (minimal - just 3 integers per episode!)
                    if image_reconstruction_clues:
                        meta_group.attrs['img_task_id'] = image_reconstruction_clues.get('task_id', task_id)
                        meta_group.attrs['img_episode_id'] = image_reconstruction_clues.get('episode_id', episode) 
                        meta_group.attrs['img_env_seed'] = image_reconstruction_clues.get('env_seed', episode)
                        print(f"[debug-image] Saved image reconstruction clues: {image_reconstruction_clues}")
                    else:
                        print(f"[debug-image] WARNING: No image reconstruction clues provided")
                        
                    print(f"[DATA_COLLECTOR] Saved metadata")
                    
                    # Save timesteps data
                    timestep_group = episode_group.create_group('timesteps')
                    timestep_group.create_dataset('timestep_ids', data=np.arange(len(hidden_states_data)))
                    print(f"[DATA_COLLECTOR] Created timestep_ids dataset")
                    
                    # Extract and save actions
                    actions_list = []
                    for timestep_data in hidden_states_data:
                        if 'actions' in timestep_data:
                            action = timestep_data['actions']
                            # Ensure action is numpy array
                            if not isinstance(action, np.ndarray):
                                action = np.array(action)
                            actions_list.append(action)
                            print(f"[debug-action] Timestep {len(actions_list)-1}: action shape {action.shape}, values {action}")
                        else:
                            print(f"[debug-action] WARNING: No action found for timestep {len(actions_list)}")
                    
                    if actions_list:
                        # Stack all actions into [timesteps, 7] array
                        actions_array = np.stack(actions_list, axis=0)
                        timestep_group.create_dataset('actions', data=actions_array)
                        print(f"[debug-action] Saved actions array with shape: {actions_array.shape}")
                    else:
                        print(f"[debug-action] WARNING: No actions to save!")
                    
                    # Extract and save vision features
                    vision_features_list = []
                    for timestep_data in hidden_states_data:
                        if 'vision_features' in timestep_data:
                            vision_feat = timestep_data['vision_features']
                            # Ensure vision features are numpy array
                            if not isinstance(vision_feat, np.ndarray):
                                vision_feat = np.array(vision_feat)
                            vision_features_list.append(vision_feat)
                            print(f"[debug-visual] Timestep {len(vision_features_list)-1}: vision shape {vision_feat.shape}, dtype {vision_feat.dtype}")
                        else:
                            print(f"[debug-visual] WARNING: No vision features found for timestep {len(vision_features_list)}")
                    
                    if vision_features_list:
                        # Stack all vision features into [timesteps, num_patches, vision_dim] array
                        vision_features_array = np.stack(vision_features_list, axis=0)
                        timestep_group.create_dataset('vision_features', data=vision_features_array)
                        print(f"[debug-visual] Saved vision features array with shape: {vision_features_array.shape}")
                    else:
                        print(f"[debug-visual] WARNING: No vision features to save!")
                    
                    # Save hidden states (per layer with generation steps)
                    hidden_group = timestep_group.create_group('hidden_states')
                    
                    # Organize hidden states by layer and handle variable sequence lengths
                    if len(hidden_states_data) > 0 and 'hidden_states' in hidden_states_data[0]:
                        # Get all layer indices from first timestep
                        layer_indices = list(hidden_states_data[0]['hidden_states'].keys())
                        print(f"[DATA_COLLECTOR] Processing layers: {layer_indices}")
                        
                        for layer_idx in layer_indices:
                            layer_subgroup = hidden_group.create_group(f'layer_{layer_idx}')
                            
                            # Collect all timesteps for this layer
                            layer_data_all_timesteps = []
                            for timestep_data in hidden_states_data:
                                if 'hidden_states' in timestep_data and layer_idx in timestep_data['hidden_states']:
                                    layer_data_all_timesteps.append(timestep_data['hidden_states'][layer_idx])
                                else:
                                    print(f"[DATA_COLLECTOR] WARNING: Missing hidden states for layer {layer_idx}")
                            
                            if layer_data_all_timesteps:
                                # Handle both stacked arrays and lists of generation steps
                                for t, data in enumerate(layer_data_all_timesteps):
                                    timestep_subgroup = layer_subgroup.create_group(f'timestep_{t}')
                                    
                                    if isinstance(data, np.ndarray):
                                        # Data is already stacked (all generation steps have same shape)
                                        timestep_subgroup.create_dataset('hidden_states', data=data)
                                    elif isinstance(data, list):
                                        # Data is a list of generation steps with different shapes
                                        for gen_step, gen_data in enumerate(data):
                                            timestep_subgroup.create_dataset(f'generation_step_{gen_step}', data=gen_data)
                                    else:
                                        print(f"[DATA_COLLECTOR] ERROR: Unknown data type for layer {layer_idx}: {type(data)}")
                                
                                print(f"[DATA_COLLECTOR] Successfully saved layer_{layer_idx} with {len(layer_data_all_timesteps)} timesteps")
                    
                    print(f"[DATA_COLLECTOR] Successfully saved episode {episode_path}")
                    
            except Exception as e:
                print(f"[DATA_COLLECTOR] ERROR saving episode data: {e}")
                import traceback
                traceback.print_exc()
    
    def get_file_info(self):
        """Debug function to print HDF5 file structure"""
        try:
            with h5py.File(self.data_file, 'r') as f:
                print(f"[DATA_COLLECTOR] HDF5 file structure:")
                
                def print_structure(name, obj):
                    print(f"  {name}: {type(obj)}")
                    if hasattr(obj, 'shape'):
                        print(f"    shape: {obj.shape}")
                    if hasattr(obj, 'attrs'):
                        for attr_name in obj.attrs.keys():
                            print(f"    attr {attr_name}: {obj.attrs[attr_name]}")
                
                f.visititems(print_structure)
        except Exception as e:
            print(f"[DATA_COLLECTOR] ERROR reading file structure: {e}")