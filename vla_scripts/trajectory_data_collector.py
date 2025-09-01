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
        success: bool
    ):
        """
        Save hidden states for a single episode trajectory.
        
        Args:
            task_id: LIBERO task ID
            episode: Episode number
            hidden_states_data: List of dicts, one per timestep, containing hidden states
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
        
        # Debug: Check first timestep structure
        first_step = hidden_states_data[0]
        if 'hidden_states' in first_step:
            print(f"[DATA_COLLECTOR] === DEBUGGING TIMESTEP DATA STRUCTURE ===")
            print(f"[DATA_COLLECTOR] Hidden states layers in first timestep: {list(first_step['hidden_states'].keys())}")
            print(f"[DATA_COLLECTOR] Total layers found: {len(first_step['hidden_states'])}")
            
            for layer_idx, layer_data in first_step['hidden_states'].items():
                print(f"[DATA_COLLECTOR] === LAYER {layer_idx} ANALYSIS ===")
                print(f"[DATA_COLLECTOR] Layer {layer_idx} data type: {type(layer_data)}")
                
                if isinstance(layer_data, np.ndarray):
                    print(f"[DATA_COLLECTOR] Layer {layer_idx} numpy array shape: {layer_data.shape}")
                    print(f"[DATA_COLLECTOR] Layer {layer_idx} numpy array dtype: {layer_data.dtype}")
                elif isinstance(layer_data, list):
                    print(f"[DATA_COLLECTOR] Layer {layer_idx} is a list with {len(layer_data)} items")
                    for i, item in enumerate(layer_data):
                        if hasattr(item, 'shape'):
                            print(f"[DATA_COLLECTOR] Layer {layer_idx} item[{i}] shape: {item.shape}")
                        else:
                            print(f"[DATA_COLLECTOR] Layer {layer_idx} item[{i}] type: {type(item)}")
                else:
                    print(f"[DATA_COLLECTOR] Layer {layer_idx} unknown type: {type(layer_data)}")
        else:
            print(f"[DATA_COLLECTOR] WARNING: No 'hidden_states' key found in timestep data!")
        
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
                    print(f"[DATA_COLLECTOR] Saved metadata")
                    
                    # Save timesteps data
                    timestep_group = episode_group.create_group('timesteps')
                    timestep_group.create_dataset('timestep_ids', data=np.arange(len(hidden_states_data)))
                    print(f"[DATA_COLLECTOR] Created timestep_ids dataset")
                    
                    # Save hidden states (per layer with generation steps)
                    hidden_group = timestep_group.create_group('hidden_states')
                    
                    # Organize hidden states by layer and handle variable sequence lengths
                    if len(hidden_states_data) > 0 and 'hidden_states' in hidden_states_data[0]:
                        # Get all layer indices from first timestep
                        layer_indices = list(hidden_states_data[0]['hidden_states'].keys())
                        print(f"[DATA_COLLECTOR] Processing layers: {layer_indices}")
                        
                        for layer_idx in layer_indices:
                            layer_subgroup = hidden_group.create_group(f'layer_{layer_idx}')
                            print(f"[DATA_COLLECTOR] === Processing Layer {layer_idx} ===")
                            
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
                                        print(f"[DATA_COLLECTOR] Layer {layer_idx}, timestep {t}: saved stacked array {data.shape}")
                                    elif isinstance(data, list):
                                        # Data is a list of generation steps with different shapes
                                        print(f"[DATA_COLLECTOR] Layer {layer_idx}, timestep {t}: handling {len(data)} generation steps")
                                        for gen_step, gen_data in enumerate(data):
                                            timestep_subgroup.create_dataset(f'generation_step_{gen_step}', data=gen_data)
                                            print(f"[DATA_COLLECTOR] Layer {layer_idx}, timestep {t}, gen_step {gen_step}: saved shape {gen_data.shape}")
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