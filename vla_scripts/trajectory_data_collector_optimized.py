"""
trajectory_data_collector_optimized.py

Optimized trajectory data collector for storing VLA evaluation data in efficient multi-file format.
- Separate files per layer for 32x I/O reduction during probing
- HDF5 compression and chunking optimizations
- Parallel processing support with temporary chunk files
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import threading
import time
import json
import tempfile
from collections import defaultdict


class OptimizedTrajectoryDataCollector:
    """
    Optimized data collector that writes to multi-file format for efficient probing.
    
    Design:
    - Collects data in memory during episode processing
    - Writes to temporary chunk files per process
    - Final combination creates optimized multi-file format
    """
    
    def __init__(self, 
                 save_dir: str, 
                 task_suite_name: str, 
                 process_id: int = 0,
                 temp_dir: Optional[str] = None):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task_suite_name = task_suite_name
        self.process_id = process_id
        
        # Setup temporary processing directory
        if temp_dir:
            self.temp_dir = Path(temp_dir) / f"gpu_process_{process_id}"
        else:
            self.temp_dir = self.save_dir / "temp_trajectory_processing" / f"gpu_process_{process_id}"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory data accumulation for new format: generation_step -> layer -> samples
        self.accumulated_data = {
            'hidden_states': defaultdict(lambda: defaultdict(list)),  # generation_step -> layer_idx -> list of samples
            'actions': [],
            'vision_features': [],
            'vlm_embeddings': [],
            'episodes': []  # Episode metadata with indexing info
        }
        
        self.current_sample_count = 0
        self.lock = threading.Lock()
        
        print(f"[OPTIMIZED_COLLECTOR] Process {process_id} initialized")
        print(f"[OPTIMIZED_COLLECTOR] Save directory: {self.save_dir}")
        print(f"[OPTIMIZED_COLLECTOR] Temp directory: {self.temp_dir}")
    
    def save_episode_data(self,
                         task_id: int,
                         episode_id: int, 
                         hidden_states_data: List[Dict],
                         task_description: str,
                         success: bool,
                         image_reconstruction_clues: Dict):
        """
        Accumulate episode data in memory for later batch writing.
        
        Args:
            task_id: LIBERO task ID
            episode_id: Episode number
            hidden_states_data: List of timestep data containing hidden_states, actions, vision_features
            task_description: Task description string
            success: Whether episode was successful
            image_reconstruction_clues: Dict with img_task_id, img_episode_id, img_env_seed
        """
        print(f"[OPTIMIZED_COLLECTOR] Accumulating episode data: task_{task_id}/episode_{episode_id}")
        print(f"[OPTIMIZED_COLLECTOR] Timesteps: {len(hidden_states_data)}, Success: {success}")
        
        if len(hidden_states_data) == 0:
            print(f"[OPTIMIZED_COLLECTOR] WARNING: No timestep data to save!")
            return
        
        with self.lock:
            episode_start_idx = self.current_sample_count
            
            # Process each timestep
            for timestep_idx, timestep_data in enumerate(hidden_states_data):
                
                # Accumulate actions
                if 'actions' in timestep_data:
                    action = timestep_data['actions']
                    if not isinstance(action, np.ndarray):
                        action = np.array(action)
                    self.accumulated_data['actions'].append(action)
                else:
                    print(f"WARNING: No actions for task_{task_id}/episode_{episode_id}/timestep_{timestep_idx}")
                
                # Accumulate vision features
                if 'vision_features' in timestep_data:
                    vision_feat = timestep_data['vision_features']
                    if not isinstance(vision_feat, np.ndarray):
                        vision_feat = np.array(vision_feat)
                    self.accumulated_data['vision_features'].append(vision_feat)
                else:
                    print(f"WARNING: No vision features for task_{task_id}/episode_{episode_id}/timestep_{timestep_idx}")
                
                # Accumulate VLM embeddings
                if 'vlm_embeddings' in timestep_data:
                    vlm_embed = timestep_data['vlm_embeddings']
                    if vlm_embed is not None:
                        if not isinstance(vlm_embed, np.ndarray):
                            vlm_embed = np.array(vlm_embed)
                        self.accumulated_data['vlm_embeddings'].append(vlm_embed)
                    else:
                        print(f"INFO: VLM embeddings is None for task_{task_id}/episode_{episode_id}/timestep_{timestep_idx}")
                else:
                    print(f"WARNING: No VLM embeddings for task_{task_id}/episode_{episode_id}/timestep_{timestep_idx}")
                
                # Accumulate hidden states by generation step and layer (NEW FORMAT)
                if 'hidden_states' in timestep_data:
                    generation_steps_data = timestep_data['hidden_states']
                    
                    # NEW FORMAT: hidden_states[generation_step][layer] = layer_data
                    for generation_step, layers_data in generation_steps_data.items():
                        for layer_idx, layer_hidden_states in layers_data.items():
                            
                            if not isinstance(layer_hidden_states, np.ndarray):
                                layer_hidden_states = np.array(layer_hidden_states)
                            
                            print(f"[COLLECTOR_DEBUG] Process {self.process_id}: Gen step {generation_step}, Layer {layer_idx}: shape {layer_hidden_states.shape}")
                            
                            # Store layer activations - consistent [1, hidden_dim] shape per layer per generation step
                            # No flattening needed since dimensions are now consistent
                            self.accumulated_data['hidden_states'][generation_step][layer_idx].append(layer_hidden_states)
                
                self.current_sample_count += 1
            
            episode_end_idx = self.current_sample_count - 1
            
            # Store episode metadata with indexing
            episode_metadata = {
                'task_id': task_id,
                'episode_id': episode_id,
                'success': success,
                'task_description': task_description,
                'num_timesteps': len(hidden_states_data),
                'start_idx': episode_start_idx,
                'end_idx': episode_end_idx,
                # Reconstruction clues
                'img_task_id': image_reconstruction_clues.get('task_id', task_id),
                'img_episode_id': image_reconstruction_clues.get('episode_id', episode_id),
                'img_env_seed': image_reconstruction_clues.get('env_seed', episode_id)
            }
            
            self.accumulated_data['episodes'].append(episode_metadata)
            
            print(f"[OPTIMIZED_COLLECTOR] Accumulated {len(hidden_states_data)} samples "
                  f"(total: {self.current_sample_count})")
    
    def save_chunk_to_temp(self):
        """
        Save accumulated data to temporary chunk files with HDF5 compression.
        """
        print(f"[OPTIMIZED_COLLECTOR] Saving accumulated data to temp files...")
        print(f"[OPTIMIZED_COLLECTOR] Total samples: {self.current_sample_count}")
        
        if self.current_sample_count == 0:
            print(f"[OPTIMIZED_COLLECTOR] No data to save!")
            return
        
        with self.lock:
            # HDF5 compression settings
            compression_kwargs = {
                'compression': 'gzip',
                'compression_opts': 6,
                'shuffle': True
            }
            
            # Save actions
            if self.accumulated_data['actions']:
                actions_path = self.temp_dir / "actions_chunk.h5"
                actions_array = np.stack(self.accumulated_data['actions'], axis=0)
                with h5py.File(actions_path, 'w') as f:
                    f.create_dataset('actions', data=actions_array, **compression_kwargs)
                print(f"[OPTIMIZED_COLLECTOR] Saved actions: {actions_array.shape}")
            
            # Save vision features
            if self.accumulated_data['vision_features']:
                vision_path = self.temp_dir / "vision_features_chunk.h5"
                vision_array = np.stack(self.accumulated_data['vision_features'], axis=0)
                with h5py.File(vision_path, 'w') as f:
                    f.create_dataset('vision_features', data=vision_array, **compression_kwargs)
                print(f"[OPTIMIZED_COLLECTOR] Saved vision features: {vision_array.shape}")
            
            # Save VLM embeddings
            if self.accumulated_data['vlm_embeddings']:
                vlm_path = self.temp_dir / "vlm_embeddings_chunk.h5"
                vlm_array = np.stack(self.accumulated_data['vlm_embeddings'], axis=0)
                with h5py.File(vlm_path, 'w') as f:
                    f.create_dataset('vlm_embeddings', data=vlm_array, **compression_kwargs)
                print(f"[OPTIMIZED_COLLECTOR] Saved VLM embeddings: {vlm_array.shape}")
            
            # Save hidden states by generation step (NEW FORMAT)
            hidden_states_dir = self.temp_dir / "hidden_states"
            hidden_states_dir.mkdir(exist_ok=True)
            
            for generation_step, layers_data in self.accumulated_data['hidden_states'].items():
                if layers_data:
                    generation_step_path = hidden_states_dir / f"generation_step_{generation_step}_chunk.h5"
                    
                    with h5py.File(generation_step_path, 'w') as f:
                        # Save each layer's data for this generation step
                        for layer_idx, layer_data in layers_data.items():
                            if layer_data:
                                
                                # Check shapes before stacking
                                sample_shapes = [sample.shape for sample in layer_data[:3]]  # Check first 3
                                
                                try:
                                    layer_array = np.stack(layer_data, axis=0)
                                    
                                    # Determine optimal chunking based on actual array shape
                                    chunk_size = min(1000, layer_array.shape[0])
                                    if len(layer_array.shape) == 4:
                                        # Shape: [samples, 1, seq_len, hidden_dim] (old format - should not happen after fix)
                                        chunks = (chunk_size, layer_array.shape[1], layer_array.shape[2], layer_array.shape[3])
                                    elif len(layer_array.shape) == 3:
                                        # Shape: [samples, 1, hidden_dim] (expected after fix)
                                        chunks = (chunk_size, layer_array.shape[1], layer_array.shape[2])
                                    elif len(layer_array.shape) == 2:
                                        # Shape: [samples, hidden_dim] (if we squeeze further)
                                        chunks = (chunk_size, layer_array.shape[1])
                                    else:
                                        chunks = True
                                    
                                    f.create_dataset(f'layer_{layer_idx:02d}',
                                                   data=layer_array,
                                                   chunks=chunks,
                                                   **compression_kwargs)
                                    
                                except Exception as e:
                                    print(f"[COLLECTOR_ERROR] Process {self.process_id}: Failed at gen step {generation_step}, layer {layer_idx}: {e}")
                                    print(f"[COLLECTOR_ERROR] Process {self.process_id}: Sample shapes were: {[s.shape for s in layer_data]}")
                                    raise e
                    
            # Save episode metadata
            episodes_path = self.temp_dir / "episodes_chunk.json"
            with open(episodes_path, 'w') as f:
                json.dump(self.accumulated_data['episodes'], f, indent=2)
            
            
            # Create processing manifest
            manifest = {
                'process_id': self.process_id,
                'task_suite_name': self.task_suite_name,
                'total_samples': self.current_sample_count,
                'total_episodes': len(self.accumulated_data['episodes']),
                'generation_steps': list(self.accumulated_data['hidden_states'].keys()),
                'layer_indices': list(set(layer_idx for layers in self.accumulated_data['hidden_states'].values() for layer_idx in layers.keys())),
                'timestamp': time.time()
            }
            
            manifest_path = self.temp_dir / "chunk_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"[OPTIMIZED_COLLECTOR] Chunk saving complete!")
            print(f"[OPTIMIZED_COLLECTOR] Temp directory: {self.temp_dir}")


def combine_chunks_to_optimized_format(
    temp_processing_dir: str,
    output_dir: str,
    task_suite_name: str
):
    """
    Combine temporary chunk files into final optimized multi-file format.
    
    Args:
        temp_processing_dir: Directory containing gpu_process_* subdirectories
        output_dir: Final output directory for optimized format
        task_suite_name: Task suite name for naming
    """
    temp_dir = Path(temp_processing_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[CHUNK_COMBINER] Combining chunks to optimized format...")
    print(f"[CHUNK_COMBINER] Temp dir: {temp_dir}")
    print(f"[CHUNK_COMBINER] Output dir: {output_dir}")
    
    # Find all process directories
    process_dirs = [d for d in temp_dir.iterdir() if d.is_dir() and d.name.startswith('gpu_process_')]
    print(f"[CHUNK_COMBINER] Found {len(process_dirs)} process directories")
    
    if not process_dirs:
        print(f"[CHUNK_COMBINER] No process directories found!")
        return
    
    # Load all manifests to understand data structure
    manifests = []
    total_samples = 0
    all_generation_steps = set()
    all_layer_indices = set()
    
    for process_dir in process_dirs:
        manifest_path = process_dir / "chunk_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                manifests.append({**manifest, 'process_dir': process_dir})
                total_samples += manifest['total_samples']
                all_generation_steps.update(manifest['generation_steps'])
                all_layer_indices.update(manifest['layer_indices'])
    
    print(f"[CHUNK_COMBINER] Total samples across all processes: {total_samples}")
    print(f"[CHUNK_COMBINER] Generation steps: {sorted(all_generation_steps)}")
    print(f"[CHUNK_COMBINER] Layer indices: {sorted(all_layer_indices)}")
    
    # HDF5 compression settings (removed conflicting chunks parameter)
    compression_kwargs = {
        'compression': 'gzip',
        'compression_opts': 6,
        'shuffle': True
    }
    
    # Combine actions
    print(f"[CHUNK_COMBINER] Combining actions...")
    actions_chunks = []
    for manifest in manifests:
        actions_path = manifest['process_dir'] / "actions_chunk.h5"
        if actions_path.exists():
            with h5py.File(actions_path, 'r') as f:
                actions_chunks.append(f['actions'][:])
    
    if actions_chunks:
        combined_actions = np.concatenate(actions_chunks, axis=0)
        actions_output_path = output_dir / "actions.h5"
        with h5py.File(actions_output_path, 'w') as f:
            f.create_dataset('actions', data=combined_actions, **compression_kwargs)
        print(f"[CHUNK_COMBINER] Saved combined actions: {combined_actions.shape}")
    
    # Combine vision features  
    print(f"[CHUNK_COMBINER] Combining vision features...")
    vision_chunks = []
    for manifest in manifests:
        vision_path = manifest['process_dir'] / "vision_features_chunk.h5"
        if vision_path.exists():
            with h5py.File(vision_path, 'r') as f:
                vision_chunks.append(f['vision_features'][:])
    
    if vision_chunks:
        combined_vision = np.concatenate(vision_chunks, axis=0)
        vision_output_path = output_dir / "vision_features.h5"
        with h5py.File(vision_output_path, 'w') as f:
            f.create_dataset('vision_features', data=combined_vision, **compression_kwargs)
        print(f"[CHUNK_COMBINER] Saved combined vision features: {combined_vision.shape}")
    
    # Combine VLM embeddings
    print(f"[CHUNK_COMBINER] Combining VLM embeddings...")
    vlm_chunks = []
    for manifest in manifests:
        vlm_path = manifest['process_dir'] / "vlm_embeddings_chunk.h5"
        if vlm_path.exists():
            with h5py.File(vlm_path, 'r') as f:
                vlm_chunks.append(f['vlm_embeddings'][:])
    
    if vlm_chunks:
        combined_vlm = np.concatenate(vlm_chunks, axis=0)
        vlm_output_path = output_dir / "vlm_embeddings.h5"
        with h5py.File(vlm_output_path, 'w') as f:
            f.create_dataset('vlm_embeddings', data=combined_vlm, **compression_kwargs)
        print(f"[CHUNK_COMBINER] Saved combined VLM embeddings: {combined_vlm.shape}")
    
    # Combine hidden states by generation step (NEW FORMAT)
    hidden_states_output_dir = output_dir / "hidden_states"
    hidden_states_output_dir.mkdir(exist_ok=True)
    
    print(f"[CHUNK_COMBINER] Combining hidden states by generation step...")
    for generation_step in sorted(all_generation_steps):
        print(f"[CHUNK_COMBINER] Processing generation step {generation_step}...")
        
        # Collect data for this generation step from all processes
        step_data = {}  # layer_idx -> list of chunks
        
        for manifest in manifests:
            step_path = manifest['process_dir'] / "hidden_states" / f"generation_step_{generation_step}_chunk.h5"
            if step_path.exists():
                with h5py.File(step_path, 'r') as f:
                    # Load all layers for this generation step from this process
                    for layer_dataset_name in f.keys():
                        if layer_dataset_name.startswith('layer_'):
                            layer_idx = int(layer_dataset_name.split('_')[1])
                            if layer_idx not in step_data:
                                step_data[layer_idx] = []
                            step_data[layer_idx].append(f[layer_dataset_name][:])
        
        # Combine and save this generation step
        if step_data:
            step_output_path = hidden_states_output_dir / f"generation_step_{generation_step}.h5"
            
            with h5py.File(step_output_path, 'w') as f:
                for layer_idx in sorted(step_data.keys()):
                    if step_data[layer_idx]:
                        combined_layer = np.concatenate(step_data[layer_idx], axis=0)
                        
                        # Optimize chunking for sequential access - handle 3D arrays
                        chunk_size = min(10000, combined_layer.shape[0])
                        if len(combined_layer.shape) == 3:
                            # Shape: [samples, 1, hidden_dim]
                            chunks = (chunk_size, combined_layer.shape[1], combined_layer.shape[2])
                        elif len(combined_layer.shape) == 2:
                            # Shape: [samples, hidden_dim]
                            chunks = (chunk_size, combined_layer.shape[1])
                        else:
                            chunks = True
                        
                        f.create_dataset(f'layer_{layer_idx:02d}',
                                       data=combined_layer,
                                       chunks=chunks,
                                       **compression_kwargs)
            
            print(f"[CHUNK_COMBINER] Saved generation step {generation_step}: {len(step_data)} layers")
    
    # Combine episode metadata and create index
    print(f"[CHUNK_COMBINER] Creating episode index...")
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
    episode_index_path = output_dir / "episode_index.h5"
    
    with h5py.File(episode_index_path, 'w') as f:
        # Save each column separately for efficient access
        for col in episode_df.columns:
            if episode_df[col].dtype == 'object':
                # String columns need special handling
                f.create_dataset(col, data=episode_df[col].astype('S'))
            else:
                f.create_dataset(col, data=episode_df[col].values, **compression_kwargs)
    
    print(f"[CHUNK_COMBINER] Saved episode index: {len(episode_df)} episodes")
    
    # Create summary metadata (UPDATED for new format)
    summary_path = output_dir / "dataset_summary.json"
    summary = {
        'task_suite_name': task_suite_name,
        'total_samples': int(total_samples),
        'total_episodes': len(all_episodes),
        'generation_steps': sorted(list(all_generation_steps)),
        'layer_indices': sorted(list(all_layer_indices)),
        'successful_episodes': int(episode_df['success'].sum()),
        'created_at': time.time(),
        'format_version': '2.0_optimized_generation_steps'
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[CHUNK_COMBINER] Combination complete!")
    print(f"[CHUNK_COMBINER] Output directory: {output_dir}")
    print(f"[CHUNK_COMBINER] Summary: {summary}")
    
    return summary


# Backwards compatibility wrapper
class TrajectoryDataCollector(OptimizedTrajectoryDataCollector):
    """Backwards compatibility wrapper"""
    def __init__(self, save_path: str, task_suite_name: str, process_id: int = 0):
        # Convert old single-file path to directory
        save_dir = Path(save_path).parent / "optimized_trajectory_data"
        super().__init__(save_dir, task_suite_name, process_id)
    
    def save_episode_hidden_states(self, *args, **kwargs):
        """Backwards compatibility method name"""
        return self.save_episode_data(*args, **kwargs)