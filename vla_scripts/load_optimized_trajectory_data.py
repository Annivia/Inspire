"""
load_optimized_trajectory_data.py

Efficient loader for optimized multi-file trajectory data format.
Designed for linear probing experiments with 32x I/O reduction per layer.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import time


class OptimizedTrajectoryLoader:
    """
    Loader for optimized multi-file trajectory data format.
    
    Key features:
    - Load only required layers (32x I/O reduction)
    - Efficient episode indexing
    - Memory-mapped access for large arrays
    - Same API as original loader for backwards compatibility
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Load dataset summary
        summary_path = self.data_dir / "dataset_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
        else:
            # Fallback - scan directory structure
            self.summary = self._scan_directory_structure()
        
        # Load episode index
        self._load_episode_index()
        
        print(f"[OPTIMIZED_LOADER] Loaded dataset: {self.data_dir}")
        print(f"[OPTIMIZED_LOADER] Episodes: {len(self.episode_index)}")
        print(f"[OPTIMIZED_LOADER] Available layers: {self.summary['layer_indices']}")
        if 'generation_steps' in self.summary:
            print(f"[OPTIMIZED_LOADER] Available generation steps: {self.summary['generation_steps']}")
        print(f"[OPTIMIZED_LOADER] Total samples: {self.summary['total_samples']}")
    
    def _scan_directory_structure(self) -> Dict:
        """Fallback method to scan directory if summary is missing."""
        hidden_states_dir = self.data_dir / "hidden_states"
        
        # Check for new generation step format first
        generation_step_files = list(hidden_states_dir.glob("generation_step_*.h5"))
        if generation_step_files:
            # New format: generation steps
            generation_steps = [int(f.stem.split('_')[2]) for f in generation_step_files]
            
            # Get layer info from first generation step file
            layer_indices = []
            if generation_step_files:
                with h5py.File(generation_step_files[0], 'r') as f:
                    layer_indices = [int(key.split('_')[1]) for key in f.keys() if key.startswith('layer_')]
            
            return {
                'generation_steps': sorted(generation_steps),
                'layer_indices': sorted(layer_indices),
                'total_samples': 0,  # Will be determined from data
                'format_version': '2.0_generation_steps'
            }
        else:
            # Legacy format: separate layer files
            layer_files = list(hidden_states_dir.glob("layer_*.h5"))
            layer_indices = [int(f.stem.split('_')[1]) for f in layer_files]
            
            return {
                'layer_indices': sorted(layer_indices),
                'total_samples': 0,  # Will be determined from data
                'format_version': 'legacy_layer_files'
            }
    
    def _load_episode_index(self):
        """Load episode metadata and indexing information."""
        episode_index_path = self.data_dir / "episode_index.h5"
        
        if not episode_index_path.exists():
            raise FileNotFoundError(f"Episode index not found: {episode_index_path}")
        
        # Load episode index from HDF5
        episode_data = {}
        with h5py.File(episode_index_path, 'r') as f:
            for col_name in f.keys():
                data = f[col_name][:]
                # Handle string columns
                if data.dtype.kind == 'S':
                    data = data.astype('U')  # Convert bytes to unicode
                episode_data[col_name] = data
        
        self.episode_index = pd.DataFrame(episode_data)
        print(f"[OPTIMIZED_LOADER] Loaded episode index: {len(self.episode_index)} episodes")
    
    def load_generation_step_data(self,
                                 generation_step: int,
                                 layer_idx: Optional[int] = None,
                                 tasks: Optional[List[int]] = None,
                                 episodes: Optional[List[int]] = None,
                                 successful_only: bool = True) -> Tuple[Union[np.ndarray, Dict[int, np.ndarray]], pd.DataFrame]:
        """
        Load hidden states for a specific generation step with episode filtering.
        
        Args:
            generation_step: Generation step to load (0-6 for 7 action tokens)
            layer_idx: Specific layer to load (None for all layers)
            tasks: List of task IDs to include (None for all)
            episodes: List of episode IDs to include (None for all)
            successful_only: Only include successful episodes
            
        Returns:
            Tuple of (hidden_states_data, episode_metadata)
            - If layer_idx is None: Dict[layer_idx] -> np.ndarray
            - If layer_idx is specified: np.ndarray
        """
        print(f"[OPTIMIZED_LOADER] Loading generation step {generation_step}...")
        
        # Check if new format is available
        if 'generation_steps' not in self.summary:
            raise ValueError(f"Generation step loading not supported in this format version: {self.summary.get('format_version', 'unknown')}")
        
        if generation_step not in self.summary['generation_steps']:
            raise ValueError(f"Generation step {generation_step} not available. Available: {self.summary['generation_steps']}")
        
        # Filter episodes based on criteria
        filtered_episodes = self._filter_episodes(tasks, episodes, successful_only)
        
        if len(filtered_episodes) == 0:
            print(f"[OPTIMIZED_LOADER] No episodes match criteria!")
            return (np.array([]) if layer_idx is not None else {}), pd.DataFrame()
        
        print(f"[OPTIMIZED_LOADER] Loading {len(filtered_episodes)} episodes for generation step {generation_step}")
        
        # Load generation step data
        generation_step_path = self.data_dir / "hidden_states" / f"generation_step_{generation_step}.h5"
        
        start_time = time.time()
        with h5py.File(generation_step_path, 'r') as f:
            # Collect sample indices for filtered episodes
            sample_indices = []
            for _, episode in filtered_episodes.iterrows():
                sample_indices.extend(range(episode['start_idx'], episode['end_idx'] + 1))
            
            if layer_idx is not None:
                # Load specific layer
                layer_dataset_name = f'layer_{layer_idx:02d}'
                if layer_dataset_name not in f:
                    raise ValueError(f"Layer {layer_idx} not found in generation step {generation_step}")
                
                layer_dataset = f[layer_dataset_name]
                
                # Load selected samples efficiently
                if len(sample_indices) < len(layer_dataset) * 0.5:
                    hidden_states = layer_dataset[sample_indices]
                else:
                    hidden_states = layer_dataset[:][sample_indices]
                
                result = hidden_states
                
            else:
                # Load all layers
                result = {}
                for layer_dataset_name in f.keys():
                    if layer_dataset_name.startswith('layer_'):
                        layer_idx_parsed = int(layer_dataset_name.split('_')[1])
                        layer_dataset = f[layer_dataset_name]
                        
                        if len(sample_indices) < len(layer_dataset) * 0.5:
                            layer_data = layer_dataset[sample_indices]
                        else:
                            layer_data = layer_dataset[:][sample_indices]
                        
                        result[layer_idx_parsed] = layer_data
        
        load_time = time.time() - start_time
        
        if layer_idx is not None:
            print(f"[OPTIMIZED_LOADER] Loaded generation step {generation_step}, layer {layer_idx}: {result.shape} in {load_time:.2f}s")
        else:
            print(f"[OPTIMIZED_LOADER] Loaded generation step {generation_step}: {len(result)} layers in {load_time:.2f}s")
        
        return result, filtered_episodes
    
    def load_layer_data(self, 
                       layer_idx: int,
                       tasks: Optional[List[int]] = None,
                       episodes: Optional[List[int]] = None,
                       successful_only: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load hidden states for a specific layer with episode filtering.
        
        Args:
            layer_idx: Layer index to load
            tasks: List of task IDs to include (None for all)
            episodes: List of episode IDs to include (None for all) 
            successful_only: Only include successful episodes
            
        Returns:
            Tuple of (hidden_states_array, episode_metadata)
        """
        print(f"[OPTIMIZED_LOADER] Loading layer {layer_idx}...")
        
        if layer_idx not in self.summary['layer_indices']:
            raise ValueError(f"Layer {layer_idx} not available. Available: {self.summary['layer_indices']}")
        
        # Filter episodes based on criteria
        filtered_episodes = self._filter_episodes(tasks, episodes, successful_only)
        
        if len(filtered_episodes) == 0:
            print(f"[OPTIMIZED_LOADER] No episodes match criteria!")
            return np.array([]), pd.DataFrame()
        
        print(f"[OPTIMIZED_LOADER] Loading {len(filtered_episodes)} episodes for layer {layer_idx}")
        
        # Load layer data
        layer_path = self.data_dir / "hidden_states" / f"layer_{layer_idx:02d}.h5"
        
        start_time = time.time()
        with h5py.File(layer_path, 'r') as f:
            layer_dataset = f['hidden_states']
            
            # Collect sample indices for filtered episodes
            sample_indices = []
            for _, episode in filtered_episodes.iterrows():
                sample_indices.extend(range(episode['start_idx'], episode['end_idx'] + 1))
            
            # Load selected samples efficiently
            if len(sample_indices) < len(layer_dataset) * 0.5:
                # If loading less than 50% of data, use fancy indexing
                hidden_states = layer_dataset[sample_indices]
            else:
                # If loading most of the data, load all and slice
                hidden_states = layer_dataset[:][sample_indices]
        
        load_time = time.time() - start_time
        print(f"[OPTIMIZED_LOADER] Loaded layer {layer_idx}: {hidden_states.shape} in {load_time:.2f}s")
        
        return hidden_states, filtered_episodes
    
    def load_actions_data(self,
                         tasks: Optional[List[int]] = None,
                         episodes: Optional[List[int]] = None,
                         successful_only: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load action data with episode filtering.
        
        Returns:
            Tuple of (actions_array, episode_metadata)
        """
        print(f"[OPTIMIZED_LOADER] Loading actions...")
        
        # Filter episodes
        filtered_episodes = self._filter_episodes(tasks, episodes, successful_only)
        
        if len(filtered_episodes) == 0:
            return np.array([]), pd.DataFrame()
        
        # Load actions
        actions_path = self.data_dir / "actions.h5"
        start_time = time.time()
        
        with h5py.File(actions_path, 'r') as f:
            actions_dataset = f['actions']
            
            # Collect sample indices
            sample_indices = []
            for _, episode in filtered_episodes.iterrows():
                sample_indices.extend(range(episode['start_idx'], episode['end_idx'] + 1))
            
            actions = actions_dataset[sample_indices]
        
        load_time = time.time() - start_time
        print(f"[OPTIMIZED_LOADER] Loaded actions: {actions.shape} in {load_time:.2f}s")
        
        return actions, filtered_episodes
    
    def load_vision_features_data(self,
                                 tasks: Optional[List[int]] = None,
                                 episodes: Optional[List[int]] = None,
                                 successful_only: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Load vision features data with episode filtering.
        
        Returns:
            Tuple of (vision_features_array, episode_metadata)
        """
        print(f"[OPTIMIZED_LOADER] Loading vision features...")
        
        # Filter episodes
        filtered_episodes = self._filter_episodes(tasks, episodes, successful_only)
        
        if len(filtered_episodes) == 0:
            return np.array([]), pd.DataFrame()
        
        # Load vision features
        vision_path = self.data_dir / "vision_features.h5"
        start_time = time.time()
        
        with h5py.File(vision_path, 'r') as f:
            vision_dataset = f['vision_features']
            
            # Collect sample indices
            sample_indices = []
            for _, episode in filtered_episodes.iterrows():
                sample_indices.extend(range(episode['start_idx'], episode['end_idx'] + 1))
            
            vision_features = vision_dataset[sample_indices]
        
        load_time = time.time() - start_time
        print(f"[OPTIMIZED_LOADER] Loaded vision features: {vision_features.shape} in {load_time:.2f}s")
        
        return vision_features, filtered_episodes
    
    def _filter_episodes(self,
                        tasks: Optional[List[int]] = None,
                        episodes: Optional[List[int]] = None,
                        successful_only: bool = True) -> pd.DataFrame:
        """Filter episodes based on criteria."""
        filtered = self.episode_index.copy()
        
        if successful_only:
            filtered = filtered[filtered['success'] == True]
        
        if tasks is not None:
            filtered = filtered[filtered['task_id'].isin(tasks)]
        
        if episodes is not None:
            filtered = filtered[filtered['episode_id'].isin(episodes)]
        
        return filtered
    
    def get_dataset_info(self) -> Dict:
        """Get comprehensive dataset information."""
        return {
            'summary': self.summary,
            'episode_stats': {
                'total_episodes': len(self.episode_index),
                'successful_episodes': int(self.episode_index['success'].sum()),
                'unique_tasks': len(self.episode_index['task_id'].unique()),
                'task_ids': sorted(self.episode_index['task_id'].unique().tolist())
            },
            'data_files': {
                'actions': (self.data_dir / "actions.h5").exists(),
                'vision_features': (self.data_dir / "vision_features.h5").exists(),
                'hidden_states': len(list((self.data_dir / "hidden_states").glob("layer_*.h5"))),
                'episode_index': (self.data_dir / "episode_index.h5").exists()
            }
        }


def load_trajectory_dataset(data_path: Union[str, Path],
                           layers: Optional[List[int]] = None,
                           tasks: Optional[List[int]] = None,
                           episodes: Optional[List[int]] = None,
                           generation_steps: Optional[List[int]] = None,
                           successful_only: bool = True,
                           **kwargs) -> Dict:
    """
    Backwards-compatible interface for loading optimized trajectory data.
    
    This function provides the same interface as the original loader but
    with dramatically improved performance for single-layer access.
    
    Args:
        data_path: Path to optimized trajectory data directory
        layers: Layer indices to load (None for all available)
        tasks: Task IDs to include (None for all)
        episodes: Episode IDs to include (None for all)
        generation_steps: Generation steps to load (None for all available, [0] for default)
        successful_only: Only load successful episodes
        
    Returns:
        Dictionary with same structure as original loader:
        - 'hidden_states': Dict[layer_idx] -> np.ndarray
        - 'actions': np.ndarray  
        - 'vision_features': np.ndarray
        - 'metadata': List of episode metadata dicts
        - 'summary': Dataset summary
    """
    loader = OptimizedTrajectoryLoader(data_path)
    
    # Load episode metadata first
    filtered_episodes = loader._filter_episodes(tasks, episodes, successful_only)
    metadata = filtered_episodes.to_dict('records')
    
    # Load actions and vision features
    actions_array, _ = loader.load_actions_data(tasks, episodes, successful_only)
    vision_array, _ = loader.load_vision_features_data(tasks, episodes, successful_only)
    
    # Load hidden states - NEW: Support both generation step and legacy formats
    hidden_states = {}
    
    if 'generation_steps' in loader.summary:
        # NEW FORMAT: Load by generation step
        if generation_steps is None:
            generation_steps = loader.summary['generation_steps']
        
        if layers is None:
            layers = loader.summary['layer_indices']
        
        # Load data organized by generation step
        for gen_step in generation_steps:
            if gen_step in loader.summary['generation_steps']:
                step_data, _ = loader.load_generation_step_data(gen_step, tasks=tasks, episodes=episodes, successful_only=successful_only)
                
                # Reorganize to match legacy format: hidden_states[layer_idx]
                for layer_idx in layers:
                    if layer_idx in step_data:
                        layer_key = f"layer_{layer_idx}_step_{gen_step}" if len(generation_steps) > 1 else layer_idx
                        hidden_states[layer_key] = step_data[layer_idx]
        
    else:
        # LEGACY FORMAT: Load by layer
        if generation_steps is not None and generation_steps != [0]:
            print(f"WARNING: Legacy format only supports generation_step_0. Ignoring: {generation_steps}")
        
        if layers is None:
            layers = loader.summary['layer_indices']
        
        for layer_idx in layers:
            if layer_idx in loader.summary['layer_indices']:
                layer_data, _ = loader.load_layer_data(layer_idx, tasks, episodes, successful_only)
                hidden_states[layer_idx] = layer_data
            else:
                print(f"WARNING: Layer {layer_idx} not available in dataset")
    
    # Create summary
    summary = {
        'total_episodes': len(loader.episode_index),
        'loaded_episodes': len(filtered_episodes),
        'successful_episodes': int(filtered_episodes['success'].sum()) if len(filtered_episodes) > 0 else 0,
        'num_layers': len(hidden_states),
        'layers': list(hidden_states.keys()),
        'tasks': list(filtered_episodes['task_id'].unique()) if len(filtered_episodes) > 0 else [],
        'file_size_mb': sum(f.stat().st_size for f in loader.data_dir.rglob("*.h5")) / (1024 * 1024)
    }
    
    print(f"\nOptimized Dataset Summary:")
    print(f"  Loaded episodes: {len(filtered_episodes)}")
    print(f"  Successful episodes: {summary['successful_episodes']}")
    print(f"  Layers loaded: {len(hidden_states)}")
    print(f"  Actions shape: {actions_array.shape}")
    print(f"  Vision features shape: {vision_array.shape}")
    
    return {
        'hidden_states': hidden_states,
        'actions': actions_array,
        'vision_features': vision_array,
        'metadata': metadata,
        'summary': summary
    }


def get_layer_data_flat(dataset: Dict,
                       layer_idx: int,
                       generation_step: int = 0,
                       include_metadata: bool = True) -> Tuple[np.ndarray, Optional[List[Dict]]]:
    """
    Extract flattened hidden states for a specific layer.
    Compatible with original API but optimized for new format.
    """
    if generation_step != 0:
        print(f"WARNING: Optimized format only supports generation_step_0. Using 0 instead of {generation_step}")
    
    if layer_idx not in dataset['hidden_states']:
        raise ValueError(f"Layer {layer_idx} not found in dataset")
    
    hidden_states = dataset['hidden_states'][layer_idx]
    
    if include_metadata:
        # Create metadata for each sample
        metadata_list = []
        sample_idx = 0
        
        for episode_meta in dataset['metadata']:
            num_timesteps = episode_meta['num_timesteps']
            for timestep_id in range(num_timesteps):
                sample_meta = {
                    'task_id': episode_meta['task_id'],
                    'episode_id': episode_meta['episode_id'],
                    'timestep_id': timestep_id,
                    'generation_step': 0,
                    'task_description': episode_meta['task_description'],
                    'success': episode_meta['success']
                }
                metadata_list.append(sample_meta)
                sample_idx += 1
        
        return hidden_states, metadata_list
    else:
        return hidden_states, None


def get_actions_data_flat(dataset: Dict,
                         include_metadata: bool = True) -> Tuple[np.ndarray, Optional[List[Dict]]]:
    """
    Extract flattened actions data.
    Compatible with original API.
    """
    actions = dataset['actions']
    
    if include_metadata:
        # Create metadata for each sample  
        metadata_list = []
        
        for episode_meta in dataset['metadata']:
            num_timesteps = episode_meta['num_timesteps']
            for timestep_id in range(num_timesteps):
                sample_meta = {
                    'task_id': episode_meta['task_id'],
                    'episode_id': episode_meta['episode_id'],
                    'timestep_id': timestep_id,
                    'task_description': episode_meta['task_description'],
                    'success': episode_meta['success']
                }
                metadata_list.append(sample_meta)
        
        return actions, metadata_list
    else:
        return actions, None


def get_vision_features_data_flat(dataset: Dict,
                                 include_metadata: bool = True) -> Tuple[np.ndarray, Optional[List[Dict]]]:
    """
    Extract flattened vision features data.
    Compatible with original API.
    """
    vision_features = dataset['vision_features']
    
    if include_metadata:
        # Create metadata for each sample
        metadata_list = []
        
        for episode_meta in dataset['metadata']:
            num_timesteps = episode_meta['num_timesteps']
            for timestep_id in range(num_timesteps):
                # For vision features, we need patch-level metadata
                # Assuming vision_features shape is [samples, num_patches, vision_dim]
                if len(vision_features.shape) == 3:
                    num_patches = vision_features.shape[1]
                    for patch_id in range(num_patches):
                        sample_meta = {
                            'task_id': episode_meta['task_id'],
                            'episode_id': episode_meta['episode_id'],
                            'timestep_id': timestep_id,
                            'patch_id': patch_id,
                            'task_description': episode_meta['task_description'],
                            'success': episode_meta['success']
                        }
                        metadata_list.append(sample_meta)
                else:
                    # Already flattened
                    sample_meta = {
                        'task_id': episode_meta['task_id'],
                        'episode_id': episode_meta['episode_id'],
                        'timestep_id': timestep_id,
                        'task_description': episode_meta['task_description'],
                        'success': episode_meta['success']
                    }
                    metadata_list.append(sample_meta)
        
        return vision_features, metadata_list
    else:
        return vision_features, None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and test optimized trajectory data")
    parser.add_argument("data_dir", help="Path to optimized trajectory data directory")
    parser.add_argument("--layers", type=int, nargs='*', help="Layer indices to test")
    parser.add_argument("--info-only", action='store_true', help="Only show dataset info")
    
    args = parser.parse_args()
    
    try:
        loader = OptimizedTrajectoryLoader(args.data_dir)
        
        # Show dataset info
        info = loader.get_dataset_info()
        print("\n=== Dataset Info ===")
        print(json.dumps(info, indent=2))
        
        if not args.info_only:
            # Test loading a layer
            test_layers = args.layers or [info['summary']['layer_indices'][0]]
            
            for layer_idx in test_layers:
                print(f"\n=== Testing Layer {layer_idx} ===")
                hidden_states, episodes = loader.load_layer_data(layer_idx)
                print(f"Hidden states shape: {hidden_states.shape}")
                print(f"Episodes: {len(episodes)}")
            
            # Test backwards-compatible interface
            print(f"\n=== Testing Backwards-Compatible Interface ===")
            dataset = load_trajectory_dataset(args.data_dir, layers=test_layers[:1])
            print(f"Loaded dataset keys: {dataset.keys()}")
            print(f"Summary: {dataset['summary']}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()