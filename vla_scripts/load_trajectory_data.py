"""
load_trajectory_data.py

Script to load trajectory data collected during VLA evaluation for linear probe training.
Provides utilities to extract hidden states from HDF5 files with proper handling of
variable-length generation sequences from autoregressive generation.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse


def load_trajectory_dataset(
    data_path: Union[str, Path], 
    layers: Optional[List[int]] = None,
    tasks: Optional[List[int]] = None,
    episodes: Optional[List[int]] = None,
    generation_steps: Optional[List[int]] = None,
    successful_only: bool = True
) -> Dict:
    """
    Load hidden states from trajectory data HDF5 file.
    
    Args:
        data_path: Path to trajectory data HDF5 file
        layers: List of layer indices to load (default: all layers)
        tasks: List of task IDs to load (default: all tasks)
        episodes: List of episode IDs to load (default: all episodes)  
        generation_steps: List of generation step indices to load (default: all steps)
        successful_only: Only load data from successful episodes
        
    Returns:
        Dictionary containing:
        - 'hidden_states': Dict[layer_idx][task_id][episode_id][timestep_id][generation_step] -> np.ndarray
        - 'metadata': List of episode metadata dictionaries
        - 'summary': Dataset summary statistics
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Trajectory data file not found: {data_path}")
    
    print(f"Loading trajectory data from: {data_path}")
    
    hidden_states = {}
    metadata = []
    
    with h5py.File(data_path, 'r') as f:
        # Get available tasks
        available_tasks = [int(k.split('_')[1]) for k in f.keys() if k.startswith('task_')]
        if tasks is None:
            tasks = available_tasks
        else:
            tasks = [t for t in tasks if t in available_tasks]
        
        print(f"Loading tasks: {tasks}")
        
        total_episodes = 0
        successful_episodes = 0
        
        for task_id in tasks:
            task_group = f'task_{task_id}'
            if task_group not in f:
                print(f"Warning: Task {task_id} not found, skipping")
                continue
                
            # Get available episodes for this task
            available_episodes = []
            for ep_key in f[task_group].keys():
                if ep_key.startswith('episode_'):
                    ep_id = int(ep_key.split('_')[1])
                    available_episodes.append(ep_id)
            
            if episodes is None:
                task_episodes = available_episodes
            else:
                task_episodes = [e for e in episodes if e in available_episodes]
            
            print(f"Task {task_id}: Loading episodes {task_episodes}")
            
            for episode_id in task_episodes:
                episode_group = f'{task_group}/episode_{episode_id}'
                if episode_group not in f:
                    continue
                
                total_episodes += 1
                
                # Load episode metadata
                meta_group = f[episode_group]['metadata']
                episode_meta = {
                    'task_id': task_id,
                    'episode_id': episode_id,
                    'success': bool(meta_group.attrs.get('success', False)),
                    'task_description': meta_group.attrs.get('task_description', ''),
                    'num_timesteps': int(meta_group.attrs.get('num_timesteps', 0))
                }
                
                # Skip unsuccessful episodes if requested
                if successful_only and not episode_meta['success']:
                    print(f"Skipping unsuccessful episode: task_{task_id}/episode_{episode_id}")
                    continue
                    
                successful_episodes += 1
                metadata.append(episode_meta)
                
                # Load hidden states
                timesteps_group = f[episode_group]['timesteps']
                if 'hidden_states' not in timesteps_group:
                    print(f"Warning: No hidden states found for task_{task_id}/episode_{episode_id}")
                    continue
                
                hidden_states_group = timesteps_group['hidden_states']
                
                # Get available layers
                available_layers = []
                for layer_key in hidden_states_group.keys():
                    if layer_key.startswith('layer_'):
                        layer_idx = int(layer_key.split('_')[1])
                        available_layers.append(layer_idx)
                
                if layers is None:
                    episode_layers = available_layers
                else:
                    episode_layers = [l for l in layers if l in available_layers]
                
                for layer_idx in episode_layers:
                    if layer_idx not in hidden_states:
                        hidden_states[layer_idx] = {}
                    if task_id not in hidden_states[layer_idx]:
                        hidden_states[layer_idx][task_id] = {}
                    if episode_id not in hidden_states[layer_idx][task_id]:
                        hidden_states[layer_idx][task_id][episode_id] = {}
                    
                    layer_group = hidden_states_group[f'layer_{layer_idx}']
                    
                    # Load timesteps for this layer
                    for timestep_key in layer_group.keys():
                        if not timestep_key.startswith('timestep_'):
                            continue
                            
                        timestep_idx = int(timestep_key.split('_')[1])
                        timestep_group = layer_group[timestep_key]
                        
                        hidden_states[layer_idx][task_id][episode_id][timestep_idx] = {}
                        
                        # Load generation steps for this timestep
                        available_gen_steps = []
                        for gen_key in timestep_group.keys():
                            if gen_key.startswith('generation_step_'):
                                gen_step = int(gen_key.split('_')[2])
                                available_gen_steps.append(gen_step)
                        
                        if generation_steps is None:
                            timestep_gen_steps = available_gen_steps
                        else:
                            timestep_gen_steps = [g for g in generation_steps if g in available_gen_steps]
                        
                        for gen_step in timestep_gen_steps:
                            gen_dataset = timestep_group[f'generation_step_{gen_step}']
                            hidden_states[layer_idx][task_id][episode_id][timestep_idx][gen_step] = np.array(gen_dataset)
    
    # Create summary
    summary = {
        'total_episodes': total_episodes,
        'successful_episodes': successful_episodes, 
        'loaded_episodes': len(metadata),
        'num_layers': len(hidden_states.keys()) if hidden_states else 0,
        'layers': sorted(hidden_states.keys()) if hidden_states else [],
        'tasks': tasks,
        'file_size_mb': data_path.stat().st_size / (1024 * 1024)
    }
    
    print(f"\nDataset Summary:")
    print(f"  Total episodes in file: {total_episodes}")
    print(f"  Successful episodes: {successful_episodes}")
    print(f"  Loaded episodes: {len(metadata)}")
    print(f"  Number of layers: {summary['num_layers']}")
    print(f"  Layers: {summary['layers']}")
    print(f"  Tasks: {summary['tasks']}")
    print(f"  File size: {summary['file_size_mb']:.2f} MB")
    
    return {
        'hidden_states': hidden_states,
        'metadata': metadata,
        'summary': summary
    }


def get_layer_data_flat(
    dataset: Dict,
    layer_idx: int,
    generation_step: int = 0,
    include_metadata: bool = True
) -> Tuple[np.ndarray, Optional[List[Dict]]]:
    """
    Extract and flatten hidden states for a specific layer and generation step.
    
    Args:
        dataset: Dataset dictionary from load_trajectory_dataset()
        layer_idx: Layer index to extract
        generation_step: Generation step index (0=full input, 1-6=autoregressive steps)
        include_metadata: Whether to return corresponding metadata
        
    Returns:
        Tuple of (flattened_hidden_states, metadata_list)
        - flattened_hidden_states: np.ndarray of shape (N, hidden_dim)
        - metadata_list: List of metadata dicts corresponding to each sample
    """
    if layer_idx not in dataset['hidden_states']:
        raise ValueError(f"Layer {layer_idx} not found in dataset")
    
    layer_data = dataset['hidden_states'][layer_idx]
    all_hidden_states = []
    sample_metadata = []
    
    for task_id in layer_data:
        for episode_id in layer_data[task_id]:
            # Find corresponding episode metadata
            episode_meta = None
            if include_metadata:
                for meta in dataset['metadata']:
                    if meta['task_id'] == task_id and meta['episode_id'] == episode_id:
                        episode_meta = meta
                        break
            
            for timestep_id in layer_data[task_id][episode_id]:
                timestep_data = layer_data[task_id][episode_id][timestep_id]
                
                if generation_step in timestep_data:
                    hidden_state = timestep_data[generation_step]  # Shape: (1, seq_len, hidden_dim)
                    
                    # Flatten: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim) -> (seq_len * hidden_dim,)
                    flattened = hidden_state.reshape(-1, hidden_state.shape[-1])
                    all_hidden_states.append(flattened)
                    
                    if include_metadata:
                        # Create sample metadata
                        sample_meta = {
                            'task_id': task_id,
                            'episode_id': episode_id,
                            'timestep_id': timestep_id,
                            'generation_step': generation_step,
                            'sequence_length': hidden_state.shape[1],
                            'task_description': episode_meta['task_description'] if episode_meta else '',
                            'success': episode_meta['success'] if episode_meta else False
                        }
                        sample_metadata.extend([sample_meta] * flattened.shape[0])
    
    if not all_hidden_states:
        return np.array([]), []
    
    # Stack all hidden states
    stacked_hidden_states = np.vstack(all_hidden_states)
    
    return stacked_hidden_states, sample_metadata if include_metadata else None


def print_dataset_structure(data_path: Union[str, Path]):
    """Print detailed structure of the HDF5 trajectory data file."""
    with h5py.File(data_path, 'r') as f:
        def print_structure(name, obj, indent=0):
            spaces = "  " * indent
            if isinstance(obj, h5py.Group):
                print(f"{spaces}{name}: group with {len(obj.keys())} items")
                if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                    for attr_name, attr_val in obj.attrs.items():
                        print(f"{spaces}  @{attr_name}: {attr_val}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{spaces}{name}: dataset {obj.shape} {obj.dtype}")
        
        print(f"HDF5 Structure of {data_path}:")
        f.visititems(lambda name, obj: print_structure(name, obj, name.count('/')))


def main():
    parser = argparse.ArgumentParser(description='Load and inspect VLA trajectory data')
    parser.add_argument('data_path', help='Path to trajectory data HDF5 file')
    parser.add_argument('--structure', action='store_true', help='Print dataset structure')
    parser.add_argument('--layers', nargs='+', type=int, help='Layer indices to load')
    parser.add_argument('--tasks', nargs='+', type=int, help='Task IDs to load')
    parser.add_argument('--generation-steps', nargs='+', type=int, help='Generation step indices to load')
    parser.add_argument('--include-failed', action='store_true', help='Include failed episodes')
    
    args = parser.parse_args()
    
    if args.structure:
        print_dataset_structure(args.data_path)
        return
    
    # Load dataset
    dataset = load_trajectory_dataset(
        args.data_path,
        layers=args.layers,
        tasks=args.tasks,
        generation_steps=args.generation_steps,
        successful_only=not args.include_failed
    )
    
    # Example: Extract layer 0, generation step 0 data
    if 0 in dataset['summary']['layers']:
        print(f"\nExample: Extracting Layer 0, Generation Step 0 data...")
        hidden_states, metadata = get_layer_data_flat(dataset, layer_idx=0, generation_step=0)
        print(f"Flattened hidden states shape: {hidden_states.shape}")
        print(f"Number of samples: {len(metadata) if metadata else 0}")
        
        if metadata and len(metadata) > 0:
            print(f"First sample metadata: {metadata[0]}")


if __name__ == "__main__":
    main()