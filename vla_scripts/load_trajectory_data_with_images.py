#!/usr/bin/env python3
"""
Load trajectory data with cross-referenced images for linear probe training.

This script provides unified access to both trajectory data (hidden states, actions, 
vision features) and corresponding images, enabling complete datasets for analysis.
"""

import h5py
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import argparse

# Import the base trajectory loader
import sys
sys.path.append('/u/xzhang42/Inspire/vla_scripts')
from load_trajectory_data import load_trajectory_dataset


class TrajectoryDataWithImages:
    """
    Unified interface for trajectory data and corresponding images.
    Provides cross-referenced access for linear probe training.
    """
    
    def __init__(self, dataset_path: str, images_dir: str):
        """
        Initialize with trajectory dataset and images directory.
        
        Args:
            dataset_path: Path to trajectory data HDF5 file
            images_dir: Directory containing generated images
        """
        self.dataset_path = Path(dataset_path)
        self.images_dir = Path(images_dir)
        
        print(f"[debug-image] Loading trajectory dataset from: {self.dataset_path}")
        print(f"[debug-image] Images directory: {self.images_dir}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Load the complete trajectory dataset
        self.dataset = load_trajectory_dataset(str(self.dataset_path))
        
        print(f"[debug-image] Dataset loaded successfully")
        print(f"[debug-image] Available data types: {list(self.dataset.keys())}")
        
    def get_episode_data(
        self, 
        task_id: int, 
        episode_id: int, 
        include_images: bool = True,
        include_hidden_states: bool = True,
        include_actions: bool = True,
        include_vision_features: bool = True,
        layers: Optional[List[int]] = None,
        generation_steps: Optional[List[int]] = None
    ) -> Dict:
        """
        Get complete episode data including images and all trajectory data.
        
        Args:
            task_id: LIBERO task ID
            episode_id: Episode ID within the task
            include_images: Whether to load corresponding images
            include_hidden_states: Whether to include hidden states
            include_actions: Whether to include actions
            include_vision_features: Whether to include vision features
            layers: List of hidden state layer indices to include (default: all)
            generation_steps: List of generation step indices to include (default: all)
            
        Returns:
            Dictionary containing requested data types with perfect correspondence
        """
        print(f"[debug-image] Loading episode data for task_{task_id}/episode_{episode_id}")
        
        result = {
            'task_id': task_id,
            'episode_id': episode_id,
            'metadata': None
        }
        
        # Find episode metadata
        episode_meta = None
        for meta in self.dataset['metadata']:
            if meta['task_id'] == task_id and meta['episode_id'] == episode_id:
                episode_meta = meta
                break
        
        if episode_meta is None:
            raise ValueError(f"Episode task_{task_id}/episode_{episode_id} not found in dataset")
        
        result['metadata'] = episode_meta
        num_timesteps = episode_meta['num_timesteps']
        
        print(f"[debug-image] Episode has {num_timesteps} timesteps, success: {episode_meta['success']}")
        
        # Load images if requested
        if include_images:
            images = self._load_episode_images(task_id, episode_id, num_timesteps)
            result['images'] = images
            print(f"[debug-image] Loaded {len(images)} images")
        
        # Load actions if requested
        if include_actions and 'actions' in self.dataset:
            if task_id in self.dataset['actions'] and episode_id in self.dataset['actions'][task_id]:
                result['actions'] = self.dataset['actions'][task_id][episode_id]
                print(f"[debug-image] Loaded actions: shape {result['actions'].shape}")
            else:
                print(f"[debug-image] WARNING: No actions found for task_{task_id}/episode_{episode_id}")
        
        # Load vision features if requested
        if include_vision_features and 'vision_features' in self.dataset:
            if task_id in self.dataset['vision_features'] and episode_id in self.dataset['vision_features'][task_id]:
                result['vision_features'] = self.dataset['vision_features'][task_id][episode_id]
                print(f"[debug-image] Loaded vision features: shape {result['vision_features'].shape}")
            else:
                print(f"[debug-image] WARNING: No vision features found for task_{task_id}/episode_{episode_id}")
        
        # Load hidden states if requested
        if include_hidden_states and 'hidden_states' in self.dataset:
            hidden_states = {}
            available_layers = list(self.dataset['hidden_states'].keys())
            
            if layers is None:
                layers = available_layers
            else:
                layers = [l for l in layers if l in available_layers]
            
            for layer_idx in layers:
                if (task_id in self.dataset['hidden_states'][layer_idx] and 
                    episode_id in self.dataset['hidden_states'][layer_idx][task_id]):
                    
                    layer_data = {}
                    episode_layer_data = self.dataset['hidden_states'][layer_idx][task_id][episode_id]
                    
                    for timestep_id in episode_layer_data:
                        timestep_data = episode_layer_data[timestep_id]
                        
                        if generation_steps is None:
                            layer_data[timestep_id] = timestep_data
                        else:
                            filtered_data = {}
                            for gen_step in generation_steps:
                                if gen_step in timestep_data:
                                    filtered_data[gen_step] = timestep_data[gen_step]
                            layer_data[timestep_id] = filtered_data
                    
                    hidden_states[layer_idx] = layer_data
                    
            result['hidden_states'] = hidden_states
            print(f"[debug-image] Loaded hidden states for layers: {list(hidden_states.keys())}")
        
        return result
    
    def _load_episode_images(self, task_id: int, episode_id: int, num_timesteps: int) -> List[np.ndarray]:
        """Load all images for an episode."""
        episode_img_dir = self.images_dir / f"task_{task_id}" / f"episode_{episode_id}"
        
        if not episode_img_dir.exists():
            raise FileNotFoundError(f"Episode images not found: {episode_img_dir}")
        
        images = []
        for timestep in range(num_timesteps):
            img_path = episode_img_dir / f"timestep_{timestep:04d}.png"
            
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            img = Image.open(img_path)
            img_array = np.array(img)
            images.append(img_array)
        
        return images
    
    def get_matched_data_for_probing(
        self,
        layer_idx: int,
        generation_step: int = 0,
        successful_only: bool = True,
        include_images: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Get perfectly matched data for linear probing: hidden states, vision features, actions, and images.
        
        Args:
            layer_idx: Hidden state layer to extract
            generation_step: Generation step for hidden states  
            successful_only: Only include successful episodes
            include_images: Whether to load corresponding images
            
        Returns:
            Tuple of (hidden_states, vision_features, actions, images, metadata)
            - hidden_states: [N, hidden_dim] 
            - vision_features: [N, num_patches, vision_dim]
            - actions: [N, 7] 
            - images: [N, height, width, 3] (if include_images=True, else None)
            - metadata: List of sample metadata
        """
        print(f"[debug-image] Extracting matched data for layer {layer_idx}, generation step {generation_step}")
        
        all_hidden_states = []
        all_vision_features = []
        all_actions = []
        all_images = []
        all_metadata = []
        
        # Filter episodes by success if requested
        episodes_to_process = []
        for meta in self.dataset['metadata']:
            if not successful_only or meta['success']:
                episodes_to_process.append((meta['task_id'], meta['episode_id']))
        
        print(f"[debug-image] Processing {len(episodes_to_process)} episodes")
        
        for task_id, episode_id in episodes_to_process:
            try:
                # Get episode data
                episode_data = self.get_episode_data(
                    task_id, episode_id,
                    include_images=include_images,
                    layers=[layer_idx],
                    generation_steps=[generation_step]
                )
                
                if ('hidden_states' not in episode_data or 
                    'actions' not in episode_data or 
                    'vision_features' not in episode_data):
                    print(f"[debug-image] Skipping task_{task_id}/episode_{episode_id} - missing data")
                    continue
                
                # Extract hidden states for this layer/generation step
                if (layer_idx in episode_data['hidden_states']):
                    layer_data = episode_data['hidden_states'][layer_idx]
                    episode_hidden_states = []
                    
                    for timestep_id in sorted(layer_data.keys()):
                        if generation_step in layer_data[timestep_id]:
                            hidden_state = layer_data[timestep_id][generation_step]
                            # Flatten: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim) -> (seq_len * hidden_dim,)
                            flattened = hidden_state.reshape(-1, hidden_state.shape[-1])
                            episode_hidden_states.append(flattened)
                    
                    if episode_hidden_states:
                        episode_hidden_states = np.vstack(episode_hidden_states)
                        all_hidden_states.append(episode_hidden_states)
                        
                        # Add corresponding vision features, actions, and images
                        all_vision_features.append(episode_data['vision_features'])
                        all_actions.append(episode_data['actions'])
                        
                        if include_images:
                            all_images.append(np.stack(episode_data['images']))
                        
                        # Add metadata for each timestep
                        for timestep in range(len(episode_data['actions'])):
                            sample_meta = {
                                'task_id': task_id,
                                'episode_id': episode_id,
                                'timestep_id': timestep,
                                'layer_idx': layer_idx,
                                'generation_step': generation_step,
                                'task_description': episode_data['metadata']['task_description'],
                                'success': episode_data['metadata']['success']
                            }
                            all_metadata.append(sample_meta)
                            
            except Exception as e:
                print(f"[debug-image] ERROR processing task_{task_id}/episode_{episode_id}: {e}")
                continue
        
        # Stack all data
        if not all_hidden_states:
            return np.array([]), np.array([]), np.array([]), None, []
        
        stacked_hidden_states = np.vstack(all_hidden_states)
        stacked_vision_features = np.vstack(all_vision_features) 
        stacked_actions = np.vstack(all_actions)
        stacked_images = np.vstack(all_images) if include_images else None
        
        print(f"[debug-image] Final data shapes:")
        print(f"[debug-image] Hidden states: {stacked_hidden_states.shape}")
        print(f"[debug-image] Vision features: {stacked_vision_features.shape}")
        print(f"[debug-image] Actions: {stacked_actions.shape}")
        if include_images:
            print(f"[debug-image] Images: {stacked_images.shape}")
        print(f"[debug-image] Metadata samples: {len(all_metadata)}")
        
        return stacked_hidden_states, stacked_vision_features, stacked_actions, stacked_images, all_metadata


def main():
    parser = argparse.ArgumentParser(description='Load trajectory data with images for linear probing')
    parser.add_argument('dataset_path', help='Path to trajectory data HDF5 file')
    parser.add_argument('images_dir', help='Directory containing generated images')
    parser.add_argument('--task-id', type=int, help='Specific task ID to load')
    parser.add_argument('--episode-id', type=int, help='Specific episode ID to load')
    parser.add_argument('--layer', type=int, default=0, help='Hidden state layer for probing example')
    
    args = parser.parse_args()
    
    # Initialize unified loader
    data_loader = TrajectoryDataWithImages(args.dataset_path, args.images_dir)
    
    if args.task_id is not None and args.episode_id is not None:
        # Load specific episode
        episode_data = data_loader.get_episode_data(args.task_id, args.episode_id)
        
        print(f"\n=== Episode Data ===")
        print(f"Task: {episode_data['metadata']['task_description']}")
        print(f"Success: {episode_data['metadata']['success']}")
        print(f"Timesteps: {episode_data['metadata']['num_timesteps']}")
        
        if 'images' in episode_data:
            print(f"Images: {len(episode_data['images'])} loaded, shape: {episode_data['images'][0].shape}")
        if 'actions' in episode_data:
            print(f"Actions: {episode_data['actions'].shape}")
        if 'vision_features' in episode_data:
            print(f"Vision features: {episode_data['vision_features'].shape}")
        if 'hidden_states' in episode_data:
            print(f"Hidden states layers: {list(episode_data['hidden_states'].keys())}")
    
    else:
        # Demonstrate matched data for probing
        print(f"\n=== Linear Probing Data Example ===")
        hidden_states, vision_features, actions, images, metadata = data_loader.get_matched_data_for_probing(
            layer_idx=args.layer,
            generation_step=0,
            include_images=True
        )
        
        print(f"\nPerfectly matched data ready for linear probing:")
        print(f"- Hidden states: {hidden_states.shape}")  
        print(f"- Vision features: {vision_features.shape}")
        print(f"- Actions: {actions.shape}")
        if images is not None:
            print(f"- Images: {images.shape}")
        print(f"- Metadata samples: {len(metadata)}")
        
        if len(metadata) > 0:
            print(f"\nFirst sample metadata: {metadata[0]}")


if __name__ == "__main__":
    main()