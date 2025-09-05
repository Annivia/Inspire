#!/usr/bin/env python3
"""
experiment_2_vision_to_actions.py

Experiment 2: [Vision encoder outputs] -> actions
Train linear regression probes for vision encoder patch features.
Evaluate with R2/MSE on held-out trajectories.

Baselines:
- Normal: Original data
- Randomized pairs: randomly shuffle vision features and action sequences on a trajectory basis  
- Noise baseline: [Vision features] -> gaussian noise with same dim as actions

This tests what action information is linearly accessible in vision encoder representations.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import time

# Add project paths
sys.path.append('/u/xzhang42/Inspire')
sys.path.append('/u/xzhang42/Inspire/vla_scripts')

from vla_scripts.load_optimized_trajectory_data import (
    load_trajectory_dataset, get_vision_features_data_flat, get_actions_data_flat
)
from probing.linear_probe import run_probe_with_baselines, save_probe_results


def run_experiment_2(
    data_path: str,
    output_dir: str,
    successful_only: bool = True,
    max_episodes: Optional[int] = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    debug: bool = False
) -> Dict:
    """
    Run Experiment 2: Test what action information is linearly accessible 
    in vision encoder outputs.
    
    Args:
        data_path: Path to trajectory data HDF5 file
        output_dir: Directory to save results
        successful_only: Only use successful episodes
        max_episodes: Maximum number of episodes to use
        test_size: Fraction of data for testing
        random_seed: Random seed for reproducibility
        debug: Enable debug output
        
    Returns:
        Dictionary containing all experiment results
    """
    if debug:
        print(f"[DEBUG] Starting Experiment 2: [Vision encoder outputs] -> actions")
        print(f"[DEBUG] Data path: {data_path}")
        print(f"[DEBUG] Output directory: {output_dir}")
    
    start_time = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load trajectory dataset
    if debug:
        print(f"[DEBUG] Loading trajectory dataset...")
    
    try:
        dataset = load_trajectory_dataset(
            data_path=data_path,
            successful_only=successful_only
        )
    except Exception as e:
        if debug:
            print(f"[DEBUG] ERROR loading dataset: {e}")
        raise
    
    if debug:
        print(f"[DEBUG] Dataset loaded successfully")
        print(f"[DEBUG] Total episodes: {dataset['summary']['loaded_episodes']}")
    
    # Extract actions data (target for all probes) 
    if debug:
        print(f"[DEBUG] Extracting actions data...")
    
    actions_flat, actions_metadata = get_actions_data_flat(dataset, include_metadata=True)
    
    if len(actions_flat) == 0:
        raise ValueError("No actions data found in dataset")
    
    if debug:
        print(f"[DEBUG] Actions shape: {actions_flat.shape}")
        print(f"[DEBUG] Actions range: [{actions_flat.min():.3f}, {actions_flat.max():.3f}]")
        print(f"[DEBUG] Actions mean: {actions_flat.mean(axis=0)}")
        print(f"[DEBUG] Actions std: {actions_flat.std(axis=0)}")
    
    # Extract vision features data (input for probes)
    if debug:
        print(f"[DEBUG] Extracting vision features data...")
    
    vision_features_flat, vision_metadata = get_vision_features_data_flat(dataset, include_metadata=True)
    
    if len(vision_features_flat) == 0:
        raise ValueError("No vision features data found in dataset")
    
    if debug:
        print(f"[DEBUG] Vision features shape: {vision_features_flat.shape}")
        print(f"[DEBUG] Vision features range: [{vision_features_flat.min():.3f}, {vision_features_flat.max():.3f}]")
    
    # Handle vision features shape - flatten patch dimensions if needed
    if len(vision_features_flat.shape) == 4:
        # Shape: (N_samples, 1, num_patches, feature_dim) -> (N_samples, num_patches * feature_dim)
        vision_features_flat = vision_features_flat.reshape(vision_features_flat.shape[0], -1)
        if debug:
            print(f"[DEBUG] Flattened vision features to: {vision_features_flat.shape}")
    elif len(vision_features_flat.shape) == 3:
        # Shape: (N_samples, num_patches, feature_dim) -> (N_samples, num_patches * feature_dim)  
        vision_features_flat = vision_features_flat.reshape(vision_features_flat.shape[0], -1)
        if debug:
            print(f"[DEBUG] Flattened vision features to: {vision_features_flat.shape}")
    
    # Limit episodes if requested
    if max_episodes is not None and len(actions_flat) > max_episodes:
        if debug:
            print(f"[DEBUG] Limiting to {max_episodes} episodes")
        indices = np.random.RandomState(random_seed).choice(len(actions_flat), max_episodes, replace=False)
        actions_flat = actions_flat[indices]
        vision_features_flat = vision_features_flat[indices]
        actions_metadata = [actions_metadata[i] for i in indices]
    
    # Ensure we have matching data
    min_samples = min(len(vision_features_flat), len(actions_flat))
    vision_features_matched = vision_features_flat[:min_samples]
    actions_matched = actions_flat[:min_samples]
    
    if debug:
        print(f"[DEBUG] Using {min_samples} matched samples")
        print(f"[DEBUG] Final shapes - Vision: {vision_features_matched.shape}, Actions: {actions_matched.shape}")
    
    # Results storage
    experiment_results = {
        'experiment_id': 2,
        'experiment_name': 'vision_encoder_to_actions',
        'description': 'Linear probes from vision encoder outputs to actions',
        'data_path': data_path,
        'dataset_summary': dataset['summary'],
        'config': {
            'successful_only': successful_only,
            'max_episodes': max_episodes,
            'test_size': test_size,
            'random_seed': random_seed,
            'final_vision_shape': vision_features_matched.shape,
            'final_actions_shape': actions_matched.shape
        },
        'results': {},
        'timestamp': time.time()
    }
    
    # Run vision encoder probe
    if debug:
        print(f"[DEBUG] Running vision encoder probe...")
    
    try:
        probe_name = "vision_encoder_to_actions"
        
        probe_results = run_probe_with_baselines(
            X=vision_features_matched,
            y=actions_matched, 
            probe_name=probe_name,
            task_type='regression',
            test_size=test_size,
            random_seed=random_seed,
            debug=debug
        )
        
        experiment_results['results'] = probe_results
        
        # Save probe results
        probe_file = output_path / 'vision_encoder_results.json'
        save_probe_results(probe_results, probe_file)
        
        if debug:
            if 'summary' in probe_results:
                summary = probe_results['summary']
                print(f"[DEBUG] Vision Encoder Results:")
                print(f"[DEBUG] - Normal R2: {probe_results['normal']['r2_test']:.4f}")
                print(f"[DEBUG] - Random R2: {probe_results['randomized']['r2_test']:.4f}")  
                print(f"[DEBUG] - Noise R2: {probe_results['noise']['r2_test']:.4f}")
                print(f"[DEBUG] - Linear separability strength: {summary['linear_separability_strength']:.4f}")
                
    except Exception as e:
        if debug:
            print(f"[DEBUG] ERROR processing vision encoder probe: {e}")
        experiment_results['results'] = {'error': str(e)}
    
    # Compute experiment summary
    if debug:
        print(f"[DEBUG] Computing experiment summary...")
    
    summary_stats = compute_experiment_summary(experiment_results, debug=debug)
    experiment_results['experiment_summary'] = summary_stats
    
    # Save complete experiment results
    experiment_file = output_path / 'experiment_2_complete_results.json'
    save_probe_results(experiment_results, experiment_file)
    
    elapsed_time = time.time() - start_time
    if debug:
        print(f"[DEBUG] Experiment 2 completed in {elapsed_time:.2f} seconds")
        print(f"[DEBUG] Results saved to: {output_path}")
    
    return experiment_results


def compute_experiment_summary(results: Dict, debug: bool = False) -> Dict:
    """
    Compute summary statistics for vision encoder experiment.
    
    Args:
        results: Complete experiment results
        debug: Enable debug output
        
    Returns:
        Dictionary with summary statistics
    """
    if debug:
        print(f"[DEBUG] Computing summary statistics...")
    
    if 'error' in results['results']:
        return {'error': 'No valid probe results found'}
    
    probe_data = results['results']
    
    if 'normal' in probe_data:
        normal_r2 = probe_data['normal'].get('r2_test', 0)
        random_r2 = probe_data['randomized'].get('r2_test', 0)
        noise_r2 = probe_data['noise'].get('r2_test', 0)
        
        summary = {
            'total_probes_run': 1,
            'normal_r2': normal_r2,
            'random_r2': random_r2, 
            'noise_r2': noise_r2,
            'best_r2': normal_r2,
            'normal_vs_random_improvement': normal_r2 - random_r2,
            'normal_vs_noise_improvement': normal_r2 - noise_r2,
            'linear_separability_strength': normal_r2
        }
        
        if debug:
            print(f"[DEBUG] Summary computed:")
            print(f"[DEBUG] - Normal R2: {normal_r2:.4f}")
            print(f"[DEBUG] - Random R2: {random_r2:.4f}")
            print(f"[DEBUG] - Noise R2: {noise_r2:.4f}")
            print(f"[DEBUG] - Linear separability strength: {normal_r2:.4f}")
        
        return summary
    else:
        return {'error': 'No valid probe results found'}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Experiment 2: Vision encoder to actions probing")
    parser.add_argument("data_path", help="Path to optimized trajectory data directory")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--successful-only", action='store_true', default=True,
                       help="Only use successful episodes")
    parser.add_argument("--max-episodes", type=int, help="Maximum episodes to use")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set fraction")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--debug", action='store_true',
                       help="Enable debug output")
    
    args = parser.parse_args()
    
    try:
        results = run_experiment_2(
            data_path=args.data_path,
            output_dir=args.output_dir,
            successful_only=args.successful_only,
            max_episodes=args.max_episodes,
            test_size=args.test_size,
            random_seed=args.random_seed,
            debug=args.debug
        )
        
        print(f"\nExperiment 2 completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        if 'experiment_summary' in results and 'error' not in results['experiment_summary']:
            summary = results['experiment_summary']
            print(f"\nSummary:")
            print(f"- Normal R2: {summary['normal_r2']:.4f}")
            print(f"- Random R2: {summary['random_r2']:.4f}")
            print(f"- Noise R2: {summary['noise_r2']:.4f}")
            print(f"- Linear separability strength: {summary['linear_separability_strength']:.4f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()