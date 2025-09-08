#!/usr/bin/env python3
"""
experiment_2_vision_to_actions.py

Experiment 2: [Vision encoder outputs] -> actions
Train linear regression probes for vision encoder features:
- Raw patch features from vision backbone (DINOv2/CLIP/SigLIP etc.)
- VLM-transformed visual embeddings (after projector)

Evaluate with R2/MSE on held-out trajectories.

Baselines:
- Normal: Original data
- Randomized pairs: randomly shuffle vision features and action sequences on a trajectory basis  
- Noise baseline: [Vision features] -> gaussian noise with same dim as actions

This tests what action information is linearly accessible in different vision representations.
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
    load_trajectory_dataset, get_vision_features_data_flat, get_actions_data_flat, get_vlm_embeddings_data_flat
)
from probing.linear_probe import run_probe_with_baselines, save_probe_results


def run_experiment_2(
    data_path: str,
    output_dir: str,
    vision_type: str = "both",  # "raw", "vlm", or "both"
    successful_only: bool = True,
    max_episodes: Optional[int] = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    debug: bool = False
) -> Dict:
    """
    Run Experiment 2: Test what action information is linearly accessible 
    in vision encoder outputs (both raw patches and VLM embeddings).
    
    Args:
        data_path: Path to trajectory data HDF5 file
        output_dir: Directory to save results
        vision_type: Which vision features to probe ("raw", "vlm", or "both")
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
    
    # Determine what vision data to load based on vision_type
    load_vision_raw = vision_type in ["raw", "both"]
    load_vision_vlm = vision_type in ["vlm", "both"]
    
    try:
        dataset = load_trajectory_dataset(
            data_path=data_path,
            successful_only=successful_only,
            load_hidden_states=False,        # Skip hidden states (not needed)
            load_actions=True,               # Need actions as targets
            load_vision_features=load_vision_raw,    # Load raw patches if requested
            load_vlm_embeddings=load_vision_vlm      # Load VLM embeddings if requested
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
    
    # Extract and prepare vision features based on vision_type
    vision_experiments = {}
    
    if vision_type in ["raw", "both"]:
        if debug:
            print(f"[DEBUG] Extracting raw vision features data...")
        
        raw_vision_flat, raw_vision_metadata = get_vision_features_data_flat(dataset, include_metadata=True)
        
        if len(raw_vision_flat) == 0:
            print(f"WARNING: No raw vision features data found")
        else:
            # Handle raw vision features shape - flatten patch dimensions if needed
            if len(raw_vision_flat.shape) == 4:
                # Shape: (N_samples, 1, num_patches, feature_dim) -> (N_samples, num_patches * feature_dim)
                raw_vision_flat = raw_vision_flat.reshape(raw_vision_flat.shape[0], -1)
            elif len(raw_vision_flat.shape) == 3:
                # Shape: (N_samples, num_patches, feature_dim) -> (N_samples, num_patches * feature_dim)  
                raw_vision_flat = raw_vision_flat.reshape(raw_vision_flat.shape[0], -1)
            
            vision_experiments['raw_patches'] = raw_vision_flat
            
            if debug:
                print(f"[DEBUG] Raw vision features shape: {raw_vision_flat.shape}")
                print(f"[DEBUG] Raw vision features range: [{raw_vision_flat.min():.3f}, {raw_vision_flat.max():.3f}]")
    
    if vision_type in ["vlm", "both"]:
        if debug:
            print(f"[DEBUG] Extracting VLM embeddings data...")
        
        vlm_embeddings_flat, vlm_metadata = get_vlm_embeddings_data_flat(dataset, include_metadata=True)
        
        if len(vlm_embeddings_flat) == 0:
            print(f"WARNING: No VLM embeddings data found")
        else:
            # VLM embeddings should already be properly shaped: (N_samples, embed_dim)
            # But handle potential extra dimensions
            if len(vlm_embeddings_flat.shape) > 2:
                vlm_embeddings_flat = vlm_embeddings_flat.reshape(vlm_embeddings_flat.shape[0], -1)
            
            vision_experiments['vlm_embeddings'] = vlm_embeddings_flat
            
            if debug:
                print(f"[DEBUG] VLM embeddings shape: {vlm_embeddings_flat.shape}")
                print(f"[DEBUG] VLM embeddings range: [{vlm_embeddings_flat.min():.3f}, {vlm_embeddings_flat.max():.3f}]")
    
    if len(vision_experiments) == 0:
        raise ValueError(f"No vision data found for vision_type: {vision_type}")
    
    # Limit episodes if requested (apply to all vision types)
    if max_episodes is not None:
        n_actions = len(actions_flat)
        if n_actions > max_episodes:
            if debug:
                print(f"[DEBUG] Limiting to {max_episodes} episodes from {n_actions}")
            indices = np.random.RandomState(random_seed).choice(n_actions, max_episodes, replace=False)
            actions_flat = actions_flat[indices]
            actions_metadata = [actions_metadata[i] for i in indices]
            
            # Apply same indices to all vision experiment data
            for exp_name, vision_data in vision_experiments.items():
                vision_experiments[exp_name] = vision_data[indices]
    
    if debug:
        print(f"[DEBUG] Final shapes - Actions: {actions_flat.shape}")
        for exp_name, vision_data in vision_experiments.items():
            print(f"[DEBUG] Final shapes - {exp_name}: {vision_data.shape}")
    
    # Results storage
    experiment_results = {
        'experiment_id': 2,
        'experiment_name': 'vision_encoder_to_actions',
        'description': 'Linear probes from vision encoder outputs to actions (both raw patches and VLM embeddings)',
        'data_path': data_path,
        'dataset_summary': dataset['summary'],
        'config': {
            'vision_type': vision_type,
            'successful_only': successful_only,
            'max_episodes': max_episodes,
            'test_size': test_size,
            'random_seed': random_seed,
            'final_actions_shape': actions_flat.shape,
            'vision_experiments': {name: data.shape for name, data in vision_experiments.items()}
        },
        'results': {},
        'timestamp': time.time()
    }
    
    # Run vision encoder probes for each vision experiment
    if debug:
        print(f"[DEBUG] Running vision encoder probes for {list(vision_experiments.keys())}...")
    
    all_probe_results = {}
    
    for exp_name, vision_features in vision_experiments.items():
        if debug:
            print(f"[DEBUG] Running probe for {exp_name}...")
        
        try:
            # Ensure we have matching data
            min_samples = min(len(vision_features), len(actions_flat))
            vision_matched = vision_features[:min_samples]
            actions_matched = actions_flat[:min_samples]
            
            probe_name = f"{exp_name}_to_actions"
            
            probe_results = run_probe_with_baselines(
                X=vision_matched,
                y=actions_matched, 
                probe_name=probe_name,
                task_type='regression',
                test_size=test_size,
                random_seed=random_seed,
                debug=debug
            )
            
            all_probe_results[exp_name] = probe_results
            
            # Save individual probe results
            probe_file = output_path / f'{exp_name}_results.json'
            save_probe_results(probe_results, probe_file)
            
            if debug:
                if 'summary' in probe_results:
                    summary = probe_results['summary']
                    print(f"[DEBUG] {exp_name} Results:")
                    print(f"[DEBUG] - Normal R2: {probe_results['normal']['r2_test']:.4f}")
                    print(f"[DEBUG] - Random R2: {probe_results['randomized']['r2_test']:.4f}")  
                    print(f"[DEBUG] - Noise R2: {probe_results['noise']['r2_test']:.4f}")
                    print(f"[DEBUG] - Linear separability strength: {summary['linear_separability_strength']:.4f}")
                    
        except Exception as e:
            if debug:
                print(f"[DEBUG] ERROR processing {exp_name} probe: {e}")
            all_probe_results[exp_name] = {'error': str(e)}
    
    experiment_results['results'] = all_probe_results
    
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
    Compute summary statistics for vision encoder experiment (multiple vision types).
    
    Args:
        results: Complete experiment results
        debug: Enable debug output
        
    Returns:
        Dictionary with summary statistics for each vision experiment
    """
    if debug:
        print(f"[DEBUG] Computing summary statistics...")
    
    all_probe_results = results['results']
    experiment_summaries = {}
    
    for exp_name, probe_data in all_probe_results.items():
        if 'error' in probe_data:
            experiment_summaries[exp_name] = {'error': 'No valid probe results found'}
            continue
        
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
            
            experiment_summaries[exp_name] = summary
            
            if debug:
                print(f"[DEBUG] Summary computed for {exp_name}:")
                print(f"[DEBUG] - Normal R2: {normal_r2:.4f}")
                print(f"[DEBUG] - Random R2: {random_r2:.4f}")
                print(f"[DEBUG] - Noise R2: {noise_r2:.4f}")
                print(f"[DEBUG] - Linear separability strength: {normal_r2:.4f}")
        else:
            experiment_summaries[exp_name] = {'error': 'No valid probe results found'}
    
    return experiment_summaries


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Experiment 2: Vision encoder to actions probing")
    parser.add_argument("data_path", help="Path to optimized trajectory data directory")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--vision-type", choices=["raw", "vlm", "both"], default="both",
                       help="Which vision features to probe (default: both)")
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
            vision_type=args.vision_type,
            successful_only=args.successful_only,
            max_episodes=args.max_episodes,
            test_size=args.test_size,
            random_seed=args.random_seed,
            debug=args.debug
        )
        
        print(f"\nExperiment 2 completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        if 'experiment_summary' in results:
            print(f"\nSummary:")
            for exp_name, summary in results['experiment_summary'].items():
                if 'error' not in summary:
                    print(f"\n{exp_name}:")
                    print(f"- Normal R2: {summary['normal_r2']:.4f}")
                    print(f"- Random R2: {summary['random_r2']:.4f}")
                    print(f"- Noise R2: {summary['noise_r2']:.4f}")
                    print(f"- Linear separability strength: {summary['linear_separability_strength']:.4f}")
                else:
                    print(f"\n{exp_name}: {summary['error']}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()