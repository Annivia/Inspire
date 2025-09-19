#!/usr/bin/env python3
"""
experiment_1_hidden_to_actions.py

Experiment 1: [Hidden state] -> actions
Train a linear regression probe for every layer's hidden state on every timestep.
Evaluate with R2/MSE on held-out trajectories.

Baselines:
- Normal: Original data
- Randomized pairs: randomly shuffle hidden states and action sequences on a trajectory basis  
- Noise baseline: [Hidden state] -> gaussian noise with same dim as actions

This tests what action information is linearly accessible in each layer's hidden states.
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
    load_trajectory_dataset, get_layer_data_flat, get_actions_data_flat
)
from probing.linear_probe import run_probe_with_baselines, save_probe_results


def run_experiment_1(
    data_path: str,
    output_dir: str,
    layers: Optional[List[int]] = None,
    generation_steps: Optional[List[int]] = None,
    successful_only: bool = True,
    max_episodes: Optional[int] = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    debug: bool = False
) -> Dict:
    """
    Run Experiment 1: Test what action information is linearly accessible 
    in hidden states from different layers.
    
    Args:
        data_path: Path to trajectory data HDF5 file
        output_dir: Directory to save results
        layers: List of layer indices to analyze (None = all layers)
        generation_steps: List of generation steps to analyze (None = [0])
        successful_only: Only use successful episodes
        max_episodes: Maximum number of episodes to use
        test_size: Fraction of data for testing
        random_seed: Random seed for reproducibility
        debug: Enable debug output
        
    Returns:
        Dictionary containing all experiment results
    """
    if debug:
        print(f"[DEBUG] Starting Experiment 1: [Hidden state] -> actions")
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
            layers=layers,
            successful_only=successful_only,
            load_hidden_states=True,  # Need hidden states
            load_actions=True,        # Need actions as targets
            load_vision_features=False,  # Skip vision features (not needed)
            load_vlm_embeddings=False    # Skip VLM embeddings (not needed)
        )
    except Exception as e:
        if debug:
            print(f"[DEBUG] ERROR loading dataset: {e}")
        raise
    
    if debug:
        print(f"[DEBUG] Dataset loaded successfully")
        print(f"[DEBUG] Available layers: {dataset['summary']['layers']}")
        print(f"[DEBUG] Total episodes: {dataset['summary']['loaded_episodes']}")
    
    # Extract actions data (target for all probes)
    if debug:
        print(f"[DEBUG] Extracting actions data...")
    
    # Use three targets from the action horizon: first, middle, and last
    actions_flat, actions_metadata = get_actions_data_flat(
        dataset, include_metadata=True, selection='first_middle_last'
    )
    
    if len(actions_flat) == 0:
        raise ValueError("No actions data found in dataset")
    
    if debug:
        print(f"[DEBUG] Actions shape: {actions_flat.shape}")
        print(f"[DEBUG] Actions range: [{actions_flat.min():.3f}, {actions_flat.max():.3f}]")
        print(f"[DEBUG] Actions mean: {actions_flat.mean(axis=0)}")
        print(f"[DEBUG] Actions std: {actions_flat.std(axis=0)}")
    
    # Limit episodes if requested
    if max_episodes is not None and len(actions_flat) > max_episodes:
        if debug:
            print(f"[DEBUG] Limiting to {max_episodes} episodes")
        indices = np.random.RandomState(random_seed).choice(len(actions_flat), max_episodes, replace=False)
        actions_flat = actions_flat[indices]
        actions_metadata = [actions_metadata[i] for i in indices]
    
    # Set default generation steps
    if generation_steps is None:
        generation_steps = [0]  # Full input processing step
    
    if debug:
        print(f"[DEBUG] Testing generation steps: {generation_steps}")
    
    # Results storage
    experiment_results = {
        'experiment_id': 1,
        'experiment_name': 'hidden_state_to_actions',
        'description': 'Linear probes from hidden states to actions',
        'data_path': data_path,
        'dataset_summary': dataset['summary'],
        'config': {
            'layers': layers,
            'generation_steps': generation_steps,
            'action_selection': 'first_middle_last',
            'successful_only': successful_only,
            'max_episodes': max_episodes,
            'test_size': test_size,
            'random_seed': random_seed,
            'final_actions_shape': actions_flat.shape
        },
        'results_by_layer': {},
        'timestamp': time.time()
    }
    
    # Run probes for each layer and generation step
    available_layers = dataset['summary']['layers']
    test_layers = layers if layers is not None else available_layers
    
    if debug:
        print(f"[DEBUG] Running probes for layers: {test_layers}")
    
    for layer_idx in test_layers:
        if layer_idx not in available_layers:
            if debug:
                print(f"[DEBUG] Layer {layer_idx} not available, skipping")
            continue
            
        if debug:
            print(f"[DEBUG] Processing layer {layer_idx}...")
        
        layer_results = {}
        
        for gen_step in generation_steps:
            if debug:
                print(f"[DEBUG] Processing layer {layer_idx}, generation step {gen_step}...")
            
            # Extract hidden states for this layer and generation step

            hidden_states_flat, hidden_metadata = get_layer_data_flat(
                dataset, layer_idx=layer_idx, generation_step=gen_step, include_metadata=True
            )
            
            if len(hidden_states_flat) == 0:
                if debug:
                    print(f"[DEBUG] No data for layer {layer_idx}, gen step {gen_step}")
                continue
            
            if debug:
                print(f"[DEBUG] Hidden states shape: {hidden_states_flat.shape}")
                print(f"[DEBUG] Hidden states range: [{hidden_states_flat.min():.3f}, {hidden_states_flat.max():.3f}]")
            
            # Ensure we have matching data
            min_samples = min(len(hidden_states_flat), len(actions_flat))
            hidden_states_matched = hidden_states_flat[:min_samples]
            actions_matched = actions_flat[:min_samples]
            
            if debug:
                print(f"[DEBUG] Using {min_samples} matched samples")
                print(f"[DEBUG] Final shapes - Hidden: {hidden_states_matched.shape}, Actions: {actions_matched.shape}")
            
            # Run linear probes with all baselines
            probe_name = f"layer_{layer_idx}_gen_{gen_step}_to_actions"
            
            probe_results = run_probe_with_baselines(
                X=hidden_states_matched,
                y=actions_matched, 
                probe_name=probe_name,
                task_type='regression',
                test_size=test_size,
                random_seed=random_seed,
                debug=debug
            )
            
            layer_results[f'generation_step_{gen_step}'] = probe_results
            
            # Save individual probe results
            probe_file = output_path / f'layer_{layer_idx}_gen_{gen_step}_results.json'
            save_probe_results(probe_results, probe_file)
            
            if debug:
                if 'summary' in probe_results:
                    summary = probe_results['summary']
                    print(f"[DEBUG] Layer {layer_idx} Gen {gen_step} Results:")
                    print(f"[DEBUG] - Normal R2: {probe_results['normal']['r2_test']:.4f}")
                    print(f"[DEBUG] - Random R2: {probe_results['randomized']['r2_test']:.4f}")  
                    print(f"[DEBUG] - Noise R2: {probe_results['noise']['r2_test']:.4f}")
                    print(f"[DEBUG] - Linear separability strength: {summary['linear_separability_strength']:.4f}")

        experiment_results['results_by_layer'][f'layer_{layer_idx}'] = layer_results
    
    # Compute experiment-wide summary statistics
    if debug:
        print(f"[DEBUG] Computing experiment summary...")
    
    summary_stats = compute_experiment_summary(experiment_results, debug=debug)
    experiment_results['experiment_summary'] = summary_stats
    
    # Save complete experiment results
    experiment_file = output_path / 'experiment_1_complete_results.json'
    save_probe_results(experiment_results, experiment_file)
    
    elapsed_time = time.time() - start_time
    if debug:
        print(f"[DEBUG] Experiment 1 completed in {elapsed_time:.2f} seconds")
        print(f"[DEBUG] Results saved to: {output_path}")
    
    return experiment_results


def compute_experiment_summary(results: Dict, debug: bool = False) -> Dict:
    """
    Compute summary statistics across all layers and generation steps.
    
    Args:
        results: Complete experiment results
        debug: Enable debug output
        
    Returns:
        Dictionary with summary statistics
    """
    if debug:
        print(f"[DEBUG] Computing summary statistics...")
    
    all_normal_r2 = []
    all_random_r2 = []
    all_noise_r2 = []
    best_layer_performance = {}
    
    for layer_key, layer_data in results['results_by_layer'].items():
        layer_idx = layer_key.split('_')[1]
        
        for gen_step_key, probe_data in layer_data.items():
            if 'error' in probe_data:
                continue
                
            if 'normal' in probe_data:
                normal_r2 = probe_data['normal'].get('r2_test', 0)
                random_r2 = probe_data['randomized'].get('r2_test', 0)
                noise_r2 = probe_data['noise'].get('r2_test', 0)
                
                all_normal_r2.append(normal_r2)
                all_random_r2.append(random_r2)
                all_noise_r2.append(noise_r2)
                
                # Track best performance per layer
                if layer_idx not in best_layer_performance or normal_r2 > best_layer_performance[layer_idx]['r2']:
                    best_layer_performance[layer_idx] = {
                        'r2': normal_r2,
                        'generation_step': gen_step_key,
                        'probe_data': probe_data
                    }
    
    if not all_normal_r2:
        return {'error': 'No valid probe results found'}
    
    # Find best overall performance
    best_r2_idx = np.argmax(all_normal_r2)
    best_r2 = all_normal_r2[best_r2_idx]
    
    summary = {
        'total_probes_run': len(all_normal_r2),
        'normal_r2_stats': {
            'mean': np.mean(all_normal_r2),
            'std': np.std(all_normal_r2),
            'min': np.min(all_normal_r2),
            'max': np.max(all_normal_r2),
            'median': np.median(all_normal_r2)
        },
        'random_r2_stats': {
            'mean': np.mean(all_random_r2),
            'std': np.std(all_random_r2),
            'min': np.min(all_random_r2),
            'max': np.max(all_random_r2)
        },
        'noise_r2_stats': {
            'mean': np.mean(all_noise_r2),
            'std': np.std(all_noise_r2),
            'min': np.min(all_noise_r2),
            'max': np.max(all_noise_r2)
        },
        'best_overall_r2': best_r2,
        'mean_normal_vs_random_improvement': np.mean(all_normal_r2) - np.mean(all_random_r2),
        'mean_normal_vs_noise_improvement': np.mean(all_normal_r2) - np.mean(all_noise_r2),
        'best_layer_performance': best_layer_performance
    }
    
    if debug:
        print(f"[DEBUG] Summary computed:")
        print(f"[DEBUG] - Total probes: {summary['total_probes_run']}")
        print(f"[DEBUG] - Best R2: {summary['best_overall_r2']:.4f}")
        print(f"[DEBUG] - Mean normal R2: {summary['normal_r2_stats']['mean']:.4f}")
        print(f"[DEBUG] - Mean random R2: {summary['random_r2_stats']['mean']:.4f}")
        print(f"[DEBUG] - Mean noise R2: {summary['noise_r2_stats']['mean']:.4f}")
    
    return summary
