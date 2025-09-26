#!/usr/bin/env python3
"""
probe.py

Main script for running linear probing experiments on VLA trajectory data.
Implements the four experiments specified in probing/README.md with comprehensive
baseline comparisons and result logging.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import time

# Add project root to path for imports
sys.path.append('/u/xzhang42/Inspire')

from probing.experiments.experiment_1_hidden_to_actions import run_experiment_1
from probing.experiments.experiment_2_vision_to_actions import run_experiment_2
from probing.experiments.experiment_3_hidden_to_concepts import run_experiment_3_general_1
from probing.experiments.experiment_4_vision_to_concepts import run_experiment_4_general_1


def main():
    parser = argparse.ArgumentParser(description='Run linear probing experiments on VLA trajectory data')
    
    # Data and experiment configuration
    parser.add_argument('--data-path', required=True, 
                       help='Path to optimized trajectory data directory')
    parser.add_argument('--experiment', type=int, required=True, choices=[1, 2, 3, 4],
                       help='Experiment number (1: hidden->actions, 2: vision->actions, 3: hidden->concepts, 4: vision->concepts)')
    parser.add_argument('--output-dir', default='./results',
                       help='Directory to save results and visualizations')
    
    # Data filtering options
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                       help='Hidden state layers to analyze (default: all layers)')
    parser.add_argument('--generation-steps', nargs='+', type=int, default=[0],
                       help='Generation steps to analyze (default: [0] for full input processing)')
    parser.add_argument('--successful-only', action='store_true', default=True,
                       help='Only use successful episodes (default: True)')
    parser.add_argument('--max-episodes', type=int, default=None,
                       help='Maximum number of episodes to use (for testing)')
    
    # Model configuration
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Experiment 2 specific options
    parser.add_argument('--vision-type', choices=["raw", "vlm", "both"], default="both",
                       help='Which vision features to probe for experiment 2 (default: both)')
    # Experiment 3/4 task category (currently only general_1 is supported)
    parser.add_argument('--task-category', choices=["general_1"], default="general_1",
                       help='Which probing task category to run (for exp 3/4).')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    
    args = parser.parse_args()
    
    print(f"=== VLA Linear Probing Experiment {args.experiment} ===")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Debug mode: {args.debug}")
    print(f"Test size: {args.test_size}")
    print(f"Random seed: {args.random_seed}")
    print(f"Layers: {args.layers}")
    print(f"Generation steps: {args.generation_steps}")
    print(f"Successful only: {args.successful_only}")
    print(f"Max episodes: {args.max_episodes}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Created output directory: {output_path}")
    
    # Save experiment configuration
    config = {
        'experiment': args.experiment,
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'layers': args.layers,
        'generation_steps': args.generation_steps,
        'successful_only': args.successful_only,
        'max_episodes': args.max_episodes,
        'test_size': args.test_size,
        'random_seed': args.random_seed,
        'timestamp': time.time(),
        'timestamp_str': time.strftime('%Y-%m-%d_%H-%M-%S')
    }
    
    config_file = output_path / f'experiment_{args.experiment}_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[DEBUG] Saved configuration to: {config_file}")
    
    # Verify data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Data path must be a directory: {args.data_path}")
    
    # Run the specified experiment
    start_time = time.time()
    
    try:
        if args.experiment == 1:
            print(f"[DEBUG] Running Experiment 1: [Hidden state] -> actions")
            results = run_experiment_1(
                data_path=args.data_path,
                output_dir=args.output_dir,
                layers=args.layers,
                generation_steps=args.generation_steps,
                successful_only=args.successful_only,
                max_episodes=args.max_episodes,
                test_size=args.test_size,
                random_seed=args.random_seed,
                debug=args.debug
            )
            
        elif args.experiment == 2:
            print(f"[DEBUG] Running Experiment 2: [Vision encoder outputs] -> actions")
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
            
        elif args.experiment == 3:
            print(f"[DEBUG] Running Experiment 3: [Hidden state] -> visual concepts (general_1)")
            results = run_experiment_3_general_1(
                data_root=args.data_path,
                output_dir=args.output_dir,
                layers=args.layers,
                generation_steps=args.generation_steps,
                successful_only=args.successful_only,
                max_tasks=args.max_episodes,  # reuse as cap on number of tasks
                test_size=args.test_size,
                random_seed=args.random_seed,
                debug=args.debug
            )
            
        elif args.experiment == 4:
            print(f"[DEBUG] Running Experiment 4: [Vision encoder outputs] -> visual concepts (general_1)")
            results = run_experiment_4_general_1(
                data_root=args.data_path,
                output_dir=args.output_dir,
                vision_type=args.vision_type,
                successful_only=args.successful_only,
                max_tasks=args.max_episodes,  # reuse as cap on number of tasks
                test_size=args.test_size,
                random_seed=args.random_seed,
                debug=args.debug
            )
            
    except Exception as e:
        print(f"[ERROR] Experiment {args.experiment} failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    elapsed_time = time.time() - start_time
    print(f"[DEBUG] Experiment {args.experiment} completed in {elapsed_time:.2f} seconds")
    
    # Save final results summary
    summary = {
        'experiment': args.experiment,
        'status': 'completed',
        'elapsed_time': elapsed_time,
        'results': results if 'results' in locals() else None,
        'config': config
    }
    
    summary_file = output_path / f'experiment_{args.experiment}_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[DEBUG] Saved experiment summary to: {summary_file}")
    
    print(f"=== Experiment {args.experiment} Completed Successfully ===")


if __name__ == "__main__":
    main()
