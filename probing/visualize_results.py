#!/usr/bin/env python3
"""
visualize_results.py

Visualization utilities for linear probing experiment results.
Generates publication-ready PNG plots without using plt.show() (for server usage).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Set style for clean plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_experiment_results(results_file: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return results


def visualize_experiment_1_results(
    results: Dict, 
    output_dir: str,
    debug: bool = False
) -> List[str]:
    """
    Create visualizations for Experiment 1: [Hidden state] -> actions.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
        debug: Enable debug output
        
    Returns:
        List of generated plot file paths
    """
    if debug:
        print(f"[DEBUG] Creating visualizations for Experiment 1")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_plots = []
    
    # Extract data for plotting
    layers, gen_steps, normal_r2, random_r2, noise_r2 = extract_experiment_1_data(results, debug)
    
    if len(normal_r2) == 0:
        if debug:
            print(f"[DEBUG] No valid data found for visualization")
        return []
    
    # Plot 1: R2 scores by layer (all baselines)
    plot1_path = create_r2_by_layer_plot(
        layers, gen_steps, normal_r2, random_r2, noise_r2, 
        output_path / "experiment_1_r2_by_layer.png", debug
    )
    generated_plots.append(str(plot1_path))
    
    # Plot 2: Linear separability analysis
    plot2_path = create_linear_separability_plot(
        layers, normal_r2, random_r2, noise_r2,
        output_path / "experiment_1_linear_separability.png", debug
    )
    generated_plots.append(str(plot2_path))
    
    # Plot 3: Per-action dimension analysis (if multi-dimensional actions)
    if check_multi_dimensional_actions(results):
        plot3_path = create_action_dimension_analysis(
            results, output_path / "experiment_1_action_dimensions.png", debug
        )
        if plot3_path:
            generated_plots.append(str(plot3_path))
    
    # Plot 4: Summary heatmap
    plot4_path = create_summary_heatmap(
        results, output_path / "experiment_1_summary_heatmap.png", debug
    )
    generated_plots.append(str(plot4_path))
    
    if debug:
        print(f"[DEBUG] Generated {len(generated_plots)} plots:")
        for plot in generated_plots:
            print(f"[DEBUG] - {plot}")
    
    return generated_plots


def extract_experiment_1_data(results: Dict, debug: bool = False) -> Tuple[List, List, List, List, List]:
    """Extract plotting data from experiment results."""
    layers = []
    gen_steps = []
    normal_r2 = []
    random_r2 = []
    noise_r2 = []
    
    for layer_key, layer_data in results['results_by_layer'].items():
        layer_idx = int(layer_key.split('_')[1])
        
        for gen_step_key, probe_data in layer_data.items():
            if 'error' in probe_data:
                continue
                
            if 'normal' in probe_data and 'randomized' in probe_data and 'noise' in probe_data:
                gen_step = int(gen_step_key.split('_')[2])
                
                layers.append(layer_idx)
                gen_steps.append(gen_step)
                normal_r2.append(probe_data['normal'].get('r2_test', 0))
                random_r2.append(probe_data['randomized'].get('r2_test', 0))
                noise_r2.append(probe_data['noise'].get('r2_test', 0))
    
    if debug:
        print(f"[DEBUG] Extracted data: {len(normal_r2)} data points")
        print(f"[DEBUG] Layers: {sorted(set(layers))}")
        print(f"[DEBUG] Generation steps: {sorted(set(gen_steps))}")
    
    return layers, gen_steps, normal_r2, random_r2, noise_r2


def create_r2_by_layer_plot(
    layers: List, gen_steps: List, normal_r2: List, random_r2: List, noise_r2: List,
    output_path: Path, debug: bool = False
) -> Path:
    """Create R2 scores by layer plot."""
    
    if debug:
        print(f"[DEBUG] Creating R2 by layer plot")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group data by layer
    unique_layers = sorted(set(layers))
    layer_positions = np.arange(len(unique_layers))
    
    normal_means = []
    random_means = []  
    noise_means = []
    normal_stds = []
    random_stds = []
    noise_stds = []
    
    for layer in unique_layers:
        layer_indices = [i for i, l in enumerate(layers) if l == layer]
        
        normal_values = [normal_r2[i] for i in layer_indices]
        random_values = [random_r2[i] for i in layer_indices]
        noise_values = [noise_r2[i] for i in layer_indices]
        
        normal_means.append(np.mean(normal_values))
        random_means.append(np.mean(random_values))
        noise_means.append(np.mean(noise_values))
        
        normal_stds.append(np.std(normal_values) if len(normal_values) > 1 else 0)
        random_stds.append(np.std(random_values) if len(random_values) > 1 else 0)
        noise_stds.append(np.std(noise_values) if len(noise_values) > 1 else 0)
    
    # Create bar plot with error bars
    width = 0.25
    x = layer_positions
    
    bars1 = ax.bar(x - width, normal_means, width, yerr=normal_stds, 
                   label='Normal (Original)', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x, random_means, width, yerr=random_stds,
                   label='Randomized (Broken Correspondence)', alpha=0.8, color='#A23B72')
    bars3 = ax.bar(x + width, noise_means, width, yerr=noise_stds,
                   label='Noise (Gaussian)', alpha=0.8, color='#F18F01')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Linear Probe Performance: Hidden States → Actions\n(Higher R² = More Linearly Separable)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(layer_positions)
    ax.set_xticklabels([f'Layer {l}' for l in unique_layers])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if debug:
        print(f"[DEBUG] Saved R2 by layer plot to: {output_path}")
    
    return output_path


def create_linear_separability_plot(
    layers: List, normal_r2: List, random_r2: List, noise_r2: List,
    output_path: Path, debug: bool = False
) -> Path:
    """Create linear separability analysis plot."""
    
    if debug:
        print(f"[DEBUG] Creating linear separability plot")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Normal vs Random (tests actual vs chance performance)
    normal_vs_random_diff = np.array(normal_r2) - np.array(random_r2)
    
    ax1.scatter(normal_r2, random_r2, alpha=0.7, s=60, c='#2E86AB')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Correlation')
    ax1.set_xlabel('Normal R² (Original Data)', fontsize=12)
    ax1.set_ylabel('Random R² (Shuffled Pairs)', fontsize=12) 
    ax1.set_title('Normal vs Random Performance\n(Points below diagonal = meaningful signal)', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add layer labels
    unique_layers = sorted(set(layers))
    for i, (nr2, rr2, layer) in enumerate(zip(normal_r2, random_r2, layers)):
        ax1.annotate(f'L{layer}', (nr2, rr2), xytext=(2, 2), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # Plot 2: Normal vs Noise (tests robustness against overfitting)
    ax2.scatter(normal_r2, noise_r2, alpha=0.7, s=60, c='#A23B72')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Correlation')
    ax2.set_xlabel('Normal R² (Original Data)', fontsize=12)
    ax2.set_ylabel('Noise R² (Gaussian Targets)', fontsize=12)
    ax2.set_title('Normal vs Noise Performance\n(Points below diagonal = not overfitting)', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add layer labels
    for i, (nr2, noise_r2_val, layer) in enumerate(zip(normal_r2, noise_r2, layers)):
        ax2.annotate(f'L{layer}', (nr2, noise_r2_val), xytext=(2, 2), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if debug:
        print(f"[DEBUG] Saved linear separability plot to: {output_path}")
    
    return output_path


def check_multi_dimensional_actions(results: Dict) -> bool:
    """Check if actions are multi-dimensional (e.g., 7D robot control)."""
    for layer_key, layer_data in results['results_by_layer'].items():
        for gen_step_key, probe_data in layer_data.items():
            if 'normal' in probe_data:
                if 'r2_test_dim_0' in probe_data['normal']:
                    return True
    return False


def create_action_dimension_analysis(
    results: Dict, output_path: Path, debug: bool = False
) -> Optional[Path]:
    """Create per-action dimension analysis plot."""
    
    if debug:
        print(f"[DEBUG] Creating action dimension analysis plot")
    
    # Extract per-dimension R2 scores
    layer_data = []
    dimension_data = {}
    
    for layer_key, layer_results in results['results_by_layer'].items():
        layer_idx = int(layer_key.split('_')[1])
        
        for gen_step_key, probe_data in layer_results.items():
            if 'normal' not in probe_data:
                continue
                
            normal_data = probe_data['normal']
            
            # Find all dimension keys
            dim_keys = [k for k in normal_data.keys() if k.startswith('r2_test_dim_')]
            
            if dim_keys:
                for dim_key in dim_keys:
                    dim_idx = int(dim_key.split('_')[3])
                    r2_value = normal_data[dim_key]
                    
                    if dim_idx not in dimension_data:
                        dimension_data[dim_idx] = []
                    dimension_data[dim_idx].append((layer_idx, r2_value))
    
    if not dimension_data:
        if debug:
            print(f"[DEBUG] No multi-dimensional action data found")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    dimensions = sorted(dimension_data.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(dimensions)))
    
    for dim_idx, color in zip(dimensions, colors):
        layers, r2_values = zip(*dimension_data[dim_idx])
        layers = np.array(layers)
        r2_values = np.array(r2_values)
        
        # Sort by layer
        sort_idx = np.argsort(layers)
        layers = layers[sort_idx]
        r2_values = r2_values[sort_idx]
        
        ax.plot(layers, r2_values, 'o-', color=color, label=f'Action Dim {dim_idx}', 
                markersize=6, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Linear Separability by Action Dimension\n(7D Robot Control Vector)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if debug:
        print(f"[DEBUG] Saved action dimension analysis to: {output_path}")
    
    return output_path


def create_summary_heatmap(results: Dict, output_path: Path, debug: bool = False) -> Path:
    """Create summary heatmap of all results."""
    
    if debug:
        print(f"[DEBUG] Creating summary heatmap")
    
    # Extract data for heatmap
    layers = []
    gen_steps = []
    r2_values = []
    
    for layer_key, layer_data in results['results_by_layer'].items():
        layer_idx = int(layer_key.split('_')[1])
        
        for gen_step_key, probe_data in layer_data.items():
            if 'normal' in probe_data:
                gen_step = int(gen_step_key.split('_')[2])
                r2_val = probe_data['normal'].get('r2_test', 0)
                
                layers.append(layer_idx)
                gen_steps.append(gen_step)
                r2_values.append(r2_val)
    
    if not r2_values:
        if debug:
            print(f"[DEBUG] No data for heatmap")
        return output_path
    
    # Create heatmap matrix
    unique_layers = sorted(set(layers))
    unique_gen_steps = sorted(set(gen_steps))
    
    heatmap_data = np.zeros((len(unique_layers), len(unique_gen_steps)))
    
    for layer, gen_step, r2_val in zip(layers, gen_steps, r2_values):
        layer_idx = unique_layers.index(layer)
        gen_step_idx = unique_gen_steps.index(gen_step)
        heatmap_data[layer_idx, gen_step_idx] = r2_val
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=0, vmax=max(r2_values))
    
    # Set ticks and labels
    ax.set_xticks(range(len(unique_gen_steps)))
    ax.set_yticks(range(len(unique_layers)))
    ax.set_xticklabels([f'Gen {g}' for g in unique_gen_steps])
    ax.set_yticklabels([f'Layer {l}' for l in unique_layers])
    
    # Add text annotations
    for i in range(len(unique_layers)):
        for j in range(len(unique_gen_steps)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="white", fontsize=10)
    
    ax.set_xlabel('Generation Step', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title('Linear Probe R² Heatmap: Hidden States → Actions\n(Higher values = better linear separability)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('R² Score', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if debug:
        print(f"[DEBUG] Saved summary heatmap to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Visualize linear probing experiment results')
    parser.add_argument('results_file', help='Path to experiment results JSON file')
    parser.add_argument('--output-dir', default='./plots', help='Output directory for plots')
    parser.add_argument('--experiment', type=int, default=1, choices=[1, 2, 3, 4],
                       help='Experiment number')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    print(f"[INFO] Loading results from: {args.results_file}")
    results = load_experiment_results(args.results_file)
    
    if args.experiment == 1:
        plots = visualize_experiment_1_results(results, args.output_dir, args.debug)
        print(f"[INFO] Generated {len(plots)} plots for Experiment 1")
        for plot in plots:
            print(f"[INFO] - {plot}")
    else:
        print(f"[ERROR] Experiment {args.experiment} visualization not yet implemented")


if __name__ == "__main__":
    main()