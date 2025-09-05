#!/usr/bin/env python3
"""
visualize_experiment_2.py

Visualization utilities for Experiment 2: [Vision encoder outputs] -> actions
Creates publication-ready plots for vision encoder linear probing results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Optional, List
import seaborn as sns

# Set publication-ready style
plt.style.use('default')
sns.set_palette("husl")


def visualize_experiment_2_results(
    results_file: str,
    output_dir: str,
    figsize: tuple = (12, 8),
    dpi: int = 300,
    show_plots: bool = False
) -> None:
    """
    Create comprehensive visualizations for Experiment 2 results.
    
    Args:
        results_file: Path to experiment_2_complete_results.json
        output_dir: Directory to save plots
        figsize: Figure size for plots
        dpi: Resolution for saved plots
        show_plots: Whether to display plots (False for server environments)
    """
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract probe results
    probe_results = results.get('results', {})
    
    if 'error' in probe_results:
        print(f"ERROR: Cannot visualize results due to error: {probe_results['error']}")
        return
    
    # 1. Baseline Comparison Plot
    create_baseline_comparison_plot(
        probe_results, output_path, figsize, dpi, show_plots
    )
    
    # 2. Action Dimension Analysis (if per-dimension data available)
    create_action_dimension_plot(
        probe_results, output_path, figsize, dpi, show_plots
    )
    
    # 3. Linear Separability Summary
    create_separability_summary_plot(
        probe_results, output_path, figsize, dpi, show_plots
    )
    
    print(f"Experiment 2 visualizations saved to: {output_path}")


def create_baseline_comparison_plot(
    probe_results: Dict,
    output_path: Path,
    figsize: tuple,
    dpi: int,
    show_plots: bool
) -> None:
    """Create bar chart comparing baseline conditions."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract metrics
    conditions = ['Normal', 'Randomized', 'Noise']
    r2_scores = []
    mse_scores = []
    
    for condition in ['normal', 'randomized', 'noise']:
        if condition in probe_results:
            r2_scores.append(probe_results[condition].get('r2_test', 0))
            mse_scores.append(probe_results[condition].get('mse_test', 0))
        else:
            r2_scores.append(0)
            mse_scores.append(0)
    
    # R2 Score comparison
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars1 = ax1.bar(conditions, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Vision Encoder Linear Separability\n(R² Score)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_ylim(0, max(max(r2_scores) * 1.2, 0.1))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MSE comparison
    bars2 = ax2.bar(conditions, mse_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Vision Encoder Prediction Error\n(MSE)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean Squared Error', fontsize=12)
    ax2.set_ylim(0, max(mse_scores) * 1.2 if max(mse_scores) > 0 else 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars2, mse_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add interpretation text
    normal_r2 = r2_scores[0]
    random_r2 = r2_scores[1]
    improvement = normal_r2 - random_r2
    
    fig.suptitle('Experiment 2: Vision Encoder → Actions Linear Probing', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Add summary text
    summary_text = f'Linear Separability: {normal_r2:.3f}\nImprovement over Random: +{improvement:.3f}'
    fig.text(0.02, 0.02, summary_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)
    
    # Save plot
    plt.savefig(output_path / 'experiment_2_baseline_comparison.png', 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def create_action_dimension_plot(
    probe_results: Dict,
    output_path: Path,
    figsize: tuple,
    dpi: int,
    show_plots: bool
) -> None:
    """Create plot showing per-action-dimension performance if available."""
    
    # Check if per-dimension metrics are available
    normal_results = probe_results.get('normal', {})
    dimension_keys = [k for k in normal_results.keys() if k.startswith('r2_test_dim_')]
    
    if not dimension_keys:
        print("No per-dimension metrics available for action dimension plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract per-dimension R2 scores for each condition
    conditions = ['normal', 'randomized', 'noise']
    condition_labels = ['Normal', 'Randomized', 'Noise']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    num_dims = len(dimension_keys)
    action_dims = [f'Action {i}' for i in range(num_dims)]
    
    x = np.arange(num_dims)
    width = 0.25
    
    for i, (condition, label, color) in enumerate(zip(conditions, condition_labels, colors)):
        if condition in probe_results:
            dim_scores = []
            for dim_idx in range(num_dims):
                dim_key = f'r2_test_dim_{dim_idx}'
                score = probe_results[condition].get(dim_key, 0)
                dim_scores.append(score)
            
            bars = ax.bar(x + i*width, dim_scores, width, label=label, 
                         color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars (only for normal condition to avoid clutter)
            if condition == 'normal':
                for bar, score in zip(bars, dim_scores):
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                               f'{score:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Linear Separability by Action Dimension\n(Vision Encoder → Actions)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Action Dimensions', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(action_dims)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path / 'experiment_2_action_dimensions.png', 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def create_separability_summary_plot(
    probe_results: Dict,
    output_path: Path,
    figsize: tuple,
    dpi: int,
    show_plots: bool
) -> None:
    """Create summary visualization of linear separability strength."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract key metrics
    normal_r2 = probe_results.get('normal', {}).get('r2_test', 0)
    random_r2 = probe_results.get('randomized', {}).get('r2_test', 0)
    noise_r2 = probe_results.get('noise', {}).get('r2_test', 0)
    
    # Create separability strength visualization
    max_possible = 1.0
    
    # Stacked bar showing different components
    categories = ['Vision\nEncoder']
    
    # Components: Random baseline, Noise overfitting, True signal
    random_component = max(random_r2, 0)
    noise_component = max(noise_r2 - random_r2, 0) 
    true_signal = max(normal_r2 - max(random_r2, noise_r2), 0)
    remaining = max_possible - normal_r2
    
    # Create stacked bars
    p1 = ax.bar(categories, [random_component], label='Random Chance', 
                color='#F18F01', alpha=0.8)
    p2 = ax.bar(categories, [noise_component], bottom=[random_component],
                label='Potential Overfitting', color='#A23B72', alpha=0.8)
    p3 = ax.bar(categories, [true_signal], 
                bottom=[random_component + noise_component],
                label='True Linear Signal', color='#2E86AB', alpha=0.8)
    p4 = ax.bar(categories, [remaining], 
                bottom=[random_component + noise_component + true_signal],
                label='Unexplained Variance', color='lightgray', alpha=0.5)
    
    # Add total score annotation
    ax.text(0, normal_r2 + 0.02, f'Total R²: {normal_r2:.3f}', 
            ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_title('Vision Encoder Linear Separability Breakdown', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('R² Score Components', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add interpretation
    if true_signal > 0.1:
        interpretation = "Strong linear separability"
    elif true_signal > 0.05:
        interpretation = "Moderate linear separability"  
    else:
        interpretation = "Limited linear separability"
    
    ax.text(0.5, 0.95, f'Interpretation: {interpretation}', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            fontsize=11)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path / 'experiment_2_separability_summary.png', 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def compare_experiments_1_and_2(
    exp1_results_file: str,
    exp2_results_file: str,
    output_dir: str,
    figsize: tuple = (12, 6),
    dpi: int = 300,
    show_plots: bool = False
) -> None:
    """
    Compare results between Experiment 1 (hidden states) and Experiment 2 (vision encoder).
    """
    
    # Load both experiments
    with open(exp1_results_file, 'r') as f:
        exp1_results = json.load(f)
    
    with open(exp2_results_file, 'r') as f:
        exp2_results = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Extract best performance from Experiment 1
    exp1_summary = exp1_results.get('experiment_summary', {})
    best_hidden_r2 = exp1_summary.get('best_overall_r2', 0)
    mean_hidden_r2 = exp1_summary.get('normal_r2_stats', {}).get('mean', 0)
    
    # Extract performance from Experiment 2
    exp2_summary = exp2_results.get('experiment_summary', {})
    vision_r2 = exp2_summary.get('normal_r2', 0)
    
    # Comparison bar chart
    categories = ['Best Hidden\nState Layer', 'Mean Hidden\nState Layers', 'Vision\nEncoder']
    scores = [best_hidden_r2, mean_hidden_r2, vision_r2]
    colors = ['#2E86AB', '#5C9BD5', '#F18F01']
    
    bars = ax1.bar(categories, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Hidden States vs Vision Encoder\nLinear Separability (R²)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_ylim(0, max(scores) * 1.2)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Relative comparison
    if vision_r2 > 0:
        relative_to_best = vision_r2 / best_hidden_r2 if best_hidden_r2 > 0 else 0
        relative_to_mean = vision_r2 / mean_hidden_r2 if mean_hidden_r2 > 0 else 0
    else:
        relative_to_best = 0
        relative_to_mean = 0
    
    comparison_labels = ['vs Best\nHidden Layer', 'vs Mean\nHidden Layers']
    comparison_values = [relative_to_best, relative_to_mean]
    comparison_colors = ['#A23B72', '#D67096']
    
    bars2 = ax2.bar(comparison_labels, comparison_values, color=comparison_colors, 
                    alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Vision Encoder Relative Performance\n(Ratio to Hidden States)', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Ratio', fontsize=12)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars2, comparison_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{score:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Experiment 1 vs Experiment 2: Representation Comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison plot
    plt.savefig(output_path / 'experiments_1_vs_2_comparison.png', 
                dpi=dpi, bbox_inches='tight', facecolor='white')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    print(f"Experiment comparison saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Experiment 2 results")
    parser.add_argument("results_file", help="Path to experiment_2_complete_results.json")
    parser.add_argument("output_dir", help="Output directory for plots")
    parser.add_argument("--exp1-results", help="Path to experiment_1_complete_results.json for comparison")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 8], help="Figure size")
    parser.add_argument("--dpi", type=int, default=300, help="Plot resolution")
    parser.add_argument("--show-plots", action='store_true', help="Display plots")
    
    args = parser.parse_args()
    
    try:
        # Create Experiment 2 visualizations
        visualize_experiment_2_results(
            results_file=args.results_file,
            output_dir=args.output_dir,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            show_plots=args.show_plots
        )
        
        # Create comparison if Experiment 1 results provided
        if args.exp1_results:
            compare_experiments_1_and_2(
                exp1_results_file=args.exp1_results,
                exp2_results_file=args.results_file,
                output_dir=args.output_dir,
                figsize=tuple(args.figsize),
                dpi=args.dpi,
                show_plots=args.show_plots
            )
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()