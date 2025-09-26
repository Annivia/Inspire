#!/usr/bin/env python3
"""
plot_probe1_first_mid_last.py

Standalone plotting for Experiment 1 (Hidden → Actions), drawing only from
existing results under /u/xzhang42/Inspire/probing/results/experiment_1.

It separates action tokenization into First | Middle | Last groups and plots
the Normal-baseline R² across layers for each group in red/blue/green.

No probing or data regeneration is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_results(results_dir: Path, results_file: str = 'experiment_1_complete_results.json') -> Dict:
    path = results_dir / results_file
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def collect_per_layer_group_means(results: Dict) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Returns layer_indices, r2_first, r2_middle, r2_last

    - Extracts per-dimension R² for the Normal baseline at each layer/gen step.
    - Averages across generation steps per layer.
    - Splits dimensions into three equal groups (First/Middle/Last) and
      averages within each group.
    """
    by_layer = results.get('results_by_layer', {})

    # Determine layer ordering
    layer_indices = []
    for lk in by_layer.keys():
        try:
            layer_indices.append(int(lk.split('_')[1]))
        except Exception:
            continue
    layers_sorted = sorted(set(layer_indices))

    # Determine number of dims from any entry
    total_dims = None
    for layer_data in by_layer.values():
        for probe_data in layer_data.values():  # generation steps
            normal = probe_data.get('normal', {})
            dim_keys = [k for k in normal.keys() if k.startswith('r2_test_dim_')]
            if dim_keys:
                dim_indices = [int(k.split('_')[-1]) for k in dim_keys]
                total_dims = max(dim_indices) + 1
                break
        if total_dims is not None:
            break
    if total_dims is None:
        raise ValueError('No per-dimension R² metrics found in results.')

    if total_dims % 3 != 0:
        # Fallback: still compute 3 groups by slicing as evenly as possible
        sizes = [total_dims // 3] * 3
        for i in range(total_dims % 3):
            sizes[i] += 1
        boundaries = np.cumsum([0] + sizes).tolist()
        groups = [range(boundaries[i], boundaries[i + 1]) for i in range(3)]
    else:
        D = total_dims // 3
        groups = [range(0, D), range(D, 2 * D), range(2 * D, 3 * D)]

    r2_first: List[float] = []
    r2_middle: List[float] = []
    r2_last: List[float] = []

    for L in layers_sorted:
        layer_key = f'layer_{L}'
        layer_data = by_layer.get(layer_key, {})

        # Collect per-dim values across generation steps, then average gen steps
        per_dim_vals: Dict[int, List[float]] = {}
        for probe_data in layer_data.values():  # gen steps
            normal = probe_data.get('normal', {})
            for d in range(total_dims):
                key = f'r2_test_dim_{d}'
                if key in normal:
                    per_dim_vals.setdefault(d, []).append(float(normal[key]))

        # Average over gen steps per dim
        dim_means = np.array([np.mean(per_dim_vals.get(d, [np.nan])) for d in range(total_dims)], dtype=float)

        # Group means (ignore NaNs)
        def group_mean(idx_range: range) -> float:
            vals = dim_means[list(idx_range)]
            vals = vals[~np.isnan(vals)]
            return float(np.mean(vals)) if vals.size else float('nan')

        r2_first.append(group_mean(groups[0]))
        r2_middle.append(group_mean(groups[1]))
        r2_last.append(group_mean(groups[2]))

    return layers_sorted, r2_first, r2_middle, r2_last


def plot_by_layer(layers: List[int], r2_first: List[float], r2_middle: List[float], r2_last: List[float], output_path: Path):
    plt.figure(figsize=(12, 7))
    plt.plot(layers, r2_first, '-o', color='red', label='First', linewidth=2, markersize=4, alpha=0.9)
    plt.plot(layers, r2_middle, '-o', color='blue', label='Middle', linewidth=2, markersize=4, alpha=0.9)
    plt.plot(layers, r2_last, '-o', color='green', label='Last', linewidth=2, markersize=4, alpha=0.9)

    plt.xlabel('Layer Index')
    plt.ylabel('R² (Normal)')
    plt.title('Hidden → Actions: First | Middle | Last (per layer)')
    plt.grid(True, alpha=0.3)
    plt.xticks(layers)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot probe1 results separating First | Middle | Last actions.')
    parser.add_argument('--results-dir', type=str, default='/u/xzhang42/Inspire/probing/results/experiment_1',
                        help='Directory containing experiment_1_complete_results.json')
    parser.add_argument('--results-file', type=str, default='experiment_1_complete_results.json',
                        help='Results JSON filename to load')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path. Default: <results-dir>/plots_custom/experiment_1_first_mid_last_by_layer.png')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir, args.results_file)

    layers, r2_first, r2_middle, r2_last = collect_per_layer_group_means(results)

    if args.output:
        out = Path(args.output)
    else:
        out = results_dir / 'plots_custom' / 'experiment_1_first_mid_last_by_layer.png'

    plot_by_layer(layers, r2_first, r2_middle, r2_last, out)
    print(f'[INFO] Saved plot to: {out}')


if __name__ == '__main__':
    main()

