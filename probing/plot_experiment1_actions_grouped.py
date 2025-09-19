#!/usr/bin/env python3
"""
plot_experiment1_actions_grouped.py

Standalone visualization for Experiment 1 per-action-dimension R², grouped as:
- First action dims (red, dark→light for dims 1→7)
- Middle action dims (blue, dark→light)
- Last action dims (green, dark→light)

It reads the existing JSON results from the experiment_1 results directory and
produces a grouped bar plot with x-ticks labeled like:
  First action dim 1 | Middle action dim 1 | Last action dim 1 | First action dim 2 | ...

No probing is run; this only plots from existing results.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def load_results(results_dir: Path, results_file: str) -> Dict:
    results_path = results_dir / results_file
    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")
    with open(results_path, 'r') as f:
        return json.load(f)


def collect_per_dim_r2(results: Dict) -> Tuple[List[float], int]:
    """
    Collect per-dimension R² across all layers/gen steps and return mean per dim.
    Returns (mean_r2_per_dim, total_dims)
    """
    total_dims = None
    per_dim_values: Dict[int, List[float]] = {}

    by_layer = results.get('results_by_layer', {})

    for layer_key, layer_data in by_layer.items():
        for gen_key, probe_data in layer_data.items():
            normal = probe_data.get('normal', {})
            dim_keys = [k for k in normal.keys() if k.startswith('r2_test_dim_')]
            if not dim_keys:
                r2_overall = normal.get('r2_test', None)
                if r2_overall is None:
                    continue
                raise ValueError("Per-dimension R² not found in results. Re-run probes with multi-output metrics.")

            if total_dims is None:
                dim_indices = [int(k.split('_')[-1]) for k in dim_keys]
                total_dims = max(dim_indices) + 1

            for k in dim_keys:
                di = int(k.split('_')[-1])
                per_dim_values.setdefault(di, []).append(normal[k])

    if total_dims is None or not per_dim_values:
        raise ValueError("Could not find any per-dimension metrics to plot.")

    mean_r2 = [float(np.mean(per_dim_values.get(i, [0.0]))) for i in range(total_dims)]
    return mean_r2, total_dims


def build_color_palettes(D: int):
    reds_cmap = mpl.cm.get_cmap('Reds')
    blues_cmap = mpl.cm.get_cmap('Blues')
    greens_cmap = mpl.cm.get_cmap('Greens')

    def shades(cmap):
        return [cmap(0.95 - 0.5 * (i / max(1, D - 1))) for i in range(D)]

    reds = shades(reds_cmap)
    blues = shades(blues_cmap)
    greens = shades(greens_cmap)
    return reds, blues, greens


def make_grouped_plot(mean_r2: List[float], total_dims: int, output_path: Path, debug: bool = False):
    if total_dims % 3 != 0:
        x = np.arange(total_dims)
        fig, ax = plt.subplots(figsize=(max(12, total_dims * 0.5), 6))
        ax.bar(x, mean_r2, color='#555555', alpha=0.85, edgecolor='black', linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Dim {i+1}' for i in x], rotation=90)
        ax.set_ylabel('R² (mean across layers)')
        ax.set_title('Actions per-dimension R² (no horizon grouping)')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    D = total_dims // 3
    reds, blues, greens = build_color_palettes(D)

    group_gap = 0.6
    cursor = 0.0
    positions = []
    colors = []
    heights = []
    labels = []

    for i in range(D):
        idx_first = i
        idx_middle = D + i
        idx_last = 2 * D + i

        positions.append(cursor)
        colors.append(reds[i])
        heights.append(mean_r2[idx_first])
        labels.append(f'First action dim {i+1}')
        cursor += 1.0

        positions.append(cursor)
        colors.append(blues[i])
        heights.append(mean_r2[idx_middle])
        labels.append(f'Middle action dim {i+1}')
        cursor += 1.0

        positions.append(cursor)
        colors.append(greens[i])
        heights.append(mean_r2[idx_last])
        labels.append(f'Last action dim {i+1}')
        cursor += 1.0 + group_gap

    fig_width = max(14, int(0.35 * len(positions)))
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    ax.bar(positions, heights, color=colors, edgecolor='black', linewidth=0.3, alpha=0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=9)

    group_edges = []
    cursor = 0.0
    for i in range(D):
        edge_x = cursor + 2.0 + 0.5
        group_edges.append(edge_x)
        cursor += 3.0 + group_gap

    ymin, ymax = 0.0, max(heights + [1.0])
    for gx in group_edges[:-1]:
        ax.vlines(gx, ymin, ymax, colors='gray', linestyles='dotted', alpha=0.3, linewidth=0.8)

    rep_red = mpl.patches.Patch(color=mpl.cm.Reds(0.9), label='First action (dims 1→7 dark→light)')
    rep_blue = mpl.patches.Patch(color=mpl.cm.Blues(0.9), label='Middle action (dims 1→7 dark→light)')
    rep_green = mpl.patches.Patch(color=mpl.cm.Greens(0.9), label='Last action (dims 1→7 dark→light)')
    ax.legend(handles=[rep_red, rep_blue, rep_green], loc='upper right')

    ax.set_ylabel('R² (mean across layers)', fontsize=12)
    ax.set_title('Hidden → Actions: Per-dimension R² grouped by horizon (First | Middle | Last)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot Experiment 1 action dimensions grouped by horizon (first/middle/last)')
    parser.add_argument('--results-dir', type=str, default='/u/xzhang42/Inspire/probing/results/experiment_1',
                        help='Directory containing experiment_1_complete_results.json')
    parser.add_argument('--results-file', type=str, default='experiment_1_complete_results.json',
                        help='Results JSON filename to load')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for the plot (PNG). Default: <results-dir>/plots_custom/experiment_1_action_dims_grouped.png')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir, args.results_file)

    if args.debug:
        cfg = results.get('config', {})
        print(f"[DEBUG] Loaded results from: {results_dir / args.results_file}")
        print(f"[DEBUG] Config: action_selection={cfg.get('action_selection')}  final_actions_shape={cfg.get('final_actions_shape')}")

    mean_r2, total_dims = collect_per_dim_r2(results)

    out_path = Path(args.output) if args.output else (results_dir / 'plots_custom' / 'experiment_1_action_dims_grouped.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.debug:
        print(f"[DEBUG] Total dims: {total_dims}")
        print(f"[DEBUG] Writing plot to: {out_path}")

    make_grouped_plot(mean_r2, total_dims, out_path, debug=args.debug)
    print(f"[INFO] Saved grouped action-dimension plot to: {out_path}")


if __name__ == '__main__':
    main()

