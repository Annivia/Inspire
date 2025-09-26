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


def collect_per_dim_r2_by_layer(results: Dict):
    """
    Build per-dimension, per-layer R² (averaged over generation steps if multiple).
    Returns: layers_sorted, total_dims, dim_to_layer_values (dim -> list aligned with layers_sorted)
    """
    by_layer = results.get('results_by_layer', {})

    # Gather all layer indices
    layer_indices = []
    for layer_key in by_layer.keys():
        try:
            layer_indices.append(int(layer_key.split('_')[1]))
        except Exception:
            continue
    layers_sorted = sorted(set(layer_indices))

    # Determine total dims from any entry
    total_dims = None
    for layer_key, layer_data in by_layer.items():
        for gen_key, probe_data in layer_data.items():
            normal = probe_data.get('normal', {})
            dim_keys = [k for k in normal.keys() if k.startswith('r2_test_dim_')]
            if dim_keys:
                dim_indices = [int(k.split('_')[-1]) for k in dim_keys]
                total_dims = max(dim_indices) + 1
                break
        if total_dims is not None:
            break

    if total_dims is None:
        raise ValueError('No per-dimension metrics found in results.')

    # Accumulate values per (dim, layer) over generation steps
    dim_layer_vals: Dict[int, Dict[int, List[float]]] = {d: {L: [] for L in layers_sorted} for d in range(total_dims)}

    for layer_key, layer_data in by_layer.items():
        try:
            L = int(layer_key.split('_')[1])
        except Exception:
            continue
        for gen_key, probe_data in layer_data.items():
            normal = probe_data.get('normal', {})
            for d in range(total_dims):
                key = f'r2_test_dim_{d}'
                if key in normal:
                    dim_layer_vals[d][L].append(normal[key])

    # Average over gen steps and align to layers_sorted
    dim_to_layer_values: Dict[int, List[float]] = {}
    for d in range(total_dims):
        values = []
        for L in layers_sorted:
            vals = dim_layer_vals[d][L]
            values.append(float(np.mean(vals)) if len(vals) else np.nan)
        dim_to_layer_values[d] = values

    return layers_sorted, total_dims, dim_to_layer_values


def build_color_palettes(D: int):
    """Build high-contrast color shades for dims 1..D (dark→light)."""
    reds_cmap = mpl.cm.get_cmap('Reds')
    blues_cmap = mpl.cm.get_cmap('Blues')
    greens_cmap = mpl.cm.get_cmap('Greens')

    # For Matplotlib 'Reds/Blues/Greens': t=0 is very light, t=1 is darkest.
    # Make dim 1 darkest → dim D lightest with a wide dynamic range.
    t_vals = np.linspace(0.98, 0.15, D)  # strong contrast

    reds = [reds_cmap(t) for t in t_vals]
    blues = [blues_cmap(t) for t in t_vals]
    greens = [greens_cmap(t) for t in t_vals]
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
    ax.legend(handles=[rep_red, rep_blue, rep_green], loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.subplots_adjust(right=0.8)

    ax.set_ylabel('R² (mean across layers)', fontsize=12)
    ax.set_title('Hidden → Actions: Per-dimension R² grouped by horizon (First | Middle | Last)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def make_by_layer_line_plot(results: Dict, output_path: Path, debug: bool = False):
    layers_sorted, total_dims, dim_to_layer_values = collect_per_dim_r2_by_layer(results)

    # Expect total_dims = 3 * D when selection is first_middle_last
    D = total_dims // 3 if total_dims % 3 == 0 else total_dims
    reds, blues, greens = build_color_palettes(D)

    fig, ax = plt.subplots(figsize=(14, 8))

    for d in range(total_dims):
        if total_dims % 3 == 0:
            block = d // D  # 0:first, 1:middle, 2:last
            idx_in_block = d % D
            if block == 0:
                color = reds[idx_in_block]
            elif block == 1:
                color = blues[idx_in_block]
            else:
                color = greens[idx_in_block]
        else:
            # Fallback single palette if not divisible by 3
            color = mpl.cm.tab20(d % 20)

        y = np.array(dim_to_layer_values[d], dtype=float)
        ax.plot(layers_sorted, y, '-o', markersize=3, linewidth=1.8, color=color, alpha=0.9)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('R² (Normal baseline)', fontsize=12)
    ax.set_title('Hidden → Actions: R² by layer (First | Middle | Last, dims 1→7 dark→light)', fontsize=14)
    ax.set_xticks(layers_sorted)
    ax.grid(True, alpha=0.3)

    # Group-level legend
    rep_red = mpl.patches.Patch(color=mpl.cm.Reds(0.9), label='First action (dims 1→7 dark→light)')
    rep_blue = mpl.patches.Patch(color=mpl.cm.Blues(0.9), label='Middle action (dims 1→7 dark→light)')
    rep_green = mpl.patches.Patch(color=mpl.cm.Greens(0.9), label='Last action (dims 1→7 dark→light)')
    ax.legend(handles=[rep_red, rep_blue, rep_green], loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.subplots_adjust(right=0.8)

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

    # Outputs
    if args.output:
        out_grouped = Path(args.output)
        out_by_layer = out_grouped.with_name(out_grouped.stem + '_by_layer.png')
    else:
        base = results_dir / 'plots_custom'
        out_grouped = base / 'experiment_1_action_dims_grouped.png'
        out_by_layer = base / 'experiment_1_action_dims_by_layer.png'

    out_grouped.parent.mkdir(parents=True, exist_ok=True)

    if args.debug:
        print(f"[DEBUG] Total dims: {total_dims}")
        print(f"[DEBUG] Writing grouped plot to: {out_grouped}")
        print(f"[DEBUG] Writing by-layer plot to: {out_by_layer}")

    make_grouped_plot(mean_r2, total_dims, out_grouped, debug=args.debug)
    make_by_layer_line_plot(results, out_by_layer, debug=args.debug)
    print(f"[INFO] Saved grouped action-dimension plot to: {out_grouped}")
    print(f"[INFO] Saved by-layer action-dimension plot to: {out_by_layer}")


if __name__ == '__main__':
    main()
