#!/usr/bin/env python3
"""
plot_probe1_columns_first_mid_last.py

Column/bar plot for probe1 (Hidden → Actions) that compares
Normal vs Randomized vs Noise while being action-horizon aware
(First | Middle | Last).

Color encodes action group:
  - First: red
  - Middle: blue
  - Last: green

Hatch encodes condition:
  - Normal: solid (no hatch)
  - Randomized: //
  - Noise: xx

Reads existing JSON results; does not re-run any probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results(results_dir: Path, results_file: str = 'experiment_1_complete_results.json') -> Dict:
    p = results_dir / results_file
    if not p.exists():
        raise FileNotFoundError(f"Results JSON not found: {p}")
    with open(p, 'r') as f:
        return json.load(f)


def _get_total_dims(results: Dict) -> int:
    by_layer = results.get('results_by_layer', {})
    total_dims = None
    for layer_data in by_layer.values():
        for probe_data in layer_data.values():  # generation steps
            for cond in ('normal', 'randomized', 'noise'):
                block = probe_data.get(cond, {})
                if not block:
                    continue
                dim_keys = [k for k in block.keys() if k.startswith('r2_test_dim_')]
                if dim_keys:
                    total_dims = max(int(k.split('_')[-1]) for k in dim_keys) + 1
                    break
            if total_dims is not None:
                break
        if total_dims is not None:
            break
    if total_dims is None:
        raise ValueError('Could not infer number of per-dimension targets from results.')
    return total_dims


def _split_groups(total_dims: int) -> List[range]:
    # Expect 3 equal groups (first/middle/last). If not divisible, split as evenly as possible.
    if total_dims % 3 == 0:
        D = total_dims // 3
        return [range(0, D), range(D, 2 * D), range(2 * D, 3 * D)]
    sizes = [total_dims // 3] * 3
    for i in range(total_dims % 3):
        sizes[i] += 1
    bounds = np.cumsum([0] + sizes)
    return [range(bounds[i], bounds[i + 1]) for i in range(3)]


def collect_group_means_by_layer(results: Dict) -> Tuple[List[int], Dict[str, Dict[str, List[float]]]]:
    """
    Returns (layers_sorted, group_means) where group_means[group][condition] → list over layers.
    group in {'first','middle','last'}, condition in {'normal','randomized','noise'}.
    """
    by_layer = results.get('results_by_layer', {})

    # Layers
    layers = []
    for lk in by_layer.keys():
        try:
            layers.append(int(lk.split('_')[1]))
        except Exception:
            continue
    layers_sorted = sorted(set(layers))

    total_dims = _get_total_dims(results)
    groups = _split_groups(total_dims)
    group_names = ['first', 'middle', 'last']

    # Initialize
    gm: Dict[str, Dict[str, List[float]]] = {g: {c: [] for c in ('normal', 'randomized', 'noise')} for g in group_names}

    for L in layers_sorted:
        layer_key = f'layer_{L}'
        layer_data = by_layer.get(layer_key, {})

        # Accumulate per-dim values across generation steps for each condition
        per_cond_dim_vals: Dict[str, Dict[int, List[float]]] = {c: {} for c in ('normal', 'randomized', 'noise')}
        for probe_data in layer_data.values():  # generation steps
            for cond in ('normal', 'randomized', 'noise'):
                block = probe_data.get(cond, {})
                for d in range(total_dims):
                    key = f'r2_test_dim_{d}'
                    if key in block:
                        per_cond_dim_vals[cond].setdefault(d, []).append(float(block[key]))

        # Average over gen steps per dim
        per_cond_dim_means: Dict[str, np.ndarray] = {}
        for cond, dim_map in per_cond_dim_vals.items():
            arr = np.array([np.mean(dim_map.get(d, [np.nan])) for d in range(total_dims)], dtype=float)
            per_cond_dim_means[cond] = arr

        # Group means
        for name, idxs in zip(group_names, groups):
            for cond in ('normal', 'randomized', 'noise'):
                vals = per_cond_dim_means[cond][list(idxs)]
                vals = vals[~np.isnan(vals)]
                gm[name][cond].append(float(np.mean(vals)) if vals.size else float('nan'))

    return layers_sorted, gm


def plot_columns(layers: List[int], gm: Dict[str, Dict[str, List[float]]], output_path: Path):
    # Visual layout params
    bar_width = 0.08
    group_gap = 0.28   # spacing between first/middle/last within a layer

    # Colors for groups
    group_color = {'first': 'red', 'middle': 'blue', 'last': 'green'}
    conditions = ['normal', 'randomized', 'noise']
    hatches = {'normal': None, 'randomized': '//', 'noise': 'xx'}

    x_positions = []
    heights = []
    colors = []
    hts = []

    for i, L in enumerate(layers):
        base_x = float(i)
        for gi, gname in enumerate(['first', 'middle', 'last']):
            group_center = base_x + (gi - 1) * group_gap
            for bi, cond in enumerate(conditions):
                x = group_center + (bi - 1) * bar_width
                x_positions.append(x)
                heights.append(gm[gname][cond][i])
                colors.append(group_color[gname])
                hts.append(hatches[cond])

    fig_w = max(14, int(0.6 * len(layers)))
    fig, ax = plt.subplots(figsize=(fig_w, 7))

    # Draw bars with hatches
    bars = ax.bar(x_positions, heights, width=bar_width, color=colors, edgecolor='black', linewidth=0.4)
    for b, hatch in zip(bars, hts):
        if hatch:
            b.set_hatch(hatch)
        b.set_alpha(0.9)

    # X ticks per layer
    ax.set_xticks([float(i) for i in range(len(layers))])
    ax.set_xticklabels([f'Layer {L}' for L in layers], rotation=45)
    ax.set_ylabel('R² (Normal / Randomized / Noise)')
    ax.set_title('Hidden → Actions: Column plot by layer (First | Middle | Last)')
    ax.grid(True, axis='y', alpha=0.3)

    # Legends
    # Group legend (color)
    group_handles = [mpatches.Patch(color=group_color[g], label=g.capitalize()) for g in ['first', 'middle', 'last']]
    # Condition legend (hatch)
    cond_handles = [mpatches.Patch(facecolor='lightgray', edgecolor='black', hatch=h if h else '', label=cond.capitalize())
                    for cond, h in zip(['normal', 'randomized', 'noise'], [hatches['normal'], hatches['randomized'], hatches['noise']])]

    leg1 = ax.legend(handles=group_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    leg2 = ax.legend(handles=cond_handles, loc='upper left', bbox_to_anchor=(1.02, 0.78), borderaxespad=0.)
    ax.add_artist(leg1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Bar plot for probe1 separating First|Middle|Last and comparing Normal/Randomized/Noise')
    parser.add_argument('--results-dir', type=str, default='/u/xzhang42/Inspire/probing/results/experiment_1',
                        help='Directory containing experiment_1_complete_results.json')
    parser.add_argument('--results-file', type=str, default='experiment_1_complete_results.json',
                        help='Results JSON filename')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path (default: <results-dir>/plots_custom/experiment_1_columns_first_mid_last.png)')
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    results = load_results(res_dir, args.results_file)
    layers, gm = collect_group_means_by_layer(results)

    if args.output:
        out = Path(args.output)
    else:
        out = res_dir / 'plots_custom' / 'experiment_1_columns_first_mid_last.png'

    plot_columns(layers, gm, out)
    print(f'[INFO] Saved plot to: {out}')


if __name__ == '__main__':
    main()

