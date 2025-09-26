#!/usr/bin/env python3
"""
visualize_experiment_3.py

Visualization for Experiment 3: [Hidden state] -> visual concepts (general_1).
Aggregates results across tasks and layers and produces publication-ready PNGs.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette('husl')


def _default_results_file() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir / 'results' / 'experiment_3' / 'experiment_3_hidden_to_concepts' / 'general_1' / 'experiment_3_general_1_complete.json'


def load_results(results_file: Path) -> Dict:
    with open(results_file, 'r') as f:
        return json.load(f)


def aggregate_by_layer(results: Dict) -> Tuple[List[int], Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]]]:
    layers = set()
    normal: Dict[int, List[float]] = {}
    randomized: Dict[int, List[float]] = {}
    noise: Dict[int, List[float]] = {}

    per_task = results.get('per_task', {})
    for _, task in per_task.items():
        by_layer = task.get('results_by_layer', {})
        for layer_key, gen_data in by_layer.items():
            try:
                L = int(layer_key.split('_')[1])
            except Exception:
                continue
            layers.add(L)
            normal.setdefault(L, [])
            randomized.setdefault(L, [])
            noise.setdefault(L, [])
            for _, res in gen_data.items():
                if not isinstance(res, dict):
                    continue
                if 'normal' in res and 'randomized' in res and 'noise' in res:
                    if 'error' in res['normal'] or 'error' in res['randomized'] or 'error' in res['noise']:
                        continue
                    normal[L].append(res['normal'].get('r2_test', 0.0))
                    randomized[L].append(res['randomized'].get('r2_test', 0.0))
                    noise[L].append(res['noise'].get('r2_test', 0.0))
    return sorted(layers), normal, randomized, noise


def aggregate_unified_by_layer(results: Dict) -> Tuple[List[int], Dict[str, Dict[int, List[float]]]]:
    """Use unified aggregate results if present. Returns per-dimension per-layer lists.
    dims are labeled as: contact_gripper, ontop_region, contact_region.
    """
    out: Dict[str, Dict[int, List[float]]] = {
        'contact_gripper': {}, 'ontop_region': {}, 'contact_region': {}
    }
    agg = results.get('aggregate', {}).get('by_layer', {})
    if not agg:
        return [], out
    layers = []
    for layer_key, gen in agg.items():
        try:
            L = int(layer_key.split('_')[1])
        except Exception:
            continue
        layers.append(L)
        res = gen.get('generation_step_0') or {}
        normal = res.get('normal', {})
        # Expect 3 dims
        vals = [normal.get('r2_test_dim_0'), normal.get('r2_test_dim_1'), normal.get('r2_test_dim_2')]
        keys = ['contact_gripper', 'ontop_region', 'contact_region']
        for k, v in zip(keys, vals):
            if v is None:
                continue
            out[k].setdefault(L, [])
            out[k][L].append(float(v))
    return sorted(set(layers)), out


def _concept_type_from_name(name: str) -> str:
    try:
        if name.startswith('contact(') and name.endswith(',gripper)'):
            return 'contact_gripper'
        if name.startswith('ontop('):
            return 'ontop_region'
        if name.startswith('contact(') and not name.endswith(',gripper)'):
            return 'contact_region'
    except Exception:
        pass
    return ''


def per_task_concept_metrics(results: Dict) -> Dict[str, Dict[int, List[float]]]:
    """Collect per-layer distributions for each concept type from per-task results."""
    out: Dict[str, Dict[int, List[float]]] = {
        'contact_gripper': {}, 'ontop_region': {}, 'contact_region': {}
    }
    per_task = results.get('per_task', {})
    for _, task in per_task.items():
        target_concepts = task.get('target_concepts', [])
        by_layer = task.get('results_by_layer', {})
        for layer_key, gen in by_layer.items():
            try:
                L = int(layer_key.split('_')[1])
            except Exception:
                continue
            res = gen.get('generation_step_0') or {}
            normal = res.get('normal', {})
            for i, cname in enumerate(target_concepts):
                ctype = _concept_type_from_name(cname)
                if not ctype:
                    continue
                key = f'r2_test_dim_{i}'
                if key in normal:
                    out[ctype].setdefault(L, [])
                    out[ctype][L].append(float(normal[key]))
    return out


def collect_dimwise_r2(results: Dict) -> List[float]:
    vals: List[float] = []
    per_task = results.get('per_task', {})
    for _, task in per_task.items():
        by_layer = task.get('results_by_layer', {})
        for _, gen_data in by_layer.items():
            for _, res in gen_data.items():
                if not isinstance(res, dict):
                    continue
                normal = res.get('normal', {})
                for k, v in normal.items():
                    if isinstance(k, str) and k.startswith('r2_test_dim_'):
                        try:
                            vals.append(float(v))
                        except Exception:
                            pass
    return vals


def plot_r2_by_layer(layers: List[int], normal: Dict[int, List[float]], randomized: Dict[int, List[float]], noise: Dict[int, List[float]], out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 8))
    unique_layers = layers
    x = np.arange(len(unique_layers))
    width = 0.25

    def mean_std(dct: Dict[int, List[float]]):
        means = [np.mean(dct.get(L, [0.0])) if dct.get(L) else 0.0 for L in unique_layers]
        stds = [np.std(dct.get(L, [0.0])) if dct.get(L) and len(dct[L]) > 1 else 0.0 for L in unique_layers]
        return means, stds

    n_means, n_stds = mean_std(normal)
    r_means, r_stds = mean_std(randomized)
    z_means, z_stds = mean_std(noise)

    b1 = ax.bar(x - width, n_means, width, yerr=n_stds, label='Normal', alpha=0.85, color='#2E86AB')
    b2 = ax.bar(x, r_means, width, yerr=r_stds, label='Randomized', alpha=0.85, color='#A23B72')
    b3 = ax.bar(x + width, z_means, width, yerr=z_stds, label='Noise', alpha=0.85, color='#F18F01')

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('R² (Concepts)', fontsize=12)
    ax.set_title('Hidden → Concepts: Mean R² by Layer (general_1)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{L}' for L in unique_layers])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_linear_separability_scatter(results: Dict, out_path: Path):
    xs, ys, labels = [], [], []
    per_task = results.get('per_task', {})
    for _, task in per_task.items():
        by_layer = task.get('results_by_layer', {})
        for layer_key, gen_data in by_layer.items():
            try:
                L = int(layer_key.split('_')[1])
            except Exception:
                L = -1
            for _, res in gen_data.items():
                if not isinstance(res, dict):
                    continue
                if 'normal' in res and 'randomized' in res:
                    if 'error' in res['normal'] or 'error' in res['randomized']:
                        continue
                    xs.append(res['normal'].get('r2_test', 0.0))
                    ys.append(res['randomized'].get('r2_test', 0.0))
                    labels.append(L)

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(xs, ys, c=labels, cmap='viridis', alpha=0.8, s=45)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('Normal R² (Original)', fontsize=12)
    ax.set_ylabel('Randomized R² (Shuffled)', fontsize=12)
    ax.set_title('Hidden → Concepts: Linear Separability (general_1)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Layer Index')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_concept_dim_distribution(results: Dict, out_path: Path):
    vals = collect_dimwise_r2(results)
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(vals, bins=30, kde=True, ax=ax, color='#2E86AB', alpha=0.85)
    ax.set_xlabel('R² Across Concept Dimensions (Normal)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Hidden → Concepts: Distribution of Per-Concept R² (general_1)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Experiment 3 (Hidden → Concepts) results')
    parser.add_argument('--results-file', type=str, default=None, help='Path to experiment_3_general_1_complete.json')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    results_file = Path(args.results_file) if args.results_file else _default_results_file()
    if args.debug:
        print(f"[DEBUG] Loading results: {results_file}")
    results = load_results(results_file)

    # Default output dir: probing/results/experiment3_visual
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).resolve().parent / 'results' / 'experiment3_visual'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try unified aggregate first for robust plots
    uni_layers, uni = aggregate_unified_by_layer(results)
    if uni_layers:
        # Per-concept figures
        for concept_key, pretty in [('contact_gripper', 'Obj–Gripper Contact'),
                                    ('ontop_region', 'Ontop Region'),
                                    ('contact_region', 'Obj–Region Contact')]:
            # Build mean per layer
            x = uni_layers
            means = [np.mean(uni[concept_key].get(L, [0.0])) for L in x]
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(x, means, 'o-', color='#2E86AB', lw=2.0)
            ax.set_xticks(x)
            ax.set_xticklabels([f'L{L}' for L in x])
            ax.set_ylabel('R² (Normal)')
            ax.set_xlabel('Layer Index')
            ax.set_title(f'Hidden → Concepts ({pretty}) — Unified across tasks')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f'experiment_3_unified_{concept_key}_by_layer.png', dpi=300, bbox_inches='tight')
            plt.close()
        # Also, create violin plots from per-task distributions if available
        dist = per_task_concept_metrics(results)
        for concept_key, pretty in [('contact_gripper', 'Obj–Gripper Contact'),
                                    ('ontop_region', 'Ontop Region'),
                                    ('contact_region', 'Obj–Region Contact')]:
            data = []
            labels = []
            for L in uni_layers:
                vals = dist.get(concept_key, {}).get(L, [])
                if not vals:
                    continue
                data.append(vals)
                labels.append(f'L{L}')
            if data:
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.violinplot(data=data, ax=ax, inner='box')
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, rotation=0)
                ax.set_ylabel('R² (Normal)')
                ax.set_title(f'Hidden → Concepts ({pretty}) — Per-task distributions')
                ax.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / f'experiment_3_violin_{concept_key}.png', dpi=300, bbox_inches='tight')
                plt.close()
    else:
        # Fall back to per-task aggregation if unified not available
        layers, normal, randomized, noise = aggregate_by_layer(results)
        if layers:
            plot_r2_by_layer(layers, normal, randomized, noise, out_dir / 'experiment_3_r2_by_layer.png')
            plot_linear_separability_scatter(results, out_dir / 'experiment_3_linear_separability.png')
            plot_concept_dim_distribution(results, out_dir / 'experiment_3_concept_dim_distribution.png')
        else:
            # Save an empty sentinel to indicate no data
            with open(out_dir / 'experiment_3_NO_VALID_DATA.txt', 'w') as f:
                f.write('No valid results found in experiment_3 results. Ensure reconstruction produced concepts and rerun probe3.sh.')

    if args.debug:
        print(f"[DEBUG] Plots saved to: {out_dir}")


if __name__ == '__main__':
    main()
