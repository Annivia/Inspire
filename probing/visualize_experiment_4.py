#!/usr/bin/env python3
"""
visualize_experiment_4.py

Visualization for Experiment 4: [Vision features] -> visual concepts (general_1).
Aggregates results across tasks and produces PNGs comparing raw vs VLM features.
"""

from pathlib import Path
from typing import Dict, List
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette('husl')


def _default_results_file() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir / 'results' / 'experiment_4' / 'experiment_4_vision_to_concepts' / 'general_1' / 'experiment_4_general_1_complete.json'


def load_results(results_file: Path) -> Dict:
    with open(results_file, 'r') as f:
        return json.load(f)


def _get_nested(d: Dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def collect_unified_concept_conditions(results: Dict) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return per-concept R² for each condition and input type from unified results.
    Structure: {concept: { 'raw': {cond: value}, 'vlm': {cond: value} }}
    Concept keys: contact_gripper, ontop_region, contact_region
    """
    concept_keys = ['contact_gripper', 'ontop_region', 'contact_region']
    dim_map = {'contact_gripper': 0, 'ontop_region': 1, 'contact_region': 2}
    out = {k: {'raw': {}, 'vlm': {}} for k in concept_keys}

    agg = _get_nested(results, ['aggregate', 'by_input'], {})
    for vt in ['raw', 'vlm']:
        unified = _get_nested(agg, [vt, 'unified'], {})
        for cond in ['normal', 'randomized', 'noise']:
            metrics = unified.get(cond) or {}
            # pull per-dimension test R² if present; else fall back to overall r2_test
            for ckey in concept_keys:
                dim = dim_map[ckey]
                val = metrics.get(f'r2_test_dim_{dim}')
                if val is None:
                    val = metrics.get('r2_test', 0.0)
                try:
                    out[ckey][vt][cond] = float(val)
                except Exception:
                    out[ckey][vt][cond] = 0.0
    return out


# (no longer using per-task aggregation in this visualizer)


def plot_concept_bars(concept_label: str, values: Dict[str, Dict[str, float]], out_path: Path):
    labels = ['Normal', 'Randomized', 'Noise']
    conds = ['normal', 'randomized', 'noise']
    x = np.arange(len(labels))
    width = 0.35

    raw_vals = [values.get('raw', {}).get(c, 0.0) for c in conds]
    vlm_vals = [values.get('vlm', {}).get(c, 0.0) for c in conds]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width/2, raw_vals, width, label='Raw patches', color='#2E86AB', alpha=0.85)
    ax.bar(x + width/2, vlm_vals, width, label='VLM embeddings', color='#F18F01', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('R² (Concept)')
    ax.set_title(f'Vision → Concepts: {concept_label} (general_1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# removed old linear separability scatter for exp4


def main():
    parser = argparse.ArgumentParser(description='Visualize Experiment 4 (Vision → Concepts) results')
    parser.add_argument('--results-file', type=str, default=None, help='Path to experiment_4_general_1_complete.json')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save plots')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    results_file = Path(args.results_file) if args.results_file else _default_results_file()
    if args.debug:
        print(f"[DEBUG] Loading results: {results_file}")
    results = load_results(results_file)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).resolve().parent / 'results' / 'experiment4_visual'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build three concept-specific bar charts from unified metrics (no per-task plots)
    data = collect_unified_concept_conditions(results)
    label_map = {
        'contact_gripper': 'Obj–Gripper Contact',
        'ontop_region': 'Ontop Region',
        'contact_region': 'Obj–Region Contact',
    }
    for key, label in label_map.items():
        plot_concept_bars(label, data.get(key, {}), out_dir / f'experiment_4_{key}_bars.png')

    if args.debug:
        print(f"[DEBUG] Plots saved to: {out_dir}")


if __name__ == '__main__':
    main()
