#!/usr/bin/env python3
"""
experiment_3_hidden_to_concepts.py

Experiment 3: [Hidden state] -> visual concepts (general_1)
Uses test/hash selection (no TXT parsing). Per-shard probing, regression on
binary concept vectors to preserve the strictly-linear framework.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import numpy as np

sys.path.append('/u/xzhang42/Inspire')

from probing.linear_probe import run_probe_with_baselines, save_probe_results
from probing.generate_concept_selections import load_hashes, is_pick_and_place, first_or_none


def _derive_general_1_targets(rel: Dict, chk: Dict) -> Tuple[str, str, List[str]]:
    meta = (rel.get('meta') if rel else None) or (chk.get('meta') if chk else None) or {}
    lang = str(meta.get('language_instruction') or '')
    involved_objs = [o for o in (meta.get('involved_objects') or []) if o != 'gripper']
    involved_sites = meta.get('involved_sites') or []
    rel_names = set((rel.get('concepts') or {}).keys())
    chk_names = set((chk.get('concepts') or {}).keys())
    all_names = rel_names | chk_names
    if not (is_pick_and_place(lang) and len(involved_objs) == 1):
        return '', '', []
    tgt = first_or_none(involved_objs)
    reg = first_or_none(involved_sites) if involved_sites else ''
    if not reg:
        cand_sites: List[str] = []
        for name in rel_names:
            if name.startswith(f'on({tgt},') or name.startswith(f'in({tgt},'):
                try:
                    site = name.split(',', 1)[1][:-1]
                    cand_sites.append(site)
                except Exception:
                    pass
        suf = f',{tgt})'
        for name in rel_names:
            if name.startswith('region_contains(') and name.endswith(suf):
                site = name.split('(', 1)[1].split(',', 1)[0]
                cand_sites.append(site)
        if cand_sites:
            cand_sites = sorted(set(cand_sites), key=lambda s: (('contain' in s.lower()), ('init' not in s.lower())), reverse=True)
            reg = cand_sites[0]
    candidates = [
        f'contact({tgt},gripper)',
        f'on({tgt},{reg})' if reg else '',
        f'ontop({tgt},{reg})' if reg else '',
        f'contact({tgt},{reg})' if reg else '',
    ]
    concepts = [c for c in candidates if c and c in all_names]
    return tgt, reg, concepts


def _to_shard_dirname(base_key: str) -> str:
    if '__' not in base_key:
        return base_key
    instr, scene = base_key.split('__', 1)
    return f"{scene}__{instr}"


def _load_hidden_states_for_shard(shard_dir: Path, layer_idx: int, gen_step: int) -> np.ndarray:
    import h5py
    fpath = shard_dir / 'hidden_states' / f'generation_step_{gen_step}.h5'
    ds_name = f'layer_{layer_idx:02d}'
    with h5py.File(fpath, 'r') as f:
        if ds_name not in f:
            raise ValueError(f"Layer {layer_idx} not found in {fpath}")
        X = f[ds_name][:]
    # Ensure strictly 2D features [N, D]
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _load_concepts_for_shard(shard_dir: Path, concept_names: List[str], successful_only: bool = True) -> Tuple[np.ndarray, List[str]]:
    import h5py
    cfile = shard_dir / 'concepts.h5'
    with h5py.File(cfile, 'r') as f:
        names = f['concept_names'][:]
        names = [n.decode('utf-8') if hasattr(n, 'decode') else str(n) for n in names]
        name_to_idx = {n: i for i, n in enumerate(names)}
        concepts_mat = f['concepts'][:]
        mask = None
        if successful_only and 'episode_success' in f:
            mask = (f['episode_success'][:] > 0)
        indices = [name_to_idx[c] for c in concept_names if c in name_to_idx]
        if not indices:
            return np.zeros((0, 0), dtype=np.float32), []
        Y = concepts_mat[:, indices].astype(np.float32)
        if mask is not None:
            Y = Y[mask]
        sel_names = [names[i] for i in indices]
    return Y, sel_names


def _load_concepts_three_core(
    shard_dir: Path,
    tgt: str,
    reg: str,
    successful_only: bool = True
) -> Optional[np.ndarray]:
    """
    Load a fixed 3-column targets matrix for unified probing across tasks:
    columns = [contact(tgt,gripper), ontop(tgt,reg), contact(tgt,reg)].
    Returns None if any required concept is missing for this shard.
    """
    import h5py
    if not tgt or not reg:
        return None
    names_needed = [f'contact({tgt},gripper)', f'ontop({tgt},{reg})', f'contact({tgt},{reg})']
    cfile = shard_dir / 'concepts.h5'
    if not cfile.exists():
        return None
    with h5py.File(cfile, 'r') as f:
        names = f['concept_names'][:]
        names = [n.decode('utf-8') if hasattr(n, 'decode') else str(n) for n in names]
        name_to_idx = {n: i for i, n in enumerate(names)}
        if not all(n in name_to_idx for n in names_needed):
            return None
        concepts_mat = f['concepts'][:]
        mask = None
        if successful_only and 'episode_success' in f:
            mask = (f['episode_success'][:] > 0)
        indices = [name_to_idx[n] for n in names_needed]
        Y = concepts_mat[:, indices].astype(np.float32)
        if mask is not None:
            Y = Y[mask]
    return Y


def run_experiment_3_general_1(
    data_root: str,
    output_dir: str,
    layers: Optional[List[int]] = None,
    generation_steps: Optional[List[int]] = None,
    successful_only: bool = True,
    max_tasks: Optional[int] = None,
    test_size: float = 0.2,
    random_seed: int = 42,
    debug: bool = False,
    unify_only: bool = True
) -> Dict:
    start_time = time.time()
    root = Path(data_root)
    data_dir = root if (root / 'episode_index.h5').exists() else (root / 'optimized_trajectory_data')
    out_root = Path(output_dir)
    out_exp = out_root / 'experiment_3_hidden_to_concepts' / 'general_1'
    out_exp.mkdir(parents=True, exist_ok=True)

    tasks = load_hashes()
    selected: List[Tuple[str, Dict, Dict, str, str, List[str]]] = []
    for base, rel, chk in tasks:
        tgt, reg, concepts = _derive_general_1_targets(rel, chk)
        if tgt and concepts:
            selected.append((base, rel, chk, tgt, reg, concepts))
    if max_tasks is not None:
        selected = selected[:max_tasks]
    if generation_steps is None:
        generation_steps = [0]
    if layers is None:
        layers = list(range(6))

    all_results = {
        'experiment_id': 3,
        'experiment_name': 'hidden_state_to_concepts_general_1',
        'task_count': len(selected),
        'layers': layers,
        'generation_steps': generation_steps,
        'data_root': str(data_dir),
        'timestamp': time.time(),
        'per_task': {}
    }

    if not unify_only:
        for base, rel, chk, tgt, reg, concept_names in selected:
            shard_name = _to_shard_dirname(base)
            shard_dir = data_dir / shard_name
            if not shard_dir.exists():
                if debug:
                    print(f"[DEBUG] Shard not found, skipping: {shard_dir}")
                continue
            task_out_dir = out_exp / shard_name
            task_out_dir.mkdir(parents=True, exist_ok=True)
            per_layer_results: Dict[str, Dict] = {}
            for L in layers:
                gen_results: Dict[str, Dict] = {}
                for gs in generation_steps:
                    try:
                        X = _load_hidden_states_for_shard(shard_dir, L, gs)
                        Y, sel_names = _load_concepts_for_shard(shard_dir, concept_names, successful_only=successful_only)
                        if X.shape[0] != Y.shape[0]:
                            n = min(X.shape[0], Y.shape[0])
                            X = X[:n]
                            Y = Y[:n]
                        if X.size == 0 or Y.size == 0:
                            raise ValueError('Empty X or Y after alignment')
                        probe_name = f'{shard_name}__layer_{L}_gen_{gs}_to_concepts'
                        res = run_probe_with_baselines(
                            X=X, y=Y, probe_name=probe_name, task_type='regression', test_size=test_size,
                            random_seed=random_seed, debug=debug
                        )
                    except Exception as e:
                        res = {'error': str(e)}
                    gen_results[f'generation_step_{gs}'] = res
                    save_probe_results(res, task_out_dir / f'layer_{L}_gen_{gs}_results.json')
                per_layer_results[f'layer_{L}'] = gen_results
            all_results['per_task'][shard_name] = {
                'target_concepts': concept_names,
                'results_by_layer': per_layer_results
            }
            with open(task_out_dir / 'task_results_summary.json', 'w') as f:
                json.dump(all_results['per_task'][shard_name], f, indent=2)

    with open(out_exp / 'experiment_3_general_1_complete.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Additionally, compute a UNIFIED probe across all selected tasks with fixed 3 targets
    aggregate_dir = out_exp / 'aggregate'
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    all_results['aggregate'] = {'by_layer': {}}

    for L in layers:
        layer_summary: Dict[str, Dict] = {}
        for gs in generation_steps:
            X_list: List[np.ndarray] = []
            Y_list: List[np.ndarray] = []
            for base, rel, chk, tgt, reg, concept_names in selected:
                shard_name = _to_shard_dirname(base)
                shard_dir = data_dir / shard_name
                if not shard_dir.exists():
                    continue
                try:
                    X = _load_hidden_states_for_shard(shard_dir, L, gs)
                    Y = _load_concepts_three_core(shard_dir, tgt, reg, successful_only=successful_only)
                    if Y is None:
                        continue
                    n = min(X.shape[0], Y.shape[0])
                    if n <= 1:
                        continue
                    X_list.append(X[:n])
                    Y_list.append(Y[:n])
                except Exception:
                    continue
            if not X_list or not Y_list:
                res = {'error': 'No aligned unified data for this layer/gen_step'}
            else:
                X_all = np.concatenate(X_list, axis=0)
                Y_all = np.concatenate(Y_list, axis=0)
                try:
                    probe_name = f'UNIFIED_layer_{L}_gen_{gs}_to_concepts'
                    res = run_probe_with_baselines(
                        X=X_all, y=Y_all, probe_name=probe_name, task_type='regression',
                        test_size=test_size, random_seed=random_seed, debug=debug
                    )
                except Exception as e:
                    res = {'error': str(e)}
            layer_summary[f'generation_step_{gs}'] = res
            save_probe_results(res, aggregate_dir / f'layer_{L}_gen_{gs}_results.json')
        all_results['aggregate']['by_layer'][f'layer_{L}'] = layer_summary

    with open(out_exp / 'experiment_3_general_1_complete.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    return all_results
