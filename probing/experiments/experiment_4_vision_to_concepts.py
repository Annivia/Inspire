#!/usr/bin/env python3
"""
experiment_4_vision_to_concepts.py

Experiment 4: [Vision features] -> visual concepts (general_1)
Supports raw vision patches and/or VLM embeddings. Uses test/hash selection
without parsing TXT files.
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


def _load_vision_for_shard(shard_dir: Path, vision_type: str = 'raw') -> np.ndarray:
    import h5py
    if vision_type == 'raw':
        fpath = shard_dir / 'vision_features.h5'
        with h5py.File(fpath, 'r') as f:
            # [N, num_patches, vision_dim] -> flatten patches
            v = f['vision_features'][:]
            N = v.shape[0]
            v = v.reshape(N, -1)
        return v
    elif vision_type == 'vlm':
        fpath = shard_dir / 'vlm_embeddings.h5'
        with h5py.File(fpath, 'r') as f:
            v = f['vlm_embeddings'][:]
        return v
    else:
        raise ValueError("vision_type must be 'raw' or 'vlm'")


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
    import h5py
    if not tgt or not reg:
        return None
    needed = [f'contact({tgt},gripper)', f'ontop({tgt},{reg})', f'contact({tgt},{reg})']
    cfile = shard_dir / 'concepts.h5'
    if not cfile.exists():
        return None
    with h5py.File(cfile, 'r') as f:
        names = f['concept_names'][:]
        names = [n.decode('utf-8') if hasattr(n, 'decode') else str(n) for n in names]
        name_to_idx = {n: i for i, n in enumerate(names)}
        if not all(n in name_to_idx for n in needed):
            return None
        mat = f['concepts'][:]
        mask = None
        if successful_only and 'episode_success' in f:
            mask = (f['episode_success'][:] > 0)
        idxs = [name_to_idx[n] for n in needed]
        Y = mat[:, idxs].astype(np.float32)
        if mask is not None:
            Y = Y[mask]
    return Y


def run_experiment_4_general_1(
    data_root: str,
    output_dir: str,
    vision_type: str = 'both',  # 'raw', 'vlm', or 'both'
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
    out_exp = out_root / 'experiment_4_vision_to_concepts' / 'general_1'
    out_exp.mkdir(parents=True, exist_ok=True)

    tasks = load_hashes()
    selected: List[Tuple[str, Dict, Dict, str, str, List[str]]] = []
    for base, rel, chk in tasks:
        tgt, reg, concepts = _derive_general_1_targets(rel, chk)
        if tgt and concepts:
            selected.append((base, rel, chk, tgt, reg, concepts))
    if max_tasks is not None:
        selected = selected[:max_tasks]

    all_results = {
        'experiment_id': 4,
        'experiment_name': 'vision_to_concepts_general_1',
        'task_count': len(selected),
        'vision_type': vision_type,
        'data_root': str(data_dir),
        'timestamp': time.time(),
        'per_task': {}
    }

    vt_list = ['raw', 'vlm'] if vision_type == 'both' else [vision_type]

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
            vt_results: Dict[str, Dict] = {}
            for vt in vt_list:
                try:
                    X = _load_vision_for_shard(shard_dir, vt)
                    Y, sel_names = _load_concepts_for_shard(shard_dir, concept_names, successful_only=successful_only)
                    if X.shape[0] != Y.shape[0]:
                        n = min(X.shape[0], Y.shape[0])
                        X = X[:n]
                        Y = Y[:n]
                    if X.size == 0 or Y.size == 0:
                        raise ValueError('Empty X or Y after alignment')
                    probe_name = f'{shard_name}__{vt}_to_concepts'
                    res = run_probe_with_baselines(
                        X=X, y=Y, probe_name=probe_name, task_type='regression', test_size=test_size,
                        random_seed=random_seed, debug=debug
                    )
                except Exception as e:
                    res = {'error': str(e)}
                vt_results[vt] = res
                save_probe_results(res, task_out_dir / f'{vt}_results.json')
            all_results['per_task'][shard_name] = {
                'target_concepts': concept_names,
                'results_by_input': vt_results
            }
            with open(task_out_dir / 'task_results_summary.json', 'w') as f:
                json.dump(all_results['per_task'][shard_name], f, indent=2)

    with open(out_exp / 'experiment_4_general_1_complete.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Unified across all tasks
    aggregate_dir = out_exp / 'aggregate'
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    all_results['aggregate'] = {'by_input': {}}

    vt_list = ['raw', 'vlm'] if vision_type == 'both' else [vision_type]
    for vt in vt_list:
        vt_summary: Dict[str, Dict] = {}
        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        for base, rel, chk, tgt, reg, concept_names in selected:
            shard_name = _to_shard_dirname(base)
            shard_dir = data_dir / shard_name
            if not shard_dir.exists():
                continue
            try:
                X = _load_vision_for_shard(shard_dir, vt)
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
            res = {'error': 'No aligned unified data for this vision type'}
        else:
            X_all = np.concatenate(X_list, axis=0)
            Y_all = np.concatenate(Y_list, axis=0)
            try:
                probe_name = f'UNIFIED_{vt}_to_concepts'
                res = run_probe_with_baselines(
                    X=X_all, y=Y_all, probe_name=probe_name, task_type='regression',
                    test_size=test_size, random_seed=random_seed, debug=debug
                )
            except Exception as e:
                res = {'error': str(e)}
        vt_summary['unified'] = res
        save_probe_results(res, aggregate_dir / f'{vt}_results.json')
        all_results['aggregate']['by_input'][vt] = vt_summary

    with open(out_exp / 'experiment_4_general_1_complete.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    return all_results
