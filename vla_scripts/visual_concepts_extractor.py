#!/usr/bin/env python3
"""
Visual Concepts Extractor (LIBERO-native)

This module computes visual / spatial relations by delegating to LIBERO's object
state wrappers and predicate implementations. It deliberately avoids any manual
geometry math and only uses:

- LIBERO env object states: `ObjectState`, `SiteObjectState`
- Region geometry: `SiteObject.in_box`, `TargetZone` logic
- Predicates: `In`, `On` (from `libero.libero.envs.predicates`)

Outputs can be consumed by both trajectory reconstruction and data collection.
It also includes a CSV recorder that logs per-timestep concept truth values per task.

Relations computed per timestep:
- contact(A, B): symmetric contact between any two non-site objects / fixtures
- in(A, B): LIBERO predicate (arg2.check_contact(arg1) and arg2.check_contain(arg1))
- on(A, B): LIBERO predicate (check_ontop)
- region_contains(R, A): region (site) containment via `SiteObjectState.check_contain`

CSV format (per task): one file per task; first column lists concept names; each
next column is the sequential truth value at timestep t0, t1, ..., tn.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Tuple, Iterable, Optional


def _get_bddl_env(env):
    """Return the underlying BDDLBaseDomain env, regardless of ControlEnv wrapper.

    Accepts either LIBERO ControlEnv (OffScreenRenderEnv / SegmentationRenderEnv) or the
    underlying BDDLBaseDomain instance. Raises if object states are not found.
    """
    # Direct
    if hasattr(env, "object_states_dict"):
        return env
    # Wrapped
    if hasattr(env, "env") and hasattr(env.env, "object_states_dict"):
        return env.env
    raise RuntimeError("Could not locate LIBERO BDDL environment with object_states_dict.")


def _get_task_identifiers(control_or_bddl_env) -> Tuple[str, str]:
    """Return (task_name, language_instruction) if available, else empty strings.
    Works with ControlEnv wrapper or raw BDDL env.
    """
    # ControlEnv exposes these directly
    problem_name = getattr(control_or_bddl_env, "problem_name", None)
    language_instruction = getattr(control_or_bddl_env, "language_instruction", None)
    if problem_name is not None:
        return str(problem_name), str(language_instruction or "")

    # Try underlying env
    try:
        bddl_env = _get_bddl_env(control_or_bddl_env)
        parsed = getattr(bddl_env, "parsed_problem", {}) or {}
        return str(parsed.get("problem_name", "")), " ".join(parsed.get("language_instruction", []))
    except Exception:
        return "", ""


def _sanitize_filename(s: str) -> str:
    """Sanitize a string for safe filesystem usage."""
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "task"


def _get_object_states(env) -> Dict[str, object]:
    """Return the mapping of name -> ObjectState / SiteObjectState."""
    bddl_env = _get_bddl_env(env)
    return getattr(bddl_env, "object_states_dict")


def _split_site_vs_non_site(object_states: Dict[str, object]) -> Tuple[List[str], List[str]]:
    """Split object names into non-site and site lists based on object_state_type."""
    non_site, site = [], []
    for name, state in object_states.items():
        t = getattr(state, "object_state_type", None)
        if t == "site":
            site.append(name)
        else:
            non_site.append(name)
    return sorted(non_site), sorted(site)


def enumerate_concept_keys(env) -> List[str]:
    """Enumerate all concept keys for the current scene in a deterministic order.

    Concepts include:
    - contact(A,B) over unordered pairs of non-site objects (A < B)
    - in(A,B) over ordered pairs: A in B, with A non-site; B can be non-site or site
    - on(A,B) over ordered pairs: A on B, with A non-site; B can be non-site or site
    - region_contains(R,A) over sites R and non-site objects A
    """
    object_states = _get_object_states(env)
    non_site, site = _split_site_vs_non_site(object_states)

    concepts: List[str] = []

    # contact over unordered pairs of non-site
    for i in range(len(non_site)):
        for j in range(i + 1, len(non_site)):
            a, b = non_site[i], non_site[j]
            concepts.append(f"contact({a},{b})")

    # in and on over ordered pairs (A non-site; B any)
    any_targets = non_site + site
    for a in non_site:
        for b in any_targets:
            if a == b:
                continue
            concepts.append(f"in({a},{b})")
    for a in non_site:
        for b in any_targets:
            if a == b:
                continue
            concepts.append(f"on({a},{b})")

    # region_contains over (site, non-site)
    for r in site:
        for a in non_site:
            concepts.append(f"region_contains({r},{a})")

    return concepts


def _eval_contact(a_state, b_state) -> bool:
    """Use LIBERO ObjectState.check_contact."""
    try:
        return bool(a_state.check_contact(b_state))
    except Exception:
        return False


def _eval_in(a_state, b_state) -> bool:
    """Use LIBERO predicate In (combines contact + contain)."""
    try:
        from libero.libero.envs.predicates import eval_predicate_fn
        return bool(eval_predicate_fn("in", a_state, b_state))
    except Exception:
        return False


def _eval_on(a_state, b_state) -> bool:
    """Use LIBERO predicate On (check_ontop)."""
    try:
        from libero.libero.envs.predicates import eval_predicate_fn
        return bool(eval_predicate_fn("on", a_state, b_state))
    except Exception:
        return False


def _eval_region_contains(r_site_state, a_state) -> bool:
    """Use SiteObjectState.check_contain (pure region containment)."""
    try:
        return bool(r_site_state.check_contain(a_state))
    except Exception:
        return False


def evaluate_concepts(env, concept_keys: Iterable[str]) -> Dict[str, int]:
    """Evaluate the given concept keys on the current simulator state.

    Returns: mapping concept_key -> {0,1}
    """
    object_states = _get_object_states(env)

    out: Dict[str, int] = {}
    for key in concept_keys:
        # key format: relation(arg1,arg2) or region_contains(region,arg)
        try:
            head, rest = key.split("(", 1)
            args = rest[:-1]  # drop trailing ')'
            if head == "contact":
                a, b = args.split(",")
                a_state, b_state = object_states.get(a), object_states.get(b)
                val = _eval_contact(a_state, b_state) if a_state and b_state else False
            elif head == "in":
                a, b = args.split(",")
                a_state, b_state = object_states.get(a), object_states.get(b)
                val = _eval_in(a_state, b_state) if a_state and b_state else False
            elif head == "on":
                a, b = args.split(",")
                a_state, b_state = object_states.get(a), object_states.get(b)
                val = _eval_on(a_state, b_state) if a_state and b_state else False
            elif head == "region_contains":
                r, a = args.split(",")
                r_state, a_state = object_states.get(r), object_states.get(a)
                val = _eval_region_contains(r_state, a_state) if r_state and a_state else False
            else:
                val = False
        except Exception:
            val = False
        out[key] = 1 if val else 0
    return out


class CSVRelationsRecorder:
    """Accumulates time series of concept truth values and writes a per-task CSV.

    Usage:
        concepts = enumerate_concept_keys(env)
        rec = CSVRelationsRecorder(task_name)
        rec.initialize(concepts)
        for t in timesteps:
            snapshot = evaluate_concepts(env, concepts)
            rec.append(snapshot)
        rec.save(output_dir)
    """

    def __init__(self, task_name: str, language: str = ""):
        self.task_name = task_name
        self.language = language
        self.concepts: List[str] = []
        self.ts: List[Dict[str, int]] = []

    def initialize(self, concepts: List[str]):
        self.concepts = list(concepts)
        self.ts = []

    def append(self, snapshot: Dict[str, int]):
        if not self.concepts:
            # infer concept order from first snapshot
            self.concepts = sorted(snapshot.keys())
        # Keep only known concepts; missing keys default to 0
        row = {k: int(snapshot.get(k, 0)) for k in self.concepts}
        self.ts.append(row)

    def save(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        base = _sanitize_filename(self.task_name or "task")
        path = os.path.join(output_dir, f"{base}__relations.csv")

        # Prepare columns: concept, t0, t1, ...
        n_steps = len(self.ts)
        headers = ["concept"] + [f"t{i}" for i in range(n_steps)]

        # Write CSV
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # Optional metadata header as a comment row
            if self.language:
                writer.writerow([f"# task={self.task_name}", f"language={self.language}"])
            else:
                writer.writerow([f"# task={self.task_name}"])

            writer.writerow(headers)
            for concept in self.concepts:
                row = [concept]
                for i in range(n_steps):
                    row.append(self.ts[i].get(concept, 0))
                writer.writerow(row)
        return path

    def save_as_task_csv(self, root_dir: str) -> str:
        """Save as a single CSV named by language instruction (preferred) or task.

        Writes to: `{root_dir}/{sanitized_language_or_task}.csv`
        """
        os.makedirs(root_dir, exist_ok=True)
        base_name = self.language.strip() if isinstance(self.language, str) else ""
        if not base_name:
            base_name = self.task_name or "task"
        base = _sanitize_filename(base_name)
        path = os.path.join(root_dir, f"{base}.csv")

        n_steps = len(self.ts)
        headers = ["concept"] + [f"t{i}" for i in range(n_steps)]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            if self.language:
                writer.writerow([f"# task={self.task_name}", f"language={self.language}"])
            else:
                writer.writerow([f"# task={self.task_name}"])
            writer.writerow(headers)
            for concept in self.concepts:
                row = [concept]
                for i in range(n_steps):
                    row.append(self.ts[i].get(concept, 0))
                writer.writerow(row)
        return path


# Backwards-compat: simple sim state getter for callers that expect it.
def extract_simulator_state(env) -> Dict[str, object]:
    """Minimal simulator state for compatibility with reconstruction scripts.

    Returns a dict with low-level MuJoCo state if available (flattened), plus
    task identifiers. This function does not compute any geometry; it only
    accesses existing env APIs.
    """
    state: Dict[str, object] = {}

    # Try to get flat MuJoCo state from ControlEnv
    try:
        if hasattr(env, "get_sim_state"):
            state["mujoco_flat_state"] = env.get_sim_state()
        elif hasattr(env, "env") and hasattr(env.env, "sim"):
            # Fallback to raw MuJoCo data getter if present
            sim = env.env.sim
            if hasattr(sim, "get_state"):
                state["mujoco_flat_state"] = sim.get_state().flatten()
    except Exception:
        pass

    task_name, language = _get_task_identifiers(env)
    state["task_name"] = task_name
    state["language_instruction"] = language
    return state


# Convenience helpers for one-shot extraction + saving
def extract_and_record_all_relations(env, output_dir: str) -> str:
    """Enumerate, evaluate, and save all relations for the current task as CSV.

    Returns the CSV path.
    """
    task_name, language = _get_task_identifiers(env)
    concepts = enumerate_concept_keys(env)
    snapshot = evaluate_concepts(env, concepts)
    rec = CSVRelationsRecorder(task_name, language)
    rec.initialize(concepts)
    rec.append(snapshot)
    return rec.save(output_dir)


def accumulate_relations_over_time(env, recorder: Optional[CSVRelationsRecorder]) -> Tuple[List[str], Dict[str, int]]:
    """Evaluate and optionally append to a recorder for time series logging.

    Returns (concepts, snapshot).
    """
    concepts = recorder.concepts if (recorder and recorder.concepts) else enumerate_concept_keys(env)
    snapshot = evaluate_concepts(env, concepts)
    if recorder is not None:
        if not recorder.concepts:
            recorder.initialize(concepts)
        recorder.append(snapshot)
    return concepts, snapshot
