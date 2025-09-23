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
    """Sanitize a string for safe filesystem usage."""
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "task"


def debug_print_sim_capabilities(env, max_contacts: int = 5) -> None:
    """Print simulator capabilities relevant for concept extraction.

    Prints:
    - Task identifiers
    - Object list (objects / fixtures / sites)
    - Object states available (name → type; key methods)
    - Contacts / collisions info (access + sample contacts)
    - Target containment region definitions for site objects

    Safe to call on either ControlEnv wrappers or raw BDDL env.
    """
    try:
        bddl_env = _get_bddl_env(env)
    except Exception as e:
        print(f"[VCE:debug] Unable to unwrap BDDL env: {e}")
        bddl_env = env

    # Task identifiers
    task_name, language = _get_task_identifiers(env)
    print("\n================= SIM CAPABILITIES =================")
    print(f"Task: {task_name}")
    if language:
        print(f"Instruction: {language}")

    # Object inventory from env dicts when available
    objects = sorted(list(getattr(bddl_env, "objects_dict", {}).keys()))
    fixtures = sorted(list(getattr(bddl_env, "fixtures_dict", {}).keys()))
    sites = sorted(list(getattr(bddl_env, "object_sites_dict", {}).keys()))
    print("-- Object List --")
    print(f"Objects  ({len(objects)}): {objects}")
    print(f"Fixtures ({len(fixtures)}): {fixtures}")
    print(f"Sites    ({len(sites)}): {sites}")

    # Object states available and their key methods
    try:
        object_states = _get_object_states(env)
    except Exception as e:
        object_states = {}
        print(f"[VCE:debug] object_states_dict not found: {e}")
    print("-- Object States --")
    for name in sorted(object_states.keys()):
        st = object_states[name]
        st_type = getattr(st, "object_state_type", type(st).__name__)
        caps = []
        for meth in ("check_contact", "check_contain", "check_ontop", "get_joint_state", "is_open", "is_close"):
            if hasattr(st, meth):
                caps.append(meth)
        extra = []
        if st_type == "site":
            # include parent and site metadata if available
            parent = getattr(st, "parent_name", None)
            if parent is not None:
                extra.append(f"parent={parent}")
        print(f"  - {name}: type={st_type}; methods={caps} {'[' + ', '.join(extra) + ']' if extra else ''}")

    # Contacts / Collisions availability
    print("-- Contacts / Collisions --")
    sim = None
    if hasattr(bddl_env, "sim"):
        sim = bddl_env.sim
    elif hasattr(bddl_env, "env") and hasattr(bddl_env.env, "sim"):
        sim = bddl_env.env.sim

    if sim is not None and hasattr(sim, "data"):
        try:
            ncon = int(getattr(sim.data, "ncon", 0))
            print(f"MuJoCo contacts accessible: ncon={ncon}")
            # Try to map a few contacts to geom / body names
            if ncon > 0 and hasattr(sim.data, "contact") and hasattr(sim.model, "geom_id2name"):
                shown = 0
                for i in range(min(ncon, max_contacts)):
                    c = sim.data.contact[i]
                    g1 = int(getattr(c, "geom1", -1))
                    g2 = int(getattr(c, "geom2", -1))
                    try:
                        g1_name = sim.model.geom_id2name(g1) if g1 >= 0 else "-"
                    except Exception:
                        g1_name = str(g1)
                    try:
                        g2_name = sim.model.geom_id2name(g2) if g2 >= 0 else "-"
                    except Exception:
                        g2_name = str(g2)
                    print(f"  contact[{i}]: {g1_name} <-> {g2_name}")
                    shown += 1
                if shown == 0:
                    print("  (contacts exist but geom names unavailable)")
        except Exception as e:
            print(f"MuJoCo contact access failed: {e}")
    else:
        print("MuJoCo sim.data not available; cannot read contacts")

    # Target containment region definition (site objects)
    print("-- Target Containment Regions (Sites) --")
    site_objs = getattr(bddl_env, "object_sites_dict", {}) or {}
    if not site_objs:
        print("  (no site objects)")
    else:
        for site_name in sorted(site_objs.keys()):
            site_obj = site_objs[site_name]
            cls = type(site_obj)
            mod = getattr(cls, "__module__", "")
            site_type = getattr(site_obj, "site_type", None)
            size = getattr(site_obj, "size", None)
            parent = getattr(site_obj, "parent_name", None)
            has_in_box = hasattr(site_obj, "in_box")
            has_under = hasattr(site_obj, "under")
            print(f"  - {site_name}: class={cls.__name__} ({mod}); site_type={site_type}; size={size}; parent={parent}; in_box={has_in_box}; under={has_under}")

    # Completion-like check: which objects are currently in which regions
    try:
        object_states = _get_object_states(env)
        # separate lists
        non_site_names, site_names = _split_site_vs_non_site(object_states)
        print("-- Region Containment Snapshot (current state) --")
        any_found = False
        for r in site_names:
            r_state = object_states.get(r)
            contained = []
            for a in non_site_names:
                a_state = object_states.get(a)
                if r_state is None or a_state is None:
                    continue
                if _eval_region_contains(r_state, a_state):
                    contained.append(a)
            if contained:
                any_found = True
                print(f"  {r} contains: {contained}")
        if not any_found:
            print("  (no objects reported inside any region)")
    except Exception as e:
        print(f"[VCE:debug] Region containment snapshot failed: {e}")

    # Goal / completion predicates from BDDL and their current truth values
    try:
        parsed = getattr(bddl_env, "parsed_problem", {}) or {}
        goal_state = parsed.get("goal_state", []) or []
        object_states = _get_object_states(env)
        print("-- Goal Predicates (and current truth) --")
        if not goal_state:
            print("  (no goal_state in parsed_problem)")
        else:
            try:
                from libero.libero.envs.predicates import eval_predicate_fn
            except Exception:
                eval_predicate_fn = None
            for st in goal_state:
                ok = False
                desc = None
                try:
                    if len(st) == 3:
                        p, a, b = st
                        a_state = object_states.get(a)
                        b_state = object_states.get(b)
                        if eval_predicate_fn and a_state and b_state:
                            ok = bool(eval_predicate_fn(p, a_state, b_state))
                        desc = f"{p}({a},{b})"
                    elif len(st) == 2:
                        p, a = st
                        a_state = object_states.get(a)
                        if eval_predicate_fn and a_state:
                            ok = bool(eval_predicate_fn(p, a_state))
                        desc = f"{p}({a})"
                except Exception:
                    ok = False
                if desc:
                    print(f"  {desc}: {int(ok)}")
    except Exception as e:
        print(f"[VCE:debug] Goal predicate evaluation failed: {e}")
    print("====================================================\n")




def should_include_under(env) -> bool:
    """Decide automatically whether to include site.under checks for this task.

    Current policy: include only if the language instruction mentions the word
    "under" (case-insensitive).
    """
    try:
        _, language = _get_task_identifiers(env)
        return isinstance(language, str) and ("under" in language.lower())
    except Exception:
        return False

def _get_object_states(env) -> Dict[str, object]:
    """Return the mapping of name -> ObjectState / SiteObjectState."""
    bddl_env = _get_bddl_env(env)
    return getattr(bddl_env, "object_states_dict")


def get_env_inventory(env) -> Dict[str, List[str]]:
    """Return names of objects, fixtures, and sites in the current scene."""
    bddl_env = _get_bddl_env(env)
    return {
        "objects": sorted(list(getattr(bddl_env, "objects_dict", {}).keys())),
        "fixtures": sorted(list(getattr(bddl_env, "fixtures_dict", {}).keys())),
        "sites": sorted(list(getattr(bddl_env, "object_sites_dict", {}).keys())),
    }


def get_goal_predicates(env) -> List[Tuple[str, int, Tuple[str, ...]]]:
    """Extract goal predicates from parsed_problem and evaluate their current truth.

    Returns list of (pretty_desc, truth_int, args_tuple) where args_tuple contains
    involved names in order (unary: (a,), binary: (a,b)).
    """
    bddl_env = _get_bddl_env(env)
    parsed = getattr(bddl_env, "parsed_problem", {}) or {}
    goal_state = parsed.get("goal_state", []) or []
    object_states = _get_object_states(env)
    try:
        from libero.libero.envs.predicates import eval_predicate_fn
    except Exception:
        eval_predicate_fn = None

    results: List[Tuple[str, int, Tuple[str, ...]]] = []
    for st in goal_state:
        try:
            if len(st) == 3:
                p, a, b = st
                a_state = object_states.get(a)
                b_state = object_states.get(b)
                ok = bool(eval_predicate_fn(p, a_state, b_state)) if (eval_predicate_fn and a_state and b_state) else False
                results.append((f"{p}({a},{b})", 1 if ok else 0, (a, b)))
            elif len(st) == 2:
                p, a = st
                a_state = object_states.get(a)
                ok = bool(eval_predicate_fn(p, a_state)) if (eval_predicate_fn and a_state) else False
                results.append((f"{p}({a})", 1 if ok else 0, (a,)))
        except Exception:
            # Fall back to false
            if len(st) == 3:
                p, a, b = st
                results.append((f"{p}({a},{b})", 0, (a, b)))
            elif len(st) == 2:
                p, a = st
                results.append((f"{p}({a})", 0, (a,)))
    return results


def get_site_parent_name(env, site_name: str) -> Optional[str]:
    """Return the parent object name for a site (region), if available."""
    try:
        bddl_env = _get_bddl_env(env)
        site_objs = getattr(bddl_env, "object_sites_dict", {}) or {}
        site_obj = site_objs.get(site_name, None)
        if site_obj is None:
            return None
        return getattr(site_obj, "parent_name", None)
    except Exception:
        return None


def _get_robot_body_ids(bddl_env) -> Optional[set]:
    """Collect all MuJoCo body IDs that belong to the robot via naming prefix.

    Uses robots[0].robot_model.naming_prefix to match body names. Returns None if unavailable.
    """
    try:
        sim = getattr(bddl_env, "sim", None)
        robots = getattr(bddl_env, "robots", None)
        if sim is None or robots is None or len(robots) == 0:
            return None
        pf = getattr(robots[0].robot_model, "naming_prefix", None)
        if not pf:
            return None
        ids = set()
        nbody = int(sim.model.nbody)
        for i in range(nbody):
            name = sim.model.body_id2name(i) or ""
            if pf in name:
                ids.add(i)
        return ids if ids else None
    except Exception:
        return None


def build_contact_index(env) -> Optional[set]:
    """Build a set of unordered body-id pairs that are in contact (MuJoCo).

    Returns a set of tuples (min(b1,b2), max(b1,b2)) or None if unavailable.
    """
    try:
        bddl_env = _get_bddl_env(env)
        sim = getattr(bddl_env, "sim", None)
        if sim is None:
            return None
        pairs = set()
        ncon = int(getattr(sim.data, "ncon", 0))
        for i in range(ncon):
            c = sim.data.contact[i]
            g1 = int(getattr(c, "geom1", -1))
            g2 = int(getattr(c, "geom2", -1))
            if g1 < 0 or g2 < 0:
                continue
            b1 = int(sim.model.geom_bodyid[g1])
            b2 = int(sim.model.geom_bodyid[g2])
            if b1 == b2:
                continue
            if b1 < b2:
                pairs.add((b1, b2))
            else:
                pairs.add((b2, b1))
        return pairs
    except Exception:
        return None


def contact_obj_with_robot(env, obj_name: str, contact_index: Optional[set] = None) -> Optional[int]:
    """Check contact(obj, robot) via MuJoCo contact table.

    Returns 1/0 if computed, or None if unavailable.
    """
    try:
        bddl_env = _get_bddl_env(env)
        sim = getattr(bddl_env, "sim", None)
        if sim is None:
            return None
        robot_bodies = _get_robot_body_ids(bddl_env)
        if not robot_bodies:
            return None
        # Get object body id
        obj_body_id_map = getattr(bddl_env, "obj_body_id", {})
        if obj_name not in obj_body_id_map:
            return None
        obj_body_id = int(obj_body_id_map[obj_name])
        # Use precomputed index if present, else scan contacts
        if contact_index is not None:
            for rb in robot_bodies:
                pair = (rb, obj_body_id) if rb < obj_body_id else (obj_body_id, rb)
                if pair in contact_index:
                    return 1
            return 0
        else:
            ncon = int(getattr(sim.data, "ncon", 0))
            if ncon <= 0:
                return 0
            for i in range(ncon):
                c = sim.data.contact[i]
                g1 = int(getattr(c, "geom1", -1))
                g2 = int(getattr(c, "geom2", -1))
                if g1 < 0 or g2 < 0:
                    continue
                b1 = int(sim.model.geom_bodyid[g1])
                b2 = int(sim.model.geom_bodyid[g2])
                if (b1 == obj_body_id and b2 in robot_bodies) or (b2 == obj_body_id and b1 in robot_bodies):
                    return 1
            return 0
    except Exception:
        return None


def evaluate_site_methods(env, site_name: str, obj_name: Optional[str] = None) -> Dict[str, int]:
    """Evaluate a subset of SiteObjectState methods on demand.

    Excludes get_joint_state; returns mapping method->0/1 where applicable.
    For unary methods (is_open, is_close), obj_name is ignored.
    """
    object_states = _get_object_states(env)
    site_state = object_states.get(site_name)
    obj_state = object_states.get(obj_name) if obj_name else None
    out: Dict[str, int] = {}
    if site_state is None:
        return out
    # Binary
    if obj_state is not None:
        for m, fn in ("check_contact", getattr(site_state, "check_contact", None)), (
            "check_contain",
            getattr(site_state, "check_contain", None),
        ), ("check_ontop", getattr(site_state, "check_ontop", None)):
            if callable(fn):
                try:
                    out[m] = 1 if bool(fn(obj_state)) else 0
                except Exception:
                    out[m] = 0
    # Unary
    for m, fn in ("is_open", getattr(site_state, "is_open", None)), (
        "is_close",
        getattr(site_state, "is_close", None),
    ):
        if callable(fn):
            try:
                out[m] = 1 if bool(fn()) else 0
            except Exception:
                out[m] = 0
    return out


def evaluate_site_geometry_methods(env, site_name: str, obj_name: str) -> Dict[str, Optional[int]]:
    """Evaluate raw geometry methods on the underlying SiteObject if available.

    Returns a dict possibly containing keys among {"in_box", "under", "on_top"}.
    Values are 1/0, or None if evaluation failed or method not present.
    """
    out: Dict[str, Optional[int]] = {}
    try:
        bddl_env = _get_bddl_env(env)
        sim = getattr(bddl_env, "sim", None)
        if sim is None:
            return out

        site_objs = getattr(bddl_env, "object_sites_dict", {}) or {}
        site_obj = site_objs.get(site_name, None)
        if site_obj is None:
            return out

        # Positions / transforms
        this_pos = sim.data.get_site_xpos(site_name)
        this_mat = sim.data.get_site_xmat(site_name)
        # Object position
        obj_body_id_map = getattr(bddl_env, "obj_body_id", {})
        if obj_name not in obj_body_id_map:
            return out
        obj_pos = sim.data.body_xpos[int(obj_body_id_map[obj_name])]

        # in_box
        if hasattr(site_obj, "in_box"):
            try:
                out["in_box"] = 1 if bool(site_obj.in_box(this_pos, this_mat, obj_pos)) else 0
            except Exception:
                out["in_box"] = 0

        # under
        if hasattr(site_obj, "under"):
            try:
                out["under"] = 1 if bool(site_obj.under(this_pos, this_mat, obj_pos)) else 0
            except Exception:
                out["under"] = 0

        # on_top (TargetZone has on_top)
        if hasattr(site_obj, "on_top"):
            try:
                out["on_top"] = 1 if bool(site_obj.on_top(this_pos, this_mat, obj_pos)) else 0
            except Exception:
                out["on_top"] = 0
    except Exception:
        pass
    return out


def _get_robot_body_ids(bddl_env) -> Optional[set]:
    try:
        sim = getattr(bddl_env, "sim", None)
        robots = getattr(bddl_env, "robots", None)
        if sim is None or robots is None or len(robots) == 0:
            return None
        pf = getattr(robots[0].robot_model, "naming_prefix", None)
        if not pf:
            return None
        ids = set()
        nbody = int(sim.model.nbody)
        for i in range(nbody):
            name = sim.model.body_id2name(i) or ""
            if pf in name:
                ids.add(i)
        return ids if ids else None
    except Exception:
        return None


def build_contact_index(env) -> Optional[set]:
    """Build a set of unordered body-id pairs that are in contact (MuJoCo).

    Returns a set of tuples (min(b1,b2), max(b1,b2)) or None if unavailable.
    """
    try:
        bddl_env = _get_bddl_env(env)
        sim = getattr(bddl_env, "sim", None)
        if sim is None:
            return None
        pairs = set()
        ncon = int(getattr(sim.data, "ncon", 0))
        for i in range(ncon):
            c = sim.data.contact[i]
            g1 = int(getattr(c, "geom1", -1))
            g2 = int(getattr(c, "geom2", -1))
            if g1 < 0 or g2 < 0:
                continue
            b1 = int(sim.model.geom_bodyid[g1])
            b2 = int(sim.model.geom_bodyid[g2])
            if b1 == b2:
                continue
            if b1 < b2:
                pairs.add((b1, b2))
            else:
                pairs.add((b2, b1))
        return pairs
    except Exception:
        return None


def contact_obj_with_robot(env, obj_name: str, contact_index: Optional[set] = None) -> Optional[int]:
    """Check contact(obj, robot) via MuJoCo contact table.

    Returns 1/0 if computed, or None if unavailable.
    """
    try:
        bddl_env = _get_bddl_env(env)
        sim = getattr(bddl_env, "sim", None)
        if sim is None:
            return None
        robot_bodies = _get_robot_body_ids(bddl_env)
        if not robot_bodies:
            return None
        # Get object body id
        obj_body_id_map = getattr(bddl_env, "obj_body_id", {})
        if obj_name not in obj_body_id_map:
            return None
        obj_body_id = int(obj_body_id_map[obj_name])
        # Use precomputed index if present, else scan contacts
        if contact_index is not None:
            for rb in robot_bodies:
                pair = (rb, obj_body_id) if rb < obj_body_id else (obj_body_id, rb)
                if pair in contact_index:
                    return 1
            return 0
        else:
            ncon = int(getattr(sim.data, "ncon", 0))
            if ncon <= 0:
                return 0
            for i in range(ncon):
                c = sim.data.contact[i]
                g1 = int(getattr(c, "geom1", -1))
                g2 = int(getattr(c, "geom2", -1))
                if g1 < 0 or g2 < 0:
                    continue
                b1 = int(sim.model.geom_bodyid[g1])
                b2 = int(sim.model.geom_bodyid[g2])
                if (b1 == obj_body_id and b2 in robot_bodies) or (b2 == obj_body_id and b1 in robot_bodies):
                    return 1
            return 0
    except Exception:
        return None


def contact_between_bodies(env, name_a: str, name_b: str, contact_index: Optional[set] = None) -> Optional[int]:
    """Check MuJoCo contact between two non-site entities (objects/fixtures).

    Returns 1/0 or None if unavailable or names not mapped to bodies.
    """
    try:
        bddl_env = _get_bddl_env(env)
        sim = getattr(bddl_env, "sim", None)
        if sim is None:
            return None
        body_map = getattr(bddl_env, "obj_body_id", {})
        if name_a not in body_map or name_b not in body_map:
            return None
        ba = int(body_map[name_a])
        bb = int(body_map[name_b])
        if contact_index is not None:
            pair = (ba, bb) if ba < bb else (bb, ba)
            return 1 if pair in contact_index else 0
        else:
            ncon = int(getattr(sim.data, "ncon", 0))
            if ncon <= 0:
                return 0
            for i in range(ncon):
                c = sim.data.contact[i]
                g1 = int(getattr(c, "geom1", -1))
                g2 = int(getattr(c, "geom2", -1))
                if g1 < 0 or g2 < 0:
                    continue
                b1 = int(sim.model.geom_bodyid[g1])
                b2 = int(sim.model.geom_bodyid[g2])
                if (b1 == ba and b2 == bb) or (b1 == bb and b2 == ba):
                    return 1
            return 0
    except Exception:
        return None


def _tokenize_name(name: str) -> List[str]:
    base = re.sub(r"\d+", " ", name)
    toks = re.split(r"[_\-\s]+", base)
    toks = [t for t in toks if t]
    return toks


def expand_overlap_objects(all_objects: List[str], involved_objects: List[str]) -> List[str]:
    """Expand object set with those sharing any token with involved objects."""
    inv_tokens = set()
    for n in involved_objects:
        inv_tokens.update(_tokenize_name(n))
    out = []
    for n in all_objects:
        toks = set(_tokenize_name(n))
        if inv_tokens & toks:
            out.append(n)
    return sorted(list(set(out)))


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
