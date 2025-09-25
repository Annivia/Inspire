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
from typing import Dict, List, Tuple, Iterable, Optional, Any
import os as _os

# Debug toggle for contact mapping. Enable by setting VCE_CONTACT_DEBUG=1
_CONTACT_DEBUG = str(_os.environ.get("VCE_CONTACT_DEBUG", "0")).strip() in ("1", "true", "True")

def _cdbg(msg: str) -> None:
    if _CONTACT_DEBUG:
        try:
            print(f"[VCE:contact] {msg}")
        except Exception:
            pass


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
    """Return the parent object/fixture name for a site (region), if available."""
    try:
        bddl_env = _get_bddl_env(env)
        site_objs = getattr(bddl_env, "object_sites_dict", {}) or {}
        site_obj = site_objs.get(site_name, None)
        if site_obj is None:
            return None
        parent = getattr(site_obj, "parent_name", None)
        return parent if isinstance(parent, str) else None
    except Exception:
        return None

def get_site_parent_map(env) -> Dict[str, str]:
    """Return mapping from site (region) name to its parent object/fixture name when available."""
    mapping: Dict[str, str] = {}
    try:
        bddl_env = _get_bddl_env(env)
        site_objs = getattr(bddl_env, "object_sites_dict", {}) or {}
        for site_name, site_obj in site_objs.items():
            parent = getattr(site_obj, "parent_name", None)
            if isinstance(parent, str) and parent:
                mapping[site_name] = parent
    except Exception:
        pass
    return mapping


def _get_robot_body_ids(bddl_env) -> Optional[set]:
    """Collect MuJoCo body IDs for the robot, including gripper bodies.

    Tries robot and gripper naming prefixes; falls back to heuristic matches for
    common gripper terms. Returns None if unavailable.
    """
    try:
        sim = getattr(bddl_env, "sim", None)
        robots = getattr(bddl_env, "robots", None)
        if sim is None or robots is None or len(robots) == 0:
            return None
        robot_pf = getattr(robots[0].robot_model, "naming_prefix", None)
        # Try to find an explicit gripper prefix if present
        gripper_pf = None
        try:
            gr = getattr(robots[0], "gripper", None)
            if gr is not None:
                gripper_pf = getattr(gr, "naming_prefix", None)
        except Exception:
            gripper_pf = None

        ids = set()
        nbody = int(sim.model.nbody)
        for i in range(nbody):
            name = sim.model.body_id2name(i) or ""
            add = False
            if robot_pf and robot_pf in name:
                add = True
            if gripper_pf and gripper_pf in name:
                add = True
            # Heuristic match for typical gripper sub-bodies
            if (not add) and ("gripper" in name or "finger" in name or name.endswith("_eef") or "eef" in name):
                add = True
            if add:
                ids.add(i)
        _cdbg(f"robot bodies collected: {len(ids)} (robot_pf={robot_pf}, gripper_pf={gripper_pf})")
        return ids if ids else None
    except Exception:
        return None

def _collect_body_subtree(sim, roots: Iterable[int]) -> set:
    """Return set of body ids in the subtree rooted at the given body ids (inclusive)."""
    try:
        roots = {int(r) for r in roots if r is not None and r >= 0}
        if not roots:
            return set()
        parent = getattr(sim.model, "body_parentid", None)
        nbody = int(sim.model.nbody)
        if parent is None:
            # Fallback: if parent array missing, just return roots
            return set(roots)
        children = {i: [] for i in range(nbody)}
        for i in range(nbody):
            p = int(parent[i])
            if p >= 0:
                children[p].append(i)
        out = set()
        stack = list(roots)
        while stack:
            b = stack.pop()
            if b in out:
                continue
            out.add(b)
            stack.extend(children.get(b, []))
        return out
    except Exception:
        return set()

def _resolve_name_to_body_roots(bddl_env, name: str) -> List[int]:
    """Resolve a scene entity name to one or more root body ids.

    Tries env.obj_body_id, then direct body name lookup; for site names, returns
    the parent body id. Returns an empty list if nothing found.
    """
    try:
        sim = getattr(bddl_env, "sim", None)
        if sim is None:
            return []
        # 1) Direct object/fixture mapping
        body_map = getattr(bddl_env, "obj_body_id", {}) or {}
        if name in body_map:
            bid = int(body_map[name])
            return [bid]
        # 2) Site → parent body id
        try:
            sid = int(sim.model.site_name2id(name))
            bid = int(sim.model.site_bodyid[sid])
            return [bid]
        except Exception:
            pass
        # 3) Raw body name
        try:
            bid = int(sim.model.body_name2id(name))
            return [bid]
        except Exception:
            pass
    except Exception:
        pass
    return []

def _resolve_name_to_body_set(bddl_env, name: str) -> set:
    """Resolve a scene entity name to a set of body ids (including subtree)."""
    try:
        sim = getattr(bddl_env, "sim", None)
        if sim is None:
            return set()
        roots = _resolve_name_to_body_roots(bddl_env, name)
        if not roots:
            _cdbg(f"name→body: '{name}' not resolved")
            return set()
        ids = _collect_body_subtree(sim, roots)
        _cdbg(f"name→body: '{name}' roots={roots} subtree_size={len(ids)}")
        return ids
    except Exception:
        return set()


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
        # Get object body id set (include subtree)
        obj_bodies = _resolve_name_to_body_set(bddl_env, obj_name)
        if not obj_bodies:
            return None
        _cdbg(f"check robot contact: obj='{obj_name}' obj_bodies={len(obj_bodies)} robot_bodies={len(robot_bodies)}")
        # Use precomputed index if present, else scan contacts
        if contact_index is not None:
            for rb in robot_bodies:
                for ob in obj_bodies:
                    pair = (rb, ob) if rb < ob else (ob, rb)
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
                if (b1 in obj_bodies and b2 in robot_bodies) or (b2 in obj_bodies and b1 in robot_bodies):
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
    """Return MuJoCo body ids that belong to the robot, including gripper bodies.

    Uses the robot model naming_prefix (e.g., 'robot0_') and also includes any
    bodies whose name contains 'gripper' (e.g., 'gripper0_leftfinger').
    """
    try:
        sim = getattr(bddl_env, "sim", None)
        robots = getattr(bddl_env, "robots", None)
        if sim is None or robots is None or len(robots) == 0:
            return None
        pf = getattr(robots[0].robot_model, "naming_prefix", None) or ""
        ids = set()
        nbody = int(sim.model.nbody)
        for i in range(nbody):
            name = (sim.model.body_id2name(i) or "").lower()
            if (pf and pf.lower() in name) or ("gripper" in name):
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
        set_a = _resolve_name_to_body_set(bddl_env, name_a)
        set_b = _resolve_name_to_body_set(bddl_env, name_b)
        if not set_a or not set_b:
            _cdbg(f"contact_between_bodies: mapping failed a='{name_a}'|{len(set_a)} b='{name_b}'|{len(set_b)}")
            return None
        _cdbg(f"contact_between_bodies: a='{name_a}' sizeA={len(set_a)} b='{name_b}' sizeB={len(set_b)}")
        if contact_index is not None:
            for ba in set_a:
                for bb in set_b:
                    pair = (ba, bb) if ba < bb else (bb, ba)
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
                if (b1 in set_a and b2 in set_b) or (b1 in set_b and b2 in set_a):
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


def evaluate_concept_expressions(env, expressions: Iterable[str], contact_index: Optional[set] = None) -> Dict[str, int]:
    """Evaluate heterogeneous concept expressions on the current simulator state.

    Supports heads among: contact, mj_contact, ontop, contain, in_box, under,
    on_top, is_open, is_close, in, on, region_contains. Unknown heads → 0.

    Notes:
    - contact(obj,gripper): uses MuJoCo robot contact
    - contact(obj,site): uses SiteObjectState.check_contact when available; else MuJoCo body contact
    - ontop/contain: use SiteObjectState methods
    - in_box/under/on_top: use raw site geometry on SiteObject
    - is_open/is_close: unary site methods
    - mj_contact(a,b): strict MuJoCo contact; supports gripper
    - in/on: delegates to predicate evaluator via evaluate_concepts
    """
    out: Dict[str, int] = {}
    object_states = _get_object_states(env)
    non_site, site = _split_site_vs_non_site(object_states)

    def _is_site(name: str) -> bool:
        return name in site

    for expr in expressions:
        try:
            base = expr.split(" ", 1)[0]
            if not ("(" in base and base.endswith(")")):
                out[expr] = 0
                continue
            head, rest = base.split("(", 1)
            args = [a.strip() for a in rest[:-1].split(",") if a.strip()]
            head_l = head.lower()

            if head_l == "contact":
                if len(args) != 2:
                    out[expr] = 0
                    continue
                a, b = args
                if b == "gripper":
                    r = contact_obj_with_robot(env, a, contact_index)
                    out[expr] = 1 if r == 1 else 0
                    continue
                if _is_site(b):
                    sm = evaluate_site_methods(env, b, a)
                    out[expr] = 1 if sm.get("check_contact") else 0
                else:
                    r = contact_between_bodies(env, a, b, contact_index)
                    out[expr] = int(r) if (r is not None) else 0
                continue

            if head_l == "ontop":
                if len(args) == 2:
                    a, s = args
                    sm = evaluate_site_methods(env, s, a)
                    out[expr] = 1 if sm.get("check_ontop") else 0
                else:
                    out[expr] = 0
                continue

            if head_l == "contain":
                if len(args) == 2:
                    a, s = args
                    sm = evaluate_site_methods(env, s, a)
                    out[expr] = 1 if sm.get("check_contain") else 0
                else:
                    out[expr] = 0
                continue

            if head_l in ("in_box", "under", "on_top"):
                if len(args) == 2:
                    a, s = args
                    gm = evaluate_site_geometry_methods(env, s, a)
                    out[expr] = 1 if gm.get(head_l) else 0
                else:
                    out[expr] = 0
                continue

            if head_l == "is_open":
                if len(args) == 1:
                    s = args[0]
                    um = evaluate_site_methods(env, s, None)
                    out[expr] = 1 if um.get("is_open") else 0
                else:
                    out[expr] = 0
                continue

            if head_l == "is_close":
                if len(args) == 1:
                    s = args[0]
                    um = evaluate_site_methods(env, s, None)
                    out[expr] = 1 if um.get("is_close") else 0
                else:
                    out[expr] = 0
                continue

            if head_l == "mj_contact":
                if len(args) != 2:
                    out[expr] = 0
                    continue
                a, b = args
                if a == "gripper" and b != "gripper":
                    r = contact_obj_with_robot(env, b, contact_index)
                    out[expr] = 1 if r == 1 else 0
                    continue
                if b == "gripper" and a != "gripper":
                    r = contact_obj_with_robot(env, a, contact_index)
                    out[expr] = 1 if r == 1 else 0
                    continue
                r = contact_between_bodies(env, a, b, contact_index)
                out[expr] = int(r) if (r is not None) else 0
                continue

            if head_l in ("in", "on", "region_contains"):
                out[expr] = int(evaluate_concepts(env, [expr]).get(expr, 0))
                continue

            out[expr] = 0
        except Exception:
            out[expr] = 0
    return out


def select_task_concepts(env) -> List[str]:
    """Select a per-task concept list for logging/probing.

    Policy:
    - Prefer curated checks from collect_scene_predicates(env)
    - Augment with mj_contact(obj,gripper) for involved/relevant objects
    - Augment with mj_contact(obj,parent_of_init_site_for_obj) when resolvable
    - If empty, fall back to enumerate_concept_keys(env)
    """
    concepts: List[str] = []
    try:
        scene_pred = collect_scene_predicates(env) or {}
        checks = scene_pred.get("checks") or []
        concepts = sorted({c.get("expr") for c in checks if isinstance(c, dict) and c.get("expr")})
        try:
            inv = get_env_inventory(env)
            objects, sites = inv.get("objects", []), inv.get("sites", [])
            parent_map = get_site_parent_map(env)
            goals = get_goal_predicates(env)
            involved_objs, _ = derive_involved_from_goals(goals, objects, sites)
            relevant_objs = expand_overlap_objects(objects, involved_objs) if involved_objs else []
            objs_interest = sorted(list({*involved_objs, *relevant_objs})) or involved_objs
            extra = []
            for obj in objs_interest:
                extra.append(f"mj_contact({obj},gripper)")
                base = "_".join(obj.split("_")[:-1]) or obj
                for s in sites:
                    sl = s.lower()
                    if "init" in sl and base in sl:
                        parent = parent_map.get(s)
                        if parent:
                            extra.append(f"mj_contact({obj},{parent})")
            if extra:
                concepts = sorted(list({*concepts, *extra}))
        except Exception:
            pass
    except Exception:
        concepts = []
    if not concepts:
        concepts = enumerate_concept_keys(env)
    return concepts


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


def derive_involved_from_goals(goals: List[Tuple[str, int, Tuple[str, ...]]],
                               objects: List[str],
                               sites: List[str]) -> Tuple[List[str], List[str]]:
    """Derive involved objects and sites from goal predicate arguments.

    Returns (involved_objects, involved_sites) sorted and deduplicated.
    """
    involved_objects: List[str] = []
    involved_sites: List[str] = []
    for _, _, args in goals:
        if len(args) == 1:
            a = args[0]
            if a in objects:
                involved_objects.append(a)
            if a in sites:
                involved_sites.append(a)
        elif len(args) == 2:
            a, b = args
            if a in objects:
                involved_objects.append(a)
            if a in sites:
                involved_sites.append(a)
            if b in objects:
                involved_objects.append(b)
            if b in sites:
                involved_sites.append(b)
    involved_objects = sorted(list({n for n in involved_objects if n in objects}))
    involved_sites = sorted(list({n for n in involved_sites if n in sites}))
    return involved_objects, involved_sites


def collect_scene_predicates(env) -> Dict[str, Any]:
    """Collect a structured snapshot of predicates for the current scene.

    Centralized policy used by data collection:
    - Goal predicates (from BDDL) first
    - Involved objects / regions parsed from goals
    - Required checks per involved obj / site:
        * SiteObjectState: check_ontop, check_contact
        * Site geometry: in_box, under (only if language contains 'under'), on_top
        * Robot contact: mj_contact(obj, robot)
    - Site-wide checks for all non-'init' sites against overlap objects
    - Strict MuJoCo contacts:
        * mj_contact(obj_i, obj_j) for pairs of involved objects
        * mj_contact(obj, parent(site)) only (no site NA lines; parents only)

    Returns a dict with fields: language, objects, sites, fixtures, predicates,
    goals (list of {expr, value}), involved_objects, involved_sites, checks (list of {expr, value}).
    """
    inv = get_env_inventory(env)
    objects, sites, fixtures = inv["objects"], inv["sites"], inv["fixtures"]
    task_name, language = _get_task_identifiers(env)
    include_under = should_include_under(env)
    goals = get_goal_predicates(env)
    involved_objects, involved_sites = derive_involved_from_goals(goals, objects, sites)
    overlap_objs = expand_overlap_objects(objects, involved_objects) if involved_objects else []
    parent_map = get_site_parent_map(env)
    contact_index = build_contact_index(env)

    # Predicate names used in goals
    pred_names = sorted(list({g[0].split("(", 1)[0] for g in goals})) if goals else []

    out: Dict[str, Any] = {
        "language": language,
        "objects": objects,
        "sites": sites,
        "fixtures": fixtures,
        "predicates": pred_names,
        "goals": [{"expr": g[0], "value": g[1]} for g in goals],
        "involved_objects": involved_objects,
        "involved_sites": involved_sites,
        "checks": [],
    }

    def add(expr: str, val):
        if val is None:
            return
        out["checks"].append({"expr": expr, "value": int(val)})

    # Robot contacts for involved objects
    for obj in involved_objects:
        add(f"contact({obj},gripper)", contact_obj_with_robot(env, obj, contact_index))

    # Per involved site/object pair
    for site in involved_sites:
        for obj in involved_objects:
            sm = evaluate_site_methods(env, site, obj)
            if "check_ontop" in sm:
                add(f"ontop({obj},{site})", sm["check_ontop"])
            if "check_contact" in sm:
                add(f"contact({obj},{site})", sm["check_contact"])
            geom = evaluate_site_geometry_methods(env, site, obj)
            if "in_box" in geom:
                add(f"in_box({obj},{site})", geom["in_box"])
            if include_under and ("under" in geom):
                add(f"under({obj},{site})", geom["under"])
            if "on_top" in geom:
                add(f"on_top({obj},{site})", geom["on_top"])

    # All non-init sites vs overlap objects
    for site in [s for s in sites if "init" not in s.lower()]:
        # Unary
        um = evaluate_site_methods(env, site, None)
        if "is_open" in um:
            add(f"is_open({site})", um["is_open"])
        if "is_close" in um:
            add(f"is_close({site})", um["is_close"])
        # Binary
        for obj in overlap_objs:
            bm = evaluate_site_methods(env, site, obj)
            if "check_contact" in bm:
                add(f"contact({obj},{site})", bm["check_contact"])
            if "check_contain" in bm:
                add(f"contain({obj},{site})", bm["check_contain"])
            if "check_ontop" in bm:
                add(f"ontop({obj},{site})", bm["check_ontop"])
            geom = evaluate_site_geometry_methods(env, site, obj)
            if "in_box" in geom:
                add(f"in_box({obj},{site})", geom["in_box"])
            if include_under and ("under" in geom):
                add(f"under({obj},{site})", geom["under"])
            if "on_top" in geom:
                add(f"on_top({obj},{site})", geom["on_top"])

    # Strict MuJoCo contacts
    # Involved object pairs
    for i in range(len(involved_objects)):
        for j in range(i + 1, len(involved_objects)):
            a, b = involved_objects[i], involved_objects[j]
            add(f"mj_contact({a},{b})", contact_between_bodies(env, a, b, contact_index))
    # Involved object vs parent(site) only
    printed_parent = set()
    for site in involved_sites:
        parent = parent_map.get(site)
        if not parent or parent not in (fixtures + objects):
            continue
        for obj in involved_objects:
            key = (obj, parent)
            if key in printed_parent:
                continue
            add(f"mj_contact({obj},{parent}) [parent_of={site}]", contact_between_bodies(env, obj, parent, contact_index))
            printed_parent.add(key)

    return out


def build_concept_hash(
    env,
    concepts: Optional[List[str]] = None,
    source: str = "relations",
    *,
    task_id: Optional[str] = None,
    scene_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a filterable concept hash table for the current scene.

    Produces a dictionary mapping concept names → attributes and fast-filter indexes.

    Args:
        env: LIBERO environment (ControlEnv or underlying BDDL env).
        concepts: Optional explicit list of concept strings to hash. If None,
            `source` controls which set is generated.
        source: One of:
            - "relations": use `enumerate_concept_keys(env)` (CSV column names)
            - "checks": use expressions collected by `collect_scene_predicates(env)`

    Returns:
        Dict with keys:
            - meta: task + scene inventory and involvement sets
            - concepts: mapping str → per-concept attributes
              Each concept has a "fields" array; for each argument/entity:
                {"name": <entity>, "kind": "object"|"site"|"robot", "parent": <parent_or_None>, "involved": <bool>}
            - index: reverse indexes for fast filtering (by entity/kind/relation/involved)
    """

    # Scene inventory and helpers
    object_states = _get_object_states(env)
    non_site, site = _split_site_vs_non_site(object_states)
    inv = get_env_inventory(env)
    objects, sites = inv["objects"], inv["sites"]
    parent_map = get_site_parent_map(env)
    task_name, language = _get_task_identifiers(env)

    # Involvement sets derived from BDDL goals
    goals = get_goal_predicates(env)
    involved_objects, involved_sites = derive_involved_from_goals(goals, objects, sites)

    # Relevance heuristics (category/variant level), disjoint from involved
    COLOR_TOKENS = {
        "white","black","red","blue","green","yellow","brown","pink","gray","grey","purple","orange",
        "silver","gold","cyan","magenta","beige","tan",
    }
    LATERALITY_TOKENS = {"left","right","middle","center","centre"}

    def _is_num(tok: str) -> bool:
        try:
            int(tok)
            return True
        except Exception:
            return False

    def _obj_signature(name: str) -> Tuple[str, ...]:
        toks = _tokenize_name(name)
        filt = [t for t in toks if t not in COLOR_TOKENS and t not in LATERALITY_TOKENS and not _is_num(t)]
        # keep order to preserve multiword categories like "paper_towel"
        return tuple(filt)

    def _site_signature(name: str) -> Tuple[str, ...]:
        toks = _tokenize_name(name)
        filt = [t for t in toks if t not in LATERALITY_TOKENS and not _is_num(t)]
        return tuple(filt)

    involved_obj_sigs = { _obj_signature(o) for o in involved_objects }
    # Objects relevant to any involved object: share the same object signature, but are different instances
    relevant_objects = sorted(
        [o for o in objects
         if o not in involved_objects and _obj_signature(o) in involved_obj_sigs and o != "gripper"]
    ) if involved_objects else []

    # Sites relevant to involved sites: siblings with same parent and same role signature (ignoring laterality / numbers)
    relevant_sites = []
    if involved_sites:
        for s in sites:
            if s in involved_sites:
                continue
            p = parent_map.get(s)
            sig = _site_signature(s)
            for isite in involved_sites:
                if parent_map.get(isite) == p and _site_signature(isite) == sig:
                    relevant_sites.append(s)
                    break
        relevant_sites = sorted(list(set(relevant_sites)))
    else:
        relevant_sites = []

    # Choose concept list
    concept_list: List[str]
    if concepts is not None:
        concept_list = list(concepts)
    elif source == "checks":
        snap = collect_scene_predicates(env)
        concept_list = [c["expr"] for c in (snap.get("checks") or []) if isinstance(c, dict) and "expr" in c]
    else:  # default to relations (CSV-compatible)
        concept_list = enumerate_concept_keys(env)

    # Helper: parse concept string
    def _parse(concept: str) -> Tuple[str, List[str]]:
        # strip any annotation suffix like " ... [parent_of=site]"
        bare = concept.split(" ", 1)[0]
        if "(" in bare and bare.endswith(")"):
            head, rest = bare.split("(", 1)
            args = rest[:-1]
            arg_list = [a.strip() for a in args.split(",") if a.strip()]
            return head.strip(), arg_list
        return bare, []

    # Storage
    concept_meta: Dict[str, Dict[str, Any]] = {}

    # Indexes
    idx_by_object: Dict[str, List[str]] = {}
    idx_by_site: Dict[str, List[str]] = {}
    idx_by_parent: Dict[str, List[str]] = {}
    idx_by_relation: Dict[str, List[str]] = {}
    idx_by_flag: Dict[str, List[str]] = {}
    idx_by_relation_type: Dict[str, List[str]] = {}
    idx_by_involved_object: Dict[str, List[str]] = {}
    idx_by_involved_site: Dict[str, List[str]] = {}
    
    def _add(index: Dict[str, List[str]], key: Optional[str], concept_name: str):
        if not key:
            return
        index.setdefault(key, []).append(concept_name)

    def _add_flag(flag_name: str, cond: bool, concept_name: str):
        if cond:
            idx_by_flag.setdefault(flag_name, []).append(concept_name)

    for key in concept_list:
        head, args = _parse(key)
        head_l = head.lower()

        # Determine involved names
        arg_set = set(args)
        concepts_objects = [a for a in args if (a in non_site) or (a == "gripper")]
        concepts_sites = [a for a in args if a in site]
        parents = sorted(list({parent_map.get(s) for s in concepts_sites if parent_map.get(s)}))

        # Flags per requested categories
        is_check_contact = (head_l == "contact")
        is_mujoco_contact = (head_l == "mj_contact") or (head_l == "contact" and ("gripper" in arg_set))
        is_on_top = (head_l in ("on", "ontop", "on_top"))
        is_under = (head_l == "under")
        is_in = (head_l == "in")
        is_region_contains = (head_l == "region_contains")
        is_is_open = (head_l == "is_open")
        is_is_close = (head_l == "is_close")

        # Relation type labeling
        if is_is_open or is_is_close:
            relation_type = "site-attribute"
        elif "gripper" in arg_set:
            relation_type = "object-robot"
        elif concepts_sites and concepts_objects:
            relation_type = "object-region"
        elif len(concepts_objects) >= 2:
            relation_type = "object-object"
        elif concepts_sites and not concepts_objects:
            relation_type = "site-site"
        else:
            relation_type = "other"

        # Task involvement labels (entity-level)
        involved_obj_flags = {o: (o in involved_objects) for o in concepts_objects if o != "gripper"}
        involved_site_flags = {s: (s in involved_sites) for s in concepts_sites}


        # Assemble metadata
        fields = []
        for a in args:
            if a == "gripper":
                fields.append({"name": a, "kind": "robot", "parent": None, "involved": False})
            elif a in site:
                fields.append({"name": a, "kind": "site", "parent": parent_map.get(a), "involved": bool(involved_site_flags.get(a, False))})
            else:
                fields.append({"name": a, "kind": "object", "parent": None, "involved": bool(involved_obj_flags.get(a, False))})
        meta = {
            "name": key,
            "relation": head_l,
            "objects": concepts_objects,
            "sites": concepts_sites,
            "site_parents": {s: parent_map.get(s) for s in concepts_sites if s in parent_map},
            "parents": parents,
            "fields": fields,
            "flags": {
                "is_check_contact": bool(is_check_contact),
                "is_mujoco_contact": bool(is_mujoco_contact),
                "is_on_top": bool(is_on_top),
                "is_under": bool(is_under),
                "is_in": bool(is_in),
                "is_region_contains": bool(is_region_contains),
                "is_open": bool(is_is_open),
                "is_close": bool(is_is_close),
            },
            "relation_type": relation_type,
            "involved_objects": involved_obj_flags,  # per-object boolean
            "involved_sites": involved_site_flags,   # per-site boolean
        }
        concept_meta[key] = meta

        # Populate indexes
        _add(idx_by_relation, head_l, key)
        for o in concepts_objects:
            _add(idx_by_object, o, key)
            if involved_obj_flags.get(o):
                _add(idx_by_involved_object, o, key)
        for s in concepts_sites:
            _add(idx_by_site, s, key)
            if involved_site_flags.get(s):
                _add(idx_by_involved_site, s, key)
        for p in parents:
            _add(idx_by_parent, p, key)
        _add_flag("is_check_contact", is_check_contact, key)
        _add_flag("is_mujoco_contact", is_mujoco_contact, key)
        _add_flag("is_on_top", is_on_top, key)
        _add_flag("is_under", is_under, key)
        _add_flag("is_in", is_in, key)
        _add_flag("is_region_contains", is_region_contains, key)
        _add_flag("is_open", is_is_open, key)
        _add_flag("is_close", is_is_close, key)
        _add(idx_by_relation_type, relation_type, key)

    # Derive scene name if not explicitly provided
    if scene_name is None:
        # Prefer ControlEnv/BDDL problem_name if it already encodes scene
        candidate = task_id or task_name
        try:
            import re as _re  # local import to avoid polluting module globals
            m = _re.match(r"^(.*?_SCENE\d+)_", str(candidate))
            scene_name = m.group(1) if m else ""
        except Exception:
            scene_name = ""

    result: Dict[str, Any] = {
        "meta": {
            "task_name": task_name,
            "task_id": task_id or task_name,
            "scene_name": scene_name or "",
            "language_instruction": language,
            "objects": objects,
            "sites": sites,
            "involved_objects": involved_objects,
            "involved_sites": involved_sites,
            "source": source,
        },
        "concepts": concept_meta,
        "index": {
            "by_object": idx_by_object,
            "by_site": idx_by_site,
            "by_parent": idx_by_parent,
            "by_relation": idx_by_relation,
            "by_flag": idx_by_flag,
            "by_relation_type": idx_by_relation_type,
            "by_involved_object": idx_by_involved_object,
            "by_involved_site": idx_by_involved_site,
        },
    }
    return result
