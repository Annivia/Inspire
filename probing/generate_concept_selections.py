#!/usr/bin/env python3
"""
Generate per-task concept selections for experiments by scanning saved concept
hash tables under test/hash and writing experiment_1.txt and experiment_2.txt.

This reads only names/metadata — no simulator calls. It uses lightweight
heuristics on language and entity names to decide which concepts to print for
each task category described by the user.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_DIR = REPO_ROOT / "test" / "hash"
OUT_1 = REPO_ROOT / "experiment_1.txt"   # legacy
OUT_2 = REPO_ROOT / "experiment_2.txt"   # legacy
OUT_GENERAL_1 = REPO_ROOT / "experiment_general_task_1.txt"
OUT_GENERAL_2 = REPO_ROOT / "experiment_general_task_2.txt"
OUT_SPATIAL_1 = REPO_ROOT / "experiment_spatial_task_1.txt"
OUT_SPATIAL_2 = REPO_ROOT / "experiment_spatial_task_2.txt"
OUT_SPATIAL_3 = REPO_ROOT / "experiment_spatial_task_3.txt"


def load_hashes() -> List[Tuple[str, Dict, Dict]]:
    """Load paired (relations, checks) hashes for each task base name."""
    rels = {p.stem.replace("__relations_hash", ""): p for p in HASH_DIR.glob("*__relations_hash.json")}
    chks = {p.stem.replace("__checks_hash", ""): p for p in HASH_DIR.glob("*__checks_hash.json")}
    bases = sorted(set(rels.keys()) | set(chks.keys()))
    out = []
    for b in bases:
        rel = {}
        chk = {}
        if b in rels:
            try:
                rel = json.loads(rels[b].read_text())
            except Exception:
                rel = {}
        if b in chks:
            try:
                chk = json.loads(chks[b].read_text())
            except Exception:
                chk = {}
        out.append((b, rel, chk))
    return out


def tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[\W_]+", " ", s)
    return [t for t in s.split() if t]


def first_or_none(xs: List[str]) -> str:
    return xs[0] if xs else ""


def pick_target_object(meta: Dict) -> str:
    objs = list(meta.get("involved_objects") or [])
    objs = [o for o in objs if o != "gripper"]
    return first_or_none(objs)


def pick_primary_site(meta: Dict) -> str:
    sites = list(meta.get("involved_sites") or [])
    return first_or_none(sites)


def has_state_change(lang: str) -> bool:
    L = lang.lower()
    return any(k in L for k in [
        "open", "close", "shut", "turn on", "turn off", "power on", "power off",
    ]) and any(k in L for k in ["drawer", "cabinet", "microwave", "oven", "stove"])


def is_pick_and_place(lang: str) -> bool:
    L = lang.lower()
    return any(k in L for k in ["put", "place", "set", "stack"]) and not has_state_change(L)


def is_specific_drawer(lang: str) -> bool:
    L = lang.lower()
    return ("drawer" in L) and any(k in L for k in ["top", "bottom", "middle"])


def has_identical_targets(meta: Dict, lang: str) -> bool:
    L = lang.lower()
    return (any(k in L for k in ["left", "right"])) and (any(k in L for k in ["plate", "bowl", "mug", "cup"]))


def is_specific_region(lang: str) -> bool:
    L = lang.lower()
    return any(k in L for k in ["front", "back", "left", "right", "middle"]) and any(
        k in L for k in ["compartment", "section", "slot", "caddy", "bin", "tray", "compartments"]
    )


def find_side_sites(all_sites: List[str], base_keyword: str) -> Tuple[str, str]:
    left = ""
    right = ""
    for s in all_sites:
        sl = s.lower()
        if base_keyword in sl and "left" in sl:
            left = s
        if base_keyword in sl and "right" in sl:
            right = s
    return left, right


def format_block(task_title: str, concepts: List[str]) -> str:
    out = [task_title]
    for c in concepts:
        out.append(f"  - {c}")
    return "\n".join(out)


def select_and_dump(task_key: str, rel: Dict, chk: Dict) -> str:
    meta = (rel.get("meta") if rel else None) or (chk.get("meta") if chk else None) or {}
    lang = str(meta.get("language_instruction") or task_key)
    involved_objs = meta.get("involved_objects") or []
    involved_sites = meta.get("involved_sites") or []
    sites_all = meta.get("sites") or []
    rel_names = set((rel.get("concepts") or {}).keys())
    chk_names = set((chk.get("concepts") or {}).keys())
    all_names = rel_names | chk_names

    def src_of(name: str) -> str:
        if name in rel_names and name in chk_names:
            return "[rel|chk]"
        if name in rel_names:
            return "[rel]"
        if name in chk_names:
            return "[chk]"
        return ""

    tgt = first_or_none([o for o in involved_objs if o != "gripper"]) or first_or_none(involved_objs) or ""
    reg = first_or_none(involved_sites) or ""

    blocks: List[str] = []
    header = f"Task: {lang}\nSources: "
    srcs = []
    if rel:
        srcs.append(f"{task_key}__relations_hash.json")
    if chk:
        srcs.append(f"{task_key}__checks_hash.json")
    blocks.append(header + ", ".join(srcs))

    # Category: General Task 2 (State Change)
    if has_state_change(lang):
        drawer = reg or first_or_none([s for s in sites_all if any(k in s.lower() for k in ("drawer","microwave","stove","oven"))])
        cat = ["General Task 2 (State Change)"]
        for name in [f"is_open({drawer})", f"is_close({drawer})"]:
            if drawer and name in all_names:
                cat.append(f"  - {name} {src_of(name)}")
        if len(cat) > 1:
            blocks.append("\n".join(cat))

    # Category: General Task 1 (Pick & Place, single target)
    is_pnp = is_pick_and_place(lang) and len([o for o in involved_objs if o != "gripper"]) == 1
    if is_pnp and tgt:
        cat = ["General Task 1 (Pick & Place)"]
        cands = [
            f"contact({tgt},gripper)",      # from checks
            f"on({tgt},{reg})",             # from relations
            f"ontop({tgt},{reg})",          # from checks
            f"contact({tgt},{reg})",        # from checks
        ]
        for name in cands:
            if tgt and reg and name in all_names:
                cat.append(f"  - {name} {src_of(name)}")
            elif name.endswith(",gripper)") and name in all_names:
                cat.append(f"  - {name} {src_of(name)}")
        if len(cat) > 1:
            blocks.append("\n".join(cat))

    # Category: Spatial Task 1 (Specific Drawer)
    if is_specific_drawer(lang):
        top = first_or_none([s for s in sites_all if ("drawer" in s.lower() and "top" in s.lower())])
        bottom = first_or_none([s for s in sites_all if ("drawer" in s.lower() and "bottom" in s.lower())])
        cat = ["Spatial Task 1 (Specific Drawer)"]
        for name in filter(None, [
            f"is_open({top})" if top else None,
            f"is_open({bottom})" if bottom else None,
            f"contact({tgt},{top})" if (tgt and top) else None,
            f"contact({tgt},{bottom})" if (tgt and bottom) else None,
        ]):
            if name in all_names:
                cat.append(f"  - {name} {src_of(name)}")
        if len(cat) > 1:
            blocks.append("\n".join(cat))

    # Category: Spatial Task 2 (Identical Targets)
    if has_identical_targets(meta, lang) and tgt:
        left = first_or_none([s for s in sites_all if ("left" in s.lower())])
        right = first_or_none([s for s in sites_all if ("right" in s.lower())])
        cat = ["Spatial Task 2 (Identical Targets)"]
        for name in filter(None, [
            f"on({tgt},{left})" if (left) else None,
            f"contact({tgt},{left})" if (left) else None,
            f"on({tgt},{right})" if (right) else None,
            f"contact({tgt},{right})" if (right) else None,
        ]):
            if name in all_names:
                cat.append(f"  - {name} {src_of(name)}")
        if len(cat) > 1:
            blocks.append("\n".join(cat))

    # Category: Spatial Task 3 (Specific Containment Region)
    if is_specific_region(lang) and tgt:
        front = first_or_none([s for s in sites_all if ("front" in s.lower())])
        back = first_or_none([s for s in sites_all if ("back" in s.lower())])
        general = reg or first_or_none(sites_all)
        cat = ["Spatial Task 3 (Specific Containment Region)"]
        for name in filter(None, [
            f"on({tgt},{front})" if front else None,
            f"on({tgt},{general})" if general else None,
            f"on({tgt},{back})" if back else None,
        ]):
            if name in all_names:
                cat.append(f"  - {name} {src_of(name)}")
        if len(cat) > 1:
            blocks.append("\n".join(cat))

    # If nothing matched, dump minimal list grounded in available names
    if len(blocks) <= 1:
        fallback = ["Available Concepts"] + [f"  - {n} {src_of(n)}" for n in sorted(all_names)]
        blocks.append("\n".join(fallback))

    return "\n".join(blocks)


def select_by_category(task_key: str, rel: Dict, chk: Dict) -> Dict[str, str]:
    """Return text blocks per category for a single task, keyed by category id.

    Keys: g1, g2, s1, s2, s3
    """
    meta = (rel.get("meta") if rel else None) or (chk.get("meta") if chk else None) or {}
    lang = str(meta.get("language_instruction") or task_key)
    involved_objs = meta.get("involved_objects") or []
    involved_sites = meta.get("involved_sites") or []
    sites_all = meta.get("sites") or []
    rel_names = set((rel.get("concepts") or {}).keys())
    chk_names = set((chk.get("concepts") or {}).keys())
    all_names = rel_names | chk_names

    def src_of(name: str) -> str:
        if name in rel_names and name in chk_names:
            return "[rel|chk]"
        if name in rel_names:
            return "[rel]"
        if name in chk_names:
            return "[chk]"
        return ""

    tgt = first_or_none([o for o in involved_objs if o != "gripper"]) or first_or_none(involved_objs) or ""

    # Infer a usable region from relations if involved_sites is empty or not helpful
    def extract_sites_for_target(t: str) -> List[str]:
        out = []
        prefix_on = f"on({t},"
        prefix_in = f"in({t},"
        for name in rel_names:
            if name.startswith(prefix_on) or name.startswith(prefix_in):
                try:
                    site = name.split(',',1)[1][:-1]
                except Exception:
                    continue
                out.append(site)
        # Also consider region_contains(site,t)
        suffix = f",{t})"
        for name in rel_names:
            if name.startswith("region_contains(") and name.endswith(suffix):
                site = name.split('(',1)[1].split(',',1)[0]
                out.append(site)
        # Deduplicate
        seen=set(); uniq=[]
        for s in out:
            if s not in seen:
                uniq.append(s); seen.add(s)
        return uniq

    def choose_primary_region(t: str) -> str:
        sites = extract_sites_for_target(t)
        if not sites:
            return ""
        # scoring: prefer contain regions, avoid init regions
        def score(s: str) -> Tuple[int,int]:
            return (1 if ("contain" in s.lower()) else 0, -1 if ("init" in s.lower()) else 0)
        sites.sort(key=score, reverse=True)
        return sites[0]

    reg = first_or_none(involved_sites) or (choose_primary_region(tgt) if tgt else "")

    header = f"Task: {lang}\nSources: "
    srcs = []
    if rel:
        srcs.append(f"{task_key}__relations_hash.json")
    if chk:
        srcs.append(f"{task_key}__checks_hash.json")
    head = header + ", ".join(srcs)

    out: Dict[str, List[str]] = {k: [head] for k in ("g1","g2","s1","s2","s3")}

    # General Task 1
    if is_pick_and_place(lang) and len([o for o in involved_objs if o != "gripper"]) == 1 and tgt:
        lines = ["General Task 1 (Pick & Place, single target)"]
        # Always try the canonical three; only include if present
        for name in [f"contact({tgt},gripper)", f"on({tgt},{reg})" if reg else "", f"ontop({tgt},{reg})" if reg else "", f"contact({tgt},{reg})" if reg else ""]:
            if name and name in all_names:
                lines.append(f"  - {name} {src_of(name)}")
        if len(lines) > 1:
            out["g1"].append("\n".join(lines))

    # General Task 2
    if has_state_change(lang):
        L = lang.lower()
        # Collect all stateful sites mentioned/available for this task
        state_sites = [
            s for s in sites_all
            if any(k in s.lower() for k in ("drawer", "microwave", "stove", "oven"))
        ]
        # If language specifies positions (top/middle/bottom), filter accordingly;
        # allow multiple sites if multiple are mentioned (e.g., top AND bottom).
        pos_filters = []
        for pos in ("top", "middle", "bottom"):
            if pos in L:
                pos_filters.append(pos)
        if pos_filters:
            selected_sites = [s for s in state_sites if any(p in s.lower() for p in pos_filters)]
        else:
            selected_sites = state_sites or ([reg] if reg else [])

        # If nothing obvious from sites list, fall back to any predicates present in hashes
        # by parsing site names from is_open()/is_close() entries.
        if not selected_sites:
            for name in all_names:
                if name.startswith("is_open(") or name.startswith("is_close("):
                    try:
                        site = name.split("(", 1)[1][:-1]
                    except Exception:
                        continue
                    if site:
                        selected_sites.append(site)
            # Deduplicate while preserving order
            seen = set(); selected_sites = [s for s in selected_sites if not (s in seen or seen.add(s))]

        if selected_sites:
            lines = ["General Task 2 (State Change)"]
            want_open = any(k in L for k in ("open", "turn on", "power on"))
            want_close = any(k in L for k in ("close", "shut", "turn off", "power off"))
            for site in selected_sites:
                # Include all matching states present in hashes for each relevant site
                for pred, wanted in (("is_open", want_open), ("is_close", want_close)):
                    name = f"{pred}({site})"
                    if (not want_open and not want_close) or wanted:
                        if name in all_names:
                            lines.append(f"  - {name} {src_of(name)}")
            if len(lines) > 1:
                out["g2"].append("\n".join(lines))

    # Spatial Task 1
    if is_specific_drawer(lang) and tgt:
        top = first_or_none([s for s in sites_all if ("drawer" in s.lower() and "top" in s.lower())])
        bottom = first_or_none([s for s in sites_all if ("drawer" in s.lower() and "bottom" in s.lower())])
        lines = ["Spatial Task 1 (Specific Drawer)"]
        for name in filter(None, [
            f"is_open({top})" if top else None,
            f"is_open({bottom})" if bottom else None,
            f"contact({tgt},{top})" if (top) else None,
            f"contact({tgt},{bottom})" if (bottom) else None,
        ]):
            if name in all_names:
                lines.append(f"  - {name} {src_of(name)}")
        if len(lines) > 1:
            out["s1"].append("\n".join(lines))

    # Spatial Task 2
    if has_identical_targets(meta, lang) and tgt:
        left = first_or_none([s for s in sites_all if ("left" in s.lower())])
        right = first_or_none([s for s in sites_all if ("right" in s.lower())])
        lines = ["Spatial Task 2 (Identical Targets)"]
        for name in filter(None, [
            f"on({tgt},{left})" if left else None,
            f"contact({tgt},{left})" if left else None,
            f"on({tgt},{right})" if right else None,
            f"contact({tgt},{right})" if right else None,
        ]):
            if name in all_names:
                lines.append(f"  - {name} {src_of(name)}")
        if len(lines) > 1:
            out["s2"].append("\n".join(lines))

    # Spatial Task 3
    if is_specific_region(lang) and tgt:
        front = first_or_none([s for s in sites_all if ("front" in s.lower())])
        back = first_or_none([s for s in sites_all if ("back" in s.lower())])
        general = reg or first_or_none(sites_all)
        lines = ["Spatial Task 3 (Specific Containment Region)"]
        for name in filter(None, [
            f"on({tgt},{front})" if front else None,
            f"on({tgt},{general})" if general else None,
            f"on({tgt},{back})" if back else None,
        ]):
            if name in all_names:
                lines.append(f"  - {name} {src_of(name)}")
        if len(lines) > 1:
            out["s3"].append("\n".join(lines))

    # Convert lists to text blocks (or empty string if nothing)
    return {k: ("\n".join(v) if len(v) > 1 else "") for k, v in out.items()}


def main():
    hashes = load_hashes()
    g1_blocks: List[str] = []
    g2_blocks: List[str] = []
    s1_blocks: List[str] = []
    s2_blocks: List[str] = []
    s3_blocks: List[str] = []

    for task_key, rel, chk in hashes:
        chosen = select_by_category(task_key, rel, chk)
        if chosen["g1"]:
            g1_blocks.extend([chosen["g1"], ""]) 
        if chosen["g2"]:
            g2_blocks.extend([chosen["g2"], ""]) 
        if chosen["s1"]:
            s1_blocks.extend([chosen["s1"], ""]) 
        if chosen["s2"]:
            s2_blocks.extend([chosen["s2"], ""]) 
        if chosen["s3"]:
            s3_blocks.extend([chosen["s3"], ""]) 

    # Write separate files per category
    OUT_GENERAL_1.write_text(("\n".join(g1_blocks)).strip() + "\n")
    OUT_GENERAL_2.write_text(("\n".join(g2_blocks)).strip() + "\n")
    OUT_SPATIAL_1.write_text(("\n".join(s1_blocks)).strip() + "\n")
    OUT_SPATIAL_2.write_text(("\n".join(s2_blocks)).strip() + "\n")
    OUT_SPATIAL_3.write_text(("\n".join(s3_blocks)).strip() + "\n")
    # Maintain legacy combined outputs for convenience
    combined = []
    combined.extend(g1_blocks)
    combined.extend(g2_blocks)
    combined.extend(s1_blocks)
    combined.extend(s2_blocks)
    combined.extend(s3_blocks)
    combined_text = ("\n".join(combined)).strip() + "\n"
    OUT_1.write_text(combined_text)
    OUT_2.write_text(combined_text)
    print("Wrote category files:")
    print(f" - {OUT_GENERAL_1}")
    print(f" - {OUT_GENERAL_2}")
    print(f" - {OUT_SPATIAL_1}")
    print(f" - {OUT_SPATIAL_2}")
    print(f" - {OUT_SPATIAL_3}")


if __name__ == "__main__":
    main()
