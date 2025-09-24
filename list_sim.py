#!/usr/bin/env python3
"""
List LIBERO simulator capabilities per task and dump instructions.

Runs with no arguments. For each task in LIBERO-90 and the 10 LIBERO-10
holdout tasks, this will:
- Instantiate the env headlessly
- Print object list, object states, contact / collision availability
- Print target containment region definitions (site objects)
- Append the language instruction to output txt files

Outputs:
- libero_90_instructions.txt
- libero_10_instructions.txt
"""

import os
import argparse
import sys
from pathlib import Path
import json

# Ensure imports resolve when running from repo root
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "LIBERO"))

# Headless MuJoCo
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env
import numpy as np
from vla_scripts.visual_concepts_extractor import (
    get_env_inventory,
    get_goal_predicates,
    contact_obj_with_robot,
    build_contact_index,
    evaluate_site_methods,
    evaluate_site_geometry_methods,
    contact_between_bodies,
    get_site_parent_name,
    should_include_under,
    collect_scene_predicates,
    expand_overlap_objects,
    build_concept_hash,
)


def get_tasks(benchmark_name):
    bm_cls = benchmark.get_benchmark(benchmark_name)
    bm = bm_cls()
    return bm.tasks


def _sanitize_filename(s: str) -> str:
    s = s.strip().lower()
    import re as _re
    s = _re.sub(r"\s+", "_", s)
    s = _re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "task"


def process_tasks(tasks, suite_label: str, instr_out_path: Path, print_path: Path):
    print(f"===== Processing {suite_label} ({len(tasks)} tasks) =====")
    all_print_lines = []
    # Where we dump concept hash tables
    hash_out_dir = REPO_ROOT / "test" / "hash"
    hash_out_dir.mkdir(parents=True, exist_ok=True)
    for idx, task in enumerate(tasks):
        print(f"[{suite_label} {idx+1}/{len(tasks)}] {task.name}")
        env = None
        try:
            env, task_desc = get_libero_env(task, model_family="openvla", resolution=64)
            try:
                env.reset()
            except Exception:
                pass
            snap = collect_scene_predicates(env)
            lang = snap.get('language') or task_desc or ''
            # Build concept hash tables (relations + checks)
            try:
                rel_hash = build_concept_hash(env, source="relations")
                chk_hash = build_concept_hash(env, source="checks")
                base_name = lang.strip() if isinstance(lang, str) and lang.strip() else task.name
                base = _sanitize_filename(base_name)
                rel_path = hash_out_dir / f"{base}__relations_hash.json"
                chk_path = hash_out_dir / f"{base}__checks_hash.json"
                with open(rel_path, "w", encoding="utf-8") as f:
                    json.dump(rel_hash, f, indent=2, ensure_ascii=False)
                with open(chk_path, "w", encoding="utf-8") as f:
                    json.dump(chk_hash, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[list_sim] Failed to build/save concept hashes: {e}")
            block = []
            block.append(lang)
            block.append(f"objects: {snap.get('objects', [])}")
            block.append(f"sites: {snap.get('sites', [])}")
            block.append(f"fixtures: {snap.get('fixtures', [])}")
            preds = snap.get('predicates', []) or []
            if preds:
                block.append("Predicates: " + ", ".join(preds))
            for g in snap.get('goals', []) or []:
                expr, val = g.get('expr'), g.get('value')
                if expr is not None and val is not None:
                    block.append(f"{expr}: {val}")
            inv_objs = snap.get('involved_objects', []) or []
            if inv_objs:
                block.append("Involved objects: " + ", ".join(inv_objs))
            inv_sites = snap.get('involved_sites', []) or []
            if inv_sites:
                block.append("Involved regions: " + ", ".join(inv_sites))
            for c in snap.get('checks', []) or []:
                expr, val = c.get('expr'), c.get('value')
                if expr is not None and val is not None:
                    block.append(f"{expr}: {val}")
            all_print_lines.append("\n".join(block))
        except Exception as e:
            print(f"[list_sim] Failed to create / introspect env for {task.name}: {e}")
        finally:
            try:
                if env is not None and hasattr(env, 'close'):
                    env.close()
            except Exception:
                pass
    # Write structured output to the suite's instructions file
    with open(instr_out_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_print_lines) + "\n")
    print(f"Wrote structured output → {instr_out_path}")


def main():
    parser = argparse.ArgumentParser(description="List LIBERO tasks and print simulator predicates")
    parser.add_argument(
        "--suite",
        choices=["libero_90", "libero_10", "both"],
        default="libero_90",
        help="Task suite to process",
    )
    args = parser.parse_args()

    if args.suite == "libero_90":
        tasks_90 = get_tasks("libero_90")
        out_instr = REPO_ROOT / "libero_90_instructions.txt"
        out_print = REPO_ROOT / "print.txt"
        process_tasks(tasks_90, "LIBERO-90", out_instr, out_print)
    elif args.suite == "libero_10":
        tasks_10 = get_tasks("libero_10")
        out_instr = REPO_ROOT / "libero_10_instructions.txt"
        out_print = REPO_ROOT / "print.txt"
        process_tasks(tasks_10, "LIBERO-10", out_instr, out_print)
    else:  # both
        tasks_90 = get_tasks("libero_90")
        tasks_10 = get_tasks("libero_10")
        process_tasks(
            tasks_90,
            "LIBERO-90",
            REPO_ROOT / "libero_90_instructions.txt",
            REPO_ROOT / "print_libero_90.txt",
        )
        process_tasks(
            tasks_10,
            "LIBERO-10",
            REPO_ROOT / "libero_10_instructions.txt",
            REPO_ROOT / "print_libero_10.txt",
        )


if __name__ == "__main__":
    main()
