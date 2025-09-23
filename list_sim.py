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
    expand_overlap_objects,
)


def get_tasks(benchmark_name):
    bm_cls = benchmark.get_benchmark(benchmark_name)
    bm = bm_cls()
    return bm.tasks


def process_tasks(tasks, suite_label: str, instr_out_path: Path, print_path: Path):
    print(f"\n===== Processing {suite_label} ({len(tasks)} tasks) =====")
    instructions = []
    all_print_lines = []
    for idx, task in enumerate(tasks):
        print(f"\n[{suite_label} {idx+1}/{len(tasks)}] {task.name}")
        env = None
        try:
            env, task_desc = get_libero_env(task, model_family="openvla", resolution=64)
            # Reset only (fast)
            try:
                env.reset()
            except Exception:
                pass

            contact_index = build_contact_index(env)
            # Prepare inventory
            inv = get_env_inventory(env)
            objects = inv["objects"]
            sites = inv["sites"]
            fixtures = inv["fixtures"]

            # Task targets: goal predicates
            goals = get_goal_predicates(env)
            involved_objects = []
            involved_sites = []
            for pretty, _, args in goals:
                if len(args) == 1:
                    a = args[0]
                    # site or object
                    involved_objects += [a]
                    if a in sites:
                        involved_sites += [a]
                elif len(args) == 2:
                    a, b = args
                    if a in objects:
                        involved_objects += [a]
                    if a in sites:
                        involved_sites += [a]
                    if b in sites:
                        involved_sites += [b]
                    if b in objects:
                        involved_objects += [b]
            # Deduplicate and split types
            involved_objects = sorted(list({n for n in involved_objects if n in objects}))
            involved_sites = sorted(list({n for n in involved_sites if n in sites}))

            # Build per-task print block
            lang = task.language or task_desc or ""
include_under = should_include_under(env)
            block = []
            block.append(lang)
            block.append(f"objects: {objects}")
            block.append(f"sites: {sites}")
            block.append(f"fixtures: {fixtures}")

            # Goal predicates first
            if goals:
                # Task-specific predicate names
                pred_names = []
                for pretty, _, _ in goals:
                    head = pretty.split("(", 1)[0]
                    pred_names.append(head)
                if pred_names:
                    uniq = sorted(list(set(pred_names)))
                    block.append(f"Predicates: {', '.join(uniq)}")
                for pretty, val, _ in goals:
                    block.append(f"{pretty}: {val}")

            # Involved sets
            if involved_objects:
                block.append(f"Involved objects: {', '.join(involved_objects)}")
            if involved_sites:
                block.append(f"Involved regions: {', '.join(involved_sites)}")

            # Required grounded checks
            for obj_name in involved_objects:
                val = contact_obj_with_robot(env, obj_name, contact_index)
                if val is None:
                    # Unknown gripper site; mark as NA
                    block.append(f"contact({obj_name},gripper): NA")
                else:
                    block.append(f"contact({obj_name},gripper): {val}")
            for site_name in involved_sites:
                for obj_name in involved_objects:
                    evals = evaluate_site_methods(env, site_name, obj_name)
                    if "check_ontop" in evals:
                        block.append(f"ontop({obj_name},{site_name}): {evals['check_ontop']}")
                    if "check_contact" in evals:
                        block.append(f"contact({obj_name},{site_name}): {evals['check_contact']}")
                    # Also evaluate raw geometry methods (SiteObject.in_box / under / TargetZone.on_top)
                    geom = evaluate_site_geometry_methods(env, site_name, obj_name)
                    if "in_box" in geom:
                        block.append(f"in_box({obj_name},{site_name}): {geom['in_box']}")
                    if include_under and ("under" in geom):
                        block.append(f"under({obj_name},{site_name}): {geom['under']}")
                    if "on_top" in geom:
                        block.append(f"on_top({obj_name},{site_name}): {geom['on_top']}")

            # Expanded site checks: all sites (exclude 'init') vs overlapping objects
            if involved_objects:
                overlap_objs = expand_overlap_objects(objects, involved_objects)
            else:
                overlap_objs = []
            for site_name in [s for s in sites if "init" not in s.lower()]:
                # Unary state
                unary = evaluate_site_methods(env, site_name, None)
                if "is_open" in unary:
                    block.append(f"is_open({site_name}): {unary['is_open']}")
                if "is_close" in unary:
                    block.append(f"is_close({site_name}): {unary['is_close']}")
                # Binary w.r.t overlap objects
                for obj_name in overlap_objs:
                    bin_evals = evaluate_site_methods(env, site_name, obj_name)
                    for key in ("check_contact", "check_contain", "check_ontop"):
                        if key in bin_evals:
                            pretty = {
                                "check_contact": "contact",
                                "check_contain": "contain",
                                "check_ontop": "ontop",
                            }[key]
                            block.append(f"{pretty}({obj_name},{site_name}): {bin_evals[key]}")
                    # Also evaluate raw geometry methods
                    geom = evaluate_site_geometry_methods(env, site_name, obj_name)
                    if "in_box" in geom:
                        block.append(f"in_box({obj_name},{site_name}): {geom['in_box']}")
                    if include_under and ("under" in geom):
                        block.append(f"under({obj_name},{site_name}): {geom['under']}")
                    if "on_top" in geom:
                        block.append(f"on_top({obj_name},{site_name}): {geom['on_top']}")

            # Stricter MuJoCo contact reporting
            # Between involved objects and all fixtures
            for obj_name in involved_objects:
                for fix_name in fixtures:
                    val = contact_between_bodies(env, obj_name, fix_name, contact_index)
                    if val is None:
                        continue
                    block.append(f"mj_contact({obj_name},{fix_name}): {val}")
            # Between involved objects themselves (unique pairs)
            for i in range(len(involved_objects)):
                for j in range(i + 1, len(involved_objects)):
                    a = involved_objects[i]
                    b = involved_objects[j]
                    val = contact_between_bodies(env, a, b, contact_index)
                    if val is None:
                        continue
                    block.append(f"mj_contact({a},{b}): {val}")
            # Between involved objects and involved sites (sites have no geoms → NA)
            for obj_name in involved_objects:
                for site_name in involved_sites:
                    block.append(f"mj_contact({obj_name},{site_name}): NA")
                    parent = get_site_parent_name(env, site_name)
                    if parent is not None and parent in fixtures + objects:
                        val = contact_between_bodies(env, obj_name, parent, contact_index)
                        if val is not None:
                            block.append(f"mj_contact({obj_name},{parent}) [parent_of={site_name}]: {val}")

            all_print_lines.append("\n".join(block))

            # Record instruction
            instructions.append((task.name, lang))
        except Exception as e:
            print(f"[list_sim] Failed to create / introspect env for {task.name}: {e}")
            instructions.append((task.name, task.language or ""))
        finally:
            # Best-effort cleanup
            try:
                if env is not None and hasattr(env, "close"):
                    env.close()
            except Exception:
                pass

    # Write instructions txt
    with open(instr_out_path, "w", encoding="utf-8") as f:
        for name, instr in instructions:
            f.write(f"{name}: {instr}\n")
    print(f"Wrote instructions → {instr_out_path}")

    # Write aggregated print.txt
    with open(print_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_print_lines) + "\n")
    print(f"Wrote predicates/states → {print_path}")


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
