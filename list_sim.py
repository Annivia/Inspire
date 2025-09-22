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
from vla_scripts.visual_concepts_extractor import debug_print_sim_capabilities


def get_tasks(benchmark_name):
    bm_cls = benchmark.get_benchmark(benchmark_name)
    bm = bm_cls()
    return bm.tasks


def process_tasks(tasks, suite_label: str, instr_out_path: Path):
    print(f"\n===== Processing {suite_label} ({len(tasks)} tasks) =====")
    instructions = []
    for idx, task in enumerate(tasks):
        print(f"\n[{suite_label} {idx+1}/{len(tasks)}] {task.name}")
        env = None
        try:
            env, task_desc = get_libero_env(task, model_family="openvla", resolution=64)
            # Reset and run a couple of quick steps to populate contacts
            try:
                env.reset()
                # step 1: zero action
                env.step([0.0] * 7)
                # step 2: tiny random nudge
                rand = (np.random.randn(7) * 0.01).tolist()
                env.step(rand)
            except Exception:
                # Fallback: ignore if stepping API differs
                pass
            # Print capabilities
            debug_print_sim_capabilities(env)
            # Record instruction
            instructions.append((task.name, task.language or task_desc or ""))
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


def main():
    out_90 = REPO_ROOT / "libero_90_instructions.txt"
    out_10 = REPO_ROOT / "libero_10_instructions.txt"

    tasks_90 = get_tasks("libero_90")
    tasks_10 = get_tasks("libero_10")

    process_tasks(tasks_90, "LIBERO-90", out_90)
    process_tasks(tasks_10, "LIBERO-10", out_10)


if __name__ == "__main__":
    main()
