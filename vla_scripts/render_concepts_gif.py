#!/usr/bin/env python3
"""
Render a combined GIF with (left) action replay frames and (right) concept states,
aligned strictly by timestep for a selected episode.

Given a dataset root that contains:
- sim_states/episodes_index.h5
- concepts/{language_or_task}.csv (per-task recorder saved by reconstruction)
and optionally an images root containing per-episode trajectory.gif, this tool will:

1) For a selected episode (by global episode_idx),
2) Slice the corresponding time window from the per-task concepts CSV,
3) Render a combined GIF where each frame shows the action frame on the left
   and the concept names with 0/1 on the right.
"""

from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vla_scripts.state_io import resolve_paths

# ---------------------- Configuration ----------------------
# Edit these values, then run: python vla_scripts/render_concepts_gif.py

DATASET_ROOT = "/work/nvme/bfbo/xzhang42/data/pilot_test/optimized_trajectory_data"
EPISODE_IDX = 0
IMAGES_ROOT = "/work/nvme/bfbo/xzhang42/data/pilot_test/reconstructed_trajectory_data/images"
ONLY_CHANGING = True
DURATION_MS = 100
CANVAS_WIDTH = 1400


def _sanitize(s: str) -> str:
    import re
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]", "", s)
    return s or "task"


def _load_episodes_index(sim_states_root: Path):
    epi_path = sim_states_root / "episodes_index.h5"
    if not epi_path.exists():
        raise FileNotFoundError(f"episodes_index.h5 not found at {epi_path}")
    with h5py.File(epi_path, "r") as f:
        def _read_str(name):
            arr = f[name][...]
            return [x.decode("utf-8") for x in arr]
        idx = f["episode_idx"][...].astype(int).tolist()
        task_name = _read_str("task_name") if "task_name" in f else [""] * len(idx)
        lang = _read_str("language_instruction") if "language_instruction" in f else [""] * len(idx)
        task_id = f["task_id"][...].astype(int).tolist() if "task_id" in f else [-1] * len(idx)
        num_ts = f["num_timesteps"][...].astype(int).tolist()
        return {
            "episode_idx": idx,
            "task_name": task_name,
            "language_instruction": lang,
            "task_id": task_id,
            "num_timesteps": num_ts,
        }


def _load_task_concepts_csv(concepts_root: Path, language_or_task: str) -> Tuple[List[str], np.ndarray]:
    base = _sanitize(language_or_task)
    path = concepts_root / f"{base}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Concepts CSV not found: {path}")
    concepts: List[str] = []
    values: List[List[int]] = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        # Skip first line if comment
        first = next(reader, None)
        if first and first[0].startswith("#"):
            header = next(reader, None)
        else:
            header = first
        # Read rows
        for row in reader:
            if not row:
                continue
            concepts.append(row[0])
            vals = [int(x) if x != '' else 0 for x in row[1:]]
            values.append(vals)
    arr = np.array(values, dtype=np.int8) if values else np.zeros((0, 0), dtype=np.int8)
    return concepts, arr


def _slice_episode_columns(epi_meta: dict, target_idx: int) -> Tuple[int, int, str, int, int]:
    # Determine episode's language key and local offset within that task's sequence
    lang_all = epi_meta["language_instruction"]
    num_ts_all = epi_meta["num_timesteps"]
    # Use language if present, else fallback to task name
    key = lang_all[target_idx] if lang_all and lang_all[target_idx] else epi_meta["task_name"][target_idx]
    # Compute offset as sum of timesteps over earlier episodes with same key
    offset = 0
    for i in range(target_idx):
        ki = lang_all[i] if lang_all and lang_all[i] else epi_meta["task_name"][i]
        if ki == key:
            offset += int(num_ts_all[i])
    length = int(num_ts_all[target_idx])
    task_ids = epi_meta.get("task_id", [-1] * len(epi_meta["episode_idx"]))
    epi_ids = epi_meta.get("episode_id", [-1] * len(epi_meta["episode_idx"]))
    return offset, length, key, int(task_ids[target_idx] if task_ids else -1), int(epi_ids[target_idx] if epi_ids else -1)


def _render_concepts_frames(concepts: List[str], values: np.ndarray,
                            only_changing: bool, width: int = 800,
                            bg_color=(20, 20, 20), on_color=(0, 200, 0), off_color=(200, 0, 0)) -> List[Image.Image]:
    # Filter concepts if only_changing
    if values.size == 0:
        return []
    keep_idx = list(range(len(concepts)))
    if only_changing:
        keep_idx = [i for i, row in enumerate(values) if np.any(row != row[0])]
        if not keep_idx:
            keep_idx = list(range(len(concepts)))  # fallback to show all

    concepts_f = [concepts[i] for i in keep_idx]
    values_f = values[keep_idx, :]

    # Layout
    rows = len(concepts_f)
    row_h = 22
    pad = 12
    height = pad * 2 + rows * row_h
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    frames: List[Image.Image] = []
    T = values_f.shape[1]
    for t in range(T):
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        y = pad
        for i, name in enumerate(concepts_f):
            val = int(values_f[i, t])
            color = on_color if val == 1 else off_color
            # Name
            draw.text((pad, y), f"{name}", fill=(220, 220, 220), font=font)
            # Numeric value right after name
            try:
                name_box = draw.textbbox((pad, y), name, font=font)
                nx = name_box[2] + 8
            except Exception:
                nx = pad + max(8, len(name) * 8)
            draw.text((nx, y), "1" if val == 1 else "0", fill=color, font=font)
            # Colored indicator box on the far right with numeric overlay
            box_w, box_h = 16, 16
            bx = max(pad, width - pad - box_w)
            by = y + max(0, (row_h - box_h) // 2)
            draw.rectangle([bx, by, bx + box_w, by + box_h], fill=color)
            digit = "1" if val == 1 else "0"
            try:
                tb = draw.textbbox((0, 0), digit, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                tw, th = (8, 12)
            tx = bx + (box_w - tw) // 2
            ty = by + (box_h - th) // 2
            draw.text((tx, ty), digit, fill=(255, 255, 255), font=font)
            y += row_h
        frames.append(img)
    return frames


def _load_gif_frames(path: Path) -> List[Image.Image]:
    frames: List[Image.Image] = []
    if not path.exists():
        return frames
    try:
        with Image.open(path) as im:
            i = 0
            while True:
                im.seek(i)
                frames.append(im.convert("RGB").copy())
                i += 1
    except EOFError:
        pass
    except Exception:
        frames = []
    return frames


def main():
    root = Path(DATASET_ROOT)
    paths = resolve_paths(str(root))
    sim_states_root = paths["sim_states"]
    concepts_root = paths["concepts"]

    epi = _load_episodes_index(sim_states_root)
    if EPISODE_IDX < 0 or EPISODE_IDX >= len(epi["episode_idx"]):
        raise IndexError(f"episode_idx out of range: {EPISODE_IDX}")

    # Find slice and identifiers
    offset, length, key, task_id, episode_id = _slice_episode_columns(epi, EPISODE_IDX)
    concepts, matrix = _load_task_concepts_csv(concepts_root, key)
    if matrix.shape[1] < offset + length:
        raise ValueError("Concepts CSV does not have enough timesteps for this episode slice")
    episode_vals = matrix[:, offset: offset + length]

    # Render concept frames
    concept_frames = _render_concepts_frames(concepts, episode_vals, only_changing=ONLY_CHANGING, width=max(600, CANVAS_WIDTH // 2))
    if not concept_frames:
        print("No concept frames rendered (empty concepts or zero timesteps)")
        return

    # Locate trajectory.gif for this episode
    action_frames: List[Image.Image] = []
    dest = None
    if IMAGES_ROOT:
        base = Path(IMAGES_ROOT)
        if task_id >= 0 and episode_id >= 0:
            traj_path = base / f"task_{task_id}" / f"episode_{episode_id}" / "trajectory.gif"
            action_frames = _load_gif_frames(traj_path)
            if action_frames:
                dest = traj_path.parent
        if not action_frames:
            # Fallback: any trajectory.gif
            try:
                cand = list(base.glob("**/trajectory.gif"))
                if cand:
                    from PIL import Image as _Image
                    action_frames = _load_gif_frames(cand[0])
                    dest = cand[0].parent
            except Exception:
                pass
    # Strict sync length
    T = episode_vals.shape[1]
    if action_frames and len(action_frames) != T:
        print(f"[warn] trajectory.gif frames ({len(action_frames)}) != episode steps ({T}); trimming to min")
        L = min(len(action_frames), T)
        action_frames = action_frames[:L]
        concept_frames = concept_frames[:L]
    else:
        L = len(concept_frames)

    # Compose side-by-side
    combined: List[Image.Image] = []
    left_w = (CANVAS_WIDTH * 3) // 5
    right_w = CANVAS_WIDTH - left_w
    for i in range(L):
        left = action_frames[i] if action_frames else Image.new("RGB", (left_w, concept_frames[i].height), color=(0, 0, 0))
        if left.width != left_w:
            ratio = left_w / float(left.width)
            left = left.resize((left_w, max(1, int(left.height * ratio))), Image.Resampling.BILINEAR)
        right = concept_frames[i]
        h = max(left.height, right.height)
        canvas = Image.new("RGB", (left_w + right_w, h), color=(0, 0, 0))
        canvas.paste(left, (0, 0))
        canvas.paste(right.resize((right_w, h), Image.Resampling.BILINEAR), (left_w, 0))
        combined.append(canvas)

    if dest is None:
        cand = [Path(IMAGES_ROOT) if IMAGES_ROOT else None, root / "reconstructed_trajectory_data" / "images", root]
        dest = next((p for p in cand if p and p.exists()), root)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "combined.gif"
    combined[0].save(out_path, save_all=True, append_images=combined[1:], duration=max(1, int(DURATION_MS)), loop=0)
    print(f"Saved combined GIF: {out_path}")


if __name__ == "__main__":
    main()
