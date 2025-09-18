#!/usr/bin/env python3
"""
Standalone test: render concept panels and compose with dummy action frames.
Does not require LIBERO.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vla_scripts.concepts_render_utils import render_concept_frames, compose_action_concepts, save_gif


def _dummy_action_frames(T: int, size=(480, 360)) -> List[Image.Image]:
    frames: List[Image.Image] = []
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for t in range(T):
        img = Image.new("RGB", size, color=(10, 10, 10))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, size[0] - 20, size[1] - 20], outline=(80, 80, 80), width=2)
        draw.text((30, 30), f"Dummy action frame t={t}", fill=(220, 220, 220), font=font)
        frames.append(img)
    return frames


def main():
    # Dummy concepts and values
    concepts = ["contact(cup,plate)", "on(cup,table)", "in(cup,drawer)", "region_contains(tray,cup)"]
    T = 40
    values = np.zeros((len(concepts), T), dtype=np.int8)
    # Create some changes
    values[0, 5:10] = 1
    values[1, 15:25] = 1
    values[2, 20:] = 1
    values[3, ::7] = 1

    # Render concept frames and dummy action frames
    c_frames = render_concept_frames(concepts, values, width=600)
    a_frames = _dummy_action_frames(T, size=(480, 360))
    combined = compose_action_concepts(a_frames, c_frames, left_width=700, right_width=600)

    out_dir = Path("./results")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "concepts_layout_test.gif"
    save_gif(combined, str(out_path), duration_ms=100)
    print(f"Saved test GIF at: {out_path}")


if __name__ == "__main__":
    main()
