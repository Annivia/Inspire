#!/usr/bin/env python3
"""
Utilities for rendering concept panels and composing combined frames with action replay.

Independent of LIBERO; imports only PIL and numpy.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_concept_frames(
    concepts: List[str],
    values: np.ndarray,
    width: int = 500,
    font_size: int = 16,
    row_height: int = 22,
    pad: int = 12,
    bg_color=(20, 20, 20),
    on_color=(220, 80, 80),   # faint red for 1
    off_color=(150, 150, 150),# fixed gray for 0
) -> List[Image.Image]:
    """Render per-timestep concept panels as a list of PIL Images.

    Args:
        concepts: list of concept names (N)
        values: array [N, T] with 0/1 values
        width: panel width
        font_size: font size for text
        row_height: pixels per concept row
        pad: outer padding in pixels
        bg_color: background color
        on_color/off_color: colors for 1 / 0
    """
    if values.size == 0:
        return []

    rows = len(concepts)
    height = pad * 2 + rows * row_height
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    frames: List[Image.Image] = []
    T = values.shape[1]
    for t in range(T):
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        y = pad
        for i, name in enumerate(concepts):
            val = int(values[i, t])
            color = on_color if val == 1 else off_color
            # Draw name
            draw.text((pad, y), name, fill=(220, 220, 220), font=font)
            # Draw numeric value immediately after the name for visibility
            try:
                # PIL >= 8 provides textbbox for accurate width
                name_box = draw.textbbox((pad, y), name, font=font)
                nx = name_box[2] + 8
            except Exception:
                # Fallback spacing
                nx = pad + max(8, len(name) * (font_size // 2))
            draw.text((nx, y), "1" if val == 1 else "0", fill=color, font=font)
            # Also draw a colored indicator box on the far right with numeric overlay
            box_w, box_h = 16, 16
            bx = max(pad, width - pad - box_w)
            by = y + max(0, (row_height - box_h) // 2)
            draw.rectangle([bx, by, bx + box_w, by + box_h], fill=color)
            # Center the digit '1' or '0' in the box for explicit binary cue
            digit = "1" if val == 1 else "0"
            try:
                tb = draw.textbbox((0, 0), digit, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                tw, th = (font.size // 2, font.size)
            tx = bx + (box_w - tw) // 2
            ty = by + (box_h - th) // 2
            draw.text((tx, ty), digit, fill=(255, 255, 255), font=font)
            y += row_height
        frames.append(img)
    return frames


def compose_action_concepts(
    action_frames: List[Image.Image],
    concept_frames: List[Image.Image],
    left_width: int = 700,
    right_width: int = 600,
    bg=(0, 0, 0),
) -> List[Image.Image]:
    """Compose action and concept frames side-by-side without distorting text.

    - Left (action) is resized to left_width preserving aspect.
    - Right (concept) is rendered at right_width and NOT scaled vertically; we pad
      to match the taller side.
    """
    L = min(len(action_frames), len(concept_frames))
    out: List[Image.Image] = []
    for i in range(L):
        left = action_frames[i].convert("RGB")
        # Resize left by width, preserve aspect
        if left.width != left_width:
            ratio = left_width / float(left.width)
            left = left.resize((left_width, max(1, int(left.height * ratio))), Image.Resampling.BILINEAR)

        right = concept_frames[i].convert("RGB")
        # Ensure right has desired width; do NOT scale by height to avoid text distortion
        if right.width != right_width:
            ratio = right_width / float(right.width)
            right = right.resize((right_width, max(1, int(right.height * ratio))), Image.Resampling.NEAREST)

        H = max(left.height, right.height)
        canvas = Image.new("RGB", (left.width + right.width, H), color=bg)
        # Paste left top-aligned
        canvas.paste(left, (0, 0))
        # Paste right vertically centered (no scaling)
        top = (H - right.height) // 2
        canvas.paste(right, (left.width, top))
        out.append(canvas)
    return out


def save_gif(frames: List[Image.Image], out_path: str, duration_ms: int = 100):
    if not frames:
        return
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=max(1, int(duration_ms)), loop=0)
