"""Deterministic multi-resolution `.ico` generator for MelodyDesk 2.0 (Luthier Rosette emblem)."""
from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw


def build_rosette_icon(out_path: Path) -> Path:
    size = 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded squircle mahogany/espresso studio badge
    pad = 56
    draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=216,
        fill=(28, 25, 23, 255),
        outline=(196, 164, 124, 255),
        width=18,
    )
    inner_pad = pad + 34
    draw.rounded_rectangle(
        [inner_pad, inner_pad, size - inner_pad, size - inner_pad],
        radius=186,
        outline=(92, 74, 58, 220),
        width=6,
    )

    cx, cy = size // 2, size // 2

    # Concentric spruce & herringbone soundhole rosette rings
    ring_specs = [
        (338, (214, 182, 140, 255), 14),
        (308, (181, 68, 42, 255), 18),
        (280, (231, 214, 190, 255), 10),
    ]
    for r, color, width in ring_specs:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)

    # Rosette radial luthiery inlays (24 subtle gold ticks between r=290..326)
    for k in range(24):
        angle = 2.0 * math.pi * k / 24.0
        x1 = cx + 292 * math.cos(angle)
        y1 = cy + 292 * math.sin(angle)
        x2 = cx + 324 * math.cos(angle)
        y2 = cy + 324 * math.sin(angle)
        draw.line([(x1, y1), (x2, y2)], fill=(236, 212, 176, 185), width=6)

    # Soundhole deep acoustic chamber
    hole_r = 250
    draw.ellipse(
        [cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r],
        fill=(15, 13, 12, 255),
        outline=(120, 96, 74, 255),
        width=8,
    )

    # 6 acoustic phosphor-bronze / silvered strings traversing the soundhole
    span = 220
    for i in range(6):
        sx = cx - span // 2 + round(i * span / 5.0)
        thickness = 12 - i  # wound bass strings on left, crisp treble on right
        tone = (218, 170, 118, 245) if i < 3 else (242, 236, 228, 245)
        draw.line([(sx, pad + 56), (sx, size - pad - 56)], fill=tone, width=thickness)

    # Vermilion playhead accent diamond at center
    d = 46
    draw.polygon(
        [(cx, cy - d), (cx + d, cy), (cx, cy + d), (cx - d, cy)],
        fill=(181, 68, 42, 255),
        outline=(251, 249, 245, 255),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    base_256 = img.resize((256, 256), Image.Resampling.LANCZOS)
    base_256.save(out_path, format="ICO", sizes=ico_sizes)
    return out_path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "assets" / "melodydesk.ico"
    saved = build_rosette_icon(target)
    print(f"Generated {saved} ({saved.stat().st_size} bytes)")
