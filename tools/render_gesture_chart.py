"""Render docs/media/gesture-chart.svg straight from the gesture vocabulary.

The chart is generated rather than drawn by hand so that it can never drift
out of sync with :mod:`aerovision.gestures`: add a row to ``GESTURE_SPECS``
and the picture in the README grows a card.

    python tools/render_gesture_chart.py            # write the SVG
    python tools/render_gesture_chart.py --png out.png   # preview (needs Pillow)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerovision.gestures import GESTURE_SPECS, Gesture  # noqa: E402

# -- palette ---------------------------------------------------------------
BACKGROUND = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
BONE = "#3fb950"
JOINT = "#39d0d8"
TITLE = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58a6ff"
# GitHub strips <style> blocks from SVG, so fonts are set per element.
FONT = "Segoe UI,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Consolas,monospace"

# -- layout ----------------------------------------------------------------
COLUMNS = 5
CARD_W, CARD_H = 232, 236
GAP = 14
MARGIN = 20
HEADER = 58

# -- hand model ------------------------------------------------------------
# Local coordinates: wrist at the origin, +y points up the hand.
WRIST = (0.0, 0.0)
FINGERS = {
    #        mcp position   base angle  segment lengths
    "index": ((-26.0, 60.0), -10.0, (30.0, 21.0, 17.0)),
    "middle": ((-3.0, 66.0), -1.0, (33.0, 23.0, 18.0)),
    "ring": ((19.0, 62.0), 9.0, (30.0, 21.0, 17.0)),
    "pinky": ((38.0, 52.0), 20.0, (24.0, 17.0, 14.0)),
}
FINGER_ORDER = ("index", "middle", "ring", "pinky")
STRAIGHT_BENDS = (6.0, 6.0, 6.0)
CURLED_BENDS = (62.0, 78.0, 62.0)
CURLED_SCALE = 0.82

THUMB_CMC = (-30.0, 20.0)
THUMB_SEGMENTS = (24.0, 20.0, 17.0)
THUMB_OUT = (-62.0, (-8.0, -10.0, -8.0))
THUMB_TUCKED = (16.0, (24.0, 28.0, 26.0))


def _chain(origin, base_angle, lengths, bends):
    """Walk a finger outwards, bending at each joint (forward kinematics)."""
    x, y = origin
    angle = base_angle
    points = [(x, y)]
    for length, bend in zip(lengths, bends, strict=True):
        angle += bend
        radians = math.radians(angle)
        x += length * math.sin(radians)
        y += length * math.cos(radians)
        points.append((x, y))
    return points


def hand_skeleton(pattern, rotation_deg=0.0):
    """Return the bones of a hand posed to match a finger pattern.

    Each bone is a list of points; ``rotation_deg`` spins the whole hand,
    which is how a thumbs-up becomes a thumbs-down.
    """
    thumb_out, *fingers = pattern
    base_angle, bends = THUMB_OUT if thumb_out else THUMB_TUCKED
    bones = [
        [WRIST, THUMB_CMC],
        _chain(THUMB_CMC, base_angle, THUMB_SEGMENTS, bends),
    ]
    knuckles = []
    for name, extended in zip(FINGER_ORDER, fingers, strict=True):
        mcp, angle, lengths = FINGERS[name]
        knuckles.append(mcp)
        if extended:
            bones.append(_chain(mcp, angle, lengths, STRAIGHT_BENDS))
        else:
            folded = tuple(length * CURLED_SCALE for length in lengths)
            bones.append(_chain(mcp, angle, folded, CURLED_BENDS))
    bones.append([WRIST, knuckles[0]])  # palm edge, thumb side
    bones.append([WRIST, knuckles[-1]])  # palm edge, pinky side
    bones.append(knuckles)  # knuckle line

    radians = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    return [
        [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in bone] for bone in bones
    ]


def place(bones, centre, box=(150.0, 150.0)):
    """Fit a posed hand into a card slot (y flips: SVG grows downwards).

    Each pose is scaled and centred on its own bounding box, so a rotated
    thumbs-down lands in the same visual slot as an upright open palm.
    """
    points = [point for bone in bones for point in bone]
    min_x = min(x for x, _ in points)
    max_x = max(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_y = max(y for _, y in points)
    scale = min(box[0] / max(max_x - min_x, 1e-6), box[1] / max(max_y - min_y, 1e-6), 1.05)
    mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    cx, cy = centre
    return [
        [(cx + (x - mid_x) * scale, cy - (y - mid_y) * scale) for x, y in bone]
        for bone in bones
    ]


def pose_for(spec):
    """Pick the drawn pattern and hand rotation for one vocabulary row."""
    if spec.gesture is Gesture.UP:
        return spec.patterns[0], -66.0
    if spec.gesture is Gesture.DOWN:
        return spec.patterns[0], 114.0
    return spec.patterns[0], 0.0


def card_geometry():
    """Yield one (spec, origin, bones) triple per gesture."""
    for index, spec in enumerate(GESTURE_SPECS):
        column, row = index % COLUMNS, index // COLUMNS
        x = MARGIN + column * (CARD_W + GAP)
        y = HEADER + MARGIN + row * (CARD_H + GAP)
        pattern, rotation = pose_for(spec)
        centre = (x + CARD_W / 2, y + 92)
        yield spec, (x, y), place(hand_skeleton(pattern, rotation), centre)


def canvas_size():
    rows = math.ceil(len(GESTURE_SPECS) / COLUMNS)
    width = MARGIN * 2 + COLUMNS * CARD_W + (COLUMNS - 1) * GAP
    height = HEADER + MARGIN * 2 + rows * CARD_H + (rows - 1) * GAP
    return width, height


def render_svg() -> str:
    width, height = canvas_size()
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="AeroVision gesture vocabulary">',
        f'<rect width="{width}" height="{height}" rx="14" fill="{BACKGROUND}"/>',
        f'<text x="{MARGIN + 4}" y="38" fill="{TITLE}" font-size="21" font-weight="600" '
        f'font-family="{FONT}">Gesture vocabulary</text>',
        f'<text x="{width - MARGIN - 4}" y="38" fill="{MUTED}" font-size="13" '
        f'text-anchor="end" font-family="{FONT}">one webcam hand, ten drone commands</text>',
    ]

    for spec, (x, y), bones in card_geometry():
        out.append(
            f'<rect x="{x}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
            f'fill="{CARD}" stroke="{BORDER}"/>'
        )
        for bone in bones:
            path = " ".join(
                f"{'M' if i == 0 else 'L'}{px:.1f} {py:.1f}" for i, (px, py) in enumerate(bone)
            )
            out.append(
                f'<path d="{path}" fill="none" stroke="{BONE}" stroke-width="3.2" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )
        for bone in bones:
            for px, py in bone:
                out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.1" fill="{JOINT}"/>')
        out.extend(
            [
                f'<text x="{x + CARD_W / 2}" y="{y + CARD_H - 44}" fill="{TITLE}" '
                f'font-size="15" font-weight="600" text-anchor="middle" '
                f'font-family="{FONT}">{spec.hand_shape}</text>',
                f'<text x="{x + CARD_W / 2}" y="{y + CARD_H - 24}" fill="{ACCENT}" '
                f'font-size="12.5" font-family="{MONO}" '
                f'text-anchor="middle">{spec.gesture.value}</text>',
                f'<text x="{x + CARD_W / 2}" y="{y + CARD_H - 8}" fill="{MUTED}" '
                f'font-size="11.5" text-anchor="middle" '
                f'font-family="{FONT}">{spec.action}</text>',
            ]
        )

    out.append("</svg>")
    return "\n".join(out)


def render_png(path: Path) -> None:
    """Rasterise the same geometry with Pillow, for quick visual checks."""
    from PIL import Image, ImageDraw

    width, height = canvas_size()
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for spec, (x, y), bones in card_geometry():
        draw.rounded_rectangle([x, y, x + CARD_W, y + CARD_H], 10, fill=CARD, outline=BORDER)
        for bone in bones:
            draw.line([tuple(point) for point in bone], fill=BONE, width=3, joint="curve")
            for px, py in bone:
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=JOINT)
        draw.text((x + CARD_W / 2, y + CARD_H - 46), spec.hand_shape, fill=TITLE, anchor="ma")
        draw.text((x + CARD_W / 2, y + CARD_H - 28), spec.gesture.value, fill=ACCENT, anchor="ma")
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs" / "media" / "gesture-chart.svg"
    )
    parser.add_argument("--png", type=Path, help="also write a raster preview")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_svg(), encoding="utf-8")
    print(f"Wrote {args.out}")
    if args.png:
        render_png(args.png)
        print(f"Wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
