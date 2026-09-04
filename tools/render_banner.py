"""Render docs/media/banner.svg, the header image used by the README.

Shares the parametric hand model with the gesture chart so the artwork and
the classifier always describe the same hand.

    python tools/render_banner.py
    python tools/render_banner.py --png out.png   # preview (needs Pillow)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from render_gesture_chart import (
    ACCENT,
    BACKGROUND,
    BONE,
    BORDER,
    FONT,
    JOINT,
    MUTED,
    TITLE,
    hand_skeleton,
    place,
)

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1200, 340

EYEBROW = "PGCC CODE CLUB  ·  2026 ENGINEERING CHALLENGE"
HEADLINE = "AeroVision"
SUBLINE = "A repaired Tello drone that flies on hand gestures"
CHIPS = ("Python 3.10+", "OpenCV", "MediaPipe", "Tello SDK", "Fusion 360")

HAND_CENTRE = (760, 170)
DRONE_CENTRE = (1040, 168)
OPEN_PALM = (True, True, True, True, True)


def drone_paths(centre):
    """Line art quadcopter: (arms, motors, propellers, body)."""
    cx, cy = centre
    arms, motors, propellers = [], [], []
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        mx, my = cx + dx * 66, cy + dy * 44
        arms.append(((cx + dx * 16, cy + dy * 12), (mx, my)))
        motors.append((mx, my, 7.0))
        propellers.append((mx, my, 30.0, 4.5))
    body = (cx - 30, cy - 21, 60, 42)
    return arms, motors, propellers, body


def signal_arcs(origin):
    """Three widening arcs suggesting the command travelling to the drone."""
    x, y = origin
    return [
        (x + step * 26, y, 20 + step * 6, 0.75 - step * 0.2) for step in range(3)
    ]


def render_svg() -> str:
    bones = place(hand_skeleton(OPEN_PALM), HAND_CENTRE, box=(190, 190))
    arms, motors, propellers, body = drone_paths(DRONE_CENTRE)
    bx, by, bw, bh = body

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="AeroVision: a repaired Tello drone that flies on hand gestures">',
        "<defs>"
        '<pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">'
        f'<path d="M26 0H0V26" fill="none" stroke="{BORDER}" stroke-width="0.6" '
        'opacity="0.45"/></pattern>'
        '<linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#0d1117" stop-opacity="1"/>'
        '<stop offset="1" stop-color="#0d1117" stop-opacity="0"/></linearGradient>'
        "</defs>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="16" fill="{BACKGROUND}"/>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="16" fill="url(#grid)"/>',
        f'<rect width="{WIDTH * 0.62:.0f}" height="{HEIGHT}" rx="16" fill="url(#fade)"/>',
        f'<text x="58" y="96" fill="{ACCENT}" font-size="13" font-weight="600" '
        f'letter-spacing="2.4" font-family="{FONT}">{EYEBROW}</text>',
        f'<text x="56" y="168" fill="{TITLE}" font-size="62" font-weight="700" '
        f'letter-spacing="-1" font-family="{FONT}">{HEADLINE}</text>',
        f'<text x="58" y="206" fill="{MUTED}" font-size="19" '
        f'font-family="{FONT}">{SUBLINE}</text>',
    ]

    x = 58
    for chip in CHIPS:
        chip_w = 15 + len(chip) * 7.6
        out.append(
            f'<rect x="{x:.0f}" y="236" width="{chip_w:.0f}" height="28" rx="14" '
            f'fill="#161b22" stroke="{BORDER}"/>'
        )
        out.append(
            f'<text x="{x + chip_w / 2:.0f}" y="255" fill="{MUTED}" font-size="12.5" '
            f'text-anchor="middle" font-family="{FONT}">{chip}</text>'
        )
        x += chip_w + 8

    for cx, cy, radius, opacity in signal_arcs((855, 170)):
        out.append(
            f'<path d="M{cx} {cy - radius} A {radius} {radius} 0 0 1 {cx} {cy + radius}" '
            f'fill="none" stroke="{ACCENT}" stroke-width="2.4" stroke-linecap="round" '
            f'opacity="{opacity:.2f}"/>'
        )

    for bone in bones:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{px:.1f} {py:.1f}" for i, (px, py) in enumerate(bone)
        )
        out.append(
            f'<path d="{path}" fill="none" stroke="{BONE}" stroke-width="3.4" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    for bone in bones:
        for px, py in bone:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.4" fill="{JOINT}"/>')

    for (x1, y1), (x2, y2) in arms:
        out.append(
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="{TITLE}" stroke-width="4" stroke-linecap="round"/>'
        )
    for mx, my, rx, ry in propellers:
        out.append(
            f'<ellipse cx="{mx:.0f}" cy="{my - 8:.0f}" rx="{rx}" ry="{ry}" '
            f'fill="{ACCENT}" opacity="0.35"/>'
        )
    for mx, my, radius in motors:
        out.append(
            f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="{radius}" fill="#161b22" '
            f'stroke="{TITLE}" stroke-width="3"/>'
        )
    out.append(
        f'<rect x="{bx:.0f}" y="{by:.0f}" width="{bw}" height="{bh}" rx="14" '
        f'fill="#161b22" stroke="{TITLE}" stroke-width="4"/>'
    )
    out.append(
        f'<circle cx="{bx + bw - 16:.0f}" cy="{by + bh / 2:.0f}" r="7" fill="{JOINT}"/>'
    )
    out.append("</svg>")
    return "\n".join(out)


def render_png(path: Path) -> None:
    """Rough raster preview so the layout can be eyeballed without a browser."""
    from PIL import Image, ImageDraw, ImageFont

    def font(size, bold=False):
        for name in (("arialbd.ttf", "segoeuib.ttf") if bold else ("arial.ttf", "segoeui.ttf")):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((58, 82), EYEBROW, fill=ACCENT, font=font(13, True))
    draw.text((56, 112), HEADLINE, fill=TITLE, font=font(58, True))
    draw.text((58, 190), SUBLINE, fill=MUTED, font=font(19))

    x = 58
    for chip in CHIPS:
        chip_w = 15 + len(chip) * 7.6
        draw.rounded_rectangle([x, 236, x + chip_w, 264], 14, fill="#161b22", outline=BORDER)
        draw.text((x + chip_w / 2, 243), chip, fill=MUTED, font=font(12), anchor="ma")
        x += chip_w + 8

    for cx, cy, radius, _ in signal_arcs((855, 170)):
        box = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.arc(box, -90, 90, fill=ACCENT, width=3)

    for bone in place(hand_skeleton(OPEN_PALM), HAND_CENTRE, box=(190, 190)):
        draw.line([tuple(point) for point in bone], fill=BONE, width=3, joint="curve")
        for px, py in bone:
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=JOINT)

    arms, motors, propellers, body = drone_paths(DRONE_CENTRE)
    for start, end in arms:
        draw.line([start, end], fill=TITLE, width=4)
    for mx, my, rx, ry in propellers:
        draw.ellipse([mx - rx, my - 8 - ry, mx + rx, my - 8 + ry], fill="#2a4a6a")
    for mx, my, radius in motors:
        draw.ellipse([mx - radius, my - radius, mx + radius, my + radius],
                     fill="#161b22", outline=TITLE, width=3)
    bx, by, bw, bh = body
    draw.rounded_rectangle([bx, by, bx + bw, by + bh], 14, fill="#161b22", outline=TITLE, width=4)
    draw.ellipse([bx + bw - 23, by + bh / 2 - 7, bx + bw - 9, by + bh / 2 + 7], fill=JOINT)
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "media" / "banner.svg")
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
