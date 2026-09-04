"""Synthetic hands so the gesture layer can be tested without a camera.

The builder produces the same 21-point layout MediaPipe emits, which lets the
tests exercise the real classifier instead of a stand-in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from aerovision.gestures import FingerPattern

# Knuckle (MCP) positions of the four fingers in a canonical upright hand.
_FINGER_MCPS = {
    "index": (0.44, 0.62),
    "middle": (0.50, 0.60),
    "ring": (0.56, 0.62),
    "pinky": (0.61, 0.66),
}
_FINGER_ORDER = ("index", "middle", "ring", "pinky")
# Landmark ids per finger: (mcp, pip, dip, tip).
_FINGER_IDS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_EXTENDED_OFFSETS = (-0.10, -0.17, -0.23)
_CURLED_OFFSETS = (-0.08, -0.06, -0.01)


@dataclass
class FakeLandmark:
    """Stand-in for a MediaPipe normalised landmark."""

    x: float
    y: float
    z: float = 0.0


def make_hand(pattern: FingerPattern, rotation_deg: float = 0.0) -> list[FakeLandmark]:
    """Build a 21-landmark hand matching ``pattern``.

    ``rotation_deg`` rotates the whole hand about the frame centre, which is
    how the tests check that classification does not depend on wrist angle.
    A 180 degree rotation of a thumbs-up is a thumbs-down.
    """
    thumb_out, *fingers = pattern
    points: dict[int, tuple[float, float]] = {
        0: (0.50, 0.95),  # wrist
        1: (0.42, 0.90),  # thumb CMC
        2: (0.36, 0.84),  # thumb MCP
        3: (0.32, 0.78),  # thumb IP
        4: (0.22, 0.70) if thumb_out else (0.45, 0.72),  # thumb tip
    }

    for name, extended in zip(_FINGER_ORDER, fingers, strict=True):
        mcp_x, mcp_y = _FINGER_MCPS[name]
        mcp_id, *joint_ids = _FINGER_IDS[name]
        points[mcp_id] = (mcp_x, mcp_y)
        offsets = _EXTENDED_OFFSETS if extended else _CURLED_OFFSETS
        for joint_id, offset in zip(joint_ids, offsets, strict=True):
            points[joint_id] = (mcp_x, mcp_y + offset)

    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    hand = []
    for index in range(21):
        x, y = points[index]
        dx, dy = x - 0.5, y - 0.5
        hand.append(
            FakeLandmark(0.5 + dx * cos_a - dy * sin_a, 0.5 + dx * sin_a + dy * cos_a)
        )
    return hand
