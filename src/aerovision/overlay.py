"""On-screen heads-up display: hand skeleton, telemetry panel, key hints."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .gestures import HAND_CONNECTIONS, Hand

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
AMBER = (0, 200, 255)
WHITE = (255, 255, 255)
RED = (60, 60, 255)


def draw_hand(frame: np.ndarray, hand: Hand, colour: tuple[int, int, int] = GREEN) -> None:
    """Draw the 21-point skeleton over a frame, in place."""
    height, width = frame.shape[:2]
    points = [(int(point.x * width), int(point.y * height)) for point in hand]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], colour, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, CYAN, -1, cv2.LINE_AA)


def draw_panel(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: tuple[int, int] = (14, 14),
    scale: float = 0.6,
    opacity: float = 0.55,
) -> None:
    """Draw text on a translucent panel so it stays readable over video."""
    if not lines:
        return
    line_height = int(30 * scale / 0.6)
    width = int(max(len(line) for line in lines) * 11 * scale) + 24
    height = line_height * len(lines) + 18
    x, y = origin

    panel = frame.copy()
    cv2.rectangle(panel, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(panel, opacity, frame, 1 - opacity, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 70, 70), 1)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + 12, y + 24 + index * line_height),
            FONT,
            scale,
            WHITE,
            1,
            cv2.LINE_AA,
        )


def draw_status(frame: np.ndarray, text: str, colour: tuple[int, int, int] = GREEN) -> None:
    """Draw a single large status line along the bottom of the frame."""
    height = frame.shape[0]
    cv2.putText(frame, text, (16, height - 20), FONT, 0.9, colour, 2, cv2.LINE_AA)


def draw_recording_dot(frame: np.ndarray) -> None:
    """Mark a frame that is being written to disk."""
    cv2.circle(frame, (frame.shape[1] - 30, 30), 9, RED, -1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (frame.shape[1] - 90, 37), FONT, 0.7, RED, 2, cv2.LINE_AA)
