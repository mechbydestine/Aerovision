"""Webcam-only gesture debugger: practise the vocabulary without flying."""

from __future__ import annotations

import time

import cv2

from .. import overlay
from ..config import Settings
from ..gestures import Gesture, GestureStabilizer, classify, describe, finger_pattern
from ..hands import HandTracker

_FINGER_LABELS = ("thumb", "index", "middle", "ring", "pinky")


def run(settings: Settings) -> int:
    """Stream the webcam, drawing the detected hand, pattern and command."""
    capture = cv2.VideoCapture(settings.tracking.camera_index)
    if not capture.isOpened():
        print(f"Could not open camera {settings.tracking.camera_index}.")
        return 1

    flight = settings.flight
    stabilizer = GestureStabilizer(
        window=flight.vote_window,
        min_votes=flight.min_votes,
        cooldown=flight.command_cooldown,
    )
    last_command = "-"

    with HandTracker(settings.tracking) as tracker:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera stopped delivering frames.")
                break
            if settings.tracking.mirror_webcam:
                frame = cv2.flip(frame, 1)

            hands = tracker.detect(frame)
            gesture = Gesture.NONE
            pattern_text = "-"
            if hands:
                overlay.draw_hand(frame, hands[0])
                gesture = classify(hands[0])
                raised = [
                    label
                    for label, up in zip(_FINGER_LABELS, finger_pattern(hands[0]), strict=True)
                    if up
                ]
                pattern_text = "+".join(raised) if raised else "none"

            confirmed = stabilizer.update(gesture, time.monotonic())
            if confirmed is not None:
                spec = describe(confirmed)
                last_command = f"{confirmed} ({spec.action})" if spec else str(confirmed)
                print(f"Command: {last_command}")

            overlay.draw_panel(
                frame,
                (
                    f"Gesture   {gesture}",
                    f"Fingers   {pattern_text}",
                    f"Confirmed {last_command}",
                    "No drone connected - practice mode",
                    "ESC  quit",
                ),
            )
            cv2.imshow("AeroVision - gesture check", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    capture.release()
    cv2.destroyAllWindows()
    return 0
