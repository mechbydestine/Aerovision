"""Gesture flight: webcam reads the operator, the drone answers.

Two video streams run in one loop. The laptop webcam is the control surface
and the drone camera is the payload, which keeps hand tracking working even
when the drone is facing away from the operator.
"""

from __future__ import annotations

import time

import cv2

from .. import overlay
from ..config import Settings
from ..drone import Drone, LowBatteryError
from ..gestures import (
    MOTION_GESTURES,
    Gesture,
    GestureStabilizer,
    classify,
    describe,
)
from ..hands import HandTracker


def _print_vocabulary() -> None:
    print("Gesture vocabulary:")
    for gesture in Gesture:
        spec = describe(gesture)
        if spec is not None:
            print(f"  {spec.hand_shape:<16} {spec.action}")
    print("  ESC              quit (lands first)")


def run(settings: Settings, dry_run: bool = False) -> int:
    """Fly by gesture. With ``dry_run`` the commands are printed, not sent."""
    capture = cv2.VideoCapture(settings.tracking.camera_index)
    if not capture.isOpened():
        print(f"Could not open camera {settings.tracking.camera_index}.")
        return 1

    drone = Drone(settings)
    try:
        drone.connect()
    except Exception as exc:
        print(f"Could not reach the drone: {exc}")
        capture.release()
        return 1

    flight = settings.flight
    stabilizer = GestureStabilizer(
        window=flight.vote_window,
        min_votes=flight.min_votes,
        cooldown=flight.command_cooldown,
    )
    _print_vocabulary()
    last_command = "-"

    with drone, HandTracker(settings.tracking) as tracker:
        while True:
            ok, operator_view = capture.read()
            if not ok:
                print("Webcam stopped delivering frames, landing.")
                break
            if settings.tracking.mirror_webcam:
                operator_view = cv2.flip(operator_view, 1)

            hands = tracker.detect(operator_view)
            gesture = Gesture.NONE
            if hands:
                overlay.draw_hand(operator_view, hands[0])
                gesture = classify(hands[0])

            drone_view = drone.read_frame()
            if drone_view is not None:
                drone.write_frame(drone_view)

            confirmed = stabilizer.update(gesture, time.monotonic())
            if confirmed is not None:
                last_command = str(confirmed)
                print(f"Command: {last_command}")
                if not dry_run:
                    _apply(drone, confirmed, drone_view)

            overlay.draw_panel(
                operator_view,
                (
                    f"Gesture   {gesture}",
                    f"Command   {last_command}",
                    f"Battery   {drone.battery}%",
                    f"Flying    {drone.is_flying}   Recording {drone.is_recording}",
                    "ESC  land and quit",
                ),
            )
            cv2.imshow("AeroVision - operator", operator_view)
            if drone_view is not None:
                if drone.is_recording:
                    overlay.draw_recording_dot(drone_view)
                cv2.imshow("AeroVision - drone camera", drone_view)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    capture.release()
    cv2.destroyAllWindows()
    return 0


def _apply(drone: Drone, gesture: Gesture, drone_view) -> None:
    """Carry out one confirmed command."""
    if gesture is Gesture.TAKEOFF:
        try:
            drone.takeoff()
        except LowBatteryError as exc:
            print(exc)
    elif gesture is Gesture.LAND:
        drone.land()
    elif gesture is Gesture.PHOTO and drone_view is not None:
        drone.save_photo(drone_view)
    elif gesture is Gesture.RECORD and drone_view is not None:
        drone.toggle_recording(drone_view)
    elif gesture in MOTION_GESTURES:
        drone.nudge(gesture)
