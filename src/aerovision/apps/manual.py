"""Live drone view with optional keyboard flight control."""

from __future__ import annotations

import cv2

from .. import overlay
from ..config import Settings
from ..drone import Drone, LowBatteryError
from ..gestures import Gesture

#: Key to flight command. Layout mirrors WASD with R/F for altitude.
KEY_COMMANDS: dict[str, Gesture] = {
    "w": Gesture.FORWARD,
    "s": Gesture.BACKWARD,
    "r": Gesture.UP,
    "f": Gesture.DOWN,
    "q": Gesture.ROTATE_LEFT,
    "e": Gesture.ROTATE_RIGHT,
}
STRAFE_KEYS = {"a": -1, "d": 1}

VIEW_ONLY_HELP = ("p  photo", "ESC  quit")
CONTROL_HELP = (
    "t/l  takeoff / land",
    "w/s  forward / back",
    "a/d  strafe left / right",
    "r/f  up / down",
    "q/e  yaw left / right",
    "p  photo    ESC  quit",
)


def run(settings: Settings, view_only: bool = False) -> int:
    """Show the drone feed; with ``view_only`` disabled, fly it from the keyboard."""
    drone = Drone(settings)
    try:
        drone.connect()
    except Exception as exc:
        print(f"Could not reach the drone: {exc}")
        return 1

    window = "AeroVision - drone camera"
    help_lines = VIEW_ONLY_HELP if view_only else CONTROL_HELP

    with drone:
        while True:
            frame = drone.read_frame()
            if frame is None:
                continue

            overlay.draw_panel(
                frame,
                (f"Battery {drone.battery}%", f"Flying {drone.is_flying}", *help_lines),
            )
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("p"):
                drone.save_photo(frame)
                continue
            if view_only or key == 255:
                continue

            char = chr(key)
            if char == "t":
                try:
                    drone.takeoff()
                except LowBatteryError as exc:
                    print(exc)
            elif char == "l":
                drone.land()
            elif char in STRAFE_KEYS and drone.is_flying:
                speed = settings.flight.speed * STRAFE_KEYS[char]
                drone.tello.send_rc_control(speed, 0, 0, 0)
                cv2.waitKey(int(settings.flight.nudge_seconds * 1000))
                drone.tello.send_rc_control(0, 0, 0, 0)
            elif char in KEY_COMMANDS:
                drone.nudge(KEY_COMMANDS[char])

    cv2.destroyAllWindows()
    return 0
