"""A safety-first wrapper around the Tello SDK.

Every flight script in this repository talks to the drone through this class,
so the battery gate, the hover-after-nudge rule and the guaranteed landing on
exit apply everywhere instead of being re-implemented per script.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from djitellopy import Tello

from .config import Settings
from .gestures import Gesture

#: Gesture to RC stick vector (left_right, forward_back, up_down, yaw).
RC_VECTORS: dict[Gesture, tuple[int, int, int, int]] = {
    Gesture.FORWARD: (0, 1, 0, 0),
    Gesture.BACKWARD: (0, -1, 0, 0),
    Gesture.UP: (0, 0, 1, 0),
    Gesture.DOWN: (0, 0, -1, 0),
    Gesture.ROTATE_LEFT: (0, 0, 0, -1),
    Gesture.ROTATE_RIGHT: (0, 0, 0, 1),
}


class LowBatteryError(RuntimeError):
    """Raised when the pack is too flat to take off safely."""


class Drone:
    """Connection, video and flight state for one Tello."""

    def __init__(self, settings: Settings | None = None, tello: Tello | None = None) -> None:
        self.settings = settings or Settings()
        self.tello = tello or Tello()
        self.is_flying = False
        self._frame_reader = None
        self._writer: cv2.VideoWriter | None = None
        self._recording_path: Path | None = None

    # -- connection ------------------------------------------------------
    def connect(self, video: bool = True) -> int:
        """Connect over Wi-Fi and return the battery percentage."""
        self.tello.connect()
        battery = self.tello.get_battery()
        print(f"Connected. Battery {battery}%  Temp {self.tello.get_temperature():.0f}C")
        if video:
            self.tello.streamoff()  # clear a stream left open by a crashed run
            self.tello.streamon()
            self._frame_reader = self.tello.get_frame_read()
            time.sleep(1.0)  # let the decoder fill its first frames
        return battery

    @property
    def battery(self) -> int:
        return self.tello.get_battery()

    def read_frame(self) -> np.ndarray | None:
        """Return the latest drone frame as BGR, or None if the stream is cold.

        djitellopy hands back RGB frames. Passing those straight to OpenCV is
        what tinted the early recordings blue, so the conversion happens once,
        here, rather than in each script.
        """
        if self._frame_reader is None:
            return None
        frame = self._frame_reader.frame
        if frame is None or frame.size == 0:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # -- flight ----------------------------------------------------------
    def takeoff(self) -> None:
        """Take off, refusing to launch on a pack that cannot land safely."""
        if self.is_flying:
            return
        battery = self.battery
        minimum = self.settings.flight.min_takeoff_battery
        if battery < minimum:
            raise LowBatteryError(f"Battery {battery}% is below the {minimum}% takeoff floor")
        self.tello.takeoff()
        self.tello.send_rc_control(0, 0, 0, 0)  # hold a stable hover
        self.is_flying = True

    def land(self) -> None:
        if not self.is_flying:
            return
        self.tello.send_rc_control(0, 0, 0, 0)
        self.tello.land()
        self.is_flying = False

    def nudge(self, gesture: Gesture) -> bool:
        """Apply a short stick input, then return to hover.

        Discrete nudges keep the drone predictable: a missed frame can never
        leave a stick held down, because the hover command always follows.
        """
        vector = RC_VECTORS.get(gesture)
        if vector is None or not self.is_flying:
            return False
        speed = self.settings.flight.speed
        left_right, forward_back, up_down, yaw = (axis * speed for axis in vector)
        self.tello.send_rc_control(left_right, forward_back, up_down, yaw)
        time.sleep(self.settings.flight.nudge_seconds)
        self.tello.send_rc_control(0, 0, 0, 0)
        return True

    # -- capture ---------------------------------------------------------
    def _capture_path(self, suffix: str) -> Path:
        directory = Path(self.settings.capture.directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"tello_{time.strftime('%Y%m%d_%H%M%S')}{suffix}"

    def save_photo(self, frame: np.ndarray) -> Path:
        path = self._capture_path(".jpg")
        cv2.imwrite(str(path), frame)
        print(f"Photo saved: {path}")
        return path

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def toggle_recording(self, frame: np.ndarray) -> bool:
        """Start or stop recording the drone feed, returning the new state."""
        if self._writer is None:
            height, width = frame.shape[:2]
            path = self._capture_path(".avi")
            fourcc = cv2.VideoWriter_fourcc(*self.settings.capture.video_codec)
            self._writer = cv2.VideoWriter(
                str(path), fourcc, self.settings.capture.video_fps, (width, height)
            )
            self._recording_path = path
            print(f"Recording started: {path}")
            return True
        self._stop_recording()
        return False

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def _stop_recording(self) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"Recording saved: {self._recording_path}")
            self._writer = None
            self._recording_path = None

    # -- teardown --------------------------------------------------------
    def close(self) -> None:
        """Land, close the stream and release files, whatever went wrong.

        Each step is isolated: a failure while stopping the video must never
        prevent the drone from landing.
        """
        for step in (self._stop_recording, self.land, self.tello.streamoff, self.tello.end):
            try:
                step()
            except Exception as exc:
                print(f"Cleanup warning ({step.__name__}): {exc}")

    def __enter__(self) -> Drone:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
