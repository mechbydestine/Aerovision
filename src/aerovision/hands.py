"""MediaPipe hand tracking, wrapped so the rest of the app stays framework free."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import MODEL_URL, TrackingSettings
from .gestures import Hand


class ModelUnavailableError(RuntimeError):
    """Raised when the hand landmark model is missing and cannot be fetched."""


def ensure_model(path: Path, url: str = MODEL_URL) -> Path:
    """Return the landmark model, downloading it on first run.

    The 7.8 MB model is a build artifact rather than source, so it is fetched
    from Google instead of being committed to the repository.
    """
    if path.exists():
        return path

    print(f"Hand landmark model not found, downloading to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise ModelUnavailableError(
            f"Could not download the hand landmark model from {url}.\n"
            f"Download it manually and save it as {path}."
        ) from exc
    print("Model ready.")
    return path


class HandTracker:
    """Detect hand landmarks in a stream of BGR frames.

    MediaPipe's VIDEO mode requires strictly increasing millisecond
    timestamps; a fast webcam can deliver two frames inside the same
    millisecond, so the clock is clamped upwards instead of trusted directly.
    """

    def __init__(self, settings: TrackingSettings | None = None) -> None:
        self.settings = settings or TrackingSettings()
        self._landmarker: mp.tasks.vision.HandLandmarker | None = None
        self._started_at = time.monotonic()
        self._last_timestamp_ms = -1

    def __enter__(self) -> HandTracker:
        model_path = ensure_model(Path(self.settings.model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.settings.num_hands,
            min_hand_detection_confidence=self.settings.min_detection_confidence,
            min_hand_presence_confidence=self.settings.min_presence_confidence,
            min_tracking_confidence=self.settings.min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def _next_timestamp_ms(self) -> int:
        elapsed_ms = int((time.monotonic() - self._started_at) * 1000)
        self._last_timestamp_ms = max(elapsed_ms, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def detect(self, frame_bgr: np.ndarray) -> list[Hand]:
        """Return every hand found in a BGR frame, nearest detection first."""
        if self._landmarker is None:
            raise RuntimeError("HandTracker must be used as a context manager")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._next_timestamp_ms())
        return list(result.hand_landmarks or [])
