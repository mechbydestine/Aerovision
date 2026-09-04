"""Tunable settings for tracking, flight and capture.

Every magic number in AeroVision lives here so that flight behaviour can be
tuned without touching the control loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Official MediaPipe hand landmark model (downloaded on first run).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def project_root() -> Path:
    """Return the AeroVision checkout root (override with ``AEROVISION_HOME``)."""
    override = os.environ.get("AEROVISION_HOME")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").exists():
        return root
    return Path.cwd()


@dataclass(frozen=True)
class TrackingSettings:
    """MediaPipe hand-landmarker configuration."""

    model_path: Path = field(
        default_factory=lambda: project_root() / "models" / "hand_landmarker.task"
    )
    num_hands: int = 1
    min_detection_confidence: float = 0.3
    min_presence_confidence: float = 0.3
    min_tracking_confidence: float = 0.3
    #: Webcam index used for the gesture camera.
    camera_index: int = 0
    #: Mirror the webcam so the operator sees themselves as in a mirror.
    mirror_webcam: bool = True


@dataclass(frozen=True)
class FlightSettings:
    """Flight envelope and command timing."""

    #: RC stick magnitude (-100..100) used for a nudge.
    speed: int = 30
    #: How long a single nudge is held before the drone returns to hover.
    nudge_seconds: float = 0.35
    #: Minimum delay between two accepted gesture commands.
    command_cooldown: float = 2.0
    #: Sliding window in which gesture votes are collected.
    vote_window: float = 0.7
    #: Votes required inside the window before a command is trusted.
    min_votes: int = 4
    #: Refuse to take off below this battery percentage.
    min_takeoff_battery: int = 15


@dataclass(frozen=True)
class CaptureSettings:
    """Where photos and recordings are written."""

    directory: Path = field(default_factory=lambda: project_root() / "captures")
    video_fps: float = 20.0
    video_codec: str = "XVID"


@dataclass(frozen=True)
class Settings:
    """Top-level settings bundle passed through the application."""

    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    flight: FlightSettings = field(default_factory=FlightSettings)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
