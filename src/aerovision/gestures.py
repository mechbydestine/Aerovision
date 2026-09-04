"""Hand-shape vocabulary, classifier and vote-based command stabiliser.

This module is deliberately free of OpenCV and MediaPipe imports: it works on
any object exposing normalised ``.x`` / ``.y`` landmark coordinates, which
makes the whole gesture layer unit-testable without a camera or a drone.

Landmark indices follow the MediaPipe hand model (21 points, wrist = 0):
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from typing import Protocol

# --------------------------------------------------------------------------
# Landmark indices
# --------------------------------------------------------------------------
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

#: (tip, pip) pairs for the four non-thumb fingers, in index..pinky order.
FINGER_JOINTS: tuple[tuple[int, int], ...] = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)

#: Bone list used for drawing the skeleton overlay.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class Landmark(Protocol):
    """Minimal structural type for a MediaPipe normalised landmark."""

    x: float
    y: float


Hand = Sequence[Landmark]
#: (thumb, index, middle, ring, pinky) extension flags.
FingerPattern = tuple[bool, bool, bool, bool, bool]


class Gesture(Enum):
    """Every command the drone understands."""

    NONE = "none"
    TAKEOFF = "takeoff"
    LAND = "land"
    FORWARD = "forward"
    BACKWARD = "backward"
    UP = "up"
    DOWN = "down"
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"
    PHOTO = "photo"
    RECORD = "record"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value.replace("_", " ").upper()


#: Commands that translate directly into RC stick input.
MOTION_GESTURES = frozenset(
    {
        Gesture.FORWARD,
        Gesture.BACKWARD,
        Gesture.UP,
        Gesture.DOWN,
        Gesture.ROTATE_LEFT,
        Gesture.ROTATE_RIGHT,
    }
)


@dataclass(frozen=True)
class GestureSpec:
    """One row of the gesture vocabulary.

    ``patterns`` holds every finger combination that maps to the gesture, so
    the table doubles as documentation and as the lookup source used by the
    classifier, the docs and the gesture chart.
    """

    gesture: Gesture
    hand_shape: str
    action: str
    patterns: tuple[FingerPattern, ...]


# Finger flags are ordered (thumb, index, middle, ring, pinky).
GESTURE_SPECS: tuple[GestureSpec, ...] = (
    GestureSpec(Gesture.TAKEOFF, "Open palm", "Take off, then hover",
                ((True, True, True, True, True), (False, True, True, True, True))),
    GestureSpec(Gesture.LAND, "Fist", "Land",
                ((False, False, False, False, False),)),
    GestureSpec(Gesture.FORWARD, "Index finger", "Nudge forward",
                ((False, True, False, False, False), (True, True, False, False, False))),
    GestureSpec(Gesture.BACKWARD, "Index + pinky", "Nudge backward",
                ((False, True, False, False, True), (True, True, False, False, True))),
    GestureSpec(Gesture.UP, "Thumb up", "Climb",
                ((True, False, False, False, False),)),
    GestureSpec(Gesture.DOWN, "Thumb down", "Descend",
                ((True, False, False, False, False),)),
    GestureSpec(Gesture.ROTATE_LEFT, "Pinky", "Yaw left",
                ((False, False, False, False, True),)),
    GestureSpec(Gesture.ROTATE_RIGHT, "Thumb + pinky", "Yaw right",
                ((True, False, False, False, True),)),
    GestureSpec(Gesture.PHOTO, "Peace sign", "Save a photo",
                ((False, True, True, False, False), (True, True, True, False, False))),
    GestureSpec(Gesture.RECORD, "Three fingers", "Start / stop recording",
                ((False, True, True, True, False), (True, True, True, True, False))),
)

#: Thumb-only shapes are split by thumb direction rather than by pattern.
_THUMB_ONLY: FingerPattern = (True, False, False, False, False)


def _build_lookup() -> dict[FingerPattern, Gesture]:
    """Index the vocabulary by finger pattern, rejecting ambiguous rows."""
    lookup: dict[FingerPattern, Gesture] = {}
    for spec in GESTURE_SPECS:
        for pattern in spec.patterns:
            if pattern == _THUMB_ONLY:
                continue  # resolved at runtime by thumb direction
            if pattern in lookup:  # pragma: no cover - guards the table
                raise ValueError(f"Ambiguous finger pattern {pattern} in GESTURE_SPECS")
            lookup[pattern] = spec.gesture
    return lookup


_PATTERN_TO_GESTURE = _build_lookup()


# --------------------------------------------------------------------------
# Finger geometry
# --------------------------------------------------------------------------
def _distance(a: Landmark, b: Landmark) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def finger_extended(hand: Hand, tip: int, pip: int, margin: float = 1.05) -> bool:
    """True when a finger is straightened out.

    Comparing the tip against the middle joint on the y axis alone only works
    while the hand stays upright, which breaks the moment the operator tilts
    their wrist or signals a thumbs-down. Measuring both joints from the wrist
    instead is orientation invariant: a curled finger folds its tip back
    towards the palm, so the tip ends up closer to the wrist than its own
    middle joint.
    """
    return _distance(hand[tip], hand[WRIST]) > _distance(hand[pip], hand[WRIST]) * margin


def thumb_extended(hand: Hand, margin: float = 1.1) -> bool:
    """True when the thumb points away from the palm.

    The thumb swings sideways instead of curling, so the usual tip-above-joint
    test is unreliable. Compare instead how far the tip and the
    interphalangeal joint sit from the index knuckle: a tucked thumb pulls its
    tip closer to that knuckle than the joint it hangs from.
    """
    tip_spread = _distance(hand[THUMB_TIP], hand[INDEX_MCP])
    joint_spread = _distance(hand[THUMB_IP], hand[INDEX_MCP])
    return tip_spread > joint_spread * margin


def thumb_points_up(hand: Hand) -> bool:
    """True for a thumbs-up, False for a thumbs-down."""
    return hand[THUMB_TIP].y < hand[WRIST].y


def finger_pattern(hand: Hand) -> FingerPattern:
    """Reduce 21 landmarks to five extension flags."""
    fingers = tuple(finger_extended(hand, tip, pip) for tip, pip in FINGER_JOINTS)
    return (thumb_extended(hand), *fingers)  # type: ignore[return-value]


def classify(hand: Hand) -> Gesture:
    """Map a hand to a Gesture, returning Gesture.NONE when unrecognised."""
    pattern = finger_pattern(hand)
    if pattern == _THUMB_ONLY:
        return Gesture.UP if thumb_points_up(hand) else Gesture.DOWN
    return _PATTERN_TO_GESTURE.get(pattern, Gesture.NONE)


def describe(gesture: Gesture) -> GestureSpec | None:
    """Return the vocabulary row for a gesture, if it has one."""
    for spec in GESTURE_SPECS:
        if spec.gesture is gesture:
            return spec
    return None


# --------------------------------------------------------------------------
# Stabiliser
# --------------------------------------------------------------------------
@dataclass
class GestureStabilizer:
    """Turn noisy per-frame guesses into a small number of trusted commands.

    A gesture only fires when it wins the votes collected inside ``window``
    and the cooldown since the previous command has elapsed. That is what
    stops a hand passing through a fist on its way to an open palm from
    landing the drone mid-flight.
    """

    window: float = 0.7
    min_votes: int = 4
    cooldown: float = 2.0
    _votes: list[tuple[Gesture, float]] = field(default_factory=list, repr=False)
    _last_fired_at: float = field(default=float("-inf"), repr=False)

    def update(self, gesture: Gesture, now: float) -> Gesture | None:
        """Record a frame's gesture and return a command when one is confirmed."""
        self._votes = [(g, t) for g, t in self._votes if now - t <= self.window]
        if gesture is not Gesture.NONE:
            self._votes.append((gesture, now))
        if not self._votes:
            return None

        counts: dict[Gesture, int] = {}
        for voted, _ in self._votes:
            counts[voted] = counts.get(voted, 0) + 1
        winner = max(counts, key=lambda key: counts[key])

        if counts[winner] < self.min_votes:
            return None
        if now - self._last_fired_at < self.cooldown:
            return None

        self._last_fired_at = now
        self._votes.clear()
        return winner

    def reset(self) -> None:
        """Forget pending votes, for example after a mode change."""
        self._votes.clear()
