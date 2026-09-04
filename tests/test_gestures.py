"""Tests for the gesture vocabulary and classifier."""

from __future__ import annotations

import pytest

from aerovision.gestures import (
    GESTURE_SPECS,
    Gesture,
    classify,
    describe,
    finger_pattern,
    thumb_extended,
)
from synthetic import make_hand

# Thumb-only shapes share one pattern and are separated by thumb direction,
# so they are checked in their own test rather than in the table sweep.
_TABLE_CASES = [
    (spec.gesture, pattern)
    for spec in GESTURE_SPECS
    if spec.gesture not in (Gesture.UP, Gesture.DOWN)
    for pattern in spec.patterns
]


@pytest.mark.parametrize("expected, pattern", _TABLE_CASES, ids=lambda value: str(value))
def test_every_documented_shape_classifies_to_its_gesture(expected, pattern):
    assert classify(make_hand(pattern)) is expected


def test_finger_pattern_round_trips_through_the_builder():
    pattern = (False, True, True, False, False)
    assert finger_pattern(make_hand(pattern)) == pattern


def test_thumb_only_shape_resolves_by_direction():
    thumbs_up = make_hand((True, False, False, False, False))
    assert classify(thumbs_up) is Gesture.UP
    thumbs_down = make_hand((True, False, False, False, False), rotation_deg=180)
    assert classify(thumbs_down) is Gesture.DOWN


@pytest.mark.parametrize("rotation", [-40, -20, 20, 40])
def test_classification_is_stable_when_the_wrist_tilts(rotation):
    """Finger extension is measured from the wrist, so tilt must not matter."""
    for pattern, expected in (
        ((False, True, True, True, True), Gesture.TAKEOFF),
        ((False, False, False, False, False), Gesture.LAND),
        ((False, True, False, False, False), Gesture.FORWARD),
        ((False, True, True, False, False), Gesture.PHOTO),
    ):
        assert classify(make_hand(pattern, rotation_deg=rotation)) is expected


def test_unmapped_shape_is_ignored():
    """A ring finger on its own means nothing, so the drone must not move."""
    assert classify(make_hand((False, False, False, True, False))) is Gesture.NONE


def test_tucked_thumb_is_not_counted_as_extended():
    assert not thumb_extended(make_hand((False, False, False, False, False)))
    assert thumb_extended(make_hand((True, False, False, False, False)))


def test_vocabulary_has_no_ambiguous_patterns():
    """Two gestures sharing a finger pattern would make commands unpredictable."""
    seen: dict[tuple[bool, ...], Gesture] = {}
    for spec in GESTURE_SPECS:
        for pattern in spec.patterns:
            if spec.gesture in (Gesture.UP, Gesture.DOWN):
                continue
            assert pattern not in seen, f"{spec.gesture} collides with {seen.get(pattern)}"
            seen[pattern] = spec.gesture


def test_every_gesture_except_none_is_documented():
    for gesture in Gesture:
        if gesture is Gesture.NONE:
            continue
        spec = describe(gesture)
        assert spec is not None and spec.action
