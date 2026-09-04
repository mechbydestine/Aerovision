"""Tests for the vote-and-cooldown gate that guards every flight command."""

from __future__ import annotations

from aerovision.gestures import Gesture, GestureStabilizer


def feed(stabilizer, gesture, times):
    """Push one gesture across a list of timestamps, returning fired commands."""
    return [fired for t in times if (fired := stabilizer.update(gesture, t)) is not None]


def test_a_single_frame_never_fires_a_command():
    stabilizer = GestureStabilizer(window=0.7, min_votes=4, cooldown=2.0)
    assert stabilizer.update(Gesture.LAND, 0.0) is None


def test_command_fires_once_enough_votes_land_inside_the_window():
    stabilizer = GestureStabilizer(window=0.7, min_votes=4, cooldown=2.0)
    fired = feed(stabilizer, Gesture.TAKEOFF, [0.0, 0.1, 0.2, 0.3])
    assert fired == [Gesture.TAKEOFF]


def test_votes_older_than_the_window_are_discarded():
    stabilizer = GestureStabilizer(window=0.5, min_votes=4, cooldown=2.0)
    assert feed(stabilizer, Gesture.LAND, [0.0, 0.4, 0.9, 1.4]) == []


def test_cooldown_blocks_a_repeat_command():
    stabilizer = GestureStabilizer(window=1.0, min_votes=2, cooldown=2.0)
    assert stabilizer.update(Gesture.FORWARD, 0.0) is None
    assert stabilizer.update(Gesture.FORWARD, 0.1) is Gesture.FORWARD
    assert feed(stabilizer, Gesture.FORWARD, [0.2, 0.3, 0.4]) == []
    assert feed(stabilizer, Gesture.FORWARD, [2.2, 2.3]) == [Gesture.FORWARD]


def test_a_stray_frame_cannot_outvote_the_held_gesture():
    """A hand passing through a fist must not land a flying drone."""
    stabilizer = GestureStabilizer(window=1.0, min_votes=4, cooldown=2.0)
    stream = [
        (Gesture.TAKEOFF, 0.0),
        (Gesture.LAND, 0.1),
        (Gesture.TAKEOFF, 0.2),
        (Gesture.NONE, 0.3),
        (Gesture.TAKEOFF, 0.4),
        (Gesture.TAKEOFF, 0.5),
    ]
    fired = [f for g, t in stream if (f := stabilizer.update(g, t)) is not None]
    assert fired == [Gesture.TAKEOFF]


def test_empty_frames_alone_never_fire():
    stabilizer = GestureStabilizer(window=1.0, min_votes=2, cooldown=0.0)
    assert feed(stabilizer, Gesture.NONE, [0.0, 0.1, 0.2, 0.3]) == []


def test_reset_clears_pending_votes():
    stabilizer = GestureStabilizer(window=1.0, min_votes=3, cooldown=0.0)
    stabilizer.update(Gesture.PHOTO, 0.0)
    stabilizer.update(Gesture.PHOTO, 0.1)
    stabilizer.reset()
    assert stabilizer.update(Gesture.PHOTO, 0.2) is None
