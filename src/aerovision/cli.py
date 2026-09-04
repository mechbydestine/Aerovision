"""Command line entry point: ``aerovision <command>``."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from . import __version__
from .config import CaptureSettings, FlightSettings, Settings, TrackingSettings
from .gestures import GESTURE_SPECS

COMMANDS = ("preflight", "camera", "keyboard", "handcheck", "fly", "gestures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerovision",
        description="Gesture and keyboard control for a Ryze Tello drone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  aerovision preflight          check battery and Wi-Fi link\n"
            "  aerovision handcheck          practise gestures, no drone needed\n"
            "  aerovision fly --dry-run      read gestures without arming motors\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"AeroVision {__version__}")
    parser.add_argument("--camera", type=int, default=None, metavar="N",
                        help="webcam index used for gesture input (default 0)")
    parser.add_argument("--speed", type=int, default=None, metavar="0-100",
                        help="RC stick magnitude for a nudge (default 30)")
    parser.add_argument("--cooldown", type=float, default=None, metavar="SECONDS",
                        help="minimum delay between accepted commands (default 2.0)")
    parser.add_argument("--model", type=Path, default=None, metavar="PATH",
                        help="path to hand_landmarker.task")
    parser.add_argument("--captures", type=Path, default=None, metavar="DIR",
                        help="where photos and recordings are written")

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    preflight = subparsers.add_parser("preflight", help="connection and battery check")
    preflight.add_argument("--hop", action="store_true", help="take off for five seconds")
    subparsers.add_parser("camera", help="live drone camera view")
    subparsers.add_parser("keyboard", help="fly with the keyboard")
    subparsers.add_parser("handcheck", help="practise gestures on the webcam only")
    fly = subparsers.add_parser("fly", help="fly with hand gestures")
    fly.add_argument("--dry-run", action="store_true",
                     help="recognise gestures but never command the drone")
    subparsers.add_parser("gestures", help="print the gesture vocabulary")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Apply command line overrides on top of the defaults in config.py."""
    tracking = TrackingSettings()
    flight = FlightSettings()
    capture = CaptureSettings()
    if args.camera is not None:
        tracking = dataclasses.replace(tracking, camera_index=args.camera)
    if args.model is not None:
        tracking = dataclasses.replace(tracking, model_path=args.model)
    if args.speed is not None:
        flight = dataclasses.replace(flight, speed=max(10, min(100, args.speed)))
    if args.cooldown is not None:
        flight = dataclasses.replace(flight, command_cooldown=args.cooldown)
    if args.captures is not None:
        capture = dataclasses.replace(capture, directory=args.captures)
    return Settings(tracking=tracking, flight=flight, capture=capture)


def print_gestures() -> int:
    """Print the vocabulary as a Markdown table (source of docs/GESTURES.md)."""
    print("| Hand shape | Command | What the drone does |")
    print("| --- | --- | --- |")
    for spec in GESTURE_SPECS:
        print(f"| {spec.hand_shape} | `{spec.gesture.value}` | {spec.action} |")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "gestures":
        return print_gestures()

    settings = settings_from_args(args)

    # Imported lazily so `gestures` and `--help` work without OpenCV installed.
    if args.command == "preflight":
        from .apps import preflight

        return preflight.run(settings, hop=args.hop)
    if args.command in ("camera", "keyboard"):
        from .apps import manual

        return manual.run(settings, view_only=args.command == "camera")
    if args.command == "handcheck":
        from .apps import handcheck

        return handcheck.run(settings)
    if args.command == "fly":
        from .apps import fly

        return fly.run(settings, dry_run=args.dry_run)
    raise AssertionError(f"unhandled command {args.command}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
