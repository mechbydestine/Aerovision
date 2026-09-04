"""Connection and telemetry check: run this before every flying session."""

from __future__ import annotations

import time

from ..config import Settings
from ..drone import Drone


def run(settings: Settings, hop: bool = False) -> int:
    """Report telemetry and, with ``hop``, fly a short take-off/land cycle."""
    drone = Drone(settings)
    try:
        battery = drone.connect(video=False)
    except Exception as exc:
        print(f"Could not reach the drone: {exc}")
        print("Check that you are joined to the TELLO-XXXXXX Wi-Fi network.")
        return 1

    with drone:
        tello = drone.tello
        print(f"  Battery      {battery}%")
        print(f"  Temperature  {tello.get_temperature():.0f} C")
        print(f"  Height       {tello.get_height()} cm")
        print(f"  Wi-Fi SNR    {tello.query_wifi_signal_noise_ratio()}")
        print(f"  Flight time  {tello.get_flight_time()} s")

        if hop:
            print("Hop test: taking off for 5 seconds ...")
            drone.takeoff()
            time.sleep(5)
    print("Preflight complete.")
    return 0
