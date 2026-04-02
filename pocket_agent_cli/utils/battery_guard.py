"""Battery guard for macOS — prevents benchmark runs on low battery.

When battery is below threshold, waits for it to charge before proceeding.
This prevents CPU/GPU throttling from power saving that skews measurements.
"""

import re
import subprocess
import time
import platform


def get_battery_level() -> dict:
    """Get current battery status on macOS.

    Returns:
        Dict with 'percent' (int), 'charging' (bool), 'ac_power' (bool).
        Returns None values on non-macOS or if battery info unavailable.
    """
    if platform.system() != "Darwin":
        return {"percent": 100, "charging": False, "ac_power": True}

    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
        )
        output = result.stdout

        ac_power = "AC Power" in output
        match = re.search(r"(\d+)%", output)
        percent = int(match.group(1)) if match else None
        charging = "charging" in output.lower() and "not charging" not in output.lower()

        return {"percent": percent, "charging": charging, "ac_power": ac_power}
    except Exception:
        return {"percent": None, "charging": False, "ac_power": False}


def wait_for_battery(min_percent: int = 90, check_interval: int = 60) -> None:
    """Block until battery is at or above min_percent.

    Args:
        min_percent: Minimum battery level to proceed (default 90%).
        check_interval: Seconds between checks (default 60).
    """
    if platform.system() != "Darwin":
        return

    while True:
        status = get_battery_level()
        pct = status["percent"]

        if pct is None:
            print("[battery] Could not read battery level, proceeding")
            return

        if pct >= min_percent:
            print(f"[battery] {pct}% — OK (threshold: {min_percent}%)")
            return

        if not status["ac_power"]:
            print(f"[battery] {pct}% — BELOW {min_percent}% and NOT plugged in! Plug in charger.")
        else:
            print(f"[battery] {pct}% — charging, waiting for {min_percent}%...")

        time.sleep(check_interval)
