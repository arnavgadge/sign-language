"""
layer1/preprocessing.py
------------------------
Pure, deterministic transforms with no learned parameters. Every function
here takes a raw signal and returns a normalized one; same input always
produces the same output, so these need no train/test split and no model
file — only correctness tests (see tests/test_layer1.py).
"""
import time

from .config import TIME_PERIOD_BOUNDARIES, MOTION_REFERENCE_SPEED_PX


def normalize_word(word: str) -> str:
    """Case/whitespace normalization, matching the convention already used
    at the Layer 5 boundary (`intent/engine.py`) so both sides agree on the
    canonical word form."""
    return (word or "").upper().strip()


def bucket_time_period(epoch_ts: float = None) -> tuple:
    """
    Returns (hour: int, period: str) for the given epoch timestamp
    (local time). Defaults to now() if no timestamp is given.
    """
    if epoch_ts is None:
        epoch_ts = time.time()
    hour = time.localtime(epoch_ts).tm_hour

    for period, (start, end) in TIME_PERIOD_BOUNDARIES.items():
        if start <= hour < end:
            return hour, period
    return hour, "night"


def normalize_motion_speed(motion_speed_px: float,
                            reference: float = MOTION_REFERENCE_SPEED_PX) -> float:
    """
    Normalize a raw px/frame motion speed to a [0, 1] urgency value.
    Values at or above `reference` saturate to 1.0; negative input
    (should not occur, but defensively clipped) floors to 0.0.
    """
    if motion_speed_px is None or reference <= 0:
        return 0.0
    return max(0.0, min(1.0, motion_speed_px / reference))
