"""
layer1/config.py
-----------------
All Layer 1 hyperparameters and paths in one place, so every experiment
run can log exactly what configuration produced its results.

Numeric defaults for the face cascade and motion normalization intentionally
match the values already used by the legacy `intent/context_layers.py`
(audited in Phase 0) so Layer 1's raw signals stay consistent with what the
rest of the system has assumed so far. They are independent constants here,
not an import from that module, so Layer 1 has no dependency on Layer 5's
implementation.
"""
from dataclasses import dataclass, field
import os

RANDOM_SEED = 42

# ── Time-period bucketing (hour -> label) ──────────────────────────────────
TIME_PERIOD_BOUNDARIES = {
    "morning":   (6, 12),
    "afternoon": (12, 17),
    "evening":   (17, 21),
    # anything else -> "night"
}

# ── Motion urgency normalization ────────────────────────────────────────────
MOTION_REFERENCE_SPEED_PX = 50.0   # px/frame that normalizes to urgency = 1.0

# ── Face detector (Haar cascade) ────────────────────────────────────────────
FACE_CASCADE_SCALE_FACTOR = 1.1
FACE_CASCADE_MIN_NEIGHBORS = 4
FACE_CASCADE_MIN_SIZE = (60, 60)

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERACTIONS_DIR = os.path.join(_ROOT, "data", "interactions")
EVAL_FACES_DIR = os.path.join(_ROOT, "data", "eval", "faces")
EVAL_REPORTS_DIR = os.path.join(_ROOT, "evaluation", "reports")


@dataclass(frozen=True)
class Layer1Config:
    """Snapshot of the active configuration, for logging alongside results."""
    random_seed: int = RANDOM_SEED
    time_period_boundaries: dict = field(default_factory=lambda: dict(TIME_PERIOD_BOUNDARIES))
    motion_reference_speed_px: float = MOTION_REFERENCE_SPEED_PX
    face_cascade_scale_factor: float = FACE_CASCADE_SCALE_FACTOR
    face_cascade_min_neighbors: int = FACE_CASCADE_MIN_NEIGHBORS
    face_cascade_min_size: tuple = FACE_CASCADE_MIN_SIZE

    def to_dict(self) -> dict:
        return {
            "random_seed": self.random_seed,
            "time_period_boundaries": self.time_period_boundaries,
            "motion_reference_speed_px": self.motion_reference_speed_px,
            "face_cascade_scale_factor": self.face_cascade_scale_factor,
            "face_cascade_min_neighbors": self.face_cascade_min_neighbors,
            "face_cascade_min_size": list(self.face_cascade_min_size),
        }
