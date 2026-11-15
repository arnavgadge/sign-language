"""
layer1/schema.py
-----------------
Output schema for Layer 1 (Context / Input Acquisition).

Layer 1 emits *raw, normalized signals only* — it does not weight, score,
or interpret them. Interpretation (reliability weighting, decision-making)
is downstream work belonging to Layer 5, which is out of scope for Phase 2.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass(frozen=True)
class FaceSignal:
    """
    Output of the one learned/pre-trained component in Layer 1: an
    OpenCV Haar-cascade face detector. `face_count` and `face_present`
    are raw detector outputs, not an interpreted reliability score.
    """
    face_count: int
    face_present: bool
    detector_available: bool  # False if the cascade failed to load

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "FaceSignal":
        return FaceSignal(
            face_count=int(d["face_count"]),
            face_present=bool(d["face_present"]),
            detector_available=bool(d["detector_available"]),
        )


@dataclass(frozen=True)
class InteractionContext:
    """
    The complete Layer 1 output: a structured, normalized representation
    of "what is happening right now" that Layers 3 and 5 will later
    consume. Every field is either a direct passthrough of a raw signal
    or a deterministic normalization of one — no learned weighting.
    """
    session_id: str
    event_id: str
    timestamp: float          # epoch seconds, capture time
    user_id: str

    word_raw: str              # exactly as received
    word: str                  # normalized (upper + stripped)

    motion_speed_px: float     # raw px/frame, as received
    motion_urgency: float      # normalized to [0, 1]

    hour: int                  # 0-23, derived from timestamp
    time_period: str           # "morning" | "afternoon" | "evening" | "night"

    face: FaceSignal

    pipeline_latency_ms: float  # wall-clock time to build this context

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "InteractionContext":
        d = dict(d)
        d["face"] = FaceSignal.from_dict(d["face"])
        return InteractionContext(**d)

    @staticmethod
    def from_json(s: str) -> "InteractionContext":
        return InteractionContext.from_dict(json.loads(s))
