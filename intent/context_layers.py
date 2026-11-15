"""
Stage 2 — Four independent context layers.

Each layer returns a LayerOutput with:
  signal_scores  dict[intent_id -> float 0-1]  how strongly this layer supports each candidate
  reliability    float 0-1                      how much to trust this layer right now
  reliability_factors  dict                     what raised/lowered reliability (for debug)

Layer 1  Visual   – face presence, scene complexity, signing urgency (motion speed)
Layer 2  Temporal – conversation history, timing patterns
Layer 3  Device   – time of day, platform metadata (GPS unavailable on desktop)
Layer 4  Personal – per-user behavioral priors from UserModel
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .ontology import IntentCandidate
from .conversation import ConversationHistory
from .user_model import UserModel

# Layers below this reliability floor are excluded from resolution entirely
RELIABILITY_FLOOR = 0.05


@dataclass
class LayerOutput:
    name: str
    signal_scores: Dict[str, float]   # intent_id -> signal (0-1)
    reliability: float                 # layer reliability (0-1)
    reliability_factors: dict          # human-readable reasons for reliability value


# ── Layer 1: Visual Context ────────────────────────────────────────────────────

class VisualContextLayer:
    """
    Uses the same camera frame already captured for sign recognition.
    No additional hardware required.

    Reliability degradation triggers:
      - Face not visible (no face detected)
      - Multiple people in frame (ambiguous signers)
      - Face detection module unavailable
    """

    def __init__(self):
        self._cascade = None
        try:
            import cv2
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(path)
        except Exception:
            pass

    def compute(
        self,
        candidates: List[IntentCandidate],
        frame=None,
        motion_speed: float = 0.0,
    ) -> LayerOutput:
        signals = {c.id: c.base_probability for c in candidates}
        reliability_factors: dict = {}
        reliability = 0.35  # pessimistic default when frame is unavailable

        if frame is not None and self._cascade is not None:
            try:
                import cv2
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
                )
                face_count = len(faces)

                if face_count == 1:
                    reliability = 0.80
                    reliability_factors["face_visible"] = True
                elif face_count == 0:
                    reliability = 0.30
                    reliability_factors["face_not_visible"] = True
                else:
                    reliability = 0.40
                    reliability_factors["multiple_people"] = int(face_count)
            except Exception as exc:
                reliability = 0.30
                reliability_factors["detection_error"] = str(exc)
        else:
            reliability_factors["no_frame_provided"] = True

        # Motion urgency: normalise to 0-1 using a 50 px/frame reference speed
        urgency = min(1.0, motion_speed / 50.0) if motion_speed > 0 else 0.0
        reliability_factors["motion_speed"] = round(motion_speed, 2)

        for c in candidates:
            base = signals[c.id]
            if c.emergency and urgency > 0.55:
                signals[c.id] = min(1.0, base + urgency * 0.30)
            elif c.emergency and urgency < 0.15:
                signals[c.id] = max(0.0, base - 0.10)

        return LayerOutput(
            name="visual",
            signal_scores=signals,
            reliability=reliability,
            reliability_factors=reliability_factors,
        )


# ── Layer 2: Temporal Context ──────────────────────────────────────────────────

class TemporalContextLayer:
    """
    Reliability degradation triggers:
      - First sign of a new conversation (time_since_last > 300 s)
      - Long gap mid-conversation (> 60 s)
    """

    def compute(
        self,
        candidates: List[IntentCandidate],
        history: ConversationHistory,
    ) -> LayerOutput:
        signals = {c.id: c.base_probability for c in candidates}
        reliability_factors: dict = {}

        time_since_last = history.time_since_last()
        recent_words = history.recent_words(seconds=120)

        if time_since_last == float("inf") or time_since_last > 300:
            reliability = 0.20
            reliability_factors["fresh_conversation"] = True
        elif time_since_last > 60:
            reliability = 0.50
            reliability_factors["gap_seconds"] = round(time_since_last, 1)
        else:
            reliability = 0.85
            reliability_factors["active_conversation"] = True

        reliability_factors["recent_word_count"] = len(recent_words)

        if not recent_words:
            return LayerOutput(
                name="temporal",
                signal_scores=signals,
                reliability=reliability,
                reliability_factors=reliability_factors,
            )

        for c in candidates:
            triggers = sum(1 for t in c.context_triggers if t in recent_words)
            conflicts = sum(1 for t in c.context_conflicts if t in recent_words)
            delta = (triggers * 0.15) - (conflicts * 0.20)
            signals[c.id] = max(0.0, min(1.0, c.base_probability + delta))

        return LayerOutput(
            name="temporal",
            signal_scores=signals,
            reliability=reliability,
            reliability_factors=reliability_factors,
        )


# ── Layer 3: Device Context ────────────────────────────────────────────────────

class DeviceContextLayer:
    """
    Uses always-available device metadata (time of day).
    GPS is unavailable on desktop → reliability is moderate and location priors
    are skipped, which is reflected in the reliability score.

    Reliability degradation triggers:
      - GPS unavailable (always true on desktop)
    """

    def compute(self, candidates: List[IntentCandidate]) -> LayerOutput:
        signals = {c.id: c.base_probability for c in candidates}
        reliability_factors: dict = {}

        hour = time.localtime().tm_hour
        if 6 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 21:
            period = "evening"
        else:
            period = "night"

        reliability_factors["time_period"] = period
        reliability_factors["hour"] = hour

        # Apply ontology-specified time-period priors
        for c in candidates:
            adj = c.time_priors.get(period, 0.0)
            if adj:
                signals[c.id] = min(1.0, signals[c.id] + adj * 0.20)

        # GPS unavailable on desktop → cannot apply location priors
        reliability = 0.45
        reliability_factors["gps_available"] = False
        reliability_factors["location_priors_skipped"] = True

        return LayerOutput(
            name="device",
            signal_scores=signals,
            reliability=reliability,
            reliability_factors=reliability_factors,
        )


# ── Layer 4: Personal Context ──────────────────────────────────────────────────

class PersonalContextLayer:
    """
    Reliability degradation triggers:
      - New user (total_interactions < 3)
      - Insufficient word-specific history (< 5 samples for this word)
    """

    def compute(
        self,
        candidates: List[IntentCandidate],
        word: str,
        user_model: UserModel,
    ) -> LayerOutput:
        signals = {c.id: c.base_probability for c in candidates}
        reliability_factors: dict = {}

        reliability = user_model.compute_reliability(word)
        reliability_factors["total_user_interactions"] = user_model.get_total_interactions()
        reliability_factors["has_word_history"] = user_model.has_sufficient_history(word)

        if reliability < 0.20:
            reliability_factors["insufficient_history"] = True
            return LayerOutput(
                name="personal",
                signal_scores=signals,
                reliability=reliability,
                reliability_factors=reliability_factors,
            )

        priors = user_model.get_intent_priors(word)
        if priors:
            for c in candidates:
                prior = priors.get(c.id)
                if prior is not None:
                    # Linear blend: higher reliability → learned prior dominates
                    signals[c.id] = (
                        (1.0 - reliability) * c.base_probability + reliability * prior
                    )

        return LayerOutput(
            name="personal",
            signal_scores=signals,
            reliability=reliability,
            reliability_factors=reliability_factors,
        )
