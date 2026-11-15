"""
layer3_temporal/config.py
---------------------------
All Layer 3 (Temporal Context) hyperparameters in one place, so every
resolution can be logged alongside the exact configuration that produced it.

Nothing here is a trained weight — these are transparent, hand-set scoring
weights for a deterministic, explainable model, per the Phase 2 requirement
("do not use an LLM", "do not introduce sophisticated NLP yet").
"""
import os
from dataclasses import dataclass, field

# ── Temporal window ─────────────────────────────────────────────────────────
TEMPORAL_WINDOW_SECONDS = 120.0   # configurable, not hardcoded elsewhere

# ── Feature normalization ───────────────────────────────────────────────────
FREQUENCY_SATURATION_COUNT = 3     # 3+ relevant trigger occurrences -> frequency = 1.0
REPETITION_SATURATION_COUNT = 2    # 2+ prior repeats of current_word -> repetition = 1.0
RICHNESS_SATURATION_COUNT = 3      # 3+ words in window -> full conversation "richness"

# ── Signal weights (must sum to 1.0 before the conflict penalty) ───────────
SIGNAL_WEIGHT_SEQUENCE = 0.50
SIGNAL_WEIGHT_RECENCY = 0.30
SIGNAL_WEIGHT_FREQUENCY = 0.20
SIGNAL_CONFLICT_PENALTY_WEIGHT = 0.50   # subtracted, scaled by conflict_strength

# ── Reliability weights (must sum to 1.0) ───────────────────────────────────
RELIABILITY_WEIGHT_RICHNESS = 0.45   # global: how much conversation exists at all
RELIABILITY_WEIGHT_RECENCY = 0.30    # candidate-specific: freshness of its evidence
RELIABILITY_WEIGHT_COHERENCE = 0.25  # candidate-specific: tightness of its evidence's gaps

RICHNESS_REPETITION_BLEND = 0.30     # fraction of "richness" contributed by repetition

# ── Paths ────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPORAL_LOG_DIR = os.path.join(_ROOT, "data", "temporal_resolutions")


@dataclass(frozen=True)
class Layer3Config:
    """Snapshot of the active configuration, for logging alongside results."""
    window_seconds: float = TEMPORAL_WINDOW_SECONDS
    frequency_saturation_count: int = FREQUENCY_SATURATION_COUNT
    repetition_saturation_count: int = REPETITION_SATURATION_COUNT
    richness_saturation_count: int = RICHNESS_SATURATION_COUNT
    signal_weight_sequence: float = SIGNAL_WEIGHT_SEQUENCE
    signal_weight_recency: float = SIGNAL_WEIGHT_RECENCY
    signal_weight_frequency: float = SIGNAL_WEIGHT_FREQUENCY
    signal_conflict_penalty_weight: float = SIGNAL_CONFLICT_PENALTY_WEIGHT
    reliability_weight_richness: float = RELIABILITY_WEIGHT_RICHNESS
    reliability_weight_recency: float = RELIABILITY_WEIGHT_RECENCY
    reliability_weight_coherence: float = RELIABILITY_WEIGHT_COHERENCE
    richness_repetition_blend: float = RICHNESS_REPETITION_BLEND

    def to_dict(self) -> dict:
        return {
            "window_seconds": self.window_seconds,
            "frequency_saturation_count": self.frequency_saturation_count,
            "repetition_saturation_count": self.repetition_saturation_count,
            "richness_saturation_count": self.richness_saturation_count,
            "signal_weight_sequence": self.signal_weight_sequence,
            "signal_weight_recency": self.signal_weight_recency,
            "signal_weight_frequency": self.signal_weight_frequency,
            "signal_conflict_penalty_weight": self.signal_conflict_penalty_weight,
            "reliability_weight_richness": self.reliability_weight_richness,
            "reliability_weight_recency": self.reliability_weight_recency,
            "reliability_weight_coherence": self.reliability_weight_coherence,
            "richness_repetition_blend": self.richness_repetition_blend,
        }
