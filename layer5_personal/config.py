"""
layer5_personal/config.py
----------------------------
All Layer 5 (Personal Context) hyperparameters and paths in one place.

Nothing here is a trained weight — deterministic, explainable scoring
weights, per the same rule Layers 1/3 followed (no LLM, no learned model).

The recency mechanism is deliberately DIFFERENT from Layer 3's: Layer 3
decays linearly over a 120-SECOND window (the current exchange); Layer 5
decays exponentially over a HALF-LIFE measured in DAYS (long-term habit),
so the two layers can never collapse into duplicates of each other.
"""
import os
from dataclasses import dataclass

# ── Storage ──────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(_ROOT, "data", "personal_context.db")

# ── Recency (long-term, exponential half-life — NOT Layer 3's linear window) ──
RECENCY_HALF_LIFE_DAYS = 30.0

# ── Feature saturation points ───────────────────────────────────────────────
OBSERVATION_SATURATION_COUNT = 10   # occurrences of ONE word->intent pair for full sufficiency
PROFILE_SATURATION_COUNT = 20       # total observations across ALL words for full profile maturity

# ── Signal weights (sum to 1.0) ─────────────────────────────────────────────
SIGNAL_WEIGHT_PRIOR = 0.70
SIGNAL_WEIGHT_RECENCY = 0.30

# ── Reliability weights (sum to 1.0) ────────────────────────────────────────
RELIABILITY_WEIGHT_PROFILE_MATURITY = 0.35   # global: has this user built any profile at all?
RELIABILITY_WEIGHT_SUFFICIENCY = 0.30        # candidate-specific: enough observations of THIS pair?
RELIABILITY_WEIGHT_CONCENTRATION = 0.20      # candidate-specific: is history dominated by one intent, or split?
RELIABILITY_WEIGHT_RECENCY = 0.15            # candidate-specific: reused from signal's recency_weight


@dataclass(frozen=True)
class Layer5Config:
    """Snapshot of the active configuration, for logging alongside results."""
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS
    observation_saturation_count: int = OBSERVATION_SATURATION_COUNT
    profile_saturation_count: int = PROFILE_SATURATION_COUNT
    signal_weight_prior: float = SIGNAL_WEIGHT_PRIOR
    signal_weight_recency: float = SIGNAL_WEIGHT_RECENCY
    reliability_weight_profile_maturity: float = RELIABILITY_WEIGHT_PROFILE_MATURITY
    reliability_weight_sufficiency: float = RELIABILITY_WEIGHT_SUFFICIENCY
    reliability_weight_concentration: float = RELIABILITY_WEIGHT_CONCENTRATION
    reliability_weight_recency: float = RELIABILITY_WEIGHT_RECENCY

    def to_dict(self) -> dict:
        return {
            "recency_half_life_days": self.recency_half_life_days,
            "observation_saturation_count": self.observation_saturation_count,
            "profile_saturation_count": self.profile_saturation_count,
            "signal_weight_prior": self.signal_weight_prior,
            "signal_weight_recency": self.signal_weight_recency,
            "reliability_weight_profile_maturity": self.reliability_weight_profile_maturity,
            "reliability_weight_sufficiency": self.reliability_weight_sufficiency,
            "reliability_weight_concentration": self.reliability_weight_concentration,
            "reliability_weight_recency": self.reliability_weight_recency,
        }
