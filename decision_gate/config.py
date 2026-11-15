"""
decision_gate/config.py
---------------------------
Thresholds for the Confidence + Conflict Decision Gate. Every threshold
is a hand-set, documented constant — no learned parameters, consistent
with every prior phase.

Note: intent/confidence_gate.py already implements a conceptually similar
idea (HIGH_CONF/MED_CONF thresholds, resolved/conflict/top_n/literal
modes) for the OLD internal intent/ pipeline (CandidateGenerator ->
4 Context Layers -> IntentResolver -> ConfidenceGate). This module is a
NEW, independent gate for the NEW pipeline (Layer 1 -> Layer 3/5 ->
Reasoning Engine) built across this session — it does not read from,
write to, or import that legacy module.
"""
from dataclasses import dataclass

MODE_RESOLVED = "resolved"
MODE_AMBIGUOUS = "ambiguous"
MODE_LITERAL_FALLBACK = "literal_fallback"

# Top candidate's final_score must meet/exceed this to even be considered
# for RESOLVED.
HIGH_CONFIDENCE_THRESHOLD = 0.65

# Gap required between rank-1 and rank-2 final_score for RESOLVED — a
# high top score that's nearly tied with the runner-up is still
# ambiguous, not confidently resolved.
MIN_MARGIN = 0.15

# Below this, the top candidate isn't even worth presenting as one option
# among several — there's no usable evidence at all, fall back to the
# literal recognized word.
LOW_CONFIDENCE_THRESHOLD = 0.25

# How many candidates to surface when the gate returns AMBIGUOUS / TOP-N.
TOP_N_COUNT = 3


@dataclass(frozen=True)
class DecisionGateConfig:
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD
    min_margin: float = MIN_MARGIN
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    top_n_count: int = TOP_N_COUNT

    def to_dict(self) -> dict:
        return {
            "high_confidence_threshold": self.high_confidence_threshold,
            "min_margin": self.min_margin,
            "low_confidence_threshold": self.low_confidence_threshold,
            "top_n_count": self.top_n_count,
        }
