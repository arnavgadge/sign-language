"""
reasoning/config.py
----------------------
Configuration for the Reasoning Engine. No trained parameters — a
transparent, hand-set threshold, per the same rule every prior layer
followed (no LLM, no learned model, deterministic and explainable).
"""
from dataclasses import dataclass

# Names of the evidence sources currently implemented. Layer 2 ("visual")
# and Layer 4 ("device") are NOT in this list — they are simply not
# implemented yet, and the engine must never fabricate evidence for a
# source that isn't here. Adding them later is a matter of the resolver
# accepting new optional parameters, not a rewrite of this module.
SOURCE_TEMPORAL = "temporal"
SOURCE_PERSONAL = "personal"

# Minimum magnitude of disagreement between two sources' top-supported
# candidates (weighted by signal * reliability) before it's flagged as a
# genuine conflict rather than noise.
CONFLICT_THRESHOLD = 0.30


@dataclass(frozen=True)
class ReasoningConfig:
    conflict_threshold: float = CONFLICT_THRESHOLD

    def to_dict(self) -> dict:
        return {"conflict_threshold": self.conflict_threshold}
