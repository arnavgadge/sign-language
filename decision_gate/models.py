"""
decision_gate/models.py
---------------------------
Output schema for the Confidence + Conflict Decision Gate.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass(frozen=True)
class GateDecision:
    word: str
    evaluated_at: float
    mode: str                            # "resolved" | "ambiguous" | "literal_fallback"

    resolved_candidate: Optional[str]    # set only when mode == "resolved"
    resolved_label: Optional[str]

    confidence: float                    # top candidate's final_score (0.0 if no candidates)
    margin: float                        # gap between rank-1 and rank-2 (== confidence if only 1 candidate)
    conflict_detected: bool

    top_n: List[dict]                    # populated only for "ambiguous"
    reason: str                          # human-readable explanation, derived from the actual numbers
    evidence: Dict                       # full inspectable numbers behind the decision

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "evaluated_at": self.evaluated_at,
            "mode": self.mode,
            "resolved_candidate": self.resolved_candidate,
            "resolved_label": self.resolved_label,
            "confidence": self.confidence,
            "margin": self.margin,
            "conflict_detected": self.conflict_detected,
            "top_n": self.top_n,
            "reason": self.reason,
            "evidence": self.evidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
