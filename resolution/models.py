"""
resolution/models.py
------------------------
The final result of one full pipeline run — what the existing recognition
system's integration point actually receives back.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json


@dataclass(frozen=True)
class ResolutionResult:
    user_id: str
    recognized_word: str
    evaluated_at: float

    status: str                        # "RESOLVED" | "AMBIGUOUS" | "LITERAL"

    selected_intent: Optional[str]     # candidate id, only set when status == RESOLVED
    selected_label: Optional[str]
    selected_lcd_text: str             # always populated (falls back to recognized_word[:16])

    final_score: float
    conflict: bool

    output: str                        # natural-language text: ontology output_text if RESOLVED, else the literal word

    ranked_candidates: List[dict]      # full ranking from the Reasoning Engine
    top_n: List[dict]                  # populated only for AMBIGUOUS

    reason: str                        # Decision Gate's explanation, for observability

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "recognized_word": self.recognized_word,
            "evaluated_at": self.evaluated_at,
            "status": self.status,
            "selected_intent": self.selected_intent,
            "selected_label": self.selected_label,
            "selected_lcd_text": self.selected_lcd_text,
            "final_score": self.final_score,
            "conflict": self.conflict,
            "output": self.output,
            "ranked_candidates": self.ranked_candidates,
            "top_n": self.top_n,
            "reason": self.reason,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class EngineTrace:
    """
    A bundle of the REAL, already-computed objects produced during one
    IntentEngine.resolve() call — Layer 1's candidates, Layer 3's and
    Layer 5's full evidence, the Reasoning Engine's full ranking, the
    Decision Gate's full decision, and the trimmed ResolutionResult.

    This performs no computation of its own — it only retains references
    to objects each layer already built, so a visualization layer (or a
    future research-data export) can inspect the complete reasoning chain
    without recalculating anything. Every nested object is a frozen
    dataclass from its owning layer/module, so nothing here can mutate or
    silently drift from what the engine actually computed.
    """
    user_id: str
    word: str
    evaluated_at: float

    candidates: list            # List[intent.ontology.IntentCandidate] — Layer 1's raw output
    temporal_result: object     # layer3_temporal.schema.TemporalResolution
    personal_result: object     # layer5_personal.schema.PersonalResolution
    reasoning_result: object    # reasoning.models.ReasoningResult
    gate_decision: object       # decision_gate.models.GateDecision
    resolution: "ResolutionResult"

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "word": self.word,
            "evaluated_at": self.evaluated_at,
            "candidates": [asdict(c) for c in self.candidates],
            "temporal_result": self.temporal_result.to_dict(),
            "personal_result": self.personal_result.to_dict(),
            "reasoning_result": self.reasoning_result.to_dict(),
            "gate_decision": self.gate_decision.to_dict(),
            "resolution": self.resolution.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
