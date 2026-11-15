"""
layer3_temporal/schema.py
----------------------------
Output schema for Layer 3 (Temporal Context).

Layer 3 produces EVIDENCE per candidate intent, never a final decision —
signal and reliability are two distinct axes (see engine.py docstring).
"""
from dataclasses import dataclass, asdict, field
from typing import List, Dict
import json


@dataclass(frozen=True)
class TemporalEvidence:
    """One candidate's temporal evidence. `signal` and `reliability` are
    independent: a candidate can have high signal (the sequence strongly
    points to it) with low reliability (not enough temporal history to
    trust that pointer), or vice versa."""
    candidate: str
    signal: float
    reliability: float
    evidence: Dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TemporalResolution:
    """The full output of one Layer 3 evaluation call — one TemporalEvidence
    per candidate, plus the shared context they were all scored against."""
    user_id: str
    current_word: str
    evaluated_at: float
    window_seconds: float
    window_word_count: int
    window_words: List[str]
    candidates: List[TemporalEvidence]

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "current_word": self.current_word,
            "evaluated_at": self.evaluated_at,
            "window_seconds": self.window_seconds,
            "window_word_count": self.window_word_count,
            "window_words": self.window_words,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
