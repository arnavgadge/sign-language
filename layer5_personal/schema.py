"""
layer5_personal/schema.py
-----------------------------
Output schema for Layer 5 (Personal Context). Signal and reliability are
independent per candidate — see engine.py / scoring.py for why.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict
import json


@dataclass(frozen=True)
class PersonalEvidence:
    candidate: str
    signal: float
    reliability: float
    evidence: Dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PersonalResolution:
    user_id: str
    current_word: str
    evaluated_at: float
    profile_total_observations: int   # global: this user's total observations across ALL words
    profile_distinct_words: int       # global: how many distinct words this user has history for
    candidates: List[PersonalEvidence]

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "current_word": self.current_word,
            "evaluated_at": self.evaluated_at,
            "profile_total_observations": self.profile_total_observations,
            "profile_distinct_words": self.profile_distinct_words,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
