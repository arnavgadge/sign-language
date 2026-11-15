"""
reasoning/models.py
----------------------
Shared data structures for the Reasoning Engine. Nothing here queries a
database or performs I/O — pure structure, matching the "Database -> Layer
-> Evidence object -> Reasoning Engine" separation requested.
"""
from dataclasses import dataclass, field
from typing import Dict, List
import json


@dataclass(frozen=True)
class EvidenceItem:
    """One (source, candidate) pair's raw signal/reliability, as produced
    by Layer 3 or Layer 5 — never invented by the Reasoning Engine."""
    signal: float
    reliability: float


@dataclass(frozen=True)
class SourceContribution:
    """What one available source contributed to one candidate's final score."""
    signal: float
    reliability: float
    weighted_contribution: float   # signal * reliability

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "reliability": self.reliability,
            "weighted_contribution": self.weighted_contribution,
        }


@dataclass(frozen=True)
class CandidateResult:
    candidate: str
    label: str
    final_score: float             # confidence-like, [0,1] — NOT a calibrated probability
    sources: Dict[str, SourceContribution]   # only keys for sources actually available
    rank: int                       # 1-based, set after ranking

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "label": self.label,
            "final_score": self.final_score,
            "rank": self.rank,
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
        }


@dataclass(frozen=True)
class ConflictReport:
    conflict_detected: bool
    conflict_score: float
    disagreements: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "conflict_detected": self.conflict_detected,
            "conflict_score": self.conflict_score,
            "disagreements": self.disagreements,
        }


@dataclass(frozen=True)
class ReasoningResult:
    word: str
    evaluated_at: float
    strategy_name: str
    available_sources: List[str]     # which context sources actually contributed
    candidates: List[CandidateResult]  # full ranking, not just the winner
    conflict: ConflictReport

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "evaluated_at": self.evaluated_at,
            "strategy_name": self.strategy_name,
            "available_sources": self.available_sources,
            "candidates": [c.to_dict() for c in self.candidates],
            "conflict": self.conflict.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
