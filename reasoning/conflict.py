"""
reasoning/conflict.py
-------------------------
Detects when available evidence sources disagree about which candidate
is best supported. Deterministic, explainable, configurable — not a
scenario-specific rule.

Method: for each available source, find the candidate with the highest
SIGNAL (what that source is asserting, independent of how much we trust
it). If two sources' top picks differ, the disagreement's magnitude is
the smaller of the two sources' own (signal * reliability) for their
respective top pick — i.e. a disagreement between two weakly-reliable
sources counts for little; a disagreement between two strongly reliable,
strongly confident sources counts for a lot. Conflict is flagged if the
largest such magnitude across all source pairs meets the configured
threshold.
"""
from typing import Dict, List

from .config import ReasoningConfig
from .models import EvidenceItem, ConflictReport


class ConflictDetector:
    def __init__(self, config: ReasoningConfig = None):
        self.config = config or ReasoningConfig()

    def detect(self, evidence_by_source: Dict[str, Dict[str, EvidenceItem]]) -> ConflictReport:
        """
        evidence_by_source: {source_name: {candidate_id: EvidenceItem}}
        (only available sources — resolver's responsibility to filter).
        """
        source_names = [s for s in evidence_by_source if evidence_by_source[s]]
        if len(source_names) < 2:
            return ConflictReport(conflict_detected=False, conflict_score=0.0, disagreements=[])

        top_picks: Dict[str, str] = {}
        for name in source_names:
            candidate_map = evidence_by_source[name]
            # Deterministic tie-break: highest signal, then candidate id.
            top_id = max(candidate_map.keys(), key=lambda cid: (candidate_map[cid].signal, cid))
            top_picks[name] = top_id

        disagreements: List[dict] = []
        max_conflict = 0.0

        for i in range(len(source_names)):
            for j in range(i + 1, len(source_names)):
                a, b = source_names[i], source_names[j]
                cand_a, cand_b = top_picks[a], top_picks[b]
                if cand_a == cand_b:
                    continue

                ev_a = evidence_by_source[a][cand_a]
                ev_b = evidence_by_source[b][cand_b]
                strength_a = ev_a.signal * ev_a.reliability
                strength_b = ev_b.signal * ev_b.reliability
                magnitude = min(strength_a, strength_b)

                disagreements.append({
                    "source_a": a, "candidate_a": cand_a, "strength_a": round(strength_a, 4),
                    "source_b": b, "candidate_b": cand_b, "strength_b": round(strength_b, 4),
                    "magnitude": round(magnitude, 4),
                })
                max_conflict = max(max_conflict, magnitude)

        return ConflictReport(
            conflict_detected=max_conflict >= self.config.conflict_threshold,
            conflict_score=round(max_conflict, 4),
            disagreements=disagreements,
        )
