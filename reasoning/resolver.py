"""
reasoning/resolver.py
-------------------------
LAYER: REASONING ENGINE.

Purpose: given the candidates Layer 1 generated, and whichever of Layer
3 (temporal) / Layer 5 (personal) actually ran, fuse their signal +
reliability evidence into a ranked list of candidates. Does NOT decide a
final answer for the system to act on — that is the Confidence Gate's
job (next phase, not built here).

Pipeline position:
    Recognized Word -> Layer 1 -> Candidate Intents
        -> Layer 3 (temporal) -----\\
        -> Layer 5 (personal) ------+--> REASONING ENGINE (this module)
                                     -> Ranked candidates + conflict report
                                     -> [future: Confidence Gate, NOT built here]

Design: IntentResolver only GATHERS evidence (deciding which sources are
available and building the per-candidate EvidenceItem maps); the actual
fusion math lives in a swappable ScoringStrategy (strategies.py), and
conflict detection lives in a separate ConflictDetector (conflict.py).
This keeps "how do we combine evidence" replaceable (e.g. a future
BayesianStrategy) without touching Layers 1, 3, or 5, or this gathering
logic.
"""
import logging
import time
from typing import Dict, List, Optional

from intent.ontology import IntentCandidate
from layer3_temporal.schema import TemporalResolution
from layer5_personal.schema import PersonalResolution

from .config import ReasoningConfig, SOURCE_TEMPORAL, SOURCE_PERSONAL
from .conflict import ConflictDetector
from .models import CandidateResult, EvidenceItem, ReasoningResult
from .strategies import ReliabilityWeightedStrategy, ScoringStrategy

logger = logging.getLogger("reasoning.resolver")


class IntentResolver:
    """
    Usage
    -----
    resolver = IntentResolver()   # default: ReliabilityWeightedStrategy

    result = resolver.resolve(
        word="HELP",
        candidates=layer1_candidates,       # List[IntentCandidate], from Layer 1 — authoritative
        temporal=temporal_resolution,        # TemporalResolution or None if unavailable
        personal=personal_resolution,        # PersonalResolution or None if unavailable
    )
    for c in result.candidates:   # already ranked, full list
        print(c.rank, c.candidate, c.final_score, c.sources)
    print(result.conflict.conflict_detected)
    """

    def __init__(self, strategy: ScoringStrategy = None,
                 conflict_detector: ConflictDetector = None,
                 config: ReasoningConfig = None):
        self.config = config or ReasoningConfig()
        self.strategy = strategy or ReliabilityWeightedStrategy()
        self.conflict_detector = conflict_detector or ConflictDetector(self.config)

    def resolve(self, word: str, candidates: List[IntentCandidate],
                temporal: Optional[TemporalResolution] = None,
                personal: Optional[PersonalResolution] = None,
                now: float = None) -> ReasoningResult:

        # Layer 1's candidate list is authoritative — the engine ranks
        # ONLY these ids, regardless of what any evidence source contains.
        candidate_ids = [c.id for c in candidates]
        label_map = {c.id: c.label for c in candidates}

        available_sources: List[str] = []
        evidence_by_source: Dict[str, Dict[str, EvidenceItem]] = {}

        if temporal is not None:
            available_sources.append(SOURCE_TEMPORAL)
            temporal_map = {ev.candidate: ev for ev in temporal.candidates}
            evidence_by_source[SOURCE_TEMPORAL] = {
                cid: EvidenceItem(temporal_map[cid].signal, temporal_map[cid].reliability)
                for cid in candidate_ids if cid in temporal_map
            }

        if personal is not None:
            available_sources.append(SOURCE_PERSONAL)
            personal_map = {ev.candidate: ev for ev in personal.candidates}
            evidence_by_source[SOURCE_PERSONAL] = {
                cid: EvidenceItem(personal_map[cid].signal, personal_map[cid].reliability)
                for cid in candidate_ids if cid in personal_map
            }
        # Layer 2 ("visual") / Layer 4 ("device") are simply absent from
        # `available_sources` — not implemented, never simulated.

        results: List[CandidateResult] = []
        for cid in candidate_ids:
            per_candidate_evidence = {
                src: evidence_by_source[src][cid]
                for src in available_sources
                if cid in evidence_by_source[src]
            }
            final_score, sources = self.strategy.score(per_candidate_evidence)
            results.append(CandidateResult(
                candidate=cid, label=label_map.get(cid, cid),
                final_score=final_score, sources=sources, rank=0,
            ))

        # Rank highest-first; deterministic tie-break by candidate id.
        results.sort(key=lambda r: (-r.final_score, r.candidate))
        ranked = [
            CandidateResult(candidate=r.candidate, label=r.label,
                             final_score=r.final_score, sources=r.sources, rank=i + 1)
            for i, r in enumerate(results)
        ]

        conflict = self.conflict_detector.detect(evidence_by_source)

        result = ReasoningResult(
            word=word.upper().strip(),
            evaluated_at=now if now is not None else time.time(),
            strategy_name=self.strategy.name,
            available_sources=available_sources,
            candidates=ranked,
            conflict=conflict,
        )
        self._log(result)
        return result

    # ── Observability ──────────────────────────────────────────────────────

    def _log(self, result: ReasoningResult) -> None:
        logger.info("WORD: %s", result.word)
        logger.info("Available sources: %s  |  Strategy: %s",
                     result.available_sources, result.strategy_name)
        for c in result.candidates:
            logger.info("-" * 30)
            logger.info("%s (%s)", c.label.upper(), c.candidate)
            for src_name, contrib in c.sources.items():
                logger.info("  %s:", src_name.capitalize())
                logger.info("    Signal      = %.4f", contrib.signal)
                logger.info("    Reliability = %.4f", contrib.reliability)
                logger.info("    Contribution= %.4f", contrib.weighted_contribution)
            logger.info("  Final Score = %.4f", c.final_score)
        logger.info("-" * 30)
        logger.info("CONFLICT: %s (score=%.4f)", result.conflict.conflict_detected, result.conflict.conflict_score)
        logger.info("RANKING:")
        for c in result.candidates:
            logger.info("  %d. %s (%.4f)", c.rank, c.label, c.final_score)
