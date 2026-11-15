"""
reasoning/strategies.py
--------------------------
Swappable scoring strategies. A ScoringStrategy only sees whatever
evidence it's handed for one candidate (already filtered to available
sources by IntentResolver) — it does not know about "missing" layers,
databases, or anything upstream. This is what makes it possible to swap
the fusion mechanism later without touching Layers 1, 3, or 5, or the
resolver's evidence-gathering logic.

ReliabilityWeightedStrategy is the PRODUCTION strategy — the one
resolution/engine.py uses by default, and the one every prior phase's
tests and evaluation were run against:

    final_score = sum(signal_i * reliability_i) / sum(reliability_i)

This is a transparent weighted average — NOT a Bayesian/probabilistic
model. A future BayesianStrategy could implement this same interface
(e.g. treating reliability as an inverse-variance weight in a proper
probabilistic fusion), but it is not built here — do not claim otherwise.

UnweightedAverageStrategy (added Phase 9) is an EVALUATION-ONLY
comparison strategy — a plain mean of the available signals, ignoring
reliability entirely:

    final_score = mean(signal_i)

It exists solely to answer Phase 9's ablation question ("does
reliability weighting contribute beyond simply combining signals?") by
giving the evaluation harness something to compare against. Nothing in
resolution/engine.py or any production code path uses it — the default
strategy everywhere except the Phase 9 ablation script remains
ReliabilityWeightedStrategy, unchanged.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from .models import EvidenceItem, SourceContribution


class ScoringStrategy(ABC):
    name: str

    @abstractmethod
    def score(self, evidence_by_source: Dict[str, EvidenceItem]) -> Tuple[float, Dict[str, SourceContribution]]:
        """
        evidence_by_source: {source_name: EvidenceItem} — ONLY sources
        available for this resolution; the strategy must not assume any
        particular set of keys is present.

        Returns (final_score, sources) where `sources` mirrors the input
        with each item's computed weighted_contribution attached, for
        observability/explanation.
        """
        raise NotImplementedError


class ReliabilityWeightedStrategy(ScoringStrategy):
    name = "reliability_weighted"

    def score(self, evidence_by_source: Dict[str, EvidenceItem]) -> Tuple[float, Dict[str, SourceContribution]]:
        numerator = 0.0
        denominator = 0.0
        sources: Dict[str, SourceContribution] = {}

        for source_name, item in evidence_by_source.items():
            contribution = item.signal * item.reliability
            numerator += contribution
            denominator += item.reliability
            sources[source_name] = SourceContribution(
                signal=item.signal,
                reliability=item.reliability,
                weighted_contribution=round(contribution, 4),
            )

        final_score = (numerator / denominator) if denominator > 0 else 0.0
        return round(final_score, 4), sources


class UnweightedAverageStrategy(ScoringStrategy):
    """EVALUATION-ONLY (Phase 9 ablation #7) — plain mean of available
    signals, reliability recorded for observability but not used in the
    score itself. Never used by resolution/engine.py."""
    name = "unweighted_average"

    def score(self, evidence_by_source: Dict[str, EvidenceItem]) -> Tuple[float, Dict[str, SourceContribution]]:
        sources: Dict[str, SourceContribution] = {}
        if not evidence_by_source:
            return 0.0, sources

        signals = []
        for source_name, item in evidence_by_source.items():
            signals.append(item.signal)
            sources[source_name] = SourceContribution(
                signal=item.signal,
                reliability=item.reliability,
                weighted_contribution=round(item.signal, 4),   # "contribution" here == its own signal, unweighted
            )

        final_score = sum(signals) / len(signals)
        return round(final_score, 4), sources
