"""
layer3_temporal/scoring.py
-----------------------------
Combines the raw temporal features (features.py) into two independent,
per-candidate scores:

  SIGNAL       "How strongly does the recent sequence support this
               candidate?" — driven by which of the candidate's
               ontology-defined trigger words appeared, how recently, how
               immediately-preceding, and how often; penalized if the
               candidate's own conflict words appeared instead.

  RELIABILITY  "How much should we trust that signal right now?" — driven
               primarily by how much conversation context exists at all
               (a single-word conversation can't be trusted much either
               way), plus how fresh and internally-coherent this specific
               candidate's evidence is.

These are deliberately computed from different weighted combinations of
the underlying features, per the Phase 2 requirement that they must not
collapse into the same number.
"""
from typing import List

from intent.conversation import ConversationEntry
from intent.ontology import IntentCandidate

from .config import Layer3Config
from .schema import TemporalEvidence
from .features import (
    matched_entries,
    compute_recency,
    compute_sequence_strength,
    compute_frequency,
    compute_time_gap_coherence,
    compute_repetition,
    conversation_length,
)


def window_richness(entries: List[ConversationEntry], repetition_score: float,
                     cfg: Layer3Config) -> float:
    """Global (candidate-independent) measure of how much there is to
    reason about: more words in the window, and more repeats of the
    current request, both increase how much we trust ANY temporal
    evidence — whether it's pointing toward or away from a candidate."""
    length_term = min(1.0, conversation_length(entries) / cfg.richness_saturation_count)
    blend = cfg.richness_repetition_blend
    return (1.0 - blend) * length_term + blend * repetition_score


def score_candidate(candidate: IntentCandidate,
                     window_entries: List[ConversationEntry],
                     current_word: str,
                     now: float,
                     cfg: Layer3Config,
                     richness: float,
                     repetition_score: float) -> TemporalEvidence:

    triggers = matched_entries(window_entries, candidate.context_triggers)
    conflicts = matched_entries(window_entries, candidate.context_conflicts)

    recency = compute_recency(triggers, now, cfg.window_seconds)
    sequence_strength = compute_sequence_strength(triggers, window_entries)
    frequency = compute_frequency(triggers, cfg.frequency_saturation_count)
    time_gap = compute_time_gap_coherence(triggers, cfg.window_seconds)

    conflict_strength = compute_sequence_strength(conflicts, window_entries)

    raw_signal = (
        cfg.signal_weight_sequence * sequence_strength +
        cfg.signal_weight_recency * recency +
        cfg.signal_weight_frequency * frequency
    )
    signal = max(0.0, min(1.0, raw_signal - cfg.signal_conflict_penalty_weight * conflict_strength))

    reliability = max(0.0, min(1.0,
        cfg.reliability_weight_richness * richness +
        cfg.reliability_weight_recency * recency +
        cfg.reliability_weight_coherence * time_gap
    ))

    evidence = {
        "relevant_words": [e.word for e in triggers],
        "conflicting_words": [e.word for e in conflicts],
        "sequence_strength": round(sequence_strength, 4),
        "recency": round(recency, 4),
        "frequency": round(frequency, 4),
        "time_gap": round(time_gap, 4),
        "conflict_strength": round(conflict_strength, 4),
        "repetition": round(repetition_score, 4),
        "conversation_length": conversation_length(window_entries),
    }

    return TemporalEvidence(
        candidate=candidate.id,
        signal=round(signal, 4),
        reliability=round(reliability, 4),
        evidence=evidence,
    )
