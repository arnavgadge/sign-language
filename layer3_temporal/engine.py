"""
layer3_temporal/engine.py
----------------------------
LAYER 3 — TEMPORAL CONTEXT.

Purpose: determine how the meaning of the currently recognized word is
influenced by the user's recent communication sequence. Takes the
candidate intents Layer 1 already generated for the current word, and
scores each one with temporal evidence (signal + reliability). It does
NOT rank, filter, or pick a winner — that is explicitly out of scope
(no reasoning engine, no confidence gate, no final decision here).

Pipeline position:
    Recognized Word -> Layer 1 (candidate generation) -> Candidate Intents
        -> LAYER 3 (this module) -> candidate-specific temporal evidence
        -> [future: Layer 5 decision-making, NOT built here]
"""
import logging
import time
from typing import List

from intent.ontology import IntentCandidate

from .config import Layer3Config, TEMPORAL_WINDOW_SECONDS
from .features import compute_repetition
from .history_store import TemporalHistoryStore
from .schema import TemporalEvidence, TemporalResolution
from .scoring import score_candidate, window_richness

logger = logging.getLogger("layer3.temporal")


class TemporalContextEngine:
    """
    Usage
    -----
    engine = TemporalContextEngine()

    # After each letter-word is accepted by the (untouched) recognition
    # pipeline, record it:
    engine.record(user_id="default", word="CHEST")
    engine.record(user_id="default", word="PAIN")

    # When Layer 1 has produced candidates for the current word:
    resolution = engine.evaluate(user_id="default", current_word="HELP",
                                  candidates=layer1_candidates)
    for ev in resolution.candidates:
        print(ev.candidate, ev.signal, ev.reliability, ev.evidence)
    """

    def __init__(self, config: Layer3Config = None, history_store: TemporalHistoryStore = None):
        self.config = config or Layer3Config()
        self.history = history_store or TemporalHistoryStore(window_seconds=self.config.window_seconds)

    # ── Public API ─────────────────────────────────────────────────────────

    def record(self, user_id: str, word: str, resolved_intent: str = "",
               timestamp: float = None) -> None:
        """Add a recognized word to this user's conversation history.
        Call this after the (untouched) recognition pipeline accepts a
        word — Layer 3 does not modify or gate word acceptance itself."""
        self.history.add(user_id, word, resolved_intent=resolved_intent, timestamp=timestamp)

    def evaluate(self, user_id: str, current_word: str,
                 candidates: List[IntentCandidate],
                 now: float = None) -> TemporalResolution:
        """
        Score each Layer-1-provided candidate with temporal evidence.
        Does NOT add `current_word` to history itself — call `record()`
        separately once the word is accepted, so evaluate() can be called
        speculatively without polluting history.
        """
        eval_time = now if now is not None else time.time()
        window_entries = self.history.get_window(user_id, now=eval_time,
                                                   window_seconds=self.config.window_seconds)

        repetition_score = compute_repetition(window_entries, current_word,
                                               self.config.repetition_saturation_count)
        richness = window_richness(window_entries, repetition_score, self.config)

        scored: List[TemporalEvidence] = [
            score_candidate(c, window_entries, current_word, eval_time, self.config,
                             richness, repetition_score)
            for c in candidates
        ]

        resolution = TemporalResolution(
            user_id=user_id,
            current_word=current_word.upper().strip(),
            evaluated_at=eval_time,
            window_seconds=self.config.window_seconds,
            window_word_count=len(window_entries),
            window_words=[e.word for e in window_entries],
            candidates=scored,
        )

        self._log(resolution)
        return resolution

    # ── Observability ──────────────────────────────────────────────────────

    def _log(self, resolution: TemporalResolution) -> None:
        logger.info(
            "[TEMPORAL] word=%s window=%ss recent=%s (n=%d)",
            resolution.current_word, resolution.window_seconds,
            resolution.window_words, resolution.window_word_count,
        )
        for ev in resolution.candidates:
            logger.info(
                "[TEMPORAL]   candidate=%-20s signal=%.4f reliability=%.4f "
                "relevant=%s conflicts=%s seq=%.3f recency=%.3f freq=%.3f gap=%.3f",
                ev.candidate, ev.signal, ev.reliability,
                ev.evidence["relevant_words"], ev.evidence["conflicting_words"],
                ev.evidence["sequence_strength"], ev.evidence["recency"],
                ev.evidence["frequency"], ev.evidence["time_gap"],
            )
