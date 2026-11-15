"""
layer5_personal/engine.py
-----------------------------
LAYER 5 — PERSONAL CONTEXT.

Purpose: determine how a SPECIFIC user's longer-term communication habits
should influence the interpretation of a recognized word. Takes the
candidate intents Layer 1 already generated and scores each one against
that user's stored history. Does NOT rank, filter, or pick a winner, and
does NOT look at the current conversation sequence — that's Layer 3's
job, not duplicated here.

Pipeline position:
    Recognized Word -> Layer 1 (candidate generation) -> Candidate Intents
        -> LAYER 5 (this module) -> candidate-specific personal evidence
        -> [future: Reasoning Engine, NOT built here]
"""
import logging
import time
from typing import List

from intent.ontology import IntentCandidate

from .config import Layer5Config
from .db import PersonalContextDB
from .schema import PersonalEvidence, PersonalResolution
from .scoring import score_candidate

logger = logging.getLogger("layer5.personal")


class PersonalContextEngine:
    """
    Usage
    -----
    engine = PersonalContextEngine()                    # real DB
    engine = PersonalContextEngine(db_path=":memory:")   # isolated, for tests

    # Clean update interface — called by the future Reasoning Engine once
    # a final intent is resolved. NOT called by this module itself, and
    # no final-resolution process is assumed or faked here.
    engine.update(user_id="001", word="HELP", resolved_intent="medical_assistance")

    # When Layer 1 has produced candidates for the current word:
    resolution = engine.evaluate(user_id="001", current_word="HELP",
                                  candidates=layer1_candidates)
    for ev in resolution.candidates:
        print(ev.candidate, ev.signal, ev.reliability, ev.evidence)
    """

    def __init__(self, config: Layer5Config = None, db_path: str = None):
        self.config = config or Layer5Config()
        self.db = PersonalContextDB(db_path=db_path) if db_path else PersonalContextDB()

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, user_id: str, word: str, resolved_intent: str,
               timestamp: float = None) -> None:
        """
        Record that `resolved_intent` was the outcome for `word` for this
        user. This is the clean interface the (not-yet-built) Reasoning
        Engine will call once it produces a final decision — Layer 5 does
        not assume or simulate that a resolver exists; it only exposes
        this update hook for one to call later.
        """
        self.db.record(user_id, word, resolved_intent, timestamp=timestamp)

    def evaluate(self, user_id: str, current_word: str,
                 candidates: List[IntentCandidate],
                 now: float = None) -> PersonalResolution:
        """Score each Layer-1-provided candidate against this user's
        stored history for `current_word`. Reads only — never mutates
        history (call `update()` separately once an intent is resolved)."""
        eval_time = now if now is not None else time.time()
        word = current_word.upper().strip()

        word_rows = self.db.get_word_rows(user_id, word)
        total_user_observations = self.db.get_user_total_observations(user_id)
        distinct_words = self.db.get_user_distinct_words(user_id)

        scored: List[PersonalEvidence] = [
            score_candidate(c, word_rows, eval_time, self.config, total_user_observations)
            for c in candidates
        ]

        resolution = PersonalResolution(
            user_id=user_id,
            current_word=word,
            evaluated_at=eval_time,
            profile_total_observations=total_user_observations,
            profile_distinct_words=distinct_words,
            candidates=scored,
        )

        self._log(resolution)
        return resolution

    def close(self) -> None:
        self.db.close()

    # ── Observability ──────────────────────────────────────────────────────

    def _log(self, resolution: PersonalResolution) -> None:
        logger.info(
            "[PERSONAL] USER: %s  WORD: %s  (profile_observations=%d, distinct_words=%d)",
            resolution.user_id, resolution.current_word,
            resolution.profile_total_observations, resolution.profile_distinct_words,
        )
        for ev in resolution.candidates:
            logger.info(
                "[PERSONAL]   Candidate: %-20s Historical count: %-4d Prior: %.3f "
                "Signal: %.3f Reliability: %.3f",
                ev.candidate, ev.evidence["historical_count"],
                ev.evidence["personal_prior"], ev.signal, ev.reliability,
            )
