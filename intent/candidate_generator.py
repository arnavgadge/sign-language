"""
Stage 1 — Three-stage candidate generation funnel.

  1A  Ontology lookup       – all intents linked to the word in the graph
  1B  Context filter        – eliminate candidates inconsistent with recent history
  1C  Personal pruning      – remove historically unlikely candidates for this user

The funnel hands a small, contextually plausible shortlist to Stage 2.
"""
from typing import List, Tuple

from .ontology import IntentOntology, IntentCandidate
from .conversation import ConversationHistory
from .user_model import UserModel

# Minimum personal-prior to survive pruning (prevents over-pruning for new users)
PERSONAL_PRUNE_THRESHOLD = 0.05
# Never prune below this many candidates
MIN_CANDIDATES = 2
# Context filter: only filter if layers differentiate by at least this margin
CONTEXT_FILTER_MIN_DELTA = 0.5


class CandidateGenerator:
    def __init__(
        self,
        ontology: IntentOntology,
        history: ConversationHistory,
        user_model: UserModel,
    ):
        self.ontology = ontology
        self.history = history
        self.user_model = user_model

    def generate(self, word: str) -> Tuple[List[IntentCandidate], dict]:
        """
        Returns (candidates, debug_info).
          candidates  – shortlist after all three stages
          debug_info  – intermediate states for display / logging
        """
        debug: dict = {}

        # ── 1A: Ontology lookup ───────────────────────────────────────────────
        raw = self.ontology.lookup(word)
        debug["1a_raw"] = [c.id for c in raw]

        if len(raw) <= 1:
            debug["1b_after_context"] = [c.id for c in raw]
            debug["1c_after_personal"] = [c.id for c in raw]
            return raw, debug

        # ── 1B: Conversation context filter ───────────────────────────────────
        after_context = self._context_filter(raw)
        debug["1b_after_context"] = [c.id for c in after_context]

        # ── 1C: Personal pattern pruning ──────────────────────────────────────
        after_personal = self._personal_prune(after_context, word)
        debug["1c_after_personal"] = [c.id for c in after_personal]

        return after_personal, debug

    # ── Stage 1B ──────────────────────────────────────────────────────────────

    def _context_filter(self, candidates: List[IntentCandidate]) -> List[IntentCandidate]:
        recent = self.history.recent_words(seconds=120)
        if not recent:
            return candidates  # No history → nothing to filter on

        scored: List[Tuple[IntentCandidate, float]] = []
        for c in candidates:
            score = 0.0
            for trigger in c.context_triggers:
                if trigger in recent:
                    score += 1.0
            for conflict in c.context_conflicts:
                if conflict in recent:
                    score -= 0.8
            scored.append((c, score))

        max_s = max(s for _, s in scored)
        min_s = min(s for _, s in scored)

        # Only filter if there is meaningful differentiation
        if (max_s - min_s) < CONTEXT_FILTER_MIN_DELTA:
            return candidates

        # Keep candidates within 0.5 of the minimum (i.e. drop clearly conflicting ones)
        surviving = [c for c, s in scored if s > min_s - 0.5]
        if len(surviving) < MIN_CANDIDATES:
            return candidates
        return surviving

    # ── Stage 1C ──────────────────────────────────────────────────────────────

    def _personal_prune(
        self, candidates: List[IntentCandidate], word: str
    ) -> List[IntentCandidate]:
        if not self.user_model.has_sufficient_history(word):
            return candidates  # Not enough data → no pruning

        priors = self.user_model.get_intent_priors(word)
        if not priors:
            return candidates

        surviving = [
            c
            for c in candidates
            # Keep if no personal record (never seen this intent) OR prior above threshold
            if priors.get(c.id, 0.0) == 0.0 or priors.get(c.id, 0.0) >= PERSONAL_PRUNE_THRESHOLD
        ]

        if len(surviving) < MIN_CANDIDATES:
            return candidates  # Safety: never drop below minimum
        return surviving
