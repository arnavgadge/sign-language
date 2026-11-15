"""
Stage 3 — Reliability-Weighted Intent Resolution with Conflict Detection.

Core formula (per candidate i):
  weighted_score_i = sum(signal(i, layer_j) * reliability(layer_j))
                     / sum(reliability(layer_j))          [only active layers]

Active layers: reliability >= RELIABILITY_FLOOR (others are excluded, not down-weighted).

Conflict detection:
  For each pair of active layers with reliability >= 0.3, check whether they
  disagree on the top candidate.  Conflict magnitude = |signal_disagreement| *
  min(reliability_a, reliability_b).  If the max conflict across all pairs
  exceeds CONFLICT_THRESHOLD the system flags it and downstream stages adapt.

Emergency override:
  If any emergency-flagged candidate receives a raw signal >= EMERGENCY_MIN_SIGNAL
  from ANY layer, it is boosted to at least that threshold before normalization.
  This ensures life-safety intents are never silently suppressed.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .ontology import IntentCandidate
from .context_layers import LayerOutput

RELIABILITY_FLOOR = 0.05       # Layers below this are excluded entirely
EMERGENCY_MIN_SIGNAL = 0.25    # Minimum raw signal to trigger emergency override
CONFLICT_THRESHOLD = 0.45      # Inter-layer disagreement magnitude for conflict flag


@dataclass
class ResolutionResult:
    candidates: List[IntentCandidate]
    ranked_scores: Dict[str, float]     # intent_id -> normalized probability
    top_intent_id: str
    top_intent_score: float
    conflict_detected: bool
    conflict_score: float
    emergency_triggered: bool
    active_layer_count: int
    layer_contributions: Dict           # debug: {intent_id -> {layer_name -> eff_score}}


class IntentResolver:

    def resolve(
        self, candidates: List[IntentCandidate], layers: List[LayerOutput]
    ) -> ResolutionResult:

        # ── Filter layers below reliability floor ─────────────────────────────
        active = [l for l in layers if l.reliability >= RELIABILITY_FLOOR]

        if not active:
            return self._fallback_base_probs(candidates)

        # ── Reliability-weighted scoring ──────────────────────────────────────
        total_rel = sum(l.reliability for l in active)
        weighted: Dict[str, float] = {}
        contributions: Dict = {}

        for c in candidates:
            w_score = sum(
                l.signal_scores.get(c.id, c.base_probability) * l.reliability
                for l in active
            )
            weighted[c.id] = (w_score / total_rel) * c.priority_weight
            contributions[c.id] = {
                l.name: round(
                    l.signal_scores.get(c.id, c.base_probability) * l.reliability, 4
                )
                for l in active
            }

        # ── Emergency override ────────────────────────────────────────────────
        emergency_triggered = False
        for c in candidates:
            if not c.emergency:
                continue
            max_raw_signal = max(
                l.signal_scores.get(c.id, 0.0) for l in active
            )
            if max_raw_signal >= EMERGENCY_MIN_SIGNAL:
                # Boost score to at least the raw signal value
                if weighted.get(c.id, 0.0) < max_raw_signal:
                    weighted[c.id] = max_raw_signal
                    emergency_triggered = True

        # ── Normalize to probability distribution ─────────────────────────────
        total = sum(weighted.values()) or 1.0
        normalized = {k: v / total for k, v in weighted.items()}

        # ── Conflict detection ────────────────────────────────────────────────
        conflict_score, conflict_detected = self._detect_conflict(candidates, active)

        top_id = max(normalized, key=normalized.get)

        return ResolutionResult(
            candidates=candidates,
            ranked_scores=normalized,
            top_intent_id=top_id,
            top_intent_score=normalized[top_id],
            conflict_detected=conflict_detected,
            conflict_score=round(conflict_score, 4),
            emergency_triggered=emergency_triggered,
            active_layer_count=len(active),
            layer_contributions=contributions,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_conflict(
        self, candidates: List[IntentCandidate], layers: List[LayerOutput]
    ) -> Tuple[float, bool]:
        """
        Returns (max_conflict_score, conflict_flag).

        A conflict occurs when two reliable layers each confidently support
        *different* top candidates.  The score captures both the magnitude of
        disagreement and the reliability of the disagreeing layers.
        """
        if len(layers) < 2:
            return 0.0, False

        max_conflict = 0.0

        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                la, lb = layers[i], layers[j]
                pair_rel = min(la.reliability, lb.reliability)
                if pair_rel < 0.30:
                    continue  # Too unreliable to create meaningful conflict

                top_a = max(candidates, key=lambda c: la.signal_scores.get(c.id, 0.0))
                top_b = max(candidates, key=lambda c: lb.signal_scores.get(c.id, 0.0))

                if top_a.id != top_b.id:
                    sig_a_in_b = lb.signal_scores.get(top_a.id, 0.0)
                    sig_a_in_a = la.signal_scores.get(top_a.id, 0.0)
                    disagreement = abs(sig_a_in_a - sig_a_in_b)
                    conflict = disagreement * pair_rel
                    max_conflict = max(max_conflict, conflict)

        return max_conflict, max_conflict >= CONFLICT_THRESHOLD

    def _fallback_base_probs(
        self, candidates: List[IntentCandidate]
    ) -> ResolutionResult:
        """All layers unreliable → fall back to ontology base probabilities."""
        scores = {c.id: c.base_probability * c.priority_weight for c in candidates}
        total = sum(scores.values()) or 1.0
        normalized = {k: v / total for k, v in scores.items()}
        top_id = max(normalized, key=normalized.get)
        return ResolutionResult(
            candidates=candidates,
            ranked_scores=normalized,
            top_intent_id=top_id,
            top_intent_score=normalized[top_id],
            conflict_detected=False,
            conflict_score=0.0,
            emergency_triggered=False,
            active_layer_count=0,
            layer_contributions={},
        )
