"""
Stage 4 — Confidence Evaluation and Output Generation.

Decision tree:
  1. Emergency override active      → always resolve, regardless of confidence
  2. High confidence + no conflict  → resolve to natural-language output
  3. Conflict detected              → partial resolve with conflict warning
  4. Medium confidence              → top-N fallback (show ranked candidates)
  5. Low confidence                 → literal translation fallback
"""
from dataclasses import dataclass, field
from typing import List, Dict

from .ontology import IntentCandidate
from .resolver import ResolutionResult

HIGH_CONF = 0.60    # Minimum score for clean resolution
MED_CONF  = 0.40    # Minimum score for top-N fallback


@dataclass
class IntentOutput:
    word_input: str
    resolved: bool              # True if a single intent was committed
    intent_id: str
    intent_label: str
    output_text: str            # Full natural-language output for screen display
    lcd_text: str               # ≤16-char version for Arduino LCD
    confidence: float           # 0-1 score of the top intent
    conflict_detected: bool
    emergency: bool
    fallback_mode: str          # "resolved" | "conflict" | "top_n" | "literal"
    top_n_candidates: List[dict]  # [{id, label, score_pct}] for non-resolved cases


class ConfidenceGate:

    def evaluate(
        self,
        word: str,
        result: ResolutionResult,
        candidates_map: Dict[str, IntentCandidate],
    ) -> IntentOutput:

        top_id = result.top_intent_id
        top_score = result.top_intent_score
        top = candidates_map.get(top_id)

        if top is None:
            return self._literal(word, result)

        # ── Emergency override ────────────────────────────────────────────────
        if result.emergency_triggered and top.emergency:
            return IntentOutput(
                word_input=word,
                resolved=True,
                intent_id=top_id,
                intent_label=top.label,
                output_text=top.output_text,
                lcd_text=top.lcd_text,
                confidence=top_score,
                conflict_detected=result.conflict_detected,
                emergency=True,
                fallback_mode="resolved",
                top_n_candidates=[],
            )

        # ── High confidence, no conflict ──────────────────────────────────────
        if top_score >= HIGH_CONF and not result.conflict_detected:
            return IntentOutput(
                word_input=word,
                resolved=True,
                intent_id=top_id,
                intent_label=top.label,
                output_text=top.output_text,
                lcd_text=top.lcd_text,
                confidence=top_score,
                conflict_detected=False,
                emergency=top.emergency,
                fallback_mode="resolved",
                top_n_candidates=[],
            )

        # ── Conflict detected ─────────────────────────────────────────────────
        if result.conflict_detected:
            top_n = self._top_n(result, candidates_map, n=3)
            return IntentOutput(
                word_input=word,
                resolved=False,
                intent_id=top_id,
                intent_label=top.label,
                output_text=f"[Ambiguous] {top.output_text}",
                lcd_text=f"?{top.lcd_text[:15]}",
                confidence=top_score,
                conflict_detected=True,
                emergency=False,
                fallback_mode="conflict",
                top_n_candidates=top_n,
            )

        # ── Medium confidence → top-N ─────────────────────────────────────────
        if top_score >= MED_CONF:
            top_n = self._top_n(result, candidates_map, n=3)
            return IntentOutput(
                word_input=word,
                resolved=False,
                intent_id=top_id,
                intent_label=top.label,
                output_text=top.output_text,
                lcd_text=top.lcd_text,
                confidence=top_score,
                conflict_detected=False,
                emergency=False,
                fallback_mode="top_n",
                top_n_candidates=top_n,
            )

        # ── Low confidence → literal ──────────────────────────────────────────
        return self._literal(word, result)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _top_n(
        self,
        result: ResolutionResult,
        candidates_map: Dict[str, IntentCandidate],
        n: int = 3,
    ) -> List[dict]:
        ranked = sorted(result.ranked_scores, key=result.ranked_scores.get, reverse=True)
        out = []
        for cid in ranked[:n]:
            c = candidates_map.get(cid)
            if c:
                out.append({
                    "id": cid,
                    "label": c.label,
                    "score_pct": round(result.ranked_scores[cid] * 100, 1),
                })
        return out

    def _literal(self, word: str, result: ResolutionResult) -> IntentOutput:
        return IntentOutput(
            word_input=word,
            resolved=False,
            intent_id="literal_translation",
            intent_label=f"Literal: {word}",
            output_text=word,
            lcd_text=word[:16],
            confidence=result.top_intent_score,
            conflict_detected=result.conflict_detected,
            emergency=False,
            fallback_mode="literal",
            top_n_candidates=[],
        )
