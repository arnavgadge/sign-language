"""
decision_gate/gate.py
-------------------------
CONFIDENCE + CONFLICT DECISION GATE.

Purpose: decide what the system should DO with a ranked, reliability-
weighted list of candidate intents (from the Reasoning Engine) — never
present a weak or ambiguous interpretation as if it were confidently
resolved.

Pipeline position:
    Recognized Word -> Layer 1 -> Layer 3/5 -> Reasoning Engine
        -> Ranked Intent Candidates + Conflict Report
        -> THIS MODULE -> GateDecision (final decision for the system to act on)

Decision logic (deterministic, in this priority order):

  1. LOW CONFIDENCE  — top candidate's final_score < low_confidence_threshold
                        -> LITERAL_FALLBACK (no usable evidence at all;
                        showing "top options" of near-zero scores isn't
                        useful either)

  2. CONFLICT         — Reasoning Engine flagged conflict_detected=True
                        -> AMBIGUOUS, even if the top score alone looks
                        confident (sources disagreeing must never be
                        silently hidden behind a single answer)

  3. HIGH CONFIDENCE  — top candidate's final_score >= high_confidence_threshold
                        AND margin over rank-2 >= min_margin
                        -> RESOLVED

  4. otherwise        -> AMBIGUOUS (moderate confidence or too close to call)

This module does not implement Layer 2, Layer 4, or any new ML model —
purely deterministic threshold logic over numbers the Reasoning Engine
already computed.
"""
import logging
import time
from typing import List

from reasoning.models import CandidateResult, ReasoningResult

from .config import (
    DecisionGateConfig,
    MODE_RESOLVED, MODE_AMBIGUOUS, MODE_LITERAL_FALLBACK,
)
from .models import GateDecision

logger = logging.getLogger("decision_gate")


class ConfidenceDecisionGate:
    """
    Usage
    -----
    gate = ConfidenceDecisionGate()
    decision = gate.evaluate(reasoning_result)   # ReasoningResult from reasoning.IntentResolver

    if decision.mode == "resolved":
        act_on(decision.resolved_candidate)
    elif decision.mode == "ambiguous":
        show_options(decision.top_n)
    else:
        fall_back_to_literal_word(decision.word)
    """

    def __init__(self, config: DecisionGateConfig = None):
        self.config = config or DecisionGateConfig()

    def evaluate(self, reasoning_result: ReasoningResult, now: float = None) -> GateDecision:
        cfg = self.config
        candidates: List[CandidateResult] = reasoning_result.candidates
        eval_time = now if now is not None else time.time()

        if not candidates:
            decision = self._build(reasoning_result, eval_time, MODE_LITERAL_FALLBACK,
                                    confidence=0.0, margin=0.0,
                                    reason="No candidates were supplied by Layer 1 — nothing to resolve.")
            self._log(decision)
            return decision

        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        confidence = top.final_score
        margin = (confidence - second.final_score) if second is not None else confidence
        conflict_detected = reasoning_result.conflict.conflict_detected

        if confidence < cfg.low_confidence_threshold:
            mode = MODE_LITERAL_FALLBACK
            reason = (f"Top candidate '{top.candidate}' confidence {confidence:.4f} "
                      f"< low_confidence_threshold {cfg.low_confidence_threshold} -> LITERAL_FALLBACK.")
        elif conflict_detected:
            mode = MODE_AMBIGUOUS
            reason = (f"Conflict detected between evidence sources "
                      f"(conflict_score={reasoning_result.conflict.conflict_score:.4f}) -> AMBIGUOUS, "
                      f"despite top candidate confidence {confidence:.4f}.")
        elif confidence >= cfg.high_confidence_threshold and margin >= cfg.min_margin:
            mode = MODE_RESOLVED
            reason = (f"Top candidate '{top.candidate}' confidence {confidence:.4f} "
                      f">= high_confidence_threshold {cfg.high_confidence_threshold}, "
                      f"margin {margin:.4f} >= min_margin {cfg.min_margin}, no conflict -> RESOLVED.")
        else:
            mode = MODE_AMBIGUOUS
            reason = (f"Top candidate '{top.candidate}' confidence {confidence:.4f} and margin {margin:.4f} "
                      f"did not both clear the resolved thresholds "
                      f"(need >= {cfg.high_confidence_threshold} confidence and "
                      f">= {cfg.min_margin} margin) -> AMBIGUOUS.")

        decision = self._build(reasoning_result, eval_time, mode, confidence, margin,
                                reason, top=top, second=second, conflict_detected=conflict_detected)
        self._log(decision)
        return decision

    # ── Internal ───────────────────────────────────────────────────────────

    def _build(self, reasoning_result: ReasoningResult, eval_time: float, mode: str,
               confidence: float, margin: float, reason: str,
               top: CandidateResult = None, second: CandidateResult = None,
               conflict_detected: bool = False) -> GateDecision:
        cfg = self.config

        top_n = []
        if mode == MODE_AMBIGUOUS:
            for c in reasoning_result.candidates[:cfg.top_n_count]:
                top_n.append({
                    "candidate": c.candidate, "label": c.label,
                    "final_score": c.final_score, "rank": c.rank,
                })

        evidence = {
            "full_ranking": [
                {"candidate": c.candidate, "label": c.label, "final_score": c.final_score, "rank": c.rank}
                for c in reasoning_result.candidates
            ],
            "top_candidate": top.candidate if top else None,
            "second_candidate": second.candidate if second else None,
            "conflict_score": reasoning_result.conflict.conflict_score,
            "available_sources": reasoning_result.available_sources,
            "thresholds": cfg.to_dict(),
        }

        return GateDecision(
            word=reasoning_result.word,
            evaluated_at=eval_time,
            mode=mode,
            resolved_candidate=top.candidate if (mode == MODE_RESOLVED and top) else None,
            resolved_label=top.label if (mode == MODE_RESOLVED and top) else None,
            confidence=round(confidence, 4),
            margin=round(margin, 4),
            conflict_detected=conflict_detected,
            top_n=top_n,
            reason=reason,
            evidence=evidence,
        )

    # ── Observability ──────────────────────────────────────────────────────

    def _log(self, decision: GateDecision) -> None:
        logger.info("WORD: %s", decision.word)
        logger.info("MODE: %s", decision.mode.upper())
        logger.info("Confidence: %.4f  Margin: %.4f  Conflict: %s",
                     decision.confidence, decision.margin, decision.conflict_detected)
        logger.info("Reason: %s", decision.reason)
        if decision.mode == MODE_RESOLVED:
            logger.info("RESOLVED -> %s (%s)", decision.resolved_candidate, decision.resolved_label)
        elif decision.mode == MODE_AMBIGUOUS:
            logger.info("TOP-N:")
            for item in decision.top_n:
                logger.info("  %d. %s (%.4f)", item["rank"], item["label"], item["final_score"])
        else:
            logger.info("LITERAL FALLBACK -> '%s'", decision.word)
