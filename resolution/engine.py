"""
resolution/engine.py
------------------------
THE SINGLE ENGINE ENTRY POINT (Step 2).

    intent_engine.resolve(user_id, recognized_word) -> ResolutionResult

Internally chains:

    Layer 1 (intent.ontology.IntentOntology)
        -> Layer 3 (layer3_temporal.TemporalContextEngine)
        -> Layer 5 (layer5_personal.PersonalContextEngine)
        -> Reasoning Engine (reasoning.IntentResolver)
        -> Decision Gate (decision_gate.ConfidenceDecisionGate)

The existing recognition system does not need to know any of this exists
— it hands over a word and a user id, and gets a ResolutionResult back.

This module does NOT duplicate any layer's database logic — it only
calls each layer's own public interface (evaluate/record/update) and
owns exactly one thing none of them owns: the resolution_log table
(resolution/store.py), plus the thin orchestration/mapping glue below.
"""
import logging
import time
from typing import Optional

from intent.ontology import IntentOntology

from layer3_temporal.engine import TemporalContextEngine
from layer3_temporal.config import Layer3Config

from layer5_personal.engine import PersonalContextEngine
from layer5_personal.config import Layer5Config

from reasoning.resolver import IntentResolver
from reasoning.config import ReasoningConfig

from decision_gate.gate import ConfidenceDecisionGate
from decision_gate.config import DecisionGateConfig, MODE_RESOLVED, MODE_AMBIGUOUS, MODE_LITERAL_FALLBACK

from .models import ResolutionResult, EngineTrace
from .store import ResolutionStore

logger = logging.getLogger("resolution.engine")

_STATUS_MAP = {
    MODE_RESOLVED: "RESOLVED",
    MODE_AMBIGUOUS: "AMBIGUOUS",
    MODE_LITERAL_FALLBACK: "LITERAL",
}


class IntentEngine:
    """
    Usage
    -----
    engine = IntentEngine()
    result = engine.resolve(user_id="default", recognized_word="HELP")

    if result.status == "RESOLVED":
        print(result.output)
    elif result.status == "AMBIGUOUS":
        for c in result.top_n:
            print(c["label"], c["final_score"])
    else:
        print(result.recognized_word)   # LITERAL
    """

    def __init__(self,
                 ontology_path: str = "data/intent_ontology.json",
                 temporal_config: Layer3Config = None,
                 personal_config: Layer5Config = None,
                 personal_db_path: str = None,
                 reasoning_config: ReasoningConfig = None,
                 gate_config: DecisionGateConfig = None,
                 resolution_db_path: str = None):
        self.ontology = IntentOntology(ontology_path=ontology_path)

        self.temporal_engine = TemporalContextEngine(config=temporal_config)

        if personal_db_path:
            self.personal_engine = PersonalContextEngine(config=personal_config, db_path=personal_db_path)
        else:
            self.personal_engine = PersonalContextEngine(config=personal_config)

        self.resolver = IntentResolver(config=reasoning_config)
        self.gate = ConfidenceDecisionGate(config=gate_config)

        if resolution_db_path:
            self.store = ResolutionStore(db_path=resolution_db_path)
        else:
            self.store = ResolutionStore()

        # Most recent EngineTrace (Phase 7) — a passthrough bundle of the
        # real objects each layer already computed during the last
        # resolve() call, for a visualization layer to read. None until
        # the first resolve() call. Not history — ResolutionStore already
        # owns persisted history (Step 6); this is only the latest trace.
        self.last_trace: EngineTrace = None

    # ── Public API ─────────────────────────────────────────────────────────

    def resolve(self, user_id: str, recognized_word: str, now: float = None) -> ResolutionResult:
        eval_time = now if now is not None else time.time()
        word = recognized_word.upper().strip()

        # Layer 1 — candidate generation
        candidates = self.ontology.lookup(word)

        # Layer 3 — temporal context (reads history BEFORE this word is recorded)
        temporal_result = self.temporal_engine.evaluate(user_id, word, candidates, now=eval_time)

        # Layer 5 — personal context
        personal_result = self.personal_engine.evaluate(user_id, word, candidates, now=eval_time)

        # Reasoning Engine — reliability-weighted fusion
        reasoning_result = self.resolver.resolve(word, candidates, temporal=temporal_result,
                                                  personal=personal_result, now=eval_time)

        # Confidence + Conflict Decision Gate
        gate_decision = self.gate.evaluate(reasoning_result, now=eval_time)
        status = _STATUS_MAP[gate_decision.mode]

        # Step 3 — record this word into temporal history AFTER evaluating it,
        # so the next word's evaluation sees it, but this word's own
        # evidence was never evaluated against itself.
        self.temporal_engine.record(user_id, word,
                                     resolved_intent=gate_decision.resolved_candidate or "",
                                     timestamp=eval_time)

        # Step 7 — only update the long-term personal profile on a genuinely
        # resolved outcome; ambiguous/literal results are not treated as
        # ground truth about what the user meant.
        if status == "RESOLVED":
            self.personal_engine.update(user_id, word, gate_decision.resolved_candidate, timestamp=eval_time)

        result = self._build_result(user_id, word, eval_time, status, gate_decision, candidates)

        # Step 6 — persist every completed resolution.
        self.store.record(result)

        # Phase 7 — retain the real intermediate objects for the dashboard.
        # No new computation happens here; every field is a reference to
        # something already built above.
        self.last_trace = EngineTrace(
            user_id=user_id, word=word, evaluated_at=eval_time,
            candidates=candidates, temporal_result=temporal_result,
            personal_result=personal_result, reasoning_result=reasoning_result,
            gate_decision=gate_decision, resolution=result,
        )

        self._log_trace(user_id, word, candidates, temporal_result, personal_result,
                         reasoning_result, gate_decision, result)
        return result

    def close(self) -> None:
        self.personal_engine.close()
        self.store.close()

    # ── Internal ───────────────────────────────────────────────────────────

    def _build_result(self, user_id, word, eval_time, status, gate_decision, candidates) -> ResolutionResult:
        label_map = {c.id: c.label for c in candidates}
        output_map = {c.id: c.output_text for c in candidates}
        lcd_map = {c.id: c.lcd_text for c in candidates}

        selected_intent: Optional[str] = None
        selected_label: Optional[str] = None
        selected_lcd_text = word[:16]
        output = word

        if status == "RESOLVED" and gate_decision.resolved_candidate:
            selected_intent = gate_decision.resolved_candidate
            selected_label = gate_decision.resolved_label
            output = output_map.get(selected_intent, word)
            selected_lcd_text = lcd_map.get(selected_intent, word[:16])

        return ResolutionResult(
            user_id=user_id,
            recognized_word=word,
            evaluated_at=eval_time,
            status=status,
            selected_intent=selected_intent,
            selected_label=selected_label,
            selected_lcd_text=selected_lcd_text,
            final_score=gate_decision.confidence,
            conflict=gate_decision.conflict_detected,
            output=output,
            ranked_candidates=gate_decision.evidence["full_ranking"],
            top_n=gate_decision.top_n,
            reason=gate_decision.reason,
        )

    # ── Observability ──────────────────────────────────────────────────────

    def _log_trace(self, user_id, word, candidates, temporal_result, personal_result,
                    reasoning_result, gate_decision, result) -> None:
        logger.info("=" * 50)
        logger.info("INTENT ENGINE TRACE")
        logger.info("=" * 50)
        logger.info("User:\n    %s", user_id)
        logger.info("Recognized:\n    %s", word)

        logger.info("Layer 1:")
        logger.info("    Candidates:")
        for c in candidates:
            logger.info("        %s", c.id)

        logger.info("Layer 3:")
        logger.info("    Relevant history:\n        %s", temporal_result.window_words)
        for ev in temporal_result.candidates:
            if ev.evidence.get("relevant_words"):
                logger.info("    Evidence[%s]: signal=%.4f reliability=%.4f relevant=%s",
                             ev.candidate, ev.signal, ev.reliability, ev.evidence["relevant_words"])

        logger.info("Layer 5:")
        logger.info("    Historical profile: total_observations=%d, distinct_words=%d",
                     personal_result.profile_total_observations, personal_result.profile_distinct_words)
        for ev in personal_result.candidates:
            if ev.evidence.get("historical_count"):
                logger.info("    Evidence[%s]: signal=%.4f reliability=%.4f count=%d",
                             ev.candidate, ev.signal, ev.reliability, ev.evidence["historical_count"])

        logger.info("Reasoning:")
        logger.info("    Conflict: %s (score=%.4f)",
                     reasoning_result.conflict.conflict_detected, reasoning_result.conflict.conflict_score)

        logger.info("Ranking:")
        for c in reasoning_result.candidates:
            logger.info("    %d. %s (%.4f)", c.rank, c.candidate, c.final_score)

        logger.info("Decision:\n    %s", result.status)
        logger.info("Final output:\n    %s", result.output)
        logger.info("=" * 50)
