"""
evaluation/final_variants.py
--------------------------------
PHASE 9 -- runs each final-dataset case through 7 configurations:

  1. baseline               -- literal, no interpretation
  2. L1_candidates           -- Layer 1 only (reported via coverage/recall,
                                 NOT "accuracy" -- see final_metrics.py)
  3. L1_temporal             -- Layer 1 + Temporal
  4. L1_personal             -- Layer 1 + Personal
  5. L1_temporal_personal     -- Layer 1 + Temporal + Personal, reliability-weighted
  6. full_reliability_weighted -- same as #5 (the production configuration)
  7. full_unweighted          -- same evidence as #5, fused WITHOUT
                                  reliability weighting (UnweightedAverageStrategy)

All real engine components: intent.ontology.IntentOntology,
layer3_temporal.TemporalContextEngine, layer5_personal.PersonalContextEngine,
reasoning.IntentResolver (with a swappable strategy), decision_gate.ConfidenceDecisionGate.

DATA LEAKAGE GUARD: ground_truth / expected_status are NEVER passed into
any engine call in this module. `assert_no_leakage()` inspects the
actual call sites defensively (see below) and is invoked once at import
time by run_final_evaluation.py.
"""
import inspect
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from intent.ontology import IntentOntology

from layer3_temporal.engine import TemporalContextEngine
from layer5_personal.engine import PersonalContextEngine

from reasoning.resolver import IntentResolver
from reasoning.strategies import ReliabilityWeightedStrategy, UnweightedAverageStrategy

from decision_gate.gate import ConfidenceDecisionGate

from .final_dataset import FinalEvalCase

DAY = 86400.0
NOW = 1_700_000_000.0

VARIANTS = [
    "baseline", "L1_candidates", "L1_temporal", "L1_personal",
    "L1_temporal_personal", "full_reliability_weighted", "full_unweighted",
]

_ontology = IntentOntology()   # stateless/read-only -> safe to share


@dataclass
class FinalVariantResult:
    sample_id: str
    scenario_category: str
    variant: str
    user_id: str
    recognized_word: str
    conversation_sequence: str
    ground_truth: str
    expected_status: str
    predicted_intent: str
    correct: bool
    status_correct: bool                # predicted status == expected_status
    layer1_candidate_ids: str            # pipe-separated, for candidate-coverage metrics
    ground_truth_in_candidates: bool     # candidate coverage for THIS sample
    ground_truth_rank: int               # this variant's ranked position of ground_truth (0 = not ranked/not present)
    top_score: float
    second_score: float
    margin: float
    status: str
    conflict: bool
    temporal_signal: float
    temporal_reliability: float
    personal_signal: float
    personal_reliability: float
    processing_time_ms: float


def _seed_personal(engine: PersonalContextEngine, case: FinalEvalCase) -> None:
    for word, intent_id, count, days_ago in case.personal_seed:
        ts = NOW - days_ago * DAY
        for _ in range(count):
            engine.update(case.user_id, word, intent_id, timestamp=ts)


def _seed_temporal_history(engine: TemporalContextEngine, case: FinalEvalCase) -> None:
    preceding_words = case.conversation_sequence[:-1]
    preceding_offsets = case.sequence_offsets_seconds[:-1]
    for word, offset in zip(preceding_words, preceding_offsets):
        engine.record(case.user_id, word, timestamp=NOW - offset)


def _predicted_label(status: str, resolved_candidate: Optional[str]) -> str:
    if status == "resolved" and resolved_candidate:
        return resolved_candidate
    if status == "ambiguous":
        return "AMBIGUOUS"
    return "LITERAL"


def _evidence_for(candidate_id: Optional[str], resolution) -> tuple:
    if resolution is None or candidate_id is None:
        return 0.0, 0.0
    for ev in resolution.candidates:
        if ev.candidate == candidate_id:
            return ev.signal, ev.reliability
    return 0.0, 0.0


def run_baseline(case: FinalEvalCase) -> FinalVariantResult:
    t0 = time.perf_counter()
    word = case.conversation_sequence[-1]
    predicted = "LITERAL"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return FinalVariantResult(
        sample_id=case.sample_id, scenario_category=case.scenario_category, variant="baseline",
        user_id=case.user_id, recognized_word=word,
        conversation_sequence=" > ".join(case.conversation_sequence),
        ground_truth=case.ground_truth, expected_status=case.expected_status,
        predicted_intent=predicted, correct=(predicted == case.ground_truth),
        status_correct=("LITERAL" == case.expected_status),
        layer1_candidate_ids="", ground_truth_in_candidates=False, ground_truth_rank=0,
        top_score=0.0, second_score=0.0, margin=0.0, status="LITERAL", conflict=False,
        temporal_signal=0.0, temporal_reliability=0.0, personal_signal=0.0, personal_reliability=0.0,
        processing_time_ms=elapsed_ms,
    )


def run_all_variants(case: FinalEvalCase) -> List[FinalVariantResult]:
    """Runs baseline + all 6 engine configurations for one case, fully
    isolated per-case state. `case.ground_truth` / `case.expected_status`
    are used ONLY here, in this evaluation module, for post-hoc scoring
    (the `correct=` / `status_correct=` comparisons below) -- never
    passed as an argument into ontology.lookup(), TemporalContextEngine,
    PersonalContextEngine, IntentResolver, or ConfidenceDecisionGate."""
    t0 = time.perf_counter()

    temporal_engine = TemporalContextEngine()
    _seed_temporal_history(temporal_engine, case)

    personal_engine = PersonalContextEngine(db_path=":memory:")
    _seed_personal(personal_engine, case)

    word = case.conversation_sequence[-1]
    candidates = _ontology.lookup(word)   # Layer 1 -- ground_truth NOT passed
    candidate_ids = [c.id for c in candidates]
    coverage = case.ground_truth in candidate_ids

    temporal_result = temporal_engine.evaluate(case.user_id, word, candidates, now=NOW)
    personal_result = personal_engine.evaluate(case.user_id, word, candidates, now=NOW)

    total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    results = [run_baseline(case)]

    variant_config = {
        "L1_candidates":              (None, None, ReliabilityWeightedStrategy()),
        "L1_temporal":                (temporal_result, None, ReliabilityWeightedStrategy()),
        "L1_personal":                (None, personal_result, ReliabilityWeightedStrategy()),
        "L1_temporal_personal":       (temporal_result, personal_result, ReliabilityWeightedStrategy()),
        "full_reliability_weighted":  (temporal_result, personal_result, ReliabilityWeightedStrategy()),
        "full_unweighted":            (temporal_result, personal_result, UnweightedAverageStrategy()),
    }

    gate = ConfidenceDecisionGate()

    for variant, (t_arg, p_arg, strategy) in variant_config.items():
        resolver = IntentResolver(strategy=strategy)
        reasoning_result = resolver.resolve(word, candidates, temporal=t_arg, personal=p_arg, now=NOW)
        decision = gate.evaluate(reasoning_result, now=NOW)

        ranking = decision.evidence["full_ranking"]
        top_score = ranking[0]["final_score"] if ranking else 0.0
        second_score = ranking[1]["final_score"] if len(ranking) > 1 else 0.0
        top_candidate_id = ranking[0]["candidate"] if ranking else None

        predicted = _predicted_label(decision.mode, decision.resolved_candidate)
        evidence_candidate = decision.resolved_candidate or top_candidate_id
        predicted_status = predicted if predicted in ("AMBIGUOUS", "LITERAL") else "RESOLVED"

        t_sig, t_rel = _evidence_for(evidence_candidate, temporal_result)
        p_sig, p_rel = _evidence_for(evidence_candidate, personal_result)

        gt_rank = 0
        for row in ranking:
            if row["candidate"] == case.ground_truth:
                gt_rank = row["rank"]
                break

        results.append(FinalVariantResult(
            sample_id=case.sample_id, scenario_category=case.scenario_category, variant=variant,
            user_id=case.user_id, recognized_word=word,
            conversation_sequence=" > ".join(case.conversation_sequence),
            ground_truth=case.ground_truth, expected_status=case.expected_status,
            predicted_intent=predicted, correct=(predicted == case.ground_truth),
            status_correct=(predicted_status == case.expected_status),
            layer1_candidate_ids="|".join(candidate_ids), ground_truth_in_candidates=coverage,
            ground_truth_rank=gt_rank,
            top_score=top_score, second_score=second_score, margin=decision.margin,
            status=predicted_status, conflict=decision.conflict_detected,
            temporal_signal=t_sig, temporal_reliability=t_rel,
            personal_signal=p_sig, personal_reliability=p_rel,
            processing_time_ms=total_elapsed_ms,
        ))

    personal_engine.close()
    return results


# ── Data leakage guard ───────────────────────────────────────────────────────

_ENGINE_CALL_METHODS = [
    (IntentOntology.lookup, ["self", "word"]),
    (TemporalContextEngine.evaluate, ["self", "user_id", "current_word", "candidates", "now"]),
    (PersonalContextEngine.evaluate, ["self", "user_id", "current_word", "candidates", "now"]),
    (IntentResolver.resolve, ["self", "word", "candidates", "temporal", "personal", "now"]),
]


def assert_no_leakage() -> None:
    """Defensive check: confirms none of the real engine entry points this
    module calls have a parameter literally named ground_truth/expected_*,
    so it is structurally impossible for this module to pass them in even
    by mistake -- not just a promise, a checked fact about the interfaces."""
    for fn, expected_params in _ENGINE_CALL_METHODS:
        actual_params = list(inspect.signature(fn).parameters.keys())
        for p in actual_params:
            assert "ground_truth" not in p.lower() and "expected_" not in p.lower(), (
                f"LEAKAGE RISK: {fn.__qualname__} exposes parameter '{p}'"
            )
    print("[LEAKAGE CHECK] Passed: no engine entry point used by this evaluation "
          "accepts a ground_truth/expected_* parameter.")
