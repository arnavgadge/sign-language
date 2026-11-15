"""
evaluation/variants.py
--------------------------
Runs ONE evaluation sample through the baseline and each ablation
variant, using the REAL engine components built in Phases 1-6
(intent.ontology.IntentOntology, layer3_temporal.TemporalContextEngine,
layer5_personal.PersonalContextEngine, reasoning.IntentResolver,
decision_gate.ConfidenceDecisionGate). No scoring math is reimplemented
here — this module only decides WHICH real evidence to feed into the
REAL resolver for each variant, exactly the way reasoning.IntentResolver
already supports (temporal=None / personal=None).

Each sample gets fully ISOLATED temporal/personal state (fresh, in-memory
per sample) so results are deterministic and independent of dataset
ordering — a genuine methodological choice, disclosed in the Phase 8
report, not an attempt to make numbers look better.
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from intent.ontology import IntentOntology

from layer3_temporal.engine import TemporalContextEngine
from layer5_personal.engine import PersonalContextEngine

from reasoning.resolver import IntentResolver

from decision_gate.gate import ConfidenceDecisionGate

from .dataset import EvalSample

DAY = 86400.0
NOW = 1_700_000_000.0   # fixed reference "now" for full reproducibility

VARIANTS = ["baseline", "L1_only", "L1_temporal", "L1_personal", "full"]

_ontology = IntentOntology()   # stateless/read-only -> safe to share across samples


@dataclass
class VariantResult:
    sample_id: str
    category: str
    variant: str
    user_id: str
    recognized_word: str
    conversation_sequence: str
    ground_truth: str
    predicted_intent: str          # intent_id | "AMBIGUOUS" | "LITERAL"
    correct: bool
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


def _seed_personal(engine: PersonalContextEngine, sample: EvalSample) -> None:
    for word, intent_id, count, days_ago in sample.personal_seed:
        ts = NOW - days_ago * DAY
        for _ in range(count):
            engine.update(sample.user_id, word, intent_id, timestamp=ts)


def _seed_temporal_history(engine: TemporalContextEngine, sample: EvalSample) -> None:
    preceding = sample.sequence[:-1]
    n = len(preceding)
    for i, word in enumerate(preceding):
        seconds_before = (n - i) * 10.0   # 10s apart, most recent closest to "now"
        engine.record(sample.user_id, word, timestamp=NOW - seconds_before)


def _predicted_label(status: str, resolved_candidate: Optional[str]) -> str:
    if status == "resolved" and resolved_candidate:
        return resolved_candidate
    if status == "ambiguous":
        return "AMBIGUOUS"
    return "LITERAL"


def _evidence_for(candidate_id: str, resolution) -> tuple:
    """(signal, reliability) for candidate_id from a TemporalResolution or
    PersonalResolution, or (0.0, 0.0) if unavailable/candidate absent."""
    if resolution is None:
        return 0.0, 0.0
    for ev in resolution.candidates:
        if ev.candidate == candidate_id:
            return ev.signal, ev.reliability
    return 0.0, 0.0


def run_baseline(sample: EvalSample) -> VariantResult:
    """Literal interpretation: the recognized word is shown as-is, no
    intent is ever selected. Correct only when ground_truth == LITERAL —
    by definition, a context-free system cannot resolve ambiguity."""
    t0 = time.perf_counter()
    word = sample.sequence[-1]
    predicted = "LITERAL"
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return VariantResult(
        sample_id=sample.sample_id, category=sample.category, variant="baseline",
        user_id=sample.user_id, recognized_word=word,
        conversation_sequence=" > ".join(sample.sequence), ground_truth=sample.ground_truth,
        predicted_intent=predicted, correct=(predicted == sample.ground_truth),
        top_score=0.0, second_score=0.0, margin=0.0, status="LITERAL", conflict=False,
        temporal_signal=0.0, temporal_reliability=0.0, personal_signal=0.0, personal_reliability=0.0,
        processing_time_ms=elapsed_ms,
    )


def run_all_variants(sample: EvalSample) -> List[VariantResult]:
    """Runs baseline + all 4 ablation variants for one sample, using the
    real engine components, with fully isolated per-sample state."""
    t0 = time.perf_counter()

    temporal_engine = TemporalContextEngine()
    _seed_temporal_history(temporal_engine, sample)

    personal_engine = PersonalContextEngine(db_path=":memory:")
    _seed_personal(personal_engine, sample)

    word = sample.sequence[-1]
    candidates = _ontology.lookup(word)   # Layer 1 — real, shared, stateless

    temporal_result = temporal_engine.evaluate(sample.user_id, word, candidates, now=NOW)   # Layer 3 — real
    personal_result = personal_engine.evaluate(sample.user_id, word, candidates, now=NOW)   # Layer 5 — real

    total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

    results = [run_baseline(sample)]

    variant_sources = {
        "L1_only":     (None, None),
        "L1_temporal": (temporal_result, None),
        "L1_personal": (None, personal_result),
        "full":        (temporal_result, personal_result),
    }

    resolver = IntentResolver()
    gate = ConfidenceDecisionGate()

    for variant, (t_arg, p_arg) in variant_sources.items():
        reasoning_result = resolver.resolve(word, candidates, temporal=t_arg, personal=p_arg, now=NOW)
        decision = gate.evaluate(reasoning_result, now=NOW)

        ranking = decision.evidence["full_ranking"]
        top_score = ranking[0]["final_score"] if ranking else 0.0
        second_score = ranking[1]["final_score"] if len(ranking) > 1 else 0.0
        top_candidate_id = ranking[0]["candidate"] if ranking else None

        predicted = _predicted_label(decision.mode, decision.resolved_candidate)
        # For LITERAL/AMBIGUOUS the "top candidate" for evidence-reporting
        # purposes is still the highest-ranked one, even though it wasn't committed to.
        evidence_candidate = decision.resolved_candidate or top_candidate_id

        t_sig, t_rel = _evidence_for(evidence_candidate, temporal_result) if evidence_candidate else (0.0, 0.0)
        p_sig, p_rel = _evidence_for(evidence_candidate, personal_result) if evidence_candidate else (0.0, 0.0)

        results.append(VariantResult(
            sample_id=sample.sample_id, category=sample.category, variant=variant,
            user_id=sample.user_id, recognized_word=word,
            conversation_sequence=" > ".join(sample.sequence), ground_truth=sample.ground_truth,
            predicted_intent=predicted, correct=(predicted == sample.ground_truth),
            top_score=top_score, second_score=second_score, margin=decision.margin,
            status=predicted if predicted in ("AMBIGUOUS", "LITERAL") else "RESOLVED",
            conflict=decision.conflict_detected,
            temporal_signal=t_sig, temporal_reliability=t_rel,
            personal_signal=p_sig, personal_reliability=p_rel,
            processing_time_ms=total_elapsed_ms,
        ))

    personal_engine.close()
    return results
