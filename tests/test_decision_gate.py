"""
tests/test_decision_gate.py
------------------------------
Automated tests for the Confidence + Conflict Decision Gate.

Run with:  venv310\\Scripts\\python.exe -m unittest tests.test_decision_gate -v
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent.ontology import IntentOntology
from layer3_temporal.engine import TemporalContextEngine
from layer5_personal.engine import PersonalContextEngine
from reasoning.resolver import IntentResolver
from reasoning.models import CandidateResult, ConflictReport, ReasoningResult

from decision_gate.config import DecisionGateConfig, MODE_RESOLVED, MODE_AMBIGUOUS, MODE_LITERAL_FALLBACK
from decision_gate.gate import ConfidenceDecisionGate

NOW = 1_700_000_000.0
ONTOLOGY = IntentOntology(ontology_path=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "intent_ontology.json"
))


def _result(scores: list, conflict_detected: bool = False, conflict_score: float = 0.0) -> ReasoningResult:
    """Build a ReasoningResult directly from [(candidate_id, label, final_score), ...]
    (already assumed sorted highest-first) for controlled gate tests."""
    candidates = [
        CandidateResult(candidate=cid, label=label, final_score=score, sources={}, rank=i + 1)
        for i, (cid, label, score) in enumerate(scores)
    ]
    return ReasoningResult(
        word="HELP", evaluated_at=NOW, strategy_name="reliability_weighted",
        available_sources=["temporal", "personal"], candidates=candidates,
        conflict=ConflictReport(conflict_detected=conflict_detected, conflict_score=conflict_score, disagreements=[]),
    )


class TestClearResolved(unittest.TestCase):
    def test_high_confidence_big_margin_resolves(self):
        result = _result([("medical_assistance", "Medical Assistance", 0.85),
                           ("task_assistance", "Task Assistance", 0.30)])
        gate = ConfidenceDecisionGate()
        decision = gate.evaluate(result, now=NOW)

        self.assertEqual(decision.mode, MODE_RESOLVED)
        self.assertEqual(decision.resolved_candidate, "medical_assistance")
        self.assertAlmostEqual(decision.confidence, 0.85, places=4)
        self.assertAlmostEqual(decision.margin, 0.55, places=4)
        self.assertEqual(decision.top_n, [])   # top_n only populated for ambiguous


class TestConflictForcesAmbiguous(unittest.TestCase):
    def test_conflict_overrides_high_confidence(self):
        result = _result([("medical_assistance", "Medical Assistance", 0.90),
                           ("task_assistance", "Task Assistance", 0.20)],
                          conflict_detected=True, conflict_score=0.81)
        gate = ConfidenceDecisionGate()
        decision = gate.evaluate(result, now=NOW)

        self.assertEqual(decision.mode, MODE_AMBIGUOUS)
        self.assertIsNone(decision.resolved_candidate)
        self.assertTrue(decision.conflict_detected)
        self.assertGreater(len(decision.top_n), 0)


class TestCloseMarginIsAmbiguous(unittest.TestCase):
    def test_narrow_margin_prevents_resolution(self):
        result = _result([("medical_assistance", "Medical Assistance", 0.70),
                           ("emergency_support", "Emergency Support", 0.68)])
        gate = ConfidenceDecisionGate()
        decision = gate.evaluate(result, now=NOW)

        self.assertEqual(decision.mode, MODE_AMBIGUOUS)
        self.assertAlmostEqual(decision.margin, 0.02, places=4)


class TestLowConfidenceFallback(unittest.TestCase):
    def test_weak_top_score_falls_back_to_literal(self):
        result = _result([("medical_assistance", "Medical Assistance", 0.10),
                           ("task_assistance", "Task Assistance", 0.08)])
        gate = ConfidenceDecisionGate()
        decision = gate.evaluate(result, now=NOW)

        self.assertEqual(decision.mode, MODE_LITERAL_FALLBACK)
        self.assertIsNone(decision.resolved_candidate)
        self.assertEqual(decision.top_n, [])


class TestEmptyCandidates(unittest.TestCase):
    def test_no_candidates_does_not_crash(self):
        result = _result([])
        gate = ConfidenceDecisionGate()
        decision = gate.evaluate(result, now=NOW)

        self.assertEqual(decision.mode, MODE_LITERAL_FALLBACK)
        self.assertEqual(decision.confidence, 0.0)


class TestSingleCandidateMargin(unittest.TestCase):
    def test_single_candidate_margin_equals_its_own_score(self):
        result = _result([("medical_assistance", "Medical Assistance", 0.80)])
        gate = ConfidenceDecisionGate()
        decision = gate.evaluate(result, now=NOW)

        self.assertEqual(decision.margin, decision.confidence)
        self.assertEqual(decision.mode, MODE_RESOLVED)


class TestThresholdConfigurability(unittest.TestCase):
    def test_same_input_different_config_different_mode(self):
        result = _result([("medical_assistance", "Medical Assistance", 0.60),
                           ("task_assistance", "Task Assistance", 0.40)])

        lenient = ConfidenceDecisionGate(DecisionGateConfig(high_confidence_threshold=0.55, min_margin=0.10))
        strict = ConfidenceDecisionGate(DecisionGateConfig(high_confidence_threshold=0.90, min_margin=0.10))

        self.assertEqual(lenient.evaluate(result, now=NOW).mode, MODE_RESOLVED)
        self.assertEqual(strict.evaluate(result, now=NOW).mode, MODE_AMBIGUOUS)


class TestTopNContent(unittest.TestCase):
    def test_top_n_capped_and_ordered(self):
        result = _result([
            ("medical_assistance", "Medical Assistance", 0.50),
            ("emergency_support", "Emergency Support", 0.48),
            ("task_assistance", "Task Assistance", 0.30),
            ("information_request", "Information Request", 0.10),
        ])
        cfg = DecisionGateConfig(top_n_count=2)
        gate = ConfidenceDecisionGate(cfg)
        decision = gate.evaluate(result, now=NOW)

        self.assertEqual(decision.mode, MODE_AMBIGUOUS)  # margin 0.02 too close
        self.assertEqual(len(decision.top_n), 2)
        self.assertEqual(decision.top_n[0]["candidate"], "medical_assistance")
        self.assertEqual(decision.top_n[1]["candidate"], "emergency_support")


class TestReasonIsExplainable(unittest.TestCase):
    def test_reason_mentions_actual_numbers(self):
        result = _result([("medical_assistance", "Medical Assistance", 0.85),
                           ("task_assistance", "Task Assistance", 0.10)])
        gate = ConfidenceDecisionGate()
        decision = gate.evaluate(result, now=NOW)
        self.assertIn("0.85", decision.reason)
        self.assertIn("RESOLVED", decision.reason)


class TestFullIntegration(unittest.TestCase):
    """Real Layer 1 -> Layer 3 -> Layer 5 -> Reasoning Engine -> Decision Gate."""

    def test_end_to_end_resolves_medical(self):
        candidates = ONTOLOGY.lookup("HELP")
        now = time.time()

        temporal_engine = TemporalContextEngine()
        temporal_engine.record("gate_user", "CHEST", timestamp=now - 20)
        temporal_engine.record("gate_user", "PAIN", timestamp=now - 10)
        temporal_result = temporal_engine.evaluate("gate_user", "HELP", candidates, now=now)

        personal_engine = PersonalContextEngine(db_path=":memory:")
        for _ in range(18):
            personal_engine.update("gate_user", "HELP", "medical_assistance", timestamp=now - 5 * 86400)
        for _ in range(3):
            personal_engine.update("gate_user", "HELP", "emergency_support", timestamp=now - 5 * 86400)
        personal_result = personal_engine.evaluate("gate_user", "HELP", candidates, now=now)

        resolver = IntentResolver()
        reasoning_result = resolver.resolve("HELP", candidates, temporal=temporal_result,
                                             personal=personal_result, now=now)

        gate = ConfidenceDecisionGate()
        decision = gate.evaluate(reasoning_result, now=now)

        self.assertEqual(decision.mode, MODE_RESOLVED)
        self.assertEqual(decision.resolved_candidate, "medical_assistance")
        self.assertFalse(decision.conflict_detected)

    def test_end_to_end_no_context_falls_back(self):
        """No temporal/personal evidence at all (both None) -> Reasoning Engine
        gives every candidate final_score=0.0 -> gate must fall back, not resolve."""
        candidates = ONTOLOGY.lookup("HELP")
        resolver = IntentResolver()
        reasoning_result = resolver.resolve("HELP", candidates, temporal=None, personal=None, now=NOW)

        gate = ConfidenceDecisionGate()
        decision = gate.evaluate(reasoning_result, now=NOW)

        self.assertEqual(decision.mode, MODE_LITERAL_FALLBACK)


class TestPerformance(unittest.TestCase):
    def test_evaluate_is_fast(self):
        import time as _time
        result = _result([("medical_assistance", "Medical Assistance", 0.85),
                           ("task_assistance", "Task Assistance", 0.10)])
        gate = ConfidenceDecisionGate()

        n = 200
        start = _time.perf_counter()
        for _ in range(n):
            gate.evaluate(result, now=NOW)
        avg_ms = (_time.perf_counter() - start) * 1000.0 / n
        print(f"\n[LATENCY] ConfidenceDecisionGate.evaluate(): avg {avg_ms:.4f} ms over {n} calls")
        self.assertLess(avg_ms, 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
