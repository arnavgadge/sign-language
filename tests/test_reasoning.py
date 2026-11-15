"""
tests/test_reasoning.py
--------------------------
Automated tests for the Reasoning Engine, covering:
  - unit tests of the fusion mathematics in isolation (no layers involved)
  - the 8 required end-to-end scenarios
  - a full Layer 1 -> Layer 3 -> Layer 5 -> Reasoning Engine integration test

Run with:  venv310\\Scripts\\python.exe -m unittest tests.test_reasoning -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent.ontology import IntentOntology
from layer3_temporal.engine import TemporalContextEngine
from layer3_temporal.schema import TemporalEvidence, TemporalResolution
from layer5_personal.engine import PersonalContextEngine
from layer5_personal.schema import PersonalEvidence, PersonalResolution

from reasoning.config import ReasoningConfig
from reasoning.conflict import ConflictDetector
from reasoning.models import EvidenceItem
from reasoning.resolver import IntentResolver
from reasoning.strategies import ReliabilityWeightedStrategy

NOW = 1_700_000_000.0
ONTOLOGY = IntentOntology(ontology_path=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "intent_ontology.json"
))


def _help_candidates():
    return ONTOLOGY.lookup("HELP")


def _find(result, candidate_id):
    for c in result.candidates:
        if c.candidate == candidate_id:
            return c
    raise AssertionError(f"candidate {candidate_id} not found")


def _temporal_resolution(pairs: dict) -> TemporalResolution:
    """Build a TemporalResolution directly from {candidate_id: (signal, reliability)}
    for controlled unit/scenario tests, without running the real temporal engine."""
    evidence = [
        TemporalEvidence(candidate=cid, signal=sig, reliability=rel, evidence={})
        for cid, (sig, rel) in pairs.items()
    ]
    return TemporalResolution(
        user_id="test", current_word="HELP", evaluated_at=NOW,
        window_seconds=120.0, window_word_count=0, window_words=[], candidates=evidence,
    )


def _personal_resolution(pairs: dict) -> PersonalResolution:
    evidence = [
        PersonalEvidence(candidate=cid, signal=sig, reliability=rel, evidence={})
        for cid, (sig, rel) in pairs.items()
    ]
    return PersonalResolution(
        user_id="test", current_word="HELP", evaluated_at=NOW,
        profile_total_observations=0, profile_distinct_words=0, candidates=evidence,
    )


class TestScoringMathematics(unittest.TestCase):
    """Independently verify the reliability-weighted formula, no layers involved."""

    def test_two_source_weighted_average(self):
        strategy = ReliabilityWeightedStrategy()
        evidence = {
            "temporal": EvidenceItem(signal=0.8, reliability=0.9),
            "personal": EvidenceItem(signal=0.6, reliability=0.5),
        }
        expected = (0.8 * 0.9 + 0.6 * 0.5) / (0.9 + 0.5)   # = 1.02 / 1.4 = 0.728571...
        final_score, sources = strategy.score(evidence)
        self.assertAlmostEqual(final_score, expected, places=4)
        self.assertAlmostEqual(sources["temporal"].weighted_contribution, 0.8 * 0.9, places=4)
        self.assertAlmostEqual(sources["personal"].weighted_contribution, 0.6 * 0.5, places=4)

    def test_single_source_equals_its_own_signal(self):
        """With one source, the weighted average must reduce EXACTLY to
        that source's own signal, regardless of its reliability (as long
        as reliability > 0) — a direct mathematical identity check."""
        strategy = ReliabilityWeightedStrategy()
        for reliability in (0.1, 0.5, 0.99):
            evidence = {"temporal": EvidenceItem(signal=0.73, reliability=reliability)}
            final_score, _ = strategy.score(evidence)
            self.assertAlmostEqual(final_score, 0.73, places=4)

    def test_zero_reliability_everywhere_yields_zero(self):
        strategy = ReliabilityWeightedStrategy()
        evidence = {
            "temporal": EvidenceItem(signal=0.95, reliability=0.0),
            "personal": EvidenceItem(signal=0.80, reliability=0.0),
        }
        final_score, _ = strategy.score(evidence)
        self.assertEqual(final_score, 0.0)

    def test_no_sources_yields_zero(self):
        strategy = ReliabilityWeightedStrategy()
        final_score, sources = strategy.score({})
        self.assertEqual(final_score, 0.0)
        self.assertEqual(sources, {})

    def test_higher_reliability_pulls_score_toward_itself(self):
        """Same two signals, but flip which one has higher reliability ->
        final score must move toward the now-more-reliable one."""
        strategy = ReliabilityWeightedStrategy()
        a, _ = strategy.score({
            "temporal": EvidenceItem(signal=0.9, reliability=0.9),
            "personal": EvidenceItem(signal=0.3, reliability=0.1),
        })
        b, _ = strategy.score({
            "temporal": EvidenceItem(signal=0.9, reliability=0.1),
            "personal": EvidenceItem(signal=0.3, reliability=0.9),
        })
        self.assertGreater(a, b)          # (a) trusts the high (0.9) signal more
        self.assertLess(b, 0.5)           # (b) trusts the low (0.3) signal more


class TestConflictDetectorMath(unittest.TestCase):
    def test_no_conflict_single_source(self):
        detector = ConflictDetector()
        report = detector.detect({"temporal": {"medical_assistance": EvidenceItem(0.9, 0.9)}})
        self.assertFalse(report.conflict_detected)

    def test_no_conflict_when_top_picks_agree(self):
        detector = ConflictDetector()
        report = detector.detect({
            "temporal": {"medical_assistance": EvidenceItem(0.9, 0.9), "task_assistance": EvidenceItem(0.1, 0.9)},
            "personal": {"medical_assistance": EvidenceItem(0.8, 0.8), "task_assistance": EvidenceItem(0.2, 0.8)},
        })
        self.assertFalse(report.conflict_detected)

    def test_conflict_detected_when_strong_disagreement(self):
        detector = ConflictDetector(ReasoningConfig(conflict_threshold=0.30))
        report = detector.detect({
            "temporal": {"medical_assistance": EvidenceItem(0.90, 0.90), "task_assistance": EvidenceItem(0.10, 0.90)},
            "personal": {"medical_assistance": EvidenceItem(0.10, 0.90), "task_assistance": EvidenceItem(0.90, 0.90)},
        })
        self.assertTrue(report.conflict_detected)
        self.assertEqual(len(report.disagreements), 1)
        self.assertAlmostEqual(report.conflict_score, 0.81, places=4)   # min(0.9*0.9, 0.9*0.9)

    def test_weak_disagreement_below_threshold_not_flagged(self):
        detector = ConflictDetector(ReasoningConfig(conflict_threshold=0.30))
        report = detector.detect({
            "temporal": {"medical_assistance": EvidenceItem(0.9, 0.9), "task_assistance": EvidenceItem(0.1, 0.9)},
            "personal": {"medical_assistance": EvidenceItem(0.4, 0.2), "task_assistance": EvidenceItem(0.45, 0.2)},
        })
        # personal's top pick differs (task, barely) but its strength is tiny (0.45*0.2=0.09)
        self.assertFalse(report.conflict_detected)

    def test_threshold_is_configurable(self):
        evidence = {
            "temporal": {"medical_assistance": EvidenceItem(0.6, 0.6), "task_assistance": EvidenceItem(0.1, 0.6)},
            "personal": {"medical_assistance": EvidenceItem(0.1, 0.6), "task_assistance": EvidenceItem(0.6, 0.6)},
        }
        strict = ConflictDetector(ReasoningConfig(conflict_threshold=0.9)).detect(evidence)
        lenient = ConflictDetector(ReasoningConfig(conflict_threshold=0.1)).detect(evidence)
        self.assertFalse(strict.conflict_detected)
        self.assertTrue(lenient.conflict_detected)


class TestScenario1ConsistentEvidence(unittest.TestCase):
    """TEST 1 — both sources favor medical -> medical ranks first."""

    def test_medical_ranks_first(self):
        temporal = _temporal_resolution({
            "medical_assistance": (0.91, 0.94), "emergency_support": (0.20, 0.90),
            "task_assistance": (0.10, 0.90), "information_request": (0.05, 0.90), "general_attention": (0.0, 0.90),
        })
        personal = _personal_resolution({
            "medical_assistance": (0.87, 0.91), "emergency_support": (0.15, 0.85),
            "task_assistance": (0.10, 0.85), "information_request": (0.05, 0.85), "general_attention": (0.0, 0.85),
        })
        resolver = IntentResolver()
        result = resolver.resolve("HELP", _help_candidates(), temporal=temporal, personal=personal, now=NOW)
        self.assertEqual(result.candidates[0].candidate, "medical_assistance")
        self.assertEqual(result.candidates[0].rank, 1)


class TestScenario2DifferentEvidence(unittest.TestCase):
    """TEST 2 — temporal favors medical, personal favors task -> engine combines, doesn't blindly follow one."""

    def test_combines_rather_than_following_one_source(self):
        temporal = _temporal_resolution({"medical_assistance": (0.91, 0.94), "task_assistance": (0.10, 0.94)})
        personal = _personal_resolution({"medical_assistance": (0.15, 0.91), "task_assistance": (0.87, 0.91)})
        resolver = IntentResolver()
        result = resolver.resolve("HELP", ONTOLOGY.lookup("HELP")[:1] +
                                   [c for c in ONTOLOGY.lookup("HELP") if c.id == "task_assistance"],
                                   temporal=temporal, personal=personal, now=NOW)

        medical = _find(result, "medical_assistance")
        task = _find(result, "task_assistance")
        expected_medical = (0.91 * 0.94 + 0.15 * 0.91) / (0.94 + 0.91)
        expected_task = (0.10 * 0.94 + 0.87 * 0.91) / (0.94 + 0.91)
        self.assertAlmostEqual(medical.final_score, expected_medical, places=4)
        self.assertAlmostEqual(task.final_score, expected_task, places=4)
        # Neither score equals either single source's raw signal in isolation —
        # proof the engine actually fused both rather than picking one.
        self.assertNotEqual(medical.final_score, 0.91)
        self.assertNotEqual(medical.final_score, 0.15)


class TestScenario3ReliabilityMatters(unittest.TestCase):
    """TEST 3 — low-reliability high-signal temporal must NOT dominate reliable personal evidence."""

    def test_low_reliability_has_reduced_influence(self):
        temporal = _temporal_resolution({"medical_assistance": (0.95, 0.20)})
        personal = _personal_resolution({"medical_assistance": (0.70, 0.90)})
        resolver = IntentResolver()
        result = resolver.resolve("HELP", [c for c in _help_candidates() if c.id == "medical_assistance"],
                                   temporal=temporal, personal=personal, now=NOW)
        medical = _find(result, "medical_assistance")

        expected = (0.95 * 0.20 + 0.70 * 0.90) / (0.20 + 0.90)
        self.assertAlmostEqual(medical.final_score, expected, places=4)
        # Closer to the reliable personal evidence (0.70) than the unreliable temporal spike (0.95).
        self.assertLess(abs(medical.final_score - 0.70), abs(medical.final_score - 0.95))


class TestScenario4EqualReliability(unittest.TestCase):
    """TEST 4 — similar evidence + similar reliability -> scores stay close."""

    def test_similar_evidence_yields_close_scores(self):
        temporal = _temporal_resolution({"medical_assistance": (0.60, 0.70), "emergency_support": (0.58, 0.70)})
        personal = _personal_resolution({"medical_assistance": (0.55, 0.65), "emergency_support": (0.57, 0.65)})
        resolver = IntentResolver()
        result = resolver.resolve("HELP", [c for c in _help_candidates() if c.id in
                                            ("medical_assistance", "emergency_support")],
                                   temporal=temporal, personal=personal, now=NOW)
        medical = _find(result, "medical_assistance")
        emergency = _find(result, "emergency_support")
        self.assertLess(abs(medical.final_score - emergency.final_score), 0.05)


class TestScenario5MissingContext(unittest.TestCase):
    """TEST 5 — only temporal available -> engine works, no fake personal evidence."""

    def test_only_temporal_available(self):
        temporal = _temporal_resolution({"medical_assistance": (0.80, 0.85)})
        resolver = IntentResolver()
        result = resolver.resolve("HELP", [c for c in _help_candidates() if c.id == "medical_assistance"],
                                   temporal=temporal, personal=None, now=NOW)

        self.assertEqual(result.available_sources, ["temporal"])
        medical = _find(result, "medical_assistance")
        self.assertNotIn("personal", medical.sources)
        # Single source -> final score reduces exactly to that source's signal.
        self.assertAlmostEqual(medical.final_score, 0.80, places=4)


class TestScenario6NoUsableEvidence(unittest.TestCase):
    """TEST 6 — all available evidence is weak/zero -> engine must not fake confidence."""

    def test_weak_evidence_yields_low_score(self):
        temporal = _temporal_resolution({cid: (0.0, 0.0) for cid in
                                          ("medical_assistance", "emergency_support", "task_assistance")})
        personal = _personal_resolution({cid: (0.0, 0.0) for cid in
                                          ("medical_assistance", "emergency_support", "task_assistance")})
        resolver = IntentResolver()
        result = resolver.resolve("HELP", [c for c in _help_candidates() if c.id in
                                            ("medical_assistance", "emergency_support", "task_assistance")],
                                   temporal=temporal, personal=personal, now=NOW)
        for c in result.candidates:
            self.assertEqual(c.final_score, 0.0)


class TestScenario7ConflictingLayers(unittest.TestCase):
    """TEST 7 — temporal strongly supports medical, personal strongly supports task -> conflict=True, ranking still produced."""

    def test_conflict_flagged_and_ranking_still_generated(self):
        temporal = _temporal_resolution({"medical_assistance": (0.90, 0.90), "task_assistance": (0.05, 0.90)})
        personal = _personal_resolution({"medical_assistance": (0.05, 0.90), "task_assistance": (0.90, 0.90)})
        resolver = IntentResolver()
        result = resolver.resolve("HELP", [c for c in _help_candidates() if c.id in
                                            ("medical_assistance", "task_assistance")],
                                   temporal=temporal, personal=personal, now=NOW)
        self.assertTrue(result.conflict.conflict_detected)
        self.assertEqual(len(result.candidates), 2)
        self.assertTrue(all(c.rank in (1, 2) for c in result.candidates))


class TestScenario8MultipleCandidates(unittest.TestCase):
    """TEST 8 — 3+ candidates -> complete ranking returned."""

    def test_complete_ranking_for_all_candidates(self):
        candidates = _help_candidates()
        self.assertGreaterEqual(len(candidates), 3)

        temporal = _temporal_resolution({c.id: (0.1 * i, 0.8) for i, c in enumerate(candidates)})
        resolver = IntentResolver()
        result = resolver.resolve("HELP", candidates, temporal=temporal, personal=None, now=NOW)

        self.assertEqual(len(result.candidates), len(candidates))
        ranks = sorted(c.rank for c in result.candidates)
        self.assertEqual(ranks, list(range(1, len(candidates) + 1)))
        scores = [c.final_score for c in result.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestCandidateAuthorityFromLayer1(unittest.TestCase):
    """The engine must never rank a candidate Layer 1 didn't supply, even
    if a context source's evidence mentions one."""

    def test_extra_evidence_candidate_is_ignored(self):
        layer1_candidates = [c for c in _help_candidates() if c.id == "medical_assistance"]
        temporal = _temporal_resolution({
            "medical_assistance": (0.8, 0.8),
            "task_assistance": (0.99, 0.99),   # NOT in Layer 1's candidate list
        })
        resolver = IntentResolver()
        result = resolver.resolve("HELP", layer1_candidates, temporal=temporal, personal=None, now=NOW)

        result_ids = {c.candidate for c in result.candidates}
        self.assertEqual(result_ids, {"medical_assistance"})
        self.assertNotIn("task_assistance", result_ids)


class TestFullIntegration(unittest.TestCase):
    """Real Layer 1 (ontology) -> real Layer 3 -> real Layer 5 -> Reasoning Engine."""

    def test_end_to_end_real_layers(self):
        candidates = ONTOLOGY.lookup("HELP")

        temporal_engine = TemporalContextEngine()
        temporal_engine.record("integration_user", "CHEST", timestamp=NOW - 20)
        temporal_engine.record("integration_user", "PAIN", timestamp=NOW - 10)
        temporal_result = temporal_engine.evaluate("integration_user", "HELP", candidates, now=NOW)

        personal_engine = PersonalContextEngine(db_path=":memory:")
        for _ in range(18):
            personal_engine.update("integration_user", "HELP", "medical_assistance", timestamp=NOW - 5 * 86400)
        for _ in range(3):
            personal_engine.update("integration_user", "HELP", "emergency_support", timestamp=NOW - 5 * 86400)
        personal_result = personal_engine.evaluate("integration_user", "HELP", candidates, now=NOW)

        resolver = IntentResolver()
        result = resolver.resolve("HELP", candidates, temporal=temporal_result, personal=personal_result, now=NOW)

        self.assertEqual(result.available_sources, ["temporal", "personal"])
        self.assertEqual(result.candidates[0].candidate, "medical_assistance")
        self.assertEqual(len(result.candidates), len(candidates))
        self.assertIn("temporal", result.candidates[0].sources)
        self.assertIn("personal", result.candidates[0].sources)


class TestPerformance(unittest.TestCase):
    def test_resolve_is_fast(self):
        import time as _time
        candidates = _help_candidates()
        temporal = _temporal_resolution({c.id: (0.5, 0.7) for c in candidates})
        personal = _personal_resolution({c.id: (0.4, 0.6) for c in candidates})
        resolver = IntentResolver()

        n = 100
        start = _time.perf_counter()
        for _ in range(n):
            resolver.resolve("HELP", candidates, temporal=temporal, personal=personal, now=NOW)
        avg_ms = (_time.perf_counter() - start) * 1000.0 / n
        print(f"\n[LATENCY] IntentResolver.resolve(): avg {avg_ms:.3f} ms over {n} calls")
        self.assertLess(avg_ms, 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
