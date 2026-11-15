"""
tests/test_layer5_personal.py
--------------------------------
Automated tests for Layer 5 (Personal Context), covering the 7 required
scenarios plus feature-level correctness checks.

Every test uses its own isolated in-memory SQLite database
(PersonalContextEngine(db_path=":memory:")) — this is the "clearly
separate test data from actual user data" requirement: test data never
touches data/personal_context.db, the real database. Real ontology data
(intent.ontology.IntentOntology) is used for candidates, same pattern as
the Layer 3 test suite, so this exercises the real Layer 1 -> 5 path.

Run with:  venv310\\Scripts\\python.exe -m unittest tests.test_layer5_personal -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent.ontology import IntentOntology
from layer5_personal.config import Layer5Config
from layer5_personal.engine import PersonalContextEngine
from layer5_personal.features import (
    personal_prior,
    recency_weight,
    observation_sufficiency,
    profile_maturity,
    concentration,
)

NOW = 1_700_000_000.0
DAY = 86400.0
ONTOLOGY = IntentOntology(ontology_path=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "intent_ontology.json"
))


def _help_candidates():
    return ONTOLOGY.lookup("HELP")


def _find(resolution, candidate_id):
    for ev in resolution.candidates:
        if ev.candidate == candidate_id:
            return ev
    raise AssertionError(f"candidate {candidate_id} not found in resolution")


def _seed(engine, user_id, intent_id, count, timestamp):
    """Seed/demo helper — NOT production data. Each test builds its own
    isolated in-memory DB, so this never touches real user records."""
    for _ in range(count):
        engine.update(user_id, "HELP", intent_id, timestamp=timestamp)


class TestFeatureFunctions(unittest.TestCase):
    def test_personal_prior_basic(self):
        self.assertAlmostEqual(personal_prior(18, 22), 18 / 22)

    def test_personal_prior_no_data(self):
        self.assertEqual(personal_prior(0, 0), 0.0)

    def test_recency_weight_at_zero_days(self):
        self.assertEqual(recency_weight(NOW, NOW, half_life_days=30), 1.0)

    def test_recency_weight_at_half_life(self):
        self.assertAlmostEqual(recency_weight(NOW - 30 * DAY, NOW, half_life_days=30), 0.5, places=4)

    def test_recency_weight_no_data(self):
        self.assertEqual(recency_weight(None, NOW, half_life_days=30), 0.0)

    def test_observation_sufficiency_saturates(self):
        self.assertEqual(observation_sufficiency(50, saturation_count=10), 1.0)

    def test_observation_sufficiency_partial(self):
        self.assertAlmostEqual(observation_sufficiency(3, saturation_count=10), 0.3)

    def test_profile_maturity_new_user(self):
        self.assertEqual(profile_maturity(0, saturation_count=20), 0.0)

    def test_profile_maturity_established_user(self):
        self.assertEqual(profile_maturity(40, saturation_count=20), 1.0)

    def test_concentration_single_intent_no_competition(self):
        self.assertEqual(concentration(1.0, num_intents_for_word=1, has_row=True), 1.0)

    def test_concentration_no_data_no_competition(self):
        self.assertEqual(concentration(0.0, num_intents_for_word=0, has_row=False), 0.0)

    def test_concentration_perfectly_split_two_way(self):
        self.assertAlmostEqual(concentration(0.5, num_intents_for_word=2, has_row=True), 0.0)

    def test_concentration_dominant_among_three(self):
        # prior 0.818 among 3 competing intents (uniform=0.333)
        c = concentration(18 / 22, num_intents_for_word=3, has_row=True)
        self.assertGreater(c, 0.6)


class TestScenario1StrongMedicalPreference(unittest.TestCase):
    """TEST 1 — User A: HELP->medical x18, emergency x3, task x1 -> medical strongest."""

    def test_medical_strongest_signal(self):
        engine = PersonalContextEngine(db_path=":memory:")
        _seed(engine, "user_A", "medical_assistance", 18, NOW - 5 * DAY)
        _seed(engine, "user_A", "emergency_support", 3, NOW - 5 * DAY)
        _seed(engine, "user_A", "task_assistance", 1, NOW - 5 * DAY)

        resolution = engine.evaluate("user_A", "HELP", _help_candidates(), now=NOW)
        medical = _find(resolution, "medical_assistance")
        emergency = _find(resolution, "emergency_support")
        task = _find(resolution, "task_assistance")

        self.assertGreater(medical.signal, emergency.signal)
        self.assertGreater(medical.signal, task.signal)
        self.assertEqual(medical.evidence["historical_count"], 18)
        self.assertEqual(medical.evidence["total_word_intent_count"], 22)
        self.assertAlmostEqual(medical.evidence["personal_prior"], 18 / 22, places=3)


class TestScenario2DifferentUser(unittest.TestCase):
    """TEST 2 — User B: HELP->task x15, medical x2 -> task stronger. Demonstrates personalization."""

    def test_task_stronger_for_user_b(self):
        engine = PersonalContextEngine(db_path=":memory:")
        _seed(engine, "user_B", "task_assistance", 15, NOW - 5 * DAY)
        _seed(engine, "user_B", "medical_assistance", 2, NOW - 5 * DAY)

        resolution = engine.evaluate("user_B", "HELP", _help_candidates(), now=NOW)
        task = _find(resolution, "task_assistance")
        medical = _find(resolution, "medical_assistance")

        self.assertGreater(task.signal, medical.signal)


class TestScenario3NewUser(unittest.TestCase):
    """TEST 3 — User C: no history -> low reliability, no fabricated preference."""

    def test_cold_start(self):
        engine = PersonalContextEngine(db_path=":memory:")
        resolution = engine.evaluate("user_C", "HELP", _help_candidates(), now=NOW)

        self.assertEqual(resolution.profile_total_observations, 0)
        for ev in resolution.candidates:
            self.assertEqual(ev.signal, 0.0)
            self.assertLess(ev.reliability, 0.2)
            self.assertEqual(ev.evidence["historical_count"], 0)


class TestScenario4ConsistentHistory(unittest.TestCase):
    """TEST 4 — User D: HELP->medical x30 (only) -> high signal, high reliability."""

    def test_high_signal_high_reliability(self):
        engine = PersonalContextEngine(db_path=":memory:")
        _seed(engine, "user_D", "medical_assistance", 30, NOW - 1 * DAY)

        resolution = engine.evaluate("user_D", "HELP", _help_candidates(), now=NOW)
        medical = _find(resolution, "medical_assistance")

        self.assertGreater(medical.signal, 0.85)
        self.assertGreater(medical.reliability, 0.85)


class TestScenario5AmbiguousHistory(unittest.TestCase):
    """TEST 5 — User E: HELP->medical x10, task x10 -> neither artificially strong."""

    def test_tied_candidates_no_artificial_strength(self):
        engine = PersonalContextEngine(db_path=":memory:")
        _seed(engine, "user_E", "medical_assistance", 10, NOW - 5 * DAY)
        _seed(engine, "user_E", "task_assistance", 10, NOW - 5 * DAY)

        resolution = engine.evaluate("user_E", "HELP", _help_candidates(), now=NOW)
        medical = _find(resolution, "medical_assistance")
        task = _find(resolution, "task_assistance")

        self.assertAlmostEqual(medical.signal, task.signal, places=4)
        # Neither should look like a confidently dominant candidate.
        self.assertLess(medical.signal, 0.85)
        self.assertLess(task.signal, 0.85)
        # The ambiguity itself should be visible in the evidence.
        self.assertAlmostEqual(medical.evidence["concentration"], 0.0, places=4)


class TestScenario6OldVsRecentHistory(unittest.TestCase):
    """TEST 6 — same observation count, different ages -> recent scores higher."""

    def test_recent_beats_old(self):
        engine_recent = PersonalContextEngine(db_path=":memory:")
        _seed(engine_recent, "user_recent", "medical_assistance", 5, NOW - 1 * DAY)
        recent = _find(
            engine_recent.evaluate("user_recent", "HELP", _help_candidates(), now=NOW),
            "medical_assistance",
        )

        engine_old = PersonalContextEngine(db_path=":memory:")
        _seed(engine_old, "user_old", "medical_assistance", 5, NOW - 90 * DAY)
        old = _find(
            engine_old.evaluate("user_old", "HELP", _help_candidates(), now=NOW),
            "medical_assistance",
        )

        # Same occurrence_count and personal_prior (both 1.0) -> the ONLY
        # difference is recency, isolating its effect on signal/reliability.
        self.assertEqual(recent.evidence["historical_count"], old.evidence["historical_count"])
        self.assertEqual(recent.evidence["personal_prior"], old.evidence["personal_prior"])
        self.assertGreater(recent.evidence["recency_weight"], old.evidence["recency_weight"])
        self.assertGreater(recent.signal, old.signal)
        self.assertGreater(recent.reliability, old.reliability)


class TestScenario7UserIsolation(unittest.TestCase):
    """TEST 7 — User A's history must never influence User B."""

    def test_users_isolated(self):
        engine = PersonalContextEngine(db_path=":memory:")
        _seed(engine, "user_iso_A", "medical_assistance", 18, NOW - 5 * DAY)
        # user_iso_B never recorded anything

        res_a = engine.evaluate("user_iso_A", "HELP", _help_candidates(), now=NOW)
        res_b = engine.evaluate("user_iso_B", "HELP", _help_candidates(), now=NOW)

        medical_a = _find(res_a, "medical_assistance")
        medical_b = _find(res_b, "medical_assistance")

        self.assertGreater(medical_a.signal, 0.0)
        self.assertEqual(medical_b.signal, 0.0)
        self.assertEqual(res_b.profile_total_observations, 0)
        self.assertEqual(medical_b.evidence["historical_count"], 0)


class TestOutputStructure(unittest.TestCase):
    def test_required_output_fields_present(self):
        engine = PersonalContextEngine(db_path=":memory:")
        _seed(engine, "user_struct", "medical_assistance", 5, NOW - 2 * DAY)
        resolution = engine.evaluate("user_struct", "HELP", _help_candidates(), now=NOW)

        for ev in resolution.candidates:
            d = ev.to_dict()
            self.assertIn("candidate", d)
            self.assertIn("signal", d)
            self.assertIn("reliability", d)
            self.assertIn("evidence", d)
            for key in ("historical_count", "total_word_intent_count", "personal_prior",
                        "last_seen", "profile_observations"):
                self.assertIn(key, d["evidence"])

    def test_signal_and_reliability_are_independent(self):
        engine = PersonalContextEngine(db_path=":memory:")
        _seed(engine, "user_indep", "medical_assistance", 1, NOW - 2 * DAY)
        resolution = engine.evaluate("user_indep", "HELP", _help_candidates(), now=NOW)
        medical = _find(resolution, "medical_assistance")
        # Single observation -> potentially decent signal, but reliability
        # should be held back by low observation_sufficiency/profile_maturity.
        self.assertNotEqual(medical.signal, medical.reliability)
        self.assertLess(medical.reliability, medical.signal)


class TestDoesNotDecide(unittest.TestCase):
    def test_evaluate_returns_evidence_not_a_decision(self):
        engine = PersonalContextEngine(db_path=":memory:")
        resolution = engine.evaluate("user_nodecide", "HELP", _help_candidates(), now=NOW)
        d = resolution.to_dict()
        self.assertNotIn("decision", d)
        self.assertNotIn("winner", d)
        self.assertNotIn("resolved_intent", d)
        self.assertIsInstance(resolution.candidates, list)

    def test_does_not_generate_new_candidates(self):
        """Layer 5 must only score candidates Layer 1 handed it, never invent new ones."""
        engine = PersonalContextEngine(db_path=":memory:")
        _seed(engine, "user_nogen", "medical_assistance", 5, NOW - 2 * DAY)
        input_candidates = _help_candidates()
        resolution = engine.evaluate("user_nogen", "HELP", input_candidates, now=NOW)

        input_ids = {c.id for c in input_candidates}
        output_ids = {ev.candidate for ev in resolution.candidates}
        self.assertEqual(input_ids, output_ids)


class TestUpdateInterface(unittest.TestCase):
    def test_update_then_evaluate_reflects_change(self):
        engine = PersonalContextEngine(db_path=":memory:")
        before = _find(engine.evaluate("user_upd", "HELP", _help_candidates(), now=NOW),
                        "medical_assistance")
        self.assertEqual(before.evidence["historical_count"], 0)

        engine.update("user_upd", "HELP", "medical_assistance", timestamp=NOW)

        after = _find(engine.evaluate("user_upd", "HELP", _help_candidates(), now=NOW),
                       "medical_assistance")
        self.assertEqual(after.evidence["historical_count"], 1)
        self.assertGreater(after.signal, before.signal)


class TestPerformance(unittest.TestCase):
    def test_evaluate_is_fast(self):
        import time as _time
        engine = PersonalContextEngine(db_path=":memory:")
        _seed(engine, "user_perf", "medical_assistance", 10, NOW - 2 * DAY)
        candidates = _help_candidates()

        n = 50
        start = _time.perf_counter()
        for _ in range(n):
            engine.evaluate("user_perf", "HELP", candidates, now=NOW)
        avg_ms = (_time.perf_counter() - start) * 1000.0 / n
        print(f"\n[LATENCY] PersonalContextEngine.evaluate(): avg {avg_ms:.3f} ms over {n} calls")
        self.assertLess(avg_ms, 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
