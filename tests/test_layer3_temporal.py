"""
tests/test_layer3_temporal.py
--------------------------------
Automated tests for Layer 3 (Temporal Context), covering the 6 required
scenarios plus feature-level correctness checks. Uses REAL ontology data
(intent.ontology.IntentOntology reading data/intent_ontology.json) rather
than mocked candidates, so these tests exercise the actual Layer 1 -> 3
integration path.

All timestamps are explicitly injected (via `now=` / `timestamp=`) so
these tests are deterministic and fast — no real-time sleeps.

Run with:  venv310\\Scripts\\python.exe -m unittest tests.test_layer3_temporal -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent.ontology import IntentOntology
from layer3_temporal.config import Layer3Config
from layer3_temporal.engine import TemporalContextEngine
from layer3_temporal.features import (
    linear_decay,
    compute_recency,
    compute_sequence_strength,
    compute_frequency,
    compute_time_gap_coherence,
    compute_repetition,
)
from intent.conversation import ConversationEntry

NOW = 1_700_000_000.0
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


class TestFeatureFunctions(unittest.TestCase):
    def test_linear_decay_at_zero(self):
        self.assertEqual(linear_decay(0, 120), 1.0)

    def test_linear_decay_at_window_edge(self):
        self.assertEqual(linear_decay(120, 120), 0.0)

    def test_linear_decay_past_window_clips(self):
        self.assertEqual(linear_decay(500, 120), 0.0)

    def test_linear_decay_midpoint(self):
        self.assertAlmostEqual(linear_decay(60, 120), 0.5)

    def test_compute_recency_no_matches(self):
        self.assertEqual(compute_recency([], NOW, 120), 0.0)

    def test_compute_recency_fresh_match(self):
        e = ConversationEntry(word="PAIN", timestamp=NOW - 5)
        self.assertAlmostEqual(compute_recency([e], NOW, 120), 1.0 - 5 / 120, places=4)

    def test_compute_sequence_strength_immediate_prior(self):
        e1 = ConversationEntry(word="CHEST", timestamp=NOW - 20)
        e2 = ConversationEntry(word="PAIN", timestamp=NOW - 10)
        all_entries = [e1, e2]
        # e2 is immediately prior (position_back=1) -> weight 1.0
        self.assertAlmostEqual(compute_sequence_strength([e2], all_entries), 1.0)
        # e1 is 2-back -> weight 0.5
        self.assertAlmostEqual(compute_sequence_strength([e1], all_entries), 0.5)
        # both matched -> 1.0 + 0.5 = 1.5, capped to 1.0
        self.assertAlmostEqual(compute_sequence_strength([e1, e2], all_entries), 1.0, places=4)

    def test_compute_frequency_saturates(self):
        matches = [ConversationEntry(word="PAIN", timestamp=NOW)] * 5
        self.assertEqual(compute_frequency(matches, saturation_count=3), 1.0)

    def test_compute_frequency_partial(self):
        matches = [ConversationEntry(word="PAIN", timestamp=NOW)]
        self.assertAlmostEqual(compute_frequency(matches, saturation_count=3), 1 / 3)

    def test_time_gap_single_match_is_neutral(self):
        e = ConversationEntry(word="PAIN", timestamp=NOW - 5)
        self.assertEqual(compute_time_gap_coherence([e], 120), 1.0)

    def test_time_gap_tight_matches_high_coherence(self):
        e1 = ConversationEntry(word="CHEST", timestamp=NOW - 20)
        e2 = ConversationEntry(word="PAIN", timestamp=NOW - 15)  # 5s gap
        self.assertAlmostEqual(compute_time_gap_coherence([e1, e2], 120), 1 - 5 / 120, places=4)

    def test_time_gap_wide_matches_low_coherence(self):
        e1 = ConversationEntry(word="CHEST", timestamp=NOW - 115)
        e2 = ConversationEntry(word="PAIN", timestamp=NOW - 5)   # 110s gap in a 120s window
        self.assertAlmostEqual(compute_time_gap_coherence([e1, e2], 120), 1 - 110 / 120, places=4)

    def test_repetition_counts_prior_occurrences(self):
        entries = [
            ConversationEntry(word="HELP", timestamp=NOW - 30),
            ConversationEntry(word="HELP", timestamp=NOW - 15),
        ]
        self.assertEqual(compute_repetition(entries, "HELP", saturation_count=2), 1.0)

    def test_repetition_zero_when_absent(self):
        entries = [ConversationEntry(word="WATER", timestamp=NOW - 10)]
        self.assertEqual(compute_repetition(entries, "HELP", saturation_count=2), 0.0)


class TestScenario1MedicalSequence(unittest.TestCase):
    """TEST 1 — CHEST -> PAIN -> HELP should favor medical_assistance over task_assistance."""

    def test_medical_beats_task(self):
        engine = TemporalContextEngine()
        engine.record("user1", "CHEST", timestamp=NOW - 20)
        engine.record("user1", "PAIN", timestamp=NOW - 10)

        resolution = engine.evaluate("user1", "HELP", _help_candidates(), now=NOW)

        medical = _find(resolution, "medical_assistance")
        task = _find(resolution, "task_assistance")

        self.assertGreater(medical.signal, task.signal)
        self.assertIn("CHEST", medical.evidence["relevant_words"])
        self.assertIn("PAIN", medical.evidence["relevant_words"])
        # PAIN is a context_conflict for task_assistance -> penalized, not just "low"
        self.assertIn("PAIN", task.evidence["conflicting_words"])
        self.assertGreater(medical.reliability, 0.0)


class TestScenario2TaskSequence(unittest.TestCase):
    """TEST 2 — PROJECT -> DEADLINE -> HELP should favor task_assistance over medical_assistance."""

    def test_task_beats_medical(self):
        engine = TemporalContextEngine()
        engine.record("user2", "PROJECT", timestamp=NOW - 20)
        engine.record("user2", "DEADLINE", timestamp=NOW - 10)

        resolution = engine.evaluate("user2", "HELP", _help_candidates(), now=NOW)

        medical = _find(resolution, "medical_assistance")
        task = _find(resolution, "task_assistance")

        self.assertGreater(task.signal, medical.signal)
        self.assertIn("PROJECT", task.evidence["relevant_words"])
        self.assertIn("DEADLINE", task.evidence["relevant_words"])


class TestScenario3NoHistory(unittest.TestCase):
    """TEST 3 — HELP with no prior history -> weak signal, low reliability."""

    def test_weak_signal_low_reliability(self):
        engine = TemporalContextEngine()
        resolution = engine.evaluate("user3", "HELP", _help_candidates(), now=NOW)

        self.assertEqual(resolution.window_word_count, 0)
        for ev in resolution.candidates:
            self.assertEqual(ev.signal, 0.0)
            self.assertLess(ev.reliability, 0.2)


class TestScenario4LongGap(unittest.TestCase):
    """TEST 4 — a large gap before CHEST substantially reduces its temporal influence."""

    def test_stale_evidence_weaker_than_fresh(self):
        cfg = Layer3Config()  # window_seconds = 120.0

        engine_fresh = TemporalContextEngine(config=cfg)
        engine_fresh.record("u_fresh", "CHEST", timestamp=NOW - 5)
        fresh = _find(engine_fresh.evaluate("u_fresh", "HELP", _help_candidates(), now=NOW),
                      "medical_assistance")

        engine_stale = TemporalContextEngine(config=cfg)
        engine_stale.record("u_stale", "CHEST", timestamp=NOW - 115)   # near the 120s edge
        stale = _find(engine_stale.evaluate("u_stale", "HELP", _help_candidates(), now=NOW),
                      "medical_assistance")

        self.assertGreater(fresh.signal, stale.signal)
        self.assertGreater(fresh.reliability, stale.reliability)

    def test_word_outside_window_is_excluded_entirely(self):
        cfg = Layer3Config()  # 120s window
        engine = TemporalContextEngine(config=cfg)
        engine.record("u_outside", "CHEST", timestamp=NOW - 200)  # outside the window

        resolution = engine.evaluate("u_outside", "HELP", _help_candidates(), now=NOW)
        medical = _find(resolution, "medical_assistance")

        self.assertEqual(resolution.window_word_count, 0)
        self.assertEqual(medical.signal, 0.0)
        self.assertNotIn("CHEST", medical.evidence["relevant_words"])

    def test_window_is_configurable_not_hardcoded(self):
        """Same gap (15s), two different configured windows -> different outcomes,
        proving TEMPORAL_WINDOW_SECONDS is a real, respected parameter."""
        short_cfg = Layer3Config(window_seconds=10.0)
        engine = TemporalContextEngine(config=short_cfg)
        engine.record("u_cfg", "CHEST", timestamp=NOW - 15)  # outside a 10s window

        resolution = engine.evaluate("u_cfg", "HELP", _help_candidates(), now=NOW)
        self.assertEqual(resolution.window_word_count, 0)  # excluded under the 10s config


class TestScenario5Repetition(unittest.TestCase):
    """TEST 5 — repeated HELP HELP HELP should register as detected repetition."""

    def test_repetition_detected(self):
        engine = TemporalContextEngine()
        engine.record("user5", "HELP", timestamp=NOW - 40)
        engine.record("user5", "HELP", timestamp=NOW - 20)

        resolution = engine.evaluate("user5", "HELP", _help_candidates(), now=NOW)

        for ev in resolution.candidates:
            self.assertEqual(ev.evidence["repetition"], 1.0)  # 2 prior repeats, saturation=2


class TestScenario6UserIsolation(unittest.TestCase):
    """TEST 6 — different users' histories must remain isolated."""

    def test_users_do_not_leak_into_each_other(self):
        engine = TemporalContextEngine()
        engine.record("userA", "CHEST", timestamp=NOW - 20)
        engine.record("userA", "PAIN", timestamp=NOW - 10)
        # userB has no history at all

        res_a = engine.evaluate("userA", "HELP", _help_candidates(), now=NOW)
        res_b = engine.evaluate("userB", "HELP", _help_candidates(), now=NOW)

        medical_a = _find(res_a, "medical_assistance")
        medical_b = _find(res_b, "medical_assistance")

        self.assertGreater(medical_a.signal, 0.0)
        self.assertEqual(medical_b.signal, 0.0)
        self.assertEqual(res_b.window_word_count, 0)
        self.assertNotIn("CHEST", medical_b.evidence["relevant_words"])


class TestOutputStructure(unittest.TestCase):
    def test_required_output_fields_present(self):
        engine = TemporalContextEngine()
        engine.record("user_struct", "CHEST", timestamp=NOW - 10)
        resolution = engine.evaluate("user_struct", "HELP", _help_candidates(), now=NOW)

        for ev in resolution.candidates:
            d = ev.to_dict()
            self.assertIn("candidate", d)
            self.assertIn("signal", d)
            self.assertIn("reliability", d)
            self.assertIn("evidence", d)
            for key in ("relevant_words", "sequence_strength", "recency",
                        "frequency", "time_gap", "repetition", "conversation_length"):
                self.assertIn(key, d["evidence"])

    def test_signal_and_reliability_are_independent(self):
        """Regression guard against the documented anti-pattern of setting
        signal == reliability by construction."""
        engine = TemporalContextEngine()
        engine.record("user_indep", "CHEST", timestamp=NOW - 5)
        resolution = engine.evaluate("user_indep", "HELP", _help_candidates(), now=NOW)
        medical = _find(resolution, "medical_assistance")
        self.assertNotEqual(medical.signal, medical.reliability)


class TestDoesNotDecide(unittest.TestCase):
    def test_evaluate_returns_evidence_not_a_decision(self):
        """Layer 3 must not pick a winner or expose a 'decision' field."""
        engine = TemporalContextEngine()
        resolution = engine.evaluate("user_nodecide", "HELP", _help_candidates(), now=NOW)
        d = resolution.to_dict()
        self.assertNotIn("decision", d)
        self.assertNotIn("winner", d)
        self.assertNotIn("resolved_intent", d)
        self.assertIsInstance(resolution.candidates, list)


class TestPerformance(unittest.TestCase):
    def test_evaluate_is_fast(self):
        import time as _time
        engine = TemporalContextEngine()
        engine.record("user_perf", "CHEST", timestamp=NOW - 20)
        engine.record("user_perf", "PAIN", timestamp=NOW - 10)
        candidates = _help_candidates()

        n = 50
        start = _time.perf_counter()
        for _ in range(n):
            engine.evaluate("user_perf", "HELP", candidates, now=NOW)
        avg_ms = (_time.perf_counter() - start) * 1000.0 / n
        print(f"\n[LATENCY] TemporalContextEngine.evaluate(): avg {avg_ms:.3f} ms over {n} calls")
        self.assertLess(avg_ms, 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
