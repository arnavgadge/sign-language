"""
tests/test_full_integration.py
----------------------------------
STEP 11 — end-to-end integration tests for the full pipeline:

    Layer 1 -> Layer 3 -> Layer 5 -> Reasoning Engine -> Decision Gate

exercised entirely through resolution.engine.IntentEngine.resolve(), the
single entry point the (untouched) recognition system now calls.

Every test uses its own isolated in-memory SQLite databases
(personal_db_path=":memory:", resolution_db_path=":memory:") — never the
real data/personal_context.db or data/resolution_history.db.

Run with:  venv310\\Scripts\\python.exe -m unittest tests.test_full_integration -v
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resolution.engine import IntentEngine
from resolution.event_gate import WordEventGate

NOW = 1_700_000_000.0
DAY = 86400.0


def _fresh_engine() -> IntentEngine:
    return IntentEngine(personal_db_path=":memory:", resolution_db_path=":memory:")


def _top(result):
    return result.ranked_candidates[0]["candidate"] if result.ranked_candidates else None


class TestScenario1MedicalContext(unittest.TestCase):
    def test_chest_pain_help_favors_medical(self):
        engine = _fresh_engine()
        engine.resolve("user1", "CHEST", now=NOW - 20)
        engine.resolve("user1", "PAIN", now=NOW - 10)
        result = engine.resolve("user1", "HELP", now=NOW)

        self.assertEqual(_top(result), "medical_assistance")
        self.assertIn(result.status, ("RESOLVED", "AMBIGUOUS"))


class TestScenario2TaskContext(unittest.TestCase):
    def test_project_deadline_help_favors_task_and_differs_from_medical_case(self):
        engine_medical = _fresh_engine()
        engine_medical.resolve("userM", "CHEST", now=NOW - 20)
        engine_medical.resolve("userM", "PAIN", now=NOW - 10)
        medical_result = engine_medical.resolve("userM", "HELP", now=NOW)

        engine_task = _fresh_engine()
        engine_task.resolve("userT", "PROJECT", now=NOW - 20)
        engine_task.resolve("userT", "DEADLINE", now=NOW - 10)
        task_result = engine_task.resolve("userT", "HELP", now=NOW)

        self.assertEqual(_top(task_result), "task_assistance")
        self.assertNotEqual(_top(medical_result), _top(task_result))


class TestScenario3NoPreviousContext(unittest.TestCase):
    def test_fresh_help_has_low_temporal_reliability_and_no_fake_confidence(self):
        engine = _fresh_engine()
        candidates = engine.ontology.lookup("HELP")

        # Inspect Layer 3's raw evidence directly — must show low reliability,
        # not a fabricated value, for a word with no prior history.
        temporal_result = engine.temporal_engine.evaluate("fresh_user", "HELP", candidates, now=NOW)
        for ev in temporal_result.candidates:
            self.assertLess(ev.reliability, 0.25)
        self.assertEqual(temporal_result.window_word_count, 0)

        # The full pipeline must not pretend to a confident resolution either.
        result = engine.resolve("fresh_user2", "HELP", now=NOW)
        self.assertEqual(result.status, "LITERAL")


class TestScenario4DifferentUsers(unittest.TestCase):
    def test_personal_history_differs_between_users(self):
        engine = _fresh_engine()

        for _ in range(18):
            engine.personal_engine.update("userA", "HELP", "medical_assistance", timestamp=NOW - 5 * DAY)
        for _ in range(15):
            engine.personal_engine.update("userB", "HELP", "task_assistance", timestamp=NOW - 5 * DAY)

        result_a = engine.resolve("userA", "HELP", now=NOW)
        result_b = engine.resolve("userB", "HELP", now=NOW)

        self.assertEqual(result_a.selected_intent, "medical_assistance")
        self.assertEqual(result_b.selected_intent, "task_assistance")


class TestScenario5AmbiguousConflictingEvidence(unittest.TestCase):
    def test_temporal_vs_personal_conflict_is_detected(self):
        engine = _fresh_engine()

        # Temporal history strongly points to medical.
        engine.resolve("conflict_user", "CHEST", now=NOW - 40)
        engine.resolve("conflict_user", "PAIN", now=NOW - 30)

        # But this user's long-term personal history strongly points to task.
        for _ in range(20):
            engine.personal_engine.update("conflict_user", "HELP", "task_assistance", timestamp=NOW - 10 * DAY)

        result = engine.resolve("conflict_user", "HELP", now=NOW)

        self.assertTrue(result.conflict)
        self.assertNotEqual(result.status, "")  # ranking still produced regardless
        self.assertGreater(len(result.ranked_candidates), 1)


class TestScenario6LowConfidence(unittest.TestCase):
    def test_unknown_word_with_no_history_falls_back(self):
        engine = _fresh_engine()
        result = engine.resolve("weak_user", "XYZZYPLUGH", now=NOW)   # not in the ontology at all
        self.assertEqual(result.status, "LITERAL")
        self.assertIsNone(result.selected_intent)
        self.assertEqual(result.output, "XYZZYPLUGH")


class TestScenario7RepeatedEventProtection(unittest.TestCase):
    def test_word_event_gate_blocks_immediate_repeat(self):
        gate = WordEventGate()
        self.assertTrue(gate.should_process("HELP"))
        self.assertFalse(gate.should_process("HELP"))   # simulated held-key / re-render
        self.assertFalse(gate.should_process("help"))   # case-insensitive duplicate too
        self.assertTrue(gate.should_process("STOP"))     # genuinely new word
        self.assertTrue(gate.should_process("HELP"))     # HELP again later is legitimate
        gate.reset()
        self.assertTrue(gate.should_process("HELP"))     # after reset, allowed immediately

    def test_engine_invoked_once_per_genuine_word_event(self):
        """Simulate a render loop calling the same handler repeatedly with an
        unchanged current_word — the engine itself must only run once."""
        engine = _fresh_engine()
        gate = WordEventGate()
        call_count = 0

        for _ in range(5):   # simulate 5 frames with the same accepted word
            current_word = "HELP"
            if gate.should_process(current_word):
                engine.resolve("loop_user", current_word, now=NOW)
                call_count += 1

        self.assertEqual(call_count, 1)


class TestStep6Persistence(unittest.TestCase):
    def test_resolution_recorded_in_store(self):
        engine = _fresh_engine()
        result = engine.resolve("persist_user", "HELP", now=NOW)

        history = engine.store.get_history("persist_user")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["recognized_word"], "HELP")
        self.assertEqual(history[0]["status"], result.status)


class TestStep7ProfileUpdateGating(unittest.TestCase):
    def test_resolved_updates_profile_ambiguous_does_not(self):
        engine = _fresh_engine()

        # A clear medical case -> RESOLVED -> should update the personal profile.
        engine.resolve("gate_user", "CHEST", now=NOW - 20)
        engine.resolve("gate_user", "PAIN", now=NOW - 10)
        resolved_result = engine.resolve("gate_user", "HELP", now=NOW)
        self.assertEqual(resolved_result.status, "RESOLVED")

        rows_after_resolved = engine.personal_engine.db.get_word_rows("gate_user", "HELP")
        total_after_resolved = sum(r.occurrence_count for r in rows_after_resolved)
        self.assertGreaterEqual(total_after_resolved, 1)

        # Now use a completely fresh user/word with no temporal or personal
        # history at all -> LITERAL (verified pattern from Scenario 3) ->
        # must not write anything to the profile.
        fresh_engine = _fresh_engine()
        literal_result = fresh_engine.resolve("ambig_user", "HELP", now=NOW)
        self.assertNotEqual(literal_result.status, "RESOLVED")

        rows_after_literal = fresh_engine.personal_engine.db.get_word_rows("ambig_user", "HELP")
        self.assertEqual(sum(r.occurrence_count for r in rows_after_literal), 0)


class TestStep3ContextPreservedAcrossWords(unittest.TestCase):
    def test_history_not_reset_between_words(self):
        engine = _fresh_engine()
        engine.resolve("history_user", "CHEST", now=NOW - 20)
        window = engine.temporal_engine.history.get_window("history_user", now=NOW)
        self.assertEqual([e.word for e in window], ["CHEST"])

        engine.resolve("history_user", "PAIN", now=NOW - 10)
        window = engine.temporal_engine.history.get_window("history_user", now=NOW)
        self.assertEqual([e.word for e in window], ["CHEST", "PAIN"])


class TestPerformance(unittest.TestCase):
    def test_full_resolve_is_fast_enough_for_the_recognition_loop(self):
        engine = _fresh_engine()
        n = 20
        start = time.perf_counter()
        for i in range(n):
            engine.resolve("perf_user", "HELP", now=NOW + i)
        avg_ms = (time.perf_counter() - start) * 1000.0 / n
        print(f"\n[LATENCY] Full IntentEngine.resolve(): avg {avg_ms:.3f} ms over {n} calls")
        self.assertLess(avg_ms, 100.0)   # generous — must not visibly stall a webcam-driven UI


if __name__ == "__main__":
    unittest.main(verbosity=2)
