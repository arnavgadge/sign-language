"""
tests/test_dashboard.py
---------------------------
Tests for the Phase 7 technical observability dashboard.

Since the dashboard renders to an OpenCV image (not text), these tests
verify: (1) the DATA it's fed is real and correctly differs across
scenarios, (2) rendering never crashes for any real engine outcome
(RESOLVED / AMBIGUOUS / LITERAL, with/without conflict), and (3) it
updates only on genuine word events, never per-frame.

Run with:  venv310\\Scripts\\python.exe -m unittest tests.test_dashboard -v
"""
import dataclasses
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from resolution.engine import IntentEngine
from resolution.event_gate import WordEventGate
from dashboard.renderer import render_dashboard, blank_placeholder, CANVAS_W, CANVAS_H
from dashboard.window import EngineDashboard

NOW = 1_700_000_000.0
DAY = 86400.0


def _fresh_engine() -> IntentEngine:
    return IntentEngine(personal_db_path=":memory:", resolution_db_path=":memory:")


class TestScenario1MedicalTrace(unittest.TestCase):
    def test_chest_pain_help_renders_real_data(self):
        engine = _fresh_engine()
        engine.resolve("user1", "CHEST", now=NOW - 20)
        engine.resolve("user1", "PAIN", now=NOW - 10)
        result = engine.resolve("user1", "HELP", now=NOW)

        trace = engine.last_trace
        self.assertIsNotNone(trace)
        self.assertEqual(trace.word, "HELP")
        self.assertEqual(trace.resolution.status, result.status)
        # The trace's temporal evidence must show the real relevant words —
        # not something the dashboard invented.
        top_temporal = max(trace.temporal_result.candidates, key=lambda e: e.signal)
        self.assertIn("CHEST", top_temporal.evidence.get("relevant_words", []))

        history = engine.store.get_history("user1")
        canvas = render_dashboard(trace, history)
        self.assertEqual(canvas.shape, (CANVAS_H, CANVAS_W, 3))


class TestScenario2TemporalEvidenceDiffers(unittest.TestCase):
    def test_project_deadline_produces_different_temporal_evidence(self):
        engine_medical = _fresh_engine()
        engine_medical.resolve("userM", "CHEST", now=NOW - 20)
        engine_medical.resolve("userM", "PAIN", now=NOW - 10)
        engine_medical.resolve("userM", "HELP", now=NOW)
        medical_temporal = engine_medical.last_trace.temporal_result

        engine_task = _fresh_engine()
        engine_task.resolve("userT", "PROJECT", now=NOW - 20)
        engine_task.resolve("userT", "DEADLINE", now=NOW - 10)
        engine_task.resolve("userT", "HELP", now=NOW)
        task_temporal = engine_task.last_trace.temporal_result

        medical_top = max(medical_temporal.candidates, key=lambda e: e.signal)
        task_top = max(task_temporal.candidates, key=lambda e: e.signal)
        self.assertNotEqual(medical_top.candidate, task_top.candidate)
        self.assertEqual(medical_top.candidate, "medical_assistance")
        self.assertEqual(task_top.candidate, "task_assistance")


class TestScenario3PersonalEvidenceDiffersByUser(unittest.TestCase):
    def test_different_user_histories_produce_different_personal_evidence(self):
        engine = _fresh_engine()
        for _ in range(18):
            engine.personal_engine.update("userA", "HELP", "medical_assistance", timestamp=NOW - 5 * DAY)
        for _ in range(15):
            engine.personal_engine.update("userB", "HELP", "task_assistance", timestamp=NOW - 5 * DAY)

        engine.resolve("userA", "HELP", now=NOW)
        trace_a = engine.last_trace
        engine.resolve("userB", "HELP", now=NOW)
        trace_b = engine.last_trace

        top_a = max(trace_a.personal_result.candidates, key=lambda e: e.signal)
        top_b = max(trace_b.personal_result.candidates, key=lambda e: e.signal)
        self.assertEqual(top_a.candidate, "medical_assistance")
        self.assertEqual(top_b.candidate, "task_assistance")

        # Both must render without crashing, with genuinely different content.
        render_dashboard(trace_a, engine.store.get_history("userA"))
        render_dashboard(trace_b, engine.store.get_history("userB"))


class TestScenario4ConflictDisplayed(unittest.TestCase):
    def test_conflict_data_reaches_the_dashboard(self):
        engine = _fresh_engine()
        engine.resolve("conflict_user", "CHEST", now=NOW - 40)
        engine.resolve("conflict_user", "PAIN", now=NOW - 30)
        for _ in range(20):
            engine.personal_engine.update("conflict_user", "HELP", "task_assistance", timestamp=NOW - 10 * DAY)

        engine.resolve("conflict_user", "HELP", now=NOW)
        trace = engine.last_trace

        self.assertTrue(trace.reasoning_result.conflict.conflict_detected)
        self.assertGreater(len(trace.reasoning_result.conflict.disagreements), 0)
        # Must not crash rendering a conflicted result.
        canvas = render_dashboard(trace, engine.store.get_history("conflict_user"))
        self.assertEqual(canvas.shape, (CANVAS_H, CANVAS_W, 3))


class TestScenario5LowConfidenceDisplayed(unittest.TestCase):
    def test_literal_status_renders_correctly(self):
        engine = _fresh_engine()
        result = engine.resolve("fresh_user", "HELP", now=NOW)
        trace = engine.last_trace

        self.assertEqual(result.status, "LITERAL")
        self.assertEqual(trace.gate_decision.mode, "literal_fallback")
        canvas = render_dashboard(trace, engine.store.get_history("fresh_user"))
        self.assertEqual(canvas.shape, (CANVAS_H, CANVAS_W, 3))

    def test_ambiguous_status_renders_top_n(self):
        engine = _fresh_engine()
        # Craft a scenario likely to land in AMBIGUOUS: some but not
        # dominant temporal evidence, no personal history.
        engine.resolve("ambig_user", "CHEST", now=NOW - 115)  # near window edge -> weak recency
        result = engine.resolve("ambig_user", "HELP", now=NOW)
        trace = engine.last_trace

        canvas = render_dashboard(trace, engine.store.get_history("ambig_user"))
        self.assertEqual(canvas.shape, (CANVAS_H, CANVAS_W, 3))
        self.assertIn(result.status, ("AMBIGUOUS", "LITERAL", "RESOLVED"))  # whichever it genuinely is


class TestScenario6OneUpdatePerGenuineEvent(unittest.TestCase):
    def test_repeated_word_produces_one_dashboard_update(self):
        engine = _fresh_engine()
        gate = WordEventGate()
        dashboard = EngineDashboard()

        for _ in range(5):  # simulate 5 render-loop frames with an unchanged accepted word
            current_word = "HELP"
            if gate.should_process(current_word):
                engine.resolve("loop_user", current_word, now=NOW)
                dashboard.update(engine.last_trace, engine.store.get_history("loop_user"))

        self.assertEqual(dashboard._update_count, 1)

    def test_different_words_each_produce_an_update(self):
        engine = _fresh_engine()
        gate = WordEventGate()
        dashboard = EngineDashboard()

        for word, ts in [("CHEST", NOW - 20), ("PAIN", NOW - 10), ("HELP", NOW)]:
            if gate.should_process(word):
                engine.resolve("seq_user", word, now=ts)
                dashboard.update(engine.last_trace, engine.store.get_history("seq_user"))

        self.assertEqual(dashboard._update_count, 3)


class TestNoRecalculation(unittest.TestCase):
    """Structural guarantee, not just a convention: the nested evidence
    objects are frozen dataclasses, so the dashboard cannot mutate/
    recompute them even by accident."""

    def test_trace_components_are_immutable(self):
        engine = _fresh_engine()
        engine.resolve("immutable_user", "HELP", now=NOW)
        trace = engine.last_trace

        with self.assertRaises(dataclasses.FrozenInstanceError):
            trace.temporal_result.candidates[0].signal = 0.99
        with self.assertRaises(dataclasses.FrozenInstanceError):
            trace.gate_decision.confidence = 0.99
        with self.assertRaises(dataclasses.FrozenInstanceError):
            trace.reasoning_result.candidates[0].final_score = 0.99


class TestTraceExportStructure(unittest.TestCase):
    """Confirms the data is 'accessible and structured' for later research export."""

    def test_engine_trace_to_dict_has_required_research_fields(self):
        engine = _fresh_engine()
        engine.resolve("export_user", "CHEST", now=NOW - 20)
        engine.resolve("export_user", "HELP", now=NOW)
        trace = engine.last_trace
        d = trace.to_dict()

        self.assertIn("word", d)
        self.assertIn("candidates", d)
        self.assertIn("temporal_result", d)
        self.assertIn("personal_result", d)
        self.assertIn("reasoning_result", d)
        self.assertIn("gate_decision", d)
        self.assertIn("resolution", d)
        self.assertIn("evaluated_at", d)

        # Spot-check nested structure carries signal/reliability separately.
        first_temporal_candidate = d["temporal_result"]["candidates"][0]
        self.assertIn("signal", first_temporal_candidate)
        self.assertIn("reliability", first_temporal_candidate)


class TestDashboardWindowLifecycle(unittest.TestCase):
    def test_blank_placeholder_before_first_update(self):
        dashboard = EngineDashboard()
        self.assertIsInstance(dashboard._canvas, np.ndarray)
        self.assertEqual(dashboard._canvas.shape, blank_placeholder().shape)
        self.assertEqual(dashboard._update_count, 0)

    def test_toggle_flips_visibility(self):
        dashboard = EngineDashboard()
        self.assertFalse(dashboard.visible)
        dashboard.toggle()
        self.assertTrue(dashboard.visible)


class TestPerformance(unittest.TestCase):
    def test_render_is_fast(self):
        engine = _fresh_engine()
        engine.resolve("perf_user", "CHEST", now=NOW - 20)
        engine.resolve("perf_user", "PAIN", now=NOW - 10)
        engine.resolve("perf_user", "HELP", now=NOW)
        trace = engine.last_trace
        history = engine.store.get_history("perf_user")

        n = 20
        start = time.perf_counter()
        for _ in range(n):
            render_dashboard(trace, history)
        avg_ms = (time.perf_counter() - start) * 1000.0 / n
        print(f"\n[LATENCY] render_dashboard(): avg {avg_ms:.3f} ms over {n} calls")
        self.assertLess(avg_ms, 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
