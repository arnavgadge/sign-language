"""
tests/test_layer1.py
----------------------
Correctness + reproducibility + latency tests for Layer 1.

Layer 1 has no trainable weights (see Phase 2 report), so there is no
train/test split here for *these* tests — they validate that deterministic
transforms produce deterministic, correct output, and that the pipeline
runs within a reasonable latency budget. The one component with real
ground-truth-based evaluation (the face detector) is evaluated separately
in evaluation/layer1_face_eval.py, which needs labeled image data.

Run with:  venv310\\Scripts\\python.exe -m unittest tests.test_layer1 -v
"""
import datetime
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer1.preprocessing import normalize_word, bucket_time_period, normalize_motion_speed
from layer1.schema import InteractionContext, FaceSignal
from layer1.face_signal import FaceDetector
from layer1.pipeline import Layer1Pipeline
from layer1.storage import InteractionLogger


def _ts_at_hour(hour: int) -> float:
    """Build an epoch timestamp for *today*, local time, at the given hour."""
    now = datetime.datetime.now()
    return now.replace(hour=hour, minute=0, second=0, microsecond=0).timestamp()


class TestNormalizeWord(unittest.TestCase):
    def test_basic_uppercase(self):
        self.assertEqual(normalize_word("help"), "HELP")

    def test_strips_whitespace(self):
        self.assertEqual(normalize_word("  help  "), "HELP")

    def test_already_normalized(self):
        self.assertEqual(normalize_word("HELP"), "HELP")

    def test_empty_string(self):
        self.assertEqual(normalize_word(""), "")

    def test_none_input(self):
        self.assertEqual(normalize_word(None), "")


class TestBucketTimePeriod(unittest.TestCase):
    def test_morning(self):
        hour, period = bucket_time_period(_ts_at_hour(8))
        self.assertEqual(hour, 8)
        self.assertEqual(period, "morning")

    def test_afternoon(self):
        _, period = bucket_time_period(_ts_at_hour(14))
        self.assertEqual(period, "afternoon")

    def test_evening(self):
        _, period = bucket_time_period(_ts_at_hour(19))
        self.assertEqual(period, "evening")

    def test_night_late(self):
        _, period = bucket_time_period(_ts_at_hour(23))
        self.assertEqual(period, "night")

    def test_night_early(self):
        _, period = bucket_time_period(_ts_at_hour(3))
        self.assertEqual(period, "night")

    def test_boundary_6am_is_morning(self):
        _, period = bucket_time_period(_ts_at_hour(6))
        self.assertEqual(period, "morning")

    def test_boundary_5am_is_night(self):
        _, period = bucket_time_period(_ts_at_hour(5))
        self.assertEqual(period, "night")


class TestNormalizeMotionSpeed(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(normalize_motion_speed(0), 0.0)

    def test_half_reference(self):
        self.assertAlmostEqual(normalize_motion_speed(25.0, reference=50.0), 0.5)

    def test_at_reference_saturates(self):
        self.assertEqual(normalize_motion_speed(50.0, reference=50.0), 1.0)

    def test_above_reference_clips(self):
        self.assertEqual(normalize_motion_speed(999.0, reference=50.0), 1.0)

    def test_negative_floors_to_zero(self):
        self.assertEqual(normalize_motion_speed(-10.0, reference=50.0), 0.0)

    def test_none_input(self):
        self.assertEqual(normalize_motion_speed(None), 0.0)


class TestSchemaRoundTrip(unittest.TestCase):
    def _sample_context(self) -> InteractionContext:
        return InteractionContext(
            session_id="s1", event_id="e1", timestamp=1234567890.0,
            user_id="default", word_raw="help", word="HELP",
            motion_speed_px=12.4, motion_urgency=0.248,
            hour=14, time_period="afternoon",
            face=FaceSignal(face_count=1, face_present=True, detector_available=True),
            pipeline_latency_ms=3.2,
        )

    def test_to_dict_from_dict(self):
        ctx = self._sample_context()
        restored = InteractionContext.from_dict(ctx.to_dict())
        self.assertEqual(ctx, restored)

    def test_to_json_from_json(self):
        ctx = self._sample_context()
        restored = InteractionContext.from_json(ctx.to_json())
        self.assertEqual(ctx, restored)


class TestFaceDetector(unittest.TestCase):
    def test_cascade_available(self):
        detector = FaceDetector()
        self.assertTrue(detector.is_available,
                         "Haar cascade failed to load — check OpenCV install")

    def test_none_frame(self):
        detector = FaceDetector()
        signal = detector.detect(None)
        self.assertEqual(signal.face_count, 0)
        self.assertFalse(signal.face_present)

    def test_blank_frame_no_faces(self):
        detector = FaceDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        signal = detector.detect(blank)
        self.assertEqual(signal.face_count, 0)
        self.assertFalse(signal.face_present)
        self.assertTrue(signal.detector_available)


class TestLayer1Pipeline(unittest.TestCase):
    def test_end_to_end_synthetic(self):
        pipeline = Layer1Pipeline(user_id="test_user", session_id="test_session")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx = pipeline.build(word="  help  ", frame=frame, motion_speed_px=25.0)

        self.assertEqual(ctx.word, "HELP")
        self.assertEqual(ctx.word_raw, "  help  ")
        self.assertAlmostEqual(ctx.motion_urgency, 0.5)
        self.assertEqual(ctx.face.face_count, 0)
        self.assertIn(ctx.time_period, ("morning", "afternoon", "evening", "night"))
        self.assertGreaterEqual(ctx.pipeline_latency_ms, 0.0)
        self.assertEqual(ctx.session_id, "test_session")
        self.assertEqual(ctx.user_id, "test_user")

    def test_reproducible_given_fixed_timestamp(self):
        pipeline = Layer1Pipeline(user_id="u", session_id="s")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fixed_ts = 1700000000.0

        ctx1 = pipeline.build(word="help", frame=frame, motion_speed_px=10.0, timestamp=fixed_ts)
        ctx2 = pipeline.build(word="help", frame=frame, motion_speed_px=10.0, timestamp=fixed_ts)

        # event_id and pipeline_latency_ms legitimately vary per call.
        self.assertEqual(ctx1.word, ctx2.word)
        self.assertEqual(ctx1.timestamp, ctx2.timestamp)
        self.assertEqual(ctx1.hour, ctx2.hour)
        self.assertEqual(ctx1.time_period, ctx2.time_period)
        self.assertEqual(ctx1.motion_urgency, ctx2.motion_urgency)
        self.assertEqual(ctx1.face, ctx2.face)

    def test_latency_budget(self):
        """Informational + regression guard: pipeline should stay well under
        real-time frame budget (33ms @ 30fps) on CPU for a single call,
        with generous headroom to avoid flakiness on slower machines."""
        pipeline = Layer1Pipeline(user_id="u", session_id="s")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        n = 20
        start = time.perf_counter()
        for _ in range(n):
            pipeline.build(word="help", frame=frame, motion_speed_px=10.0)
        total_ms = (time.perf_counter() - start) * 1000.0
        avg_ms = total_ms / n

        print(f"\n[LATENCY] Layer1Pipeline.build(): avg {avg_ms:.2f} ms over {n} calls")
        self.assertLess(avg_ms, 500.0, "Layer 1 pipeline unexpectedly slow")


class TestInteractionLogger(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_append_and_read_all(self):
        logger = InteractionLogger(session_id="roundtrip_test", base_dir=self.tmp_dir)
        pipeline = Layer1Pipeline(user_id="u", session_id="roundtrip_test")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        ctx1 = pipeline.build(word="help", frame=frame, motion_speed_px=5.0)
        ctx2 = pipeline.build(word="stop", frame=frame, motion_speed_px=15.0)
        logger.append(ctx1)
        logger.append(ctx2)

        records = logger.read_all()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].word, "HELP")
        self.assertEqual(records[1].word, "STOP")

    def test_read_all_missing_file_returns_empty(self):
        logger = InteractionLogger(session_id="does_not_exist", base_dir=self.tmp_dir)
        self.assertEqual(logger.read_all(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
