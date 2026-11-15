"""
layer1/pipeline.py
--------------------
Orchestrates the complete Layer 1 inference pipeline: takes the raw
signals available at the moment a user commits an interaction, and
produces one InteractionContext.

This is the single entry point Layers 3 and 5 (Phase 3+) will call —
everything upstream of it (recognition system) and downstream of it
(personalization, decision-making) is out of scope for Phase 2.
"""
import time
import uuid

from .face_signal import FaceDetector
from .preprocessing import normalize_word, bucket_time_period, normalize_motion_speed
from .schema import InteractionContext


class Layer1Pipeline:
    """
    Usage
    -----
    pipeline = Layer1Pipeline(user_id="default", session_id="session-1")
    ctx = pipeline.build(word="HELP", frame=frame_bgr, motion_speed_px=12.4)
    """

    def __init__(self, user_id: str = "default", session_id: str = None,
                 face_detector: FaceDetector = None):
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        self.face_detector = face_detector or FaceDetector()

    def build(self, word: str, frame=None, motion_speed_px: float = 0.0,
              timestamp: float = None) -> InteractionContext:
        t0 = time.perf_counter()

        ts = timestamp if timestamp is not None else time.time()

        word_norm = normalize_word(word)
        hour, period = bucket_time_period(ts)
        urgency = normalize_motion_speed(motion_speed_px)
        face_signal = self.face_detector.detect(frame)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return InteractionContext(
            session_id=self.session_id,
            event_id=str(uuid.uuid4()),
            timestamp=ts,
            user_id=self.user_id,
            word_raw=word if word is not None else "",
            word=word_norm,
            motion_speed_px=float(motion_speed_px or 0.0),
            motion_urgency=urgency,
            hour=hour,
            time_period=period,
            face=face_signal,
            pipeline_latency_ms=elapsed_ms,
        )
