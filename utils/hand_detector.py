"""
utils/hand_detector.py
----------------------
Drop-in hand detector built on the MediaPipe Tasks API
(compatible with mediapipe >= 0.10.x, no cvzone required).

Downloads hand_landmarker.task (~8 MB) to model/ on first run.
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
_MODEL_PATH = os.path.join("model", "hand_landmarker.task")

# Pairs of landmark indices that form the hand skeleton
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm base
]


def _ensure_model() -> str:
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print("[HandDetector] Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[HandDetector] Saved -> {_MODEL_PATH}")
    return _MODEL_PATH


class HandDetector:
    """
    Detects one hand per frame using MediaPipe Tasks HandLandmarker.

    Usage
    -----
    detector = HandDetector(max_hands=1, detection_confidence=0.8)
    hands, frame = detector.find_hands(frame_bgr, draw=True)
    if hands:
        bbox = hands[0]['bbox']   # (x, y, w, h) in pixels
    detector.close()
    """

    def __init__(self, max_hands: int = 1,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5):
        model_path = _ensure_model()
        base_opts  = mp_tasks.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._detector  = mp_vision.HandLandmarker.create_from_options(opts)
        self._t0_ms     = int(time.time() * 1000)
        self._last_ts   = 0

    # ── Public ────────────────────────────────────────────────────────────────

    def find_hands(self, frame_bgr: np.ndarray,
                   draw: bool = True) -> tuple:
        """
        Detect hands in frame_bgr (BGR, uint8).

        Returns
        -------
        hands     : list of dicts  [{'bbox': (x,y,w,h), 'lmList': [(x,y),...]}]
        frame_bgr : same array, annotated if draw=True
        """
        h, w = frame_bgr.shape[:2]

        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamps must be strictly monotonic for VIDEO mode
        ts_ms = max(self._last_ts + 1,
                    int(time.time() * 1000) - self._t0_ms)
        self._last_ts = ts_ms

        result = self._detector.detect_for_video(mp_img, ts_ms)

        hands = []
        for landmarks in (result.hand_landmarks or []):
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]

            x1 = max(0, min(xs));  y1 = max(0, min(ys))
            x2 = min(w, max(xs));  y2 = min(h, max(ys))

            hands.append({
                'bbox'  : (x1, y1, x2 - x1, y2 - y1),
                'lmList': list(zip(xs, ys)),
                'lmRaw' : landmarks,   # NormalizedLandmark objects for feature extraction
            })

            if draw:
                for s, e in _CONNECTIONS:
                    cv2.line(frame_bgr, (xs[s], ys[s]), (xs[e], ys[e]),
                             (0, 220, 0), 2, cv2.LINE_AA)
                for px, py in zip(xs, ys):
                    cv2.circle(frame_bgr, (px, py), 4, (255, 80, 0), -1)

        return hands, frame_bgr

    def close(self) -> None:
        self._detector.close()
