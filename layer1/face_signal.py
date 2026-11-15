"""
layer1/face_signal.py
-----------------------
The one component in Layer 1 that is an actual classifier: an OpenCV
Haar-cascade face detector. It is PRE-TRAINED (shipped by OpenCV) — this
module does not train it, only runs inference and reports raw output.

This is the only Layer 1 sub-component evaluated with classification
metrics (precision/recall/F1/confusion matrix) in Phase 2, because it is
the only one that makes a prediction which can be right or wrong against
ground truth. See evaluation/layer1_face_eval.py.
"""
import os

from .config import (
    FACE_CASCADE_SCALE_FACTOR,
    FACE_CASCADE_MIN_NEIGHBORS,
    FACE_CASCADE_MIN_SIZE,
)
from .schema import FaceSignal


class FaceDetector:
    """
    Usage
    -----
    detector = FaceDetector()
    signal = detector.detect(frame_bgr)   # FaceSignal(face_count, face_present, ...)
    """

    def __init__(self,
                 scale_factor: float = FACE_CASCADE_SCALE_FACTOR,
                 min_neighbors: int = FACE_CASCADE_MIN_NEIGHBORS,
                 min_size: tuple = FACE_CASCADE_MIN_SIZE):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        self._cascade = None
        self._cascade_path = None
        try:
            import cv2
            self._cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            cascade = cv2.CascadeClassifier(self._cascade_path)
            if not cascade.empty():
                self._cascade = cascade
        except Exception:
            self._cascade = None

    @property
    def is_available(self) -> bool:
        return self._cascade is not None

    @property
    def cascade_path(self):
        return self._cascade_path

    def detect(self, frame_bgr) -> FaceSignal:
        """
        frame_bgr : np.ndarray (H, W, 3) BGR uint8, or None.
        Returns FaceSignal with face_count=0, face_present=False,
        detector_available=False if the frame or cascade is unavailable.
        """
        if frame_bgr is None or self._cascade is None:
            return FaceSignal(
                face_count=0,
                face_present=False,
                detector_available=self._cascade is not None,
            )

        import cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        count = len(faces)
        return FaceSignal(
            face_count=count,
            face_present=count > 0,
            detector_available=True,
        )
