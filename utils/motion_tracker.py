"""
utils/motion_tracker.py
------------------------
Track fingertip trajectories to detect motion-based signs J and Z.

J: pinky traces a downward hook (I → draw J)
Z: index traces a zig-zag (1 → draw Z)
"""

from collections import deque
import numpy as np


class MotionTracker:
    """
    Buffer recent landmark positions and detect J / Z trajectories.

    Usage
    -----
    tracker = MotionTracker()
    # each frame after hand is detected:
    tracker.update(hand['lmRaw'])
    if tracker.detect_j(): ...
    if tracker.detect_z(): ...
    # when hand disappears / letter accepted:
    tracker.clear()
    """

    def __init__(self, buffer_size: int = 30):
        self._buf  = deque(maxlen=buffer_size)
        self._size = buffer_size

    # ── Public ────────────────────────────────────────────────────────────────

    def update(self, landmarks) -> None:
        """Add current frame's landmark positions to the trajectory buffer."""
        if hasattr(landmarks[0], 'x'):
            pts = [(lm.x, lm.y) for lm in landmarks]
        else:
            pts = list(landmarks)
        self._buf.append(pts)

    def clear(self) -> None:
        self._buf.clear()

    @property
    def ready(self) -> bool:
        return len(self._buf) >= 15

    def detect_j(self) -> bool:
        """
        J: pinky tip (landmark 20) moves predominantly DOWNWARD (y increases
        in normalised image coords) with a small rightward hook at the end.
        """
        if not self.ready:
            return False
        ys = [f[20][1] for f in self._buf]
        xs = [f[20][0] for f in self._buf]
        dy = ys[-1] - ys[0]             # positive = downward in image
        x_range = max(xs) - min(xs)
        # Must move down significantly more than sideways
        return dy > 0.14 and dy > x_range * 0.8

    def detect_z(self) -> bool:
        """
        Z: index tip (landmark 8) traces a zig-zag — at least two clear
        direction reversals in x, with meaningful total horizontal travel.
        """
        if not self.ready:
            return False
        xs = [f[8][0] for f in self._buf]
        # Compute smoothed directional changes
        dx = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
        # Threshold small jitter
        dx = [d if abs(d) > 0.008 else 0.0 for d in dx]
        # Count sign changes
        sign_changes = 0
        prev = 0.0
        for d in dx:
            if d != 0.0:
                if prev != 0.0 and d * prev < 0:
                    sign_changes += 1
                prev = d
        x_range = max(xs) - min(xs)
        return sign_changes >= 2 and x_range > 0.18

    def pinky_extended(self) -> bool:
        """True if pinky tip is currently above its MCP (roughly extended)."""
        if not self._buf:
            return False
        last = self._buf[-1]
        return last[20][1] < last[17][1]   # tip y < MCP y (higher in frame)

    def index_extended(self) -> bool:
        """True if index tip is currently above its MCP."""
        if not self._buf:
            return False
        last = self._buf[-1]
        return last[8][1] < last[5][1]
