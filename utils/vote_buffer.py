"""
utils/vote_buffer.py
---------------------
Temporal majority voting over a sliding window of raw predictions.

Stabilises flickering predictions before the LetterStabilizer sees them.
Example: [C C D C C C] → C  even if individual frame confidence is moderate.
"""

from collections import deque, Counter


class VoteBuffer:
    """
    Collect raw per-frame predictions and return the majority letter
    over the last `window` frames.

    Usage
    -----
    buf = VoteBuffer(window=12)
    buf.update("C")
    letter, fraction = buf.vote()   # ("C", 0.83)
    buf.clear()
    """

    def __init__(self, window: int = 12):
        self._buf    = deque(maxlen=window)
        self._window = window

    def update(self, letter: str) -> None:
        if letter:
            self._buf.append(letter)

    def vote(self) -> tuple:
        """Return (majority_letter, agreement_fraction)."""
        if not self._buf:
            return "", 0.0
        counts  = Counter(self._buf)
        winner, count = counts.most_common(1)[0]
        return winner, count / len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self._window // 2
