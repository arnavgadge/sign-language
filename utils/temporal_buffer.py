"""
utils/temporal_buffer.py
------------------------
TEMPORAL SEQUENCE-BASED RECOGNITION

Instead of making decisions from single frames, this buffer:
1. Stores 15 frames of predictions with their confidence scores
2. Performs weighted majority voting based on confidence and persistence
3. Only accepts a letter if it persists across multiple frames
4. Ignores single-frame anomalies (temporal smoothing)

Key insight:
  ASL is a temporal hand topology, not a single-image classification problem.
  TIME CONSISTENCY matters more than single-frame certainty.
"""

from collections import deque, Counter
from typing import Tuple


class TemporalBuffer:
    """
    Store predictions + confidence over 15 frames.
    Perform weighted majority voting to stabilize recognition.

    Usage
    -----
    buf = TemporalBuffer(window=15)
    
    # Each frame:
    buf.update("D", confidence=0.92)  
    
    # Query the result:
    letter, weighted_confidence, persistence = buf.vote()
    """

    def __init__(self, window: int = 15):
        """
        Parameters
        ----------
        window : int
            Number of frames to keep in the buffer. Default = 15 (~0.5 sec at 30 FPS).
        """
        self._predictions = deque(maxlen=window)  # stores (letter, confidence) tuples
        self._window = window
        self._persistence = {}  # tracks consecutive frame counts per letter

    def update(self, letter: str, confidence: float) -> None:
        """
        Add a new frame's prediction to the buffer.
        
        Parameters
        ----------
        letter : str
            The predicted letter (e.g., 'D', 'E', 'A')
        confidence : float
            Model confidence (0.0 to 1.0)
        """
        if letter:
            self._predictions.append((letter, confidence))

    def vote(self) -> Tuple[str, float, int]:
        """
        Perform weighted majority voting over the buffer.
        
        Returns
        -------
        letter : str
            The winning letter (from majority voting)
        weighted_confidence : float
            Weighted confidence (0.0 to 1.0)
        persistence : int
            How many consecutive frames the winner has appeared
        """
        if not self._predictions:
            return "", 0.0, 0

        # Count votes weighted by confidence
        weighted_votes = {}
        for letter, conf in self._predictions:
            weighted_votes[letter] = weighted_votes.get(letter, 0.0) + conf

        # Winner is the letter with highest cumulative confidence
        winner = max(weighted_votes, key=weighted_votes.get)
        total_confidence = sum(weighted_votes.values())
        weighted_confidence = weighted_votes[winner] / total_confidence if total_confidence > 0 else 0.0

        # Calculate persistence: how many recent consecutive frames show the winner
        persistence = 0
        for letter, _ in reversed(list(self._predictions)):
            if letter == winner:
                persistence += 1
            else:
                break

        return winner, weighted_confidence, persistence

    def clear(self) -> None:
        """Clear all buffered predictions."""
        self._predictions.clear()
        self._persistence.clear()

    @property
    def ready(self) -> bool:
        """Return True if buffer has at least half-window of data."""
        return len(self._predictions) >= self._window // 2

    @property
    def buffer_size(self) -> int:
        """Current number of frames in buffer."""
        return len(self._predictions)

    @property
    def buffer_status(self) -> dict:
        """Return diagnostic info about current buffer state."""
        if not self._predictions:
            return {"frames": 0, "letters": []}
        
        letters = [letter for letter, _ in self._predictions]
        counts = Counter(letters)
        
        return {
            "frames": len(self._predictions),
            "letters": letters,
            "distribution": dict(counts),
            "most_common": counts.most_common(1)[0] if counts else ("", 0),
        }
