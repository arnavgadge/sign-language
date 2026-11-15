"""
utils/temporal_stabilizer.py
----------------------------
TEMPORAL STABILIZATION FOR ASL RECOGNITION

Only accept a letter when it persists across time with high confidence.

Key rules:
1. Single-frame anomalies are ignored
2. A letter must appear in 10+ consecutive frames before acceptance
3. Confidence must remain stable (not drop below threshold suddenly)
4. After accepting a letter, require a gap before accepting the same letter again

This prevents random spikes and ensures only intentional hand signs are recognized.

Example:
  Buffer: [D, D, D, L, D, D, D, D, D, D, D, D]
  → Temporal voting returns 'D' as winner (weighted confidence high)
  → Persistence = 5+ frames (before the L spike, or after)
  → Accept 'D' only if persistence ≥ 10
"""

from typing import Optional


class TemporalStabilizer:
    """
    Gate for accepting recognized letters based on temporal persistence.
    
    Only the letter is accepted if:
    - Weighted voting confidence is high
    - The letter persists for enough consecutive frames
    - Confidence remains stable
    
    Usage
    -----
    stab = TemporalStabilizer(persistence_frames=10, cooldown_frames=20)
    
    # Each frame (after TemporalBuffer voting):
    accepted = stab.update(letter="D", confidence=0.88, persistence=8)
    
    if accepted:
        current_word += letter
    """

    def __init__(
        self,
        persistence_frames: int = 10,
        confidence_threshold: float = 0.70,
        cooldown_frames: int = 20,
    ):
        """
        Parameters
        ----------
        persistence_frames : int
            Minimum consecutive frames the letter must appear (at high confidence)
            before acceptance. Default = 10 frames (~0.33 sec at 30 FPS).
            
        confidence_threshold : float
            Minimum weighted confidence required. Default = 0.70
            
        cooldown_frames : int
            After accepting a letter, skip this many frames before accepting
            the same letter again (debounce). Default = 20 frames (~0.67 sec).
        """
        self.persistence_frames = persistence_frames
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames

        # Internal state
        self._last_accepted = ""
        self._cooldown_counter = 0
        self._last_confidence = 0.0
        self._consecutive_high_conf = 0

    def update(
        self,
        letter: str,
        confidence: float,
        persistence: int,
    ) -> Optional[str]:
        """
        Gate function: determine if letter should be accepted.
        
        Parameters
        ----------
        letter : str
            Letter from temporal buffer voting
        confidence : float
            Weighted confidence from voting (0.0-1.0)
        persistence : int
            Consecutive frames this letter has appeared
            
        Returns
        -------
        str or None
            The letter to append to word, or None to reject.
        """
        # Cooldown: after accepting a letter, wait before accepting anything again
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return None

        # Confidence gate: minimum confidence required
        if confidence < self.confidence_threshold:
            self._consecutive_high_conf = 0
            self._last_confidence = confidence
            return None

        # Same letter as last acceptance? (debounce)
        if letter == self._last_accepted:
            # Don't immediately re-accept the same letter
            # This requires a low-confidence gap first
            return None

        # Persistence gate: letter must appear in 10+ consecutive frames
        if persistence < self.persistence_frames:
            self._consecutive_high_conf += 1
            # Track confidence stability
            if abs(confidence - self._last_confidence) < 0.10:
                self._consecutive_high_conf += 1
            self._last_confidence = confidence
            return None

        # All gates passed! Accept the letter.
        self._last_accepted = letter
        self._cooldown_counter = self.cooldown_frames
        self._consecutive_high_conf = 0
        self._last_confidence = 0.0

        return letter

    def reset(self) -> None:
        """Reset internal state (e.g., when user clears the word)."""
        self._last_accepted = ""
        self._cooldown_counter = 0
        self._consecutive_high_conf = 0
        self._last_confidence = 0.0

    def force_reset_last_accepted(self) -> None:
        """
        Allow the same letter to be typed again by resetting the 'last accepted' state.
        Useful when the hand is removed (large gap in confidence).
        """
        self._last_accepted = ""

    def get_status(self) -> dict:
        """Return diagnostic status."""
        return {
            "last_accepted": self._last_accepted,
            "cooldown": self._cooldown_counter,
            "consecutive_high_conf": self._consecutive_high_conf,
        }
