"""
utils/stabilizer.py
--------------------
Temporal stabilization of per-frame letter predictions.

Problem
-------
A CNN running on a 30 fps webcam stream produces a new prediction every
~33 ms.  Without stabilization:
  - A single letter would be appended dozens of times per second.
  - Brief mis-classifications would inject noise into the word.

Solution — four-layer defence
-------------------------------
1. Confidence gate     – reject predictions below a threshold.
2. Hold-frame counter  – require the *same* letter for N consecutive frames
                         before accepting it as stable.
3. Cooldown timer      – after appending, ignore all input for M frames
                         (prevents the same letter being immediately re-appended).
4. Gap detector        – if confidence stays low for G frames (hand removed),
                         reset last_appended so the same letter CAN be typed
                         again in the next gesture.

Typical tuning (at ~30 fps)
----------------------------
  confidence_threshold = 0.80   →  80 % confidence required
  required_frames      = 20     →  hold for ~0.67 s
  cooldown_frames      = 35     →  ~1.17 s lock after append
  gap_frames           = 12     →  ~0.4 s of low-confidence clears the lock
"""


class LetterStabilizer:
    """
    Stateful stabilizer: call update() once per frame.

    Parameters
    ----------
    confidence_threshold : float
        Minimum model confidence to consider a prediction valid.
    required_frames : int
        How many consecutive frames the same letter must appear before
        it is accepted.
    cooldown_frames : int
        Frames to ignore after a letter is appended (debounce).
    gap_frames : int
        Frames of low-confidence input required to allow the same letter
        to be typed again.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        required_frames: int = 20,
        cooldown_frames: int = 35,
        gap_frames: int = 12,
    ):
        self.confidence_threshold = confidence_threshold
        self.required_frames = required_frames
        self.cooldown_frames = cooldown_frames
        self.gap_frames = gap_frames

        # Internal state
        self._hold_letter: str = ""
        self._hold_count: int = 0
        self._last_appended: str = ""
        self._cooldown: int = 0
        self._gap_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, letter: str, confidence: float) -> str | None:
        """
        Feed one frame's prediction.

        Returns
        -------
        str   – the letter to append to current_word, OR
        None  – nothing to append this frame.
        """
        # ── Layer 3: Cooldown ───────────────────────────────────────────
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        # ── Layer 1: Confidence gate ────────────────────────────────────
        if confidence < self.confidence_threshold:
            self._gap_count += 1
            # Layer 4: Gap detection – allow same letter again after gap
            if self._gap_count >= self.gap_frames:
                self._last_appended = ""
                self._hold_letter = ""
                self._hold_count = 0
            return None
        else:
            self._gap_count = 0  # confident frame – reset gap counter

        # ── Layer 2: Hold-frame counter ─────────────────────────────────
        if letter == self._hold_letter:
            self._hold_count += 1
        else:
            self._hold_letter = letter
            self._hold_count = 1

        # ── Emit letter when stable AND different from last appended ────
        if (
            self._hold_count >= self.required_frames
            and self._hold_letter != self._last_appended
        ):
            accepted = self._hold_letter
            self._last_appended = accepted
            self._cooldown = self.cooldown_frames
            self._hold_count = 0
            return accepted

        return None

    def get_status(self) -> dict:
        """
        Return current internal state for HUD display.

        Keys
        ----
        hold_letter    : letter currently being held
        hold_count     : consecutive frames held so far
        required_frames: target frames to accept
        cooldown       : remaining cooldown frames
        last_appended  : last letter that was emitted
        progress       : float 0–1, hold_count / required_frames
        """
        return {
            "hold_letter": self._hold_letter,
            "hold_count": self._hold_count,
            "required_frames": self.required_frames,
            "cooldown": self._cooldown,
            "last_appended": self._last_appended,
            "progress": min(self._hold_count / self.required_frames, 1.0),
        }

    def reset(self) -> None:
        """Fully reset the stabilizer (e.g., after clearing the word)."""
        self._hold_letter = ""
        self._hold_count = 0
        self._last_appended = ""
        self._cooldown = 0
        self._gap_count = 0
