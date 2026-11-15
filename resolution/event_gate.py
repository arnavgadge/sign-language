"""
resolution/event_gate.py
----------------------------
STEP 8 — duplicate event protection.

The existing recognition loop runs continuously (webcam frame rate); the
Enter-key handler already only fires on an actual keypress, not every
frame, but a held key (OS auto-repeat) or an accidental double-press
without any new letters typed would otherwise resubmit the SAME
unchanged word to the Intent Engine repeatedly. This gate makes that
explicit and independently testable, rather than relying only on
"the key handler happened to only fire once."
"""


class WordEventGate:
    """
    Usage
    -----
    gate = WordEventGate()
    if gate.should_process(current_word):
        engine.resolve(user_id, current_word)

    Only blocks IMMEDIATE repeats of the identical word — a word that
    reoccurs later after a different word in between is processed again,
    since that's a genuine new event, not a duplicate of the same one.
    """

    def __init__(self):
        self._last_word = None

    def should_process(self, word: str) -> bool:
        normalized = (word or "").upper().strip()
        if not normalized or normalized == self._last_word:
            return False
        self._last_word = normalized
        return True

    def reset(self) -> None:
        self._last_word = None
