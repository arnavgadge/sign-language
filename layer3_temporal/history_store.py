"""
layer3_temporal/history_store.py
-----------------------------------
Reuses the EXISTING conversation-history mechanism (intent.conversation.
ConversationHistory / ConversationEntry) rather than duplicating it. This
module only adds per-user isolation on top, since the current running
system only ever instantiates one ConversationHistory (main.py creates a
single IntentEngine(user_id="default")) with no user-keying at all.

No changes were made to ConversationHistory's storage logic — two small,
backward-compatible additions were made to intent/conversation.py:
  - an optional `timestamp` parameter on `add()` (defaults to now()), so
    tests can inject deterministic capture times instead of real sleeps;
  - a public `entries_since(cutoff)` accessor, so callers can read full
    entries (with timestamps) relative to a caller-supplied `now` instead
    of always measuring from time.time() internally.
Both are additive; every existing call site behaves identically.
"""
from typing import List

from intent.conversation import ConversationHistory, ConversationEntry
from .config import TEMPORAL_WINDOW_SECONDS


class TemporalHistoryStore:
    """
    Usage
    -----
    store = TemporalHistoryStore()
    store.add("user_1", "CHEST")
    store.add("user_1", "PAIN")
    window = store.get_window("user_1", now=time.time())
    """

    def __init__(self, window_seconds: float = TEMPORAL_WINDOW_SECONDS, max_entries: int = 50):
        self.window_seconds = window_seconds
        self.max_entries = max_entries
        self._histories: dict = {}   # user_id -> ConversationHistory

    def _get_or_create(self, user_id: str) -> ConversationHistory:
        if user_id not in self._histories:
            self._histories[user_id] = ConversationHistory(
                window_seconds=self.window_seconds, max_entries=self.max_entries
            )
        return self._histories[user_id]

    def add(self, user_id: str, word: str, resolved_intent: str = "",
            timestamp: float = None) -> None:
        history = self._get_or_create(user_id)
        history.add(word, resolved_intent=resolved_intent, timestamp=timestamp)

    def get_window(self, user_id: str, now: float, window_seconds: float = None) -> List[ConversationEntry]:
        """
        Return all entries for this user within `window_seconds` of `now`,
        oldest first. Entries belonging to other users are never visible
        here — each user_id maps to its own ConversationHistory instance.
        """
        if user_id not in self._histories:
            return []
        history = self._histories[user_id]
        w = window_seconds if window_seconds is not None else self.window_seconds
        cutoff = now - w
        return history.entries_since(cutoff)

    def clear(self, user_id: str) -> None:
        if user_id in self._histories:
            self._histories[user_id].clear()
