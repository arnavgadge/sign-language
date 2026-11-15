"""
Sliding-window conversation history for the context layer.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List


@dataclass
class ConversationEntry:
    word: str
    timestamp: float
    resolved_intent: str = ""


class ConversationHistory:
    def __init__(self, window_seconds: float = 120.0, max_entries: int = 20):
        self.window_seconds = window_seconds
        self._entries: deque = deque(maxlen=max_entries)

    def add(self, word: str, resolved_intent: str = "", timestamp: float = None) -> None:
        """`timestamp` is optional and defaults to now() — added so callers
        (e.g. deterministic tests) can inject a specific capture time without
        changing any existing call site's behavior."""
        self._entries.append(ConversationEntry(
            word=word.upper(),
            timestamp=timestamp if timestamp is not None else time.time(),
            resolved_intent=resolved_intent
        ))

    def recent_words(self, seconds: float = None) -> List[str]:
        cutoff = time.time() - (seconds if seconds is not None else self.window_seconds)
        return [e.word for e in self._entries if e.timestamp >= cutoff]

    def entries_since(self, cutoff: float) -> List[ConversationEntry]:
        """Full entries (word + timestamp + resolved_intent) at or after
        `cutoff` (an epoch timestamp), oldest first. Unlike `recent_words()`,
        this exposes timestamps and accepts an explicit cutoff instead of
        always measuring from time.time(), so callers can evaluate a window
        relative to a caller-supplied `now`."""
        return [e for e in self._entries if e.timestamp >= cutoff]

    def time_since_last(self) -> float:
        """Seconds since last entry, or inf if history is empty."""
        if not self._entries:
            return float("inf")
        return time.time() - self._entries[-1].timestamp

    def is_fresh_conversation(self, gap_threshold: float = 60.0) -> bool:
        return self.time_since_last() > gap_threshold

    def clear(self) -> None:
        self._entries.clear()
