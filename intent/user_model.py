"""
Per-user behavioral model stored as JSON.

Tracks intent frequency per word (what the user has historically meant),
word-pair sequences, and session statistics.  Used by the Personal Context
Layer (Stage 2) and Personal Pattern Pruning (Stage 1C).
"""
import json
import os
import time
from typing import Dict


class UserModel:
    def __init__(self, user_id: str = "default", data_dir: str = "data/users"):
        self.user_id = user_id
        self.data_dir = data_dir
        self.path = os.path.join(data_dir, f"{user_id}.json")
        self._data = self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r") as fh:
                return json.load(fh)
        return {
            "user_id": self.user_id,
            "created": time.time(),
            "total_interactions": 0,
            "word_intent_counts": {},   # word -> {intent_id -> count}
            "word_pairs": {},           # word -> {next_word -> count}
            "session_count": 0,
        }

    def save(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(self._data, fh, indent=2)

    # ── Write ──────────────────────────────────────────────────────────────────

    def record_intent(self, word: str, intent_id: str) -> None:
        word = word.upper()
        wic = self._data["word_intent_counts"]
        wic.setdefault(word, {})[intent_id] = wic.get(word, {}).get(intent_id, 0) + 1
        self._data["total_interactions"] += 1
        self.save()

    def record_pair(self, word1: str, word2: str) -> None:
        w1, w2 = word1.upper(), word2.upper()
        wp = self._data["word_pairs"]
        wp.setdefault(w1, {})[w2] = wp.get(w1, {}).get(w2, 0) + 1
        self.save()

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_intent_priors(self, word: str) -> Dict[str, float]:
        """Normalized frequency-based prior for each intent the user has used."""
        word = word.upper()
        counts = self._data["word_intent_counts"].get(word, {})
        total = sum(counts.values())
        if not total:
            return {}
        return {intent_id: cnt / total for intent_id, cnt in counts.items()}

    def get_total_interactions(self) -> int:
        return self._data.get("total_interactions", 0)

    def has_sufficient_history(self, word: str, min_samples: int = 5) -> bool:
        word = word.upper()
        counts = self._data["word_intent_counts"].get(word, {})
        return sum(counts.values()) >= min_samples

    def compute_reliability(self, word: str) -> float:
        """
        0.0 → new user with no data
        1.0 → user has 20+ recorded interactions for this word
        """
        total = self.get_total_interactions()
        if total < 3:
            return 0.1
        word_samples = sum(
            self._data["word_intent_counts"].get(word.upper(), {}).values()
        )
        return min(1.0, 0.1 + (word_samples / 20.0) * 0.9)
