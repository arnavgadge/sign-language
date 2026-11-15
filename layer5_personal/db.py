"""
layer5_personal/db.py
------------------------
SQLite storage for Layer 5 (Personal Context). One aggregate row per
(user_id, word, intent_id) triple — matching exactly the schema shape
requested: user_id, word, intent, occurrence_count, first_seen, last_seen.

A new database (data/personal_context.db by default) — there was no
pre-existing SQLite database anywhere in this project to reuse (verified
by inspection; only prior personal-data mechanism was the JSON-based
intent/user_model.py, which belongs to Layer 1's dependency chain and is
left untouched here).
"""
import os
import sqlite3
import time
from typing import List, NamedTuple, Optional

from .config import DEFAULT_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS personal_word_intent (
    user_id          TEXT    NOT NULL,
    word             TEXT    NOT NULL,
    intent_id        TEXT    NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    first_seen       REAL    NOT NULL,
    last_seen        REAL    NOT NULL,
    PRIMARY KEY (user_id, word, intent_id)
);
CREATE INDEX IF NOT EXISTS idx_personal_user_word
    ON personal_word_intent(user_id, word);
"""


class WordIntentRow(NamedTuple):
    intent_id: str
    occurrence_count: int
    first_seen: float
    last_seen: float


class PersonalContextDB:
    """
    Usage
    -----
    db = PersonalContextDB()                      # real DB, data/personal_context.db
    db = PersonalContextDB(db_path=":memory:")     # isolated, for tests

    db.record(user_id="001", word="HELP", intent_id="medical_assistance")
    rows = db.get_word_rows(user_id="001", word="HELP")
    total = db.get_user_total_observations(user_id="001")
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        if db_path != ":memory:":
            directory = os.path.dirname(db_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────

    def record(self, user_id: str, word: str, intent_id: str,
               timestamp: float = None) -> None:
        """Increment occurrence_count for (user_id, word, intent_id),
        creating the row if it doesn't exist yet. Updates last_seen;
        first_seen is set once and never changed."""
        ts = timestamp if timestamp is not None else time.time()
        word = word.upper().strip()

        cur = self._conn.execute(
            "SELECT occurrence_count FROM personal_word_intent "
            "WHERE user_id=? AND word=? AND intent_id=?",
            (user_id, word, intent_id),
        )
        row = cur.fetchone()

        if row is None:
            self._conn.execute(
                "INSERT INTO personal_word_intent "
                "(user_id, word, intent_id, occurrence_count, first_seen, last_seen) "
                "VALUES (?, ?, ?, 1, ?, ?)",
                (user_id, word, intent_id, ts, ts),
            )
        else:
            self._conn.execute(
                "UPDATE personal_word_intent SET occurrence_count = ?, last_seen = ? "
                "WHERE user_id=? AND word=? AND intent_id=?",
                (row[0] + 1, ts, user_id, word, intent_id),
            )
        self._conn.commit()

    # ── Read ───────────────────────────────────────────────────────────────

    def get_word_rows(self, user_id: str, word: str) -> List[WordIntentRow]:
        word = word.upper().strip()
        cur = self._conn.execute(
            "SELECT intent_id, occurrence_count, first_seen, last_seen "
            "FROM personal_word_intent WHERE user_id=? AND word=?",
            (user_id, word),
        )
        return [WordIntentRow(*r) for r in cur.fetchall()]

    def get_user_total_observations(self, user_id: str) -> int:
        cur = self._conn.execute(
            "SELECT COALESCE(SUM(occurrence_count), 0) FROM personal_word_intent WHERE user_id=?",
            (user_id,),
        )
        return int(cur.fetchone()[0])

    def get_user_distinct_words(self, user_id: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(DISTINCT word) FROM personal_word_intent WHERE user_id=?",
            (user_id,),
        )
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self._conn.close()
