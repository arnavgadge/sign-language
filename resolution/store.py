"""
resolution/store.py
------------------------
STEP 6 — database persistence for completed resolutions.

A NEW table (`resolution_log`) — nothing like it exists in any prior
phase's schema, so this isn't a duplicate. Kept in its OWN database file
(data/resolution_history.db), separate from Layer 5's
data/personal_context.db, so Layer 5 stays exclusively responsible for
personal word->intent history (single responsibility, no cross-layer
coupling) and this module owns only resolution-outcome records.
"""
import json
import os
import sqlite3

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(_ROOT, "data", "resolution_history.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS resolution_log (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id               TEXT    NOT NULL,
    recognized_word       TEXT    NOT NULL,
    timestamp             REAL    NOT NULL,
    status                TEXT    NOT NULL,
    selected_intent       TEXT,
    final_score           REAL    NOT NULL,
    conflict_detected     INTEGER NOT NULL,
    ranked_candidates_json TEXT   NOT NULL,
    output_text           TEXT
);
CREATE INDEX IF NOT EXISTS idx_resolution_user_time
    ON resolution_log(user_id, timestamp);
"""


class ResolutionStore:
    """
    Usage
    -----
    store = ResolutionStore()                      # real DB
    store = ResolutionStore(db_path=":memory:")     # isolated, for tests

    store.record(resolution_result)
    rows = store.get_history(user_id="001")
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

    def record(self, result) -> None:
        """`result` is a resolution.models.ResolutionResult."""
        self._conn.execute(
            "INSERT INTO resolution_log "
            "(user_id, recognized_word, timestamp, status, selected_intent, "
            " final_score, conflict_detected, ranked_candidates_json, output_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.user_id, result.recognized_word, result.evaluated_at, result.status,
                result.selected_intent, result.final_score, int(result.conflict),
                json.dumps(result.ranked_candidates), result.output,
            ),
        )
        self._conn.commit()

    def get_history(self, user_id: str, limit: int = 50) -> list:
        cur = self._conn.execute(
            "SELECT user_id, recognized_word, timestamp, status, selected_intent, "
            "final_score, conflict_detected, ranked_candidates_json, output_text "
            "FROM resolution_log WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        )
        rows = []
        for r in cur.fetchall():
            rows.append({
                "user_id": r[0], "recognized_word": r[1], "timestamp": r[2], "status": r[3],
                "selected_intent": r[4], "final_score": r[5], "conflict_detected": bool(r[6]),
                "ranked_candidates": json.loads(r[7]), "output_text": r[8],
            })
        return rows

    def close(self) -> None:
        self._conn.close()
