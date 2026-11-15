"""
evaluation/results_store.py
-------------------------------
Structured, reproducible storage for evaluation results — every field
Phase 8 asked for, stored per (sample, variant) row. SQLite for
queryability + a CSV export for spreadsheet/pandas use, both generated
programmatically from the same VariantResult objects (nothing here is a
manually-maintained spreadsheet).
"""
import csv
import os
import sqlite3
import time
from dataclasses import fields
from typing import List

from .variants import VariantResult

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "evaluation", "results")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_results (
    sample_id             TEXT NOT NULL,
    category              TEXT NOT NULL,
    variant               TEXT NOT NULL,
    user_id               TEXT NOT NULL,
    recognized_word       TEXT NOT NULL,
    conversation_sequence TEXT NOT NULL,
    ground_truth          TEXT NOT NULL,
    predicted_intent      TEXT NOT NULL,
    correct               INTEGER NOT NULL,
    top_score             REAL NOT NULL,
    second_score          REAL NOT NULL,
    margin                REAL NOT NULL,
    status                TEXT NOT NULL,
    conflict              INTEGER NOT NULL,
    temporal_signal       REAL NOT NULL,
    temporal_reliability  REAL NOT NULL,
    personal_signal       REAL NOT NULL,
    personal_reliability  REAL NOT NULL,
    processing_time_ms    REAL NOT NULL,
    PRIMARY KEY (sample_id, variant)
);
"""

_FIELD_NAMES = [f.name for f in fields(VariantResult)]


def save_results(results: List[VariantResult], run_label: str = None) -> dict:
    """Writes results to a fresh timestamped SQLite DB and a matching CSV.
    Returns the paths written."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = run_label or str(int(time.time()))

    db_path = os.path.join(RESULTS_DIR, f"eval_run_{ts}.db")
    csv_path = os.path.join(RESULTS_DIR, f"eval_run_{ts}.csv")

    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    for r in results:
        conn.execute(
            f"INSERT INTO eval_results ({', '.join(_FIELD_NAMES)}) "
            f"VALUES ({', '.join(['?'] * len(_FIELD_NAMES))})",
            [int(getattr(r, f)) if isinstance(getattr(r, f), bool) else getattr(r, f) for f in _FIELD_NAMES],
        )
    conn.commit()
    conn.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_FIELD_NAMES)
        for r in results:
            writer.writerow([getattr(r, f) for f in _FIELD_NAMES])

    return {"db_path": db_path, "csv_path": csv_path, "n_rows": len(results)}


def load_results_from_db(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    cur = conn.execute(f"SELECT {', '.join(_FIELD_NAMES)} FROM eval_results")
    rows = [dict(zip(_FIELD_NAMES, row)) for row in cur.fetchall()]
    conn.close()
    return rows
