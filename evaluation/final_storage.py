"""
evaluation/final_storage.py
-------------------------------
PHASE 9 -- writes all required final data files, generated
programmatically from real objects (no hand-typed numbers).
"""
import csv
import json
import os
from dataclasses import asdict, fields
from typing import List

from .final_dataset import FinalEvalCase
from .final_variants import FinalVariantResult

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "evaluation", "results")


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_dataset(cases: List[FinalEvalCase], path: str) -> None:
    rows = []
    for c in cases:
        rows.append({
            "sample_id": c.sample_id, "scenario_category": c.scenario_category,
            "recognized_word": c.recognized_word,
            "conversation_sequence": " > ".join(c.conversation_sequence),
            "sequence_offsets_seconds": "|".join(f"{o:.2f}" for o in c.sequence_offsets_seconds),
            "user_id": c.user_id, "ground_truth": c.ground_truth, "expected_status": c.expected_status,
            "personal_seed": ";".join(f"{w}:{i}:{cnt}:{d}d" for w, i, cnt, d in c.personal_seed),
            "notes": c.notes,
        })
    _write_csv(path, rows, list(rows[0].keys()) if rows else [])


def save_predictions(results: List[FinalVariantResult], path: str) -> None:
    field_names = [f.name for f in fields(FinalVariantResult)]
    rows = [asdict(r) for r in results]
    _write_csv(path, rows, field_names)


def save_metrics(metrics_dict: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics_dict, fh, indent=2, default=str)


def save_error_analysis(error_rows: List[dict], path: str) -> None:
    if not error_rows:
        _write_csv(path, [], ["sample_id"])
        return
    _write_csv(path, error_rows, list(error_rows[0].keys()))


def save_ablation_results(ablation_rows: List[dict], path: str) -> None:
    _write_csv(path, ablation_rows, list(ablation_rows[0].keys()) if ablation_rows else [])


def save_latency_results(latency_rows: List[dict], path: str) -> None:
    _write_csv(path, latency_rows, list(latency_rows[0].keys()) if latency_rows else [])


def save_text_report(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
