"""
evaluation/final_error_analysis.py
--------------------------------------
PHASE 9 -- classifies every incorrect full-engine prediction into a
failure category. Heuristic, best-effort -- the FULL evidence is saved
alongside the category so a human can re-classify if the heuristic is
wrong. Nothing is hidden; every failure is reported.
"""
from typing import List

from .final_variants import FinalVariantResult

CATEGORIES = [
    "candidate-generation failure", "adversarial trigger",
    "reliability-weighting / conflict failure", "personal-context (cold start) failure",
    "insufficient evidence", "confidence-gate false-confident resolution",
    "confidence-gate over-cautious", "other",
]


def classify_failure(r: FinalVariantResult) -> str:
    if r.ground_truth not in ("AMBIGUOUS", "LITERAL") and not r.ground_truth_in_candidates:
        return "candidate-generation failure"
    if r.scenario_category == "adversarial":
        return "adversarial trigger"
    if r.scenario_category == "conflict":
        return "reliability-weighting / conflict failure"
    if r.scenario_category == "cold_start":
        return "personal-context (cold start) failure"
    if r.temporal_signal == 0.0 and r.personal_signal == 0.0 and r.ground_truth not in ("LITERAL", "AMBIGUOUS"):
        return "insufficient evidence"
    if r.status == "RESOLVED" and not r.correct:
        return "confidence-gate false-confident resolution"
    if r.status in ("AMBIGUOUS", "LITERAL") and r.ground_truth not in ("AMBIGUOUS", "LITERAL"):
        return "confidence-gate over-cautious"
    return "other"


def build_error_analysis(results: List[FinalVariantResult], variant: str = "full_reliability_weighted") -> list:
    subset = [r for r in results if r.variant == variant and not r.correct]
    rows = []
    for r in subset:
        rows.append({
            "sample_id": r.sample_id, "scenario_category": r.scenario_category,
            "ground_truth": r.ground_truth, "predicted_intent": r.predicted_intent,
            "recognized_word": r.recognized_word, "conversation_sequence": r.conversation_sequence,
            "user_id": r.user_id, "layer1_candidates": r.layer1_candidate_ids,
            "temporal_signal": r.temporal_signal, "temporal_reliability": r.temporal_reliability,
            "personal_signal": r.personal_signal, "personal_reliability": r.personal_reliability,
            "top_score": r.top_score, "second_score": r.second_score, "margin": r.margin,
            "conflict": r.conflict, "status": r.status,
            "failure_category": classify_failure(r),
        })
    return rows


def failure_category_counts(error_rows: list) -> dict:
    counts = {}
    for row in error_rows:
        cat = row["failure_category"]
        counts[cat] = counts.get(cat, 0) + 1
    return counts
