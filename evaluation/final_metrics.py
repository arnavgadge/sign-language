"""
evaluation/final_metrics.py
-------------------------------
PHASE 9 -- all metrics, computed directly from stored FinalVariantResult
rows. Nothing here re-runs the engine, and nothing is adjusted after
computation.
"""
import statistics
from typing import List

from sklearn.metrics import precision_recall_fscore_support

from .final_variants import FinalVariantResult, VARIANTS


def filter_variant(results: List[FinalVariantResult], variant: str) -> List[FinalVariantResult]:
    return [r for r in results if r.variant == variant]


# ── A. Candidate-generation metrics (Layer 1 -- NOT reported as "accuracy") ──

def candidate_coverage(results: List[FinalVariantResult]) -> dict:
    """Fraction of samples where the independently-defined ground_truth
    intent was present in Layer 1's candidate set at all. AMBIGUOUS/
    LITERAL ground-truth samples are excluded -- coverage only makes
    sense against a specific intent_id."""
    subset = [r for r in filter_variant(results, "L1_candidates")
              if r.ground_truth not in ("AMBIGUOUS", "LITERAL")]
    if not subset:
        return {"n": 0, "coverage": float("nan")}
    covered = sum(1 for r in subset if r.ground_truth_in_candidates)
    return {"n": len(subset), "coverage": covered / len(subset)}


def top_k_candidate_recall(results: List[FinalVariantResult], k_values=(1, 3, 5)) -> dict:
    """Among samples where ground truth WAS in Layer 1's candidate set,
    what fraction does the full reliability-weighted engine rank it
    within the top K, using the REAL recorded rank (ground_truth_rank),
    not an approximation."""
    full = filter_variant(results, "full_reliability_weighted")
    eligible = [r for r in full if r.ground_truth not in ("AMBIGUOUS", "LITERAL")
                and r.ground_truth_in_candidates]
    out = {"n_eligible": len(eligible)}
    if not eligible:
        for k in k_values:
            out[f"top_{k}"] = float("nan")
        return out
    for k in k_values:
        hits = sum(1 for r in eligible if 1 <= r.ground_truth_rank <= k)
        out[f"top_{k}"] = hits / len(eligible)
    return out


# ── B. Classification metrics ────────────────────────────────────────────────

def accuracy(results: List[FinalVariantResult]) -> float:
    if not results:
        return float("nan")
    return sum(1 for r in results if r.correct) / len(results)


def status_accuracy(results: List[FinalVariantResult]) -> float:
    """Fraction where the DECISION STATUS (RESOLVED/AMBIGUOUS/LITERAL)
    matched expected_status, regardless of which specific intent."""
    if not results:
        return float("nan")
    return sum(1 for r in results if r.status_correct) / len(results)


def macro_prf1(results: List[FinalVariantResult]) -> dict:
    if not results:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    y_true = [r.ground_truth for r in results]
    y_pred = [r.predicted_intent for r in results]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"precision": p, "recall": r, "f1": f1}


# ── C. Decision metrics ──────────────────────────────────────────────────────

def decision_rates(results: List[FinalVariantResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "resolved_rate": sum(1 for r in results if r.status == "RESOLVED") / n,
        "ambiguous_rate": sum(1 for r in results if r.status == "AMBIGUOUS") / n,
        "literal_rate": sum(1 for r in results if r.status == "LITERAL") / n,
    }


def false_confident_resolution_rate(results: List[FinalVariantResult]) -> dict:
    resolved = [r for r in results if r.status == "RESOLVED"]
    if not resolved:
        return {"n_resolved": 0, "rate": float("nan")}
    wrong = sum(1 for r in resolved if not r.correct)
    return {"n_resolved": len(resolved), "n_wrong": wrong, "rate": wrong / len(resolved)}


def confidence_gate_breakdown(results: List[FinalVariantResult]) -> dict:
    """Six-way breakdown requested by Phase 9: correct/incorrect crossed
    with resolved/ambiguous/literal."""
    out = {}
    for status in ("RESOLVED", "AMBIGUOUS", "LITERAL"):
        subset = [r for r in results if r.status == status]
        correct = sum(1 for r in subset if r.status_correct)
        out[status] = {"n": len(subset), "status_correct": correct,
                        "status_incorrect": len(subset) - correct}
    return out


# ── D. Conflict metrics ──────────────────────────────────────────────────────

def conflict_metrics(results: List[FinalVariantResult]) -> dict:
    """Precision/recall/F1 for conflict detection, using scenario_category
    == 'conflict' AND expected_status == 'AMBIGUOUS' as the positive class
    (a conflict case where evidence was designed to be genuinely balanced)."""
    tp = fp = fn = tn = 0
    for r in results:
        designed_conflict = (r.scenario_category == "conflict" and r.expected_status == "AMBIGUOUS")
        if designed_conflict and r.conflict:
            tp += 1
        elif designed_conflict and not r.conflict:
            fn += 1
        elif not designed_conflict and r.conflict:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision == precision and recall == recall and (precision + recall) > 0) else float("nan"))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


# ── E. Personalization metrics ──────────────────────────────────────────────

def personalization_by_history_size(cold_start_results: List[FinalVariantResult],
                                     cases_by_id: dict) -> List[dict]:
    """cases_by_id: sample_id -> FinalEvalCase (to recover observation count)."""
    rows = []
    for r in cold_start_results:
        case = cases_by_id.get(r.sample_id)
        if case is None or not case.personal_seed:
            obs = 0
        else:
            obs = case.personal_seed[0][2]
        rows.append({
            "sample_id": r.sample_id, "observation_count": obs,
            "personal_signal": r.personal_signal, "personal_reliability": r.personal_reliability,
            "correct": r.correct,
        })
    return sorted(rows, key=lambda x: x["observation_count"])


# ── F. Performance metrics ──────────────────────────────────────────────────

def latency_stats(results: List[FinalVariantResult]) -> dict:
    times = [r.processing_time_ms for r in results]
    if not times:
        return {}
    return {
        "mean_ms": statistics.mean(times), "median_ms": statistics.median(times),
        "min_ms": min(times), "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


# ── G. Robustness ────────────────────────────────────────────────────────────

def adversarial_accuracy(results: List[FinalVariantResult]) -> dict:
    subset = [r for r in results if r.scenario_category == "adversarial"]
    return {"n": len(subset), "accuracy": accuracy(subset)}


def ambiguous_case_handling(results: List[FinalVariantResult]) -> dict:
    subset = [r for r in results if r.scenario_category == "genuinely_ambiguous"]
    correctly_flagged = sum(1 for r in subset if r.predicted_intent == "AMBIGUOUS")
    return {"n": len(subset),
            "correctly_flagged_ambiguous": correctly_flagged,
            "rate": correctly_flagged / len(subset) if subset else float("nan")}


def cold_start_behavior(results: List[FinalVariantResult]) -> dict:
    subset = [r for r in results if r.scenario_category == "cold_start"]
    return {"n": len(subset), "status_accuracy": status_accuracy(subset)}


def full_summary(results: List[FinalVariantResult]) -> dict:
    out = {}
    for variant in VARIANTS:
        subset = filter_variant(results, variant)
        out[variant] = {
            "n": len(subset),
            "accuracy": accuracy(subset),
            "status_accuracy": status_accuracy(subset),
            **{f"macro_{k}": v for k, v in macro_prf1(subset).items()},
            "decision_rates": decision_rates(subset),
            "false_confident_resolution": false_confident_resolution_rate(subset),
            "latency": latency_stats(subset),
        }
    return out
