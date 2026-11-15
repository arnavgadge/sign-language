"""
evaluation/metrics.py
-------------------------
All metrics computed directly from stored VariantResult rows — nothing
here re-runs the engine or adjusts a result after the fact.
"""
import statistics
from typing import List

from sklearn.metrics import precision_recall_fscore_support

from .variants import VariantResult


def filter_variant(results: List[VariantResult], variant: str) -> List[VariantResult]:
    return [r for r in results if r.variant == variant]


def accuracy(results: List[VariantResult]) -> float:
    if not results:
        return float("nan")
    return sum(1 for r in results if r.correct) / len(results)


def macro_prf1(results: List[VariantResult]) -> dict:
    if not results:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    y_true = [r.ground_truth for r in results]
    y_pred = [r.predicted_intent for r in results]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"precision": p, "recall": r, "f1": f1}


def ambiguity_detection_rate(results: List[VariantResult]) -> dict:
    """Among samples DESIGNED to be ambiguous (ground_truth == 'AMBIGUOUS'),
    what fraction did the system correctly flag as AMBIGUOUS?
    Also reports the broader 'did not overinterpret' rate (AMBIGUOUS or LITERAL)."""
    targets = [r for r in results if r.ground_truth == "AMBIGUOUS"]
    if not targets:
        return {"n": 0, "exact_ambiguous_rate": float("nan"), "non_overinterpret_rate": float("nan")}
    exact = sum(1 for r in targets if r.predicted_intent == "AMBIGUOUS") / len(targets)
    restrained = sum(1 for r in targets if r.predicted_intent in ("AMBIGUOUS", "LITERAL")) / len(targets)
    return {"n": len(targets), "exact_ambiguous_rate": exact, "non_overinterpret_rate": restrained}


def false_confident_resolution_rate(results: List[VariantResult]) -> dict:
    """Among all RESOLVED predictions, what fraction were WRONG? This is
    the "confident but wrong" failure mode."""
    resolved = [r for r in results if r.status == "RESOLVED"]
    if not resolved:
        return {"n_resolved": 0, "rate": float("nan")}
    wrong = sum(1 for r in resolved if not r.correct)
    return {"n_resolved": len(resolved), "rate": wrong / len(resolved)}


def literal_fallback_rate(results: List[VariantResult]) -> float:
    if not results:
        return float("nan")
    return sum(1 for r in results if r.status == "LITERAL") / len(results)


def conflict_detection_rates(results: List[VariantResult]) -> dict:
    conflict_category = [r for r in results if r.category == "conflict"]
    non_conflict_category = [r for r in results if r.category != "conflict"]
    return {
        "true_positive_rate": (sum(1 for r in conflict_category if r.conflict) / len(conflict_category)
                                if conflict_category else float("nan")),
        "n_conflict_category": len(conflict_category),
        "false_positive_rate": (sum(1 for r in non_conflict_category if r.conflict) / len(non_conflict_category)
                                 if non_conflict_category else float("nan")),
        "n_non_conflict_category": len(non_conflict_category),
    }


def latency_stats(results: List[VariantResult]) -> dict:
    times = [r.processing_time_ms for r in results]
    if not times:
        return {}
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def summarize_variant(results: List[VariantResult], variant: str) -> dict:
    subset = filter_variant(results, variant)
    return {
        "variant": variant,
        "n_samples": len(subset),
        "accuracy": accuracy(subset),
        **{f"macro_{k}": v for k, v in macro_prf1(subset).items()},
        "ambiguity_detection": ambiguity_detection_rate(subset),
        "false_confident_resolution": false_confident_resolution_rate(subset),
        "literal_fallback_rate": literal_fallback_rate(subset),
        "conflict_detection": conflict_detection_rates(subset),
        "latency": latency_stats(subset),
    }
