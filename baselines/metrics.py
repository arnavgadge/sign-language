"""
baselines/metrics.py
------------------------
Metrics over the rows produced by baselines.common.run_method. Nothing here
re-runs a method. scenario_category is used only for per-category reporting.

Conventions
  forced  -- the 378 cases whose ground_truth is an intent id; every method's
             top-ranked intent is scored (the engine's resolver top-1,
             regardless of gate decision).
  full    -- all 500 cases; baselines always answer with an intent (so they
             can be right on at most 378/500 = 75.6%), the engine is scored on
             its gated label exactly as evaluation/final_variants.py does.
  risk-coverage -- forced cases sorted by the method's confidence. Tied
             confidences are treated as randomly ordered (expected accuracy),
             so the curve does not depend on dataset order. AURC = mean risk
             over every coverage level k/n, k = 1..n (lower is better).
"""
import math
import random
import statistics
from collections import defaultdict
from typing import Dict, List

from sklearn.metrics import precision_recall_fscore_support

BOOTSTRAP_SEED = 20240521
BOOTSTRAP_RESAMPLES = 1000
COVERAGE_GRID = [round(0.1 * i, 1) for i in range(1, 11)]


def _acc(flags: List[bool]) -> float:
    return sum(flags) / len(flags) if flags else float("nan")


def bootstrap_ci(flags: List[bool], seed: int = BOOTSTRAP_SEED,
                 n_resamples: int = BOOTSTRAP_RESAMPLES, alpha: float = 0.05) -> tuple:
    """Percentile CI. Same seed for every method -> identical (paired) resample indices."""
    rng = random.Random(seed)
    n = len(flags)
    stats = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    lo = stats[int(math.floor(alpha / 2 * n_resamples))]
    hi = stats[int(math.ceil((1 - alpha / 2) * n_resamples)) - 1]
    return lo, hi


def mcnemar_exact(ref: List[bool], other: List[bool]) -> Dict:
    """Exact two-sided McNemar (binomial on discordant pairs).
    b = ref right / other wrong, c = ref wrong / other right."""
    b = sum(1 for r, o in zip(ref, other) if r and not o)
    c = sum(1 for r, o in zip(ref, other) if o and not r)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}


def expected_correct_prefix(scores: List[float], correct: List[bool]) -> List[float]:
    """E[#correct among the k most confident], k = 0..n, ties in random order."""
    groups = defaultdict(lambda: [0, 0])
    for s, ok in zip(scores, correct):
        groups[s][0] += 1
        groups[s][1] += int(ok)
    out, base = [0.0], 0
    for s in sorted(groups, reverse=True):
        size, n_ok = groups[s]
        out.extend(base + n_ok * j / size for j in range(1, size + 1))
        base += n_ok
    return out


def risk_coverage(scores: List[float], correct: List[bool]) -> Dict:
    n = len(scores)
    prefix = expected_correct_prefix(scores, correct)
    grid = []
    for cov in COVERAGE_GRID:
        k = max(1, math.ceil(cov * n - 1e-9))
        acc = prefix[k] / k
        grid.append({"coverage": cov, "n_answered": k, "accuracy": acc, "risk": 1 - acc})
    aurc = statistics.mean(1 - prefix[k] / k for k in range(1, n + 1))
    return {"grid": grid, "aurc": aurc}


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    return precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]


def gated_decision_metrics(rows: List[Dict]) -> Dict:
    n = len(rows)
    resolved = [r for r in rows if r["gated_status"] == "RESOLVED"]
    wrong = sum(1 for r in resolved if not r["correct_full"])
    return {
        "resolved_rate": len(resolved) / n,
        "ambiguous_rate": sum(r["gated_status"] == "AMBIGUOUS" for r in rows) / n,
        "literal_rate": sum(r["gated_status"] == "LITERAL" for r in rows) / n,
        "false_confident": {"n_resolved": len(resolved), "n_wrong": wrong,
                            "rate": wrong / len(resolved) if resolved else float("nan")},
    }


def compute_all(rows_by_method: Dict[str, List[Dict]], reference: str = "engine") -> Dict:
    """rows_by_method: method name -> rows in identical case order."""
    ref_forced = [r["correct_forced"] for r in rows_by_method[reference] if r["intent_gold"]]
    n_total = len(rows_by_method[reference])
    n_intent = len(ref_forced)

    out = {"n_total": n_total, "n_intent_gold": n_intent,
           "non_abstaining_ceiling": n_intent / n_total, "methods": {}}
    for name, rows in rows_by_method.items():
        forced = [r for r in rows if r["intent_gold"]]
        f_ok = [r["correct_forced"] for r in forced]
        full_ok = [r["correct_full"] for r in rows]
        abstains = "gated_status" in rows[0]   # only abstaining methods report a gated status

        by_cat = defaultdict(list)
        for r in forced:
            by_cat[r["scenario_category"]].append(r["correct_forced"])

        m = {
            "forced_accuracy": _acc(f_ok),
            "forced_accuracy_ci95": bootstrap_ci(f_ok),
            "forced_macro_f1": macro_f1([r["ground_truth"] for r in forced],
                                        [r["predicted_intent"] for r in forced]),
            "full_accuracy": _acc(full_ok),
            "full_accuracy_ci95": bootstrap_ci(full_ok),
            "full_ceiling": None if abstains else n_intent / n_total,
            "risk_coverage": risk_coverage([r["score"] for r in forced], f_ok),
            "mcnemar_vs_" + reference: None if name == reference else mcnemar_exact(ref_forced, f_ok),
            "latency_ms": {"mean": statistics.mean(r["latency_ms"] for r in rows),
                           "median": statistics.median(r["latency_ms"] for r in rows)},
            "per_category_forced": {c: {"n": len(v), "accuracy": _acc(v)} for c, v in by_cat.items()},
        }
        if "gated_status" in rows[0]:
            m["gated_decisions"] = gated_decision_metrics(rows)
        if "used_fallback" in rows[0]:
            m["n_used_fallback"] = sum(bool(r["used_fallback"]) for r in rows)
            m["n_used_fallback_intent_gold"] = sum(bool(r["used_fallback"]) for r in forced)
        out["methods"][name] = m
    return out
