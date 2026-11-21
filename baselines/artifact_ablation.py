"""
baselines/artifact_ablation.py
----------------------------------
Re-runs the tuned LR and RF classifiers (same out-of-fold protocol as
run_baselines.py) with the dataset-artifact features removed:
seq_len, has_prev, last_offset, and every filler-word (SAFE_DISTRACTORS)
context / interaction feature. See classifier_features.featurize_no_artifacts.

    python -m baselines.artifact_ablation       -> baselines/results/artifact_ablation.json
"""
import json
import logging
import os
from collections import Counter

from evaluation.scenario_spec import SAFE_DISTRACTORS

from .classifier import classifier_lr, classifier_rf
from .classifier_features import SEQUENCE_SHAPE_FEATURES, featurize_no_artifacts
from .common import RESULTS_DIR, load_cases, run_method
from .engine_ref import EngineRef
from .metrics import compute_all

KEEP = ("forced_accuracy", "forced_accuracy_ci95", "forced_macro_f1", "full_accuracy",
        "full_accuracy_ci95", "gated_decisions", "mcnemar_vs_engine", "per_category_forced")


def dataset_artifact_evidence(cases) -> dict:
    amb = [c for c in cases if c.ground_truth == "AMBIGUOUS"]
    rest = [c for c in cases if c.ground_truth != "AMBIGUOUS"]
    has_filler = lambda c: any(w in SAFE_DISTRACTORS for w in c.conversation_sequence)
    return {
        "ambiguous_gold_with_filler_word": f"{sum(map(has_filler, amb))}/{len(amb)}",
        "other_gold_with_filler_word": f"{sum(map(has_filler, rest))}/{len(rest)}",
        "earlier_word_count_ambiguous_gold": dict(Counter(len(c.conversation_sequence) - 1 for c in amb)),
        "earlier_word_count_other_gold": dict(Counter(len(c.conversation_sequence) - 1 for c in rest)),
    }


def main():
    logging.disable(logging.CRITICAL)
    cases = load_cases()
    methods = [
        EngineRef(),
        classifier_lr("classifier_lr_no_artifacts", featurize_no_artifacts),
        classifier_rf("classifier_rf_no_artifacts", featurize_no_artifacts),
    ]
    rows = {}
    for m in methods:
        rows[m.name] = run_method(m, cases)
        print(f"[ABLATION] ran {m.name}")
    metrics = compute_all(rows, reference="engine")

    out = {
        "removed_features": {"sequence_shape": list(SEQUENCE_SHAPE_FEATURES),
                             "filler_words (ctx= and wx=...|W)": list(SAFE_DISTRACTORS)},
        "dataset_artifact_evidence": dataset_artifact_evidence(cases),
        "methods": {name: {k: m[k] for k in KEEP if k in m}
                    for name, m in metrics["methods"].items()},
    }
    for m in methods:
        if hasattr(m, "report"):
            out["methods"][m.name]["folds"] = m.report()["folds"]

    full_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(full_path):   # with-artifact numbers from run_baselines.py, for side-by-side
        with open(full_path, encoding="utf-8") as fh:
            full = json.load(fh)["methods"]
        out["with_artifact_features_reference"] = {
            n: {k: full[n][k] for k in KEEP if k in full[n] and k != "per_category_forced"}
            for n in ("classifier_lr", "classifier_rf") if n in full}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "artifact_ablation.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    for name, m in out["methods"].items():
        g = m.get("gated_decisions", {})
        print(f"{name:28s} forced {m['forced_accuracy']:.3f} {tuple(round(x, 3) for x in m['forced_accuracy_ci95'])}  "
              f"full {m['full_accuracy']:.3f} {tuple(round(x, 3) for x in m['full_accuracy_ci95'])}  "
              f"R/A/L {g.get('resolved_rate', 0):.3f}/{g.get('ambiguous_rate', 0):.3f}/{g.get('literal_rate', 0):.3f}  "
              f"false-conf {g.get('false_confident', {}).get('rate', float('nan')):.3f}")


if __name__ == "__main__":
    main()
