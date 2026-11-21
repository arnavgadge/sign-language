"""
baselines/leave_category_out.py
-----------------------------------
Generalization to an unseen scenario type: for each scenario_category, the
tuned LR and RF (same inner-CV protocol as run_baselines.py) are trained on all
OTHER categories and tested on that category only. scenario_category decides
the split and nothing else; it is never a feature.

Engine numbers need no training: they are the engine's per-case predictions,
recomputed by the same deterministic EngineRef used in run_baselines.py
(identical to baselines/results/predictions.csv).

genuinely_ambiguous has no intent-gold cases, so only full accuracy
(did it answer AMBIGUOUS) is reported for it.

    python -m baselines.leave_category_out   -> baselines/results/leave_category_out.csv
"""
import csv
import logging
import os

from .classifier import classifier_lr, classifier_rf
from .common import NON_INTENT_LABELS, RESULTS_DIR, load_cases, run_method, to_input
from .engine_ref import EngineRef


def _acc(flags):
    return sum(flags) / len(flags) if flags else ""


def main():
    logging.disable(logging.CRITICAL)
    cases = load_cases()
    engine = run_method(EngineRef(), cases)
    categories = list(dict.fromkeys(c.scenario_category for c in cases))

    out = []
    for cat in categories:
        test = [i for i, c in enumerate(cases) if c.scenario_category == cat]
        train = [i for i, c in enumerate(cases) if c.scenario_category != cat]
        gold = [cases[i].ground_truth for i in test]
        is_intent = [g not in NON_INTENT_LABELS for g in gold]
        train_labels = {cases[i].ground_truth for i in train}
        base = {"held_out_category": cat, "n_test": len(test), "n_test_intent_gold": sum(is_intent),
                "n_train": len(train),
                "n_test_gold_label_unseen_in_train": sum(g not in train_labels for g in gold)}

        out.append({**base, "method": "engine",
                    "forced_acc": _acc([engine[i]["correct_forced"] for i, ok in zip(test, is_intent) if ok]),
                    "full_acc": _acc([engine[i]["correct_full"] for i in test])})

        for make in (classifier_lr, classifier_rf):
            clf = make()
            clf.fit([to_input(cases[i]) for i in train], [cases[i].ground_truth for i in train])
            preds = clf.predict_many([to_input(cases[i]) for i in test])
            out.append({**base, "method": clf.name,
                        "forced_acc": _acc([top == g for (_, top), g, ok in zip(preds, gold, is_intent) if ok]),
                        "full_acc": _acc([gated == g for (gated, _), g in zip(preds, gold)]),
                        "tau": clf.tau, "params": clf.fold_log[-1]["params"]})
        print(f"[LOCO] {cat:28s} " + "  ".join(
            f"{r['method']}: forced {r['forced_acc'] if r['forced_acc'] == '' else round(r['forced_acc'], 3)}"
            f" full {round(r['full_acc'], 3)}" for r in out[-3:]))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fields = list(dict.fromkeys(k for r in out for k in r))
    with open(os.path.join(RESULTS_DIR, "leave_category_out.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
