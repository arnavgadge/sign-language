"""
baselines/learning_curve.py
-------------------------------
How much labelled data does a classifier need to match the engine?

Same 5 outer stratified folds as run_baselines.py (seed 20240521). In each
outer training fold the training set is subsampled to N cases, 5 subsample
seeds per N; LR and RF are trained with FIXED hyperparameters (the most
frequently tuned values, classifier.FIXED_LR / FIXED_RF) and predict the whole
outer test fold. One (N, seed) therefore yields a prediction for all 500 cases.

Subsampling: if N >= number of classes in the training fold, one random case per
class first, then the remainder allocated proportionally to class size
(largest-remainder rounding); otherwise plain random. N equal to the fold size
(400) is the whole fold, so its 5 seeds are identical.

    python -m baselines.learning_curve      -> baselines/results/learning_curve.csv
"""
import csv
import logging
import os
import random
import statistics
from collections import defaultdict

from .classifier import FIXED_LR, FIXED_RF, fixed_lr, fixed_rf
from .common import (NON_INTENT_LABELS, OUTER_FOLD_SEED, OUTER_FOLDS, RESULTS_DIR,
                     load_cases, run_method, stratified_folds, to_input)
from .engine_ref import EngineRef

SIZES = (25, 50, 100, 200, 400)
N_SEEDS = 5
BASE_SEED = 20240521


def subsample(train_idx, labels, n, rng):
    if n >= len(train_idx):
        return list(train_idx)
    by_class = defaultdict(list)
    for i in train_idx:
        by_class[labels[i]].append(i)
    classes = sorted(by_class)
    if n < len(classes):
        return rng.sample(list(train_idx), n)

    chosen = [rng.choice(by_class[c]) for c in classes]
    left = {c: [i for i in by_class[c] if i not in chosen] for c in classes}
    remaining, pool = n - len(chosen), sum(len(v) for v in left.values())
    quota = {c: remaining * len(left[c]) / pool for c in classes}
    take = {c: int(quota[c]) for c in classes}
    for c in sorted(classes, key=lambda c: quota[c] - take[c], reverse=True)[:remaining - sum(take.values())]:
        take[c] += 1
    for c in classes:
        chosen += rng.sample(left[c], min(take[c], len(left[c])))
    return chosen


def main():
    logging.disable(logging.CRITICAL)
    cases = load_cases()
    labels = [c.ground_truth for c in cases]
    inputs = [to_input(c) for c in cases]
    intent_gold = [l not in NON_INTENT_LABELS for l in labels]
    folds = stratified_folds(labels, OUTER_FOLDS, OUTER_FOLD_SEED)

    engine = run_method(EngineRef(), cases)
    eng_forced = sum(r["correct_forced"] for r in engine if r["intent_gold"]) / sum(intent_gold)
    eng_full = sum(r["correct_full"] for r in engine) / len(engine)

    out_rows = []
    for make, fixed in ((fixed_lr, FIXED_LR), (fixed_rf, FIXED_RF)):
        model_name = make().name
        for n in SIZES:
            forced_accs, full_accs = [], []
            for s in range(N_SEEDS):
                gated, forced = [None] * len(cases), [None] * len(cases)
                for f, (tr, te) in enumerate(folds):
                    rng = random.Random(BASE_SEED * 1_000_000 + f * 100_000 + n * 10 + s)
                    idx = subsample(tr, labels, n, rng)
                    clf = make()
                    clf.fit([inputs[i] for i in idx], [labels[i] for i in idx])
                    for i, (g, top) in zip(te, clf.predict_many([inputs[i] for i in te])):
                        gated[i], forced[i] = g, top
                forced_accs.append(sum(forced[i] == labels[i] for i in range(len(cases)) if intent_gold[i])
                                   / sum(intent_gold))
                full_accs.append(sum(g == l for g, l in zip(gated, labels)) / len(cases))
            row = {
                "model": model_name, "tau": fixed["tau"], "params": fixed["params"], "N": n,
                "n_seeds": N_SEEDS,
                "forced_mean": statistics.mean(forced_accs), "forced_std": statistics.stdev(forced_accs),
                "full_mean": statistics.mean(full_accs), "full_std": statistics.stdev(full_accs),
                "engine_forced": eng_forced, "engine_full": eng_full,
            }
            row["reaches_engine_forced"] = row["forced_mean"] >= eng_forced
            row["reaches_engine_full"] = row["full_mean"] >= eng_full
            out_rows.append(row)
            print(f"[LC] {model_name:20s} N={n:3d}  forced {row['forced_mean']:.3f}±{row['forced_std']:.3f}  "
                  f"full {row['full_mean']:.3f}±{row['full_std']:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "learning_curve.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"[LC] engine reference: forced {eng_forced:.3f}, full {eng_full:.3f}")
    for model in dict.fromkeys(r["model"] for r in out_rows):
        rs = [r for r in out_rows if r["model"] == model]
        for key in ("forced", "full"):
            hit = next((r["N"] for r in rs if r[f"reaches_engine_{key}"]), None)
            print(f"[LC] {model}: smallest N reaching engine {key} accuracy (mean over seeds): "
                  f"{hit if hit is not None else 'never (up to N=400)'}")


if __name__ == "__main__":
    main()
