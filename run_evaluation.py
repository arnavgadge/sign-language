"""
run_evaluation.py
--------------------
PHASE 8 -- single reproducible entry point for the full experimental
evaluation. No manual editing of source files is needed between runs.

Usage
-----
    python run_evaluation.py

What it does
------------
1. Loads the ground-truth evaluation dataset (evaluation/dataset.py).
2. Runs the literal baseline + 4 ablation variants (L1 only, L1+Temporal,
   L1+Personal, full) against every sample, using the REAL engine
   components built in Phases 1-6.
3. Saves every (sample, variant) result to SQLite + CSV
   (evaluation/results/).
4. Runs Experiment 4 (reliability weighting, controlled/synthetic),
   Experiment 5 (cold start sweep), and Experiment 9 (latency).
5. Computes all metrics and prints/saves the full paper-ready report
   (evaluation/results/report_<timestamp>.txt).

Nothing here fabricates or adjusts a result -- every number printed
comes directly from a real call into intent.ontology, layer3_temporal,
layer5_personal, reasoning, or decision_gate.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.dataset import load_dataset, dataset_summary
from evaluation.variants import run_all_variants
from evaluation.results_store import save_results
from evaluation import metrics as M
from evaluation import reliability_experiment, cold_start_experiment, latency_experiment
from evaluation.report import full_report


def main():
    print("[EVAL] Loading dataset...")
    dataset = load_dataset()
    summary = dataset_summary()
    print(f"[EVAL] {summary['n_samples']} samples, {summary['n_distinct_words']} distinct words, "
          f"{summary['n_distinct_users']} users, {summary['n_distinct_ground_truth_classes']} ground-truth classes")

    print("[EVAL] Running baseline + ablation variants for every sample...")
    all_results = []
    for sample in dataset:
        all_results.extend(run_all_variants(sample))
    print(f"[EVAL] Collected {len(all_results)} (sample, variant) result rows")

    run_label = str(int(time.time()))
    paths = save_results(all_results, run_label=run_label)
    print(f"[EVAL] Saved results -> {paths['db_path']}")
    print(f"[EVAL]                -> {paths['csv_path']}")

    print("[EVAL] Running Experiment 4 (reliability weighting, controlled)...")
    reliability_result = reliability_experiment.run()

    print("[EVAL] Running Experiment 5 (cold start sweep)...")
    cold_start_rows = cold_start_experiment.run()
    monotonic = cold_start_experiment.check_monotonic_reliability(cold_start_rows)
    print(f"[EVAL] Reliability monotonically non-decreasing with more observations: {monotonic}")

    print("[EVAL] Running Experiment 9 (latency, N=300 per stage)...")
    latency_result = latency_experiment.run()

    print("[EVAL] Generating report...")
    report_text = full_report(all_results, reliability_result, cold_start_rows, latency_result, summary)

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "evaluation", "results", f"report_{run_label}.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    print()
    print(report_text)
    print()
    print(f"[EVAL] Full report saved -> {report_path}")


if __name__ == "__main__":
    main()
