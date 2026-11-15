"""
run_final_evaluation.py
---------------------------
PHASE 9 -- THE single reproducible command for the final evaluation.

Usage
-----
    python run_final_evaluation.py

No manual editing is required between runs. This script:
  1. loads/generates the final ~500-case dataset (deterministic, seeded)
  2. verifies the data-leakage guard
  3. runs baseline + 6 engine configurations against every case
  4. saves raw predictions + the dataset itself
  5. computes every requested metric category (A-G)
  6. runs the reliability-vs-unweighted controlled experiment
  7. runs the ontology overlap/ambiguity analysis
  8. runs the latency experiment
  9. builds error analysis with failure-category counts
  10. writes all final_*.csv / .json / .txt files
  11. prints the final evaluation summary

Nothing is fabricated, adjusted, or cherry-picked -- every printed
number is read directly from a real call into the frozen engine.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.final_dataset import generate_final_dataset, dataset_composition
from evaluation.final_variants import run_all_variants, assert_no_leakage
from evaluation import final_metrics as M
from evaluation.final_error_analysis import build_error_analysis
from evaluation.final_storage import (
    save_dataset, save_predictions, save_metrics, save_error_analysis,
    save_ablation_results, save_latency_results, save_text_report, RESULTS_DIR,
)
from evaluation.final_report import full_report, _VARIANT_LABEL
from evaluation import reliability_experiment
from evaluation import ontology_overlap
from evaluation.latency_experiment import run as run_latency_experiment
from evaluation.scenario_spec import verify_all_representable


def main():
    print("[FINAL EVAL] Verifying scenario templates are representable in the real ontology...")
    rep = verify_all_representable()
    assert rep["all_representable"], f"Non-representable templates found: {rep}"
    print(f"[FINAL EVAL] All {len(rep['details'])} scenario templates verified representable.")

    print("[FINAL EVAL] Verifying no data-leakage path exists in the engine interfaces used...")
    assert_no_leakage()

    print("[FINAL EVAL] Generating the final evaluation dataset (deterministic, seeded)...")
    cases = generate_final_dataset()
    cases_by_id = {c.sample_id: c for c in cases}
    comp = dataset_composition(cases)
    print(f"[FINAL EVAL] {comp['n_total']} cases, {comp['n_distinct_words']} words, "
          f"{comp['n_distinct_users']} users, {comp['n_ground_truth_classes']} ground-truth classes")

    print("[FINAL EVAL] Running baseline + 6 engine configurations for every case "
          "(this is the main computation)...")
    t0 = time.time()
    all_results = []
    for case in cases:
        all_results.extend(run_all_variants(case))
    print(f"[FINAL EVAL] Collected {len(all_results)} (case, variant) rows in {time.time() - t0:.2f}s")

    run_label = str(int(time.time()))

    print("[FINAL EVAL] Saving final data files...")
    save_dataset(cases, os.path.join(RESULTS_DIR, "final_evaluation_dataset.csv"))
    save_predictions(all_results, os.path.join(RESULTS_DIR, "final_predictions.csv"))

    print("[FINAL EVAL] Computing metrics (A-G)...")
    coverage = M.candidate_coverage(all_results)
    top_k = M.top_k_candidate_recall(all_results)
    conflict_m = M.conflict_metrics(all_results)
    adversarial_m = M.adversarial_accuracy([r for r in all_results if r.variant == "full_reliability_weighted"])
    ambiguous_m = M.ambiguous_case_handling([r for r in all_results if r.variant == "full_reliability_weighted"])
    cold_start_full = [r for r in all_results if r.variant == "full_reliability_weighted"
                        and r.scenario_category == "cold_start"]
    cold_start_rows = M.personalization_by_history_size(cold_start_full, cases_by_id)
    full_summary = M.full_summary(all_results)

    print("[FINAL EVAL] Running reliability-weighted vs unweighted controlled experiment...")
    reliability_exp = reliability_experiment.run()

    print("[FINAL EVAL] Running ontology overlap / ambiguity analysis (read-only)...")
    ontology_summary = ontology_overlap.summary()

    print("[FINAL EVAL] Running latency experiment (N=300/stage)...")
    latency_exp = run_latency_experiment()

    print("[FINAL EVAL] Building error analysis...")
    error_rows = build_error_analysis(all_results)

    metrics_dict = {
        "dataset_composition": comp,
        "candidate_coverage": coverage,
        "top_k_candidate_recall": top_k,
        "conflict_metrics": conflict_m,
        "adversarial_accuracy": adversarial_m,
        "ambiguous_case_handling": ambiguous_m,
        "cold_start_by_history_size": cold_start_rows,
        "per_variant_summary": full_summary,
        "reliability_vs_unweighted_experiment": reliability_exp,
        "latency_by_stage": latency_exp,
        "ontology_overlap_summary": {k: v for k, v in ontology_summary.items()},
    }
    save_metrics(metrics_dict, os.path.join(RESULTS_DIR, "final_metrics.json"))
    save_error_analysis(error_rows, os.path.join(RESULTS_DIR, "final_error_analysis.csv"))

    ablation_rows = [
        {"configuration": _VARIANT_LABEL[v], "variant_key": v,
         "n": full_summary[v]["n"], "accuracy": full_summary[v]["accuracy"],
         "status_accuracy": full_summary[v]["status_accuracy"],
         "macro_precision": full_summary[v]["macro_precision"],
         "macro_recall": full_summary[v]["macro_recall"], "macro_f1": full_summary[v]["macro_f1"]}
        for v in full_summary
    ]
    save_ablation_results(ablation_rows, os.path.join(RESULTS_DIR, "final_ablation_results.csv"))

    latency_rows = [{"stage": stage, **stats} for stage, stats in latency_exp.items()]
    save_latency_results(latency_rows, os.path.join(RESULTS_DIR, "final_latency_results.csv"))

    print("[FINAL EVAL] Generating final report...")
    report_text = full_report(all_results, cases, coverage, top_k, conflict_m, cold_start_rows,
                               adversarial_m, ambiguous_m, reliability_exp, latency_exp,
                               ontology_summary, error_rows, comp)
    save_text_report(report_text, os.path.join(RESULTS_DIR, "final_evaluation_report.txt"))

    print()
    print(report_text)
    print()
    print(f"[FINAL EVAL] All files saved under {RESULTS_DIR}")
    print(f"[FINAL EVAL] Run label: {run_label}")


if __name__ == "__main__":
    main()
