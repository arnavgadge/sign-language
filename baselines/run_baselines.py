"""
baselines/run_baselines.py
------------------------------
Runs the engine reference and every baseline on the 500 final-evaluation
cases and writes baselines/results/.

    python -m baselines.run_baselines                 (engine, priors, classifiers, Gemini)
    python -m baselines.run_baselines --llm local     (... + local Qwen instead of Gemini)
    --llm {none,cloud,local,all}, default cloud. LLM runs resume from their cache;
    an incomplete LLM run is reported as progress only, never scored.
    If another run is active, --llm local collects Qwen into baselines/results/tmp/,
    waits for the other run to finish, then merges and scores (one command).

To add a method, append one instance to METHODS. Methods with trainable=True
are predicted out-of-fold automatically (see baselines/common.run_method).
"""
import argparse
import csv
import json
import logging
import os
import sys
import time

from evaluation.final_variants import run_all_variants

from .base_prior import BasePrior
from .classifier import classifier_lr, classifier_rf
from .common import load_cases, run_method, to_input
from .engine_ref import EngineRef
from .llm import TMP_DIR, build_llm_methods, merge_temp_caches, other_run_active
from .metrics import compute_all
from .user_majority import UserMajority

METHODS = [
    EngineRef(),      # must stay first: the reference for McNemar and the sanity check
    BasePrior(),
    UserMajority(),
    classifier_lr(),
    classifier_rf(),
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EXPECTED_GATED_CORRECT = 419   # 0.838 * 500, evaluation/results/final_ablation_results.csv
EXPECTED_TOP1_CORRECT = 369    # 0.976 * 378, final_metrics.json top_k_candidate_recall.top_1


def sanity_check(cases, engine_rows) -> None:
    gated = sum(r["correct_full"] for r in engine_rows)
    top1 = sum(r["correct_forced"] for r in engine_rows if r["intent_gold"])
    n_intent = sum(r["intent_gold"] for r in engine_rows)
    print(f"[SANITY] engine gated accuracy {gated}/{len(engine_rows)} = {gated / len(engine_rows):.3f} "
          f"(expected {EXPECTED_GATED_CORRECT})")
    print(f"[SANITY] engine top-1 on intent-gold {top1}/{n_intent} = {top1 / n_intent:.3f} "
          f"(expected {EXPECTED_TOP1_CORRECT})")

    # Per-case cross-check against the existing evaluation's own code path.
    mismatches = []
    for case, row in zip(cases, engine_rows):
        ref = next(r for r in run_all_variants(case) if r.variant == "full_reliability_weighted")
        if ref.predicted_intent != row["gated_label"] or \
                (row["intent_gold"] and (ref.ground_truth_rank == 1) != row["correct_forced"]):
            mismatches.append(case.sample_id)
    print(f"[SANITY] per-case mismatches vs evaluation/final_variants.py: {len(mismatches)}")

    if gated != EXPECTED_GATED_CORRECT or top1 != EXPECTED_TOP1_CORRECT or mismatches:
        sys.exit(f"[SANITY] FAILED -- not writing results. Mismatched cases: {mismatches[:20]}")


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _summary_rows(metrics):
    rows = []
    for name, m in metrics["methods"].items():
        mc = m.get("mcnemar_vs_engine") or {}
        gd = m.get("gated_decisions", {})
        rows.append({
            "method": name,
            "forced_acc": m["forced_accuracy"],
            "forced_ci_lo": m["forced_accuracy_ci95"][0], "forced_ci_hi": m["forced_accuracy_ci95"][1],
            "forced_macro_f1": m["forced_macro_f1"],
            "full_acc": m["full_accuracy"],
            "full_ci_lo": m["full_accuracy_ci95"][0], "full_ci_hi": m["full_accuracy_ci95"][1],
            "full_ceiling": "" if m["full_ceiling"] is None else m["full_ceiling"],
            "aurc": m["risk_coverage"]["aurc"],
            "mcnemar_b_engine_only": mc.get("b", ""), "mcnemar_c_method_only": mc.get("c", ""),
            "mcnemar_p": mc.get("p", ""),
            "latency_mean_ms": m["latency_ms"]["mean"], "latency_median_ms": m["latency_ms"]["median"],
            "resolved_rate": gd.get("resolved_rate", ""), "ambiguous_rate": gd.get("ambiguous_rate", ""),
            "literal_rate": gd.get("literal_rate", ""),
            "false_confident_rate": gd.get("false_confident", {}).get("rate", ""),
            "n_used_fallback": m.get("n_used_fallback", ""),
            "fit_seconds_total": sum(f["fit_seconds_incl_tuning"] for f in m["training"]["folds"])
                                 if "training" in m else "",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=("none", "cloud", "local", "all"), default="cloud")
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)
    cases = load_cases()
    print(f"[BASELINES] {len(cases)} cases")

    busy = other_run_active()
    if busy:
        # Another run owns baselines/results/. Local Qwen is collected in parallel into
        # baselines/results/tmp/; this command then waits for that run, merges and scores.
        if args.llm != "local":
            sys.exit(f"[BASELINES] another baselines run is active (pid {busy}); wait for it to finish, "
                     f"or use --llm local to collect Qwen responses into temporary storage.")
        local = [m for m in build_llm_methods("local", write_to_temp=True) if m.name == "llm_local"]
        for m in local:
            m.run_all([to_input(c) for c in cases])
            rep = m.run_report([to_input(c) for c in cases])
            print(f"[BASELINES] collected {rep['n_cases_done']}/{len(cases)} Qwen cases into {TMP_DIR} "
                  f"(invalid {rep['n_invalid']}). Nothing in baselines/results/ was overwritten.")
        # One command: wait for the other run to finish, then merge and score below.
        while other_run_active():
            print(f"[BASELINES] waiting for the other run (pid {other_run_active()}) to finish "
                  f"before merging and scoring...", flush=True)
            time.sleep(60)
        print("[BASELINES] other run finished; merging temporary storage and scoring.")

    merged = merge_temp_caches()
    if merged:
        print(f"[BASELINES] merged temporary LLM caches into baselines/results/: {merged}")

    # LLMs: resolve every case first (cache, then API); only complete runs are scored.
    inputs = [to_input(c) for c in cases]
    llm_runs, scored_llms = {}, []
    for m in build_llm_methods(args.llm) if args.llm != "none" else []:
        complete = m.run_all(inputs)
        llm_runs[m.name] = m.run_report(inputs)
        llm_runs[m.name]["latency_note"] = ("per-case wall time of the API call(s), INCLUDING network time"
                                            if m.name == "llm_cloud" else "per-case local inference wall time")
        if complete:
            scored_llms.append(m)
        else:
            print(f"[BASELINES] {m.name} incomplete ({llm_runs[m.name]['n_cases_done']}/{len(cases)} cases): "
                  f"not scored. Re-run the same command to resume.")

    rows_by_method = {}
    for method in METHODS + scored_llms:
        rows_by_method[method.name] = run_method(method, cases)
        print(f"[BASELINES] ran {method.name}")

    sanity_check(cases, rows_by_method["engine"])
    metrics = compute_all(rows_by_method, reference="engine")
    for method in METHODS:
        if hasattr(method, "report"):
            metrics["methods"][method.name]["training"] = method.report()
    metrics["llm_runs"] = llm_runs

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = [r for rows in rows_by_method.values() for r in rows]
    fields = list(dict.fromkeys(k for r in all_rows for k in r))
    _write_csv(os.path.join(RESULTS_DIR, "predictions.csv"), all_rows, fields)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    summary = _summary_rows(metrics)
    _write_csv(os.path.join(RESULTS_DIR, "summary.csv"), summary, list(summary[0].keys()))

    rc = [{"method": name, **pt} for name, m in metrics["methods"].items()
          for pt in m["risk_coverage"]["grid"]]
    _write_csv(os.path.join(RESULTS_DIR, "risk_coverage.csv"), rc, list(rc[0].keys()))

    for s in summary:
        print(f"{s['method']:14s} forced {s['forced_acc']:.3f} [{s['forced_ci_lo']:.3f},{s['forced_ci_hi']:.3f}]  "
              f"full {s['full_acc']:.3f} [{s['full_ci_lo']:.3f},{s['full_ci_hi']:.3f}]  "
              f"AURC {s['aurc']:.3f}  p={s['mcnemar_p']}  {s['latency_mean_ms']:.3f} ms")
    print(f"[BASELINES] wrote {RESULTS_DIR}")


if __name__ == "__main__":
    main()
