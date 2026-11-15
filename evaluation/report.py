"""
evaluation/report.py
------------------------
Generates the paper-ready tables and error analysis directly from stored
VariantResult rows and experiment outputs — no numbers are typed in here
by hand.
"""
from typing import List

from .variants import VariantResult, VARIANTS
from . import metrics as M

_VARIANT_LABEL = {
    "baseline": "Literal baseline",
    "L1_only": "Layer 1 only",
    "L1_temporal": "Layer 1 + Temporal",
    "L1_personal": "Layer 1 + Personal",
    "full": "Full implemented engine",
}


def table1_baseline_vs_proposed(results: List[VariantResult]) -> str:
    lines = [
        "TABLE 1 -- Baseline vs Proposed",
        f"{'Method':<28}{'Accuracy':<12}{'Macro F1':<12}{'Mean Latency (ms)':<20}",
        "-" * 72,
    ]
    for variant in VARIANTS:
        subset = M.filter_variant(results, variant)
        acc = M.accuracy(subset)
        f1 = M.macro_prf1(subset)["f1"]
        lat = M.latency_stats(subset).get("mean_ms", float("nan"))
        lines.append(f"{_VARIANT_LABEL[variant]:<28}{acc:<12.3f}{f1:<12.3f}{lat:<20.4f}")
    return "\n".join(lines)


def table2_ablation(results: List[VariantResult]) -> str:
    config = {
        "baseline":     (False, False, False),
        "L1_only":      (False, False, True),
        "L1_temporal":  (True, False, True),
        "L1_personal":  (False, True, True),
        "full":         (True, True, True),
    }
    lines = [
        "TABLE 2 -- Ablation",
        f"{'Configuration':<20}{'Temporal':<10}{'Personal':<10}{'ReliabWeight':<14}{'Accuracy':<10}{'Macro F1'}",
        "-" * 78,
    ]
    for variant in VARIANTS:
        t, p, rw = config[variant]
        subset = M.filter_variant(results, variant)
        acc = M.accuracy(subset)
        f1 = M.macro_prf1(subset)["f1"]
        lines.append(f"{_VARIANT_LABEL[variant]:<20}{str(t):<10}{str(p):<10}{str(rw):<14}{acc:<10.3f}{f1:.3f}")
    return "\n".join(lines)


def table3_user_history(cold_start_rows: list) -> str:
    lines = [
        "TABLE 3 -- User History (Cold Start, Experiment 5)",
        f"{'Observations':<14}{'Signal':<10}{'Reliability':<14}{'Profile Maturity':<18}{'Obs. Sufficiency'}",
        "-" * 66,
    ]
    for row in cold_start_rows:
        lines.append(f"{row['observation_count']:<14}{row['signal']:<10.4f}{row['reliability']:<14.4f}"
                      f"{row['profile_maturity']:<18.4f}{row['observation_sufficiency']:.4f}")
    return "\n".join(lines)


def table4_decision_outcomes(results: List[VariantResult]) -> str:
    subset = M.filter_variant(results, "full")
    n = len(subset)
    resolved = sum(1 for r in subset if r.status == "RESOLVED")
    ambiguous = sum(1 for r in subset if r.status == "AMBIGUOUS")
    literal = sum(1 for r in subset if r.status == "LITERAL")
    incorrect_confident = sum(1 for r in subset if r.status == "RESOLVED" and not r.correct)
    lines = [
        "TABLE 4 -- Decision Outcomes (full engine)",
        f"{'Outcome':<24}{'Count':<10}{'% of samples'}",
        "-" * 46,
        f"{'RESOLVED':<24}{resolved:<10}{resolved / n * 100:.1f}%" if n else "n/a",
        f"{'AMBIGUOUS':<24}{ambiguous:<10}{ambiguous / n * 100:.1f}%" if n else "n/a",
        f"{'LITERAL':<24}{literal:<10}{literal / n * 100:.1f}%" if n else "n/a",
        f"{'Incorrect confident':<24}{incorrect_confident:<10}{incorrect_confident / n * 100:.1f}%" if n else "n/a",
    ]
    return "\n".join(lines)


def error_analysis(results: List[VariantResult]) -> str:
    subset = M.filter_variant(results, "full")
    failures = [r for r in subset if not r.correct]
    lines = [f"ERROR ANALYSIS -- full engine: {len(failures)}/{len(subset)} samples incorrect", ""]
    for r in failures:
        lines.append(f"[{r.sample_id}] category={r.category}")
        lines.append(f"    Sequence: {r.conversation_sequence}")
        lines.append(f"    User: {r.user_id}")
        lines.append(f"    Ground truth: {r.ground_truth}")
        lines.append(f"    Predicted: {r.predicted_intent}  (status={r.status}, conflict={r.conflict})")
        lines.append(f"    Top score: {r.top_score:.4f}  Second score: {r.second_score:.4f}  Margin: {r.margin:.4f}")
        lines.append(f"    Temporal evidence: signal={r.temporal_signal:.4f} reliability={r.temporal_reliability:.4f}")
        lines.append(f"    Personal evidence: signal={r.personal_signal:.4f} reliability={r.personal_reliability:.4f}")
        lines.append("")
    if not failures:
        lines.append("(no incorrect predictions in the full-engine variant on this dataset)")
    return "\n".join(lines)


def full_report(results: List[VariantResult], reliability_exp: dict, cold_start_rows: list,
                 latency_exp: dict, dataset_summary: dict) -> str:
    sections = [
        "=" * 70,
        "PHASE 8 EVALUATION REPORT",
        "=" * 70,
        "",
        "DATASET SUMMARY",
        "-" * 70,
        f"Samples: {dataset_summary['n_samples']}",
        f"Distinct words: {dataset_summary['n_distinct_words']}",
        f"Distinct users: {dataset_summary['n_distinct_users']}",
        f"Distinct ground-truth classes: {dataset_summary['n_distinct_ground_truth_classes']}",
        f"Categories: {dataset_summary['categories']}",
        "",
        table1_baseline_vs_proposed(results),
        "",
        table2_ablation(results),
        "",
        table3_user_history(cold_start_rows),
        "",
        table4_decision_outcomes(results),
        "",
        "EXPERIMENT 4 -- RELIABILITY WEIGHTING (controlled, synthetic)",
        "-" * 70,
        f"Config A winner: {reliability_exp['config_a']['winner']} "
        f"(medical={reliability_exp['config_a']['scores']['medical_assistance']['final_score']:.4f}, "
        f"task={reliability_exp['config_a']['scores']['task_assistance']['final_score']:.4f})",
        f"Config B winner: {reliability_exp['config_b']['winner']} "
        f"(medical={reliability_exp['config_b']['scores']['medical_assistance']['final_score']:.4f}, "
        f"task={reliability_exp['config_b']['scores']['task_assistance']['final_score']:.4f})",
        f"Ranking flipped when reliability reversed (signals unchanged): {reliability_exp['ranking_flipped']}",
        f"Conflict detected in config A: {reliability_exp['config_a']['conflict_detected']} "
        f"(score={reliability_exp['config_a']['conflict_score']:.4f})",
        f"Conflict detected in config B: {reliability_exp['config_b']['conflict_detected']} "
        f"(score={reliability_exp['config_b']['conflict_score']:.4f})",
        "",
        "EXPERIMENT 9 -- LATENCY (ms, N=300 per stage unless noted)",
        "-" * 70,
    ]
    for stage, stats in latency_exp.items():
        sections.append(f"{stage:<28} mean={stats['mean_ms']:.4f}  median={stats['median_ms']:.4f}  "
                         f"min={stats['min_ms']:.4f}  max={stats['max_ms']:.4f}  stdev={stats['stdev_ms']:.4f}")
    sections += ["", error_analysis(results)]
    return "\n".join(sections)
