"""
evaluation/final_report.py
------------------------------
PHASE 9 -- builds the final text report from real computed values only.
"""
from typing import List

from .final_variants import FinalVariantResult, VARIANTS
from . import final_metrics as M
from .final_error_analysis import failure_category_counts

_VARIANT_LABEL = {
    "baseline": "Literal baseline",
    "L1_candidates": "Layer 1 candidate generation",
    "L1_temporal": "Layer 1 + Temporal",
    "L1_personal": "Layer 1 + Personal",
    "L1_temporal_personal": "Layer 1 + Temporal + Personal",
    "full_reliability_weighted": "Full engine (reliability-weighted)",
    "full_unweighted": "Full engine WITHOUT reliability weighting",
}


def table_ablation(results: List[FinalVariantResult]) -> str:
    lines = [
        "TABLE -- Ablation (7 configurations, same 500 cases)",
        f"{'Configuration':<42}{'Accuracy':<12}{'Status Acc.':<14}{'Macro F1':<12}{'Mean Lat.(ms)'}",
        "-" * 92,
    ]
    for variant in VARIANTS:
        subset = M.filter_variant(results, variant)
        acc = M.accuracy(subset)
        sacc = M.status_accuracy(subset)
        f1 = M.macro_prf1(subset)["f1"]
        lat = M.latency_stats(subset).get("mean_ms", float("nan"))
        lines.append(f"{_VARIANT_LABEL[variant]:<42}{acc:<12.3f}{sacc:<14.3f}{f1:<12.3f}{lat:.4f}")
    return "\n".join(lines)


def table_decision_outcomes(results: List[FinalVariantResult]) -> str:
    subset = M.filter_variant(results, "full_reliability_weighted")
    breakdown = M.confidence_gate_breakdown(subset)
    lines = ["TABLE -- Confidence Gate Breakdown (full engine, n=%d)" % len(subset),
             f"{'Status':<14}{'n':<8}{'Status Correct':<18}{'Status Incorrect'}", "-" * 55]
    for status, d in breakdown.items():
        lines.append(f"{status:<14}{d['n']:<8}{d['status_correct']:<18}{d['status_incorrect']}")
    fcr = M.false_confident_resolution_rate(subset)
    lines.append("")
    lines.append(f"FALSE CONFIDENT RESOLUTION RATE: {fcr['rate']:.4f}  "
                 f"({fcr.get('n_wrong', '?')}/{fcr['n_resolved']} RESOLVED predictions were wrong)")
    return "\n".join(lines)


def table_class_distribution(cases) -> str:
    from .final_dataset import dataset_composition
    comp = dataset_composition(cases)
    lines = ["TABLE -- Ground-Truth Class Distribution",
             f"n_classes={comp['n_ground_truth_classes']}  "
             f"min_class_count={comp['min_class_count']}  max_class_count={comp['max_class_count']}"]
    return "\n".join(lines)


def full_report(results: List[FinalVariantResult], cases, coverage: dict, top_k: dict,
                 conflict_m: dict, cold_start_rows: list, adversarial_m: dict, ambiguous_m: dict,
                 reliability_exp: dict, latency_exp: dict, ontology_summary: dict,
                 error_rows: list, dataset_comp: dict) -> str:
    lines = [
        "=" * 78, "PHASE 9 -- FINAL EVALUATION REPORT (500-case dataset)", "=" * 78, "",
        "DATASET COMPOSITION", "-" * 78,
        f"Total cases: {dataset_comp['n_total']}",
        f"By category: {dataset_comp['by_category']}",
        f"Distinct words: {dataset_comp['n_distinct_words']}  "
        f"Distinct users: {dataset_comp['n_distinct_users']}  "
        f"Ground-truth classes: {dataset_comp['n_ground_truth_classes']}",
        f"Min class count: {dataset_comp['min_class_count']}  Max class count: {dataset_comp['max_class_count']}",
        "",
        "A. CANDIDATE-GENERATION METRICS (Layer 1 -- NOT reported as accuracy)", "-" * 78,
        f"Candidate coverage: {coverage['coverage']:.4f}  (n={coverage['n']})",
        f"Top-K candidate recall: {top_k}",
        "",
        table_ablation(results), "",
        table_decision_outcomes(results), "",
        "D. CONFLICT DETECTION METRICS", "-" * 78,
        f"Precision={conflict_m['precision']:.4f}  Recall={conflict_m['recall']:.4f}  "
        f"F1={conflict_m['f1']:.4f}  (tp={conflict_m['tp']} fp={conflict_m['fp']} "
        f"fn={conflict_m['fn']} tn={conflict_m['tn']})",
        "",
        "G. ROBUSTNESS", "-" * 78,
        f"Adversarial accuracy: {adversarial_m['accuracy']:.4f}  (n={adversarial_m['n']})",
        f"Ambiguous-case correctly flagged AMBIGUOUS: {ambiguous_m['rate']:.4f}  (n={ambiguous_m['n']})",
        "",
        "EXPERIMENT -- RELIABILITY WEIGHTING vs UNWEIGHTED (controlled, synthetic)", "-" * 78,
        f"Reliability-weighted: ranking flipped when reliability reversed = {reliability_exp['ranking_flipped']}",
        f"Unweighted:           ranking flipped when reliability reversed = "
        f"{reliability_exp['unweighted_comparison']['ranking_flipped']}",
        "",
        "LATENCY (ms)", "-" * 78,
    ]
    for stage, stats in latency_exp.items():
        lines.append(f"{stage:<28} mean={stats['mean_ms']:.4f}  median={stats['median_ms']:.4f}  "
                      f"min={stats['min_ms']:.4f}  max={stats['max_ms']:.4f}  stdev={stats['stdev_ms']:.4f}")
    lines += [
        "",
        "ONTOLOGY OVERLAP / ROBUSTNESS", "-" * 78,
        f"Distinct trigger words: {ontology_summary['n_distinct_trigger_words']}",
        f"High-ambiguity trigger words (5+ associated intents): {ontology_summary['n_high_ambiguity_triggers']}",
        f"  {[r['trigger_word'] for r in ontology_summary['high_ambiguity_triggers']]}",
        f"Sparse intents (<3 triggers): {ontology_summary['n_sparse_intents']}",
        "",
        "ERROR ANALYSIS -- failure category counts (full engine)", "-" * 78,
        str(failure_category_counts(error_rows)),
        f"Total incorrect (full engine, intent-level): {len(error_rows)} / "
        f"{len(M.filter_variant(results, 'full_reliability_weighted'))}",
    ]
    return "\n".join(lines)
