"""
evaluation/reliability_experiment.py
----------------------------------------
EXPERIMENT 4 — Reliability weighting.

A controlled, synthetic experiment (deliberately NOT dataset-driven):
directly constructs EvidenceItem values for two sources (temporal,
personal) and two candidates (medical_assistance, task_assistance),
computed through the REAL reasoning.ReliabilityWeightedStrategy — then
reverses ONLY the reliability values (signals held fixed) and re-runs
the SAME real function, to verify the ranking flips as reliability
weighting predicts it should.

This is intentionally a mechanism-verification experiment using
hand-specified numbers, not a claim about real engine behavior on
real data — that's Experiment 1-3, 6-8's job. Distinguishing the two is
the whole point of the strict-rules section for this phase.
"""
from reasoning.models import EvidenceItem
from reasoning.strategies import ReliabilityWeightedStrategy, UnweightedAverageStrategy
from reasoning.conflict import ConflictDetector


def run() -> dict:
    strategy = ReliabilityWeightedStrategy()
    unweighted_strategy = UnweightedAverageStrategy()   # Phase 9 comparison
    detector = ConflictDetector()

    # Configuration A: temporal is highly reliable, personal is not.
    config_a = {
        "temporal": {"medical_assistance": EvidenceItem(0.90, 0.90),
                     "task_assistance": EvidenceItem(0.10, 0.90)},
        "personal": {"medical_assistance": EvidenceItem(0.15, 0.20),
                     "task_assistance": EvidenceItem(0.85, 0.20)},
    }
    # Configuration B: SAME signals, reliabilities reversed.
    config_b = {
        "temporal": {"medical_assistance": EvidenceItem(0.90, 0.20),
                     "task_assistance": EvidenceItem(0.10, 0.20)},
        "personal": {"medical_assistance": EvidenceItem(0.15, 0.90),
                     "task_assistance": EvidenceItem(0.85, 0.90)},
    }

    def score_config(cfg):
        scores = {}
        for candidate in ("medical_assistance", "task_assistance"):
            per_candidate = {src: cfg[src][candidate] for src in cfg}
            final_score, sources = strategy.score(per_candidate)
            scores[candidate] = {"final_score": final_score,
                                  "sources": {k: v.to_dict() for k, v in sources.items()}}
        conflict_input = {src: {cid: cfg[src][cid] for cid in ("medical_assistance", "task_assistance")}
                           for src in cfg}
        conflict = detector.detect(conflict_input)
        return scores, conflict

    scores_a, conflict_a = score_config(config_a)
    scores_b, conflict_b = score_config(config_b)

    winner_a = max(scores_a, key=lambda c: scores_a[c]["final_score"])
    winner_b = max(scores_b, key=lambda c: scores_b[c]["final_score"])

    # Phase 9 -- same two configs, scored WITHOUT reliability weighting.
    # Signals are identical between config_a and config_b, so an unweighted
    # mean should NOT change winner between them if reliability weighting
    # is the thing doing the work.
    def score_unweighted(cfg):
        scores = {}
        for candidate in ("medical_assistance", "task_assistance"):
            per_candidate = {src: cfg[src][candidate] for src in cfg}
            final_score, _ = unweighted_strategy.score(per_candidate)
            scores[candidate] = final_score
        return scores

    unweighted_a = score_unweighted(config_a)
    unweighted_b = score_unweighted(config_b)
    unweighted_winner_a = max(unweighted_a, key=unweighted_a.get)
    unweighted_winner_b = max(unweighted_b, key=unweighted_b.get)

    return {
        "unweighted_comparison": {
            "config_a_scores": unweighted_a, "config_a_winner": unweighted_winner_a,
            "config_b_scores": unweighted_b, "config_b_winner": unweighted_winner_b,
            "ranking_flipped": unweighted_winner_a != unweighted_winner_b,
            "note": "Signals are IDENTICAL between config_a and config_b -- an unweighted "
                    "mean of signals is mathematically guaranteed to give the same winner "
                    "for both, regardless of reliability. Any difference between the "
                    "reliability-weighted result (which DOES flip) and this result (which "
                    "cannot flip, by construction) is the measured contribution of "
                    "reliability weighting.",
        },
        "config_a": {"description": "temporal reliability=0.90 (high), personal reliability=0.20 (low), "
                                     "signals unchanged from config_b",
                      "scores": scores_a, "winner": winner_a,
                      "conflict_detected": conflict_a.conflict_detected,
                      "conflict_score": conflict_a.conflict_score},
        "config_b": {"description": "SAME signals as config_a, reliabilities reversed: "
                                     "temporal=0.20 (low), personal=0.90 (high)",
                      "scores": scores_b, "winner": winner_b,
                      "conflict_detected": conflict_b.conflict_detected,
                      "conflict_score": conflict_b.conflict_score},
        "ranking_flipped": winner_a != winner_b,
        "score_delta_medical": scores_b["medical_assistance"]["final_score"] - scores_a["medical_assistance"]["final_score"],
        "score_delta_task": scores_b["task_assistance"]["final_score"] - scores_a["task_assistance"]["final_score"],
    }
