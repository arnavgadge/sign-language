"""
baselines/common.py
-----------------------
Shared case loading, method interface and timing runner for the baseline
comparison. Read-only with respect to the engine and evaluation/.

Leakage guard: methods never see a FinalEvalCase. They receive a
CaseInput that carries ONLY the allowed inputs (recognized_word,
conversation_sequence, sequence_offsets_seconds, personal_seed); the
candidate list comes from ONTOLOGY.lookup(word) inside predict(). The
scoring fields (ground_truth, expected_status, scenario_category,
sample_id, user_id, notes) stay with the runner and are used only to
score the returned prediction.
"""
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import StratifiedKFold

from evaluation.final_dataset import FinalEvalCase, SEED, generate_final_dataset
from intent.ontology import IntentOntology

ONTOLOGY = IntentOntology()   # read-only; same default path final_variants.py uses

NON_INTENT_LABELS = ("AMBIGUOUS", "LITERAL")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

Ranking = List[Tuple[str, float]]
Prediction = Tuple[str, float, Ranking]   # (top_intent_id, confidence_score, full_ranking)


@dataclass(frozen=True)
class CaseInput:
    """Everything a method is allowed to see about one case."""
    recognized_word: str
    conversation_sequence: Tuple[str, ...]        # earlier words, target word LAST
    sequence_offsets_seconds: Tuple[float, ...]
    personal_seed: Tuple[Tuple[str, str, int, float], ...]


def to_input(case: FinalEvalCase) -> CaseInput:
    return CaseInput(
        recognized_word=case.recognized_word,
        conversation_sequence=tuple(case.conversation_sequence),
        sequence_offsets_seconds=tuple(case.sequence_offsets_seconds),
        personal_seed=tuple(tuple(s) for s in case.personal_seed),
    )


def load_cases() -> List[FinalEvalCase]:
    """The same deterministic 500 cases run_final_evaluation.py uses."""
    assert SEED == 20240521, f"Unexpected dataset seed {SEED}"
    return generate_final_dataset()


class Method:
    """
    Interface every compared method implements.

    prepare(inp)  -- optional, UNTIMED per-case setup (e.g. seeding in-memory history)
    predict(inp)  -- TIMED; returns (top_intent_id, confidence_score, full_ranking)
    extras()      -- optional extra columns for the row just predicted
    cleanup()     -- optional, untimed per-case teardown
    abstains      -- True only if the method can output AMBIGUOUS/LITERAL
                     (it must then put that label in extras()["gated_label"])
    trainable     -- True if the method learns from labels. The runner then
                     predicts out-of-fold: fit() gets ONLY training-fold inputs
                     and their ground_truth labels; test-fold cases are predicted
                     by a model that never saw them.
    """
    name = "method"
    abstains = False
    trainable = False

    def fit(self, train_inputs: List[CaseInput], train_labels: List[str]) -> None:
        pass

    def prepare(self, inp: CaseInput) -> None:
        pass

    def predict(self, inp: CaseInput) -> Prediction:
        raise NotImplementedError

    def extras(self) -> Dict:
        return {}

    def cleanup(self) -> None:
        pass


OUTER_FOLDS = 5
OUTER_FOLD_SEED = 20240521


def stratified_folds(labels: List[str], n_splits: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """StratifiedKFold(shuffle=True). Classes with fewer members than n_splits are kept;
    sklearn's 'least populated class' warning for them is expected and silenced here."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The least populated class in y has only")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(list(tr), list(te)) for tr, te in skf.split([[0]] * len(labels), labels)]


def run_method(method: Method, cases: List[FinalEvalCase]) -> List[Dict]:
    inputs = [to_input(c) for c in cases]
    if method.trainable:
        labels = [c.ground_truth for c in cases]   # used only as training labels below
        folds = stratified_folds(labels, OUTER_FOLDS, OUTER_FOLD_SEED)
    else:
        folds = [(None, list(range(len(cases))))]

    rows: List[Optional[Dict]] = [None] * len(cases)
    for fold, (train_idx, test_idx) in enumerate(folds):
        if train_idx is not None:
            method.fit([inputs[i] for i in train_idx], [labels[i] for i in train_idx])
        for i in test_idx:
            rows[i] = _predict_row(method, cases[i], inputs[i],
                                   fold if train_idx is not None else None)
    return rows


def _predict_row(method: Method, case: FinalEvalCase, inp: CaseInput, fold: Optional[int]) -> Dict:
    method.prepare(inp)
    t0 = time.perf_counter()
    top, score, ranking = method.predict(inp)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    extra = method.extras()
    method.cleanup()
    # A method may report its own latency (the LLM reports the recorded API
    # round-trip of the cached response rather than the cache-replay time).
    latency_ms = extra.pop("latency_ms", latency_ms)
    if fold is not None:
        extra["fold"] = fold

    gated_label: Optional[str] = extra.pop("gated_label", None) if method.abstains else top
    return {
        "method": method.name,
        "sample_id": case.sample_id,
        "scenario_category": case.scenario_category,
        "recognized_word": case.recognized_word,
        "ground_truth": case.ground_truth,
        "expected_status": case.expected_status,
        "intent_gold": case.ground_truth not in NON_INTENT_LABELS,
        "predicted_intent": top,
        "score": float(score),
        "correct_forced": top == case.ground_truth,
        "gated_label": gated_label,
        "correct_full": gated_label == case.ground_truth,
        "latency_ms": latency_ms,
        "ranking": json.dumps([[c, round(float(s), 6)] for c, s in ranking]),
        **extra,
    }
