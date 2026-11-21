"""
baselines/classifier.py
---------------------------
Trained-classifier baseline (Logistic Regression or Random Forest) over
baselines/classifier_features.py, evaluated out-of-fold by the runner in
baselines/common.py (fit() only ever sees training-fold inputs and labels).

Inside each fit(): hyperparameters (tau x model grid) are chosen by a 3-fold
stratified CV on the training fold only, selecting by pooled macro-F1 of the
gated (masked, abstaining) answer. The final model is refit on the whole
training fold with the chosen setting.

Prediction: probabilities over all 33 training classes are masked to the
word's candidate intents + AMBIGUOUS + LITERAL and renormalized.
  gated answer   = argmax over the masked classes (may abstain)
  forced answer  = argmax over candidate intents only
  confidence     = renormalized probability of that forced answer
"""
import time
import warnings
from collections import defaultdict
from typing import Callable, Dict, List, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .classifier_features import featurize
from .common import NON_INTENT_LABELS, ONTOLOGY, CaseInput, Method, stratified_folds

TAUS = (30, 60, 120, 300, 600)
INNER_FOLDS = 3
INNER_SEED = 20240521
RF_SEED = 20240521


def masked_answer(classes: Sequence[str], proba: Sequence[float], candidate_ids: List[str]):
    """-> (gated_label, forced_top, forced_score, ranking over candidate intents)."""
    allowed = list(candidate_ids) + list(NON_INTENT_LABELS)
    p = dict.fromkeys(allowed, 0.0)
    for c, pr in zip(classes, proba):
        if c in p:
            p[c] = float(pr)
    total = sum(p.values())
    p = {c: (v / total if total > 0 else 1.0 / len(allowed)) for c, v in p.items()}

    gated = max(allowed, key=lambda c: p[c])          # ties -> earliest in allowed order
    ranking = sorted(((c, p[c]) for c in candidate_ids),
                     key=lambda cs: -cs[1])            # stable -> ontology order on ties
    return gated, ranking[0][0], ranking[0][1], ranking


def _status(label: str) -> str:
    return label if label in NON_INTENT_LABELS else "RESOLVED"


class Classifier(Method):
    abstains = True
    trainable = True

    def __init__(self, name: str, make_model: Callable[..., object], grid: List[Dict],
                 taus: Sequence[float] = TAUS, featurizer: Callable = featurize):
        """A single tau and a single grid point means fixed hyperparameters (no inner CV)."""
        self.name = name
        self.make_model = make_model
        self.grid = grid
        self.taus = tuple(taus)
        self.featurize = featurizer
        self.fold_log: List[Dict] = []
        self.fold_models: List[tuple] = []
        self._extras: Dict = {}

    # ── training ──────────────────────────────────────────────────────────

    def _fit_one(self, feats: List[Dict], labels: List[str], params: Dict, n_jobs: int):
        vec = DictVectorizer(sparse=False)
        X = vec.fit_transform(feats)
        model = self.make_model(n_jobs=n_jobs, **params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, labels)
        n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        return vec, model, n_conv

    def _inner_score(self, inputs, feats, labels, params) -> float:
        preds = [None] * len(labels)
        for tr, va in stratified_folds(labels, INNER_FOLDS, INNER_SEED):
            vec, model, _ = self._fit_one([feats[i] for i in tr], [labels[i] for i in tr], params, n_jobs=-1)
            proba = model.predict_proba(vec.transform([feats[i] for i in va]))
            for i, row in zip(va, proba):
                cands = [c.id for c in ONTOLOGY.lookup(inputs[i].recognized_word)]
                preds[i] = masked_answer(model.classes_, row, cands)[0]
        return f1_score(labels, preds, average="macro", zero_division=0)

    def fit(self, train_inputs: List[CaseInput], train_labels: List[str]) -> None:
        t0 = time.perf_counter()
        if len(self.taus) == 1 and len(self.grid) == 1:
            best = (None, self.taus[0], self.grid[0])
        else:
            best = None
            for tau in self.taus:
                feats = [self.featurize(inp, tau) for inp in train_inputs]
                for params in self.grid:
                    score = self._inner_score(train_inputs, feats, train_labels, params)
                    if best is None or score > best[0]:       # ties keep the earlier grid point
                        best = (score, tau, params)
        score, tau, params = best
        vec, model, n_conv = self._fit_one([self.featurize(i, tau) for i in train_inputs],
                                           train_labels, params, n_jobs=-1)
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1                              # single-case prediction latency
        self.vec, self.model, self.tau = vec, model, tau
        self.fold_models.append((vec, model))
        self.fold_log.append({
            "fold": len(self.fold_log), "n_train": len(train_labels),
            "n_train_classes": len(set(train_labels)), "tau": tau, "params": params,
            "inner_macro_f1": score, "final_fit_convergence_warnings": n_conv,
            "fit_seconds_incl_tuning": time.perf_counter() - t0,
        })

    # ── prediction ────────────────────────────────────────────────────────

    def predict(self, inp: CaseInput):
        cands = [c.id for c in ONTOLOGY.lookup(inp.recognized_word)]
        proba = self.model.predict_proba(self.vec.transform([self.featurize(inp, self.tau)]))[0]
        gated, top, score, ranking = masked_answer(self.model.classes_, proba, cands)
        self._extras = {"gated_label": gated, "gated_status": _status(gated)}
        return top, score, ranking

    def predict_many(self, inputs: List[CaseInput]) -> List[tuple]:
        """Batch version of predict() for experiments where latency is not measured.
        -> [(gated_label, forced_top), ...] using the same masking."""
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = -1
        proba = self.model.predict_proba(self.vec.transform([self.featurize(i, self.tau) for i in inputs]))
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        out = []
        for inp, row in zip(inputs, proba):
            cands = [c.id for c in ONTOLOGY.lookup(inp.recognized_word)]
            gated, top, _, _ = masked_answer(self.model.classes_, row, cands)
            out.append((gated, top))
        return out

    def extras(self):
        return dict(self._extras)

    # ── reporting ─────────────────────────────────────────────────────────

    def inspection(self, example_classes=("AMBIGUOUS", "LITERAL", "halt_action"), top_k=15) -> Dict:
        """Importances / coefficients averaged over the outer-fold models
        (a feature absent from a fold's vocabulary counts as 0 there)."""
        n = len(self.fold_models)
        out = {}
        if all(hasattr(m, "feature_importances_") for _, m in self.fold_models):
            imp = defaultdict(float)
            for vec, m in self.fold_models:
                for f, v in zip(vec.feature_names_, m.feature_importances_):
                    imp[f] += v / n
            out["rf_top_importances"] = sorted(imp.items(), key=lambda kv: -kv[1])[:top_k]
        else:
            per_class = {}
            for cls in example_classes:
                coef = defaultdict(float)
                for vec, m in self.fold_models:
                    lr = m[-1]
                    if cls in lr.classes_:
                        row = lr.coef_[list(lr.classes_).index(cls)]
                        for f, v in zip(vec.feature_names_, row):
                            coef[f] += v / n
                per_class[cls] = sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:10]
            out["lr_top_coefficients_standardized"] = per_class
        return out

    def report(self) -> Dict:
        return {"folds": self.fold_log, "inspection": self.inspection()}


def make_lr(n_jobs: int = None, C: float = 1.0):
    # lbfgs + multiclass labels -> multinomial (softmax) loss in scikit-learn >= 1.5
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def make_rf(n_jobs: int = -1, n_estimators: int = 200):
    return RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                  random_state=RF_SEED, n_jobs=n_jobs)


def classifier_lr(name: str = "classifier_lr", featurizer: Callable = featurize) -> Classifier:
    return Classifier(name, make_lr, [{"C": c} for c in (0.1, 1, 10)], featurizer=featurizer)


def classifier_rf(name: str = "classifier_rf", featurizer: Callable = featurize) -> Classifier:
    return Classifier(name, make_rf, [{"n_estimators": n} for n in (200, 500)], featurizer=featurizer)


# Most frequently chosen setting across the 5 outer folds of the tuned run
# (baselines/results/metrics.json -> methods.<name>.training.folds):
#   LR: tau 120 (3/5 folds), C 0.1 (3/5)      RF: tau 60 (3/5), n_estimators 500 (3/5)
FIXED_LR = {"tau": 120, "params": {"C": 0.1}}
FIXED_RF = {"tau": 60, "params": {"n_estimators": 500}}


def fixed_lr(name: str = "classifier_lr_fixed") -> Classifier:
    return Classifier(name, make_lr, [FIXED_LR["params"]], taus=(FIXED_LR["tau"],))


def fixed_rf(name: str = "classifier_rf_fixed") -> Classifier:
    return Classifier(name, make_rf, [FIXED_RF["params"]], taus=(FIXED_RF["tau"],))
