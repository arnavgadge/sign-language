"""
evaluation/layer1_face_eval.py
--------------------------------
Classification evaluation for the ONE learned component in Layer 1: the
pre-trained Haar-cascade face detector (layer1/face_signal.py).

Why only this component: every other Layer 1 output (word normalization,
time-period bucketing, motion-speed normalization) is a deterministic
formula with no "ground truth" to be right or wrong about — see
tests/test_layer1.py for their correctness tests instead. Face detection
is the one place Layer 1 makes a real prediction, so it's the one place
accuracy/precision/recall/F1/confusion-matrix are meaningful metrics.

There are no trainable weights here either (the cascade is pre-trained by
OpenCV) — so there is no "training split". Instead:
  - a VALIDATION split supports optional hyperparameter tuning (--tune),
  - a TEST split (never touched by tuning) produces the final reported
    numbers.

If no labeled data exists yet, this script says so explicitly and exits
without producing any numbers — see collect_face_eval_samples.py to
generate real ground truth.

Usage
-----
  venv310\\Scripts\\python.exe evaluation\\layer1_face_eval.py
  venv310\\Scripts\\python.exe evaluation\\layer1_face_eval.py --tune
"""
import argparse
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from layer1.config import EVAL_FACES_DIR, EVAL_REPORTS_DIR, RANDOM_SEED, Layer1Config
from layer1.face_signal import FaceDetector

LABELS_CSV = os.path.join(EVAL_FACES_DIR, "labels.csv")

VAL_FRAC = 0.30   # held out ONLY for optional hyperparameter tuning
# remainder (0.70) is TEST — never touched by tuning, final numbers only


def _load_labels() -> list:
    if not os.path.exists(LABELS_CSV):
        return []
    rows = []
    with open(LABELS_CSV, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            path = os.path.join(EVAL_FACES_DIR, row["filename"])
            if os.path.exists(path):
                rows.append((path, int(row["face_present"])))
    return rows


def _split(rows: list, seed: int = RANDOM_SEED) -> tuple:
    rows = list(rows)
    random.Random(seed).shuffle(rows)   # reproducible, seeded shuffle
    n_val = int(len(rows) * VAL_FRAC)
    return rows[:n_val], rows[n_val:]   # (val, test)


def _confusion_and_metrics(detector: FaceDetector, rows: list, scale_factor, min_neighbors, min_size) -> dict:
    tp = fp = tn = fn = 0
    latencies_ms = []

    for path, label in rows:
        frame = cv2.imread(path)
        if frame is None:
            continue

        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector._cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        pred = 1 if len(faces) > 0 else 0

        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    return {
        "n_samples": n,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_inference_ms": round(sum(latencies_ms) / len(latencies_ms), 3) if latencies_ms else None,
    }


def run(tune: bool = False) -> dict:
    rows = _load_labels()

    if len(rows) == 0:
        msg = (
            "NO LABELED DATA FOUND.\n"
            f"Expected: {LABELS_CSV}\n"
            "This evaluation cannot fabricate results without ground truth.\n"
            "Run collect_face_eval_samples.py to build a real labeled set "
            "(aim for 30+ examples of each class), then re-run this script."
        )
        print(msg)
        report = {"status": "no_data", "message": msg, "config": Layer1Config().to_dict()}
        _save_report(report)
        return report

    n_present = sum(1 for _, l in rows if l == 1)
    n_absent = len(rows) - n_present
    print(f"[INFO] Loaded {len(rows)} labeled samples ({n_present} present, {n_absent} absent)")

    val_rows, test_rows = _split(rows)
    print(f"[INFO] Split: {len(val_rows)} validation, {len(test_rows)} test (seed={RANDOM_SEED})")

    detector = FaceDetector()
    if not detector.is_available:
        msg = "Haar cascade failed to load — cannot evaluate."
        print(msg)
        report = {"status": "detector_unavailable", "message": msg}
        _save_report(report)
        return report

    cfg = Layer1Config()
    chosen_min_neighbors = cfg.face_cascade_min_neighbors
    tuning_results = None

    if tune and len(val_rows) > 0:
        print("[INFO] --tune: sweeping minNeighbors on VALIDATION split only")
        candidates = [3, 4, 5, 6, 7]
        tuning_results = []
        best_f1, best_mn = -1.0, chosen_min_neighbors
        for mn in candidates:
            m = _confusion_and_metrics(detector, val_rows, cfg.face_cascade_scale_factor,
                                        mn, cfg.face_cascade_min_size)
            tuning_results.append({"min_neighbors": mn, **m})
            print(f"    minNeighbors={mn}: F1={m['f1']}  (val n={m['n_samples']})")
            if m["f1"] > best_f1:
                best_f1, best_mn = m["f1"], mn
        chosen_min_neighbors = best_mn
        print(f"[INFO] Selected minNeighbors={chosen_min_neighbors} from validation sweep")
    elif tune:
        print("[WARN] --tune requested but validation split is empty; using default config")

    print(f"[INFO] Evaluating on TEST split (n={len(test_rows)}) — never used for tuning")
    test_metrics = _confusion_and_metrics(
        detector, test_rows, cfg.face_cascade_scale_factor,
        chosen_min_neighbors, cfg.face_cascade_min_size
    )

    cascade_size_bytes = os.path.getsize(detector.cascade_path) if detector.cascade_path else None

    report = {
        "status": "ok",
        "config": {**cfg.to_dict(), "min_neighbors_used": chosen_min_neighbors},
        "dataset": {
            "total_labeled": len(rows), "n_present": n_present, "n_absent": n_absent,
            "n_validation": len(val_rows), "n_test": len(test_rows),
            "random_seed": RANDOM_SEED,
        },
        "tuning": tuning_results,
        "test_metrics": test_metrics,
        "model_info": {
            "detector": "cv2.CascadeClassifier (haarcascade_frontalface_default.xml)",
            "trained_by": "OpenCV (pre-trained, not trained in this project)",
            "file_size_bytes": cascade_size_bytes,
            "file_size_kb": round(cascade_size_bytes / 1024, 1) if cascade_size_bytes else None,
        },
    }

    print(json.dumps(report["test_metrics"], indent=2))
    _save_report(report)
    return report


def _save_report(report: dict) -> None:
    os.makedirs(EVAL_REPORTS_DIR, exist_ok=True)
    path = os.path.join(EVAL_REPORTS_DIR, f"layer1_face_eval_{int(time.time())}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[SAVED] {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                         help="Sweep minNeighbors on the validation split before final test evaluation")
    args = parser.parse_args()
    run(tune=args.tune)
