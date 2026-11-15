"""
train_geometric_classifier.py
------------------------------
Train a Random Forest on geometric hand features extracted from:
  1. Kaggle ASL images  (300 per letter — diverse hand shapes)
  2. dataset_custom     (all webcam samples — your specific camera)

Geometric features are invariant to lighting, texture, and background.
No domain gap. Works across any environment.

Output: model/geometric_classifier.pkl

Usage
-----
  python train_geometric_classifier.py
"""

import os
import sys
import pickle
import random

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

sys.path.insert(0, ".")
from utils.hand_detector import _ensure_model
from utils.geometric_features import extract, FEATURE_DIM

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Config ────────────────────────────────────────────────────────────────────
KAGGLE_DIR         = os.path.join("AE Cp", "dekstop files",
                                  "asl_alphabet_train", "asl_alphabet_train")
CUSTOM_DIR         = "dataset_custom"
OUTPUT_PATH        = os.path.join("model", "geometric_classifier.pkl")
KAGGLE_PER_LETTER  = 0      # 0 = skip Kaggle entirely (use only your webcam data)
WEBCAM_WEIGHT      = 1      # no duplication needed when you have 150+ real samples

# ── Setup MediaPipe IMAGE mode ─────────────────────────────────────────────────
print("[INFO] Loading MediaPipe ...")
model_path = _ensure_model()
opts = mp_vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
    num_hands=1,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(opts)
print("[INFO] Ready.\n")


def landmarks_from_image(path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    if result.hand_landmarks:
        return result.hand_landmarks[0]
    return None


# ── Feature cache (skip MediaPipe pass on re-runs) ────────────────────────────
CACHE_PATH = os.path.join("model", "feature_cache.pkl")

def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as fh:
            return pickle.load(fh)
    return {"X": [], "y": []}

def _save_cache(X, y):
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump({"X": X, "y": y}, fh)
    print(f"[INFO] Feature cache saved -> {CACHE_PATH}")

CSV_LANDMARKS = os.path.join("data", "landmarks.csv")

cache = _load_cache()
if cache["X"]:
    print(f"[INFO] Using cached features ({len(cache['X'])} samples). "
          "Delete model/feature_cache.pkl to force re-extraction.\n")
    X, y = cache["X"], cache["y"]

elif os.path.exists(CSV_LANDMARKS):
    # ── Fast path: load pre-extracted landmarks from collect_guided.py ────────
    import csv
    X, y = [], []
    with open(CSV_LANDMARKS, "r", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
    from collections import Counter
    dist = dict(sorted(Counter(y).items()))
    print(f"[INFO] Loaded {len(X)} pre-extracted landmark rows from {CSV_LANDMARKS}")
    print(f"[INFO] Samples per letter: {dist}\n")
    detector.close()
    _save_cache(X, y)

else:
    # ── Fallback: extract features from images (slow) ─────────────────────────
    X, y = [], []

    if KAGGLE_PER_LETTER > 0:
        print("[INFO] Processing Kaggle images ...")
        for letter in sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            folder = os.path.join(KAGGLE_DIR, letter)
            if not os.path.isdir(folder):
                continue
            files = [f for f in os.listdir(folder)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if len(files) > KAGGLE_PER_LETTER:
                files = random.sample(files, KAGGLE_PER_LETTER)
            detected = 0
            for f in files:
                lm = landmarks_from_image(os.path.join(folder, f))
                if lm is not None:
                    X.append(extract(lm))
                    y.append(letter)
                    detected += 1
            print(f"  {letter}: {detected}/{len(files)} Kaggle")

    print("\n[INFO] Processing webcam images ...")
    for letter in sorted(os.listdir(CUSTOM_DIR)):
        folder = os.path.join(CUSTOM_DIR, letter)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            continue
        detected = 0
        for f in files:
            lm = landmarks_from_image(os.path.join(folder, f))
            if lm is not None:
                feat = extract(lm)
                for _ in range(WEBCAM_WEIGHT):
                    X.append(feat)
                    y.append(letter)
                detected += 1
        print(f"  {letter}: {detected}/{len(files)} webcam (x{WEBCAM_WEIGHT})")

    detector.close()

    if not X:
        sys.exit("[ERROR] No data. Run collect_guided.py first.")

    _save_cache(X, y)

if not X:
    sys.exit("[ERROR] No data collected.")

X = np.array(X, dtype=np.float32)
le = LabelEncoder()
Y  = le.fit_transform(y)

print(f"\n[INFO] Total samples : {len(X)}")
print(f"[INFO] Feature dim   : {X.shape[1]}")
print(f"[INFO] Classes       : {list(le.classes_)}\n")

# ── Train / validation split ──────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.15, random_state=42, stratify=Y
)

# ── Train Random Forest ───────────────────────────────────────────────────────
print("[INFO] Training Random Forest ...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
clf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
val_acc = clf.score(X_val, y_val)
print(f"\n[INFO] Validation accuracy: {val_acc*100:.1f}%")
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred, target_names=le.classes_))

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "wb") as fh:
    pickle.dump({"classifier": clf, "label_encoder": le}, fh)

print(f"[DONE] Saved -> {OUTPUT_PATH}")
print("Run:  python main.py")
