"""
build_landmark_prototypes.py
----------------------------
Run MediaPipe on Kaggle training images to extract hand landmarks,
then build one mean landmark prototype per letter.

Landmark features are COMPLETELY invariant to background and lighting —
no webcam samples needed, works anywhere.

Output: model/landmark_prototypes.pkl

Usage
-----
  python build_landmark_prototypes.py
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

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ── Config ────────────────────────────────────────────────────────────────────
KAGGLE_DIR = os.path.join(
    "AE Cp", "dekstop files",
    "asl_alphabet_train", "asl_alphabet_train"
)
OUTPUT_PATH        = os.path.join("model", "landmark_prototypes.pkl")
SAMPLES_PER_LETTER = 300   # images to try per letter (detected subset used)

# ── Setup IMAGE-mode MediaPipe detector ───────────────────────────────────────
print("[INFO] Loading MediaPipe hand landmarker ...")
model_path = _ensure_model()
base_opts  = mp_tasks.BaseOptions(model_asset_path=model_path)
opts = mp_vision.HandLandmarkerOptions(
    base_options=base_opts,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(opts)
print("[INFO] Ready.\n")


# ── Landmark normalisation ────────────────────────────────────────────────────
def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Convert 21 NormalizedLandmarks -> 42-D feature vector.
    Wrist (lm 0) at origin, scaled by wrist-to-middle-fingertip distance.
    Result is position and scale invariant.
    """
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]                                          # wrist at origin
    scale = np.linalg.norm(pts[12]) + 1e-8                # middle fingertip scale
    return (pts / scale).flatten()                         # (42,)


# ── Build prototypes ──────────────────────────────────────────────────────────
prototypes = {}

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    folder = os.path.join(KAGGLE_DIR, letter)
    if not os.path.isdir(folder):
        print(f"  {letter}: folder not found — skipped")
        continue

    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(files) > SAMPLES_PER_LETTER:
        files = random.sample(files, SAMPLES_PER_LETTER)

    feats = []
    for f in files:
        bgr = cv2.imread(os.path.join(folder, f))
        if bgr is None:
            continue
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)
        if result.hand_landmarks:
            feats.append(normalize_landmarks(result.hand_landmarks[0]))

    if feats:
        proto = np.mean(feats, axis=0)
        prototypes[letter] = proto / (np.linalg.norm(proto) + 1e-8)
        print(f"  {letter}: {len(feats)}/{len(files)} detected -> OK")
    else:
        print(f"  {letter}: 0 hands detected — skipped")

detector.close()

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "wb") as fh:
    pickle.dump(prototypes, fh)

print(f"\n[DONE] {len(prototypes)} landmark prototypes saved -> {OUTPUT_PATH}")
print("Run:  python main.py")
