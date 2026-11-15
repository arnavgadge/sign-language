"""
build_webcam_landmark_prototypes.py
------------------------------------
Run MediaPipe on the existing dataset_custom images (already collected)
to extract landmark features and build webcam-calibrated prototypes.

No new data collection needed — uses what's already there.

Output: model/landmark_prototypes.pkl  (overwrites Kaggle-based version)
"""

import os, sys, pickle, random
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

sys.path.insert(0, ".")
from utils.hand_detector import _ensure_model

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

CUSTOM_DIR  = "dataset_custom"
OUTPUT_PATH = os.path.join("model", "landmark_prototypes.pkl")


def normalize_landmarks(landmarks) -> np.ndarray:
    pts   = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    pts  -= pts[0]
    scale = np.linalg.norm(pts[12]) + 1e-8
    return (pts / scale).flatten()


print("[INFO] Loading MediaPipe hand landmarker ...")
model_path = _ensure_model()
base_opts  = mp_tasks.BaseOptions(model_asset_path=model_path)
opts = mp_vision.HandLandmarkerOptions(
    base_options=base_opts,
    num_hands=1,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(opts)
print("[INFO] Ready.\n")

prototypes = {}

for letter in sorted(os.listdir(CUSTOM_DIR)):
    folder = os.path.join(CUSTOM_DIR, letter)
    if not os.path.isdir(folder):
        continue

    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        print(f"  {letter}: no images — skipped")
        continue

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

with open(OUTPUT_PATH, "wb") as fh:
    pickle.dump(prototypes, fh)

print(f"\n[DONE] {len(prototypes)} webcam landmark prototypes -> {OUTPUT_PATH}")
print("Run:  python main.py")
