"""
extract_landmarks_from_images.py
---------------------------------
One-time script: extract landmark feature vectors from existing dataset_custom images.

Fixes the tight-crop detection problem by padding each image with a gray border
before running MediaPipe — this gives the hand detector the context it needs.

Output: data/landmarks.csv  (used directly by train_geometric_classifier.py)
"""

import os
import sys
import csv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

sys.path.insert(0, ".")
from utils.hand_detector import _ensure_model
from utils.geometric_features import extract

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

CUSTOM_DIR = "dataset_custom"
CSV_PATH   = os.path.join("data", "landmarks.csv")
PAD        = 80   # gray border pixels added on each side before detection

os.makedirs("data", exist_ok=True)

print("[INFO] Loading MediaPipe ...")
model_path = _ensure_model()
opts = mp_vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
    num_hands=1,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.1,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(opts)
print("[INFO] Ready.\n")


def detect_with_padding(bgr_img, pad=PAD):
    """Add gray border, run MediaPipe, return landmarks or None."""
    padded = cv2.copyMakeBorder(bgr_img, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(128, 128, 128))
    rgb    = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    if result.hand_landmarks:
        return result.hand_landmarks[0]
    return None


total_saved  = 0
total_missed = 0

with open(CSV_PATH, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)

    for letter in sorted(os.listdir(CUSTOM_DIR)):
        folder = os.path.join(CUSTOM_DIR, letter)
        if not os.path.isdir(folder) or len(letter) != 1:
            continue

        files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if not files:
            print(f"  {letter}: no images — skipped")
            continue

        saved  = 0
        missed = 0
        for fname in files:
            bgr = cv2.imread(os.path.join(folder, fname))
            if bgr is None:
                continue
            lm = detect_with_padding(bgr)
            if lm is not None:
                feat = extract(lm)
                writer.writerow([letter] + [f"{v:.6f}" for v in feat])
                saved += 1
            else:
                missed += 1

        total_saved  += saved
        total_missed += missed
        status = "OK" if missed == 0 else f"{missed} missed"
        print(f"  {letter}: {saved}/{len(files)} saved  ({status})")

detector.close()

print(f"\n[DONE] {total_saved} rows written to {CSV_PATH}")
print(f"       {total_missed} images where MediaPipe still couldn't detect a hand")
if total_missed > 0:
    print(f"       (those images are just skipped — enough good data remains)")
print("\nNow run:  python train_geometric_classifier.py")
