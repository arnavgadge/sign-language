"""
build_prototypes.py
-------------------
Extract 256-D feature vectors from your webcam samples using the CNN's
penultimate Dense layer, then compute one mean prototype per letter.

At inference, main.py finds the nearest prototype by cosine similarity
instead of using the raw 26-class softmax — this works much better with
only 5 samples per letter because we bypass the Kaggle-calibrated Dense
head entirely.

Usage
-----
  python build_prototypes.py

Run once after collecting webcam samples. Re-run if you add more samples.
Output: model/prototypes.pkl
"""

import os
import sys
import pickle

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

CUSTOM_DIR      = "dataset_custom"
MODEL_PATH      = os.path.join("model", "model.h5")      # use base model
PROTO_PATH      = os.path.join("model", "prototypes.pkl")
IMG_SIZE        = 64

# ── Load model + build feature extractor ──────────────────────────────────────
print(f"[INFO] Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Dense(256) is the third-from-last layer — a 256-D representation of the hand.
# We use this instead of the final softmax because the softmax was calibrated on
# Kaggle data. The 256-D features are more transferable across domains.
feature_layer   = model.layers[-3]           # Dense(256) after ReLU
# Warm up the model so input/output tensors are defined
model.predict(np.zeros((1, 64, 64, 3), dtype=np.float32), verbose=0)
feature_model   = tf.keras.Model(inputs=model.inputs,
                                 outputs=feature_layer.output)
print(f"[INFO] Feature extractor output: {feature_model.output_shape}")

# ── Load and encode webcam samples ────────────────────────────────────────────
prototypes      = {}   # {letter: mean_feature_vector (256,)}
all_features    = {}   # {letter: [feature_vectors]} — kept for diagnostics

letters_found   = sorted([
    d for d in os.listdir(CUSTOM_DIR)
    if os.path.isdir(os.path.join(CUSTOM_DIR, d))
])

if not letters_found:
    sys.exit(f"[ERROR] No subfolders found in '{CUSTOM_DIR}/'. "
             "Collect webcam samples first (collect_samples.py).")

print(f"\n[INFO] Processing {len(letters_found)} letters...\n")

for letter in letters_found:
    folder = os.path.join(CUSTOM_DIR, letter)
    files  = [os.path.join(folder, f) for f in os.listdir(folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        continue

    feats = []
    for path in files:
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        norm    = resized.astype(np.float32) / 255.0
        feat    = feature_model.predict(norm[np.newaxis], verbose=0)[0]
        feats.append(feat)

    if feats:
        prototypes[letter]   = np.mean(feats, axis=0)   # mean prototype
        all_features[letter] = feats
        print(f"  {letter}: {len(feats)} samples → prototype shape {prototypes[letter].shape}")

# ── Normalise prototypes to unit length (cosine similarity → dot product) ──────
for letter in prototypes:
    v = prototypes[letter]
    prototypes[letter] = v / (np.linalg.norm(v) + 1e-8)

# ── Save ──────────────────────────────────────────────────────────────────────
with open(PROTO_PATH, "wb") as fh:
    pickle.dump(prototypes, fh)

print(f"\n[INFO] Saved {len(prototypes)} prototypes → {PROTO_PATH}")

# ── Self-test: accuracy on the training samples themselves ────────────────────
print("\n[INFO] Self-test (should be high — same images used to build prototypes):")
correct = total = 0
for true_letter, feats in all_features.items():
    for feat in feats:
        feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
        scores    = {l: float(np.dot(feat_norm, p)) for l, p in prototypes.items()}
        predicted = max(scores, key=scores.get)
        if predicted == true_letter:
            correct += 1
        total += 1

print(f"  Prototype self-accuracy: {correct}/{total} = {100*correct/total:.1f}%")
print("\n[DONE] Run  python main.py  — it will automatically use prototypes.")
