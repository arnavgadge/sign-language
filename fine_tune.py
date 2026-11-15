"""
fine_tune.py
------------
Fine-tune the top Dense layers of the existing model using your webcam
samples + a small Kaggle subset per class.

Why this is fast (~10-20 min vs 6 hrs)
---------------------------------------
- Starts from your existing model.h5 (already trained weights)
- Freezes ALL conv layers (no conv retraining)
- Only retrains the last Dense(256) + Dense(26) layers
- Uses small data: webcam samples + 50 Kaggle images per class

How to collect webcam samples first
-------------------------------------
  1. Run:  python main.py
  2. Press S  →  collection mode ON
  3. Show each ASL sign → press that letter key (5 times per letter)
  4. Press S  →  collection mode OFF, then Q to quit
  5. Run:  python fine_tune.py

Usage
------
  python fine_tune.py
"""

import os
import sys
import pickle
import random

import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.path.join("model", "model.h5")
FINETUNED_PATH = os.path.join("model", "model_finetuned.h5")
LABELS_PATH    = os.path.join("model", "labels.pkl")
CUSTOM_DIR     = "dataset_custom"

# Kaggle training dataset (the same one used in train_model.py)
KAGGLE_DIR = os.path.join(
    "AE Cp", "dekstop files",
    "asl_alphabet_train", "asl_alphabet_train"
)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE         = 64     # must match training
BATCH_SIZE       = 32
EPOCHS           = 25
LR               = 5e-5   # very low LR — shift weights gently, don't destroy them
KAGGLE_PER_CLASS = 30     # smaller Kaggle subset so webcam data has more influence
WEBCAM_WEIGHT    = 15     # repeat each webcam sample N times before augmentation

# ── Sanity checks ─────────────────────────────────────────────────────────────
for path, label in [(MODEL_PATH, "Model"), (LABELS_PATH, "Labels")]:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] {label} not found: {path}\n  Run train_model.py first.")

if not os.path.exists(CUSTOM_DIR) or not os.listdir(CUSTOM_DIR):
    sys.exit(
        f"[ERROR] No webcam samples found in '{CUSTOM_DIR}/'.\n"
        "  Open main.py → press S → press A-Z to save samples → press Q."
    )

# ── Load label map ────────────────────────────────────────────────────────────
with open(LABELS_PATH, "rb") as fh:
    label_map = pickle.load(fh)          # {0:'A', 1:'B', ...}
label_to_idx = {v: k for k, v in label_map.items()}
all_letters  = [label_map[i] for i in range(26)]

# ── Image loader ──────────────────────────────────────────────────────────────
def load_img(path: str) -> "np.ndarray | None":
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def load_folder(folder: str, limit: int = 0) -> list:
    """Return list of loaded images from a folder."""
    if not os.path.isdir(folder):
        return []
    exts  = {".jpg", ".jpeg", ".png"}
    files = [os.path.join(folder, f)
             for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in exts]
    if limit and len(files) > limit:
        files = random.sample(files, limit)
    imgs = [load_img(p) for p in files]
    return [im for im in imgs if im is not None]


# ── Build dataset ─────────────────────────────────────────────────────────────
print("\n[INFO] Loading images...")
X, y = [], []
webcam_total = 0
kaggle_total = 0

for letter in all_letters:
    idx = label_to_idx[letter]

    # Webcam samples — oversampled so they influence the loss proportionally
    webcam_imgs = load_folder(os.path.join(CUSTOM_DIR, letter))
    for img in webcam_imgs:
        for _ in range(WEBCAM_WEIGHT):
            X.append(img)
            y.append(idx)
    webcam_total += len(webcam_imgs) * WEBCAM_WEIGHT

    # Kaggle subset — keeps the model from forgetting the other letters
    kaggle_imgs = load_folder(os.path.join(KAGGLE_DIR, letter), limit=KAGGLE_PER_CLASS)
    for img in kaggle_imgs:
        X.append(img)
        y.append(idx)
    kaggle_total += len(kaggle_imgs)

if len(X) == 0:
    sys.exit("[ERROR] No images loaded. Check KAGGLE_DIR and CUSTOM_DIR paths.")

X = np.array(X, dtype=np.float32)
Y = tf.keras.utils.to_categorical(y, num_classes=26)

# Report which letters have webcam samples
webcam_letters = sorted([
    l for l in all_letters
    if os.path.isdir(os.path.join(CUSTOM_DIR, l))
    and len(os.listdir(os.path.join(CUSTOM_DIR, l))) > 0
])
missing = [l for l in all_letters if l not in webcam_letters]

print(f"[INFO] Webcam samples : {webcam_total // WEBCAM_WEIGHT} raw "
      f"({webcam_total} after {WEBCAM_WEIGHT}x oversample)")
print(f"[INFO] Kaggle samples : {kaggle_total}")
print(f"[INFO] Total           : {len(X)}")
print(f"[INFO] Letters with webcam data: {webcam_letters}")
if missing:
    print(f"[WARN] No webcam data for   : {missing}")
    print(f"[WARN] Those letters rely on Kaggle data only — collect them later for best results.")

# ── Train/val split ───────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.15, random_state=42, stratify=np.array(y)
)
print(f"\n[INFO] Train: {len(X_train)}  Val: {len(X_val)}")

# ── Load model and freeze conv layers ─────────────────────────────────────────
print(f"\n[INFO] Loading base model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Unfreeze ALL layers — full domain adaptation at very low LR.
# Previous run only trained the Dense head; the frozen conv layers were still
# extracting Kaggle-specific features, so the Dense head couldn't compensate.
for layer in model.layers:
    layer.trainable = True

trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
total_p   = sum(int(np.prod(w.shape)) for w in model.weights)
print(f"[INFO] Trainable: {trainable:,} / {total_p:,} params "
      f"({100 * trainable / total_p:.1f}%)")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ── Augmentation ──────────────────────────────────────────────────────────────
from tensorflow.keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.20,
    brightness_range=[0.55, 1.45],
    shear_range=0.10,
    fill_mode="nearest",
    horizontal_flip=False,   # flipping breaks ASL hand shapes
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        FINETUNED_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ── Train ─────────────────────────────────────────────────────────────────────
steps = max(1, len(X_train) // BATCH_SIZE)
print(f"\n[INFO] Fine-tuning for up to {EPOCHS} epochs "
      f"({steps} steps/epoch)...")
print("[INFO] This should take ~10-20 minutes on CPU.\n")

model.fit(
    aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=steps,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"\n[INFO] Fine-tuned model saved → {FINETUNED_PATH}")
print("[INFO] main.py will automatically load this model next time you run it.")
