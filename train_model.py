"""
train_model.py
--------------
Train a lightweight CNN on the ASL Alphabet dataset (A–Z only).

Dataset expected layout
-----------------------
  asl_alphabet_train/
  asl_alphabet_train/   ← inner folder (Kaggle download structure)
    A/   B/   C/  ...  Z/   del/  nothing/  space/

Only A–Z folders are used; del/nothing/space are ignored.

Outputs
-------
  model/model.h5       – saved Keras model
  model/labels.pkl     – dict  {index: letter},  e.g. {0:'A', 1:'B', …}
  model/training_history.png  – accuracy / loss curves

Usage
-----
  python train_model.py
"""

import os
import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ═══════════════════════════════════════════════════════
#  CONFIG  –  edit DATASET_PATH if your folder differs
# ═══════════════════════════════════════════════════════
DATASET_PATH = r"AE Cp\dekstop files\asl_alphabet_train\asl_alphabet_train"
MODEL_DIR    = "model"
IMG_SIZE     = 64          # must match utils/preprocessing.py
BATCH_SIZE   = 64
EPOCHS       = 25
LR           = 1e-3

# Only the 26 ASL letters – skip del / nothing / space
VALID_CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]  # ['A'..'Z']
NUM_CLASSES   = len(VALID_CLASSES)   # 26


# ═══════════════════════════════════════════════════════
#  CNN ARCHITECTURE
# ═══════════════════════════════════════════════════════
def build_model(num_classes: int) -> tf.keras.Model:
    """
    Lightweight sequential CNN.

    Layer flow
    ----------
    Input (64, 64, 3)
      ↓
    Block 1 : Conv(32)  → BN → Conv(32)  → BN → MaxPool → Dropout(0.25)
    Block 2 : Conv(64)  → BN → Conv(64)  → BN → MaxPool → Dropout(0.25)
    Block 3 : Conv(128) → BN             → MaxPool → Dropout(0.25)
      ↓
    Flatten → Dense(256, relu) → Dropout(0.50) → Dense(26, softmax)

    Feature map sizes (64 input):
      After Block1 pool : 32×32×32
      After Block2 pool : 16×16×64
      After Block3 pool :  8× 8×128  → 8192 flatten units

    Total params ≈ 3.4 M  — fast enough for real-time 30 fps inference.
    """
    model = models.Sequential([
        # ── Block 1 ──────────────────────────────────────────────────
        layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 2 ──────────────────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Block 3 ──────────────────────────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ── Classifier head ──────────────────────────────────────────
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.50),
        layers.Dense(num_classes, activation="softmax"),
    ], name="asl_cnn")

    return model


# ═══════════════════════════════════════════════════════
#  DATA GENERATORS
# ═══════════════════════════════════════════════════════
def make_generators(dataset_path: str):
    """
    Build train / validation generators with data augmentation.

    Augmentation choices
    --------------------
    • rotation_range = 10°   – slight tilt variation
    • width/height shift = 10% – minor position shifts
    • zoom_range = 0.10       – small scale variation
    • horizontal_flip = False – MUST be False: flipped hands change meaning
    • brightness = (0.8, 1.2) – lighting robustness

    Validation split = 20 %
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.20,
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        brightness_range=(0.8, 1.2),
        horizontal_flip=False,   # ← keep False; flipping breaks letter shapes
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.20,
    )

    common = dict(
        directory=dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=VALID_CLASSES,
        shuffle=True,
        seed=42,
    )

    train_gen = train_datagen.flow_from_directory(**common, subset="training")
    val_gen   = val_datagen.flow_from_directory(**common,   subset="validation")

    return train_gen, val_gen


# ═══════════════════════════════════════════════════════
#  SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════
def save_labels(class_indices: dict, out_dir: str) -> None:
    """
    class_indices : {'A': 0, 'B': 1, …} (from Keras generator)
    Saves the *inverted* map {0: 'A', 1: 'B', …} as labels.pkl
    so that during inference we can do  label_map[np.argmax(pred)]
    """
    label_map = {v: k for k, v in class_indices.items()}
    path = os.path.join(out_dir, "labels.pkl")
    with open(path, "wb") as fh:
        pickle.dump(label_map, fh)
    print(f"[SAVE] label map  → {path}")
    print(f"       classes     : {[label_map[i] for i in sorted(label_map)]}")


def plot_history(history, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train acc")
    axes[0].plot(history.history["val_accuracy"], label="Val acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"],     label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Val loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    path = os.path.join(out_dir, "training_history.png")
    fig.savefig(path, dpi=120)
    print(f"[SAVE] training plot  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main() -> None:
    print("=" * 60)
    print("  AE-CP  |  ASL CNN Training")
    print("=" * 60)

    # ── Verify dataset path ───────────────────────────────────────
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {DATASET_PATH!r}\n"
            "  Edit the DATASET_PATH variable at the top of train_model.py."
        )

    # ── Create output directory ───────────────────────────────────
    pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # ── Build data generators ─────────────────────────────────────
    print("\n[DATA] Building generators …")
    train_gen, val_gen = make_generators(DATASET_PATH)
    print(f"       Train batches : {len(train_gen)}")
    print(f"       Val   batches : {len(val_gen)}")
    print(f"       Classes       : {list(train_gen.class_indices.keys())}")

    # ── Build & compile model ─────────────────────────────────────
    print("\n[MODEL] Building CNN …")
    model = build_model(NUM_CLASSES)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ────────────────────────────────────────────────
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n[TRAIN] Starting training  (up to {EPOCHS} epochs) …\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[cb_checkpoint, cb_early_stop, cb_reduce_lr],
    )

    # ── Save labels & plot ────────────────────────────────────────
    save_labels(train_gen.class_indices, MODEL_DIR)
    plot_history(history, MODEL_DIR)

    # ── Final report ─────────────────────────────────────────────
    best_val_acc = max(history.history["val_accuracy"])
    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Best val accuracy : {best_val_acc * 100:.2f} %")
    print(f"  Model saved to    : {MODEL_DIR}/model.h5")
    print(f"  Labels saved to   : {MODEL_DIR}/labels.pkl")
    print("=" * 60)


if __name__ == "__main__":
    # Optional: limit GPU memory growth to avoid OOM on smaller cards
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    main()
