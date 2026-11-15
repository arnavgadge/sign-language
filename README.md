# AE-CP — Real-Time ASL Sign Language Recognition → Arduino LCD

A real-time American Sign Language (ASL) letter recognition system using a lightweight CNN, OpenCV, and an Arduino UNO with a 16×2 I2C LCD display.

```
Webcam (hand signs) → CNN prediction → Temporal stabilization → Word accumulation
                                                                        ↓
                                                              Press Q → Serial
                                                                        ↓
                                                              Arduino LCD displays word
```

---

## Project Structure

```
AE-CP/
│
├── train_model.py          ← Train the CNN model
├── main.py                 ← Run real-time detection + word builder
├── requirements.txt        ← Python dependencies
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py    ← Frame pre-processing (BGR→RGB, resize, normalise)
│   └── stabilizer.py       ← Temporal stabilization logic
│
├── model/                  ← Created after training
│   ├── model.h5
│   ├── labels.pkl
│   └── training_history.png
│
├── arduino/
│   └── lcd_display.ino     ← Upload this to your Arduino UNO
│
└── AE Cp/dekstop files/    ← Dataset (already present)
    └── asl_alphabet_train/
        └── asl_alphabet_train/
            ├── A/  B/  C/  …  Z/
            └── del/  nothing/  space/  (ignored during training)
```

---

## Quick Start

### Step 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> Tested with Python 3.9 – 3.11.  Use a virtual environment if possible.

---

### Step 2 — Train the model

```bash
python train_model.py
```

- Training uses only A–Z folders (26 classes × 3,000 images = 78,000 images).
- 20 % of images are used for validation automatically.
- Data augmentation: ±10° rotation, ±10% shift, ±10% zoom, 0.8–1.2× brightness.
- `horizontal_flip = False` (flipping breaks letter orientation).
- Best model checkpoint saved to `model/model.h5`.
- Label map saved to `model/labels.pkl`.
- Training curves saved to `model/training_history.png`.
- Expected val accuracy: **95–99 %** on the Kaggle ASL dataset.

---

### Step 3 — Upload Arduino sketch

1. Open **Arduino IDE**.
2. Install the **LiquidCrystal_I2C** library:
   - Sketch → Include Library → Manage Libraries → search `LiquidCrystal_I2C` (by Frank de Brabander).
3. Open `arduino/lcd_display.ino`.
4. Select board: **Arduino UNO** and correct COM port.
5. Upload.

**LCD Wiring:**

| LCD Pin | Arduino UNO |
|---------|------------|
| VCC     | 5V         |
| GND     | GND        |
| SDA     | A4         |
| SCL     | A5         |

> **I2C address**: the sketch defaults to `0x27`.  If the LCD stays blank, run an I2C Scanner sketch to find the correct address (common alternative: `0x3F`) and update line 47 of `lcd_display.ino`.

---

### Step 4 — Run real-time detection

```bash
python main.py
```

Before running, open `main.py` and edit:

```python
COM_PORT = "COM3"    # ← change to your Arduino's COM port
```

---

## How It Works

### CNN Architecture

| Layer            | Output shape     | Notes                         |
|------------------|-----------------|-------------------------------|
| Input            | 64 × 64 × 3     | RGB, normalised [0, 1]        |
| Conv2D(32) + BN  | 64 × 64 × 32    | padding='same'                |
| Conv2D(32) + BN  | 64 × 64 × 32    |                               |
| MaxPool + Drop   | 32 × 32 × 32    | Dropout 0.25                  |
| Conv2D(64) + BN  | 32 × 32 × 64    |                               |
| Conv2D(64) + BN  | 32 × 32 × 64    |                               |
| MaxPool + Drop   | 16 × 16 × 64    | Dropout 0.25                  |
| Conv2D(128) + BN | 16 × 16 × 128   |                               |
| MaxPool + Drop   | 8 × 8 × 128     | Dropout 0.25                  |
| Flatten          | 8192            |                               |
| Dense(256) + Drop| 256             | Dropout 0.50                  |
| Dense(26, softmax)| 26             | One output per letter         |

~3.4 M parameters — fast enough for 25–30 fps real-time inference on a modern CPU.

---

### Temporal Stabilization (4-layer defence)

Raw CNN predictions at 30 fps are noisy — without stabilization, a single hand sign would append the same letter 20+ times per second.

```
Layer 1 — Confidence gate
  Only accept predictions where model confidence ≥ 80 %.

Layer 2 — Hold-frame counter
  The same letter must appear for 20 consecutive confident frames (~0.67 s)
  before it is accepted.

Layer 3 — Cooldown timer
  After appending a letter, ignore all input for 35 frames (~1.17 s).
  This prevents the same letter being re-appended immediately.

Layer 4 — Gap detector
  If confidence stays below threshold for 12 frames (~0.4 s),
  reset the "last appended" lock.  This allows the same letter to be
  typed again after clearly removing the hand.
```

**To type "ARNAV":**
1. Show A → system holds for ~0.67 s → A appended → cooldown
2. Show R → R appended
3. Show N → N appended
4. Show A → because you removed your hand (gap reset), A is accepted again
5. Show V → V appended
6. Press Q → "ARNAV" sent to Arduino

---

### Serial Communication

Python sends the final word as a UTF-8 string terminated by `\n`:

```python
arduino.write((current_word + "\n").encode("utf-8"))
```

Arduino reads until `\n`, then displays on LCD.  The app continues working even if no Arduino is connected (word is printed to console instead).

---

## On-Screen HUD

```
┌──────────────────────────────────────────────────────────────────┐
│ FPS: 29.8                           ┌──────────────────────────┐ │
│                                     │                          │ │
│  A  ← predicted letter             │    PLACE HAND HERE       │ │
│                                     │                          │ │
│ Conf:  94.2%  [████████████░░░░]   │    (green box ROI)       │ │
│ Hold:  15/20  [████████████░░░░]   │                          │ │
│ Cooldown: 30  [█████████░░░░░░░]   └──────────────────────────┘ │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│ Word: A R N A V                                                   │
│ [Q] Quit & Send to Arduino   [C] Clear   [Backspace] Delete      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Keyboard Shortcuts

| Key        | Action                              |
|------------|-------------------------------------|
| Q  or ESC  | Quit and send final word to Arduino |
| C          | Clear the current word              |
| Backspace  | Delete last letter                  |

---

## Tips for Better Accuracy

1. **Lighting**: ensure your hand is well-lit from the front; avoid backlight.
2. **Background**: plain, contrasting background works best.
3. **Hand position**: keep your hand fully inside the green ROI box.
4. **Distance**: 30–50 cm from the webcam is ideal.
5. **Stabilization tuning** (in `main.py`):
   - Raise `CONF_THRESHOLD` (e.g. 0.90) to reduce false positives.
   - Raise `REQUIRED_FRAMES` (e.g. 25) for more deliberate pacing.
   - Lower `REQUIRED_FRAMES` (e.g. 15) for faster typing.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Run `python train_model.py` first |
| LCD stays blank | Check I2C address (run I2C Scanner); confirm wiring |
| `Cannot open webcam` | Change `VideoCapture(0)` to `VideoCapture(1)` in main.py |
| Serial port error | Check COM_PORT in main.py; install drivers if needed |
| Low accuracy | Ensure hand is inside ROI, good lighting, re-train if needed |
| Letters flickering | Raise CONF_THRESHOLD or REQUIRED_FRAMES in main.py |

---

## Hardware Requirements

- Arduino UNO (or compatible)
- 16×2 LCD with I2C backpack (PCF8574)
- USB cable (Arduino ↔ PC)
- Webcam (built-in or external)
