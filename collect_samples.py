"""
collect_samples.py
------------------
Standalone webcam sample collector using MediaPipe hand detection.
No TensorFlow — starts in seconds.

Saves the exact same blue-background hand crop that main.py uses for
inference, so training and inference are perfectly aligned.

Usage
-----
  python collect_samples.py

Controls
--------
  A-Z       : save current hand crop for that letter
  Backspace : delete the last saved sample for that letter
  Q / ESC   : quit
"""

import os
import sys
import time

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))
from utils.hand_detector import HandDetector
from utils.preprocessing import mediapipe_hand_crop

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR  = "dataset_custom"
SAVE_SIZE = 200    # saved image size in px
TARGET    = 30     # samples per letter to aim for

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
GREEN  = (0, 220,   0)
YELLOW = (0, 220, 220)
WHITE  = (255, 255, 255)
GRAY   = (60,  60,  60)
RED    = (0,   0,  220)
CYAN   = (255, 220,  0)


# ── File helpers ──────────────────────────────────────────────────────────────

def count_saved() -> dict:
    counts = {}
    if not os.path.isdir(SAVE_DIR):
        return counts
    for letter in sorted(os.listdir(SAVE_DIR)):
        folder = os.path.join(SAVE_DIR, letter)
        if os.path.isdir(folder):
            counts[letter] = len([
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
    return counts


def save_canvas(hand_canvas: np.ndarray, letter: str) -> str:
    folder = os.path.join(SAVE_DIR, letter)
    os.makedirs(folder, exist_ok=True)
    img  = cv2.resize(hand_canvas, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    path = os.path.join(folder, f"{letter}_{int(time.time()*1000)}.jpg")
    cv2.imwrite(path, img)
    return path


def delete_last(letter: str) -> bool:
    folder = os.path.join(SAVE_DIR, letter)
    if not os.path.isdir(folder):
        return False
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    if files:
        os.remove(files[-1])
        return True
    return False


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_hand_preview(frame: np.ndarray, hand_canvas: np.ndarray) -> None:
    """150x150 preview inset — shows exactly what will be saved."""
    fw = frame.shape[1]
    size = 150
    preview = cv2.resize(hand_canvas, (size, size), interpolation=cv2.INTER_AREA)
    x1, y1 = fw - size - 10, 10
    frame[y1:y1+size, x1:x1+size] = preview
    cv2.rectangle(frame, (x1, y1), (x1+size, y1+size), WHITE, 1)
    cv2.putText(frame, "Will save this:",
                (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)


def draw_progress(frame: np.ndarray, counts: dict, h: int, w: int) -> None:
    bar_h   = 110
    cell_w  = w // 26
    start_y = h - bar_h

    cv2.rectangle(frame, (0, start_y), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, f"Progress  (target: {TARGET} per letter — press A-Z to save):",
                (8, start_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)

    for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        cnt  = counts.get(letter, 0)
        done = cnt >= TARGET
        cx   = i * cell_w + cell_w // 2
        cy   = start_y + 35
        clr  = GREEN if done else (YELLOW if cnt > 0 else GRAY)

        cv2.putText(frame, letter,   (cx - 6, cy),      cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)
        cv2.putText(frame, str(cnt), (cx - 5, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1, cv2.LINE_AA)

        bx, by = i * cell_w + 2, start_y + 58
        bw, bh = cell_w - 4, 10
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), GRAY, -1)
        fill = int(bw * min(cnt / TARGET, 1.0))
        cv2.rectangle(frame, (bx, by), (bx+fill, by+bh), GREEN if done else YELLOW, -1)

    total   = sum(counts.values())
    done_ct = sum(1 for c in counts.values() if c >= TARGET)
    cv2.putText(frame, f"Total: {total}   Complete: {done_ct}/26",
                (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, WHITE, 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[INFO] Starting MediaPipe hand detector...")
    detector = HandDetector(max_hands=1, detection_confidence=0.7,
                            tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counts      = count_saved()
    flash_msg   = ""
    flash_frames = 0
    last_letter = "A"
    hand_canvas = None

    print(f"[INFO] Ready.  Press A-Z to save.  Target: {TARGET} per letter.  Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        frame_clean = frame.copy()
        hands, frame = detector.find_hands(frame, draw=True)

        hand_detected = bool(hands)
        if hands:
            bbox = hands[0]['bbox']
            hand_canvas, _ = mediapipe_hand_crop(frame_clean, bbox)
        else:
            hand_canvas = None

        # ── Status ────────────────────────────────────────────────────
        status_txt = "HAND: YES" if hand_detected else "HAND: NO — show hand"
        status_clr = GREEN if hand_detected else RED
        cv2.putText(frame, status_txt,
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_clr, 2, cv2.LINE_AA)
        cv2.putText(frame, "Press A-Z to save sample for that letter",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, GRAY, 1, cv2.LINE_AA)

        cnt = counts.get(last_letter, 0)
        cv2.putText(frame, f"Last: {last_letter}  ({cnt}/{TARGET})",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)

        # ── Hand crop preview ─────────────────────────────────────────
        if hand_canvas is not None:
            draw_hand_preview(frame, hand_canvas)

        # ── Flash ─────────────────────────────────────────────────────
        if flash_frames > 0:
            cv2.putText(frame, flash_msg,
                        (w//2 - 120, h//2 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 3, cv2.LINE_AA)
            flash_frames -= 1

        draw_progress(frame, counts, h, w)
        cv2.imshow("AE-CP | Sample Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        elif key == 8:
            if delete_last(last_letter):
                counts = count_saved()
                flash_msg    = f"DELETED last {last_letter}"
                flash_frames = 25
                print(f"[DELETE] Removed last '{last_letter}' sample.")

        elif ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
            letter = chr(key).upper()
            if hand_canvas is not None:
                path = save_canvas(hand_canvas, letter)
                counts      = count_saved()
                last_letter = letter
                flash_msg   = f"SAVED  {letter}  ({counts.get(letter,0)}/{TARGET})"
                flash_frames = 20
                print(f"[SAVE] {letter} -> {path}  ({counts.get(letter,0)}/{TARGET})")
            else:
                flash_msg    = "No hand detected!"
                flash_frames = 20

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\n[DONE] Final counts:")
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        cnt = counts.get(letter, 0)
        bar = "#" * min(cnt, TARGET) + "." * max(0, TARGET - cnt)
        status = "OK" if cnt >= TARGET else f"need {TARGET - cnt} more"
        print(f"  {letter}: [{bar}] {cnt:2d}  {status}")

    total   = sum(counts.values())
    done_ct = sum(1 for c in counts.values() if c >= TARGET)
    print(f"\n  Total: {total} images  |  {done_ct}/26 letters complete")
    if done_ct == 26:
        print("\n  Run:  python build_prototypes.py")
    else:
        missing = [l for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                   if counts.get(l, 0) < TARGET]
        print(f"  Still need samples for: {', '.join(missing)}")
        print("  Re-run this script to top them up, then run build_prototypes.py")


if __name__ == "__main__":
    main()
