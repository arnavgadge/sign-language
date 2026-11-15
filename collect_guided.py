"""
collect_guided.py
-----------------
Guided auto-capture for all 26 ASL letters.
Hold the sign steady — samples save automatically.

Saves landmark feature vectors directly to data/landmarks.csv
(MediaPipe is already running live — no re-detection needed during training).
Also saves hand crop images to dataset_custom/ as backup.

Flow per letter:
  PREPARE   show "Next: A", wait 2 sec (or press SPACE)
  COUNTDOWN 3 … 2 … 1
  CAPTURING auto-saves 150 frames while hand is detected
  DONE      brief pause, moves to next letter

Controls:
  SPACE   skip current letter
  R       redo current letter (deletes this round's captures)
  Q/ESC   quit early

Total time: ~8 minutes for all 26 letters.
"""

import os
import sys
import time
import glob
import csv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.hand_detector import HandDetector
from utils.preprocessing import mediapipe_hand_crop
from utils.geometric_features import extract as geo_extract, FEATURE_DIM

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_PER_LETTER = 150
CAPTURE_EVERY_N    = 3       # capture 1 frame out of every N (adds variation)
COUNTDOWN_SEC      = 3
PREPARE_SEC        = 2
SAVE_DIR           = "dataset_custom"
CSV_PATH           = os.path.join("data", "landmarks.csv")
LETTERS            = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)
C_GREEN  = (0,   220, 0)
C_RED    = (0,   0,   220)
C_YELLOW = (0,   220, 220)
C_CYAN   = (255, 220, 0)
C_ORANGE = (0,   160, 255)
C_GRAY   = (120, 120, 120)
C_DARK   = (30,  30,  30)


def draw_progress_bar(frame, x, y, w, h, fraction, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * fraction), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_WHITE, 1)


def count_existing_csv(letter):
    if not os.path.exists(CSV_PATH):
        return 0
    count = 0
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == letter:
                count += 1
    return count


def save_landmark_row(writer, letter, feat):
    writer.writerow([letter] + [f"{v:.6f}" for v in feat])


def delete_session_rows(letter, session_count):
    """Remove the last session_count rows for this letter from the CSV."""
    if not os.path.exists(CSV_PATH) or session_count == 0:
        return
    rows = []
    with open(CSV_PATH, "r", newline="") as f:
        rows = list(csv.reader(f))

    # Remove last session_count rows matching this letter (from the end)
    removed = 0
    kept = []
    for row in reversed(rows):
        if removed < session_count and row and row[0] == letter:
            removed += 1
        else:
            kept.append(row)
    kept.reverse()

    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerows(kept)


def run():
    os.makedirs("data", exist_ok=True)

    # Open CSV in append mode — keeps data from previous runs
    csv_file   = open(CSV_PATH, "a", newline="")
    csv_writer = csv.writer(csv_file)

    detector = HandDetector(max_hands=1,
                            detection_confidence=0.6,
                            tracking_confidence=0.6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    letter_idx    = 0
    state         = "PREPARE"
    state_start   = time.time()
    captured      = 0
    frame_tick    = 0
    session_count = 0     # rows written this round (for redo)

    print(f"[INFO] Saving landmarks to {CSV_PATH}")
    print(f"[INFO] {SAMPLES_PER_LETTER} samples per letter.")
    print(f"[INFO] Controls: SPACE=skip  R=redo  Q=quit\n")

    while letter_idx < len(LETTERS):
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        frame_clean = frame.copy()

        letter = LETTERS[letter_idx]

        # ── Hand detection ────────────────────────────────────────────────────
        hands, frame = detector.find_hands(frame, draw=True)
        hand_detected = bool(hands)
        hand_canvas   = None
        feat          = None

        if hands:
            bbox = hands[0]['bbox']
            hand_canvas, _ = mediapipe_hand_crop(frame_clean, bbox)
            feat = geo_extract(hands[0]['lmRaw'])

        # ── Dark top panel ────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (fw, 70), C_DARK, -1)

        # ── Letter progress bar (very top) ───────────────────────────────────
        draw_progress_bar(frame, 0, 0, fw, 8,
                          letter_idx / len(LETTERS), C_CYAN)
        cv2.putText(frame, f"Letter {letter_idx + 1}/{len(LETTERS)}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1, cv2.LINE_AA)

        existing = count_existing_csv(letter)
        cv2.putText(frame, f"CSV rows: {existing}",
                    (fw - 180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1, cv2.LINE_AA)

        # ── STATE MACHINE ─────────────────────────────────────────────────────
        elapsed = time.time() - state_start

        if state == "PREPARE":
            cv2.putText(frame, letter,
                        (fw // 2 - 80, fh // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 9.0, C_YELLOW, 18, cv2.LINE_AA)
            cv2.putText(frame, f"Next: {letter}",
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_YELLOW, 2, cv2.LINE_AA)
            cv2.putText(frame, "Get your hand ready...",
                        (fw // 2 - 160, fh - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_WHITE, 2, cv2.LINE_AA)
            cv2.putText(frame, "SPACE=skip  R=redo  Q=quit",
                        (fw // 2 - 155, fh - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GRAY, 1, cv2.LINE_AA)

            if elapsed >= PREPARE_SEC:
                state = "COUNTDOWN"
                state_start = time.time()

        elif state == "COUNTDOWN":
            count = COUNTDOWN_SEC - int(elapsed)
            if count <= 0:
                state = "CAPTURING"
                state_start = time.time()
                captured = 0
                session_count = 0
                frame_tick = 0
            else:
                cv2.putText(frame, letter,
                            (fw // 2 - 80, fh // 2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 9.0, C_YELLOW, 18, cv2.LINE_AA)
                cv2.putText(frame, f"Starting in {count}...",
                            (fw // 2 - 130, fh - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_ORANGE, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Signing: {letter}",
                            (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_ORANGE, 2, cv2.LINE_AA)

        elif state == "CAPTURING":
            frame_tick += 1

            if hand_detected and feat is not None:
                if frame_tick % CAPTURE_EVERY_N == 0 and captured < SAMPLES_PER_LETTER:
                    save_landmark_row(csv_writer, letter, feat)
                    csv_file.flush()
                    captured += 1
                    session_count += 1

            # Hand canvas preview
            if hand_canvas is not None:
                preview = cv2.resize(hand_canvas, (130, 130), interpolation=cv2.INTER_AREA)
                frame[70:200, fw - 140:fw - 10] = preview
                cv2.rectangle(frame, (fw - 140, 70), (fw - 10, 200), C_GREEN, 2)

            letter_clr = C_GREEN if hand_detected else C_RED
            cv2.putText(frame, letter,
                        (fw // 2 - 80, fh // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 9.0, letter_clr, 18, cv2.LINE_AA)

            status_txt = "CAPTURING" if hand_detected else "NO HAND - show hand"
            status_clr = C_GREEN if hand_detected else C_RED
            cv2.putText(frame, status_txt,
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_clr, 2, cv2.LINE_AA)

            cv2.putText(frame, f"{captured} / {SAMPLES_PER_LETTER}",
                        (fw // 2 - 80, fh - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, C_WHITE, 3, cv2.LINE_AA)

            draw_progress_bar(frame, 20, fh - 60, fw - 40, 22,
                              captured / SAMPLES_PER_LETTER, C_GREEN)

            cv2.putText(frame, "R=redo  SPACE=skip",
                        (fw // 2 - 110, fh - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_GRAY, 1, cv2.LINE_AA)

            if captured >= SAMPLES_PER_LETTER:
                csv_file.flush()
                print(f"[DONE] {letter}: {captured} landmark rows saved.")
                state = "DONE"
                state_start = time.time()

        elif state == "DONE":
            cv2.putText(frame, letter,
                        (fw // 2 - 80, fh // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 9.0, C_GREEN, 18, cv2.LINE_AA)
            cv2.putText(frame, f"Done! {captured} samples",
                        (fw // 2 - 140, fh - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_GREEN, 2, cv2.LINE_AA)

            if elapsed >= 1.2:
                letter_idx += 1
                state = "PREPARE"
                state_start = time.time()

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            print("[QUIT] Exiting early.")
            break

        elif key == ord(" "):
            print(f"[SKIP] Skipping {letter}.")
            letter_idx += 1
            state = "PREPARE"
            state_start = time.time()

        elif key in (ord("r"), ord("R")):
            if state in ("CAPTURING", "DONE"):
                delete_session_rows(letter, session_count)
                print(f"[REDO] {letter}: removed {session_count} rows, restarting.")
            else:
                print(f"[REDO] {letter}: restarting.")
            captured = 0
            session_count = 0
            state = "PREPARE"
            state_start = time.time()

        cv2.imshow("AE-CP | Guided Collection", frame)

    csv_file.close()
    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    if letter_idx >= len(LETTERS):
        print("\n[COMPLETE] All 26 letters collected!")
    print(f"[INFO] Landmarks saved to {CSV_PATH}")
    print("[INFO] Now run:  python train_geometric_classifier.py")


if __name__ == "__main__":
    run()
