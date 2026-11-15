"""
collect_face_eval_samples.py
------------------------------
Interactive tool to build the ONE labeled dataset Layer 2 (Phase 2) needs:
a small, human-labeled set of frames for evaluating the face-presence
detector in layer1/face_signal.py.

No such dataset exists in this repository or anywhere else in this
environment (checked: no bundled sample images, no scikit-image install).
Rather than fabricate evaluation numbers, this script lets you generate
real ground truth from your own webcam.

For each captured frame YOU decide (by keypress) whether a face is
present — that human judgment is the ground truth a face detector is
evaluated against.

Usage
-----
  python collect_face_eval_samples.py

Controls
--------
  Y / SPACE  : save current frame, labeled face_present = 1
  N          : save current frame, labeled face_present = 0
  Q / ESC    : quit

Output
------
  data/eval/faces/<uuid>.jpg
  data/eval/faces/labels.csv   (filename,face_present)

Aim for at least ~30 of each label (60+ total) before running
evaluation/layer1_face_eval.py — small sets produce noisy metrics.
"""
import csv
import os
import sys
import uuid

import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))
from layer1.config import EVAL_FACES_DIR

LABELS_CSV = os.path.join(EVAL_FACES_DIR, "labels.csv")


def _load_existing_counts() -> tuple:
    pos = neg = 0
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, "r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["face_present"] == "1":
                    pos += 1
                else:
                    neg += 1
    return pos, neg


def _append_label(filename: str, face_present: int) -> None:
    write_header = not os.path.exists(LABELS_CSV)
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["filename", "face_present"])
        writer.writerow([filename, face_present])


def run() -> None:
    os.makedirs(EVAL_FACES_DIR, exist_ok=True)
    pos, neg = _load_existing_counts()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam.")

    print("[INFO] Y/SPACE = label face present | N = label face absent | Q = quit")
    print(f"[INFO] Existing labels: {pos} present, {neg} absent")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        hud = frame.copy()
        cv2.putText(hud, f"Present: {pos}  Absent: {neg}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(hud, "Y/SPACE=face present  N=face absent  Q=quit", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Face Eval Sample Collector", hud)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        elif key in (ord("y"), ord("Y"), 32):
            fname = f"{uuid.uuid4()}.jpg"
            cv2.imwrite(os.path.join(EVAL_FACES_DIR, fname), frame)
            _append_label(fname, 1)
            pos += 1
            print(f"[SAVE] {fname} -> face_present=1  (total present={pos})")

        elif key in (ord("n"), ord("N")):
            fname = f"{uuid.uuid4()}.jpg"
            cv2.imwrite(os.path.join(EVAL_FACES_DIR, fname), frame)
            _append_label(fname, 0)
            neg += 1
            print(f"[SAVE] {fname} -> face_present=0  (total absent={neg})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Final counts: {pos} present, {neg} absent -> {LABELS_CSV}")


if __name__ == "__main__":
    run()
