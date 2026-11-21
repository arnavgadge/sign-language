"""
main.py - TEMPORAL SEQUENCE-BASED ASL RECOGNITION
===================================================
FINAL ARCHITECTURE FOR FULL ASL SUPPORT

Core insight:
  ASL is not single-image classification.
  ASL is TEMPORAL HAND TOPOLOGY.

Pipeline:
  Webcam (30 FPS)
    ↓
  MediaPipe HandLandmarker (lightweight)
    ↓
  Feature Extraction (90-D: finger states, curl, compactness, etc.)
    ↓
  RandomForest Per-Frame Prediction
    ↓
  TemporalBuffer (15 frames, weighted voting)
    ↓
  TemporalStabilizer (persistence gate)
    ↓
  Word Accumulation
    ↓
  Arduino LCD Output

Key changes from old system:
- ✗ REMOVED: apply_rules() decision tree (handcrafted per-letter hacks)
- ✗ REMOVED: geometric_bonus() confidence boosting
- ✓ KEPT: RandomForest predictions (train once, predict always)
- ✓ KEPT: Feature extraction (finger states, curl, compactness)
- ✓ NEW: Temporal buffer (15 frames weighted voting)
- ✓ NEW: Persistence-based gating (10+ frames before acceptance)
- ✓ NEW: Temporal smoothing (ignore single-frame spikes)

Keyboard shortcuts:
  Q/ESC    - Quit and send final word to Arduino
  C        - Clear current word
  Backspace - Delete last letter
  M        - Toggle debug window
  S        - Sample collection mode (A-Z to save)

Results:
- Stable letter recognition across time
- No random spikes (single-frame anomalies ignored)
- Improved D/E detection via finger states
- Recovered 25-30 FPS stable tracking
- Support for significantly more letters robustly
"""

import os
import sys
import time
import pickle

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARNING] pyserial not installed - serial communication disabled.")

sys.path.insert(0, os.path.dirname(__file__))
from utils.preprocessing import mediapipe_hand_crop
from utils.hand_detector import HandDetector
from utils.geometric_features import extract as geo_extract, summary as geo_summary
from utils.motion_tracker import MotionTracker
from utils.temporal_buffer import TemporalBuffer
from utils.temporal_stabilizer import TemporalStabilizer
from resolution.engine import IntentEngine
from resolution.models import ResolutionResult
from resolution.event_gate import WordEventGate
from dashboard.window import EngineDashboard
from stt import SpeechListener


# ==============================================================
#  CONFIG
# ==============================================================
_PROTO_PATH = os.path.join("model", "prototypes.pkl")
_FINETUNED  = os.path.join("model", "model_finetuned.h5")
MODEL_PATH  = (os.path.join("model", "model.h5") if os.path.exists(_PROTO_PATH)
               else (_FINETUNED if os.path.exists(_FINETUNED)
                     else os.path.join("model", "model.h5")))
LABELS_PATH = os.path.join("model", "labels.pkl")

COM_PORT  = "COM5"
BAUD_RATE = 9600

USER_ID = "default"   # Intent Engine user identity; single-user until the UI exposes profile selection

DETECTION_CONFIDENCE = 0.8
TRACKING_CONFIDENCE  = 0.8

# TEMPORAL ARCHITECTURE PARAMETERS
TEMPORAL_WINDOW = 15            # frames to buffer for voting
PERSISTENCE_FRAMES = 10         # frames letter must persist before acceptance
TEMPORAL_CONFIDENCE = 0.65      # weighted confidence threshold
COOLDOWN_FRAMES = 20            # debounce after accepting a letter

CUSTOM_DATASET_DIR = "dataset_custom"


# ==============================================================
#  COLORS (BGR)
# ==============================================================
CLR_GREEN   = (0,   220,   0)
CLR_BLUE    = (255,  80,   0)
CLR_RED     = (0,    0,  220)
CLR_WHITE   = (255, 255, 255)
CLR_BLACK   = (0,    0,   0)
CLR_YELLOW  = (0,   220, 220)
CLR_GRAY    = (60,  60,   60)
CLR_ORANGE  = (0,  140,  255)
CLR_CYAN    = (255, 220,   0)


# ==============================================================
#  LOAD MODEL & LABELS
# ==============================================================
def load_model_and_labels():
    """Load RF classifier or CNN model."""
    # ── 1. Geometric RF classifier (highest priority) ─────────────────────────
    geo_path = os.path.join("model", "geometric_classifier.pkl")
    if os.path.exists(geo_path):
        with open(geo_path, "rb") as fh:
            data = pickle.load(fh)
        print(f"[INFO] Geometric RF classifier loaded")
        print(f"[INFO] Classes: {list(data['label_encoder'].classes_)}")
        print(f"[INFO] Mode: TEMPORAL + GEOMETRIC RF\n")
        return None, data['label_encoder'], None, data['classifier']

    # ── 2. Landmark prototypes fallback ───────────────────────────────────────
    lm_proto_path = os.path.join("model", "landmark_prototypes.pkl")
    if os.path.exists(lm_proto_path):
        with open(lm_proto_path, "rb") as fh:
            lm_prototypes = pickle.load(fh)
        print(f"[INFO] Landmark prototypes: {sorted(lm_prototypes.keys())}")
        print(f"[INFO] Mode: TEMPORAL + LANDMARK\n")
        return None, None, None, lm_prototypes

    # ── Fallback: CNN model ───────────────────────────────────────────────────
    import tensorflow as tf
    for path, label in [(MODEL_PATH, "Model"), (LABELS_PATH, "Labels")]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {label} not found: {path}")

    print(f"[INFO] Loading model  ...  {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "rb") as fh:
        label_map = pickle.load(fh)
    print(f"[INFO] Classes : {[label_map[i] for i in sorted(label_map)]}\n")

    proto_path    = os.path.join("model", "prototypes.pkl")
    feature_model = None
    prototypes    = None
    if os.path.exists(proto_path):
        with open(proto_path, "rb") as fh:
            prototypes = pickle.load(fh)
        feature_model = tf.keras.Model(inputs=model.inputs,
                                       outputs=model.layers[-3].output)
        print(f"[INFO] CNN prototypes loaded: {sorted(prototypes.keys())}\n")
    return model, label_map, feature_model, prototypes


# ==============================================================
#  SERIAL (ARDUINO)
# ==============================================================
def connect_arduino():
    if not SERIAL_AVAILABLE:
        return None
    try:
        arduino = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"[SERIAL] Arduino connected on {COM_PORT} @ {BAUD_RATE} baud")
        return arduino
    except Exception as exc:
        print(f"[SERIAL] Arduino not connected ({exc})")
        print("[SERIAL] Running in display-only mode.")
        return None


def send_word(arduino, word: str) -> None:
    if not word:
        print("[SERIAL] Nothing to send - word is empty.")
        return
    if arduino and arduino.is_open:
        arduino.write((word + "\n").encode("utf-8"))
        arduino.flush()
        print(f"[SERIAL] Sent -> '{word}'")
    else:
        print(f"[SERIAL] (no connection) Final word = '{word}'")


# ==============================================================
#  SAMPLE COLLECTION
# ==============================================================
def save_sample(hand_canvas: np.ndarray, letter: str) -> str:
    """Save the 300x300 hand crop to dataset_custom/<letter>/."""
    folder = os.path.join(CUSTOM_DATASET_DIR, letter)
    os.makedirs(folder, exist_ok=True)
    preview = cv2.resize(hand_canvas, (200, 200), interpolation=cv2.INTER_AREA)
    ts   = int(time.time() * 1000)
    path = os.path.join(folder, f"{letter}_{ts}.jpg")
    cv2.imwrite(path, preview)
    return path


# ==============================================================
#  DRAWING HELPERS (MINIMIZED)
# ==============================================================
def draw_fps(frame: np.ndarray, fps: float, hand_detected: bool) -> None:
    """Draw FPS and hand detection status."""
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_WHITE, 2, cv2.LINE_AA)
    txt = "HAND: YES" if hand_detected else "HAND: NO"
    clr = CLR_GREEN if hand_detected else CLR_RED
    cv2.putText(frame, txt,
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 1, cv2.LINE_AA)


def draw_hand_preview(frame: np.ndarray, hand_canvas: np.ndarray) -> None:
    """150x150 preview in top-right corner."""
    fw   = frame.shape[1]
    size = 150
    preview = cv2.resize(hand_canvas, (size, size), interpolation=cv2.INTER_AREA)
    x1, y1 = fw - size - 10, 10
    frame[y1:y1 + size, x1:x1 + size] = preview
    cv2.rectangle(frame, (x1, y1), (x1 + size, y1 + size), CLR_WHITE, 1)
    cv2.putText(frame, "Model sees:",
                (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_WHITE, 1, cv2.LINE_AA)


def draw_prediction_panel(frame: np.ndarray,
                          display_letter: str,
                          weighted_conf: float,
                          persistence: int) -> None:
    """Draw letter, confidence, and persistence information."""
    px, py = 80, 120
    letter_clr = CLR_YELLOW
    
    # Large letter
    cv2.putText(frame, display_letter if display_letter else "?",
                (px, py), cv2.FONT_HERSHEY_SIMPLEX, 6.0, letter_clr, 10, cv2.LINE_AA)
    
    # Weighted confidence
    conf_text = f"WConf: {weighted_conf * 100:.1f}%"
    cv2.putText(frame, conf_text,
                (px + 20, py + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    
    # Persistence (frames)
    persist_text = f"Persist: {persistence}/{PERSISTENCE_FRAMES}"
    cv2.putText(frame, persist_text,
                (px + 20, py + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLR_CYAN, 1, cv2.LINE_AA)


def draw_word_bar(frame: np.ndarray, word: str, h: int,
                  collection_mode: bool) -> None:
    """Draw word accumulation bar at bottom."""
    cv2.rectangle(frame, (0, h - 80), (frame.shape[1], h), (20, 20, 20), -1)
    cv2.putText(frame, "Word:",
                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.70, CLR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, word if word else "(empty)",
                (80, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.90, CLR_YELLOW, 2, cv2.LINE_AA)

    if collection_mode:
        hints = "[A-Z] Save sample   [S] Exit collection   [Q] Quit"
        cv2.putText(frame, hints,
                    (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, CLR_ORANGE, 1, cv2.LINE_AA)
    else:
        hints = "[Enter] Resolve Intent   [Q] Quit   [C] Clear   [Backspace] Del   [M] Debug   [D] Dashboard"
        cv2.putText(frame, hints,
                    (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)


def draw_resolution_panel(frame: np.ndarray, resolution: "ResolutionResult", h: int) -> None:
    """Draw the Intent Engine's resolution result (Layer 1 -> Layer 3/5 ->
    Reasoning Engine -> Confidence Gate) in the strip between the
    prediction panel and the word bar."""
    if resolution is None:
        return

    panel_top = h - 162
    fw = frame.shape[1]

    cv2.rectangle(frame, (0, panel_top), (fw, h - 82), (10, 10, 30), -1)
    cv2.line(frame, (0, panel_top), (fw, panel_top), (60, 60, 110), 1)

    mode_color = {
        "RESOLVED":  CLR_GREEN,
        "AMBIGUOUS": CLR_YELLOW,
        "LITERAL":   (160, 160, 160),
    }.get(resolution.status, CLR_WHITE)

    label = resolution.selected_label or resolution.recognized_word
    cv2.putText(frame, f"[{resolution.status}] {label}",
                (10, panel_top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)

    # Confidence bar
    bx, by, bw, bh = 10, panel_top + 24, 180, 8
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * resolution.final_score), by + bh), mode_color, -1)
    cv2.putText(frame, f"{resolution.final_score * 100:.0f}%",
                (bx + bw + 5, by + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, resolution.output[:72],
                (10, panel_top + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, CLR_WHITE, 1, cv2.LINE_AA)

    if resolution.top_n:
        for idx, cand in enumerate(resolution.top_n[:3]):
            txt = f"{cand['label'][:20]}: {cand['final_score'] * 100:.0f}%"
            cv2.putText(frame, txt, (10 + idx * 210, panel_top + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (170, 170, 170), 1, cv2.LINE_AA)

    if resolution.conflict:
        cv2.putText(frame, "[CONFLICT]", (fw - 110, panel_top + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, CLR_ORANGE, 1, cv2.LINE_AA)


def draw_flash(frame: np.ndarray, letter: str) -> None:
    """Flash when letter is accepted."""
    fh, fw = frame.shape[:2]
    cv2.putText(frame, letter,
                (fw // 2 - 60, fh // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 5.0, CLR_GREEN, 10, cv2.LINE_AA)


def draw_collection_banner(frame: np.ndarray) -> None:
    """Banner for collection mode."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 36), (60, 0, 60), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "  COLLECTION MODE - Press A-Z to save a sample for that letter",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_ORANGE, 1, cv2.LINE_AA)


STT_PANEL_W = 300   # width of the speech panel appended to the right of the frame


def draw_stt_panel(canvas: np.ndarray, history: list, is_listening: bool) -> None:
    """Draw incoming-speech panel on the right-hand side of the combined canvas."""
    ch, full_w = canvas.shape[:2]
    px = full_w - STT_PANEL_W   # left edge of panel

    cv2.rectangle(canvas, (px, 0), (full_w, ch), (15, 15, 15), -1)
    cv2.line(canvas, (px, 0), (px, ch), (70, 70, 70), 2)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr_clr = CLR_GREEN if is_listening else (100, 100, 100)
    cv2.putText(canvas, "INCOMING SPEECH",
                (px + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54, hdr_clr, 1, cv2.LINE_AA)

    # Recording dot
    dot_clr = (0, 0, 220) if is_listening else (60, 60, 60)
    cv2.circle(canvas, (full_w - 16, 14), 7, dot_clr, -1)
    status = "LIVE" if is_listening else "OFF"
    cv2.putText(canvas, status,
                (full_w - 50, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot_clr, 1, cv2.LINE_AA)

    cv2.line(canvas, (px + 6, 32), (full_w - 6, 32), (55, 55, 55), 1)

    # ── Transcription history (newest on top) ─────────────────────────────────
    max_chars = 30          # chars before wrapping
    y = 54
    line_h = 18
    gap_h = 8

    for utterance in reversed(history[-7:]):
        words = utterance.split()
        line_buf = ""
        lines_this = []
        for word in words:
            candidate = (line_buf + " " + word).strip()
            if len(candidate) <= max_chars:
                line_buf = candidate
            else:
                if line_buf:
                    lines_this.append(line_buf)
                line_buf = word
        if line_buf:
            lines_this.append(line_buf)

        for ln in lines_this:
            if y > ch - 30:
                break
            cv2.putText(canvas, ln,
                        (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (215, 215, 215), 1, cv2.LINE_AA)
            y += line_h
        y += gap_h

    if not history:
        cv2.putText(canvas, "Waiting for speech...",
                    (px + 10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.41,
                    (70, 70, 70), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Other person speaks here",
                (px + 8, ch - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (55, 55, 55), 1, cv2.LINE_AA)


def draw_debug_window(hand_canvas: np.ndarray, geo_metrics: dict) -> None:
    """Minimal debug window showing geometric features."""
    size      = 300
    debug_img = cv2.resize(hand_canvas, (size, size), interpolation=cv2.INTER_AREA)
    cv2.putText(debug_img, "Hand crop",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    y = 34
    for key, val in list(geo_metrics.items())[:10]:  # Show only first 10 metrics
        cv2.putText(debug_img, f"{key}: {val}",
                    (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_CYAN, 1, cv2.LINE_AA)
        y += 16
    cv2.imshow("AE-CP | Debug", debug_img)


# ==============================================================
#  MAIN LOOP
# ==============================================================
def _init_pilot():
    """Pilot-study logging, enabled ONLY by `python main.py --pilot <participant_id>`.
    Returns None (and does nothing at all) when the flag is absent."""
    if "--pilot" not in sys.argv:
        return None
    i = sys.argv.index("--pilot")
    if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("--"):
        sys.exit("[PILOT][ERROR] Usage: python main.py --pilot <participant_id>")
    from utils.pilot_logger import PilotSession
    base = os.path.dirname(os.path.abspath(__file__))
    return PilotSession(sys.argv[i + 1],
                        os.path.join(base, "pilot_items.csv"),
                        os.path.join(base, "pilot_logs"))


def run() -> None:
    pilot = _init_pilot()   # None unless --pilot <id> was given
    import tensorflow as tf

    model, label_map, feature_model, prototypes = load_model_and_labels()
    arduino    = connect_arduino()
    
    # NEW ARCHITECTURE: Temporal buffer + stabilizer
    temporal_buf = TemporalBuffer(window=TEMPORAL_WINDOW)
    stabilizer   = TemporalStabilizer(
        persistence_frames=PERSISTENCE_FRAMES,
        confidence_threshold=TEMPORAL_CONFIDENCE,
        cooldown_frames=COOLDOWN_FRAMES,
    )
    
    detector = HandDetector(max_hands=1,
                            detection_confidence=DETECTION_CONFIDENCE,
                            tracking_confidence=TRACKING_CONFIDENCE)

    # Determine inference mode
    from sklearn.ensemble import RandomForestClassifier as _RFC
    geometric_mode = isinstance(prototypes, _RFC)
    landmark_mode  = (model is None) and not geometric_mode
    _infer         = None

    if not geometric_mode and not landmark_mode:
        import tensorflow as tf
        _sig = [tf.TensorSpec(shape=(1, 64, 64, 3), dtype=tf.float32)]
        if feature_model is not None and prototypes:
            @tf.function(input_signature=_sig)
            def _infer(x):
                return feature_model(x, training=False)
        else:
            @tf.function(input_signature=_sig)
            def _infer(x):
                return model(x, training=False)
        print("[INFO] Warming up inference graph...")
        _infer(tf.constant(np.zeros((1, 64, 64, 3), dtype=np.float32)))

    print("[INFO] Ready.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam.")
    
    # OPTIMIZE FOR FPS: 640x480, max_num_hands=1 already set
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    motion_tracker  = MotionTracker(buffer_size=30)

    current_word    = ""
    flash_frames    = 0
    last_accepted   = ""
    show_debug      = False
    collection_mode = False
    hand_canvas     = None
    last_resolution = None   # most recent ResolutionResult — displayed until next resolution

    intent_engine = IntentEngine()
    word_event_gate = WordEventGate()   # STEP 8: prevents re-resolving an unchanged word
    engine_dashboard = EngineDashboard()   # Phase 7: technical observability window (toggle: D)

    stt_listener = SpeechListener()
    stt_listener.start()

    fps         = 0.0
    fps_timer   = time.time()
    frame_count = 0

    print("[INFO] Webcam running.  Show your hand anywhere in the frame.")
    print("[INFO] Keys: Enter=resolve intent | Q=quit&send | C=clear | Backspace=delete | "
          "M=debug | D=engine dashboard | S=collect\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Dropped frame.")
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 0.5:
            fps         = frame_count / (now - fps_timer)
            fps_timer   = now
            frame_count = 0

        # Keep a clean copy for cropping
        frame_clean = frame.copy()

        # ── MediaPipe hand detection ──────────────────────────────────
        hands, frame = detector.find_hands(frame, draw=True)

        raw_letter       = ""
        weighted_conf    = 0.0
        persistence      = 0
        hand_detected    = bool(hands)
        geo_metrics      = {}

        if hands:
            hand        = hands[0]
            bbox        = hand['bbox']
            hand_canvas, img_input = mediapipe_hand_crop(frame_clean, bbox)
            motion_tracker.update(hand['lmRaw'])

            if geometric_mode:
                # ── RandomForest on geometric features ──────────────────────────────
                geo_feat   = geo_extract(hand['lmRaw'])
                geo_metrics = geo_summary(geo_feat)
                
                # Direct RF prediction (NO apply_rules, NO geometric_bonus)
                proba      = prototypes.predict_proba([geo_feat])[0]
                idx        = int(np.argmax(proba))
                raw_letter = label_map.classes_[idx]
                raw_confidence = float(proba[idx])

            elif landmark_mode:
                # ── Landmark cosine-similarity ────────────────────────────
                from utils.preprocessing import normalize_landmarks
                feat_n     = normalize_landmarks(hand['lmRaw'])
                feat_n     = feat_n / (np.linalg.norm(feat_n) + 1e-8)
                scores     = {l: float(np.dot(feat_n, p)) for l, p in prototypes.items()}
                raw_letter = max(scores, key=scores.get)
                raw_confidence = float((scores[raw_letter] + 1.0) / 2.0)

            elif feature_model is not None and prototypes:
                # ── CNN prototype classifier ──────────────────────────────
                import tensorflow as tf
                feat       = _infer(tf.constant(img_input)).numpy()[0]
                feat_n     = feat / (np.linalg.norm(feat) + 1e-8)
                scores     = {l: float(np.dot(feat_n, p)) for l, p in prototypes.items()}
                raw_letter = max(scores, key=scores.get)
                raw_confidence = float((scores[raw_letter] + 1.0) / 2.0)

            else:
                # ── Raw softmax ───────────────────────────────────────────
                import tensorflow as tf
                preds      = _infer(tf.constant(img_input)).numpy()[0]
                idx        = int(np.argmax(preds))
                raw_confidence = float(preds[idx])
                raw_letter = label_map.get(idx, "?")
        else:
            hand_canvas = None
            geo_metrics = {}
            motion_tracker.clear()
            temporal_buf.clear()
            stabilizer.force_reset_last_accepted()  # allow same letter after hand removed
            raw_letter = ""
            raw_confidence = 0.0

        # ── TEMPORAL BUFFER: Weighted majority voting over 15 frames ──────────────
        temporal_buf.update(raw_letter, raw_confidence)
        voted_letter, weighted_conf, persistence = temporal_buf.vote()

        # Display the voted letter only if confident enough
        display_letter = voted_letter if weighted_conf >= TEMPORAL_CONFIDENCE else "?"

        # ── TEMPORAL STABILIZER: Gate based on persistence ────────────────────────
        # Only accept if:
        # - High weighted confidence
        # - Persistence (consecutive frames >= PERSISTENCE_FRAMES)
        new_letter = stabilizer.update(voted_letter, weighted_conf, persistence)

        if new_letter:
            current_word += new_letter
            motion_tracker.clear()
            temporal_buf.clear()
            last_accepted = new_letter
            flash_frames  = 12
            print(f"[WORD] Letter accepted: '{new_letter}'  ->  '{current_word}'")
            send_word(arduino, current_word)

        # ── Drawing ───────────────────────────────────────────────────
        if collection_mode:
            draw_collection_banner(frame)

        draw_fps(frame, fps, hand_detected)
        draw_prediction_panel(frame, display_letter, weighted_conf, persistence)
        draw_resolution_panel(frame, last_resolution, h)
        draw_word_bar(frame, current_word, h, collection_mode)

        if hand_detected and hand_canvas is not None:
            draw_hand_preview(frame, hand_canvas)

        if show_debug:
            if hand_detected and hand_canvas is not None:
                draw_debug_window(hand_canvas, geo_metrics)
            else:
                placeholder = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No hand detected", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imshow("AE-CP | Debug", placeholder)

        if engine_dashboard.visible:
            engine_dashboard.show()   # cheap redisplay of the cached canvas, no recompute

        if flash_frames > 0:
            draw_flash(frame, last_accepted)
            flash_frames -= 1

        # ── Combine frame + STT panel into one canvas ─────────────────
        canvas = np.zeros((h, w + STT_PANEL_W, 3), dtype=np.uint8)
        canvas[:h, :w] = frame
        draw_stt_panel(canvas, stt_listener.get_history(), stt_listener.is_listening)
        cv2.imshow("AE-CP | ASL Recognition", canvas)

        # ── Keyboard handling ─────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and current_word:  # ENTER → resolve intent
            if word_event_gate.should_process(current_word):
                last_resolution = intent_engine.resolve(
                    user_id=USER_ID,
                    recognized_word=current_word,
                )
                if pilot is not None:
                    pilot.log(current_word, last_resolution)
                print(f"[INTENT] '{current_word}' → [{last_resolution.status}] "
                      f"{last_resolution.selected_label or current_word} "
                      f"({last_resolution.final_score * 100:.1f}%)")
                print(f"[INTENT] Output: {last_resolution.output}")
                if last_resolution.conflict:
                    print("[INTENT] *** CONFLICTING EVIDENCE DETECTED ***")
                if last_resolution.status == "RESOLVED":
                    send_word(arduino, last_resolution.selected_lcd_text)
                if last_resolution.top_n:
                    for c in last_resolution.top_n:
                        print(f"         {c['label']}: {c['final_score'] * 100:.1f}%")
                # Phase 7: refresh the dashboard's cached content — ONLY here,
                # on a genuine resolution, never per-frame.
                engine_dashboard.update(intent_engine.last_trace,
                                         intent_engine.store.get_history(USER_ID))
            else:
                print(f"[INTENT] '{current_word}' already resolved — "
                      f"change the word before resolving again.")

        elif key in (ord("q"), ord("Q"), 27):
            print(f"\n[QUIT] Final word: '{current_word}'")
            send_word(arduino, current_word)
            break

        elif key in (ord("c"), ord("C")):
            current_word = ""
            stabilizer.reset()
            temporal_buf.clear()
            word_event_gate.reset()
            print("[CLEAR] Word cleared.")

        elif key == 8:
            if current_word:
                current_word = current_word[:-1]
                stabilizer.reset()
                temporal_buf.clear()
                print(f"[DELETE] Word: '{current_word}'")

        elif key in (ord("m"), ord("M")):
            show_debug = not show_debug
            if not show_debug:
                try:
                    cv2.destroyWindow("AE-CP | Debug")
                except cv2.error:
                    pass
            print(f"[DEBUG] Window {'ON' if show_debug else 'OFF'}")

        elif key in (ord("d"), ord("D")):
            is_visible = engine_dashboard.toggle()
            print(f"[DASHBOARD] Window {'ON' if is_visible else 'OFF'}")

        elif key in (ord("s"), ord("S")):
            collection_mode = not collection_mode
            if collection_mode:
                print("[COLLECT] Collection mode ON.  Show a sign then press its letter key.")
            else:
                print("[COLLECT] Collection mode OFF.")

        # Pilot mode only: N = next item, R = count a retry. Skipped while S
        # collection mode is on so N/R can still be saved as letter samples.
        elif pilot is not None and not collection_mode and key in (ord("n"), ord("N")):
            pilot.next_item()

        elif pilot is not None and not collection_mode and key in (ord("r"), ord("R")):
            pilot.retry()

        elif collection_mode and hand_canvas is not None:
            if ord("a") <= key <= ord("z"):
                letter_key = chr(key).upper()
                path = save_sample(hand_canvas, letter_key)
                print(f"[COLLECT] Saved '{letter_key}' -> {path}")
            elif ord("A") <= key <= ord("Z"):
                letter_key = chr(key)
                path = save_sample(hand_canvas, letter_key)
                print(f"[COLLECT] Saved '{letter_key}' -> {path}")

    cap.release()
    detector.close()
    stt_listener.stop()
    intent_engine.close()
    if pilot is not None:
        pilot.close()
    cv2.destroyAllWindows()
    if arduino and arduino.is_open:
        arduino.close()
        print("[SERIAL] Connection closed.")
    print("[INFO] Done.")


if __name__ == "__main__":
    run()
