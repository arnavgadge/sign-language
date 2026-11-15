"""
generate_report.py
------------------
Generates the AE-CP technical documentation report as a Word .docx file.
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
section = doc.sections[0]
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)
section.left_margin   = Cm(3.0)
section.right_margin  = Cm(2.5)

# ── Styles helper ─────────────────────────────────────────────────────────────
def set_font(run, name="Calibri", size=11, bold=False, color=None):
    run.font.name  = name
    run.font.size  = Pt(size)
    run.font.bold  = bold
    if color:
        run.font.color.rgb = RGBColor(*color)

def heading(text, level=1, color=(0, 70, 127)):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = RGBColor(*color)
        run.font.name = "Calibri"
    return p

def body(text, bold=False, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Cm(1)
    run = p.add_run(text)
    set_font(run, bold=bold)
    return p

def bullet(text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(1 + level * 0.5)
    run = p.add_run(text)
    set_font(run, size=11)
    return p

def add_table(headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        set_font(run, bold=True, size=10)
        # shade header
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd  = OxmlElement("w:shd")
        shd.set(qn("w:val"),   "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"),  "1F618D")
        tcPr.append(shd)
        run.font.color.rgb = RGBColor(255, 255, 255)
    # Data rows
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        for ci, cell_text in enumerate(row_data):
            cell = row.cells[ci]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(cell_text))
            set_font(run, size=10)
            if ri % 2 == 1:
                tc   = cell._tc
                tcPr = tc.get_or_add_tcPr()
                shd  = OxmlElement("w:shd")
                shd.set(qn("w:val"),   "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"),  "D6EAF8")
                tcPr.append(shd)
    if col_widths:
        for ci, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[ci].width = Cm(w)
    return table


# ══════════════════════════════════════════════════════════════════════════════
#  TITLE PAGE
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()

title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title_p.add_run("AE-CP")
r.font.name  = "Calibri"
r.font.size  = Pt(36)
r.font.bold  = True
r.font.color.rgb = RGBColor(0, 70, 127)

sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub_p.add_run("ASL Real-Time Recognition System")
r.font.name  = "Calibri"
r.font.size  = Pt(18)
r.font.color.rgb = RGBColor(40, 40, 40)

doc.add_paragraph()

desc_p = doc.add_paragraph()
desc_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = desc_p.add_run("Technical Documentation Report")
r.font.name  = "Calibri"
r.font.size  = Pt(14)
r.font.italic = True
r.font.color.rgb = RGBColor(80, 80, 80)

doc.add_paragraph()
doc.add_paragraph()

date_p = doc.add_paragraph()
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = date_p.add_run(f"Date: {datetime.date.today().strftime('%B %d, %Y')}")
set_font(r, size=12)

doc.add_page_break()


# ══════════════════════════════════════════════════════════════════════════════
#  1. PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
heading("1. Project Overview", 1)
body(
    "AE-CP is a real-time American Sign Language (ASL) letter recognition system. "
    "It captures hand signs via a webcam, recognizes the signed letter using a machine "
    "learning model, and displays the resulting word on an Arduino-driven 16×2 LCD screen. "
    "The system supports the full ASL alphabet (A–Z) and builds words letter-by-letter in real time."
)

doc.add_paragraph()
heading("Objectives", 2)
bullet("Recognize ASL hand signs in real time using a standard webcam")
bullet("Achieve stable, noise-resistant letter detection without flickering")
bullet("Output the recognized word to a physical LCD display via Arduino")
bullet("Be lighting- and skin-tone-independent")


# ══════════════════════════════════════════════════════════════════════════════
#  2. SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("2. System Architecture", 1)
body(
    "The pipeline consists of six sequential stages, from raw webcam input to LCD output:"
)

stages = [
    ("Stage", "Component", "Description"),
]
stage_rows = [
    ("1", "Webcam Capture",           "640×480 @ 30 FPS via OpenCV"),
    ("2", "Hand Detection",           "MediaPipe HandLandmarker — detects 21 landmark points on the hand"),
    ("3", "Feature Extraction",       "~90 geometric features computed from landmark coordinates"),
    ("4", "RandomForest Classifier",  "Predicts letter A–Z from geometric features"),
    ("5", "Temporal Buffer",          "15-frame weighted vote — smooths out single-frame noise"),
    ("6", "Temporal Stabilizer",      "Requires 10 consecutive frames before accepting a letter"),
    ("7", "Arduino LCD Output",       "Accepted letter/word sent over Serial (COM5, 9600 baud)"),
]
add_table(["Stage", "Component", "Description"], stage_rows, col_widths=[1.5, 4.0, 8.5])

doc.add_paragraph()
body(
    "The key architectural decision was moving away from pixel-based CNN classification to "
    "geometry-based classification using hand landmark coordinates. This eliminates the "
    "domain gap between clean training images and noisy real-world webcam frames."
)


# ══════════════════════════════════════════════════════════════════════════════
#  3. COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("3. Core Components", 1)

heading("3.1 MediaPipe Hand Detection", 2)
body(
    "Google's MediaPipe HandLandmarker model detects the hand in each frame and returns "
    "21 landmarks — one per key joint (wrist, knuckles, fingertips). These landmarks are "
    "normalized 3D coordinates (x, y, z) relative to the wrist. Detection and tracking "
    "confidence thresholds are both set to 0.80."
)

heading("3.2 Geometric Feature Extraction", 2)
body(
    "Approximately 90 numeric features are computed from the 21 landmarks. These include:"
)
bullet("Finger extension state (is each finger straight or curled?)")
bullet("Per-finger curl angle (how bent is each finger?)")
bullet("Hand compactness (how tightly packed are the fingertips?)")
bullet("Relative distances between fingertip pairs")
bullet("Thumb direction and opposition angle")
body(
    "These features are invariant to lighting conditions, skin tone, background clutter, "
    "and camera angle — making the system robust in real-world environments."
)

heading("3.3 RandomForest Classifier", 2)
body(
    "A RandomForest classifier (300 trees, balanced class weights) is trained on geometric "
    "features extracted from webcam-collected hand images. At inference time, the classifier "
    "outputs a probability distribution over all 26 letters. The letter with the highest "
    "probability is taken as the raw prediction for that frame."
)

heading("3.4 Temporal Buffer", 2)
body(
    "A sliding window of the last 15 frames is maintained. A weighted majority vote is run "
    "across this window — more recent frames are weighted higher. This produces a "
    "weighted_confidence score and a voted_letter, smoothing out momentary prediction spikes."
)

heading("3.5 Temporal Stabilizer", 2)
body(
    "Before a letter is accepted and added to the word, it must:"
)
bullet("Remain the voted letter for at least 10 consecutive frames")
bullet("Maintain a weighted confidence of at least 65%")
bullet("Pass a cooldown gate (20 frames) to prevent the same letter being double-accepted")
body(
    "This design means the user must intentionally hold a sign for ~0.3 seconds before "
    "it registers — preventing accidental or transitional signs from being accepted."
)

heading("3.6 Arduino LCD Output", 2)
body(
    "An Arduino UNO receives the growing word over USB Serial (9600 baud, COM5). "
    "Every time a new letter is accepted, the current word is sent immediately. "
    "The 16×2 I2C LCD displays the last accepted letter centred on line 1 and the "
    "full word so far on line 2."
)


# ══════════════════════════════════════════════════════════════════════════════
#  4. MODEL EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("4. Model Evolution", 1)
body(
    "The project went through two distinct architectural phases:"
)

heading("Phase 1 — CNN on Kaggle Images (Abandoned)", 2)
body(
    "A lightweight CNN (~3.4M parameters) was trained on the Kaggle ASL Alphabet dataset "
    "(~90,000 images, 26 classes, 64×64 RGB). The architecture consisted of three "
    "convolutional blocks followed by a dense classifier head."
)
bullet("Training validation accuracy: ~94%")
bullet("Real-world webcam accuracy: ~50–60%")
bullet("Root cause: domain gap — Kaggle images are clean studio photos; webcam frames are noisy")
bullet("Mitigation attempted: YCrCb skin masking, hand centering, tighter ROI")
bullet("Outcome: insufficient improvement; architecture abandoned")

heading("Phase 2 — Geometric RandomForest (Current)", 2)
body(
    "The system was rebuilt around MediaPipe landmark coordinates and geometric features "
    "to eliminate the domain gap entirely. Since features are computed from joint positions "
    "rather than pixel values, the model is inherently environment-independent."
)
bullet("Training data: webcam-collected samples from dataset_custom/ (A, B, C, D)")
bullet("Classifier: RandomForest (300 trees)")
bullet("No domain gap — features are the same whether collected in a lab or at home")

add_table(
    ["Model", "Trained On", "Currently Active", "Notes"],
    [
        ("geometric_classifier.pkl", "dataset_custom/ (webcam)", "Yes — primary model", "Only A–D reliable currently"),
        ("model.h5",                 "90K Kaggle images",        "No — bypassed",       "CNN, good val acc, poor real-world"),
        ("landmark_prototypes.pkl",  "Webcam samples",           "No — bypassed",       "Cosine similarity fallback"),
    ],
    col_widths=[4.5, 4.5, 3.5, 4.5]
)


# ══════════════════════════════════════════════════════════════════════════════
#  5. FILE STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("5. Key Files", 1)

add_table(
    ["File", "Purpose"],
    [
        ("main.py",                           "Main real-time inference loop + HUD + serial output"),
        ("train_geometric_classifier.py",     "Trains the RandomForest on geometric features"),
        ("train_model.py",                    "Trains the (now unused) CNN on Kaggle images"),
        ("utils/hand_detector.py",            "MediaPipe HandLandmarker wrapper"),
        ("utils/geometric_features.py",       "Extracts ~90 geometric features from landmarks"),
        ("utils/temporal_buffer.py",          "15-frame weighted voting buffer"),
        ("utils/temporal_stabilizer.py",      "Persistence-based letter acceptance gate"),
        ("utils/preprocessing.py",            "Hand crop and landmark normalization"),
        ("model/geometric_classifier.pkl",    "Trained RandomForest classifier (active model)"),
        ("model/model.h5",                    "Trained CNN (inactive, bypassed)"),
        ("dataset_custom/",                   "Webcam-collected training samples (A, B, C, D)"),
        ("arduino/lcd_display.ino",           "Arduino sketch for I2C LCD display"),
    ],
    col_widths=[6.5, 10.5]
)


# ══════════════════════════════════════════════════════════════════════════════
#  6. HARDWARE
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("6. Hardware Setup", 1)

add_table(
    ["Component", "Specification"],
    [
        ("Computer",      "Windows 11, Python 3.10"),
        ("Webcam",        "Standard USB webcam, 640×480 @ 30 FPS"),
        ("Microcontroller", "Arduino UNO"),
        ("Display",       "16×2 LCD with I2C backpack (PCF8574, address 0x27)"),
        ("Connection",    "USB Serial — COM5, 9600 baud"),
        ("I2C Wiring",    "SDA → A4, SCL → A5, VCC → 5V, GND → GND"),
    ],
    col_widths=[5.0, 12.0]
)


# ══════════════════════════════════════════════════════════════════════════════
#  7. SOFTWARE STACK
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("7. Software Stack", 1)

add_table(
    ["Library", "Version", "Purpose"],
    [
        ("Python",        "3.10",   "Core language"),
        ("OpenCV",        "4.x",    "Webcam capture, image processing, HUD rendering"),
        ("MediaPipe",     "0.10+",  "Hand landmark detection (21 points)"),
        ("scikit-learn",  "1.x",    "RandomForest classifier"),
        ("TensorFlow/Keras", "2.x", "CNN model (inactive)"),
        ("NumPy",         "1.x",    "Feature computation"),
        ("pyserial",      "3.x",    "Arduino serial communication"),
        ("Arduino IDE",   "2.x",    "LCD sketch upload"),
        ("LiquidCrystal_I2C", "—",  "Arduino I2C LCD library"),
    ],
    col_widths=[4.5, 2.5, 10.0]
)


# ══════════════════════════════════════════════════════════════════════════════
#  8. HOW TO RUN
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("8. How to Run", 1)

heading("8.1 Upload Arduino Sketch", 2)
bullet("Open arduino/lcd_display.ino in Arduino IDE")
bullet("Install LiquidCrystal_I2C library (Library Manager → Frank de Brabander)")
bullet("Select Board: Arduino UNO, Port: COM5")
bullet("Click Upload")

heading("8.2 Run the Recognition System", 2)
body("From the project directory:")
code_p = doc.add_paragraph()
code_p.paragraph_format.left_indent = Cm(1.5)
r = code_p.add_run("venv310\\Scripts\\python main.py")
r.font.name = "Courier New"
r.font.size = Pt(10)
r.font.color.rgb = RGBColor(0, 100, 0)

heading("8.3 Keyboard Controls", 2)
add_table(
    ["Key", "Action"],
    [
        ("Q / ESC",     "Quit and send final word to Arduino"),
        ("C",           "Clear the current word"),
        ("Backspace",   "Delete last letter"),
        ("M",           "Toggle debug window (hand crop + geometric metrics)"),
        ("S",           "Toggle sample collection mode"),
        ("A–Z (in collection mode)", "Save current hand crop to dataset_custom/<letter>/"),
    ],
    col_widths=[5.0, 12.0]
)


# ══════════════════════════════════════════════════════════════════════════════
#  9. LIMITATIONS & FUTURE WORK
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("9. Limitations & Future Work", 1)

heading("Current Limitations", 2)
bullet("Only A, B, C, D are reliably recognized — only 4 letters have training data in dataset_custom/")
bullet("Dynamic signs (J, Z) require motion — not yet supported")
bullet("Single-hand only")

heading("To Improve", 2)
bullet("Set KAGGLE_PER_LETTER = 300 in train_geometric_classifier.py and retrain — adds diverse data for all 26 letters from the Kaggle dataset via MediaPipe landmark extraction")
bullet("Collect more webcam samples per letter using the S key for better personal accuracy")
bullet("Add word-level autocorrect or a simple language model for post-processing")


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════════════════════
out_path = "AE-CP_Technical_Report.docx"
doc.save(out_path)
print(f"[DONE] Report saved -> {out_path}")
