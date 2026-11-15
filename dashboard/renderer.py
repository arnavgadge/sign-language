"""
dashboard/renderer.py
-------------------------
Pure OpenCV drawing functions for the technical observability dashboard
(Phase 7). Every function here only READS fields already computed by the
engine (resolution.models.EngineTrace) — none of them perform any signal,
reliability, score, ranking, or conflict computation. Every nested object
in EngineTrace is a frozen dataclass, so there is nothing here even could
mutate to "recalculate" a result.

Uses the same technology already in the project (OpenCV drawing +
cv2.imshow) rather than introducing a new UI framework — see
dashboard/window.py for the window lifecycle this renders into.
"""
import time

import cv2
import numpy as np

from resolution.models import EngineTrace

# ── Layout ───────────────────────────────────────────────────────────────────
CANVAS_W = 1000
CANVAS_H = 1080
LEFT_X = 14
RIGHT_X = 512
COL_W = 480

# ── Colors (BGR) ─────────────────────────────────────────────────────────────
BG        = (18, 18, 24)
HEADER    = (255, 255, 255)
GREEN     = (0, 200, 0)
YELLOW    = (0, 210, 210)
ORANGE    = (0, 140, 255)
GRAY      = (170, 170, 170)
DIM       = (110, 110, 110)
CYAN      = (220, 200, 80)
ACCENT    = (255, 180, 90)
RULE      = (70, 70, 90)

_STATUS_COLOR = {"RESOLVED": GREEN, "AMBIGUOUS": YELLOW, "LITERAL": (160, 160, 160)}
_GATE_MODE_COLOR = {"resolved": GREEN, "ambiguous": YELLOW, "literal_fallback": (160, 160, 160)}


class _Writer:
    """Small cursor helper so panel functions don't hand-place pixel coordinates."""

    def __init__(self, canvas, x, y, line_height=18):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.line_height = line_height

    def line(self, text, color=GRAY, scale=0.42, thickness=1, dx=0):
        cv2.putText(self.canvas, str(text), (self.x + dx, self.y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        self.y += self.line_height

    def gap(self, px=6):
        self.y += px

    def rule(self, width):
        cv2.line(self.canvas, (self.x, self.y), (self.x + width, self.y), RULE, 1)
        self.y += 10


def _header(wr: _Writer, text: str, width: int) -> None:
    wr.line(text, HEADER, 0.55, 1)
    wr.rule(width)


# ── Section 1: Recognition Output ───────────────────────────────────────────

def draw_recognition_panel(canvas, x, y, w, trace: EngineTrace) -> int:
    wr = _Writer(canvas, x, y)
    _header(wr, "RECOGNITION OUTPUT", w)
    wr.line(trace.word, GREEN, 0.85, 2)
    wr.gap(4)
    ts = time.strftime("%H:%M:%S", time.localtime(trace.evaluated_at))
    wr.line(f"User ID: {trace.user_id}", GRAY)
    wr.line(f"Timestamp: {ts}", GRAY)
    wr.line(f"Conversation length (active window): {trace.temporal_result.window_word_count}", GRAY)
    return wr.y


# ── Section 2: Candidate Generation (Layer 1) ───────────────────────────────

def draw_candidates_panel(canvas, x, y, w, trace: EngineTrace) -> int:
    wr = _Writer(canvas, x, y)
    _header(wr, "INTENT CANDIDATES  (Layer 1)", w)
    if not trace.candidates:
        wr.line("(none)", DIM)
    for c in trace.candidates:
        wr.line(f"- {c.label}", (220, 220, 220))
    wr.gap(2)
    wr.line("No separate filtering stage exists in this pipeline -", DIM, 0.33)
    wr.line("this is Layer 1's full, final shortlist.", DIM, 0.33)
    return wr.y


# ── Section 3: Temporal Context (Layer 3) ───────────────────────────────────

def draw_temporal_panel(canvas, x, y, w, trace: EngineTrace) -> int:
    tr = trace.temporal_result
    wr = _Writer(canvas, x, y)
    _header(wr, "TEMPORAL CONTEXT  (Layer 3)", w)
    wr.line(f"Active window: {tr.window_seconds:.0f}s", GRAY)
    words = ", ".join(tr.window_words) if tr.window_words else "(empty)"
    wr.line(f"Recent words: {words}", GRAY)
    wr.gap(4)

    ranked = sorted(tr.candidates, key=lambda e: -e.signal)
    for ev in ranked[:4]:
        wr.line(ev.candidate, CYAN, 0.44, 1)
        wr.line(f"   Signal: {ev.signal:.4f}   Reliability: {ev.reliability:.4f}", (200, 200, 200), 0.37)
        rel = ev.evidence.get("relevant_words", [])
        if rel:
            wr.line(f"   Relevant: {rel}", (190, 190, 190), 0.33)
            wr.line(f"   seq={ev.evidence.get('sequence_strength', 0):.2f}  "
                    f"recency={ev.evidence.get('recency', 0):.2f}  "
                    f"freq={ev.evidence.get('frequency', 0):.2f}  "
                    f"gap={ev.evidence.get('time_gap', 0):.2f}", DIM, 0.32)
    if not ranked:
        wr.line("(no candidates)", DIM)
    return wr.y


# ── Section 4: Personal Context (Layer 5) ───────────────────────────────────

def draw_personal_panel(canvas, x, y, w, trace: EngineTrace) -> int:
    pr = trace.personal_result
    wr = _Writer(canvas, x, y)
    _header(wr, "PERSONAL CONTEXT  (Layer 5)", w)
    wr.line(f"User: {pr.user_id}   Profile observations: {pr.profile_total_observations}   "
            f"Distinct words: {pr.profile_distinct_words}", GRAY, 0.38)
    wr.gap(4)

    wr.line(f"'{trace.word}' history:", (220, 220, 220))
    ranked_by_count = sorted(pr.candidates, key=lambda e: -e.evidence.get("historical_count", 0))
    any_history = False
    for ev in ranked_by_count:
        count = ev.evidence.get("historical_count", 0)
        if count > 0:
            any_history = True
            wr.line(f"   {ev.candidate}: {count}", (200, 200, 200))
    if not any_history:
        wr.line("   (no prior history for this word)", DIM)

    wr.gap(4)
    for ev in sorted(pr.candidates, key=lambda e: -e.signal)[:3]:
        wr.line(f"{ev.candidate}   Signal={ev.signal:.4f}  Reliability={ev.reliability:.4f}", CYAN, 0.4)
    return wr.y


# ── Section 5: Reasoning Engine ─────────────────────────────────────────────

def draw_reasoning_panel(canvas, x, y, w, trace: EngineTrace) -> int:
    rr = trace.reasoning_result
    wr = _Writer(canvas, x, y)
    _header(wr, "RELIABILITY-WEIGHTED REASONING", w)
    wr.line(f"Strategy: {rr.strategy_name}   Sources: {', '.join(rr.available_sources) or '(none)'}", GRAY, 0.36)
    wr.gap(4)

    for c in rr.candidates[:4]:
        wr.line(c.candidate, ACCENT, 0.46, 1)
        for src_name, contrib in c.sources.items():
            wr.line(f"   {src_name.upper()}:  signal={contrib.signal:.4f}  "
                    f"reliability={contrib.reliability:.4f}  "
                    f"contribution={contrib.weighted_contribution:.4f}", (200, 200, 200), 0.35)
        color = GREEN if c.rank == 1 else GRAY
        wr.line(f"   Final Score: {c.final_score:.4f}", color, 0.42, 1)
        wr.gap(3)
    if len(rr.candidates) > 4:
        wr.line(f"... +{len(rr.candidates) - 4} more (see Ranking below)", DIM, 0.33)
    return wr.y


# ── Section 6: Ranking ───────────────────────────────────────────────────────

def draw_ranking_panel(canvas, x, y, w, trace: EngineTrace) -> int:
    rr = trace.reasoning_result
    wr = _Writer(canvas, x, y)
    _header(wr, "INTENT RANKING", w)
    for c in rr.candidates:
        color = GREEN if c.rank == 1 else (200, 200, 200)
        wr.line(f"{c.rank}. {c.label}      {c.final_score:.4f}", color)
    if not rr.candidates:
        wr.line("(no candidates)", DIM)
    return wr.y


# ── Section 7: Conflict Detection ───────────────────────────────────────────

def draw_conflict_panel(canvas, x, y, w, trace: EngineTrace) -> int:
    conflict = trace.reasoning_result.conflict
    wr = _Writer(canvas, x, y)
    _header(wr, "CONFLICT STATUS", w)
    if conflict.conflict_detected:
        wr.line(f"[!] Conflict detected  (score={conflict.conflict_score:.4f})", ORANGE, 0.46, 1)
        for d in conflict.disagreements:
            wr.line(f"   {d['source_a'].capitalize()} -> {d['candidate_a']}  (strength={d['strength_a']:.3f})",
                     (230, 210, 170), 0.36)
            wr.line(f"   {d['source_b'].capitalize()} -> {d['candidate_b']}  (strength={d['strength_b']:.3f})",
                     (230, 210, 170), 0.36)
    else:
        wr.line("[OK] No significant conflict", GREEN, 0.46, 1)
    return wr.y


# ── Section 8: Decision Gate ─────────────────────────────────────────────────

def draw_decision_panel(canvas, x, y, w, trace: EngineTrace) -> int:
    gd = trace.gate_decision
    wr = _Writer(canvas, x, y)
    _header(wr, "DECISION GATE", w)

    ranking = gd.evidence.get("full_ranking", [])
    top_score = ranking[0]["final_score"] if len(ranking) > 0 else 0.0
    second_score = ranking[1]["final_score"] if len(ranking) > 1 else 0.0
    thresholds = gd.evidence.get("thresholds", {})

    wr.line(f"Top Score: {top_score:.4f}", GRAY)
    wr.line(f"Second Score: {second_score:.4f}", GRAY)
    wr.line(f"Score Margin: {gd.margin:.4f}", GRAY)
    wr.line(f"Resolution Threshold: {thresholds.get('high_confidence_threshold', '?')}   "
            f"Min Margin: {thresholds.get('min_margin', '?')}", GRAY, 0.36)
    wr.gap(4)
    status_color = _GATE_MODE_COLOR.get(gd.mode, HEADER)
    wr.line(f"Status: {gd.mode.upper()}", status_color, 0.5, 2)
    wr.line(gd.reason[:95], DIM, 0.30)
    return wr.y


# ── Section 9: Final Communication Output ───────────────────────────────────

def draw_final_output_panel(canvas, x, y, w, trace: EngineTrace) -> int:
    res = trace.resolution
    wr = _Writer(canvas, x, y)
    _header(wr, "FINAL COMMUNICATION", w)

    status_color = _STATUS_COLOR.get(res.status, HEADER)
    if res.status == "RESOLVED":
        wr.line(f'"{res.output}"', GREEN, 0.58, 2)
    elif res.status == "AMBIGUOUS":
        wr.line(f'"{res.recognized_word}"', YELLOW, 0.58, 2)
        wr.gap(2)
        wr.line("Possible interpretations:", GRAY, 0.38)
        for c in res.top_n:
            wr.line(f"   - {c['label']}   ({c['final_score'] * 100:.0f}%)", (210, 210, 210), 0.38)
    else:
        wr.line(f'"{res.recognized_word}"', (200, 200, 200), 0.58, 2)

    wr.gap(4)
    wr.line(f"Status: {res.status}", status_color, 0.46, 1)
    wr.line(f"Confidence-like Score: {res.final_score:.4f}", GRAY, 0.4)
    return wr.y


# ── Resolution History ───────────────────────────────────────────────────────

def draw_history_panel(canvas, x, y, w, history_rows: list) -> int:
    wr = _Writer(canvas, x, y)
    _header(wr, "RESOLUTION HISTORY", w)
    wr.line(f"{'Time':<10}{'Word':<12}{'Result':<26}{'Score':<8}Status", GRAY, 0.36)
    if not history_rows:
        wr.line("(no resolution history yet)", DIM)
    for row in history_rows[:8]:
        t = time.strftime("%H:%M:%S", time.localtime(row["timestamp"]))
        label = row["selected_intent"] or row["output_text"] or row["recognized_word"]
        wr.line(f"{t:<10}{row['recognized_word']:<12}{str(label)[:24]:<26}"
                 f"{row['final_score']:<8.2f}{row['status']}", (210, 210, 210), 0.34)
    return wr.y


# ── Layer 2 / Layer 4 architecture indicator (no fake scores) ──────────────

def draw_architecture_indicator(canvas, x, y) -> int:
    wr = _Writer(canvas, x, y, line_height=16)
    wr.line("Layer 2 (Visual Context):  FUTURE  (not implemented, no data shown)", DIM, 0.34)
    wr.line("Layer 4 (Device Context):  FUTURE  (not implemented, no data shown)", DIM, 0.34)
    return wr.y


# ── Compositor ───────────────────────────────────────────────────────────────

def render_dashboard(trace: EngineTrace, history_rows: list) -> np.ndarray:
    """Build the complete dashboard canvas from ONE real EngineTrace and
    the real, persisted resolution history. No computation happens here —
    every value drawn is read directly from objects the engine already
    produced."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
    cv2.putText(canvas, "AE-CP INTENT ENGINE  -  TECHNICAL DASHBOARD", (LEFT_X, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, HEADER, 2, cv2.LINE_AA)
    cv2.line(canvas, (0, 36), (CANVAS_W, 36), RULE, 1)

    y = draw_recognition_panel(canvas, LEFT_X, 60, COL_W, trace)
    y = draw_candidates_panel(canvas, LEFT_X, y + 14, COL_W, trace)
    y = draw_temporal_panel(canvas, LEFT_X, y + 14, COL_W, trace)
    left_end = draw_personal_panel(canvas, LEFT_X, y + 14, COL_W, trace)

    y = draw_reasoning_panel(canvas, RIGHT_X, 60, COL_W, trace)
    y = draw_ranking_panel(canvas, RIGHT_X, y + 14, COL_W, trace)
    y = draw_conflict_panel(canvas, RIGHT_X, y + 14, COL_W, trace)
    y = draw_decision_panel(canvas, RIGHT_X, y + 14, COL_W, trace)
    right_end = draw_final_output_panel(canvas, RIGHT_X, y + 14, COL_W, trace)

    history_y = min(max(left_end, right_end) + 20, CANVAS_H - 220)
    y = draw_history_panel(canvas, LEFT_X, history_y, CANVAS_W - 2 * LEFT_X, history_rows)
    draw_architecture_indicator(canvas, LEFT_X, min(y + 10, CANVAS_H - 24))

    return canvas


def blank_placeholder() -> np.ndarray:
    """Shown before the first resolve() event has happened."""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
    cv2.putText(canvas, "AE-CP INTENT ENGINE  -  TECHNICAL DASHBOARD", (LEFT_X, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, HEADER, 2, cv2.LINE_AA)
    cv2.line(canvas, (0, 36), (CANVAS_W, 36), RULE, 1)
    cv2.putText(canvas, "Waiting for the first accepted word to be resolved...", (LEFT_X, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)
    return canvas
