"""
utils/geometric_features.py
----------------------------
Geometric + topological feature extraction from MediaPipe landmarks.
Letter-specific rule overrides for hard signs.
"""

import numpy as np

# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP,  THUMB_TIP  = 1,  2,  3,  4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9,  10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
PIPS = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _angle(a, b, c) -> float:
    v1 = a - b;  v2 = c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _dist(a, b) -> float:
    return float(np.linalg.norm(a - b))


def _canonical_rotate(pts: np.ndarray) -> np.ndarray:
    """
    Rotate landmarks into a canonical orientation so the feature vector
    is invariant to wrist rotation.

    Reference vector: wrist → middle-MCP (stable palm direction).
    Target direction: straight up = (0, -1) in image coordinates.

    After this transform, the same ASL sign held at any wrist angle
    produces nearly identical coordinates.
    """
    ref  = pts[MIDDLE_MCP]           # wrist already at origin
    norm = np.linalg.norm(ref)
    if norm < 1e-6:
        return pts                   # degenerate — return unchanged

    # Angle from +x axis to the reference vector
    theta = np.arctan2(ref[1], ref[0])
    # Rotation needed to align ref with "up" = (0, -1) = -π/2 from +x
    alpha = (-np.pi / 2.0) - theta

    cos_a = float(np.cos(alpha))
    sin_a = float(np.sin(alpha))
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float32)

    return pts @ R.T                 # shape (21, 2) rotated


# ── Feature extraction ────────────────────────────────────────────────────────

def extract(landmarks) -> np.ndarray:
    """
    Full invariant feature pipeline:
      1. Translation normalisation  — wrist at origin
      2. Scale normalisation        — divide by wrist→middle-tip distance
      3. Rotation normalisation     — align palm direction to canonical "up"
      4. Extract distances, angles, states (all invariant by construction)

    Result: same ASL sign ≈ same feature vector regardless of position,
    distance, or wrist rotation.
    """
    if hasattr(landmarks[0], 'x'):
        pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    else:
        pts = np.asarray(landmarks, dtype=np.float32)

    # 1. Translation
    pts = pts - pts[WRIST]

    # 2. Scale
    scale = np.linalg.norm(pts[MIDDLE_TIP]) + 1e-8
    pts  /= scale

    # 3. Rotation — canonical palm orientation
    pts = _canonical_rotate(pts)

    feats = []

    # Block 1: Rotation-invariant normalised coords (42)  indices 0-41
    feats.extend(pts.flatten())

    # Block 2: Extension states (5)    indices 42-46
    for tip, pip in zip(TIPS, PIPS):
        feats.append(float(np.linalg.norm(pts[tip]) > np.linalg.norm(pts[pip])))

    # Block 3: Curl angles (5)         indices 47-51
    for a, b, c in [
        (THUMB_MCP,  THUMB_IP,   THUMB_TIP),
        (INDEX_MCP,  INDEX_PIP,  INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP,   RING_PIP,   RING_TIP),
        (PINKY_MCP,  PINKY_PIP,  PINKY_TIP),
    ]:
        feats.append(_angle(pts[a], pts[b], pts[c]))

    # Block 4: Pairwise distances (12)  indices 52-63
    for i, j in [
        (THUMB_TIP,  INDEX_TIP),    # 52 thumb-index
        (INDEX_TIP,  MIDDLE_TIP),   # 53 *** U vs V ***
        (MIDDLE_TIP, RING_TIP),     # 54
        (RING_TIP,   PINKY_TIP),    # 55
        (INDEX_TIP,  RING_TIP),     # 56
        (THUMB_TIP,  PINKY_TIP),    # 57
        (THUMB_TIP,  MIDDLE_TIP),   # 58 thumb-middle
        (INDEX_TIP,  PINKY_TIP),    # 59
        (WRIST,      INDEX_TIP),    # 60
        (WRIST,      PINKY_TIP),    # 61
        (THUMB_TIP,  INDEX_MCP),    # 62 thumb tip vs index root
        (THUMB_TIP,  MIDDLE_MCP),   # 63 thumb tip vs middle root
    ]:
        feats.append(_dist(pts[i], pts[j]))

    # Block 5: Spread angles (4)       indices 64-67
    feats.append(_angle(pts[INDEX_TIP],  pts[INDEX_MCP],  pts[MIDDLE_TIP]))
    feats.append(_angle(pts[MIDDLE_TIP], pts[MIDDLE_MCP], pts[RING_TIP]))
    feats.append(_angle(pts[RING_TIP],   pts[RING_MCP],   pts[PINKY_TIP]))
    feats.append(_angle(pts[THUMB_TIP],  pts[THUMB_MCP],  pts[INDEX_TIP]))

    # Block 6: Thumb vs palm (4)       indices 68-71
    feats.append(pts[THUMB_TIP, 0] - pts[INDEX_MCP,  0])   # 68
    feats.append(pts[THUMB_TIP, 1] - pts[INDEX_MCP,  1])   # 69
    feats.append(pts[THUMB_TIP, 0] - pts[MIDDLE_MCP, 0])   # 70
    feats.append(pts[THUMB_TIP, 1] - pts[MIDDLE_MCP, 1])   # 71

    # Block 7: Palm shape (3)          indices 72-74
    palm = pts[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]]
    pw   = float(palm[:, 0].max() - palm[:, 0].min())
    ph   = float(palm[:, 1].max() - palm[:, 1].min())
    feats.extend([pw, ph, pw / (ph + 1e-8)])

    # Block 8: Fingertip heights vs palm centre (5)  indices 75-79
    palm_cy = pts[[INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], 1].mean()
    for tip in TIPS:
        feats.append(pts[tip, 1] - palm_cy)

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK 9: EXPLICIT FINGER STATE VECTOR (4)  indices 80-83
    # ─────────────────────────────────────────────────────────────────────────
    # Symbolic binary states based on joint ordering (y-coordinates).
    # A finger is EXTENDED if: tip_y < pip_y < mcp_y (upward pointing).
    # This directly captures finger structure for D [1,0,0,0] and E [0,0,0,0].
    
    # Index: tip < pip < mcp means extended upward
    idx_extended = float(
        (pts[INDEX_TIP, 1] < pts[INDEX_PIP, 1]) and 
        (pts[INDEX_PIP, 1] < pts[INDEX_MCP, 1])
    )
    feats.append(idx_extended)  # 80
    
    # Middle: tip < pip < mcp means extended upward
    mid_extended = float(
        (pts[MIDDLE_TIP, 1] < pts[MIDDLE_PIP, 1]) and 
        (pts[MIDDLE_PIP, 1] < pts[MIDDLE_MCP, 1])
    )
    feats.append(mid_extended)  # 81
    
    # Ring: tip < pip < mcp means extended upward
    ring_extended = float(
        (pts[RING_TIP, 1] < pts[RING_PIP, 1]) and 
        (pts[RING_PIP, 1] < pts[RING_MCP, 1])
    )
    feats.append(ring_extended)  # 82
    
    # Pinky: tip < pip < mcp means extended upward
    pinky_extended = float(
        (pts[PINKY_TIP, 1] < pts[PINKY_PIP, 1]) and 
        (pts[PINKY_PIP, 1] < pts[PINKY_MCP, 1])
    )
    feats.append(pinky_extended)  # 83

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK 10: THUMB STATE & VISIBILITY (3)  indices 84-86
    # ─────────────────────────────────────────────────────────────────────────
    # Thumb visibility: is thumb tip above index MCP (thumb folded inward)?
    thumb_visible = float(pts[THUMB_TIP, 1] < pts[INDEX_MCP, 1])
    feats.append(thumb_visible)  # 84
    
    # Thumb-to-index distance (normalized by palm size)
    thumb_index_dist = _dist(pts[THUMB_TIP], pts[INDEX_TIP])
    feats.append(thumb_index_dist)  # 85
    
    # Thumb fold state: is thumb curled toward palm (angle check)?
    thumb_curl = _angle(pts[THUMB_MCP], pts[THUMB_IP], pts[THUMB_TIP])
    feats.append(thumb_curl)  # 86

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCK 11: COMPACTNESS & OPENNESS METRICS (3)  indices 87-89
    # ─────────────────────────────────────────────────────────────────────────
    # Compactness: mean distance of fingertips from palm center
    fingertip_heights = [
        abs(pts[INDEX_TIP, 1] - palm_cy),
        abs(pts[MIDDLE_TIP, 1] - palm_cy),
        abs(pts[RING_TIP, 1] - palm_cy),
        abs(pts[PINKY_TIP, 1] - palm_cy),
    ]
    mean_fingertip_height = float(np.mean(fingertip_heights))
    compactness = float(1.0 - np.clip(mean_fingertip_height / 0.50, 0.0, 1.0))
    feats.append(compactness)  # 87
    
    # Openness: are fingers spread apart? Use average distance between adjacent tips
    finger_tip_spreads = [
        _dist(pts[INDEX_TIP], pts[MIDDLE_TIP]),
        _dist(pts[MIDDLE_TIP], pts[RING_TIP]),
        _dist(pts[RING_TIP], pts[PINKY_TIP]),
    ]
    avg_spread = float(np.mean(finger_tip_spreads))
    palm_openness = float(np.clip(avg_spread / 0.25, 0.0, 1.0))
    feats.append(palm_openness)  # 88
    
    # Overall fist state: number of non-thumb fingers extended (0-4)
    num_extended = float(idx_extended + mid_extended + ring_extended + pinky_extended)
    feats.append(num_extended / 4.0)  # 89 (normalized 0-1)

    return np.array(feats, dtype=np.float32)


FEATURE_DIM = 90

# ── Feature index constants ───────────────────────────────────────────────────
# Block 2 — extension
EXT_THUMB,  EXT_INDEX, EXT_MID, EXT_RING, EXT_PINKY = 42, 43, 44, 45, 46
# Block 3 — curl
CURL_THUMB, CURL_IDX,  CURL_MID, CURL_RING, CURL_PINKY = 47, 48, 49, 50, 51
# Block 4 — distances
DIST_THUMB_IDX  = 52   # thumb tip  ↔ index tip
DIST_IDX_MID    = 53   # index tip  ↔ middle tip  (U vs V)
DIST_MID_RING   = 54
DIST_RING_PINKY = 55
DIST_IDX_RING   = 56
DIST_THUMB_PINKY= 57
DIST_THUMB_MID  = 58
DIST_IDX_PINKY  = 59
DIST_W_IDX      = 60   # wrist ↔ index tip
DIST_W_PINKY    = 61   # wrist ↔ pinky tip
DIST_THUMB_IMCP = 62   # thumb tip ↔ index MCP (for K/T)
DIST_THUMB_MMCP = 63   # thumb tip ↔ middle MCP
# Block 5 — spread angles
ANG_IDX_MID  = 64
ANG_MID_RING = 65
ANG_RING_PINKY=66
ANG_THUMB_IDX= 67
# Block 6 — thumb vs palm
THUMB_X_IMCP = 68   # thumb.x - index_MCP.x
THUMB_Y_IMCP = 69   # thumb.y - index_MCP.y
THUMB_X_MMCP = 70
THUMB_Y_MMCP = 71
# Block 7 — palm
PALM_W, PALM_H, PALM_RATIO = 72, 73, 74
# Block 8 — heights (neg = above palm)
H_THUMB, H_INDEX, H_MID, H_RING, H_PINKY = 75, 76, 77, 78, 79
# Block 9 — EXPLICIT FINGER STATE VECTOR (symbolic [0,0,0,0] to [1,1,1,1])
STATE_INDEX, STATE_MID, STATE_RING, STATE_PINKY = 80, 81, 82, 83
# Block 10 — thumb state & visibility
THUMB_VISIBLE, THUMB_IDX_DIST, THUMB_CURL = 84, 85, 86
# Block 11 — compactness & openness
COMPACTNESS, PALM_OPENNESS, FIST_STATE = 87, 88, 89


# ── Letter-specific geometric rules ──────────────────────────────────────────

# Raw landmark coordinate offsets inside Block 1 (pts flattened, shape 21×2)
# INDEX_TIP = lm 8 → y at index 8*2+1 = 17
# WRIST     = lm 0 → y at index 1
_LM_IDX_TIP_Y  = 17   # normalised index-tip y (+ = below wrist = pointing down)
_LM_PINKY_TIP_Y= 41   # normalised pinky-tip y
_LM_THUMB_TIP_X= 8    # normalised thumb-tip x
_LM_THUMB_TIP_Y= 9    # normalised thumb-tip y


def c_score(f: np.ndarray) -> float:
    """
    0-1 confidence score for letter C using exactly 3 stable features.

    thumb_score : how open the C curve is (large thumb-index gap)
    open_score  : how open the palm is (fingers not fisted)
    curl_score  : how curved the fingers are (ideal ≈ 2.2 rad)

    All distances already palm-size-normalised by extract().
    """
    d_TI = f[DIST_THUMB_IDX]

    # 1. Thumb-index gap — smooth ramp from 0 at 0.20 to 1 at 0.50
    thumb_score = float(np.clip((d_TI - 0.20) / 0.30, 0.0, 1.0))

    # 2. Palm openness — all 4 fingers at least partially open (> 1.6 rad)
    min_curl    = float(min(f[CURL_IDX + i] for i in range(4)))
    open_score  = float(np.clip((min_curl - 1.6) / 0.60, 0.0, 1.0))

    # 3. Average curl near the C ideal of 2.2 rad
    avg_curl    = float(np.mean([f[CURL_IDX + i] for i in range(4)]))
    curl_score  = float(max(0.0, 1.0 - abs(avg_curl - 2.2) / 0.70))

    return thumb_score * 0.40 + open_score * 0.40 + curl_score * 0.20


def d_score(f: np.ndarray) -> dict:
    """
    Lightweight geometric scoring for letter D.
    
    D is characterized by EXPLICIT FINGER STRUCTURE:
      - Index EXTENDED (state = 1)
      - Middle CURLED (state = 0)
      - Ring CURLED (state = 0)
      - Pinky CURLED (state = 0)
      - Thumb folded inward
    
    Expected finger state vector: [1, 0, 0, 0]
    
    Returns
    -------
    dict with keys: 'score', 'index_extension', 'curled_fingers', 'dominance', 'finger_match'
    """
    # CRITICAL: Match finger state vector [1, 0, 0, 0]
    finger_state_match = (
        f[STATE_INDEX] > 0.5 and
        f[STATE_MID] < 0.5 and
        f[STATE_RING] < 0.5 and
        f[STATE_PINKY] < 0.5
    )
    finger_match_score = float(finger_state_match)
    
    # 1. Index extension: direct from explicit state
    index_ext = f[STATE_INDEX]
    
    # 2. Other fingers curled: direct from explicit states (all should be ~0)
    avg_others = float(np.mean([f[STATE_MID], f[STATE_RING], f[STATE_PINKY]]))
    curled_fingers = float(1.0 - avg_others)  # invert: 1 when all curled
    
    # 3. Single-finger dominance: index tip well above middle
    idx_y = f[17]    # index tip canonical y
    mid_y = f[25]    # middle tip canonical y  
    dominance = float(np.clip((mid_y - idx_y) / 0.35, 0.0, 1.0))
    
    # 4. Thumb should be folded inward
    thumb_inward = 1.0 - f[THUMB_VISIBLE]  # visible < 0.5 is good
    
    # Combined score: heavily weight finger state match
    final_score = (
        finger_match_score * 0.50 +
        index_ext * 0.25 +
        curled_fingers * 0.15 +
        dominance * 0.10
    )
    
    return {
        'score': float(np.clip(final_score, 0.0, 1.0)),
        'finger_match': finger_match_score,
        'index_extension': index_ext,
        'curled_fingers': curled_fingers,
        'dominance': dominance
    }


def e_score(f: np.ndarray) -> dict:
    """
    Lightweight geometric scoring for letter E.
    
    E is characterized by:
      - Compact hand (fingertips close to palm center)
      - Low openness (all fingers folded, NOT open)
      - High average curl (all fingers bent inward ~1.7–2.1 rad)
      - Dense closed-hand geometry
    
    Returns
    -------
    dict with keys: 'score', 'compactness', 'low_openness', 'curl_average'
    """
    # CRITICAL: Match finger state vector [0, 0, 0, 0] — ALL CURLED
    finger_state_match = (
        f[STATE_INDEX] < 0.5 and
        f[STATE_MID] < 0.5 and
        f[STATE_RING] < 0.5 and
        f[STATE_PINKY] < 0.5
    )
    finger_match_score = float(finger_state_match)
    
    # 1. Fingertip-to-palm compactness (from Block 11)
    compactness = f[COMPACTNESS]
    
    # 2. Low openness: hand is NOT open (all fingers bent)
    #    Use inverse of palm_openness: high compactness = low spread
    low_openness = 1.0 - f[PALM_OPENNESS]
    
    # 3. Average finger curl: all 4 fingers should curl similarly inward
    avg_curl = float(np.mean([f[CURL_IDX + i] for i in range(4)]))
    # Gaussian-like: reward ~1.9, penalize extremes
    curl_diff = abs(avg_curl - 1.90)
    curl_average = float(np.clip(1.0 - curl_diff / 0.80, 0.0, 1.0))
    
    # 4. Thumb folded inward (not visible = good for E)
    thumb_folded = 1.0 - f[THUMB_VISIBLE]
    
    # Combined score: heavily weight finger state match
    final_score = (
        finger_match_score * 0.55 +
        compactness * 0.25 +
        low_openness * 0.15 +
        curl_average * 0.05
    )
    
    return {
        'score': float(np.clip(final_score, 0.0, 1.0)),
        'finger_match': finger_match_score,
        'compactness': compactness,
        'low_openness': low_openness,
        'curl_average': curl_average
    }


def geometric_bonus(letter: str, f: np.ndarray) -> float:
    """
    DEPRECATED — No longer used in temporal architecture.
    
    The new system trusts RandomForest predictions directly with 
    temporal smoothing (TemporalBuffer + TemporalStabilizer).
    
    Returns 0.0 always.
    """
    return 0.0


def apply_rules(letter: str, f: np.ndarray) -> str:
    """
    DEPRECATED — No longer used in temporal architecture.
    
    The new system trusts RandomForest predictions directly with 
    temporal smoothing (TemporalBuffer + TemporalStabilizer).
    
    This was the giant per-letter decision tree that caused:
    - instability
    - overengineering
    - excessive hacks
    
    Returns the letter unchanged now.
    """
    return letter


# ── Debug summary ─────────────────────────────────────────────────────────────

def summary(f: np.ndarray) -> dict:
    """
    Return debug metrics for overlay display.
    Now includes EXPLICIT FINGER STATE VECTOR for D and E detection.
    """
    # Extension states from Block 2 (older feature)
    ext  = "".join(
        c for c, e in zip("TIMRP", f[EXT_THUMB:EXT_THUMB+5]) if e > 0.5
    ) or "none"
    
    # Fist check from curl angles
    fist = all(f[CURL_IDX + i] < 2.1 for i in range(4))
    
    cs = c_score(f)
    ds = d_score(f)
    es = e_score(f)
    
    # Build finger state string for display: [I, M, R, P] as 0/1
    finger_state_str = "".join([
        str(int(f[STATE_INDEX])),
        str(int(f[STATE_MID])),
        str(int(f[STATE_RING])),
        str(int(f[STATE_PINKY])),
    ])
    
    return {
        # ─── FINGER STATE VECTOR (Block 9) ───
        "FState"     : f"[{finger_state_str}]",  # e.g., "[1,0,0,0]" for D, "[0,0,0,0]" for E
        "Index"      : "EXT" if f[STATE_INDEX] > 0.5 else "CRL",
        "Middle"     : "EXT" if f[STATE_MID] > 0.5 else "CRL",
        "Ring"       : "EXT" if f[STATE_RING] > 0.5 else "CRL",
        "Pinky"      : "EXT" if f[STATE_PINKY] > 0.5 else "CRL",
        
        # ─── THUMB STATE ───
        "Thumb"      : "VIS" if f[THUMB_VISIBLE] > 0.5 else "HID",
        
        # ─── OLD EXTENSION STATES (for reference) ───
        "ext_block2" : ext,
        "fist"       : "Y" if fist else "N",
        
        # ─── C SCORE ───
        "C-score"    : f"{cs:.2f}",
        
        # ─── D SCORE (with finger match) ───
        "D-score"    : f"{ds['score']:.2f}",
        "D-fmatch"   : f"{ds['finger_match']:.2f}",
        "D-index"    : f"{ds['index_extension']:.2f}",
        "D-curled"   : f"{ds['curled_fingers']:.2f}",
        
        # ─── E SCORE (with finger match) ───
        "E-score"    : f"{es['score']:.2f}",
        "E-fmatch"   : f"{es['finger_match']:.2f}",
        "E-compact"  : f"{es['compactness']:.2f}",
        "E-openness" : f"{es['low_openness']:.2f}",
        
        # ─── GEOMETRY ───
        "compact"    : f"{f[COMPACTNESS]:.2f}",
        "spread"     : f"{f[PALM_OPENNESS]:.2f}",
        "TI-dist"    : f"{f[DIST_THUMB_IDX]:.2f}",
    }

