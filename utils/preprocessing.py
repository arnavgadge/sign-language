"""
utils/preprocessing.py
-----------------------
Frame pre-processing for ASL inference.

Two hand-isolation strategies are available and can be used together:

  BackgroundSubtractor  (recommended)
    Captures one background frame (hand absent), then detects the hand as
    whatever has changed.  Works for any skin tone and any background colour.
    Press 'B' in main.py to capture.

  Skin-colour masking   (fallback)
    YCrCb thresholding.  Less reliable but needs no background capture.

Both strategies share the same downstream pipeline:
  foreground mask
    → largest central contour
    → convex-hull fill
    → tight square crop + gray background
    → BGR→RGB → 64×64 → /255 → (1,64,64,3)
"""

import cv2
import numpy as np
import math

IMG_SIZE = 64    # must match training
_BG_GRAY = 128   # neutral fill (used by BackgroundSubtractor path)
# Kaggle ASL alphabet training images have a pure blue background (BGR 244,1,1).
# Matching it gives the CNN the domain it was trained on.
_BG_KAGGLE = np.array([244, 1, 1], dtype=np.uint8)

# YCrCb skin-colour range (fallback only)
_SKIN_MIN = np.array([0,   133,  77], dtype=np.uint8)
_SKIN_MAX = np.array([255, 173, 127], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  BackgroundSubtractor
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundSubtractor:
    """
    One-shot background model using per-pixel frame differencing.

    Usage
    -----
    bg = BackgroundSubtractor()

    # With hand OUT of the ROI:
    bg.capture(roi_bgr)

    # Every inference frame:
    mask = bg.get_foreground_mask(roi_bgr)   # uint8 single-channel
    isolated = isolate_with_mask(roi_bgr, mask)
    """

    def __init__(self, threshold: int = 28):
        """
        Parameters
        ----------
        threshold : pixel-difference value (0-255) below which a pixel is
                    considered background.  28 works well under typical
                    indoor lighting; raise it if noise is high.
        """
        self._bg        : np.ndarray | None = None  # float32 grayscale
        self.threshold  = threshold

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._bg is not None

    def capture(self, roi_bgr: np.ndarray) -> None:
        """
        Store the current ROI as the background reference.
        Call this when the hand is NOT inside the ROI box.
        """
        gray    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        # Slight blur to reduce sensor noise in the reference frame.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self._bg = blurred.astype(np.float32)

    def get_foreground_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        Return a binary uint8 mask: 255 = foreground (hand), 0 = background.
        Returns a blank mask if capture() has not been called yet.
        """
        if self._bg is None:
            return np.zeros(roi_bgr.shape[:2], dtype=np.uint8)

        gray    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        diff    = cv2.absdiff(blurred, self._bg)

        _, mask = cv2.threshold(diff.astype(np.uint8),
                                self.threshold, 255, cv2.THRESH_BINARY)

        # Close: fill holes inside the hand (gaps between fingers).
        k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_big, iterations=2)

        # Open: remove small noise specks.
        k_sml = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_sml, iterations=1)

        # Dilate: expand slightly to recapture hand edges.
        mask  = cv2.dilate(mask, k_sml, iterations=2)

        return mask


# ─────────────────────────────────────────────────────────────────────────────
#  Shared isolation core
# ─────────────────────────────────────────────────────────────────────────────

def isolate_with_mask(roi_bgr: np.ndarray,
                      fg_mask: np.ndarray) -> np.ndarray:
    """
    Given any binary foreground mask, find the hand, crop it tight, and
    composite it onto a neutral gray background.

    Contour selection
    -----------------
    - Discard contours < 5 % of ROI area (noise).
    - Among the rest, pick the one whose centroid is closest to the ROI centre
      so that arm fragments entering from the edge are ignored.

    Fill strategy
    -------------
    Convex hull of the selected contour is filled solid before compositing.
    This avoids patchy gaps from imperfect masks while preserving the true
    hand silhouette.

    Falls back to the original ROI when no valid contour is found.
    """
    h, w = roi_bgr.shape[:2]

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi_bgr

    min_area = 0.05 * h * w
    valid    = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return roi_bgr

    # Closest centroid to the ROI centre wins.
    cx_roi, cy_roi = w / 2.0, h / 2.0

    def _dist(c: np.ndarray) -> float:
        M = cv2.moments(c)
        if M["m00"] == 0:
            return float("inf")
        return ((M["m10"] / M["m00"] - cx_roi) ** 2 +
                (M["m01"] / M["m00"] - cy_roi) ** 2)

    hand = min(valid, key=_dist)

    # Convex hull fill
    hull      = cv2.convexHull(hand)
    fill_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(fill_mask, hull, 255)

    # Tight square crop, centred on the bounding box with 20 % padding
    bx, by, bw, bh = cv2.boundingRect(hand)
    side   = max(bw, bh)
    pad    = int(side * 0.20)
    cx, cy = bx + bw // 2, by + bh // 2
    half   = side // 2 + pad

    x1 = max(0, cx - half);  y1 = max(0, cy - half)
    x2 = min(w, cx + half);  y2 = min(h, cy + half)

    hand_crop = roi_bgr[y1:y2, x1:x2]
    crop_fill = fill_mask[y1:y2, x1:x2]

    output  = np.full_like(hand_crop, _BG_GRAY)
    mask_3c = cv2.merge([crop_fill, crop_fill, crop_fill])
    return np.where(mask_3c > 0, hand_crop, output)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_roi(roi_bgr   : np.ndarray,
                   img_size  : int          = IMG_SIZE,
                   use_skin_mask: bool      = False,
                   fg_mask   : np.ndarray | None = None) -> np.ndarray:
    """
    Prepare a BGR ROI for model inference.

    Isolation priority
    ------------------
    1. fg_mask provided (BackgroundSubtractor result)  → isolate_with_mask
    2. use_skin_mask=True                              → YCrCb skin masking
    3. Neither                                         → raw ROI crop

    Then: BGR→RGB → resize to img_size × img_size → /255 → (1, H, W, 3)
    """
    if fg_mask is not None:
        working = isolate_with_mask(roi_bgr, fg_mask)
    elif use_skin_mask:
        working = _isolate_hand_skin(roi_bgr)
    else:
        working = roi_bgr

    roi_rgb     = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (img_size, img_size),
                             interpolation=cv2.INTER_AREA)
    roi_norm    = roi_resized.astype(np.float32) / 255.0
    return np.expand_dims(roi_norm, axis=0)


def get_skin_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """Return the single-channel YCrCb skin mask (for debug display)."""
    return _build_skin_mask(roi_bgr)


def get_processed_preview(roi_bgr     : np.ndarray,
                          preview_size: int = 128,
                          use_skin_mask: bool = False,
                          fg_mask     : np.ndarray | None = None) -> np.ndarray:
    """
    BGR preview of what the model sees (before final resize to 64×64).
    Pass the same fg_mask used for inference so the preview is accurate.
    """
    if fg_mask is not None:
        working = isolate_with_mask(roi_bgr, fg_mask)
    elif use_skin_mask:
        working = _isolate_hand_skin(roi_bgr)
    else:
        working = roi_bgr

    return cv2.resize(working, (preview_size, preview_size),
                      interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────────────
#  MediaPipe / CVZone hand crop
# ─────────────────────────────────────────────────────────────────────────────

def mediapipe_hand_crop(frame_bgr: np.ndarray,
                        hand_bbox: tuple,
                        img_size: int = IMG_SIZE,
                        offset_frac: float = 0.20) -> tuple:
    """
    Crop the detected hand from a clean frame, centre it on a gray square,
    and return a BGR canvas plus a normalised model input tensor.

    Parameters
    ----------
    frame_bgr   : full BGR frame WITHOUT MediaPipe drawings (use frame.copy()
                  before calling detector.findHands)
    hand_bbox   : (x, y, w, h) from cvzone hand['bbox']
    img_size    : CNN input side length (default 64)
    offset_frac : padding around bbox as a fraction of the longer side

    Returns
    -------
    canvas      : (300, 300, 3) uint8 BGR — hand centred on gray bg (for preview)
    model_input : (1, img_size, img_size, 3) float32 normalised
    """
    CANVAS = 300
    x, y, w, h = hand_bbox
    offset = int(max(w, h) * offset_frac)

    fh, fw = frame_bgr.shape[:2]
    y1 = max(0, y - offset);  y2 = min(fh, y + h + offset)
    x1 = max(0, x - offset);  x2 = min(fw, x + w + offset)

    img_crop = frame_bgr[y1:y2, x1:x2]

    # CLAHE on the hand crop before placing on canvas — normalises brightness
    # so predictions stay consistent across different room lighting conditions.
    if img_crop.size > 0:
        lab  = cv2.cvtColor(img_crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_crop = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]),
                                cv2.COLOR_LAB2BGR)

    canvas = np.full((CANVAS, CANVAS, 3), _BG_KAGGLE, dtype=np.uint8)

    if img_crop.size > 0:
        ch, cw  = img_crop.shape[:2]
        aspect  = ch / (cw + 1e-8)
        if aspect > 1:
            new_h = CANVAS
            new_w = max(1, math.ceil(CANVAS / aspect))
            resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gap = (CANVAS - new_w) // 2
            canvas[:, gap:gap + new_w] = resized
        else:
            new_w = CANVAS
            new_h = max(1, math.ceil(CANVAS * aspect))
            resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            gap = (CANVAS - new_h) // 2
            canvas[gap:gap + new_h, :] = resized

    rgb         = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    small       = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    model_input = np.expand_dims(small.astype(np.float32) / 255.0, 0)

    return canvas, model_input


# ─────────────────────────────────────────────────────────────────────────────
#  Private — skin-colour fallback
# ─────────────────────────────────────────────────────────────────────────────

def _build_skin_mask(roi_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(ycrcb, _SKIN_MIN, _SKIN_MAX)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def _isolate_hand_skin(roi_bgr: np.ndarray) -> np.ndarray:
    """YCrCb skin mask → isolate_with_mask (fallback path)."""
    mask = _build_skin_mask(roi_bgr)
    return isolate_with_mask(roi_bgr, mask)
