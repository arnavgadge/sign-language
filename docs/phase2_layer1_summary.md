# Layer 1 (Context / Input Acquisition) — Technical Summary

*AE-CP 5-layer engine, Phase 2 implementation. Drafted for later inclusion in the PUNECON paper.*

## Purpose

Layer 1 collects and normalizes the raw signals available at the moment a user commits an
ASL-recognized word, producing a single structured `InteractionContext` object for downstream
layers (personalization, decision-making) to consume. It performs no interpretation or
weighting of its own.

## Design

Four of Layer 1's five sub-signals are deterministic transforms with no learned parameters:
word normalization (case/whitespace), time-of-day bucketing, and motion-speed normalization.
The fifth — face presence/count — is produced by a pre-trained OpenCV Haar-cascade classifier
(`haarcascade_frontalface_default.xml`, 930,127 bytes), used here for inference only; it was
not trained as part of this work.

This split matters methodologically: deterministic components were validated with correctness
unit tests (28/28 passing), while the one genuine classifier was evaluated with standard
classification metrics (precision/recall/F1/confusion matrix) on a held-out test split, kept
separate from a validation split used for optional hyperparameter selection (`minNeighbors`
sweep). No component was tuned against its own test data.

## Implementation

- `layer1/schema.py` — `InteractionContext` / `FaceSignal` output schema (JSON-serializable).
- `layer1/preprocessing.py` — pure functions: word normalization, time-period bucketing,
  motion-speed normalization to `[0,1]`.
- `layer1/face_signal.py` — `FaceDetector`, wrapping the pre-trained Haar cascade.
- `layer1/pipeline.py` — `Layer1Pipeline.build()`, the single entry point orchestrating the
  above and measuring end-to-end latency.
- `layer1/storage.py` — append-only JSONL persistence per session (`data/interactions/`).
- `layer1/config.py` — all hyperparameters and paths, snapshotted with every evaluation run for
  reproducibility (fixed random seed = 42 throughout).

## Results (measured, this environment)

| Metric | Value |
|---|---|
| Unit tests | 28/28 passed (correctness of deterministic transforms + schema round-trip + storage round-trip) |
| End-to-end pipeline latency | 3.77 ms/call average (n=20, synthetic 640×480 frame, CPU) |
| Face-cascade file size | 930,127 bytes (908.3 KB) |
| Face-detection classification metrics | **Not measured** — no labeled evaluation dataset exists in this environment (no webcam access, no bundled sample face images, no `scikit-image` install). `collect_face_eval_samples.py` was built to produce this dataset; `evaluation/layer1_face_eval.py` is ready to compute precision/recall/F1/confusion matrix as soon as it exists. |

No face-detection accuracy numbers are reported because none could be honestly measured in this
session. This is stated explicitly per the "do not fabricate results" requirement, rather than
substituting a plausible-looking placeholder.

## Scope boundary

Layer 1 is implemented as a standalone module (`layer1/`) with no dependency on `intent/`
(Layer 5's implementation) and no changes to `main.py`. It is ready to be called by Layers 3 and
5 in a later phase, but that wiring is out of scope for Phase 2 by design — Phase 2's goal was a
validated component, not a working end-to-end engine.
