# Experiment log (newest first)

Data: WLASL100 clips available in the Voxel51 HF mirror (1,007 of 2,038; official splits: 744 train / 163 val / 100 test), see DATA.md.
Landmarks: MediaPipe 1.2.0 HolisticLandmarker. Model: BiGRU + attention pooling (539k params), 32 frames, AdamW 2e-3, OneCycle,
label smoothing 0.1, ≤120 epochs, early stopping on val top-1 (patience 30). Seeds 0, 1, 2. All runs 2026-11-03, CPU.
Raw results: `results/ablation.json`, `results/ablation_log.txt`, `results/consistency.json`.

## EXP-003 — Cumulative ablation: what moved accuracy (test = 100 clips, mean ± std over 3 seeds)

| Variant | Change (cumulative) | Test top-1 | Test top-5 | Val top-1 |
|---|---|---|---|---|
| V0 | baseline: raw image x,y, missing → 0, no augmentation | 0.453 ± 0.031 | 0.793 | 0.495 |
| V1 | + body-centred normalisation (shoulder midpoint, shoulder width) | 0.557 ± 0.005 | 0.807 | 0.515 |
| V2 | + hand-local shape coordinates (wrist-relative, hand-size scaled) | 0.617 ± 0.012 | 0.873 | 0.630 |
| V3 | + gap interpolation (≤8 frames) + 3-frame median smoothing | 0.643 ± 0.009 | 0.853 | 0.611 |
| V4 | + idle-frame trimming + dominant-hand mirroring | **0.673 ± 0.021** | 0.860 | 0.601 |
| V5 | + velocity features | 0.657 ± 0.029 | 0.850 | 0.607 |
| **V6** | + augmentation (affine ±8°, ±10% scale, temporal crop, 10% hand dropout) | 0.660 ± 0.014 | 0.867 | **0.667** |
| V7 | + drop 31 consistency-flagged TRAIN clips | 0.663 ± 0.005 | 0.870 | 0.638 |

- **Selected model: V6** (best validation top-1). Its test top-1 is 0.660. V4 has the best *test* mean, but choosing it would be test-set selection.
- **What moved accuracy:** feature representation and data-consistency repairs — normalisation (+10.4 pts), hand-local coordinates (+6.0),
  interpolation/smoothing (+2.6), trimming/mirroring (+3.0). Together V0 → V4: **+22 points**. The model never changed.
- **What did not:** velocity features (−1.6, within noise), augmentation (≈0 on test, +6.0 on val), removing flagged train clips (≈0).
- **Surprise:** val and test disagree by up to 7 points. With 100 test clips (1 clip = 1 point) and 3 seeds, differences < ~3 points are noise.

## EXP-002 — Data-consistency checks (`scripts/check_consistency.py`)

- Label checks: 100 glosses, no duplicate video ids, no video in two splits, every gloss present in train (2–12 clips each).
  Test covers only 72 of the 100 glosses, val 88.
- Signer overlap: 31 of 34 test signers also appear in train → this is **not** a signer-independent evaluation.
- Tracking (mean over clips): pose missing 0.3% of frames; **left hand missing 66.9%**, right hand missing 38.0%; **no hand at all in 34.3% of frames**
  (mostly idle frames before/after the sign). Wrist jitter p50 0.030, p90 0.075 shoulder-widths/frame².
- v1 thresholds (no hand > 90% of frames, > 3 wrist jumps) flagged **0 clips** — too loose for this data. v2 thresholds set from the observed
  distribution (no hand > 75%, ≥1 wrist jump > 1 shoulder width, jitter > 0.12) flag 46 clips: train 31, val 9, test 6.
  Only train clips are ever excluded (V7); val/test are never filtered.
- The 34% hand-free frames and the 67% left-hand dropout are what V3/V4 address (interpolation, trimming, mirroring).

## EXP-001 — Pipeline bring-up

- MediaPipe 1.x removed `mp.solutions`; switched to the Tasks `HolisticLandmarker`. Needed `libegl1` (crashed with `libEGL.so.1` missing).
- Extraction: 1,007 clips, 0 errors, 711 s wall for the final resumed run on 4 processes (the first run was killed by a 30-min job limit and resumed from cache).
- Smoke test initially failed (learned nothing in 25 epochs × 2 steps); the training loop was fine, the test had too few optimiser steps.

## Resume-claim comparison

> Resume claim: 40% → 80% accuracy | Reproduced here on WLASL100 (1,007 available clips, official splits, 100-clip test set, 3 seeds): 45.3% → 66.0% top-1 (V0 → V6, model chosen on validation); best test mean of any variant 67.3% | Gap / why: the original data, class set and metric are not available; this public set has only ~7 training clips per gloss and half of WLASL100 is missing from the mirror. The *direction and source* of the gain (feature extraction + data consistency, not model changes) reproduces; the 80% endpoint does not.
