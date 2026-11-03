# Data provenance

## Dataset: WLASL100 (subset of WLASL v0.3)

| Item | Value |
|---|---|
| Name | WLASL — Word-Level American Sign Language, v0.3; subset **WLASL100** = the first 100 glosses of `WLASL_v0.3.json` (the subset definition used in the WLASL paper) |
| Paper | Li et al., "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison", WACV 2020 |
| Labels / splits / signer ids | `WLASL_v0.3.json` from https://github.com/dxli94/WLASL (`start_kit/`), fetched 2026-11-03 |
| Video files | Hugging Face mirror https://huggingface.co/datasets/Voxel51/WLASL (`data/data_*/<video_id>.mp4`), fetched 2026-11-03 |
| License | WLASL is for academic, computational, non-commercial use under the **C-UDA** (Computational Use of Data Agreement), as stated on the mirror's dataset card. Videos are **not** redistributed in this repo; `scripts/download_wlasl.py` fetches them. |
| How obtained | `python scripts/download_wlasl.py --out data/raw --glosses 100` |

### What was actually available

| | count |
|---|---|
| WLASL100 instances listed in `WLASL_v0.3.json` | 2,038 |
| … present in the HF mirror (the mirror holds 11,980 clips across all 2,000 glosses; I did not verify why the rest are absent) | 1,013 |
| … downloadable (6 files listed in the mirror metadata returned HTTP 404) | **1,007** |
| Split of the 1,007 (official WLASL splits) | train 744 / val 163 / test 100 |
| Glosses | 100 (all present in train; 2–12 train clips per gloss) |
| Signers | 64 distinct `signer_id`s |
| Video (from mirror metadata) | fps: 30 (743 clips), 24 (212), 25 (51), 50 (1). Resolution: 288×192 (224), 1920×1080 (169), 640×480 (134), 720×400 (113), 656×370 (103), 1280×720 (95), other. Length 19–195 frames, median 65. All 1,007 have `frame_start=1, frame_end=-1` (whole file is the sign). |
| Raw size | 478 MB mp4 |

**Consequence:** results here are on ~49% of the official WLASL100 data, with the official split assignment.
They are **not directly comparable** to published WLASL100 numbers, which used all 2,038 clips.

## Preprocessing

1. `scripts/extract_landmarks.py`: MediaPipe Tasks `HolisticLandmarker` (float16 model, VIDEO running mode) on every frame,
   frames downscaled so the long side ≤ 640 px. Saves pose (33×4), left hand (21×3), right hand (21×3) per frame as `.npz`;
   missing detections are NaN. Face landmarks are not stored.
2. `scripts/check_consistency.py`: label, frame and tracking checks → `results/consistency.json` (see README / EXPERIMENTS).
3. `src/slr/features.py`: baseline vs improved feature pipelines (normalisation, gap interpolation, smoothing, mirroring …), resample to 32 frames.

## Limitations

- Only ~5–7 training clips per gloss on average: a strong data-scarcity regime; variance between seeds is high.
- Test set is 100 clips → one clip = 1 percentage point. We report mean ± std over 3 training seeds.
- The mirror keeps only clips whose source links survived; that may bias toward particular sources/signers.
- MediaPipe hand tracking fails on fast motion / motion blur / low resolution; those frames become NaN (handled, not fixed).
