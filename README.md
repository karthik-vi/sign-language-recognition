# sign-language-recognition

Video → MediaPipe landmarks → data-consistency checks → feature pipeline → BiGRU classifier, on public WLASL100 clips,
with an ablation that shows which preprocessing changes moved accuracy, and a video/webcam demo.

## What this is / is not

| | |
|---|---|
| **Reconstructs** | Resume (Marvel Technology Solutions, AI/ML Engineer Intern, Jun 2026 – Present), verbatim: *"Boosted sign-language recognition accuracy from 40% to 80% by building end-to-end video ML training pipelines with Python and MediaPipe, improving feature extraction and data consistency."* |
| **Is** | A from-scratch portfolio reconstruction on the public **WLASL100** dataset (C-UDA, non-commercial; videos downloaded by script, not committed). All numbers measured here (`docs/EXPERIMENTS.md`). |
| **Is not** | Marvel's code, data, client, or model. The 40% → 80% figures are not reproduced (see Results). |

### Period freeze
Marvel, Jun 2026 – Present, **cutoff Nov 2026**. Python 3.12; MediaPipe 1.2.0 (2026-10-27) Tasks `HolisticLandmarker` (float16 model, sha256 `e2dab611…`);
PyTorch 2.15.0 CPU; OpenCV 5.0.0.93; NumPy 2.5.4; scikit-learn 1.9.1. All released before the cutoff. Details: `docs/DECISIONS.md`.

## Problem
Isolated-sign recognition from ordinary video: few examples per sign, signers with different handedness, positions and camera framing,
and hand tracking that drops out on fast motion.

## Architecture
```
mp4 ─► MediaPipe HolisticLandmarker (VIDEO mode, ≤640 px) ─► per-frame pose(33) + left/right hand(21) landmarks, NaN when missing
   ─► consistency checks (labels, splits, frame counts, hand dropout, jitter, wrist jumps) ─► flags (train-only exclusion)
   ─► features: body-centred normalisation │ hand-local shape │ gap interpolation │ median smoothing │ idle trim │
                dominant-hand mirroring │ (velocity) ─► resample to 32 frames
   ─► BiGRU(2×128) + attention pooling ─► 100-way softmax
demo: webcam/video ─► sliding 64-frame window ─► top-3 glosses every N frames
```

## Quickstart
```bash
sudo apt-get install -y libegl1 libgles2
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 && pip install -r requirements.txt
pytest -q                                              # runs without the dataset
bash scripts/download_models.sh
python scripts/download_wlasl.py --out data/raw        # ~480 MB
python scripts/extract_landmarks.py --workers 4        # ~12 min on 4 vCPU
python scripts/check_consistency.py
python scripts/run_experiments.py --seeds 0,1,2        # ~1.5 h on 4 vCPU
python scripts/train_best.py && python scripts/demo.py --video data/raw/videos/<id>.mp4
```

## Results (WLASL100 official test split, 100 clips, mean ± std over 3 seeds)

| | Test top-1 | Test top-5 |
|---|---|---|
| V0 baseline (raw coordinates) | 0.453 ± 0.031 | 0.793 |
| V4 + normalisation, hand-local coords, interpolation/smoothing, trimming/mirroring | 0.673 ± 0.021 | 0.860 |
| **V6 selected on validation** (+ velocity + augmentation) | **0.660 ± 0.014** | 0.867 |

> Resume claim: 40% → 80% accuracy | Reproduced here on WLASL100 (1,007 available clips, 100-clip test set, 3 seeds): 45.3% → 66.0% top-1 | Gap / why: original data/classes/metric unavailable; ~7 training clips per gloss and half of WLASL100 missing from the public mirror. The source of the gain (feature extraction + data consistency, with the model unchanged) does reproduce.

Full ablation and consistency findings: `docs/EXPERIMENTS.md`.

## Limitations
- Not signer-independent (31/34 test signers seen in training). Not comparable to published WLASL100 numbers (half the data).
- 100 test clips: ±3 points is noise. Val and test rank variants differently.
- Webcam path (`--show`) implemented but not tested here (no camera/display); video-file demo tested.

## How to extend
- Signer-independent split; full WLASL100 if more clips can be obtained legally.
- Hand-crop pixel features (CNN) for handshape detail lost when landmarks fail; a temporal-conv or transformer encoder.
- Use the consistency flags to target re-annotation/re-recording rather than dropping clips.
