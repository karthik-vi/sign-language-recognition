# Architecture Decision Records

## Period freeze

| | |
|---|---|
| Work period reconstructed | Marvel Technology Solutions, AI/ML Engineer Intern, Jun 2026 – present |
| Cutoff | Nov 2026 |
| Python | 3.12 |
| Landmarks | MediaPipe 1.2.0 (PyPI upload 2026-10-27), Tasks API `HolisticLandmarker`; model `holistic_landmarker.task` float16 (GCS `latest`, Last-Modified 2023-12-21, sha256 prefix `e2dab61191e2dcd0`) |
| Video IO | opencv-python-headless 5.0.0.93 (2026-07-02) |
| Training | PyTorch 2.15.0 CPU (2026-10-28), NumPy 2.5.4 (2026-10-10), scikit-learn 1.9.1 (2026-09-10) |

All releases predate the Nov 2026 cutoff (checked 2026-11-03). Exceptions: none. Note: MediaPipe 1.x removed the legacy
`mp.solutions.holistic` API, so this repo uses the Tasks API; the system libraries `libegl1 libgles2` are required at runtime.

---

## ADR-001 Dataset: WLASL100 via the Voxel51 HF mirror — 2026-11-03

- **Context.** Need a public, video-based, isolated-sign dataset with a license that allows research use, downloadable without manual approval.
- **Options.** WLASL (C-UDA, URLs largely dead, HF mirror exists); MS-ASL (YouTube links only); Kaggle "ASL Signs" (landmarks only, needs Kaggle login, no video → no MediaPipe stage to reconstruct); ASL alphabet images (static, not video); LSA64 (Argentinian SL; license not confirmed during my search).
- **Decision.** WLASL100 with official splits, videos from the Voxel51 mirror, labels from the official WLASL json. 1,007 of 2,038 clips available.
- **Tradeoffs.** Half the official data → numbers not comparable to papers. Very few clips per class.
- **Would change my mind.** Access to the full WLASL video set, or a dataset with ≥20 clips per class and a clear license.

## ADR-002 Representation: MediaPipe Holistic landmarks, not raw pixels — 2026-11-03

- **Context.** CPU-only environment; ~1k training clips.
- **Options.** (a) RGB video models (I3D/VideoMAE) — GPU-bound, overfit at this size; (b) landmarks + light sequence model.
- **Decision.** (b). Pose (25 upper-body points) + both hands → 32-frame sequences → BiGRU with attention pooling (~0.5M params).
- **Tradeoffs.** Loses facial expression and fine handshape detail when hand tracking fails; gains speed (CPU training in minutes) and robustness to background/clothing.
- **Would change my mind.** If hand-tracking failure rates (see consistency report) were the dominant error source, a pixel-based hand crop model would be worth it.

## ADR-003 Data consistency as a first-class step — 2026-11-03

- **Context.** The resume story is that data-consistency work, not model changes, drove most of the gain.
- **Decision.** Two separate mechanisms, measured separately:
  1. *Repairs in the feature pipeline* (applied to every split): gap interpolation, median smoothing, idle-frame trimming, body-centred normalisation, dominant-hand mirroring.
  2. *Exclusion of flagged clips* — training split only. Val/test clips are never removed, because dropping hard test clips inflates accuracy.
- **Tradeoffs.** Exclusion shrinks an already tiny training set.

## ADR-004 Evaluation — 2026-11-03

- Top-1 and top-5 accuracy on the official WLASL test split (100 clips here); model selection on the official val split only.
- 3 training seeds per variant, report mean ± std. Ablation is cumulative so each row isolates one change.

## ADR-005 Demo / deployment — 2026-11-03

- `scripts/demo.py` runs on a video file headlessly (prints top-3 per window) or on a webcam with `--show` (needs GUI OpenCV).
- No server. Docker image runs the video demo / tests. A web service is out of scope for this reconstruction.
