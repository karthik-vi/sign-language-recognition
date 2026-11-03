"""Video -> per-frame landmarks with MediaPipe Tasks HolisticLandmarker.

Output per video (.npz):
  pose  (T, 33, 4)  x, y, z, visibility   (NaN if no person)
  lhand (T, 21, 3)  x, y, z               (NaN if hand not detected)
  rhand (T, 21, 3)
  fps, n_frames_video, frame_idx (T,)
Coordinates are MediaPipe-normalised image coordinates (x, y in [0, 1]).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def make_landmarker(model_path: Path | None = None):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode

    opts = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path or MODEL_DIR / "holistic_landmarker.task")),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return HolisticLandmarker.create_from_options(opts), mp


def _arr(lms, n, dims):
    if not lms:
        return np.full((n, dims), np.nan, np.float32)
    out = np.array([[getattr(p, a) if getattr(p, a, None) is not None else np.nan
                     for a in ("x", "y", "z", "visibility")[:dims]] for p in lms], np.float32)
    return out


def extract_video(path: Path, frame_start: int = 1, frame_end: int = -1, max_side: int = 640) -> dict:
    """Run the landmarker on every frame in [frame_start, frame_end] (1-indexed, inclusive; -1 = last)."""
    landmarker, mp = make_landmarker()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose, lh, rh, idx = [], [], [], []
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i < frame_start or (frame_end > 0 and i > frame_end):
                continue
            h, w = frame.shape[:2]
            if max(h, w) > max_side:
                s = max_side / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = landmarker.detect_for_video(img, int(1000 * i / fps))
            pose.append(_arr(res.pose_landmarks, 33, 4))
            lh.append(_arr(res.left_hand_landmarks, 21, 3))
            rh.append(_arr(res.right_hand_landmarks, 21, 3))
            idx.append(i)
    finally:
        cap.release()
        landmarker.close()
    T = len(idx)
    return {
        "pose": np.stack(pose) if T else np.zeros((0, 33, 4), np.float32),
        "lhand": np.stack(lh) if T else np.zeros((0, 21, 3), np.float32),
        "rhand": np.stack(rh) if T else np.zeros((0, 21, 3), np.float32),
        "frame_idx": np.array(idx, np.int32),
        "fps": np.float32(fps),
        "n_frames_video": np.int32(n_total),
    }
