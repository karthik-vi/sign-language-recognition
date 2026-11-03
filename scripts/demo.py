"""Live webcam or video-file demo: MediaPipe landmarks -> sliding window -> top-3 glosses.

  python scripts/demo.py --video data/raw/videos/00623.mp4            # prints predictions (headless OK)
  python scripts/demo.py --webcam 0 --show                             # needs a display + opencv-python (not headless)
  python scripts/demo.py --video in.mp4 --save out.mp4                 # annotated video
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slr.extract import _arr, make_landmarker  # noqa: E402
from slr.features import FeatureConfig, build, feature_dim  # noqa: E402
from slr.model import SignGRU  # noqa: E402

ap = argparse.ArgumentParser()
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument("--video", type=Path)
src.add_argument("--webcam", type=int)
ap.add_argument("--model", type=Path, default=Path("artifacts/best/model.pt"))
ap.add_argument("--window", type=int, default=64, help="frames in the sliding window")
ap.add_argument("--every", type=int, default=8, help="predict every N frames")
ap.add_argument("--show", action="store_true")
ap.add_argument("--save", type=Path)
a = ap.parse_args()

ck = torch.load(a.model, map_location="cpu", weights_only=False)
feat = FeatureConfig(**ck["feature_config"])
model = SignGRU(feature_dim(feat), len(ck["glosses"]), ck["hidden"], ck["dropout"])
model.load_state_dict(ck["state_dict"])
model.eval()

landmarker, mp = make_landmarker()
cap = cv2.VideoCapture(str(a.video) if a.video else a.webcam)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
buf = {k: deque(maxlen=a.window) for k in ("pose", "lhand", "rhand")}
writer, i, label = None, 0, ""
while True:
    ok, frame = cap.read()
    if not ok:
        break
    i += 1
    res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(1000 * i / fps))
    buf["pose"].append(_arr(res.pose_landmarks, 33, 4))
    buf["lhand"].append(_arr(res.left_hand_landmarks, 21, 3))
    buf["rhand"].append(_arr(res.right_hand_landmarks, 21, 3))
    if i % a.every == 0 and len(buf["pose"]) >= 16:
        x = build({k: np.stack(v) for k, v in buf.items()}, feat)
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(x)[None]), -1)[0]
        top = p.topk(3)
        label = " | ".join(f"{ck['glosses'][j]} {float(s):.2f}" for s, j in zip(top.values, top.indices))
        print(f"frame {i:5d}: {label}", flush=True)
    if a.show or a.save:
        cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if a.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(a.save), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if a.show:
            cv2.imshow("sign-language-recognition", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
cap.release()
landmarker.close()
if writer:
    writer.release()
