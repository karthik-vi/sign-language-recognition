"""Run MediaPipe HolisticLandmarker over every video in the manifest -> data/landmarks/<video_id>.npz

  python scripts/extract_landmarks.py --raw data/raw --out data/landmarks --workers 4
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def job(args):
    from slr.extract import extract_video
    import numpy as np

    r, raw, out = args
    dest = out / f"{r['video_id']}.npz"
    if dest.exists():
        return r["video_id"], "cached"
    try:
        d = extract_video(raw / r["file"], r.get("frame_start", 1), r.get("frame_end", -1))
        np.savez_compressed(dest, **d)
        return r["video_id"], f"ok {len(d['frame_idx'])}"
    except Exception as e:
        return r["video_id"], f"error {e!r}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/landmarks"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (a.raw / "manifest.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if (a.raw / r["file"]).exists()]
    t0 = time.time()
    errs = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (vid, status) in enumerate(ex.map(job, [(r, a.raw, a.out) for r in rows], chunksize=4)):
            errs += status.startswith("error")
            if status.startswith("error") or i % 100 == 0:
                print(f"[{i + 1}/{len(rows)}] {vid} {status} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {len(rows)} videos, {errs} errors, {time.time() - t0:.0f}s")
