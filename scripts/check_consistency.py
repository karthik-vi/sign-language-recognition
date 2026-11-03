"""Data-consistency report over manifest + extracted landmarks.

  python scripts/check_consistency.py --raw data/raw --landmarks data/landmarks --out results/consistency.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slr.consistency import check_dataset  # noqa: E402
from slr.train import load_samples  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", type=Path, default=Path("data/raw"))
ap.add_argument("--landmarks", type=Path, default=Path("data/landmarks"))
ap.add_argument("--out", type=Path, default=Path("results/consistency.json"))
a = ap.parse_args()
manifest = [json.loads(l) for l in (a.raw / "manifest.jsonl").read_text().splitlines() if l.strip()]
samples = load_samples(a.landmarks, manifest)
res = check_dataset(manifest, samples, vocab=sorted({r["gloss"] for r in manifest}))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, indent=2))
print(json.dumps(res["report"], indent=2))
by_split = {}
split_of = {r["video_id"]: r["split"] for r in manifest}
for vid in res["flags"]:
    by_split[split_of.get(vid, "?")] = by_split.get(split_of.get(vid, "?"), 0) + 1
print("flagged videos by split:", by_split)
