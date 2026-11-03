"""Train the variant with the best mean VALIDATION top-1 from results/ablation.json and save it for the demo.

  python scripts/train_best.py --out artifacts/best
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import STEPS  # noqa: E402
from slr.train import TrainConfig, load_samples, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ablation", type=Path, default=Path("results/ablation.json"))
ap.add_argument("--raw", type=Path, default=Path("data/raw"))
ap.add_argument("--landmarks", type=Path, default=Path("data/landmarks"))
ap.add_argument("--flags", type=Path, default=Path("results/consistency.json"))
ap.add_argument("--out", type=Path, default=Path("artifacts/best"))
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
summ = json.loads(a.ablation.read_text())["summary"]
best = max(summ, key=lambda k: summ[k]["val_top1_mean"])  # selection on validation, never on test
key, desc, feat, tkw = next(s for s in STEPS if s[0] == best)
manifest = [json.loads(l) for l in (a.raw / "manifest.jsonl").read_text().splitlines() if l.strip()]
flags = json.loads(a.flags.read_text())["flags"] if a.flags.exists() else {}
res = run(manifest, load_samples(a.landmarks, manifest), feat, TrainConfig(seed=a.seed, **tkw), flags, out_dir=a.out)
print(best, desc, {"val": res["val"], "test": res["test"]})
