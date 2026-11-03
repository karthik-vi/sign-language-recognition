"""Cumulative ablation: baseline -> improved, several seeds each, official WLASL100 splits.

  python scripts/run_experiments.py --variants all --seeds 0,1,2 --out results/ablation.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slr.features import BASELINE  # noqa: E402
from slr.train import TrainConfig  # noqa: E402

B = BASELINE
STEPS = [
    ("V0", "baseline: raw image coords, NaN->0, no aug", B, dict(augment=False, exclude_flagged_train=False)),
    ("V1", "+ body-centred normalisation", replace(B, normalize=True), dict(augment=False, exclude_flagged_train=False)),
    ("V2", "+ hand-local shape coords", replace(B, normalize=True, hand_local=True), dict(augment=False, exclude_flagged_train=False)),
    ("V3", "+ gap interpolation + median smoothing", replace(B, normalize=True, hand_local=True, interpolate=True, smooth=True), dict(augment=False, exclude_flagged_train=False)),
    ("V4", "+ idle trim + dominant-hand mirroring", replace(B, normalize=True, hand_local=True, interpolate=True, smooth=True, trim_idle=True, dominant=True), dict(augment=False, exclude_flagged_train=False)),
    ("V5", "+ velocity features", replace(B, normalize=True, hand_local=True, interpolate=True, smooth=True, trim_idle=True, dominant=True, velocity=True), dict(augment=False, exclude_flagged_train=False)),
    ("V6", "+ augmentation (affine, temporal crop, hand dropout)", replace(B, normalize=True, hand_local=True, interpolate=True, smooth=True, trim_idle=True, dominant=True, velocity=True), dict(augment=True, exclude_flagged_train=False)),
    ("V7", "+ drop consistency-flagged TRAIN clips", replace(B, normalize=True, hand_local=True, interpolate=True, smooth=True, trim_idle=True, dominant=True, velocity=True), dict(augment=True, exclude_flagged_train=True)),
]


def _job(args):
    import torch

    torch.set_num_threads(1)
    from slr.train import load_samples, run

    key, feat, tkw, seed, raw, lm, flags_path = args
    manifest = [json.loads(l) for l in (raw / "manifest.jsonl").read_text().splitlines() if l.strip()]
    samples = load_samples(lm, manifest)
    flags = json.loads(flags_path.read_text())["flags"] if flags_path.exists() else {}
    res = run(manifest, samples, feat, TrainConfig(seed=seed, **tkw), flags)
    return key, seed, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--landmarks", type=Path, default=Path("data/landmarks"))
    ap.add_argument("--flags", type=Path, default=Path("results/consistency.json"))
    ap.add_argument("--variants", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/ablation.json"))
    a = ap.parse_args()
    steps = STEPS if a.variants == "all" else [s for s in STEPS if s[0] in a.variants.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = [(k, f, t, s, a.raw, a.landmarks, a.flags) for k, _, f, t in steps for s in seeds]
    results = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for key, seed, res in ex.map(_job, jobs):
            results.setdefault(key, {})[seed] = res
            print(f"{key} seed={seed} val={res['val']['top1']:.3f} test_top1={res['test']['top1']:.3f} "
                  f"test_top5={res['test']['top5']:.3f} ep={res['epochs_run']} {res['train_s']}s", flush=True)
    import numpy as np

    summary = {}
    for k, desc, _, _ in steps:
        r = results[k]
        t1 = [r[s]["test"]["top1"] for s in seeds]
        t5 = [r[s]["test"]["top5"] for s in seeds]
        v1 = [r[s]["val"]["top1"] for s in seeds]
        summary[k] = {"desc": desc, "test_top1_mean": float(np.mean(t1)), "test_top1_std": float(np.std(t1)),
                      "test_top5_mean": float(np.mean(t5)), "val_top1_mean": float(np.mean(v1)),
                      "n_test": r[seeds[0]]["test"]["n"], "n_train": r[seeds[0]]["n_train"]}
        print(f"{k} {desc:55s} test top1 {np.mean(t1):.3f}±{np.std(t1):.3f}  top5 {np.mean(t5):.3f}  val {np.mean(v1):.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"seeds": seeds, "summary": summary, "runs": results}, indent=2))
