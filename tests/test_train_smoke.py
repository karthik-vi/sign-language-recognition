"""End-to-end smoke test of training + evaluation on synthetic landmark sequences (no video needed)."""
import numpy as np

from slr.features import IMPROVED
from slr.train import TrainConfig, run


def _class_sample(c, rng, T=30):
    t = np.linspace(0, 1, T)[:, None]
    pose = np.full((T, 33, 4), 0.5, np.float32)
    pose[:, 11, :2], pose[:, 12, :2] = [0.4, 0.5], [0.6, 0.5]
    rh = np.zeros((T, 21, 3), np.float32)
    rh[..., 0] = 0.5 + 0.2 * np.sin(2 * np.pi * (c + 1) * t) + rng.normal(0, 0.01, (T, 21))
    rh[..., 1] = 0.6 + 0.02 * np.arange(21)[None] + rng.normal(0, 0.01, (T, 21))
    lh = np.full((T, 21, 3), np.nan, np.float32)
    return {"pose": pose, "lhand": lh, "rhand": rh}


def test_train_runs_and_learns():
    rng = np.random.default_rng(0)
    manifest, samples = [], {}
    for split, n in (("train", 12), ("val", 4), ("test", 4)):
        for c in range(3):
            for i in range(n):
                vid = f"{split}{c}{i}"
                samples[vid] = _class_sample(c, rng)
                manifest.append({"video_id": vid, "gloss": f"g{c}", "split": split, "signer_id": 0})
    res = run(manifest, samples, IMPROVED, TrainConfig(epochs=60, patience=60, seed=0, batch_size=8))
    assert res["test"]["n"] == 12
    assert res["test"]["top1"] > 0.6
