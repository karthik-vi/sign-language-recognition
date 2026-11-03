"""Training / evaluation loop (CPU-friendly)."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import FeatureConfig, build, feature_dim
from .model import SignGRU


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    hidden: int = 128
    dropout: float = 0.3
    augment: bool = True
    exclude_flagged_train: bool = True
    seed: int = 0
    patience: int = 30


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_samples(lm_dir: Path, manifest: list[dict]) -> dict[str, dict]:
    out = {}
    for r in manifest:
        p = lm_dir / f"{r['video_id']}.npz"
        if p.exists():
            with np.load(p) as z:
                out[r["video_id"]] = {k: z[k] for k in z.files}
    return out


def augment(s: dict, rng: np.random.Generator) -> dict:
    """Spatial affine + temporal crop + simulated missed hand detections, on raw landmarks."""
    T = len(s["pose"])
    if T >= 8:
        keep = int(T * rng.uniform(0.8, 1.0))
        a = rng.integers(0, T - keep + 1)
        sl = slice(a, a + keep)
    else:
        sl = slice(0, T)
    ang = np.deg2rad(rng.uniform(-8, 8))
    sc = rng.uniform(0.9, 1.1)
    sh = rng.uniform(-0.03, 0.03, 2)
    R = sc * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
    out = {}
    for k in ("pose", "lhand", "rhand"):
        x = s[k][sl].copy()
        xy = x[..., :2] - 0.5
        x[..., :2] = xy @ R.T + 0.5 + sh
        if k != "pose":
            drop = rng.random(len(x)) < 0.1
            x[drop] = np.nan
        out[k] = x
    return out


def _batches(X, y, bs, shuffle, rng):
    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i: i + bs]
        yield torch.from_numpy(np.stack([X[k] for k in j])), torch.from_numpy(y[j])


@torch.no_grad()
def evaluate(model, X, y) -> dict:
    model.eval()
    if len(X) == 0:
        return {}
    logits = model(torch.from_numpy(np.stack(X)))
    top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices.numpy()
    yt = np.asarray(y)
    return {"n": int(len(yt)), "top1": float((top5[:, 0] == yt).mean()), "top5": float((top5 == yt[:, None]).any(1).mean())}


def run(manifest: list[dict], samples: dict, feat: FeatureConfig, tc: TrainConfig, flags: dict | None = None,
        out_dir: Path | None = None) -> dict:
    set_seed(tc.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = np.random.default_rng(tc.seed)
    glosses = sorted({r["gloss"] for r in manifest})
    gid = {g: i for i, g in enumerate(glosses)}
    split = {k: [r for r in manifest if r["split"] == k and r["video_id"] in samples] for k in ("train", "val", "test")}
    excluded = 0
    if tc.exclude_flagged_train and flags:
        keep = [r for r in split["train"] if r["video_id"] not in flags]
        excluded = len(split["train"]) - len(keep)
        split["train"] = keep
    Xv = [build(samples[r["video_id"]], feat) for r in split["val"]]
    yv = np.array([gid[r["gloss"]] for r in split["val"]], np.int64)
    Xt = [build(samples[r["video_id"]], feat) for r in split["test"]]
    yt = np.array([gid[r["gloss"]] for r in split["test"]], np.int64)
    ytr = np.array([gid[r["gloss"]] for r in split["train"]], np.int64)
    Xtr_static = [build(samples[r["video_id"]], feat) for r in split["train"]]

    model = SignGRU(feature_dim(feat), len(glosses), tc.hidden, tc.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=tc.epochs * ((len(ytr) + tc.batch_size - 1) // tc.batch_size))
    lossf = nn.CrossEntropyLoss(label_smoothing=tc.label_smoothing)
    best, best_state, bad, hist = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(tc.epochs):
        model.train()
        Xtr = ([build(augment(samples[r["video_id"]], rng), feat) for r in split["train"]] if tc.augment else Xtr_static)
        tot = 0.0
        for xb, yb in _batches(Xtr, ytr, tc.batch_size, True, rng):
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
        v = evaluate(model, Xv, yv)
        hist.append({"epoch": ep, "loss": tot / len(ytr), "val_top1": v["top1"]})
        if v["top1"] > best:
            best, bad = v["top1"], 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc.patience:
                break
    model.load_state_dict(best_state)
    res = {
        "feature_config": asdict(feat), "train_config": asdict(tc),
        "n_train": len(ytr), "n_train_excluded_by_flags": excluded, "n_classes": len(glosses),
        "best_val_epoch": int(np.argmax([h["val_top1"] for h in hist])), "epochs_run": len(hist),
        "val": evaluate(model, Xv, yv), "test": evaluate(model, Xt, yt), "train_s": round(time.time() - t0, 1),
    }
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "glosses": glosses, "feature_config": asdict(feat),
                    "hidden": tc.hidden, "dropout": tc.dropout}, out_dir / "model.pt")
        (out_dir / "metrics.json").write_text(json.dumps({**res, "history": hist}, indent=2))
    return res
