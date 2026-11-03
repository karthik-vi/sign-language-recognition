"""Landmark sequences -> fixed-size model inputs.

Two feature pipelines are kept side by side so the experiment log can show
what actually moved accuracy:

  baseline  raw image-space x,y of upper-body pose + both hands, NaN -> 0,
            uniform resample to T frames.
  improved  each step can be toggled with FeatureConfig (used for ablations):
    normalize    body-centred coords (shoulder midpoint origin, shoulder-width scale)
    hand_local   hand shape relative to its wrist, scaled by hand size
    interpolate  fill short gaps in hand tracks (missing detections) linearly
    smooth       temporal median filter (landmark jitter)
    trim_idle    drop leading/trailing frames where no hand is visible
    dominant     mirror so the more active hand is always the 'right' channel
    velocity     first differences of body-frame hand/wrist positions
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UPPER_POSE = list(range(0, 25))  # face + arms + hips (legs are rarely in frame)
L_SH, R_SH = 11, 12


@dataclass(frozen=True)
class FeatureConfig:
    T: int = 32
    normalize: bool = True
    hand_local: bool = True
    interpolate: bool = True
    max_gap: int = 8
    smooth: bool = True
    trim_idle: bool = True
    dominant: bool = True
    velocity: bool = True


BASELINE = FeatureConfig(normalize=False, hand_local=False, interpolate=False, smooth=False,
                         trim_idle=False, dominant=False, velocity=False)
IMPROVED = FeatureConfig()


def resample_idx(n: int, T: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(T, int)
    return np.clip(np.round(np.linspace(0, n - 1, T)).astype(int), 0, n - 1)


def interp_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """x: (T, K, D) with NaN rows for missing frames. Linearly fill interior gaps <= max_gap;
    edge gaps are filled by nearest value only if within max_gap."""
    x = x.copy()
    present = ~np.isnan(x[:, 0, 0])
    if present.sum() == 0:
        return x
    idx = np.where(present)[0]
    T = len(x)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                t = k / (gap + 1)
                x[a + k] = (1 - t) * x[a] + t * x[b]
    if idx[0] > 0 and idx[0] <= max_gap:
        x[: idx[0]] = x[idx[0]]
    if T - 1 - idx[-1] > 0 and T - 1 - idx[-1] <= max_gap:
        x[idx[-1] + 1:] = x[idx[-1]]
    return x


def median_smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Temporal median filter that ignores NaNs (keeps NaN where centre is NaN)."""
    if len(x) < k:
        return x
    pad = k // 2
    xp = np.concatenate([x[:1].repeat(pad, 0), x, x[-1:].repeat(pad, 0)])
    win = np.stack([xp[i: i + len(x)] for i in range(k)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.nanmedian(win, axis=0)
    out[np.isnan(x[..., :1]).repeat(x.shape[-1], -1)] = np.nan
    return out


def _activity(h: np.ndarray) -> float:
    w = h[:, 0, :2]
    ok = ~np.isnan(w[:, 0])
    if ok.sum() < 2:
        return 0.0
    d = np.diff(w[ok], axis=0)
    return float(np.nansum(np.linalg.norm(d, axis=1))) + 0.01 * ok.sum()


def build(sample: dict, cfg: FeatureConfig) -> np.ndarray:
    """sample: dict with pose (T,33,4), lhand (T,21,3), rhand (T,21,3). Returns (cfg.T, F) float32."""
    pose = sample["pose"][:, UPPER_POSE, :3].astype(np.float32)
    lh, rh = sample["lhand"].astype(np.float32), sample["rhand"].astype(np.float32)
    n = len(pose)
    if n == 0:
        return np.zeros((cfg.T, feature_dim(cfg)), np.float32)

    if cfg.trim_idle:
        anyhand = ~np.isnan(lh[:, 0, 0]) | ~np.isnan(rh[:, 0, 0])
        if anyhand.any():
            a, b = np.where(anyhand)[0][[0, -1]]
            pose, lh, rh = pose[a: b + 1], lh[a: b + 1], rh[a: b + 1]

    if cfg.dominant and _activity(lh) > _activity(rh):
        # mirror: swap hands and flip x so the dominant hand always lands in the 'right' channel
        lh, rh = rh.copy(), lh.copy()
        for arr in (pose, lh, rh):
            arr[..., 0] = 1.0 - arr[..., 0]
        perm = list(range(25))
        for a, b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]:
            perm[a], perm[b] = b, a
        pose = pose[:, perm]

    if cfg.interpolate:
        lh, rh = interp_gaps(lh, cfg.max_gap), interp_gaps(rh, cfg.max_gap)
        pose = interp_gaps(pose, cfg.max_gap)
    if cfg.smooth:
        lh, rh, pose = median_smooth(lh), median_smooth(rh), median_smooth(pose)

    present = np.stack([~np.isnan(lh[:, 0, 0]), ~np.isnan(rh[:, 0, 0])], -1).astype(np.float32)

    if cfg.normalize:
        sh = pose[:, [L_SH, R_SH], :2]
        centre = np.nanmedian(sh.mean(1), axis=0)
        width = np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))
        if not np.isfinite(width) or width < 1e-3:
            width = 0.2
        if not np.all(np.isfinite(centre)):
            centre = np.array([0.5, 0.5], np.float32)
        norm = lambda a: (a[..., :2] - centre) / width  # noqa: E731
    else:
        norm = lambda a: a[..., :2]  # noqa: E731

    parts = [norm(pose).reshape(len(pose), -1)]
    for h in (lh, rh):
        if cfg.hand_local:
            wrist = h[:, :1, :2]
            size = np.linalg.norm(h[:, 9, :2] - h[:, 0, :2], axis=-1)[:, None, None]  # wrist -> middle MCP
            size = np.where(np.isfinite(size) & (size > 1e-4), size, 1.0)
            parts += [norm(h[:, 0]).reshape(len(h), -1), ((h[..., :2] - wrist) / size).reshape(len(h), -1)]
        else:
            parts.append(norm(h).reshape(len(h), -1))
    x = np.concatenate(parts, -1)
    if cfg.velocity:
        v = np.diff(np.concatenate([norm(lh[:, 0]), norm(rh[:, 0])], -1), axis=0, prepend=np.nan * np.ones((1, 4), np.float32))
        x = np.concatenate([x, v], -1)
    x = np.concatenate([x, present], -1)
    x = np.nan_to_num(x[resample_idx(len(x), cfg.T)], nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def feature_dim(cfg: FeatureConfig) -> int:
    d = 25 * 2
    d += 2 * ((2 + 21 * 2) if cfg.hand_local else 21 * 2)
    d += 4 if cfg.velocity else 0
    return d + 2
