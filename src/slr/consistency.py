"""Data-consistency checks for a landmark dataset.

Checks (per video unless stated):
  - label:   gloss vocabulary, duplicate video ids, same video id in >1 split,
             classes without training examples, signer overlap across splits (report only)
  - frames:  unreadable / zero extracted frames, very short clips, trim window past video end
  - tracking: frames with no pose, frames with no hand, wrist jitter (median frame-to-frame
             acceleration in shoulder widths), wrist jumps (> JUMP_SW shoulder widths in one frame,
             usually a left/right hand swap or a false detection)

Policy: flags may be used to EXCLUDE TRAINING samples only. Val/test samples are never dropped,
otherwise reported accuracy would be inflated.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

MIN_FRAMES = 12
JUMP_SW = 1.0
MAX_NOHAND_FRAC = 0.9


def tracking_stats(s: dict) -> dict:
    pose, lh, rh = s["pose"], s["lhand"], s["rhand"]
    T = len(pose)
    if T == 0:
        return {"n_frames": 0}
    sh = pose[:, [11, 12], :2]
    sw = float(np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))) if np.isfinite(sh).any() else np.nan
    sw = sw if np.isfinite(sw) and sw > 1e-3 else 0.2
    out = {
        "n_frames": int(T),
        "pose_missing_frac": float(np.isnan(pose[:, 0, 0]).mean()),
        "lhand_missing_frac": float(np.isnan(lh[:, 0, 0]).mean()),
        "rhand_missing_frac": float(np.isnan(rh[:, 0, 0]).mean()),
        "nohand_frac": float((np.isnan(lh[:, 0, 0]) & np.isnan(rh[:, 0, 0])).mean()),
    }
    jitter, jumps = [], 0
    for h in (lh, rh):
        w = h[:, 0, :2] / sw
        d = np.diff(w, axis=0)
        ok = ~np.isnan(d[:, 0])
        jumps += int((np.linalg.norm(d[ok], axis=1) > JUMP_SW).sum())
        a = np.diff(d, axis=0)
        ok2 = ~np.isnan(a[:, 0])
        if ok2.sum() > 2:
            jitter.append(float(np.median(np.linalg.norm(a[ok2], axis=1))))
    out["wrist_jitter_sw"] = float(np.mean(jitter)) if jitter else None
    out["wrist_jumps"] = jumps
    return out


def check_dataset(manifest: list[dict], samples: dict[str, dict], vocab: list[str] | None = None) -> dict:
    flags: dict[str, list[str]] = defaultdict(list)
    report: dict = {"n_manifest": len(manifest)}

    ids = Counter(r["video_id"] for r in manifest)
    for vid, c in ids.items():
        if c > 1:
            flags[vid].append("duplicate_video_id")
    splits_of = defaultdict(set)
    for r in manifest:
        splits_of[r["video_id"]].add(r["split"])
    for vid, s in splits_of.items():
        if len(s) > 1:
            flags[vid].append("video_in_multiple_splits")
    if vocab is not None:
        for r in manifest:
            if r["gloss"] not in vocab:
                flags[r["video_id"]].append("gloss_not_in_vocab")
    by_split = defaultdict(Counter)
    for r in manifest:
        by_split[r["split"]][r["gloss"]] += 1
    report["class_counts_per_split"] = {k: {"n": sum(v.values()), "classes": len(v), "min": min(v.values()), "max": max(v.values())}
                                        for k, v in by_split.items()}
    report["classes_missing_in_train"] = sorted(set().union(*[set(v) for v in by_split.values()]) - set(by_split.get("train", {})))
    signers = {k: {r["signer_id"] for r in manifest if r["split"] == k} for k in by_split}
    if "train" in signers and "test" in signers:
        report["signer_overlap_train_test"] = {
            "test_signers": len(signers["test"]),
            "test_signers_seen_in_train": len(signers["test"] & signers["train"]),
        }

    stats = {}
    for r in manifest:
        vid = r["video_id"]
        s = samples.get(vid)
        if s is None:
            flags[vid].append("no_landmarks_file")
            continue
        st = tracking_stats(s)
        stats[vid] = st
        if st["n_frames"] == 0:
            flags[vid].append("zero_frames")
            continue
        if st["n_frames"] < MIN_FRAMES:
            flags[vid].append("too_short")
        if r.get("frame_end", -1) > 0 and int(s.get("n_frames_video", 0)) and r["frame_end"] > int(s["n_frames_video"]) + 2:
            flags[vid].append("trim_window_past_video_end")
        if st["pose_missing_frac"] > 0.5:
            flags[vid].append("pose_mostly_missing")
        if st["nohand_frac"] > MAX_NOHAND_FRAC:
            flags[vid].append("hands_mostly_missing")
        if st["wrist_jumps"] > 3:
            flags[vid].append("wrist_jumps")
    report["flag_counts"] = dict(Counter(f for fs in flags.values() for f in fs))
    report["n_flagged"] = len(flags)
    agg = defaultdict(list)
    for st in stats.values():
        for k, v in st.items():
            if isinstance(v, (int, float)) and v is not None:
                agg[k].append(v)
    report["tracking_summary"] = {k: {"mean": float(np.mean(v)), "p50": float(np.median(v)), "p90": float(np.percentile(v, 90))}
                                  for k, v in agg.items()}
    return {"report": report, "flags": dict(flags), "stats": stats}
