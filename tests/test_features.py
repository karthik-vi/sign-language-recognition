import numpy as np

from slr.consistency import check_dataset, tracking_stats
from slr.features import BASELINE, IMPROVED, build, feature_dim, interp_gaps


def fake_sample(T=40, seed=0, drop_left=True):
    rng = np.random.default_rng(seed)
    pose = rng.uniform(0.3, 0.7, (T, 33, 4)).astype(np.float32)
    pose[:, 11, :2] = [0.4, 0.5]
    pose[:, 12, :2] = [0.6, 0.5]
    lh = rng.uniform(0.3, 0.7, (T, 21, 3)).astype(np.float32)
    rh = rng.uniform(0.3, 0.7, (T, 21, 3)).astype(np.float32)
    if drop_left:
        lh[5:9] = np.nan
    return {"pose": pose, "lhand": lh, "rhand": rh, "n_frames_video": np.int32(T)}


def test_shapes_and_finite():
    s = fake_sample()
    for cfg in (BASELINE, IMPROVED):
        x = build(s, cfg)
        assert x.shape == (cfg.T, feature_dim(cfg))
        assert np.isfinite(x).all()


def test_empty_sequence():
    s = {"pose": np.zeros((0, 33, 4), np.float32), "lhand": np.zeros((0, 21, 3), np.float32), "rhand": np.zeros((0, 21, 3), np.float32)}
    assert build(s, IMPROVED).shape == (IMPROVED.T, feature_dim(IMPROVED))


def test_interp_fills_short_gap_only():
    x = np.arange(20, dtype=np.float32).reshape(20, 1, 1).repeat(3, -1)
    x[5:8] = np.nan
    x[10:19] = np.nan
    y = interp_gaps(x, max_gap=4)
    assert np.allclose(y[5:8, 0, 0], [5, 6, 7])
    assert np.isnan(y[12, 0, 0])


def test_consistency_flags():
    s = fake_sample(T=5)
    manifest = [{"video_id": "a", "gloss": "x", "split": "train", "signer_id": 1},
                {"video_id": "a", "gloss": "x", "split": "test", "signer_id": 2},
                {"video_id": "b", "gloss": "y", "split": "test", "signer_id": 2}]
    res = check_dataset(manifest, {"a": s}, vocab=["x"])
    f = res["flags"]
    assert "duplicate_video_id" in f["a"] and "video_in_multiple_splits" in f["a"] and "too_short" in f["a"]
    assert "no_landmarks_file" in f["b"] and "gloss_not_in_vocab" in f["b"]
    assert tracking_stats(fake_sample())["lhand_missing_frac"] > 0
