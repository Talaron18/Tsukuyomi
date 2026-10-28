"""
flow.actionhead's numpy side, on system python (no torch): the eval
metrics on hand-made cases with known answers -- relative MSE (the motion
around the posture), velocity MSE, the predicted to true std (a constant
prediction has none), beat alignment of a motion with the audio's accents
(a copy of the RMS lines up, a motion offset from it does not, a constant
one is undefined: None, not NaN) -- the per-avatar and pooled
normalisation, the history corruption, the prosody features and the
anchor of delta targets.
"""

from __future__ import annotations

import json
import math

import numpy as np

from flow import actionhead as ah


def sc(x, y, m=None, sd=1.0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.ones(y.shape, bool) if m is None else m
    return ah.motion_scores(x, y, m, np.full(y.shape[1], sd))


def test_posture_offset_is_in_mse_but_not_in_relative_mse():
    t = np.linspace(0, 6 * math.pi, 120)
    y = np.sin(t)[:, None]
    p = y + 3.0                                     # the right motion, 3 off
    s = ah.score_summary(sc(p, y))
    assert s["normalised_mse"] == 9.0
    assert s["relative_mse"] < 1e-12 and s["velocity_mse"] < 1e-12
    assert abs(s["std_ratio"] - 1) < 1e-9
    # holding still: the whole motion is the error
    r = ah.score_summary(sc(np.zeros_like(y), y))
    assert abs(r["relative_mse"] - np.var(y)) < 1e-9
    assert abs(r["velocity_mse"] - np.mean(np.diff(y, axis=0) ** 2)) < 1e-9


def test_constant_prediction_has_no_std_and_half_the_motion_is_half():
    y = np.sin(np.linspace(0, 20, 200))[:, None] * 2
    assert ah.score_summary(sc(np.full_like(y, 5.0), y))["std_ratio"] == 0.0
    s = ah.score_summary(sc(0.5 * y, y))
    assert abs(s["std_ratio"] - 0.5) < 1e-9
    assert abs(s["velocity_std_ratio"] - 0.5) < 1e-9
    # a target that does not move: the ratio is undefined, not infinite
    assert ah.score_summary(sc(y, np.ones_like(y)))["std_ratio"] is None


def test_scores_count_only_the_known_values_and_use_the_std():
    y = np.array([[0.0], [2.0], [4.0], [6.0]])
    m = np.array([[1], [1], [0], [1]], bool)
    x = np.array([[0.0], [2.0], [100.0], [8.0]])
    s = sc(x, y, m, sd=2.0)
    assert s["n"][0] == 3 and s["se"][0] == 4.0 and s["sen"][0] == 1.0
    assert s["nv"][0] == 1                       # only frames 0-1 are a pair


def test_normalisation_per_avatar(tmp_path):
    class D:
        vocab = ["a", "b"]
        avatars = ["?", "x", "y"]

        def __init__(self):
            self.its = [
                {"a": 1, "y": np.array([[0.], [2.]], np.float32).repeat(2, 1),
                 "m": np.array([[1, 1], [1, 1]], bool)},
                {"a": 2, "y": np.array([[10.], [14.]], np.float32).repeat(
                    2, 1), "m": np.array([[1, 0], [1, 0]], bool)}]

        def item(self, i):
            return self.its[i]

    d = D()
    m0, s0 = ah.norm_stats(d, [0, 1])
    assert m0.shape == (2,) and abs(m0[0] - 6.5) < 1e-5
    m, s = ah.norm_stats(d, [0, 1], per_avatar=True)
    assert m.shape == (3, 2)
    np.testing.assert_allclose(m[1], [1, 1])            # x's own
    np.testing.assert_allclose(m[2, 0], 12)             # y's own
    np.testing.assert_allclose(s[2, 0], 2)
    np.testing.assert_allclose(m[2, 1], m0[1])          # y has no b: pooled
    np.testing.assert_allclose(m[0], m0)                # unknown: pooled
    assert ah.stats_for(m, s, 2)[0][0] == 12
    assert ah.stats_for(m, s, 9)[0][0] == m0[0]         # past the table
    assert ah.stats_for(m0, s0, 2)[0] is m0             # pooled passes
    pm, ps = ah.pooled(m, s)
    np.testing.assert_allclose(ps, s0)


def accents(n=300, period=30, decay=6.0):
    """An RMS with a sharp rise every `period` frames and a slow fall."""
    r = np.zeros(n)
    for k in range(5, n, period):
        j = np.arange(n - k)
        r[k:] = np.maximum(r[k:], np.exp(-j / decay))
    return r


def test_beat_alignment_of_a_copy_of_the_rms():
    for r in (accents(), 0.5 - 0.5 * np.cos(np.linspace(0, 20 * math.pi,
                                                        300))):
        y = r[:, None]                              # symmetric too
        b = ah.beat_scores({"copy": [(y, np.ones(y.shape, bool),
                                      np.ones(1))]}, {"g": [0]}, [r], 30.0)
        g = b["copy"]["g"]
        assert g["corr"] > 0.99, g
        assert g["consistency"] == 1.0 and g["precision"] == 1.0, g


def test_beat_alignment_of_motion_between_the_accents_and_of_none():
    r = accents()
    off = np.roll(r, 15)[:, None]                   # peaks half a period away
    one = np.ones(1)
    ok = np.ones(off.shape, bool)
    g = ah.beat_scores({"x": [(off, ok, one)]}, {"g": [0]}, [r],
                       30.0)["x"]["g"]
    assert g["consistency"] == 0.0 and g["precision"] == 0.0
    assert g["corr"] < 0.3
    # a constant motion: undefined, and JSON-safe
    c = ah.beat_scores({"x": [(np.ones((300, 1)), ok, one)]}, {"g": [0]},
                       [r], 30.0)["x"]["g"]
    assert c["corr"] is None and c["precision"] is None
    assert c["consistency"] == 0.0              # no beat of its own: no hit
    assert json.loads(json.dumps(c))["corr"] is None
    # a group not known all along is skipped
    bad = ok.copy()
    bad[10] = False
    s = ah.beat_scores({"x": [(off, bad, one)]}, {"g": [0]}, [r],
                       30.0)["x"]["g"]
    assert s["segments"] == 0 and s["corr"] is None


def test_history_corruption_drops_and_noises():
    rng = np.random.default_rng(0)
    y = np.ones((30, 4), np.float32)
    m = np.ones((30, 4), bool)
    ys, ms = ah.corrupt_history(y, m, rng, noise=0, drop_frame=0,
                                drop_var=0, drop_all=0)
    assert ms.all() and (ys == 1).all()
    ys, ms = ah.corrupt_history(y, m, rng, noise=0, drop_frame=0,
                                drop_var=0, drop_all=1)
    assert not ms.any() and (ys == 0).all()
    kept, noisy = [], []
    for _ in range(200):
        ys, ms = ah.corrupt_history(y, m, rng, noise=0.3, drop_frame=0.2,
                                    drop_var=0.2, drop_all=0.0)
        kept.append(ms.mean())
        noisy.append(np.abs(ys[ms] - 1).mean() if ms.any() else 0)
        assert (ys[~ms] == 0).all()
    assert 0.55 < np.mean(kept) < 0.75              # 0.8 * 0.8
    assert 0.02 < np.mean(noisy) < 0.3


def test_last_valid_frame_is_the_anchor():
    y = np.array([[1, 10], [2, 20], [3, 30]], np.float32)
    m = np.array([[1, 1], [1, 1], [0, 1]], bool)
    np.testing.assert_array_equal(ah.last_valid(y, m), [2, 30])
    np.testing.assert_array_equal(ah.last_valid(y, m & False), [0, 0])


def test_prosody_features():
    p = np.array([[0.1, 150.0, 1], [0.0, 0.0, 0], [0.01, 300.0, 1]],
                 np.float32)
    f = ah.pros_feat(p)
    assert f.shape == (3, 3) and np.isfinite(f).all()
    assert f[0, 1] == 0 and f[1, 1] == 0 and f[2, 1] > 0   # log f0 / 150
    assert f[0, 0] > f[2, 0] > f[1, 0]                     # louder is more
    np.testing.assert_array_equal(f[:, 2], [1, 0, 1])
