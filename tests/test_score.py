"""
flow.score on synthetic data with known answers.

Authored motion scored against itself must pass with every ratio at 1 -- the
sanity check on the metric code. A frozen trajectory (the collapsed prior's
failure) must fail the std check, and a clean loop must fail periodicity.
The synthetic authored motion is smoothed noise, deliberately aperiodic: a
sine here would fail the absolute periodicity bound against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from flow.score import periodic_peak, score

PIDS = ["ParamAngleX", "ParamAngleY", "ParamBodyAngleX", "ParamEyeBallX",
        "ParamArmLA"]
GROUPS = ["head", "head", "head", "gaze", "arm"]   # BodyAngle as group_of has it
FPS = 30.0


def _smooth_noise(rng, n, p, width=25):
    k = np.hanning(width)
    k /= k.sum()
    x = rng.standard_normal((n + width, p)).cumsum(axis=0)
    x = np.stack([np.convolve(x[:, j], k, "same") for j in range(p)], 1)[:n]
    x -= x.mean(axis=0)
    return np.clip(0.5 * x / (np.abs(x).max(axis=0) + 1e-9), -1, 1)


@pytest.fixture()
def files(tmp_path):
    rng = np.random.default_rng(0)
    clips = {}
    for i, n in enumerate((400, 300, 500)):
        x = _smooth_noise(rng, n, len(PIDS)).astype(np.float32)
        mask = np.ones(len(PIDS), bool)
        if i == 1:
            mask[4] = False                 # this clip leaves the arm alone
            x[:, 4] = 0.0
        clips[f"clip{i}"], clips[f"mask{i}"] = x, mask
    cpath = tmp_path / "clips.npz"
    np.savez(cpath, **clips, pid=np.array(PIDS, dtype=object),
             param_row=np.arange(len(PIDS), dtype=np.int32),
             fps=np.float32(FPS), n_clips=np.int32(3))
    rpath = tmp_path / "rig.npz"
    np.savez(rpath, pid=np.array(PIDS, dtype=object),
             group=np.array(GROUPS, dtype=object),
             minimum=np.full(len(PIDS), -30, np.float32),
             maximum=np.full(len(PIDS), 30, np.float32))
    return tmp_path, str(cpath), str(rpath)


def _dump(path, traj):
    np.savez(path, traj=traj.astype(np.float32),
             mask=np.ones(traj.shape, bool),
             segment=np.zeros(len(traj), np.int32),
             pid=np.array(PIDS, dtype=object),
             param_row=np.arange(len(PIDS), dtype=np.int32),
             fps=np.float32(FPS), mode=np.array("test"))
    return str(path)


def test_authored_against_itself_passes(files):
    _, clips, rig = files
    res = score(clips, clips, rig)
    assert res["pass"], res["checks"]
    o = res["overall"]
    for k in ("std", "dmean", "dp95", "jerk"):
        assert o[k] == pytest.approx(1.0)
    # The rig file says head; score splits BodyAngle back out.
    assert set(res["groups"]) == {"head", "body", "gaze", "arm"}
    assert res["groups"]["body"]["params"] == ["ParamBodyAngleX"]


def test_frozen_trajectory_fails_std(files):
    tmp, clips, rig = files
    traj = np.zeros((900, len(PIDS)))
    res = score(_dump(tmp / "frozen.npz", traj), clips, rig)
    assert not res["checks"]["std"]
    assert not res["pass"]
    assert res["overall"]["std"] < 0.05
    assert res["groups"]["head"]["still_gen"] == 1.0


def test_loop_fails_periodicity(files):
    tmp, clips, rig = files
    t = np.arange(1800) / FPS
    base = 0.3 * np.sin(2 * np.pi * t / 4.0)
    traj = np.repeat(base[:, None], len(PIDS), axis=1)
    res = score(_dump(tmp / "loop.npz", traj), clips, rig)
    assert not res["checks"]["period"]
    assert res["groups"]["head"]["period_gen"] > 0.9


def test_mismatched_columns_are_refused(files, tmp_path):
    _, clips, rig = files
    bad = tmp_path / "bad.npz"
    np.savez(bad, traj=np.zeros((10, 2), np.float32),
             pid=np.array(["A", "B"], dtype=object), fps=np.float32(FPS))
    with pytest.raises(ValueError):
        score(str(bad), clips, rig)


def test_periodic_peak_ignores_drift_and_constants():
    t = np.arange(900) / FPS
    assert periodic_peak(np.zeros(900), FPS) == (0.0, 0.0)
    drift = np.tanh((t - 15) / 4)           # one slow turn, no repeat
    peak, plain = periodic_peak(drift, FPS)
    assert peak == 0.0 and plain > 0.6      # the plain max would misfire
    assert periodic_peak(np.sin(2 * np.pi * t / 3), FPS)[0] > 0.9


def test_constant_clips_report_no_data_not_fail(tmp_path):
    # 魔女's clips hold every driven parameter constant. Authored motion scored
    # against itself must not FAIL; there is simply nothing to compare.
    x = np.full((200, len(PIDS)), 0.25, np.float32)
    cpath = tmp_path / "flat.npz"
    np.savez(cpath, clip0=x, mask0=np.ones(len(PIDS), bool),
             pid=np.array(PIDS, dtype=object),
             param_row=np.arange(len(PIDS), dtype=np.int32),
             fps=np.float32(FPS), n_clips=np.int32(1))
    rpath = tmp_path / "rig.npz"
    np.savez(rpath, pid=np.array(PIDS, dtype=object),
             group=np.array(GROUPS, dtype=object),
             minimum=np.full(len(PIDS), -30, np.float32),
             maximum=np.full(len(PIDS), 30, np.float32))
    res = score(str(cpath), str(cpath), str(rpath))
    assert res["pass"] is None and "no_data" in res
