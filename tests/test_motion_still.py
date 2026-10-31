"""scripts/motion_still.py: the share of still 0.1 s bins and the burstiness
of a render's motion, on synthetic signals and on MotionStream plans (no
GL)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from flow import stream as fs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "motion_still", os.path.join(ROOT, "scripts", "motion_still.py"))
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)
FPS = 30.0


def test_still_share_and_burst_on_synthetic_signals():
    n = int(20 * FPS)
    t = np.arange(n) / FPS
    flat = np.zeros(n)
    r = ms.still_stats(flat, FPS, 0.01)
    assert r["still"] == 1.0 and r["burst"] == float("inf")
    even = np.full(n, 0.2)
    r = ms.still_stats(even, FPS, 0.01)
    assert r["still"] == 0.0 and abs(r["burst"] - 1.0) < 1e-9
    # spikes of 0.3 s every 2 s in silence: 85% of the bins still
    spikes = np.where((t % 2.0) < 0.3, 1.0, 0.0)
    r = ms.still_stats(spikes, FPS, 0.01)
    assert abs(r["still"] - 0.85) < 0.02 and r["burst"] == float("inf")
    # a quiet floor under the same spikes: nothing still, strongly bursty
    r = ms.still_stats(spikes + 0.05, FPS, 0.01)
    assert r["still"] == 0.0 and r["burst"] > 10


def test_bins_are_a_tenth_of_a_second_and_drop_the_tail():
    sp = np.arange(31, dtype=float)
    assert list(ms.bin_speed(sp, 30.0)) == [1.0, 4.0, 7.0, 10.0, 13.0, 16.0,
                                            19.0, 22.0, 25.0, 28.0]
    assert ms.still_stats(np.zeros(2), 30.0, 0.01)["bins"] == 0


def test_plan_npz_speed_leaves_the_mouth_and_blinks_out(tmp_path):
    n = int(5 * FPS)
    t = np.arange(n) / FPS
    traj = np.zeros((n, 3), np.float32)
    traj[:, 1] = np.sin(2 * np.pi * 5 * t)             # the mouth
    traj[:, 2] = np.where(t > 2.5, -1.0, 1.0)          # a blink-like step
    np.savez(tmp_path / "p.npz", traj=traj, fps=np.float32(FPS),
             pid=np.array(["ParamAngleX", "ParamMouthOpenY", "ParamEyeLOpen"],
                          dtype=object))
    sp, fps = ms.plan_speed(str(tmp_path / "p.npz"))
    assert fps == FPS and not sp.any()
    sp, _ = ms.plan_speed(str(tmp_path / "p.npz"), keep_all=True)
    assert sp.max() > 1
    assert ms.main([str(tmp_path / "p.npz")]) == 1      # all still: fails
    traj[:, 0] = 0.3 * np.sin(2 * np.pi * 0.3 * t)
    np.savez(tmp_path / "q.npz", traj=traj, fps=np.float32(FPS),
             pid=np.array(["ParamAngleX", "ParamMouthOpenY", "ParamEyeLOpen"],
                          dtype=object))
    assert ms.main([str(tmp_path / "q.npz")]) == 0


def _stream_speed(rig, drift, ease, seed, beats=True, n=300):
    st = fs.MotionStream(rig, seed=seed, idle=True, autonomic=False,
                         head_drift=drift, pose_ease=ease,
                         log=lambda s: None)
    if beats:
        # four short head beats in the manner of the beats prompt:
        # dur 0.2, hold 0.4, a release of 0.5
        st.direct([{"t": t, "type": "pose", "role": "pose", "params": {
            "ParamAngleX": x, "ParamAngleY": y}, "parts": {},
            "parts_from": {}, "step": [], "duration": 0.2, "hold": 0.4,
            "release": 0.5, "groups": ["head"], "variant": "v",
            "src": "director"} for t, x, y in (
                (1.0, 6, -3), (2.6, -5, 2), (4.2, 4, 3), (5.8, -4, -2))])
    ex, lo = rig.expressive, rig.lo
    span = np.maximum(rig.hi[ex] - lo[ex], 1e-6)
    tr = np.empty((n, len(ex)), np.float32)
    for f in range(n):
        over, _ = st.frame()
        for c, i in enumerate(ex):
            tr[f, c] = (over.get(rig.pids[i], rig.dflt[i]) - lo[i]) \
                / span[c] * 2 - 1
    cols = [c for c, i in enumerate(ex) if not ms.QUIET.search(rig.pids[i])]
    d = np.diff(tr[:, cols], axis=0) * FPS / 2
    return np.concatenate([[0.0], np.linalg.norm(d, axis=1)])


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    from tests.test_speech import _mouth_rig
    tmp = tmp_path_factory.mktemp("still")
    model, rig_ = _mouth_rig(tmp / "x")
    return fs.EventRig(str(model), str(rig_), "default", None, None, FPS,
                       str(tmp), log=lambda s: None)


def test_drift_and_eased_beats_fix_the_stills_and_the_bursts(rig):
    """Four short head beats over the calm idle, 10 s, 8 seeds: before (no
    drift, beats as written) a quarter of the bins are still, with drift
    and eased beats under a tenth, and the spikes are lower."""
    def avg(drift, ease, beats):
        rs = [ms.still_stats(_stream_speed(rig, drift, ease, s, beats), FPS,
                             ms.EPS_PLAN) for s in range(8)]
        return (np.mean([r["still"] for r in rs]),
                np.mean([r["burst"] for r in rs]))
    still0, burst0 = avg(None, False, True)
    still1, burst1 = avg(1.0, True, True)
    assert still0 > 0.15 and still1 < ms.STILL_TARGET
    assert burst1 < 0.8 * burst0
    # the calm idle alone, no beats: drift covers the fixations
    idle0, _ = avg(None, False, False)
    idle1, _ = avg(1.0, True, False)
    assert idle0 > 0.25 and idle1 < ms.STILL_TARGET


def test_drift_covers_the_head_and_sways_the_body(rig):
    d = fs.HeadDrift(rig, 0, 1.0)
    pids = {a[0] for a in d.axes}
    assert {"ParamAngleX", "ParamAngleY"} <= pids
    # a rig with body angle roles: a breathing sway of under a degree, slow
    import types
    r = types.SimpleNamespace(
        pids=["ParamBodyAngleX", "ParamBodyAngleZ"], expressive=[0, 1],
        lo=np.array([-10.0, -10.0]), hi=np.array([10.0, 10.0]),
        role_cols=[("body_x", 0, 1.0), ("body_z", 1, 1.0)])
    sw = fs.HeadDrift(r, 3, 1.0)
    tr = np.array([[sw.offsets(t)[p] for p in r.pids]
                   for t in np.arange(0, 60, 0.1)])
    assert 0.2 < np.abs(tr).max() <= 0.8 + 0.6 + 1e-9
    assert np.abs(tr[:, 0]).max() <= fs.SWAY_DEG["body_x"] + 1e-9
    assert np.abs(tr[:, 1]).max() <= fs.SWAY_DEG["body_z"] + 1e-9
    assert np.abs(np.diff(tr, axis=0)).max() < 0.12   # slow: < 1.2 deg/s
