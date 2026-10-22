"""
flow.behaviors: the hand-made baseline. Pure numpy.

Binding must take Cubism 2 ids (PARAM_ANGLE_X) like Cubism 3 ones and let an
override win. The generated motion must not be periodic (the scorer's check
would catch a clock-driven scheduler), every move must start and end at rest
(minimum jerk), and fitting must never read a held-out rig.
"""

from __future__ import annotations

import json

import numpy as np

from flow.behaviors import (DEFAULTS, ROLES, bind_roles, fit, generate,
                            min_jerk)
from flow.runtime import Autonomic
from flow.score import periodic_peak


def test_binding_standard_ids_aliases_and_override():
    pids = ["PARAM_ANGLE_X", "ParamAngleY", "PARAM_EYE_BALL_X", "ParamFoo",
            "ParamBodyAngleZ"]
    b = bind_roles(pids)
    assert b["head_x"] == ("PARAM_ANGLE_X", 1.0, "id")
    assert b["gaze_x"][0] == "PARAM_EYE_BALL_X"
    assert b["body_z"][0] == "ParamBodyAngleZ"
    assert "gaze_y" not in b                    # no fallback without a rig
    b = bind_roles(pids, override={"head_x": ["ParamFoo", -1]})
    assert b["head_x"] == ("ParamFoo", -1.0, "override")
    # `allowed` restricts candidates (e.g. to the expressive band).
    assert "head_x" not in bind_roles(pids, allowed=["ParamAngleY"])


def test_min_jerk_ends_at_rest():
    u = np.linspace(0, 1, 1001)
    s = min_jerk(u)
    v = np.gradient(s, u)
    assert s[0] == 0 and abs(s[-1] - 1) < 1e-12
    assert abs(v[0]) < 1e-2 and abs(v[-1]) < 1e-2


def test_generated_motion_is_aperiodic_and_bounded():
    fps = 30.0
    for seed in (0, 1, 2):
        out, blinks = generate(DEFAULTS, list(ROLES), 60.0, fps,
                               np.random.default_rng(seed))
        for r, v in out.items():
            assert len(v) == 1800 and np.abs(v).max() <= 1
            peak, _ = periodic_peak(v.astype(np.float64), fps)
            assert peak <= 0.6, (seed, r, peak)
        assert out["gaze_x"].std() > 0.05 and out["head_x"].std() > 0.02
        assert all(0 <= t < 60 for t in blinks)
    a, _ = generate(DEFAULTS, ["gaze_x"], 10, fps, np.random.default_rng(0))
    b, _ = generate(DEFAULTS, ["gaze_x"], 10, fps, np.random.default_rng(1))
    assert np.abs(a["gaze_x"] - b["gaze_x"]).mean() > 1e-3   # seeds differ


def test_blink_request_fires_and_schedule_resumes():
    fps = 30.0
    auto = Autonomic(fps, np.random.default_rng(0), mean_interval=1000.0,
                     requests=[1.0])
    eye = np.array([auto.step()[0] for _ in range(int(3 * fps))])
    closed = np.flatnonzero(eye < 0.5) / fps
    assert len(closed) and 0.95 <= closed[0] <= 1.15
    assert closed[-1] < 1.3                     # one blink, then open again
    assert 1.3 < auto.next_blink                # natural schedule rescheduled


def test_fit_skips_holdout(tmp_path):
    rng = np.random.default_rng(0)
    pids = np.array(["ParamAngleX", "ParamEyeBallX"], dtype=object)
    for name, amp in (("Train", 0.2), ("Held", 0.9)):
        x = (amp * np.sin(np.linspace(0, 20, 300))[:, None]
             * np.ones((1, 2)) + 0.01 * rng.standard_normal((300, 2)))
        np.savez(tmp_path / f"{name}.npz", clip0=x.astype(np.float32),
                 mask0=np.ones(2, bool), pid=pids, fps=np.float32(30),
                 n_clips=np.int32(1))
    st = fit([str(tmp_path / "Train.npz"), str(tmp_path / "Held.npz")],
             holdout={"Held"})
    assert st["rigs"] == ["Train"]
    assert st["roles"]["head_x"]["std"] < 0.3
    json.dumps(st)                               # writable as-is


def test_abtest_hides_sources_and_tallies(tmp_path):
    from flow.abtest import make, tally
    vids = []
    for src in ("beh", "flow", "static"):
        p = tmp_path / f"{src}.mp4"
        p.write_bytes(src.encode())
        vids.append(f"Natori/{src}={p}")
    out = tmp_path / "ab"
    assert make(str(out), vids, seed=0) == 0
    key = json.loads((out / "answer_key.json").read_text())
    assert len(key) == 3
    for pid, k in key.items():
        # Files carry no source name; contents match the key.
        assert (out / f"p{pid}_A.mp4").read_bytes() == k["A"].encode()
    lines = [f"{pid} A" if key[pid]["A"] == "static" or key[pid]["B"] != "static"
             else f"{pid} B" for pid in key]
    (out / "judgments.txt").write_text("\n".join(lines))
    assert tally(str(out)) == 0
