"""
motion3.json parsing: curve evaluation and switch detection.

toggle_params decides which parameters the runtime snaps. It must flag a
parameter that authored motion uses as a 0/1 switch and leave alone one that
moves smoothly, or snapping would turn a head turn into a jump.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from l2dprobe.motions import (authored_motion, band_of, eval_segments,
                              toggle_params)


def test_linear_and_clamping():
    segs = [0.0, 0.0, 0, 1.0, 1.0]
    out = eval_segments(segs, np.array([-1, 0, 0.25, 0.5, 1.0, 2.0]))
    assert out == pytest.approx([0, 0, 0.25, 0.5, 1.0, 1.0])


def test_stepped_and_inverse_stepped():
    t = np.array([0.5])
    assert eval_segments([0, 0, 2, 1, 1], t)[0] == 0.0
    assert eval_segments([0, 0, 3, 1, 1], t)[0] == 1.0


def test_bezier_with_linear_handles_is_linear():
    segs = [0, 0, 1, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1]
    t = np.array([0.1, 0.5, 0.9])
    assert eval_segments(segs, t) == pytest.approx(t, abs=1e-3)


def test_multiple_segments_chain():
    segs = [0, 0, 0, 1, 2, 2, 2, 5, 0, 3, -1]
    out = eval_segments(segs, np.array([0.5, 1.5, 2.5, 3.0]))
    assert out == pytest.approx([1.0, 2.0, 2.0, -1.0])


def _write_rig(tmp_path, curves, duration=6.0):
    motion = {"Version": 3, "Meta": {"Duration": duration, "Fps": 30},
              "Curves": curves}
    (tmp_path / "m.motion3.json").write_text(json.dumps(motion))
    model = {"Version": 3, "FileReferences": {
        "Moc": "x.moc3", "Motions": {"Idle": [{"File": "m.motion3.json"}]}}}
    path = tmp_path / "x.model3.json"
    path.write_text(json.dumps(model))
    return str(path)


def test_toggle_params_flags_switch_not_sine(tmp_path):
    # A 0/1 swap with a 0.1 s fade, held either side.
    switch = [0, 0, 0, 2.9, 0, 0, 3.0, 1, 0, 6.0, 1]
    # A smooth sine as a chain of short linear pieces.
    sine = [0.0, 0.0]
    for k in range(1, 61):
        t = k * 0.1
        sine += [0, round(t, 3), math.sin(2 * math.pi * t / 3.0)]
    model = _write_rig(tmp_path, [
        {"Target": "Parameter", "Id": "ParamArmSwap", "Segments": switch},
        {"Target": "Parameter", "Id": "ParamAngleX", "Segments": sine},
        {"Target": "PartOpacity", "Id": "PartArm", "Segments": switch}])
    found = toggle_params(model, 30.0)
    assert "ParamArmSwap" in found
    frac, lo, hi = found["ParamArmSwap"]
    assert frac < 0.05 and lo == pytest.approx(0) and hi == pytest.approx(1)
    assert "ParamAngleX" not in found
    assert "PartArm" not in found          # parts are not parameters

    m = authored_motion(str(tmp_path / "m.motion3.json"), 30.0)
    assert m["frames"] == 180
    assert set(m["params"]) == {"ParamArmSwap", "ParamAngleX"}
    assert set(m["parts"]) == {"PartArm"}


def test_toggle_params_ignores_short_and_flat(tmp_path):
    model = _write_rig(tmp_path, [
        {"Target": "Parameter", "Id": "Flat", "Segments": [0, 1, 0, 6, 1]}])
    assert toggle_params(model, 30.0) == {}


def test_vowels_are_mouth_band():
    # ParamA/I/U/E/O drive lip-sync shapes; as expressive parameters the prior
    # generated vowels and the runtime snapped them as switches.
    for pid in ("ParamA", "ParamI", "ParamU", "ParamE", "ParamO", "PARAM_A"):
        assert band_of(pid, "continuous", "other", set()) == "mouth", pid
    for pid in ("ParamAngleX", "ParamEyeLOpen", "ParamArmA", "ParamE2"):
        assert band_of(pid, "continuous", "other", set()) != "mouth", pid


def test_body_angle_groups_as_body_not_head():
    from l2dprobe.analyze import group_of
    for pid in ("ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ"):
        assert group_of(pid) == "body", pid
    for pid in ("ParamAngleX", "ParamAngleY", "ParamAngleZ"):
        assert group_of(pid) == "head", pid
