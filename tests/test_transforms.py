"""
Whole-model transform detection from probe support maps.

The runtime holds what this flags. It must flag an effect that reaches the
top and bottom of the character and covers most of it (ParamAllX moves the
feet too), and must not flag a body angle that bends around fixed feet.
"""

from __future__ import annotations

import numpy as np

from l2dprobe.transforms import THRESHOLD, global_score, whole_model_params


def test_whole_body_move_is_flagged_body_bend_is_not():
    whole = np.ones((32, 32), np.float32)
    bend = np.zeros((32, 32), np.float32)
    bend[:20] = 1.0                       # upper body only; feet stay put
    assert global_score(whole) >= THRESHOLD
    assert global_score(bend) < THRESHOLD
    assert global_score(np.zeros((32, 32), np.float32)) == 0.0


def test_whole_model_params_reads_probe_npz(tmp_path):
    sup = np.zeros((2, 32, 32), np.float32)
    sup[0] = 1.0
    sup[1, :10] = 1.0
    np.savez(tmp_path / "rig_X.npz", support=sup,
             pids=np.array(["ParamAllX", "ParamAngleX"], dtype=object))
    assert set(whole_model_params(str(tmp_path), "X")) == {"ParamAllX"}
    assert whole_model_params(str(tmp_path), "missing") is None
