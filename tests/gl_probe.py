#!/usr/bin/env python3
"""
GL-side halves of the render regression tests, one check per process.

The Live2D backend shares one GL context and one Cubism framework per process
(backend._SHARED), fixed at the size of the first backend opened. pytest runs
every test in one process, so each render check runs here instead, in a
fresh interpreter, and prints one JSON object on its last line for the test
to judge. Keeping the measurement here and the thresholds in the tests means
a failing run can be re-run by hand and its numbers read directly:

    xvfb-run -a python tests/gl_probe.py physics path/to/Haru.model3.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from l2dprobe.backend import Live2DBackend  # noqa: E402

SIZE = 256


def _diff(a, b) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def physics(model: str) -> dict:
    """Does secondary motion keep moving after the head stops?

    Push the head and body to one side, hold, then release. With the solver
    live, hair and cloth keep swinging for a while after the release, so the
    frames right after it differ from each other. The same run with write
    order "after" is the control: that order resets every parameter after
    Update() -- physics outputs included -- so the post-release frames are
    frozen. The control proves the check can fail.
    """
    out = {}
    with Live2DBackend(model, size=SIZE, strip_dynamics=False, order="after",
                       fps=30) as be:
        out["order_after_open"] = be.order
        be.calibrate()
        out["order"] = be.order
        out["timestep"] = be._api_report.get("timestep", "")
        push = {}
        for pid in ("ParamAngleX", "ParamBodyAngleX", "ParamAngleZ"):
            spec = next((s for s in be.params if s.pid == pid), None)
            if spec is not None:
                push[pid] = spec.maximum
        out["pushed"] = sorted(push)

        def run(order):
            for _ in range(30):
                be.render({}, order=order)
            for _ in range(20):
                be.render(push, order=order)
            frames = [be.render({}, order=order) for _ in range(12)]
            # Skip the release frame itself: the head snaps back there in
            # either order, which is not secondary motion.
            return [_diff(frames[k], frames[k + 1]) for k in range(1, 11)]

        out["post_release_live"] = run(be.order)
        out["post_release_after"] = run("after")
    return out


def second_backend(model_a: str, model_b: str) -> dict:
    """Open two backends in one process; is the second one textured?

    Measures the mean RGB of drawn pixels (alpha > 0) in each. Tearing the GL
    context down between rigs used to leave the second drawing a correct
    silhouette in solid black.
    """
    out = {}
    for key, model in (("first", model_a), ("second", model_b)):
        with Live2DBackend(model, size=SIZE) as be:
            be.calibrate()
            img = be.render()
        drawn = img[..., 3] > 0
        out[key] = {"coverage": float(drawn.mean()),
                    "mean_rgb": float(img[..., :3][drawn].mean())
                    if drawn.any() else 0.0}
    return out


def make_rig(model: str, dest: str) -> dict:
    """Minimal rigs/<name>.npz for the runtime: ids, ranges, defaults.

    The runtime's --static path reads the rig file only for parameter ids,
    bands and ranges, so the descriptor can be empty. This keeps the
    PartOpacity test independent of a probe run.
    """
    from l2dprobe.analyze import group_of
    with Live2DBackend(model, size=SIZE) as be:
        specs = list(be.params)
        parts = be._api_report.get("parts", "")
    pids = [s.pid for s in specs]
    np.savez_compressed(
        dest, pid=np.array(pids, dtype=object),
        kind=np.array(["continuous"] * len(pids), dtype=object),
        group=np.array([group_of(p) for p in pids], dtype=object),
        minimum=np.array([s.minimum for s in specs], np.float32),
        maximum=np.array([s.maximum for s in specs], np.float32),
        default=np.array([s.default for s in specs], np.float32),
        descriptor=np.zeros((len(pids), 8), np.float32),
        mirror=np.full(len(pids), -1, np.int32))
    return {"n": len(pids), "dest": dest, "parts": parts}


def main(argv) -> int:
    cmd, args = argv[0], argv[1:]
    fn = {"physics": physics, "second_backend": second_backend,
          "make_rig": make_rig}[cmd]
    print(json.dumps(fn(*args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
