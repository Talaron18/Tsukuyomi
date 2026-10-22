"""
Render regressions. Each GL job runs in its own process (see gl_probe.py).

Needs live2d-py, a display or xvfb-run, and TSUKUYOMI_SAMPLES pointing at a
CubismWebSamples checkout; skips otherwise. Thresholds below were set from
measurements at 256 px with live2d-py 0.8.1 (numbers in each test) and leave
a wide margin on both sides, because the point is to separate "works" from
"broken", not to pin pixels.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from conftest import gl_probe, run_child, sample_model


def test_physics_live_with_fixed_step(samples, gl_prefix):
    """Dynamics on: order is never "after", the step is fixed, hair swings.

    Hiyori, 10 frame-to-frame diffs right after releasing a held head/body
    push. Measured: live 0.17-0.40 mean abs (uint8 units over RGBA), order
    "after" exactly 0.0 -- it resets the solver's outputs every frame, which
    is the bug this guards against. Threshold: live max > 0.05, control max
    < 0.01 (the control proves the check can fail).
    """
    r = gl_probe(gl_prefix, "physics", sample_model(samples, "Hiyori"))
    ts = r["timestep"]
    if "no fixed-step API" in ts:
        pytest.skip(f"this live2d-py build has no fixed-step Update ({ts})")
    assert r["order_after_open"] != "after", r
    assert r["order"] != "after", r
    assert ts.startswith("fixed"), f"fixed-step API exists but unused: {ts}"
    assert r["pushed"], "no head/body parameters found to push"
    assert max(r["post_release_live"]) > 0.05, r
    assert max(r["post_release_after"]) < 0.01, r


def test_second_backend_is_textured(samples, gl_prefix):
    """Two backends in one process; the second must not render black.

    Measured mean RGB of drawn pixels: Haru (first) 84, Hiyori (second) 161.
    The old failure drew the second rig's silhouette solid black (~0).
    """
    r = gl_probe(gl_prefix, "second_backend",
                 sample_model(samples, "Haru"), sample_model(samples, "Hiyori"))
    for key in ("first", "second"):
        assert r[key]["coverage"] > 0.02, r
        assert r[key]["mean_rgb"] > 40, r


def _no_pose_copy(model: str, dest_dir) -> str:
    """Haru without its pose3 file: same files, Pose line removed.

    The model3.json is edited as text: a json.dump rewrite made the Cubism
    parser reject the file ("Json parse error @line 1"). Every other entry is
    symlinked, so nothing is written into the samples checkout.
    """
    src = os.path.dirname(model)
    for name in os.listdir(src):
        if not name.endswith(".model3.json"):
            os.symlink(os.path.join(src, name), os.path.join(dest_dir, name))
    with open(model, encoding="utf-8") as fh:
        text = fh.read()
    out = re.sub(r'\s*"Pose"\s*:\s*"[^"]*"\s*,', "", text, count=1)
    if out == text:                       # Pose was the last entry
        out = re.sub(r',\s*"Pose"\s*:\s*"[^"]*"', "", text, count=1)
    assert out != text and '"Pose"' not in out
    dest = os.path.join(dest_dir, os.path.basename(model))
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(out)
    return dest


def _static_frame(prefix, model, rig, out, frame, extra=()):
    run_child(prefix, ["-m", "flow.runtime", "--model", model, "--rig", rig,
                       "--static", "--seconds", f"{(frame + 2) / 30:.3f}",
                       "--size", "256", "--no-autonomic", "--out", str(out),
                       *extra])
    from PIL import Image
    return np.asarray(Image.open(os.path.join(out, f"f{frame:05d}.png")),
                      dtype=np.int16)


def test_part_opacity_applied_without_pose(samples, gl_prefix, tmp_path):
    """HaruNoPose under --static matches Haru with its pose file.

    Live2D Native ignores PartOpacity curves on a rig with no pose3 file and
    draws every arm variant; the runtime applies them itself. The "native"
    side here is the runtime with --no-parts rather than Live2D's own motion
    player: it applies the same authored parameters in the same order and
    differs only in part opacity, whereas the native player picks a
    different motion and timing, so its diff would measure that instead.

    Frame 120, mean abs diff over RGBA. Measured: with pose vs without
    0.000; with pose vs --no-parts 0.46 (frames 0-240 all 0.46-0.48).
    Thresholds: < 0.1 and > 0.25.
    """
    haru = sample_model(samples, "Haru")
    nopose_dir = tmp_path / "HaruNoPose"
    nopose_dir.mkdir()
    nopose = _no_pose_copy(haru, str(nopose_dir))
    rig = str(tmp_path / "Haru.npz")
    gl_probe(gl_prefix, "make_rig", haru, rig)

    pose = _static_frame(gl_prefix, haru, rig, tmp_path / "pose", 120)
    fixed = _static_frame(gl_prefix, nopose, rig, tmp_path / "nopose", 120)
    native = _static_frame(gl_prefix, nopose, rig, tmp_path / "native", 120,
                           ["--no-parts"])
    d_fixed = float(np.abs(pose - fixed).mean())
    d_native = float(np.abs(pose - native).mean())
    assert d_fixed < 0.1, (d_fixed, d_native)
    assert d_native > 0.25, (d_fixed, d_native)
