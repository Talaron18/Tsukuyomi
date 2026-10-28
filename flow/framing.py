#!/usr/bin/env python3
"""
Camera framing for the Live2D renders: a "bust" view (head and chest, as a
streamer's camera shows them) instead of the whole canvas.

  how       The body file (flow.body) has the chin and the head height (hu)
            in model units. The Core's visible vertices at rest give the
            figure's extent in model units, the rest render's alpha its
            extent in pixels: two points per axis fix the linear map from
            model units to pixels at scale 1. The window kept is from
            `above` hu over the chin to `below` hu under it (default 1.5 and
            1.6: hair and ears to the middle of the chest), centred on the
            chin horizontally; live2d-py's SetScale scales about the canvas
            centre and SetOffset then shifts by half-canvas units (+x right,
            +y up), measured on Hiyori: an offset of 0.5 moves the figure
            128 px of 512 whatever the scale.
  full      --frame full keeps the old whole-canvas view (scale 1, no
            offset).

Rendering only: the parameters, the body model and every measurement are
untouched.
"""

from __future__ import annotations

import os

import numpy as np

FRAMES = ("bust", "full")
ABOVE, BELOW = 1.5, 1.6


def body_path_for(rig: str | None, model_json: str | None = None):
    """The body file for a render: <rig's folder>/<rig name>.body.json
    (rigs/Hiyori.npz -> rigs/Hiyori.body.json), else rigs/<model name>.
    body.json; None when neither exists."""
    cands = []
    if rig:
        cands.append(os.path.splitext(rig)[0] + ".body.json")
    if model_json:
        stem = os.path.basename(model_json)
        for ext in (".model3.json", ".json"):
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
                break
        cands.append(os.path.join("rigs", f"{stem}.body.json"))
    return next((c for c in cands if os.path.exists(c)), None)


def _alpha_box(img):
    a = img[..., 3] > 10
    if not a.any():
        return None
    ys, xs = np.where(a)
    return ys.min(), ys.max(), xs.min(), xs.max()


def bust_view(backend, body_path, model_json, above=ABOVE, below=BELOW,
              log=print):
    """(scale, offset_x, offset_y) for backend.set_view, or None when the
    pieces are missing (then the caller keeps the full view)."""
    import json
    try:
        from l2dprobe import core
    except ImportError:
        return None
    if not body_path or not os.path.exists(body_path):
        log("framing: no body file; full view")
        return None
    with open(body_path, encoding="utf-8") as fh:
        frame = json.load(fh).get("frame") or {}
    if "origin" not in frame or "unit" not in frame:
        log("framing: the body file has no frame; full view")
        return None
    ox, oy = (float(v) for v in frame["origin"])
    unit = float(frame["unit"])
    backend.set_view(1.0, 0.0, 0.0)
    box = _alpha_box(backend.render())
    if box is None:
        return None
    m = core.load(model_json)
    m.set({})
    op = m.opacities()
    vis = [m.vertices(i) for i in range(len(m.drawable_ids)) if op[i] > 0.5]
    if not vis:
        return None
    v = np.concatenate(vis)
    (x0, y0), (x1, y1) = v.min(0), v.max(0)
    t, b, l, r = box
    if y1 - y0 <= 0 or x1 - x0 <= 0:
        return None
    # pixels = p0 + k * model (y flipped)
    ky, kx = (b - t) / (y1 - y0), (r - l) / (x1 - x0)
    k = 0.5 * (ky + kx)                         # one scale for both axes
    py = lambda y: t + (y1 - y) * k             # noqa: E731
    px = lambda x: l + (x - x0) * k             # noqa: E731
    top, bot = py(oy + above * unit), py(oy - below * unit)
    size = backend.size
    h = bot - top
    if h <= 0:
        return None
    s = max(1.0, size / h)
    cy, cx = 0.5 * (top + bot), px(ox)
    half = size / 2.0
    off_y = s * (cy - half) / half
    off_x = -s * (cx - half) / half
    log(f"framing: bust, {above}+{below} hu around the chin -> scale "
        f"{s:.2f}, offset ({off_x:+.2f}, {off_y:+.2f})")
    return s, off_x, off_y
