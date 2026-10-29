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

  switch    The avatar may switch framing itself: a [frame bust] / [frame
            full] marker (flow.pose) becomes a "frame" token
            (frame_token), and Camera plays it as an ease of scale and
            offset between the two set_view states (VIEW_EASE seconds,
            minimum jerk) from the marker's time on. The offline render
            (flow.runtime) and the live render loop (flow.live) both ask
            Camera.at(t) each frame.
  budget    What a body can do, measured from the body file (budget): the
            share of a gesture zone each hand can reach (the per-variable
            ranges of body.limits combined as one box per arm drawing),
            how much of it the bust picture shows, the face controls the
            prompt offers. flow.pose words it as one prompt line.

Rendering only: the parameters, the body model and every measurement are
untouched.
"""

from __future__ import annotations

import os
import random
import zlib

import numpy as np

FRAMES = ("bust", "full")
ABOVE, BELOW = 1.5, 1.6
VIEW_EASE = 0.6          # seconds a [frame] switch takes
BUST_SHARE = 0.7         # sessions of generated data that start in bust
# The prompt's face controls (the same prefixes as flow.dataexport's face).
FACE_PREFIX = ("brow.", "eye.", "mouth.", "cheek.", "gaze.")
# The gesture zone a hand's reach is measured over, in hu from the chin: x
# outward from the midline on the hand's own side, y from where the lower
# hand hangs at rest to ABOVE_HEAD over the head landmark.
ZONE_X, ZONE_ABOVE_HEAD = (0.15, 1.1), 0.6
# Reach shares (zone share a hand reaches) that split the budget's tiers.
LIMITED, CAPABLE = 0.35, 0.8


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


# ----------------------------------------------------- what the picture shows

def window(above=ABOVE, below=BELOW) -> dict:
    """The bust picture in hu from the chin (the body frame): `top` and
    `bottom` y, `half_width` (the window is square on a square canvas, and
    centred on the chin horizontally)."""
    return {"top": float(above), "bottom": -float(below),
            "half_width": 0.5 * (float(above) + float(below))}


def visible(x, y, win=None) -> bool:
    """True when a point (hu from the chin) is inside the bust picture."""
    w = win or window()
    return w["bottom"] <= y <= w["top"] and abs(x) <= w["half_width"]


def can_frame(bm) -> bool:
    """True for a body whose framing the avatar may switch: a Live2D body
    with a frame (chin, hu) to place the window by. A VRM body is drawn by
    the browser page at one camera (flow.vrmbody.camera): always full."""
    return not hasattr(bm, "vrm") and "frame" in (bm.blob or {})


def _boxes(bm, side):
    """[(y lo, y hi, x lo, x hi)] per arm drawing of a hand: the reach of
    each variable alone (body.limits), combined as a box."""
    out = []
    for k in range(len(bm.variants)):
        ly = bm.limits(f"hand.{side}.y", k) if f"hand.{side}.y" in bm.fidx \
            else None
        lx = bm.limits(f"hand.{side}.x", k) if f"hand.{side}.x" in bm.fidx \
            else None
        if ly is not None and lx is not None:
            out.append((ly[0], ly[1], lx[0], lx[1]))
    return out


def rest_hands(bm) -> dict:
    """{"L"/"R": (x, y)} where the hands rest (the first arm drawing)."""
    v0, out = bm.variants[0], {}
    for s in "LR":
        ix, iy = f"hand.{s}.x", f"hand.{s}.y"
        if ix in bm.fidx and v0["present"][bm.fidx[ix]]:
            out[s] = (float(v0["rest"][bm.fidx[ix]]),
                      float(v0["rest"][bm.fidx[iy]]))
    return out


def hand_reach(bm, win=None) -> dict:
    """The measured reach of the hands over the gesture zone (ZONE_*): the
    share of its grid points (8 heights x 6 widths, each side) that some
    arm drawing's box holds, "all"; the share of the points reached that the
    bust picture shows (win), "bust"; the lowest and highest y a hand
    reaches (any drawing); the arm drawings."""
    w = win or window()
    L = bm.landmarks
    rest = rest_hands(bm)
    y0 = min([y for _x, y in rest.values()] or [L["chest"][1] - 0.5])
    y1 = L["head"][1] + ZONE_ABOVE_HEAD
    ys = np.linspace(y0, y1, 8)
    xs = np.linspace(ZONE_X[0], ZONE_X[1], 6)
    tot = hit = bhit = 0
    lo, hi = [], []
    for s in "LR":
        boxes = _boxes(bm, s)
        sg = 1.0 if s == "L" else -1.0
        for b in boxes:
            lo.append(b[0])
            hi.append(b[1])
        for y in ys:
            for x in xs:
                X = sg * x
                h = bool(any(b[0] <= y <= b[1] and b[2] <= X <= b[3]
                             for b in boxes))
                tot += 1
                hit += h
                bhit += h and visible(X, y, w)
    return {"all": round(hit / max(tot, 1), 3),
            "bust": round(bhit / max(hit, 1), 3),
            "low": round(float(min(lo)), 3) if lo else None,
            "high": round(float(max(hi)), 3) if hi else None,
            "drawings": len(bm.variants)}


def budget(bm, above=ABOVE, below=BELOW) -> dict:
    """What this body can do, measured from its body file (the numbers
    flow.pose words as the motion-budget line, and avatars.json keeps):
      kind          "live2d" / "vrm"
      face          the face controls the prompt offers (brow, eye, mouth,
                    cheek, gaze variables of pose.offered)
      offered       all variables it offers
      torso         torso variables offered
      reach         hand_reach(): zone shares "all" and "bust", lowest and
                    highest hand y, arm drawings
      tier          "limited" (< LIMITED of the zone), "capable" (>= CAPABLE)
                    or "moderate"
      hand_moves    most hand moves a reply per framing, {"bust", "full"}:
                    limited 1 / 1, moderate 1 / 2, capable 1 / 3; 0 in
                    bust where a hand reaches nothing the picture shows
      window        the bust picture (window())
      rest_visible  {side: the resting hand is in the bust picture}
      frames        the framings it can switch between
    """
    try:
        from . import pose as pz
    except ImportError:
        from flow import pose as pz
    off = pz.offered(bm)
    w = window(above, below)
    r = hand_reach(bm, w)
    tier = ("limited" if r["all"] < LIMITED else
            "capable" if r["all"] >= CAPABLE else "moderate")
    full = {"limited": 1, "moderate": 2, "capable": 3}[tier]
    return {"kind": "vrm" if hasattr(bm, "vrm") else "live2d",
            "face": sum(v.startswith(FACE_PREFIX) for v in off),
            "offered": len(off),
            "torso": sum(v.startswith("torso.") for v in off),
            "reach": r, "tier": tier,
            "hand_moves": {"bust": 1 if r["bust"] > 0 and can_frame(bm)
                           else 0, "full": full},
            "window": w,
            "rest_visible": {s: visible(x, y, w)
                             for s, (x, y) in rest_hands(bm).items()},
            "frames": list(FRAMES) if can_frame(bm) else ["full"]}


# --------------------------------------------------------------- switching

def frame_token(view, t, ease=VIEW_EASE) -> dict:
    """The event-stream token of a [frame] marker acted at stream time t."""
    return {"t": round(float(t), 4), "type": "frame", "role": "camera",
            "view": view, "duration": float(ease), "src": "director"}


def frame_tokens(acts, t0=0.0) -> list:
    """Tokens of the "frame" acts among `acts` ({"t", "tag", "arg"}), t0
    seconds later."""
    return [frame_token(x["arg"], float(x["t"]) + t0) for x in acts
            if x.get("tag") == "frame" and x.get("arg") in FRAMES]


def last_frame(tags, default=None):
    """The framing after a reply: that of its last [frame] marker (tags or
    acts, in order), else `default`."""
    cur = default
    for t in tags:
        if t.get("tag") == "frame" and t.get("arg") in FRAMES:
            cur = t["arg"]
    return cur


def switches(tags, start) -> int:
    """How many times a reply's frame markers change the framing it
    starts in."""
    cur, n = start, 0
    for t in tags:
        if t.get("tag") == "frame" and t.get("arg") in FRAMES:
            n += t["arg"] != cur
            cur = t["arg"]
    return n


def start_frame(seed, sid, share=BUST_SHARE) -> str:
    """The framing a generated session starts in: bust with probability
    `share`. Drawn from its own generator of the run seed and session id,
    so no other draw of the session moves."""
    r = random.Random(zlib.crc32(f"{seed}:{sid}:frame".encode()) & 0x7FFFFFFF)
    return "bust" if r.random() < share else "full"


class Camera:
    """The view over time: `views` {"bust": (scale, off_x, off_y), "full":
    ...} the set_view states, `start` the framing at t = 0. commit() adds
    a switch at a time (stream seconds); at(t) is the view then, each
    switch an ease (minimum jerk) of scale and offset over `ease` seconds
    from wherever the view was when it began (a switch back mid-ease
    turns around smoothly). Pure in t: any order of at() calls agrees."""

    def __init__(self, views, start="full", ease=VIEW_EASE):
        self.views = {k: tuple(float(x) for x in v) for k, v in views.items()}
        self.start = start if start in self.views else "full"
        self.ease = float(ease)
        self.segs = []                       # (t0, from view, to name, ease)

    def commit(self, t, view, ease=None):
        if view not in self.views:
            return False
        t = float(t)
        if self.segs:
            t = max(t, self.segs[-1][0])
        self.segs.append((t, self.at(t), view,
                          self.ease if ease is None else float(ease)))
        return True

    def at(self, t):
        cur = self.views[self.start]
        for t0, frm, to, ease in reversed(self.segs):
            if t >= t0:
                u = (t - t0) / ease if ease > 0 else 1.0
                if u >= 1.0:
                    return self.views[to]
                a = float(_ease(u))
                return tuple(f + a * (g - f)
                             for f, g in zip(frm, self.views[to]))
        return cur

    def target(self):
        """The framing the last switch heads to (the start when none)."""
        return self.segs[-1][2] if self.segs else self.start

    def commit_token(self, k):
        return self.commit(k["t"], k.get("view"), k.get("duration"))


def _ease(u):
    try:
        from .behaviors import min_jerk
    except ImportError:
        from flow.behaviors import min_jerk
    return min_jerk(u)


def camera_for(backend, body_path, model_json, start="bust", log=print):
    """(Camera, the views) for a Live2D backend: bust_view's state for
    "bust" and the full canvas, the backend left in `start`. None when
    there is no bust view (then the render stays full)."""
    v = bust_view(backend, body_path, model_json, log=log)
    if v is None:
        return None
    cam = Camera({"bust": v, "full": (1.0, 0.0, 0.0)}, start)
    backend.set_view(*cam.at(0.0))
    return cam


def view_tokens(tokens, cam):
    """Commit the "frame" tokens among `tokens` (sorted by time) to a
    Camera; the number committed."""
    n = 0
    for k in sorted((k for k in tokens if k.get("type") == "frame"),
                    key=lambda k: float(k["t"])):
        n += bool(cam.commit_token(k))
    return n
