#!/usr/bin/env python3
"""
Motion primitives: rhythmic, continuous movement the language model calls
by name next to its poses, written inline like a pose marker:

    Let's dance! [move {"name": "dance", "beats": 8, "tempo": 120}] Come on!

Body mode (flow.pose) let the model place its body by numbers: precise
asks ("tilt your head", "raise your right hand", "hands together") came out
well, open ones ("dance for me") as a few held poses one after another --
nothing on the beat, nothing continuous. A pose is a target; a dance is a
curve. Here a primitive is a curve over canonical variables (flow.body's
frame: hu, degrees, the character's own left and right), so the same name
plays on any body, Live2D or VRM, solved on that body.

  primitives  PRIMS below: sway, bounce, head_bob, shimmy, look_around (the
            head and trunk), clap, cheer, wave, twirl (the arms) and dance,
            a sequence of them. Each is a function of the beat phase b (b =
            1 on the first beat after its start): additive offsets on
            declared variables (head, torso, shoulders, gaze, brows; and on
            a VRM the whole body's sway and bob, "hips.x" / "hips.y") and
            absolute hand targets (hand.S.x/y, on a VRM also z and the
            "@together" composite for a clap). Periods are whole beats or
            simple fractions of one: sway 2 beats, look_around 4, shimmy a
            half, the rest one. Shapes are sinusoids on the beat (pulse(b)
            = 1 on the beat, 0 between) and, for the clap, |sin| -- a kink
            at contact, the impact, not a soft touch.
            Chosen for what open requests need ("dance", "celebrate",
            "cheer", "groove", "I'm so excited") and what the bodies here
            can show from the front: side to side (sway), up and down on the
            beat (bounce, head_bob), quick and small (shimmy), looking about
            (look_around), and four arm figures. nod_along is head_bob at
            half tempo and smaller, fist_pump is cheer, arm_wave is wave:
            aliases, not entries, so the menu stays short and the parser
            still takes what a model writes.
  capability  caps(bm), once per body (cached): what each primitive needs
            (the variables of its main motion; for an arm figure, hands
            that reach its targets). A VRM body reaches everything. A Live2D
            body draws its arms in a few variants with limited reach: an arm
            figure is tried in each variant whose reach box covers its
            heights, its targets clipped into that box, and is offered when
            the hands still travel half of what the figure means (and, for a
            clap, meet); cheer and wave per side. The prompt lists only what
            passes (prompt_lines); the parser drops the rest with a warning;
            a side a body lacks is degraded to the side it has, logged.
  clip      build(): keyframes every 1/K of a beat (K = 8, more for a
            faster period: at least 8 per period, so a shimmy gets 16 a
            beat), additive variables mapped to parameters (flow.body's
            declared map: ParamAngleZ = -head.tilt...; a VRM's are its own),
            hand targets solved with the body's own solver on each keyframe
            -- a VRM's collision-aware IK (a clap's hands meet, never pass),
            a Live2D forward-model solve with the variant fixed -- the prior
            the previous keyframe. Repeating keyframes (a periodic figure)
            are solved once: an 8-beat clap is 8 solves, not 64. One cycle
            is solved twice when its first keyframe, solved again from its
            last, lands elsewhere (the swivel of a VRM elbow can).
            dance(): segments of 2 or 4 beats, each a trunk figure plus an
            arm figure (or none) from what the body can do, no figure twice
            in a row, mixed over half a beat at each join -- in target space
            and solved there, so a mix between two figures is itself a
            collision-checked pose (a mix of joint angles need not be).
  play      MoveLayer: per parameter the clip's keyframes through a
            monotone cubic (PCHIP: no overshoot past a keyframe; a
            Catmull-Rom spline overshot the clap's contact into the other
            hand), under an envelope that eases in over the first beat and
            out over the last (min-jerk, 0.3 to 0.8 s). Additive channels
            add to the frame the idle built (idle head moves, saccades,
            breath keep going under a head_bob); absolute ones blend from
            it. A later move on the same parameters takes them over as it
            eases in (e_i = w_i * prod over later j of (1 - w_j)), so
            nothing jumps. A Live2D arm variant swaps its parts at the
            envelope's midpoints, as a pose does. The pose layer runs after
            it: a pose during a move takes over the parameters it plays and
            gives them back as it releases; held, it keeps no LIFE on the
            joints a move drives absolutely (a share of a clap leaking into
            a held hand would be a joint-space mix the IK never checked).
  timing    a marker is a tag ({"tag": "move", "arg": {...}, "at": n}),
            timed from the words or moras as poses are; solve() schedules
            moves on shared parameters one after another (the later easing
            in over the earlier's ease-out), on different ones together.

Without [move] markers nothing here runs: tag mode and body-mode outputs are
the bytes they were, and --no-body-moves gives the prompt and parser of
before (flow.pose.system_prompt / parser_for without moves).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time

import numpy as np

try:
    from . import body as fb
    from .behaviors import min_jerk
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import body as fb
    from flow.behaviors import min_jerk

MOVE = "move"
TEMPO, TEMPO_RANGE = 110.0, (60.0, 180.0)
BEATS, BEATS_RANGE = 8.0, (1.0, 32.0)
AMP, AMP_RANGE = 0.7, (0.1, 1.0)
SIDES = ("both", "L", "R")
# Keyframes per beat, and at least KF per period (a shimmy's half beat).
KF, KF_MAX = 8, 16
# The ease in and out: one beat, kept between these (s). At 110 bpm one
# beat is 0.55 s, about the fitted idle head move's median 1.3 s halved.
RAMP_RANGE = (0.3, 0.8)
ARM_RAMP = 0.9          # s: the ease of a clip that moves an arm
# Joint change (deg / units) between a cycle's first keyframe solved from
# the rest and from the cycle's own end above which the cycle is solved
# again from its end.
RESOLVE_JUMP = 5.0
# Hips sway and bob are a VRM render offset of the whole avatar (hu); the
# frame dict carries them only while a move writes them.
HIPS = ("hips.x", "hips.y")
# A hand no figure holds, under a trunk figure on a VRM, is carried this
# share of the way the chest would carry it (the rest hangs): at 0.5 the
# targets sat in the hips' surface and each solve searched its way out
# (51 ms a keyframe, 2.8 s for a 16-beat dance).
HANG = 0.25
# A dance's figures are mixed over this many beats at each join, in target
# space. Half a beat moved a fist from its peak to the hips in 8 frames.
MIX = 1.0
BULGE = 0.35            # hu forward at the middle of a mix
# Live2D arm solves: targets weighed 0.3 against the prior (continuity
# from the previous keyframe; Hiyori's dance: largest step between
# keyframes 6.3 -> 2.8 on ParamHandRB, span 20), and at most this many
# spans a second (a wave at 120 bpm needs 1.6).
L2D_WEIGHT, L2D_RATE = 0.3, 2.5
# A Live2D arm figure is offered when its hands travel at least this share
# of the figure's own travel within one drawing, and a clap's hands come
# this close (hu).
TRAVEL_MIN = 0.5
CLAP_GAP = 0.25
# ... or at least this far (hu): a fist pump within a small drawing.
TRAVEL_HU = 0.25
ALIASES = {"nod_along": ("head_bob", {"tempo_scale": 0.5, "amp_scale": 0.7}),
           "nod": ("head_bob", {"tempo_scale": 0.5, "amp_scale": 0.7}),
           "fist_pump": ("cheer", {}), "pump": ("cheer", {}),
           "arm_wave": ("wave", {}), "wave_hand": ("wave", {}),
           "shoulder_shake": ("shimmy", {}), "shake_shoulders": ("shimmy", {}),
           "twirl_hands": ("twirl", {}), "roll": ("twirl", {}),
           "bop": ("head_bob", {}), "groove": ("bounce", {}),
           "look": ("look_around", {}), "celebrate": ("dance", {}),
           "hands_up": ("cheer", {"side": "both"})}


# ----------------------------------------------------------------- shapes

def pulse(b):
    """1 on the beat (b whole), 0 half way between."""
    return 0.5 * (1.0 + math.cos(2.0 * math.pi * b))


def swing(b, period):
    return math.sin(2.0 * math.pi * b / period)


class Ctx:
    """What a primitive reads of a body: landmarks and the variables it
    has, in its own frame (hu, degrees)."""

    def __init__(self, bm):
        L = bm.landmarks
        self.vrm = getattr(bm, "kind", None) == "vrm"
        self.has = set(available_vars(bm))
        if self.vrm:
            self.has |= set(HIPS)
        self.head_y = float(L.get("head", [0, 0.45])[1])
        self.sh_y = float(L.get("shoulder.y", -0.25))
        self.chest = list(L.get("chest", [0.0, self.sh_y - 0.4]))
        self.sh_x = {s: float(L.get(f"shoulder.{s}", [(1 if s == "L" else -1)
                                                       * 0.4, 0])[0])
                     for s in "LR"}
        self.depth = getattr(bm, "depth", {}) or {}
        self.bm = bm
        # Live2D draws pitch about half as deep as a VRM head turns
        # (flow.vrm.ROLE_MAP: head_y 1 = 15 deg on a VRM, ParamAngleY 30):
        # a Live2D body gets twice the number for the same look.
        self.scale = {} if self.vrm else {"head.pitch": 2.0}

    def sgn(self, s):
        return 1.0 if s == "L" else -1.0


def _clap_y(c):
    return c.chest[1] + 0.55 * (c.sh_y - c.chest[1])


def _twirl_c(c):
    return (0.75 * (abs(c.sh_x["L"]) + abs(c.sh_x["R"])) / 2,
            c.chest[1] + 0.45 * (c.sh_y - c.chest[1]))


# Each primitive: f(b, amp, side, ctx) -> (additive {var: offset},
# hand targets {var: value}). Signs as flow.body: x toward the character's
# own left, head.tilt / torso.tilt + toward its right.

def p_sway(b, a, side, c):
    s = swing(b, 2.0)
    add = {"torso.tilt": 9.0 * a * s, "torso.yaw": 2.0 * a * s,
           "head.tilt": -4.0 * a * s, "head.yaw": -1.5 * a * s}
    if c.vrm:
        # The hips go the other way from the lean (a hip sway), and dip a
        # little on every step.
        add["hips.x"] = 0.08 * a * s
        add["hips.y"] = -0.02 * a * pulse(b)
    return add, {}


def p_bounce(b, a, side, c):
    d = pulse(b)
    add = {"head.pitch": -4.0 * a * pulse(b - 0.1),
           "shoulders.up": 0.3 * a * (0.5 - d)}
    if c.vrm:
        add["hips.y"] = -0.06 * a * d
        add["torso.pitch"] = -2.5 * a * d
    else:
        add["torso.pitch"] = -4.0 * a * d
        add["shoulders.up"] = 0.5 * a * (0.5 - d)
    return add, {}


def p_head_bob(b, a, side, c):
    return {"head.pitch": -8.0 * a * pulse(b - 0.05),
            "torso.pitch": -1.5 * a * pulse(b)}, {}


def p_shimmy(b, a, side, c):
    s = swing(b, 0.5)
    add = {"torso.yaw": 4.0 * a * s, "head.yaw": -2.0 * a * s}
    if {"shoulder.L.up", "shoulder.R.up"} <= c.has:
        add.update({"shoulder.L.up": 0.6 * a * s, "shoulder.R.up": -0.6 * a * s})
    else:
        add["shoulders.up"] = 0.4 * a * abs(s) - 0.2 * a
    return add, {}


def p_look_around(b, a, side, c):
    s = swing(b, 4.0)
    return {"head.yaw": 22.0 * a * s, "torso.yaw": 4.0 * a * s,
            "gaze.x": 0.5 * a * swing(b + 0.3, 4.0),
            "head.pitch": 3.0 * a * swing(b, 2.0)}, {}


def p_clap(b, a, side, c):
    sep = abs(math.sin(math.pi * b))
    add = {"head.pitch": -3.0 * a * pulse(b), "shoulders.up": 0.1 * a
           * (1.0 - sep)}
    w = 0.08 + 0.32 * a * sep
    if c.vrm:
        hands = {"hands": "@together", "hand.L.x": w, "hand.R.x": -w}
    else:
        x0, y = float(c.chest[0]), _clap_y(c)
        hands = {"hand.L.x": x0 + w - 0.06, "hand.L.y": y,
                 "hand.R.x": x0 - w + 0.06, "hand.R.y": y}
    return add, hands


def _side_list(side):
    return ["L", "R"] if side == "both" else [side]


def p_cheer(b, a, side, c):
    d = pulse(b)
    add = {"head.pitch": 5.0 * a * d, "torso.pitch": 1.5 * a * d,
           "brow.L.lift": 0.4 * a * d, "brow.R.lift": 0.4 * a * d}
    hands = {}
    # From beside the head to above it: a pump of ~0.6 hu at amp 0.7.
    # From the shoulder (1.4 hu) the fist crossed it in 8 frames at 110 bpm,
    # 0.47 hu a frame -- a blur, not a pump.
    # Out to the side and up, a V: a fist by the ear read as a hand on the
    # head (Seed-san's first render).
    low, top = c.head_y - 0.1, c.head_y + 0.35 + 0.5 * a
    for s in _side_list(side):
        hands[f"hand.{s}.x"] = c.sh_x[s] * (1.9 - 0.4 * d)
        hands[f"hand.{s}.y"] = low + (top - low) * d
        if c.vrm:
            hands[f"hand.{s}.z"] = float(c.depth.get("chest", 0.3)) + 0.1
    return add, hands


def p_wave(b, a, side, c):
    s = swing(b, 1.0)
    sides = _side_list(side)
    add = {"head.tilt": 2.0 * a * s}
    if len(sides) == 2:
        add["torso.tilt"] = 3.0 * a * s
    hands = {}
    for k in sides:
        sg = c.sgn(k)
        # One hand well out beside the head: at 1.5 shoulder widths its
        # inward swing put the palm on the forehead (Seed-san's dance).
        hands[f"hand.{k}.x"] = sg * abs(c.sh_x[k]) * (2.0 if len(sides) == 1
                                                       else 1.3) \
            + 0.25 * a * s
        hands[f"hand.{k}.y"] = c.head_y + (0.1 if len(sides) == 1 else 0.6)
        if c.vrm:
            hands[f"hand.{k}.z"] = float(c.depth.get("chest", 0.3)) + 0.05
    return add, hands


def p_twirl(b, a, side, c):
    th = 2.0 * math.pi * b
    cx, cy = _twirl_c(c)
    r = 0.04 + 0.16 * a
    hands = {"hand.L.x": cx + r * math.cos(th), "hand.L.y": cy + r * math.sin(th),
             "hand.R.x": -cx + r * math.cos(th),
             "hand.R.y": cy - r * math.sin(th)}
    if c.vrm:
        bm = c.bm
        for s in "LR":
            x, y = hands[f"hand.{s}.x"], hands[f"hand.{s}.y"]
            z = bm._surface_z(x, y) if hasattr(bm, "_surface_z") else None
            hands[f"hand.{s}.z"] = (z if z is not None else 0.3) + 0.45
    return {"head.pitch": -2.0 * a * pulse(b)}, hands


# name: (function, period in beats, kind, main variables, one-line meaning)
PRIMS = {
    "sway": (p_sway, 2.0, "trunk", ("torso.tilt", "torso.yaw"),
             "sway your upper body side to side, your head countering it"),
    "bounce": (p_bounce, 1.0, "trunk", ("torso.pitch", "shoulders.up",
                                         "hips.y"),
               "bounce down on every beat (knees, shoulders, a little nod)"),
    "head_bob": (p_head_bob, 1.0, "head", ("head.pitch",),
                 "bob your head on the beat, as to music"),
    "shimmy": (p_shimmy, 0.5, "trunk", ("torso.yaw", "shoulder.L.up"),
               "shake your shoulders quickly, twice a beat"),
    "look_around": (p_look_around, 4.0, "head", ("head.yaw",),
                    "look slowly around, side to side"),
    "clap": (p_clap, 1.0, "arms", (),
             "clap your hands on every beat, in front of your chest"),
    "cheer": (p_cheer, 1.0, "arms", (),
              "pump your fist up on every beat (\"side\" L, R or both)"),
    "wave": (p_wave, 1.0, "arms", (),
             "wave a raised hand side to side (\"side\" L or R, default R; "
             "both: both hands up, swaying together)"),
    "twirl": (p_twirl, 1.0, "arms", (),
              "roll your hands around each other in front of your chest"),
}
TRUNK = ("bounce", "sway", "shimmy")
ARM_FIGS = ("clap", "cheer", "wave", "twirl")
ORDER = ("sway", "bounce", "head_bob", "shimmy", "look_around", "clap",
         "cheer", "wave", "twirl", "dance")
DANCE_DESC = ("a little dance: the motions above in a row, different each "
              "time")


# --------------------------------------------------------------- the body

def available_vars(bm):
    """The canonical variables a primitive may drive on this body: the
    declared ones some variant has."""
    out = set()
    for f in bm.features:
        k = bm.fidx[f]
        if any(v["present"][k] for v in bm.variants):
            out.add(f)
    return out


def _declared(bm):
    """{var: (param, scale)}: flow.body's declared map (a VRM's are its own
    parameters at scale 1)."""
    out = {}
    for var, d in (bm.blob.get("declared") or {}).items():
        if isinstance(d, (list, tuple)) and len(d) >= 2 and d[0] in bm.params:
            out[var] = (d[0], float(d[1]))
    return out


def _arm_set(bm):
    """Parameters an arm figure may play absolutely: those that move a
    hand (flow.pose.arm_params) but no declared head / torso / shoulder /
    face parameter (the trunk figures add to those)."""
    try:
        from . import pose as pz
    except ImportError:
        from flow import pose as pz
    dec = _declared(bm)
    trunk = {dec[n][0] for n, *_ in fb.DECLARED if n in dec}
    return set(pz.arm_params(bm)) - trunk


def _reach_box(bm, k, s):
    """(x lo, x hi, y lo, y hi) of hand s in Live2D variant k moved by its
    arm alone (_arm_set): a figure plays its hands with the arm, the
    trunk figures having the body angles (Hiyori's raised drawing reaches
    y -0.32 to -0.02 with the body leaning and the shoulder up; with the
    arm alone -0.19 to -0.15). Cached on the body."""
    cache = bm.__dict__.setdefault("_move_reach", {})
    if (k, s) in cache:
        return cache[(k, s)]
    v = bm.variants[k]
    out = None
    if all(v["present"][bm.fidx[f"hand.{s}.{q}"]] for q in "xy"):
        only = _arm_set(bm)
        ends = []
        for q in "xy":
            got = []
            for far in (-50.0, 50.0):
                r = bm._solve_variant({f"hand.{s}.{q}": far}, {}, k, {}, only)
                got.append(None if r is None else
                           r["report"][f"hand.{s}.{q}"]["got"])
            if None in got:
                break
            ends += [float(min(got)), float(max(got))]
        if len(ends) == 4:
            out = tuple(ends)
    cache[(k, s)] = out
    return out


def _clip_hands(hands, boxes, shift=None):
    """Hand targets moved by `shift` ({side: (dx, dy)}: the figure placed
    where this drawing reaches) and clipped into each side's reach box."""
    out = {}
    for k, v in hands.items():
        if k.startswith("hand.") and k[-2:] in (".x", ".y") and \
                boxes.get(k[5]) is not None:
            x0, x1, y0, y1 = boxes[k[5]]
            ax = k.endswith(".x")
            lo, hi = (x0, x1) if ax else (y0, y1)
            v = v + (shift or {}).get(k[5], (0.0, 0.0))[0 if ax else 1]
            m = 0.02 * (hi - lo)
            v = float(np.clip(v, lo + m, hi - m))
        out[k] = v
    return out


def _cycle(name, side, ctx, n=8):
    fn, period, *_ = PRIMS[name]
    return [fn(float(b), 1.0, side, ctx)[1]
            for b in np.arange(n) * period / n]


def _fit(name, side, boxes, ctx):
    """{side: (dx, dy)} placing an arm figure inside the reach boxes: a
    figure that fits is moved in whole (a fist pump at the height this
    drawing has, not squashed against its edge), one larger than the box
    keeps its top (the pump's peak, the wave's height). Two-hand figures
    move up or down together and never sideways (the hands must meet)."""
    cyc = _cycle(name, side, ctx)
    sides = sorted({k[5] for h in cyc for k in h if k.startswith("hand.")})
    sh = {}
    for s in sides:
        xs = [h[f"hand.{s}.x"] for h in cyc if f"hand.{s}.x" in h]
        ys = [h[f"hand.{s}.y"] for h in cyc if f"hand.{s}.y" in h]
        x0, x1, y0, y1 = boxes[s]

        def need(a, b, lo, hi):
            if b - a > hi - lo:
                return hi - b
            return (lo - a) if a < lo else (hi - b) if b > hi else 0.0
        sh[s] = (need(min(xs), max(xs), x0, x1),
                 need(min(ys), max(ys), y0, y1))
    if name in ("clap", "twirl"):
        dy = float(np.mean([sh[s][1] for s in sides]))
        sh = {s: (0.0, dy) for s in sides}
    return sh


def _score(name, side, boxes, ctx):
    """Share of the figure's extent inside the boxes (no solve)."""
    cyc = _cycle(name, side, ctx)
    tot = 0.0
    for s, bx in boxes.items():
        xs = [h[f"hand.{s}.x"] for h in cyc if f"hand.{s}.x" in h]
        ys = [h[f"hand.{s}.y"] for h in cyc if f"hand.{s}.y" in h]
        if not xs:
            continue
        x0, x1, y0, y1 = bx
        ox = max(0.0, min(max(xs), x1) - max(min(xs), x0)) + 0.05
        oy = max(0.0, min(max(ys), y1) - max(min(ys), y0)) + 0.05
        # A box that cannot hold the figure's height still lets it move
        # there; one far from it is worth less.
        tot += min(ox / (np.ptp(xs) + 0.05), 1) * min(
            (y1 - y0 + 0.05) / (np.ptp(ys) + 0.05), 1) * (
            1.0 if oy > 0.05 else 0.5)
    return tot


def _probe(bm, ctx, name, side, k):
    """(ok, travel, why, shift) of an arm figure in Live2D variant k: one
    cycle at full size, placed and clipped into the variant's reach
    (_fit), solved there."""
    fn, period, *_ = PRIMS[name]
    sides = ("L", "R") if name in ("clap", "twirl") else tuple(_side_list(side))
    boxes = {s: _reach_box(bm, k, s) for s in sides}
    if any(b is None for b in boxes.values()):
        return False, 0.0, "a hand it needs is not drawn", None
    shift = _fit(name, side, boxes, ctx)
    want, got = [], []
    prev = None
    for b in np.arange(0.0, period, period / 3) + 0.5 * (name == "clap"):
        _a, hands = fn(float(b), 1.0, side, ctx)
        tg = _clip_hands(hands, boxes, shift)
        r = bm.solve(tg, current=prev, variant=k, only=_arm_set(bm))
        if r.get("variant") is None:
            return False, 0.0, "no drawing has these hands", None
        prev = {**bm.default_values(), **r["params"]}
        want.append([hands[f"hand.{s}.{q}"] for s in sides for q in "xy"])
        got.append([r["report"][f"hand.{s}.{q}"]["got"] or 0.0
                    for s in sides for q in "xy"])
    want, got = np.array(want), np.array(got)
    tw = float(np.ptp(want, axis=0).sum())
    tg_ = float(np.ptp(got, axis=0).sum())
    if name == "clap":
        r = bm.solve(_clip_hands(fn(0.0, 1.0, side, ctx)[1], boxes, shift),
                     current=prev, variant=k, only=_arm_set(bm))
        gap = abs(r["report"]["hand.L.x"]["got"] - r["report"]["hand.R.x"]
                  ["got"])
        dy = abs(r["report"]["hand.L.y"]["got"] - r["report"]["hand.R.y"]
                 ["got"])
        if gap > CLAP_GAP or dy > CLAP_GAP:
            return False, tg_, (f"the hands do not meet (they stay "
                                f"{gap:.2f} hu apart)"), None
    if name in ("cheer", "wave"):
        top = min(got[:, 2 * i + 1].max() for i in range(len(sides)))
        floor = ctx.sh_y - (0.1 if name == "wave" else -0.05)
        if top < floor:
            return False, tg_, (f"the hand does not get above shoulder "
                                f"height (highest {top:.2f})"), None
    if tg_ < min(TRAVEL_MIN * tw, TRAVEL_HU):
        return False, tg_, (f"the hands move {tg_:.2f} hu of the figure's "
                            f"{tw:.2f} within reach"), None
    return True, tg_, "", shift


def caps(bm):
    """{name: {"ok", "why", "sides" (arm figures: the sides it can do),
    "variant" {side: the Live2D arm drawing it plays in}, "shift" {side:
    the figure's placement}}} for this body; cached on it (the prompt is
    built from it once, byte-stable per body file). A Live2D figure is
    probed in the drawing whose reach holds most of it, then the next
    (Natori has 12: probing all took 66 s)."""
    got = bm.__dict__.get("_move_caps")
    if got is not None:
        return got
    ctx = Ctx(bm)
    out = {}
    for name in ORDER[:-1]:
        fn, period, kind, main, _d = PRIMS[name]
        if kind != "arms":
            have = [v for v in main if v in ctx.has]
            out[name] = {"ok": bool(have), "why": "" if have else
                         "needs " + " or ".join(main)}
            continue
        if ctx.vrm:
            out[name] = {"ok": True, "why": "", "sides": list(SIDES),
                         "variant": {s: 0 for s in SIDES}, "shift": {}}
            continue
        options = ("both",) if name in ("clap", "twirl") else SIDES
        sides, why, var, shifts = [], "", {}, {}
        for sd in options:
            need = ("L", "R") if sd == "both" else (sd,)
            cand, seen = [], set()
            for k in range(len(bm.variants)):
                boxes = {s: _reach_box(bm, k, s) for s in need}
                if any(b is None for b in boxes.values()):
                    continue
                key = tuple(round(x, 2) for s in need for x in boxes[s])
                if key in seen:
                    continue
                seen.add(key)
                cand.append((-_score(name, sd, boxes, ctx), k))
            best = None
            for _sc, k in sorted(cand)[:2]:
                ok, trav, w, shift = _probe(bm, ctx, name, sd, k)
                if ok:
                    best = (trav, k, shift)
                    break
                if not why:
                    why = w
            if best is not None:
                sides.append(sd)
                var[sd] = best[1]
                shifts[sd] = {s: [round(v, 4) for v in d]
                              for s, d in best[2].items()}
        out[name] = {"ok": bool(sides), "why": "" if sides else why,
                     "sides": sides, "variant": var, "shift": shifts}
    trunk = [n for n in TRUNK if out[n]["ok"]]
    out["dance"] = {"ok": bool(trunk), "why": "" if trunk else
                    "no trunk motion"}
    bm.__dict__["_move_caps"] = out
    return out


def offered(bm):
    """The primitives this body can do, in menu order."""
    c = caps(bm)
    return [n for n in ORDER if c[n]["ok"]]


def prompt_lines(bm) -> str:
    """The [move] entry of the body-mode prompt: the marker, its
    parameters and the primitives this body can do."""
    names = offered(bm)
    if not names:
        return ""
    c = caps(bm)
    L = ['[move {"name": "sway", "beats": 8, "tempo": 110, "amp": 0.7}]  '
         "a rhythmic motion that keeps going on the beat: \"beats\" how long "
         f"(1 to 32, default {BEATS:g}), \"tempo\" beats per minute (60 to "
         f"180, default {TEMPO:g}; faster when excited, slower when calm), "
         "\"amp\" its size (0.1 to 1, default 0.7). It eases in and out over "
         "your idle and plays while you keep talking. A pose during it takes "
         "over the parts it names, then gives them back. Two motions on "
         "different parts at one place play together ([move {\"name\": "
         "\"sway\"}][move {\"name\": \"clap\"}]); on the same parts, one "
         "after the other. Your motions:"]
    for n in names:
        if n == "dance":
            L.append(f"  dance: {DANCE_DESC}")
            continue
        d = PRIMS[n][4]
        if n == "bounce" and not getattr(bm, "kind", None) == "vrm":
            d = d.replace("knees, ", "")
        sides = c[n].get("sides")
        if n in ("cheer", "wave") and sides and set(sides) != set(SIDES):
            only = [s for s in sides if s != "both"]
            d = d.split(" (")[0] + (f" (only your {'left' if only[0] == 'L' else 'right'} "
                                    f"hand: \"side\": \"{only[0]}\")"
                                    if len(only) == 1 and "both" not in sides
                                    else " (\"side\" L, R or both)")
        L.append(f"  {n}: {d}")
    return "\n".join(L) + "\n"


PROMPT_RULE = ("- When asked to dance, celebrate or cheer, or when the moment "
               "is rhythmic, joyful or excited, use a motion rather than a "
               "chain of poses: [move {\"name\": \"dance\"}] dances for you; "
               "a pose or two on top of it (a smile, a look) make it yours.\n")
PROMPT_EXAMPLE = ('Yay, let\'s dance! [move {"name": "dance", "beats": 8, '
                  '"tempo": 120}] Come on, [pose {"mouth.form": 1, '
                  '"eye.L.smile": 0.8, "eye.R.smile": 0.8}] move with me!')


# ----------------------------------------------------------------- parse

def normalize(name):
    n = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    extra = {}
    if n in ALIASES:
        n, extra = ALIASES[n]
        extra = dict(extra)
    return n, extra


def parse_spec(body, names, short):
    """(arg dict, warnings) for a [move ...] marker's body: JSON (tolerant,
    flow.pose.loads_lenient), or a name optionally followed by JSON
    ("sway", "sway {\"beats\": 4}", ":clap"). names: the primitives this
    body offers. arg None: dropped."""
    try:
        from . import pose as pz
    except ImportError:
        from flow import pose as pz
    warn = []
    s = body.strip().lstrip(":").strip()
    d, name = {}, None
    m = re.match(r"([A-Za-z_][\w\-]*)\s*(.*)$", s, re.S)
    if m and not s.startswith("{") and not re.match(r"\w+\s*:", s):
        name, s = m.group(1), m.group(2).strip().lstrip(":").strip()
    if s:
        try:
            d, repaired = pz.loads_lenient(s)
        except ValueError as exc:
            if name is None:
                return None, [f"[{short}]: bad JSON ({exc}); dropped"]
            warn.append(f"[{short}]: bad JSON after the name ({exc}); "
                        "defaults used")
            d, repaired = {}, False
        if repaired:
            warn.append(f"[{short}]: JSON repaired")
    if name is None:
        name = d.pop("name", None) or d.pop("move", None) or d.pop(
            "motion", None)
    else:
        d.pop("name", None)
    if not name:
        return None, warn + [f"[{short}]: no \"name\"; dropped"]
    n, extra = normalize(name)
    if n not in PRIMS and n != "dance":
        return None, warn + [f"[{short}]: no motion {name!r} (have "
                             f"{', '.join(names)}); dropped"]
    if n not in names:
        return None, warn + [f"[{short}]: {n} is not a motion this body can "
                             "do; dropped"]
    arg = {"name": n}
    for k, v in d.items():
        key = str(k).strip().lower()
        key = {"bpm": "tempo", "amplitude": "amp", "size": "amp",
               "hand": "side", "length": "beats", "count": "beats"}.get(
            key, key)
        if key in ("beats", "tempo", "amp", "dur", "duration"):
            try:
                x = float(v)
            except (TypeError, ValueError):
                warn.append(f"[{short}]: {key} {v!r} is not a number; "
                            "ignored")
                continue
            if not math.isfinite(x):
                continue
            if key in ("dur", "duration"):
                arg["seconds"] = round(min(max(x, 0.5), 20.0), 3)
            else:
                lo, hi = {"beats": BEATS_RANGE, "tempo": TEMPO_RANGE,
                          "amp": AMP_RANGE}[key]
                arg[key] = round(min(max(x, lo), hi), 3)
        elif key == "side":
            sv = str(v).strip().lower()
            sd = {"l": "L", "left": "L", "r": "R", "right": "R",
                  "both": "both", "two": "both"}.get(sv)
            if sd is None:
                warn.append(f"[{short}]: side {v!r} is not L, R or both; "
                            "ignored")
                continue
            arg["side"] = sd
        elif key == "seed":
            try:
                arg["seed"] = int(v)
            except (TypeError, ValueError):
                pass
        else:
            warn.append(f"[{short}]: {k!r} ignored")
    if "side" in extra and "side" not in arg:
        arg["side"] = extra["side"]
    if "tempo_scale" in extra:
        arg["tempo_scale"] = extra["tempo_scale"]
    if "amp_scale" in extra:
        arg["amp_scale"] = extra["amp_scale"]
    return arg, warn


def tag_text(arg) -> str:
    return f"[move {json.dumps(arg, ensure_ascii=False)}]"


# ------------------------------------------------------------------ clips

def _segments(name, beats, side, cp, rng):
    """[(b0, b1, [trunk / head figure names], arm figure or None, side)]."""
    if name != "dance":
        kind = PRIMS[name][2]
        if kind == "arms":
            return [(0.0, beats, [], name, side)]
        return [(0.0, beats, [name], None, side)]
    trunk = [n for n in TRUNK if cp[n]["ok"]]
    arms = [n for n in ARM_FIGS if cp[n]["ok"]]
    bob = cp["head_bob"]["ok"]
    segs, b, last_t, last_a = [], 0.0, None, "start"
    while b < beats - 1e-9:
        n = 4.0 if beats - b >= 6 and rng.random() < 0.5 else 2.0
        if beats - b < 2 + 2:
            n = beats - b
        tr = [x for x in trunk if x != last_t] or trunk
        t_ = tr[int(rng.integers(len(tr)))]
        # Shimmy is a short figure: only for 2 beats.
        if t_ == "shimmy" and n > 2 and len(tr) > 1:
            t_ = [x for x in tr if x != "shimmy"][int(rng.integers(
                len(tr) - 1))]
        # Arms most of the time: a segment of trunk alone reads as a pause
        # in a dance (Seed-san's first: 2 of 6 without arms).
        ar = [x for x in arms if x != last_a]
        if last_a and last_a != "start" and rng.random() < 0.2:
            ar = [None]
        if last_a == "start" and arms:
            ar = [x for x in arms if x in ("clap", "twirl")] or arms
        a_ = ar[int(rng.integers(len(ar)))] if ar else None
        sd = "both"
        if a_ in ("cheer", "wave"):
            opts = cp[a_]["sides"]
            sd = opts[int(rng.integers(len(opts)))]
        figs = [t_]
        if bob and a_ != "clap" and t_ != "shimmy" and rng.random() < 0.5:
            figs.append("head_bob")
        segs.append((b, b + n, figs, a_, sd))
        last_t, last_a = t_, a_
        b += n
    return segs


def _seg_weights(segs, b, mix=0.5):
    """[weight per segment] at beat b: 1 inside, min-jerk mixed over `mix`
    beats around each join; they sum to 1."""
    w = []
    for i, (b0, b1, *_r) in enumerate(segs):
        lo = 1.0 if i == 0 else float(min_jerk(np.clip(
            (b - (b0 - mix / 2)) / mix, 0, 1)))
        hi = 1.0 if i == len(segs) - 1 else 1.0 - float(min_jerk(np.clip(
            (b - (b1 - mix / 2)) / mix, 0, 1)))
        w.append(lo * hi)
    tot = sum(w) or 1.0
    return [x / tot for x in w]


def _rest_hands(bm, k, ctx):
    v = bm.variants[k]
    out = {}
    for s in "LR":
        for q in ("x", "y") + (("z",) if ctx.vrm else ()):
            f = f"hand.{s}.{q}"
            if f in bm.fidx and v["present"][bm.fidx[f]]:
                out[f] = float(v["rest"][bm.fidx[f]])
    return out


def _follow(bm, base, trunk, hands, wfig):
    """Hand targets carried with the chest as the trunk offsets turn it
    (a VRM): a figure's hands move with the upper body; a hand no figure
    holds hangs, HANG of the way the chest would carry it."""
    if not trunk:
        return hands
    top = next((b for b in ("upperChest", "chest", "spine") if b in bm.bones),
               None)
    if top is None:
        return hands
    W0, P0 = bm.fk(base)
    W1, P1 = bm.fk({**base, **trunk})
    R = W1[top] @ W0[top].T
    out = dict(hands)
    for s in "LR":
        ks = [f"hand.{s}.{q}" for q in "xyz"]
        if not all(k in hands for k in ks):
            continue
        p = np.array([hands[k] for k in ks], float)
        w = bm.to_world(p)
        w2 = P1[top] + R @ (w - P0[top])
        p2 = bm.to_body(w2)
        k_ = 1.0 - (1.0 - HANG) * (1.0 - min(max(wfig.get(s, 0.0), 0.0), 1.0))
        q = p + k_ * (p2 - p)
        for k, v in zip(ks, q):
            out[k] = float(v)
    return out


def _resolve_hands(bm, hands):
    """Composite hand targets as numbers (a VRM's "hands": "@together"),
    and the composite key kept for its wrist orientation."""
    comp = {k: hands[k] for k in ("hands",) if k in hands}
    plain = {k: v for k, v in hands.items() if k not in comp}
    if comp:
        ex = bm.resolve(dict(comp))
        for k, v in plain.items():
            ex[k] = v
        return ex, comp
    return plain, {}


def build(bm, arg, current=None, parts=None, log=print, seed=0,
          caps_=None):
    """A move marker's clip on this body (module docstring, "clip"):
    {name, spec, seconds, ramp_in, ramp_out, dt, add {param: [keyframes]},
    abs {param: [keyframes]}, step, parts, parts_from, variant, segments,
    solves, solve_seconds}; None when the body cannot do it (logged)."""
    t_start = time.monotonic()
    cp = caps_ or caps(bm)
    name = arg.get("name")
    if name not in cp or not cp[name]["ok"]:
        why = cp.get(name, {}).get("why", "unknown motion")
        log(f"move: {name}: not on this body ({why}); skipped")
        return None
    ctx = Ctx(bm)
    tempo = float(arg.get("tempo", TEMPO)) * float(arg.get("tempo_scale",
                                                           1.0))
    tempo = min(max(tempo, TEMPO_RANGE[0] / 2), TEMPO_RANGE[1])
    amp = float(arg.get("amp", AMP)) * float(arg.get("amp_scale", 1.0))
    spb = 60.0 / tempo
    beats = float(arg.get("beats", BEATS))
    if arg.get("seconds") is not None and "beats" not in arg:
        beats = max(1.0, round(float(arg["seconds"]) / spb))
    # A wave is one hand unless asked (both up is a crowd's sway).
    side = arg.get("side", "R" if name == "wave" else "both")
    kind = PRIMS[name][2] if name in PRIMS else "dance"
    if kind == "arms" and name in ("cheer", "wave"):
        sides = cp[name]["sides"]
        if side not in sides:
            new = side if side in sides else (
                "R" if "R" in sides else sides[0]) if side == "both" else (
                "both" if "both" in sides else sides[0])
            if "side" in arg:
                log(f"move: {name} side {side} not on this body; {new} "
                    "instead")
            side = new
    rng = np.random.default_rng([int(arg.get("seed", seed)) & 0xFFFFFFFF,
                                 11])
    segs = _segments(name, beats, side, cp, rng)
    figs_all = sorted({f for s in segs for f in s[2]}
                      | {s[3] for s in segs if s[3]})
    periods = [PRIMS[f][1] for f in figs_all]
    K = int(min(KF_MAX, max(KF, math.ceil(KF / min(periods)))))
    n = int(round(beats * K)) + 1
    dt = spb / K
    # Live2D: one arm drawing for the whole clip.
    k_var = 0
    arm_figs = [s[3] for s in segs if s[3]]
    if arm_figs and not ctx.vrm:
        vs = [cp[f]["variant"].get(s[4] if f in ("cheer", "wave") else
                                   "both") for s in segs for f in [s[3]]
              if f]
        # The drawing most of the figures play in; a figure that needs
        # another one is dropped from the clip (logged).
        k_var = max(set(vs), key=vs.count)
        segs2 = []
        for s in segs:
            f = s[3]
            if f and cp[f]["variant"].get(s[4] if f in ("cheer", "wave")
                                          else "both") != k_var:
                log(f"move: {f} needs another arm drawing than the rest of "
                    f"the {name}; left out")
                s = (s[0], s[1], s[2], None, s[4])
            segs2.append(s)
        segs = segs2
    dec = _declared(bm)
    add_rows, hand_rows, wclap, wfig = [], [], [], []
    missing = set()
    rest_h = _rest_hands(bm, k_var, ctx)
    # A VRM's arms hang from its chest: swayed or shimmied with it they
    # swing into the hips (Seed-san: 0.15 hu at torso.tilt 7), so under a
    # trunk figure they are solved too, hanging (module docstring).
    follow = bool(ctx.vrm and getattr(bm, "coll", None)
                  and any(PRIMS[f][2] == "trunk" for s in segs
                          for f in s[2]))
    any_arms = any(s[3] for s in segs) or follow
    for i in range(n):
        b = i / K
        ws = _seg_weights(segs, b, MIX) if len(segs) > 1 else [1.0]
        add, hands, wc, wf = {}, {}, 0.0, {"L": 0.0, "R": 0.0}
        for w, (b0, b1, figs, af, sd) in zip(ws, segs):
            if w <= 0:
                continue
            lb = b - b0
            for f in figs:
                a_, _h = PRIMS[f][0](lb, amp, sd, ctx)
                for var, x in a_.items():
                    add[var] = add.get(var, 0.0) + w * x
            if any_arms:
                if af:
                    a_, h = PRIMS[af][0](lb, amp, sd, ctx)
                    for var, x in a_.items():
                        add[var] = add.get(var, 0.0) + w * x
                    h, _comp = _resolve_hands(bm, h)
                    if not ctx.vrm:
                        sh = cp[af].get("shift", {}).get(
                            sd if af in ("cheer", "wave") else "both", {})
                        h = {k2: v2 + sh.get(k2[5], (0.0, 0.0))[
                            0 if k2.endswith(".x") else 1]
                            if k2[-2:] in (".x", ".y") else v2
                            for k2, v2 in h.items()}
                    if af == "clap":
                        wc += w
                    for sd2 in "LR":
                        if f"hand.{sd2}.x" in h:
                            wf[sd2] += w
                else:
                    h = {}
                for var in set(rest_h) | set(h):
                    if var.startswith("hand.") and (var in h or var in rest_h):
                        hands[var] = hands.get(var, 0.0) + w * h.get(
                            var, rest_h.get(var, 0.0))
        if ctx.vrm and len([w for w in ws if w > 0]) > 1 and hands:
            # Between two figures the hands pass in front (BULGE): the
            # straight line from a hip to a raised fist ran along the body,
            # and the IK took the elbow behind it on the way.
            mixw = 4.0 * float(np.prod(sorted(ws)[-2:]))
            for s_ in "LR":
                if f"hand.{s_}.z" in hands:
                    hands[f"hand.{s_}.z"] += BULGE * mixw
        add_rows.append(add)
        hand_rows.append(hands)
        wclap.append(wc)
        wfig.append(wf)
    # Additive variables -> parameters.
    add_p = {}
    for i, add in enumerate(add_rows):
        for var, x in add.items():
            x *= ctx.scale.get(var, 1.0)
            if var in HIPS:
                if ctx.vrm:
                    add_p.setdefault(var, np.zeros(n))[i] = x
                continue
            if var not in ctx.has or var not in dec:
                missing.add(var)
                continue
            pid, sc = dec[var]
            add_p.setdefault(pid, np.zeros(n))[i] += sc * x
    for pid, arr in add_p.items():
        if pid in bm.params:
            e = bm.params[pid]
            span = e["max"] - e["min"]
            # Leave the idle a tenth of the range on top.
            arr[:] = np.clip(arr, -0.45 * span, 0.45 * span)
    # Hands: solved on the body, each distinct keyframe once.
    abs_p, step, new_parts, parts_from = {}, [], {}, {}
    solves = 0
    if hand_rows and hand_rows[0]:
        allowed = _arm_set(bm)
        base = bm.default_values()
        base.update(current or {})
        if not ctx.vrm:
            v = bm.variants[k_var]
            base.update(v["switches"])
        boxes = ({} if ctx.vrm else
                 {s: _reach_box(bm, k_var, s) for s in "LR"})
        cache = {}
        orient_key = {"hands": "@together"} if ctx.vrm else {}

        TRUNK_P = ("torso.yaw", "torso.pitch", "torso.tilt", "shoulders.up",
                   "shoulder.L.up", "shoulder.R.up")
        trunk_rows = [{p: float(add_p[p][i]) for p in TRUNK_P if p in add_p}
                      for i in range(n)] if ctx.vrm else [{}] * n
        if ctx.vrm and any(trunk_rows):
            hand_rows = [_follow(bm, base, tr, tg, wf) for tr, tg, wf
                         in zip(trunk_rows, hand_rows, wfig)]

        def key(tg, oc, tr=None):
            return (tuple(sorted((k2, round(v2, 4)) for k2, v2 in tg.items())),
                    oc, tuple(sorted((k2, round(v2, 3)) for k2, v2 in
                                     (tr or {}).items())))

        def solve(tg, prior, oc, tr=None):
            nonlocal solves
            kk = key(tg, oc, tr)
            if kk in cache:
                return cache[kk]
            if tr:
                prior = {**prior, **tr}
            if ctx.vrm and not oc:
                # Only a clap turns the wrists: any other keyframe is
                # solved (and played) with them where they rest.
                prior = {**prior, **{p: base.get(p, 0.0) for s_ in "LR"
                                     for p in (f"hand.{s_}.roll",
                                               f"hand.{s_}.flex",
                                               f"hand.{s_}.dev")}}
            t2 = dict(tg)
            if oc:
                t2.update(orient_key)   # explicit numbers win (resolve)
            if ctx.vrm:
                r = bm.solve(t2, current=prior, via=False)
            else:
                # The first keyframe from the box's middle too (a prior at
                # the rest drawing's angles is no start in the raised one);
                # the rest stay on its branch.
                r = bm.solve(_clip_hands(t2, boxes), current=prior,
                             variant=k_var, only=allowed,
                             restart=not cache,
                             weights={t_: L2D_WEIGHT for t_ in t2})
            solves += 1
            sol = {p: float(x) for p, x in r["params"].items()
                   if p in allowed}
            cache[kk] = sol
            return sol
        # Every other keyframe is solved, the ones between interpolated in
        # the parameters (a quarter beat at most between solves: half the
        # cost, an 8-beat dance 0.55 s on Seed-san instead of 1.1).
        stride = 2 if K >= 8 and (n - 1) % 2 == 0 else 1
        idx = list(range(0, n, stride))
        rows = []
        prior = dict(base)
        # A periodic clip's cycle: solve its first keyframes again from the
        # cycle's end when the swivel there landed elsewhere.
        cyc = int(round(max(periods) * K)) if len(segs) == 1 else 0
        if cyc % stride:
            cyc = 0
        oc_of = lambda i: bool(ctx.vrm and wclap[i] >= 0.5)  # noqa: E731
        j = 0
        while j < len(idx):
            i = idx[j]
            sol = solve(hand_rows[i], prior, oc_of(i), trunk_rows[i])
            if cyc and i == cyc and rows:
                kk = key(hand_rows[0], oc_of(0), trunk_rows[0])
                cache.pop(kk, None)
                s2 = solve(hand_rows[0], {**prior, **rows[-1]}, oc_of(0),
                           trunk_rows[0])
                jump = max((abs(s2.get(p, 0.0) - rows[0].get(p, 0.0))
                            for p in s2), default=0.0)
                if jump > RESOLVE_JUMP:
                    # Solve the cycle again from its own end.
                    cache.clear()
                    rows = []
                    p2 = {**prior, **s2}
                    for i2 in idx[:j + 1]:
                        s3 = solve(hand_rows[i2], p2, oc_of(i2),
                                   trunk_rows[i2])
                        rows.append(s3)
                        p2 = {**p2, **s3}
                    prior = p2
                    j += 1
                    continue
                cache[kk] = rows[0]
            rows.append(sol)
            prior = {**prior, **sol}
            j += 1
        keys = sorted({p for r in rows for p in r})
        for p in keys:
            arr = np.array([r.get(p, np.nan) for r in rows], float)
            if np.isnan(arr).any():
                good = ~np.isnan(arr)
                fill = base.get(p, bm.params[p]["default"]) \
                    if p in bm.params else 0.0
                arr[~good] = np.interp(np.flatnonzero(~good),
                                       np.flatnonzero(good), arr[good]) \
                    if good.any() else fill
            if stride > 1:
                Y = arr[None, :]
                M = pchip_slopes(Y)
                arr = np.array([pchip_at(Y, M, i / stride)[0]
                                for i in range(n)])
            if not ctx.vrm and p in bm.params:
                # A Live2D arm has two parameters for one hand position
                # (Hiyori's ParamArmLB and ParamHandLB both move hand.L.x),
                # and its piecewise-linear model has kinks: between two
                # keyframes a solve can change branch. No parameter moves
                # faster than L2D_RATE spans a second.
                e = bm.params[p]
                lim = L2D_RATE * (e["max"] - e["min"]) * dt
                for i in range(1, n):
                    arr[i] = arr[i - 1] + float(np.clip(arr[i] - arr[i - 1],
                                                        -lim, lim))
            abs_p[p] = arr
        if ctx.vrm:
            # The wrist: the clap's palms-facing turn mixed out with the
            # clap's weight (only clap keyframes are solved with it).
            wc = np.array(wclap)
            for s in "LR":
                for p in (f"hand.{s}.roll", f"hand.{s}.flex",
                          f"hand.{s}.dev"):
                    if p in abs_p and 0 < wc.max():
                        ref = abs_p[p][wc >= 0.999]
                        ref = float(np.median(ref)) if len(ref) else \
                            float(abs_p[p][int(np.argmax(wc))])
                        abs_p[p] = np.where(wc >= 0.999, abs_p[p],
                                            wc * ref + (1 - wc)
                                            * base.get(p, 0.0))
        if not ctx.vrm:
            # The arm drawing the clip plays in, swapped in at the ease-in's
            # midpoint and back at the ease-out's (as a pose carries it).
            v = bm.variants[k_var]
            cur = {**bm.default_values(), **(current or {})}
            cur_k = bm.match_variant(cur, parts)
            if v["parts"]:
                new_parts = {p: float(x) for p, x in v["parts"].items()}
                prt = dict(parts if parts is not None else
                           {**bm.blob.get("rest_parts", {}),
                            **bm.variants[cur_k]["parts"]})
                parts_from = {q: float(prt.get(q, 1.0 - x))
                              for q, x in new_parts.items()}
            for p, x in v["switches"].items():
                abs_p[p] = np.full(n, float(x))
                step.append(p)
            step += [p for p in abs_p if bm.params.get(p, {}).get("role")
                     in ("switch", "variant") and p not in step]
    if missing:
        log(f"move: {name}: this body has no {', '.join(sorted(missing))}; "
            "left out")
    seconds = beats * spb
    ramp = float(np.clip(spb, *RAMP_RANGE))
    if abs_p:
        # An arm raised from hanging to overhead travels ~2.7 hu (Seed-san):
        # over one beat at 110 bpm that peaked at 0.44 hu a frame.
        ramp = max(ramp, ARM_RAMP)
    ramp = min(ramp, seconds / 2)
    rnd = lambda a_: [round(float(x), 5) for x in a_]  # noqa: E731
    rec = {"name": name, "spec": dict(arg), "tempo": round(tempo, 3),
           "beats": beats, "amp": round(amp, 3), "side": side,
           "seconds": round(seconds, 4), "ramp_in": round(ramp, 4),
           "ramp_out": round(ramp, 4), "dt": round(dt, 6),
           "add": {p: rnd(a_) for p, a_ in sorted(add_p.items())},
           "abs": {p: rnd(a_) for p, a_ in sorted(abs_p.items())},
           "step": sorted(step), "parts": new_parts,
           "parts_from": parts_from, "variant": k_var,
           "segments": [{"beats": [s[0], s[1]], "figures": s[2],
                         "arms": s[3], "side": s[4]} for s in segs],
           "solves": solves,
           "solve_seconds": round(time.monotonic() - t_start, 3)}
    return rec


def solve_moves(bm, acts, current=None, parts=None, log=print, seed=0):
    """The move actions among acts ({"t", "tag": "move", "arg", "at"}),
    each built on this body and scheduled: one on parameters an earlier
    move still plays starts as that one eases out. Returns records (build's
    plus act, t, at, start)."""
    out = []
    order = sorted((i for i, x in enumerate(acts) if x.get("tag") == MOVE),
                   key=lambda i: float(acts[i]["t"]))
    cp = caps(bm)
    for idx in order:
        a = acts[idx]
        t = float(a["t"])
        rec = build(bm, a.get("arg") or {}, current, parts, log,
                    seed=hash_seed(seed, a.get("at"), idx), caps_=cp)
        if rec is None:
            continue
        start = t
        mine = set(rec["add"]) | set(rec["abs"])
        for q in out:
            q_end = q["start"] + q["seconds"]
            if start < q_end and mine & (set(q["add"]) | set(q["abs"])):
                start = max(start, q_end - min(q["ramp_out"],
                                               rec["ramp_in"]))
        rec.update(act=idx, t=round(t, 4), at=a.get("at"),
                   start=round(start, 4))
        out.append(rec)
        log(f"move: {rec['name']} at {start:.2f}s, {rec['beats']:g} beats "
            f"at {rec['tempo']:g} bpm, amp {rec['amp']:g}"
            + (f", {' / '.join('+'.join(s['figures'] + ([s['arms']] if s['arms'] else [])) for s in rec['segments'])}"
               if rec["name"] == "dance" else "")
            + f" ({rec['solves']} solves, {rec['solve_seconds']}s)")
    return out


def move_seed(a, f0):
    """flow.live's seed for a sentence's moves: the run's --seed and a
    number for the sentence (turn and sentence index), so every turn
    dances anew; flow.live writes it on the marker, so --avatar both's two
    bodies draw the same."""
    return (int(getattr(a, "seed", 0)) * 7919 + int(f0)) & 0x7FFFFFFF


def hash_seed(seed, at, idx):
    """A dance's own generator: from the run's seed and where the marker
    is, never from the director's or the live loop's generators."""
    return (int(seed) * 1000003 + int(at or 0) * 131 + int(idx)) & 0x7FFFFFFF


def drop_clashing_gestures(acts, moves, motions, log=print):
    """--body-gestures: a directed gesture whose play overlaps a move
    that plays an arm is dropped (the move wins, as a pose does: flow.pose.
    drop_clashing_gestures). acts is changed in place, each move's "act"
    follows. Returns the dropped gesture actions."""
    try:
        from . import director as dr
    except ImportError:
        from flow import director as dr
    win = windows(moves)
    if not win:
        return []
    labels = dr.gesture_labels(motions)
    keep, gone = [], []
    for x in acts:
        name = labels.get(str(x.get("arg")).lower()) \
            if x.get("tag") == "gesture" else None
        if name is not None:
            t0 = float(x["t"])
            t1 = t0 + min(float(motions[name]), dr.GESTURE_MAX) + dr.SETTLE
            if any(t0 < b and t1 > a for a, b in win):
                gone.append(x)
                log(f"move: [gesture:{x['arg']}] at {t0:.2f}s dropped: a "
                    "motion plays the arms then")
                continue
        keep.append(x)
    if gone:
        new = {id(x): i for i, x in enumerate(keep)}
        for m in moves:
            m["act"] = new.get(id(acts[m["act"]]), m["act"])
        acts[:] = keep
    return gone


def end_time(moves) -> float:
    return max((m["start"] + m["seconds"] for m in moves), default=0.0)


def windows(moves):
    """(start, end) of the moves that play an arm: an earlier gesture gives
    way to them, as to an arm pose."""
    return [(m["start"], m["start"] + m["seconds"]) for m in moves
            if m["abs"] or m["parts"]]


def tokens(moves, arms=()):
    """Move tokens for the event stream: type "move", group "arms" when
    they play an arm (director.windows drops the idle's gestures then; the
    head, gaze and body moves of the idle keep running under them)."""
    out = []
    for m in moves:
        groups = ["arms"] if m["abs"] or m["parts"] else []
        out.append({"t": m["start"], "type": MOVE, "role": MOVE,
                    "name": m["name"], "duration": m["seconds"],
                    "ramp_in": m["ramp_in"], "ramp_out": m["ramp_out"],
                    "dt": m["dt"], "add": m["add"], "abs": m["abs"],
                    "step": m["step"], "parts": m["parts"],
                    "parts_from": m.get("parts_from", {}),
                    "groups": groups, "src": "director"})
    return out


# ------------------------------------------------------------------- play

def pchip_slopes(Y):
    """Fritsch-Carlson slopes per keyframe (rows: channels, uniform step)."""
    Y = np.asarray(Y, float)
    n = Y.shape[1]
    if n < 2:
        return np.zeros_like(Y)
    d = np.diff(Y, axis=1)
    m = np.zeros_like(Y)
    m[:, 0], m[:, -1] = d[:, 0], d[:, -1]
    if n > 2:
        a, b = d[:, :-1], d[:, 1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(a * b > 0, 2 * a * b / (a + b), 0.0)
        m[:, 1:-1] = h
    # Ends: keep monotone.
    for j in (0, -1):
        m[:, j] = np.where(np.sign(m[:, j]) != np.sign(d[:, j]), 0.0, m[:, j])
    return m


def pchip_at(Y, M, u):
    """Values at fractional keyframe index u (clamped)."""
    n = Y.shape[1]
    if n == 1:
        return Y[:, 0]
    u = min(max(u, 0.0), n - 1.0)
    i = min(int(u), n - 2)
    s = u - i
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return (h00 * Y[:, i] + h10 * M[:, i] + h01 * Y[:, i + 1]
            + h11 * M[:, i + 1])


def envelope(tau, dur, rin, rout):
    if tau <= 0 or tau >= dur:
        return 0.0
    w = 1.0
    if tau < rin:
        w = float(min_jerk(tau / rin))
    if tau > dur - rout:
        w = min(w, float(min_jerk((dur - tau) / rout)))
    return w


class MoveLayer:
    """Move tokens played over the frame the idle built (module docstring,
    "play"). commit(tok) queues a token; apply(f, over, parts) is frame f.
    `written`: the parameters this frame wrote; `holding`: those a move
    plays absolutely now (the pose layer keeps no life on them)."""

    def __init__(self, fps, defaults=None, part_rest=None):
        self.fps = float(fps)
        self.defaults = dict(defaults or {})
        self.part_rest = dict(part_rest or {})
        self.queue = []
        self.active = []
        self.touched = set()
        self.written = set()
        self.holding = set()

    def commit(self, tok):
        self.queue.append(tok)

    def busy(self):
        return bool(self.queue or self.active or self.touched)

    def _start(self, tok):
        chans = []
        for mode in ("add", "abs"):
            items = tok.get(mode) or {}
            if not items:
                chans.append((mode, [], None, None))
                continue
            keys = sorted(items)
            Y = np.array([items[k] for k in keys], float)
            chans.append((mode, keys, Y, pchip_slopes(Y)))
        for i, v in (tok.get("parts_from") or {}).items():
            self.part_rest.setdefault(i, float(v))
        self.active.append({"tok": tok, "t0": float(tok["t"]),
                            "dur": float(tok["duration"]),
                            "rin": max(float(tok.get("ramp_in", 0.3)),
                                       1.0 / self.fps),
                            "rout": max(float(tok.get("ramp_out", 0.3)),
                                        1.0 / self.fps),
                            "dt": float(tok["dt"]), "chans": chans,
                            "step": set(tok.get("step") or ())})

    def apply(self, f, over, parts):
        self.written, self.holding = set(), set()
        if not self.busy():
            return over, parts
        tf = f / self.fps
        while self.queue and int(math.ceil(float(self.queue[0]["t"])
                                           * self.fps - 1e-9)) <= f:
            self._start(self.queue.pop(0))
        over = dict(over)
        parts_in = parts
        parts = dict(parts or {})
        # Each active token's weight and channel values this frame.
        live = []
        for a in self.active:
            tau = tf - a["t0"]
            if tau >= a["dur"]:
                continue
            w = envelope(tau, a["dur"], a["rin"], a["rout"])
            vals = {}
            for mode, keys, Y, M in a["chans"]:
                if not keys:
                    continue
                y = pchip_at(Y, M, tau / a["dt"])
                for k, x in zip(keys, y):
                    vals[k] = (mode, float(x))
            live.append((a, w, vals))
        self.active = [a for a in self.active
                       if tf - a["t0"] < a["dur"]]
        under = {}
        contrib = {}
        # Later tokens take over shared parameters as they ease in.
        for j in range(len(live) - 1, -1, -1):
            a, w, vals = live[j]
            for p, (mode, x) in vals.items():
                rem = contrib.setdefault(p, [1.0, []])
                e = w * rem[0]
                rem[0] *= (1.0 - w)
                rem[1].append((mode, x, e, w, p in a["step"]))
        for p, (_rem, items) in contrib.items():
            u = float(over.get(p, self.defaults.get(p, 0.0)))
            under[p] = u
            stepped = [it for it in items if it[4]]
            if stepped:
                # A switch: the latest token's value from its midpoint.
                on = [it for it in stepped if it[3] >= 0.5]
                if on:
                    over[p] = on[0][1]
                    self.written.add(p)
                continue
            v = u
            for mode, x, e, w, _s in items:
                v += e * (x if mode == "add" else x - u)
                if mode == "abs" and e > 0:
                    self.holding.add(p)
            over[p] = v
            self.written.add(p)
        # Part opacities: the latest token that carries them, from the
        # midpoints of its ease in to its ease out.
        on_parts = {}
        for a, w, _v in live:
            for i, x in (a["tok"].get("parts") or {}).items():
                self.touched.add(i)
                if w >= 0.5:
                    on_parts[i] = float(x)
        for i in self.touched:
            parts[i] = on_parts.get(i, float(
                (parts_in or {}).get(i, self.part_rest.get(i, 1.0))))
        if not self.active and not self.queue:
            self.touched = set()
        if parts_in is None and not parts:
            parts = None
        return over, parts


def layer_for(tokens_, fps, defaults, part_rest=None):
    """A MoveLayer with every move token of a stream committed; None when
    there are none (nothing changes)."""
    toks = [k for k in tokens_ if k.get("type") == MOVE]
    if not toks:
        return None
    lay = MoveLayer(fps, defaults, part_rest)
    for k in sorted(toks, key=lambda k: float(k["t"])):
        lay.commit(k)
    return lay


def report_lines(moves, spoken=""):
    out = []
    for m in moves:
        at = m.get("at")
        words = (f" at {spoken[at:at + 16].strip()!r}"
                 if spoken and at is not None else "")
        segs = " / ".join("+".join(s["figures"] + ([s["arms"] + (
            f"({s['side']})" if s["arms"] in ("cheer", "wave") else "")]
            if s["arms"] else [])) for s in m["segments"])
        out.append(f"move {m['start']:.2f}s{words}: {m['name']} "
                   f"{m['beats']:g} beats at {m['tempo']:g} bpm, amp "
                   f"{m['amp']:g} [{segs}] {len(m['add'])} added, "
                   f"{len(m['abs'])} played parameters")
    return out


if __name__ == "__main__":
    ap_ = sys.argv[1:]
    bm_ = fb.load(ap_[0])
    print(prompt_lines(bm_))
    for n_, c_ in caps(bm_).items():
        print(n_, c_)
