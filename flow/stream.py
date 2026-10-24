#!/usr/bin/env python3
"""
Streaming motion: the --events runtime one frame at a time, for flow.live.

flow.runtime plans a whole run before it renders: schedule the idle over a
fixed duration, merge the director's tokens, render every role trajectory
and every gesture frame, then loop over the frames. A live character has no
end time and learns what it will say while it is already moving, so this
module keeps the same pieces as running state instead:

  idle       events.IdleScheduler / GestureScheduler, advanced in rolling
             windows (`horizon` seconds ahead, refilled when less than
             `refill` is left). They are resumable, so the windows join
             with no seam: the tokens are the ones a single schedule() call
             with the same generators emits (tests/test_events). Gestures
             draw from their own generator, so the window size cannot
             change the moves.
  pending    tokens not yet started, sorted by (t, type) as events.render
             and director.merge sort them. direct() merges a reply's tokens
             into them with director.merge semantics: an idle move or
             gesture overlapping a directed window on its group is dropped,
             blinks and untouched groups stay. Only pending tokens can be
             dropped: an idle move already in flight finishes first (render
             never retargets mid-flight), so a directed move on a busy group
             waits for it -- the one difference from the offline merge,
             which can drop a move that started before the directed window.
             The wait is logged per directed move.
  commit     each frame, every pending token with onset <= the frame time
             starts, in that order: moves go to MoveTrack (events.render
             one frame at a time), gestures to the GesturePlayer's current
             and previous gesture, blinks to Autonomic.
  frame      what runtime.main writes for that frame: the base state, the
             role columns, gesture columns, switches snapped with their
             hysteresis, the mouth, part opacities, then blink and breath.
             Fed the same tokens, frame(f) equals runtime --tokens-in
             --dump-only's plan frame for frame (tests/test_live).
  mouth      a sentence's mouth curve (speech.mouth_track) starts on a frame
             (flow.live snaps the audio start to the frame grid), so frame f
             reads its curve at f - start exactly. Between sentences the
             mouth stays at the curve's closed value (the calibrated map's
             rest) rather than falling back to the rig default, which with
             a map whose rest is not the default would pop at every
             sentence end. Offline this never arises: one mouth token spans
             the whole render.

Everything here is numpy and the standard library; the GL backend stays in
flow.live's render loop.
"""

from __future__ import annotations

import math
import os
import sys
import threading
from collections import deque

import numpy as np

try:
    from . import events as ev
    from . import runtime as rt
    from .behaviors import min_jerk
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import events as ev
    from flow import runtime as rt
    from flow.behaviors import min_jerk

MOVE_TYPES = ("saccade", "head_turn", "body_shift", "brow")
# A directed move that starts this much after its intended time (its group
# busy with an idle move already in flight) is logged.
LATE_LOG = 0.1


class EventRig:
    """Everything runtime.main sets up for an --events run, once: columns,
    held parameters, bound roles and adapted statistics, the gesture
    library (calm under the calm profile) and settle motion, base state,
    the GesturePlayer, snapped switches, autonomic and mouth targets. Built
    from the same runtime helpers in the same order, so a stream on it draws
    what the offline render draws; tests/test_live checks that frame for
    frame."""

    def __init__(self, model_json, rig, events="default", clips=None,
                 binding=None, fps=30.0, probe_dir="out",
                 idle_profile="calm", no_snap=False, pool_only=False,
                 snap_ramp=6, hold="", hold_global=True, log=print):
        self.model_json, self.fps = model_json, float(fps)
        col = rt.rig_columns(rig, model_json, clips, log)
        self.col = col
        pids, groups = col["pids"], col["groups"]
        lo, hi, dflt = col["lo"], col["hi"], col["default"]
        phys, bands, ex = col["phys"], col["bands"], col["expressive"]
        if not ex:
            raise SystemExit("no expressive parameters")
        self.pids, self.lo, self.hi, self.dflt = pids, lo, hi, dflt
        self.bands, self.expressive = bands, ex
        self.held = rt.held_params(rig, probe_dir, pids, groups, ex, hold,
                                   hold_global)
        if self.held:
            log(f"holding {len(self.held)} whole-model parameters still: "
                + ", ".join(sorted(self.held)))
        span = np.maximum(hi[ex] - lo[ex], 1e-6)
        self.span_e = span
        self.default_n = np.clip((dflt[ex] - lo[ex]) / span * 2 - 1, -1, 1)
        motions = rt.authored_motions(model_json, fps)
        self.motions = motions
        self.motion_seconds = {n: m["frames"] / fps for n, m in motions}
        band_by_id = dict(zip(pids, bands))
        stats, roles, _ = rt.bind_events(
            events, model_json, pids, [pids[i] for i in ex], lo, hi, motions,
            fps, binding, no_snap, pool_only, log)
        stats["idle"]["profile"] = idle_profile
        self.stats, self.roles = stats, roles
        self.start = {r: stats["roles"][r]["center"] for r in roles}
        role_ids = {p for p, _, _ in roles.values()}
        pid_set = set(pids)
        gkeep = rt.base_keep(role_ids | self.held, phys, band_by_id)
        keep = lambda p: p in pid_set and gkeep(p)      # noqa: E731
        found = {} if no_snap else rt.toggle_params(model_json, fps)
        sw = {p: (v[1], v[2]) for p, v in rt.snap_set(
            found, pids, groups, bands, ex, col["continuous"]).items()}
        gest = ev.gesture_library(motions, keep, dict(zip(pids, lo.tolist())),
                                  dict(zip(pids, hi.tolist())), sw, fps)
        self.settle = None
        if idle_profile == "calm" and gest:
            gest, self.settle = rt.calm_library(gest, model_json, log)
        self.gest = gest
        model_driven = role_ids - self.held
        _, self.base_params, self.base_parts = rt.base_state(
            motions, rt.base_keep(model_driven, phys, band_by_id))
        loose = []
        if self.settle is not None:
            # As runtime.main for its underlay token: from the rig and the
            # settle loop alone.
            sw_all = rt.toggle_params(model_json, fps)
            und_m = dict(motions).get(self.settle, {}).get("params", {})
            loose = ev.loose_params(
                [pids[i] for i in ex], [groups[i] for i in ex], sw_all,
                lambda p: keep(p) and np.ptp(und_m.get(p, [0.0])) > 0)
        self.player = ev.GesturePlayer(
            dict(motions), keep, self.base_params, self.base_parts,
            {p: float(v) for p, v in zip(pids, dflt)}, fps, loose)
        self.loose = loose
        self.col_of = {pids[i]: c for c, i in enumerate(ex)}
        self.role_cols = [(r, self.col_of[p], float(sg))
                          for r, (p, sg, _) in roles.items()]
        self.gesture_cols = [c for c, i in enumerate(ex)
                             if pids[i] not in role_ids
                             and pids[i] not in self.held]
        self.expr_ids = {pids[i] for i in ex}
        # The columns runtime writes from the trajectory, and those it takes
        # from the base state (held, or banded otherwise since the clips).
        self.in_band = [(c, i) for c, i in enumerate(ex)
                        if bands[i] == "expressive"
                        and pids[i] not in self.held]
        live = rt.snap_set(found, pids, groups, bands, ex,
                           col["continuous"])
        fade_n = int(round(stats["gesture"]["fade"] * fps))
        self.switch_states = {p: (v[1], v[2]) for p, v in live.items()}
        self.switch_ramp = max(snap_ramp, fade_n)
        self.eye_idx = [i for i, p in enumerate(pids)
                        if bands[i] == "autonomic" and "open" in p.lower()]
        self.breath_idx = [i for i, p in enumerate(pids)
                           if "breath" in p.lower()]

    def switches(self):
        """A fresh runtime.Switches (stateful: one per stream)."""
        if not self.switch_states:
            return None
        return rt.Switches(self.switch_states, self.switch_ramp,
                           shape="minjerk")

    def mouth_curves(self, tok, n_frames):
        """{pid: values} of a mouth token over n_frames from its own start
        (runtime.mouth_frames with t = 0)."""
        return rt.mouth_frames(dict(tok, t=0.0), self.pids, self.bands,
                               self.hi, self.dflt, self.fps, n_frames,
                               lo=self.lo)


class MoveTrack:
    """events.render one frame at a time. commit() takes move tokens in the
    order render() would read them; at(f) is frame f of every role. A move
    starts at max(its onset, the end of its group's current move), runs a
    minimum-jerk profile from the previous target, and holds after."""

    def __init__(self, roles, fps, start=None):
        self.roles = list(roles)
        self.fps = float(fps)
        self.gbusy = {}
        self.last = {r: float((start or {}).get(r, 0.0)) for r in self.roles}
        self.queue = {r: deque() for r in self.roles}
        # (t0, d, from, to) of the move each role is in or last finished.
        self.cur = {r: None for r in self.roles}
        self.hold = dict(self.last)

    def commit(self, tok):
        """Returns the move's actual start (seconds)."""
        g = tok["role"]
        t0 = max(float(tok["t"]), self.gbusy.get(g, 0.0))
        d = max(float(tok["duration"]), 1.0 / self.fps)
        self.gbusy[g] = t0 + d
        for r, v in tok["target"].items():
            if r in self.queue:
                a = int(np.ceil(t0 * self.fps - 1e-9))
                self.queue[r].append((a, t0, d, self.last[r], float(v)))
                self.last[r] = float(v)
        return t0

    def busy_until(self, g):
        return self.gbusy.get(g, 0.0)

    def at(self, f):
        tf = f / self.fps
        out = {}
        for r in self.roles:
            q = self.queue[r]
            while q and q[0][0] <= f:
                self.cur[r] = q.popleft()[1:]
            c = self.cur[r]
            if c is None:
                y = self.hold[r]
            else:
                t0, d, frm, to = c
                y = frm + (to - frm) * min_jerk(np.array([(tf - t0) / d]))[0]
            out[r] = np.float32(np.clip(y, -1, 1))
        return out

    def position(self, r):
        """Where role r ends once its committed moves are done."""
        return self.last.get(r)


class MotionStream:
    """The live token stream on an EventRig, rendered a frame at a time.

    idle: generate the idle (False: blinks only, as runtime --no-idle).
    tokens: a fixed token list to play instead (a --tokens file), with no
    idle generated; what the equivalence test feeds it. mouth_map: the
    rig's calibrated mouth map (flow.mouth), whose rest the mouth holds
    between sentences. Thread safety: direct()/add_mouth() may be called
    from any thread while the render thread calls frame().
    """

    def __init__(self, rig, seed=0, idle=True, tokens=None, horizon=20.0,
                 refill=10.0, mouth_map=None, log=print, autonomic=True):
        self.rig, self.fps, self.log = rig, rig.fps, log
        self.lock = threading.Lock()
        self.horizon, self.refill = float(horizon), float(refill)
        roles = list(rig.roles)
        self.track = MoveTrack(roles, rig.fps, rig.start)
        self.pending = []
        self.sched = self.gsched = None
        self.idle = idle
        self.und = None
        self.ahead = 0.0                   # idle generated up to here
        if tokens is not None:
            self.pending = sorted((dict(k) for k in tokens
                                   if k["type"] not in ("underlay", "mouth")),
                                  key=lambda k: (float(k["t"]), k["type"]))
            und = next((k for k in tokens if k["type"] == "underlay"), None)
            self.und = rig.player.underlay(und)
            self.ahead = math.inf
        else:
            self.sched = ev.IdleScheduler(rig.stats, roles,
                                          np.random.default_rng(seed),
                                          rig.start)
            if idle and rig.gest:
                self.gsched = ev.GestureScheduler(
                    self.sched.st, rig.gest, np.random.default_rng([seed, 2]))
            if idle:
                # One underlay for the whole session (runtime: one per run).
                und = ev.underlay_token(rig.stats, rig.settle, 1e9)
                self.und = rig.player.underlay(und)
        self.ges_cur = self.ges_prev = None
        self.switches = rig.switches()
        self.auto = (rt.Autonomic(rig.fps, np.random.default_rng([seed, 3]))
                     if autonomic else None)
        self.mouths = deque()              # (first frame, {pid: values})
        self.mouth_rest = {}
        if mouth_map:
            rest = rig.mouth_curves({"fps": rig.fps, "open": [0.0],
                                     "map": mouth_map}, 1)
            self.mouth_rest = {p: float(v[0]) for p, v in rest.items()}
        self.f = 0                          # the next frame to compute
        self.late = []                      # (intended, actual) directed
        self.extend(0.0)

    # ------------------------------------------------------------ tokens

    def extend(self, now):
        """Generate the idle until now + horizon when less than `refill`
        seconds of it are left; new idle tokens are merged against the
        directed ones already pending."""
        if self.sched is None or self.ahead - now >= self.refill:
            return 0
        until = now + self.horizon
        new = self.sched.advance(until)
        self.sched.toks.clear()
        if self.gsched is not None:
            new = new + self.gsched.advance(until)
        if not self.idle:
            new = [k for k in new if k["type"] == "blink"]
        for k in new:
            if "follows" in k:
                # An emission number; nothing renders it, and merge would
                # renumber it as an index into a list it is not in.
                k["follows"] = None
        self.ahead = until
        with self.lock:
            self._merge(new, [])
        return len(new)

    def _merge(self, idle_new, directed_new):
        try:
            from . import director as dr
        except ImportError:
            from flow import director as dr
        idle = [k for k in self.pending if k.get("src") != "director"]
        dirs = [k for k in self.pending if k.get("src") == "director"]
        self.pending = dr.merge(idle + idle_new, dirs + directed_new)

    def direct(self, tokens):
        """Merge directed tokens (director.realize, absolute stream times)
        into the future. A token whose onset has already passed starts on
        the next frame instead: a move begun in the past would jump to
        partway through its profile. Returns how many idle tokens it
        displaced."""
        with self.lock:
            nxt = self.f / self.fps
            toks = []
            for k in tokens:
                k = dict(k)
                if float(k["t"]) < nxt:
                    k["t"] = round(nxt, 4)
                if k.get("follows") is not None:
                    k["follows"] = None
                toks.append(k)
            n0 = len(self.pending)
            self._merge([], toks)
            return n0 + len(toks) - len(self.pending)

    def add_mouth(self, first_frame, tok, n_frames):
        """A mouth token (speech.mouth_track payload) starting on frame
        `first_frame`, over n_frames."""
        curves = self.rig.mouth_curves(tok, n_frames)
        with self.lock:
            self.mouths.append((int(first_frame), curves))

    def position(self, r):
        return self.track.position(r)

    def pose(self):
        """{role: value} each role ends at once its committed moves finish;
        director.realize's start positions (it sizes durations by them)."""
        return {r: self.track.position(r) for r in self.rig.roles}

    def _commit(self, tf):
        while self.pending and float(self.pending[0]["t"]) <= tf + 1e-9:
            k = self.pending.pop(0)
            typ = k["type"]
            if typ in MOVE_TYPES:
                t0 = self.track.commit(k)
                if k.get("src") == "director" and t0 - float(k["t"]) > LATE_LOG:
                    self.late.append((float(k["t"]), t0))
                    self.log(f"stream: directed {typ} for {k['t']:.2f}s "
                             f"starts {t0 - float(k['t']):.2f}s late (its "
                             f"group is finishing an idle move)")
            elif typ == "gesture":
                self.ges_prev, self.ges_cur = self.ges_cur, k
            elif typ == "blink" and self.auto is not None:
                self.auto.request(float(k["t"]))

    # ------------------------------------------------------------- frames

    def frame(self, f=None):
        """(overrides, parts) for the next frame, as runtime.main writes
        them (autonomic included unless the stream was made without).
        Frames are computed in order; skipping one for the GL render still
        needs this call, since switches, blinks and the token commit are
        stateful."""
        rig = self.rig
        if f is not None and f != self.f:
            raise ValueError(f"frame {f} asked, {self.f} next")
        f = self.f
        tf = f / self.fps
        self.extend(tf)
        with self.lock:
            self._commit(tf)
            while self.mouths and self.mouths[0][0] + len(
                    next(iter(self.mouths[0][1].values()), ())) <= f:
                self.mouths.popleft()
            mouth = list(self.mouths)
            self.f += 1
        gen = self.track.at(f)
        gp, parts, _ = rig.player.at(tf, self.ges_cur, self.ges_prev,
                                     self.und)
        ex, pids, lo = rig.expressive, rig.pids, rig.lo
        traj = rig.default_n.astype(np.float32)
        for r, c, sg in rig.role_cols:
            traj[c] = np.clip(sg * gen[r], -1, 1)
        for c in rig.gesture_cols:
            i = ex[c]
            p = pids[i]
            v = np.float32(gp.get(p, rig.base_params.get(p, rig.dflt[i])))
            traj[c] = np.clip((v - lo[i]) / rig.span_e[c] * 2 - 1, -1, 1)
        over = dict(rig.base_params)
        vals = rt.denorm(traj, lo[ex], rig.hi[ex])
        sw = self.switches
        for c, i in rig.in_band:
            v = float(vals[c])
            if sw is not None and pids[i] in sw.states:
                v = sw(pids[i], v)
            over[pids[i]] = v
        for p, v in self.mouth_rest.items():
            over[p] = v
        for f0, curves in mouth:
            if f0 <= f:
                for p, v in curves.items():
                    over[p] = float(v[f - f0])
        for p, v in gp.items():
            if p not in rig.expr_ids:
                over[p] = v
        if self.auto is not None:
            eye, breath = self.auto.step()
            for i in rig.eye_idx:
                if pids[i] not in over:
                    over[pids[i]] = float(lo[i] + eye * (rig.hi[i] - lo[i]))
            for i in rig.breath_idx:
                if pids[i] not in over:
                    over[pids[i]] = float(lo[i] + breath
                                          * (rig.hi[i] - lo[i]))
        return over, parts
