#!/usr/bin/env python3
"""
Event-based idle motion: a scheduler of discrete events and a movement layer
that renders them, both fitted to authored clips rather than trained.

Why events
----------
Every continuous generator tried so far (flow/flowprior.py in all its
variants) produces motion that never rests: gaze still 0% of the time against
18-85% authored, brows likewise, and a character that reads as stiff and
non-human. Human idle motion is made of events -- fixate, saccade, hold; the
head follows the eyes a beat later; the weight shifts; the brows change and
stay. This module has that structure by construction and takes every timing
and amplitude it can from the authored clips.

Two layers
----------
  event layer     `schedule(stats, roles, seconds, rng, ...)` returns a list of
                  event tokens, plain dicts sorted by onset `t` (seconds):

                    {"t": 3.21, "type": "saccade", "role": "gaze",
                     "target": {"gaze_x": 0.31, "gaze_y": -0.12},
                     "duration": 0.28, "hold": 1.9}

                  types: saccade (gaze), fixate (gaze, informational: the
                  hold that follows a saccade), head_turn (head; "follows"
                  is the index of the saccade it follows, or null), body_shift
                  (body), brow (all bound brow columns jointly, change and
                  hold), gesture (arms and everything else the rig's own
                  motion files drive; "motion", "start", "fade"), blink.
                  Targets are in the clips' normalised [-1, 1] units, keyed by
                  role. This list is the interface a learned model or a VLM
                  will later emit or condition on; `--tokens out.json` dumps
                  it and `--tokens-in` replays one instead of scheduling.
  movement layer  `render(tokens, roles, seconds, fps, start)` is
                  deterministic in the tokens. Each move is a minimum-jerk
                  profile from wherever the role is to the target; between
                  moves the value is held exactly (zero velocity). A move is
                  never retargeted mid-flight -- restarting from the current
                  position with zero velocity is a velocity step -- so a token
                  arriving during a move waits for it to end.

Fitting (`--fit clips2 --out events.json`)
-----------------------------------------
Per training rig and role group (gaze, head, body, brow; columns bound by
standard ID only), each clip is cut into holds -- runs where every column of
the group changes by less than flow.score.STILL_EPS per frame, the judge's own
definition of still -- and moves between them (amplitude, duration, peak
velocity). Clips are 2-10 s, so a hold that touches a clip end is truncated:
it is a lower bound, and the hold-duration lognormal is fitted by censored
maximum likelihood (complete holds contribute the density, censored ones the
survival function). A clip that keys a group but never moves it is a single
censored hold. Columns a clip does not key (mask False) are not data.

Rigs key motion very unevenly (some barely key the eyes). Every statistic is
fitted per rig, then pooled as the median over the rigs that have enough data
for it (MIN_EVENTS complete events), so one sparse rig cannot set the global
hold time; the per-rig values are kept in the stats file and printed.
Target spread is the std over all frames pooled, not about each clip's mean:
that is what flow.score compares against.

Gestures
--------
Arm parameters are rig-specific (pinyin names, ...T switches) and cannot be
bound by role. Gesture events replay segments of the rig's OWN motion files
(every non-role expressive parameter and part opacity), cross-faded in, back
to back with short rests. Against the rig's own clips that part will score as
copying -- expected. Novelty is measured on the role columns, and the claim
of new motion rests on the event sequencing and targets there.

Blinks are not fitted (eye-open is the autonomic band and not in the clips):
a saccade larger than BLINK_AMP requests a blink with probability P_BLINK,
fed to runtime.Autonomic, which keeps its own Poisson blinks and breath.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

try:
    from .behaviors import (HOLDOUT, ROLES, bind_roles, canon, load_binding,
                            min_jerk)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow.behaviors import (HOLDOUT, ROLES, bind_roles, canon,
                                load_binding, min_jerk)

# Brows are judged as a group of up to eight columns (Hiyori: LY/RY, LX/RX,
# LAngle/RAngle, LForm/RForm); driving only the two heights leaves six held
# and the group's std ratio near a quarter. These stay local:
# behaviors.ROLES is what flow.score uses to pick its novelty columns.
EXTRA_ROLES = {
    "brow_lx": "ParamBrowLX", "brow_rx": "ParamBrowRX",
    "brow_la": "ParamBrowLAngle", "brow_ra": "ParamBrowRAngle",
    "brow_lf": "ParamBrowLForm", "brow_rf": "ParamBrowRForm",
}
ALL_ROLES = {**ROLES, **EXTRA_ROLES}
GROUPS = {
    "gaze": ("gaze_x", "gaze_y"),
    "head": ("head_x", "head_y", "head_z"),
    "body": ("body_x", "body_y", "body_z"),
    "brow": ("brow_l", "brow_r", "brow_lx", "brow_rx", "brow_la", "brow_ra",
             "brow_lf", "brow_rf"),
}
GROUP_OF = {r: g for g, rs in GROUPS.items() for r in rs}
MIRROR = ("head_x", "head_z", "body_x", "body_z", "gaze_x")

STILL_EPS = 0.002       # = flow.score.STILL_EPS (not imported: score imports us
#                         indirectly through behaviors, keep the judge apart)
MIN_HOLD = 3            # frames of velocity below STILL_EPS to call a hold
MIN_EVENTS = 3          # complete events a rig needs to count for a statistic
BLINK_AMP = 0.25        # saccade amplitude (normalised) that may bring a blink
P_BLINK = 0.5

# Normalised units; used for a group no training rig has data for.
DEFAULTS = {
    "groups": {
        g: {"hold": [m, 0.8], "move": [d, 0.5], "amp": [a, 0.6],
            "dur_amp_slope": 0.3}
        for g, m, d, a in (("gaze", 1.5, 0.3, 0.3), ("head", 0.6, 1.2, 0.15),
                           ("body", 0.5, 1.5, 0.1), ("brow", 2.0, 0.4, 0.2))},
    "roles": {r: {"center": 0.0, "std": s, "lo": -2.5 * s, "hi": 2.5 * s}
              for r, s in (("head_x", 0.25), ("head_y", 0.2), ("head_z", 0.15),
                           ("body_x", 0.15), ("body_y", 0.1), ("body_z", 0.1),
                           ("gaze_x", 0.35), ("gaze_y", 0.25),
                           ("brow_l", 0.2), ("brow_r", 0.2),
                           ("brow_lx", 0.1), ("brow_rx", 0.1),
                           ("brow_la", 0.15), ("brow_ra", 0.15),
                           ("brow_lf", 0.15), ("brow_rf", 0.15))},
    "head_follow": {"p": 0.6, "lag": 0.2, "gain": [0.5, 0.3]},
    "brow_loading": {r: 1.0 for r in GROUPS["brow"]},
    "brow_on_saccade": 0.3,
    "body_head_slope": 0.0,
    "gesture": {"seg": [4.0, 0.4], "rest": [0.5, 0.6], "p_rest": 0.3,
                "fade": 0.5, "calm_of": 1, "flip_cost": 0.5},
    "rigs": [], "n_clips": 0, "per_rig": {},
}


# ---------------------------------------------------------------- binding

def bind(pids, allowed=None, override=None):
    """{role: (pid, sign, source)} by standard ID or --binding override only.

    No descriptor fallback: a wrong head or gaze binding is worse than none,
    so a rig without standard IDs needs --binding file.json.
    """
    return bind_roles(pids, allowed, rig_npz=None, override=override,
                      roles=ALL_ROLES)


# ------------------------------------------------------------- extraction

def segment(x, fps, eps=STILL_EPS, min_hold=MIN_HOLD):
    """Holds and moves of one group in one clip.

    x is (T, k): the group's keyed columns. Velocity is the largest per-frame
    change over the columns, so a hold means every column is still -- the
    judge's notion. Returns (holds, moves):
      holds  [(start s, duration s, censored bool, position (k,))]
      moves  [(start s, duration s, censored bool, delta (k,), peak vel /s)]
    Frame indices are of velocity samples: hold [a, b) means x[a..b] is still.
    """
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    T = len(x)
    if T < 2:
        return [], []
    sp = np.abs(np.diff(x, axis=0)).max(axis=1)
    still = sp < eps
    d = np.diff(np.r_[0, still.astype(np.int8), 0])
    runs = [(a, b) for a, b in zip(np.flatnonzero(d == 1),
                                   np.flatnonzero(d == -1))
            if b - a >= min_hold or (a == 0 and b == len(sp))]
    holds, moves = [], []
    n = len(sp)
    for a, b in runs:
        holds.append((a / fps, (b - a) / fps, bool(a == 0 or b == n),
                      x[a].copy()))
    # Moves are the gaps between holds (short still stretches inside a move
    # are part of it: an ease-in-ease-out passes near zero velocity).
    edges = [(0, 0)] + runs + [(n, n)]
    for (_, e0), (s1, _) in zip(edges[:-1], edges[1:]):
        if s1 - e0 <= 0:
            continue
        cens = e0 == 0 or s1 == n
        moves.append((e0 / fps, (s1 - e0) / fps, bool(cens),
                      x[s1] - x[e0], float(sp[e0:s1].max() * fps)))
    return holds, moves


# --------------------------------------------------------------- fitting

def _ndtr_upper(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def censored_lognormal(complete, censored):
    """(median, sigma) of a lognormal by censored maximum likelihood.

    complete: observed durations. censored: lower bounds (the hold was still
    going when the clip ended). Grid search over (mu, sigma), refined twice;
    no scipy. Returns None without at least one complete observation -- with
    only censored data the likelihood grows without bound in mu.
    """
    c = np.log(np.maximum(np.asarray(complete, float), 1e-3))
    r = np.log(np.maximum(np.asarray(censored, float), 1e-3))
    if len(c) == 0:
        return None

    def ll(mu, s):
        z = (c - mu) / s
        out = float((-0.5 * z * z - math.log(s)).sum())
        for v in r:
            out += math.log(max(_ndtr_upper((v - mu) / s), 1e-300))
        return out

    lo_mu, hi_mu = math.log(0.03), math.log(120.0)
    lo_s, hi_s = 0.15, 2.5
    best = None
    for _ in range(3):
        mus = np.linspace(lo_mu, hi_mu, 41)
        ss = np.linspace(lo_s, hi_s, 31)
        for mu in mus:
            for s in ss:
                v = ll(mu, s)
                if best is None or v > best[0]:
                    best = (v, mu, s)
        dm, ds = (hi_mu - lo_mu) / 20, (hi_s - lo_s) / 15
        lo_mu, hi_mu = best[1] - dm, best[1] + dm
        lo_s, hi_s = max(0.1, best[2] - ds), best[2] + ds
    return [float(math.exp(best[1])), float(best[2])]


def kaplan_meier_median(complete, censored):
    """Median duration from the Kaplan-Meier survival curve, or None when
    survival never drops to 0.5 (too much censoring to say)."""
    ev = sorted([(float(x), 1) for x in complete]
                + [(float(x), 0) for x in censored])
    n, s = len(ev), 1.0
    for i, (t, e) in enumerate(ev):
        if e:
            s *= 1 - 1 / (n - i)
            if s <= 0.5:
                return t
    return None


def _lognorm_fit(xs):
    xs = np.asarray([x for x in xs if x > 0], float)
    if len(xs) < MIN_EVENTS:
        return None
    lx = np.log(xs)
    return [float(np.exp(np.median(lx))), float(max(lx.std(), 0.1))]


def _clip_iter(path):
    z = np.load(path, allow_pickle=True)
    pid = [str(p) for p in z["pid"]]
    col = {canon(p): j for j, p in enumerate(pid)}
    fps = float(z["fps"]) if "fps" in z else 30.0
    for i in range(int(z["n_clips"])):
        x = z[f"clip{i}"].astype(np.float32)
        m = z[f"mask{i}"].astype(bool)
        m = m if m.ndim == 1 else m.all(axis=0)
        have = {r: x[:, col[canon(s)]] for r, s in ALL_ROLES.items()
                if canon(s) in col and m[col[canon(s)]]}
        yield fps, have


def fit_rig(path):
    """Per-rig statistics, or None when the rig binds no role by ID."""
    hold = {g: ([], []) for g in GROUPS}
    move = {g: [] for g in GROUPS}         # (dur, amp, peak)
    onsets = {g: [] for g in GROUPS}       # per clip: list of (t, delta, dict)
    frames = {r: [] for r in ALL_ROLES}
    pairs_bh = []
    brow_pairs = {r: [] for r in GROUPS["brow"]}
    follow = {"n": 0, "hit": 0, "lag": [], "gx": [], "gy": []}
    brow_sacc = {"n": 0, "hit": 0}
    n_clips = 0
    fps = 30.0
    for fps, have in _clip_iter(path):
        if not have:
            continue
        n_clips += 1
        for r, v in have.items():
            frames[r].append(v)
        clip_moves = {}
        for g, rs in GROUPS.items():
            cols = [r for r in rs if r in have]
            if not cols:
                continue
            x = np.stack([have[r] for r in cols], 1)
            hs, ms = segment(x, fps)
            for _, d, cens, _ in hs:
                hold[g][1 if cens else 0].append(d)
            clip_moves[g] = (cols, ms)
            for t0, d, cens, delta, pk in ms:
                if not cens:
                    move[g].append((d, float(np.linalg.norm(delta)), pk))
        # Head following gaze: for each complete saccade, the head move whose
        # onset is nearest within [-0.2, 0.6] s.
        if "gaze" in clip_moves and "head" in clip_moves:
            gc, gm = clip_moves["gaze"]
            hc, hm = clip_moves["head"]
            for t0, d, cens, delta, _ in gm:
                if cens or np.linalg.norm(delta) < 0.05:
                    continue
                follow["n"] += 1
                cand = [(abs(h[0] - t0 - 0.15), h) for h in hm
                        if -0.2 <= h[0] - t0 <= 0.6]
                if not cand:
                    continue
                follow["hit"] += 1
                h = min(cand, key=lambda c: c[0])[1]
                follow["lag"].append(h[0] - t0)
                g = dict(zip(gc, delta))
                hd = dict(zip(hc, h[3]))
                if "gaze_x" in g and "head_x" in hd:
                    follow["gx"].append((g["gaze_x"], hd["head_x"]))
                if "gaze_y" in g and "head_y" in hd:
                    follow["gy"].append((g["gaze_y"], hd["head_y"]))
        if "gaze" in clip_moves and "brow" in clip_moves:
            _, gm = clip_moves["gaze"]
            _, bm = clip_moves["brow"]
            for b in bm:
                if b[2]:
                    continue
                brow_sacc["n"] += 1
                brow_sacc["hit"] += any(abs(b[0] - g0[0]) <= 0.3 for g0 in gm)
        if "body_x" in have and "head_x" in have:
            pairs_bh.append((have["body_x"] - have["body_x"].mean(),
                             have["head_x"] - have["head_x"].mean()))
        ref = [have[r] for r in ("brow_l", "brow_r") if r in have]
        if ref:
            e = np.mean(ref, 0)
            e = e - e.mean()
            for r in GROUPS["brow"]:
                if r in have:
                    brow_pairs[r].append((e, have[r] - have[r].mean()))
    if not n_clips:
        return None
    out = {"n_clips": n_clips, "fps": fps, "groups": {}, "roles": {}}
    for g in GROUPS:
        comp, cens = hold[g]
        mv = move[g]
        if not comp and not cens and not mv:
            continue
        gs = {"n_hold": len(comp), "n_censored": len(cens), "n_move": len(mv),
              "hold": (censored_lognormal(comp, cens)
                       if len(comp) >= MIN_EVENTS else None),
              "hold_km_median": kaplan_meier_median(comp, cens),
              "hold_naive_median": (float(np.median(comp)) if comp else None),
              "move": _lognorm_fit([m[0] for m in mv]),
              "amp": _lognorm_fit([m[1] for m in mv]),
              "peak_vel": (float(np.median([m[2] for m in mv])) if mv else None),
              "dur_amp_slope": None}
        if len(mv) >= MIN_EVENTS:
            la = np.log(np.maximum([m[1] for m in mv], 1e-3))
            ld = np.log([m[0] for m in mv])
            if la.std() > 1e-6:
                gs["dur_amp_slope"] = float(np.clip(
                    np.cov(la, ld)[0, 1] / la.var(), 0.0, 0.8))
        out["groups"][g] = gs
    for r, vs in frames.items():
        if vs:
            v = np.concatenate(vs)
            c = float(np.median(v))
            # Left and right are mirror images for an idle: with three
            # training rigs, a side bias (Mao's gaze sits right) is an
            # accident of the sample, and clipping targets to it halved gaze
            # spread. The range of a left-right role is taken symmetric.
            w = np.r_[v, 2 * c - v] if r in MIRROR else v
            out["roles"][r] = {"center": c, "std": float(v.std()),
                               "lo": float(np.percentile(w, 2)),
                               "hi": float(np.percentile(w, 98)),
                               "n": int(len(v))}
    if follow["n"] >= MIN_EVENTS:
        hf = {"p": follow["hit"] / follow["n"], "n": follow["n"],
              "lag": (float(np.median(follow["lag"])) if follow["lag"]
                      else None), "gain": [None, None]}
        for k, key in ((0, "gx"), (1, "gy")):
            ps = np.asarray(follow[key])
            if len(ps) >= MIN_EVENTS:
                hf["gain"][k] = float(np.clip(
                    (ps[:, 0] * ps[:, 1]).sum() / ((ps[:, 0] ** 2).sum() + 1e-9),
                    -1.0, 1.5))
        out["head_follow"] = hf
    if brow_sacc["n"] >= MIN_EVENTS:
        out["brow_on_saccade"] = brow_sacc["hit"] / brow_sacc["n"]
    if pairs_bh:
        b = np.concatenate([p[0] for p in pairs_bh])
        h = np.concatenate([p[1] for p in pairs_bh])
        out["body_head_slope"] = float(np.clip(
            (b * h).sum() / ((h * h).sum() + 1e-9), -1.5, 1.5))
    load = {}
    for r, ps in brow_pairs.items():
        if ps:
            e = np.concatenate([p[0] for p in ps])
            y = np.concatenate([p[1] for p in ps])
            if e.std() > 1e-6 and y.std() > 1e-6:
                load[r] = float(np.corrcoef(e, y)[0, 1])
    if load:
        out["brow_loading"] = load
    return out


def _median(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def pool(per_rig):
    """Rig-balanced pooled statistics: the median of per-rig values over the
    rigs that have the statistic at all."""
    st = json.loads(json.dumps(DEFAULTS))
    st["per_rig"] = per_rig
    st["rigs"] = sorted(per_rig)
    st["n_clips"] = sum(p["n_clips"] for p in per_rig.values())
    st["sources"] = {}
    rigs = list(per_rig.values())
    for g in GROUPS:
        dst = st["groups"][g]
        for k in ("hold", "move", "amp"):
            got = [(n, p["groups"][g][k]) for n, p in per_rig.items()
                   if g in p["groups"] and p["groups"][g][k]]
            if got:
                dst[k] = [_median([v[0] for _, v in got]),
                          _median([v[1] for _, v in got])]
                st["sources"][f"{g}.{k}"] = sorted(n for n, _ in got)
        s = _median([p["groups"][g]["dur_amp_slope"] for p in rigs
                     if g in p["groups"]])
        if s is not None:
            dst["dur_amp_slope"] = s
    for r in ALL_ROLES:
        got = [p["roles"][r] for p in rigs if r in p["roles"]]
        if got:
            st["roles"][r] = {k: _median([v[k] for v in got])
                              for k in ("center", "std", "lo", "hi")}
            st["sources"][f"role.{r}"] = sorted(
                n for n, p in per_rig.items() if r in p["roles"])
    hf = [p["head_follow"] for p in rigs if "head_follow" in p]
    if hf:
        st["head_follow"] = {
            "p": _median([h["p"] for h in hf]),
            "lag": _median([h["lag"] for h in hf]) or DEFAULTS["head_follow"]["lag"],
            "gain": [(_median([h["gain"][k] for h in hf])
                      if any(h["gain"][k] is not None for h in hf)
                      else DEFAULTS["head_follow"]["gain"][k])
                     for k in (0, 1)]}
        st["head_follow"]["lag_fitted"] = st["head_follow"]["lag"]
        # The authored lag is what it is; physiologically the head trails the
        # eyes by 100-300 ms, and that bound is kept.
        st["head_follow"]["lag"] = float(np.clip(st["head_follow"]["lag"],
                                                 0.1, 0.3))
    v = _median([p.get("brow_on_saccade") for p in rigs])
    if v is not None:
        st["brow_on_saccade"] = v
    v = _median([p.get("body_head_slope") for p in rigs])
    if v is not None:
        st["body_head_slope"] = v
    bl = {}
    for r in GROUPS["brow"]:
        v = _median([p.get("brow_loading", {}).get(r) for p in rigs])
        if v is not None:
            bl[r] = v
    st["brow_loading"].update(bl)
    return st


def fit(clip_files, holdout=()):
    per_rig = {}
    skipped = []
    for path in clip_files:
        name = os.path.basename(path)[:-4]
        if name in holdout:
            continue
        r = fit_rig(path)
        if r is None:
            skipped.append(name)
            continue
        per_rig[name] = r
    st = pool(per_rig)
    st["holdout"] = sorted(holdout)
    st["skipped_no_roles"] = skipped
    return st


def load_stats(path):
    base = json.loads(json.dumps(DEFAULTS))
    if not path or path == "default":
        return base
    with open(path, encoding="utf-8") as fh:
        st = json.load(fh)
    for k in ("groups", "roles", "brow_loading", "gesture", "head_follow"):
        base[k].update(st.get(k, {}))
        st[k] = base[k]
    return {**base, **st}


# ------------------------------------------------------------ event layer

def _lognormal(rng, median_sigma):
    m, s = median_sigma
    return float(rng.lognormal(math.log(max(m, 1e-3)), s))


def _move_dur(st, g, amp, rng):
    """Move duration from the fitted main sequence (duration grows with
    amplitude) with lognormal scatter, never below 3 frames' worth."""
    G = st["groups"][g]
    med_amp = max(G["amp"][0], 1e-3)
    d = G["move"][0] * (max(amp, 1e-3) / med_amp) ** G["dur_amp_slope"]
    d *= math.exp(rng.normal(0, 0.5 * G["move"][1]))
    return float(np.clip(d, 0.1, 4.0 * G["move"][0] + 0.1))


def _target(st, roles, rng, scale=1.0, base=None, dev=None):
    """A target for `roles`: base (default the role centre) + dev + spread."""
    out = {}
    for r in roles:
        R = st["roles"][r]
        c = R["center"] if base is None else base[r]
        v = c + (dev or {}).get(r, 0.0) + scale * rng.normal(0, 1) * R["std"]
        out[r] = float(np.clip(v, max(R["lo"], -1), min(R["hi"], 1)))
    return out


def schedule(st, roles, seconds, rng, gestures=None, start=None):
    """Event tokens for `seconds` of idle motion. Semi-Markov per group: each
    group's next event comes after a random hold drawn from its fitted
    duration distribution, and what it is depends on the current state, never
    on a clock.

    roles: role names that are bound (others get no tokens).
    gestures: [(name, n_seconds)] of the rig's motions, or None.
    start: {role: value} initial positions (default the centres).
    """
    roles = [r for r in roles if r in ALL_ROLES]
    have = {g: [r for r in rs if r in roles] for g, rs in GROUPS.items()}
    pos = {r: (start or {}).get(r, st["roles"][r]["center"]) for r in roles}
    toks = []
    busy = {g: 0.0 for g in GROUPS}       # when the group's current move ends
    hf = st["head_follow"]

    def emit(tok):
        toks.append(tok)
        return len(toks) - 1

    def move(g, t, target, extra=None, typ=None):
        amp = float(np.sqrt(sum((target[r] - pos[r]) ** 2 for r in target)))
        d = _move_dur(st, g, amp, rng)
        typ = typ or {"gaze": "saccade", "head": "head_turn",
                      "body": "body_shift", "brow": "brow"}[g]
        tok = {"t": round(t, 4), "type": typ, "role": g,
               "target": {r: round(v, 5) for r, v in target.items()},
               "duration": round(d, 4)}
        tok.update(extra or {})
        pos.update(target)
        busy[g] = t + d
        return emit(tok), amp, d

    nxt = {g: (rng.uniform(0.2, 1.0) if have[g] else math.inf)
           for g in GROUPS}
    head_due = []
    t = 0.0
    while True:
        t = min(min(nxt.values()), min((h[0] for h in head_due), default=math.inf))
        if t >= seconds:
            break
        if head_due and head_due[0][0] <= t:
            due, idx, gt = head_due.pop(0)
            t = max(due, busy["head"])
            gc = {"gaze_x": 0, "gaze_y": 1}
            dev = {}
            for gr, hr in (("gaze_x", "head_x"), ("gaze_y", "head_y")):
                if gr in gt and hr in have["head"]:
                    dev[hr] = hf["gain"][gc[gr]] * (
                        gt[gr] - st["roles"][gr]["center"])
            tgt = _target(st, have["head"], rng, scale=0.5, dev=dev)
            _, _, d = move("head", t, tgt, {"follows": idx})
            nxt["head"] = t + d + _lognormal(rng, st["groups"]["head"]["hold"])
            continue
        g = min(nxt, key=nxt.get)
        t = max(nxt[g], busy[g])
        if g == "gaze":
            tgt = _target(st, have["gaze"], rng)
            idx, amp, d = move("gaze", t, tgt)
            hold = _lognormal(rng, st["groups"]["gaze"]["hold"])
            toks[idx]["hold"] = round(hold, 4)
            emit({"t": round(t + d, 4), "type": "fixate", "role": "gaze",
                  "duration": round(hold, 4)})
            if amp > BLINK_AMP and rng.random() < P_BLINK:
                emit({"t": round(t, 4), "type": "blink", "role": "eyes"})
            if have["head"] and rng.random() < hf["p"]:
                head_due.append((t + hf["lag"], idx, tgt))
                head_due.sort(key=lambda h: h[0])
            if (have["brow"] and amp > BLINK_AMP and t >= busy["brow"]
                    and rng.random() < st["brow_on_saccade"]):
                _brow(st, have["brow"], t, rng, move, nxt)
            nxt["gaze"] = t + d + hold
        elif g == "head":
            tgt = _target(st, have["head"], rng)
            _, _, d = move("head", t, tgt, {"follows": None})
            nxt["head"] = t + d + _lognormal(rng, st["groups"]["head"]["hold"])
        elif g == "body":
            dev = {}
            if "body_x" in have["body"] and "head_x" in pos:
                dev["body_x"] = st["body_head_slope"] * (
                    pos["head_x"] - st["roles"]["head_x"]["center"])
            tgt = _target(st, have["body"], rng, dev=dev)
            _, _, d = move("body", t, tgt)
            nxt["body"] = t + d + _lognormal(rng, st["groups"]["body"]["hold"])
        elif g == "brow":
            _brow(st, have["brow"], t, rng, move, nxt)
    if gestures:
        toks.extend(schedule_gestures(st, gestures, seconds, rng))
    toks.sort(key=lambda k: (k["t"], k["type"]))
    return toks


def _brow(st, cols, t, rng, move, nxt):
    """Brow change and hold: one shared expression value, loaded onto every
    bound brow column by its fitted correlation with the brow heights."""
    e = rng.normal(0, 1)
    tgt = {}
    for r in cols:
        R = st["roles"][r]
        k = float(np.clip(st["brow_loading"].get(r, 1.0), -1, 1))
        v = R["center"] + R["std"] * (k * e + math.sqrt(max(1 - k * k, 0))
                                      * rng.normal(0, 1))
        tgt[r] = float(np.clip(v, max(R["lo"], -1), min(R["hi"], 1)))
    _, _, d = move("brow", t, tgt)
    nxt["brow"] = t + d + _lognormal(rng, st["groups"]["brow"]["hold"])


def gesture_library(motions, keep, lo, hi, switches, fps):
    """What the scheduler needs to know about the rig's own motions.

    motions: [(name, authored_motion dict)]. keep(pid): parameters a gesture
    may write. lo/hi: {pid: range} for normalising. switches: {pid: (a, b)}
    the two authored states of switch-like parameters (runtime.Switches).
    Each entry: name, seconds, `states` (T, k) int8 per switch, 1/0 for the
    state nearer b/a and -1 where the motion does not key it, and `vel` (T,)
    the largest normalised per-frame change over the gesture's parameters.
    """
    sw = sorted(switches)
    lib = []
    for name, m in motions:
        T = int(m["frames"])
        if T < 2:
            continue
        st = np.full((T, len(sw)), -1, np.int8)
        for j, p in enumerate(sw):
            if p in m["params"] and keep(p):
                a, b = switches[p]
                v = np.asarray(m["params"][p], float)
                st[:, j] = (np.abs(v - b) < np.abs(v - a)).astype(np.int8)
        vel = np.zeros(T)
        for p, v in m["params"].items():
            if keep(p) and p in lo and hi[p] > lo[p]:
                d = np.abs(np.diff(np.asarray(v, float))) * 2 / (hi[p] - lo[p])
                vel[1:] = np.maximum(vel[1:], d)
        lib.append({"name": name, "seconds": T / fps, "fps": fps,
                    "states": st, "vel": vel})
    return lib


def _flips(a, b):
    """Switches that change state between two state rows (unkeyed = no flip)."""
    if a is None or b is None or not len(a):
        return 0
    both = (a >= 0) & (b >= 0)
    return int((a[both] != b[both]).sum())


def schedule_gestures(st, gestures, seconds, rng):
    """Gesture tokens: segments of the rig's own motions, back to back with
    occasional short rests. Segment starts are random so replay does not loop,
    never the same motion twice in a row when there is a choice.

    gestures: gesture_library() entries, or plain (name, seconds) pairs.
    With switch states known, a candidate segment is weighted by exp(-flips):
    each arm swap it would force at the seam is a snapped switch ramp, which
    authored clips (one arm variant per motion) almost never contain. With
    velocities known, a segment starts at the calmest frame of a few drawn,
    so the cross-fade joins two poses rather than two moves.
    """
    G = st["gesture"]
    lib = [g if isinstance(g, dict) else {"name": g[0], "seconds": g[1]}
           for g in gestures]
    lib = [g for g in lib if g["seconds"] > 0.3]
    out, t, last, cur_state = [], 0.0, None, None
    if not lib:
        return out
    while t < seconds:
        pool = [g for g in lib if g["name"] != last] or lib
        cands = []
        for g in pool:
            length = g["seconds"]
            seg = min(_lognormal(rng, G["seg"]), length)
            seg = max(seg, min(length, 1.0))
            starts = rng.uniform(0, max(length - seg, 0),
                                 max(int(G.get("calm_of", 1)), 1))
            if "vel" in g:
                fps = g["fps"]
                k = [min(int(s * fps), len(g["vel"]) - 1) for s in starts]
                s0 = float(starts[int(np.argmin(g["vel"][k]))])
            else:
                s0 = float(starts[0])
            st0 = (g["states"][min(int(s0 * g["fps"]), len(g["states"]) - 1)]
                   if "states" in g else None)
            cands.append((g, s0, seg, _flips(cur_state, st0)))
        w = np.exp(-G.get("flip_cost", 1.0)
                   * np.array([c[3] for c in cands], float))
        g, s0, seg, _ = cands[int(rng.choice(len(cands), p=w / w.sum()))]
        out.append({"t": round(t, 4), "type": "gesture", "role": "arms",
                    "motion": g["name"], "start": round(s0, 4),
                    "duration": round(seg, 4), "fade": G["fade"]})
        if "states" in g:
            row = g["states"][min(int((s0 + seg) * g["fps"]),
                                  len(g["states"]) - 1)]
            cur_state = row if cur_state is None else np.where(
                row >= 0, row, cur_state)
        last = g["name"]
        t += seg
        if rng.random() < G["p_rest"]:
            t += _lognormal(rng, G["rest"])
    return out


# --------------------------------------------------------- movement layer

def render(tokens, roles, seconds, fps, start=None):
    """{role: (T,) float32} from move tokens, plus blink request times.

    Deterministic in the tokens. A move starts at max(token time, end of the
    role group's current move): it is never retargeted mid-flight. Between
    moves values are held exactly.
    """
    T = int(round(seconds * fps))
    out = {}
    tracks = {r: [] for r in roles}        # [(t0, dur, target)]
    gbusy = {}
    for tok in tokens:
        if tok["type"] not in ("saccade", "head_turn", "body_shift", "brow"):
            continue
        g = tok["role"]
        t0 = max(float(tok["t"]), gbusy.get(g, 0.0))
        d = max(float(tok["duration"]), 1.0 / fps)
        gbusy[g] = t0 + d
        for r, v in tok["target"].items():
            if r in tracks:
                tracks[r].append((t0, d, float(v)))
    tt = np.arange(T) / fps
    for r in roles:
        v0 = float((start or {}).get(r, 0.0))
        y = np.full(T, v0)
        cur = v0
        for t0, d, tgt in tracks[r]:
            a = int(np.ceil(t0 * fps - 1e-9))
            if a >= T:
                break
            u = (tt[a:] - t0) / d
            y[a:] = cur + (tgt - cur) * min_jerk(u)
            cur = tgt
        out[r] = np.clip(y, -1, 1).astype(np.float32)
    blinks = [float(k["t"]) for k in tokens if k["type"] == "blink"]
    return out, blinks


def generate(st, roles, seconds, fps, rng, gestures=None, start=None):
    """schedule + render: (trajectories, blink times, tokens)."""
    if start is None:
        start = {r: st["roles"][r]["center"] for r in roles if r in ALL_ROLES}
    toks = schedule(st, roles, seconds, rng, gestures, start)
    traj, blinks = render(toks, roles, seconds, fps, start)
    return traj, blinks, toks


class GesturePlayer:
    """Per-frame parameter and part values from gesture tokens.

    motions: {name: authored_motion dict}. keep(pid) says which parameters a
    gesture may write (not roles, held transforms, autonomic or mouth).
    base_params/base_parts: what is shown before the first gesture. A segment
    plays from its `start` for its `duration` and then holds its last frame
    (a rest) until the next gesture. Each gesture fades in over `fade`
    seconds while the previous one keeps playing, as Live2D's own motion
    fade does, with a minimum-jerk weight (a raised cosine steps the
    acceleration at both ends). A parameter a motion does not key sits at
    its base value (else the rig default); a part, at its base opacity.
    """

    def __init__(self, motions, keep, base_params, base_parts, default, fps):
        self.motions = motions
        self.keep = keep
        self.base_params = dict(base_params)
        self.base_parts = dict(base_parts or {})
        self.default = default
        self.fps = fps
        self.param_ids = sorted(
            {p for m in motions.values() for p in m["params"] if keep(p)}
            | set(self.base_params))
        self.part_ids = sorted({p for m in motions.values() for p in m["parts"]}
                               | set(self.base_parts))

    def _rest(self, pid):
        return self.base_params.get(pid, self.default.get(pid, 0.0))

    def base(self):
        return ({p: self._rest(p) for p in self.param_ids},
                {p: self.base_parts.get(p, 1.0) for p in self.part_ids})

    def sample(self, tok, t):
        """Complete (params, parts) of one gesture token at absolute time t."""
        m = self.motions.get(tok["motion"])
        if m is None:
            return self.base()
        el = min(max(t - float(tok["t"]), 0.0), float(tok["duration"]))
        k = int(round((float(tok["start"]) + el) * self.fps))
        k = min(max(k, 0), m["frames"] - 1)
        params = {p: (float(m["params"][p][k]) if p in m["params"]
                      else self._rest(p)) for p in self.param_ids}
        parts = {p: (float(m["parts"][p][k]) if p in m["parts"]
                     else self.base_parts.get(p, 1.0)) for p in self.part_ids}
        return params, parts

    def keys(self, tok):
        m = self.motions.get(tok["motion"])
        return {p for p in m["params"] if self.keep(p)} if m else set()

    def frames(self, tokens, n_frames):
        """[(params, parts)] per frame. Also sets self.keyed: per frame, the
        parameters some playing motion actually keys. The others sit at
        their rest value only because nothing animates them -- the clips
        mark such columns unkeyed, and so does the runtime's --dump, as it
        does under --static."""
        ges = sorted((k for k in tokens if k["type"] == "gesture"),
                     key=lambda k: k["t"])
        out, self.keyed = [], []
        gi = -1
        for f in range(n_frames):
            t = f / self.fps
            while gi + 1 < len(ges) and float(ges[gi + 1]["t"]) <= t + 1e-9:
                gi += 1
            if gi < 0:
                out.append(self.base())
                self.keyed.append(set())
                continue
            cur = ges[gi]
            new = self.sample(cur, t)
            keyed = self.keys(cur)
            fade = max(float(cur.get("fade", 0.5)), 0.0)
            el = t - float(cur["t"])
            if fade > 0 and el < fade:
                old = self.sample(ges[gi - 1], t) if gi > 0 else self.base()
                if gi > 0:
                    keyed = keyed | self.keys(ges[gi - 1])
                w = float(min_jerk(el / fade))
                new = tuple({p: (1 - w) * o[p] + w * n[p] for p in n}
                            for o, n in zip(old, new))
            out.append(new)
            self.keyed.append(keyed)
        return out


# ---------------------------------------------------------------- report

def report(st):
    lines = [f"fitted on {len(st['rigs'])} rigs, {st['n_clips']} clips "
             f"({', '.join(st['rigs'])}); held out "
             f"{', '.join(st.get('holdout', []))}"]
    if st.get("skipped_no_roles"):
        lines.append("  skipped (no standard-ID roles): "
                     + ", ".join(st["skipped_no_roles"]))
    f = lambda v: "-" if v is None else (f"{v:.2f}" if not isinstance(v, list)  # noqa: E731
                                         else "/".join(f"{x:.2f}" for x in v))
    for g in GROUPS:
        G = st["groups"][g]
        lines.append(f"  {g:<5} pooled hold {f(G['hold'])} (median s / sigma)"
                     f"  move {f(G['move'])}  amp {f(G['amp'])}  "
                     f"dur~amp^{G['dur_amp_slope']:.2f}  from "
                     f"{', '.join(st.get('sources', {}).get(g + '.hold', [])) or 'defaults'}")
        for n, p in sorted(st["per_rig"].items()):
            pg = p["groups"].get(g)
            if not pg:
                continue
            lines.append(
                f"      {n:<10} holds {pg['n_hold']:>3} (+{pg['n_censored']} "
                f"censored)  hold {f(pg['hold'])}  KM median "
                f"{f(pg['hold_km_median'])}  naive {f(pg['hold_naive_median'])}"
                f"  moves {pg['n_move']:>3} dur {f(pg['move'])} amp "
                f"{f(pg['amp'])} peak {f(pg['peak_vel'])}/s")
    hf = st["head_follow"]
    lines.append(f"  head follows gaze: p {hf['p']:.2f}  lag {hf['lag']:.2f}s "
                 f"(fitted {f(hf.get('lag_fitted'))})  gain {f(hf['gain'])}")
    lines.append(f"  brow on saccade {st['brow_on_saccade']:.2f}  "
                 f"body/head slope {st['body_head_slope']:+.2f}")
    for r in ALL_ROLES:
        R = st["roles"][r]
        lines.append(f"  {r:<8} centre {R['center']:+.3f} std {R['std']:.3f} "
                     f"p2..p98 {R['lo']:+.2f}..{R['hi']:+.2f}  from "
                     f"{', '.join(st.get('sources', {}).get('role.' + r, [])) or 'defaults'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fit", required=True, metavar="CLIPS",
                    help="clips directory (or a glob of clips .npz)")
    ap.add_argument("--holdout", default=HOLDOUT,
                    help="comma-separated rig names never fitted on")
    ap.add_argument("--out", default="events.json")
    a = ap.parse_args(argv)
    files = (sorted(glob.glob(os.path.join(a.fit, "*.npz")))
             if os.path.isdir(a.fit) else sorted(glob.glob(a.fit)))
    hold = {h.strip() for h in a.holdout.split(",") if h.strip()}
    st = fit(files, hold)
    if not st["rigs"]:
        print("no training clips (all held out?)", file=sys.stderr)
        return 1
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1)
    print(report(st))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
