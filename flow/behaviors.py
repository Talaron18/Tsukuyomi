#!/usr/bin/env python3
"""
Hand-made idle behaviours: the baseline a learned motion model must beat.

The long-term plan is a learned generator conditioned on a language model's
hidden state. A generator that cannot out-score a few hundred lines of
procedural gaze and head motion is not worth its training run, so this is the
floor, scored with the same tools (flow.score) as every learned candidate.

Roles
-----
Only eight to ten parameters are driven: head yaw/pitch/roll, body X/Y/Z,
gaze X/Y and the two brow heights. They are bound by standard ID
(ParamAngleX, PARAM_ANGLE_X, ...). A rig without the ID for a role falls back
to binding_eval's descriptor retrieval, which is right about three times in
four, so every fallback is printed for a person to confirm. `--binding
file.json` ({"head_x": "ParamFoo"} or {"head_x": ["ParamFoo", -1]}) overrides
both. Every other expressive parameter is held at the idle motion's first
frame (or the rig default when there is no idle): a baseline that invents arm
gestures would be measuring my taste, not a floor.

Primitives
----------
  gaze   saccade-and-fixate. A semi-Markov chain over "looking at the viewer"
         and "looking away", each with its own lognormal dwell. No fixed
         period anywhere: the scorer's periodicity check must pass.
  head   follows each gaze target after a 100-300 ms lag, with the authored
         head-on-gaze slope (smaller amplitude), plus occasional nods (Y dip
         and return) and tilts (new Z target).
  body   slow sway of its own plus a lagged response to the head, with the
         authored body-on-head slope -- negative is counter-sway.
  brow   occasional raise on a large gaze shift.
  blink  a large gaze shift requests a blink at saccade onset (people blink
         with big eye movements). Runtime feeds these to Autonomic, which
         keeps its own Poisson blinks and breath.

Every transition is a minimum-jerk profile (Flash & Hogan 1985). Saccade
durations come from the authored clips, not physiology: a real 40 ms saccade
is 1-2 frames, and authored Live2D gaze is eased over many more, so the
physiological value would fail the jerk check against the rig's own motion.

Statistics
----------
`python -m flow.behaviors --fit clips/ --holdout ... --out stats.json` pools
the standard-ID columns of every TRAINING rig's clips (normalised [-1,1]
units) into per-role centre/spread, transition durations, inter-event
intervals and the head/gaze and body/head slopes. Held-out rigs are skipped
by name. Without a fit, DEFAULTS are used.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HOLDOUT = "aidang_2,Natori,Hiyori,lafei,xuefeng"

# role -> standard ID. Matching ignores case and underscores, so
# PARAM_ANGLE_X (Cubism 2 style) binds the same as ParamAngleX.
ROLES = {
    "head_x": "ParamAngleX", "head_y": "ParamAngleY", "head_z": "ParamAngleZ",
    "body_x": "ParamBodyAngleX", "body_y": "ParamBodyAngleY",
    "body_z": "ParamBodyAngleZ",
    "gaze_x": "ParamEyeBallX", "gaze_y": "ParamEyeBallY",
    "brow_l": "ParamBrowLY", "brow_r": "ParamBrowRY",
}
# Roles binding_eval can retrieve by descriptor. Brow has no reference there.
FALLBACK_ROLES = {"head_x", "head_y", "head_z", "body_x", "body_y", "body_z",
                  "gaze_x", "gaze_y"}

# Normalised units. Rough mid-values of the Cubism samples, for a rig with no
# fit; `--fit` replaces them.
DEFAULTS = {
    "roles": {r: {"center": 0.0, "std": s, "lo": -2 * s, "hi": 2 * s}
              for r, s in (("head_x", 0.25), ("head_y", 0.2), ("head_z", 0.15),
                           ("body_x", 0.15), ("body_y", 0.1), ("body_z", 0.1),
                           ("gaze_x", 0.35), ("gaze_y", 0.25),
                           ("brow_l", 0.2), ("brow_r", 0.2))},
    "gaze_move_s": 0.25, "head_move_s": 0.6, "body_move_s": 1.6,
    "gaze_dwell_s": [1.2, 0.6],     # lognormal (median s, sigma)
    "head_dwell_s": [2.0, 0.8],
    "body_dwell_s": [3.0, 0.8],
    "head_gaze_slope": [0.5, 0.5],  # x, y
    "body_head_slope": -0.3,
    "n_clips": 0, "rigs": [],
}


def canon(pid: str) -> str:
    return pid.lower().replace("_", "")


def min_jerk(u):
    """Position profile 10u^3 - 15u^4 + 6u^5: zero velocity and acceleration
    at both ends, so chaining moves never kicks the jerk metric."""
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


# ------------------------------------------------------------------ binding

def bind_roles(pids, allowed=None, rig_npz=None, override=None, roles=None):
    """{role: (pid, sign, source)}. `allowed` limits candidates (column ids).

    `roles` ({role: standard id}, default ROLES) lets flow.events bind its
    extra brow columns the same way.

    source is "id", "override" or "fallback". Fallback needs rig_npz to sit in
    a directory with other exported rigs, whose standard-ID parameters are the
    reference -- binding_eval.rank_roles, CSLS over descriptors.
    """
    allowed = list(pids) if allowed is None else list(allowed)
    by_canon = {canon(p): p for p in allowed}
    out = {}
    for role, sid in (ROLES if roles is None else roles).items():
        if canon(sid) in by_canon:
            out[role] = (by_canon[canon(sid)], 1.0, "id")
    for role, v in (override or {}).items():
        pid, sign = (v, 1.0) if isinstance(v, str) else (v[0], float(v[1]))
        if pid in allowed:
            out[role] = (pid, sign, "override")
    missing = [r for r in FALLBACK_ROLES if r not in out]
    if missing and rig_npz:
        out.update(_fallback(missing, rig_npz, allowed, out))
    return out


def _fallback(missing, rig_npz, allowed, bound):
    try:
        from ..l2dprobe.binding_eval import load_rig_files, rank_roles
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        from l2dprobe.binding_eval import load_rig_files, rank_roles
    name = os.path.basename(rig_npz)[:-4]
    packs = load_rig_files(os.path.dirname(os.path.abspath(rig_npz)))
    if name not in packs or len(packs) < 2:
        return {}
    ranked, _ = rank_roles(packs, name, use_csls=True)
    used = {p for p, _, _ in bound.values()}
    std_ids = {canon(s) for s in ROLES.values()}
    ok = set(allowed)
    out = {}
    for role in missing:
        for pid in ranked.get(ROLES[role], []):
            if pid in ok and pid not in used and canon(pid) not in std_ids:
                out[role] = (pid, 1.0, "fallback")
                used.add(pid)
                break
    return out


def load_binding(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ fitting

def _events(v, fps, thr):
    """(durations s, sizes, onset times s) of runs where |dv| exceeds thr."""
    sp = np.abs(np.diff(v, axis=0))
    sp = sp.max(axis=1) if sp.ndim > 1 else sp
    moving = np.r_[0, (sp > thr).astype(np.int8), 0]
    d = np.diff(moving)
    on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    dur, size = [], []
    for a, b in zip(on, off):
        dur.append((b - a) / fps)
        size.append(float(np.linalg.norm(np.atleast_1d(v[min(b, len(v) - 1)]
                                                       - v[a]))))
    return dur, size, list(on / fps)


def _lognorm(xs, default):
    xs = [x for x in xs if x > 0]
    if len(xs) < 4:
        return default
    lx = np.log(xs)
    return [float(np.exp(np.median(lx))), float(np.clip(lx.std(), 0.5, 1.0))]


def fit(clip_files, holdout=()):
    """Pool standard-ID role columns of the training rigs' clips."""
    vals = {r: [] for r in ROLES}
    pair = {"hg_x": [], "hg_y": [], "bh": []}
    ev = {"gaze": ([], []), "head": ([], []), "body": ([], [])}
    rigs, n = [], 0
    thr = {"gaze": 0.01, "head": 0.004, "body": 0.002}
    for path in clip_files:
        name = os.path.basename(path)[:-4]
        if name in holdout:
            continue
        z = np.load(path, allow_pickle=True)
        col = {canon(str(p)): j for j, p in enumerate(z["pid"])}
        fps = float(z["fps"]) if "fps" in z else 30.0
        rigs.append(name)
        for i in range(int(z["n_clips"])):
            x, m = z[f"clip{i}"].astype(np.float32), z[f"mask{i}"].astype(bool)
            n += 1
            have = {r: x[:, col[canon(s)]] for r, s in ROLES.items()
                    if canon(s) in col and m[col[canon(s)]]}
            for r, v in have.items():
                vals[r].append(v)
            for a, b, k in (("head_x", "gaze_x", "hg_x"),
                            ("head_y", "gaze_y", "hg_y"),
                            ("body_x", "head_x", "bh")):
                if a in have and b in have:
                    pair[k].append((have[a] - have[a].mean(),
                                    have[b] - have[b].mean()))
            for g, keys in (("gaze", ("gaze_x", "gaze_y")),
                            ("head", ("head_x", "head_y")),
                            ("body", ("body_x", "body_z"))):
                cols = [have[k] for k in keys if k in have]
                if not cols:
                    continue
                dur, _, on = _events(np.stack(cols, 1), fps, thr[g])
                ev[g][0].extend(dur)
                ev[g][1].extend(np.diff(on).tolist())
    st = json.loads(json.dumps(DEFAULTS))
    st.update(n_clips=n, rigs=rigs, holdout=sorted(holdout))
    for r, vs in vals.items():
        if not vs:
            continue
        v = np.concatenate(vs)
        st["roles"][r] = {
            "center": float(np.median(v)),
            # Spread about each clip's own mean: one clip parked at an extreme
            # is a pose, not motion amplitude.
            "std": float(np.sqrt(np.mean(np.concatenate(
                [(u - u.mean()) ** 2 for u in vs])))),
            "lo": float(np.percentile(v, 5)), "hi": float(np.percentile(v, 95)),
            "n": int(len(v))}
    for k, out in (("hg_x", ("head_gaze_slope", 0)),
                   ("hg_y", ("head_gaze_slope", 1)), ("bh", ("body_head_slope",
                                                             None))):
        if not pair[k]:
            continue
        a = np.concatenate([p[0] for p in pair[k]])
        b = np.concatenate([p[1] for p in pair[k]])
        slope = float(np.clip((a * b).sum() / ((b * b).sum() + 1e-9), -1.5, 1.5))
        if out[1] is None:
            st[out[0]] = slope
        else:
            st[out[0]][out[1]] = slope
    for g in ("gaze", "head", "body"):
        dur, gaps = ev[g]
        if len(dur) >= 4:
            st[f"{g}_move_s"] = float(np.median(dur))
        st[f"{g}_dwell_s"] = _lognorm(gaps, st[f"{g}_dwell_s"])
    return st


def load_stats(path):
    if not path or path == "default":
        return json.loads(json.dumps(DEFAULTS))
    with open(path, encoding="utf-8") as fh:
        st = json.load(fh)
    base = json.loads(json.dumps(DEFAULTS))
    base["roles"].update(st.get("roles", {}))
    st["roles"] = base["roles"]
    return {**base, **st}


# --------------------------------------------------------------- generation

class Track:
    """One scalar driven by chained minimum-jerk moves."""

    def __init__(self, v):
        self.v0 = self.v1 = float(v)
        self.t0, self.dur = 0.0, 1.0

    def value(self, t):
        return self.v0 + (self.v1 - self.v0) * float(
            min_jerk((t - self.t0) / self.dur))

    def busy(self, t):
        return t < self.t0 + self.dur

    def go(self, t, target, dur):
        self.v0, self.v1 = self.value(t), float(target)
        self.t0, self.dur = t, max(dur, 1e-3)


def generate(stats, roles, seconds, fps, rng):
    """{role: (T,) normalised values} for the roles present, and blink times.

    An event loop over frames. Each primitive owns its next-event time drawn
    from a lognormal dwell, so the schedule is semi-Markov: what happens next
    and when depends on the current state, never on a clock.
    """
    R = stats["roles"]
    T = int(round(seconds * fps))
    dwell = lambda k: float(rng.lognormal(np.log(stats[k][0]),  # noqa: E731
                                          stats[k][1]))
    clip = lambda r, v: float(np.clip(v, max(R[r]["lo"], -1),  # noqa: E731
                                      min(R[r]["hi"], 1)))
    c = {r: R[r]["center"] for r in ROLES}
    tr = {r: Track(c[r]) for r in ROLES}
    nod = Track(0.0)
    gaze_mv = max(stats["gaze_move_s"], 0.2)
    head_mv = max(stats["head_move_s"], 0.35)
    body_mv = max(stats["body_move_s"], 1.0)
    sx, sy = stats["head_gaze_slope"]
    away = False
    t_gaze, t_body, t_tilt = 0.3, dwell("body_dwell_s"), dwell("head_dwell_s")
    t_nod, t_brow_back = rng.exponential(8.0), None
    head_due = None
    blinks = []
    # Body's response to the head goes through a critically damped
    # second-order lag (~1 s): smooth enough that it adds no jerk.
    lag_x = lag_v = 0.0
    w = 2 * np.pi / 3.0
    out = {r: np.empty(T, np.float32) for r in roles}
    for f in range(T):
        t = f / fps
        # A move is never retargeted mid-flight: restarting from the current
        # position with zero velocity is a velocity step, and the jerk metric
        # (rightly) reads that as a twitch. Events wait for the move to end.
        if t >= t_gaze and not tr["gaze_x"].busy(t):
            # Looking away dwells shorter than holding the viewer's eye.
            away = (not away) if rng.random() < 0.7 else away
            if away:
                gx = clip("gaze_x", c["gaze_x"] + rng.normal(0, 1.3) * R["gaze_x"]["std"])
                gy = clip("gaze_y", c["gaze_y"] + rng.normal(0, 1.3) * R["gaze_y"]["std"])
            else:
                gx = c["gaze_x"] + rng.normal(0, 0.15) * R["gaze_x"]["std"]
                gy = c["gaze_y"] + rng.normal(0, 0.15) * R["gaze_y"]["std"]
            jump = np.hypot(gx - tr["gaze_x"].value(t),
                            gy - tr["gaze_y"].value(t))
            tr["gaze_x"].go(t, gx, gaze_mv * rng.uniform(0.8, 1.3))
            tr["gaze_y"].go(t, gy, gaze_mv * rng.uniform(0.8, 1.3))
            big = jump > 1.5 * max(R["gaze_x"]["std"], 0.05)
            if big and rng.random() < 0.6:
                blinks.append(t)
            if (big and t_brow_back is None and not tr["brow_l"].busy(t)
                    and rng.random() < 0.25):
                # Varied height and hold: identical pulses repeat, and the
                # periodicity check sees the repeat.
                amp, d = rng.uniform(0.5, 1.6), rng.uniform(0.25, 0.5)
                for r in ("brow_l", "brow_r"):
                    tr[r].go(t, clip(r, c[r] + amp * R[r]["std"]), d)
                t_brow_back = t + d + rng.lognormal(np.log(1.0), 0.5)
            elif (t_brow_back is None and not tr["brow_l"].busy(t)
                  and rng.random() < 0.3):
                # Small unprompted brow shifts between raises. Raises alone
                # are sparse identical-looking pulses, which a 60 s window
                # can read as a loop.
                for r in ("brow_l", "brow_r"):
                    tr[r].go(t, clip(r, c[r] + rng.normal(0, 0.3) * R[r]["std"]),
                             rng.uniform(0.4, 0.9))
            head_due = (t + rng.uniform(0.1, 0.3), gx, gy)
            t_gaze = t + dwell("gaze_dwell_s") * (0.6 if away else 1.0)
        if head_due and t >= head_due[0] and not tr["head_x"].busy(t):
            _, gx, gy = head_due
            hx = c["head_x"] + sx * (gx - c["gaze_x"]) + rng.normal(0, 0.3) * R["head_x"]["std"]
            hy = c["head_y"] + sy * (gy - c["gaze_y"]) + rng.normal(0, 0.3) * R["head_y"]["std"]
            d = head_mv * rng.uniform(0.8, 1.4)
            tr["head_x"].go(t, clip("head_x", hx), d)
            tr["head_y"].go(t, clip("head_y", hy), d)
            head_due = None
        if t >= t_tilt and not tr["head_z"].busy(t):
            tr["head_z"].go(t, clip("head_z", c["head_z"] + rng.normal(0, 1.0)
                                    * R["head_z"]["std"]),
                            head_mv * rng.uniform(1.0, 1.8))
            t_tilt = t + dwell("head_dwell_s") * 1.5
        if t >= t_nod and not nod.busy(t) and nod.v1 == 0.0:
            amp = -rng.uniform(0.6, 1.2) * R["head_y"]["std"]
            nod.go(t, amp, rng.uniform(0.25, 0.4))
            t_nod = t + rng.exponential(9.0) + 1.5
        if nod.v1 != 0.0 and not nod.busy(t):
            nod.go(t, 0.0, rng.uniform(0.35, 0.6))
        if t_brow_back is not None and t >= t_brow_back:
            if not tr["brow_l"].busy(t):
                d = rng.uniform(0.4, 0.8)
                for r in ("brow_l", "brow_r"):
                    tr[r].go(t, c[r], d)
                t_brow_back = None
        if t >= t_body and not tr["body_x"].busy(t):
            d = body_mv * rng.uniform(0.8, 1.5)
            for r in ("body_x", "body_y", "body_z"):
                tr[r].go(t, clip(r, c[r] + rng.normal(0, 0.8) * R[r]["std"]), d)
            # Two states: swaying, or settled for a long stretch. One dwell
            # distribution alone left sway near-regular over a minute.
            t_body = t + d + dwell("body_dwell_s") * (
                rng.uniform(2.0, 4.0) if rng.random() < 0.25 else 1.0)
        # second-order lag of head X, for the body's coupled component
        hx_now = tr["head_x"].value(t) - c["head_x"]
        lag_v += (w * w * (hx_now - lag_x) - 2 * w * lag_v) / fps
        lag_x += lag_v / fps
        for r in roles:
            v = tr[r].value(t)
            if r == "head_y":
                v += nod.value(t)
            elif r == "body_x":
                v += stats["body_head_slope"] * lag_x
            out[r][f] = np.clip(v, -1, 1)
    return out, blinks


# ---------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fit", required=True, metavar="CLIPS",
                    help="clips directory (or a glob of clips .npz)")
    ap.add_argument("--holdout", default=HOLDOUT,
                    help="comma-separated rig names never fitted on")
    ap.add_argument("--out", default="behaviors.json")
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
    print(f"fitted on {len(st['rigs'])} rigs, {st['n_clips']} clips "
          f"({', '.join(st['rigs'])}); held out {', '.join(sorted(hold))}")
    for r, s in st["roles"].items():
        print(f"  {r:<7} centre {s['center']:+.3f}  std {s['std']:.3f}  "
              f"p5..p95 {s['lo']:+.2f}..{s['hi']:+.2f}  n={s.get('n', 0)}")
    print(f"  moves s: gaze {st['gaze_move_s']:.2f} head "
          f"{st['head_move_s']:.2f} body {st['body_move_s']:.2f}; dwell "
          f"(median, sigma) gaze {st['gaze_dwell_s']} head "
          f"{st['head_dwell_s']} body {st['body_dwell_s']}")
    print(f"  slopes head/gaze {st['head_gaze_slope']}  body/head "
          f"{st['body_head_slope']:+.2f}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
