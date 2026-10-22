#!/usr/bin/env python3
"""
Score a generated motion trajectory against the rig's own authored motion.

Every judgement of the prior so far has been a look at a video. That does not
scale past one rig, and it is the wrong instrument for the failures that keep
recurring: a prior that collapses to a held pose looks "calm", one that loops a
sinusoid looks "alive" for the first ten seconds, and a switch parameter left
hovering at 0.4 is a faint ghost limb that is easy to miss at 256 px. Each of
those has a number, and this script computes them.

The reference is the rig's own authored clips (clips/<name>.npz), measured the
same way. A ratio of 1.0 means "moves like the rigger's own animation". The
comparison is per functional group because the failures are: a model can match
overall variance while the head is frozen and the arms thrash.

Metrics, each generated / authored unless stated:

  std      temporal std per parameter, averaged over the group. Collapse shows
           as a ratio near 0, thrashing as a ratio well above 1.
  |d|      mean and 95th-percentile per-frame change. Speed, and burstiness.
  jerk     mean |third difference|. Smoothness: jitter is invisible in std but
           dominates jerk.
  still    fraction of 1 s windows whose largest per-frame change is below
           STILL_EPS (generated vs authored, both shown). Authored idles hold
           poses; a prior that never rests reads as nervous, one that always
           rests is dead.
  period   peak normalised autocorrelation of the group's mean signal at lags
           of 2-20 s. Flags sinusoid loops. See periodic_peak for why this is
           not a plain maximum.
  toggle   for parameters that authored motion uses as switches
           (motions.toggle_params), fraction of frames strictly inside 10-90%
           of the authored range. Each such frame is a partially drawn limb.

Only columns the authored clips actually drive are scored: a column no clip
touches has zero authored variance, so any ratio on it is meaningless, and the
masks are what says which columns those are.

Pass/fail, per rig (the roadmap thresholds):
  overall std ratio in [0.7, 1.3]   within 30% of authored liveliness
  overall jerk ratio <= 1.5         no more than 50% rougher than authored
  toggle in-between <= authored + 0.05
  no group with periodic peak > max(0.6, authored peak + 0.1)
The 0.6 cut sits between incidental repetition and a loop that dominates the
signal: a pure sinusoid scores ~1.0, a sinusoid under noise of equal power
~0.5-0.7, and most authored clips 0 (no qualifying peak) to ~0.45. The
roadmap states it as an absolute bound, but that fails authored motion
against itself on the samples: Mao's body scores 0.71 on a 7.8 s clip that
repeats a gesture, and Mark's idle arm and body sways are authored loops
(0.99, 1.00). So a group fails only when it is
also clearly more periodic than the rig's own animation of that group. Where
no authored clip is long enough to judge (see periodicity), the authored peak
counts as 0 and the bound is the absolute 0.6.

Usage:
    python -m flow.runtime --model M --rig rigs/Mao.npz --clips clips/Mao.npz \\
        --ckpt prior.pt --seconds 90 --dump run.npz --dump-only
    python -m flow.score --traj run.npz --clips clips/Mao.npz \\
        --rig rigs/Mao.npz --model M --json score.json

`--traj` may also be the clips file itself: authored scored against itself
must pass, which is the sanity check on the metric code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    from ..l2dprobe.motions import toggle_params
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from l2dprobe.motions import toggle_params

STD_BAND = (0.7, 1.3)
JERK_MAX = 1.5
TOGGLE_SLACK = 0.05
PERIOD_MAX = 0.6
# Per-frame change, in normalised units (range 2), below which a window counts
# as held. 0.002 is 0.1% of the range per frame, 3% per second: below what
# reads as deliberate motion at 30 fps, above bezier ease-out tails.
STILL_EPS = 0.002
# A periodic peak must rise at least this far above the lowest correlation at
# a shorter lag. Slow drift gives a high but monotonically falling
# autocorrelation; only a real repeat dips and comes back.
PERIOD_TROUGH = 0.3


# ------------------------------------------------------------------ loading

def segments_of(path: str) -> dict:
    """A trajectory file as a list of (X (T,P), M (T,P) bool) segments.

    Accepts both formats. A clips file is a set of clips each with a per-
    parameter mask; a runtime --dump is one rollout whose `segment` array
    marks where motion files were concatenated (under --static), with a
    per-frame mask. Statistics never span a segment boundary, so a seam
    between two clips is not scored as a jerk.
    """
    z = np.load(path, allow_pickle=True)
    pid = [str(p) for p in z["pid"]]
    fps = float(z["fps"]) if "fps" in z else 30.0
    segs = []
    if "n_clips" in z:
        for i in range(int(z["n_clips"])):
            x = z[f"clip{i}"].astype(np.float32)
            m = z[f"mask{i}"].astype(bool)
            segs.append((x, np.broadcast_to(m, x.shape)))
        kind = "clips"
    elif "traj" in z:
        x = z["traj"].astype(np.float32)
        m = (z["mask"].astype(bool) if "mask" in z
             else np.ones(x.shape, bool))
        seg = z["segment"] if "segment" in z else np.zeros(len(x), int)
        cuts = np.flatnonzero(np.diff(seg)) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(x)]):
            segs.append((x[a:b], m[a:b]))
        kind = "dump:" + (str(z["mode"]) if "mode" in z else "?")
    else:
        raise ValueError(f"{path}: neither a clips file nor a runtime --dump")
    row = z["param_row"].astype(int) if "param_row" in z else None
    return {"pid": pid, "fps": fps, "segs": segs, "kind": kind, "row": row}


def group_labels(rig_npz: str, pid: list[str], row) -> list[str]:
    """Functional group per column, from the rig file's `group` field.

    Rig files exported before analyze.GROUPS put body ahead of head label
    ParamBodyAngleX/Y/Z as head. Head and body are the two groups whose
    behaviour differs most (body sways slowly, head darts), so they are split
    back apart here for those files rather than scored as one.
    """
    z = np.load(rig_npz, allow_pickle=True)
    rpid = [str(p) for p in z["pid"]]
    rgroup = [str(g) for g in z["group"]]
    index = {p: i for i, p in enumerate(rpid)}
    out = []
    for c, p in enumerate(pid):
        i = int(row[c]) if row is not None else index.get(p)
        if i is None or i >= len(rpid) or rpid[i] != p:
            i = index.get(p)
        g = rgroup[i] if i is not None else "other"
        if "bodyangle" in p.lower():
            g = "body"
        out.append(g)
    return out


# ------------------------------------------------------------------ metrics

def _runs(m: np.ndarray):
    """Start/stop of runs of True in a 1-D mask."""
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def column_stats(segs, cols) -> dict:
    """Per-column std, mean/p95 |d|, mean |d3|, over driven frames only."""
    out = {}
    for c in cols:
        vals, d1, d3 = [], [], []
        for x, m in segs:
            for a, b in _runs(m[:, c]):
                v = x[a:b, c]
                vals.append(v)
                if len(v) > 1:
                    d1.append(np.abs(np.diff(v)))
                if len(v) > 3:
                    d3.append(np.abs(np.diff(v, 3)))
        if not vals or sum(len(v) for v in vals) < 2:
            continue
        v = np.concatenate(vals)
        d1 = np.concatenate(d1) if d1 else np.zeros(1)
        d3 = np.concatenate(d3) if d3 else np.zeros(1)
        out[c] = {"std": float(v.std()), "dmean": float(d1.mean()),
                  "dp95": float(np.percentile(d1, 95)),
                  "jerk": float(d3.mean())}
    return out


def stillness(segs, cols, fps: float) -> float:
    """Fraction of 1 s windows in which nothing in `cols` moves noticeably."""
    w = max(int(round(fps)), 2)
    n = still = 0
    for x, m in segs:
        for s in range(0, len(x) - w + 1, w):
            live = [c for c in cols if m[s:s + w, c].all()]
            if not live:
                continue
            d = np.abs(np.diff(x[s:s + w][:, live], axis=0)).max()
            n += 1
            still += d < STILL_EPS
    return still / n if n else float("nan")


def autocorr(sig: np.ndarray, max_lag: int) -> np.ndarray:
    """Pearson correlation of sig with itself at lags 0..max_lag."""
    r = np.zeros(max_lag + 1)
    r[0] = 1.0
    for k in range(1, max_lag + 1):
        a, b = sig[:-k], sig[k:]
        sa, sb = a.std(), b.std()
        r[k] = 0.0 if sa < 1e-9 or sb < 1e-9 else float(
            ((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return r


def periodic_peak(sig: np.ndarray, fps: float,
                  lo_s: float = 2.0, hi_s: float = 20.0) -> tuple[float, float]:
    """(peak, plain max) of the autocorrelation at lags lo_s..hi_s.

    The plain maximum is what the roadmap names, and it is also reported, but
    on its own it fires on any slow drift: a head that turns and stays turned
    is correlated with itself 2 s later without repeating anything. A peak
    here must be a local maximum that the correlation climbed back to from a
    trough at least PERIOD_TROUGH lower -- the dip-and-return signature of a
    repeat. Lags are capped at a third of the segment: at half, a 5.6 s Mao
    clip had a handful of lags whose correlation rested on under three
    seconds of data and scored 0.92 on an ordinary authored arm gesture. A constant signal (a collapsed prior) scores 0:
    it is broken, but not periodic, and std already says so.
    """
    n = len(sig)
    lo, hi = int(round(lo_s * fps)), min(int(round(hi_s * fps)), n // 3)
    if hi <= lo or sig.std() < 1e-6:
        return 0.0, 0.0
    r = autocorr(sig, hi + 1)
    plain = float(r[lo:hi + 1].max())
    peak = 0.0
    for k in range(lo, hi + 1):
        if r[k] >= r[k - 1] and r[k] >= r[k + 1]:
            if r[1:k].min() <= r[k] - PERIOD_TROUGH:
                peak = max(peak, float(r[k]))
    return peak, plain


def periodicity(segs, cols, fps: float) -> tuple[float, float, int]:
    """Worst periodic peak of the group's mean signal over segments.

    Returns (peak, plain max, number of segments long enough to judge).
    """
    best = plain = 0.0
    judged = 0
    for x, m in segs:
        live = [c for c in cols if m[:, c].all()]
        if not live or len(x) < 3 * int(round(2.0 * fps)) + 3:
            continue
        sig = x[:, live].mean(axis=1)
        p, q = periodic_peak(sig, fps)
        best, plain = max(best, p), max(plain, q)
        judged += 1
    return best, plain, judged


def inbetween(segs, cols, states: dict) -> tuple[float, int]:
    """Fraction of driven frames strictly inside 10-90% of the switch range."""
    hit = tot = 0
    for c in cols:
        lo, hi = states[c]
        if hi - lo < 1e-6:
            continue
        for x, m in segs:
            v = x[m[:, c], c]
            u = (v - lo) / (hi - lo)
            hit += int(((u > 0.1) & (u < 0.9)).sum())
            tot += len(v)
    return (hit / tot if tot else float("nan")), tot


def _ratio(g: float, a: float) -> float:
    if a < 1e-9:
        return float("nan") if g < 1e-9 else float("inf")
    return g / a


def summarize(gen, ref, cols, fps):
    """Ratios and side-by-side numbers for one set of columns."""
    gs, rs = column_stats(gen, cols), column_stats(ref, cols)
    # A column that clips drive but hold constant (Mao's ParamCheek) has no
    # authored motion to compare against; any movement would score inf.
    both = [c for c in cols if c in gs and c in rs and rs[c]["std"] > 1e-6]
    if not both:
        return None
    mean = lambda d, k: float(np.mean([d[c][k] for c in both]))  # noqa: E731
    out = {"n": len(both)}
    for k in ("std", "dmean", "dp95", "jerk"):
        out[k] = _ratio(mean(gs, k), mean(rs, k))
        out[k + "_gen"], out[k + "_ref"] = mean(gs, k), mean(rs, k)
    out["still_gen"] = stillness(gen, both, fps)
    out["still_ref"] = stillness(ref, both, fps)
    p, q, n = periodicity(gen, both, fps)
    pr, qr, nr = periodicity(ref, both, fps)
    out.update(period_gen=p, period_gen_plain=q, period_gen_n=n,
               period_ref=pr, period_ref_plain=qr, period_ref_n=nr)
    return out


def score(traj_path: str, clips_path: str, rig_path: str,
          model_json: str | None = None) -> dict:
    gen = segments_of(traj_path)
    ref = segments_of(clips_path)
    if gen["pid"] != ref["pid"]:
        raise ValueError(
            "trajectory and clips have different parameter columns "
            f"({len(gen['pid'])} vs {len(ref['pid'])}); dump the trajectory "
            "with the same rig file the clips were built from")
    fps = ref["fps"]
    if abs(gen["fps"] - fps) > 1e-3:
        raise ValueError(f"fps differs: trajectory {gen['fps']} vs clips {fps}")
    pid = ref["pid"]
    groups = group_labels(rig_path, pid, ref["row"])
    driven = [c for c in range(len(pid))
              if any(m[:, c].any() for _, m in ref["segs"])]

    res = {"traj": traj_path, "kind": gen["kind"], "clips": clips_path,
           "fps": fps, "groups": {}}
    for g in sorted({groups[c] for c in driven}):
        cols = [c for c in driven if groups[c] == g]
        s = summarize(gen["segs"], ref["segs"], cols, fps)
        if s is not None:
            s["params"] = [pid[c] for c in cols]
            res["groups"][g] = s
    res["overall"] = summarize(gen["segs"], ref["segs"], driven, fps)

    res["toggle"] = None
    if model_json:
        rz = np.load(rig_path, allow_pickle=True)
        rpid = [str(p) for p in rz["pid"]]
        lo = {p: float(v) for p, v in zip(rpid, rz["minimum"])}
        hi = {p: float(v) for p, v in zip(rpid, rz["maximum"])}
        found = toggle_params(model_json, fps)
        states = {}
        for c in driven:
            p = pid[c]
            if p in found and p in lo and hi[p] > lo[p]:
                _, a, b = found[p]
                states[c] = tuple((v - lo[p]) / (hi[p] - lo[p]) * 2 - 1
                                  for v in (a, b))
        if states:
            fg, ng = inbetween(gen["segs"], list(states), states)
            fr, nr = inbetween(ref["segs"], list(states), states)
            res["toggle"] = {"n_params": len(states), "gen": fg, "ref": fr,
                             "params": sorted(pid[c] for c in states)}

    o = res["overall"] or {}
    checks = {
        "std": bool(o) and STD_BAND[0] <= o["std"] <= STD_BAND[1],
        "jerk": bool(o) and o["jerk"] <= JERK_MAX,
        "toggle": (res["toggle"] is None
                   or not res["toggle"]["gen"] <= res["toggle"]["gen"]  # nan
                   or res["toggle"]["gen"]
                   <= res["toggle"]["ref"] + TOGGLE_SLACK),
        "period": all(s["period_gen"]
                      <= max(PERIOD_MAX, s["period_ref"] + 0.1)
                      for s in res["groups"].values()),
    }
    res["checks"] = checks
    res["pass"] = all(checks.values())
    res["thresholds"] = {"std": STD_BAND, "jerk_max": JERK_MAX,
                         "toggle_slack": TOGGLE_SLACK,
                         "period_max": PERIOD_MAX, "still_eps": STILL_EPS,
                         "period_rule": "gen <= max(period_max, ref + 0.1)"}
    return res


def _fmt(v, w=6, p=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-".rjust(w)
    if isinstance(v, float) and np.isinf(v):
        return "inf".rjust(w)
    return f"{v:{w}.{p}f}"


def print_table(res: dict) -> None:
    print(f"{res['traj']}  [{res['kind']}]  vs  {res['clips']}")
    print(f"{'group':<8}{'n':>4} {'std':>6} {'|d|':>6} {'p95':>6} {'jerk':>6}"
          f"  {'still g/a':>11}  {'period g/a':>11}")
    rows = list(res["groups"].items()) + [("OVERALL", res["overall"])]
    for g, s in rows:
        if not s:
            continue
        print(f"{g:<8}{s['n']:>4} {_fmt(s['std'])} {_fmt(s['dmean'])} "
              f"{_fmt(s['dp95'])} {_fmt(s['jerk'])}  "
              f"{_fmt(s['still_gen'], 5)}/{_fmt(s['still_ref'], 5)}  "
              f"{_fmt(s['period_gen'], 5)}/{_fmt(s['period_ref'], 5)}")
    t = res["toggle"]
    if t:
        print(f"toggle in-between ({t['n_params']} switch params): "
              f"gen {t['gen']:.3f}  authored {t['ref']:.3f}")
    c = res["checks"]
    print("checks: " + "  ".join(f"{k}={'ok' if v else 'FAIL'}"
                                 for k, v in c.items())
          + f"   => {'PASS' if res['pass'] else 'FAIL'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traj", required=True,
                    help="runtime --dump file, or a clips file")
    ap.add_argument("--clips", required=True, help="clips/<name>.npz")
    ap.add_argument("--rig", required=True, help="rigs/<name>.npz")
    ap.add_argument("--model", default=None,
                    help=".model3.json; enables the toggle check")
    ap.add_argument("--json", default=None, help="write the full result here")
    a = ap.parse_args(argv)

    res = score(a.traj, a.clips, a.rig, a.model)
    print_table(res)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=float)
        print(f"-> {a.json}")
    return 0 if res["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
