#!/usr/bin/env python3
"""
Rig-calibrated lip sync: which mouth parameters a rig talks with, and over
what range, read from its own authored motions.

The lip sync used to drive the mouth-open parameter from its default to its
maximum on every rig (and ParamA/I/U/E/O only under --lipsync vowels). A rig
whose artist shapes speech with the vowel parameters, or opens the mouth
only part of the way while talking, or moves the mouth form with it, then
talks unlike its own voice lines. Many rigs ship voice-line motions with
keyed lip flaps; those curves say how this rig talks.

  talking    A motion talks where an articulator (the open parameter or a
             vowel, flow.runtime.mouth_targets) flaps: its normalised value
             (rest 0, the far end of its range 1) rises and falls by
             FLAP_DEPTH at least MIN_FLAPS times with peaks at most FLAP_GAP
             apart. Held shapes -- the sample rigs' smiles and open-mouthed
             expressions (Hiyori m02/m05 hold OpenY at 1 for seconds, Mao
             special_01 holds ParamA at 1, Wanko's idles pant at ~1 s) --
             are not talking. The frames from half a gap before the first
             peak to half a gap after the last are the talking frames.
  range      Over the talking frames where it is active (past ACTIVE of
             its range), each articulator's speaking extreme is the 95th
             percentile of its value (toward the end of its range it opens
             to); it is used when that is SPEAK_USED of the way there. Another mouth parameter (form, smile) follows the
             articulation when a least-squares line on it explains it
             (|r| >= FOLLOW_R) with a slope of FOLLOW_SLOPE of its range or
             more: it is then driven rest -> rest + slope with the openness.
  drive      open used, vowels not: the open parameter, rest -> p95.
             vowels used, open not: each used vowel, rest -> its p95, with
             openness times the vowel shape (they sum to the openness), so
             the mouth opens no further than one vowel at full; a single
             used vowel gets the openness alone (no shapes to choose).
             both: both, as the artist co-keys them.
             nothing talks: None -- the caller keeps the open-only lip sync
             of before (rest -> maximum on the open parameter).
  level      The median normalised articulation over the talking frames;
             level_gamma shapes the speech openness so its voiced median
             lands there (an artist who keys flaps to the full range wants
             a fuller mouth than a level curve that dips between syllables).

Run as a script to print a rig's calibration:
    python -m flow.mouth --model-json M.model3.json --rig rigs/R.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    from ..l2dprobe.motions import band_of, is_vowel, physics_params
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from l2dprobe.motions import band_of, is_vowel, physics_params

# A flap is a rise and fall of at least FLAP_DEPTH of the articulator's
# range; MIN_FLAPS of them with peaks no more than FLAP_GAP apart are
# talking. Lip flaps keyed by hand run ~3-5 per second (spoken Japanese is
# ~7-8 morae/s, animators key every other one or so); 0.6 s admits 1.7 Hz
# and still excludes Wanko's ~1 s panting.
FLAP_DEPTH = 0.3
MIN_FLAPS = 3
FLAP_GAP = 0.6
SPEAK_USED = 0.2
# An articulator's speaking range is read over the talking frames where it
# is past ACTIVE of its range, if that is ACTIVE_SHARE of them or more.
ACTIVE = 0.05
ACTIVE_SHARE = 0.05
SPEAK_PCT = 95
FOLLOW_R = 0.5
FOLLOW_SLOPE = 0.15
# Bounds on the level exponent: the shaping may open a voiced median of
# 0.75 (the server's edge runs: 0.68-0.82) to at most 0.87 or close it to
# at least 0.56 -- a calibration should tune, not overturn, the level map.
GAMMA_LO, GAMMA_HI = 0.5, 2.0


def _flaps(a, fps):
    """Peak frames of rises and falls of FLAP_DEPTH (hysteresis scan)."""
    peaks = []
    lo, hi, hi_i = a[0], a[0], 0
    rising = False
    for i, v in enumerate(a):
        if not rising:
            lo = min(lo, v)
            if v >= lo + FLAP_DEPTH:
                rising, hi, hi_i = True, v, i
        else:
            if v > hi:
                hi, hi_i = v, i
            if v <= hi - FLAP_DEPTH:
                peaks.append(hi_i)
                rising, lo = False, v
    return peaks


def talking_frames(a, fps):
    """Boolean mask of the frames inside talking stretches of the
    articulation `a` (normalised, 0 rest .. 1 open)."""
    peaks = _flaps(np.asarray(a, float), fps)
    mask = np.zeros(len(a), bool)
    gap = FLAP_GAP * fps
    k = 0
    while k < len(peaks):
        j = k
        while j + 1 < len(peaks) and peaks[j + 1] - peaks[j] <= gap:
            j += 1
        if j - k + 1 >= MIN_FLAPS:
            s = max(int(peaks[k] - gap / 2), 0)
            e = min(int(peaks[j] + gap / 2) + 1, len(a))
            mask[s:e] = True
        k = j + 1
    return mask


def _norm(v, rest, ext):
    d = ext - rest
    return np.clip((v - rest) / d, 0, 1) if abs(d) > 1e-9 else v * 0


def calibrate(model_json, rig, fps=30.0, motions=None):
    """The rig's talking calibration (see the module docstring): {"rig",
    "motions" (talking motion names), "frames", "params" {pid: stats},
    "drive" ("open" / "vowels" / "both" / None), "map" {pid: {"curve",
    "lo", "hi"}}, "level", "why"}. `motions`: [(name, authored_motion)]
    (runtime.authored_motions), read from the model when None."""
    try:
        from .runtime import authored_motions, mouth_targets
    except ImportError:
        from flow.runtime import authored_motions, mouth_targets
    z = np.load(rig, allow_pickle=True)
    pids = [str(x) for x in z["pid"]]
    lo = z["minimum"].astype(float)
    hi = z["maximum"].astype(float)
    dflt = z["default"].astype(float)
    phys, _ = physics_params(model_json)
    bands = [band_of(p, k, g, phys) for p, k, g in
             zip(pids, (str(x) for x in z["kind"]),
                 (str(x) for x in z["group"]))]
    op_i, vow = mouth_targets(pids, bands)
    mouth = [i for i, b in enumerate(bands) if b == "mouth"]
    name = os.path.splitext(os.path.basename(rig))[0]
    out = {"rig": name, "motions": [], "frames": 0, "params": {},
           "drive": None, "map": {}, "level": None, "why": ""}
    if motions is None:
        motions = authored_motions(model_json, fps)
    arts = ([op_i] if op_i is not None else []) + sorted(vow.values())

    def ext(i):            # the end of the range the parameter opens to
        if i in arts and is_vowel(pids[i]):
            return hi[i]
        return hi[i] if hi[i] - dflt[i] >= dflt[i] - lo[i] else lo[i]

    if not arts:
        out["why"] = "no mouth-open or vowel parameter"
        return out
    vals = {i: [] for i in mouth}
    art_all = []
    for mname, m in motions:
        T = m["frames"]
        cur = {i: (np.asarray(m["params"][pids[i]], float)
                   if pids[i] in m["params"] else np.full(T, dflt[i]))
               for i in mouth}
        if not any(pids[i] in m["params"] for i in arts):
            continue
        norm = np.stack([_norm(cur[i], dflt[i], ext(i)) for i in arts])
        a = norm.max(0)
        mask = talking_frames(a, fps)
        if not mask.any():
            continue
        out["motions"].append(mname)
        for i in mouth:
            vals[i].append(cur[i][mask])
        art_all.append(a[mask])
    if not art_all:
        out["why"] = (f"no talking curves in {len(motions)} motions (no lip "
                      "flaps on the open/vowel parameters)")
        return out
    art = np.concatenate(art_all)
    out["frames"] = int(len(art))
    used = []
    for i in mouth:
        v = np.concatenate(vals[i])
        st = {"rest": round(float(dflt[i]), 4),
              "range": [round(float(lo[i]), 4), round(float(hi[i]), 4)]}
        if i in arts:
            e = ext(i)
            # While it is active: a vowel shapes one flap in three or five,
            # and its p95 over all talking frames would be its rest.
            act = v[_norm(v, dflt[i], e) > ACTIVE]
            frac, p = 0.0, dflt[i]
            if len(act) >= max(3, ACTIVE_SHARE * len(v)):
                p = np.percentile(act, SPEAK_PCT if e >= dflt[i] else
                                  100 - SPEAK_PCT)
                frac = (p - dflt[i]) / (e - dflt[i]) if e != dflt[i] else 0.0
            st.update(role="open" if i == op_i else "vowel",
                      p95=round(float(p), 4), share=round(float(frac), 3),
                      active=round(len(act) / max(len(v), 1), 3),
                      used=bool(frac >= SPEAK_USED))
            if st["used"]:
                used.append(i)
        else:
            span = hi[i] - lo[i]
            st["role"] = "other"
            if np.std(v) > 1e-6 and np.std(art) > 1e-6 and span > 0:
                r = float(np.corrcoef(art, v)[0, 1])
                slope = float(np.polyfit(art, v, 1)[0])
                st.update(r=round(r, 3), slope=round(slope, 4))
                if abs(r) >= FOLLOW_R and abs(slope) >= FOLLOW_SLOPE * span:
                    st["follow"] = True
            else:
                st.update(r=0.0, slope=0.0)
        out["params"][pids[i]] = st
    v_used = [i for i in used if i != op_i]
    o_used = op_i in used
    if not used:
        out["why"] = (f"talking motions found but no articulator reaches "
                      f"{SPEAK_USED:.0%} of its range while talking")
        return out
    out["drive"] = ("both" if o_used and v_used else
                    "open" if o_used else "vowels")
    mp = {}
    if o_used:
        mp[pids[op_i]] = {"curve": "open", "lo": float(dflt[op_i]),
                          "hi": out["params"][pids[op_i]]["p95"]}
    letter = {i: k for k, i in vow.items()}
    for i in v_used:
        mp[pids[i]] = {"curve": letter[i] if len(v_used) > 1 else "open",
                       "lo": float(dflt[i]),
                       "hi": out["params"][pids[i]]["p95"]}
    for p, st in out["params"].items():
        if st.get("follow"):
            i = pids.index(p)
            mp[p] = {"curve": "open", "lo": float(dflt[i]),
                     "hi": round(float(np.clip(dflt[i] + st["slope"],
                                               lo[i], hi[i])), 4)}
    out["map"] = mp
    # In the drive's units: openness 1 is the p95, not the range's end.
    ref = max(out["params"][pids[i]]["share"] for i in used)
    talk = np.minimum(art[art > 0.05] / ref, 1.0)
    out["level"] = round(float(np.median(talk)), 4) if len(talk) else None
    return out


def level_gamma(open_, level):
    """Exponent g so the voiced frames' median openness ** g is `level`
    (bounded GAMMA_LO..GAMMA_HI); 1 without a level or voiced frames."""
    v = np.asarray(open_, float)
    v = v[v > 0]
    if level is None or not len(v):
        return 1.0
    m = float(np.median(v))
    if not 0 < m < 1:
        return 1.0
    level = min(max(float(level), 1e-3), 0.999)
    return float(np.clip(np.log(level) / np.log(m), GAMMA_LO, GAMMA_HI))


def describe(cal):
    """One log line: what the rig talks with and what will be driven."""
    name = cal["rig"]
    if not cal["map"]:
        return (f"mouth: {name}: {cal['why']} -> open-only on the "
                "mouth-open parameter, rest to maximum (as before)")
    ps = cal["params"]
    arts = [(p, s) for p, s in ps.items() if s.get("role") in ("open",
                                                                "vowel")]
    used = ", ".join(f"{p} (p95 {s['p95']:g}, {s['share']:.0%})"
                     for p, s in arts if s["used"])
    unused = [p for p, s in arts if not s["used"]]
    follow = [f"{p} {cal['map'][p]['lo']:g}->{cal['map'][p]['hi']:g}"
              for p, s in ps.items() if s.get("follow")]
    what = {"open": "driving the open parameter",
            "vowels": "driving vowels" if sum(
                m["curve"] != "open" for m in cal["map"].values()) > 1
            else "driving the vowel with the openness",
            "both": "driving open and vowels as co-keyed"}[cal["drive"]]
    return (f"mouth: {name} talks with {used} in {len(cal['motions'])} "
            f"motions ({cal['frames']} talking frames)"
            + (f", {', '.join(unused)} unused" if unused else "")
            + (f"; follows: {', '.join(follow)}" if follow else "")
            + f"; level {cal['level']} -> {what}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-json", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--json", action="store_true",
                    help="print the whole calibration as JSON too")
    a = ap.parse_args(argv)
    cal = calibrate(a.model_json, a.rig, a.fps)
    print(describe(cal))
    for p, s in cal["params"].items():
        print(f"  {p}: " + ", ".join(f"{k} {v}" for k, v in s.items()))
    if a.json:
        print(json.dumps(cal, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
