#!/usr/bin/env python3
"""
Parse authored Live2D animation into trajectories on the Block 1 interface.

sanitize.py stripped `Motions` from FileReferences so probing would be
deterministic, but the .motion3.json files were never touched. They are
keyframed parameter curves authored by professional riggers -- exactly the
supervision the motion prior needs, requiring no video, no inverse renderer and
no text.

Bands
-----
Only the expressive band is emitted as a target. Four groups are excluded, each
for its own reason:

  physics   Hair, ribbons and accessories are computed by Live2D's solver from
            head and body motion at runtime. Generating them means fighting the
            solver and getting doubled motion. physics3.json lists these
            parameters explicitly, so this is exact rather than heuristic.
  autonomic Blink and breath are stochastic and carry no semantic content. A
            model would burn capacity learning noise that is cheaper and better
            generated procedurally.
  mouth     Lip sync must be frame-accurate to audio. A generative model drifts;
            phoneme timing does not.
  discrete  Categorical parameters are part swaps and opacity toggles.
            Interpolating through them yields half-visible limbs.

Values are normalised to [-1, 1] across each parameter's own range, so a
trajectory means the same thing on every rig.

Usage:
    python -m l2dprobe.motions --models 'CubismWebSamples/**/*.model3.json' \
        --models 'live2d-models/**/*.model3.json' --rigs rigs/ --out clips/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

AUTONOMIC = ("breath", "blink")
MOUTH = ("mouth", "lip")


def bezier_x(p, s):
    a, b, c, d = p
    m = 1 - s
    return (m ** 3) * a + 3 * (m ** 2) * s * b + 3 * m * (s ** 2) * c + (s ** 3) * d


def eval_segments(segs: list[float], times: np.ndarray) -> np.ndarray:
    """Evaluate a motion3 curve at `times`.

    Segment encoding: the list opens with (t0, v0), then repeats a type code
    followed by its points. 0 linear (t,v), 1 bezier (t,v)x3, 2 stepped (t,v),
    3 inverse-stepped (t,v).
    """
    out = np.empty(len(times), np.float32)
    t0, v0 = float(segs[0]), float(segs[1])
    i = 2
    knots = [(t0, v0)]
    pieces = []
    while i < len(segs):
        kind = int(segs[i])
        if kind == 1:
            t1, v1, t2, v2, t3, v3 = map(float, segs[i + 1:i + 7])
            pieces.append(("bezier", knots[-1], (t1, v1), (t2, v2), (t3, v3)))
            knots.append((t3, v3))
            i += 7
        else:
            t1, v1 = float(segs[i + 1]), float(segs[i + 2])
            name = {0: "linear", 2: "stepped", 3: "istepped"}[kind]
            pieces.append((name, knots[-1], (t1, v1)))
            knots.append((t1, v1))
            i += 3

    for k, t in enumerate(times):
        if t <= knots[0][0] or not pieces:
            out[k] = knots[0][1]
            continue
        if t >= knots[-1][0]:
            out[k] = knots[-1][1]
            continue
        for piece in pieces:
            a = piece[1]
            b = piece[-1]
            if not (a[0] <= t <= b[0]):
                continue
            span = b[0] - a[0]
            u = 0.0 if span <= 0 else (t - a[0]) / span
            if piece[0] == "linear":
                out[k] = a[1] + (b[1] - a[1]) * u
            elif piece[0] == "stepped":
                out[k] = a[1]
            elif piece[0] == "istepped":
                out[k] = b[1]
            else:
                _, p0, p1, p2, p3 = piece
                xs = (p0[0], p1[0], p2[0], p3[0])
                lo, hi = 0.0, 1.0
                for _ in range(24):     # bisect for s where x(s) == t
                    mid = 0.5 * (lo + hi)
                    if bezier_x(xs, mid) < t:
                        lo = mid
                    else:
                        hi = mid
                s = 0.5 * (lo + hi)
                out[k] = bezier_x((p0[1], p1[1], p2[1], p3[1]), s)
            break
        else:
            out[k] = knots[-1][1]
    return out


def physics_params(model_json: str) -> tuple[set[str], bool]:
    """Parameters driven by the physics solver, and whether a file was found.

    The distinction matters. A rig with no physics3.json has no solver, so its
    hair must be keyframed and belongs in the expressive band. A rig that has
    one keeps hair out of the band because Live2D computes it at runtime.
    Reporting zero for both cases hides a real inconsistency in supervision:
    the model would see hair sway as a target on some rigs and not others.
    """
    try:
        with open(model_json, encoding="utf-8") as fh:
            refs = json.load(fh).get("FileReferences", {})
        rel = refs.get("Physics")
        if not rel:
            return set(), False
        path = os.path.join(os.path.dirname(model_json), rel)
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception:  # noqa: BLE001
        return set(), False
    out = set()
    for setting in blob.get("PhysicsSettings", []):
        for o in setting.get("Output", []):
            pid = o.get("Destination", {}).get("Id")
            if pid:
                out.add(pid)
    return out, True


def motion_files(model_json: str) -> list[str]:
    """Motion files referenced by the model, falling back to a directory scan."""
    root = os.path.dirname(model_json)
    found = []
    try:
        with open(model_json, encoding="utf-8") as fh:
            groups = json.load(fh).get("FileReferences", {}).get("Motions", {})
        for entries in groups.values():
            for e in entries:
                f = e.get("File")
                if f:
                    found.append(os.path.join(root, f))
    except Exception:  # noqa: BLE001
        pass
    if not found:
        for dirpath, _d, files in os.walk(root):
            found += [os.path.join(dirpath, f) for f in files
                      if f.endswith(".motion3.json")]
    return sorted({p for p in found if os.path.exists(p)})


def band_of(pid: str, kind: str, group: str, phys: set[str]) -> str:
    if pid in phys:
        return "physics"
    if kind == "categorical":
        return "discrete"
    low = pid.lower()
    if any(k in low for k in AUTONOMIC) or group == "breath":
        return "autonomic"
    if any(k in low for k in MOUTH) or group == "mouth":
        return "mouth"
    if group in ("eyelid",) and "open" in low:
        return "autonomic"     # driven by the blink generator at runtime
    return "expressive"


def build(model_json: str, rig_npz: str, fps: float) -> dict | None:
    z = np.load(rig_npz, allow_pickle=True)
    pids = [str(x) for x in z["pid"]]
    kinds = [str(x) for x in z["kind"]]
    groups = [str(x) for x in z["group"]]
    lo = z["minimum"].astype(np.float32)
    hi = z["maximum"].astype(np.float32)
    dflt = z["default"].astype(np.float32)
    phys, has_physics = physics_params(model_json)

    bands = [band_of(p, k, g, phys) for p, k, g in zip(pids, kinds, groups)]
    expressive = [i for i, b in enumerate(bands) if b == "expressive"]
    if not expressive:
        return None
    index = {p: i for i, p in enumerate(pids)}

    clips, covered = [], Counter()
    for path in motion_files(model_json):
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        dur = float(blob.get("Meta", {}).get("Duration", 0) or 0)
        if dur <= 0.05:
            continue
        times = np.arange(0.0, dur, 1.0 / fps, dtype=np.float32)
        if len(times) < 4:
            continue

        # Undriven parameters must rest at their DEFAULT, not at zero. After
        # normalisation zero is the midpoint of the range, which is a different
        # pose: ParamEyeLOpen on [0, 1.2] with default 1.0 normalises to +0.67,
        # so zero-filling would silently half-close the eye in every clip that does
        # not animate it. Most clips drive only a subset, so this affected a
        # large fraction of every target tensor.
        span_e = np.maximum(hi[expressive] - lo[expressive], 1e-6)
        default_n = np.clip(
            (dflt[expressive] - lo[expressive]) / span_e * 2 - 1, -1, 1)
        frame = np.tile(default_n.astype(np.float32), (len(times), 1))
        driven = np.zeros(len(expressive), bool)
        hit = 0
        for curve in blob.get("Curves", []):
            if curve.get("Target") != "Parameter":
                continue
            i = index.get(curve.get("Id"))
            if i is None or bands[i] != "expressive":
                continue
            col = expressive.index(i)
            vals = eval_segments(curve.get("Segments", []), times)
            span = max(float(hi[i] - lo[i]), 1e-6)
            frame[:, col] = np.clip((vals - lo[i]) / span * 2 - 1, -1, 1)
            driven[col] = True
            hit += 1
            covered[curve["Id"]] += 1
        if hit:
            # The mask lets training ignore parameters this clip never touched,
            # rather than teaching the model that they are deliberately held.
            clips.append((frame, driven.copy()))

    if not clips:
        return None
    return {
        "clips": [c for c, _ in clips],
        "masks": [m for _, m in clips],
        "default_n": default_n.astype(np.float32),
        "pid": np.array([pids[i] for i in expressive], dtype=object),
        "param_row": np.asarray(expressive, np.int32),
        "bands": Counter(bands),
        "fps": fps,
        "has_physics": has_physics,
        "n_curves_used": len(covered),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="append", required=True)
    ap.add_argument("--rigs", default="rigs")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args(argv)

    paths = []
    for pattern in a.models:
        paths += [p for p in sorted(glob.glob(pattern, recursive=True))
                  if p.endswith(".model3.json") and ".probe." not in p]
    if not paths:
        print("no models matched", file=sys.stderr)
        return 1

    os.makedirs(a.out, exist_ok=True)
    total_s = total_clips = 0
    band_totals = Counter()
    # Rigs are keyed by the model filename, so two directories holding the same
    # model under different names -- ANIYA/ and ANIYA-walk/, which are separate
    # motion sets for one character -- collide and the second silently
    # overwrites the first. Merge their clips instead of losing one.
    seen: dict[str, str] = {}
    for path in paths:
        name = os.path.basename(path).split(".")[0]
        if name in seen:
            print(f"{name:<24} NAME COLLISION with {seen[name]}; "
                  "merging clips into the existing file")
        seen.setdefault(name, path)
        rig_npz = os.path.join(a.rigs, f"{name}.npz")
        if not os.path.exists(rig_npz):
            continue
        try:
            pack = build(path, rig_npz, a.fps)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<24} FAILED {type(exc).__name__}: {exc}")
            continue
        if pack is None:
            print(f"{name:<24} no usable motion")
            continue

        secs = sum(len(c) for c in pack["clips"]) / a.fps
        cov = float(np.mean([m.mean() for m in pack["masks"]]))
        band_totals += pack["bands"]

        dest = os.path.join(a.out, f"{name}.npz")
        if os.path.exists(dest) and seen.get(name) != path:
            # Same character under two directories (ANIYA and ANIYA-walk are
            # separate motion sets for one rig). Without this the second write
            # silently discards the first set's clips.
            try:
                old = np.load(dest, allow_pickle=True)
                if list(old["pid"]) == list(pack["pid"]):
                    n_old = int(old["n_clips"])
                    pack["clips"] = [old[f"clip{i}"] for i in range(n_old)] \
                        + list(pack["clips"])
                    pack["masks"] = [old[f"mask{i}"] for i in range(n_old)] \
                        + list(pack["masks"])
                    secs = sum(len(c) for c in pack["clips"]) / a.fps
                else:
                    print(f"  {name}: parameter sets differ, not merging")
            except Exception as exc:  # noqa: BLE001
                print(f"  {name}: could not merge ({exc})")

        np.savez_compressed(
            dest,
            **{f"clip{i}": c for i, c in enumerate(pack["clips"])},
            **{f"mask{i}": m for i, m in enumerate(pack["masks"])},
            pid=pack["pid"], param_row=pack["param_row"],
            default_n=pack["default_n"],
            fps=np.float32(a.fps), n_clips=np.int32(len(pack["clips"])))
        b = pack["bands"]
        print(f"{name:<24} {len(pack['clips']):>3} clips  {secs:>7.1f}s  "
              f"{len(pack['pid']):>3} expressive  "
              f"(phys {b['physics']}, auto {b['autonomic']}, "
              f"mouth {b['mouth']}, disc {b['discrete']})  "
              f"{cov:.0%} driven/clip"
              f"{'' if pack['has_physics'] else '  [NO physics3.json]'}")
        total_s += secs
        total_clips += len(pack["clips"])

    print(f"\n{total_clips} clips, {total_s / 60:.1f} minutes at {a.fps:g} fps")
    print(f"bands across all rigs: {dict(band_totals)}")
    print("\nMinutes of animation is the number that decides the model size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
