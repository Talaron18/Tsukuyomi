#!/usr/bin/env python3
"""
What can each rig actually express, and does the training data exercise it?

Motivated by a real observation: previews look like swaying and blinking. Two
separate causes are possible and they need different responses.

  the rig cannot        A parameter is a deformation the rigger authored over
                        the mesh. If a character's hand is one flat texture with
                        no deformer, no parameter exists and none can be added
                        in code -- that is Cubism Editor work, per rig, by hand.
                        Surveyed across 80 rigs: arms 92%, hands 30%,
                        shoulders 21%, fingers 4%. A third of rigs with arms
                        have only one swing axis.

  the data never did    Parameters can exist and sit motionless through every
                        authored clip. Then the model never learns to move them
                        no matter how good the rig is, and the fix is data, not
                        rigging.

This reports both per region, so the two are distinguishable. `coverage` is what
the rig affords; `motion` is what the clips actually do with it.

Usage:
    python -m l2dprobe.capability --rigs rigs/ --clips clips/
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np

REGIONS = {
    "head":     ("angle", "head", "顔", "首"),
    "eyes":     ("eye", "iris", "目", "瞳"),
    "brow":     ("brow", "眉"),
    "mouth":    ("mouth", "lip", "口"),
    "body":     ("body", "waist", "chest", "breast", "体", "腰"),
    "shoulder": ("shoulder", "肩"),
    "arm":      ("arm", "elbow", "腕", "肘"),
    "hand":     ("hand", "wrist", "palm", "手"),
    "finger":   ("finger", "thumb", "指"),
    "hair":     ("hair", "髪"),
    "leg":      ("leg", "foot", "knee", "脚", "足"),
}


def region_of(pid: str) -> str:
    low = pid.lower()
    # Ordered longest-specific first: "hand" must not swallow parameters that
    # are really fingers, and "arm" must not swallow shoulders.
    for name in ("finger", "hand", "shoulder", "arm", "brow", "eyes",
                 "mouth", "hair", "leg", "body", "head"):
        if any(k in low for k in REGIONS[name]):
            return name
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigs", default="rigs")
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--min-motion", type=float, default=0.01,
                    help="temporal std below this counts as motionless")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(a.rigs, "*.npz")))
    paths = [p for p in paths if not os.path.basename(p).startswith("_")]
    if not paths:
        print(f"no rigs in {a.rigs}", file=sys.stderr)
        return 1

    have = defaultdict(int)          # rigs possessing any parameter in region
    animated = defaultdict(int)      # rigs whose clips move that region
    per_rig = []

    for path in paths:
        name = os.path.basename(path).replace(".npz", "")
        z = np.load(path, allow_pickle=True)
        pids = [str(x) for x in z["pid"]]
        regions = [region_of(p) for p in pids]
        counts = defaultdict(int)
        for r in regions:
            counts[r] += 1
        for r in counts:
            have[r] += 1

        moved = {}
        cpath = os.path.join(a.clips, f"{name}.npz")
        if os.path.exists(cpath):
            cz = np.load(cpath, allow_pickle=True)
            cpids = [str(x) for x in cz["pid"]]
            clips = [cz[f"clip{i}"] for i in range(int(cz["n_clips"]))]
            if clips:
                allc = np.concatenate(clips)
                std = allc.std(axis=0)
                by_region = defaultdict(list)
                for c, p in enumerate(cpids):
                    by_region[region_of(p)].append(std[c])
                for r, vals in by_region.items():
                    m = float(np.mean(vals))
                    moved[r] = m
                    if m > a.min_motion:
                        animated[r] += 1
        per_rig.append((name, counts, moved))

    n = len(paths)
    print(f"{n} rigs\n")
    print(f"{'region':<10} {'rigs have':>10} {'%':>5} {'rigs animate':>13} "
          f"{'%':>5}   verdict")
    order = ["head", "eyes", "brow", "mouth", "body", "hair",
             "shoulder", "arm", "hand", "finger", "leg"]
    # Regions the pipeline deliberately excludes from the expressive band
    # never appear in clips/, so "animate" would read 0% and look like a data
    # failure. Mouth belongs to phoneme timing, hair to the physics solver.
    EXCLUDED = {"mouth": "mouth band -> TTS phoneme timing",
                "hair": "largely physics-driven -> solver"}
    for r in order:
        h, an = have[r], animated[r]
        if r in EXCLUDED:
            verdict = f"excluded by design ({EXCLUDED[r]})"
        elif h == 0:
            verdict = "absent everywhere"
        elif an == 0:
            verdict = "present but never animated -> DATA problem"
        elif an / max(h, 1) < 0.4:
            verdict = "rarely animated -> data problem"
        elif h / n < 0.4:
            verdict = "rig problem: most rigs lack it"
        else:
            verdict = "ok"
        print(f"{r:<10} {h:>10} {h/n:>4.0%} {an:>13} {an/max(h,1):>4.0%}   "
              f"{verdict}")

    print("\nper-rig arm/hand articulation (params, and mean motion in clips):")
    rows = []
    for name, counts, moved in per_rig:
        limbs = counts["arm"] + counts["hand"] + counts["finger"] + counts["shoulder"]
        mot = np.mean([moved.get(r, 0.0) for r in
                       ("arm", "hand", "finger", "shoulder")])
        rows.append((limbs, mot, name, counts))
    rows.sort(key=lambda r: -r[0])
    for limbs, mot, name, counts in rows[:20]:
        print(f"  {name[:24]:<24} arm {counts['arm']:>3} hand {counts['hand']:>3} "
              f"finger {counts['finger']:>2} shoulder {counts['shoulder']:>2}"
              f"   clip motion {mot:.4f}")
    dead = [n for l, m, n, _ in rows if l >= 2 and m < a.min_motion]
    if dead:
        print(f"\n{len(dead)} rigs have limb parameters that never move in their "
              f"clips:\n  {', '.join(dead[:12])}")
        print("  Those are wasted capacity: the parameters exist and the model "
              "has no example of using them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
