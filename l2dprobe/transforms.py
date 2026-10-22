#!/usr/bin/env python3
"""
Find a rig's whole-model transform parameters from what they do on screen.

Many rigs carry parameters that move, scale or rotate the entire character at
once -- ParamAllX/Y/Rotate, ParamModelX, PARAM_moveY, a body_Scale, or pinyin
ids such as zongtiZ (whole-body Z). They exist for staging, not for acting.
A motion model that drives them freely moves the character off the canvas,
shrinks it to a dot or tips it over, which is what the flow prior did on
lafei, xuefeng and Natori. They have to be held still.

Names cannot find them: half the ids are pinyin or numbered. The probe's
support maps can. A body or head angle bends the character around fixed feet;
a whole-model transform moves the feet too, and touches most of the
character. The score is

    reach x cover

  reach  how strongly the effect reaches BOTH the top and the bottom rows of
         the character box (min of the two), relative to its peak
  cover  fraction of the character box the effect touches at all

Measured on the Cubism samples: every whole-model parameter scores >= 0.30
(Mao/Natori ParamAllX/Y/Rotate, Rice ParamAllZ), every body or head angle
<= 0.20 (Mark ParamBodyAngleY is the closest). A global colour tint
(Mao ParamAllColor1/2) also scores high; holding it is harmless.

Usage:
    python -m l2dprobe.transforms out/ [--rig NAME] [--threshold 0.25]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

THRESHOLD = 0.25
EDGE_ROWS = 3         # of the 32-row support map: top and bottom ~10%
COVER_EPS = 0.05


def global_score(support: np.ndarray) -> float:
    """reach x cover for one (H, W) support map; 0 for a dead parameter."""
    peak = float(support.max())
    if peak <= 0:
        return 0.0
    s = support / peak
    reach = min(float(s[:EDGE_ROWS].max()), float(s[-EDGE_ROWS:].max()))
    cover = float((s > COVER_EPS).mean())
    return reach * cover


def scores_for(probe_dir: str, rig: str) -> dict[str, float] | None:
    """{pid: score} from out/rig_<rig>.npz, or None if the probe is missing."""
    path = os.path.join(probe_dir, f"rig_{rig}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    if "support" not in z or "pids" not in z:
        return None
    return {str(p): global_score(s) for p, s in zip(z["pids"], z["support"])}


def whole_model_params(probe_dir: str, rig: str,
                       threshold: float = THRESHOLD) -> dict[str, float] | None:
    """Parameters scoring at or above the threshold, with their scores."""
    sc = scores_for(probe_dir, rig)
    if sc is None:
        return None
    return {p: v for p, v in sc.items() if v >= threshold}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe_dir")
    ap.add_argument("--rig", default=None, help="one rig; default every rig")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--top", type=int, default=6,
                    help="also list this many highest-scoring parameters "
                         "below the threshold, to check the margin")
    a = ap.parse_args(argv)

    rigs = ([a.rig] if a.rig else
            sorted(os.path.basename(p)[4:-4]
                   for p in glob.glob(os.path.join(a.probe_dir, "rig_*.npz"))))
    if not rigs:
        print(f"no rig_*.npz in {a.probe_dir}", file=sys.stderr)
        return 1
    out = {}
    for r in rigs:
        sc = scores_for(a.probe_dir, r)
        if sc is None:
            print(f"{r}: no probe support maps")
            continue
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        held = [(p, v) for p, v in ranked if v >= a.threshold]
        near = [(p, v) for p, v in ranked if v < a.threshold][:a.top]
        out[r] = [p for p, _ in held]
        print(f"{r}: hold {', '.join(f'{p} {v:.2f}' for p, v in held) or '-'}"
              f"   | next: {', '.join(f'{p} {v:.2f}' for p, v in near)}")
    with open(os.path.join(a.probe_dir, "whole_model.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
