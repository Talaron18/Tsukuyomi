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

Two corrections from the full 59-rig set:

  protected  On rigs drawn without feet (busts, chibis) a body angle reaches
             both edges of the box and scores high: Lumi BodyAngleZ 0.67,
             lafei 0.45, z23 PARAM_BODY_ANGLE_Z 0.51. Angles and gaze are never
             held, whatever they score.
  hinted     A whole-model transform that moves the character only a little
             scores low: taiyuan_2 ParamZTY 0.21, banrenma_2 zongtiZ 0.11,
             lafei_4 shentidaxiao ("body size") 0.06. Ids that name the whole
             model -- All, Model, Main, Position, Scale, ZT/zongti (whole),
             daxiao (size) -- need only HINTED_THRESHOLD.

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
HINTED_THRESHOLD = 0.05
# Matched at the start of the id once a Param/PARAM_ prefix is removed.
HINT_PREFIXES = ("all", "model", "main", "move", "pos", "zt", "zong",
                 "camera", "base")
# Matched anywhere in the id.
HINT_SUBSTRINGS = ("position", "scale", "size", "zongti", "daxiao")
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


def _stem(pid: str) -> str:
    low = pid.lower()
    for prefix in ("param_", "param"):
        if low.startswith(prefix):
            return low[len(prefix):]
    return low


def name_hints_whole_model(pid: str) -> bool:
    stem, low = _stem(pid), pid.lower()
    return (stem.startswith(HINT_PREFIXES)
            or any(h in low for h in HINT_SUBSTRINGS))


def protected(pid: str, rig_group: str = "") -> bool:
    """Angles and gaze are never held as whole-model transforms.

    analyze.GROUPS files AllRotate under body, and ParamAllRotate is exactly
    the kind of parameter to hold, so the body group is not protected.
    """
    try:
        from .analyze import group_of
    except ImportError:
        from analyze import group_of
    low = pid.lower()
    return ("angle" in low or "eyeball" in low
            or rig_group in ("head", "gaze") or group_of(pid) in ("head", "gaze"))


def decide(scores: dict[str, float], groups: dict[str, str] | None = None,
           threshold: float = THRESHOLD) -> tuple[dict, dict]:
    """(held, protected) -- each {pid: score}. The runtime holds `held`."""
    groups = groups or {}
    held, prot = {}, {}
    for p, v in scores.items():
        bar = HINTED_THRESHOLD if name_hints_whole_model(p) else threshold
        if v < bar:
            continue
        (prot if protected(p, groups.get(p, "")) else held)[p] = v
    return held, prot


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
                       threshold: float = THRESHOLD,
                       groups: dict[str, str] | None = None
                       ) -> dict[str, float] | None:
    """The parameters to hold, with their scores, after protection and hints."""
    sc = scores_for(probe_dir, rig)
    if sc is None:
        return None
    return decide(sc, groups, threshold)[0]


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
        held, prot = decide(sc, None, a.threshold)
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        near = [(p, v) for p, v in ranked
                if p not in held and p not in prot][:a.top]
        fmt = lambda d: ", ".join(f"{p} {v:.2f}" for p, v in  # noqa: E731
                                  sorted(d.items(), key=lambda kv: -kv[1]))
        out[r] = sorted(held)
        print(f"{r}: hold {fmt(held) or '-'}"
              + (f"   | protected {fmt(prot)}" if prot else "")
              + f"   | next: {', '.join(f'{p} {v:.2f}' for p, v in near)}")
    with open(os.path.join(a.probe_dir, "whole_model.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\nThe runtime holds the 'hold' lists, limited to parameters the "
          "model drives\n(breath, mouth and switches are handled elsewhere).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
