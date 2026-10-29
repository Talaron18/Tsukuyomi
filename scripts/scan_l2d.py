#!/usr/bin/env python3
"""
Rank the Live2D models under a folder by what a talking avatar can use:
face controls (eyes, brows, mouth, cheeks), arm and hand controls, arm
drawings (parts switched in and out for other arm poses) and body angles.
Reads the .moc3 through the Cubism Core only: no rendering, no GL.

    python scripts/scan_l2d.py ~/live2d-models [--json scan.json]

The score is a rough sort key, not a verdict: face controls count once,
arm and hand controls twice, extra arm drawings three times. Check the
top of the list with flow.body before adding a model to the avatars.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l2dprobe import core  # noqa: E402

FACE = re.compile(r"eye|brow|mouth|cheek|tere|blush|tear|face", re.I)
ARM = re.compile(r"arm|hand|elbow|finger|wrist|ude|te[_A-Z0-9]", re.I)
BODY = re.compile(r"body|bust|shoulder|waist|hip|leg", re.I)
ARM_PART = re.compile(r"arm|hand|ude", re.I)


def scan(model_json):
    m = core.load(model_json)
    ids = m.param_ids
    face = [p for p in ids if FACE.search(p)]
    arm = [p for p in ids if ARM.search(p) and not FACE.search(p)]
    body = [p for p in ids if BODY.search(p) and p not in arm]
    arm_parts = [p for p in m.part_ids if ARM_PART.search(p)]
    with open(model_json, encoding="utf-8") as fh:
        refs = json.load(fh).get("FileReferences", {})
    motions = sum(len(v) for v in (refs.get("Motions") or {}).values())
    return {"model": model_json, "params": len(ids), "face": len(face),
            "arm": len(arm), "body": len(body),
            "arm_parts": len(arm_parts), "motions": motions,
            "expressions": len(refs.get("Expressions") or []),
            "score": len(face) + 2 * len(arm) + 3 * max(0, len(arm_parts) - 2),
            "arm_ids": arm, "arm_part_ids": arm_parts}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--json", default=None, help="write every row here")
    a = ap.parse_args()
    if not core.available():
        sys.exit("no Cubism Core library found (see l2dprobe/core.py)")
    rows, bad = [], []
    for dp, _, fs in os.walk(a.root):
        for f in fs:
            if f.endswith(".model3.json"):
                p = os.path.join(dp, f)
                try:
                    rows.append(scan(p))
                except Exception as e:  # an old or broken moc: list, go on
                    bad.append((p, f"{type(e).__name__}: {e}"))
    rows.sort(key=lambda r: -r["score"])
    print(f"{'score':>5} {'face':>4} {'arm':>4} {'armpt':>5} {'body':>4} "
          f"{'mot':>4} {'exp':>4}  model")
    for r in rows:
        print(f"{r['score']:5d} {r['face']:4d} {r['arm']:4d} "
              f"{r['arm_parts']:5d} {r['body']:4d} {r['motions']:4d} "
              f"{r['expressions']:4d}  {os.path.relpath(r['model'], a.root)}")
    for p, e in bad:
        print(f"  skipped {os.path.relpath(p, a.root)}: {e}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "skipped": bad}, fh, ensure_ascii=False,
                      indent=1)


if __name__ == "__main__":
    main()
