#!/usr/bin/env python3
"""
Blind A/B judging of rendered clips.

Numbers say whether motion is lively, smooth and aperiodic; only a person can
say whether it looks like a character. Seeing which source made a clip biases
that judgement, so this hides it: every pair of sources on the same rig is
shown as A and B in random order, the answer key goes to a separate file,
and the tally is computed only after the judgements are written.

    python -m flow.abtest make ab/ Natori/behaviors=runs/beh/preview.mp4 \\
        Natori/flow=runs/flow/preview.mp4 Natori/static=runs/static/preview.mp4
    # open ab/index.html, write one line per pair in ab/judgments.txt:
    #   01 A        (or B, or = for no preference)
    python -m flow.abtest tally ab/

`make` copies the videos to ab/pNN_A.mp4 / pNN_B.mp4, writes index.html and
judgments.txt (to fill in) and answer_key.json. Do not open the key before
judging.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np


def make(out: str, items: list[str], seed: int | None) -> int:
    by_rig = defaultdict(dict)
    for it in items:
        name, _, path = it.partition("=")
        rig, _, src = name.rpartition("/")
        if not path or not os.path.exists(path):
            print(f"missing video: {it}", file=sys.stderr)
            return 1
        by_rig[rig or "-"][src] = path
    rng = np.random.default_rng(seed)
    pairs = [(rig, a, b) for rig, srcs in sorted(by_rig.items())
             for a, b in itertools.combinations(sorted(srcs), 2)]
    order = rng.permutation(len(pairs))
    os.makedirs(out, exist_ok=True)
    key, rows = {}, []
    for n, k in enumerate(order, 1):
        rig, a, b = pairs[k]
        if rng.random() < 0.5:
            a, b = b, a
        pid = f"{n:02d}"
        for side, src in (("A", a), ("B", b)):
            shutil.copy(by_rig[rig][src], os.path.join(out, f"p{pid}_{side}.mp4"))
        key[pid] = {"rig": rig, "A": a, "B": b}
        rows.append(
            f"<h3>{pid} &middot; {html.escape(rig)}</h3><div class=row>"
            f"<figure><video src=p{pid}_A.mp4 controls loop muted></video>"
            f"<figcaption>A</figcaption></figure>"
            f"<figure><video src=p{pid}_B.mp4 controls loop muted></video>"
            f"<figcaption>B</figcaption></figure></div>")
    with open(os.path.join(out, "answer_key.json"), "w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=1)
    with open(os.path.join(out, "judgments.txt"), "w", encoding="utf-8") as fh:
        fh.write("# one line per pair: <pair> A|B|=   (which looks more alive "
                 "and natural)\n")
        fh.writelines(f"{pid} \n" for pid in sorted(key))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><meta charset=utf-8><title>Blind A/B</title>"
                 "<style>body{font-family:sans-serif;max-width:1100px;margin:"
                 "auto;padding:16px}.row{display:flex;gap:12px;flex-wrap:wrap}"
                 "video{width:100%;max-width:520px}figure{margin:0}</style>"
                 "<p>For each pair, write A, B or = into judgments.txt. The "
                 "sources are hidden; do not open answer_key.json first.</p>"
                 + "".join(rows))
    print(f"{len(pairs)} pairs -> {out}/index.html; fill {out}/judgments.txt, "
          f"then: python -m flow.abtest tally {out}")
    return 0


def tally(out: str) -> int:
    with open(os.path.join(out, "answer_key.json"), encoding="utf-8") as fh:
        key = json.load(fh)
    wins, games = defaultdict(float), defaultdict(int)
    head = defaultdict(lambda: [0.0, 0])
    n = 0
    with open(os.path.join(out, "judgments.txt"), encoding="utf-8") as fh:
        for line in fh:
            parts = line.split("#")[0].split()
            if len(parts) < 2 or parts[0] not in key:
                continue
            pick = parts[1].upper()
            k = key[parts[0]]
            if pick not in ("A", "B", "="):
                continue
            n += 1
            for side in ("A", "B"):
                src = k[side]
                games[src] += 1
                w = 0.5 if pick == "=" else float(pick == side)
                wins[src] += w
                pair = tuple(sorted((k["A"], k["B"])))
                if src == pair[0]:
                    head[pair][0] += w
                    head[pair][1] += 1
    if not n:
        print("no judgments yet", file=sys.stderr)
        return 1
    print(f"{n} judged pairs\nsource          win rate")
    for src in sorted(wins, key=lambda s: -wins[s] / games[s]):
        print(f"  {src:<14}{wins[src] / games[src]:>6.0%}  ({games[src]} pairs)")
    print("head to head (first source's win rate, ties count half):")
    for (a, b), (w, g) in sorted(head.items()):
        print(f"  {a} vs {b}: {w / g:.0%} of {g}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("out")
    m.add_argument("items", nargs="+", help="rig/source=video.mp4")
    m.add_argument("--seed", type=int, default=None,
                   help="fix the shuffle (default: fresh each time)")
    t = sub.add_parser("tally")
    t.add_argument("out")
    a = ap.parse_args(argv)
    return make(a.out, a.items, a.seed) if a.cmd == "make" else tally(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
