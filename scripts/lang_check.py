#!/usr/bin/env python3
"""
List the turns of a flow.dataset run whose reply is not in the language the
simulated user was asked to speak, with the user line and the reply, and
count them by avatar, persona and (asked -> reply).

    python scripts/lang_check.py data/datagen/r2pilot2 [--max 40]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data")
    ap.add_argument("--max", type=int, default=40, help="turns to print")
    a = ap.parse_args()
    n = 0
    bad = []
    for f in sorted(glob.glob(os.path.join(a.data, "shards", "*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                lang = r.get("lang") or {}
                n += 1
                if lang.get("reply") != lang.get("asked"):
                    bad.append(r)
    print(f"{len(bad)} of {n} turns reply in another language than asked")
    for key in ("avatar", "persona", "pair"):
        c = collections.Counter()
        for r in bad:
            lang = r["lang"]
            c[{"avatar": (r.get("avatar") or {}).get("name"),
               "persona": r.get("persona"),
               "pair": f"{lang.get('asked')}->{lang.get('reply')}"}[key]] += 1
        print(f"  by {key}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    for r in bad[:a.max]:
        lang = r["lang"]
        print(f"\n{r['id']}  asked {lang.get('asked')} mix {lang.get('mix')} "
              f"user-detected {lang.get('user')} reply {lang.get('reply')}")
        print(f"   U: {r.get('user', '')[:110]}")
        print(f"   A: {r.get('spoken', '')[:110]}")


if __name__ == "__main__":
    main()
