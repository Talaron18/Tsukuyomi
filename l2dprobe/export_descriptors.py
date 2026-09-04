#!/usr/bin/env python3
"""
Freeze Block 1 into a rig interface for the motion model.

Block 1 asked: can a Live2D parameter be described by what it DOES rather than
by its name, well enough that an unseen rig is usable without a hand-written
mapping? The answer is a qualified yes. Hand-designed descriptors reach ~49.5%
exact retrieval and ~76% functional-group accuracy on a held-out rig, which is
the number that matters downstream: the motion model needs an unseen rig's head
rotation to land where it has learned to drive head rotation, not to be named
correctly.

A learned encoder was built and evaluated leave-one-rig-out. It improves with
data (36.9% -> 43.7% going from 7 to 41 rigs with balanced sampling) but has
not overtaken the hand-designed features, and fusing the two is worth about
+1-2 points, within run-to-run noise. So the default here is hand-designed,
with fusion available behind a flag.

What this writes, per rig:

    descriptor  (P, D)  the frozen parameter representation
    pid         (P,)    parameter ids
    kind        (P,)    continuous | categorical
    minimum, maximum, default, span
    mirror      (P,)    index of the L/R partner, or -1
    group       (P,)    coarse function label where inferable

`kind` matters for Block 2: categorical parameters are part swaps and opacity
toggles, so a continuous motion model must not interpolate through them. They
need a discrete head. Interpolating a part swap produces half-visible limbs.

`mirror` is name-based here rather than descriptor-based. Descriptor mirror
detection failed on five of seven rigs -- laterality is the weakest part of the
representation -- and L/R substring matching is both more reliable and free.

Usage:
    python -m l2dprobe.export_descriptors --probe 'out/rig_*.json' --out rigs/
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    from .analyze import (csls, featurize, fit_scalar_stats, group_of, load_rig)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze import csls, featurize, fit_scalar_stats, group_of, load_rig


def score(packs, holdout: str, targets: set | None = None) -> dict:
    """Restricted retrieval, same metric used throughout Block 1."""
    by = {n: (f, pids) for n, f, pids, _ in packs}
    if holdout not in by:
        return {}
    fa, pa = by[holdout]
    n = t1 = g1 = gn = 0
    mrr = 0.0
    for other, (fb, pb) in by.items():
        if other == holdout or (targets is not None and other not in targets):
            continue
        shared = set(pa) & set(pb)
        cand = [j for j, q in enumerate(pb) if q in shared]
        if not cand:
            continue
        na = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-9)
        nb = fb[cand] / (np.linalg.norm(fb[cand], axis=1, keepdims=True) + 1e-9)
        sim = csls(na @ nb.T)
        for i, pid in enumerate(pa):
            if pid not in shared:
                continue
            ranked = [pb[cand[j]] for j in np.argsort(-sim[i])]
            n += 1
            t1 += ranked[0] == pid
            mrr += 1.0 / (ranked.index(pid) + 1)
            g = group_of(pid)
            if g != "other":
                gn += 1
                g1 += group_of(ranked[0]) == g
    if not n:
        return {}
    return {"n": n, "top1": t1 / n, "mrr": mrr / n,
            "group": g1 / gn if gn else float("nan")}

# Ordered so the longer, more specific patterns are tried first.
LR_PATTERNS = (
    (re.compile(r"(?i)(.*)left(.*)"), (re.compile(r"(?i)(.*)right(.*)"))),
    (re.compile(r"^(Param)([A-Za-z]*)L([A-Z].*|)$"),
     re.compile(r"^(Param)([A-Za-z]*)R([A-Z].*|)$")),
    (re.compile(r"^(.*)_L$"), re.compile(r"^(.*)_R$")),
)


def mirror_index(pids: list[str]) -> np.ndarray:
    """Pair L/R parameters by name.

    Name matching beats the descriptor here. The flow-based detector found zero
    pairs on five of seven rigs, because on a full-body rig the two eyes sit two
    or three grid cells apart and the blur that makes near-misses match erases
    the difference. Names are unreliable in general -- that is the premise of
    this whole block -- but laterality is the one thing they encode consistently.
    """
    out = np.full(len(pids), -1, dtype=np.int32)
    index = {p: i for i, p in enumerate(pids)}
    for i, pid in enumerate(pids):
        if out[i] != -1:
            continue
        for cand in _lr_partners(pid):
            j = index.get(cand)
            if j is not None and j != i and out[j] == -1:
                out[i], out[j] = j, i
                break
    return out


def _lr_partners(pid: str) -> list[str]:
    """Candidate names for the mirror partner of `pid`, both directions."""
    cands = []
    for a, b in (("Left", "Right"), ("left", "right"), ("_L", "_R"), ("_l", "_r")):
        if a in pid:
            cands.append(pid.replace(a, b))
        if b in pid:
            cands.append(pid.replace(b, a))
    # ParamEyeLOpen <-> ParamEyeROpen: a bare L/R before a capital or end.
    m = re.match(r"^(Param[A-Za-z]*?)([LR])([A-Z].*|)$", pid)
    if m:
        head, side, tail = m.groups()
        cands.append(f"{head}{'R' if side == 'L' else 'L'}{tail}")
    return cands


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="out/rig_*.json")
    ap.add_argument("--out", default="rigs")
    ap.add_argument("--fuse-npz", default=None,
                    help="optional learned embeddings; adds ~1-2 points, "
                         "within noise, so off by default")
    ap.add_argument("--fuse-weight", type=float, default=2.0)
    ap.add_argument("--eval-holdout", default=None,
                    help="sweep PCA widths and report retrieval, then exit "
                         "logic continues with --pca")
    ap.add_argument("--eval-rigs", default=None,
                    help="comma-separated retrieval targets for --eval-holdout")
    ap.add_argument("--pca", type=int, default=128,
                    help="reduce descriptor width with a shared PCA basis; "
                         "0 keeps the full ~2062 dims")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(a.probe))
    if not paths:
        print(f"no probe files matched {a.probe}", file=sys.stderr)
        return 1

    rigs = [load_rig(p) for p in paths]
    for r in rigs:
        n = r["name"]
        r["name"] = n[4:].split(".")[0] if n.startswith("rig_") else n.split(".")[0]

    # out/ can hold two naming schemes from different runs (rig_Haru.json from
    # pipeline.py and rig_Haru.model3.json from early --batch runs), so the same
    # rig arrives twice. The .npz output would be fine -- second write wins --
    # but the scalar statistics below are pooled over every rig, and
    # double-counting seven of them skews the shared feature space toward them.
    seen, deduped, dupes = set(), [], []
    for r in rigs:
        if r["name"] in seen:
            dupes.append(r["name"])
            continue
        seen.add(r["name"])
        deduped.append(r)
    if dupes:
        print(f"ignored {len(dupes)} duplicate rig files: "
              f"{', '.join(sorted(set(dupes)))}\n")
    rigs = deduped

    # Scalar standardisation must use statistics pooled over every rig, or two
    # rigs exported separately would land in different spaces.
    stats = fit_scalar_stats(rigs)
    learned = np.load(a.fuse_npz, allow_pickle=True) if a.fuse_npz else None

    os.makedirs(a.out, exist_ok=True)

    # Featurize everything first: PCA needs a single basis shared by all rigs,
    # or two rigs would land in incomparable spaces.
    packs = []
    for r in rigs:
        feats, pids, kept = featurize(r, stats)
        if len(feats) == 0:
            print(f"{r['name']}: no live parameters, skipped")
            continue
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        if learned is not None and r["name"] in learned:
            emb = learned[r["name"]]
            if len(emb) == len(feats):
                feats = np.concatenate([emb, a.fuse_weight * feats], axis=1)
                feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
            else:
                print(f"  {r['name']}: learned embedding has {len(emb)} rows "
                      f"vs {len(feats)} parameters, not fused")
        packs.append((r["name"], feats, pids, kept))

    if a.eval_holdout and packs:
        # Variance retained is the wrong thing to judge this on: these
        # descriptors are sparse and high-rank, so variance spreads thin while
        # the cosine structure retrieval depends on may survive fine. Measure
        # retrieval at each width instead of inferring it.
        targets = ({x.strip() for x in a.eval_rigs.split(",")}
                   if a.eval_rigs else None)
        allf = np.concatenate([f for _, f, _, _ in packs])
        mu = allf.mean(axis=0, keepdims=True)
        _, sv, vt = np.linalg.svd(allf - mu, full_matrices=False)
        print(f"retrieval vs PCA width (holdout {a.eval_holdout}):")
        for k in (32, 64, 128, 256, 512, 0):
            if k:
                b = vt[:k]
                red = [(n, ((f - mu) @ b.T), pids, kept)
                       for n, f, pids, kept in packs]
                var = float((sv[:k] ** 2).sum() / (sv ** 2).sum())
            else:
                red, var = packs, 1.0
            r = score(red, a.eval_holdout, targets)
            if r:
                label = f"{k:>4}" if k else "full"
                print(f"  dim={label}  var={var:6.1%}  n={r['n']}  "
                      f"top1={r['top1']:6.1%}  mrr={r['mrr']:.3f}  "
                      f"group={r['group']:6.1%}")
        print()

    basis = mean = None
    if a.pca and packs:
        # 2062 dimensions is mostly empty support cells, heavily correlated
        # across parameters. Block 2 conditions on one descriptor per parameter
        # and a rig can have 150+, so the raw width is a real cost there.
        allf = np.concatenate([f for _, f, _, _ in packs])
        mean = allf.mean(axis=0, keepdims=True)
        _, sv, vt = np.linalg.svd(allf - mean, full_matrices=False)
        k = min(a.pca, vt.shape[0])
        basis = vt[:k]
        kept_var = float((sv[:k] ** 2).sum() / (sv ** 2).sum())
        print(f"PCA {allf.shape[1]} -> {k} dims, {kept_var:.1%} variance "
              f"retained (basis shared by all rigs)\n")
        np.savez_compressed(os.path.join(a.out, "_pca.npz"),
                            basis=basis.astype(np.float32),
                            mean=mean.astype(np.float32))

    total = 0
    for name, feats, pids, kept in packs:
        if basis is not None:
            feats = (feats - mean) @ basis.T
            feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

        dest = os.path.join(a.out, f"{name}.npz")
        mir = mirror_index(pids)
        np.savez_compressed(
            dest,
            descriptor=feats.astype(np.float32),
            pid=np.array(pids, dtype=object),
            kind=np.array([p["kind"] for p in kept], dtype=object),
            minimum=np.array([p["minimum"] for p in kept], np.float32),
            maximum=np.array([p["maximum"] for p in kept], np.float32),
            default=np.array([p["default"] for p in kept], np.float32),
            mirror=mir,
            group=np.array([group_of(p) for p in pids], dtype=object),
        )
        n_cat = sum(p["kind"] == "categorical" for p in kept)
        print(f"{name:<24} {len(pids):>4} params  dim={feats.shape[1]:<5} "
              f"{n_cat:>3} categorical  {int((mir >= 0).sum()):>3} mirrored")
        total += len(pids)

    print(f"\n{total} parameters across {len(rigs)} rigs written to {a.out}/")
    print("Block 2 consumes these as the rig interface: one descriptor per")
    print("parameter, plus the range and default needed to emit valid values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
