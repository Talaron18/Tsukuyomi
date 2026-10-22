#!/usr/bin/env python3
"""
Can an unseen rig's head, body and gaze parameters be found from descriptors?

The motion-primitive plan drives a handful of roles directly -- head yaw,
pitch and roll, body X/Y/Z, gaze X/Y -- and so needs to know, on a rig it has
never seen, which parameter fills each role. Block 1's retrieval numbers
(~50% exact, ~76% group) average over every parameter, most of which no
primitive will ever drive, and they ask "which parameter of rig B is this
parameter of rig A", not "which parameter of this rig is head yaw". This asks
the second question directly, for the eight roles that matter.

Method, leave-one-rig-out:

  For each held-out rig H, the reference is every OTHER rig's parameter that
  carries the role's standard ID. Each of H's live (non-dead) parameters is
  scored against that role's reference set, and the top-ranked parameter is
  the prediction. Ground truth is H's own parameter with the standard ID,
  used only for scoring -- the rig's names never enter the matching. Rigs
  without the standard ID for a role are not scored on it.

  Scores are CSLS over the pooled reference (all live parameters of the other
  rigs), so a diffuse parameter that is close to everything cannot win every
  role; plain cosine is reported beside it. Roles are ranked independently,
  so two roles can pick the same parameter; the "1:1" column assigns them
  greedily without reuse, which is what a binding step would actually do.

Sign, from the probe's flow fields (--probe only): a primitive also needs to
know which way is "positive". The signed scalar is the net horizontal flow
from min to max for X roles, vertical for Y roles, and curl for Z roles. The
column reports how often the held-out rig's true parameter has the same sign
as the median of the reference rigs -- i.e. whether polarity could be taken
from other rigs without a look at the rig. Gaze moves a few pixels at probe
resolution, so its sign is the noisiest.

Two input modes:

  --probe 'out/rig_*.json'  (default) Recomputes descriptors from the probe
            files with scalar statistics fitted on the reference rigs only,
            so nothing about H leaks into the feature space. Gives sign.
  --rigs rigs/   Uses the exported descriptors. Their shared PCA basis was fit
            on every rig, H included -- unsupervised, but a leak. No sign.

Usage:
    python -m l2dprobe.binding_eval --probe 'out/rig_*.json'
    python -m l2dprobe.binding_eval --rigs rigs/ --json binding.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from .analyze import csls, featurize, fit_scalar_stats, load_rig
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze import csls, featurize, fit_scalar_stats, load_rig

ROLES = ("ParamAngleX", "ParamAngleY", "ParamAngleZ",
         "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
         "ParamEyeBallX", "ParamEyeBallY")


def _unit(f: np.ndarray) -> np.ndarray:
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)


def signed_scalar(role: str, flow_lo, flow_hi) -> float:
    """Which way the parameter moves the image as it goes from min to max."""
    if flow_lo is None or flow_hi is None:
        return 0.0
    d = np.asarray(flow_hi, np.float64) - np.asarray(flow_lo, np.float64)
    if role.endswith("Z"):
        # Same curl as probe._curl_div, on the min->max field.
        dx, dy = d[..., 0], d[..., 1]
        return float((np.gradient(dy, axis=1) - np.gradient(dx, axis=0)).mean())
    mag = np.linalg.norm(d, axis=-1)
    if mag.sum() <= 0:
        return 0.0
    w = mag / mag.sum()
    return float((d[..., 0 if role.endswith("X") else 1] * w).sum())


# ------------------------------------------------------------------ loading

def load_probe(pattern: str) -> list[dict]:
    """Probe files, one per rig name (two naming schemes can coexist)."""
    rigs, seen = [], set()
    for path in sorted(glob.glob(pattern)):
        r = load_rig(path)
        n = r["name"]
        r["name"] = n[4:].split(".")[0] if n.startswith("rig_") else n.split(".")[0]
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        rigs.append(r)
    return rigs


def probe_packs(rigs: list[dict], holdout: str) -> dict:
    """{name: (features, pids, signed-scalar lookup)} with H-free statistics."""
    stats = fit_scalar_stats([r for r in rigs if r["name"] != holdout])
    out = {}
    for r in rigs:
        feats, pids, kept = featurize(r, stats)
        if not len(pids):
            continue
        row = {p["pid"]: i for i, p in enumerate(r["params"])}

        def sign_of(pid, role, r=r, row=row):
            i = row.get(pid)
            if i is None or r.get("flow_lo") is None:
                return 0.0
            return signed_scalar(role, r["flow_lo"][i], r["flow_hi"][i])
        out[r["name"]] = (_unit(feats), list(pids), sign_of)
    return out


def load_rig_files(directory: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        name = os.path.basename(path)[:-4]
        if name.startswith("_"):
            continue                      # _pca.npz
        z = np.load(path, allow_pickle=True)
        if "descriptor" not in z:
            continue
        pids = [str(p) for p in z["pid"]]
        kinds = [str(k) for k in z["kind"]] if "kind" in z else []
        keep = [i for i, k in enumerate(kinds or ["x"] * len(pids))
                if k != "dead"]
        out[name] = (_unit(z["descriptor"][keep].astype(np.float32)),
                     [pids[i] for i in keep], None)
    return out


# --------------------------------------------------------------- evaluation

def rank_roles(packs: dict, holdout: str, use_csls: bool):
    """{role: candidate pids of H, best first} plus the greedy 1:1 binding."""
    fh, ph, _ = packs[holdout]
    ref_f, ref_pid = [], []
    for name, (f, pids, _) in packs.items():
        if name == holdout:
            continue
        ref_f.append(f)
        ref_pid += pids
    if not ref_f:
        return {}, {}
    ref = np.concatenate(ref_f)
    sim = fh @ ref.T
    if use_csls:
        sim = csls(sim)
    ref_pid = np.array(ref_pid, dtype=object)
    scores = {}
    for role in ROLES:
        cols = np.flatnonzero(ref_pid == role)
        if len(cols):
            scores[role] = sim[:, cols].max(axis=1)
    ranked = {r: [ph[i] for i in np.argsort(-s)] for r, s in scores.items()}

    # Greedy one-to-one: the most confident (role, parameter) pair first.
    pairs = sorted(((float(s[i]), r, i) for r, s in scores.items()
                    for i in range(len(ph))), reverse=True)
    bound, used = {}, set()
    for _, r, i in pairs:
        if r in bound or i in used:
            continue
        bound[r] = ph[i]
        used.add(i)
    return ranked, bound


def evaluate(packs_for, names: list[str], use_csls: bool, with_sign: bool):
    per_role = {r: {"n": 0, "top1": 0, "top3": 0, "one2one": 0,
                    "sign_n": 0, "sign_ok": 0, "misses": []} for r in ROLES}
    for h in names:
        packs = packs_for(h)
        if h not in packs:
            continue
        ranked, bound = rank_roles(packs, h, use_csls)
        _, ph, sign_h = packs[h]
        for role, order in ranked.items():
            if role not in ph:
                continue                   # H has no standard ID for it
            s = per_role[role]
            s["n"] += 1
            s["top1"] += int(order[0] == role)
            s["top3"] += int(role in order[:3])
            s["one2one"] += int(bound.get(role) == role)
            if order[0] != role:
                s["misses"].append(f"{h}:{order[0]}")
            if with_sign and sign_h is not None:
                ref = [p[2](role, role) for n, p in packs.items()
                       if n != h and role in p[1]]
                ref = [v for v in ref if v != 0.0]
                truth = sign_h(role, role)
                if ref and truth != 0.0:
                    s["sign_n"] += 1
                    s["sign_ok"] += int(np.sign(truth) == np.sign(np.median(ref)))
    return per_role


def print_table(per_role: dict, title: str, with_sign: bool) -> dict:
    print(title)
    print(f"  {'role':<16}{'n':>3}{'top1':>7}{'top3':>7}{'1:1':>7}"
          + (f"{'sign':>10}" if with_sign else "") + "   top-1 misses")
    tot = defaultdict(int)
    out = {}
    for role, s in per_role.items():
        n = s["n"]
        if not n:
            print(f"  {role:<16}{0:>3}   (no held-out rig has this ID)")
            continue
        for k in ("n", "top1", "top3", "one2one", "sign_n", "sign_ok"):
            tot[k] += s[k]
        sign = (f"{s['sign_ok']:>4}/{s['sign_n']:<4}" if s["sign_n"]
                else "     -    ")
        print(f"  {role:<16}{n:>3}{s['top1'] / n:>7.0%}{s['top3'] / n:>7.0%}"
              f"{s['one2one'] / n:>7.0%}" + (f"{sign:>10}" if with_sign else "")
              + "   " + ", ".join(s["misses"]))
        out[role] = {"n": n, "top1": s["top1"] / n, "top3": s["top3"] / n,
                     "one2one": s["one2one"] / n, "sign_ok": s["sign_ok"],
                     "sign_n": s["sign_n"], "misses": s["misses"]}
    n = tot["n"]
    if n:
        print(f"  {'ALL':<16}{n:>3}{tot['top1'] / n:>7.0%}{tot['top3'] / n:>7.0%}"
              f"{tot['one2one'] / n:>7.0%}"
              + (f"{tot['sign_ok']:>4}/{tot['sign_n']:<4}".rjust(10)
                 if with_sign else ""))
        out["ALL"] = {"n": n, "top1": tot["top1"] / n, "top3": tot["top3"] / n,
                      "one2one": tot["one2one"] / n,
                      "sign_ok": tot["sign_ok"], "sign_n": tot["sign_n"]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--probe", default=None,
                     help="probe files, e.g. 'out/rig_*.json' (the default)")
    src.add_argument("--rigs", default=None,
                     help="directory of exported rigs/<name>.npz")
    ap.add_argument("--json", default=None, help="write results here")
    a = ap.parse_args(argv)

    if a.rigs:
        packs = load_rig_files(a.rigs)
        names = sorted(packs)
        packs_for = lambda h: packs  # noqa: E731
        mode, with_sign = f"rigs {a.rigs} (shared PCA basis: mild leak)", False
    else:
        pattern = a.probe or "out/rig_*.json"
        rigs = load_probe(pattern)
        names = [r["name"] for r in rigs]
        cache = {}

        def packs_for(h):
            if h not in cache:
                cache.clear()
                cache[h] = probe_packs(rigs, h)
            return cache[h]
        mode, with_sign = f"probe {pattern}", True
    if len(names) < 2:
        print("need at least two rigs", file=sys.stderr)
        return 1

    print(f"{len(names)} rigs from {mode}: {', '.join(names)}")
    print("leave-one-rig-out; n = held-out rigs that have the standard ID\n")
    res = {"mode": mode, "rigs": names}
    for label, use in (("csls", True), ("cosine", False)):
        per = evaluate(packs_for, names, use, with_sign)
        res[label] = print_table(per, f"[{label}]", with_sign)
        print()
    if len(names) < 15:
        print(f"Only {len(names)} rigs: each cell is a handful of trials and "
              "moves by 1/n per hit. Read it as a smoke test, not an estimate.")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
