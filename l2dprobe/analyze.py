#!/usr/bin/env python3
"""
Does the descriptor space actually generalise across rigs?

This is the experiment Block 1 exists to pass. The claim is that a parameter
can be identified by what it *does* to the image rather than by its name. If
that holds, a parameter on an unseen rig is usable without a hand-written
mapping. If it does not, the zero-shot premise fails and Block 2 would be built
on sand.

Method
------
Probe several rigs. Many of them share standard Cubism IDs (ParamAngleX,
ParamEyeLOpen, ...). Take a parameter from rig A, find its nearest neighbour in
rig B *using descriptors only*, then check whether the neighbour happens to
have the same ID.

Names are ground truth for scoring and are never an input to the matching. A
high score means the visual descriptor recovers parameter identity on its own.

The baseline to beat is chance, which for ~70 candidates is about 1.4%. The
number that matters more is top-1 on the common standard parameters, since
those are the ones where "same name" reliably means "same function".

Usage
-----
    python -m l2dprobe.analyze out/rig_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Scalar descriptor fields appended to the flattened support map.
SCALARS = ("area", "linearity", "stepiness", "alpha_ratio",
           "curl", "divergence", "travel_lo", "travel_hi")
# Derived in analyze from the saved maps rather than stored per parameter.
DERIVED = ("lateral", "flow_dx", "flow_dy", "flow_aniso")
SUPPORT_WEIGHT = 1.0
FLOW_WEIGHT = 1.5       # displacement: direction, and appearance-invariant
SCALAR_WEIGHT = 1.2     # standardised, so this is now a real contribution
CENTROID_WEIGHT = 1.0
BLUR_PASSES = 2         # soften support so near-misses still overlap
CSLS_K = 8              # neighbourhood size for hubness correction


# Functional grouping of the standard Cubism parameter IDs. Used only for
# scoring, never as a matching input.
#
# Exact-name retrieval is a harsher target than the pipeline actually needs.
# The motion model does not identify parameters by name; it needs functionally
# similar parameters to sit near each other, so that an unseen rig's head
# rotation lands where the model has learned to drive head rotation. Retrieving
# ParamBodyAngleZ for ParamAngleZ is a near hit for that purpose, but exact-name
# scoring calls it a total miss. Group accuracy measures the real requirement.
GROUPS = (
    ("gaze", ("EyeBall",)),
    ("eyelid", ("EyeLOpen", "EyeROpen", "EyeSmile", "EyeForm")),
    ("brow", ("Brow",)),
    ("mouth", ("Mouth",)),
    ("head", ("AngleX", "AngleY", "AngleZ")),
    ("body", ("BodyAngle", "AllRotate")),
    ("arm", ("Arm", "Hand", "Elbow", "Shoulder")),
    ("hair", ("Hair",)),
    ("face", ("Cheek", "Tere", "Blush")),
    ("breath", ("Breath",)),
)


def group_of(pid: str) -> str:
    for name, keys in GROUPS:
        if any(k.lower() in pid.lower() for k in keys):
            return name
    return "other"


def load_rig(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as fh:
        blob = json.load(fh)
    npz_path = json_path.replace(".json", ".npz")
    blob["support"] = blob["flow_lo"] = blob["flow_hi"] = None
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=True) as z:
            for key in ("support", "flow_lo", "flow_hi"):
                if key in z:
                    blob[key] = z[key]
    blob["name"] = os.path.basename(json_path).replace(".json", "")
    return blob


def scalar_vector(p: dict, flow_lo=None, flow_hi=None) -> np.ndarray:
    """Scalar features for one parameter, including derived ones.

    Area is log-scaled: gaze touches ~0.3% of the character and head roll ~30%,
    a hundredfold difference nearly invisible on a linear axis.

    Two derived families close gaps that showed up directly in the confusion
    table:

    lateral   signed offset of the support centroid from the character midline.
              Left and right eyelids produce near-identical descriptors and were
              mutually confused 26x and 23x; laterality is the one bit that
              separates them, and blurring on a full-body grid had erased it.

    flow_dx/dy  net translation direction. curl and divergence were measured but
              net direction never was, so horizontal gaze and vertical head
              pitch had nothing distinguishing them (EyeBallX -> AngleY, 27x).
    """
    vals = []
    for k in SCALARS:
        v = float(p.get(k, 0.0))
        if k == "area":
            v = np.log10(max(v, 1e-6))
        vals.append(v)

    cx = float(p.get("centroid", (0.5, 0.5))[0])
    vals.append((cx - 0.5) * 2.0)  # lateral, in [-1, 1]

    flow = None
    for f in (flow_hi, flow_lo):
        if f is not None and np.abs(f).max() > 0:
            flow = f
            break
    if flow is None:
        vals.extend([0.0, 0.0, 0.0])
    else:
        mag = np.linalg.norm(flow, axis=-1)
        w = mag / (mag.sum() + 1e-9)
        dx = float((flow[..., 0] * w).sum())
        dy = float((flow[..., 1] * w).sum())
        vals.extend([dx, dy])
        # anisotropy: +1 purely horizontal, -1 purely vertical, 0 isotropic
        ax, ay = abs(dx), abs(dy)
        vals.append((ax - ay) / (ax + ay + 1e-9))
    return np.asarray(vals, np.float32)


def fit_scalar_stats(rigs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of each scalar feature across every parameter of every rig.

    Without this the scalars go in raw: linearity is ~0.9, area ~0.003, curl
    ~1e-4. One shared weight then lets linearity dominate the block while curl
    and divergence contribute nothing at all -- which is why rotation never got
    separated from translation, despite curl being measured specifically for
    that purpose. Standardising gives each feature comparable influence.
    """
    rows = []
    for r in rigs:
        for i, p in enumerate(r["params"]):
            if p["kind"] == "dead":
                continue
            lo = r["flow_lo"][i] if r.get("flow_lo") is not None else None
            hi = r["flow_hi"][i] if r.get("flow_hi") is not None else None
            rows.append(scalar_vector(p, lo, hi))
    if not rows:
        n = len(SCALARS) + len(DERIVED)
        return np.zeros(n, np.float32), np.ones(n, np.float32)
    m = np.stack(rows)
    return m.mean(axis=0), np.maximum(m.std(axis=0), 1e-6)


def _blur(m: np.ndarray, passes: int = BLUR_PASSES) -> np.ndarray:
    """Cheap separable box blur.

    Small-support parameters (an eye) occupy only a couple of grid cells, so
    tiny differences in face proportion between rigs drive their overlap to
    zero. Softening the map lets a near-miss still score.
    """
    out = m.astype(np.float32)
    for _ in range(passes):
        pad = np.pad(out, 1, mode="edge")
        out = (pad[:-2, 1:-1] + pad[2:, 1:-1] +
               pad[1:-1, :-2] + pad[1:-1, 2:] + 4 * out) / 8.0
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def csls(sim: np.ndarray, k: int = CSLS_K) -> np.ndarray:
    """Cross-domain similarity local scaling.

    Plain cosine retrieval has hubs: a broad, diffuse descriptor such as
    ParamArmL sits close to everything, so localized parameters all retrieve it
    instead of their true match. CSLS subtracts each item's mean similarity to
    its own k nearest neighbours in the other rig, penalising items that are
    close to everything.
    """
    kq = min(k, sim.shape[1])
    kt = min(k, sim.shape[0])
    rq = np.sort(sim, axis=1)[:, -kq:].mean(axis=1, keepdims=True)
    rt = np.sort(sim, axis=0)[-kt:, :].mean(axis=0, keepdims=True)
    return 2 * sim - rq - rt


def featurize(rig: dict, stats=None) -> tuple[np.ndarray, list[str], list[dict]]:
    """Build one feature vector per live parameter."""
    feats, pids, keep = [], [], []
    for i, p in enumerate(rig["params"]):
        if p["kind"] == "dead":
            continue
        sup = (_unit(_blur(rig["support"][i]).ravel())
               if rig["support"] is not None else np.zeros(1, np.float32))

        # Normalise the two flow sides JOINTLY, never separately. A parameter
        # whose default sits at one end of its range has a near-zero field on
        # the dead side; normalising that side on its own would turn rounding
        # noise into a unit-length vector and drown the real signal.
        parts_flow = []
        for key in ("flow_lo", "flow_hi"):
            arr = rig.get(key)
            parts_flow.append(arr[i].ravel().astype(np.float32)
                              if arr is not None else np.zeros(1, np.float32))
        flow = _unit(np.concatenate(parts_flow))

        lo_arr = rig["flow_lo"][i] if rig.get("flow_lo") is not None else None
        hi_arr = rig["flow_hi"][i] if rig.get("flow_hi") is not None else None
        scalars = scalar_vector(p, lo_arr, hi_arr)
        if stats is not None:
            scalars = (scalars - stats[0]) / stats[1]
        scalars = scalars / np.sqrt(len(SCALARS) + len(DERIVED))  # keep block norm comparable
        parts = [SUPPORT_WEIGHT * sup,
                 FLOW_WEIGHT * flow,
                 CENTROID_WEIGHT * np.asarray(p["centroid"], np.float32),
                 SCALAR_WEIGHT * scalars]
        feats.append(np.concatenate(parts))
        pids.append(p["pid"])
        keep.append(p)
    if not feats:
        return np.zeros((0, 1), np.float32), [], []
    return np.stack(feats), pids, keep


def cross_rig_retrieval(rigs: list[dict]) -> None:
    stats = fit_scalar_stats(rigs)
    packs = {r["name"]: featurize(r, stats) for r in rigs}

    print("=" * 68)
    print("CROSS-RIG RETRIEVAL  (descriptors in, names used only for scoring)")
    print("=" * 68)

    totals = {"n": 0, "top1": 0, "top3": 0, "grp1": 0, "grp_n": 0,
              "rtop1": 0, "mrr": 0.0}
    confusions: dict[str, list[str]] = defaultdict(list)

    for a in rigs:
        fa, pa, _ = packs[a["name"]]
        for b in rigs:
            if a is b:
                continue
            fb, pb, _ = packs[b["name"]]
            if len(fa) == 0 or len(fb) == 0:
                continue
            shared = set(pa) & set(pb)
            if not shared:
                continue

            # cosine distance in descriptor space
            na = fa / (np.linalg.norm(fa, axis=1, keepdims=True) + 1e-9)
            nb = fb / (np.linalg.norm(fb, axis=1, keepdims=True) + 1e-9)
            sim = csls(na @ nb.T)

            # Restricted pool: candidates limited to the parameters the two
            # rigs share. The full-pool number conflates representation quality
            # with candidate count -- a 92-parameter rig like Mao, most of whose
            # parameters are character-specific props with no counterpart
            # anywhere, presents far more distractors than a 19-parameter rig.
            # The restricted number isolates the representation.
            cand = [j for j, q in enumerate(pb) if q in shared]
            sim_r = csls(na @ nb[cand].T) if cand else None

            n = top1 = top3 = 0
            grp1 = grp_n = 0
            rtop1 = 0
            mrr = 0.0
            for i, pid in enumerate(pa):
                if pid not in shared:
                    continue
                order = np.argsort(-sim[i])
                ranked = [pb[j] for j in order[:3]]
                n += 1
                if ranked[0] == pid:
                    top1 += 1
                else:
                    confusions[pid].append(ranked[0])
                if pid in ranked:
                    top3 += 1
                g = group_of(pid)
                if g != "other":
                    grp_n += 1
                    if group_of(ranked[0]) == g:
                        grp1 += 1

                if sim_r is not None:
                    ro = np.argsort(-sim_r[i])
                    rranked = [pb[cand[j]] for j in ro]
                    if rranked[0] == pid:
                        rtop1 += 1
                    if pid in rranked:
                        mrr += 1.0 / (rranked.index(pid) + 1)

            if n:
                grp_s = f" grp={grp1 / grp_n:5.1%}" if grp_n else ""
                print(f"  {a['name'][4:18]:>14} -> {b['name'][4:18]:<14} "
                      f"n={n:<3} pool={len(pb):<3} "
                      f"top1={top1 / n:5.1%} top3={top3 / n:5.1%}"
                      f"{grp_s} | restricted top1={rtop1 / n:5.1%} "
                      f"mrr={mrr / n:.3f}")
                totals["n"] += n
                totals["top1"] += top1
                totals["top3"] += top3
                totals["grp1"] += grp1
                totals["grp_n"] += grp_n
                totals["rtop1"] += rtop1
                totals["mrr"] += mrr

    if not totals["n"]:
        print("  no shared parameter IDs between rigs; cannot score")
        return

    n = totals["n"]
    print("-" * 68)
    print(f"  OVERALL   n={n}   top1={totals['top1'] / n:.1%}   "
          f"top3={totals['top3'] / n:.1%}")
    print(f"  RESTRICTED  top1={totals['rtop1'] / n:.1%}   "
          f"MRR={totals['mrr'] / n:.3f}   "
          "(candidates limited to shared parameters)")
    if totals["grp_n"]:
        gn = totals["grp_n"]
        print(f"  GROUP     n={gn}   top1={totals['grp1'] / gn:.1%}   "
              f"(retrieved parameter has the same function, ~10 groups)")

    if confusions:
        print("\n  most confused parameters:")
        ranked = sorted(confusions.items(), key=lambda kv: -len(kv[1]))[:10]
        for pid, wrong in ranked:
            common = max(set(wrong), key=wrong.count)
            print(f"    {pid:<26} -> {common:<26} ({len(wrong)}x)")


def detect_mirrors(rig: dict, iou_min: float = 0.35) -> list[tuple[str, str]]:
    """Find L/R parameter pairs from the saved maps.

    Redone here rather than in probe.py because five of seven rigs reported
    zero mirrored pairs, which is not a property of rigs that visibly have two
    eyes. Doing it in analyze also means it can be re-run without re-probing.

    A true mirror pair satisfies two conditions, not one: the support maps
    overlap after a horizontal flip, AND the horizontal flow components are
    opposed. Support alone matches any two things in the same place, which is
    why a bare IoU threshold failed.
    """
    sup, flo = rig.get("support"), rig.get("flow_hi")
    if sup is None:
        return []
    live = [(i, p) for i, p in enumerate(rig["params"]) if p["kind"] != "dead"]
    pairs, used = [], set()
    for a_pos, (i, pa) in enumerate(live):
        if pa["pid"] in used:
            continue
        flipped = np.fliplr(sup[i])
        best, best_score = None, 0.0
        for j, pb in live[a_pos + 1:]:
            if pb["pid"] in used:
                continue
            inter = np.minimum(flipped, sup[j]).sum()
            union = np.maximum(flipped, sup[j]).sum() + 1e-9
            score = float(inter / union)
            if abs(pa["centroid"][0] - pb["centroid"][0]) < 0.02:
                continue  # same side of the midline: not a mirror pair
            if flo is not None:
                fa, fb = flo[i], flo[j]
                if np.abs(fa).max() > 0 and np.abs(fb).max() > 0:
                    # horizontal components should oppose after the flip
                    opp = -float((np.fliplr(fa[..., 0]) * fb[..., 0]).sum())
                    denom = (np.linalg.norm(fa[..., 0]) *
                             np.linalg.norm(fb[..., 0]) + 1e-9)
                    score *= 0.5 + 0.5 * np.clip(opp / denom, -1, 1)
            if score > best_score:
                best, best_score = (j, pb), score
        if best is not None and best_score >= iou_min:
            pairs.append((pa["pid"], best[1]["pid"]))
            used.update({pa["pid"], best[1]["pid"]})
    return pairs


def summarize(rigs: list[dict]) -> None:
    print("\n" + "=" * 68)
    print("PER-RIG SUMMARY")
    print("=" * 68)
    for r in rigs:
        m = r["meta"]
        # Canvas coverage is the first thing to check when one rig scores badly
        # against every other. A character occupying a small or oddly shaped
        # slice of the canvas yields support maps built from very few pixels,
        # and every descriptor derived from them is noise.
        bbox = m.get("content_bbox")
        size = m.get("size", 512)
        if bbox:
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cover = f"bbox={bw}x{bh} ({bw * bh / (size * size):5.1%} of canvas)"
        else:
            cover = "bbox=<not recorded, re-probe>"
        print(f"  {r['name']:<24} {m['n_params']:>3} params  "
              f"{m.get('n_continuous', 0):>3} cont  "
              f"{m.get('n_categorical', 0):>3} categ  "
              f"{m.get('n_dead', 0):>3} dead  "
              f"{m.get('n_mirrored', 0):>3} mirr  {cover}")

    print("\n  mirror pairs re-detected in analyze (support flip + opposed flow):")
    for r in rigs:
        pairs = detect_mirrors(r)
        sample = ", ".join(f"{a}~{b}" for a, b in pairs[:3])
        print(f"    {r['name']:<24} {len(pairs):>3} pairs   {sample}")

    print("\n  per-rig mean support area (fraction of character box):")
    for r in rigs:
        areas = [p["area"] for p in r["params"] if p["kind"] != "dead"]
        if not areas:
            continue
        a = np.asarray(areas)
        print(f"    {r['name']:<24} mean={a.mean():.4f}  median="
              f"{np.median(a):.4f}  min={a.min():.5f}  max={a.max():.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rigs", nargs="+", help="probed *.json files (globs ok)")
    a = ap.parse_args(argv)

    paths = []
    for pattern in a.rigs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    rigs = [load_rig(p) for p in paths if os.path.exists(p)]
    if len(rigs) < 2:
        print("need at least two probed rigs to compare", file=sys.stderr)
        return 1

    summarize(rigs)
    cross_rig_retrieval(rigs)

    print("\nReading this: top1 well above chance means the descriptor recovers")
    print("parameter identity from appearance alone, and the rig abstraction")
    print("holds. Near chance means Block 2 needs a different representation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
