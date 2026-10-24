#!/usr/bin/env python3
"""
Block 3, step 1: synthetic supervision for the inverse renderer.

The inverse problem is image -> parameters. Its training data is free and exact,
because the forward direction is a renderer we already drive: sample a pose,
render it, keep the pair. No annotation, no tracking, no ambiguity about ground
truth.

Two choices shape whether the result transfers to real video.

Pose sampling
-------------
Uniform random parameters produce poses no character ever holds -- eyes crossed
while the head is bent backwards -- and a model trained on them wastes capacity
on a region of pose space that video never visits. Poses are therefore drawn
from the authored clips where they exist, with perturbation for coverage, and
fall back to perturbed defaults for rigs without motion.

Band selection
--------------
Only the expressive band is a target, matching the prior. Physics-driven
parameters are excluded for a reason specific to this problem: they are a
function of motion history, not of the current pose, so they are not recoverable
from a single frame even in principle. Predicting them would be fitting noise.
Rendering also strips physics so that a parameter vector maps to exactly one
image -- otherwise identical inputs would carry different labels.

Domain gap
----------
Renders are clean and centred on transparency; video is compressed, backgrounded
and arbitrarily framed. Augmentation belongs at training time rather than here,
so it can be varied without re-rendering, but this writes the alpha channel so
compositing over random backgrounds stays possible.

Usage:
    python -m l2dprobe.render_dataset --manifest manifest.json \\
        --rigs rigs/ --clips clips/ --out synth/ --samples 4000 --jobs 12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# Imported at module level, not inside the worker: ProcessPoolExecutor
# re-imports this module in each child, so module-level imports resolve there
# too. Doing it in the worker also meant hardcoding the package directory name,
# which broke as soon as the package was not called l2dprobe.
try:
    from ...l2dprobe.backend import Live2DBackend, ensure_display
    from ...l2dprobe.motions import band_of, physics_params
except ImportError:
    if PKG_DIR not in sys.path:
        sys.path.insert(0, PKG_DIR)
    from l2dprobe.backend import Live2DBackend, ensure_display
    from l2dprobe.motions import band_of, physics_params


def sample_poses(clips_npz: str | None, default_n: np.ndarray, n: int,
                 rng) -> np.ndarray:
    """Draw plausible normalised poses, shape (n, P).

    Real frames are the backbone. Interpolating between two random frames adds
    poses that lie between authored ones, which is where video actually spends
    most of its time. Noise widens the distribution without leaving the region
    the rig was designed for.
    """
    pool = []
    if clips_npz and os.path.exists(clips_npz):
        z = np.load(clips_npz, allow_pickle=True)
        for i in range(int(z["n_clips"])):
            pool.append(z[f"clip{i}"].astype(np.float32))
    if pool:
        frames = np.concatenate(pool)
        idx = rng.integers(0, len(frames), n)
        poses = frames[idx].copy()
        # Half the samples are blends of two real frames.
        blend = rng.random(n) < 0.5
        j = rng.integers(0, len(frames), n)
        w = rng.random((n, 1)).astype(np.float32)
        poses[blend] = (poses[blend] * w[blend]
                        + frames[j][blend] * (1 - w[blend]))
    else:
        poses = np.tile(default_n.astype(np.float32), (n, 1))

    poses = poses + rng.normal(0, 0.12, poses.shape).astype(np.float32)
    return np.clip(poses, -1, 1)


def build_rig(args) -> dict:
    path, rig_npz, clips_npz, out_dir, n, size, seed = args
    name = os.path.basename(rig_npz).replace(".npz", "")
    rng = np.random.default_rng(seed)
    z = np.load(rig_npz, allow_pickle=True)
    pids = [str(x) for x in z["pid"]]
    kinds = [str(x) for x in z["kind"]]
    groups = [str(x) for x in z["group"]]
    lo, hi = z["minimum"].astype(np.float32), z["maximum"].astype(np.float32)
    dflt = z["default"].astype(np.float32)

    phys, _ = physics_params(path)
    bands = [band_of(p, k, g, phys) for p, k, g in zip(pids, kinds, groups)]
    expressive = [i for i, b in enumerate(bands) if b == "expressive"]
    if not expressive:
        return {"name": name, "ok": False, "reason": "no expressive params"}

    span = np.maximum(hi[expressive] - lo[expressive], 1e-6)
    default_n = np.clip((dflt[expressive] - lo[expressive]) / span * 2 - 1, -1, 1)
    poses = sample_poses(clips_npz, default_n, n, rng)

    imgs = np.zeros((n, size, size, 4), np.uint8)
    try:
        # strip_dynamics=True: physics and pose carry history, so leaving them
        # on would give the same parameter vector different images.
        with Live2DBackend(path, size=size, strip_dynamics=True) as be:
            be.calibrate()
            by_id = {s.pid: s for s in be.params}
            for k in range(n):
                vals = lo[expressive] + (poses[k] + 1) * 0.5 * span
                over = {pids[i]: float(vals[c])
                        for c, i in enumerate(expressive) if pids[i] in by_id}
                imgs[k] = be.render(over)
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "ok": False, "reason": f"{type(exc).__name__}: {exc}"}

    dest = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(
        dest,
        images=imgs,
        poses=poses.astype(np.float16),
        pid=np.array([pids[i] for i in expressive], dtype=object),
        param_row=np.asarray(expressive, np.int32),
        minimum=lo[expressive], maximum=hi[expressive])
    return {"name": name, "ok": True, "n": n, "P": len(expressive),
            "mb": os.path.getsize(dest) / 1e6}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--models", action="append", default=[])
    ap.add_argument("--rigs", default="rigs")
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--out", default="synth")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 3))
    ap.add_argument("--limit-rigs", type=int, default=0)
    a = ap.parse_args(argv)
    # Workers run in this process's children, so there is no external wrapper
    # to supply a display; re-exec the whole command under one instead.
    ensure_display(["-m", f"{__package__ or 'l2dprobe'}.render_dataset"]
                   + (argv if argv is not None else sys.argv[1:]))

    paths = []
    if a.manifest:
        with open(a.manifest, encoding="utf-8") as fh:
            paths += [r["path"] for r in json.load(fh)["usable"]]
    for pattern in a.models:
        paths += [p for p in sorted(glob.glob(pattern, recursive=True))
                  if p.endswith(".model3.json") and ".probe." not in p]
    paths = list(dict.fromkeys(paths))

    work = []
    for i, p in enumerate(paths):
        name = os.path.basename(p).split(".")[0]
        rig_npz = os.path.join(a.rigs, f"{name}.npz")
        if not os.path.exists(rig_npz):
            continue
        if os.path.exists(os.path.join(a.out, f"{name}.npz")):
            continue                      # resumable
        clips = os.path.join(a.clips, f"{name}.npz")
        work.append((p, rig_npz, clips if os.path.exists(clips) else None,
                     a.out, a.samples, a.size, i))
    if a.limit_rigs:
        work = work[:a.limit_rigs]

    os.makedirs(a.out, exist_ok=True)
    est = a.samples * a.size * a.size * 4 / 1e9
    print(f"{len(work)} rigs x {a.samples} samples @ {a.size}px "
          f"(~{est:.1f} GB raw per rig before compression)")
    if not work:
        print("nothing to do (all rigs already generated?)")
        return 0

    ok = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(build_rig, w) for w in work]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            ok, fail = ok + bool(r["ok"]), fail + (not r["ok"])
            eta = (time.time() - t0) / i * (len(work) - i) / 60
            if r["ok"]:
                print(f"[{i:>3}/{len(work)}] ok   {r['name'][:24]:<24} "
                      f"{r['n']} imgs  {r['P']:>3} params  {r['mb']:.0f} MB"
                      f"   eta {eta:.0f}m")
            else:
                print(f"[{i:>3}/{len(work)}] FAIL {r['name'][:24]:<24} "
                      f"{r['reason'][:50]}")

    print(f"\n{ok} rigs generated, {fail} failed, "
          f"{(time.time() - t0) / 60:.1f} min")
    print("Rerun to resume; existing outputs are skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
