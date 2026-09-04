#!/usr/bin/env python3
"""
Build the training set for the learned parameter encoder.

probe.py keeps only two 16x16 flow maps per parameter, which is where
laterality was lost: on a full-body rig the two eyes land 2-3 cells apart and
the blur used to make near-misses match erased the difference. This dumps the
flow field at higher resolution across the whole sweep so a learned encoder can
pool it adaptively instead of through hand-picked global statistics.

Per rig we store, for every live parameter and every sweep value:

    flow    (P, K, R, R, 2)  displacement relative to the default pose
    alpha   (R, R)           character silhouette, shared across parameters
    values  (P, K)           the parameter value at each sweep step
    tnorm   (P, K)           that value mapped to [0, 1] across the range

Flow rather than raw pixels on purpose. It is appearance-invariant -- moving
right is +dx whether the hair is dark or light -- and switching to it produced
the largest single gain in the hand-designed descriptor. What failed there was
the pooling, not the representation, so the encoder keeps flow as input and
replaces only the pooling.

Usage:
    xvfb-run -s "-screen 0 1024x1024x24" \
        python -m l2dprobe.dataset './CubismWebSamples/**/*.model3.json' \
            --out data/ --res 64 --steps 7
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

try:
    from .backend import IncompatibleRig, Live2DBackend
    from .probe import _flow_field, _to_gray, content_bbox
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from yachiyo.l2dprobe.backend import IncompatibleRig, Live2DBackend
    from probe import _flow_field, _to_gray, content_bbox


def alpha_map(base: np.ndarray, bbox, res: int) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop = (base[y0:y1, x0:x1, 3].astype(np.float32) / 255.0)
    h, w = crop.shape
    if h == 0 or w == 0:
        return np.zeros((res, res), np.float32)
    yi = np.clip((np.arange(res) + 0.5) * h / res, 0, h - 1).astype(int)
    xi = np.clip((np.arange(res) + 0.5) * w / res, 0, w - 1).astype(int)
    return crop[np.ix_(yi, xi)].astype(np.float32)


def build_rig(path: str, res: int, steps: int, size: int,
              min_travel: float = 0.05) -> dict | None:
    be = Live2DBackend(path, size=size).open()
    try:
        be.calibrate()
        base = be.render()
        bbox = content_bbox(base)

        flows, values, tnorms, pids, kinds = [], [], [], [], []
        for spec in be.params:
            if spec.span <= 0:
                continue
            ts = np.linspace(0.0, 1.0, steps)
            per_val, per_flow = [], []
            for t in ts:
                v = spec.sample(float(t))
                # Skip sweep points that sit essentially at the default: their
                # flow is numerical noise, and normalising noise is what
                # poisoned the eyelid descriptors last time.
                if abs(v - spec.default) / (spec.span + 1e-9) < min_travel:
                    per_flow.append(np.zeros((res, res, 2), np.float32))
                else:
                    img = be.render({spec.pid: v})
                    per_flow.append(_flow_field(base, img, bbox, grid=res))
                per_val.append(v)

            stack = np.stack(per_flow)
            if float(np.abs(stack).max()) <= 0:
                continue  # parameter does nothing visible from the default pose

            flows.append(stack.astype(np.float16))
            values.append(np.asarray(per_val, np.float32))
            tnorms.append(ts.astype(np.float32))
            pids.append(spec.pid)
            kinds.append("continuous")

        if not flows:
            return None
        return {
            "flow": np.stack(flows),
            "values": np.stack(values),
            "tnorm": np.stack(tnorms),
            "alpha": alpha_map(base, bbox, res),
            "pids": np.array(pids, dtype=object),
            "bbox": np.asarray(bbox, np.int32),
            "rig": os.path.basename(path).split(".")[0],
        }
    finally:
        be.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default="data")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args(argv)

    paths = []
    for pattern in a.models:
        paths.extend(sorted(glob.glob(pattern, recursive=True)) or [pattern])
    paths = [p for p in paths
             if p.endswith(".model3.json") and ".probe." not in p]
    if not paths:
        print("no models found", file=sys.stderr)
        return 1

    os.makedirs(a.out, exist_ok=True)
    total = 0
    for path in paths:
        name = os.path.basename(path).split(".")[0]
        try:
            pack = build_rig(path, a.res, a.steps, a.size)
        except IncompatibleRig as exc:
            print(f"{name}: SKIPPED -- {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: FAILED -- {type(exc).__name__}: {exc}")
            continue
        if pack is None:
            print(f"{name}: no usable parameters")
            continue

        dest = os.path.join(a.out, f"{name}.npz")
        np.savez_compressed(dest, **pack)
        p, k = pack["flow"].shape[:2]
        mb = os.path.getsize(dest) / 1e6
        print(f"{name}: {p} params x {k} steps @ {a.res}x{a.res}  "
              f"-> {dest} ({mb:.1f} MB)")
        total += p

    print(f"\n{total} parameters across {len(paths)} rigs")
    print("Sanity check before training: parameters should be roughly the")
    print("'continuous' counts from the probe run, and flow magnitude should")
    print("be clearly nonzero for head parameters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
