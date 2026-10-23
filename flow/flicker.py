#!/usr/bin/env python3
"""
Measure frame-to-frame flicker in rendered output, and find its cause.

On the server, preview videos flickered: xuefeng switched outfits on
alternate frames, aidang_2 arm variants -- in authored replay and in the
behaviour baseline alike, though the baseline writes the same part opacities
every frame. Locally (live2d-py 0.8.1) the same replays are smooth, so the
cause lives in the server's build, write order or rigs.

Two modes:

  frames DIR      count jumps and flip-flops (frame f differs from f+1 but
                  matches f+2) in a runtime --out directory of f*.png
  still MODEL     render one fixed input repeatedly under each write order,
                  with and without part opacities and with motions stopped
                  every frame; any change between frames is the renderer
                  fighting itself, since nothing we write changes

Usage:
    python -m flow.flicker frames vids/xuefeng/beh
    xvfb-run -a python -m flow.flicker still path/to/xuefeng.model3.json
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def _gray(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), np.float32)


def flicker_stats(frames: np.ndarray) -> dict:
    """Jumps and flip-flops in a (T, H, W) stack."""
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    med = float(np.median(d)) if len(d) else 0.0
    jump = d > 4 * med + 0.5
    two = np.abs(frames[2:] - frames[:-2]).mean(axis=(1, 2))
    flip = jump[:-1] & (two < 0.5 * d[:-1])
    return {"frames": int(len(frames)), "median_delta": med,
            "jumps": int(jump.sum()), "flip_flops": int(flip.sum()),
            "first": np.flatnonzero(flip)[:10].tolist()}


def cmd_frames(a) -> int:
    paths = sorted(glob.glob(os.path.join(a.dir, "f*.png")))
    if len(paths) < 3:
        print(f"no f*.png frames in {a.dir}", file=sys.stderr)
        return 1
    s = flicker_stats(np.stack([_gray(p) for p in paths]))
    print(f"{a.dir}: {s['frames']} frames  median change "
          f"{s['median_delta']:.3f}  jumps {s['jumps']}  flip-flops "
          f"{s['flip_flops']} {s['first']}")
    return 0


def cmd_still(a) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from l2dprobe.backend import Live2DBackend, ensure_display
    from l2dprobe.motions import motion_files
    from flow.runtime import authored_motions, base_state
    ensure_display(["-m", "flow.flicker"] + sys.argv[1:])

    motions = authored_motions(a.model, 30.0)
    _, _, parts = base_state(motions, lambda p: False)
    print(f"{len(motion_files(a.model))} motion files, {len(parts)} parts "
          "with authored opacity")
    with Live2DBackend(a.model, size=256, strip_dynamics=False,
                       fps=30) as be:
        be.calibrate()
        rep = be._api_report
        print(f"backend: calibrated order={rep.get('order')} "
              f"timestep={rep.get('timestep')} parts={rep.get('parts')}")
        try:
            import live2d
            print(f"live2d-py {getattr(live2d, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            pass
        for order in ("before", "sandwich"):
            for with_parts in (False, True):
                for stop in (False, True):
                    imgs = []
                    for _ in range(90):
                        if stop:
                            fn = getattr(be.model, "StopAllMotions", None)
                            if fn:
                                fn()
                        imgs.append(be.render(
                            {}, order=order,
                            parts=parts if with_parts else None
                        ).astype(np.float32).mean(axis=2))
                    s = flicker_stats(np.stack(imgs[30:]))
                    fin = getattr(be.model, "IsMotionFinished", None)
                    print(f"  order={order:<8} parts={'on ' if with_parts else 'off'} "
                          f"stop-motions={'on ' if stop else 'off'}  "
                          f"change {s['median_delta']:.3f}  jumps "
                          f"{s['jumps']:>2}  flip-flops {s['flip_flops']:>2}"
                          f"  motion finished={fin() if fin else '?'}")
    print("\nNothing changes between these frames, so any change or "
          "flip-flop is the renderer, not the motion.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("dir")
    s = sub.add_parser("still")
    s.add_argument("model")
    a = ap.parse_args(argv)
    return cmd_frames(a) if a.cmd == "frames" else cmd_still(a)


if __name__ == "__main__":
    raise SystemExit(main())
