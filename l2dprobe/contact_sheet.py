#!/usr/bin/env python3
"""
Render a contact sheet so the probe images can be looked at directly.

Every descriptor is derived from these frames. If a rig's base render is tiny
in canvas, cropped, or blank, then its support maps, flow fields and mirror
detection are all noise, and no amount of reweighting in analyze.py will fix
it. Numbers cannot distinguish "bad descriptor" from "bad pixels"; eyes can.

Each row is one rig: the base pose, its alpha bounding box, then a few standard
parameters driven to an extreme. Compare rows. A rig that scores badly against
every other rig should look visibly different here.

Usage:
    xvfb-run -s "-screen 0 1024x1024x24" \
        python -m l2dprobe.contact_sheet './CubismWebSamples/**/*.model3.json' \
            --out dbg/sheet.png
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

try:
    from .backend import IncompatibleRig, Live2DBackend
    from .probe import content_bbox
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from yachiyo.l2dprobe.backend import IncompatibleRig, Live2DBackend
    from probe import content_bbox

# Parameters worth eyeballing: one clear translation, one rotation, one small
# local deformation, one gaze shift.
COLUMNS = ("ParamAngleX", "ParamAngleZ", "ParamEyeLOpen", "ParamEyeBallX")


def composite(img: np.ndarray) -> np.ndarray:
    """RGBA over a checkerboard, so transparent and black are distinguishable."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    check = np.where(((yy // 16) + (xx // 16)) % 2 == 0, 210, 170).astype(np.float32)
    a = img[..., 3:4].astype(np.float32) / 255.0
    return (img[..., :3].astype(np.float32) * a
            + check[..., None] * (1 - a)).astype(np.uint8)


def outline(img: np.ndarray, bbox) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    out = img.copy()
    x1 = min(x1, out.shape[1] - 1)
    y1 = min(y1, out.shape[0] - 1)
    for c, v in enumerate((255, 0, 0)):
        out[y0, x0:x1, c] = v
        out[y1, x0:x1, c] = v
        out[y0:y1, x0, c] = v
        out[y0:y1, x1, c] = v
    return out


def rig_row(path: str, size: int) -> tuple[str, np.ndarray, str] | None:
    try:
        be = Live2DBackend(path, size=size).open()
    except IncompatibleRig as exc:
        print(f"  skip {os.path.basename(path)}: {exc}")
        return None
    try:
        be.calibrate()
        base = be.render()
        bbox = content_bbox(base)
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        note = (f"{os.path.basename(path).split('.')[0]}  "
                f"bbox {bw}x{bh} = {bw * bh / (size * size):.1%} of canvas")
        print(f"  {note}")

        tiles = [outline(composite(base), bbox)]
        by_id = {s.pid: s for s in be.params}
        for pid in COLUMNS:
            spec = by_id.get(pid)
            if spec is None:
                tiles.append(np.full((size, size, 3), 60, np.uint8))
                continue
            # Drive AWAY from default, not blindly to max. ParamEyeLOpen
            # defaults to 1.0 in a [0, 1.2] range, so max is nearly a no-op
            # and min is where the eye actually closes.
            far = (spec.maximum
                   if abs(spec.maximum - spec.default) >= abs(spec.default - spec.minimum)
                   else spec.minimum)
            img = be.render({pid: far})
            d = np.abs(img.astype(np.int16) - base.astype(np.int16))
            print(f"      {pid:<18} mean|delta|={d.mean():7.4f} "
                  f"max={d.max():3.0f}")
            # Amplified difference. At full-body scale a head turn is a few
            # percent of the pixels and invisible side by side, so show what
            # actually changed rather than relying on spotting it.
            amp = np.clip(d[..., :3].mean(axis=2) * 8, 0, 255).astype(np.uint8)
            tiles.append(np.stack([amp, amp // 3, 255 - amp], axis=2))
        return os.path.basename(path), np.concatenate(tiles, axis=1), note
    finally:
        be.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default="dbg/sheet.png")
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args(argv)

    paths = []
    for pattern in a.models:
        paths.extend(sorted(glob.glob(pattern, recursive=True)) or [pattern])
    paths = [p for p in paths if p.endswith(".model3.json")]
    if not paths:
        print("no models found", file=sys.stderr)
        return 1

    print(f"columns: base(+bbox), then AMPLIFIED DIFF (x8) for "
          + ", ".join(COLUMNS))
    rows = []
    for path in paths:
        try:
            row = rig_row(path, a.size)
        except Exception as exc:  # noqa: BLE001
            print(f"  {os.path.basename(path)}: FAILED {type(exc).__name__}: {exc}")
            continue
        if row:
            rows.append(row[1])

    if not rows:
        print("nothing rendered", file=sys.stderr)
        return 1

    sheet = np.concatenate(rows, axis=0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(sheet).save(a.out)
    except ImportError:
        np.save(a.out.replace(".png", ".npy"), sheet)
    print(f"\nwrote {a.out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    print("Rows are rigs, columns are parameters. A rig that scores badly")
    print("against all others should look visibly wrong in its row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
