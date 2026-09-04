#!/usr/bin/env python3
"""
Answer two questions that the calibration failure left open.

  1. Is anything being drawn at all?
     calibrate() only compares renders to each other, so an all-blank canvas
     scores zero exactly like an unresponsive parameter. We dump the base
     render to PNG and report pixel statistics so the two cases separate.

  2. What is the real LAppModel API?
     Our theory is that LAppModel::Update() does LoadParameters() first (wiping
     writes made before it) and csmUpdateModel() last (so writes made after it
     never reach the mesh). If true, no ordering around Update() can work and we
     need a different entry point. The full member list settles it.

Usage:
    xvfb-run -s "-screen 0 1024x1024x24" \
        python -m l2dprobe.inspect_tool <model.model3.json> --dump dbg/
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

import numpy as np

from .backend import Live2DBackend


def describe(obj, label: str) -> None:
    print(f"\n=== {label}: {type(obj).__name__} ===")
    for name in sorted(n for n in dir(obj) if not n.startswith("_")):
        try:
            attr = getattr(obj, name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<32} <raised {type(exc).__name__}>")
            continue
        if callable(attr):
            try:
                sig = str(inspect.signature(attr))
            except (TypeError, ValueError):
                sig = "(?)"
            doc = (inspect.getdoc(attr) or "").strip().split("\n")[0][:60]
            print(f"  {name}{sig}".ljust(56) + (f"  # {doc}" if doc else ""))
        else:
            print(f"  {name:<32} = {attr!r}"[:110])


def stats(img: np.ndarray, label: str) -> dict:
    alpha = img[..., 3]
    rgb = img[..., :3]
    d = {
        "nonzero_alpha_frac": float((alpha > 0).mean()),
        "alpha_max": int(alpha.max()),
        "rgb_mean": float(rgb.mean()),
        "rgb_max": int(rgb.max()),
        "unique_colors": int(len(np.unique(rgb.reshape(-1, 3), axis=0))),
    }
    print(f"  {label:<22} " + "  ".join(f"{k}={v}" for k, v in d.items()))
    return d


def save(img: np.ndarray, path: str) -> None:
    try:
        from PIL import Image
    except ImportError:
        np.save(path.replace(".png", ".npy"), img)
        return
    Image.fromarray(img, "RGBA").save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dump", default="dbg", help="directory for PNG dumps")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no-strip", action="store_true",
                    help="load the original model3.json, physics and all")
    ap.add_argument("--gl", default=None, metavar="MAJOR.MINOR",
                    help="request a specific GL version (default: driver's choice)")
    ap.add_argument("--renderer", action="store_true",
                    help="call CreateRenderer() explicitly after load")
    a = ap.parse_args(argv)

    os.makedirs(a.dump, exist_ok=True)

    gl_version = None
    if a.gl:
        major, minor = a.gl.split(".")
        gl_version = (int(major), int(minor))

    be = Live2DBackend(a.model, size=a.size, strip_dynamics=not a.no_strip,
                       gl_version=gl_version).open()
    try:
        print("\n=== GL context we actually got ===")
        for k, v in be.gl_info().items():
            print(f"  {k:<10} {v}")

        # Plumbing check before anything Live2D-specific. If red does not come
        # back, the FBO or readback is broken and the model is irrelevant.
        ok, px = be.canary()
        print(f"\n=== canary (clear to red, read back) ===\n"
              f"  centre pixel RGBA = {px}  -> {'OK' if ok else 'FAILED'}")
        if not ok:
            print("  Readback is broken. Everything below is meaningless.")

        if a.renderer:
            print("\n=== calling CreateRenderer() explicitly ===")
            be.model.CreateRenderer()
            be.model.Resize(be.size, be.size)

        # Each of these is a diagnostic, not a requirement. Any one of them
        # may be unimplemented or may throw in this build, and none of them is
        # worth losing the rest of the report over.
        print("\n=== model state ===")
        for label, fn in (
            ("canvas size", lambda: be.model.GetCanvasSize()),
            ("canvas size (px)", lambda: be.model.GetCanvasSizePixel()),
            ("pixels per unit", lambda: be.model.GetPixelsPerUnit()),
            ("drawables", lambda: len(be.model.GetDrawableIds())),
            ("parts", lambda: be.model.GetPartCount()),
            ("param ids", lambda: len(be.model.GetParamIds())),
            ("ParamAngleX value", lambda: be.model.GetParameterValue(0)),
        ):
            try:
                print(f"  {label:<18} {fn()}")
            except BaseException as exc:  # noqa: BLE001
                print(f"  {label:<18} <{type(exc).__name__}: {exc}>")

        describe(be.model, "LAppModel")
        if be.params:
            raw = be.model.GetParameter(0)
            describe(raw, "Parameter[0]")

        print("\n=== is anything rendering? ===")
        base = be.render()
        stats(base, "base(fbo)")
        save(base, os.path.join(a.dump, "base_fbo.png"))

        # Same draw, read from the default framebuffer instead. If this one has
        # content and the FBO one does not, live2d-py is rebinding framebuffer 0
        # during Draw and our offscreen target is never touched.
        try:
            from OpenGL.GL import GL_FRAMEBUFFER, glBindFramebuffer
            import live2d.v3 as live2d
            from OpenGL.GL import (GL_RGBA, GL_UNSIGNED_BYTE, glReadPixels,
                                   glViewport)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, be.size, be.size)
            live2d.clearBuffer(*be.bg)
            be.model.Update()
            be.model.Draw()
            buf = glReadPixels(0, 0, be.size, be.size, GL_RGBA, GL_UNSIGNED_BYTE)
            arr = (buf.astype(np.uint8, copy=False) if isinstance(buf, np.ndarray)
                   else np.frombuffer(buf, dtype=np.uint8))
            win = np.flipud(arr.reshape(be.size, be.size, 4)).copy()
            stats(win, "base(default fb)")
            save(win, os.path.join(a.dump, "base_default_fb.png"))
        except Exception as exc:  # noqa: BLE001
            print(f"  default-framebuffer read failed: {exc!r}")

        print("\n=== does one parameter move anything? ===")
        target = next((s for s in be.params if s.pid == "ParamAngleX"), be.params[0])
        print(f"  probing {target.pid} over [{target.minimum}, {target.maximum}]")
        for order in be.ORDERS:
            img = be.render({target.pid: target.maximum}, order=order)
            delta = float(np.abs(img.astype(np.int16)
                                 - base.astype(np.int16)).mean())
            print(f"  order={order:<9} mean|delta|={delta:.4f}")
            save(img, os.path.join(a.dump, f"{target.pid}_max_{order}.png"))
    finally:
        be.close()

    print(f"\nPNGs written to {a.dump}/ -- open base_fbo.png first.")
    print("If it is empty, the problem is rendering, not parameters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
