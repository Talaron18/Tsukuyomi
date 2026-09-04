#!/usr/bin/env python3
r"""
Block 1 entry point: probe a Live2D rig and emit grounded parameter descriptors.

    python run_probe.py path\to\model.model3.json --smoke

Windows needs no virtual display; a hidden GLFW window uses the real driver.
On a headless Linux box, prefix with:
    xvfb-run -s "-screen 0 1024x1024x24"

Start with --smoke on one rig. It prints which live2d-py API names resolved,
whether renders are deterministic, and how many parameters actually move.
If any of those look wrong, stop and fix before probing the whole sample set.
"""

from __future__ import annotations

import argparse
import os

# PyOpenGL binds its GL library at import time, so the OSMesa selection has to
# happen before anything pulls in OpenGL.GL. argparse has not run yet, so we
# peek at sys.argv directly.
if "--osmesa" in __import__("sys").argv:
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import glob
import sys

try:
    from .backend import IncompatibleRig, Live2DBackend
    from .probe import probe_rig, save
except ImportError:  # running as loose scripts rather than a package
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend import IncompatibleRig, Live2DBackend
    from probe import probe_rig, save


def smoke(model_json: str, size: int, osmesa: bool = False) -> int:
    with Live2DBackend(model_json, size=size, osmesa=osmesa) as be:
        print(f"model      : {model_json}")
        print(f"api        : {be._api_report}")
        print(f"parameters : {len(be.params)}")
        for s in be.params[:8]:
            print(f"   {s.pid:<28} [{s.minimum:>7.2f}, {s.maximum:>7.2f}] "
                  f"default {s.default:>7.2f}")
        if len(be.params) > 8:
            print(f"   ... and {len(be.params) - 8} more")

        print("calibrating parameter write order:")
        order = be.calibrate(verbose=True)
        print(f"order      : {order}")

        ok, worst = be.check_determinism()
        print(f"determinism: {'OK' if ok else 'FAIL'} (max pixel delta {worst})")

        leak = be.check_state_leak()
        print(f"state leak  : param={leak['param']} "
              f"effect={leak['effect_size']:.4f} "
              f"repeat_drift={leak['repeat_drift']:.4f} "
              f"neutral_drift={leak['neutral_drift']:.4f} "
              f"-> {'LAGGED' if leak['lagged'] else 'clean'}")
        if leak["lagged"]:
            print("  !! parameter writes are landing one frame late. Every "
                  "descriptor would describe the previously probed parameter.")

        alive = be.check_responsiveness()
        print(f"responsive : {len(alive)}/{len(be.params)} parameters move")
        if not alive:
            print("  !! nothing moved. SetParameterValue is not surviving "
                  "model.Update(). Try setting parameters after Update() and "
                  "before Draw().")
            return 2
        if not ok:
            print("  !! non-deterministic. Physics or an idle motion is still "
                  "running; descriptors would be noise.")
            return 3
    return 0


def find_models(root: str) -> list[str]:
    """Recursively locate every .model3.json under `root`.

    Skips two things that are not rigs and would otherwise be probed as if they
    were: the stripped copies sanitize.py writes next to each model, and the
    probe's own output files, which share the .model3.json suffix and live in
    out/. The latter is where the duplicate rigs in the descriptor export came
    from.
    """
    hits = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {"out", "clips", "rigs", "data"}]
        for f in files:
            if not f.endswith(".model3.json"):
                continue
            if ".probe." in f or f.startswith("rig_"):
                continue
            hits.append(os.path.join(dirpath, f))
    return sorted(hits)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?",
                    help=".model3.json path, or a glob with --batch")
    ap.add_argument("--find", metavar="DIR",
                    help="recursively list .model3.json files and exit")
    ap.add_argument("--out", default="out/rig")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="diagnostics only")
    ap.add_argument("--batch", action="store_true", help="treat model as a glob")
    ap.add_argument("--osmesa", action="store_true",
                    help="software rendering, no X server needed at all")
    ap.add_argument("--loose", action="store_true",
                    help="probe even if renders are non-deterministic")
    a = ap.parse_args(argv)

    if a.find:
        found = find_models(a.find)
        if not found:
            print(f"no .model3.json under {a.find}", file=sys.stderr)
            return 1
        for path in found:
            print(path)
        print(f"\n{len(found)} model(s). Probe all of them with:\n"
              f"  python -m l2dprobe.run_probe '{a.find}/**/*.model3.json' "
              f"--batch --out out/rig")
        return 0

    if not a.model:
        ap.error("a model path is required unless --find is given")

    targets = (sorted(glob.glob(a.model, recursive=True))
               if a.batch else [a.model])
    if not targets:
        print("no models matched", file=sys.stderr)
        return 1

    if a.smoke:
        return smoke(targets[0], a.size, osmesa=a.osmesa)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    skipped: list[str] = []
    for path in targets:
        name = os.path.splitext(os.path.basename(path))[0]
        prefix = a.out if len(targets) == 1 else f"{a.out}_{name}"
        try:
            descs, meta = probe_rig(path, size=a.size, steps=a.steps,
                                    strict=not a.loose, osmesa=a.osmesa)
        except IncompatibleRig as exc:
            print(f"{name}: SKIPPED -- {exc}")
            skipped.append(name)
            continue
        except Exception as exc:  # noqa: BLE001
            # One bad rig should not cost the whole batch.
            print(f"{name}: FAILED -- {type(exc).__name__}: {exc}")
            skipped.append(name)
            continue
        save(descs, meta, prefix)
        print(f"{name}: {meta['n_params']} params -> "
              f"{meta['n_continuous']} continuous, "
              f"{meta['n_categorical']} categorical, "
              f"{meta['n_dead']} dead, "
              f"{meta['n_mirrored']} mirrored  ->  {prefix}.json")

    if skipped:
        print(f"\nskipped {len(skipped)}: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
