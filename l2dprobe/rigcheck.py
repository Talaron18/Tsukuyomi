#!/usr/bin/env python3
"""
Tell apart the reasons a rig draws several arm variants at once.

On aidang_2 the runtime's authored replay (--static) showed two or three arms
fully opaque in the same frame. Three different causes produce that picture,
and each needs a different fix:

  rig/build   Live2D's own motion player shows it too. Nothing we drop is to
              blame; the rig or this live2d-py / Cubism Core build is.
  dropped     The native player is clean but our replay is not. Our replay
              writes only expressive parameters: discrete parameters stay at
              their default and PartOpacity curves are never read, so whatever
              selects the arm pose is lost.
  default     The rig's default pose already shows every variant, and authored
              motion is what normally hides the spares.

This renders the native player and the bare default pose, and prints the facts
that decide between them: which dynamic files the rig has, which parts the
motions animate through PartOpacity, and whether the discrete parameters ever
leave their default in authored motion.

Usage:
    xvfb-run -a python -m l2dprobe.rigcheck path/to/model.model3.json \\
        --rig rigs/<name>.npz --out rigcheck_<name>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    from .backend import Live2DBackend, ensure_display
    from .motions import (band_of, eval_segments, motion_files,
                          physics_params, toggle_params)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from l2dprobe.backend import Live2DBackend, ensure_display
    from l2dprobe.motions import (band_of, eval_segments, motion_files,
                                  physics_params, toggle_params)


def authored_curves(model_json: str, fps: float):
    """Every motion curve, grouped by target, sampled at fps."""
    out = {"Parameter": defaultdict(list), "PartOpacity": defaultdict(list),
           "Model": defaultdict(list)}
    files = motion_files(model_json)
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        dur = float(blob.get("Meta", {}).get("Duration", 0) or 0)
        if dur <= 0.05:
            continue
        times = np.arange(0.0, dur, 1.0 / fps, dtype=np.float32)
        for c in blob.get("Curves", []):
            target = c.get("Target")
            if target not in out or not c.get("Id"):
                continue
            try:
                out[target][c["Id"]].append(eval_segments(c["Segments"], times))
            except Exception:  # noqa: BLE001
                continue
    return out, files


def report_files(model_json: str):
    with open(model_json, encoding="utf-8") as fh:
        refs = json.load(fh).get("FileReferences", {})
    print("FileReferences:", ", ".join(sorted(refs)))
    for key in ("Physics", "Pose"):
        print(f"  {key:<8} {'present: ' + str(refs[key]) if refs.get(key) else 'ABSENT'}")
    pose = refs.get("Pose")
    if pose:
        try:
            with open(os.path.join(os.path.dirname(model_json), pose),
                      encoding="utf-8") as fh:
                groups = json.load(fh).get("Groups", [])
            for g in groups:
                print("  pose group: " + " | ".join(e.get("Id", "?") for e in g))
        except Exception as exc:  # noqa: BLE001
            print(f"  pose file unreadable: {exc}")
    phys, has = physics_params(model_json)
    print(f"  physics outputs: {len(phys)}" + ("" if has else "  (no physics file)"))


def report_motion(curves, files, rig_npz, model_json):
    print(f"\nmotion files: {len(files)}")
    parts = curves["PartOpacity"]
    print(f"PartOpacity curves: {sum(len(v) for v in parts.values())} "
          f"across {len(parts)} parts")
    for pid in sorted(parts):
        v = np.concatenate(parts[pid])
        uniq = np.unique(np.round(v, 2))
        frac_mid = float(((v > 0.05) & (v < 0.95)).mean())
        print(f"  {pid:<32} values {uniq[:6].tolist()}"
              f"{' ...' if len(uniq) > 6 else ''}  in-between {frac_mid:.2f}")
    if curves["Model"]:
        print(f"Model-target curves: {sorted(curves['Model'])}")

    if not rig_npz or not os.path.exists(rig_npz):
        print("\n(no --rig given: skipping band and discrete-parameter report)")
        return
    z = np.load(rig_npz, allow_pickle=True)
    pids = [str(x) for x in z["pid"]]
    kinds = [str(x) for x in z["kind"]]
    groups = [str(x) for x in z["group"]]
    lo, hi, dflt = z["minimum"], z["maximum"], z["default"]
    phys, _ = physics_params(model_json)
    bands = [band_of(p, k, g, phys) for p, k, g in zip(pids, kinds, groups)]
    print("\nbands: " + ", ".join(f"{b} {n}" for b, n in Counter(bands).items()))

    params = curves["Parameter"]
    print("discrete parameters (held at default by the runtime):")
    for i, b in enumerate(bands):
        if b != "discrete":
            continue
        line = (f"  {pids[i]:<32} default {float(dflt[i]):g} "
                f"range [{float(lo[i]):g}, {float(hi[i]):g}]")
        if pids[i] in params:
            v = np.concatenate(params[pids[i]])
            off = float((np.abs(v - float(dflt[i])) > 1e-3).mean())
            vals = np.unique(np.round(v, 2))
            line += (f"  authored off-default {off:.2f}, values "
                     f"{vals[:6].tolist()}{' ...' if len(vals) > 6 else ''}")
        else:
            line += "  never animated"
        print(line)

    tog = toggle_params(model_json)
    expressive = [pids[i] for i, b in enumerate(bands) if b == "expressive"]
    snapped = sorted(p for p in expressive if p in tog)
    print(f"\nwould snap {len(snapped)} in model mode: {', '.join(snapped)}")
    missing = sorted(set(params) - set(pids))
    if missing:
        print(f"animated but not in the rig file ({len(missing)}): "
              + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))


def save_png(img, path):
    from PIL import Image
    Image.fromarray(img, "RGBA").save(path)


def encode(frames_dir, fps, dest):
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                        str(fps), "-i", os.path.join(frames_dir, "f%05d.png"),
                        "-vf", "format=yuv420p", "-c:v", "libx264", dest],
                       check=True, timeout=300)
        print(f"video -> {dest}")
    except Exception as exc:  # noqa: BLE001
        print(f"ffmpeg failed ({exc}); PNGs are in {frames_dir}/")


def render(model_json, out, fps, size, seconds, group_pick):
    os.makedirs(out, exist_ok=True)
    with Live2DBackend(model_json, size=size, strip_dynamics=False,
                       fps=fps) as be:
        be.calibrate()
        print("\nbackend: " + ", ".join(
            f"{k}={be._api_report.get(k)}" for k in ("order", "timestep")))
        try:
            import live2d
            print(f"live2d-py {getattr(live2d, '__version__', '?')}")
        except Exception:  # noqa: BLE001
            pass

        # Default pose with dynamics on, nothing written: what the rig looks
        # like before any motion picks an arm.
        for _ in range(int(fps)):
            img = be.render()
        save_png(img, os.path.join(out, "neutral.png"))
        print(f"neutral pose -> {out}/neutral.png")

        try:
            groups = be.model.GetMotionGroups()
        except Exception as exc:  # noqa: BLE001
            print(f"GetMotionGroups failed: {exc!r}; no native playback")
            return
        print(f"motion groups: {groups}")
        if not groups:
            print("no motion groups registered in the model json; "
                  "no native playback")
            return
        names = list(groups)
        group = (group_pick if group_pick in groups else
                 next((g for g in names if "idle" in g.lower()), names[0]))

        frames_dir = os.path.join(out, "native")
        os.makedirs(frames_dir, exist_ok=True)
        n_total = int(seconds * fps)
        idx, f = 0, 0
        count = int(groups[group]) if str(groups[group]).isdigit() else 1
        while f < n_total:
            try:
                be.model.StartMotion(group, idx % max(count, 1), 3)
            except Exception as exc:  # noqa: BLE001
                import inspect
                sig = "?"
                try:
                    sig = str(inspect.signature(be.model.StartMotion))
                except Exception:  # noqa: BLE001
                    pass
                print(f"StartMotion({group!r}, {idx}, 3) failed: {exc!r}; "
                      f"signature {sig}")
                break
            print(f"native: playing {group}[{idx % max(count, 1)}]")
            started = f
            while f < n_total:
                save_png(be.render_native(),
                         os.path.join(frames_dir, f"f{f:05d}.png"))
                f += 1
                done = getattr(be.model, "IsMotionFinished", None)
                if done is not None and f - started > 2 and done():
                    break
            idx += 1
        if f:
            encode(frames_dir, fps, os.path.join(out, "native.mp4"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--rig", default=None, help="rigs/<name>.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--group", default=None,
                    help="motion group to play natively (default: an idle "
                         "group if there is one, else the first)")
    a = ap.parse_args(argv)
    ensure_display(["-m", "l2dprobe.rigcheck"]
                   + (argv if argv is not None else sys.argv[1:]))

    if not os.path.isfile(a.model):
        print(f"not a file: {a.model!r} (quote paths with spaces or brackets)",
              file=sys.stderr)
        return 1
    name = os.path.basename(a.model).split(".")[0]
    out = a.out or f"rigcheck_{name}"

    report_files(a.model)
    curves, files = authored_curves(a.model, a.fps)
    report_motion(curves, files, a.rig, a.model)
    render(a.model, out, a.fps, a.size, a.seconds, a.group)
    print("\nCompare native.mp4 with preview_<name>_static_nosnap/preview.mp4:"
          "\n  native also shows extra arms  -> rig or live2d-py build"
          "\n  native clean, ours not        -> our replay drops the switch"
          "\n  neutral.png shows every arm   -> the default pose is the cause")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
