#!/usr/bin/env python3
"""
Run probe + dataset across many rigs in parallel, resumably.

Three things make this necessary at 34+ rigs rather than 7:

  isolation   The Cubism Core segfaults on some models. In-process that kills
              everything; six of the Azur Lane rigs do it. One subprocess per
              rig turns a fatal crash into one skipped row.

  resumability ~38,000 renders is hours of work. Anything that cannot resume
              will eventually lose a long run to one bad model or a stray
              Ctrl-C. Completed rigs are detected by their output files and
              skipped.

  parallelism Rigs are completely independent. Each worker gets its own X
              display via `xvfb-run -a`, so there is no contention over the GL
              context.

Usage:
    python -m l2dprobe.pipeline --manifest manifest.json --jobs 12
    python -m l2dprobe.pipeline --models './CubismWebSamples/**/*.model3.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rig_name(path: str) -> str:
    return os.path.basename(path).split(".")[0]


def run_one(args) -> dict:
    path, probe_dir, data_dir, size, steps, res, timeout, use_xvfb = args
    name = rig_name(path)
    out = {"name": name, "path": path, "probe": None, "data": None}

    prefix = ["xvfb-run", "-a", "-s", "-screen 0 1024x1024x24"] if use_xvfb else []
    jobs = [
        ("probe", prefix + [sys.executable, "-m", "l2dprobe.run_probe", path,
                            "--out", os.path.join(probe_dir, f"rig_{name}"),
                            "--size", str(size)]),
        ("data", prefix + [sys.executable, "-m", "l2dprobe.dataset", path,
                           "--out", data_dir, "--res", str(res),
                           "--steps", str(steps), "--size", str(size)]),
    ]
    for key, cmd in jobs:
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=REPO)
        except subprocess.TimeoutExpired:
            out[key] = f"timeout>{timeout}s"
            continue
        if proc.returncode == 0:
            out[key] = f"ok {time.time() - t0:.0f}s"
        elif proc.returncode < 0:
            out[key] = f"segfault({-proc.returncode})"
        else:
            tail = (proc.stderr or "").strip().splitlines()
            out[key] = f"exit{proc.returncode}: {tail[-1][:60]}" if tail else \
                f"exit{proc.returncode}"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None,
                    help="manifest.json from ingest; uses only 'usable' rigs")
    ap.add_argument("--models", default=None, help="glob, alternative to manifest")
    ap.add_argument("--probe-dir", default="out")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--force", action="store_true", help="redo completed rigs")
    a = ap.parse_args(argv)

    paths = []
    if a.manifest:
        with open(a.manifest, encoding="utf-8") as fh:
            paths += [r["path"] for r in json.load(fh)["usable"]]
    if a.models:
        paths += [p for p in sorted(glob.glob(a.models, recursive=True))
                  if p.endswith(".model3.json") and ".probe." not in p]
    # Same rig can arrive from both sources.
    paths = list(dict.fromkeys(paths))
    if not paths:
        print("no models given (use --manifest or --models)", file=sys.stderr)
        return 1

    os.makedirs(a.probe_dir, exist_ok=True)
    os.makedirs(a.data_dir, exist_ok=True)

    todo = []
    for p in paths:
        n = rig_name(p)
        done = (os.path.exists(os.path.join(a.probe_dir, f"rig_{n}.json"))
                and os.path.exists(os.path.join(a.data_dir, f"{n}.npz")))
        if done and not a.force:
            continue
        todo.append(p)
    print(f"{len(paths)} rigs, {len(paths) - len(todo)} already complete, "
          f"{len(todo)} to run on {a.jobs} workers")
    if not todo:
        return 0

    use_xvfb = shutil.which("xvfb-run") is not None and not os.environ.get("DISPLAY")
    if use_xvfb:
        print("using xvfb-run -a (one display per worker)")

    work = [(p, a.probe_dir, a.data_dir, a.size, a.steps, a.res,
             a.timeout, use_xvfb) for p in todo]

    ok = fail = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, w): w[0] for w in work}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            good = str(r["probe"]).startswith("ok") and str(r["data"]).startswith("ok")
            ok, fail = ok + good, fail + (not good)
            elapsed = time.time() - t0
            eta = elapsed / i * (len(work) - i)
            print(f"[{i:>3}/{len(work)}] {'ok  ' if good else 'FAIL'} "
                  f"{r['name'][:26]:<26} probe={r['probe']} data={r['data']}  "
                  f"eta {eta / 60:.0f}m")

    print(f"\n{ok} succeeded, {fail} failed in {(time.time() - t0) / 60:.1f} min")
    print("Rerun the same command to retry failures; completed rigs are skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
