#!/usr/bin/env python3
"""
How bursty is a render's motion?  Prints the share of 0.1 s bins in which
nearly nothing moves and the burstiness of the rest.

  python scripts/motion_still.py runs/demo/Mao_10/plan.npz
  python scripts/motion_still.py runs/demo/Mao_10/demo.mp4   (needs ffmpeg)

  plan.npz  what flow.runtime --dump writes: the parameters as applied each
            frame, normalised to +-1 (the director's run folder keeps one).
            Speed of a bin = the mean over its frames of the L2 norm of the
            frame-to-frame change, per second, over the columns that move
            the figure: the mouth, blinks and breath are left out (the
            voice moves the mouth all the time, and a blink is no sway),
            --all keeps them.
  video     mean absolute difference of gray 64 px frames, per second,
            binned the same (the mouth is in it).

  still     share of bins whose speed is under --eps (plan: 0.01 of a
            parameter's range a second, a third of a degree of head; video:
            0.3 gray levels a second).  Target for the offline render: < 10%.
  burst     p95 / median of the bin speeds: 1 is an even motion, 10 and up
            is a few spikes between stillness (inf: more than half the
            bins are exactly still).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys

import numpy as np

BIN = 0.1               # seconds
EPS_PLAN, EPS_VIDEO = 0.01, 0.3
STILL_TARGET = 0.10
# Columns that are not the figure's sway: the mouth, blinks, breath.
QUIET = re.compile(r"mouth|eye.*open|eyeopen|breath|lip|tongue", re.I)


def bin_speed(speed, fps, width=BIN):
    """Mean of a per-frame speed array over consecutive bins of `width`
    seconds (a short last bin is dropped)."""
    n = max(int(round(width * fps)), 1)
    k = len(speed) // n
    return np.asarray(speed[:k * n], float).reshape(k, n).mean(1)


def still_stats(speed, fps, eps, width=BIN) -> dict:
    """{bins, still (share of bins under eps), median, p95, burst (p95 /
    median; inf when the median is 0)} for per-frame speeds."""
    b = bin_speed(speed, fps, width)
    if not len(b):
        return {"bins": 0, "still": float("nan"), "median": 0.0, "p95": 0.0,
                "burst": float("nan")}
    med, p95 = float(np.median(b)), float(np.percentile(b, 95))
    return {"bins": int(len(b)), "still": float((b < eps).mean()),
            "median": med, "p95": p95,
            "burst": p95 / med if med > 0 else float("inf")}


def plan_speed(path, keep_all=False):
    """(per-frame speed in range units a second, fps) of a plan.npz."""
    z = np.load(path, allow_pickle=True)
    traj, fps = np.asarray(z["traj"], float), float(z["fps"])
    pid = [str(p) for p in z["pid"]]
    cols = [c for c, p in enumerate(pid) if keep_all or not QUIET.search(p)]
    if not cols:
        raise SystemExit(f"{path}: no columns left to measure")
    d = np.diff(traj[:, cols], axis=0) * fps / 2.0      # +-1 -> range units
    return np.concatenate([[0.0], np.linalg.norm(d, axis=1)]), fps


def video_speed(path, size=64):
    """(per-frame mean abs gray difference a second, fps) of a video."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg and ffprobe are needed for a video")
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    num, _, den = pr.partition("/")
    fps = float(num) / float(den or 1)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf",
         f"scale={size}:{size},format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, size * size).astype(float)
    d = np.abs(np.diff(fr, axis=0)).mean(1) * fps
    return np.concatenate([[0.0], d]), fps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="+", help="plan.npz or a video")
    ap.add_argument("--eps", type=float, default=None,
                    help=f"speed under which a bin is still [{EPS_PLAN} for "
                         f"a plan, {EPS_VIDEO} for a video]")
    ap.add_argument("--bin", type=float, default=BIN, help="bin, seconds")
    ap.add_argument("--all", action="store_true",
                    help="plan: keep the mouth, blink and breath columns")
    a = ap.parse_args(argv)
    bad = 0
    for path in a.path:
        plan = path.lower().endswith(".npz")
        speed, fps = (plan_speed(path, a.all) if plan else video_speed(path))
        eps = a.eps if a.eps is not None else (EPS_PLAN if plan
                                               else EPS_VIDEO)
        r = still_stats(speed, fps, eps, a.bin)
        ok = r["still"] < STILL_TARGET
        bad += not ok
        print(f"{path}: {r['bins']} bins of {a.bin:g} s, still "
              f"{100 * r['still']:.1f}% (< {100 * STILL_TARGET:g}% target: "
              f"{'ok' if ok else 'NO'}), burst p95/median "
              f"{r['burst']:.2f} (median {r['median']:.4f}, p95 "
              f"{r['p95']:.4f})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
