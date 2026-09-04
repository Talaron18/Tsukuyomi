#!/usr/bin/env python3
"""
Triage a directory of downloaded Live2D models before probing them.

Scaling from 7 rigs to dozens means most failures will be data problems, not
code problems: wrong Cubism version, moc3 too new for the installed Core,
missing textures, duplicate models under different names, or rigs whose
parameter IDs share nothing with anything else.

This loads each candidate once, records why it failed if it did, and writes a
manifest of the ones worth spending render time on.

It also reports parameter-ID overlap with the standard Cubism set. That matters
for evaluation, not for training: retrieval is scored on shared IDs, so a rig
with no standard parameters contributes to the encoder but cannot be scored.

Usage:
    xvfb-run -s "-screen 0 1024x1024x24" \
        python -m l2dprobe.ingest ~/live2d-models --out manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter

try:
    from .backend import IncompatibleRig, Live2DBackend
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from yachiyo.l2dprobe.backend import IncompatibleRig, Live2DBackend

STANDARD = {
    "ParamAngleX", "ParamAngleY", "ParamAngleZ", "ParamEyeLOpen", "ParamEyeROpen",
    "ParamEyeLSmile", "ParamEyeRSmile", "ParamEyeBallX", "ParamEyeBallY",
    "ParamBrowLY", "ParamBrowRY", "ParamBrowLX", "ParamBrowRX",
    "ParamBrowLForm", "ParamBrowRForm", "ParamBrowLAngle", "ParamBrowRAngle",
    "ParamMouthForm", "ParamMouthOpenY", "ParamCheek", "ParamBreath",
    "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
    "ParamHairFront", "ParamHairSide", "ParamHairBack",
}


def find_candidates(root: str) -> tuple[list[str], list[str]]:
    """Return (cubism3 model3.json paths, cubism2 model.json paths)."""
    v3, v2 = [], []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            path = os.path.join(dirpath, f)
            if f.endswith(".model3.json") and ".probe." not in f:
                v3.append(path)
            elif f == "model.json" or f.endswith(".model.json"):
                v2.append(path)
    return sorted(v3), sorted(v2)


def moc_digest(model_json: str) -> str | None:
    """Hash the .moc3 to catch the same rig appearing under several names."""
    try:
        with open(model_json, encoding="utf-8") as fh:
            moc = json.load(fh).get("FileReferences", {}).get("Moc")
        if not moc:
            return None
        path = os.path.join(os.path.dirname(model_json), moc)
        with open(path, "rb") as fh:
            return hashlib.sha1(fh.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def json_ok(path: str) -> str:
    """Return '' if the file parses as JSON, else why not.

    aierdeliqi_5 produced 'Json parse error : @line 1' inside the Cubism Core
    and then a segfault. Checking in Python first turns a native crash into a
    readable skip reason, and catches BOMs and truncated files besides.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        if not raw.strip():
            return "empty file"
        if raw[:3] == b"\xef\xbb\xbf":
            raw = raw[3:]
        json.loads(raw.decode("utf-8", errors="strict"))
        return ""
    except UnicodeDecodeError as exc:
        return f"not utf-8: {exc}"
    except json.JSONDecodeError as exc:
        return f"invalid json: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def probe_one(path: str, size: int = 256) -> dict:
    rec = {"path": path, "name": os.path.basename(path).split(".")[0]}
    bad = json_ok(path)
    if bad:
        rec.update(ok=False, reason=bad)
        return rec
    try:
        be = Live2DBackend(path, size=size).open()
    except IncompatibleRig as exc:
        rec.update(ok=False, reason=f"incompatible: {exc}")
        return rec
    except Exception as exc:  # noqa: BLE001
        rec.update(ok=False, reason=f"{type(exc).__name__}: {exc}")
        return rec
    try:
        be.calibrate()
        base = be.render()
        coverage = float((base[..., 3] > 8).mean())
        pids = [s.pid for s in be.params]
        rec.update(
            ok=coverage > 0.01,
            reason="" if coverage > 0.01 else f"renders almost nothing ({coverage:.4f})",
            n_params=len(pids),
            coverage=round(coverage, 4),
            standard=len(set(pids) & STANDARD),
        )
    except Exception as exc:  # noqa: BLE001
        rec.update(ok=False, reason=f"{type(exc).__name__}: {exc}")
    finally:
        be.close()
    return rec


def probe_isolated(path: str, size: int, timeout: int = 120) -> dict:
    """Probe one model in a child process.

    The Cubism Core is native code and a malformed model can segfault it. A
    segfault is not a Python exception -- try/except cannot catch it, and it
    takes the whole interpreter down. At 7 models that is a nuisance; across
    164 it means one bad rig destroys the entire run. Isolating each load costs
    a process spawn and converts a fatal crash into a skipped row.
    """
    cmd = [sys.executable, "-m", "l2dprobe.ingest", "--single", path,
           "--size", str(size)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout,
                              cwd=os.path.dirname(os.path.dirname(
                                  os.path.abspath(__file__))))
    except subprocess.TimeoutExpired:
        return {"path": path, "name": os.path.basename(path).split(".")[0],
                "ok": False, "reason": f"timed out after {timeout}s"}

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            try:
                return json.loads(line[len("__RESULT__"):])
            except json.JSONDecodeError:
                break
    reason = "segfault" if proc.returncode and proc.returncode < 0 else \
             f"exit {proc.returncode}"
    tail = (proc.stderr or "").strip().splitlines()
    if tail:
        reason += f": {tail[-1][:80]}"
    return {"path": path, "name": os.path.basename(path).split(".")[0],
            "ok": False, "reason": reason}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--single", default=None,
                    help="internal: probe one model and print __RESULT__ json")
    ap.add_argument("--out", default="manifest.json")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    if a.single:
        rec = probe_one(a.single, a.size)
        print("__RESULT__" + json.dumps(rec))
        return 0

    # Remove stripped copies left behind by earlier crashed runs; a truncated
    # one would be picked up as a model and fail confusingly.
    stale = 0
    for dirpath, _dirs, files in os.walk(a.root):
        for f in files:
            if f.endswith(".probe.model3.json"):
                try:
                    os.remove(os.path.join(dirpath, f))
                    stale += 1
                except OSError:
                    pass
    if stale:
        print(f"removed {stale} stale .probe.model3.json files")

    v3, v2 = find_candidates(a.root)
    print(f"found {len(v3)} Cubism 3 (.model3.json), {len(v2)} Cubism 2 (model.json)")
    if v2:
        print("  Cubism 2 models need live2d.v2, which this backend does not "
              "bind. Skipped for now.")
    if a.limit:
        v3 = v3[:a.limit]

    seen: dict[str, str] = {}
    records, dupes = [], 0
    for i, path in enumerate(v3, 1):
        digest = moc_digest(path)
        if digest and digest in seen:
            dupes += 1
            continue
        if digest:
            seen[digest] = path
        rec = probe_isolated(path, a.size)
        records.append(rec)
        flag = "ok " if rec.get("ok") else "SKIP"
        print(f"[{i:>3}/{len(v3)}] {flag} {rec['name'][:28]:<28} "
              f"params={rec.get('n_params', '-'):>4} "
              f"std={rec.get('standard', '-'):>3} {rec.get('reason', '')[:44]}")

    good = [r for r in records if r.get("ok")]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"usable": good, "all": records}, fh, indent=2)

    print(f"\n{len(good)}/{len(v3)} usable, {dupes} duplicate mocs skipped")
    if good:
        std = Counter(r["standard"] for r in good)
        print(f"  standard-parameter overlap: "
              + ", ".join(f"{k} params x{v}" for k, v in sorted(std.items())))
        print(f"  rigs with >=10 standard params: "
              f"{sum(1 for r in good if r['standard'] >= 10)}")
    print(f"\nmanifest -> {a.out}")
    print("Rigs with few standard parameters still help the encoder train;")
    print("they just cannot be scored by name-based retrieval.")
    if any(r.get("reason", "").startswith("incompatible") for r in records):
        print("Some rigs need a newer Cubism Core than live2d-py 0.7.0.4 ships.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
