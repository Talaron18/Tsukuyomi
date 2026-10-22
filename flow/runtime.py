#!/usr/bin/env python3
"""
Block 2 runtime: drive a real rig with the trained prior and render it.

The prior only generates the expressive band. The other four bands are filled
the way the design always intended, and each for its own reason:

  autonomic  Blink and breath are procedural here. They are stochastic and
             carry no semantic content, so a learned model would spend capacity
             on noise that a few lines of code reproduce better.
  physics    Left unset so Live2D's solver drives them from head and body
             motion. Whether a parameter is solver-driven is a property of the
             target rig, not of the parameter, so no training-time choice could
             be right for every rig -- masking at playback is the fix.
  mouth      Held at default. Lip sync belongs to phoneme timing from TTS, and
             is Block 4.
  discrete   Held at default. Part swaps cannot be interpolated.

Expressive parameters that the rig's own motions use as switches -- arm and
hand swaps, effect fades, which the probe often labels continuous -- are snapped
to one of their two authored states as they play, with hysteresis and a short
ramp. A model trained with squared error hedges between modes, and a swap
parameter left at 0.4 draws both arms at partial opacity.

Note the probe stripped physics to make rendering deterministic. The runtime
does the opposite and loads the rig intact, because secondary motion is most of
what makes the result look alive.

Usage:
    xvfb-run -a python -m flow.runtime \\
        --model CubismWebSamples/Samples/Resources/Hiyori/Hiyori.model3.json \\
        --rig rigs/Hiyori.npz --ckpt prior.pt --seconds 8 --out preview/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

try:
    from ..l2dprobe.backend import Live2DBackend, ensure_display
    from ..l2dprobe.motions import (authored_motion, band_of, motion_files,
                                    physics_params, toggle_params)
    from .prior import Prior
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from l2dprobe.backend import Live2DBackend, ensure_display
    from l2dprobe.motions import (authored_motion, band_of, motion_files,
                                  physics_params, toggle_params)
    from prior import Prior

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"needs pytorch: {exc}") from exc


class Autonomic:
    """Procedural blink and breath.

    Blink intervals are drawn from an exponential distribution because real
    blinking is close to a Poisson process; a fixed period reads as mechanical
    immediately. The closing motion is faster than the opening one, which is
    also true of real eyelids and is the detail that makes it look right.
    """

    def __init__(self, fps: float, rng, mean_interval: float = 4.0):
        self.fps, self.rng = fps, rng
        self.mean_interval = mean_interval
        self.next_blink = rng.exponential(mean_interval)
        self.t = 0.0
        self.close, self.open_ = 0.06, 0.12

    def step(self) -> tuple[float, float]:
        """Returns (eye_open in [0,1], breath in [0,1])."""
        self.t += 1.0 / self.fps
        eye = 1.0
        dt = self.t - self.next_blink
        if 0 <= dt < self.close + self.open_:
            if dt < self.close:
                eye = 1.0 - dt / self.close
            else:
                eye = (dt - self.close) / self.open_
            eye = float(np.clip(eye, 0.0, 1.0))
        elif dt >= self.close + self.open_:
            self.next_blink = self.t + self.rng.exponential(self.mean_interval)
        breath = 0.5 * (1 + np.sin(2 * np.pi * 0.25 * self.t))
        return eye, float(breath)


class Switches:
    """Drive switch-like parameters the way a rigger keys them.

    Each switch holds one of its two authored states. It changes state only
    when the model's value crosses most of the way to the other one -- the
    hysteresis keeps a value hovering at the midpoint from flickering -- and
    then ramps across in a few frames, which is how authored swaps look.
    Everything in between is visible only during that ramp.
    """

    def __init__(self, states: dict[str, tuple[float, float]], ramp: int,
                 enter: float = 0.7):
        self.states = states
        self.ramp = max(int(ramp), 1)
        self.enter = enter
        self.side: dict[str, int] = {}
        self.pos: dict[str, float] = {}

    def __call__(self, pid: str, value: float) -> float:
        lo, hi = self.states[pid]
        u = (value - lo) / (hi - lo)
        if pid not in self.side:
            self.side[pid] = int(u >= 0.5)
            self.pos[pid] = float(self.side[pid])
        side = self.side[pid]
        if side == 0 and u > self.enter:
            side = 1
        elif side == 1 and u < 1 - self.enter:
            side = 0
        self.side[pid] = side
        step = 1.0 / self.ramp
        p = self.pos[pid]
        p = min(p + step, 1.0) if side else max(p - step, 0.0)
        self.pos[pid] = p
        return lo + p * (hi - lo)


def authored_motions(model_json: str, fps: float):
    """(name, motion) for every motion the rig ships, in motion_files order."""
    out = []
    for path in motion_files(model_json):
        m = authored_motion(path, fps)
        if m is not None and m["frames"]:
            out.append((os.path.basename(path), m))
    return out


def base_state(motions, keep):
    """Part opacities and held parameters from the first frame of the idle.

    Many rigs leave every arm and hand variant visible at the default pose and
    rely on motion to hide the spares. Holding those at an authored frame
    instead of the default gives the model a costume that is actually worn.
    `keep(pid)` says which parameters the base may set: not the ones the
    model drives, and not physics outputs.
    """
    if not motions:
        return None, {}, {}
    name, m = next(((n, m) for n, m in motions if "idle" in n.lower()),
                   motions[0])
    parts = {p: float(v[0]) for p, v in m["parts"].items()}
    params = {p: float(v[0]) for p, v in m["params"].items() if keep(p)}
    return name, params, parts


def denorm(n: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """[-1,1] back to the rig's own units."""
    return lo + (np.clip(n, -1, 1) + 1) * 0.5 * (hi - lo)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".model3.json")
    ap.add_argument("--rig", required=True, help="rigs/<name>.npz")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", default=None,
                    help="clips/<name>.npz to seed context from real motion")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="preview")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-autonomic", action="store_true")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="0..1 blend of real motion into every predicted frame. "
                         "Continuous, so unlike --reseed there are no seams to "
                         "hide. 0.15-0.3 keeps the prior stable; the printed "
                         "mix says how much of the result is really the model.")
    ap.add_argument("--reseed", type=float, default=0.0,
                    help="seconds between refreshing the context from real "
                         "motion. Demo scaffold, not the design: the prior is "
                         "stable while fed real frames and collapses only on "
                         "its own output, so re-seeding keeps it in the regime "
                         "it works in. Needs --clips.")
    ap.add_argument("--static", action="store_true",
                    help="render the seed clip verbatim, no model. Isolates "
                         "runtime parameter application from model output: if "
                         "this looks wrong too, the model is not the problem.")
    ap.add_argument("--neutral", action="store_true",
                    help="render the rig default pose only. If THIS looks "
                         "wrong, the costume is broken before any parameter "
                         "we set, and the discrete band is the cause.")
    ap.add_argument("--no-snap", action="store_true",
                    help="drive switch-like parameters continuously, as before. "
                         "Expect ghosted limbs on rigs with fading swaps.")
    ap.add_argument("--snap-ramp", type=int, default=6,
                    help="frames a snapped switch takes to change state")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="seconds of simulation on the first pose before "
                         "recording, so physics starts at rest instead of "
                         "springing out of the default pose on frame one")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="scale motion about its own mean. A post-hoc knob, "
                         "not a fix: it magnifies the model's mistakes as much "
                         "as its motion, and clamps at the parameter range.")
    a = ap.parse_args(argv)
    ensure_display(["-m", f"{__package__ or 'l2dprobe'}.runtime"]
                   + (argv if argv is not None else sys.argv[1:]))

    rng = np.random.default_rng(a.seed)
    z = np.load(a.rig, allow_pickle=True)
    pids = [str(x) for x in z["pid"]]
    kinds = [str(x) for x in z["kind"]]
    groups = [str(x) for x in z["group"]]
    lo, hi = z["minimum"].astype(np.float32), z["maximum"].astype(np.float32)
    dflt = z["default"].astype(np.float32)
    desc_all = z["descriptor"].astype(np.float32)

    phys, has_phys = physics_params(a.model)
    bands = [band_of(p, k, g, phys) for p, k, g in zip(pids, kinds, groups)]
    expressive = [i for i, b in enumerate(bands) if b == "expressive"]
    print(f"{len(pids)} params: "
          + ", ".join(f"{b} {bands.count(b)}" for b in
                      ("expressive", "physics", "autonomic", "mouth", "discrete")))
    if not expressive:
        print("no expressive parameters", file=sys.stderr)
        return 1

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = Prior(ck["ddim"], ck["state"], window=ck["window"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    window = ck["window"]
    if ck.get("holdout"):
        print(f"checkpoint was trained holding out {ck['holdout']}")

    desc = torch.from_numpy(desc_all[expressive])
    span_e = np.maximum(hi[expressive] - lo[expressive], 1e-6)
    default_n = np.clip((dflt[expressive] - lo[expressive]) / span_e * 2 - 1, -1, 1)

    # Seed from real motion when available: starting from a frozen default pose
    # gives the model a context that never occurs in training.
    if a.clips and os.path.exists(a.clips):
        cz = np.load(a.clips, allow_pickle=True)
        c0 = cz["clip0"].astype(np.float32)
        seed = (c0[:window] if len(c0) >= window
                else np.repeat(c0[:1], window, axis=0))
        print(f"seeded from {a.clips}")
    else:
        seed = np.tile(default_n.astype(np.float32), (window, 1))
        print("seeded from the default pose (no clips available)")

    n_frames = int(a.seconds * a.fps)
    if a.neutral:
        traj = np.tile(default_n.astype(np.float32), (n_frames, 1))
        print("NEUTRAL: rig defaults only, model bypassed")
    elif a.static:
        # Real authored frames, looped. Any visual defect here belongs to how
        # the runtime applies parameters, not to what the model produced.
        # Whole clips back to back: looping only the seed window replayed 1.6s
        # with a hard seam each time, which reads as a defect of its own.
        src = seed
        if a.clips and os.path.exists(a.clips):
            cz = np.load(a.clips, allow_pickle=True)
            src = np.concatenate([cz[f"clip{i}"].astype(np.float32)
                                  for i in range(int(cz["n_clips"]))])
        reps = int(np.ceil(n_frames / len(src)))
        traj = np.tile(src, (reps, 1))[:n_frames].astype(np.float32)
    else:
        pool = None
        if (a.anchor > 0 or a.reseed > 0) and a.clips and os.path.exists(a.clips):
            cz = np.load(a.clips, allow_pickle=True)
            pool = [cz[f"clip{i}"].astype(np.float32)
                    for i in range(int(cz["n_clips"]))
                    if len(cz[f"clip{i}"]) > window + 2]

        buf = torch.from_numpy(seed[None].copy())
        traj = []
        if a.anchor > 0 and pool:
            # Continuous anchoring. Every predicted frame is pulled a fixed
            # fraction toward the next real frame, so the model never drifts far
            # from the manifold and there is no periodic discontinuity. Periodic
            # re-seeding sustained motion but jumped 5-6x real magnitude at each
            # seam, which reads as a jerk once a second.
            cur = pool[rng.integers(len(pool))]
            pos = int(rng.integers(0, max(len(cur) - window - 1, 1)))
            lam = float(np.clip(a.anchor, 0, 1))
            print(f"anchoring {lam:.0%} real motion per frame "
                  f"({len(pool)} clips)")
            with torch.no_grad():
                for _ in range(n_frames):
                    pred = model.predict(buf, desc)[:, -1].clamp(-1, 1).numpy()[0]
                    pos += 1
                    if pos >= len(cur):
                        cur = pool[rng.integers(len(pool))]
                        pos = int(rng.integers(0, max(len(cur) - 1, 1)))
                    ref = cur[min(pos, len(cur) - 1)]
                    nxt = (1 - lam) * pred + lam * ref
                    traj.append(nxt.astype(np.float32))
                    buf = torch.cat(
                        [buf[:, 1:],
                         torch.from_numpy(nxt[None, None].astype(np.float32))],
                        dim=1)
        else:
            every = int(a.reseed * a.fps) if a.reseed > 0 else 0
            with torch.no_grad():
                for f in range(n_frames):
                    if pool and every and f > 0 and f % every == 0:
                        c = pool[rng.integers(len(pool))]
                        st = rng.integers(0, len(c) - window)
                        buf = torch.from_numpy(c[st:st + window][None].copy())
                    nxt = model.predict(buf, desc)[:, -1].clamp(-1, 1)
                    traj.append(nxt.numpy()[0])
                    buf = torch.cat([buf[:, 1:], nxt[:, None]], dim=1)
        traj = np.stack(traj)
    if a.gain != 1.0:
        centre = traj.mean(axis=0, keepdims=True)
        traj = np.clip(centre + (traj - centre) * a.gain, -1, 1)
    if a.anchor > 0:
        print(f"NOTE: {a.anchor:.0%} of each frame is real motion. This is a "
              f"demo scaffold, not evidence the prior works.")
    print(f"generated {n_frames} frames  std={traj.std(axis=0).mean():.4f}  "
          f"per-frame delta={np.abs(np.diff(traj, axis=0)).mean():.5f}"
          + (f"  (gain {a.gain}x)" if a.gain != 1.0 else ""))

    # Autonomic targets, found by band rather than by name where possible.
    eye_idx = [i for i, p in enumerate(pids)
               if bands[i] == "autonomic" and "open" in p.lower()]
    breath_idx = [i for i, p in enumerate(pids) if "breath" in p.lower()]
    auto = Autonomic(a.fps, rng)

    motions = authored_motions(a.model, a.fps)
    n_parts = len({p for _, m in motions for p in m["parts"]})
    if n_parts and not has_pose(a.model):
        print(f"{n_parts} parts are switched by PartOpacity curves and the rig "
              "has no pose3 file; applying them directly (Live2D Native would "
              "ignore them and draw every variant)")
    model_driven = {pids[i] for i in expressive}
    band_by_id = dict(zip(pids, bands))
    base_name, base_params, base_parts = base_state(
        motions, lambda p: p not in model_driven
        and band_by_id.get(p) != "physics" and p not in phys)
    if base_name and not (a.static or a.neutral):
        print(f"base state from {base_name}: {len(base_parts)} part "
              f"opacities, {len(base_params)} held parameters")

    static_frames = None
    if a.static and motions:
        # Replay the motion files themselves, every curve in the rig's own
        # units. The clips hold only the expressive band, normalised, so a
        # replay from them drops part opacities, discrete parameters, and any
        # parameter the probe never kept -- the arm switches among them.
        seq = [(m, t) for _, m in motions for t in range(m["frames"])]
        static_frames = [seq[f % len(seq)] for f in range(n_frames)]
        print(f"STATIC: replaying {len(motions)} motion files curve for curve "
              f"({len(seq) / a.fps:.1f}s incl. part opacities)")
    elif a.static:
        print(f"STATIC: no motion files found; replaying {len(traj)} frames of "
              "the expressive band from the clips, model bypassed")

    switches = None
    # --static replays authored values unchanged; it is the reference the
    # other modes are judged against, so nothing may rewrite it. Snapping did:
    # on aidang_2 it changed which arm was drawn during authored playback.
    if not a.no_snap and not (a.static or a.neutral):
        found = toggle_params(a.model, a.fps)
        live = {pids[i]: found[pids[i]] for i in expressive if pids[i] in found}
        if live:
            switches = Switches({p: (v[1], v[2]) for p, v in live.items()},
                                a.snap_ramp)
            print(f"snapping {len(live)} switch-like parameters: "
                  + ", ".join(sorted(live)))

    os.makedirs(a.out, exist_ok=True)
    # strip_dynamics=False: physics and pose stay live, which is the whole point
    # of previewing rather than reading numbers.
    # fps: physics and pose advance by one video frame per render, not by
    # however long the render happened to take.
    with Live2DBackend(a.model, size=a.size, strip_dynamics=False,
                       fps=a.fps) as be:
        be.calibrate()
        print("backend: " + ", ".join(
            f"{k}={be._api_report[k]}" for k in ("order", "timestep")
            if k in be._api_report))
        by_id = {s.pid: s for s in be.params}

        def frame_overrides(f):
            if static_frames is not None:
                m, t = static_frames[f]
                return ({p: float(v[t]) for p, v in m["params"].items()},
                        {p: float(v[t]) for p, v in m["parts"].items()})
            if a.neutral:
                return {}, None
            over = dict(base_params)
            vals = denorm(traj[f], lo[expressive], hi[expressive])
            for col, i in enumerate(expressive):
                if pids[i] in by_id:
                    v = float(vals[col])
                    if switches is not None and pids[i] in switches.states:
                        v = switches(pids[i], v)
                    over[pids[i]] = v
            return over, base_parts

        if a.warmup > 0 and n_frames:
            first, first_parts = frame_overrides(0)
            for _ in range(int(a.warmup * a.fps)):
                be.render(first, parts=first_parts)

        written = 0
        for f in range(n_frames):
            over, parts = frame_overrides(f)
            if not a.no_autonomic:
                eye, breath = auto.step()
                # Under --static the authored curve wins where there is one.
                for i in eye_idx:
                    if pids[i] in by_id and pids[i] not in over:
                        over[pids[i]] = float(lo[i] + eye * (hi[i] - lo[i]))
                for i in breath_idx:
                    if pids[i] in by_id and pids[i] not in over:
                        over[pids[i]] = float(lo[i] + breath * (hi[i] - lo[i]))
            img = be.render(over, parts=parts)
            _save(img, os.path.join(a.out, f"f{f:05d}.png"))
            written += 1
    print(f"rendered {written} frames -> {a.out}/")

    _contact_sheet(a.out, n_frames)
    _encode(a.out, a.fps)
    return 0


def has_pose(model_json: str) -> bool:
    try:
        with open(model_json, encoding="utf-8") as fh:
            return bool(json.load(fh).get("FileReferences", {}).get("Pose"))
    except Exception:  # noqa: BLE001
        return False


def _save(img, path):
    try:
        from PIL import Image
        Image.fromarray(img, "RGBA").save(path)
    except ImportError:
        np.save(path.replace(".png", ".npy"), img)


def _contact_sheet(out_dir: str, n_frames: int, cols: int = 8):
    """Always produce one viewable artifact, even with no video encoder."""
    try:
        from PIL import Image
    except ImportError:
        return
    picks = np.linspace(0, n_frames - 1, cols).astype(int)
    tiles = []
    for i in picks:
        p = os.path.join(out_dir, f"f{i:05d}.png")
        if os.path.exists(p):
            im = Image.open(p).convert("RGB").resize((256, 256))
            tiles.append(np.asarray(im))
    if tiles:
        sheet = np.concatenate(tiles, axis=1)
        dest = os.path.join(out_dir, "sheet.png")
        Image.fromarray(sheet).save(dest)
        print(f"contact sheet -> {dest}")


def _encode(out_dir: str, fps: float):
    dest = os.path.join(out_dir, "preview.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps),
             "-i", os.path.join(out_dir, "f%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", dest],
            capture_output=True, check=True, timeout=300)
        print(f"video -> {dest}")
    except FileNotFoundError:
        print("ffmpeg not found; PNG sequence and contact sheet written instead")
    except Exception as exc:  # noqa: BLE001
        print(f"ffmpeg failed ({exc}); PNGs are still in {out_dir}/")


if __name__ == "__main__":
    raise SystemExit(main())
