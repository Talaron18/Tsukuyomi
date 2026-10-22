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

Expressive sources, one per run:
  --ckpt prior.pt        deterministic next-frame prior (flow.prior)
  --ckpt flowprior.pt    flow-matching prior (flow.flowprior --save), sampled
                         a chunk at a time; --seed picks the noise
  --behaviors stats.json hand-made baseline (flow.behaviors) driving only the
                         head/body/gaze/brow roles; other expressive
                         parameters are held at the idle's first frame and
                         recorded as applied, so the scorer sees them held
  --static / --neutral   authored replay / rig defaults
All of them fill the same planned frames, so --dump, base state, part
opacities and snapping behave identically.

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

# Only the pure-numpy pieces are imported at module level. The backend needs
# live2d-py and a GL display, and the prior needs torch; importing either here
# made Switches, Autonomic and base_state untestable without both, and made
# --static and --dump-only runs depend on things they never use.
try:
    from ..l2dprobe.analyze import group_of
    from ..l2dprobe.motions import (authored_motion, band_of, motion_files,
                                    physics_params, toggle_params)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from l2dprobe.analyze import group_of
    from l2dprobe.motions import (authored_motion, band_of, motion_files,
                                  physics_params, toggle_params)

# Groups that are continuous by nature. Animators often park gaze or the head
# at an extreme and hold it, which toggle_params reads as a switch -- on
# aidang_2 it snapped ParamEyeBallX/Y and gaze stopped moving entirely. A head
# turn or a glance is never a part swap, so these are never snapped.
NEVER_SNAP = ("gaze", "head", "body")


def snappable(pid: str, rig_group: str) -> bool:
    return (rig_group not in NEVER_SNAP
            and group_of(pid) not in NEVER_SNAP)


def _backend():
    try:
        from ..l2dprobe.backend import Live2DBackend, ensure_display
    except ImportError:
        from l2dprobe.backend import Live2DBackend, ensure_display
    return Live2DBackend, ensure_display


def _prior():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"needs pytorch: {exc}") from exc
    try:
        from .prior import Prior
    except ImportError:
        from flow.prior import Prior
    return torch, Prior


class Autonomic:
    """Procedural blink and breath.

    Blink intervals are drawn from an exponential distribution because real
    blinking is close to a Poisson process; a fixed period reads as mechanical
    immediately. The closing motion is faster than the opening one, which is
    also true of real eyelids and is the detail that makes it look right.
    """

    def __init__(self, fps: float, rng, mean_interval: float = 4.0,
                 requests=()):
        self.fps, self.rng = fps, rng
        self.mean_interval = mean_interval
        self.next_blink = rng.exponential(mean_interval)
        self.t = 0.0
        self.close, self.open_ = 0.06, 0.12
        # Blink requests from behaviour (times in s): a large gaze shift
        # brings a blink with it. A request moves the next blink earlier; the
        # Poisson schedule resumes after it.
        self.requests = sorted(requests)

    def request(self, t: float) -> None:
        self.requests.append(t)
        self.requests.sort()

    def step(self) -> tuple[float, float]:
        """Returns (eye_open in [0,1], breath in [0,1])."""
        self.t += 1.0 / self.fps
        eye = 1.0
        while self.requests and self.requests[0] <= self.t:
            r = self.requests.pop(0)
            if self.t - self.next_blink < 0:      # not mid-blink
                self.next_blink = max(r, self.t - 1.0 / self.fps)
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


def base_keep(model_driven, phys, band_by_id):
    """Which parameters base_state may hold at the idle's first frame.

    Not the ones the model drives, and not physics outputs. Autonomic and
    mouth stay out too: blink and breath are generated procedurally and mouth
    waits for lip sync, so freezing them at an idle frame would switch both
    off -- which is what happened before this was split out.
    """
    def keep(pid):
        return (pid not in model_driven and pid not in phys
                and band_by_id.get(pid) not in ("physics", "autonomic",
                                                "mouth"))
    return keep


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
    ap.add_argument("--ckpt", default=None,
                    help="prior checkpoint: the deterministic prior "
                         "(flow.prior) or a flow-matching one (flow.flowprior "
                         "--save), told apart by their config keys; not "
                         "needed for --static/--neutral/--behaviors")
    ap.add_argument("--behaviors", default=None, metavar="STATS.json",
                    help="hand-made behaviour baseline (flow.behaviors) with "
                         "these fitted statistics, or 'default'. Drives head, "
                         "body, gaze and brow roles; every other expressive "
                         "parameter is held at the idle's first frame")
    ap.add_argument("--binding", default=None, metavar="FILE.json",
                    help="--behaviors role binding override: "
                         '{"head_x": "ParamFoo"} or {"head_x": ["ParamFoo", -1]}')
    ap.add_argument("--solver-steps", type=int, default=None,
                    help="flow checkpoint: Euler steps per chunk "
                         "(default: what training evaluated with)")
    ap.add_argument("--device", default="auto",
                    help="flow checkpoint: cpu, cuda, or auto")
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
    ap.add_argument("--dump", default=None, metavar="PATH.npz",
                    help="write what is actually applied each frame -- the "
                         "expressive band normalised to [-1,1] in the clips' "
                         "column order, after gain and snapping, plus part "
                         "opacities -- for python -m flow.score")
    ap.add_argument("--dump-only", action="store_true",
                    help="write --dump and stop before opening a renderer. "
                         "Scoring needs no pixels, so this needs no GL either")
    ap.add_argument("--no-parts", action="store_true",
                    help="never write part opacities, as Live2D Native does "
                         "for a rig without a pose3 file. Diagnostic: shows "
                         "what PartOpacity application is fixing")
    a = ap.parse_args(argv)
    if a.dump_only and not a.dump:
        ap.error("--dump-only needs --dump")
    if not (a.static or a.neutral or a.ckpt or a.behaviors):
        ap.error("--ckpt is required unless --static, --neutral or "
                 "--behaviors")
    if not a.dump_only:
        _, ensure_display = _backend()
        ensure_display(["-m", f"{__package__ or 'flow'}.runtime"]
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
    if a.clips and os.path.exists(a.clips):
        # Clips record which rig rows they hold. Built under older band rules
        # (vowels counted as expressive until recently) their columns no longer
        # match, and seeding or anchoring from them would misalign every
        # parameter. Their rows are what any checkpoint trained on them saw, so
        # they win; parameters now banded otherwise are just not written.
        rows = np.load(a.clips, allow_pickle=True)["param_row"].astype(int)
        if list(rows) != expressive:
            gone = [pids[i] for i in rows if bands[i] != "expressive"]
            print(f"clips use {len(rows)} columns, current bands give "
                  f"{len(expressive)}; following the clips"
                  + (f", not writing {', '.join(gone)}" if gone else "")
                  + ". Regenerate clips with l2dprobe.motions to align.")
            expressive = [int(i) for i in rows]

    model = desc = flow = None
    window = 48
    use_model = bool(a.ckpt) and not (a.static or a.neutral or a.behaviors)
    if use_model:
        torch, Prior = _prior()
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        if ck.get("holdout"):
            print(f"checkpoint was trained holding out {ck['holdout']}")
        desc = torch.from_numpy(desc_all[expressive])
        want = ck.get("ddim")
        if want is not None and int(want) != desc_all.shape[1]:
            # A checkpoint is tied to the descriptor width of the rig files it
            # was trained against (export_descriptors --pca). Feeding another
            # width fails deep inside a matmul with no hint of the cause.
            print(f"ERROR: {a.ckpt} expects {want}-dim descriptors but "
                  f"{a.rig} has {desc_all.shape[1]}. The rig files were "
                  "re-exported with different settings since this checkpoint "
                  "was trained: use the rigs/ it was trained with, or "
                  "retrain on the current ones.", file=sys.stderr)
            return 1
        if "chunk" in ck and "ctx" in ck:
            # Flow-matching checkpoint: generated a chunk at a time from
            # noise. --anchor/--reseed call the deterministic next-frame
            # predict() and have no meaning here.
            if a.anchor > 0 or a.reseed > 0:
                ap.error("--anchor/--reseed apply to the deterministic prior "
                         "only")
            try:
                from .flowprior import FlowStream, load_checkpoint
            except ImportError:
                from flow.flowprior import FlowStream, load_checkpoint
            device = ("cuda" if torch.cuda.is_available() else "cpu") \
                if a.device == "auto" else a.device
            fmodel, cfg = load_checkpoint(a.ckpt, device)
            window = cfg["ctx"]
            steps = a.solver_steps or cfg.get("solver_steps", 6)
            flow = (fmodel, cfg, device, steps)
            print(f"flow prior: ctx {cfg['ctx']} chunk {cfg['chunk']} "
                  f"base {cfg['base_mode']} cond_dim {cfg['cond_dim']} "
                  f"solver {steps} steps on {device}")
        else:
            model = Prior(ck["ddim"], ck["state"], window=ck["window"])
            model.load_state_dict(ck["state_dict"])
            model.eval()
            window = ck["window"]
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
    blink_requests = []
    roles = {}
    if a.behaviors and not (a.static or a.neutral):
        try:
            from .behaviors import bind_roles, generate, load_binding, load_stats
        except ImportError:
            from flow.behaviors import (bind_roles, generate, load_binding,
                                        load_stats)
        stats = load_stats(a.behaviors)
        roles = bind_roles(pids, [pids[i] for i in expressive], a.rig,
                           load_binding(a.binding))
        for r, (p, sg, src) in sorted(roles.items()):
            print(f"  role {r:<7} -> {p}{'' if sg > 0 else ' (sign -1)'}  "
                  f"[{src}]" + ("  FALLBACK: confirm by hand"
                               if src == "fallback" else ""))
        if not roles:
            print("no roles bound; everything will be held", file=sys.stderr)
        gen, blink_requests = generate(stats, list(roles), a.seconds, a.fps,
                                       rng)
        # Non-role columns are filled from the base state below, once it is
        # known; until then they sit at the default.
        traj = np.tile(default_n.astype(np.float32), (n_frames, 1))
        col_of = {pids[i]: c for c, i in enumerate(expressive)}
        for r, (p, sg, _) in roles.items():
            traj[:, col_of[p]] = np.clip(sg * gen[r][:n_frames], -1, 1)
        print(f"behaviors: {len(roles)} roles from "
              f"{'defaults' if a.behaviors == 'default' else a.behaviors}, "
              f"{len(blink_requests)} gaze-coupled blinks requested")
    elif flow is not None:
        fmodel, cfg, device, steps = flow
        stream = FlowStream(fmodel, desc, seed, steps=steps, seed=a.seed,
                            device=device)
        traj = stream.generate(n_frames)
    elif a.neutral:
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
    if use_model:
        # Only model output is "generated"; under --static this described the
        # clips fallback trajectory, not what gets replayed.
        print(f"generated {n_frames} frames  std={traj.std(axis=0).mean():.4f}  "
              f"per-frame delta={np.abs(np.diff(traj, axis=0)).mean():.5f}"
              + (f"  (gain {a.gain}x)" if a.gain != 1.0 else ""))

    # Autonomic targets, found by band rather than by name where possible.
    eye_idx = [i for i, p in enumerate(pids)
               if bands[i] == "autonomic" and "open" in p.lower()]
    breath_idx = [i for i, p in enumerate(pids) if "breath" in p.lower()]
    auto = Autonomic(a.fps, rng, requests=blink_requests)

    motions = authored_motions(a.model, a.fps)
    n_parts = len({p for _, m in motions for p in m["parts"]})
    if n_parts and not has_pose(a.model):
        print(f"{n_parts} parts are switched by PartOpacity curves and the rig "
              "has no pose3 file; applying them directly (Live2D Native would "
              "ignore them and draw every variant)")
    # Under --behaviors only the bound roles are driven; the base state holds
    # every other expressive parameter at the idle's first frame.
    model_driven = ({p for p, _, _ in roles.values()} if a.behaviors
                    else {pids[i] for i in expressive})
    band_by_id = dict(zip(pids, bands))
    base_name, base_params, base_parts = base_state(
        motions, base_keep(model_driven, phys, band_by_id))
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

    if a.behaviors and not (a.static or a.neutral):
        bound = model_driven
        for col, i in enumerate(expressive):
            if pids[i] not in bound and pids[i] in base_params:
                traj[:, col] = np.clip(
                    (base_params[pids[i]] - lo[i]) / span_e[col] * 2 - 1, -1, 1)

    switches = None
    # --static replays authored values unchanged; it is the reference the
    # other modes are judged against, so nothing may rewrite it. Snapping did:
    # on aidang_2 it changed which arm was drawn during authored playback.
    if not a.no_snap and not (a.static or a.neutral):
        found = toggle_params(a.model, a.fps)
        live = {pids[i]: found[pids[i]] for i in expressive
                if pids[i] in found and bands[i] == "expressive"
                and snappable(pids[i], groups[i])}
        if live:
            switches = Switches({p: (v[1], v[2]) for p, v in live.items()},
                                a.snap_ramp)
            print(f"snapping {len(live)} switch-like parameters: "
                  + ", ".join(sorted(live)))

    # Decide every frame's writes before a renderer exists. The values are the
    # same either way; doing it here means --dump records exactly what the
    # render loop will apply (after gain and snapping) and --dump-only can stop
    # without GL. Switches are stateful, so they must run once per frame in
    # order -- the old loop also ran them for every warmup render of frame 0.
    frames = []
    applied = np.empty((n_frames, len(expressive)), np.float32)
    applied_mask = np.ones((n_frames, len(expressive)), bool)
    segment = np.zeros(n_frames, np.int32)
    span_x = np.maximum(hi[expressive] - lo[expressive], 1e-6)
    seg_id = -1
    for f in range(n_frames):
        if static_frames is not None:
            m, t = static_frames[f]
            if t == 0:
                seg_id += 1
            segment[f] = max(seg_id, 0)
            over = {p: float(v[t]) for p, v in m["params"].items()}
            parts = {p: float(v[t]) for p, v in m["parts"].items()}
            for col, i in enumerate(expressive):
                if pids[i] in over:
                    applied[f, col] = (over[pids[i]] - lo[i]) / span_x[col] * 2 - 1
                else:
                    applied[f, col] = default_n[col]
                    applied_mask[f, col] = False
        elif a.neutral:
            over, parts = {}, None
            applied[f] = default_n
        else:
            over = dict(base_params)
            vals = denorm(traj[f], lo[expressive], hi[expressive])
            for col, i in enumerate(expressive):
                if bands[i] != "expressive":
                    applied[f, col] = default_n[col]
                    applied_mask[f, col] = False
                    continue
                v = float(vals[col])
                if switches is not None and pids[i] in switches.states:
                    v = switches(pids[i], v)
                over[pids[i]] = v
                applied[f, col] = (v - lo[i]) / span_x[col] * 2 - 1
            parts = base_parts
        if a.no_parts:
            parts = None
        frames.append((over, parts))
    np.clip(applied, -1, 1, out=applied)

    if a.dump:
        part_ids = sorted({p for _, pt in frames if pt for p in pt})
        opac = np.full((n_frames, len(part_ids)), np.nan, np.float32)
        col_of = {p: j for j, p in enumerate(part_ids)}
        for f, (_, pt) in enumerate(frames):
            for p, v in (pt or {}).items():
                opac[f, col_of[p]] = v
        mode = ("static" if a.static else "neutral" if a.neutral
                else f"behaviors seed={a.seed}" if a.behaviors
                else f"flow steps={flow[3]} seed={a.seed}" if flow
                else f"prior anchor={a.anchor:g} reseed={a.reseed:g}")
        os.makedirs(os.path.dirname(os.path.abspath(a.dump)), exist_ok=True)
        np.savez_compressed(
            a.dump, traj=applied, mask=applied_mask, segment=segment,
            pid=np.array([pids[i] for i in expressive], dtype=object),
            param_row=np.asarray(expressive, np.int32),
            fps=np.float32(a.fps), part_id=np.array(part_ids, dtype=object),
            part_opacity=opac, mode=np.array(mode))
        print(f"dump -> {a.dump}  ({n_frames} frames, {len(expressive)} "
              f"expressive, {len(part_ids)} parts, mode {mode})")
        if a.dump_only:
            return 0

    Live2DBackend, _ = _backend()
    os.makedirs(a.out, exist_ok=True)
    # strip_dynamics=False: physics and pose stay live, which is the whole point
    # of previewing rather than reading numbers.
    # fps: physics and pose advance by one video frame per render, not by
    # however long the render happened to take.
    with Live2DBackend(a.model, size=a.size, strip_dynamics=False,
                       fps=a.fps) as be:
        be.calibrate()
        print("backend: " + ", ".join(
            f"{k}={be._api_report[k]}" for k in ("order", "timestep", "parts")
            if k in be._api_report))
        if n_parts and not be.part_index:
            print("ERROR: this rig switches parts with PartOpacity curves but "
                  "this live2d-py build has no GetPartIds/SetPartOpacity, so "
                  "every variant would be drawn. Upgrade live2d-py (>=0.8.1 "
                  "is tested).", file=sys.stderr)
            return 1
        by_id = {s.pid: s for s in be.params}
        # Expressive ids the loaded model lacks (a rig file from another
        # build of the model) are dropped, as before; authored replay is not
        # filtered.
        absent = ({pids[i] for i in expressive} - set(by_id)
                  if static_frames is None else set())

        def frame_overrides(f):
            over, parts = frames[f]
            over = {p: v for p, v in over.items() if p not in absent}
            return over, parts

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
