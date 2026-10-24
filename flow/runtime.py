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
  mouth      Held at default, unless the event stream carries a "mouth" token
             (flow.director --tts: a sampled openness curve from the speech
             audio, flow.speech), which drives the mouth-open parameter and,
             in vowel mode, ParamA/I/U/E/O; under --lipsync auto the token
             names the parameters and their speaking ranges, calibrated from
             the rig's own talking motions (flow.mouth). Otherwise mouth
             form is left to the rig.
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
  --ckpt prior.pt        deterministic next-frame prior (flow.legacy.prior)
  --ckpt flowprior.pt    flow-matching prior (flow.legacy.flowprior --save), sampled
                         a chunk at a time; --seed picks the noise
  --events stats.json    event scheduler (flow.events, fitted --fit): gaze,
                         head, body and brow roles as fixate/saccade/hold
                         events, arms as cross-faded segments of the rig's
                         own motion files; --tokens dumps the event stream,
                         --direct merges flow.director's timed action tags;
                         --idle-profile calm (default) rests the body and
                         head and plays only calm motions, rarely, with hair
                         and accessories on the idle loop in between
                         (events.idle_view, underlay_token)
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
import contextlib
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
    from ..l2dprobe.transforms import protected, whole_model_params
    from ..l2dprobe.motions import (authored_motion, band_of, is_vowel,
                                    motion_files, physics_params,
                                    toggle_params)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from l2dprobe.analyze import group_of
    from l2dprobe.transforms import protected, whole_model_params
    from l2dprobe.motions import (authored_motion, band_of, is_vowel,
                                  motion_files, physics_params, toggle_params)

# Groups that are continuous by nature. Animators often park gaze or the head
# at an extreme and hold it, which toggle_params reads as a switch -- on
# aidang_2 it snapped ParamEyeBallX/Y and gaze stopped moving entirely. A head
# turn or a glance is never a part swap, so these are never snapped.
NEVER_SNAP = ("gaze", "head", "body")


def snappable(pid: str, rig_group: str) -> bool:
    """Whether a switch-like parameter may be snapped to its two states.

    Arm and hand rotations are continuous too, even when animators park them:
    aidang_2 snapped ParamArmForeLDAngel, ParamArmUpperLDAngel and others
    ("Angel" is the rig's spelling), and its arms moved 0.28x authored frame
    to frame; unsnapped they reached 0.48x. Arm swaps (…T, …Switch) stay
    snappable -- unsnapped, switch params sat in between 42% of the time.
    """
    low = pid.lower()
    arm = rig_group == "arm" or group_of(pid) == "arm"
    rotation = any(k in low for k in ("angle", "angel", "rotat"))
    return (rig_group not in NEVER_SNAP
            and group_of(pid) not in NEVER_SNAP
            and not (arm and rotation))


def face_kind(pid: str, rig_group: str, band: str) -> str:
    """"eye" (open, lid, blink, smile), "mouth" or "": the face parameters
    a Core geometry check may overrule the probe and the motions on.

    Brows are left out on purpose: Hiyori's ParamBrowLX/RX and Natori's
    ParamBrowLY/RY/RForm are clean deformers that are snapped today, and
    Natori's ParamBrowLForm is banded discrete; widening this to them would
    change both rigs. Gaze is never snapped anyway (NEVER_SNAP)."""
    low = pid.lower()
    if "eye" in low and (rig_group == "eyelid" or any(
            k in low for k in ("open", "lid", "blink", "smile"))):
        return "eye"
    if (band == "mouth" or rig_group == "mouth" or is_vowel(pid)
            or any(k in low for k in ("mouth", "lip"))):
        return "mouth"
    return ""


def _load_core(model_json):
    """A Core model for the geometry checks, or None (no Core, no moc, a moc
    the Core refuses). Split out so tests can hand in a fake."""
    try:
        try:
            from ..l2dprobe import core
        except ImportError:
            from l2dprobe import core
        if not core.available():
            return None
        return core.load(model_json)
    except Exception:  # noqa: BLE001
        return None


@contextlib.contextmanager
def _stdout_to_stderr():
    """Finding the Core imports live2d, whose banner goes to stdout, and the
    Core prints its version there from C; director --print-system writes
    the prompt to stdout. Both go to stderr, at the descriptor."""
    sys.stdout.flush()
    try:
        saved = os.dup(1)
    except OSError:
        yield
        return
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


_CONTINUOUS_CACHE: dict = {}


def continuous_face(model_json, pids, groups, bands) -> dict[str, str]:
    """{pid: evidence} for the eye and mouth parameters (face_kind) the
    Cubism Core shows are continuous deformers: sweeping them moves
    vertices smoothly and changes no drawable opacity
    (l2dprobe.core.continuous_deformer). Physics outputs are not judged.
    Empty without a Core or a loadable moc, which is the behaviour before
    this check existed."""
    cand = [p for p, g, b in zip(pids, groups, bands)
            if b != "physics" and face_kind(p, g, b)]
    if not cand:
        return {}
    try:
        st = os.stat(model_json)
        key = (os.path.abspath(model_json), st.st_mtime_ns, tuple(cand))
    except OSError:
        return {}
    if key in _CONTINUOUS_CACHE:
        return dict(_CONTINUOUS_CACHE[key])
    out = {}
    with _stdout_to_stderr():
        m = _load_core(model_json)
        if m is not None:
            try:
                from ..l2dprobe.core import continuous_deformer
            except ImportError:
                from l2dprobe.core import continuous_deformer
            for p in cand:
                try:
                    ok, why = continuous_deformer(m, p)
                except Exception:  # noqa: BLE001
                    continue
                if ok:
                    out[p] = why
    _CONTINUOUS_CACHE[key] = out
    return dict(out)


def reband(pids, kinds, groups, bands, model_json):
    """(bands, continuous): the probe's discrete band overruled for eye and
    mouth parameters the Core shows are continuous deformers.

    The probe classifies from pixels, and an eyelid closing over the iris
    can read as one jump or as mostly an opacity change (xuefeng's
    ParamEyeL/ROpen: categorical, so discrete, so no procedural blink and
    the authored 0/1 step blinks played instead). An eye open, lid or blink
    goes to the autonomic band, a mouth parameter to the mouth band. Nothing
    is moved into the expressive band: that would shift the clips' and the
    prior's columns. `continuous` is the evidence for every eye and mouth
    deformer, re-banded or not (snap_set uses it)."""
    cont = continuous_face(model_json, pids, groups, bands)
    out = list(bands)
    for i, (p, g, b) in enumerate(zip(pids, groups, bands)):
        if b != "discrete" or p not in cont:
            continue
        kind = face_kind(p, g, b)
        low = p.lower()
        if kind == "eye" and any(k in low for k in ("open", "lid", "blink")):
            out[i] = "autonomic"
        elif kind == "mouth":
            out[i] = "mouth"
    return out, cont


def snap_set(found, pids, groups, bands, rows, continuous=()):
    """{pid: found[pid]} for the rows to snap: toggle_params' switch-like
    parameters in the expressive band that snappable allows, less the eye
    and mouth parameters the Core shows are continuous deformers (an eye
    smile keyed 0/1 by the motions still eases)."""
    return {pids[i]: found[pids[i]] for i in rows
            if pids[i] in found and bands[i] == "expressive"
            and snappable(pids[i], groups[i])
            and pids[i] not in continuous}


def protected_from_hold(pid: str, rig_group: str) -> bool:
    """Angles and gaze are never held; see l2dprobe.transforms.protected."""
    return protected(pid, rig_group)


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
        from .legacy.prior import Prior
    except ImportError:
        from flow.legacy.prior import Prior
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
                 enter: float = 0.7, shape: str = "linear"):
        self.states = states
        self.ramp = max(int(ramp), 1)
        self.enter = enter
        # "minjerk" eases the ramp in and out instead of stepping velocity at
        # both ends; --events uses it, every other mode keeps the linear ramp.
        self.shape = shape
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
        if self.shape == "minjerk":
            p = p * p * p * (10 - 15 * p + 6 * p * p)
        return lo + p * (hi - lo)


def mouth_targets(pids, bands):
    """(index of the mouth-open parameter or None, {vowel: index}). The open
    parameter is a mouth-band id containing "open", ParamMouthOpenY first;
    the vowels are l2dprobe.motions.is_vowel ids (ParamA ... PARAM_O)."""
    mouth = [i for i, b in enumerate(bands) if b == "mouth"]
    opens = sorted((i for i in mouth if "open" in pids[i].lower()),
                   key=lambda i: (not pids[i].lower().endswith("openy"),
                                  pids[i]))
    vow = {pids[i].lower()[-1]: i for i in mouth if is_vowel(pids[i])}
    return (opens[0] if opens else None), vow


def mouth_frames(tok, pids, bands, hi, dflt, fps, n_frames, lo=None):
    """{parameter id: values per frame} from a "mouth" token: {"t", "fps",
    "open": [...], "vowels": {"a": [...]}?, "map": {...}?}. Openness 0 is
    the parameter's default (a rig's rest mouth is closed) and 1 its
    maximum. Sampled at this run's frame times; outside the curve the mouth
    is closed. A rig with no open parameter gets the openness on ParamA.

    With "map" (--lipsync auto on a rig whose motions talk, flow.mouth):
    exactly the parameters it names, each from its "lo" (rest) to its "hi"
    (the authored speaking extreme) along its "curve" ("open" or a vowel),
    clamped to the rig's range (`lo`, maximum)."""
    if tok.get("map"):
        return _mapped_mouth(tok, pids, hi, lo, fps, n_frames)
    op_i, vow = mouth_targets(pids, bands)
    src = float(tok.get("fps", fps))
    t0 = float(tok.get("t", 0.0))

    def at(curve):
        c = np.asarray(curve, np.float64)
        if src == fps and t0 == 0.0:
            out = np.zeros(n_frames)
            m = min(n_frames, len(c))
            out[:m] = c[:m]
            return out
        return np.interp(np.arange(n_frames) / fps - t0,
                         np.arange(len(c)) / src, c, left=0.0, right=0.0)

    out = {}
    curves = dict(tok.get("vowels") or {})
    if op_i is not None:
        curves.setdefault("open", tok["open"])
        idx = {"open": op_i, **vow}
    elif "a" in vow:
        curves.setdefault("a", tok["open"])
        idx = vow
    else:
        return out
    for k, c in curves.items():
        i = idx.get(k)
        if i is None:
            continue
        out[pids[i]] = (float(dflt[i])
                        + at(c) * (float(hi[i]) - float(dflt[i])))
    return out


def _sample_curve(curve, src, t0, fps, n_frames):
    c = np.asarray(curve, np.float64)
    if src == fps and t0 == 0.0:
        out = np.zeros(n_frames)
        m = min(n_frames, len(c))
        out[:m] = c[:m]
        return out
    return np.interp(np.arange(n_frames) / fps - t0,
                     np.arange(len(c)) / src, c, left=0.0, right=0.0)


def _mapped_mouth(tok, pids, hi, lo, fps, n_frames):
    src = float(tok.get("fps", fps))
    t0 = float(tok.get("t", 0.0))
    curves = {"open": tok["open"], **(tok.get("vowels") or {})}
    index = {p: i for i, p in enumerate(pids)}
    out = {}
    for p, m in tok["map"].items():
        i = index.get(p)
        if i is None or m.get("curve") not in curves:
            continue
        v = float(m["lo"]) + _sample_curve(curves[m["curve"]], src, t0, fps,
                                           n_frames) * (float(m["hi"])
                                                        - float(m["lo"]))
        if lo is not None:
            v = np.clip(v, float(lo[i]), float(hi[i]))
        out[p] = v
    return out


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


def smooth_causal(traj: np.ndarray, tau: float, fps: float) -> np.ndarray:
    """Critically damped follower: two cascaded one-pole lags, past frames only.

    The v3 flow prior's large-scale motion was about right but carried
    high-frequency wobble: jerk 42-248x authored on Hiyori/Natori. A
    non-causal blur fixed it entirely, which says the excess is all high
    frequency; this is the real-time version. tau 0.08 s (~160 ms lag) took
    jerk to 2.3x (Hiyori) and 13x (Natori) with the amount of motion
    unchanged. A stopgap until the model itself generates smooth chunks.
    """
    if tau <= 0 or len(traj) == 0:
        return traj
    a = 1.0 - np.exp(-1.0 / (tau * fps))
    y1 = traj[0].astype(np.float64).copy()
    y2 = y1.copy()
    out = np.empty_like(traj)
    for f in range(len(traj)):
        y1 += a * (traj[f] - y1)
        y2 += a * (y1 - y2)
        out[f] = y2
    return out


def denorm(n: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """[-1,1] back to the rig's own units."""
    return lo + (np.clip(n, -1, 1) + 1) * 0.5 * (hi - lo)


def rig_columns(rig, model_json, clips=None, log=print):
    """The rig file's columns and bands, and which rows are expressive.

    Split out of main so flow.director binds exactly the roles the runtime
    will (its prompt offers only tags the rig can show) without a GL backend.
    `expressive` is empty when the rig has none; the clips are then not read.
    """
    z = np.load(rig, allow_pickle=True)
    pids = [str(x) for x in z["pid"]]
    kinds = [str(x) for x in z["kind"]]
    groups = [str(x) for x in z["group"]]
    phys, has_phys = physics_params(model_json)
    bands = [band_of(p, k, g, phys) for p, k, g in zip(pids, kinds, groups)]
    probe_bands = bands
    bands, continuous = reband(pids, kinds, groups, bands, model_json)
    rebanded = {pids[i]: (probe_bands[i], b) for i, b in enumerate(bands)
                if b != probe_bands[i]}
    if rebanded:
        log("continuous deformers in the Core, not switches: "
            + ", ".join(f"{p} {a}->{b}" for p, (a, b)
                        in sorted(rebanded.items())))
    expressive = [i for i, b in enumerate(bands) if b == "expressive"]
    log(f"{len(pids)} params: "
        + ", ".join(f"{b} {bands.count(b)}" for b in
                    ("expressive", "physics", "autonomic", "mouth", "discrete")))
    if expressive and clips and os.path.exists(clips):
        # Clips record which rig rows they hold. Built under older band rules
        # (vowels counted as expressive until recently) their columns no longer
        # match, and seeding or anchoring from them would misalign every
        # parameter. Their rows are what any checkpoint trained on them saw, so
        # they win; parameters now banded otherwise are just not written.
        rows = np.load(clips, allow_pickle=True)["param_row"].astype(int)
        if list(rows) != expressive:
            gone = [pids[i] for i in rows if bands[i] != "expressive"]
            log(f"clips use {len(rows)} columns, current bands give "
                f"{len(expressive)}; following the clips"
                + (f", not writing {', '.join(gone)}" if gone else "")
                + ". Regenerate clips with l2dprobe.motions to align.")
            expressive = [int(i) for i in rows]
    return {"pids": pids, "kinds": kinds, "groups": groups,
            "lo": z["minimum"].astype(np.float32),
            "hi": z["maximum"].astype(np.float32),
            "default": z["default"].astype(np.float32),
            "descriptor": z["descriptor"].astype(np.float32),
            "phys": phys, "has_phys": has_phys, "bands": bands,
            "expressive": expressive, "continuous": continuous,
            "rebanded": rebanded}


def held_params(rig, probe_dir, pids, groups, expressive, hold="",
                glob_=True, warn=True):
    """The whole-model transform parameters main holds still: --hold's ids
    plus, with glob_, those the probe output finds (whole_model_params),
    limited to the expressive columns. Shared with flow.gestures, which
    must not describe a held parameter as a gesture's motion."""
    held = set(p for p in hold.split(",") if p)
    if glob_:
        rig_name = os.path.basename(rig).rsplit(".npz", 1)[0]
        found = whole_model_params(probe_dir, rig_name,
                                   groups=dict(zip(pids, groups)))
        if found is None:
            if warn:
                print(f"WARNING: no probe output {probe_dir}/rig_{rig_name}"
                      ".npz; whole-model transform parameters are not held "
                      "(pass --probe-dir, or list them with --hold)",
                      file=sys.stderr)
        else:
            # Angles and gaze are protected inside transforms.decide.
            held |= set(found)
    return held & {pids[i] for i in expressive}


def bind_events(events, model_json, pids, allowed, lo, hi, motions, fps,
                binding=None, no_snap=False, pool_only=False, log=print):
    """(stats, roles, switch roles) for an --events run: the fitted
    statistics, adapted to the rig's own motions unless pool_only, and the
    roles bound on the `allowed` (expressive) pids. `motions` as
    authored_motions; lo/hi the rig's ranges, indexed like pids.

    Shared with flow.director, which needs the same bound roles and adapted
    centres before it writes the prompt."""
    try:
        from . import events as ev
    except ImportError:
        from flow import events as ev
    stats = ev.load_stats(events)
    roles = ev.bind(pids, allowed, ev.load_binding(binding))
    # Role columns the rig's own motions use as switches go to the
    # gesture layer (see events.drop_switch_roles); --no-snap keeps them.
    roles, sw_roles = ev.drop_switch_roles(
        roles, {} if no_snap else toggle_params(model_json, fps))
    if sw_roles:
        log("events: switch-like role columns left to the rig's own "
            "motions: " + ", ".join(f"{r}={v[0]}" for r, v
                                    in sorted(sw_roles.items())))
    for r, (p, sg, src) in sorted(roles.items()):
        log(f"  role {r:<8} -> {p}{'' if sg > 0 else ' (sign -1)'}  "
            f"[{src}]")
    missing = sorted(set(ev.ALL_ROLES) - set(roles) - set(sw_roles))
    if missing:
        log(f"events: unbound roles (no standard id; --binding to add): "
            f"{', '.join(missing)}")
    if not pool_only:
        # Body statistics shrunk toward the rig's own motion files (the
        # files the gestures replay): see events.adapt. Done before the
        # start pose so that --tokens-in replays bit for bit.
        own = ev.fit_clips(ev.motion_clips(
            motions, roles, dict(zip(pids, lo.tolist())),
            dict(zip(pids, hi.tolist())), fps))
        stats = ev.adapt(stats, own)
        for g, w in stats.get("adapted", {}).items():
            G = stats["groups"][g]
            O = (own or {}).get("groups", {}).get(g, {})
            log(f"events: {g} adapted to the rig's own motions "
                f"(weights {w}; own holds {O.get('n_hold', 0)} "
                f"+{O.get('n_censored', 0)} censored, "
                f"{O.get('n_unanimated', 0)} unanimated clips, moves "
                f"{O.get('n_move', 0)}): hold {G['hold'][0]:.2f}s "
                f"move {G['move'][0]:.2f}s p_chain "
                f"{G.get('p_chain', 0):.2f}")
    return stats, roles, sw_roles


def motion_groups(model_json):
    """{motion basename: model3 motion group} ("Idle", "TapBody", ...)."""
    try:
        with open(model_json, encoding="utf-8") as fh:
            groups = json.load(fh).get("FileReferences", {}).get("Motions",
                                                                 {})
    except Exception:  # noqa: BLE001
        return {}
    return {os.path.basename(e["File"]): g for g, es in groups.items()
            for e in es if isinstance(e, dict) and e.get("File")}


def calm_library(gest, model_json, log=print):
    """(calm gesture_library entries, settle motion name) for the calm idle
    profile: the motions events.calm_motions picks, logged with every
    motion's mean speed (normalised units per frame over the parameters a
    gesture writes), the only view the log gives of a rig's motions. The
    settle motion is the one a directed gesture returns to: the idle loop
    the base state was taken from where there is one."""
    try:
        from . import events as ev
    except ImportError:
        from flow import events as ev
    energy = {g["name"]: float(np.mean(g["vel"])) for g in gest if "vel" in g}
    names, how = ev.calm_motions([g["name"] for g in gest],
                                 motion_groups(model_json), energy)
    calm = [g for g in gest if g["name"] in names]
    log(f"events: idle profile calm; gestures from {len(calm)} of "
        f"{len(gest)} motions (by {how}): "
        + ", ".join(f"{g['name'].replace('.motion3.json', '')}"
                    f"{'*' if g['name'] in names else ''} "
                    f"{energy.get(g['name'], float('nan')):.4f}"
                    for g in gest)
        + "  (* calm; mean speed/frame)")
    settle = next((n for n in names if "idle" in n.lower()), names[0]
                  if names else None)
    return calm, settle


def event_roles(model_json, rig, events, clips=None, binding=None, fps=30.0,
                no_snap=False, pool_only=False, log=print):
    """bind_events for a rig on disk, as main does it for --events: what
    flow.director calls to know which roles a run will have."""
    col = rig_columns(rig, model_json, clips, log)
    pids, ex = col["pids"], col["expressive"]
    return bind_events(events, model_json, pids, [pids[i] for i in ex],
                       col["lo"], col["hi"], authored_motions(model_json, fps),
                       fps, binding, no_snap, pool_only, log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".model3.json")
    ap.add_argument("--rig", required=True, help="rigs/<name>.npz")
    ap.add_argument("--probe-dir", default="out",
                    help="probe outputs (rig_<name>.npz) used to find the "
                         "whole-model transform parameters to hold still")
    ap.add_argument("--no-hold-global", action="store_true",
                    help="let the model drive whole-model position, scale and "
                         "rotation (it moved characters off the canvas)")
    ap.add_argument("--hold", default="",
                    help="extra comma-separated parameter ids to hold still")
    ap.add_argument("--ckpt", default=None,
                    help="prior checkpoint: the deterministic prior "
                         "(flow.legacy.prior) or a flow-matching one (flow.legacy.flowprior "
                         "--save), told apart by their config keys; not "
                         "needed for --static/--neutral/--behaviors")
    ap.add_argument("--behaviors", default=None, metavar="STATS.json",
                    help="hand-made behaviour baseline (flow.behaviors) with "
                         "these fitted statistics, or 'default'. Drives head, "
                         "body, gaze and brow roles; every other expressive "
                         "parameter is held at the idle's first frame")
    ap.add_argument("--events", default=None, metavar="STATS.json",
                    help="event-based motion (flow.events) with these fitted "
                         "statistics, or 'default'. Roles bind by standard "
                         "ID or --binding only (no descriptor fallback)")
    ap.add_argument("--events-pool-only", action="store_true",
                    help="--events: use the pooled statistics as fitted, "
                         "without adapting the body to the rig's own motion "
                         "files (events.adapt)")
    ap.add_argument("--idle-profile", default="calm",
                    choices=("calm", "authored"),
                    help="--events: how the scheduled idle moves. calm "
                         "[default]: long body and head holds, small sway, a "
                         "calm motion (idle/main/home, or the model's Idle "
                         "group) every ~10-40 s, hair and accessories on the "
                         "idle loop between them (events.idle_view, "
                         "underlay_token); authored: the "
                         "fitted statistics as is, every motion file back "
                         "to back (the scheduler before the calm profile)")
    ap.add_argument("--tokens", default=None, metavar="OUT.json",
                    help="--events: write the scheduled event tokens here")
    ap.add_argument("--tokens-in", default=None, metavar="IN.json",
                    help="--events: render these event tokens instead of "
                         "scheduling (a --tokens file, or a list of tokens)")
    ap.add_argument("--direct", default=None, metavar="ACTIONS.json",
                    help="--events: merge these timed action tags (written "
                         "by flow.director) into the scheduled idle tokens; "
                         "a directed move overrides the idle on the role "
                         "groups it touches for as long as it lasts")
    ap.add_argument("--no-idle", action="store_true",
                    help="--events: schedule blinks only (no idle head, body, "
                         "gaze, brow moves or gestures), so --direct tags are "
                         "seen alone; idle sway hid them on aidang_2")
    ap.add_argument("--binding", default=None, metavar="FILE.json",
                    help="--behaviors/--events role binding override: "
                         '{"head_x": "ParamFoo"} or {"head_x": ["ParamFoo", -1]}')
    ap.add_argument("--static-fade", type=float, default=0.5,
                    help="--static: seconds to cross-fade from one motion "
                         "file into the next (Live2D fades too); 0 cuts")
    ap.add_argument("--smooth", type=float, default=0.08,
                    help="flow checkpoint: seconds of causal smoothing on the "
                         "generated motion (~2x this in lag); 0 disables")
    ap.add_argument("--keep", type=int, default=None,
                    help="flow checkpoint: frames committed per sampled chunk "
                         "(default half the chunk; the full chunk pulsed at "
                         "the chunk rate)")
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
    if not (a.static or a.neutral or a.ckpt or a.behaviors or a.events):
        ap.error("--ckpt is required unless --static, --neutral, "
                 "--behaviors or --events")
    if a.events and a.behaviors:
        ap.error("--events and --behaviors are separate sources; pick one")
    if a.direct and (a.tokens_in or not a.events):
        ap.error("--direct merges into a scheduled --events run; a "
                 "--tokens-in stream is already merged")
    if not a.dump_only:
        _, ensure_display = _backend()
        ensure_display(["-m", f"{__package__ or 'flow'}.runtime"]
                       + (argv if argv is not None else sys.argv[1:]))

    rng = np.random.default_rng(a.seed)
    col = rig_columns(a.rig, a.model, a.clips)
    pids, kinds, groups = col["pids"], col["kinds"], col["groups"]
    lo, hi, dflt = col["lo"], col["hi"], col["default"]
    desc_all = col["descriptor"]
    phys, has_phys = col["phys"], col["has_phys"]
    bands, expressive = col["bands"], col["expressive"]
    deformers = col["continuous"]
    if not expressive:
        print("no expressive parameters", file=sys.stderr)
        return 1

    # Whole-model transforms (ParamAllX, PARAM_moveY, zongtiZ ...) are for
    # staging. Driven freely they carried characters off the canvas, so they
    # are held: at the idle's first frame when it sets them, else at default.
    held = held_params(a.rig, a.probe_dir, pids, groups, expressive, a.hold,
                       not a.no_hold_global and not (a.static or a.neutral))
    if held:
        print(f"holding {len(held)} whole-model parameters still: "
              + ", ".join(sorted(held)))

    model = desc = flow = None
    window = 48
    use_model = bool(a.ckpt) and not (a.static or a.neutral or a.behaviors
                                      or a.events)
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
                from .legacy.flowprior import FlowStream, load_checkpoint
            except ImportError:
                from flow.legacy.flowprior import FlowStream, load_checkpoint
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
    motions = authored_motions(a.model, a.fps)
    band_by_id = dict(zip(pids, bands))
    events_on = bool(a.events) and not (a.static or a.neutral)
    pid_index0 = set(pids)
    tokens = None
    if events_on:
        try:
            from . import events as ev
        except ImportError:
            from flow import events as ev
        stats, roles, sw_roles = bind_events(
            a.events, a.model, pids, [pids[i] for i in expressive], lo, hi,
            motions, a.fps, a.binding, a.no_snap, a.events_pool_only)
        stats["idle"]["profile"] = a.idle_profile
        start = {r: stats["roles"][r]["center"] for r in roles}
        settle = None
        if a.tokens_in:
            with open(a.tokens_in, encoding="utf-8") as fh:
                blob = json.load(fh)
            tokens = blob["tokens"] if isinstance(blob, dict) else blob
            print(f"events: rendering {len(tokens)} tokens from {a.tokens_in}")
        else:
            gkeep0 = base_keep({p for p, _, _ in roles.values()} | held,
                               phys, band_by_id)
            found_sw = ({} if a.no_snap else toggle_params(a.model, a.fps))
            sw = {p: (v[1], v[2]) for p, v in snap_set(
                found_sw, pids, groups, bands, expressive,
                deformers).items()}
            gest = ev.gesture_library(
                motions, lambda p: p in pid_index0 and gkeep0(p),
                dict(zip(pids, lo.tolist())), dict(zip(pids, hi.tolist())),
                sw, a.fps)
            if a.idle_profile == "calm" and gest:
                gest, settle = calm_library(gest, a.model)
            tokens = ev.schedule(stats, list(roles), a.seconds, rng, gest,
                                 start)
            und = ev.underlay_token(stats, settle, a.seconds)
            if und is not None:
                tokens.insert(0, und)
            if a.no_idle:
                tokens = [k for k in tokens if k["type"] == "blink"]
            if a.direct:
                try:
                    from . import director as dr
                except ImportError:
                    from flow import director as dr
                with open(a.direct, encoding="utf-8") as fh:
                    blob = json.load(fh)
                acts = blob["actions"] if isinstance(blob, dict) else blob
                # Own generator: the idle stream stays what the same --seed
                # schedules without tags, whatever the tags draw.
                directed = dr.realize(
                    acts, stats, list(roles),
                    {n: m["frames"] / a.fps for n, m in motions},
                    np.random.default_rng([a.seed, 1]), start,
                    settle=settle)
                n_idle = len(tokens)
                tokens = dr.merge(tokens, directed)
                print(f"director: {len(acts)} actions -> {len(directed)} "
                      f"tokens; {n_idle + len(directed) - len(tokens)} idle "
                      "tokens overridden")
                if isinstance(blob, dict) and blob.get("mouth"):
                    # After the merge (and the --no-idle filter above): the
                    # mouth curve is no move, nothing may drop it. Read from
                    # the JSON as written, so --tokens-in replays the same
                    # floats.
                    tokens.append({"t": 0.0, "type": "mouth",
                                   "role": "mouth", "src": "director",
                                   **blob["mouth"]})
        gen, blink_requests = ev.render(tokens, list(roles), a.seconds, a.fps,
                                        start)
        traj = np.tile(default_n.astype(np.float32), (n_frames, 1))
        col_of = {pids[i]: c for c, i in enumerate(expressive)}
        for r, (p, sg, _) in roles.items():
            traj[:, col_of[p]] = np.clip(sg * gen[r][:n_frames], -1, 1)
        kinds_n = {}
        for k in tokens:
            kinds_n[k["type"]] = kinds_n.get(k["type"], 0) + 1
        print(f"events: {len(roles)} roles from "
              f"{'defaults' if a.events == 'default' else a.events}; tokens "
              + ", ".join(f"{k} {v}" for k, v in sorted(kinds_n.items())))
        if a.tokens:
            os.makedirs(os.path.dirname(os.path.abspath(a.tokens)),
                        exist_ok=True)
            with open(a.tokens, "w", encoding="utf-8") as fh:
                json.dump({"seconds": a.seconds, "fps": a.fps,
                           "seed": a.seed,
                           "idle_profile": (None if a.tokens_in
                                            else a.idle_profile),
                           "roles": {r: [p, sg] for r, (p, sg, _)
                                     in roles.items()},
                           "tokens": tokens}, fh, indent=0)
            print(f"tokens -> {a.tokens}")
    elif a.behaviors and not (a.static or a.neutral):
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
        keep = a.keep or max(1, cfg["chunk"] // 2)
        stream = FlowStream(fmodel, desc, seed, steps=steps, seed=a.seed,
                            device=device, keep=keep)
        print(f"flow: committing {keep} of every {cfg['chunk']} sampled frames")
        traj = stream.generate(n_frames)
        if a.smooth > 0:
            traj = smooth_causal(traj, a.smooth, a.fps).astype(np.float32)
            print(f"flow: causal smoothing tau {a.smooth:g}s "
                  f"(~{2 * a.smooth * 1000:.0f} ms lag)")
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

    n_parts = len({p for _, m in motions for p in m["parts"]})
    if n_parts and not has_pose(a.model):
        print(f"{n_parts} parts are switched by PartOpacity curves and the rig "
              "has no pose3 file; applying them directly (Live2D Native would "
              "ignore them and draw every variant)")
    # Under --behaviors only the bound roles are driven; the base state holds
    # every other expressive parameter at the idle's first frame. --events
    # starts from the same base and lets gestures move the rest.
    model_driven = ({p for p, _, _ in roles.values()}
                    if a.behaviors or events_on
                    else {pids[i] for i in expressive}) - held
    base_name, base_params, base_parts = base_state(
        motions, base_keep(model_driven, phys, band_by_id))
    if base_name and not (a.static or a.neutral):
        print(f"base state from {base_name}: {len(base_parts)} part "
              f"opacities, {len(base_params)} held parameters")

    mouth = {}
    mtok = next((k for k in tokens if k["type"] == "mouth"),
                None) if events_on else None
    if mtok is not None:
        mouth = mouth_frames(mtok, pids, bands, hi, dflt, a.fps, n_frames,
                             lo=lo)
        print("mouth: lip sync on " + (", ".join(mouth) or
              "nothing (the rig has no mouth-open or ParamA parameter)"))

    static_frames = None
    static_fade = 0
    if a.static and motions:
        # Replay the motion files themselves, every curve in the rig's own
        # units. The clips hold only the expressive band, normalised, so a
        # replay from them drops part opacities, discrete parameters, and any
        # parameter the probe never kept -- the arm switches among them.
        seq = [(m, t) for _, m in motions for t in range(m["frames"])]
        static_frames = [seq[f % len(seq)] for f in range(n_frames)]
        # Live2D fades one motion into the next instead of cutting. Without
        # it every file boundary popped to a new pose: 2-44 jumps per 20 s
        # preview on the server, read as blinking.
        static_fade = int(round(a.static_fade * a.fps))
        print(f"STATIC: replaying {len(motions)} motion files curve for curve "
              f"({len(seq) / a.fps:.1f}s incl. part opacities)")
    elif a.static:
        print(f"STATIC: no motion files found; replaying {len(traj)} frames of "
              "the expressive band from the clips, model bypassed")

    gesture_frames = None
    if events_on:
        # Gestures write what the base state may hold (not roles, physics,
        # autonomic, mouth) and nothing held still as a whole-model transform.
        role_ids = {p for p, _, _ in roles.values()}
        gkeep = base_keep(role_ids | held, phys, band_by_id)
        pid_set = set(pids)
        # What an underlay token may move, from the rig and its motions
        # alone, so a --tokens-in replay renders what the run rendered.
        loose = []
        und = next((k for k in tokens if k["type"] == "underlay"), None)
        if und is not None:
            sw_all = toggle_params(a.model, a.fps)
            loose = ev.loose_params(
                [pids[i] for i in expressive], [groups[i] for i in expressive],
                sw_all, lambda p: p in pid_set and gkeep(p)
                and np.ptp(dict(motions).get(und["motion"], {}).get(
                    "params", {}).get(p, [0.0])) > 0)
            print(f"events: underlay {und['motion']} x{und['gain']:g} during "
                  f"rests on {len(loose)} loose parameters: "
                  + (", ".join(loose) or "none"))
        player = ev.GesturePlayer(
            {n: m for n, m in motions}, lambda p: p in pid_set and gkeep(p),
            base_params, base_parts,
            {p: float(v) for p, v in zip(pids, dflt)}, a.fps, loose)
        gesture_frames = player.frames(tokens, n_frames)
        gesture_cols = []
        for col, i in enumerate(expressive):
            p = pids[i]
            if p in role_ids or p in held:
                continue
            gesture_cols.append(col)
            vals = np.array([gp.get(p, base_params.get(p, dflt[i]))
                             for gp, _ in gesture_frames], np.float32)
            traj[:, col] = np.clip((vals - lo[i]) / span_e[col] * 2 - 1, -1, 1)
        n_g = sum(k["type"] == "gesture" for k in tokens)
        print(f"events: {n_g} gesture segments from {len(motions)} motion "
              "files (arms will score as copying against the rig's own "
              "clips; novelty is judged on the role columns)")

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
        live = snap_set(found, pids, groups, bands, expressive, deformers)
        if live:
            if events_on:
                # Gesture seams flip arm swaps that authored clips (one arm
                # variant per motion) never flip inside a clip; a 6-frame
                # linear ramp there cost Natori 7x authored arm jerk. Under
                # --events a flip eases over the gesture fade instead.
                fade_n = int(round(stats["gesture"]["fade"] * a.fps))
                switches = Switches({p: (v[1], v[2]) for p, v in live.items()},
                                    max(a.snap_ramp, fade_n), shape="minjerk")
            else:
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
    fade_from = None
    pid_index = {p: i for i, p in enumerate(pids)}
    expr_ids = {pids[i] for i in expressive}
    for f in range(n_frames):
        if static_frames is not None:
            m, t = static_frames[f]
            if t == 0:
                seg_id += 1
                fade_from = (frames[-1] if frames else None)
            segment[f] = max(seg_id, 0)
            over = {p: float(v[t]) for p, v in m["params"].items()}
            parts = {p: float(v[t]) for p, v in m["parts"].items()}
            if static_fade > 0 and fade_from is not None and t < static_fade:
                w = 0.5 - 0.5 * np.cos(np.pi * (t + 1) / (static_fade + 1))
                po, pp = fade_from
                # A parameter the new motion leaves alone returns to its
                # default once the fade ends, so fade toward that rather than
                # hold it and drop at the end. Parts keep their opacity.
                for p, v in po.items():
                    over[p] = (1 - w) * v + w * over.get(
                        p, float(dflt[pid_index[p]]) if p in pid_index else v)
                # Part visibility is 0 or 1 in authored motion; blending it
                # drew both arm variants half visible for the whole fade.
                # Switch at the midpoint instead.
                if w < 0.5:
                    for p, v in (pp or {}).items():
                        parts[p] = v
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
                if bands[i] != "expressive" or pids[i] in held:
                    applied[f, col] = (
                        (over[pids[i]] - lo[i]) / span_x[col] * 2 - 1
                        if pids[i] in over else default_n[col])
                    applied_mask[f, col] = False
                    continue
                v = float(vals[col])
                if switches is not None and pids[i] in switches.states:
                    v = switches(pids[i], v)
                over[pids[i]] = v
                applied[f, col] = (v - lo[i]) / span_x[col] * 2 - 1
            for p, v in mouth.items():
                over[p] = float(v[f])
            parts = base_parts
            if gesture_frames is not None:
                # Discrete and other non-expressive parameters the gesture
                # keys (arm swaps among them) go with their part opacities.
                gp, parts = gesture_frames[f]
                for p, v in gp.items():
                    if p not in expr_ids:
                        over[p] = v
                for col in gesture_cols:
                    if pids[expressive[col]] not in player.keyed[f]:
                        applied_mask[f, col] = False
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
                else f"events seed={a.seed}" if events_on
                else f"flow steps={flow[3]} seed={a.seed}" if flow
                else f"prior anchor={a.anchor:g} reseed={a.reseed:g}")
        os.makedirs(os.path.dirname(os.path.abspath(a.dump)), exist_ok=True)
        np.savez_compressed(
            a.dump, traj=applied, mask=applied_mask, segment=segment,
            pid=np.array([pids[i] for i in expressive], dtype=object),
            param_row=np.asarray(expressive, np.int32),
            fps=np.float32(a.fps), part_id=np.array(part_ids, dtype=object),
            part_opacity=opac, mode=np.array(mode),
            **({"mouth": np.stack([v for v in mouth.values()], 1).astype(
                np.float32), "mouth_pid": np.array(list(mouth), dtype=object)}
               if mouth else {}))
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
