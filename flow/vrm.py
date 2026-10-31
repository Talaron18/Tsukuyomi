#!/usr/bin/env python3
"""
A VRM avatar in the pipeline: the idle, poses and voice on VRM parameters,
streamed as body states.

flow.stream runs a Live2D rig's frame loop from an EventRig (columns, roles,
gestures, base state, autonomic). VRMRig gives MotionStream the same
surface for a VRM body (flow.vrmbody), so the idle scheduler, the director's
realize/merge, the pose layer, blinks, breath and lip sync all run
unchanged; each frame's parameter dict becomes a body state
(VRMBody.state: bone rotations, look-at, expressions) instead of a GL
render. The browser draws it (flow/web/vrm.js); offline, flow.vrmrender
does in headless Chromium.

  roles     the event idle's roles drive canonical variables, in the units
            the calm profile was fitted in: a role's normalised [-1, 1] is
            the Cubism standard parameter's range, so head_x 1 = ParamAngleX
            30 = head.yaw 30 deg (ROLE_MAP; Live2D's ranges by convention,
            flow.body.DECLARED, not the VRM's joint limits, which are wider)
            -- but head_y 1 = 15 deg of pitch (Live2D draws pitch shallower;
            ROLE_MAP's comment has the measurement).
            Brow roles without a VRM shape (slide, angle) are not bound.
  arms      no gestures (VRM avatars ship no authored motions here): the
            arms rest relaxed (flow.vrmbody's rest pose, out of the T-pose)
            and move only by poses.
  breath    the autonomic breath (0..1 at 0.25 Hz) on the "breath"
            parameter: shoulders up and the upper chest back, calibrated in
            the body file so the shoulders rise BREATH_TARGET_HU, as every
            Live2D rig's breath does.
  mouth     the voice's mouth track with vowel shares (speech.mouth_track
            mode "vowels": VOICEVOX moras, else the formant estimate) on
            viseme.a/i/u/e/o -> the VRM presets aa/ih/ou/ee/oh.
  record    states.json: {"fps", "body", "vrm", "states": [state per
            frame]} (and "t0": the stream time of frame 0) -- what
            flow.vrmrender renders and what a live session records.

  search    --gesture-search (flow.gesture_search): real co-speech motion
            retrieved per sentence from a BEAT2 library, played by a
            GestureLayer under the poses and moves (MotionStream.gsearch);
            nothing of it runs, or is imported, without the flag.

Offline (flow.director --avatar vrm) a reply is planned as on Live2D --
tags or pose markers, TTS timing, subtitles -- and the stream is run for its
length into states.json; --render makes render/preview.mp4 and demo.mp4.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

try:
    from . import body as fb
    from . import events as ev
    from . import stream as fs
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import body as fb
    from flow import events as ev
    from flow import stream as fs

# role -> (canonical variable, sign, units at normalised 1). Signs as the
# Live2D standard ids relate to the canonical variables (flow.body's
# DECLARED / SIGN, Hiyori's measured file): ParamAngleZ + rolls clockwise
# as seen, head.tilt + counter-clockwise, so head_z -> -head.tilt.
# head_y 1 = 15 deg of pitch, not ParamAngleY's nominal 30: Live2D draws
# its pitch shallower than its yaw. Measured on the rigs' body files (the
# mouth's vertical shift at ParamAngleY +-30 over its sideways shift at
# ParamAngleX 30): Hiyori 0.70 / 0.39 (up / down), Natori 1.19 / 0.98,
# lafei 0.46 / 0.48, xuefeng 0.45 / 0.48, median 0.48; on a VRM head the
# same ratio is a pitch of 14 deg (Seed-san 15 deg: 0.51, three-vrm-girl
# 0.50). At 30 the fitted idle's head_y (p2 -1.0, its downward tail the
# long one) bent the head 18 deg down at its 2nd percentile.
ROLE_MAP = {
    "head_x": ("head.yaw", 1.0, 30.0),
    "head_y": ("head.pitch", 1.0, 15.0),
    "head_z": ("head.tilt", -1.0, 30.0),
    "body_x": ("torso.yaw", 1.0, 10.0),
    "body_y": ("torso.pitch", 1.0, 10.0),
    "body_z": ("torso.tilt", -1.0, 10.0),
    "gaze_x": ("gaze.x", 1.0, 1.0),
    "gaze_y": ("gaze.y", 1.0, 1.0),
    "brow_l": ("brow.L.lift", 1.0, 1.0),
    "brow_r": ("brow.R.lift", 1.0, 1.0),
    "brow_lf": ("brow.L.form", 1.0, 1.0),
    "brow_rf": ("brow.R.form", 1.0, 1.0),
}


# --avatar both: the reply's pose markers are written in the Live2D body's
# units (its prompt), where a canonical angle is the Cubism parameter's
# value (head.pitch -20 = ParamAngleY -20). On the VRM that angle is
# ROLE_MAP's: a Live2D marker scaled by role unit / Cubism range.
L2D_SCALE = {var: unit / deg for _r, (var, _sg, unit) in ROLE_MAP.items()
             for n, _u, _sid, deg in fb.DECLARED
             if n == var and deg and abs(unit / deg - 1.0) > 1e-9}


def from_live2d(acts):
    """acts (parsed on the Live2D body) with their pose targets in VRM
    units (L2D_SCALE); landmark references are left as written."""
    out = []
    for x in acts:
        tg = (x.get("arg") or {}).get("targets") if x.get("tag") == "pose" \
            else None
        if tg and any(k in L2D_SCALE and isinstance(v, (int, float))
                      for k, v in tg.items()):
            tg = {k: (v * L2D_SCALE[k] if k in L2D_SCALE and
                      isinstance(v, (int, float)) else v)
                  for k, v in tg.items()}
            x = dict(x, arg=dict(x["arg"], targets=tg))
        out.append(x)
    return out


class _NoGestures:
    """GesturePlayer's surface with nothing to play."""

    def at(self, tf, cur, prev, und):
        return {}, None, None

    def underlay(self, tok):
        return None


class VRMRig:
    """An EventRig for a VRM body (module docstring): what MotionStream and
    flow.live's Performer read from a rig."""

    def __init__(self, bm, events="default", fps=30.0, idle_profile="calm",
                 log=print):
        self.body = bm
        self.fps = float(fps)
        self.breath_gain = 1.0
        stats = ev.load_stats(events)
        stats["idle"]["profile"] = idle_profile
        self.stats = stats
        params = bm.params
        roles = {}
        for r, (var, sg, _u) in ROLE_MAP.items():
            if var in params:
                roles[r] = (var, sg, "vrm")
        self.roles = roles
        log("vrm: roles " + ", ".join(f"{r}->{v[0]}" for r, v in
                                      roles.items()))
        self.start = {r: stats["roles"][r]["center"] for r in roles}
        self.pids = list(params)
        role_pid = {v[0]: ROLE_MAP[r][2] for r, v in roles.items()}
        lo, hi = [], []
        for p in self.pids:
            if p in role_pid:
                lo.append(-role_pid[p])
                hi.append(role_pid[p])
            else:
                lo.append(params[p]["min"])
                hi.append(params[p]["max"])
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        self.dflt = np.array([params[p]["default"] for p in self.pids], float)
        self.bands = [("autonomic" if p == "breath" or p.endswith(".open")
                       and p.startswith("eye.") else params[p]["band"])
                      for p in self.pids]
        self.expressive = [i for i, p in enumerate(self.pids)
                           if p in role_pid]
        ex = self.expressive
        self.span_e = np.maximum(self.hi[ex] - self.lo[ex], 1e-6)
        self.default_n = np.clip((self.dflt[ex] - self.lo[ex]) / self.span_e
                                 * 2 - 1, -1, 1)
        self.col_of = {self.pids[i]: c for c, i in enumerate(ex)}
        self.role_cols = [(r, self.col_of[p], float(sg))
                          for r, (p, sg, _) in roles.items()]
        self.gesture_cols = []
        self.in_band = [(c, i) for c, i in enumerate(ex)]
        self.expr_ids = {self.pids[i] for i in ex}
        self.held = set()
        self.gest = []
        self.settle = None
        self.motions = []
        self.motion_seconds = {}
        self.player = _NoGestures()
        # Everything at rest but what the autonomic layer writes (blink,
        # breath): MotionStream writes those only where nothing else did.
        auto = {p for p, b in zip(self.pids, self.bands) if b == "autonomic"}
        self.base_params = {p: float(params[p]["default"])
                            for p in self.pids if p not in auto}
        self.base_parts = {}
        self.eye_idx = [i for i, p in enumerate(self.pids)
                        if p.startswith("eye.") and p.endswith(".open")]
        self.breath_idx = [i for i, p in enumerate(self.pids)
                           if p == "breath"]

    def switches(self):
        return None

    def mouth_curves(self, tok, n_frames):
        """{viseme.k: values} over n_frames from a mouth token (speech.
        mouth_track): its vowel curves, or the openness on viseme.a."""
        src = float(tok.get("fps", self.fps))
        t0 = float(tok.get("t", 0.0))

        def at(curve):
            c = np.asarray(curve, np.float64)
            if src == self.fps and t0 == 0.0:
                out = np.zeros(n_frames)
                m = min(n_frames, len(c))
                out[:m] = c[:m]
                return out
            return np.interp(np.arange(n_frames) / self.fps - t0,
                             np.arange(len(c)) / src, c, left=0.0, right=0.0)
        vow = tok.get("vowels")
        if vow:
            out = {f"viseme.{k}": at(vow.get(k, ())) if vow.get(k) else
                   np.zeros(n_frames) for k in "aiueo"}
        else:
            out = {f"viseme.{k}": np.zeros(n_frames) for k in "aiueo"}
            out["viseme.a"] = at(tok.get("open", ()))
        return {p: v for p, v in out.items() if p in self.body.params}


def load_body(path):
    bm = fb.load(path)
    if getattr(bm, "kind", None) != "vrm":
        raise SystemExit(f"{path}: not a VRM body file (python -m "
                         "flow.vrmbody --vrm PATH)")
    return bm


def body_for(vrm_path=None, body_path=None, log=print):
    """The VRMBody for --vrm / --vrm-body: the body file when given (or
    rigs/<vrm name>.body.json when it exists and names this file), else
    built from the .vrm now."""
    try:
        from . import vrmbody as vb
    except ImportError:
        from flow import vrmbody as vb
    if body_path:
        bm = load_body(body_path)
        if bm.vrm.get("rev", 1) < vb.BODY_REV:
            log(f"vrm: {body_path} is body file revision "
                f"{bm.vrm.get('rev', 1)} (no collision model: hands may pass "
                "through the body); rebuild it: python -m flow.vrmbody "
                f"--vrm PATH --out {body_path}")
        return bm, body_path
    if not vrm_path:
        raise SystemExit("--avatar vrm needs --vrm PATH or --vrm-body FILE")
    stem = os.path.splitext(os.path.basename(vrm_path))[0]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(root, "rigs", stem + ".body.json"),):
        if os.path.exists(cand):
            bm = fb.load(cand)
            if getattr(bm, "kind", None) == "vrm" and \
                    bm.vrm["file"] == os.path.basename(vrm_path):
                if bm.vrm.get("rev", 1) >= vb.BODY_REV:
                    return bm, cand
                # Built by an older flow.vrmbody (no collision model, the
                # look-at's output scale unknown): built again from the .vrm
                # it names, in place.
                log(f"vrm: {cand} is body file revision "
                    f"{bm.vrm.get('rev', 1)}, rebuilding (revision "
                    f"{vb.BODY_REV})")
                blob = vb.build(vrm_path, log=log)
                with open(cand, "w", encoding="utf-8") as fh:
                    json.dump(blob, fh, ensure_ascii=False, indent=1)
                return vb.VRMBody(blob), cand
    # A dropped-in avatar: its body file is built now and kept under
    # rigs/, as flow.vrmbody --out would write it.
    out = os.path.join(root, "rigs", stem + ".body.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} describes another .vrm; pass --vrm-body "
                         "(python -m flow.vrmbody --vrm PATH --out FILE)")
    blob = vb.build(vrm_path, log=log)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=1)
    log(f"vrm: body file {out}")
    return vb.VRMBody(blob), out


def stream_states(stream, bm, n_frames):
    """Run a MotionStream n_frames and return the body states."""
    out = []
    for _ in range(n_frames):
        over, _parts = stream.frame()
        out.append(bm.state(over))
    return out


def save_states(path, states, fps, body_path=None, vrm=None, t0=0.0,
                extra=None):
    rec = {"format": "tsukuyomi.vrmstates/1", "fps": float(fps),
           "t0": float(t0), "body": body_path, "vrm": vrm,
           "states": states, **(extra or {})}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, separators=(",", ":"))
    return path


# ---------------------------------------------------------------- director

def prepare(a, bm, body_path, info=sys.stdout):
    """director.prepare for a VRM avatar: {"labels", "offered", "system",
    "motions", "stats", "roles", ...}. Body mode: the VRM body's prompt and
    parser (no gestures: none are authored for VRM). Tag mode: the tags the
    VRM roles can play."""
    try:
        from . import director as dr
        from . import pose as pz
    except ImportError:
        from flow import director as dr
        from flow import pose as pz
    rig = VRMRig(bm, a.events, a.fps, a.idle_profile, log=lambda s: None)
    if getattr(a, "mode", "tags") == "body":
        mlines, mnames = dr.move_menu(a, bm)
        print(f"director: body mode, VRM {bm.vrm['file']}: "
              f"{len(pz.offered(bm))} variables, no gestures"
              + (f", motions {' '.join(mnames)}" if mnames else ""),
              file=info)
        return {"labels": {}, "offered": None, "descriptions": {},
                "system": pz.system_prompt(bm, a.persona, False, "",
                                           mlines, getattr(a, "body_prompt",
                                                           "v1")),
                "motions": {}, "stats": rig.stats, "roles": rig.roles,
                "mode": "body", "body": bm, "body_path": body_path,
                "core": None, "images": False, "max_tag": pz.MAX_MARKER,
                "parser": lambda: pz.parser_for(bm, moves=bool(mnames)),
                "vrm_rig": rig}
    offered, out = dr.offer(rig.stats, rig.roles)
    if out:
        print("director: not offered on this avatar: "
              + dr.left_out_text(out), file=info)
    system = dr.system_prompt(a.persona, {}, offered, {}, {})
    return {"labels": {}, "offered": offered, "system": system,
            "motions": {}, "stats": rig.stats, "roles": rig.roles,
            "descriptions": {}, "vrm_rig": rig, "body": bm,
            "body_path": body_path}


def plan(a, reply, out_dir, run, log=print, tts=None):
    """director.plan for a VRM avatar: reply.txt, spoken.txt, actions.json,
    subs.srt/json, speech.wav (with tts), tokens.json (the directed tokens)
    and states.json (the stream's body state per frame, idle included).
    Returns (seconds, states path) or None for a refusal."""
    try:
        from . import director as dr
        from . import pose as pz
        from . import speech as sp
    except ImportError:
        from flow import director as dr
        from flow import pose as pz
        from flow import speech as sp
    os.makedirs(out_dir, exist_ok=True)
    if reply["refused"]:
        with open(os.path.join(out_dir, "refusal.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(f"stop_reason: {reply['stop_reason']}\n")
        log("director: the model refused this turn; no states written")
        return None
    bm = run["body"]
    rig = run.get("vrm_rig") or VRMRig(bm, a.events, a.fps, a.idle_profile)
    fps = float(a.fps)
    for name, text in (("reply.txt", reply["raw"]),
                       ("spoken.txt", reply["spoken"] + "\n")):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    extra = {}
    mouth = None
    said = None
    if tts is not None and reply["spoken"]:
        moods = None
        if getattr(a, "mood", None) != "off":
            moods = sp.sentence_moods(reply["spoken"], reply["tags"])
        said = sp.speak(reply["spoken"], reply["tags"], tts,
                        lambda ch: 1.0 / (a.cjk_rate if dr._is_cjk(ch)
                                          else a.latin_rate), log=log,
                        moods=moods)
        acts, spans, end = said["acts"], said["spans"], said["end"]
        sp.write_wav(os.path.join(out_dir, "speech.wav"), said["audio"],
                     said["sr"])
        extra["tts"] = {"backend": getattr(tts, "kind", None) or a.tts,
                        "voice": tts.last_voice, "timing": said["method"]}
        if (getattr(a, "lipsync", None) or "vowels") != "off":
            nf = int(math.ceil(end * fps)) + 2
            mouth = sp.mouth_track(said["audio"], said["sr"], fps, nf,
                                   "vowels", a.mouth_gain,
                                   moras=said.get("moras"))
            extra["mouth"] = {"mode": "vowels", "frames": nf}
    else:
        stale = os.path.join(out_dir, "speech.wav")
        if os.path.exists(stale):
            os.remove(stale)
        acts, spans, end = dr.timeline(reply["spoken"], reply["tags"],
                                       a.cjk_rate, a.latin_rate)
    seconds = end + a.tail
    poses = []
    if any(x.get("tag") == pz.POSE for x in acts):
        poses = pz.solve_poses(bm, acts, log=log)
        arms = pz.arm_params(bm)
        for rec in poses:
            rec["arm_params"] = sorted(set(rec["params"]) & arms)
            acts[rec["act"]]["solved"] = rec
        for line in pz.report_lines(poses, reply["spoken"]):
            log("pose: " + line)
        rs = pz.reach_summary(poses)
        log(f"pose: {rs['markers']} markers, {rs['ok']}/{rs['targets']} "
            f"targets reached (solve {rs['solve_seconds']}s)")
        extra.update(mode="body", reach=rs)
        seconds = max(seconds, pz.end_time(
            poses, getattr(a, "pose_ease", False)) + a.tail)
    mvs = []
    if any(x.get("tag") == "move" for x in acts):
        try:
            from . import moves as mv
        except ImportError:
            from flow import moves as mv
        mvs = mv.solve_moves(bm, acts, log=log, seed=a.seed)
        for rec in mvs:
            acts[rec["act"]]["solved"] = rec
        for line in mv.report_lines(mvs, reply["spoken"]):
            log("move: " + line)
        if mvs:
            extra.update(mode="body", moves=len(mvs))
            seconds = max(seconds, mv.end_time(mvs) + a.tail)
    rng = np.random.default_rng([int(a.seed), 1])
    stream = fs.MotionStream(rig, seed=a.seed, idle=not a.no_idle,
                             head_drift=fs.drift_from_args(a),
                             pose_ease=fs.ease_from_args(a), log=log)
    directed = dr.realize(acts, rig.stats, list(rig.roles), {}, rng,
                          stream.pose(), settle=None)
    if poses:
        directed += pz.tokens(poses, rig.roles, pz.arm_params(bm))
    if mvs:
        directed += mv.tokens(mvs)
    if said is not None and run.get("mode") == "body":
        # Speech-driven micro-expressions (flow.micro), as the Live2D
        # render and flow.live have them.
        micro_toks = dr.micro_plan(a, run, said, acts, reply["spoken"], log)
        if micro_toks:
            directed += micro_toks
            extra["micro_events"] = len(micro_toks)
    gs = run.get("gsearch")
    gs_base = None
    if gs is not None:
        # Gesture search (flow.gesture_search): the retrieved motion under
        # the markers; needs the TTS audio (prosody, word times).
        if said is None:
            log("gesture search: needs the spoken audio (--tts); this turn "
                "plays without it")
        else:
            gs_base = list(directed)
            stream.gsearch = gs.layer(dict(zip(rig.pids, rig.dflt.tolist())))
            toks, rep = gs.plan_reply(reply["spoken"], said)
            directed += toks
            extra["gesture_search"] = {"tracks": rep, "stats": dict(gs.stats),
                                       "seed": gs.p.seed, "amp": gs.p.amp,
                                       "parts": gs.p.parts}
            log(f"gesture search: {len(toks)} tracks, "
                f"{sum(len(r['windows']) for r in rep)} windows")
    stream.direct(directed)
    n = int(round(seconds, 2) * fps) + 1
    if mouth is not None:
        stream.add_mouth(0, mouth, len(mouth["open"]))
    states = stream_states(stream, bm, n)
    states_base = None
    if gs_base is not None and getattr(a, "gs_baseline", False):
        # The same turn without the retrieved motion (the demo's other
        # panel): a stream of its own with the same seeds.
        s2 = fs.MotionStream(rig, seed=a.seed, idle=not a.no_idle,
                             head_drift=fs.drift_from_args(a),
                             pose_ease=fs.ease_from_args(a), log=log)
        s2.direct(gs_base)
        if mouth is not None:
            s2.add_mouth(0, mouth, len(mouth["open"]))
        states_base = stream_states(s2, bm, n)
    segs = dr.subtitles(reply["spoken"], spans)
    with open(os.path.join(out_dir, "actions.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"speech_seconds": round(end, 3),
                   "seconds": round(seconds, 3), "actions": acts,
                   "warnings": reply["warnings"], "avatar": "vrm",
                   "model_seconds": reply.get("model_seconds"),
                   "stop_reason": reply.get("stop_reason"), **extra}, fh,
                  indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "tokens.json"), "w",
              encoding="utf-8") as fh:
        json.dump(directed, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "subs.srt"), "w", encoding="utf-8") as fh:
        fh.write(dr.srt(segs))
    with open(os.path.join(out_dir, "subs.json"), "w", encoding="utf-8") as fh:
        json.dump(segs, fh, indent=1, ensure_ascii=False)
    path = save_states(os.path.join(out_dir, "states.json"), states, fps,
                       os.path.abspath(run["body_path"]),
                       os.path.abspath(a.vrm) if getattr(a, "vrm", None)
                       else None)
    if states_base is not None:
        save_states(os.path.join(out_dir, "states_markers.json"),
                    states_base, fps, os.path.abspath(run["body_path"]),
                    os.path.abspath(a.vrm) if getattr(a, "vrm", None)
                    else None)
    log(f"director: {len(reply['spoken'])} spoken characters, {end:.2f}s "
        f"speech, {len(acts)} actions, {len(segs)} subtitles; {n} VRM "
        f"states -> {path}")
    return seconds, path


# -------------------------------------------------------------------- live

class StateLoop:
    """flow.live's render loop for a VRM avatar: frame f at f / fps of the
    server clock, its parameters from the stream, its body state to the hub
    (the page draws it at server time + its playout delay, as it draws
    Live2D frames) and to a StateLog. A state costs well under a
    millisecond, so no frame is skipped: the page interpolates between
    them at its own rate."""

    def __init__(self, stream, bm, clock, hub, fps, log=print, record=None,
                 avatar="vrm", report=10.0):
        self.stream, self.bm, self.clock, self.hub = stream, bm, clock, hub
        self.fps, self.log, self.record = float(fps), log, record
        self.avatar, self.report = avatar, report
        self.f = 0
        self.cost = []

    def step(self):
        import time
        t0 = time.perf_counter()
        over, _parts = self.stream.frame()
        st = self.bm.state(over)
        self.hub.state(self.f, self.f / self.fps, st, self.avatar)
        if self.record is not None:
            self.record.add(self.f, st)
        self.cost.append(time.perf_counter() - t0)
        if len(self.cost) > 300:
            self.cost = self.cost[-300:]
        self.f += 1

    def run(self, stop, max_frames=None):
        import time
        t_rep = self.clock.now()
        while not stop.is_set():
            if max_frames is not None and self.f >= max_frames:
                return
            now = self.clock.now()
            due = int(now * self.fps)
            if self.f > due:
                time.sleep(min(self.f / self.fps - now, 0.05))
                continue
            self.step()
            if self.report and now - t_rep >= self.report:
                self.log(f"live: vrm states {1000 * np.median(self.cost):.2f}"
                         " ms per frame")
                t_rep = now


class StateLog:
    """--record with a VRM avatar: every state, written to states.json
    at close (render it with flow.vrmrender --run DIR, which muxes the
    session's speech.wav and subs.srt)."""

    def __init__(self, out_dir, fps, body_path, vrm_path):
        self.dir, self.fps = out_dir, fps
        self.body_path, self.vrm_path = body_path, vrm_path
        self.states = []

    def add(self, f, st):
        while len(self.states) < f:
            self.states.append(self.states[-1] if self.states else st)
        self.states.append(st)

    def close(self):
        os.makedirs(self.dir, exist_ok=True)
        return save_states(os.path.join(self.dir, "states.json"),
                           self.states, self.fps,
                           os.path.abspath(self.body_path)
                           if self.body_path else None,
                           os.path.abspath(self.vrm_path)
                           if self.vrm_path else None)


class Follower:
    """--avatar both: the VRM plays every sentence the Live2D rig plays,
    from the same parsed actions at the same server time. Its own idle,
    realize (its own roles and rng) and, in body mode, its own solve of the
    same pose markers -- landmark references resolve on its own body,
    angles in Live2D units are converted (from_live2d), and variables it
    lacks are dropped by flow.pose (reported). Composite targets ("hands":
    "@together") never reach it: the prompt and the parser are the Live2D
    body's, which has none (its prompt stays the bytes it was; it cannot
    draw a hand behind its back). Two hands it sends to one place meet
    side by side on the VRM (flow.vrmbody's collision model) instead of
    passing through each other. Its mouth follows the same audio with
    vowel shapes."""

    def __init__(self, a, bm, rig, stream, log=print):
        self.a, self.bm, self.rig, self.stream, self.log = a, bm, rig, \
            stream, log
        self.rng = np.random.default_rng([int(a.seed), 7])
        self._pre = None            # moves built ahead (build_moves)
        try:
            from . import micro as mc
        except ImportError:
            from flow import micro as mc
        self.micro = mc.planner_for(a, bm, log)    # flow.micro, or None

    def build_moves(self, acts, seed=0):
        """The sentence's move markers on the VRM's own body (its reach
        and collisions; the seed flow.live wrote on each marker: the same
        dance as the Live2D side). flow.live calls it before the sentence's
        time is taken, as it builds its own: built in sentence(), after
        the poses, a dance at the start of a sentence began 0.1-0.6 s
        after the Live2D one."""
        try:
            from . import moves as mv
        except ImportError:
            from flow import moves as mv
        cur, _ = self.stream.body_now()
        self._pre = mv.solve_moves(self.bm, acts, cur, {}, log=self.log,
                                   seed=seed)
        return self._pre

    def sentence(self, acts, t_start, f0, spoken, audio, end):
        try:
            from . import moves as mv
        except ImportError:
            from flow import moves as mv
        try:
            from . import director as dr
            from . import pose as pz
            from . import speech as sp
        except ImportError:
            from flow import director as dr
            from flow import pose as pz
            from flow import speech as sp
        a, fps = self.a, self.rig.fps
        acts = from_live2d(acts)
        poses = []
        if any(x.get("tag") == pz.POSE for x in acts):
            cur, _ = self.stream.body_now()
            poses = pz.solve_poses(self.bm, acts, cur, {}, log=self.log)
            for rec in poses:
                rec["start"] = round(rec["start"] + t_start, 4)
            for line in pz.report_lines(poses, spoken):
                self.log(f"live: vrm {line}")
            for rec in poses:
                rec["start"] = round(rec["start"] - t_start, 4)
        mvs = self._pre if self._pre is not None else \
            self.build_moves(acts, mv.move_seed(a, f0)) \
            if any(x.get("tag") == "move" for x in acts) else []
        self._pre = None
        for line in mv.report_lines(mvs, spoken):
            self.log(f"live: vrm {line}")
        shifted = [dict(x, t=round(float(x["t"]) + t_start, 4)) for x in acts
                   if x.get("tag") != "gesture"]
        directed = dr.realize(shifted, self.rig.stats, list(self.rig.roles),
                              {}, self.rng, self.stream.pose())
        for k in pz.tokens(poses, self.rig.roles, pz.arm_params(self.bm)):
            k["t"] = round(float(k["t"]) + t_start, 4)
            directed.append(k)
        if mvs:
            for k in mv.tokens(mvs):
                k["t"] = round(float(k["t"]) + t_start, 4)
                directed.append(k)
        if self.micro is not None and audio is not None:
            directed += self.micro.plan(audio[0], audio[1], acts, t_start,
                                        f0)
        if directed:
            self.stream.direct(directed)
        if audio is not None and (a.lipsync or "auto") != "off":
            x, sr, moras = audio
            nf = int(math.ceil(end * fps)) + 2
            tr = sp.mouth_track(x, sr, fps, nf, "vowels", a.mouth_gain,
                                moras=moras)
            self.stream.add_mouth(f0, tr, len(tr["open"]))


def live_setup(a, log=print):
    """(body, body path, rig, page config) for --avatar vrm / both."""
    try:
        from . import vrmrender as vr
    except ImportError:
        from flow import vrmrender as vr
    bm, bpath = body_for(a.vrm, a.vrm_body, log=log)
    vrm = a.vrm or vr.vrm_path(bm)
    rig = VRMRig(bm, a.events, a.fps, a.idle_profile, log=log)
    bg = tuple(float(x) for x in a.bg.split(","))
    cfg = dict(vr.page_config(bm, bg), avatar=a.avatar, fps=a.fps)
    return bm, bpath, rig, {"path": vrm, "config": cfg}
