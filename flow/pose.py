#!/usr/bin/env python3
"""
Body mode: the language model writes body targets, flow.body solves them,
and a pose layer plays them over the idle.

Stage E2 of docs/embodiment.md. The tag pipeline (flow.director) asks the
model to pick symbols ([nod], [look:left]); here it gets a description of
its own measured body (flow.body: parts, the character's left and right,
reach per side, landmarks, what it cannot do), its current pose as numbers
and, where the backend takes images, a picture of itself, and writes
continuous targets inline in its speech:

    Hi! [pose {"hand.R.y": "@shoulder.y", "head.tilt": 8}] So good to see you.

  parse     PoseParser: the director's incremental parser with markers
            scanned by brace depth (a JSON body is longer than any tag and
            may hold brackets), tolerant JSON (single quotes, bare keys,
            trailing commas are repaired; anything else is dropped with a
            warning, never spoken). Each target must be a variable this
            body has, a number or a landmark reference ("@shoulder.y",
            "@chest.x+0.2"). Optional "dur" (seconds the move takes) and
            "hold" (seconds kept before relaxing to the idle).
  mode      Body targets, plus [blink], [pause:x] and, with
            --body-gestures (the default since E2's A/B), the rig's own
            motions as tag mode offers them. Face shapes (brows,
            eye smile, mouth form, blush) are canonical variables like any
            joint, so they go in pose markers; a blink is a transient the
            autonomic layer owns (a held eye.L.open would stop blinking),
            and a pause is timing, not body. Gestures were left out at
            first (an authored clip hands the motion back to the rigger,
            the baseline); but lafei's tag replies won the A/B with one
            (login) where her body replies moved a hand twice in 12 poses,
            so they are offered next to the poses, and a gesture gives way
            to a pose that moves an arm (drop_clashing_gestures).
            --no-body-gestures is the pure condition. Nods and
            shakes are pose sequences (markers on the same variables play
            one after another).
  time      a marker is a tag ({"tag": "pose", "arg": {...}, "at": n}), so
            director.timeline and speech.speak time it from the words /
            moras exactly as they time tags, and the live loop's sentence
            streaming carries it unchanged.
  solve     solve_poses: each marker through BodyModel.solve with the prior
            toward the pose the body is in at that moment (the previous
            poses still held, else the rest), then only what the pose
            moves is played (parameters that change by >= PLAY_EPS of their
            span, plus each target's main mover); the rest stays with the
            idle. The report per target (forward model, and the Core's own
            geometry when a model is loaded: body.exact_state) is computed
            on that played set, so it says what renders.
  play      PoseLayer: per parameter a minimum-jerk move from what was on
            screen to the target, a hold that keeps a little life (LIFE of
            the idle's motion underneath, settling SETTLE of the way toward
            the idle), then a minimum-jerk release back
            to whatever the idle is doing, on top of the frame the idle
            built (runtime.main's frame loop and stream.MotionStream.frame
            both call it, so live draws what offline draws). Arm variant
            changes swap parts (and switch parameters) at the midpoint of
            the move, as the gesture layer does: never two arms drawn.
            Parameters the probe banded discrete but the body file says are
            solvable joints (lafei's PARAM_Arm_LT_Angle, Natori's
            ParamArmAL01) are written as solved, never snapped.
  merge     a pose token moving an arm takes over "arms" for its move
            plus hold (director.merge): idle gestures starting in that
            window are dropped. The idle's head, gaze, brow and body moves
            keep running under every pose -- on what the pose does not play
            fully, on what it holds at LIFE. (They were dropped for the
            role groups a pose touched, which froze Hiyori's head and gaze
            in 58% of the E2 hold frames.)

Usage:
    python -m flow.pose --body rigs/Hiyori.body.json --prompt
    python -m flow.pose --body rigs/Hiyori.body.json --reply \\
        'Hi! [pose {"hand.R.y": "@shoulder.y"}] Look.' [--model-json M]
    python -m flow.pose summary runs/e2/Hiyori        # body vs tags table
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys

import numpy as np

try:
    from . import body as fb
    from . import director as dr
    from . import events as ev
    from . import framing as fr
    from . import speech as sp
    from .behaviors import min_jerk
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import body as fb
    from flow import director as dr
    from flow import events as ev
    from flow import framing as fr
    from flow import speech as sp
    from flow.behaviors import min_jerk

POSE = "pose"
FRAME = "frame"
# A marker's JSON may run to a few hundred characters (six targets with
# landmark references is ~150); past this it is a runaway, dropped unspoken.
MAX_MARKER = 600
BODY_TAGS = ("pose", "blink", "pause", "gesture")
# The voice drives these; a target on them would fight the lip sync.
NOT_TARGETS = {"mouth.open": "the voice drives the mouth"}
# Timing, hand-set. A pose's default move time grows with its size, from a
# conversational stroke (director.NOD_STROKE, 0.3 s) for a small change to
# about the fitted idle head move (median 1.3 s) for a full-span one.
DUR_MIN, DUR_MAX = 0.3, 1.2
DUR_RANGE = (0.15, 3.0)
# Default hold: director.MIN_HOLD, what a tag's tilt or look holds. It was
# 1.5 s; in E2 the body replies read as long static holds next to the tags'
# quick nods and looks (Hiyori: 13 poses held 1.73 s on average, head and
# gaze frozen in 58% of the hold frames).
POSE_HOLD = 1.0
HOLD_MAX = 8.0
RELEASE_MIN = 0.5       # relaxing back is never quicker than this
# A held pose is not frozen. LIFE: the share of the idle's own motion since
# the pose started that still shows on a held target (director.merge keeps
# the idle's moves running under a pose). The calm idle moves Hiyori's
# head yaw 3.5 % of its span (std, 60 s), her gaze 15.6 % with ~0.6
# saccades a second; at 0.35 a held head drifts ~1 % (0.7 deg) and held
# eyes still make small saccades. SETTLE: how far a held target relaxes
# toward the idle by the end of its hold (eased, so the hold starts and
# ends at rest) -- people sag out of a held pose rather than stop dead,
# and a long hold the model asked for keeps moving even where the idle
# has nothing to give (face shapes, arms).
LIFE = 0.35
SETTLE = 0.2
# Organic playback (PoseLayer(ease=True); the offline director render and
# flow.live, not the dataset). The model's beats are short eases (0.15-0.3
# s) to a held target and a release; played as written, a head or torso
# beat is a spike with nothing moving around it. How a target is
# interpolated, never what it is or when it starts: head, torso and body
# angles take at least ANGLE_DUR to arrive and ANGLE_RELEASE to relax (the
# hold is as long as written, from when the move is done), the face at
# least FACE_DUR. All of it minimum jerk, an S-curve.
ANGLE_DUR, FACE_DUR, ANGLE_RELEASE = 0.35, 0.12, 0.7
_ANGLE_ID = re.compile(r"(^|\.|Param)(Body)?Angle[XYZ]$|^(head|torso|chest|"
                       r"spine|neck|hips)\.", re.I)


def angle_params(pids, roles=()) -> set:
    """The parameters that take ANGLE_DUR: those bound to a head or body
    role (runtime's {role: (pid, sign, source)}), and any other whose id is
    an angle of the head or body (ParamAngleX, ParamBodyAngleZ, head.yaw,
    torso.tilt). Brow and eye angles are the face's."""
    out = {p for r, (p, _s, _src) in dict(roles).items()
           if ev.GROUP_OF.get(r) in ("head", "body")}
    out |= {p for p in pids if _ANGLE_ID.search(p)}
    return out
# A solved parameter is played when it moves at least this fraction of its
# span from where it was (or is a target's main mover); below it the idle
# keeps it. Hiyori's hand.R.y = shoulder solve also nudged BodyAngleX by
# 0.03 of 20 (0.15 %) and ParamShoulder by 0.9 %: freezing those would stop
# the idle's sway for nothing visible.
PLAY_EPS = 0.01
# Variables offered in the prompt besides the declared ones: hand positions.
# The other measured regions (neck, torso, upper arm centroids) follow from
# them and from the declared angles, and listing them invited targets that
# fight each other.
HAND_VARS = ("hand.L.x", "hand.L.y", "hand.R.x", "hand.R.y")
# Plain words for declared variables (flow.body.SIGN has the head/torso ones).
MEANING = {
    "head.yaw": "+ turns your face toward your left (the viewer's right)",
    "head.pitch": "+ looks up",
    "head.tilt": "+ tilts your head counter-clockwise as the viewer sees it "
                 "(toward your right shoulder)",
    "torso.yaw": "+ turns your upper body toward your left",
    "torso.pitch": "+ leans back, - leans in toward the viewer",
    "torso.tilt": "+ tilts your upper body counter-clockwise as seen "
                  "(toward your right)",
    "torso.forward": "+ toward the viewer",
    "gaze.x": "+ looks toward your left (the viewer's right)",
    "gaze.y": "+ looks up",
    "eye.L.open": "0 closed, 1 open", "eye.R.open": "0 closed, 1 open",
    "eye.L.smile": "+ smiling eye", "eye.R.smile": "+ smiling eye",
    "brow.L.lift": "+ raises the brow", "brow.R.lift": "+ raises the brow",
    "brow.L.slide": "brow sideways", "brow.R.slide": "brow sideways",
    "brow.L.angle": "brow angle", "brow.R.angle": "brow angle",
    "brow.L.form": "- angry, + gentle", "brow.R.form": "- angry, + gentle",
    "mouth.form": "- frown, + smile",
    "cheek.blush": "+ blush",
    "eye.L.squint": "+ squints", "eye.R.squint": "+ squints",
    "cheek.puff": "+ puffs the cheek",
    "mouth.pucker": "- purses the lips, + widens the mouth",
    "mouth.funnel": "+ rounds the lips into an O",
    "tongue.out": "+ sticks the tongue out",
    "eye.tears": "+ tears",
    "shoulders.up": "+ shrugs", "shoulder.L.up": "+ raises the shoulder",
    "shoulder.R.up": "+ raises the shoulder",
    "hand.L.x": "your left hand, + toward your left",
    "hand.L.y": "your left hand height",
    "hand.R.x": "your right hand, + toward your left",
    "hand.R.y": "your right hand height",
}
IMAGE_BACKENDS = ("anthropic", "siliconflow", "ollama")


# ----------------------------------------------------------------- parse

def loads_lenient(s: str) -> tuple[dict, bool]:
    """(object, repaired) from a marker's JSON body. Repairs what language
    models write instead of JSON: single quotes, bare keys, trailing
    commas, missing braces. Raises ValueError when nothing works."""
    s = s.strip()
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            return d, False
    except ValueError:
        pass
    t = s
    a, b = t.find("{"), t.rfind("}")
    if a >= 0 and b > a:
        t = t[a:b + 1]
    elif a < 0 and b < 0:
        t = "{" + t + "}"
    if '"' not in t:
        t = t.replace("'", '"')
    t = re.sub(r'([{,]\s*)([A-Za-z_@][\w.@+\-]*)\s*:', r'\1"\2":', t)
    t = re.sub(r'(:\s*)(@[\w.]+(?:[+-][\d.]+)?)(\s*[,}])', r'\1"\2"\3', t)
    t = re.sub(r",\s*}", "}", t)
    try:
        d = json.loads(t)
    except ValueError as exc:
        raise ValueError(str(exc)) from None
    if not isinstance(d, dict):
        raise ValueError("not an object")
    return d, True


def _landmark_ok(ref: str, landmarks) -> bool:
    m = re.match(r"@([\w.]+?)([+-][\d.]+)?$", ref)
    if not m:
        return False
    name = m.group(1)
    return name in landmarks or (name[-2:] in (".x", ".y")
                                 and name[:-2] in landmarks)


class PoseParser(dr.TagParser):
    """director.TagParser for body mode: [pose {...}] markers (also
    "[pose: {...}]" and a bare "[{...}]"), [blink] and [pause:x]; any other
    tag is dropped with a warning. `variables`: the canonical variables
    this body has (None: any); `landmarks`: names a "@" reference may
    use. A marker split across chunks parses as it would whole."""

    def __init__(self, variables=None, landmarks=(), max_marker=MAX_MARKER,
                 gestures=(), moves=None, frames=False):
        # gestures: the labels [gesture:NAME] may name (--body-gestures);
        # none, and a gesture marker is dropped like any other tag.
        # moves: the motion primitives [move {...}] may name (flow.moves,
        # --body-moves); None, and a move marker is dropped as before.
        # frames: take [frame bust] / [frame full] (flow.framing); off, and
        # a frame marker is dropped like any other tag.
        super().__init__(gestures, None)
        self.frames = bool(frames)
        self.moves = None if moves is None else list(moves)
        self.variables = None if variables is None else set(variables)
        self.landmarks = set(landmarks or ())
        self.max_marker = max_marker
        self._depth, self._instr, self._esc, self._long = 0, False, False, \
            False

    def _open(self):
        self._buf = ""
        self._depth, self._instr, self._esc, self._long = 0, False, False, \
            False

    def feed(self, chunk):
        done = []
        chunk = chunk.translate(dr._FULLWIDTH)
        text = []
        for ch in chunk:
            if self._buf is None:
                if ch == "[":
                    self._text("".join(text))
                    text = []
                    self._open()
                else:
                    text.append(ch)
                continue
            if self._instr:
                if self._esc:
                    self._esc = False
                elif ch == "\\":
                    self._esc = True
                elif ch == '"':
                    self._instr = False
            elif ch == '"' and self._depth > 0:
                self._instr = True
            elif ch == "[" and self._depth == 0:
                # A new '[' before this one closed: the first was not a
                # marker. Dropped, as the tag parser drops it.
                self.warnings.append(f"unclosed '[{self._buf[:40]}' dropped")
                self._open()
                continue
            elif ch in "{[":
                self._depth += 1
            elif ch in "}]" and self._depth > 0:
                self._depth -= 1
            elif ch == "]":
                inner, long_ = self._buf, self._long
                self._buf = None
                if long_:
                    self.warnings.append(f"marker longer than "
                                         f"{self.max_marker} characters "
                                         "dropped")
                    continue
                t = self._marker(inner)
                if t:
                    done.append(t)
                continue
            if not self._long:
                self._buf += ch
                if len(self._buf) > self.max_marker:
                    if "{" in self._buf:
                        # JSON that never ends: drop it all, never speak it.
                        self._long = True
                    else:
                        self.warnings.append(
                            f"'[' never closed within {self.max_marker} "
                            "characters; kept as text")
                        rest, self._buf = self._buf, None
                        self._text(rest)
        if text:
            self._text("".join(text))
        return done

    def close(self):
        if self._buf is not None:
            self.warnings.append(f"unterminated marker '[{self._buf[:40]}' at "
                                 "the end dropped")
            self._buf = None
        return super().close()

    def _marker(self, inner):
        s = inner.strip()
        m = re.match(r"(?is)pose\b\s*:?\s*(.*)$", s)
        if m or s.startswith("{"):
            return self._pose(m.group(1) if m else s, inner)
        name = re.split(r"[\s:]", s, maxsplit=1)[0].lower() if s else ""
        if self.moves is not None:
            mm = re.match(r"(?is)move\b\s*(.*)$", s)
            if mm:
                return self._move(mm.group(1), inner)
            if name in self.moves:
                # A bare primitive name ("[dance]", "[clap {...}]").
                return self._move(s, inner)
        if self.frames and name == "frame":
            return self._frame(re.sub(r"(?is)^frame\b\s*:?\s*", "", s),
                               inner)
        ok = ("blink", "pause") + (("gesture",) if self.gestures else ())
        if name not in ok:
            self.warnings.append(f"[{inner}] is not a body-mode marker "
                                 f"({', '.join((POSE,) + ok)}); dropped")
            return None
        return self._tag(inner)

    def _frame(self, body, inner):
        view = body.strip().strip("\"'").strip().lower()
        if view not in fr.FRAMES:
            self.warnings.append(f"[{inner}]: 'frame' takes "
                                 f"{'/'.join(fr.FRAMES)}; dropped")
            return None
        tag = {"tag": FRAME, "arg": view, "at": len(self.spoken)}
        self.tags.append(tag)
        return tag

    def _move(self, body, inner):
        try:
            from . import moves as mv
        except ImportError:
            from flow import moves as mv
        short = inner if len(inner) <= 80 else inner[:77] + "..."
        arg, warn = mv.parse_spec(body, self.moves, short)
        self.warnings += warn
        if arg is None:
            return None
        tag = {"tag": mv.MOVE, "arg": arg, "at": len(self.spoken)}
        self.tags.append(tag)
        return tag

    def _pose(self, body, inner):
        short = inner if len(inner) <= 80 else inner[:77] + "..."
        try:
            d, repaired = loads_lenient(body)
        except ValueError as exc:
            self.warnings.append(f"[{short}]: bad JSON ({exc}); dropped")
            return None
        if repaired:
            self.warnings.append(f"[{short}]: JSON repaired")
        targets, spec = {}, {}
        for k, v in d.items():
            key = str(k).strip()
            if key in ("dur", "duration", "hold"):
                try:
                    x = float(v)
                except (TypeError, ValueError):
                    self.warnings.append(f"[{short}]: {key} {v!r} is not a "
                                         "number; ignored")
                    continue
                if not math.isfinite(x):
                    continue
                if key == "hold":
                    spec["hold"] = round(min(max(x, 0.0), HOLD_MAX), 3)
                else:
                    spec["dur"] = round(min(max(x, DUR_RANGE[0]),
                                            DUR_RANGE[1]), 3)
                continue
            if key in NOT_TARGETS:
                self.warnings.append(f"[{short}]: {key} is not yours to set "
                                     f"({NOT_TARGETS[key]}); ignored")
                continue
            if self.variables is not None and key not in self.variables:
                self.warnings.append(f"[{short}]: no variable {key!r} on "
                                     "this body; ignored")
                continue
            if isinstance(v, bool) or v is None:
                self.warnings.append(f"[{short}]: {key}: {v!r} ignored")
                continue
            if isinstance(v, (int, float)):
                if math.isfinite(float(v)):
                    targets[key] = float(v)
                continue
            sv = str(v).strip()
            if sv.startswith("@"):
                if self.landmarks and not _landmark_ok(sv, self.landmarks):
                    self.warnings.append(f"[{short}]: {key}: no landmark "
                                         f"{sv!r}; ignored")
                    continue
                targets[key] = sv
                continue
            try:
                targets[key] = float(sv)
            except ValueError:
                self.warnings.append(f"[{short}]: {key}: {sv!r} is not a "
                                     "number or @landmark; ignored")
        if not targets:
            self.warnings.append(f"[{short}]: no usable targets; dropped")
            return None
        tag = {"tag": POSE, "arg": {"targets": targets, **spec},
               "at": len(self.spoken)}
        self.tags.append(tag)
        return tag


def parser_for(bm, max_marker=MAX_MARKER, gestures=(), moves=False,
               frames=False):
    # A VRM body also takes composite targets ("hands": "@together");
    # a Live2D body has none (the same parser as before them). moves: take
    # [move {...}] markers for the primitives this body can do. frames: take
    # [frame bust] / [frame full] (a body whose framing can switch).
    names = tuple(getattr(bm, "composite_landmarks", ()))
    keys = list(getattr(bm, "composite_keys", ())) if names else []
    mvs = None
    if moves:
        try:
            from . import moves as mv
        except ImportError:
            from flow import moves as mv
        mvs = mv.offered(bm)
    return PoseParser(available(bm) + keys,
                      list(bm.landmarks) + list(names) if names
                      else bm.landmarks, max_marker, gestures, mvs,
                      frames and fr.can_frame(bm))


def tag_text(t) -> str:
    """A tag or pose action as written: "[nod]", "[look:left]",
    '[pose {"head.tilt": 8}]'."""
    arg = t.get("arg")
    if t.get("tag") == POSE and isinstance(arg, dict):
        d = dict(arg.get("targets", {}))
        d.update({k: arg[k] for k in ("dur", "hold") if k in arg})
        return f"[pose {json.dumps(d, ensure_ascii=False)}]"
    if t.get("tag") == FRAME:
        return f"[frame {arg}]"
    if t.get("tag") == "move" and isinstance(arg, dict):
        return f"[move {json.dumps(arg, ensure_ascii=False)}]"
    return f"[{t['tag']}{':' + str(arg) if arg else ''}]"


# --------------------------------------------------------- the body in words

def available(bm) -> list[str]:
    """Canonical variables some variant of this body has and can move:
    what a marker may target. Sorted (byte-stable prompts)."""
    out = []
    for f in bm.features:
        k = bm.fidx[f]
        if not any(v["present"][k] for v in bm.variants):
            continue
        if f in NOT_TARGETS:
            continue
        out.append(f)
    return sorted(out)


def _ranges(bm, var):
    """[(variant, lo, hi)] of a variable's reach alone, per variant."""
    out = []
    for k in range(len(bm.variants)):
        lim = bm.limits(var, k)
        if lim is not None:
            out.append((k, float(lim[0]), float(lim[1])))
    return out


def offered(bm) -> list[str]:
    """The variables the prompt lists: declared ones that move, then hand
    positions; in flow.body's DECLARED order, then HAND_VARS."""
    have = set(available(bm))
    out = []
    for n, _u, _sid, _deg in fb.DECLARED:
        if n in have and n not in NOT_TARGETS:
            r = _ranges(bm, n)
            if r and max(h for _, _, h in r) - min(lo for _, lo, _ in r) \
                    >= fb.TOL[fb.feature_unit(n)]:
                out.append(n)
    out += [v for v in HAND_VARS if v in have]
    return out


def _fmt(x, unit):
    s = f"{x:.0f}" if unit == "deg" else f"{x:.2f}"
    return s.lstrip("-") if float(s) == 0 else s


def _level(bm, y):
    """Plain words for a height in hu."""
    L = bm.landmarks
    head = L.get("head", [0, 0.4])[1]
    sh = L.get("shoulder.y", -0.2)
    chest = L.get("chest", [0, sh - 0.4])[1]
    if y >= head + 0.25:
        return "above your head"
    if y >= 0.0:
        return "face height"
    if y >= sh - 0.05:
        return "shoulder height"
    if y >= chest - 0.05:
        return "chest height"
    if y >= chest - 0.6:
        return "waist height"
    return "hanging low"


def _union(ints, tol):
    """Merge [(lo, hi)] that overlap within tol."""
    out = []
    for lo, hi in sorted(ints):
        if out and lo <= out[-1][1] + tol:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _side(s):
    return "left" if s == "L" else "right"


def _redrawn(bm, k):
    """Which hands a switch from the rest variant to variant k redraws:
    those whose rest position or presence differs."""
    v0, vk = bm.variants[0], bm.variants[k]
    out = []
    for s in "LR":
        ix, iy = bm.fidx[f"hand.{s}.x"], bm.fidx[f"hand.{s}.y"]
        p0, pk = bool(v0["present"][ix]), bool(vk["present"][ix])
        if p0 != pk or (p0 and np.hypot(v0["rest"][ix] - vk["rest"][ix],
                                         v0["rest"][iy] - vk["rest"][iy])
                        > 0.05):
            out.append(s)
    return out


def describe_body(bm) -> str:
    """The body description of the body-mode prompt, from the body file
    alone (no Core, no model file): byte-stable per file."""
    L = []
    regs = set(bm.blob.get("regions", {}))
    names = []
    face = [w for r, w in (("eye.L", "eyes"), ("brow.L", "brows"),
                           ("mouth", "mouth"), ("cheek", "cheeks"),
                           ("ear", "ears")) if r in regs or
            r.replace(".L", ".R") in regs]
    names.append("head" + (f" ({', '.join(face)})" if face else ""))
    for r, w in (("neck", "neck"), ("torso", "torso"), ("hips", "hips"),
                 ("legs", "legs (drawn, not movable)")):
        if r in regs:
            names.append(w)
    arms = [w for r, w in (("upper_arm", "upper arms"),
                           ("forearm", "forearms"), ("hand", "hands"),
                           ("arm", "arms")) if f"{r}.L" in regs
            or f"{r}.R" in regs]
    if arms:
        names.append(", ".join(arms) + " (left and right)")
    if "hand.L" not in regs and "hand.R" not in regs and arms:
        names.append("no separate hands: hand.L / hand.R are the ends of "
                     "your arms")
    if "hair" in regs:
        names.append("hair (it swings by itself)")
    L.append("Parts: " + "; ".join(names) + ".")
    L.append("Left and right are always your own: hand.L is your left hand, "
             "which the viewer sees on the right side of the picture. "
             "Coordinates: origin at your chin, x toward your own left, y "
             "up, unit hu = one head height; angles in degrees; face values "
             "in your rig's own units.")
    lm = []
    for k, v in bm.landmarks.items():
        if isinstance(v, list):
            lm.append(f"{k} ({v[0]:.2f}, {v[1]:.2f})")
        else:
            lm.append(f"{k} {v:.2f}")
    L.append("Landmarks (hu; use \"@name\", \"@name.x\" / \"@name.y\", "
             "optionally with an offset like \"@shoulder.y+0.1\"): "
             + ", ".join(lm) + ". hand.L / hand.R are where your hands "
             "rest.")
    nv = len(bm.variants)
    L.append("What you can move (variable: rest -> reachable range, each "
             "alone):")
    for var in offered(bm):
        u = fb.feature_unit(var)
        r = _ranges(bm, var)
        rest = float(bm.variants[0]["rest"][bm.fidx[var]]) \
            if bm.variants[0]["present"][bm.fidx[var]] else None
        ints = _union([(lo, hi) for _, lo, hi in r], fb.TOL[u])
        rng = " or ".join(f"[{_fmt(lo, u)}, {_fmt(hi, u)}]" for lo, hi in ints)
        mean = MEANING.get(var) or (fb.SIGN[var][2] if var in fb.SIGN else "")
        unit = {"deg": "deg", "hu": "hu", "unit": "", "pct": "%"}[u]
        L.append(f"  {var}: " + (f"{_fmt(rest, u)} -> " if rest is not None
                                 else "") + rng + (f" {unit}" if unit else "")
                 + (f"; {mean}" if mean else ""))
    if nv > 1:
        L.append("Your arms have more than one drawing. A target only "
                 "another drawing reaches switches to it, halfway through "
                 "the move (the arm is redrawn, not moved there).")
    L.append("Reach of your hands:")
    cannot = []
    sh = bm.landmarks.get("shoulder.y")
    for s in "LR":
        ry, rx = _ranges(bm, f"hand.{s}.y"), _ranges(bm, f"hand.{s}.x")
        if not ry:
            L.append(f"  {_side(s)} hand: not drawn")
            cannot.append(f"you have no {_side(s)} hand to move")
            continue
        # One entry per distinct reach: Natori's 12 variants are its left
        # and right arm drawings combined, so each hand has 3 of its own.
        groups = {}
        for (k, lo, hi), (_, xlo, xhi) in zip(ry, rx):
            key = tuple(round(x, 2) for x in (lo, hi, xlo, xhi))
            groups.setdefault(key, []).append(k)
        parts = []
        for n, ((lo, hi, xlo, xhi), ks) in enumerate(groups.items(), 1):
            lv = (_level(bm, lo) if _level(bm, lo) == _level(bm, hi)
                  else f"{_level(bm, lo)} to {_level(bm, hi)}")
            what = f"y {lo:.2f} to {hi:.2f} ({lv}), x {xlo:.2f} to {xhi:.2f}"
            if len(groups) > 1:
                red = [] if 0 in ks else min(
                    (_redrawn(bm, k) for k in ks), key=len)
                what = (f"drawing {n}{' (rest)' if 0 in ks else ''}: " + what
                        + (" -- switching to it redraws "
                           + ("both arms" if len(red) > 1
                              else f"your {_side(red[0])} arm")
                           if red else ""))
            parts.append(what)
        L.append(f"  {_side(s)} hand: " + "; ".join(parts))
        top = max(hi for _, _, hi in ry)
        ints = _union([(lo, hi) for _, lo, hi in ry], fb.TOL["hu"])
        if sh is not None and top < sh - fb.TOL["hu"]:
            cannot.append(f"your {_side(s)} hand cannot reach shoulder height "
                          f"(y {sh:.2f}); its highest is {top:.2f}, "
                          f"{_level(bm, top)}")
        if len(ints) > 1:
            gaps = ", ".join(f"{a[1]:.2f} to {b[0]:.2f}"
                             for a, b in zip(ints, ints[1:]))
            cannot.append(f"your {_side(s)} hand cannot stop between heights "
                          f"{gaps} (no drawing holds it there)")
        xs = [(lo, hi) for _, lo, hi in rx]
        if s == "L" and min(lo for lo, _ in xs) > 0:
            cannot.append(f"your left hand cannot cross to your right side "
                          f"(x stays above {min(lo for lo, _ in xs):.2f})")
        if s == "R" and max(hi for _, hi in xs) < 0:
            cannot.append(f"your right hand cannot cross to your left side "
                          f"(x stays below {max(hi for _, hi in xs):.2f})")
    have = set(offered(bm))
    missing = [n for n, *_ in fb.DECLARED if n not in have
               and n not in NOT_TARGETS and n not in fb.EXTENDED]
    if missing:
        cannot.append("these do not exist on this body: " + ", ".join(missing))
    cannot.append("no walking, no legs, no fingers; nothing outside the "
                  "ranges above (a target outside is reached as far as it "
                  "goes)")
    notes = getattr(bm, "prompt_notes", None)
    if notes is not None:
        L += notes()
    L.append("What you cannot do:")
    L += [f"  - {c}" for c in cannot]
    return "\n".join(L)


BODY_TEMPLATE = """\
You are {persona}

You are {kind} talking with the user face to face. Reply with \
exactly what you say aloud: one to three short spoken sentences, in the \
user's language. You move your own body by writing pose markers inline, at \
the moment in your speech where the move happens. Markers are silent; they \
are never read out.

Your body, measured from your own model:
{body}

Pose markers:
[pose {{"var": value, ...}}]  move to these targets. A value is a number in \
the variable's unit or a landmark reference ("@shoulder.y"). Optional "dur": \
seconds the move takes (0.15 to 3; default by its size), "hold": seconds to \
keep it before relaxing back into your idle (0 to 8; default {hold:g}). \
Variables you leave out keep moving with your idle, and a held pose still \
moves a little with it. Markers on the same variables play one after \
another, so a nod is two quick markers: \
[pose {{"head.pitch": -12, "dur": 0.25, "hold": 0}}]\
[pose {{"head.pitch": 0, "dur": 0.3, "hold": 0}}]
{moves}{gestures}[blink]  a deliberate blink
[pause:0.5]  stay silent for that many seconds (0.2 to 3)
Each user message ends with "(your body now: ...)", where your body is at \
that moment in the same variables{image}.

How to move:
- Move in brief beats, as people do while they talk: a quick glance away \
and back (gaze with a little head turn), a nod or a shake as a quick pair \
of markers, brows up on a surprise, a tilt on a question. The markers of \
one nod or one glance-and-back count as one beat.
- Keep holds short: leave "hold" out, or 0 for a quick beat. Hold 2 seconds \
or more only for the one moment in a reply you mean to stress.
{whole}{move_rule}{beats}- Vary your moves from reply to reply; do not open every reply the \
same way or repeat the same pose.
- Do not hold your eyes half shut with eye.L.open / eye.R.open: held eyes \
stop blinking. A blink is [blink].

Rules:
{sentence_rule}- Only the variables listed above, inside their ranges. Valid JSON.
- No other tags or markup: no asterisk actions, no parentheses stage \
directions, no emoji, no lists, no narration.
- Never explain or mention the markers or the numbers.

Examples of placement (not of content; never copy them):
{examples}"""


# The prompts of the teacher: "v1" (the first data run, r1; the bytes of
# before the expression beats), "beats" (r2: a streamer's face, see
# _beats_rule; its bytes are what tsk-q8r2 is served with and stay as they
# are) and "beats2" (r3: beats with several dense examples, the [frame full]
# entry before whole-body moves, and a reply-language line on every user
# turn). flow.dataset's --teacher-prompt picks; the director's own default
# stays v1, the prompt the first students were taught with.
PROMPTS = ("v1", "beats", "beats2")
# The prompts with the expression beats, the camera line and [frame] markers.
FRAMED = ("beats", "beats2")


def beats_like(prompt) -> bool:
    return prompt in FRAMED


# The beats prompt's reply-language rule: the pilot's replies kept the
# language of the earlier ones after the user switched.
LANG_RULE = ("- Answer in the language of the user's latest message, even "
             "if your earlier replies used another language. When the user "
             "mixes languages, answer in their main language (a mixed-in "
             "word or two may be mirrored).\n")
# beats2: the language is also written on the user turn ("(reply in: ...)",
# reply_line), and the opening of a reply is asked to vary.
LANG_RULE2 = ("- Answer in the language of the user's latest message, even "
              "if your earlier replies used another language: each user "
              "message ends with \"(reply in: English)\" (or Japanese, "
              "Chinese), which is the language of this reply. When the user "
              "mixes languages, answer in their main language (a mixed-in "
              "word or two may be mirrored).\n")
OPEN_RULE = ("- Vary how your replies open: never start every reply with the "
             "same expression, and do not take the opening of an example; "
             "begin with what this line makes you feel.\n")
SENTENCE_RULE = {
    "v1": "- One or two beats per sentence, for what that sentence means. "
          "Put a marker right before the words it goes with, often "
          "mid-sentence.\n",
    "beats": "- Make expression beats in every sentence: at least one, two or "
             "three in a longer one (the face and head beats above), each "
             "in its own marker right before the words it reacts to, often "
             "mid-sentence.\n" + LANG_RULE}
SENTENCE_RULE["beats2"] = (SENTENCE_RULE["beats"][:-len(LANG_RULE)]
                           + LANG_RULE2 + OPEN_RULE)


def _bothsides(have, base, kinds):
    """The variables of base.L / base.R + kind this body has, e.g.
    ("eye", ("smile",)) -> ["eye.L.smile", "eye.R.smile"]."""
    return [f"{base}.{s}.{k}" for k in kinds for s in "LR"
            if f"{base}.{s}.{k}" in have]


def _beats_rule(bm, v2=False) -> str:
    """The "face beats" lines of the r2 teacher prompt, worded from the
    variables this body has (an avatar without eye.*.smile is told to close
    its eyes; one without brow.*.lift is not told of brows). Streamers move
    the face several times a sentence, the head a few degrees with each
    change and the rest of the body hardly at all; the r1 replies held one
    expression a reply."""
    have = set(offered(bm))
    eyes = []
    smile = _bothsides(have, "eye", ("smile",))
    shut = _bothsides(have, "eye", ("open",))
    if smile:
        eyes.append("a smiling eye at something warm or funny ("
                    + ", ".join(f'"{v}": 0.8' for v in smile) + ")")
    if shut:
        eyes.append("eyes closed for a moment, a laugh or a happy squeeze ("
                    + ", ".join(f'"{v}": 0' for v in shut)
                    + ", with a short hold)")
    face = []
    if eyes:
        face.append("the eyes: " + "; ".join(eyes))
    if "mouth.form" in have:
        face.append('the mouth: "mouth.form" up for a smile, down for a '
                    "pout or a worry")
    brows = _bothsides(have, "brow", ("lift",))
    if brows:
        face.append("the brows: " + " and ".join(f'"{v}"' for v in brows)
                    + " up on a surprise or a question, down when it is "
                    "serious")
    if "cheek.blush" in have:
        face.append('a blush ("cheek.blush") when shy')
    squint = _bothsides(have, "eye", ("squint",))
    if squint:
        face.append("a squint for doubt or suspicion ("
                    + ", ".join(f'"{v}"' for v in squint) + " up)")
    extras = []
    if "cheek.puff" in have:
        extras.append('puffed cheeks ("cheek.puff") when sulking')
    if "mouth.pucker" in have:
        extras.append('pursed lips for a pout ("mouth.pucker" down), a wide '
                      'grin ("mouth.pucker" up)')
    if "mouth.funnel" in have:
        extras.append('an O mouth ("mouth.funnel") on a surprise')
    if "tongue.out" in have:
        extras.append('the tongue out ("tongue.out") when teasing')
    if "eye.tears" in have:
        extras.append('tears ("eye.tears") when moved or sad')
    if extras:
        face.append("small tricks of the face: " + "; ".join(extras))
    head = [v for v in ("head.tilt", "head.yaw", "head.pitch") if v in have]
    vary = []
    if eyes:
        vary.append("the eyes")
    if brows:
        vary.append("the brows")
    if "mouth.form" in have:
        vary.append("mouth shapes")
    if "cheek.blush" in have:
        vary.append("a blush")
    if squint or extras:
        vary.append("squints and other small face tricks")
    if {"gaze.x", "gaze.y"} & have:
        vary.append("the gaze")
    if head:
        vary.append("small head moves")
    if face:
        L = ["- Your face does most of the talking, as a streamer's does, "
             "and it never holds still: every sentence gets at least one "
             "expression beat, a longer sentence two or three, so that "
             "something changes about every 1.5 to 2 seconds of speech. A "
             "beat for each thing you react to (a joke, a surprise, a "
             "doubt, a warm word) and for the small turns between them. "
             "Vary them, not the same smile each time"
             + (": " + ", ".join(vary) if vary else "")
             + ". A beat is a quick change of " + "; ".join(face) + ". Use dur 0.15 to 0.4 "
             "and hold 0.3 to 1 for a beat, and the next beat or your idle "
             "takes it back; both eyes together unless you mean a wink.\n"]
    else:
        L = ["- Move your head in brief beats, several times a reply.\n"]
    if head and face:
        L.append("- Each beat carries a small head move in the same marker, "
                 "a few degrees (" + ", ".join(head) + " between -8 and 8): "
                 "a tilt with a smile, a small turn away with a doubt, a "
                 "little chin down when shy. Never write that head move as "
                 "a marker of its own.\n")
    L.append("- Keep the rest of your body for the moments that call for it: "
             "a greeting, a celebration, showing a size or a shape, pointing "
             "at what you mean, an emphasis, or when the user asks you to "
             "show or do something. Then let the hand fall back; the rest "
             "of the time the face and head do it.\n")
    L.append(camera_rule(bm))
    L.append(budget_line(bm, v2))
    return "".join(L)


# ------------------------------------------------- camera and motion budget

FRAME_ENTRY = (
    "[frame bust] or [frame full]  switch the camera between bust (head and "
    "chest) and full (the whole figure), as a streamer changes scenes. Go "
    "full only for a whole-body moment: a dance, showing an outfit or a "
    "pose, a big celebration with the body; not for a greeting wave or a "
    "single hand move. It eases over in about {ease:g} seconds at that point "
    "of your speech and lasts until the end of this reply; the next reply "
    "starts back in your home framing. At most once in a reply.\n")


def frame_entry(bm) -> str:
    """The [frame] line of the marker list ("" for a body whose framing
    cannot switch: a VRM is drawn at one camera)."""
    return FRAME_ENTRY.format(ease=fr.VIEW_EASE) if fr.can_frame(bm) else ""


# beats2: the data of r2 played dances in bust and used [frame full] for a
# single wave, so the entry says when, in terms of what the body does.
FRAME_ENTRY2 = (
    "[frame bust] or [frame full]  switch the camera between bust (head and "
    "chest) and full (the whole figure), as a streamer changes scenes. Write "
    "[frame full] right before {whole} and before showing an outfit, so the "
    "whole figure is in the picture; never for a wave or a single hand move, "
    "which stay in bust. It eases over in about {ease:g} seconds at that "
    "point of your speech and lasts until the end of this reply; the next "
    "reply starts back in your home framing. At most once in a reply.\n")


def whole_body_moves(bm, moves=True) -> list[str]:
    """The whole-body moves (flow.framing.WHOLE_BODY) this body can do, in
    that order; [] when the [move] menu is off."""
    if not moves:
        return []
    try:
        from . import moves as mv
    except ImportError:
        from flow import moves as mv
    have = set(mv.offered(bm))
    return [n for n in fr.WHOLE_BODY if n in have]


def frame_entry2(bm, moves=True) -> str:
    """The [frame] line of the beats2 prompt: [frame full] right before a
    dance or another whole-body move this body does, and before an outfit;
    never for a wave or one hand."""
    if not fr.can_frame(bm):
        return ""
    names = whole_body_moves(bm, moves)
    whole = ("a dance or another whole-body move ("
             + ", ".join(f"[move {{\"name\": \"{n}\"}}]" for n in names)
             + ")") if names else "a dance or another whole-body move"
    return FRAME_ENTRY2.format(whole=whole, ease=fr.VIEW_EASE)


def _ys(vals):
    return " and ".join(f"{v:.2f}" for v in vals)


def camera_rule(bm) -> str:
    """The camera line of the beats prompt, from the numbers: the bust
    picture's window in hu (flow.framing ABOVE / BELOW) against where this
    body's hands rest and how low they reach. Each user message then ends
    with camera_now()."""
    if not fr.can_frame(bm):
        return ("- Camera: you are filmed full, the whole figure, and each "
                "user message ends with \"(camera: full ...)\". Your body "
                "moves are seen.\n")
    b = fr.budget(bm)
    w, rest, low = b["window"], fr.rest_hands(bm), b["reach"]["low"]
    bot = w["bottom"]
    ys = [y for _x, y in rest.values()]
    vis = list(b["rest_visible"].values())
    if not any(vis):
        where = (f"your hands rest at y {_ys(ys)}, below the picture, and a "
                 f"hand shows only once raised above y {bot:.2f}")
    elif all(vis):
        where = (f"your hands rest at y {_ys(ys)}, in the lower part of the "
                 "picture")
    else:
        where = (f"your hands rest at y {_ys(ys)}, one of them below the "
                 "picture")
    hidden = (not all(vis)) or (low is not None and low < bot - fb.TOL["hu"])
    if hidden:
        tail = (f"So in bust a hand move below y {bot:.2f} is not seen: skip "
                "it and let the face carry the moment (brows, eyes, mouth, "
                "head) instead of switching to full.")
    else:
        tail = ("All your hand moves stay in that picture, but the face "
                "still carries the performance.")
    return (f"- Camera: you are filmed either in bust (head and chest) or in "
            f"full (the whole figure); each user message ends with "
            f"\"(camera: bust ...)\" or \"(camera: full ...)\", the framing "
            f"you are in now. The bust picture runs from {w['top']:.1f} hu "
            f"above your chin to {-bot:.1f} hu below it, {2 * w['half_width']:.1f} "
            f"hu wide; {where}. {tail} In full your whole body is in the "
            "picture and body moves are seen. Your home is the framing each "
            "reply starts in, the one the camera line of the user message shows; a "
            "switch lasts for this reply, and the next one starts back in "
            "your home unless you write the marker again.\n")


def camera_now(bm, framing) -> str:
    """The user turn's camera line: "(camera: bust, ...)" / "(camera:
    full, ...)", the framing now; numbers from the same window."""
    if framing == "bust" and fr.can_frame(bm):
        w = fr.window()
        return (f"(camera: bust, head and chest; hands below y "
                f"{w['bottom']:.2f} are out of frame)")
    return "(camera: full, the whole body in frame)"


def _times(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def budget_line(bm, v2=False) -> str:
    """The motion-budget line of the beats prompt: what this body can do
    (flow.framing.budget: the share of the gesture zone its hands reach,
    the face controls offered, its kind) as how much to move it."""
    b = fr.budget(bm)
    r, hm = b["reach"], b["hand_moves"]
    pct = f"{r['all']:.0%}"
    arms = (f"a hand reaches about {pct} of the places a hand goes when "
            f"talking, through {_times(r['drawings'], 'arm drawing')}")
    face = f"{_times(b['face'], 'face control')} offered"
    if r["all"] <= 0:
        what = (f"your arms do not move on their own (no control reaches a "
                f"place a hand goes when talking; they only sway with the "
                f"body) and you have {face}: the face and head carry the "
                "whole performance, with no hand moves")
    elif b["tier"] == "limited":
        what = (f"your arms are limited ({arms}) and you have {face}: carry "
                "the performance with the face and head, and move a hand "
                "rarely, at most "
                f"{_times(hm['full'], 'hand move')} in a reply (a greeting, "
                "a celebration, a size, pointing, an emphasis)")
    elif b["tier"] == "capable":
        what = (f"your arms reach almost anywhere ({arms}; a "
                f"{'3D' if b['kind'] == 'vrm' else 'Live2D'} body with "
                f"joints) and you have {face}: in full use your body, up to "
                f"{_times(hm['full'], 'hand move')} a reply, a lean or a "
                "turn of the torso, a dance when it fits"
                + (f"; in bust only a hand raised into the picture shows, so "
                   f"at most {_times(hm['bust'], 'hand move')} a reply there"
                   if "bust" in b["frames"] else ""))
    else:
        what = (f"your arms reach part of the way ({arms}) and you have "
                f"{face}: the face still does most of the talking; a hand "
                f"move is an option at the right moment, up to "
                f"{_times(hm['full'], 'hand move')} a reply"
                + (f" in full and {hm['bust']} in bust"
                   if "bust" in b["frames"] else ""))
    sw = ("; switch to full only for a whole-body moment (a dance, an outfit, "
          "a pose, a big celebration), never for a single hand move"
          if "bust" in b["frames"] else "")
    if v2 and "bust" in b["frames"]:
        sw = ("; switch to full only right before a dance or another "
              "whole-body move or an outfit, never for a wave or a single "
              "hand move")
    return f"- Motion budget: {what}{sw}.\n"


def _best_hand(bm):
    """(side, highest y) of the hand that reaches highest, or None."""
    best = None
    for s in "RL":
        r = _ranges(bm, f"hand.{s}.y")
        if r:
            top = max(hi for _, _, hi in r)
            if best is None or top > best[1] + 1e-9:
                best = (s, top)
    return best


def _whole_body(bm, gestures=False) -> str:
    """The "use your whole body" line: the hand that reaches highest and
    how high, the upper body where it turns, the rig's own motions when
    offered. E2's replies stayed on the face (lafei's 12 poses: two moved
    a hand) although lafei's right hand reaches face height."""
    have = set(offered(bm))
    bits = []
    best = _best_hand(bm)
    if best is not None:
        s, top = best
        bits.append(f"your {_side(s)} hand reaches up to {_level(bm, top)}, "
                    "so raise it to greet, point or show something, then let "
                    "it fall back")
    if have & {"torso.yaw", "torso.pitch", "torso.tilt"}:
        bits.append("lean in or turn your upper body when it gets personal")
    if gestures:
        bits.append("now and then one of your own motions, at most one per "
                    "reply and never while you pose a hand")
    if not bits:
        return ""
    return "- Use your whole body, not only your face: " + "; ".join(bits) \
        + ".\n"


def _dense_example(bm, have) -> str:
    """One whole reply written the way the beats rule asks for: four
    sentences (about 10 s of speech), eight expression beats, one at nearly
    every clause, each with its small head move, no hand. Built from the
    variables this body offers, like the other examples (lafei closes its
    eyes with eye.*.open, a body without brows or blush skips them); ""
    when fewer than five of the beats survive on this body."""
    def mk(d):
        d = {k: v for k, v in d.items() if k in have or k in ("dur", "hold")}
        return "[pose " + json.dumps(d) + "]" \
            if any(k in have for k in d) else ""
    smile = {v: 0.8 for v in _bothsides(have, "eye", ("smile",))}
    shut = {v: 0 for v in _bothsides(have, "eye", ("open",))}
    # No smiling eye (lafei): a squint, half shut; the laugh closes them.
    squeeze = smile or {v: 0.45 for v in shut}

    def brows(x):
        return {"brow.L.lift": x, "brow.R.lift": x}
    beats = [
        mk({**brows(0.7), "head.pitch": 4, "head.yaw": -4, "dur": 0.2,
            "hold": 0.4}),
        mk({**squeeze, "mouth.form": 0.9, "head.tilt": 6, "dur": 0.2,
            "hold": 0.7}),
        mk({**shut, "mouth.form": 1, "head.tilt": -5, "head.pitch": 3,
            "dur": 0.15, "hold": 0.5}),
        mk({"gaze.x": -0.6, "gaze.y": 0.2, "head.yaw": -6, "dur": 0.25,
            "hold": 0.6}),
        mk({"gaze.x": 0, "gaze.y": 0, **brows(-0.5), "mouth.form": -0.4,
            "head.yaw": 2, "head.tilt": -4, "dur": 0.25, "hold": 0.5}),
        mk({"cheek.blush": 0.6, "mouth.form": 0.4, "head.pitch": 5,
            "head.tilt": 4, "dur": 0.3, "hold": 0.8}),
        mk({**brows(0.5), "head.pitch": -3, "head.yaw": 4, "dur": 0.2,
            "hold": 0.4}),
        mk({**squeeze, "mouth.form": 0.7, "head.tilt": 5, "dur": 0.25,
            "hold": 0.6}),
    ]
    if sum(1 for x in beats if x) < 5:
        return ""
    b = [x + " " if x else "" for x in beats]
    return (f"{b[0]}Oh, you really did that? {b[1]}That is the funniest "
            f"thing, {b[2]}and I am not even sorry. {b[3]}Honestly, I was "
            f"{b[4]}a little scared, {b[5]}but it was brave, {b[6]}wasn't "
            f"it? {b[7]}Now tell me the rest.")


def _dense_variants(bm, have) -> list[str]:
    """The dense examples of the beats2 prompt: whole replies of about
    eight expression beats (the _dense_example density), each opening in
    another way (an eye smile, a look away, a laugh with closed eyes, a
    pout with worried brows, a blush, a calm slow tilt), with its own
    numbers and order. Built from the variables this body offers like the
    other examples; a variant with fewer than five surviving beats is left
    out, so the list is the same for a body every time."""
    def mk(d):
        d = {k: v for k, v in d.items() if k in have or k in ("dur", "hold")}
        return "[pose " + json.dumps(d) + "]" \
            if any(k in have for k in d) else ""
    shut = {v: 0 for v in _bothsides(have, "eye", ("open",))}

    def sm(x):
        # No smiling eye (lafei): half shut instead.
        return {v: x for v in _bothsides(have, "eye", ("smile",))} \
            or {v: 0.45 for v in shut}

    def br(x):
        return {"brow.L.lift": x, "brow.R.lift": x}

    def gz(x, y):
        return {"gaze.x": x, "gaze.y": y}
    V = [
        # 0: a smiling eye first
        [({**sm(0.8), "mouth.form": 0.8, "head.tilt": 6, "dur": 0.2,
           "hold": 0.6}, "Hehe, you remembered?"),
         ({**br(0.6), "head.pitch": 3, "head.yaw": 4, "dur": 0.2,
           "hold": 0.4}, "That is really sweet of you,"),
         ({**shut, "mouth.form": 1, "head.tilt": -4, "head.pitch": 2,
           "dur": 0.15, "hold": 0.5}, "and I am a little embarrassed."),
         ({**gz(0.6, -0.2), "head.yaw": 7, "dur": 0.25, "hold": 0.6},
          "Look, I even tidied up"),
         ({**gz(0, 0), "cheek.blush": 0.5, "mouth.form": 0.4,
           "head.pitch": 4, "dur": 0.3, "hold": 0.8}, "just in case."),
         ({**br(-0.4), "mouth.form": -0.3, "head.tilt": 4, "head.yaw": -3,
           "dur": 0.25, "hold": 0.5}, "Is it strange"),
         ({**sm(0.7), "mouth.form": 0.7, "head.tilt": -5, "dur": 0.25,
           "hold": 0.6}, "that I am this happy?"),
         ({**br(0.4), "head.pitch": -3, "head.yaw": 2, "dur": 0.2,
           "hold": 0.4}, "Stay a while.")],
        # 1: a look away first
        [({**gz(-0.5, 0.15), "head.yaw": -7, "dur": 0.25, "hold": 0.7},
          "Hmm, let me think about that."),
         ({**gz(0, 0), **br(0.5), "head.pitch": 3, "dur": 0.2,
           "hold": 0.4}, "Oh, wait,"),
         ({**sm(0.7), "mouth.form": 0.6, "head.tilt": 5, "dur": 0.2,
           "hold": 0.6}, "I know a place,"),
         ({**br(0.8), "head.yaw": 5, "head.pitch": 4, "dur": 0.15,
           "hold": 0.4}, "a tiny cafe by the river!"),
         ({**shut, "mouth.form": 1, "head.tilt": -6, "dur": 0.15,
           "hold": 0.5}, "The cake there is silly good."),
         ({**gz(0.5, 0), "head.yaw": 6, "cheek.blush": 0.4,
           "mouth.form": 0.3, "dur": 0.25, "hold": 0.7},
          "I went there alone once,"),
         ({**br(-0.5), "mouth.form": -0.4, "head.tilt": 4, "dur": 0.25,
           "hold": 0.5}, "and it was a bit lonely."),
         ({**gz(0, 0), **sm(0.8), "mouth.form": 0.9, "head.pitch": 3,
           "dur": 0.2, "hold": 0.6}, "So come with me this time.")],
        # 2: a small laugh with closed eyes first
        [({**shut, "mouth.form": 1, "head.tilt": 6, "head.pitch": 4,
           "dur": 0.15, "hold": 0.7}, "Ahaha, stop it,"),
         ({**sm(0.8), "mouth.form": 0.8, "head.yaw": -4, "dur": 0.2,
           "hold": 0.5}, "you are too funny!"),
         ({**br(0.6), "head.pitch": -3, "dur": 0.2, "hold": 0.4},
          "Wait, was that real?"),
         ({**gz(0.5, 0.1), "head.yaw": 5, "dur": 0.25, "hold": 0.6},
          "Say it again slowly,"),
         ({**gz(0, 0), **br(-0.3), "mouth.form": -0.2, "head.tilt": -5,
           "dur": 0.25, "hold": 0.5}, "I want to be sure I heard it."),
         ({**shut, "mouth.form": 0.9, "head.tilt": 5, "head.pitch": 3,
           "dur": 0.15, "hold": 0.5}, "Pfft, no, I cannot breathe."),
         ({"cheek.blush": 0.5, "mouth.form": 0.5, "head.pitch": 5,
           "dur": 0.3, "hold": 0.8}, "Sorry, sorry,"),
         ({**br(0.4), **sm(0.6), "mouth.form": 0.6, "head.yaw": -3,
           "dur": 0.2, "hold": 0.5}, "keep going.")],
        # 3: a pout / worried brows first
        [({**br(-0.6), "mouth.form": -0.5, "head.tilt": -6,
           "head.pitch": 3, "dur": 0.25, "hold": 0.8},
          "Eh, you stayed up all night again?"),
         ({**gz(-0.4, -0.2), "mouth.form": -0.3, "head.yaw": -5,
           "dur": 0.25, "hold": 0.6}, "That is not good for you,"),
         ({**br(0.7), "head.pitch": 4, "head.yaw": 3, "dur": 0.2,
           "hold": 0.5}, "you know that?"),
         ({**gz(0, 0), **br(-0.4), "mouth.form": -0.6, "head.tilt": 5,
           "dur": 0.25, "hold": 0.6}, "I was worried,"),
         ({"cheek.blush": 0.6, "mouth.form": 0.2, "head.pitch": 5,
           "head.tilt": -4, "dur": 0.3, "hold": 0.8},
          "really, not just a little."),
         ({**sm(0.5), "mouth.form": 0.5, "head.yaw": 4, "dur": 0.25,
           "hold": 0.6}, "But fine, I forgive you"),
         ({**br(0.5), "mouth.form": 0.8, "head.tilt": 6, "dur": 0.2,
           "hold": 0.5}, "if you eat something warm."),
         ({**sm(0.8), "mouth.form": 0.9, "head.pitch": -3, "dur": 0.2,
           "hold": 0.6}, "Deal?")],
        # 4: a blush first
        [({"cheek.blush": 0.7, "mouth.form": 0.4, "head.pitch": 6,
           "head.tilt": -4, "dur": 0.3, "hold": 0.9},
          "Um, you say that so suddenly."),
         ({**gz(-0.5, -0.3), "head.yaw": -6, "dur": 0.25, "hold": 0.7},
          "I do not know where to look."),
         ({**gz(0, 0), **br(0.6), "head.pitch": 3, "dur": 0.2,
           "hold": 0.4}, "Did you mean it?"),
         ({**sm(0.6), "mouth.form": 0.6, "head.tilt": 5, "dur": 0.2,
           "hold": 0.6}, "Then I am happy,"),
         ({**shut, "mouth.form": 0.8, "head.pitch": 4, "dur": 0.15,
           "hold": 0.5}, "really happy."),
         ({**br(-0.3), "mouth.form": -0.2, "head.yaw": 3, "head.tilt": -3,
           "dur": 0.25, "hold": 0.5}, "But do not tease me,"),
         ({"cheek.blush": 0.4, **sm(0.7), "mouth.form": 0.7, "dur": 0.3,
           "hold": 0.8}, "or I will hide."),
         ({**br(0.4), "head.pitch": -3, "head.yaw": -2, "dur": 0.2,
           "hold": 0.4}, "Say it once more?")],
        # 5: calm, a slow head tilt first
        [({"head.tilt": 7, "mouth.form": 0.3, "head.pitch": 2, "dur": 0.45,
           "hold": 1}, "Mm, I see."),
         ({**br(0.3), "head.pitch": 3, "dur": 0.25, "hold": 0.5},
          "So that is how it went."),
         ({**gz(0.4, -0.1), "head.yaw": 5, "dur": 0.3, "hold": 0.7},
          "Let me think for a moment,"),
         ({**gz(0, 0), "mouth.form": 0.2, "head.tilt": -4, "dur": 0.3,
           "hold": 0.6}, "because it matters."),
         ({**br(0.5), "head.yaw": -4, "head.pitch": 3, "dur": 0.2,
           "hold": 0.4}, "I think you did the right thing,"),
         ({**sm(0.6), "mouth.form": 0.6, "head.tilt": 5, "dur": 0.25,
           "hold": 0.7}, "even if it was hard."),
         ({**br(-0.2), "mouth.form": 0.1, "head.pitch": 4, "dur": 0.3,
           "hold": 0.6}, "Tomorrow will be easier,"),
         ({**sm(0.7), "mouth.form": 0.7, "head.pitch": -2, "head.yaw": 3,
           "dur": 0.25, "hold": 0.8}, "and I will be here.")],
    ]
    out = []
    for beats in V:
        ms = [mk(d) for d, _t in beats]
        if sum(1 for m in ms if m) < 5:
            continue
        out.append(" ".join((m + " " if m else "") + t
                            for m, (_d, t) in zip(ms, beats)))
    return out


def dense_variants(bm) -> list[str]:
    """_dense_variants for a body (its offered variables)."""
    return _dense_variants(bm, set(offered(bm)))


def variant_count(bm) -> int:
    """How many dense example variants the beats2 prompt has for this
    body (at least 1: a body with none shows no dense example)."""
    return max(1, len(dense_variants(bm)))


def variant_for(seed, bm) -> int:
    """The dense example variant of a generated session, from its seed
    alone (flow.dataset: the session's seed, itself from the run seed and
    the session id); serving uses variant 0."""
    return int(seed) % variant_count(bm)


def _beat_examples2(bm, variant=0) -> list[str]:
    """Placement examples for the "beats2" prompt: one dense reply (the
    `variant`th of _dense_variants) and two short ones that also change
    with it, so no opening is shown every time."""
    have = set(offered(bm))

    def mk(d):
        d = {k: v for k, v in d.items() if k in have or k in ("dur", "hold")}
        return "[pose " + json.dumps(d) + "]" \
            if any(k in have for k in d) else ""
    smile = {v: 0.8 for v in _bothsides(have, "eye", ("smile",))} \
        or {v: 0 for v in _bothsides(have, "eye", ("open",))}
    shut = {v: 0 for v in _bothsides(have, "eye", ("open",))}
    br = lambda x: {"brow.L.lift": x, "brow.R.lift": x}  # noqa: E731
    pool = [
        (mk({"gaze.x": 0.5, "gaze.y": -0.1, "head.yaw": 6, "dur": 0.25,
             "hold": 0.6}), "Hmm, I did not expect that.",
         mk({**smile, "mouth.form": 0.5, "head.tilt": -4, "dur": 0.25,
             "hold": 0.6}), "Well, maybe."),
        (mk({"cheek.blush": 0.5, "mouth.form": 0.4, "head.pitch": 5,
             "dur": 0.3, "hold": 0.8}), "Aww, really?",
         mk({**smile, "mouth.form": 0.9, "head.tilt": 5, "dur": 0.25,
             "hold": 0.6}), "You are too kind."),
        (mk({**br(-0.4), "mouth.form": -0.4, "head.tilt": 5, "dur": 0.25,
             "hold": 0.6}), "Hm.",
         mk({"gaze.x": -0.4, "head.yaw": -5, "dur": 0.25, "hold": 0.6}),
         "I am not sure about that."),
        (mk({**br(0.7), "head.pitch": 4, "head.yaw": -5, "dur": 0.2,
             "hold": 0.5}), "Wait, you did what?",
         mk({**smile, "mouth.form": 0.8, "head.tilt": 6, "dur": 0.25,
             "hold": 0.6}), "That is hilarious!"),
        (mk({**shut, "mouth.form": 1, "head.tilt": -5, "head.pitch": 3,
             "dur": 0.15, "hold": 0.5}), "Hahaha, stop it.",
         mk({**br(0.5), "mouth.form": 0.8, "head.pitch": -3, "dur": 0.2,
             "hold": 0.5}), "I cannot breathe."),
        (mk({"head.tilt": 7, "mouth.form": 0.3, "dur": 0.45, "hold": 1}),
         "Mm, I see.",
         mk({**br(0.3), "head.pitch": 3, "dur": 0.25, "hold": 0.5}),
         "Go on, I am listening."),
    ]
    ex = []
    dense = dense_variants(bm)
    if dense:
        ex.append(dense[variant % len(dense)])
    for k in (variant, variant + 3):
        a, ta, b, tb = pool[k % len(pool)]
        if a and b:
            ex.append(f"{a} {ta} {b} {tb}")
    return ex


def _beat_examples(bm) -> list[str]:
    """Placement examples for the "beats" prompt: face beats with their head
    move in the marker, from the variables this body has ([] when it has no
    face variable)."""
    have = set(offered(bm))

    def mk(d):
        d = {k: v for k, v in d.items() if k in have or k in ("dur", "hold")}
        return "[pose " + json.dumps(d) + "]" \
            if any(k in have for k in d) else ""
    smile = {v: 0.8 for v in _bothsides(have, "eye", ("smile",))} \
        or {v: 0 for v in _bothsides(have, "eye", ("open",))}
    a = mk({**smile, "mouth.form": 0.8, "head.tilt": 6, "dur": 0.25,
            "hold": 0.6})
    b = mk({"brow.L.lift": 0.7, "brow.R.lift": 0.7, "head.pitch": 4,
            "head.yaw": -5, "dur": 0.2, "hold": 0.5})
    c = mk({"mouth.form": -0.5, "brow.L.lift": -0.4, "brow.R.lift": -0.4,
            "head.tilt": -5, "head.yaw": 4, "dur": 0.25, "hold": 0.6})
    ex = []
    dense = _dense_example(bm, have)
    if dense:
        ex.append(dense)
    if a and b:
        ex.append(f"{b} Wait, you did what? {a} That's hilarious!")
    if c and a:
        ex.append(f"Hmm, {c} I'm not sure about that, {a} but it sounds "
                  "fun.")
    return ex


def _examples(bm, beats=False) -> list[str]:
    have = set(offered(bm))
    if beats:
        ex = _beat_examples(bm)
        if ex:
            return ex
    ex = []
    best = _best_hand(bm)
    if best is not None:
        s, top = best
        sh = bm.landmarks.get("shoulder.y")
        y = ('"@shoulder.y"' if sh is not None and top >= sh - fb.TOL["hu"]
             else f"{top - 0.05:.2f}")
        tilt = ', "head.tilt": 8' if "head.tilt" in have else ""
        ex.append(f"Hi! [pose {{\"hand.{s}.y\": {y}{tilt}}}] It's so good to "
                  "see you.")
    if "head.pitch" in have:
        ex.append("Hmm, [pose {\"head.pitch\": -12, \"dur\": 0.25, \"hold\": "
                  "0}][pose {\"head.pitch\": 0, \"dur\": 0.3, \"hold\": 0}] "
                  "yes, that works for me.")
    face = [v for v in ("brow.L.lift", "brow.R.lift") if v in have]
    look = [v for v in ("gaze.x", "head.yaw") if v in have]
    if face and look:
        # A glance away and back, quick: the brief beat the tags' look:away
        # gave and E2's body replies (holds of 2-3 s) did not.
        away = {"gaze.x": -0.6, "head.yaw": -12}
        out_ = ", ".join(f"\"{v}\": {away[v]:g}" for v in look)
        back = ", ".join(f"\"{v}\": 0" for v in look)
        ex.append("[pose {" + ", ".join(f"\"{v}\": 0.8" for v in face)
                  + ", \"dur\": 0.2}] Really? [pose {" + out_
                  + ", \"dur\": 0.3, \"hold\": 0.5}] I hadn't [pose {" + back
                  + ", \"dur\": 0.35, \"hold\": 0}] noticed.")
    if not ex:
        ex.append("Oh... [pause:0.5] [blink] I didn't expect that.")
    return ex


def system_prompt(bm, persona=None, images=False, gestures="",
                  moves="", prompt="v1", variant=0) -> str:
    """Byte-stable for a body file, persona, image setting and gesture
    lines: the current pose goes in the user message (pose_now), never
    here. gestures: the [gesture:NAME] entry (director.gesture_lines) when
    body mode offers the rig's own motions (--body-gestures). moves: the
    [move] entry (flow.moves.prompt_lines) when it offers motion
    primitives (--body-moves); "" is the prompt of before them. prompt: "v1"
    the first teacher prompt, "beats" the expression beats, "beats2" the
    same with several dense examples (`variant` picks the one shown: the
    data uses all of them, serving variant 0), the [frame full] entry
    before whole-body moves and the reply-language line (PROMPTS)."""
    if prompt not in PROMPTS:
        raise ValueError(f"prompt {prompt!r}: one of {', '.join(PROMPTS)}")
    beats = beats_like(prompt)
    v2 = prompt == "beats2"
    ex = (_beat_examples2(bm, variant) or _examples(bm, False)) if v2 \
        else _examples(bm, beats)
    if moves:
        try:
            from . import moves as mv
        except ImportError:
            from flow import moves as mv
        ex.append(mv.PROMPT_EXAMPLE.replace("[move", "[frame full][move", 1)
                  if v2 and fr.can_frame(bm) else mv.PROMPT_EXAMPLE)
    persona = (persona or "a friendly, expressive character.").strip()
    # A VRM body says what it is (flow.vrmbody); a Live2D one is the
    # default, its prompt the same bytes as before E3.
    return BODY_TEMPLATE.format(
        kind=getattr(bm, "prompt_kind", "a Live2D character"),
        persona=persona, body=describe_body(bm), hold=POSE_HOLD,
        image=(", and with it a picture of how you look right now, as the "
               "viewer sees you (your left is on the picture's right)"
               if images else ""),
        gestures=(gestures or "") + (
            frame_entry2(bm, bool(moves)) if v2 else
            frame_entry(bm) if beats else ""),
        whole="" if beats else _whole_body(bm, bool(gestures)),
        beats=_beats_rule(bm, v2) if beats else "",
        sentence_rule=SENTENCE_RULE[prompt], moves=moves or "", move_rule=(mv.PROMPT_RULE if moves else ""),
        examples="\n".join(ex) + "\n")


# ---------------------------------------- reply language and the turn's tail

LANG_NAMES = {"en": "English", "ja": "Japanese", "zh": "Chinese (Simplified)",
              "ko": "Korean"}
# The names the beats2 user turn carries ("(reply in: English)").
REPLY_NAMES = {"en": "English", "ja": "Japanese", "zh": "Chinese",
               "ko": "Korean"}


def detect_lang(text):
    """The main language of a line by script: 'en', 'ja', 'zh', 'ko' or
    None (flow.speech.script_language; Han alone is Chinese). A line that
    mixes scripts is the language of most of its letters: a line of English
    with a kanji or a kana word in it is English."""
    text = text or ""
    lang = sp.script_language(text)
    lang = {"han": "zh"}.get(lang, lang)
    if lang in ("ja", "zh", "ko"):
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        other = sum(1 for c in text if 0x1100 <= ord(c) <= 0x11FF
                    or 0x3040 <= ord(c) <= 0x30FF
                    or 0x3400 <= ord(c) <= 0x9FFF
                    or 0xAC00 <= ord(c) <= 0xD7AF)
        if latin >= 8 and latin > 3 * other:
            return "en"
    return lang


def reply_line(text) -> str:
    """The user turn's reply-language line of the beats2 prompt:
    "(reply in: English)", from the user's latest line (detect_lang; ""
    when it settles nothing, e.g. "OK!"). Training data and serving both
    write it with this function."""
    name = REPLY_NAMES.get(detect_lang(text))
    return f"(reply in: {name})" if name else ""


def turn_tail(bm, framing, text, prompt) -> str:
    """What follows the pose line of a user turn under `prompt`: "" for v1;
    "\n(camera: ...)" for beats; for beats2 also "\n(reply in: ...)" next
    to it. `framing`: the framing the turn starts in; `text` the user's
    line. The data generator, the director and the live loop all call
    this, so what the student was taught is what it is served."""
    if not beats_like(prompt):
        return ""
    out = "\n" + camera_now(bm, framing)
    if prompt == "beats2":
        line = reply_line(text)
        if line:
            out += "\n" + line
    return out


def pose_now(bm, values=None, parts=None) -> str:
    """"(your body now: ...)": the offered variables at these parameter
    values, those away from rest by more than their tolerance, and the
    hands always."""
    values = dict(values or {})
    k = bm.match_variant(values, parts)
    st = bm.perceive({**bm.default_values(), **values}, k)
    v0 = bm.variants[0]
    items = []
    for var in offered(bm):
        if var in HAND_VARS or var not in st:
            continue
        u = fb.feature_unit(var)
        rest = float(v0["rest"][bm.fidx[var]])
        if abs(st[var] - rest) > fb.TOL[u]:
            items.append(f"{var} {_fmt(st[var], u)}")
    for s in "LR":
        if f"hand.{s}.x" in st:
            items.append(f"hand.{s} ({st[f'hand.{s}.x']:.2f}, "
                         f"{st[f'hand.{s}.y']:.2f})")
    if len(bm.variants) > 1 and k:
        items.insert(0, "arms in another drawing")
    rest_word = "" if any(not i.startswith("hand.") for i in items) else \
        "at rest; "
    return "(your body now: " + rest_word + ", ".join(items) + ")"


def user_message(text, backend=None, image=None):
    """A user turn, with a PNG (bytes) when the backend takes images:
    Anthropic content blocks, OpenAI-compatible image_url data URLs
    (SiliconFlow's VLMs), Ollama's images field. Other backends (DeepSeek
    chat is text only) get the text."""
    if image is None or backend not in IMAGE_BACKENDS:
        return {"role": "user", "content": text}
    b64 = base64.b64encode(image).decode("ascii")
    if backend == "anthropic":
        return {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": b64}},
            {"type": "text", "text": text}]}
    if backend == "ollama":
        return {"role": "user", "content": text, "images": [b64]}
    return {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url":
                                            f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": text}]}


def text_only(msg):
    """The message without its image (history keeps the words only: an old
    picture of oneself is stale and costs a thousand tokens a turn)."""
    c = msg.get("content")
    if isinstance(c, list):
        c = "".join(b.get("text", "") for b in c if b.get("type") == "text")
    return {"role": msg["role"], "content": c}


# ----------------------------------------------------------------- solve

def _main_movers(bm, k, targets, free):
    """{target: the free parameter that moves it most} in variant k."""
    v = bm.variants[k]
    out = {}
    for t in targets:
        j = bm.fidx[t]
        best, g = None, 0.0
        for p in free:
            D = v["sweeps"][p][1]
            m = float(np.abs(D[:, j]).max())
            if m > g:
                best, g = p, m
        if best is not None:
            out[t] = best
    return out


def solve_poses(bm, acts, current=None, parts=None, core=None, log=print,
                keep_shown=False):
    """Solve the pose actions among `acts` (timeline / speech.speak output:
    {"t", "tag": "pose", "arg": {"targets", "dur"?, "hold"?}, "at"}).

    Each is solved with the prior toward the pose the body is in at its
    time: `current` (parameter values; default the rest) with the earlier
    poses still in their move or hold laid over. A pose on variables an
    earlier one is still moving starts when that move ends (after its hold
    too, when the hold was written). Returns one record per pose: {t (as
    written), start, dur, hold, release, targets (as written), resolved,
    params (the played set), parts (only on a variant change), step
    (parameters that switch rather than blend), variant, variant_name,
    report {var: {target, got, error, unit, ok}}, exact {var: value} and
    exact_ok with a Core, ok, solve_seconds, act (its index in acts)}.
    """
    import time
    base0 = bm.default_values()
    base0.update(current or {})
    parts0 = dict(parts if parts is not None else
                  {**bm.blob.get("rest_parts", {}),
                   **bm.variants[bm.match_variant(base0)]["parts"]})
    base_k = bm.match_variant(base0, parts0)
    arms = arm_params(bm)
    out = []
    order = sorted((i for i, x in enumerate(acts) if x.get("tag") == POSE),
                   key=lambda i: float(acts[i]["t"]))
    for idx in order:
        a = acts[idx]
        spec = a.get("arg") or {}
        raw = dict(spec.get("targets") or {})
        t = float(a["t"])
        try:
            resolved = bm.resolve(raw)
        except (ValueError, KeyError) as exc:
            log(f"pose: {raw}: {exc}; dropped")
            continue
        resolved = {k: v for k, v in resolved.items() if k in bm.fidx}
        if not resolved:
            continue
        # Where the body is at t: the rest, under every earlier pose still
        # moving or held.
        vals, prt = dict(base0), dict(parts0)
        start = t
        for q in out:
            q_end = q["start"] + q["dur"] + q["hold"]
            if t < q_end and (not q.get("via") or q["start"] <= t):
                vals.update(q["params"])
                prt.update(q["parts"])
            if set(q["vars"]) & set(resolved):
                wait = q["start"] + q["dur"] + (q["hold"] if q["hold_set"]
                                                 else 0.0)
                start = max(start, wait if t < wait else start)
        # A detour back out (a via leg, below) not begun when this pose on
        # the same variables starts is dropped: this pose leaves from where
        # that one held, by its own way.
        out = [q for q in out if not (q.get("via") == "out" and
                                      set(q["vars"]) & set(resolved) and
                                      q["start"] >= start)]
        cur_k = bm.match_variant(vals, prt)
        t0 = time.monotonic()
        # Composite targets (a VRM body's "hands": "@together") go to the
        # solver as written too: it turns the hands from them.
        comp = {c: raw[c] for c in getattr(bm, "composite_keys", ())
                if c in raw}
        res = bm.solve({**resolved, **comp} if comp else resolved,
                       current=vals, parts=prt)
        sec = time.monotonic() - t0
        k = res.get("variant")
        if k is None:
            log(f"pose: {raw}: no variant has these variables; dropped")
            continue
        v = bm.variants[k]
        movers = _main_movers(bm, k, resolved, res.get("free", []))
        played = {}
        amp = 0.0
        for p, x in res["params"].items():
            e = bm.params[p]
            span = max(e["max"] - e["min"], 1e-9)
            d = abs(x - vals.get(p, e["default"])) / span
            if d >= PLAY_EPS or p in movers.values():
                played[p] = float(x)
                amp = max(amp, d)
        step = sorted(p for p in v["switches"])
        new_parts = {}
        if k != cur_k or k != base_k or set(played) & arms:
            # The pose carries its arm drawing whenever it moves an arm or
            # is off the rest drawing, even when the body is already in it:
            # an earlier pose's release swapped Hiyori's arms back to the
            # hanging drawing in the middle of a later pose's hold (her
            # wave-and-point line), and an idle gesture can be playing the
            # other drawing (Hiyori_m06 and m08, both calm, key PartArmB).
            played.update({p: float(x) for p, x in v["switches"].items()})
            new_parts = {p: float(x) for p, x in v["parts"].items()}
            if k != cur_k:
                amp = max(amp, 0.5)
        else:
            step = []
        step += sorted(p for p in played if bm.params[p]["role"] in
                       ("switch", "variant") and p not in step)
        dur = spec.get("dur")
        dur = float(dur) if dur is not None else round(
            DUR_MIN + (DUR_MAX - DUR_MIN) * min(amp, 1.0), 3)
        hold_set = spec.get("hold") is not None
        hold = float(spec["hold"]) if hold_set else POSE_HOLD
        # What renders: the played set over the pose the body was in.
        shown = {**vals, **played}
        F = bm.forward(shown, k)
        report = {}
        for var, y in resolved.items():
            got = F[bm.fidx[var]]
            u = fb.feature_unit(var)
            if math.isnan(got):
                report[var] = {"target": y, "got": None, "unit": u,
                               "ok": False}
                continue
            err = float(got - y)
            report[var] = {"target": round(y, 5), "got": round(float(got), 5),
                           "error": round(err, 5), "unit": u,
                           "ok": abs(err) <= fb.TOL[u]}
        # The solver's own verdicts (a VRM body): hands placed together by
        # a composite are where they meet; a hand only reaching through
        # the body is a miss.
        for var in res.get("met", ()):
            if var in report:
                report[var]["ok"] = True
        for var, why in res.get("why", {}).items():
            if var in report and why and var not in res.get("met", ()) \
                    and not res["report"].get(var, {}).get("ok", True):
                report[var]["ok"] = False
        rec = {"act": idx, "t": round(t, 4), "at": a.get("at"),
               "start": round(start, 4), "dur": round(dur, 4),
               "hold": round(hold, 4), "hold_set": hold_set,
               "release": round(max(dur, RELEASE_MIN), 4),
               "targets": raw, "resolved": {k2: round(v2, 5) for k2, v2
                                            in resolved.items()},
               "vars": sorted(resolved),
               "params": {p: round(x, 6) for p, x in sorted(played.items())},
               "parts": new_parts,
               "parts_from": {q: float(prt.get(q, 1.0 - x))
                              for q, x in new_parts.items()},
               "step": step, "variant": k,
               "variant_name": v["name"], "report": report,
               "ok": all(e["ok"] for e in report.values()),
               "solve_seconds": round(sec, 3)}
        if not rec["ok"]:
            rec["hints"] = hints_text(bm, res, report)
        rec["_shown"] = shown
        via = res.get("via")
        if via:
            # The straight way there (and back) goes through the body (a
            # VRM body: flow.vrmbody._via): pass through `via` on the way
            # in and again before the release, each leg half the move.
            half = round(max(dur / 2, DUR_RANGE[0]), 4)

            def leg(start_, which):
                return {"act": idx, "t": rec["t"], "at": rec["at"],
                        "start": round(start_, 4), "dur": half, "hold": 0.0,
                        "hold_set": True,
                        "release": round(max(half, RELEASE_MIN), 4),
                        "targets": {}, "resolved": {}, "vars": rec["vars"],
                        "params": dict(via), "parts": {}, "parts_from": {},
                        "step": [], "variant": k, "variant_name": v["name"],
                        "report": {}, "ok": True, "solve_seconds": 0.0,
                        "via": which}
            out.append(leg(start, "in"))
            rec["start"] = round(start + half, 4)
            rec["dur"] = half
            # Held still: the hold's settle drifts every angle toward the
            # rest along the straight line (0.2 of it put the hands 0.32-
            # 0.43 hu into the hips by the end of a hold behind the back).
            rec["settle"] = 0.0
            out.append(rec)
            out.append(leg(rec["start"] + rec["dur"] + rec["hold"], "out"))
            continue
        out.append(rec)
    if core is not None:
        add_exact(bm, core, out, log)
    if not keep_shown:
        for rec in out:
            rec.pop("_shown", None)
    return out


def add_exact(bm, core, poses, log=print):
    """The Core's own geometry at each pose as played (body.exact_state):
    rec["exact"] {var: value}, rec["exact_ok"] {var: within tolerance}.
    Needs the records' "_shown" (solve_poses keep_shown=True, or core
    given there), which it removes. flow.live runs it after the poses are
    on the stream, off the path to the first audio."""
    for rec in poses:
        shown = rec.pop("_shown", None)
        if shown is None or core is None:
            continue
        try:
            ex = fb.exact_state(core, bm, shown, rec["variant"])
        except Exception as exc:  # noqa: BLE001 - a check, not the play
            log(f"pose: exact check failed: {exc!r}")
            continue
        rec["exact"] = {var: round(ex[var], 5) for var in rec["resolved"]
                        if var in ex}
        rec["exact_ok"] = {var: abs(rec["exact"][var] - rec["resolved"][var])
                           <= fb.TOL[fb.feature_unit(var)]
                           for var in rec["exact"]}


def hints_text(bm, res, report) -> dict:
    """{var: words} for the missed targets (body.miss_hints)."""
    r = dict(res, report=report)
    out = {}
    for t, h in bm.miss_hints(r).items():
        msg = []
        if h["reach"]:
            lo, hi = h["reach"]
            msg.append(f"alone it reaches {lo:.2f} to {hi:.2f} in this pose")
        for name, lim in h["variants"].items():
            msg.append(f"another drawn pose reaches {lim[0]:.2f} to "
                       f"{lim[1]:.2f}")
        if h["mirror"] and h["mirror"][1]:
            m, lim = h["mirror"]
            msg.append(f"{m} reaches {lim[0]:.2f} to {lim[1]:.2f}")
        if h.get("why"):
            msg.insert(0, h["why"])
        out[t] = "; ".join(msg)
    return out


def reach_summary(poses) -> dict:
    """{markers, targets, ok, miss, exact_ok, exact_n, solve_seconds}."""
    n = ok = eok = en = 0
    poses = [p for p in poses if not p.get("via")]
    for p in poses:
        for var, e in p["report"].items():
            n += 1
            ok += bool(e["ok"])
        for var, good in p.get("exact_ok", {}).items():
            en += 1
            eok += bool(good)
    return {"markers": len(poses), "targets": n, "ok": ok, "miss": n - ok,
            "exact_ok": eok, "exact_n": en,
            "solve_seconds": round(sum(p["solve_seconds"] for p in poses), 3)}


def report_lines(poses, spoken="") -> list[str]:
    """One line per pose for the logs: targets, got, ok/MISS, Core check."""
    out = []
    for p in poses:
        if p.get("via"):
            continue
        bits = []
        for var, e in p["report"].items():
            u = e["unit"]
            if e["got"] is None:
                bits.append(f"{var} {_fmt(e['target'], u)}: not drawn MISS")
                continue
            ex = p.get("exact", {}).get(var)
            bits.append(f"{var} {_fmt(e['target'], u)} -> "
                        f"{_fmt(e['got'], u)}"
                        + (f" (Core {_fmt(ex, u)})" if ex is not None else "")
                        + (" ok" if e["ok"] else " MISS"))
        at = p.get("at")
        words = (f" at {spoken[at:at + 16].strip()!r}"
                 if spoken and at is not None else "")
        out.append(f"pose {p['start']:.2f}s{words} [{p['variant_name']}]: "
                   + "; ".join(bits)
                   + "".join(f"  ({v}: {h})" for v, h in
                             p.get("hints", {}).items()))
    return out


def drop_clashing_gestures(acts, poses, motions, log=print):
    """--body-gestures: a directed gesture plays the rig's own arm motion
    (director.realize: up to GESTURE_MAX seconds, then SETTLE of the idle
    loop); a pose that moves an arm or swaps its drawing at the same time
    would write the same parameters and parts. The pose wins -- the model
    placed that hand on purpose, a gesture is the rig's -- and a gesture
    whose play overlaps such a pose's move, hold or release is dropped
    with a line. acts is changed in place and each pose's "act" index
    follows; returns the dropped gesture actions. motions: {motion name:
    seconds}."""
    labels = dr.gesture_labels(motions)
    arm = [(p["start"], p["start"] + p["dur"] + p["hold"] + p["release"])
           for p in poses if p.get("arm_params") or p.get("parts")]
    if not arm:
        return []
    keep, gone = [], []
    for x in acts:
        name = labels.get(str(x.get("arg")).lower()) \
            if x.get("tag") == "gesture" else None
        if name is not None:
            t0 = float(x["t"])
            t1 = t0 + min(float(motions[name]), dr.GESTURE_MAX) + dr.SETTLE
            if any(t0 < b and t1 > a for a, b in arm):
                gone.append(x)
                log(f"pose: [gesture:{x['arg']}] at {t0:.2f}s dropped: a "
                    "pose moves an arm then")
                continue
        keep.append(x)
    if gone:
        new = {id(x): i for i, x in enumerate(keep)}
        for p in poses:
            p["act"] = new.get(id(acts[p["act"]]), p["act"])
        acts[:] = keep
    return gone


def end_time(poses, ease=False) -> float:
    """When the last pose has relaxed back into the idle. ease: as played
    under organic playback (a head beat arrives and relaxes more slowly)."""
    def end(p):
        d, he, rel = (eased(p["dur"], p["dur"] + p["hold"], p["release"],
                            True, 30.0) if ease else
                      (p["dur"], p["dur"] + p["hold"], p["release"]))
        return p["start"] + he + rel
    return max((end(p) for p in poses), default=0.0)


# ----------------------------------------------------------------- tokens

def arm_params(bm) -> set:
    """Parameters that move a hand or an arm by more than 0.02 hu in some
    variant: an unbound one takes the "arms" group (idle gestures)."""
    cached = bm.__dict__.get("_arm_params")
    if cached is not None:
        return cached
    out = set()
    regs = [r for r in ("arm.L", "arm.R", "hand.L", "hand.R")]
    for v in bm.variants:
        for p, (_g, D) in v["sweeps"].items():
            for r in regs:
                kx, ky = bm.fidx[f"{r}.x"], bm.fidx[f"{r}.y"]
                if np.hypot(D[:, kx], D[:, ky]).max() > 0.02:
                    out.add(p)
                    break
    bm.__dict__["_arm_params"] = out
    return out


def tokens(poses, roles, arms=()) -> list[dict]:
    """Pose tokens for the event stream. roles: runtime's {role: (pid,
    sign, source)}; a played parameter bound to a role takes that role's
    group (director.merge then drops the idle's moves on it for the move
    plus hold), an unbound one in `arms` takes "arms"."""
    group_of_pid = {p: ev.GROUP_OF.get(r) for r, (p, _s, _src) in
                    roles.items()}
    arms = set(arms)
    out = []
    for p in poses:
        groups = set()
        for pid in list(p["params"]) + list(p["parts"]):
            g = group_of_pid.get(pid)
            if g:
                groups.add(g)
            elif pid in arms or pid in p["parts"]:
                groups.add("arms")
        out.append({"t": p["start"], "type": POSE, "role": POSE,
                    "params": p["params"], "parts": p["parts"],
                    "parts_from": p.get("parts_from", {}),
                    "step": p["step"], "duration": p["dur"],
                    "hold": p["hold"], "release": p["release"],
                    "groups": sorted(groups), "variant": p["variant_name"],
                    "src": "director",
                    **({"settle": p["settle"]} if "settle" in p else {})})
    return out


# ------------------------------------------------------------------- play

def eased(d, he, rel, angle, fps, t0=0.0):
    """(move, hold end, release) of one continuous parameter under organic
    playback: the move at least ANGLE_DUR (angle) or FACE_DUR, the hold as
    long as written (it starts when the longer move is done), the release
    at least ANGLE_RELEASE (angle) or RELEASE_MIN."""
    d2 = max(d, ANGLE_DUR if angle else FACE_DUR)
    return (d2, he + (d2 - d),
            max(rel, ANGLE_RELEASE if angle else RELEASE_MIN))


class PoseLayer:
    """Pose tokens played over the frame the idle built, a frame at a time.

    commit(tok) queues a token; apply(f, over, parts) is frame f: tokens
    whose onset frame (ceil(t * fps)) has come start there, each of their
    parameters moving from what this frame would show without the new
    token (an earlier pose's value, or the idle's) to the target with a
    minimum-jerk profile over `duration`, held until duration + hold, then
    released over `release` toward the idle's value of each frame -- so
    nothing jumps when a pose starts, is replaced, or ends. Parameters in
    `step` and part opacities switch at the midpoint of the move (and of
    the release) instead of blending: two arm variants are never drawn at
    once. A part the frame leaves unset is taken at `part_rest`, else at the
    token's "parts_from" (the body's opacity before the swap). Parts, once
    written, keep being written (the backend keeps a
    part's last opacity, while parameters reset every frame), with the
    idle's value or `part_rest`. `written` is the set of parameters this
    frame wrote, for runtime's --dump mask.

    Held targets keep a little life (continuous parameters only): `life`
    times what the idle under the parameter has moved since the token
    started is added (eased in over the move), and over the hold the
    target settles `settle` of the way toward the idle (min-jerk in the
    hold's time), the release then starting from there. life = settle = 0
    is the frozen hold of before.

    ease (organic playback, see ANGLE_DUR): a continuous parameter takes at
    least ANGLE_DUR (those in `angles`: head, torso, body angles) or
    FACE_DUR seconds to arrive and at least ANGLE_RELEASE / RELEASE_MIN to
    relax; onsets and targets are the token's, the hold lasts as long as it
    was written to, from when the longer move is done. Off: the
    token's own times, as the dataset and the action head's data play them.
    """

    def __init__(self, fps, defaults=None, part_rest=None, life=LIFE,
                 settle=SETTLE, ease=False, angles=()):
        self.fps = float(fps)
        self.ease, self.angles = bool(ease), set(angles)
        self.life, self.settle = float(life), float(settle)
        self.defaults = dict(defaults or {})
        self.part_rest = dict(part_rest or {})
        self.queue = []
        self.state = {}
        self.touched = set()
        self.written = set()

    def commit(self, tok):
        self.queue.append(tok)

    def busy(self):
        return bool(self.queue or self.state or self.touched)

    def _eval(self, s, tf, under, life=True):
        # State: (t0, duration, from, to, hold end, release, step, the
        # idle's value at t0, the token's own settle or None);
        # stream.body_now reads "to" as s[3]. life False: none of the
        # motion underneath shows (a joint a move plays, flow.moves).
        t0, d, frm, to, he, rel, step, u0, settle = s
        settle = self.settle if settle is None else settle
        if step:
            if tf < t0 + d:
                return (to if (tf - t0) / d >= 0.5 else frm), False
            if tf < he:
                return to, False
            if tf < he + rel:
                return (under if (tf - he) / rel >= 0.5 else to), False
            return under, True
        life = self.life * (under - u0) if life else 0.0
        if tf < t0 + d:
            w = float(min_jerk(max((tf - t0) / d, 0.0)))
            return frm + (to - frm) * w + life * w, False
        held = to + life
        hold = he - (t0 + d)
        if settle and hold > 0:
            tau = min(max((tf - t0 - d) / hold, 0.0), 1.0)
            held += settle * float(min_jerk(tau)) * (under - held)
        if tf < he:
            return held, False
        if tf < he + rel:
            u = (tf - he) / rel
            return held + (under - held) * float(min_jerk(u)), False
        return under, True

    def apply(self, f, over, parts, still=()):
        """still: parameters whose held targets keep no life (those a
        flow.moves clip plays absolutely under the pose)."""
        self.written = set()
        if not self.busy():
            return over, parts
        tf = f / self.fps
        over = dict(over)
        parts_in = parts
        parts = dict(parts or {})

        def under(key):
            kind, i = key
            if kind == "p":
                return float(over.get(i, self.defaults.get(i, 0.0)))
            return float(parts.get(i, self.part_rest.get(i, 1.0)))

        while self.queue and int(math.ceil(float(self.queue[0]["t"])
                                           * self.fps - 1e-9)) <= f:
            tok = self.queue.pop(0)
            t0 = float(tok["t"])
            d = max(float(tok.get("duration", DUR_MIN)), 1.0 / self.fps)
            he = t0 + d + max(float(tok.get("hold", 0.0)), 0.0)
            rel = max(float(tok.get("release", d)), 1.0 / self.fps)
            step = set(tok.get("step", ()))
            for i, v in tok.get("parts_from", {}).items():
                # Where the frame itself sets no opacity for a swapped part
                # (a rig whose motions never key it), the body's own state
                # before the swap stands in: never both arms at 1.
                self.part_rest.setdefault(i, float(v))
            for kind, items in (("p", tok.get("params", {})),
                                ("q", tok.get("parts", {}))):
                for i, v in items.items():
                    key = (kind, i)
                    u = under(key)
                    frm = (self._eval(self.state[key], tf, u,
                                      i not in still)[0]
                           if key in self.state else u)
                    d_i, he_i, rel_i = d, he, rel
                    if self.ease and kind == "p" and i not in step:
                        d_i, he_i, rel_i = eased(
                            d, he, rel, i in self.angles, self.fps, t0)
                    self.state[key] = (t0, d_i, frm, float(v), he_i, rel_i,
                                       kind == "q" or i in step, u,
                                       tok.get("settle"))
                    if kind == "q":
                        self.touched.add(i)
        for key in sorted(self.state):
            kind, i = key
            val, done = self._eval(self.state[key], tf, under(key),
                                   i not in still)
            if done:
                del self.state[key]
            if kind == "p":
                if not done:
                    over[i] = val
                    self.written.add(i)
            else:
                parts[i] = val
        for i in self.touched:
            if ("q", i) not in self.state:
                parts[i] = under(("q", i))
        if parts_in is None and not parts:
            parts = None
        return over, parts


def layer_for(tokens_, fps, defaults, part_rest=None, ease=False,
              angles=()):
    """A PoseLayer with every pose token of a stream committed, in order;
    None when there are none (tag mode: nothing changes)."""
    toks = [k for k in tokens_ if k.get("type") == POSE]
    if not toks:
        return None
    lay = PoseLayer(fps, defaults, part_rest, ease=ease, angles=angles)
    for k in sorted(toks, key=lambda k: float(k["t"])):
        lay.commit(k)
    return lay


# ------------------------------------------------------------- self-check

CHECK_PROMPT = (
    "This is what your body did with your markers (target -> reached; ok, "
    "or MISS when it is off by more than its tolerance):\n{lines}\n{image}"
    "If every pose did what you meant, reply with just OK. Otherwise reply "
    "with your whole reply again, the same words, with corrected markers.")


def check_message(poses, spoken, image_note=False) -> str:
    lines = []
    for n, p in enumerate([q for q in poses if not q.get("via")], 1):
        at = p.get("at")
        words = (spoken[at:at + 24].strip() if spoken and at is not None
                 else "")
        bits = []
        for var, e in p["report"].items():
            u = e["unit"]
            got = p.get("exact", {}).get(var, e["got"])
            if got is None:
                bits.append(f"{var} {_fmt(e['target'], u)} -> not drawn MISS")
                continue
            ok = p.get("exact_ok", {}).get(var, e["ok"])
            bits.append(f"{var} {_fmt(e['target'], u)} -> {_fmt(got, u)} "
                        + ("ok" if ok else "MISS"))
        h = "".join(f" ({v}: {t})" for v, t in p.get("hints", {}).items())
        lines.append(f"{n}. before {words!r}: " + "; ".join(bits) + h)
    return CHECK_PROMPT.format(
        lines="\n".join(lines),
        image=("The picture shows you at the last pose.\n" if image_note
               else ""))


def is_ok_reply(text) -> bool:
    return bool(re.match(r"\s*\**\s*ok\b[\s.!]*$", text or "", re.I))


# ---------------------------------------------------------------- summary

def summarize(root) -> list[dict]:
    """Rows per (mode, condition) under root/<cond>/<i>/: replies, actions
    per reply, reach, MISS, Core checks, model latency, score checks."""
    rows = []
    if not os.path.isdir(root):
        return rows
    for cond in sorted(os.listdir(root)):
        d = os.path.join(root, cond)
        if not os.path.isdir(d):
            continue
        r = {"cond": cond, "replies": 0, "actions": 0, "markers": 0,
             "targets": 0, "ok": 0, "miss": 0, "exact_ok": 0, "exact_n": 0,
             "warn": 0, "model": [], "first": [], "solve": 0.0,
             "score_pass": 0, "scored": 0, "checks": {}, "fixed": [],
             "kinds": {}}
        for i in sorted(os.listdir(d)):
            f = os.path.join(d, i, "actions.json")
            if not os.path.exists(f):
                continue
            with open(f, encoding="utf-8") as fh:
                a = json.load(fh)
            r["replies"] += 1
            acts = a.get("actions", [])
            r["actions"] += len(acts)
            for x in acts:
                key = x["tag"] if x["tag"] in (POSE, "pause", "gesture") \
                    else f"move:{x['arg'].get('name')}" if x["tag"] == "move" \
                    else f"{x['tag']}:{x['arg']}" if x.get("arg") else x["tag"]
                r["kinds"][key] = r["kinds"].get(key, 0) + 1
            r["warn"] += len(a.get("warnings", []))
            rs = a.get("reach")
            if rs:
                for k in ("markers", "targets", "ok", "miss", "exact_ok",
                          "exact_n"):
                    r[k] += rs.get(k, 0)
                r["solve"] += rs.get("solve_seconds", 0.0)
            if a.get("model_seconds") is not None:
                r["model"].append(a["model_seconds"])
            if a.get("first_text_seconds") is not None:
                r["first"].append(a["first_text_seconds"])
            sf = os.path.join(d, i, "score.json")
            if os.path.exists(sf):
                with open(sf, encoding="utf-8") as fh:
                    s = json.load(fh)
                r["scored"] += 1
                r["score_pass"] += bool(s.get("pass"))
                for k, v in (s.get("checks") or {}).items():
                    r["checks"][k] = r["checks"].get(k, 0) + bool(v)
            cf = os.path.join(d, i, "check", "check.json")
            if os.path.exists(cf):
                with open(cf, encoding="utf-8") as fh:
                    c = json.load(fh)
                r["fixed"].append(c)
        if r["replies"]:
            rows.append(r)
    return rows


def print_summary(rows, out=sys.stdout):
    avg = lambda v: f"{sum(v) / len(v):6.1f}" if v else f"{'-':>6}"  # noqa
    print(f"{'condition':<24} {'n':>2} {'act/r':>5} {'mark':>4} {'tgt':>4} "
          f"{'reach':>6} {'MISS':>4} {'Core':>7} {'warn':>4} {'model':>6} "
          f"{'first':>6} {'solve':>6} {'score':>6}  checks / kinds", file=out)
    for r in rows:
        reach = f"{r['ok'] / r['targets']:.0%}" if r["targets"] else "-"
        core = (f"{r['exact_ok']}/{r['exact_n']}" if r["exact_n"] else "-")
        sc = f"{r['score_pass']}/{r['scored']}" if r["scored"] else "-"
        checks = ", ".join(f"{k} {v}/{r['scored']}" for k, v in
                           sorted(r["checks"].items()))
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(
            r["kinds"].items(), key=lambda x: -x[1]))
        print(f"{r['cond']:<24} {r['replies']:>2} "
              f"{r['actions'] / r['replies']:>5.1f} {r['markers']:>4} "
              f"{r['targets']:>4} {reach:>6} {r['miss']:>4} {core:>7} "
              f"{r['warn']:>4} {avg(r['model'])} {avg(r['first'])} "
              f"{r['solve']:>6.1f} {sc:>6}  {checks}" + (f" | {kinds}"
                                                          if kinds else ""),
              file=out)
        if r["fixed"]:
            b = sum(c["before"]["ok"] for c in r["fixed"])
            bt = sum(c["before"]["targets"] for c in r["fixed"])
            a = sum(c["after"]["ok"] for c in r["fixed"])
            at = sum(c["after"]["targets"] for c in r["fixed"])
            n_ok = sum(c.get("model_ok", False) for c in r["fixed"])
            print(f"{'':<24}    self-check: {len(r['fixed'])} rounds, "
                  f"{n_ok} answered OK; reach {b}/{bt} -> {a}/{at}", file=out)


# -------------------------------------------------------------------- cli

def snapshot_main(argv) -> int:
    """`snapshot`: the body at some parameter values, as a PNG on a light
    background (what a vision model is shown). Needs GL."""
    ap = argparse.ArgumentParser(prog="flow.pose snapshot")
    ap.add_argument("--body", required=True)
    ap.add_argument("--model-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--current", default=None, help="JSON parameter values")
    ap.add_argument("--parts", default=None, help="JSON part opacities")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args(argv)
    try:
        from ..l2dprobe.backend import Live2DBackend, ensure_display
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        from l2dprobe.backend import Live2DBackend, ensure_display
    ensure_display(["-m", "flow.pose", "snapshot"] + list(argv))
    from PIL import Image
    bm = fb.load(a.body)
    cur = json.loads(a.current) if a.current else {}
    parts = json.loads(a.parts) if a.parts else None
    k = bm.match_variant(cur, parts)
    vals = {**bm.variants[k]["switches"], **cur}
    prt = {**bm.blob.get("rest_parts", {}), **bm.variants[k]["parts"],
           **(parts or {})}
    with Live2DBackend(a.model_json, size=a.size) as be:
        be.calibrate()
        img = be.render(vals, parts=prt)
    im = Image.fromarray(img, "RGBA")
    bg = Image.new("RGBA", im.size, (236, 236, 240, 255))
    Image.alpha_composite(bg, im).convert("RGB").save(a.out)
    print(f"snapshot -> {a.out}")
    return 0


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["snapshot"]:
        return snapshot_main(argv[1:])
    if argv[:1] == ["summary"]:
        rows = []
        for root in argv[1:]:
            rows += summarize(root)
        print_summary(rows)
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--body", required=True, help="rigs/<rig>.body.json")
    ap.add_argument("--prompt", action="store_true",
                    help="print the body-mode system prompt")
    ap.add_argument("--images", action="store_true",
                    help="--prompt as for a backend that gets images")
    ap.add_argument("--persona", default=None)
    ap.add_argument("--reply", default=None,
                    help="a body-mode reply to parse, time and solve")
    ap.add_argument("--model-json", default=None,
                    help="check each solve on the Core's own geometry")
    a = ap.parse_args(argv)
    bm = fb.load(a.body)
    if a.prompt:
        print(system_prompt(bm, a.persona, a.images))
        print("\n--- user message suffix at rest:\n" + pose_now(bm))
    if a.reply:
        p = parser_for(bm)
        p.feed(a.reply)
        p.close()
        for w in p.warnings:
            print(f"pose: {w}")
        acts, _spans, end = dr.timeline(p.spoken, p.tags)
        core = None
        if a.model_json:
            try:
                from ..l2dprobe import core as l2dcore
            except ImportError:
                from l2dprobe import core as l2dcore
            core = l2dcore.load(a.model_json)
        poses = solve_poses(bm, acts, core=core)
        print(f"spoken ({end:.2f}s): {p.spoken}")
        for line in report_lines(poses, p.spoken):
            print("  " + line)
        for q in poses:
            print(f"    params {q['params']}" + (f" parts {q['parts']}"
                                                 if q["parts"] else ""))
        print("  reach: " + json.dumps(reach_summary(poses)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
