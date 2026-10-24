#!/usr/bin/env python3
"""
A body the model knows: canonical body state, Live2D geometry, and a solver.

Stage E1 of docs/embodiment.md. Three pieces:

  canonical  The body state a language model (or anything else) thinks in,
             shared by every rig: head yaw / pitch / roll, gaze, eyelids,
             brows, mouth, torso, and per arm the pose of the upper arm,
             forearm and hand plus the hand's position. Positions are in body
             coordinates: origin at the chin, x towards the character's own
             left (which is screen right for a rig facing the viewer, and +X
             in VRM / glTF, whose humanoid faces +Z), y up, unit one head
             height (hu). See CANON for every variable, its unit and source.

  forward    Per rig, measured from mesh vertices (l2dprobe.core reads them
             from the Cubism Core with no GL): drawables are mapped to body
             regions by name and checked against motion (assign_regions),
             every parameter is swept from the rest pose, and each region's
             centroid, rotation and scale is recorded (measure). The result
             is a piecewise-linear additive model with pairwise interaction
             terms for parameters that move the same region (BodyModel),
             saved as rigs/<rig>.body.json (format below).

  solver     Body targets (a partial dict of canonical variables) -> rig
             parameter values and part opacities: bounded least squares on
             the forward model with a prior toward the current pose, arm
             variants (pose3 part swaps, opacity switches) tried as discrete
             choices, whole-model transforms and physics outputs held
             (BodyModel.solve). The reverse, parameters -> canonical state,
             is BodyModel.perceive, for self-perception.

Measured versus declared. A 2D rig turns its head by deforming the face, not
by rotating anything, so head yaw and pitch in degrees cannot be measured from
vertices: the parallax is there (eyes and nose shift against the outline) but
has no depth to convert it with. Those variables are declared: Cubism's
standard ParamAngleX/Y span +-30 and mean degrees by convention (the editor's
default and every sample rig), ParamBodyAngleX/Y +-10 likewise. Head and torso
roll, region positions, hand positions and arm segment angles are measured.
Face shapes (eyes, brows, mouth, cheek) are declared from standard ids too: a
lid's openness is the parameter's meaning, and measuring it as mesh extent
would add noise, not information. CANON lists the source of each variable.

File format (rigs/<rig>.body.json, FORMAT "tsukuyomi.body/2"):

  rig, model_json, core_version
  frame       origin (model units, the chin at rest), unit (model units per
              hu), axes
  side_naming {drawables, params: "character" | "screen", votes}: how the
              rig's ids name sides, found from rest geometry (detect_naming).
              Every ".L"/".R" in this file is the character's own side
  landmarks   rest positions in body units: chin (0, 0), head centre,
              shoulder.L/R, shoulder.y, chest, hips, hand.L/R
  drawables   [{id, part, region, side, confidence, evidence, guess}] -- one
              per Core drawable; region "unclear" where names and motion do
              not say, "effect" / "hitarea" for effects and hit-test meshes
              (never body)
  params      {pid: {min, max, default, band, role, name, declares}}: role is
              solvable / held / physics / switch / variant / effect / inert
  effects     {params: {pid: {min, max, default, name, drawables}}, parts:
              {part: {rest, name, drawables}}, drawables}: effects (PartFx,
              PARAM_FXhand...), never variants or solved, kept for deliberate
              use
  declared    {variable: [pid, scale]}: value = parameter x scale (scale < 0
              where the rig's sign is the other way, see meaning)
  meaning     {variable: which way + goes, and the measured evidence}
  rest_parts  part opacities of the measured rest (pose3 first members, or
              the calm motion's first frame for parts switched directly)
  features    names "<region>.<x|y|roll|scale>" (hu, hu, deg CCW, percent
              log-scale) and declared variables ("head.yaw", "mouth.open"...)
  variants    [{name, parts, switches, rest, present, sweeps, pairs}]: one
              per discrete arm configuration. rest[f] is the feature at the
              rest pose, present[f] whether it exists in this variant (a
              hidden arm has no hand); sweeps {pid: {grid, delta: {feature:
              [delta at each grid value]}}} sparse, deltas from rest; pairs
              [[pid_i, pid_j, {feature: [r--, r-+, r+-, r++]}]] the
              interaction residual at the four corners (- = toward min, + =
              toward max from the default), applied bilinearly
  regions     {region: {drawables, vertices, confidence}} summary

Format 1 files (sides named as the ids name them, effects in the body) load
through upgrade_v1, which fixes what the stored data allows and lists the
rest under "caveats"; re-measure to clear them.

Measuring takes 3 s on Hiyori (2 variants) and 35 s on Natori (12); a solve
0.2-3 s; the files are 0.34 and 3.2 MB.

Usage:
    python -m flow.body --model-json M --rig rigs/Hiyori.npz --measure
    python -m flow.body --body rigs/Hiyori.body.json --describe
    python -m flow.body --body rigs/Hiyori.body.json \\
        --solve '{"head.yaw": 20}' [--render out.png --model-json M]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys

import numpy as np

FORMAT = "tsukuyomi.body/2"
# Older formats load() still reads, upgraded in memory (upgrade_v1).
FORMAT_V1 = "tsukuyomi.body/1"

# ------------------------------------------------------------ canonical

# Leaf regions a drawable can be assigned to. Sided ones carry ".L"/".R":
# the character's own side, decided from rest geometry (x of the drawable
# against the face's). Cubism's samples name sides the same way
# (ParamEyeLOpen closes the eye at +x, measured on Hiyori), but rigs need
# not: the Azur Lane rigs name some or all sides by screen side (lafei's
# arm_L1 and PARAM_Arm_LT_Angle are the arm at -x). detect_naming finds a
# rig's convention, per namespace (drawable / part ids, parameter ids), and
# the body file records it as side_naming. "effect" holds drawables that are
# effects (blurs, flashes, PartFx), "hitarea" hit-test meshes (TouchHead):
# neither is body, and neither is ever a variant or solved for.
SIDED = ("eye", "brow", "upper_arm", "forearm", "hand", "arm")
LEAVES = ("face", "eye", "brow", "mouth", "nose", "ear", "cheek", "neck",
          "torso", "hips", "legs", "upper_arm", "forearm", "hand", "arm",
          "hair", "accessory", "background", "effect", "hitarea")
NONBODY = ("effect", "hitarea")
# Composite regions: unions of leaves. "arm.L" is both the leaf for an arm the
# rig does not split and the union of the left arm's segments. The head's pose
# is taken from its rigid-ish parts only: with the eyes in the union, closing
# Hiyori's eyelids read as a 4 degree head roll (Procrustes on a mesh that
# deforms), and the features have regions of their own.
COMPOSITES = {
    "head": ("face", "nose", "ear"),
    "arm.L": ("upper_arm.L", "forearm.L", "hand.L", "arm.L"),
    "arm.R": ("upper_arm.R", "forearm.R", "hand.R", "arm.R"),
}
# Regions whose pose is a feature. Hair and accessories are measured for the
# description but are physics or decoration, never a target.
POSE_REGIONS = ("head", "face", "eye.L", "eye.R", "brow.L", "brow.R", "mouth",
                "neck", "torso", "hips", "legs", "arm.L", "arm.R",
                "upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R",
                "hand.L", "hand.R", "hair")
POSE_Q = ("x", "y", "roll", "scale")
# Regions whose pose describe() lists. Eyes, brows and the mouth deform rather
# than move, so their Procrustes "roll" is shape change (Hiyori's eyelid
# reads as 11-15 degrees); their declared variables describe them instead.
DESCRIBE_REGIONS = ("head", "neck", "torso", "hips", "arm.L", "arm.R",
                    "upper_arm.L", "upper_arm.R", "forearm.L", "forearm.R",
                    "hand.L", "hand.R")

# Declared variables: (name, unit, standard id, degrees at the end of a
# normalised range or None). A rig whose angle parameter spans no more than
# +-2 is taken as normalised and scaled to the nominal degrees.
DECLARED = (
    ("head.yaw", "deg", "ParamAngleX", 30.0),
    ("head.pitch", "deg", "ParamAngleY", 30.0),
    ("head.tilt", "deg", "ParamAngleZ", 30.0),
    ("torso.yaw", "deg", "ParamBodyAngleX", 10.0),
    ("torso.pitch", "deg", "ParamBodyAngleY", 10.0),
    ("torso.tilt", "deg", "ParamBodyAngleZ", 10.0),
    ("torso.forward", "unit", "ParamBodyPosition", None),
    ("gaze.x", "unit", "ParamEyeBallX", None),
    ("gaze.y", "unit", "ParamEyeBallY", None),
    ("eye.L.open", "unit", "ParamEyeLOpen", None),
    ("eye.R.open", "unit", "ParamEyeROpen", None),
    ("eye.L.smile", "unit", "ParamEyeLSmile", None),
    ("eye.R.smile", "unit", "ParamEyeRSmile", None),
    # Not brow.L.y / .x: those are the brow region's measured position.
    ("brow.L.lift", "unit", "ParamBrowLY", None),
    ("brow.R.lift", "unit", "ParamBrowRY", None),
    ("brow.L.slide", "unit", "ParamBrowLX", None),
    ("brow.R.slide", "unit", "ParamBrowRX", None),
    ("brow.L.angle", "unit", "ParamBrowLAngle", None),
    ("brow.R.angle", "unit", "ParamBrowRAngle", None),
    ("brow.L.form", "unit", "ParamBrowLForm", None),
    ("brow.R.form", "unit", "ParamBrowRForm", None),
    ("mouth.open", "unit", "ParamMouthOpenY", None),
    ("mouth.form", "unit", "ParamMouthForm", None),
    ("cheek.blush", "unit", "ParamCheek", None),
    ("shoulders.up", "unit", "ParamShoulder", None),
    ("shoulder.L.up", "unit", "ParamLeftShoulderUp", None),
    ("shoulder.R.up", "unit", "ParamRightShoulderUp", None),
)

# Which way + goes for a declared variable, and the measured proxy that
# checks the rig agrees: (proxy feature of the head, sign the proxy must take
# when the variable is +). Cubism's own conventions where they are fixed
# (ParamAngleY + looks up), screen conventions where they are not: Hiyori's
# ParamAngleZ +30 rolls her face 10 degrees clockwise, so head.tilt is the
# negated parameter and turns the same way as the measured head.roll. A rig
# whose proxy disagrees gets its scale negated at measure time.
SIGN = {
    "head.yaw": ("head.x", +1, "+ turns the face toward the character's "
                 "left (screen right)"),
    "head.pitch": ("head.y", +1, "+ looks up"),
    "head.tilt": ("head.roll", +1, "+ tilts the head counter-clockwise as "
                  "seen, like head.roll"),
    "torso.yaw": ("head.x", +1, "+ turns the upper body toward the "
                  "character's left"),
    "torso.pitch": ("head.y", +1, "+ leans back (head rises); - leans in"),
    "torso.tilt": ("head.roll", +1, "+ tilts the upper body counter-"
                   "clockwise as seen"),
    "torso.forward": ("head.scale", +1, "+ brings the body toward the "
                      "viewer"),
}

UNITS = {"deg": "degrees", "hu": "head heights", "pct": "percent",
         "unit": "the rig's own parameter units"}
# One sigma of target error per unit, for the solver's weights, and the
# tolerance a solved target is reported "ok" within.
SIGMA = {"deg": 1.0, "hu": 0.02, "pct": 1.0, "unit": 0.02}
TOL = {"deg": 2.0, "hu": 0.05, "pct": 3.0, "unit": 0.05}


def feature_unit(name: str) -> str:
    for n, unit, _sid, _deg in DECLARED:
        if n == name:
            return unit
    q = name.rsplit(".", 1)[-1]
    return {"x": "hu", "y": "hu", "roll": "deg", "scale": "pct"}.get(q, "unit")


def canon_spec() -> dict:
    """{variable: {unit, source, meaning}} -- the canonical body description,
    before any rig: what flow.body can say about a body at all. A rig has
    the subset its measured file lists as present."""
    out = {}
    for n, unit, sid, _deg in DECLARED:
        out[n] = {"unit": unit, "source": "declared",
                  "meaning": f"the rig's {sid} (standard id)"
                  + (" in degrees" if unit == "deg" else "")
                  + (f"; {SIGN[n][2]}" if n in SIGN else "")}
    for r in POSE_REGIONS:
        out[f"{r}.x"] = {"unit": "hu", "source": "measured",
                         "meaning": f"{r} centroid, + toward the "
                                    "character's left"}
        out[f"{r}.y"] = {"unit": "hu", "source": "measured",
                         "meaning": f"{r} centroid height above the chin"}
        out[f"{r}.roll"] = {"unit": "deg", "source": "measured",
                            "meaning": f"{r} rotation from rest, + counter-"
                                       "clockwise as seen"}
        out[f"{r}.scale"] = {"unit": "pct", "source": "measured",
                             "meaning": f"{r} size change from rest "
                                        "(100 log ratio)"}
    return out


# ----------------------------------------------------------- name hints

# (leaf, pattern) over lower-case ids, part ids and cdi3 display names; first
# match at the most specific level (drawable id, then its part, then the
# part's parents) wins. Order matters within a level: brow before eye
# (Natori's PartEyeBlow is its brows), hand before neck (手首 is the wrist),
# forearm before arm, hair before face. Pinyin from the Azur Lane rigs:
# shou hand (not shoulder), bi / gebo arm, tou head, yan eye, mei brow, zui
# mouth, toufa / fa hair, lian face, shen / shenti body, tui leg.
RULES = (
    ("background", r"background|背景|sketch|下絵|credit|クレジット"),
    ("brow", r"brow|blow|眉|(^|[^a-z])mei"),
    ("hand", r"hand|finger|wrist|glove|手|(^|[^a-z])shou(?!ld)|zhi"),
    ("forearm", r"forearm|fore|lowerarm|lower_arm|前腕|elbow|肘"),
    ("upper_arm", r"upperarm|upper_arm|上腕"),
    ("arm", r"arm|腕|(^|[^a-z])bi($|[^a-z])|gebo|shoubi"),
    ("eye", r"eye|目|瞳|iris|pupil|lash|まつ毛|(^|[^a-z])yan"),
    ("mouth", r"mouth|口|lip|teeth|tooth|歯|舌|tongue|(^|[^a-z])zui"),
    ("nose", r"nose|鼻|(^|[^a-z])bizi"),
    ("cheek", r"cheek|頬|blush|照れ|tere"),
    ("ear", r"(^|[^a-z])ear($|[^a-z])|耳|(^|[^a-z])erduo"),
    ("hair", r"hair|髪|ahoge|アホ毛|bang|ponytail|sideup|サイドアップ|ツインテール|"
             r"おさげ|twin|braid|toufa|"
             r"(^|[^a-z])fa($|[^a-z])"),
    ("neck", r"neck|首|bozi"),
    ("legs", r"leg|あし|足|脚|foot|feet|knee|(^|[^a-z])tui"),
    ("hips", r"waist|腰|hip|下半身|lowerbody|lower_body|skirt|スカート|pelvis"),
    ("torso", r"body|体|上半身|upperbody|chest|bust|胸|shirt|jacket|ジャケット|"
              r"cloth|服|shen|xiong|torso"),
    ("face", r"face|顔|輪郭|lian|肌"),
    ("head", r"head|頭|(^|[^a-z])tou($|[^a-z])"),
    ("accessory", r"glass|メガネ|眼鏡|watch|時計|ribbon|リボン|hat|帽|bag|"
                  r"weapon|武器|wing|tail|chain|necklace|earring|acc"),
)
_RULES = tuple((leaf, re.compile(p)) for leaf, p in RULES)
# Joint words in parameter ids and cdi3 names, for splitting an arm the rig
# draws as one part (Hiyori's PartArmA) into upper arm, forearm and hand.
JOINT_WRIST = re.compile(r"hand|wrist|finger|手|shou(?!ld)")
JOINT_ELBOW = re.compile(r"elbow|肘|fore|前腕")
SIDE_JA = re.compile(r"([左右])")
# A joint or rotation in a parameter's id or display name: continuous, even
# when its motions park it or the probe called it categorical (Natori's
# ParamArmAL02 左肩A 肘回転, the elbow, is in toggle_params' list, and the rig
# file bands ParamArmAL01 左腕A 肩, the shoulder, as discrete).
JOINT_NAME = re.compile(r"肩|肘|手首|回転|前後|角度|rotat|angle|angel|elbow|"
                        r"shoulder|wrist")
SWITCH_NAME = re.compile(r"表示|切替|切り替え|switch|toggle|show|visible|"
                         r"(^|[^a-z])on($|[^a-z])|(?<=[a-z])on$")


# Effects and hit areas, over lower-case ids, part ids and cdi3 names. FX as
# a word or a word's end (PARAM_FXhand, PARAM_beixin_FX, PARAM_arm_LFX,
# PartFx, D_FX_03, PARAM_handFX_move); pinyin texiao / xiaoguo.
EFFECT_NAME = re.compile(r"(^|[^a-z])fx|fx($|[^a-z])|effect|エフェクト|特效|"
                         r"效果|texiao|xiaoguo")
HITAREA_NAME = re.compile(r"(^|[^a-z])touch|hit_?area|当たり判定")


def nonbody_leaf(texts) -> tuple[str, str]:
    """("effect" | "hitarea", evidence) when any of a drawable's own texts
    (its id, its parts below those all drawables share, their names) marks
    it so: an effect anywhere above it is still an effect (lafei's
    D_BODY_70 sits under PartFx)."""
    for t in texts:
        low = (t or "").lower()
        if not low:
            continue
        if EFFECT_NAME.search(low):
            return "effect", f"effect<-{t}"
        if HITAREA_NAME.search(low):
            return "hitarea", f"hitarea<-{t}"
    return "", ""


def own_texts(texts) -> list[list[str]]:
    """Each drawable's texts without the parts every drawable shares (and
    their names): texts[d] = [id, part, name, parent, name, ...]."""
    common = set.intersection(*(set(t[1::2]) for t in texts)) if texts \
        else set()
    out = []
    for t in texts:
        o = [t[0]]
        for p, n in zip(t[1::2], t[2::2]):
            if p not in common:
                o += [p, n]
        out.append(o)
    return out


def effect_name(*texts) -> bool:
    """A parameter or part named as an effect."""
    return any(EFFECT_NAME.search((t or "").lower()) for t in texts)


def name_leaf(texts) -> tuple[str, str]:
    """(leaf, evidence) from the first text a rule matches, most specific
    first. "head" (a head part with no finer name) maps to face; nothing
    matching is "". Effects and hit areas are nonbody_leaf's, over the
    drawable's own texts (own_texts)."""
    for t in texts:
        low = (t or "").lower()
        for leaf, rx in _RULES:
            if rx.search(low):
                return ("face" if leaf == "head" else leaf), f"{leaf}<-{t}"
    return "", ""


def text_side(texts) -> str:
    """"L"/"R" from ids (flow.gestures.side_of: ParamArmLA, PartEyeL,
    PartArmAL) or 左 / 右 in display names, most specific first."""
    try:
        from .gestures import side_of
    except ImportError:
        from flow.gestures import side_of
    for t in texts:
        if not t:
            continue
        m = SIDE_JA.search(t)
        if m:
            return "L" if m.group(1) == "左" else "R"
        s = side_of(t.replace("Part", "Param", 1) if t.startswith("Part")
                    else t)
        if s:
            return s
    return ""


def param_meta(pid: str, name: str = "", group: str = "") -> dict:
    """Side and joint of a parameter from its id and cdi3 display name."""
    try:
        from ..l2dprobe.analyze import group_of
    except ImportError:
        from l2dprobe.analyze import group_of
    texts = [name, pid]
    low = (pid + " " + (name or "")).lower()
    g = group or group_of(pid)
    joint = ""
    if JOINT_WRIST.search(low):
        joint = "wrist"
    elif JOINT_ELBOW.search(low):
        joint = "elbow"
    arm = g == "arm" or bool(re.search(r"arm|hand|腕|手|shou|bi", low))
    return {"side": text_side(texts), "joint": joint if arm else "",
            "arm": arm, "group": g,
            "head": g == "head" or bool(re.search(r"^param_?angle_?[xyz]$",
                                                  pid.lower()))}


def flip_side(s: str) -> str:
    return {"L": "R", "R": "L"}.get(s, s)


def mirror_id(pid: str) -> str:
    """A standard id with its side swapped: ParamEyeLOpen -> ParamEyeROpen,
    ParamBrowLY -> ParamBrowRY, ParamLeftShoulderUp -> ParamRightShoulderUp."""
    sw = {"Left": "Right", "Right": "Left", "L": "R", "R": "L"}
    return re.sub(r"Left|Right|(?<=[a-z])[LR](?=[A-Z]|$)",
                  lambda m: sw[m.group(0)], pid)


def _sided_param(p: dict) -> bool:
    return bool(p["side"]) and (p["arm"] or p["group"] in ("eyelid", "brow"))


def detect_naming(texts, rest_xy, response, params, face_x, margin=0.0,
                  motion_floor=1e-6) -> dict:
    """The rig's side-naming convention, per namespace, by majority vote of
    name against rest geometry. "character" is Cubism's (L = the
    character's left = +x), "screen" the viewer's (L = -x).

    drawables  every sided-leaf drawable (eye, brow, arm...) whose id or
               part names a side and whose centroid is clear of the face's x
               by `margin`: its name against its x
    params     every sided arm / eyelid / brow parameter that moves sided
               drawables clear of the midline: its name against the side
               (by x) it moves, when that side carries > 3x the other's
               motion

    Returns {drawables, params, votes: {drawables: [agree, disagree],
    params: [...]}}; a namespace with no votes, or a tie, is "character".
    """
    n = len(texts)
    R = np.asarray(response, float).reshape(n, len(params))
    rest_xy = np.asarray(rest_xy, float).reshape(n, 2)
    own = own_texts(texts)
    dx = rest_xy[:, 0] - face_x
    clear = np.abs(dx) > margin
    sided = np.array([not nonbody_leaf(own[d])[0]
                      and name_leaf(texts[d])[0] in SIDED for d in range(n)],
                     bool)
    vd = [0, 0]
    for d in range(n):
        s = text_side(texts[d])
        if sided[d] and clear[d] and s:
            vd[0 if (dx[d] > 0) == (s == "L") else 1] += 1
    vp = [0, 0]
    peak = R.max(axis=0) if n else np.zeros(len(params))
    for j, p in enumerate(params):
        if not _sided_param(p) or peak[j] <= motion_floor:
            continue
        m = sided & clear
        left = float(R[m & (dx > 0), j].sum())
        right = float(R[m & (dx < 0), j].sum())
        if max(left, right) <= motion_floor or \
                max(left, right) <= 3 * min(left, right):
            continue
        geo = "L" if left > right else "R"
        vp[0 if geo == p["side"] else 1] += 1

    def conv(v):
        return "screen" if v[1] > v[0] else "character"
    return {"drawables": conv(vd), "params": conv(vp),
            "votes": {"drawables": vd, "params": vp}}


def assign_regions(texts, rest_xy, response, params, face_x=None,
                   motion_floor=1e-6, naming=None, margin=0.0):
    """Body region for each drawable.

    texts[d]     names for drawable d, most specific first (its id, then its
                 part chain's ids and cdi3 names)
    rest_xy[d]   its rest centroid (model units, +x = character's left)
    response     (n_drawables, n_params) mean vertex displacement at each
                 parameter's far value
    params       [param_meta(...)] in response's column order

    naming       detect_naming's result: the drawables' name sides are read
                 through it; params' sides must already be the character's
    margin       model units: a drawable this close to the face's x has no
                 side by position

    Returns [{region, leaf, side, confidence, evidence, guess}]. Names decide
    the leaf. The side is the character's: the name read through the rig's
    convention (which rest geometry decided, detect_naming), else the sided
    arm, eye or brow parameters that move it, else its rest x against the
    face's. Position does not override a name drawable by drawable: a limb
    can cross the midline (Natori's crossed hands). Motion splits an arm
    named only as an arm into upper arm, forearm and hand by which joint
    parameters move it (a wrist or hand parameter moving it at >= SPLIT of
    its peak makes it a hand). Motion
    also checks names: a face-family drawable no head parameter moves, or an
    arm one no arm parameter moves, is kept but marked low. A drawable no
    name places stays "unclear", with a motion guess when one family of
    parameters clearly dominates it.
    """
    SPLIT = 0.3
    n = len(texts)
    R = np.asarray(response, float).reshape(n, len(params))
    rest_xy = np.asarray(rest_xy, float).reshape(n, 2)
    if face_x is None:
        face_x = float(np.median(rest_xy[:, 0])) if n else 0.0
    heads = [j for j, p in enumerate(params) if p["head"]]
    arms = [j for j, p in enumerate(params) if p["arm"]]
    peak = R.max(axis=0) if n else np.zeros(len(params))
    # Each drawable's outermost part that is not shared by all drawables
    # (Natori: PartArmALFore's drawables and PartArmAL's are one arm).
    chains = [list(t[1::2]) for t in texts]
    common = set.intersection(*(set(c) for c in chains)) if chains else set()
    group = [next((p for p in reversed(c) if p not in common), "")
             for c in chains]
    own = own_texts(texts)
    screen = (naming or {}).get("drawables") == "screen"
    out = []
    for d in range(n):
        leaf, ev = nonbody_leaf(own[d])
        if leaf:
            out.append({"region": leaf, "leaf": leaf, "side": "",
                        "confidence": "high", "evidence": ev, "guess": ""})
            continue
        leaf, ev = name_leaf(texts[d])
        side_name = text_side(texts[d])
        if screen:
            side_name = flip_side(side_name)
        # Motion side: the sided arm / eye / brow parameters that move it.
        mv = {"L": 0.0, "R": 0.0}
        for j, p in enumerate(params):
            if p["side"] and (p["arm"] or p["group"] in ("eyelid", "brow")):
                if peak[j] > motion_floor:
                    mv[p["side"]] = max(mv[p["side"]], R[d, j] / peak[j])
        side_motion = ""
        if max(mv.values()) > 0.05:
            hi_s = max(mv, key=mv.get)
            lo_s = "R" if hi_s == "L" else "L"
            if mv[hi_s] > 3 * mv[lo_s]:
                side_motion = hi_s
        side_pos = "L" if rest_xy[d, 0] >= face_x else "R"
        pos_clear = abs(rest_xy[d, 0] - face_x) > margin
        conf = "high" if leaf else "unclear"
        guess = ""
        if leaf in ("arm", "forearm"):
            # Split by the joint parameters of the drawable's side that act
            # within its own part: Hiyori's ParamHandL turns the A arm's
            # hand and nothing in PartArmB, whose hand is not split this
            # way (it stays "arm" and gets the distal hand).
            s = side_name or side_motion or side_pos
            mates = [e for e in range(n) if group[e] == group[d]]

            def acts(j):
                return peak[j] > motion_floor and \
                    R[mates, j].max() >= SPLIT * peak[j]
            wr = [j for j, p in enumerate(params) if p["arm"]
                  and p["joint"] == "wrist" and p["side"] in (s, "")
                  and acts(j)]
            el = [j for j, p in enumerate(params) if p["arm"]
                  and p["joint"] == "elbow" and p["side"] in (s, "")
                  and acts(j)]
            if any(R[d, j] >= SPLIT * peak[j] for j in wr):
                ev += "; moved by wrist/hand param"
                leaf = "hand"
            elif leaf == "arm" and any(R[d, j] >= SPLIT * peak[j]
                                       for j in el):
                ev += "; moved by elbow param"
                leaf = "forearm"
            elif leaf == "arm" and (wr or el):
                ev += "; not moved by wrist/elbow params"
                leaf = "upper_arm"
        side = ""
        if leaf in SIDED:
            if side_name and side_motion and side_name != side_motion:
                conf = "low"
                ev += f"; name says {side_name}, motion {side_motion}"
            side = side_name or side_motion or side_pos
            ev += ("; side from name" if side_name else
                   "; side from motion" if side_motion else
                   "; side from position")
            if (side_name or side_motion) and pos_clear and \
                    side != side_pos:
                # Natori's crossed hands (PartHand21, 61) lie past the
                # midline: the name, read through the convention, wins.
                ev += f"; lies on the {side_pos} side"
            if not side_name and not side_motion and conf == "high":
                conf = "medium"
        # Motion check of the name.
        fam_head = leaf in ("face", "eye", "brow", "mouth", "nose", "ear",
                            "cheek")
        h = max((R[d, j] / peak[j] for j in heads if peak[j] > motion_floor),
                default=None)
        a = max((R[d, j] / peak[j] for j in arms if peak[j] > motion_floor),
                default=None)
        if fam_head and h is not None and h < 0.02:
            conf, ev = "low", ev + "; no head parameter moves it"
        if leaf in ("arm", "upper_arm", "forearm", "hand") and a is not None \
                and a < 0.02:
            conf, ev = "low", ev + "; no arm parameter moves it"
        region = (f"{leaf}.{side}" if leaf in SIDED else leaf) if leaf \
            else "unclear"
        if not leaf:
            if h is not None and h > 0.3 and (a is None or h > 2 * a):
                guess = "head"
            elif a is not None and a > 0.3 and (h is None or a > 2 * h):
                guess = "arm." + (side_pos if pos_clear else
                                  side_motion or side_pos)
        out.append({"region": region, "leaf": leaf, "side": side,
                    "confidence": conf, "evidence": ev, "guess": guess})
    return out


# ------------------------------------------------------------- geometry

def procrustes(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(rotation in degrees counter-clockwise, scale) taking point set a to
    b, same vertex order. Vertex correspondence is exact (the Core keeps
    indices), so this is the region's rotation, unlike a principal axis,
    which is undefined for a round region and flips for a square one."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    dot = float((a * b).sum())
    crs = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())
    na = float((a * a).sum())
    if na <= 0:
        return 0.0, 1.0
    return math.degrees(math.atan2(crs, dot)), math.sqrt(float((b * b).sum())
                                                         / na)


def region_features(V, idx, V0, origin, unit) -> list[float]:
    """[x, y, roll, scale] of the vertices idx of V against rest V0."""
    if idx is None or len(idx) < 3:
        return [math.nan] * 4
    p = V[idx]
    c = (p.mean(0) - origin) / unit
    rot, sc = procrustes(V0[idx], p)
    return [float(c[0]), float(c[1]), rot, 100.0 * math.log(max(sc, 1e-9))]


def distal(V0, idx, origin, frac=0.15):
    """The hand of an arm drawn as one piece: the `frac` of its vertices
    farthest from its root (the arm vertex nearest the chin), chosen at rest
    and tracked by index."""
    idx = np.asarray(idx)
    if len(idx) < 8:
        return None
    p = V0[idx]
    root = p[np.argmin(np.linalg.norm(p - origin, axis=1))]
    d = np.linalg.norm(p - root, axis=1)
    k = max(3, int(round(frac * len(idx))))
    return idx[np.argsort(d)[-k:]]


def region_members(leaf_of, off, vis, region):
    """Stacked vertex indices of a (leaf or composite) region's visible
    drawables, or None."""
    names = COMPOSITES.get(region, (region,))
    ds = [d for d in range(len(leaf_of)) if leaf_of[d] in names and vis[d]]
    if not ds:
        return None
    return np.concatenate([np.arange(off[d], off[d + 1]) for d in ds])


def region_index(leaf_of, off, vis, Vrest, origin) -> dict:
    """{pose region: vertex indices} for one variant, with the distal
    fallback for a hand the rig does not draw separately."""
    regidx = {r: region_members(leaf_of, off, vis, r) for r in POSE_REGIONS}
    for s in "LR":
        if regidx.get(f"hand.{s}") is None and \
                regidx.get(f"arm.{s}") is not None:
            regidx[f"hand.{s}"] = distal(Vrest, regidx[f"arm.{s}"], origin)
    return regidx


def feature_vector(V, Vrest, regidx, values, origin, unit, declared,
                   defaults) -> np.ndarray:
    """Features in BodyModel order: POSE_REGIONS x POSE_Q, then declared
    variables sorted by name (parameter value x scale)."""
    out = []
    for r in POSE_REGIONS:
        out += region_features(V, regidx.get(r), Vrest, origin, unit)
    for name in sorted(declared):
        p, s = declared[name]
        out.append(float(values.get(p, defaults[p])) * s)
    return np.array(out)


def exact_state(core, bm: "BodyModel", values: dict, variant: int) -> dict:
    """Canonical state measured on the Core itself at these values (not
    the forward model): the check that a solve did what the model says."""
    blob = bm.blob
    v = bm.variants[variant]
    origin = np.array(blob["frame"]["origin"])
    unit = float(blob["frame"]["unit"])
    leaf_of = [d["region"] for d in blob["drawables"]]
    off = core.offsets
    dflt = {p: e["default"] for p, e in blob["params"].items()}
    declared = {n: (p, s) for n, (p, s) in blob["declared"].items()}
    parts = {**blob.get("rest_parts", {}), **v["parts"]}
    core.set(dict(v["switches"]), parts)
    Vr, vis = core.all_vertices(), core.opacities() > 0.5
    regidx = region_index(leaf_of, off, vis, Vr, origin)
    vals = {**v["switches"], **values}
    core.set(vals, parts)
    F = feature_vector(core.all_vertices(), Vr, regidx, vals, origin, unit,
                       declared, dflt)
    return {f: float(F[k]) for k, f in enumerate(bm.features)
            if not math.isnan(F[k])}


# ----------------------------------------------------------------- rig io

def read_rig_info(model_json: str, rig_npz: str | None = None,
                  probe_dir: str = "out", hold: str = "") -> dict:
    """Everything measure needs from files: cdi3 names, pose groups, the
    runtime's bands (runtime.rig_columns), switches (motions.toggle_params),
    held whole-model transforms (runtime.held_params, when probe output
    exists) and the authored motions' part-opacity and parameter curves."""
    try:
        from ..l2dprobe.motions import (eval_segments, motion_files,
                                        physics_params, toggle_params,
                                        band_of)
        from ..l2dprobe.analyze import group_of
    except ImportError:
        from l2dprobe.motions import (eval_segments, motion_files,
                                      physics_params, toggle_params, band_of)
        from l2dprobe.analyze import group_of
    root = os.path.dirname(model_json)
    with open(model_json, encoding="utf-8") as fh:
        refs = json.load(fh).get("FileReferences", {})
    info = {"part_names": {}, "param_names": {}, "pose_groups": [],
            "bands": {}, "groups": {}, "toggles": {}, "held": set(),
            "motion_frames": []}
    cdi = refs.get("DisplayInfo")
    if cdi and os.path.exists(os.path.join(root, cdi)):
        with open(os.path.join(root, cdi), encoding="utf-8") as fh:
            blob = json.load(fh)
        info["part_names"] = {p["Id"]: p.get("Name", "")
                              for p in blob.get("Parts", [])}
        info["param_names"] = {p["Id"]: p.get("Name", "")
                               for p in blob.get("Parameters", [])}
    pose = refs.get("Pose")
    if pose and os.path.exists(os.path.join(root, pose)):
        with open(os.path.join(root, pose), encoding="utf-8") as fh:
            blob = json.load(fh)
        info["pose_groups"] = [[e["Id"] for e in g]
                               for g in blob.get("Groups", [])]
    phys, _ = physics_params(model_json)
    info["physics"] = phys
    if rig_npz and os.path.exists(rig_npz):
        try:
            from .runtime import rig_columns, held_params
        except ImportError:
            from flow.runtime import rig_columns, held_params
        col = rig_columns(rig_npz, model_json, log=lambda *a: None)
        info["bands"] = dict(zip(col["pids"], col["bands"]))
        info["groups"] = dict(zip(col["pids"], col["groups"]))
        info["held"] = held_params(rig_npz, probe_dir, col["pids"],
                                   col["groups"], col["expressive"],
                                   hold=hold, warn=False)
    else:
        info["held"] = set(p for p in hold.split(",") if p)
    info["band_of"] = lambda pid: info["bands"].get(pid) or band_of(
        pid, "continuous", info["groups"].get(pid) or group_of(pid), phys)
    info["toggles"] = toggle_params(model_json)
    # Authored frames (part opacities and parameters), for the arm
    # configurations the rig actually shows.
    fps = 10.0
    for path in motion_files(model_json):
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        dur = float(blob.get("Meta", {}).get("Duration", 0) or 0)
        if dur <= 0.05:
            continue
        times = np.arange(0.0, dur, 1.0 / fps, dtype=np.float32)
        frame = {}
        for curve in blob.get("Curves", []):
            tgt, cid = curve.get("Target"), curve.get("Id")
            if tgt not in ("PartOpacity", "Parameter") or not cid:
                continue
            try:
                frame[(tgt, cid)] = eval_segments(curve.get("Segments", []),
                                                  times)
            except Exception:  # noqa: BLE001
                continue
        frame[("Meta", "name")] = os.path.basename(path)
        info["motion_frames"].append(frame)
    return info


# ---------------------------------------------------------------- measure

K_GRID = 9            # sweep points per parameter (plus the default)
MOVE_EPS = 1e-3       # hu: a region feature change below this is no motion
ROLL_EPS = 0.05       # deg
PAIR_MIN = 0.02       # hu or deg/10 of region motion for a pair to be fitted
MAX_VARIANTS = 24
MAX_PAIRS = 400


def _far(lo, hi, d):
    return hi if abs(hi - d) >= abs(d - lo) else lo


def whole_model(V0, V1, eps=1e-9) -> bool:
    """A parameter that moves the whole character as one rigid piece.

    l2dprobe.transforms finds these from pixels (reach x cover); with exact
    vertices the test is direct: it moves >= 85 % of all vertices noticeably
    (10 % of its peak), and one similarity transform (shift, rotation,
    scale) explains the motion to within 15 % of its RMS. Natori's
    ParamAllX/Y/Rotate are rigid (residual ~0); a body angle or a lean
    bends around the feet and leaves most of its motion unexplained
    (ParamBodyPosition, 体の前後, moves 89 % of vertices but is no transform).
    Angles are protected by name anyway (transforms.protected): on bust rigs
    a body angle can look rigid.
    """
    d = np.linalg.norm(V1 - V0, axis=1)
    if d.max() <= eps:
        return False
    if float((d > 0.1 * d.max()).mean()) < 0.85:
        return False
    m0, m1 = V0.mean(0), V1.mean(0)
    rot, sc = procrustes(V0, V1)
    t = math.radians(rot)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    pred = (V0 - m0) @ R.T * sc + m1
    resid = float(np.sqrt(((pred - V1) ** 2).sum(1).mean()))
    rms = float(np.sqrt((d ** 2).mean()))
    return resid <= 0.15 * rms


def measure(core, info: dict, log=print) -> dict:
    """The body file for one rig, from a Core model (l2dprobe.core.CoreModel
    or anything with its interface) and read_rig_info's dict."""
    try:
        from ..l2dprobe.transforms import protected
        from .runtime import snappable
    except ImportError:
        from l2dprobe.transforms import protected
        from flow.runtime import snappable
    pids = list(core.param_ids)
    lo, hi = core.param_min.astype(float), core.param_max.astype(float)
    dflt = core.param_default.astype(float)
    n_d = len(core.drawable_ids)
    off = core.offsets
    groups = info.get("pose_groups", [])
    rest_parts = {g[0]: 1.0 for g in groups}
    rest_parts.update({p: 0.0 for g in groups for p in g[1:]})
    # Parts the motions switch directly, with no pose3 (Azur Lane rigs: their
    # Unity runtime applies PartOpacity curves as they are, and the Core
    # leaves every part at opacity 1, so all arm variants draw at once).
    # Their rest is what the calm motion shows at its start.
    frames = info.get("motion_frames", [])
    pose_parts = {p for g in groups for p in g}
    curve_parts = sorted({cid for f in frames for (tgt, cid) in f
                          if tgt == "PartOpacity" and cid not in pose_parts})
    if curve_parts:
        calm = [f for f in frames if re.search(
            r"idle|home|main", str(f.get(("Meta", "name"), "")).lower())]
        for part in curve_parts:
            src = next((f for f in calm + frames
                        if ("PartOpacity", part) in f), None)
            rest_parts[part] = 1.0 if src is None else float(
                src[("PartOpacity", part)][0] > 0.5)

    def geom(values=None, parts=None):
        core.set(values or {}, {**rest_parts, **(parts or {})})
        return core.all_vertices(), core.opacities()

    # Responses are read with every pose-group member shown: a hidden part's
    # drawables report opacity 0 whatever a switch does, and Natori's hand
    # switches (ParamArmBRHand01 右手01 表示) live under the hidden PartArmBR.
    all_on = {p: 1.0 for g in groups for p in g}
    all_on.update({p: 1.0 for p in curve_parts})
    V0, _ = geom()
    _, O0 = geom(parts=all_on)
    cent = np.array([V0[off[d]:off[d + 1]].mean(0) for d in range(n_d)])

    # Responses: mean displacement per drawable at each end of each param.
    live = [i for i in range(len(pids)) if hi[i] > lo[i]]
    resp = np.zeros((n_d, len(live)))
    opac_switch = {}
    flips_all = {}      # every param that flips a drawable, switch or not
    held_geo = set()
    for c, i in enumerate(live):
        for v in (lo[i], hi[i]):
            if v == dflt[i]:
                continue
            V1, O1 = geom({pids[i]: v}, all_on)
            disp = np.linalg.norm(V1 - V0, axis=1)
            per = np.add.reduceat(disp, off[:-1]) / np.maximum(
                core.vertex_counts, 1)
            resp[:, c] = np.maximum(resp[:, c], per)
            flip = np.nonzero(np.abs(O1 - O0) > 0.5)[0]
            # Opacity that changes with a parameter is a switch only when
            # the rig uses it as one (its motions jump between two states,
            # or its name says show / switch) and it is not a joint: a rig
            # may swap meshes as a limb turns, and that limb is still
            # continuous (runtime.snappable, JOINT_NAME). Natori's flips
            # are all real switches (hand and teeth 表示); a rig whose joints
            # flip opacity shows them in describe()'s opacity-flip line.
            nm = (pids[i] + " " + info.get("param_names", {}).get(
                pids[i], "")).lower()
            is_switch = snappable(pids[i], info.get("groups", {}).get(
                pids[i], "")) and not JOINT_NAME.search(nm) and (
                pids[i] in info.get("toggles", {})
                or bool(SWITCH_NAME.search(nm)))
            if len(flip):
                flips_all.setdefault(pids[i], set()).update(flip.tolist())
            if len(flip) and is_switch:
                opac_switch.setdefault(pids[i], set()).update(flip.tolist())
            if whole_model(V0, V1) and not protected(
                    pids[i], info.get("groups", {}).get(pids[i], "")):
                held_geo.add(pids[i])

    # Regions.
    part_names = info.get("part_names", {})
    texts = []
    for d in range(n_d):
        chain = core.part_chain(int(core.drawable_parent[d]))
        t = [core.drawable_ids[d]]
        for p in chain:
            t += [p, part_names.get(p, "")]
        texts.append(t)
    pmeta = [param_meta(pids[i], info.get("param_names", {}).get(pids[i], ""),
                        info.get("groups", {}).get(pids[i], ""))
             for i in live]
    # Face centre for sides: the drawables names place on the face.
    own = own_texts(texts)
    face0 = [d for d in range(n_d) if not nonbody_leaf(own[d])[0]
             and name_leaf(texts[d])[0] in ("face", "eye", "brow", "mouth",
                                            "nose")]
    face_x = float(np.median(cent[face0, 0])) if face0 else \
        float(np.median(cent[:, 0]))
    # A drawable within 2 % of the drawn width of the face's x has no side
    # by position.
    margin = 0.02 * float(np.ptp(cent[:, 0])) if n_d else 0.0
    naming = detect_naming(texts, cent, resp, pmeta, face_x, margin)
    if naming["params"] == "screen":
        for p in pmeta:
            p["side"] = flip_side(p["side"])
    log(f"body: side naming drawables {naming['drawables']} "
        f"{naming['votes']['drawables']}, params {naming['params']} "
        f"{naming['votes']['params']} (agree, disagree with geometry)")
    regs = assign_regions(texts, cent, resp, pmeta, face_x, naming=naming,
                          margin=margin)
    leaf_of = [r["region"] for r in regs]

    # Effects: drawables named so (or under a part named so), parameters
    # named so or moving / flipping only effect drawables, parts named so
    # or holding only effect drawables. They are never variants, never
    # solved; the file lists them under "effects" for deliberate use.
    nonbody = np.array([r["region"] in NONBODY for r in regs], bool)
    eff_draw = np.array([r["region"] == "effect" for r in regs], bool)
    eff_params = {}
    for c, i in enumerate(live):
        p = pids[i]
        col = resp[:, c]
        touched = set(np.nonzero(col > 0.05 * col.max())[0].tolist()) \
            if col.max() > 1e-6 else set()
        touched |= flips_all.get(p, set())
        by_name = effect_name(p, info.get("param_names", {}).get(p, ""))
        only_fx = bool(touched) and all(nonbody[d] for d in touched) and \
            any(eff_draw[d] for d in touched)
        if by_name or only_fx:
            eff_params[p] = sorted(touched)
    all_parts = pose_parts | set(curve_parts)
    eff_parts = {}
    for part in sorted(all_parts):
        ds = [d for d in range(n_d) if part in texts[d][1::2]]
        if effect_name(part, part_names.get(part, "")) or (
                ds and all(eff_draw[d] for d in ds)):
            eff_parts[part] = ds

    def members(region, vis):
        return region_members(leaf_of, off, vis, region)

    # Body frame from the rest pose: chin = lowest face vertex, at the face
    # centroid's x; head height = vertical extent of the head regions.
    vis0 = O0 > 0.5
    head_idx = members("head", vis0)
    face_idx = members("face", vis0)
    if head_idx is None:
        raise RuntimeError("no drawable maps to the face or head: cannot "
                           "set the body frame (see the drawables listing "
                           "with --regions)")
    fi = face_idx if face_idx is not None else head_idx
    origin = np.array([V0[fi, 0].mean(), V0[fi, 1].min()])
    unit = float(V0[head_idx, 1].max() - V0[head_idx, 1].min())
    if unit <= 0:
        raise RuntimeError("head region has no height")

    # Parameter roles.
    toggles = info.get("toggles", {})
    held = set(info.get("held", set())) | held_geo
    band_of = info.get("band_of", lambda pid: "expressive")
    declared = {}
    by_canon = {re.sub(r"[_\s]", "", p.lower()): p for p in pids}
    sided_decl = {}
    for name, unit_, sid, deg in DECLARED:
        if re.search(r"\.[LR]\.", name) and naming["params"] == "screen":
            # The character's left eye is the rig's ParamEyeROpen.
            sided_decl[name] = sid = mirror_id(sid)
        p = by_canon.get(sid.lower())
        if p is None:
            continue
        i = pids.index(p)
        scale = 1.0
        if unit_ == "deg" and max(abs(lo[i]), abs(hi[i])) <= 2.0:
            scale = float(deg)
        declared[name] = (p, scale)
    # Signs from the measured proxy (SIGN): the head's shift, roll or size
    # between the parameter's two ends.
    meaning = {}
    hidx = members("head", vis0)
    for name, (p, scale) in list(declared.items()):
        if name not in SIGN:
            continue
        proxy, want, text = SIGN[name]
        i = pids.index(p)
        Va, _ = geom({p: lo[i]})
        Vb, _ = geom({p: hi[i]})
        q = proxy.rsplit(".", 1)[1]
        fa = region_features(Va, hidx, V0, origin, unit)
        fb = region_features(Vb, hidx, V0, origin, unit)
        k = POSE_Q.index(q)
        d = fb[k] - fa[k]
        floor = 0.05 if q in ("roll", "scale") else 1e-3
        if abs(d) < floor:
            meaning[name] = f"{text} (sign unverified: {proxy} moves "\
                            f"{d:+.4f} over {p}'s range)"
            continue
        if math.copysign(1, d) != want:
            scale = -scale
            declared[name] = (p, scale)
        meaning[name] = f"{text} ({'-' if scale < 0 else ''}{p}; {proxy} "\
                        f"{d:+.3f} over its range)"
    for name in sided_decl:
        if name in declared:
            side = "left" if ".L." in name else "right"
            meaning[name] = (f"the character's {side} side: "
                             f"{declared[name][0]} (the rig names "
                             "parameters by screen side)")
    arm_regions = {"arm.L", "arm.R", "upper_arm.L", "upper_arm.R",
                   "forearm.L", "forearm.R", "hand.L", "hand.R"}
    variant_switch = {p for p, ds in opac_switch.items()
                      if p not in eff_params
                      and any(leaf_of[d] in arm_regions for d in ds)}
    declared_pids = {p for p, _s in declared.values()}
    roles = {}
    for i, p in enumerate(pids):
        band = band_of(p)
        if hi[i] <= lo[i]:
            role = "inert"
        elif p in held:
            role = "held"
        elif p in eff_params:
            role = "effect"
        elif band == "physics" or p in info.get("physics", set()):
            role = "physics"
        elif p in variant_switch:
            role = "variant"
        elif p in opac_switch and p not in declared_pids:
            # An opacity switch with no standard meaning (a prop shown or
            # hidden). Declared ones stay solvable: ParamCheek fades a blush
            # in and ParamMouthForm swaps mouth meshes, and both are meant
            # to be set continuously.
            role = "switch"
        elif band in ("expressive", "autonomic", "mouth"):
            role = "solvable"
        elif band == "discrete" and JOINT_NAME.search(
                (p + " " + info.get("param_names", {}).get(p, "")).lower()):
            # The probe labelled it categorical, so the runtime holds it,
            # but it is a joint (Natori's ParamArmAL01 左腕A 肩): the body
            # can use it; E2's player has to let it through.
            role = "solvable"
        else:
            role = "switch" if band == "discrete" else "inert"
        roles[p] = {"min": float(lo[i]), "max": float(hi[i]),
                    "default": float(dflt[i]), "band": band, "role": role,
                    "name": info.get("param_names", {}).get(p, ""),
                    "toggle": p in toggles}
    for name, (p, s) in declared.items():
        roles[p]["declares"] = name

    # Variants: the arm configurations the authored motions show, per side,
    # crossed. A dimension is a pose-group part, a part the motions switch
    # directly (no pose3), or an opacity switch, that touches an arm region.
    side_of_dim = {}
    for g in groups:
        for part in g:
            if part in eff_parts:
                continue
            ds = [d for d in range(n_d) if part in texts[d][1::2]]
            sides = {regs[d]["side"] for d in ds if leaf_of[d] in arm_regions}
            if sides:
                side_of_dim[("PartOpacity", part)] = (
                    sides.pop() if len(sides) == 1 else "")
    # One side per pose group: Hiyori's PartArmA holds both arms, and a
    # group whose members fell on different sides would be enumerated per
    # side with two members shown at once.
    for g in groups:
        dg = [("PartOpacity", part) for part in g
              if ("PartOpacity", part) in side_of_dim]
        if len({side_of_dim[d] for d in dg}) > 1:
            for d in dg:
                side_of_dim[d] = ""
    for part in curve_parts:
        if part in eff_parts:
            continue
        sides = {regs[d]["side"] for d in range(n_d)
                 if part in texts[d][1::2] and leaf_of[d] in arm_regions}
        if sides:
            side_of_dim[("PartOpacity", part)] = (
                sides.pop() if len(sides) == 1 else "")
    # As with pose groups: a switched part holding both arms ties the sides,
    # so all directly switched parts are then judged together.
    cdims = [("PartOpacity", p) for p in curve_parts
             if ("PartOpacity", p) in side_of_dim]
    if any(side_of_dim[d] == "" for d in cdims):
        for d in cdims:
            side_of_dim[d] = ""
    for p in variant_switch:
        sides = {regs[d]["side"] for d in opac_switch[p]
                 if leaf_of[d] in arm_regions}
        side_of_dim[("Parameter", p)] = sides.pop() if len(sides) == 1 else ""
    switch_parts = {p: {q for d in opac_switch[p] for q in texts[d][1::2]
                        if q in pose_parts or q in curve_parts}
                    for p in variant_switch}
    variants = _variants(side_of_dim, frames, groups,
                         {p: (roles[p]["min"], roles[p]["max"],
                              roles[p]["default"]) for p in variant_switch},
                         switch_parts,
                         {p: rest_parts[p] for p in curve_parts})
    log(f"body: {n_d} drawables, {len(live)} live params, "
        f"{len(variants)} arm variant(s), held {sorted(held) or '-'}")

    feats = [f"{r}.{q}" for r in POSE_REGIONS for q in POSE_Q] + \
        sorted(declared)
    fidx = {f: k for k, f in enumerate(feats)}

    dflt_of = dict(zip(pids, dflt.tolist()))

    def features(V, Vrest, regidx, values):
        return feature_vector(V, Vrest, regidx, values, origin, unit,
                              declared, dflt_of)

    solve_pids = [p for p in pids if roles[p]["role"] in
                  ("solvable", "physics")]
    out_variants = []
    for var in variants:
        Vr, Or = geom(var["switches"], var["parts"])
        vis = Or > 0.5
        regidx = region_index(leaf_of, off, vis, Vr, origin)
        base = dict(var["switches"])
        F0 = features(Vr, Vr, regidx, base)
        present = ~np.isnan(F0)
        visverts = np.concatenate([np.arange(off[d], off[d + 1])
                                   for d in range(n_d) if vis[d]])
        sweeps = {}
        moves = {}
        for p in solve_pids:
            i = pids.index(p)
            grid = np.unique(np.concatenate([np.linspace(lo[i], hi[i],
                                                         K_GRID), [dflt[i]]]))
            D = np.zeros((len(grid), len(feats)))
            any_move = False
            for k, v in enumerate(grid):
                if v == dflt[i]:
                    continue
                V1, _ = geom({**base, p: v}, var["parts"])
                if np.abs(V1[visverts] - Vr[visverts]).max() > 1e-7:
                    any_move = True
                D[k] = features(V1, Vr, regidx, {**base, p: v}) - F0
            D[:, ~present] = 0.0
            D = np.nan_to_num(D)
            if not any_move and not any(p == q for q, _ in declared.values()):
                continue
            sweeps[p] = (grid, D)
            reach = np.abs(D).max(0)
            moves[p] = {r for r in POSE_REGIONS
                        if reach[fidx[f"{r}.x"]] > PAIR_MIN
                        or reach[fidx[f"{r}.y"]] > PAIR_MIN
                        or reach[fidx[f"{r}.roll"]] > 10 * PAIR_MIN}
        # Pairwise interactions between solvable params moving one region.
        pairs = []
        body_regions = ("head", "torso", "arm.L", "arm.R", "hand.L", "hand.R",
                        "forearm.L", "forearm.R", "upper_arm.L",
                        "upper_arm.R")
        cand = [p for p in sweeps if roles[p]["role"] == "solvable"]
        for a, b in itertools.combinations(cand, 2):
            shared = moves.get(a, set()) & moves.get(b, set()) & \
                set(body_regions)
            if not shared or len(pairs) >= MAX_PAIRS:
                continue
            ia, ib = pids.index(a), pids.index(b)
            Rq = np.zeros((4, len(feats)))
            ga, Da = sweeps[a]
            gb, Db = sweeps[b]
            for q, (va, vb) in enumerate(itertools.product(
                    (lo[ia], hi[ia]), (lo[ib], hi[ib]))):
                if va == dflt[ia] or vb == dflt[ib]:
                    continue
                V1, _ = geom({**base, a: va, b: vb}, var["parts"])
                f = features(V1, Vr, regidx, {**base, a: va, b: vb}) - F0
                Rq[q] = np.nan_to_num(
                    f - Da[np.searchsorted(ga, va)] - Db[np.searchsorted(gb,
                                                                          vb)])
            Rq[:, ~present] = 0.0
            if np.abs(Rq).max() > MOVE_EPS:
                pairs.append((a, b, Rq))
        out_variants.append({
            "name": var["name"], "parts": var["parts"],
            "switches": var["switches"], "weight": var["weight"],
            "rest": F0, "present": present, "sweeps": sweeps,
            "pairs": pairs,
            "front": _front(core, regs, vis, leaf_of)})
        log(f"  variant {var['name']}: {len(sweeps)} params move it, "
            f"{len(pairs)} interaction pairs")

    # Landmarks at the first variant's rest.
    V0v, O0v = geom(variants[0]["switches"], variants[0]["parts"])
    vis = O0v > 0.5

    def lm(idx):
        return None if idx is None else [round(float(x), 4) for x in
                                         ((V0v[idx].mean(0) - origin) / unit)]
    marks = {"chin": [0.0, 0.0], "head": lm(members("head", vis))}
    for s in "LR":
        a = members(f"arm.{s}", vis)
        if a is not None:
            top = V0v[a, 1] >= np.quantile(V0v[a, 1], 0.9)
            marks[f"shoulder.{s}"] = lm(a[top])
    sh = [marks[k][1] for k in ("shoulder.L", "shoulder.R") if k in marks]
    if sh:
        marks["shoulder.y"] = round(float(np.mean(sh)), 4)
    t = members("torso", vis)
    if t is not None:
        yy = (V0v[t, 1] - origin[1]) / unit
        top = marks.get("shoulder.y", float(yy.max()))
        chest = t[(yy <= top) & (yy >= top - 1.0)]
        marks["chest"] = lm(chest if len(chest) >= 3 else t)
    h = members("hips", vis)
    if h is not None:
        marks["hips"] = lm(h)
    F00 = out_variants[0]["rest"]
    for s in "LR":
        x, y = F00[fidx[f"hand.{s}.x"]], F00[fidx[f"hand.{s}.y"]]
        if not math.isnan(x):
            marks[f"hand.{s}"] = [round(float(x), 4), round(float(y), 4)]

    regions = {}
    for d, r in enumerate(regs):
        e = regions.setdefault(r["region"], {"drawables": [], "vertices": 0,
                                             "confidence": {}})
        e["drawables"].append(core.drawable_ids[d])
        e["vertices"] += int(core.vertex_counts[d])
        e["confidence"][r["confidence"]] = e["confidence"].get(
            r["confidence"], 0) + 1
    return {
        "format": FORMAT,
        "frame": {"origin": [float(origin[0]), float(origin[1])],
                  "unit": unit, "unit_name": "hu (head height)",
                  "axes": "x toward the character's left, y up"},
        "landmarks": marks,
        "drawables": [{"id": core.drawable_ids[d],
                       "part": texts[d][1] if len(texts[d]) > 1 else "",
                       **{k: r[k] for k in ("region", "side", "confidence",
                                            "evidence", "guess")}}
                      for d, r in enumerate(regs)],
        "regions": regions,
        "params": roles,
        "rest_parts": {p: float(v) for p, v in rest_parts.items()},
        # Params that flip an arm drawable's opacity but are solved as
        # continuous (a joint swapping meshes as it turns, like Natori's
        # elbows -- or a missed arm swap, which the solver could park half
        # way and draw two arms with). describe() lists them.
        "opacity_flips": {p: sorted({leaf_of[d] for d in ds})
                          for p, ds in flips_all.items()
                          if roles[p]["role"] == "solvable"
                          and any(leaf_of[d] in arm_regions for d in ds)},
        "declared": {n: [p, s] for n, (p, s) in declared.items()},
        "meaning": meaning,
        "side_naming": naming,
        "effects": {
            "params": {p: {"min": roles[p]["min"], "max": roles[p]["max"],
                           "default": roles[p]["default"],
                           "name": roles[p]["name"],
                           "drawables": [core.drawable_ids[d] for d in ds]}
                       for p, ds in eff_params.items()},
            "parts": {q: {"rest": float(rest_parts.get(q, 1.0)),
                          "name": part_names.get(q, ""),
                          "drawables": [core.drawable_ids[d] for d in ds]}
                      for q, ds in eff_parts.items()},
            "drawables": [core.drawable_ids[d] for d in range(n_d)
                          if eff_draw[d]]},
        "features": feats,
        "variants": out_variants,
    }


def _front(core, regs, vis, leaf_of):
    """{hand.S: True/False} whether the hand draws over the torso (render
    order), the only depth a 2D rig has: "in front of the chest"."""
    orders = core.render_orders()
    if orders is None:
        return {}
    out = {}
    torso = [orders[d] for d in range(len(regs)) if leaf_of[d] == "torso"
             and vis[d]]
    if not torso:
        return {}
    for s in "LR":
        h = [orders[d] for d in range(len(regs))
             if leaf_of[d] == f"hand.{s}" and vis[d]]
        if h:
            out[f"hand.{s}"] = bool(np.median(h) > np.median(torso))
    return out


def _variants(side_of_dim, frames, groups, switch_ranges, switch_parts=None,
              part_rest=None):
    """Discrete arm configurations: [{name, parts, switches, weight}].

    A dimension is a pose-group part, a part the motions switch directly
    (part_rest {part: rest opacity}), or an opacity switch on an arm. The
    configurations are those the authored motions show (which part of each
    pose group, where they park each switch), per side and crossed, so a
    combination is offered when each side's half was authored; a motion that
    does not touch a dimension leaves it at rest. With no authored evidence
    for a side, each pose group's members and each switch's two ends are
    enumerated. A switch whose drawables sit under a pose part that is off
    is moot and kept at rest (switch_parts {pid: pose parts above its
    drawables}). The rest configuration (each pose group's first member,
    switches at default) comes first, then by how often motions show it.
    """
    dims = sorted(side_of_dim)
    if not dims:
        return [{"name": "rest", "parts": {}, "switches": {}, "weight": 1.0}]
    switch_parts = switch_parts or {}
    part_rest = part_rest or {}
    group_of_part = {p: gi for gi, g in enumerate(groups) for p in g}

    def rest_value(d):
        tgt, cid = d
        if tgt == "PartOpacity":
            gi = group_of_part.get(cid)
            if gi is None:
                return part_rest.get(cid, 1.0)
            return 1.0 if groups[gi][0] == cid else 0.0
        return switch_ranges[cid][2]

    def snap(d, x):
        tgt, cid = d
        if tgt == "PartOpacity":
            return 1.0 if x > 0.5 else 0.0
        lo, hi, _d = switch_ranges[cid]
        return hi if abs(x - hi) < abs(x - lo) else lo

    def canonical(sd, key):
        """Moot switches (their pose part is off) back to rest."""
        on = {d[1] for d, v in zip(sd, key) if d[0] == "PartOpacity"
              and v > 0.5}
        known = {d[1] for d in sd if d[0] == "PartOpacity"}
        out = []
        for d, v in zip(sd, key):
            if d[0] == "Parameter":
                ps = switch_parts.get(d[1], set()) & known
                if ps and not ps & on:
                    v = rest_value(d)
            out.append(v)
        return tuple(out)

    def valid(sd, key):
        # Parts outside any pose group: some arm part must show (a frame
        # with none is a fade, or a motion hiding the arm for an effect).
        free = [v for d, v in zip(sd, key) if d[0] == "PartOpacity"
                and d[1] not in group_of_part]
        if free and not any(v > 0.5 for v in free):
            return False
        for g in groups:
            mem = [v for d, v in zip(sd, key) if d[0] == "PartOpacity"
                   and d[1] in g]
            if mem and sum(v > 0.5 for v in mem) != 1:
                return False
        return True

    per_side = {}
    for s in sorted(set(side_of_dim.values())):
        sd = [d for d in dims if side_of_dim[d] == s]
        seen = {}
        for frame in frames:
            if not any(d in frame for d in sd):
                continue
            n = max((len(frame[d]) for d in sd if d in frame), default=0)
            for t in range(n):
                key = tuple(snap(d, float(frame[d][min(t, len(frame[d]) - 1)]))
                            if d in frame else rest_value(d) for d in sd)
                key = canonical(sd, key)
                # A frame mid-fade shows no member or two; skip it.
                if valid(sd, key):
                    seen[key] = seen.get(key, 0) + 1
        if not seen:
            opts = []
            for d in sd:
                if d[0] == "Parameter":
                    lo, hi, _ = switch_ranges[d[1]]
                    opts.append((lo, hi))
                else:
                    opts.append((0.0, 1.0))
            for combo in itertools.product(*opts):
                if valid(sd, combo):
                    seen[canonical(sd, combo)] = 1
        rk = canonical(sd, tuple(rest_value(d) for d in sd))
        seen.setdefault(rk, 0)
        total = sum(seen.values()) or 1
        keys = sorted(seen, key=lambda k: (k != rk, -seen[k]))
        per_side[s] = [(sd, k, seen[k] / total) for k in keys]
    out = []
    for combo in itertools.product(*per_side.values()):
        parts, switches, names, w = {}, {}, [], 1.0
        for sd, key, frac in combo:
            w *= max(frac, 1e-3)
            label = []
            for d, v in zip(sd, key):
                if d[0] == "PartOpacity":
                    parts[d[1]] = v
                    if v > 0.5:
                        label.append(d[1])
                else:
                    switches[d[1]] = v
                    if v != switch_ranges[d[1]][2]:
                        label.append(f"{d[1]}={v:g}")
            names.append("+".join(label) or "rest")
        out.append({"name": " | ".join(names), "parts": parts,
                    "switches": switches, "weight": w})
    first = out[0]
    rest = [first] + sorted(out[1:], key=lambda v: -v["weight"])
    return rest[:MAX_VARIANTS]


# ---------------------------------------------------------------- save/load

def _r(x, nd=5):
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{nd}g}")


# Deltas below this are not stored: 1e-4 hu is 0.02 px on a 512 px render of
# Hiyori, 1e-4 deg nothing. At 1e-6 Natori's file was 4.9 MB.
SPARSE_EPS = 1e-4


def save(body: dict, path: str) -> None:
    """Write body (measure's dict, or BodyModel.to_dict) as JSON: sweeps
    and pairs sparse, keyed by feature name."""
    feats = body["features"]
    out = {k: v for k, v in body.items() if k != "variants"}
    out["variants"] = []
    for v in body["variants"]:
        sw = {}
        for p, (grid, D) in v["sweeps"].items():
            nz = np.nonzero(np.abs(D).max(0) > SPARSE_EPS)[0]
            sw[p] = {"grid": [_r(g) for g in grid],
                     "delta": {feats[k]: [_r(x, 4) for x in D[:, k]]
                               for k in nz}}
        pairs = []
        for a, b, Rq in v["pairs"]:
            nz = np.nonzero(np.abs(Rq).max(0) > SPARSE_EPS)[0]
            pairs.append([a, b, {feats[k]: [_r(x, 4) for x in Rq[:, k]]
                                 for k in nz}])
        out["variants"].append({
            "name": v["name"], "parts": v["parts"],
            "switches": v["switches"], "weight": _r(v.get("weight", 1.0)),
            "front": v.get("front", {}),
            "rest": [_r(x) for x in v["rest"]],
            "present": [bool(x) for x in v["present"]],
            "sweeps": sw, "pairs": pairs})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)


def load(path: str) -> "BodyModel":
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    if blob.get("format") not in (FORMAT, FORMAT_V1):
        raise ValueError(f"{path}: not a {FORMAT} file "
                         f"(format {blob.get('format')!r})")
    feats = blob["features"]
    fidx = {f: k for k, f in enumerate(feats)}
    for v in blob["variants"]:
        sw = {}
        for p, e in v["sweeps"].items():
            grid = np.array(e["grid"], float)
            D = np.zeros((len(grid), len(feats)))
            for f, col in e["delta"].items():
                D[:, fidx[f]] = np.nan_to_num(np.array(col, float))
            sw[p] = (grid, D)
        pairs = []
        for a, b, d in v["pairs"]:
            Rq = np.zeros((4, len(feats)))
            for f, col in d.items():
                Rq[:, fidx[f]] = np.nan_to_num(np.array(col, float))
            pairs.append((a, b, Rq))
        v["sweeps"], v["pairs"] = sw, pairs
        v["rest"] = np.array([np.nan if x is None else x for x in v["rest"]],
                             float)
        v["present"] = np.array(v["present"], bool)
    if blob["format"] == FORMAT_V1:
        blob = upgrade_v1(blob)
    return BodyModel(blob)


# Region families whose side is swapped together: an arm's segments and the
# composite, an eye, a brow.
FAMILIES = {"arm": ("arm", "upper_arm", "forearm", "hand"),
            "eye": ("eye",), "brow": ("brow",)}
MID_EPS = 0.05     # hu: a region centroid this close to the chin's x has no
#                    side by position


def upgrade_v1(blob: dict) -> dict:
    """A tsukuyomi.body/1 blob (as load() holds it, sweeps as arrays) in
    format 2, from what the file stores. v1 named sides by the ids, so a
    rig named by screen side had its features mirrored; here each region
    family (arm, eye, brow) takes its side from its rest centroid's x
    (else from the drawables' detected convention), declared sided
    variables (eye.L.open...) follow the parameters' convention, effects
    and hit areas leave the body, and effect variants are dropped.

    What v1 measured with a wrong assignment cannot be undone without the
    Core: an effect or hit-area drawable inside a body region moved its
    features (and a hit area in the face, the frame), and a region whose
    drawables were sided by a parameter named the other way mixes both
    sides. Those land in blob["caveats"]; re-measure to clear them.
    """
    feats = list(blob["features"])
    fidx = {f: k for k, f in enumerate(feats)}
    v0 = blob["variants"][0]
    caveats = []

    def rest_x(region):
        k = fidx.get(f"{region}.x")
        x = None if k is None else v0["rest"][k]
        return None if x is None or math.isnan(x) else float(x)

    # Effects and hit areas out of the body.
    moved_out = {}
    for d in blob["drawables"]:
        leaf, ev = nonbody_leaf([d["id"], d.get("part", "")])
        if leaf and d["region"] != leaf:
            if d["region"] != "unclear":
                moved_out.setdefault(d["region"], []).append(d["id"])
            d.update(region=leaf, side="", confidence="high",
                     evidence=ev + "; v1 upgrade", guess="")
    for r, ids in sorted(moved_out.items()):
        caveats.append(f"{r} was measured with {', '.join(ids)} (now "
                       f"effect / hitarea)" + ("; the frame (chin, hu) "
                                               "too" if r in COMPOSITES[
                                                   "head"] else ""))

    # Naming votes, as detect_naming but from region centroids.
    vd = [0, 0]
    for d in blob["drawables"]:
        leaf, _, s = d["region"].partition(".")
        ns = text_side([d["id"], d.get("part", "")])
        x = rest_x(d["region"]) if leaf in SIDED else None
        if ns and x is not None and abs(x) > MID_EPS:
            vd[0 if (x > 0) == (ns == "L") else 1] += 1
    vp = [0, 0]
    sided_regions = [f"{f}.{s}" for f in ("arm", "eye", "brow")
                     for s in "LR"]
    for p, e in blob["params"].items():
        meta = param_meta(p, e.get("name", ""))
        sw = v0["sweeps"].get(p)
        if not _sided_param(meta) or sw is None:
            continue
        D = sw[1]
        lr = [0.0, 0.0]
        for r in sided_regions:
            x = rest_x(r)
            if x is None or abs(x) <= MID_EPS:
                continue
            mv = float(np.hypot(D[:, fidx[f"{r}.x"]],
                                D[:, fidx[f"{r}.y"]]).max())
            lr[0 if x > 0 else 1] += mv
        if max(lr) <= MOVE_EPS or max(lr) <= 3 * min(lr):
            continue
        geo = "L" if lr[0] > lr[1] else "R"
        vp[0 if geo == meta["side"] else 1] += 1
    naming = {"drawables": "screen" if vd[1] > vd[0] else "character",
              "params": "screen" if vp[1] > vp[0] else "character",
              "votes": {"drawables": vd, "params": vp}}

    # Families whose labels are mirrored.
    swap = set()
    for fam in FAMILIES:
        xs = {s: rest_x(f"{fam}.{s}") for s in "LR"}
        wrong = [s for s, x in xs.items() if x is not None and abs(x) >
                 MID_EPS and (x > 0) != (s == "L")]
        right = [s for s, x in xs.items() if x is not None and abs(x) >
                 MID_EPS and (x > 0) == (s == "L")]
        if wrong and not right:
            swap.add(fam)
        elif not wrong and not right and naming["drawables"] == "screen":
            swap.add(fam)
        mid = [f"{fam}.{s} (x {x:+.3f})" for s, x in xs.items()
               if x is not None and abs(x) <= MID_EPS]
        if mid or (wrong and right):
            caveats.append(f"{fam}: " + ", ".join(
                mid or [f"{fam}.{s} on the same side" for s in "LR"])
                + " -- its drawables mix both sides")
    if naming["drawables"] != naming["params"]:
        mixed = sorted({d["region"] for d in blob["drawables"]
                        if "side from motion" in d["evidence"]})
        if mixed:
            caveats.append(f"{', '.join(mixed)}: drawables sided by "
                           "parameters named the other way round; these "
                           "regions mix both sides")
    leaves = {lf for fam in swap for lf in FAMILIES[fam]}

    def rename_region(r):
        leaf, dot, s = r.partition(".")
        return f"{leaf}.{flip_side(s)}" if dot and leaf in leaves else r

    def rename_feature(f):
        region, _, q = f.rpartition(".")
        if q in POSE_Q and region in POSE_REGIONS:
            return f"{rename_region(region)}.{q}"
        return f
    # Declared sided variables by the parameters' convention.
    declared = dict(blob["declared"])
    if naming["params"] == "screen":
        new = {}
        for n, ps in declared.items():
            m = re.match(r"(.+)\.([LR])\.(.+)$", n)
            new[f"{m.group(1)}.{flip_side(m.group(2))}.{m.group(3)}"
                if m else n] = ps
        dmap = {n: m for n, m in zip(declared, new)}
        declared = new
    else:
        dmap = {n: n for n in declared}
    region_feats = [f for f in feats if f not in dmap]
    new_feats = region_feats + sorted(declared)
    inv = {rename_feature(f): f for f in region_feats}
    inv.update({m: n for n, m in dmap.items()})
    cols = np.array([fidx[inv[f]] for f in new_feats])
    for v in blob["variants"]:
        v["rest"] = v["rest"][cols]
        v["present"] = v["present"][cols]
        v["sweeps"] = {p: (g, D[:, cols]) for p, (g, D) in
                       v["sweeps"].items()}
        v["pairs"] = [(a, b, R[:, cols]) for a, b, R in v["pairs"]]
        v["front"] = {rename_region(k): x for k, x in
                      v.get("front", {}).items()}
    blob["features"] = new_feats
    blob["declared"] = declared
    for p, e in blob["params"].items():
        e.pop("declares", None)
    renamed = {m for o, m in dmap.items() if o != m}
    for n, (p, _s) in declared.items():
        blob["params"][p]["declares"] = n
        if n in renamed:
            blob.setdefault("meaning", {})[n] = (
                f"the character's {'left' if '.L.' in n else 'right'} side:"
                f" {p} (the rig names parameters by screen side)")
    for d in blob["drawables"]:
        r = rename_region(d["region"])
        if r != d["region"]:
            d["region"], d["side"] = r, flip_side(d["side"])
            d["evidence"] += "; side mirrored by rest geometry (v1 upgrade)"
    regions = {}
    for d in blob["drawables"]:
        e = regions.setdefault(d["region"], {"drawables": [], "vertices": 0,
                                             "confidence": {}})
        e["drawables"].append(d["id"])
        e["confidence"][d["confidence"]] = e["confidence"].get(
            d["confidence"], 0) + 1
    old_regions = blob.get("regions", {})
    for r, e in regions.items():
        # Vertex counts per drawable are not in v1; a region whose members
        # did not change keeps its total.
        src = next((o for o in old_regions.values()
                    if sorted(o["drawables"]) == sorted(e["drawables"])),
                   None)
        e["vertices"] = src["vertices"] if src else None
    blob["regions"] = regions
    blob["opacity_flips"] = {p: sorted({rename_region(r) for r in rs})
                             for p, rs in blob.get("opacity_flips",
                                                   {}).items()}
    marks = dict(blob.get("landmarks", {}))
    if "arm" in swap:
        sh = {s: marks.pop(f"shoulder.{s}", None) for s in "LR"}
        for s in "LR":
            if sh[flip_side(s)] is not None:
                marks[f"shoulder.{s}"] = sh[flip_side(s)]
    nf = {f: k for k, f in enumerate(new_feats)}
    for s in "LR":
        marks.pop(f"hand.{s}", None)
        x, y = v0["rest"][nf[f"hand.{s}.x"]], v0["rest"][nf[f"hand.{s}.y"]]
        if not math.isnan(x):
            marks[f"hand.{s}"] = [round(float(x), 4), round(float(y), 4)]
    blob["landmarks"] = marks

    # Effects: parameters and parts named so; variants that show one.
    eff_params = [p for p, e in blob["params"].items()
                  if effect_name(p, e.get("name", ""))]
    for p in eff_params:
        blob["params"][p]["role"] = "effect"
    eff_parts = sorted({q for q in blob.get("rest_parts", {})
                        if effect_name(q)} |
                       {d["part"] for d in blob["drawables"]
                        if effect_name(d.get("part", ""))})
    keep = []
    for k, v in enumerate(blob["variants"]):
        shows = [p for p in eff_params if p in v["switches"] and
                 v["switches"][p] != blob["params"][p]["default"]]
        shows += [q for q in eff_parts if q in v["parts"] and
                  v["parts"][q] != blob.get("rest_parts", {}).get(q, 0.0)]
        if k and shows:
            caveats.append(f"variant {v['name']!r} dropped: it shows "
                           f"effect {', '.join(shows)}")
            continue
        for p in eff_params:
            v["switches"].pop(p, None)
        keep.append(v)
    blob["variants"] = keep
    blob["effects"] = {
        "params": {p: {k: blob["params"][p][k] for k in
                       ("min", "max", "default", "name")} | {"drawables": []}
                   for p in eff_params},
        "parts": {q: {"rest": float(blob.get("rest_parts", {}).get(q, 1.0)),
                      "name": "",
                      "drawables": [d["id"] for d in blob["drawables"]
                                    if d.get("part") == q]}
                  for q in eff_parts},
        "drawables": [d["id"] for d in blob["drawables"]
                      if d["region"] == "effect"]}
    blob["side_naming"] = naming
    blob["format"] = FORMAT
    blob["upgraded_from"] = FORMAT_V1
    blob["caveats"] = caveats
    return blob


# ----------------------------------------------------------------- model

class BodyModel:
    """A rig's forward model, its inverse (solve) and perceive."""

    # Prior weight per (deviation / span)^2, in sigma^2. At 0.02 an
    # unreachable Hiyori hand target (A arm, 0.3 hu off its arc) pulled every
    # body angle and ParamBreath to a bound for 0.01 hu; at 0.5 a full-span
    # move costs what a 0.7 sigma miss does, and a reachable target needing
    # one still lands within 0.2 % (hand gain ~0.5 hu per span).
    LAMBDA = 0.5
    SWITCH_COST = 0.5  # cost of changing arm variant, in sigma^2

    def __init__(self, blob: dict):
        self.blob = blob
        self.features = list(blob["features"])
        self.fidx = {f: k for k, f in enumerate(self.features)}
        self.params = blob["params"]
        self.variants = blob["variants"]
        self.landmarks = blob.get("landmarks", {})

    def to_dict(self) -> dict:
        return self.blob

    # ---------------------------------------------------------- forward

    def default_values(self) -> dict:
        return {p: e["default"] for p, e in self.params.items()}

    def solvable(self, variant: int = 0) -> list[str]:
        v = self.variants[variant]
        return [p for p in v["sweeps"] if self.params[p]["role"] ==
                "solvable"]

    def forward(self, values: dict, variant: int = 0,
                cols=None) -> np.ndarray:
        """Feature vector at `values` (missing params at default) in a
        variant; NaN where the variant lacks the region. `cols` (feature
        indices) restricts it to those features, which is what the solver
        evaluates: on Natori that cut a 12-variant solve from 8 s."""
        F0, sweeps, pairs = self._compiled(variant, cols)
        add = np.zeros(len(F0))
        for p, grid, D in sweeps:
            x = values.get(p)
            if x is not None:
                add += _interp(grid, D, float(x))
        for a, b, Rq in pairs:
            ua, sa = self._u(a, values.get(a))
            ub, sb = self._u(b, values.get(b))
            if ua and ub:
                add += Rq[2 * sa + sb] * ua * ub
        return F0 + add

    def _compiled(self, variant, cols):
        """(rest, [(pid, grid, delta)], [(pid, pid, residuals)]) for a
        variant, sliced to cols and without terms that are zero there."""
        key = (variant, None if cols is None else tuple(int(c) for c in cols))
        cache = self.__dict__.setdefault("_comp", {})
        if key not in cache:
            v = self.variants[variant]
            sl = slice(None) if cols is None else np.asarray(cols, int)
            F0 = v["rest"].copy()
            F0[~v["present"]] = np.nan
            sweeps = [(p, g, D[:, sl]) for p, (g, D) in v["sweeps"].items()
                      if cols is None or np.abs(D[:, sl]).max() > 0]
            pairs = [(a, b, R[:, sl]) for a, b, R in v["pairs"]
                     if cols is None or np.abs(R[:, sl]).max() > 0]
            cache[key] = (F0[sl], sweeps, pairs)
        return cache[key]

    def _u(self, p, x):
        """(fraction of the way from default to the end, end: 0 min 1 max)."""
        if x is None:
            return 0.0, 0
        e = self.params[p]
        d = e["default"]
        if x >= d:
            span = e["max"] - d
            return ((x - d) / span if span > 0 else 0.0), 1
        span = d - e["min"]
        return ((d - x) / span if span > 0 else 0.0), 0

    def perceive(self, values: dict, variant: int | None = None,
                 parts: dict | None = None) -> dict:
        """Canonical state {variable: value} at these parameter values (and
        part opacities, which pick the variant when `variant` is None)."""
        if variant is None:
            variant = self.match_variant(values, parts)
        F = self.forward(values, variant)
        out = {f: float(F[k]) for k, f in enumerate(self.features)
               if not math.isnan(F[k])}
        out["arm.variant"] = self.variants[variant]["name"]
        return out

    def match_variant(self, values: dict | None = None,
                      parts: dict | None = None) -> int:
        values, parts = values or {}, parts or {}
        best, bd = 0, math.inf
        for k, v in enumerate(self.variants):
            d = sum(abs(parts.get(p, o) - o) for p, o in v["parts"].items())
            d += sum(abs(values.get(p, self.params[p]["default"]) - s) /
                     max(self.params[p]["max"] - self.params[p]["min"], 1e-9)
                     for p, s in v["switches"].items())
            if d < bd:
                best, bd = k, d
        return best

    # ------------------------------------------------------------ solve

    def resolve(self, targets: dict) -> dict:
        """Targets with landmark references resolved: "@shoulder.y",
        "@chest.x+0.2", "@hand.R.y-0.1"."""
        out = {}
        for k, t in targets.items():
            if isinstance(t, str) and t.startswith("@"):
                m = re.match(r"@([\w.]+?)([+-][\d.]+)?$", t)
                if not m:
                    raise ValueError(f"bad landmark reference {t!r}")
                name, offs, axis = m.group(1), m.group(2), None
                if name not in self.landmarks and name[-2:] in (".x", ".y"):
                    name, axis = name[:-2], name[-1]
                val = self.landmarks.get(name)
                if val is None:
                    raise ValueError(f"no landmark {name!r}; have "
                                     f"{sorted(self.landmarks)}")
                if isinstance(val, list):
                    val = val[0 if (axis or k[-1]) == "x" else 1]
                out[k] = float(val) + float(offs or 0)
            else:
                out[k] = float(t)
        return out

    def solve(self, targets: dict, current: dict | None = None,
              variant: int | None = None, parts: dict | None = None,
              weights: dict | None = None) -> dict:
        """Parameter values that bring the canonical variables in `targets`
        to their values. Unnamed variables are free; their parameters stay
        near `current` (default: rest) by the prior. Each arm variant is
        solved and the cheapest kept unless `variant` fixes one.

        Returns {params, parts, variant, cost, report: {var: {target, got,
        error, unit, ok}}, missing: [vars no variant has]}.
        """
        targets = self.resolve(targets)
        weights = weights or {}
        current = dict(current or {})
        cur_var = self.match_variant(current, parts)
        tried = [variant] if variant is not None else \
            range(len(self.variants))
        unknown = [t for t in targets if t not in self.fidx]
        if unknown:
            raise KeyError(f"unknown variables {unknown}; see --canon")
        best = None
        for k in tried:
            r = self._solve_variant(targets, current, k, weights)
            if r is None:
                continue
            r["cost"] += self.SWITCH_COST * self._changed(cur_var, k)
            if best is None or r["cost"] < best["cost"]:
                best = r
        if best is None:
            return {"params": {}, "parts": {}, "variant": None, "cost":
                    math.inf, "report": {}, "missing": list(targets)}
        v = self.variants[best["variant"]]
        best["parts"] = dict(v["parts"])
        best["params"].update(v["switches"])
        best["variant_name"] = v["name"]
        return best

    def _solve_variant(self, targets, current, k, weights):
        v = self.variants[k]
        tk = [t for t in targets if v["present"][self.fidx[t]]]
        missing = [t for t in targets if t not in tk]
        if not tk:
            return None
        ti = np.array([self.fidx[t] for t in tk])
        y = np.array([targets[t] for t in tk])
        sig = np.array([SIGMA[feature_unit(t)] / max(weights.get(t, 1.0),
                                                     1e-6) for t in tk])
        # Autonomic and mouth parameters (blink, breath, lip sync) are
        # rewritten by the runtime every frame, so a pose that leans on them
        # does not survive playback: Hiyori's solves used ParamBreath up to
        # 1.0 for hand height. They are free only when their own declared
        # variable is a target (eye.L.open, mouth.open).
        free = [p for p in self.solvable(k)
                if np.abs(v["sweeps"][p][1][:, ti]).max() > 1e-9
                and (self.params[p].get("band") not in ("autonomic", "mouth")
                     or self.params[p].get("declares") in targets)]
        base = {p: e["default"] for p, e in self.params.items()}
        base.update({p: x for p, x in current.items() if p in base})
        base.update(v["switches"])
        lo = np.array([self.params[p]["min"] for p in free])
        hi = np.array([self.params[p]["max"] for p in free])
        span = np.maximum(hi - lo, 1e-9)
        prior = np.clip(np.array([base[p] for p in free], float), lo, hi)
        # Moving more of the body costs more: a hand target should be
        # reached with the arm, not by tilting the torso. Each param's prior
        # weight grows with the body regions it moves; without it Hiyori
        # reached a hand target with BodyAngleY 7.4 and ParamShoulder 0.87.
        lam = np.sqrt(self.LAMBDA * np.array([1.0 + self._reach(p, k)
                                              for p in free]))

        def resid(x):
            vals = dict(base)
            vals.update(zip(free, x))
            F = self.forward(vals, k, ti)
            return np.concatenate([(F - y) / sig, lam * (x - prior) / span])

        def cost(x):
            r = resid(x)
            return float(r @ r)

        x = prior.copy()
        if free:
            x = _lm(resid, x, lo, hi, span)
            # Piecewise-linear models have kinks; one restart from the
            # middle of the box catches a start stuck at a bound.
            x2 = _lm(resid, np.clip((lo + hi) / 2, lo, hi), lo, hi, span)
            if cost(x2) < cost(x):
                x = x2
        vals = dict(base)
        vals.update(zip(free, x))
        F = self.forward(vals, k)
        report = {}
        for t in targets:
            got = F[self.fidx[t]]
            unit = feature_unit(t)
            if math.isnan(got):
                report[t] = {"target": targets[t], "got": None,
                             "unit": unit, "ok": False}
                continue
            err = float(got - targets[t])
            report[t] = {"target": targets[t], "got": float(got),
                         "error": err, "unit": unit,
                         "ok": abs(err) <= TOL[unit]}
        c = cost(x) + 1e3 * len(missing)
        return {"params": {p: float(vals[p]) for p in free},
                "free": free, "variant": k, "cost": c, "report": report,
                "missing": missing}

    def _changed(self, a: int, b: int) -> int:
        """Discrete dimensions (pose parts shown, switch states) that differ
        between two variants: changing the right arm's pose for a left-hand
        target costs, so the untargeted side keeps its configuration."""
        va, vb = self.variants[a], self.variants[b]
        n = sum(abs(va["parts"].get(p, 0) - x) > 0.5
                for p, x in vb["parts"].items())
        n += sum(va["switches"].get(p) != x for p, x in vb["switches"].items())
        return n

    def _reach(self, p, k) -> int:
        """How many DESCRIBE_REGIONS a param moves in variant k."""
        key = (p, k)
        cache = self.__dict__.setdefault("_reach_cache", {})
        if key not in cache:
            D = self.variants[k]["sweeps"][p][1]
            n = 0
            for r in DESCRIBE_REGIONS:
                kx, ky = self.fidx[f"{r}.x"], self.fidx[f"{r}.y"]
                if np.hypot(D[:, kx], D[:, ky]).max() > 0.02:
                    n += 1
            cache[key] = n
        return cache[key]

    # ----------------------------------------------------------- limits

    def limits(self, var: str, variant: int = 0) -> tuple[float, float] | None:
        """Reachable range of one variable in a variant: solved toward a far
        target on each side with everything else free."""
        cache = self.__dict__.setdefault("_limits", {})
        if (var, variant) in cache:
            return cache[(var, variant)]
        v = self.variants[variant]
        k = self.fidx[var]
        if not v["present"][k]:
            return None
        far = 1e3 if feature_unit(var) in ("deg", "pct") else 50.0
        ends = []
        for t in (-far, far):
            r = self._solve_variant({var: t}, {}, variant, {})
            ends.append(r["report"][var]["got"] if r else None)
        out = None if None in ends else (float(min(ends)), float(max(ends)))
        cache[(var, variant)] = out
        return out

    def miss_hints(self, res: dict, max_variants: int = 8) -> dict:
        """For each target a solve missed: {reach: its range alone in the
        solved variant, variants: {name: range} of the (first
        max_variants) other variants whose range holds the target, mirror:
        (the other side's variable, its range)}. Ranges are single-variable
        (everything else free), so a target inside one can still miss in
        combination with others."""
        out = {}
        k0 = res.get("variant")
        for t, e in res.get("report", {}).items():
            if e.get("ok"):
                continue
            h = {"reach": None if k0 is None else self.limits(t, k0),
                 "variants": {}, "mirror": None}
            for k in range(min(len(self.variants), max_variants)):
                if k == k0:
                    continue
                lim = self.limits(t, k)
                if lim and lim[0] - TOL[feature_unit(t)] <= e["target"] \
                        <= lim[1] + TOL[feature_unit(t)]:
                    h["variants"][self.variants[k]["name"]] = lim
            m = re.sub(r"\.([LR])\.", lambda g: f".{flip_side(g.group(1))}.",
                       t)
            if m != t and m in self.fidx and k0 is not None:
                h["mirror"] = (m, self.limits(m, k0))
            out[t] = h
        return out

    def movers(self, region: str, variant: int = 0, top: int = 6) -> list:
        """[(pid, |dx,dy| max hu, roll max deg, scale max pct)] of the params
        that move a region most, from the sweeps."""
        v = self.variants[variant]
        kx, ky = self.fidx[f"{region}.x"], self.fidx[f"{region}.y"]
        kr, ks = self.fidx[f"{region}.roll"], self.fidx[f"{region}.scale"]
        rows = []
        for p, (grid, D) in v["sweeps"].items():
            if self.params[p]["role"] == "effect":
                continue
            mv = float(np.hypot(D[:, kx], D[:, ky]).max())
            rows.append((p, mv, float(np.abs(D[:, kr]).max()),
                         float(np.abs(D[:, ks]).max())))
        rows = [r for r in rows if r[1] > MOVE_EPS * 5 or r[2] > 0.5]
        return sorted(rows, key=lambda r: -(r[1] + r[2] / 30))[:top]


def _interp(grid, D, x):
    if x <= grid[0]:
        return D[0]
    if x >= grid[-1]:
        return D[-1]
    j = int(np.searchsorted(grid, x)) - 1
    w = (x - grid[j]) / (grid[j + 1] - grid[j])
    return D[j] * (1 - w) + D[j + 1] * w


def _lm(resid, x, lo, hi, span, iters=60):
    """Projected Levenberg-Marquardt with forward differences: few (< ~40)
    bounded variables, a piecewise-linear model, no scipy needed."""
    mu = 1e-3
    r = resid(x)
    c = float(r @ r)
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            h = 1e-4 * span[j]
            step = h if x[j] + h <= hi[j] else -h
            xp = x.copy()
            xp[j] += step
            J[:, j] = (resid(xp) - r) / step
        g = J.T @ r
        A = J.T @ J
        improved = False
        for _ in range(8):
            try:
                dx = -np.linalg.solve(A + mu * np.diag(np.diag(A) + 1e-12),
                                      g)
            except np.linalg.LinAlgError:
                mu *= 10
                continue
            xn = np.clip(x + dx, lo, hi)
            rn = resid(xn)
            cn = float(rn @ rn)
            if cn < c - 1e-12:
                x, r, c = xn, rn, cn
                mu = max(mu / 3, 1e-9)
                improved = True
                break
            mu *= 10
        if not improved or np.abs(dx).max() < 1e-7 * span.max():
            break
    return x


# --------------------------------------------------------------- describe

def describe(bm: BodyModel, variant: int = 0) -> str:
    """The measured body as text: frame, landmarks, regions with what moves
    them and how far, and each canonical variable's reachable range."""
    b = bm.blob
    L = [f"body: unit 1 hu = {b['frame']['unit']:.4f} model units; "
         f"{b['frame']['axes']}; origin chin"]
    sn = b.get("side_naming")
    if sn:
        L.append(f"side naming: drawables {sn['drawables']}, params "
                 f"{sn['params']} (votes agree/disagree with rest geometry: "
                 f"{sn['votes']['drawables']}, {sn['votes']['params']}); "
                 "sides below are the character's")
    for c in b.get("caveats", []):
        L.append(f"caveat ({b.get('upgraded_from', 'v1')} file, re-measure "
                 f"to clear): {c}")
    L.append("landmarks (hu): " + ", ".join(
        f"{k} {v}" for k, v in b["landmarks"].items()))
    L.append("regions:")
    for r, e in sorted(b["regions"].items()):
        conf = ", ".join(f"{k} {n}" for k, n in sorted(e["confidence"]
                                                        .items()))
        vt = "?" if e.get("vertices") is None else e["vertices"]
        L.append(f"  {r:<12} {len(e['drawables']):>3} drawables "
                 f"{vt:>5} verts  [{conf}]")
    roles = {}
    for p, e in b["params"].items():
        roles.setdefault(e["role"], []).append(p)
    for role in ("held", "variant", "switch", "physics"):
        if roles.get(role):
            L.append(f"{role}: {', '.join(sorted(roles[role]))}")
    fx = b.get("effects", {})
    if fx.get("params") or fx.get("parts"):
        L.append("effects (never variants or solved; set them on purpose): "
                 + ", ".join(sorted(fx.get("params", {})))
                 + ("; parts " + ", ".join(sorted(fx["parts"]))
                    if fx.get("parts") else "")
                 + f"; {len(fx.get('drawables', []))} effect drawables")
    flips = b.get("opacity_flips", {})
    if flips:
        L.append("opacity-flipping params solved as continuous (joints that "
                 "swap meshes, or missed arm swaps -- check): " + ", ".join(
                     f"{p} ({'/'.join(r)})" for p, r in sorted(flips.items())))
    L.append("variants: " + "; ".join(f"[{k}] {v['name']}" for k, v in
                                       enumerate(bm.variants)))
    v = bm.variants[variant]
    L.append(f"what moves each region (variant [{variant}] {v['name']}; "
             "max |shift| hu / roll deg / scale %):")
    for r in POSE_REGIONS:
        if not v["present"][bm.fidx[f"{r}.x"]]:
            continue
        mv = bm.movers(r, variant)
        if mv:
            L.append(f"  {r:<12} " + "; ".join(
                f"{p} {d:.2f}/{a:.0f}/{s:.0f}" for p, d, a, s in mv))
    L.append(f"reach by side (variant [{variant}]; the character's sides, "
             "each variable alone):")
    for s in "LR":
        r = f"hand.{s}"
        if not v["present"][bm.fidx[f"{r}.x"]]:
            L.append(f"  {r:<7} not drawn")
            continue
        rng = []
        for q in "xy":
            lim = bm.limits(f"{r}.{q}", variant)
            rng.append(f"{q} [{lim[0]:.3f}, {lim[1]:.3f}]" if lim else
                       f"{q} -")
        mv = bm.movers(r, variant, top=1)
        L.append(f"  {r:<7} rest ({v['rest'][bm.fidx[r + '.x']]:.3f}, "
                 f"{v['rest'][bm.fidx[r + '.y']]:.3f}) -> " + ", ".join(rng)
                 + (f" hu; mainly {mv[0][0]}" if mv else " hu"))
    L.append(f"canonical variables (variant [{variant}]): rest -> range")
    for f in bm.features:
        k = bm.fidx[f]
        if not v["present"][k]:
            continue
        region, q = f.rsplit(".", 1)
        if q in POSE_Q and region in POSE_REGIONS and (
                region not in DESCRIBE_REGIONS or (
                q == "scale" and region != "head")):
            continue
        lim = bm.limits(f, variant)
        if lim is None or lim[1] - lim[0] < TOL[feature_unit(f)]:
            continue
        L.append(f"  {f:<16} {v['rest'][k]:>8.3f} -> [{lim[0]:.3f}, "
                 f"{lim[1]:.3f}] {feature_unit(f)}")
    return "\n".join(L)


# -------------------------------------------------------------------- cli

def _render(model_json, frames, path, size=512):
    """Render [(values, parts)] side by side to path (needs GL)."""
    try:
        from ..l2dprobe.backend import Live2DBackend, ensure_display
    except ImportError:
        from l2dprobe.backend import Live2DBackend, ensure_display
    ensure_display()
    from PIL import Image
    with Live2DBackend(model_json, size=size) as be:
        be.calibrate()
        tiles = [be.render(vals, parts=parts) for vals, parts in frames]
    Image.fromarray(np.concatenate(tiles, axis=1), "RGBA").save(path)
    print(f"render -> {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-json", help=".model3.json (to measure/render)")
    ap.add_argument("--rig", help="rigs/<name>.npz: the runtime's bands")
    ap.add_argument("--body", help="rigs/<name>.body.json to read or write")
    ap.add_argument("--probe-dir", default="out")
    ap.add_argument("--hold", default="", help="extra params to hold")
    ap.add_argument("--measure", action="store_true",
                    help="measure the rig and write --body")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--regions", action="store_true",
                    help="list every drawable's region and evidence")
    ap.add_argument("--canon", action="store_true",
                    help="print the canonical variables")
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--solve", help="JSON targets, e.g. '{\"head.yaw\": 20}'")
    ap.add_argument("--current", help="JSON current params (prior)")
    ap.add_argument("--render", help="PNG: rest | solved (needs GL)")
    args = ap.parse_args(argv)

    if args.canon:
        for k, e in canon_spec().items():
            print(f"{k:<20} {e['unit']:<5} {e['source']:<9} {e['meaning']}")
        return 0
    body_path = args.body
    if body_path is None and args.rig:
        body_path = args.rig.rsplit(".npz", 1)[0] + ".body.json"
    if args.measure or (body_path and not os.path.exists(body_path)):
        if not args.model_json:
            ap.error("--measure needs --model-json")
        try:
            from ..l2dprobe import core as l2dcore
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            from l2dprobe import core as l2dcore
        info = read_rig_info(args.model_json, args.rig, args.probe_dir,
                             args.hold)
        blob = measure(l2dcore.load(args.model_json), info)
        blob["rig"] = os.path.basename(args.model_json).split(".")[0]
        # The file name only: the checkout differs between machines (pass
        # --model-json to render or check).
        blob["model_json"] = os.path.basename(args.model_json)
        if body_path is None:
            body_path = args.model_json.rsplit(".model3.json", 1)[0] + \
                ".body.json"
        save(blob, body_path)
        print(f"body -> {body_path}")
    if body_path is None:
        ap.error("need --body or --rig")
    bm = load(body_path)
    if args.regions:
        for d in bm.blob["drawables"]:
            print(f"{d['id']:<24} {d['part']:<22} {d['region']:<12} "
                  f"{d['confidence']:<7} {d['evidence']}"
                  + (f"  guess {d['guess']}" if d.get("guess") else ""))
    if args.describe:
        print(describe(bm, args.variant))
    if args.solve:
        targets = json.loads(args.solve)
        current = json.loads(args.current) if args.current else {}
        res = bm.solve(targets, current)
        print(f"variant [{res['variant']}] {res.get('variant_name')}  "
              f"cost {res['cost']:.3f}")
        hints = bm.miss_hints(res)
        for t, e in res["report"].items():
            if e["got"] is None:
                print(f"  {t:<14} target {e['target']:.3f}: not available "
                      "MISS")
            else:
                print(f"  {t:<14} target {e['target']:8.3f} got "
                      f"{e['got']:8.3f} err {e['error']:+.3f} {e['unit']}"
                      f" {'ok' if e['ok'] else 'MISS'}")
            h = hints.get(t)
            if not h:
                continue
            msg = []
            if h["reach"]:
                msg.append(f"alone it reaches [{h['reach'][0]:.3f}, "
                           f"{h['reach'][1]:.3f}] in this variant")
            for name, lim in h["variants"].items():
                msg.append(f"variant {name!r} reaches [{lim[0]:.3f}, "
                           f"{lim[1]:.3f}]")
            if h["mirror"] and h["mirror"][1]:
                m, lim = h["mirror"]
                msg.append(f"the other side's {m} reaches [{lim[0]:.3f}, "
                           f"{lim[1]:.3f}]")
            if msg:
                print("      " + "; ".join(msg))
        n_miss = sum(not e["ok"] for e in res["report"].values())
        print(f"  MISS: {n_miss} of {len(res['report'])} targets outside "
              "tolerance" + (" (rendering the best effort)" if args.render
                             else "") if n_miss else
              f"  ok: all {len(res['report'])} targets within tolerance")
        dflt = bm.default_values()
        moved = {p: x for p, x in res["params"].items()
                 if abs(x - dflt.get(p, x)) > 1e-4}
        print("  params: " + ", ".join(f"{p}={x:.3f}" for p, x in
                                        sorted(moved.items())))
        if res["parts"]:
            print("  parts: " + ", ".join(f"{p}={x:g}" for p, x in
                                          res["parts"].items()))
        if args.model_json:
            # The forward model's claim against the Core's own geometry.
            try:
                from ..l2dprobe import core as l2dcore
            except ImportError:
                from l2dprobe import core as l2dcore
            ex = exact_state(l2dcore.load(args.model_json), bm,
                             res["params"], res["variant"])
            print("  exact (Core): " + ", ".join(
                f"{t} {ex.get(t, float('nan')):.3f}" for t in res["report"]))
        if args.render:
            mj = args.model_json
            if not mj:
                ap.error("--render needs --model-json")
            v0 = bm.variants[0]
            # Every part the measurement set, not only the arm variant's: a
            # rig without pose3 draws all its switched parts otherwise.
            rp = bm.blob.get("rest_parts", {})
            _render(mj, [(dict(v0["switches"]), {**rp, **v0["parts"]}),
                         (res["params"], {**rp, **res["parts"]})],
                    args.render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
