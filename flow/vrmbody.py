#!/usr/bin/env python3
"""
A VRM avatar as a body the model knows: body file, kinematics, IK solver.

Stage E3 of docs/embodiment.md. flow.body describes a Live2D rig by
measuring its mesh (a forward model, solved by least squares); a VRM avatar
has a declared humanoid skeleton and expression presets, so the same
canonical description (flow.body's frame, sides, landmarks and variables)
comes from the skeleton directly and the solver is analytic.

  file      python -m flow.vrmbody --vrm PATH --out rigs/<name>.body.json
            writes a tsukuyomi.body/2 file with "kind": "vrm" (format below):
            the humanoid bones at rest in the body frame, landmarks, joint
            limits, the expression map, the camera framing, the reach of
            each hand. flow.body.load reads it into a VRMBody, which has
            BodyModel's interface (features, params, variants, landmarks,
            perceive, solve, limits, miss_hints), so flow.pose's prompt,
            solver loop, reach report and play work on it unchanged.
  frame     as flow.body: origin at the chin, x toward the character's own
            left, y up, unit one head height (hu) -- chin to the top of the
            skull, from the vertices skinned to the head bone (hair meshes
            left out, as Live2D's head height leaves the hair out). z
            toward the viewer (the avatar faces +Z in VRM 1.0; a VRM 0
            avatar faces -Z and is turned round, as three-vrm turns it).
  params    the body's "parameters" are its own joint angles in degrees and
            face weights, all canonical names (PARAMS): head / torso yaw,
            pitch, tilt split over neck+head and spine+chest+upperChest;
            per arm the upper arm's elevation (0 hanging, 90 sideways or
            forward, 170 overhead), plane of elevation (0 out to the side,
            90 forward, 130 across the body, -30 behind), axial twist (+
            outward) and the elbow's bend; shoulders.up / shoulder.L.up;
            gaze, eyelids, eye smile, brows, mouth form and open, blush
            where the avatar has a shape for it; the breath and the five
            visemes for the autonomic layer and the voice. Each declared
            variable is its own parameter (declared {var: [var, 1]}).
  kinematics  three-vrm's normalized humanoid: every bone at rest has the
            identity rotation and its world rest position, parented to the
            nearest present humanoid ancestor (VRMHumanBoneParentMap). A
            pose is a local rotation per bone; the stream sends exactly
            those quaternions (state()), so Python's forward kinematics and
            the browser's agree by construction (tests/test_vrm checks it
            in Chromium). The hand is rigid with the forearm (no wrist, no
            fingers); hand.L / hand.R are the palm centres.
  solver    VRMBody.solve(targets, current): declared variables and joint
            angles are set directly (clamped to the limits); a hand target
            is analytic two-bone IK with a pole: the elbow bend from the
            shoulder-palm distance (exact, by bisection on the actual bone
            vectors), then the rotation of the whole arm onto the target,
            searched over the swivel about the shoulder-palm axis for the
            elbow nearest out-down-back (POLE) within the joint limits and
            near the current pose. A target given only in x and y gets a
            depth: where the palm is now, pushed in front of the torso or
            the face when it is in front of them (no hand inside the body),
            and pulled within reach. Out-of-reach targets stretch toward
            the target; the report says what the forward kinematics reach.
            perceive(values) is the reverse (forward kinematics), the self
            check.

Joint limits (LIMITS) are human active ranges cut to what a talking avatar
does and its skin survives: neck rotation 80 deg, flexion 50 / extension 60,
lateral flexion 45 (AAOS norms, Greene & Heckman 1994) -> head yaw +-50,
pitch +-35 (Live2D's ParamAngleY is +-30), tilt +-30; thoracolumbar rotation
30, lateral 25 (same source) for a standing body with the hips still ->
torso yaw +-25, pitch +-15, tilt +-12; shoulder flexion / abduction 180,
horizontal adduction 135, horizontal abduction 45, rotation 70 internal / 90
external -> elevation 0..170, plane -30..130, twist +-70; elbow flexion 150
-> bend 0..145.

Format (tsukuyomi.body/2, kind "vrm"): the keys pose.py reads (frame,
landmarks [head, chest, hips, shoulder.L/R, hand.L/R as [x, y]; shoulder.y
a number], regions, params {min, max, default, band, role, declares},
declared, meaning, features, variants [one, "rest": rest values and
presence per feature; no sweeps stored, VRMBody computes them]) plus
"depth" {landmark: z hu}, "limits_source", "reach" {hand.L/R: {x, y, z}
ranges}, and "vrm" {file, version, faces, license, bones {name: {rest
[x, y, z] metres in the 1.0 frame, parent}}, palm {L, R}, torso_box,
head_box, face_map, visemes, blink, look_at, breath, camera, springs}.

Usage:
    python -m flow.vrmbody --vrm assets/vrm/VRM1_Constraint_Twist_Sample.vrm \\
        --out rigs/seed.body.json
    python -m flow.vrmbody --body rigs/seed.body.json --describe
    python -m flow.vrmbody --body rigs/seed.body.json \\
        --solve '{"hand.R.y": "@head.y", "hand.R.x": -0.9}'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

try:
    from . import body as fb
    from .gltf import GLTF, vrm_info
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import body as fb
    from flow.gltf import GLTF, vrm_info

KIND = "vrm"
# three-vrm's VRMHumanBoneParentMap for the bones driven here; a missing
# parent falls back to its own parent, as three-vrm's normalized rig does.
PARENT = {"hips": None, "spine": "hips", "chest": "spine",
          "upperChest": "chest", "neck": "upperChest", "head": "neck",
          "leftShoulder": "upperChest", "leftUpperArm": "leftShoulder",
          "leftLowerArm": "leftUpperArm", "leftHand": "leftLowerArm",
          "rightShoulder": "upperChest", "rightUpperArm": "rightShoulder",
          "rightLowerArm": "rightUpperArm", "rightHand": "rightLowerArm"}
BONES = tuple(PARENT)
REQUIRED = ("hips", "spine", "head", "leftUpperArm", "leftLowerArm",
            "leftHand", "rightUpperArm", "rightLowerArm", "rightHand")
SIDE = {"L": "left", "R": "right"}
SIGMA_SIDE = {"L": 1.0, "R": -1.0}       # the T-pose arm points +X / -X
# How a head or torso angle splits over the bones that carry it (a bone
# the avatar lacks hands its share to the next one in the list).
HEAD_SPLIT = (("neck", 0.4), ("head", 0.6))
TORSO_SPLIT = (("spine", 0.3), ("chest", 0.3), ("upperChest", 0.4))
SHOULDER_DEG = 12.0     # shoulders.up = 1: the shoulder bone raised 12 deg

# (name, unit, lo, hi, rest). Rest of the arms: relaxed, hanging a little
# out and forward with the elbow soft -- not the file's T-pose.
LIMITS = (
    ("head.yaw", "deg", -50.0, 50.0, 0.0),
    ("head.pitch", "deg", -35.0, 35.0, 0.0),
    ("head.tilt", "deg", -30.0, 30.0, 0.0),
    ("torso.yaw", "deg", -25.0, 25.0, 0.0),
    ("torso.pitch", "deg", -15.0, 15.0, 0.0),
    ("torso.tilt", "deg", -12.0, 12.0, 0.0),
    ("shoulders.up", "unit", -1.0, 1.0, 0.0),
    ("shoulder.L.up", "unit", -1.0, 1.0, 0.0),
    ("shoulder.R.up", "unit", -1.0, 1.0, 0.0),
    ("upper_arm.L.elev", "deg", 0.0, 170.0, 8.0),
    ("upper_arm.L.plane", "deg", -30.0, 130.0, 20.0),
    ("upper_arm.L.twist", "deg", -70.0, 70.0, 0.0),
    ("forearm.L.bend", "deg", 0.0, 145.0, 14.0),
    ("upper_arm.R.elev", "deg", 0.0, 170.0, 8.0),
    ("upper_arm.R.plane", "deg", -30.0, 130.0, 20.0),
    ("upper_arm.R.twist", "deg", -70.0, 70.0, 0.0),
    ("forearm.R.bend", "deg", 0.0, 145.0, 14.0),
)
JOINTS = tuple(n for n, *_ in LIMITS)
ARM = {s: (f"upper_arm.{s}.elev", f"upper_arm.{s}.plane",
           f"upper_arm.{s}.twist", f"forearm.{s}.bend") for s in "LR"}
# Face and autonomic parameters: (name, lo, hi, rest), units. Ranges as
# the Live2D standard ids they stand for (flow.body.DECLARED), so a marker
# means the same number on both bodies.
FACE = (
    ("gaze.x", -1.0, 1.0, 0.0), ("gaze.y", -1.0, 1.0, 0.0),
    ("eye.L.open", 0.0, 1.0, 1.0), ("eye.R.open", 0.0, 1.0, 1.0),
    ("eye.L.smile", 0.0, 1.0, 0.0), ("eye.R.smile", 0.0, 1.0, 0.0),
    ("brow.L.lift", -1.0, 1.0, 0.0), ("brow.R.lift", -1.0, 1.0, 0.0),
    ("brow.L.form", -1.0, 1.0, 0.0), ("brow.R.form", -1.0, 1.0, 0.0),
    ("mouth.form", -1.0, 1.0, 0.0), ("mouth.open", 0.0, 1.0, 0.0),
    ("cheek.blush", 0.0, 1.0, 0.0),
)
# Never targets: the breath and the visemes are the autonomic layer's and
# the voice's (stream.MotionStream writes them every frame).
AUTO = (("breath", 0.0, 1.0, 0.0),) + tuple(
    (f"viseme.{v}", 0.0, 1.0, 0.0) for v in "aiueo")
VISEMES = {"a": "aa", "i": "ih", "u": "ou", "e": "ee", "o": "oh"}
HAND_FEATS = tuple(f"hand.{s}.{q}" for s in "LR" for q in "xyz")
# The elbow's preferred direction (body frame, per side: out, down, back)
# and how much the swivel search weighs it against the current pose and
# the twist; a reach far across the body has its elbow up and out.
POLE = (0.5, -1.0, -0.6)
W_POLE, W_PRIOR, W_TWIST = 1.0, 0.5, 0.3
SWIVEL_STEPS = 72
# Depth a palm keeps in front of the torso / face surface (hu): about half
# a hand's thickness plus a gap, so a hand "at the chest" is in front of it.
HAND_CLEAR = 0.3
# The breath (stream's Autonomic, 0..1 at 0.25 Hz): the shoulders rise by
# runtime.BREATH_TARGET_HU at full breath, as the Live2D breath is
# calibrated to (0.0075 hu on every rig), with the upper chest leaning back
# a third of that angle.
CAMERA_FOV = 24.0      # deg, vertical
CAMERA_HALF_WIDTH = 2.2   # hu: a hand pointing out at 2 hu stays in frame


# ------------------------------------------------------------- rotations

def rx(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def axis_angle(axis, d):
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(d)
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def between(u, v):
    """The smallest rotation taking direction u to direction v."""
    u = np.asarray(u, float) / max(np.linalg.norm(u), 1e-12)
    v = np.asarray(v, float) / max(np.linalg.norm(v), 1e-12)
    c = float(np.clip(u @ v, -1, 1))
    ax = np.cross(u, v)
    if np.linalg.norm(ax) < 1e-9:
        if c > 0:
            return np.eye(3)
        ax = np.cross(u, [1, 0, 0] if abs(u[0]) < 0.9 else [0, 1, 0])
    return axis_angle(ax, math.degrees(math.acos(c)))


def quat(R):
    """(x, y, z, w) of a rotation matrix, w >= 0."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2 * math.sqrt(tr + 1)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s,
             (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4,
             (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def euler(yaw=0.0, pitch=0.0, tilt=0.0):
    """A head / torso bone's rotation in canonical terms: yaw + turns the
    front (+Z) toward the character's left (+X), pitch + looks up / leans
    back (the front rises), tilt + counter-clockwise as the viewer sees it
    (the top toward the character's right, -X)."""
    return ry(yaw) @ rx(-pitch) @ rz(tilt)


def arm_rotation(side, elev, plane, twist):
    """The upper arm's rotation from the T-pose (arm along sigma*X, palm
    down): elevation from hanging in the plane `plane` (0 the frontal
    plane, out to the side; 90 the sagittal plane, forward), then twist
    about the arm's own axis (+ external: a forward forearm turns up)."""
    sg = SIGMA_SIDE[side]
    return ry(-sg * plane) @ rz(sg * (elev - 90.0)) @ rx(-twist)


def arm_angles(side, A, plane0=0.0):
    """(elev, plane, twist) of an arm rotation (arm_rotation's inverse).
    The plane is undefined when the arm hangs straight down or points
    straight up; plane0 is used there."""
    sg = SIGMA_SIDE[side]
    a = A @ np.array([sg, 0.0, 0.0])
    elev = math.degrees(math.acos(float(np.clip(-a[1], -1, 1))))
    if math.hypot(a[0], a[2]) < 1e-6:
        plane = plane0
    else:
        plane = math.degrees(math.atan2(a[2], sg * a[0]))
    M = rz(sg * (elev - 90.0)).T @ ry(-sg * plane).T @ A
    twist = -math.degrees(math.atan2(M[2, 1], M[1, 1]))
    return elev, plane, twist


# ------------------------------------------------------------- the file

def _flip(v, faces):
    """A VRM 0 position (-Z facing) in the 1.0 frame: 180 deg about Y."""
    v = np.asarray(v, float)
    return v * np.array([-1.0, 1.0, -1.0]) if faces == "-z" else v


def _morph(names, suffix):
    for n in names:
        if n.endswith(suffix):
            return n
    return None


def face_map(info, morphs):
    """{variable: {"+": [[target, gain]], "-": [...]}} for the face
    variables this avatar can show; target "expr:<name>" (a VRM
    expression, preset or custom, through three-vrm's expression manager)
    or "morph:<name>" (a morph target by name, written directly). VRoid's
    standard face morphs (Fcl_BRW_*, Fcl_EYE_Joy_L/R, Fcl_MTH_Fun) are
    used where they exist: they move one feature, where the emotion
    presets (happy, angry, sad, relaxed, surprised) move the whole face.
    Without them the presets stand in at reduced gain (MAP_PRESET). A
    variable with neither is absent (not offered)."""
    ex = {n: e for n, e in info["expressions"].items()
          if e["binds"] or e["materials"]}
    lower = {n.lower(): n for n in ex}
    out = {}

    def m(suffix, gain=1.0):
        n = _morph(morphs, suffix)
        return [f"morph:{n}", gain] if n else None

    def e(name, gain):
        n = lower.get(name.lower())
        return [f"expr:{n}", gain] if n else None
    for s in "LR":
        smile = m(f"Fcl_EYE_Joy_{s}") or e("happy", 0.5)
        if smile:
            out[f"eye.{s}.smile"] = {"+": [smile]}
        up = m("Fcl_BRW_Surprised") or e("surprised", 0.5)
        down = m("Fcl_BRW_Angry", 0.7) or e("angry", 0.4)
        if up or down:
            out[f"brow.{s}.lift"] = {"+": [x for x in (up,) if x],
                                     "-": [x for x in (down,) if x]}
        soft = m("Fcl_BRW_Fun") or e("relaxed", 0.5)
        hard = m("Fcl_BRW_Angry") or e("angry", 0.5)
        if soft or hard:
            out[f"brow.{s}.form"] = {"+": [x for x in (soft,) if x],
                                     "-": [x for x in (hard,) if x]}
    smile = m("Fcl_MTH_Fun") or e("happy", 0.4)
    frown = m("Fcl_MTH_Sorrow", 0.8) or e("sad", 0.4)
    if smile or frown:
        out["mouth.form"] = {"+": [x for x in (smile,) if x],
                             "-": [x for x in (frown,) if x]}
    # Blush: a custom expression or a morph named for the cheeks. VRoid
    # avatars draw it in the texture (no morph); then it is absent.
    blush = next((n for n in ex if re.search(r"blush|cheek|照れ|頬", n,
                                             re.I)), None)
    if blush:
        out["cheek.blush"] = {"+": [[f"expr:{blush}", 1.0]]}
    else:
        mn = next((n for n in morphs if re.search(
            r"blush|cheek|照れ|頬", n, re.I)), None)
        if mn:
            out["cheek.blush"] = {"+": [[f"morph:{mn}", 1.0]]}
    return out


def build(path: str, log=print) -> dict:
    """The body file of a .vrm (module docstring)."""
    g = GLTF.load(path)
    info = vrm_info(g)
    faces = info["faces"]
    hb = info["humanoid"]
    miss = [b for b in REQUIRED if b not in hb]
    if miss:
        raise ValueError(f"{path}: humanoid lacks {miss}")
    pos = {b: _flip(g.node_position(hb[b]), faces) for b in BONES if b in hb}
    # Palms: halfway from the wrist to the middle finger's first knuckle.
    palm = {}
    for s in "LR":
        hand = pos[f"{SIDE[s]}Hand"]
        knuckle = hb.get(f"{SIDE[s]}MiddleProximal")
        if knuckle is not None:
            palm[s] = hand + 0.5 * (_flip(g.node_position(knuckle), faces)
                                    - hand)
        else:
            fore = hand - pos[f"{SIDE[s]}LowerArm"]
            palm[s] = hand + 0.3 * fore
    # Vertices: the head's for the frame, the torso's for its front.
    verts = g.skinned_vertices()
    morphs = sorted({n for _m, _x, _d, names in verts for n in names})
    X = np.concatenate([_flip(x, faces) for _n, x, _d, _ in verts])
    D = np.concatenate([d for _n, _x, d, _ in verts])
    hair = np.concatenate([np.full(len(x), bool(re.search(r"hair", n, re.I)))
                           for n, x, _d, _ in verts])
    head_n = hb["head"]
    hv = X[(D == head_n) & ~hair]
    if len(hv) < 20:
        hv = X[D == head_n]
    hpos = pos["head"]
    front = hv[hv[:, 2] > hpos[2]]
    if len(front) == 0:
        front = hv
    k = int(np.argmin(front[:, 1]))
    chin = np.array([hpos[0], front[k, 1], front[k, 2]])
    top = float(hv[:, 1].max())
    hu = top - chin[1]
    if not 0.05 < hu < 1.0:
        raise ValueError(f"{path}: head height {hu:.3f} m from the head "
                         "vertices is implausible")
    log(f"vrmbody: VRM {info['version']}, chin y {chin[1]:.3f} m, top of "
        f"head {top:.3f} m: 1 hu = {hu:.4f} m")

    def body(p):
        return (np.asarray(p, float) - chin) / hu
    torso_ids = {hb[b] for b in ("spine", "chest", "upperChest") if b in hb}
    tv = body(X[np.isin(D, list(torso_ids)) & ~hair])
    if len(tv) < 10:
        tv = body(X[np.isin(D, [hb["spine"]])])
    sh = {s: body(pos[f"{SIDE[s]}UpperArm"]) for s in "LR"}
    sh_y = float(np.mean([sh[s][1] for s in "LR"]))
    hips = body(pos["hips"])
    torso_box = {"x": float(np.percentile(np.abs(tv[:, 0]), 98)),
                 "y0": float(hips[1]), "y1": float(sh_y + 0.15),
                 "z": float(np.percentile(tv[:, 2], 99.5))}
    hvb = body(hv)
    head_box = {"x": float(np.percentile(np.abs(hvb[:, 0]), 98)),
                "y0": -0.15, "y1": float(hvb[:, 1].max() + 0.1),
                "z": float(np.percentile(hvb[:, 2], 99.5))}
    # The chest landmark: the front of the chest, at the height where the
    # torso is deepest above the waist (a bust, or the pectorals).
    band = tv[(tv[:, 1] > hips[1] + 0.3 * (sh_y - hips[1]))
              & (tv[:, 1] < sh_y) & (np.abs(tv[:, 0]) < 0.5)]
    chest_y = float(band[np.argmax(band[:, 2]), 1]) if len(band) else \
        float(0.5 * (sh_y + hips[1]))
    bones = {}
    for b in BONES:
        if b not in hb:
            continue
        par = PARENT[b]
        while par is not None and par not in hb:
            par = PARENT[par]
        bones[b] = {"rest": [round(float(x), 6) for x in pos[b]],
                    "parent": par, "node": int(hb[b])}
    look = info["look_at"]
    fmap = face_map(info, morphs)
    ex = {n for n, e in info["expressions"].items()
          if e["binds"] or e["materials"]}
    blink = {s: (f"blink{SIDE[s].title()}" if f"blink{SIDE[s].title()}" in ex
                 else "blink" if "blink" in ex else None) for s in "LR"}
    vis = {v: (n if n in ex else None) for v, n in VISEMES.items()}
    lic = info["license"]
    blob = {
        "format": fb.FORMAT, "kind": KIND,
        "rig": os.path.splitext(os.path.basename(path))[0],
        "vrm_file": os.path.basename(path),
        "frame": {"origin": [round(float(x), 6) for x in chin],
                  "unit": round(float(hu), 6), "unit_name": "hu (head height)",
                  "axes": "x toward the character's left, y up, z toward "
                          "the viewer"},
        "vrm": {"file": os.path.basename(path), "version": info["version"],
                "faces": faces, "license": lic, "bones": bones,
                "palm": {s: [round(float(x), 6) for x in palm[s]]
                         for s in "LR"},
                "torso_box": torso_box, "head_box": head_box,
                "face_map": fmap, "visemes": vis, "blink": blink,
                "look_at": look, "springs": info["springs"],
                "expressions": sorted(ex), "morphs": len(morphs)},
    }
    blob["landmarks"] = {
        "chin": [0.0, 0.0],
        "head": [0.0, round(float(hvb[:, 1].max() / 2), 4)],
        "shoulder.L": [round(float(sh["L"][0]), 4),
                       round(float(sh["L"][1]), 4)],
        "shoulder.R": [round(float(sh["R"][0]), 4),
                       round(float(sh["R"][1]), 4)],
        "shoulder.y": round(sh_y, 4),
        "chest": [0.0, round(chest_y, 4)],
        "hips": [0.0, round(float(hips[1]), 4)],
    }
    blob["depth"] = {"chin": 0.0, "head": round(head_box["z"], 4),
                     "face": round(head_box["z"], 4),
                     "chest": round(torso_box["z"], 4),
                     "shoulder": round(float(np.mean([sh[s][2]
                                                      for s in "LR"])), 4)}
    blob.update(_parameters(fmap, blink, vis, info["springs"]))
    # Hands at rest and their reach need the kinematics: built on the blob.
    bm = VRMBody(blob)
    rest = bm.palms(bm.default_values())
    for s in "LR":
        blob["landmarks"][f"hand.{s}"] = [round(float(rest[s][0]), 4),
                                          round(float(rest[s][1]), 4)]
    blob["reach"] = {f"hand.{s}": {q: [round(v, 3) for v in bm.limits(
        f"hand.{s}.{q}")] for q in "xyz"} for s in "LR"}
    blob["vrm"]["breath"] = bm.breath_cal()
    blob["vrm"]["camera"] = camera(blob)
    blob["limits_source"] = ("human active ranges (AAOS norms, Greene & "
                             "Heckman 1994) cut to a talking avatar; "
                             "flow/vrmbody.py LIMITS")
    v = blob["variants"][0]
    v["rest"] = [None if math.isnan(x) else round(float(x), 5)
                 for x in bm.forward(bm.default_values())]
    return blob


def _parameters(fmap, blink, vis, springs=0) -> dict:
    """params, declared, meaning, features, regions, variants (no rest
    yet) for an avatar with this face map."""
    params, declared, meaning = {}, {}, {}
    for n, unit, lo, hi, rest in LIMITS:
        params[n] = {"min": lo, "max": hi, "default": rest,
                     "band": "expressive", "role": "solvable", "unit": unit,
                     "declares": n}
    face = []
    for n, lo, hi, rest in FACE:
        if n.startswith("gaze."):
            ok = True
        elif n.startswith("eye.") and n.endswith(".open"):
            ok = blink[n.split(".")[1]] is not None
        elif n == "mouth.open":
            ok = any(vis.values())
        else:
            ok = n in fmap
        if ok:
            face.append(n)
            params[n] = {"min": lo, "max": hi, "default": rest,
                         "band": ("autonomic" if n.endswith(".open")
                                  and n.startswith("eye") else
                                  "mouth" if n == "mouth.open" else
                                  "expressive"),
                         "role": "solvable", "unit": "unit", "declares": n}
    for n, lo, hi, rest in AUTO:
        params[n] = {"min": lo, "max": hi, "default": rest,
                     "band": "autonomic" if n == "breath" else "mouth",
                     "role": "held", "unit": "unit"}
    for n in list(JOINTS) + face:
        declared[n] = [n, 1.0]
        meaning[n] = (fb.SIGN[n][2] if n in fb.SIGN else
                      JOINT_MEANING.get(n.split(".", 1)[0] + "." +
                                        n.rsplit(".", 1)[-1], ""))
    features = [n for n in list(JOINTS) + face] + list(HAND_FEATS)
    regions = {r: {"drawables": [], "vertices": None, "confidence": {}}
               for r in ("face", "eye.L", "eye.R", "mouth", "neck", "torso",
                         "hips", "upper_arm.L", "upper_arm.R", "forearm.L",
                         "forearm.R", "hand.L", "hand.R")}
    if any(f.startswith("brow.") for f in face):
        regions["brow.L"] = regions["brow.R"] = {"drawables": [],
                                                 "vertices": None,
                                                 "confidence": {}}
    if springs:
        regions["hair"] = {"drawables": [], "vertices": None,
                           "confidence": {}}
    if "cheek.blush" in face:
        regions["cheek"] = {"drawables": [], "vertices": None,
                            "confidence": {}}
    variant = {"name": "rest", "parts": {}, "switches": {}, "weight": 1.0,
               "front": {}, "rest": [], "present": [True] * len(features),
               "sweeps": {}, "pairs": []}
    return {"params": params, "declared": declared, "meaning": meaning,
            "features": features, "regions": regions,
            "variants": [variant], "rest_parts": {}, "side_naming": None}


JOINT_MEANING = {
    "upper_arm.elev": "0 hanging, 90 out level with the shoulder, 170 "
                      "overhead",
    "upper_arm.plane": "0 out to your side, 90 forward, 130 across your "
                       "body, -30 behind",
    "upper_arm.twist": "+ turns the upper arm outward",
    "forearm.bend": "0 straight, 145 fully bent",
}


def camera(blob) -> dict:
    """Framing in the 1.0 frame (metres): from just below the resting
    hands to a raised hand's height above the head, and at least
    CAMERA_HALF_WIDTH to each side (a pointing hand); the live page and
    the offline renderer both use it."""
    hu = blob["frame"]["unit"]
    o = np.array(blob["frame"]["origin"])
    L = blob["landmarks"]
    y0 = min(L["hips"][1], L["hand.L"][1], L["hand.R"][1]) - 0.3
    y1 = L["head"][1] * 2 + 0.9
    yc = 0.5 * (y0 + y1)
    h = max(y1 - y0, 2 * CAMERA_HALF_WIDTH) * hu
    dist = 0.5 * h / math.tan(math.radians(CAMERA_FOV / 2))
    target = o + np.array([0.0, yc * hu, 0.0])
    return {"fov": CAMERA_FOV, "target": [round(float(x), 4) for x in target],
            "position": [round(float(target[0]), 4),
                         round(float(target[1]), 4),
                         round(float(target[2] + dist), 4)]}


# ------------------------------------------------------------- the body

class VRMBody(fb.BodyModel):
    """A VRM body with BodyModel's interface: one variant (no arm
    drawings), parameters = joint angles and face weights, forward() by
    forward kinematics, solve() analytic."""

    def __init__(self, blob: dict):
        blob = dict(blob)
        vs = []
        for v in blob["variants"]:
            v = dict(v)
            n = len(blob["features"])
            r = v.get("rest") or [None] * n
            v["rest"] = np.array([np.nan if x is None else x for x in r],
                                 float)
            v["present"] = np.array(v.get("present") or [True] * n, bool)
            v["sweeps"], v["pairs"] = {}, []
            vs.append(v)
        blob["variants"] = vs
        super().__init__(blob)
        self.vrm = blob["vrm"]
        self.hu = float(blob["frame"]["unit"])
        self.origin = np.array(blob["frame"]["origin"], float)
        self.depth = blob.get("depth", {})
        self.bones = {b: e for b, e in self.vrm["bones"].items()}
        self.order = [b for b in BONES if b in self.bones]
        rest = {b: np.array(e["rest"], float) for b, e in self.bones.items()}
        self.rest = rest
        self.palm_rest = {s: np.array(self.vrm["palm"][s], float)
                          for s in "LR"}
        # T-pose alignment: the actual rest directions of each upper arm
        # and forearm (a file's "T" can droop a few degrees) are rotated
        # onto +-X first, so the arm angles mean what they say.
        self.align = {}
        for s in "LR":
            sg = SIGMA_SIDE[s]
            u, lo, h = (rest[f"{SIDE[s]}UpperArm"], rest[f"{SIDE[s]}LowerArm"],
                        rest[f"{SIDE[s]}Hand"])
            self.align[s] = (between(lo - u, [sg, 0, 0]),
                             between(h - lo, [sg, 0, 0]))
        self.kind = KIND
        self.prompt_kind = "a 3D (VRM) character"
        if "breath" in self.vrm:
            self._breath = self.vrm["breath"]
        else:
            self._breath = None
        # Sweeps for flow.pose's helpers (arm_params, main movers): one
        # parameter at a time over its range, from forward kinematics.
        if vs and not vs[0]["sweeps"]:
            vs[0]["sweeps"] = self._sweeps()
            if np.isnan(vs[0]["rest"]).all():
                vs[0]["rest"] = self.forward(self.default_values())

    # ------------------------------------------------------- kinematics

    def default_values(self) -> dict:
        return {p: e["default"] for p, e in self.params.items()}

    def local_rotations(self, values: dict) -> dict:
        """{bone: local rotation (3x3)} in the normalized rig."""
        v = {**self.default_values(), **(values or {})}
        out = {b: np.eye(3) for b in self.order}

        def split(pairs):
            have = [(b, w) for b, w in pairs if b in self.bones]
            tot = sum(w for _b, w in pairs)
            got = sum(w for _b, w in have)
            return [(b, w * tot / got) for b, w in have] if got else []
        for b, w in split(TORSO_SPLIT):
            out[b] = euler(w * v["torso.yaw"], w * v["torso.pitch"],
                           w * v["torso.tilt"]) @ out[b]
        for b, w in split(HEAD_SPLIT):
            out[b] = euler(w * v["head.yaw"], w * v["head.pitch"],
                           w * v["head.tilt"]) @ out[b]
        br = self._breath or {}
        b_up = float(v.get("breath", 0.0)) * br.get("shoulder_deg", 0.0)
        b_ch = float(v.get("breath", 0.0)) * br.get("chest_deg", 0.0)
        top = "upperChest" if "upperChest" in self.bones else "chest" \
            if "chest" in self.bones else "spine"
        if b_ch:
            out[top] = out[top] @ rx(-b_ch)
        for s in "LR":
            sg = SIGMA_SIDE[s]
            up = SHOULDER_DEG * (v["shoulders.up"] + v[f"shoulder.{s}.up"]) \
                + b_up
            sb = f"{SIDE[s]}Shoulder"
            if sb in self.bones:
                out[sb] = rz(sg * up)
            e, p, t, bend = (v[k] for k in ARM[s])
            a_u, a_f = self.align[s]
            Q_ua = arm_rotation(s, e, p, t) @ a_u
            if sb not in self.bones and up:
                Q_ua = rz(sg * up) @ Q_ua
            out[f"{SIDE[s]}UpperArm"] = Q_ua
            out[f"{SIDE[s]}LowerArm"] = a_u.T @ ry(-sg * bend) @ a_f
        return out

    def fk(self, values: dict):
        """({bone: world rotation}, {bone: world position}) in the 1.0
        frame, metres."""
        L = self.local_rotations(values)
        W, P = {}, {}
        for b in self.order:
            par = self.bones[b]["parent"]
            if par is None:
                W[b], P[b] = L[b], self.rest[b].copy()
            else:
                W[b] = W[par] @ L[b]
                P[b] = P[par] + W[par] @ (self.rest[b] - self.rest[par])
        return W, P

    def palms(self, values: dict, W=None, P=None) -> dict:
        """{side: palm centre in hu}."""
        if W is None:
            W, P = self.fk(values)
        out = {}
        for s in "LR":
            h = f"{SIDE[s]}Hand"
            p = P[h] + W[h] @ (self.palm_rest[s] - self.rest[h])
            out[s] = (p - self.origin) / self.hu
        return out

    def to_body(self, p):
        return (np.asarray(p, float) - self.origin) / self.hu

    def to_world(self, b):
        return self.origin + self.hu * np.asarray(b, float)

    def forward(self, values: dict, variant: int = 0,
                cols=None) -> np.ndarray:
        v = {**self.default_values(), **(values or {})}
        F = np.full(len(self.features), np.nan)
        for k, f in enumerate(self.features):
            if f in v:
                F[k] = float(v[f])
        pal = self.palms(v)
        for s in "LR":
            for i, q in enumerate("xyz"):
                F[self.fidx[f"hand.{s}.{q}"]] = float(pal[s][i])
        return F if cols is None else F[np.asarray(cols, int)]

    def perceive(self, values: dict, variant=None, parts=None) -> dict:
        """Canonical state at these values, plus the elbows and shoulders
        (hu) for the self check."""
        v = {**self.default_values(), **(values or {})}
        F = self.forward(v)
        out = {f: float(F[k]) for k, f in enumerate(self.features)
               if not math.isnan(F[k])}
        _W, P = self.fk(v)
        for s in "LR":
            for name, b in (("elbow", "LowerArm"), ("shoulder", "UpperArm")):
                p = self.to_body(P[f"{SIDE[s]}{b}"])
                out[f"{name}.{s}"] = [round(float(x), 4) for x in p]
        out["arm.variant"] = "rest"
        return out

    def match_variant(self, values=None, parts=None) -> int:
        return 0

    def _sweeps(self, n=9) -> dict:
        base = self.default_values()
        F0 = self.forward(base)
        out = {}
        for p in JOINTS:
            e = self.params[p]
            grid = np.linspace(e["min"], e["max"], n)
            D = np.zeros((n, len(self.features)))
            for i, x in enumerate(grid):
                D[i] = np.nan_to_num(self.forward({**base, p: x}) - F0)
            out[p] = (grid, D)
        for p in self.params:
            if p not in out and p in self.fidx:
                e = self.params[p]
                grid = np.linspace(e["min"], e["max"], n)
                D = np.zeros((n, len(self.features)))
                D[:, self.fidx[p]] = grid - e["default"]
                out[p] = (grid, D)
        return out

    # ------------------------------------------------------------ solve

    def resolve(self, targets: dict) -> dict:
        """flow.body's landmark references, plus depth: "@chest.z", or a
        bare "@chest" on a .z variable, reads the landmark's depth."""
        plain, out = {}, {}
        for k, t in targets.items():
            if isinstance(t, str) and t.startswith("@"):
                m = re.match(r"@([\w.]+?)([+-][\d.]+)?$", t)
                if m and (m.group(1).endswith(".z") or k.endswith(".z")):
                    name = m.group(1)[:-2] if m.group(1).endswith(".z") \
                        else m.group(1)
                    if name not in self.depth:
                        raise ValueError(f"no depth for {name!r}; have "
                                         f"{sorted(self.depth)}")
                    out[k] = float(self.depth[name]) + float(m.group(2) or 0)
                    continue
            plain[k] = t
        out.update(fb.BodyModel.resolve(self, plain))
        return {k: out[k] for k in targets}

    def solve(self, targets: dict, current: dict | None = None,
              variant=None, parts=None, weights=None) -> dict:
        """BodyModel.solve's contract (module docstring, "solver")."""
        targets = self.resolve(targets)
        unknown = [t for t in targets if t not in self.fidx]
        if unknown:
            raise KeyError(f"unknown variables {unknown}")
        cur = {**self.default_values(), **(current or {})}
        vals = dict(cur)
        free = []
        for t, y in targets.items():
            if t in self.params:
                e = self.params[t]
                vals[t] = float(np.clip(y, e["min"], e["max"]))
                free.append(t)
        for s in "LR":
            want = {q: targets[f"hand.{s}.{q}"] for q in "xyz"
                    if f"hand.{s}.{q}" in targets}
            if not want or any(p in targets for p in ARM[s]):
                continue
            sol = self.arm_ik(s, want, vals)
            vals.update(sol)
            free += list(ARM[s])
        F = self.forward(vals)
        report = {}
        cost = 0.0
        for t, y in targets.items():
            got = float(F[self.fidx[t]])
            unit = fb.feature_unit(t)
            err = got - float(y)
            cost += (err / fb.SIGMA[unit]) ** 2
            report[t] = {"target": float(y), "got": got, "error": err,
                         "unit": unit, "ok": abs(err) <= fb.TOL[unit]}
        return {"params": {p: float(vals[p]) for p in free}, "free": free,
                "variant": 0, "variant_name": "rest", "parts": {},
                "cost": cost, "report": report, "missing": []}

    def _box_clear(self, x, y, z):
        """The smallest depth a palm at (x, y) may have: in front of the
        torso or the face where it is in front of them."""
        zmin = -math.inf
        for box in (self.vrm["torso_box"], self.vrm["head_box"]):
            if abs(x) < box["x"] + 0.15 and box["y0"] <= y <= box["y1"]:
                zmin = max(zmin, box["z"] + HAND_CLEAR)
        return zmin

    def arm_ik(self, s, want: dict, values: dict) -> dict:
        """Joint angles of arm s putting its palm at `want` ({x, y, z},
        hu; missing components as module docstring), from the pose
        `values` (torso, shoulders, the arm's current angles)."""
        sg = SIGMA_SIDE[s]
        W, P = self.fk(values)
        cur = self.palms(values, W, P)[s]
        ub, lb, hb = (f"{SIDE[s]}UpperArm", f"{SIDE[s]}LowerArm",
                      f"{SIDE[s]}Hand")
        S = P[ub]
        par = self.bones[ub]["parent"]
        Rp = W[par]
        a_u, a_f = self.align[s]
        # Upper arm and palm offset in the arm's aligned frame (upper arm
        # along sigma*X): u1, and c (elbow -> palm at bend 0).
        u1 = a_u @ (self.rest[lb] - self.rest[ub])
        c = a_f @ (self.palm_rest[s] - self.rest[lb])
        reach = float(np.linalg.norm(u1 + c))
        tb = np.array([want.get("x", cur[0]), want.get("y", cur[1]),
                       want.get("z", np.nan)], float)
        Sb = self.to_body(S)
        if math.isnan(tb[2]):
            z = max(float(cur[2]), self._box_clear(tb[0], tb[1], cur[2]))
            r2 = (tb[0] - Sb[0]) ** 2 + (tb[1] - Sb[1]) ** 2
            R = reach / self.hu * 0.995
            if r2 < R * R:
                dz = math.sqrt(R * R - r2)
                z = float(np.clip(z, Sb[2] - dz, Sb[2] + dz))
            # Too near the shoulder for the arm to fold that far: forward
            # until it can (the palm meets the target in x and y).
            r_min = 1.02 * float(np.linalg.norm(
                u1 + ry(-sg * self.params[ARM[s][3]]["max"]) @ c)) / self.hu
            if r2 + (z - Sb[2]) ** 2 < r_min * r_min and r2 < r_min * r_min:
                z = Sb[2] + math.sqrt(r_min * r_min - r2)
            tb[2] = z
        T = self.to_world(tb)
        t = Rp.T @ (T - S)
        d = float(np.linalg.norm(t))
        lo_b, hi_b = (self.params[ARM[s][3]]["min"],
                      self.params[ARM[s][3]]["max"])

        def length(b):
            return float(np.linalg.norm(u1 + ry(-sg * b) @ c))
        # |u1 + R(b) c| falls as the elbow bends: bisection.
        if d >= length(lo_b):
            bend = lo_b
        elif d <= length(hi_b):
            bend = hi_b
        else:
            a, b = lo_b, hi_b
            for _ in range(50):
                m = 0.5 * (a + b)
                if length(m) > d:
                    a = m
                else:
                    b = m
            bend = 0.5 * (a + b)
        sv = u1 + ry(-sg * bend) @ c
        th = t / max(d, 1e-9)
        A0 = between(sv, th)
        pole = Rp.T @ np.array([sg * POLE[0], POLE[1], POLE[2]])
        pole = pole - (pole @ th) * th
        pn = np.linalg.norm(pole)
        pole = pole / pn if pn > 1e-9 else pole
        e0, p0, w0, _b0 = (values[k] for k in ARM[s])
        lim = [(self.params[k]["min"], self.params[k]["max"]) for k in ARM[s]]
        best = None
        for i in range(SWIVEL_STEPS):
            psi = -180.0 + 360.0 * i / SWIVEL_STEPS
            A = axis_angle(th, psi) @ A0
            e, p, w = arm_angles(s, A, p0)
            viol = sum(max(lo - x, 0.0, x - hi) for x, (lo, hi)
                       in zip((e, p, w), lim[:3]))
            elbow = A @ np.array([sg, 0.0, 0.0])
            elbow = elbow - (elbow @ th) * th
            en = np.linalg.norm(elbow)
            pc = 1.0 - (float(elbow @ pole) / en if en > 1e-6 else 0.0)
            prior = ((e - e0) / 170) ** 2 + ((p - p0) / 160) ** 2 + \
                ((w - w0) / 140) ** 2
            cost = 1e3 * viol + W_POLE * pc + W_PRIOR * prior + \
                W_TWIST * (w / 70.0) ** 2
            if best is None or cost < best[0]:
                best = (cost, e, p, w)
        _c, e, p, w = best
        out = {}
        for k, x, (lo, hi) in zip(ARM[s], (e, p, w, bend), lim + [(lo_b,
                                                                     hi_b)]):
            out[k] = float(np.clip(x, lo, hi))
        return out

    # ----------------------------------------------------------- limits

    def limits(self, var: str, variant: int = 0):
        cache = self.__dict__.setdefault("_limits", {})
        if var in cache:
            return cache[var]
        if var in self.params:
            e = self.params[var]
            out = (float(e["min"]), float(e["max"]))
        elif var in HAND_FEATS:
            s, q = var.split(".")[1], var.split(".")[2]
            pts = self._workspace(s)
            i = "xyz".index(q)
            out = (float(pts[:, i].min()), float(pts[:, i].max()))
        else:
            out = None
        cache[var] = out
        return out

    def _workspace(self, s):
        """Palm positions over a grid of arm joint angles (torso at
        rest): what the hand can reach."""
        cache = self.__dict__.setdefault("_ws", {})
        if s in cache:
            return cache[s]
        base = self.default_values()
        lim = {k: (self.params[k]["min"], self.params[k]["max"])
               for k in ARM[s]}
        pts = []
        ek, pk, tk, bk = ARM[s]
        for e in np.linspace(*lim[ek], 18):
            for p in np.linspace(*lim[pk], 17):
                for b in np.linspace(*lim[bk], 8):
                    v = dict(base)
                    v.update({ek: e, pk: p, tk: 0.0, bk: b})
                    pts.append(self.palms(v)[s])
        cache[s] = np.array(pts)
        return cache[s]

    def movers(self, region, variant=0, top=6):
        return []

    def breath_cal(self) -> dict:
        """{shoulder_deg, chest_deg}: the full breath raises the shoulder
        joints by runtime.BREATH_TARGET_HU (module docstring)."""
        try:
            from .runtime import BREATH_TARGET_HU
        except ImportError:
            from flow.runtime import BREATH_TARGET_HU
        old = self._breath
        self._breath = {"shoulder_deg": 1.0, "chest_deg": 1.0 / 3}
        base = self.default_values()
        y0 = np.mean([self.to_body(self.fk(base)[1][f"{SIDE[s]}UpperArm"])[1]
                      for s in "LR"])
        y1 = np.mean([self.to_body(self.fk({**base, "breath": 1.0})[1][
            f"{SIDE[s]}UpperArm"])[1] for s in "LR"])
        self._breath = old
        k = BREATH_TARGET_HU / max(abs(y1 - y0), 1e-9)
        return {"shoulder_deg": round(k, 4), "chest_deg": round(k / 3, 4),
                "target_hu": BREATH_TARGET_HU}

    # ------------------------------------------------------------ state

    def state(self, values: dict) -> dict:
        """What the browser applies for one frame: {"bones": {humanoid
        bone: [x, y, z, w] local rotation in the normalized rig (VRM 1.0
        frame; the page negates x and z for a VRM 0 avatar, as three-vrm
        examples do)}, "look": [yaw, pitch] deg for vrm.lookAt, "expr":
        {expression: weight}, "morph": {morph name: weight}}."""
        v = {**self.default_values(), **(values or {})}
        L = self.local_rotations(v)
        bones = {}
        for b in self.order:
            q = quat(L[b])
            if b == "hips" and abs(q[3] - 1) < 1e-9:
                continue
            bones[b] = [round(float(x), 5) for x in q]
        la = self.vrm["look_at"]
        gx, gy = float(v.get("gaze.x", 0)), float(v.get("gaze.y", 0))
        # three-vrm's look-at pitch is + down (eye direction (0, -0.17,
        # 0.98) at pitch +90 on both samples, in Chromium), and it maps a
        # negative pitch through rangeMapVerticalDown: gaze.y + (up) is a
        # negative pitch over that map's input range.
        look = [round(gx * la["h"], 3),
                round(-gy * (la["v_down"] if gy > 0 else la["v_up"]), 3)]
        expr, morph = {}, {}

        def add(target, w):
            if w <= 0:
                return
            kind, name = target.split(":", 1)
            d = expr if kind == "expr" else morph
            d[name] = min(1.0, d.get(name, 0.0) + w)
        for var, m in self.vrm["face_map"].items():
            x = float(v.get(var, 0.0))
            if var.startswith("brow."):
                # One brow morph for both sides (VRoid draws them together):
                # each side gives half.
                x *= 0.5
            for sign, lst in m.items():
                k = x if sign == "+" else -x
                if k > 0:
                    for target, gain in lst:
                        add(target, gain * k)
        for s in "LR":
            name = self.vrm["blink"][s]
            if name:
                w = 1.0 - float(np.clip(v.get(f"eye.{s}.open", 1.0), 0, 1))
                if name == "blink":
                    w *= 0.5
                if w > 0:
                    expr[name] = min(1.0, expr.get(name, 0.0) + w)
        vis = self.vrm["visemes"]
        any_v = False
        for k, name in vis.items():
            w = float(v.get(f"viseme.{k}", 0.0))
            if name and w > 0:
                expr[name] = min(1.0, expr.get(name, 0.0) + w)
                any_v = True
        if not any_v and vis.get("a"):
            w = float(v.get("mouth.open", 0.0))
            if w > 0:
                expr[vis["a"]] = min(1.0, w)
        return {"bones": bones, "look": look,
                "expr": {k: round(x, 4) for k, x in sorted(expr.items())},
                "morph": {k: round(x, 4) for k, x in sorted(morph.items())}}


def describe(bm: VRMBody) -> str:
    b = bm.blob
    v = bm.vrm
    L = [f"VRM {v['version']} avatar {v['file']}: 1 hu = {bm.hu:.4f} m; "
         f"{b['frame']['axes']}; origin chin",
         "license: " + json.dumps(v["license"], ensure_ascii=False),
         "landmarks (hu): " + ", ".join(f"{k} {x}" for k, x in
                                        b["landmarks"].items()),
         "depth (hu): " + ", ".join(f"{k} {x}" for k, x in
                                    b["depth"].items()),
         "bones: " + ", ".join(bm.order),
         f"look-at {v['look_at']['type']}, +-{v['look_at']['h']:g} deg at "
         f"gaze 1; spring chains {v['springs']}",
         "face map: " + "; ".join(
             f"{var} " + " ".join(f"{sg}{','.join(f'{t}x{g:g}' for t, g in l)}"
                                  for sg, l in m.items() if l)
             for var, m in v["face_map"].items()),
         "blink: " + json.dumps(v["blink"]) + "; visemes: "
         + json.dumps(v["visemes"]),
         "parameters (rest, range):"]
    for p, e in bm.params.items():
        L.append(f"  {p:<18} {e['default']:>6g}  [{e['min']:g}, "
                 f"{e['max']:g}] {e.get('unit', '')} ({e['role']})")
    L.append("reach (hu, each component alone): " + json.dumps(b["reach"]))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vrm", help="a .vrm avatar to describe as a body")
    ap.add_argument("--out", help="body file to write (rigs/<name>.body.json)")
    ap.add_argument("--body", help="a VRM body file to read")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--solve", default=None, metavar="JSON",
                    help="targets to solve, e.g. '{\"hand.R.y\": 0.4}'")
    a = ap.parse_args(argv)
    if a.vrm:
        blob = build(a.vrm)
        out = a.out or os.path.join("rigs", blob["rig"] + ".body.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=1)
        print(f"vrmbody: {out}")
        bm = VRMBody(blob)
    elif a.body:
        bm = fb.load(a.body)
    else:
        ap.error("--vrm or --body")
    if a.describe or not a.solve:
        print(describe(bm))
    if a.solve:
        res = bm.solve(json.loads(a.solve))
        print(json.dumps({k: res[k] for k in ("params", "report")},
                         indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
