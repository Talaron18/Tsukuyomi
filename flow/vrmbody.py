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
            90 forward, 130 across the body, -90 straight back), axial
            twist (+ outward, measured from the swing out of hanging) and
            the elbow's bend; the hand's roll, flexion and deviation (the
            wrist, turned only by composite targets); shoulders.up /
            shoulder.L.up;
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
            in Chromium). The hand turns at the wrist (WRIST, on the hand
            bone), no fingers; hand.L / hand.R are the palm centres.
  solver    VRMBody.solve(targets, current): declared variables and joint
            angles are set directly (clamped to the limits); a hand target
            is analytic two-bone IK with a pole: the elbow bend from the
            shoulder-palm distance (exact, by bisection on the actual bone
            vectors), then the rotation of the whole arm onto the target,
            searched over the swivel about the shoulder-palm axis: within
            the joint limits first, out of the body and the other arm next
            (collision), then the elbow nearest out-down-back (POLE) and
            near the current pose. A target given only in x and y gets a
            depth: where the palm is now, pushed in front of the torso or
            the face when it is in front of them, and pulled within reach.
            Out-of-reach targets stretch toward the target; the report says
            what the forward kinematics reach. perceive(values) is the
            reverse (forward kinematics), the self check.
  collision the trunk, neck and head as one signed distance, lofted
            through SLAB_H slabs of the skinned rest mesh (a rounded box
            each, hair left out), narrowed where the resting arms sank in;
            the arms as capsules (upper arm past the shoulder, forearm,
            two across the palm), radii from their own vertices. A hand
            target inside the body moves onto its surface; one no swivel
            reaches clear of the body (or, one hand moving, of the other
            arm) moves to the nearest clear target; either is reported
            (solve()'s "why", a MISS in flow.pose with the reason). Two
            hands moving together are solved each for the body, and while
            they overlap their targets part (sideways when the left hand
            is on the left), half each, until they touch. When the straight
            path in joint angles (what the pose layer plays) goes through
            the body, solve() gives a pose to pass through ("via"; flow.pose
            plays it on the way in and before the release). Measured on the
            samples: the old solver put hands together 0.08-0.09 hu into
            each other and the chest, crossed forearms 0.18-0.20 through
            each other, behind the back 0.30-0.32 into the hips; now <=
            PEN_TOL (0.02 hu, ~5 mm) in the pose and on its path.
  composites  "hands": "@together" (palm to palm before the chest),
            "@clasped" (before the waist), "@behind_back"; "hand.L" /
            "hand.R": "@<landmark>" (the palm on the body's front there),
            "@hip", "@back" -- hand positions plus a hand orientation the
            wrist turns to (composite(), composite_orient()); flow.pose's
            parser takes them for a body with composite_keys, the prompt
            lists them (prompt_notes). A Live2D body has none.
  gaze      gaze 1 is the look-at range map's input maximum: the eyes at
            the avatar's own extremes (Seed-san 10 deg, three-vrm-girl 8-12),
            as Live2D's EyeBall 1 is the drawn eye's edge; capped at
            EYE_MAX_DEG for a bone look-at whose output range is wider. The
            camera sits at the eyes' height, so gaze 0 with the head at rest
            looks into it.

Joint limits (LIMITS) are human active ranges cut to what a talking avatar
does and its skin survives: neck rotation 80 deg, flexion 50 / extension 60,
lateral flexion 45 (AAOS norms, Greene & Heckman 1994) -> head yaw +-50,
pitch +-35 (Live2D's ParamAngleY is +-30), tilt +-30; thoracolumbar rotation
30, lateral 25 (same source) for a standing body with the hips still ->
torso yaw +-25, pitch +-15, tilt +-12; shoulder flexion / abduction 180,
horizontal adduction 135, horizontal abduction 45, extension 60 ->
elevation 0..170, plane -90..130 with the extension (the arm behind the
frontal plane) held to 60 deg hanging, 45 level, 30 overhead (EXT_*);
rotation 70 internal (at 90 abduction; a hand behind the back takes ~90
with the arm down) / 90 external -> twist -100..90; elbow flexion 150 ->
bend 0..145; wrist as WRIST's comment.

Format (tsukuyomi.body/2, kind "vrm"): the keys pose.py reads (frame,
landmarks [head, chest, hips, shoulder.L/R, hand.L/R as [x, y]; shoulder.y
a number], regions, params {min, max, default, band, role, declares},
declared, meaning, features, variants [one, "rest": rest values and
presence per feature; no sweeps stored, VRMBody computes them]) plus
"depth" {landmark: z hu}, "limits_source", "reach" {hand.L/R: {x, y, z}
ranges}, and "vrm" {file, version, faces, license, bones {name: {rest
[x, y, z] metres in the 1.0 frame, parent}}, palm {L, R}, torso_box,
head_box, face_map, visemes, blink, look_at (with the output scales),
breath, camera, springs, rev (BODY_REV), eyes [x, y, z] hu, collision
{slabs, arms, rest_fit}}.

Usage:
    python -m flow.vrmbody --vrm assets/vrm/VRM1_Constraint_Twist_Sample.vrm \\
        --out rigs/seed.body.json
    python -m flow.vrmbody --body rigs/seed.body.json --describe
    python -m flow.vrmbody --body rigs/seed.body.json \\
        --solve '{"hand.R.y": "@head.y", "hand.R.x": -0.9}'
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
# out and forward with the elbow soft -- not the file's T-pose (twist -20:
# the rest pose of the Euler twist before it, the palms toward the thighs).
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
    ("upper_arm.L.plane", "deg", -90.0, 130.0, 20.0),
    ("upper_arm.L.twist", "deg", -100.0, 90.0, -20.0),
    ("forearm.L.bend", "deg", 0.0, 145.0, 14.0),
    ("upper_arm.R.elev", "deg", 0.0, 170.0, 8.0),
    ("upper_arm.R.plane", "deg", -90.0, 130.0, 20.0),
    ("upper_arm.R.twist", "deg", -100.0, 90.0, -20.0),
    ("forearm.R.bend", "deg", 0.0, 145.0, 14.0),
    # The wrist and the forearm's roll, both on the hand bone (VRM has no
    # forearm twist bone in the humanoid; three-vrm's twist constraints,
    # where an avatar has them, spread it). From the T-pose's palm down:
    # roll + supinates (the palm turns forward, then up): AAOS pronation
    # 80 / supination 80 about thumb-up, palm-down being ~80 pronated, so
    # -20..160; flex + toward the palm 80, - extension 70; dev + radial
    # (toward the thumb) 20, - ulnar 30. Only composite targets (the hands
    # placed together, behind the back, on the chest) turn them.
    ("hand.L.roll", "deg", -20.0, 160.0, 0.0),
    ("hand.L.flex", "deg", -70.0, 80.0, 0.0),
    ("hand.L.dev", "deg", -30.0, 20.0, 0.0),
    ("hand.R.roll", "deg", -20.0, 160.0, 0.0),
    ("hand.R.flex", "deg", -70.0, 80.0, 0.0),
    ("hand.R.dev", "deg", -30.0, 20.0, 0.0),
)
JOINTS = tuple(n for n, *_ in LIMITS)
ARM = {s: (f"upper_arm.{s}.elev", f"upper_arm.{s}.plane",
           f"upper_arm.{s}.twist", f"forearm.{s}.bend") for s in "LR"}
WRIST = {s: (f"hand.{s}.roll", f"hand.{s}.flex", f"hand.{s}.dev")
         for s in "LR"}
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
# Shoulder extension (the arm behind the frontal plane) is limited by how
# high the arm is: 60 deg hanging (AAOS extension 60), 45 at shoulder
# height (horizontal abduction 45), less overhead. plane -90 is straight
# back, so behind the back (elevation ~30-50, plane -40..-80) is inside.
EXT_HANGING, EXT_LEVEL, EXT_OVERHEAD = 60.0, 45.0, 30.0
# The body file's revision: 2 adds the collision model ("collision"),
# the eyes and the look-at output scales. flow.vrm.body_for rebuilds an
# older file it built itself.
BODY_REV = 2
# Collision model (module docstring, "collision"): the trunk, neck and
# head as a stack of SLAB_H-high slabs, each a rounded box measured from
# the skinned rest mesh; the arm segments as capsules.
SLAB_H = 0.2            # hu
TRUNK_BELOW_HIPS = 0.8  # hu of hips and thighs below the hips joint
# A segment may sink this far into a surface (hu, ~5 mm at 1 hu = 0.24
# m): skin and cloth give, and the measured surfaces are percentiles.
PEN_TOL = 0.02
HAND_GAP = 0.01         # hu between two hands that touch
# The upper arm's first 40% is the shoulder: it starts inside the trunk.
UA_SKIP = 0.4
W_PEN = 1e3             # per hu of penetration: limits > penetration > pole
W_WRIST = 0.05          # per deg of wrist past its limits (composites)
ESCAPE_STEP, ESCAPE_MAX = 0.04, 1.2   # hu: moving a target out of the body
# Eye rotation at gaze 1 is the avatar's own look-at range (its author's
# extremes, as Live2D's EyeBall +-1 is the drawn eye's), capped here for a
# bone look-at: the samples' ranges are 8-12 deg (three-vrm-girl, a VRoid
# export: in 8, out 12, up / down 10; Seed-san 10), and an export with an
# output range of 90 would turn gaze 1 into eyes rolled out of sight.
EYE_MAX_DEG = 15.0
# Composite targets: two-hand poses by name, and one hand onto a place.
MEET = "your hands meet here: they stop where they touch"
TWO_HANDS = ("together", "clasped", "behind_back")
ONE_HAND = ("hip", "back")
ALIASES = {"hands_together": "together", "behind": "behind_back",
           "behind_the_back": "behind_back", "chest.front": "chest",
           "folded": "clasped"}
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
    down): the arm hung down, turned about its own axis by `twist` (+
    external: with the elbow bent, a forward forearm turns out), then
    swung the shortest way from hanging to elevation `elev` in the plane
    `plane` (0 the frontal plane, out to the side; 90 the sagittal plane,
    forward; -90 straight back). Twist is measured from that swing (the
    swing-twist split of the shoulder; Codman's paradox makes any Euler
    twist's zero move with the plane), so the same twist means the same
    axial rotation wherever the arm points and its limits are the joint's
    own; it made behind-the-back reachable (an Euler twist needed -110 deg
    there, its zero turning with the plane)."""
    sg = SIGMA_SIDE[side]
    e, p = math.radians(elev), math.radians(plane)
    a = np.array([sg * math.sin(e) * math.cos(p), -math.cos(e),
                  math.sin(e) * math.sin(p)])
    return between([0.0, -1.0, 0.0], a) @ rz(-sg * 90.0) @ rx(-twist)


def wrist_rotation(side, roll, flex, dev):
    """The hand's rotation in its forearm's aligned frame (forearm along
    sigma*X, the T-pose palm facing -Y, the thumb +Z): roll about the
    forearm, then flexion and deviation at the wrist (WRIST, LIMITS)."""
    sg = SIGMA_SIDE[side]
    return rx(-roll) @ rz(-sg * flex) @ ry(-sg * dev)


def wrist_angles(side, R):
    """(roll, flex, dev) of wrist_rotation's R (its inverse)."""
    sg = SIGMA_SIDE[side]
    b = math.asin(float(np.clip(-R[0, 1], -1, 1)))
    c = math.atan2(R[0, 2], R[0, 0])
    a = math.atan2(R[2, 1], R[1, 1])
    return -math.degrees(a), -sg * math.degrees(b), -sg * math.degrees(c)


def _wrist_rotation_batch(side, roll, flex, dev):
    """wrist_rotation over arrays (n,): (n, 3, 3)."""
    sg = SIGMA_SIDE[side]

    def R(axis, d):
        c, s_ = np.cos(np.radians(d)), np.sin(np.radians(d))
        M = np.zeros((len(d), 3, 3))
        i, j = [(1, 2), (2, 0), (0, 1)][axis]
        M[:, axis, axis] = 1.0
        M[:, i, i], M[:, j, j] = c, c
        M[:, i, j], M[:, j, i] = -s_, s_
        return M
    return R(0, -np.asarray(roll)) @ R(2, -sg * np.asarray(flex)) @ \
        R(1, -sg * np.asarray(dev))


def _wrist_angles_batch(side, R):
    """wrist_angles over (n, 3, 3): [roll, flex, dev] arrays."""
    sg = SIGMA_SIDE[side]
    b = np.arcsin(np.clip(-R[:, 0, 1], -1, 1))
    c = np.arctan2(R[:, 0, 2], R[:, 0, 0])
    a = np.arctan2(R[:, 2, 1], R[:, 1, 1])
    return [-np.degrees(a), -sg * np.degrees(b), -sg * np.degrees(c)]


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
    M = (between([0.0, -1.0, 0.0], a) @ rz(-sg * 90.0)).T @ A
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


def _closest(a, b, c, d):
    """The closest points of segments a-b and c-d."""
    u, v, w = b - a, d - c, a - c
    A, B, C = float(u @ u), float(u @ v), float(v @ v)
    D, E = float(u @ w), float(v @ w)
    den = A * C - B * B
    s = np.clip((B * E - C * D) / den, 0.0, 1.0) if den > 1e-12 else 0.0
    t = (B * s + E) / C if C > 1e-12 else 0.0
    if t < 0.0 or t > 1.0:
        t = float(np.clip(t, 0.0, 1.0))
        s = float(np.clip((B * t - D) / A, 0.0, 1.0)) if A > 1e-12 else 0.0
    return a + s * u, c + t * v


def _closest_batch(a, b, c, d):
    """_closest over rows: a, b, c, d (n, 3)."""
    u, v, w = b - a, d - c, a - c
    A = np.einsum("ij,ij->i", u, u)
    B = np.einsum("ij,ij->i", u, v)
    C = np.einsum("ij,ij->i", v, v)
    D = np.einsum("ij,ij->i", u, w)
    E = np.einsum("ij,ij->i", v, w)
    den = A * C - B * B
    s = np.where(den > 1e-12, np.clip((B * E - C * D)
                                      / np.where(den > 1e-12, den, 1), 0, 1),
                 0.0)
    t = np.where(C > 1e-12, (B * s + E) / np.where(C > 1e-12, C, 1), 0.0)
    tc = np.clip(t, 0.0, 1.0)
    s = np.where(t != tc, np.where(A > 1e-12, np.clip(
        (B * tc - D) / np.where(A > 1e-12, A, 1), 0, 1), 0.0), s)
    return a + s[:, None] * u, c + tc[:, None] * v


def _arm_angles_batch(side, A, plane0=0.0):
    """arm_angles over n rotations A (n, 3, 3): (elev, plane, twist)."""
    sg = SIGMA_SIDE[side]
    a = A[:, :, 0] * sg
    elev = np.degrees(np.arccos(np.clip(-a[:, 1], -1, 1)))
    h = np.hypot(a[:, 0], a[:, 2])
    plane = np.where(h < 1e-6, plane0,
                     np.degrees(np.arctan2(a[:, 2], sg * a[:, 0])))
    # The swing from hanging: about (-a_z, 0, a_x) by the elevation.
    sn = np.sin(np.radians(elev))
    k = np.stack([-a[:, 2], np.zeros(len(a)), a[:, 0]], 1) / \
        np.maximum(sn, 1e-12)[:, None]
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    cs = np.cos(np.radians(elev))
    Sw = np.eye(3)[None] + sn[:, None, None] * K + \
        (1 - cs)[:, None, None] * (K @ K)
    Sw = np.where((sn < 1e-9)[:, None, None], np.eye(3)[None], Sw)
    M = np.swapaxes(Sw @ rz(-sg * 90.0)[None], 1, 2) @ A
    twist = -np.degrees(np.arctan2(M[:, 2, 1], M[:, 1, 1]))
    return elev, plane, twist


def _seg_dist(p, a, b):
    """Distances of points p (n, 3) to the segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def collision_mesh(g, info, hb, X, D, body, pos, log=print) -> dict:
    """The collision model's measurements from the skinned rest mesh (body
    frame, hu; module docstring "collision"): {"slabs": [[y0, y1, hx, c,
    hz]] -- each SLAB_H band of the trunk, neck and head vertices as a
    rounded box |x| <= hx, |z - c| <= hz with its short side round --
    "arms": {L/R: {"upper_r", "fore_r", "hand_r", "hand": [[a, b], [a,
    b]]}}: capsule radii (the median distance of the bone's vertices from
    it) and the hand as two capsules side by side across the palm (rest
    positions, metres, VRM 1.0 frame), so two hands side by side keep a
    hand's width apart and palm to palm a hand's thickness}. Hair is left
    out (vertices on a joint named for hair, or in a hair mesh): it swings
    and is pushed, it does not stop a hand."""
    nodes = g.json.get("nodes", [])
    par = g.parents()
    node2h = {int(n): b for b, n in hb.items()}
    owner = {}
    for d in np.unique(D):
        n, hairy = int(d), False
        while n >= 0 and n not in node2h:
            if re.search(r"hair", str(nodes[n].get("name", "")), re.I):
                hairy = True
            n = par.get(n, -1)
        owner[int(d)] = (node2h.get(n), hairy)
    own = np.array([owner[int(d)][0] or "" for d in D])
    hairy = np.array([owner[int(d)][1] for d in D])
    B = body(X)
    side = {"L": "left", "R": "right"}
    trunk = ("hips", "spine", "chest", "upperChest", "neck", "head",
             "leftUpperLeg", "rightUpperLeg")
    T = B[np.isin(own, trunk) & ~hairy]
    hips_y = float(body(pos["hips"])[1])
    y_lo, y_hi = hips_y - TRUNK_BELOW_HIPS, float(T[:, 1].max())
    slabs = []
    y = y_lo
    while y < y_hi - 1e-6:
        q = T[(T[:, 1] >= y) & (T[:, 1] < y + SLAB_H)]
        if len(q) >= 10:
            z0, z1 = np.percentile(q[:, 2], [2, 98])
            slabs.append([round(y, 4), round(min(y + SLAB_H, y_hi), 4),
                          round(float(np.percentile(np.abs(q[:, 0]), 98)), 4),
                          round(float(0.5 * (z0 + z1)), 4),
                          round(float(0.5 * (z1 - z0)), 4)])
        y += SLAB_H
    arms = {}
    for s in "LR":
        ua, lo, hd = (f"{side[s]}UpperArm", f"{side[s]}LowerArm",
                      f"{side[s]}Hand")
        a, e, w = body(pos[ua]), body(pos[lo]), body(pos[hd])

        def radius(bone, p0, p1):
            v = B[(own == bone) & ~hairy]
            if len(v) < 10:
                return 0.25 * float(np.linalg.norm(p1 - p0))
            return float(np.median(_seg_dist(v, p0, p1)))
        upper_r, fore_r = radius(ua, a, e), radius(lo, e, w)
        # The hand: the hand bone's and the fingers' vertices (not the
        # thumb's, which sticks out of the palm's plane) in a frame along
        # the hand (u), across the palm (wd) and through it (t).
        fingers = [b for b in hb if b.startswith(side[s]) and any(
            f in b for f in ("Index", "Middle", "Ring", "Little"))]
        v = B[np.isin(own, [hd] + fingers) & ~hairy]
        kn = hb.get(f"{side[s]}MiddleProximal")
        u = body(_flip(g.node_position(kn), info["faces"])) - w \
            if kn is not None else w - e
        u = u / np.linalg.norm(u)
        t = np.array([0.0, 1.0, 0.0]) - u[1] * u
        t = t / np.linalg.norm(t)
        wd = np.cross(u, t)
        if len(v) >= 10:
            r_ = v - w
            al, th, ac = r_ @ u, r_ @ t, r_ @ wd
            length = float(np.percentile(al, 99))
            t0, t1 = np.percentile(th, [5, 95])
            c0, c1 = np.percentile(ac, [3, 97])
        else:
            length = 0.6 * float(np.linalg.norm(w - e))
            t0, t1, c0, c1 = -0.06, 0.06, -0.12, 0.12
        hand_r = max(0.5 * float(t1 - t0), 0.03)
        off = max(0.5 * float(c1 - c0) - hand_r, 0.0)
        tc, cc = 0.5 * float(t0 + t1), 0.5 * float(c0 + c1)
        segs = []
        for sg in (-1.0, 1.0):
            base = w + tc * t + (cc + sg * off) * wd
            pa, pb = base + hand_r * u, base + max(length - hand_r,
                                                   hand_r) * u
            segs.append([[round(float(x), 6) for x in body.inv(p)]
                         for p in (pa, pb)])
        arms[s] = {"upper_r": round(upper_r, 4), "fore_r": round(fore_r, 4),
                   "hand_r": round(hand_r, 4), "hand": segs}
    if len(slabs) < 3:
        log("vrmbody: too few trunk vertices for a collision model; none "
            "(hands may pass through the body)")
        return None
    log(f"vrmbody: collision model: {len(slabs)} trunk/head slabs, arm "
        f"radii {arms['L']['upper_r']:.3f}/{arms['L']['fore_r']:.3f} hu, "
        f"hand {2 * arms['L']['hand_r']:.3f} hu thick")
    return {"slabs": slabs, "arms": arms}


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
    body.inv = lambda b: chin + hu * np.asarray(b, float)
    torso_ids = {hb[b] for b in ("spine", "chest", "upperChest") if b in hb}
    tv = body(X[np.isin(D, list(torso_ids)) & ~hair])
    if len(tv) == 0:
        # No vertex on the torso bones: everything between hips and chin.
        allb = body(X[~hair])
        tv = allb[(allb[:, 1] < 0) & (allb[:, 1] > body(pos["hips"])[1])]
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
    # The eyes: between the eye bones (the look-at origin), else the head
    # bone plus the look-at offset.
    eyes = [body(_flip(g.node_position(hb[b]), faces)) for b in
            ("leftEye", "rightEye") if b in hb]
    eye_mid = (np.mean(eyes, axis=0) if eyes else
               body(hpos + _flip(look.get("offset", [0, 0.06, 0]), faces)))
    coll = collision_mesh(g, info, hb, X, D, body, pos, log=log)
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
                "expressions": sorted(ex), "morphs": len(morphs),
                "rev": BODY_REV,
                "eyes": [round(float(x), 4) for x in eye_mid],
                "collision": coll},
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
    # The trunk slabs are fitted to the rest pose first: a slab the resting
    # arms sink into is measured too wide there (a loose shirt's sides and
    # sleeves are skinned to the chest), so it is narrowed until they clear.
    bm = VRMBody(blob)
    if coll:
        coll["rest_fit"] = bm.fit_rest()
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
                       "body, -90 straight back",
    "upper_arm.twist": "+ turns the upper arm outward",
    "forearm.bend": "0 straight, 145 fully bent",
    "hand.roll": "+ turns the palm up",
    "hand.flex": "+ bends the hand toward the palm",
    "hand.dev": "+ bends the hand toward the thumb",
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
    # The camera at the eyes' height, looking down at the framing's
    # centre: the viewer is face to face, as a Live2D rig is drawn, so a
    # gaze straight ahead (gaze 0, head at rest) meets the camera. At the
    # framing's height it sat 0.7 hu below the eyes on both samples: the
    # rest gaze passed ~4 deg over the viewer and every face was seen from
    # below.
    eye = (np.array(blob["vrm"]["eyes"], float) if "eyes" in blob["vrm"]
           else np.array([0.0, L["head"][1], 0.0]))
    pos = target + np.array([0.0, 0.0, dist])
    pos[1] = o[1] + eye[1] * hu
    return {"fov": CAMERA_FOV, "target": [round(float(x), 4) for x in target],
            "position": [round(float(x), 4) for x in pos]}


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
        # The collision model (absent from a rev-1 body file: no
        # collisions then, as before it).
        coll = self.vrm.get("collision")
        self.coll = coll
        if coll:
            self._slabs = np.array(coll["slabs"], float).reshape(-1, 5)
            self._arm_r = {s: (coll["arms"][s]["upper_r"],
                               coll["arms"][s]["fore_r"],
                               coll["arms"][s]["hand_r"]) for s in "LR"}
            self._hand_rest = {s: np.array(coll["arms"][s]["hand"], float)
                               for s in "LR"}
        # flow.pose.arm_params reads this cache: the joints that move an
        # arm (the torso moves the hands too, but a torso pose is not an
        # arm pose).
        self._arm_params = {p for s in "LR" for p in ARM[s] + WRIST[s]} | {
            "shoulders.up", "shoulder.L.up", "shoulder.R.up"}
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
            wr = [float(v.get(k, 0.0)) for k in WRIST[s]]
            if any(wr):
                out[f"{SIDE[s]}Hand"] = a_f.T @ wrist_rotation(s, *wr) @ a_f
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

    # -------------------------------------------------------- collision

    def body_sdf(self, P) -> np.ndarray:
        """Signed distance (hu, + outside) of points P (n, 3; body frame,
        hu) to the trunk, neck and head: lofted through the slabs -- at a
        point's height the slab shape is interpolated between the slab
        centres (a rounded box in x-z, its short side a half circle), and
        the stack ends flat at its lowest and highest slab. (A union of
        separate slabs has their shared faces inside the body: a point in
        the chest was "nearest" a seam, and moving out of it went down.)"""
        P = np.atleast_2d(np.asarray(P, float))
        S = self._slabs
        yc = 0.5 * (S[:, 0] + S[:, 1])
        y = P[:, 1]
        hx, c, hz = (np.interp(y, yc, S[:, i]) for i in (2, 3, 4))
        r = np.minimum(hx, hz)
        qx = np.abs(P[:, 0]) - (hx - r)
        qz = np.abs(P[:, 2] - c) - (hz - r)
        d2 = np.hypot(np.maximum(qx, 0), np.maximum(qz, 0)) + \
            np.minimum(np.maximum(qx, qz), 0) - r
        dy = np.maximum(S[0, 0] - y, y - S[-1, 1])
        return np.minimum(np.maximum(d2, dy), 0) + \
            np.hypot(np.maximum(d2, 0), np.maximum(dy, 0))

    def arm_caps(self, s, values=None, W=None, P=None):
        """Arm s's capsules at a pose (parameter values, or its forward
        kinematics): [(name, a, b, r)] in the body frame, hu -- the upper
        arm past UA_SKIP, the forearm, the hand's two."""
        if W is None:
            W, P = self.fk(values)
        ub, lb, hb = (f"{SIDE[s]}UpperArm", f"{SIDE[s]}LowerArm",
                      f"{SIDE[s]}Hand")
        ru, rf, rh = self._arm_r[s]
        sb, eb, wb = (self.to_body(P[b]) for b in (ub, lb, hb))
        out = [("upper", sb + UA_SKIP * (eb - sb), eb, ru),
               ("fore", eb, wb, rf)]
        for i, (pa, pb) in enumerate(self._hand_rest[s]):
            out.append((f"hand{i}",
                        self.to_body(P[hb] + W[hb] @ (pa - self.rest[hb])),
                        self.to_body(P[hb] + W[hb] @ (pb - self.rest[hb])),
                        rh))
        return out

    def _body_pen(self, caps, n=9):
        """(deepest penetration of these capsules into the trunk / head,
        hu, >= 0; the capsule)."""
        u = np.linspace(0.0, 1.0, n)[:, None]
        pts = np.concatenate([a + u * (b - a) for _n, a, b, _r in caps])
        r = np.repeat([c[3] for c in caps], n)
        pen = r - self.body_sdf(pts)
        k = int(np.argmax(pen))
        return max(float(pen[k]), 0.0), caps[k // n][0]

    @staticmethod
    def _pair_pen(c1, c2, gap=HAND_GAP):
        """(deepest overlap between two arms' capsules, hu, >= 0; the
        unit direction from arm 2's to arm 1's closest points there). The
        upper arms are left out of each other's way (they cannot meet)."""
        best, d_best = 0.0, None
        for n1, a, b, r1 in c1:
            for n2, c, d, r2 in c2:
                if n1 == "upper" and n2 == "upper":
                    continue
                p, q = _closest(a, b, c, d)
                dist = float(np.linalg.norm(p - q))
                pen = r1 + r2 + gap - dist
                if pen > best:
                    best = pen
                    d_best = (p - q) / dist if dist > 1e-9 else \
                        np.array([1.0, 0.0, 0.0])
        return best, d_best

    def penetration(self, values) -> dict:
        """{"body.L", "body.R": deepest arm penetration into the trunk /
        head (hu), "hands": deepest overlap between the two arms}."""
        if not self.coll:
            return {"body.L": 0.0, "body.R": 0.0, "hands": 0.0}
        W, P = self.fk(values)
        caps = {s: self.arm_caps(s, values, W, P) for s in "LR"}
        return {"body.L": self._body_pen(caps["L"])[0],
                "body.R": self._body_pen(caps["R"])[0],
                "hands": self._pair_pen(caps["L"], caps["R"], 0.0)[0]}

    def fit_rest(self) -> list:
        """Narrow the slabs the resting arms sink into until they clear
        (build(): a loose shirt's sides are skinned to the chest); the
        amount taken off each slab's half width, hu."""
        if not self.coll:
            return []
        rest = self.default_values()
        W, P = self.fk(rest)
        caps = [c for s in "LR" for c in self.arm_caps(s, rest, W, P)]
        u = np.linspace(0.0, 1.0, 9)[:, None]
        pts = np.concatenate([a + u * (b - a) for _n, a, b, _r in caps])
        r = np.repeat([c[3] for c in caps], 9)
        yc = 0.5 * (self._slabs[:, 0] + self._slabs[:, 1])
        cut = np.zeros(len(self._slabs))
        for _ in range(400):
            pen = r - self.body_sdf(pts)
            k = int(np.argmax(pen))
            if pen[k] <= 0.0:
                break
            # The slabs whose shapes make that height (interpolated).
            j = int(np.clip(np.searchsorted(yc, pts[k, 1]), 1,
                            len(yc) - 1))
            for i in {j - 1, j}:
                step = min(float(pen[k]) + 0.005, self._slabs[i, 2] - 0.05)
                if step > 0:
                    self._slabs[i, 2] -= step
                    cut[i] += step
        for i in range(len(self._slabs)):
            self.coll["slabs"][i][2] = round(float(self._slabs[i, 2]), 4)
        return [round(float(x), 4) for x in cut]

    # ------------------------------------------------------------ solve

    # Composite targets (module docstring, "composites"): keys a marker
    # may use beside the variables, and the names they take.
    composite_keys = ("hands", "hand.L", "hand.R")

    @property
    def composite_landmarks(self):
        return (tuple(TWO_HANDS) + tuple(ONE_HAND) + tuple(ALIASES)
                if self.coll else ())

    def prompt_notes(self) -> list:
        """The body-mode prompt's lines on two-hand poses and touching the
        body (flow.pose.describe_body calls this)."""
        if not self.coll:
            return []
        return [
            "Your hands never pass through your body or each other: a "
            "target that would is stopped where they touch. Two-hand "
            "poses and touching your body have their own targets:",
            '  "hands": "@together" palms together in front of your chest; '
            '"@clasped" hands held together in front of your waist; '
            '"@behind_back" both hands behind your lower back.',
            '  "hand.L" / "hand.R": "@chest" (or another landmark: that '
            'hand flat on your body there, e.g. "hand.R": "@shoulder.L" on '
            'your other shoulder, "+0.2" higher), "@hip" (on your hip, '
            'elbow out), "@back" (behind your back).',
            '  For example [pose {"hands": "@together", "head.tilt": 6}] or '
            '[pose {"hand.R": "@chest", "hold": 1.5}]. Two hands at the same '
            "x and y meet side by side.",
        ]

    def _surface_z(self, x, y, back=False):
        """The trunk's / head's front (or back) surface depth at (x, y),
        hu; None where (x, y) is off the body."""
        zs = np.linspace(2.0, -2.0, 401)
        if back:
            zs = zs[::-1]
        sd = self.body_sdf(np.c_[np.full(len(zs), x), np.full(len(zs), y),
                                 zs])
        k = np.nonzero(sd <= 0)[0]
        return None if not len(k) else float(zs[k[0]])

    def _surface_x(self, y, z, s):
        """The body's side surface toward side s at height y, depth z."""
        xs = SIGMA_SIDE[s] * np.linspace(2.0, 0.0, 201)
        sd = self.body_sdf(np.c_[xs, np.full(len(xs), y),
                                 np.full(len(xs), z)])
        k = np.nonzero(sd <= 0)[0]
        return None if not len(k) else float(xs[k[0]])

    def hand_orient(self, s, fingers, palm):
        """The hand's world rotation (from its rest) that points its
        fingers along `fingers` with its palm facing `palm` (body-frame
        directions; palm is made square to fingers)."""
        hb = f"{SIDE[s]}Hand"
        f0 = self.palm_rest[s] - self.rest[hb]
        f0 = f0 / np.linalg.norm(f0)
        n0 = np.array([0.0, -1.0, 0.0]) - (f0 @ [0.0, -1.0, 0.0]) * f0
        n0 = n0 / np.linalg.norm(n0)
        f1 = np.asarray(fingers, float)
        f1 = f1 / np.linalg.norm(f1)
        n1 = np.asarray(palm, float) - (np.asarray(palm, float) @ f1) * f1
        n1 = n1 / np.linalg.norm(n1)
        F0 = np.stack([f0, n0, np.cross(f0, n0)], 1)
        F1 = np.stack([f1, n1, np.cross(f1, n1)], 1)
        return F1 @ F0.T

    def composite_orient(self, key, ref) -> dict:
        """{side: the hand's world rotation} a composite target turns the
        hands to (composite())."""
        name = ALIASES.get(str(ref).strip().lstrip("@").split("+")[0]
                           .split("-")[0], str(ref).strip().lstrip("@")
                           .split("+")[0].split("-")[0])
        out = {}
        sides = "LR" if key == "hands" else key[-1]
        for s in sides:
            sg = SIGMA_SIDE[s]
            if name in ("together", "clasped", "hip"):
                # Palm to palm, fingers up and a little forward (together)
                # or forward and down (clasped); on the hip, the palm on it.
                f = {"together": (0.0, 1.0, 0.35), "clasped": (0.0, -0.4, 1.0),
                     "hip": (0.0, -0.7, 0.7)}[name]
                out[s] = self.hand_orient(s, f, (-sg, 0.0, 0.0))
            elif name in ("behind_back", "back"):
                # Behind the lower back: palms out, fingers in and down.
                out[s] = self.hand_orient(s, (-sg, -0.6, 0.0),
                                          (0.0, 0.0, -1.0))
            else:
                # On the body: the palm against it, fingers up and in.
                out[s] = self.hand_orient(s, (-sg * 0.5, 0.85, 0.0),
                                          (0.0, 0.0, -1.0))
        return out

    def composite(self, key, ref) -> dict:
        """{hand.S.x/y/z: hu} for a composite target: "hands": "@together"
        (palms together in front of the chest), "@clasped" (folded in
        front of the waist), "@behind_back" (behind the lower back); or
        "hand.L" / "hand.R": "@<landmark>" (the palm on the body's front
        there; "+0.2" offsets it up), "@hip" (on its own hip, elbow out),
        "@back" (behind the lower back)."""
        m = re.match(r"@([\w.]+?)([+-][\d.]+)?$", str(ref).strip())
        if not m or not self.coll:
            raise ValueError(f"bad composite {key}: {ref!r}")
        name, off = m.group(1), float(m.group(2) or 0)
        name = ALIASES.get(name, name)
        L = self.landmarks
        hr = max(self._arm_r[s][2] for s in "LR")
        chest_y, hips_y = L["chest"][1], L["hips"][1]
        out = {}
        if key == "hands":
            if name not in TWO_HANDS:
                raise ValueError(f"hands: {ref!r}; have "
                                 f"{sorted(TWO_HANDS)}")
            if name == "behind_back":
                for s in "LR":
                    out.update(self.composite(f"hand.{s}", "@back"))
                return out
            y = (chest_y if name == "together" else
                 hips_y + 0.35 * (chest_y - hips_y)) + off
            z = (self._surface_z(0.0, y) or 0.0) + (
                0.45 if name == "together" else 0.3)
            for s in "LR":
                sg = SIGMA_SIDE[s]
                out.update({f"hand.{s}.x": sg * (hr + 0.5 * HAND_GAP),
                            f"hand.{s}.y": y, f"hand.{s}.z": z})
            return out
        s = key[-1]
        sg = SIGMA_SIDE[s]
        if name == "back":
            y = hips_y + 0.25 + off
            # The palm half a hand from the spine: fingers turned in (the
            # composite's wrist), the two hands' fingertips meet there.
            x = sg * float(np.linalg.norm(self.palm_rest[s]
                                          - self.rest[f"{SIDE[s]}Hand"])) \
                / self.hu
            z = (self._surface_z(x, y, back=True) or -0.6) - hr - 0.15
        elif name == "hip":
            y = hips_y + 0.2 + off
            c = float(np.interp(y, self._slabs[:, 0], self._slabs[:, 3]))
            x = (self._surface_x(y, c, s) or sg * 0.5) + sg * (hr + 0.04)
            z = c
        else:
            ref2 = L.get(name)
            if not isinstance(ref2, list):
                raise ValueError(f"{key}: no landmark {name!r} (have "
                                 f"{sorted(k for k, v in L.items() if isinstance(v, list))}, "
                                 f"{', '.join(ONE_HAND)})")
            x, y = float(ref2[0]), float(ref2[1]) + off
            front = self._surface_z(x, y)
            # On the body: the palm on its surface. Off it (a hand's rest
            # place): the landmark's own depth, the palm where it is.
            z = (front + hr + 0.03) if front is not None else None
        out = {f"hand.{s}.x": x, f"hand.{s}.y": y}
        if z is not None:
            out[f"hand.{s}.z"] = z
        return out

    def resolve(self, targets: dict) -> dict:
        """flow.body's landmark references, plus depth: "@chest.z", or a
        bare "@chest" on a .z variable, reads the landmark's depth; and
        the composite targets ("hands": "@together", "hand.R": "@chest";
        composite()), expanded into hand variables (a variable given
        explicitly beside one wins)."""
        targets = dict(targets)
        expanded = {}
        for k in self.composite_keys:
            if k in targets:
                expanded.update(self.composite(k, targets.pop(k)))
        for k, v in expanded.items():
            targets.setdefault(k, v)
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
        """BodyModel.solve's contract (module docstring, "solver"), plus
        "why": {target: the reason a hand target was moved} for the
        targets the collision model kept the hand from (flow.pose's hints
        say it)."""
        orient = {}
        for k in self.composite_keys:
            if k in targets and self.coll:
                orient.update(self.composite_orient(k, targets[k]))
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
        want = {}
        for s in "LR":
            w = {q: targets[f"hand.{s}.{q}"] for q in "xyz"
                 if f"hand.{s}.{q}" in targets}
            if w and not any(p in targets for p in ARM[s]):
                want[s] = w
        orient = {s: R for s, R in orient.items() if s in want}
        why = {}
        if want:
            sol, why_s = self._solve_hands(want, vals, orient)
            vals.update(sol)
            for s in want:
                free += list(ARM[s]) + (list(WRIST[s]) if s in orient
                                        else [])
                if why_s.get(s):
                    for q in want[s]:
                        why[f"hand.{s}.{q}"] = why_s[s]
        via = self._via(cur, vals, list(want)) if want and self.coll \
            else None
        F = self.forward(vals)
        report = {}
        cost = 0.0
        met = []
        for t, y in targets.items():
            got = float(F[self.fidx[t]])
            unit = fb.feature_unit(t)
            err = got - float(y)
            cost += (err / fb.SIGMA[unit]) ** 2
            ok = abs(err) <= fb.TOL[unit]
            if t in why and why[t].startswith("!"):
                ok = False          # reached, but only through the body
            elif t in why and why[t].startswith(MEET) and \
                    t[:6] in {f"hand.{s}" for s in orient}:
                # A composite puts the hands together: where they meet is
                # the target (its numbers are where they would be, were
                # the hands as thin as the composite assumes).
                ok = True
                met.append(t)
            report[t] = {"target": float(y), "got": got, "error": err,
                         "unit": unit, "ok": ok}
        why = {t: w.lstrip("!") for t, w in why.items()
               if not report[t]["ok"]}
        out = {"params": {p: float(vals[p]) for p in free}, "free": free,
               "variant": 0, "variant_name": "rest", "parts": {},
               "cost": cost, "report": report, "missing": [], "why": why}
        if via:
            out["via"] = via
        if met:
            out["met"] = met
        return out

    def path_pen(self, a, b, n=11) -> float:
        """The deepest penetration (body or arms, hu) on the straight path
        in parameters from pose a to pose b -- the path flow.pose's
        PoseLayer plays (every parameter on the same min-jerk profile)."""
        worst = 0.0
        for u in np.linspace(0.0, 1.0, n):
            v = {k: a[k] + u * (b[k] - a[k]) for k in a}
            worst = max(worst, max(self.penetration(v).values()))
        return worst

    def _via(self, cur, vals, sides):
        """{param: value} of a pose to pass through on the way from cur to
        vals (and back) when the straight way goes through the body: the
        arm swung first and turned after, turned first, or out to the side
        between -- whichever path is clearest; None when the straight one
        is clear (or no detour is clearer). Behind the back from the rest
        went 0.33-0.41 hu deep through the hips straight (both samples)."""
        a = {k: float(cur[k]) for k in vals}
        b = {k: float(vals[k]) for k in vals}
        direct = self.path_pen(a, b)
        if direct <= 2 * PEN_TOL:
            return None
        keys = [k for s in sides for k in ARM[s] + WRIST[s]]
        cands = []
        # Detours: the arm raised to 35 deg in the plane it goes to, its
        # elbow straightened (20), as it was, or as it will be, turned as
        # it was or as it will be, the wrist likewise (a straight arm turns
        # about itself without sweeping the forearm through anything);
        # and the swing first / the turn first.
        for bend_of, twist_of, wrist_of in itertools.product(
                ("straight", "a", "b"), ("a", "b"), ("a", "b")):
            v = dict(b)
            for s in sides:
                e, p, w, bend = ARM[s]
                v[e] = max(35.0, min(a[e], b[e]))
                v[bend] = {"straight": 20.0, "a": a[bend],
                           "b": b[bend]}[bend_of]
                v[w] = (a if twist_of == "a" else b)[w]
                v.update({k: (a if wrist_of == "a" else b)[k]
                          for k in WRIST[s]})
            cands.append(v)
        for kind in ("swing", "turn"):
            v = dict(b)
            for s in sides:
                e, p, w, bend = ARM[s]
                src = ([w, bend] + list(WRIST[s]) if kind == "swing"
                       else [e, p])
                v.update({k: a[k] for k in src})
            cands.append(v)
        cands = [(max(self.path_pen(a, v, 6), self.path_pen(v, b, 6)), i, v)
                 for i, v in enumerate(cands)]
        pen, _kind, v = min(cands, key=lambda c: c[0])
        if pen >= direct - PEN_TOL:
            return None
        return {k: round(v[k], 4) for k in keys}

    def miss_hints(self, res: dict, max_variants: int = 8) -> dict:
        out = super().miss_hints(res, max_variants)
        for t, h in out.items():
            if res.get("why", {}).get(t):
                h["why"] = res["why"][t]
        return out

    def _solve_hands(self, want: dict, values: dict, orient=None):
        """Arm angles for the hand targets `want` ({side: {x, y, z}}) from
        the pose `values`, kept out of the body and out of each other
        (module docstring, "collision"). One hand: the other arm is an
        obstacle where it is. Two: each arm in turn, the other weighing on
        its swivel; hands that still overlap have their targets pushed
        apart along the line between them, half each, until they touch.
        Returns ({param: value}, {side: why or ""}); a why starting "!"
        marks a hand that only reaches its target through something."""
        vals = dict(values)
        orient = orient or {}
        # Two hands: each solved for the body alone (so the two come out
        # mirror images, not the second bent round the first), their
        # meeting then shared out below. One: the other arm is in the way.
        mode = "none" if len(want) == 2 else "escape"
        tgt = {s: self._hand_target(s, want[s], vals) for s in want}
        inside = {s: self.__dict__.pop("_inside_" + s, False) for s in want}
        moved = {s: np.zeros(3) for s in want}
        why = {s: "" for s in want}

        def run():
            for s in want:
                sol, info = self._arm_solve(s, tgt[s] + moved[s], vals, mode,
                                            orient.get(s))
                vals.update(sol)
                why[s] = info["why"]
        run()
        if len(want) == 2 and self.coll:
            for _ in range(10):
                W, P = self.fk(vals)
                cL, cR = (self.arm_caps(s, vals, W, P) for s in "LR")
                pen, d = self._pair_pen(cL, cR)
                if pen <= PEN_TOL:
                    break
                # Hands side by side (the left one on the left, as in
                # every two-hand pose but crossed arms) part sideways: the
                # closest points of two parallel hands are anywhere along
                # them, and parting up / down left one hand above the other.
                if tgt["L"][0] + moved["L"][0] > tgt["R"][0] + moved["R"][0]:
                    d = np.array([1.0, 0.0, 0.0])
                moved["L"] = moved["L"] + d * (0.5 * pen + 0.005)
                moved["R"] = moved["R"] - d * (0.5 * pen + 0.005)
                run()
            W, P = self.fk(vals)
            if self._pair_pen(*(self.arm_caps(s, vals, W, P)
                                for s in "LR"), 0.0)[0] > PEN_TOL:
                for s in want:
                    why[s] = why[s] or "!your arms would pass through each other"
            for s in want:
                if not why[s] and np.linalg.norm(moved[s]) > fb.TOL["hu"]:
                    why[s] = MEET
        for s in want:
            if inside[s] and not why[s].startswith("!"):
                why[s] = "that is inside your body: the hand stops on it"
        return {k: vals[k] for s in want for k in ARM[s] + WRIST[s]}, why

    def arm_ik(self, s, want: dict, values: dict) -> dict:
        """Joint angles of arm s putting its palm at `want` ({x, y, z},
        hu; missing components as module docstring), from the pose
        `values` (torso, shoulders, the other arm, the arm's current
        angles), out of the body and the other arm."""
        sol, _info = self._arm_solve(s, self._hand_target(s, want, values),
                                     values)
        return sol

    def _box_clear(self, x, y, z):
        """The smallest depth a palm at (x, y) may have: in front of the
        torso or the face where it is in front of them."""
        zmin = -math.inf
        for box in (self.vrm["torso_box"], self.vrm["head_box"]):
            if abs(x) < box["x"] + 0.15 and box["y0"] <= y <= box["y1"]:
                zmin = max(zmin, box["z"] + HAND_CLEAR)
        return zmin

    def _arm_frame(self, s, values, to_wrist=False):
        """(shoulder world position, parent world rotation, u1, c, reach
        m, current palm hu, W, P) of arm s at `values`; c ends at the
        wrist instead of the palm with to_wrist."""
        W, P = self.fk(values)
        ub, lb, hb = (f"{SIDE[s]}UpperArm", f"{SIDE[s]}LowerArm",
                      f"{SIDE[s]}Hand")
        a_u, a_f = self.align[s]
        # Upper arm and palm offset in the arm's aligned frame (upper arm
        # along sigma*X): u1, and c (elbow -> palm at bend 0, the wrist as
        # it is in `values`).
        u1 = a_u @ (self.rest[lb] - self.rest[ub])
        Rw = wrist_rotation(s, *(float(values.get(k, 0.0))
                                 for k in WRIST[s]))
        c = a_f @ (self.rest[hb] - self.rest[lb])
        if not to_wrist:
            c = c + Rw @ a_f @ (self.palm_rest[s] - self.rest[hb])
        return (P[ub], W[self.bones[ub]["parent"]], u1, c,
                float(np.linalg.norm(u1 + c)), self.palms(values, W, P)[s],
                W, P)

    def _hand_target(self, s, want, values):
        """The palm target (hu) of a hand target: a missing x or y is
        where the palm is; a missing depth is where the palm is, pushed in
        front of the torso or the face when it is in front of them (no
        hand inside the body) and pulled within reach."""
        sg = SIGMA_SIDE[s]
        S, _Rp, u1, c, reach, cur, _W, _P = self._arm_frame(s, values)
        tb = np.array([want.get("x", cur[0]), want.get("y", cur[1]),
                       want.get("z", np.nan)], float)
        Sb = self.to_body(S)
        if not math.isnan(tb[2]) and self.coll:
            # A depth given inside the body: the palm onto its surface,
            # the nearest place a hand can be.
            hr = self._arm_r[s][2] + 0.01
            for _ in range(40):
                d = float(self.body_sdf(tb[None])[0])
                if d >= hr:
                    break
                if d < 0:
                    self.__dict__["_inside_" + s] = True
                g = self._clear_grad(tb, None)
                if g is None:
                    g = np.array([0.0, 0.0, 1.0])
                tb = tb + g * max(hr - d, 0.01)
        elif math.isnan(tb[2]):
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
        return tb

    def _arm_solve(self, s, tb, values, other_mode="escape", orient=None):
        """Two-bone IK for arm s onto the palm target tb (hu): the elbow
        bend from the shoulder-palm distance (exact, by bisection on the
        actual bone vectors), then the rotation of the whole arm onto the
        target, searched over the swivel about the shoulder-palm axis --
        within the joint limits first, out of the body and the other arm
        next, then the elbow nearest out-down-back (POLE) and near the
        current pose. A target no swivel reaches without passing through
        the body (or, other_mode "escape", the other arm where it is) is
        moved out -- along the clearance field's gradient, or forward, or
        outward -- to the nearest one that is clear. other_mode "cost":
        the other arm only weighs on the swivel (two hands moving at once:
        _solve_hands shares their meeting out); "none": ignored. orient:
        the hand's world rotation wanted (a composite target): the arm
        then reaches the wrist that puts the palm at tb so turned, and the
        wrist (WRIST) turns the hand, the swivel also weighing what the
        wrist's limits allow. Returns ({param: value}, {"why", "pen",
        "target"})."""
        sg = SIGMA_SIDE[s]
        S, Rp, u1, c, _reach, _cur, W, P = self._arm_frame(
            s, values, to_wrist=orient is not None)
        hb = f"{SIDE[s]}Hand"
        a_f = self.align[s][1]
        palm_t = np.asarray(tb, float)
        if orient is not None:
            tb = np.asarray(tb, float) - orient @ (
                self.palm_rest[s] - self.rest[hb]) / self.hu
            wl = [(self.params[k]["min"], self.params[k]["max"])
                  for k in WRIST[s]]
        Lh_cur = a_f.T @ wrist_rotation(s, *(float(values.get(k, 0.0))
                                             for k in WRIST[s])) @ a_f
        other = None
        if self.coll and other_mode != "none":
            o = "R" if s == "L" else "L"
            other = self.arm_caps(o, values, W, P)
        e0, p0, w0, _b0 = (values[k] for k in ARM[s])
        lo_b, hi_b = (self.params[ARM[s][3]]["min"],
                      self.params[ARM[s][3]]["max"])
        psi = np.radians(-180.0 + 360.0 * np.arange(SWIVEL_STEPS)
                         / SWIVEL_STEPS)

        def length(b):
            return float(np.linalg.norm(u1 + ry(-sg * b) @ c))

        def search(tb):
            T = self.to_world(tb)
            t = Rp.T @ (T - S)
            d = float(np.linalg.norm(t))
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
            # Every swivel at once: A = rot(th, psi) A0 (Rodrigues).
            K = np.array([[0, -th[2], th[1]], [th[2], 0, -th[0]],
                          [-th[1], th[0], 0]])
            R = (np.eye(3)[None] + np.sin(psi)[:, None, None] * K[None]
                 + (1 - np.cos(psi))[:, None, None] * (K @ K)[None])
            A = R @ A0
            e, p, w = _arm_angles_batch(s, A, p0)
            viol = self._viol_batch(s, e, p, w)
            pole = Rp.T @ np.array([sg * POLE[0], POLE[1], POLE[2]])
            pole = pole - (pole @ th) * th
            pn = np.linalg.norm(pole)
            pole = pole / pn if pn > 1e-9 else pole
            elbow = A @ np.array([sg, 0.0, 0.0])
            elbow = elbow - (elbow @ th)[:, None] * th[None]
            en = np.linalg.norm(elbow, axis=1)
            pc = 1.0 - np.where(en > 1e-6, (elbow @ pole)
                                / np.maximum(en, 1e-6), 0.0)
            prior = ((e - e0) / 170) ** 2 + ((p - p0) / 160) ** 2 + \
                ((w - w0) / 140) ** 2
            Lh, wa, wviol = Lh_cur, None, 0.0
            if orient is not None:
                # The wrist that turns the hand to `orient` on each
                # candidate's forearm, within its limits.
                Wf = Rp[None] @ A @ (ry(-sg * bend) @ a_f)
                Rw = a_f[None] @ np.swapaxes(Wf, 1, 2) @ orient[None] \
                    @ a_f.T[None]
                wa = _wrist_angles_batch(s, Rw)
                wviol = sum(np.maximum(np.maximum(lo - x, x - hi), 0.0)
                            for x, (lo, hi) in zip(wa, wl))
                wa = [np.clip(x, lo, hi) for x, (lo, hi) in zip(wa, wl)]
                Lh = a_f.T[None] @ _wrist_rotation_batch(s, *wa) @ a_f[None]
            pen_b = np.zeros(len(psi))
            pen_a = np.zeros(len(psi))
            if self.coll:
                caps = self._caps_batch(s, Rp, S, A, bend, Lh)
                pen_b = self._body_pen_batch(caps)
                if other is not None:
                    pen_a = self._pair_pen_batch(caps, other)
            cost = 1e6 * viol + W_PEN * (np.maximum(pen_b - PEN_TOL, 0)
                                         + np.maximum(pen_a - PEN_TOL, 0)) \
                + W_POLE * pc + W_PRIOR * prior + W_TWIST * (w / 70.0) ** 2
            # A wrist past its limits leaves the hand short of `orient`:
            # weighed below the body (a hand turned a little off is better
            # than one in the chest), above the elbow's preferences.
            cost = cost + W_WRIST * wviol
            k = int(np.argmin(cost))
            out = {"e": float(e[k]), "p": float(p[k]), "w": float(w[k]),
                   "bend": bend, "body": float(pen_b[k]),
                   "arm": float(pen_a[k]), "viol": float(viol[k])}
            if wa is not None:
                out["wrist"] = [float(x[k]) for x in wa]
                out["wviol"] = float(np.asarray(wviol)[k])
            return out

        def blocked(b):
            return b["body"] > PEN_TOL or (other_mode == "escape"
                                           and b["arm"] > PEN_TOL)
        best = search(tb)
        why = ""
        target = np.array(tb, float)
        if self.coll and blocked(best):
            what = ("your other arm" if best["arm"] > best["body"] and
                    other_mode == "escape" else "your body")
            depth = max(best["arm"], best["body"])
            dirs = []
            g = self._clear_grad(tb, other if other_mode == "escape"
                                 else None)
            if g is not None:
                dirs.append(g)
            dirs += [np.array([0.0, 0.0, 1.0]), np.array([sg, 0.0, 0.0])]
            found = None
            k = 1
            while found is None and k * ESCAPE_STEP <= ESCAPE_MAX + 1e-9:
                for dvec in dirs:
                    t2 = tb + k * ESCAPE_STEP * dvec
                    b2 = search(t2)
                    if not blocked(b2):
                        found, target = b2, t2
                        break
                k += 1
            if found is not None:
                best = found
                why = (f"that would pass through {what}; the nearest clear "
                       "pose stops short")
            else:
                why = f"!no pose clears {what} ({depth:.2f} hu into it)"
        lim = [(self.params[k]["min"], self.params[k]["max"])
               for k in ARM[s]]
        out = {}
        for k, x, (lo, hi) in zip(ARM[s], (best["e"], best["p"], best["w"],
                                           best["bend"]), lim):
            out[k] = float(np.clip(x, lo, hi))
        if "wrist" in best:
            out.update(zip(WRIST[s], best["wrist"]))
            if best.get("wviol", 0.0) > 1.0:
                # The wrist could not turn the hand all the way: the palm,
                # placed for the full turn, would miss. Reach the palm
                # again with the wrist as it can be, the swivel near this.
                sol, info = self._arm_solve(s, palm_t, {**values, **out},
                                            other_mode)
                out.update(sol)
                return out, info
        return out, {"why": why, "pen": max(best["body"], best["arm"]),
                     "target": target}

    def _viol_batch(self, s, e, p, w):
        """arm_viol over arrays."""
        lim = [(self.params[k]["min"], self.params[k]["max"])
               for k in ARM[s][:3]]
        v = sum(np.maximum(np.maximum(lo - x, x - hi), 0.0)
                for x, (lo, hi) in zip((e, p, w), lim))
        er, pr = np.radians(e), np.radians(p)
        back = -np.sin(er) * np.sin(pr)
        up = -np.cos(er)
        ext = np.degrees(np.arcsin(np.clip(back, 0.0, 1.0)))
        ext_max = EXT_LEVEL + (EXT_HANGING - EXT_LEVEL) * np.maximum(-up, 0) \
            - (EXT_LEVEL - EXT_OVERHEAD) * np.maximum(up, 0)
        return v + np.maximum(ext - ext_max, 0.0)

    def arm_viol(self, s, e, p, w) -> float:
        """How far (deg) arm angles are outside the joint limits: the box
        (LIMITS) and the shoulder's extension, which shrinks as the arm
        rises (EXT_*: hanging 60, level 45, overhead 30)."""
        return float(self._viol_batch(s, np.array([e], float),
                                      np.array([p], float),
                                      np.array([w], float))[0])

    def _caps_batch(self, s, Rp, S, A, bend, Lh):
        """arm_caps for n upper-arm rotations A (n, 3, 3) at one bend, the
        hand's local rotation Lh ((3, 3) or (n, 3, 3)): {name: (a (n, 3),
        b (n, 3), r)}, body frame hu."""
        sg = SIGMA_SIDE[s]
        ub, lb, hb = (f"{SIDE[s]}UpperArm", f"{SIDE[s]}LowerArm",
                      f"{SIDE[s]}Hand")
        a_u, a_f = self.align[s]
        RA = Rp[None] @ A
        elbow = S[None] + RA @ (a_u @ (self.rest[lb] - self.rest[ub]))
        Wf = RA @ (ry(-sg * bend) @ a_f)
        wrist = elbow + Wf @ (self.rest[hb] - self.rest[lb])
        ru, rf, rh = self._arm_r[s]

        def B(x):
            return (x - self.origin) / self.hu
        sb, eb, wb = B(S)[None], B(elbow), B(wrist)
        out = {"upper": (sb + UA_SKIP * (eb - sb), eb, ru),
               "fore": (eb, wb, rf)}
        Wh = Wf @ Lh
        for i, (pa, pb) in enumerate(self._hand_rest[s]):
            out[f"hand{i}"] = (B(wrist + Wh @ (pa - self.rest[hb])),
                               B(wrist + Wh @ (pb - self.rest[hb])), rh)
        return out

    def _body_pen_batch(self, caps, n=9):
        u = np.linspace(0.0, 1.0, n)[None, :, None]
        pens = []
        for a, b, r in caps.values():
            pts = a[:, None] + u * (b - a)[:, None]
            m = len(a)
            sd = self.body_sdf(pts.reshape(-1, 3)).reshape(m, n)
            pens.append((r - sd).max(axis=1))
        return np.maximum(np.max(pens, axis=0), 0.0)

    @staticmethod
    def _pair_pen_batch(caps, other, gap=HAND_GAP):
        """_pair_pen of n candidate arms against one other arm: (n,)."""
        best = None
        for n1, (a, b, r1) in caps.items():
            for n2, c, d, r2 in other:
                if n1 == "upper" and n2 == "upper":
                    continue
                p, q = _closest_batch(a, b, np.broadcast_to(c, a.shape),
                                      np.broadcast_to(d, a.shape))
                pen = r1 + r2 + gap - np.linalg.norm(p - q, axis=1)
                best = pen if best is None else np.maximum(best, pen)
        return np.maximum(best, 0.0)

    def _clear_grad(self, tb, other):
        """The direction (unit, hu) in which a palm at tb gets out of the
        body and the other arm fastest: the gradient of the clearance
        (the least of the body's signed distance and the distance to the
        other arm's capsules), by central differences; None when flat."""
        def clear(p):
            d = float(self.body_sdf(p[None, :])[0])
            for _n, a, b, r in other or ():
                d = min(d, float(_seg_dist(p[None, :], a, b)[0]) - r)
            return d
        h = 0.02
        g = np.array([clear(tb + h * e) - clear(tb - h * e)
                      for e in np.eye(3)])
        n = float(np.linalg.norm(g))
        return g / n if n > 1e-6 else None

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
        """Palm positions over a grid of arm joint angles (torso and wrist
        at rest) within the joint limits and, with a collision model, out
        of the body: what the hand can reach."""
        cache = self.__dict__.setdefault("_ws", {})
        if s in cache:
            return cache[s]
        sg = SIGMA_SIDE[s]
        base = self.default_values()
        W, P = self.fk(base)
        ub, lb, hb = (f"{SIDE[s]}UpperArm", f"{SIDE[s]}LowerArm",
                      f"{SIDE[s]}Hand")
        S, Rp = P[ub], W[self.bones[ub]["parent"]]
        a_u, a_f = self.align[s]
        u1 = a_u @ (self.rest[lb] - self.rest[ub])
        c = a_f @ (self.palm_rest[s] - self.rest[lb])
        lim = {k: (self.params[k]["min"], self.params[k]["max"])
               for k in ARM[s]}
        ek, pk, tk, bk = ARM[s]
        E, Pl, T = np.meshgrid(np.linspace(*lim[ek], 18),
                               np.linspace(*lim[pk], 23),
                               np.linspace(*lim[tk], 5), indexing="ij")
        E, Pl, T = E.ravel(), Pl.ravel(), T.ravel()
        ok = self._viol_batch(s, E, Pl, T) <= 1e-9
        E, Pl, T = E[ok], Pl[ok], T[ok]
        er, pr = np.radians(E), np.radians(Pl)
        a = np.stack([sg * np.sin(er) * np.cos(pr), -np.cos(er),
                      np.sin(er) * np.sin(pr)], 1)
        A = np.stack([between([0.0, -1.0, 0.0], x) for x in a]) \
            @ rz(-sg * 90.0)[None] @ np.stack([rx(-t) for t in T])
        pts = []
        for b in np.linspace(*lim[bk], 8):
            palm = S[None] + (Rp[None] @ A) @ (u1 + ry(-sg * b) @ c)
            keep = np.ones(len(A), bool)
            if self.coll:
                caps = self._caps_batch(s, Rp, S, A, b, np.eye(3))
                keep = self._body_pen_batch(caps) <= PEN_TOL
            pts.append((palm[keep] - self.origin) / self.hu)
        cache[s] = np.concatenate(pts)
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

    def look_scale(self):
        """(yaw, pitch up, pitch down) look-at input degrees at gaze 1: the
        range map's input maximum (the avatar's own extremes), scaled down
        where its output would turn a bone look-at's eyes past
        EYE_MAX_DEG. The up map is three-vrm's rangeMapVerticalDown for a
        negative pitch and the other way round (state())."""
        la = self.vrm["look_at"]

        def k(inp, out):
            if la.get("type", "bone") != "bone" or out is None or out <= 0:
                return inp
            return inp * min(1.0, EYE_MAX_DEG / out)
        h_out = max(la.get("h_out", 0) or 0, la.get("h_in", 0) or 0) or None
        return (k(la["h"], h_out), k(la["v_down"], la.get("v_down_out")),
                k(la["v_up"], la.get("v_up_out")))

    def eye_degrees(self):
        """(yaw, up, down) eye rotation (deg) at gaze 1 for a bone
        look-at, None for an expression one or an old body file."""
        la = self.vrm["look_at"]
        if la.get("type", "bone") != "bone" or "h_out" not in la:
            return None
        kx, ku, kd = self.look_scale()
        return (max(la["h_out"], la["h_in"]) * kx / la["h"],
                la["v_down_out"] * ku / la["v_down"],
                la["v_up_out"] * kd / la["v_up"])

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
        gx, gy = float(v.get("gaze.x", 0)), float(v.get("gaze.y", 0))
        # three-vrm's look-at pitch is + down (eye direction (0, -0.17,
        # 0.98) at pitch +90 on both samples, in Chromium), and it maps a
        # negative pitch through rangeMapVerticalDown: gaze.y + (up) is a
        # negative pitch over that map's input range.
        kx, k_up, k_down = self.look_scale()
        look = [round(gx * kx, 3), round(-gy * (k_up if gy > 0 else k_down),
                                         3)]
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
         f"look-at {v['look_at']['type']}: gaze 1 = look-at input "
         f"+-{bm.look_scale()[0]:g} deg"
         + (" = eyes {:.0f} deg sideways, {:.0f} up, {:.0f} down".format(
             *bm.eye_degrees()) if bm.eye_degrees() else "")
         + f"; spring chains {v['springs']}",
         "collision: " + (f"{len(v['collision']['slabs'])} trunk slabs, "
                          "arm radii " + ", ".join(
                              f"{s} {a['upper_r']:.3f}/{a['fore_r']:.3f}/"
                              f"{a['hand_r']:.3f}" for s, a in
                              v['collision']['arms'].items()) + " hu"
                          if v.get("collision") else "none"),
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
