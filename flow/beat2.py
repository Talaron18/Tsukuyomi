#!/usr/bin/env python3
"""
BEAT2 co-speech motion on a VRM body: real human gestures as the action
head's targets (docs/embodiment.md E4).

The teacher LLM's marker-driven trajectories are partly arbitrary, and the
head learned little from speech alone (flow.actionhead). BEAT2 (SMPL-X
mocap of conversations, with audio and word timings) is what people do
while they talk. This module reads it, puts it on a VRM avatar by rotation
transfer, checks it against the avatar's own collision model and writes it
in the format `flow.dataexport action` writes, so scripts/extract_hidden.py
--context none and `python -m flow.actionhead train` take it unchanged.

  input     <beat2>/smplxflame_30/<clip>.npz: poses [N, 165] (55 SMPL-X
            joints x axis-angle: global orient, 21 body, jaw, 2 eyes, 2 x 15
            hand; [N, 55, 3] is accepted), trans [N, 3] metres, optional
            mocap_frame_rate / mocap_framerate / fps (30 assumed, and said in
            the report, when absent; any other rate is resampled to 30 fps
            with quaternion slerp, trans linear); <beat2>/wave16k/<clip>.wav;
            <beat2>/textgrid/<clip>.TextGrid (Praat, long or short text
            format, UTF-8 or UTF-16; the "words" tier; empty / sp / sil
            intervals are silence). Nothing here is checked against the real
            files: a missing key or folder stops with what was found.
            Speakers are told by the clip name ("2_scott_0_1_1": id 2, name
            scott). SMPL-X's body model (its mesh and licence) is not
            needed, only its joint rotations.
  retarget  Rotation transfer, no joint positions. Both skeletons are
            T-poses whose bones all have the identity rotation at rest (the
            SMPL-X template, three-vrm's normalized humanoid, which is what
            flow.vrmbody drives), y up, facing +Z, the character's left +X:
            one frame, so a joint's world rotation is the same rotation on
            both (VRM 0 is the 1.0 frame here too, as everywhere in
            flow.vrmbody: the page turns it round; nothing in this module
            knows the version). The world rotations are fitted to the
            body's own parameters (vrmbody.JOINTS) from the top down, each
            joint measured against what the joints above it achieved after
            their limits, so a clamped torso does not move the hands:
            torso = spine3, head = head against the torso, shoulders = the
            collar's rotation about z, upper arm elevation / plane from its
            direction in the shoulder's frame, the elbow's bend from the
            angle to the forearm and the twist chosen so the forearm lands
            on the mocap forearm (SMPL-X splits the axial turn between
            shoulder and elbow unreliably), the wrist from what is left.
            Forearm roll goes to the wrist's roll, the VRM has no forearm
            twist bone. A droop in the avatar's own T-pose (vrmbody's
            `align`) drops out: the parameters are measured from the
            ideal T. Root: the global orientation's yaw is the clip's
            facing, a circular moving mean over --facing-window seconds
            (0: the clip's mean), taken off every rotation, so the avatar
            keeps facing the camera while its torso turns a little about
            it; the translation, in the same frame and with its own moving
            mean off, is the hips' sway and bob (hips.x, hips.y, hu,
            clipped). Fingers (30 bones, where the avatar has them) are
            carried for the preview only; --fingers stores them. The legs
            are not used. An up axis that is not y (a z-up clip) is found
            from the spine and turned (reported).
  variables the columns flow.dataset.traj_vars gives this body, from its own
            forward model (the palms by forward kinematics of the fitted
            parameters, the head / torso / shoulder parameters as they are,
            hips.x / hips.y); the face (gaze, eyes, brows, mouth, visemes)
            is NaN, which the action head's loss masks: the voice owns the
            mouth, BEAT2's FLAME face is not mapped. The fitted parameters
            themselves are stored too (<key>.joints).
  filter    GMR's lesson: retargeting artifacts are learned. Every frame is
            measured on the fitted values before the limits clamp them: arm
            penetration into the trunk / head and arm into arm (the body's
            collision model, hu), the palms' speed (hu/s), how far a joint
            is past its limit (deg; the shoulder's extension too). A frame
            past --max-pen / --max-speed / --max-limit is bad, with
            --bad-pad s around it; utterances are cut at bad runs and the
            pieces left kept if they are --min-seg s or longer. Rotation
            transfer keeps angles, not contacts: a hand on a thin actor's
            belly is inside a bulkier avatar, so the defaults drop only
            gross cases (vrmbody's PEN_TOL, 0.02 hu, is a solver's
            tolerance, not a filter), and the report has the distributions
            to choose from.
  segments  One utterance per row, cut at silences of --gap s between words
            (the TextGrid), 3-15 s (--min-seg, --max-seg: longer ones split
            at the longest silence near the middle, shorter ones merged
            with a neighbour when close, else dropped), with --pad s of
            motion either side.
  export    <out>/action.jsonl and action.npz as flow.dataexport action
            writes them (one row per segment: id "beat2.<clip>/sNN", avatar =
            the body file's name, vars, fps 30, spoken = the words joined,
            reply_raw = spoken, chars [len(spoken), 2] start, end seconds
            from the word times, relative to the segment, traj [frames,
            vars] float16) plus:
              <key>.joints     [frames, 23] vrmbody.JOINTS, float16
              <key>.prosody    [frames, 3] float16: RMS (linear, 1 = full
                               scale), f0 in Hz (0 when unvoiced), voiced
                               (0 / 1), at frame f's time f / fps from the
                               wav (RMS over one frame, f0 YIN over 64 ms);
                               NOT read by flow.actionhead yet -- the key
                               the head can later read
              <key>.fingers    [frames, 30, 4] quaternions (--fingers)
            and in the row: clip, speaker, t0, t1 (clip seconds),
            prosody / joints column names. The avatar must have a
            rigs/<avatar>.body.json (flow.train.hidden's parser).
  files     <out>/parts/<clip>.{jsonl,npz} (one clip's pieces; a clip done is
            not redone without --force), <out>/reports/<clip>.json,
            <out>/summary.json (`report`), <out>/meta.json. --out must be
            under data/ (gitignored): BEAT2's licence is not ours to
            redistribute, retargeted mocap is a derivative; never commit it.
  preview   the segment's fitted joints as states.json (flow.vrm's record)
            and a flow.vrmrender mp4, mouth opening with the audio's RMS,
            fingers if stored; --also-vrm renders the same segment on a
            second avatar beside it (re-retargeted from the BEAT2 folder).

Usage:
    python -m flow.beat2 retarget --beat2 ~/BEAT2 \\
        --vrm assets/vrm/VRM1_Constraint_Twist_Sample.vrm \\
        --body rigs/seed-san.body.json --out data/beat2/seed-san \\
        [--speakers 2,scott] [--limit 20] [--jobs 4] [--fingers]
    python -m flow.beat2 report --out data/beat2/seed-san
    python -m flow.beat2 preview --out data/beat2/seed-san --beat2 ~/BEAT2 \\
        --id beat2.2_scott_0_1_1/s03
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import struct
import sys
import unicodedata
import zipfile
from collections import Counter

import numpy as np

try:
    from . import body as fb
    from . import vrmbody as vb
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import body as fb
    from flow import vrmbody as vb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 30.0
SOURCE = "beat2"

# SMPL-X's 55 joints in the order of BEAT2's 165 pose values.
BODY = ("pelvis", "left_hip", "right_hip", "spine1", "left_knee",
        "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
        "left_foot", "right_foot", "neck", "left_collar", "right_collar",
        "head", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye",
        "right_eye")
HAND = ("index", "middle", "pinky", "ring", "thumb")
JOINTS55 = BODY + tuple(f"{s}_{f}{k}" for s in ("left", "right")
                        for f in HAND for k in (1, 2, 3))
J = {n: i for i, n in enumerate(JOINTS55)}
SMPL_PARENT = {"left_hip": "pelvis", "right_hip": "pelvis",
               "spine1": "pelvis", "left_knee": "left_hip",
               "right_knee": "right_hip", "spine2": "spine1",
               "left_ankle": "left_knee", "right_ankle": "right_knee",
               "spine3": "spine2", "left_foot": "left_ankle",
               "right_foot": "right_ankle", "neck": "spine3",
               "left_collar": "spine3", "right_collar": "spine3",
               "head": "neck", "left_shoulder": "left_collar",
               "right_shoulder": "right_collar", "left_elbow": "left_shoulder",
               "right_elbow": "right_shoulder", "left_wrist": "left_elbow",
               "right_wrist": "right_elbow", "jaw": "head",
               "left_eye": "head", "right_eye": "head"}
for _s in ("left", "right"):
    for _f in HAND:
        SMPL_PARENT[f"{_s}_{_f}1"] = f"{_s}_wrist"
        SMPL_PARENT[f"{_s}_{_f}2"] = f"{_s}_{_f}1"
        SMPL_PARENT[f"{_s}_{_f}3"] = f"{_s}_{_f}2"
PARENT = [J[SMPL_PARENT[n]] if n in SMPL_PARENT else -1 for n in JOINTS55]
# The three-vrm bone of each finger joint (the 1.0 vocabulary; a VRM 0 file
# names the thumb's bones one step down, FINGER_V0).
VRM_FINGER = {"index": "Index", "middle": "Middle", "pinky": "Little",
              "ring": "Ring", "thumb": "Thumb"}
VRM_KNUCKLE = {"thumb": ("Metacarpal", "Proximal", "Distal")}
VRM_KNUCKLE_DEFAULT = ("Proximal", "Intermediate", "Distal")
FINGER_V0 = {"ThumbMetacarpal": "ThumbProximal",
             "ThumbProximal": "ThumbIntermediate"}
NAN_PREFIX = ("gaze.", "eye.", "brow.", "mouth.", "cheek.", "viseme.")
PROSODY_COLS = ("rms", "f0", "voiced")
SILENCE = {"", "sp", "sil", "spn", "sp1", "<unk>", "<eps>", "<sil>", "<sp>",
           "{sl}", "{ls}", "{lg}", "{ns}", "noise", "[silence]", "pau"}

# Defaults of the filter and the segmenter (module docstring).
CFG = dict(facing_window=20.0, max_pen=0.10, max_speed=25.0, max_limit=25.0,
           bad_pad=0.15, min_seg=3.0, max_seg=15.0, gap=0.35, pad=0.3,
           hips_x_max=0.15, hips_y_max=0.08, fingers=False, check_text=True)


def log(s):
    print(s, flush=True)


# ------------------------------------------------------------------ math

def skew(k):
    z = np.zeros(k.shape[:-1])
    return np.stack([np.stack([z, -k[..., 2], k[..., 1]], -1),
                     np.stack([k[..., 2], z, -k[..., 0]], -1),
                     np.stack([-k[..., 1], k[..., 0], z], -1)], -2)


def aa_to_mat(aa):
    """Rotation matrices [..., 3, 3] of axis-angle vectors [..., 3]."""
    aa = np.asarray(aa, float)
    th = np.linalg.norm(aa, axis=-1)[..., None, None]
    K = skew(aa / np.maximum(th[..., 0], 1e-12))
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def aa_to_quat(aa):
    """(x, y, z, w) of axis-angle vectors [..., 3], w >= 0."""
    aa = np.asarray(aa, float)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    q = np.concatenate([k * np.sin(th / 2), np.cos(th / 2)], -1)
    return np.where(q[..., 3:] < 0, -q, q)


def quat_to_mat(q):
    q = np.asarray(q, float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                  2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                  2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                  1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_quat(R):
    """(x, y, z, w), w >= 0, of rotation matrices [..., 3, 3] (Shepperd)."""
    R = np.asarray(R, float)
    m = R.reshape(-1, 3, 3)
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.zeros((len(m), 4))
    pick = np.argmax(np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2], tr], 1), 1)
    for case in range(4):
        i = pick == case
        if not i.any():
            continue
        a = m[i]
        if case == 3:
            s = 2 * np.sqrt(np.maximum(tr[i] + 1, 1e-12))
            q[i] = np.stack([(a[:, 2, 1] - a[:, 1, 2]) / s,
                             (a[:, 0, 2] - a[:, 2, 0]) / s,
                             (a[:, 1, 0] - a[:, 0, 1]) / s, s / 4], 1)
        elif case == 0:
            s = 2 * np.sqrt(np.maximum(1 + a[:, 0, 0] - a[:, 1, 1]
                                       - a[:, 2, 2], 1e-12))
            q[i] = np.stack([s / 4, (a[:, 0, 1] + a[:, 1, 0]) / s,
                             (a[:, 0, 2] + a[:, 2, 0]) / s,
                             (a[:, 2, 1] - a[:, 1, 2]) / s], 1)
        elif case == 1:
            s = 2 * np.sqrt(np.maximum(1 + a[:, 1, 1] - a[:, 0, 0]
                                       - a[:, 2, 2], 1e-12))
            q[i] = np.stack([(a[:, 0, 1] + a[:, 1, 0]) / s, s / 4,
                             (a[:, 1, 2] + a[:, 2, 1]) / s,
                             (a[:, 0, 2] - a[:, 2, 0]) / s], 1)
        else:
            s = 2 * np.sqrt(np.maximum(1 + a[:, 2, 2] - a[:, 0, 0]
                                       - a[:, 1, 1], 1e-12))
            q[i] = np.stack([(a[:, 0, 2] + a[:, 2, 0]) / s,
                             (a[:, 1, 2] + a[:, 2, 1]) / s, s / 4,
                             (a[:, 1, 0] - a[:, 0, 1]) / s], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(q[:, 3:] < 0, -q, q)
    return q.reshape(R.shape[:-2] + (4,))


def rot(axis, deg):
    """Batch rotation about x, y or z (axis 0, 1, 2) by deg [n]."""
    d = np.radians(np.atleast_1d(np.asarray(deg, float)))
    c, s = np.cos(d), np.sin(d)
    M = np.zeros((len(d), 3, 3))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    M[:, axis, axis] = 1.0
    M[:, i, i], M[:, j, j] = c, c
    M[:, i, j], M[:, j, i] = -s, s
    return M


def euler_batch(yaw, pitch, tilt):
    """vrmbody.euler over arrays."""
    return rot(1, yaw) @ rot(0, -np.asarray(pitch)) @ rot(2, tilt)


def euler_angles(R):
    """(yaw, pitch, tilt) in degrees of euler_batch's R (its inverse)."""
    yaw = np.degrees(np.arctan2(R[:, 0, 2], R[:, 2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(R[:, 1, 2], -1, 1)))
    tilt = np.degrees(np.arctan2(R[:, 1, 0], R[:, 1, 1]))
    return yaw, pitch, tilt


def angle_between(A, B):
    """Degrees of the rotation taking A to B (n, 3, 3)."""
    c = (np.einsum("nij,nij->n", A, B) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def down_to(v):
    """Rotations taking (0, -1, 0) to the unit vectors v (n, 3)."""
    u = np.array([0.0, -1.0, 0.0])
    c = v @ u
    K = skew(np.cross(np.broadcast_to(u, v.shape), v))
    return np.eye(3) + K + (K @ K) / (1.0 + c)[:, None, None]


def wrap180(d):
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def moving_mean(x, win):
    """Centred moving mean over win samples along axis 0, the window cut at
    the ends."""
    x = np.asarray(x, float)
    if win <= 1:
        return x.copy()
    n, h = len(x), win // 2
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)])
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))


# ------------------------------------------------------------- the clips

def _first_dir(root, names):
    for n in names:
        d = os.path.join(root, n)
        if os.path.isdir(d):
            return d
    for d in sorted(glob.glob(os.path.join(root, names[0].split("_")[0]
                                           + "*"))):
        if os.path.isdir(d):
            return d
    return None


def layout(root) -> dict:
    """{poses, wave, textgrid}: BEAT2's three folders under root (the
    names of the HF dataset; the first that exists, or one starting like
    it), or SystemExit saying what is there."""
    def look(r):
        return {"poses": _first_dir(r, ("smplxflame_30", "smplxflame")),
                "wave": _first_dir(r, ("wave16k", "wave")),
                "textgrid": _first_dir(r, ("textgrid", "TextGrid"))}
    out = look(root)
    miss = [k for k, v in out.items() if v is None]
    if miss and os.path.isdir(root):
        # One level down (a dataset that keeps them in a subfolder): taken
        # when exactly one subfolder holds all three.
        nested = [d for d in sorted(glob.glob(os.path.join(root, "*")))
                  if os.path.isdir(d) and None not in look(d).values()]
        if len(nested) == 1:
            return look(nested[0])
        if len(nested) > 1:
            raise SystemExit(f"beat2: {len(nested)} subfolders of {root} hold "
                             f"the three folders: {nested}; pass one as "
                             "--beat2")
    if miss:
        have = sorted(os.listdir(root)) if os.path.isdir(root) else "no such folder"
        raise SystemExit(f"beat2: {root}: no folder for {miss} (expected "
                         "smplxflame_30/, wave16k/, textgrid/); found: "
                         f"{have}")
    return out


def _find(folder, stem, exts):
    for e in exts:
        p = os.path.join(folder, stem + e)
        if os.path.exists(p):
            return p
    return None


def speaker_of(stem):
    """(id, name) from a BEAT2 clip name such as 2_scott_0_1_1."""
    t = stem.split("_")
    return (t[0] if t and t[0].isdigit() else None,
            t[1].lower() if len(t) > 1 else None)


def list_clips(root, speakers=None, limit=0) -> list[dict]:
    """[{stem, npz, wav, textgrid}] of the clips that have all three
    files, sorted; speakers: ids or names ("2,scott")."""
    lay = layout(root)
    want = {s.strip().lower() for s in (speakers or "").split(",")
            if s.strip()}
    out = []
    # Per speaker: npz files, and how many of them have a wav / a TextGrid
    # (a partial download shows up here).
    count = {}
    for p in sorted(glob.glob(os.path.join(lay["poses"], "*.npz"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        sid, name = speaker_of(stem)
        wav = _find(lay["wave"], stem, (".wav", ".WAV"))
        tg = _find(lay["textgrid"], stem, (".TextGrid", ".textgrid",
                                           ".Textgrid"))
        c = count.setdefault(f"{sid}_{name}", [0, 0, 0])
        c[0] += 1
        c[1] += wav is not None
        c[2] += tg is not None
        if want and not ({sid, name} & want):
            continue
        if wav and tg:
            out.append({"stem": stem, "npz": p, "wav": wav, "textgrid": tg})
    if not out:
        def k(x):
            h = x.split("_")[0]
            return (int(h) if h.isdigit() else 1 << 30, x)
        have = ", ".join(f"{s} {c[0]}/{c[1]}/{c[2]}"
                         for s, c in sorted(count.items(),
                                            key=lambda kv: k(kv[0])))
        raise SystemExit(
            f"beat2: no clip with an npz, a wav and a TextGrid under {root} "
            f"(speakers {sorted(want) or 'all'}). Per speaker, npz / with a "
            f"wav / with a TextGrid: {have or 'no npz at all'}. Folders: "
            f"{lay['poses']}, {lay['wave']}, {lay['textgrid']}")
    return out[:limit] if limit else out


def resample(poses, trans, src, dst=FPS):
    """poses [N, 55, 3] axis-angle, trans [N, 3] at src fps -> at dst fps:
    quaternions slerped (the shorter way round), translation linear."""
    n = len(poses)
    if abs(src - dst) < 1e-6 or n < 2:
        return poses, trans
    m = int(math.floor((n - 1) / src * dst + 1e-9)) + 1
    t = np.arange(m) * (src / dst)
    i0 = np.minimum(t.astype(int), n - 2)
    u = (t - i0)[:, None, None]
    q = aa_to_quat(poses)
    a, b = q[i0], q[i0 + 1]
    b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)
    d = np.clip((a * b).sum(-1, keepdims=True), -1, 1)
    om = np.arccos(d)
    sn = np.sin(om)
    near = sn < 1e-6
    w0 = np.where(near, 1 - u, np.sin((1 - u) * om) / np.where(near, 1, sn))
    w1 = np.where(near, u, np.sin(u * om) / np.where(near, 1, sn))
    r = w0 * a + w1 * b
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(np.linalg.norm(r[..., :3], axis=-1), r[..., 3])
    ax = r[..., :3] / np.maximum(np.linalg.norm(r[..., :3], axis=-1,
                                                keepdims=True), 1e-12)
    out = ax * ang[..., None]
    u1 = (t - i0)[:, None]
    return out, trans[i0] * (1 - u1) + trans[i0 + 1] * u1


def read_clip(path, fps=FPS, assume_fps=30.0) -> dict:
    """{poses [N, 55, 3], trans [N, 3], fps, src_fps, fps_source, notes} of a
    BEAT2 smplxflame npz, at `fps` (module docstring)."""
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"{path}: not readable as an npz: {e}") from e
    keys = list(z.files)
    if "poses" not in keys:
        raise ValueError(f"{path}: no 'poses' (keys: {keys}); BEAT2's "
                         "smplxflame_30 npz has poses [N, 165]")
    p = np.asarray(z["poses"], float)
    if p.ndim == 3 and p.shape[1:] == (55, 3):
        p = p.reshape(len(p), 165)
    if p.ndim != 2 or p.shape[1] != 165:
        raise ValueError(f"{path}: poses is {p.shape}, expected [N, 165] "
                         "(55 SMPL-X joints x 3 axis-angle)")
    notes = []
    if "trans" in keys:
        tr = np.asarray(z["trans"], float)
        if tr.shape != (len(p), 3):
            raise ValueError(f"{path}: trans is {tr.shape}, poses {p.shape}")
    else:
        tr = np.zeros((len(p), 3))
        notes.append("no trans: no hips sway")
    src, how = None, "assumed"
    for k in ("mocap_frame_rate", "mocap_framerate", "fps", "frame_rate"):
        if k in keys:
            src, how = float(np.asarray(z[k]).reshape(-1)[0]), k
            break
    if src is None:
        src = float(assume_fps)
        notes.append(f"no frame rate key (keys: {keys}): {src:g} fps assumed")
    if not 1.0 <= src <= 480.0:
        raise ValueError(f"{path}: frame rate {src}")
    if not np.isfinite(p).all() or not np.isfinite(tr).all():
        raise ValueError(f"{path}: NaN or inf in poses / trans")
    pose, tr = resample(p.reshape(-1, 55, 3), tr, src, fps)
    if abs(src - fps) > 1e-6:
        notes.append(f"resampled {src:g} -> {fps:g} fps")
    return {"poses": pose, "trans": tr, "fps": float(fps), "src_fps": src,
            "fps_source": how, "notes": notes}


# ------------------------------------------------------ TextGrid and wav

_TOK = re.compile(r'"(?:[^"]|"")*"|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')
_HEAD = re.compile(r"^\s*(?:item|intervals|points)\s*\[\d*\]\s*:\s*$", re.M)


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeError:
            continue
    return raw.decode("latin-1")


def parse_textgrid(text: str) -> dict:
    """{tier name: [(xmin, xmax, text)]} of a Praat TextGrid in the long or
    the short text format (point tiers: (time, time, mark))."""
    text = _HEAD.sub("", text)
    toks = _TOK.findall(text)

    def s(t):
        if not t.startswith('"'):
            raise ValueError(f"TextGrid: a string expected, got {t!r}")
        return t[1:-1].replace('""', '"')
    if len(toks) < 5 or s(toks[1]) != "TextGrid":
        raise ValueError("not a Praat TextGrid (no 'TextGrid' header)")
    n_tiers = int(float(toks[4]))
    i, tiers = 5, {}
    for _ in range(n_tiers):
        cls, name = s(toks[i]), s(toks[i + 1])
        n = int(float(toks[i + 4]))
        i += 5
        rows = []
        for _k in range(n):
            if cls == "IntervalTier":
                rows.append((float(toks[i]), float(toks[i + 1]),
                             s(toks[i + 2])))
                i += 3
            else:
                t = float(toks[i])
                rows.append((t, t, s(toks[i + 1])))
                i += 2
        tiers[name] = rows
    return tiers


def read_textgrid(path, tier="words") -> list[tuple]:
    """[(start, end, text)] of the words tier (case-insensitive; else the
    first tier with 'word' in its name), silences included."""
    with open(path, "rb") as fh:
        tiers = parse_textgrid(_decode(fh.read()))
    for name, rows in tiers.items():
        if name.lower() == tier:
            return rows
    for name, rows in tiers.items():
        if tier in name.lower():
            return rows
    raise ValueError(f"{path}: no '{tier}' tier (tiers: {list(tiers)})")


def clean_word(w: str) -> str:
    """A word as spoken text: the parser (flow.pose) would eat [..] and
    *, silence labels are nothing."""
    w = unicodedata.normalize("NFKC", w or "")
    w = re.sub(r"[\[\]*]", "", w)
    w = re.sub(r"\s+", " ", w).strip()
    return "" if w.lower() in SILENCE else w


def words_of(rows) -> list[tuple]:
    """[(start, end, word)] of the spoken intervals, in time order."""
    out = []
    for a, b, t in rows:
        w = clean_word(t)
        if w and b > a:
            out.append((float(a), float(b), w))
    return sorted(out)


def _cjk(ch):
    return unicodedata.name(ch, "").startswith(("CJK", "HIRAGANA", "KATAKANA",
                                                 "HANGUL"))


def spoken_text(words, t0=0.0):
    """(spoken, chars [len, 2]) of [(start, end, word)]: words joined with
    a space (none between two CJK characters), each word's characters
    spread evenly over its interval, a space spanning the gap before the
    next word, so a token that starts with it is timed from where the last
    word stopped. Times relative to t0."""
    text, spans = "", []
    for k, (a, b, w) in enumerate(words):
        if k:
            pa, pb, pw = words[k - 1]
            if not (_cjk(pw[-1]) and _cjk(w[0])):
                text += " "
                spans.append((pb - t0, max(a - t0, pb - t0)))
        n = len(w)
        for c in range(n):
            spans.append((a - t0 + (b - a) * c / n,
                          a - t0 + (b - a) * (c + 1) / n))
        text += w
    return text, np.asarray(spans, np.float32).reshape(-1, 2)


def read_wav(path):
    """(mono float32 samples in [-1, 1], sample rate) of a RIFF wav: PCM
    8 / 16 / 24 / 32 bit, float 32 / 64, any channels (averaged)."""
    with open(path, "rb") as fh:
        b = fh.read()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF wav")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        body = b[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            if fmt[0] == 0xFFFE and len(body) >= 26:
                fmt = (struct.unpack("<H", body[24:26])[0],) + fmt[1:]
        elif cid == b"data":
            data = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path}: no fmt / data chunk")
    tag, ch, sr, _br, _ba, bits = fmt
    if tag == 1 and bits == 8:
        x = (np.frombuffer(data, np.uint8).astype(np.float32) - 128) / 128
    elif tag == 1 and bits == 16:
        x = np.frombuffer(data[:len(data) // 2 * 2], "<i2") / 32768.0
    elif tag == 1 and bits == 24:
        d = np.frombuffer(data[:len(data) // 3 * 3], np.uint8).reshape(-1, 3)
        v = (d[:, 0].astype(np.int32) | d[:, 1].astype(np.int32) << 8
             | d[:, 2].astype(np.int32) << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v) / 8388608.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(data[:len(data) // 4 * 4], "<i4") / 2147483648.0
    elif tag == 3 and bits in (32, 64):
        x = np.frombuffer(data[:len(data) // (bits // 8) * (bits // 8)],
                          "<f4" if bits == 32 else "<f8")
    else:
        raise ValueError(f"{path}: wav format {tag}, {bits} bit")
    x = np.asarray(x, np.float32)
    if ch > 1:
        x = x[:len(x) // ch * ch].reshape(-1, ch).mean(1)
    return x, int(sr)


def write_wav(path, x, sr):
    """A mono 16-bit wav of float samples."""
    import wave
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(y.tobytes())


def prosody(x, sr, fps, n, fmin=70.0, fmax=400.0, thresh=0.15) -> np.ndarray:
    """[n, 3] (rms, f0 Hz, voiced) at frame f's time f / fps: the RMS over
    one frame centred there; f0 by YIN (de Cheveigne & Kawahara 2002: the
    difference function, its cumulative mean normalisation, the first dip
    under `thresh`, parabolic interpolation) over 64 ms, voiced when a dip
    is found and the frame is not near silence. Frames past the audio are
    silent."""
    out = np.zeros((n, 3), np.float32)
    if len(x) == 0 or n == 0:
        return out
    x = np.asarray(x, np.float64)
    c = np.round(np.arange(n) / fps * sr).astype(int)
    h = max(int(round(sr / fps / 2)), 1)
    pad = np.concatenate([np.zeros(h), x, np.zeros(h)])
    csum = np.concatenate([[0.0], np.cumsum(pad ** 2)])
    lo = np.clip(c, 0, len(x))
    hi = np.clip(c + 2 * h, 0, len(pad))
    out[:, 0] = np.sqrt(np.maximum(csum[hi] - csum[lo], 0) / (2 * h))
    W = int(sr * 0.032)
    tmax = int(sr / fmin) + 1
    tmin = max(int(sr / fmax), 2)
    L = W + tmax + 1
    xp = np.concatenate([np.zeros(W // 2), x, np.zeros(L + W)])
    nfft = 1 << int(np.ceil(np.log2(L + W)))
    tau = np.arange(tmax + 1)
    floor = max(0.02 * (out[:, 0].max() + 1e-9), 1e-4)
    live = np.flatnonzero(out[:, 0] >= floor)       # near silence: unvoiced
    for b0 in range(0, len(live), 1024):            # blocks: bounded memory
        idx = live[b0:b0 + 1024]
        fr = np.stack([xp[c[f]:c[f] + L] for f in idx])   # [:W] centred on c
        fr = fr - fr.mean(1, keepdims=True)
        A = np.fft.rfft(fr[:, :W], nfft)
        B = np.fft.rfft(fr, nfft)
        r = np.fft.irfft(np.conj(A) * B, nfft)[:, :tmax + 1]   # sum x_j x_j+t
        cs = np.concatenate([np.zeros((len(fr), 1)),
                             np.cumsum(fr ** 2, 1)], 1)
        d = cs[:, W:W + 1] + (cs[:, W + tau] - cs[:, tau]) - 2 * r
        d[:, 0] = 0
        cm = np.cumsum(d[:, 1:], 1) / np.arange(1, tmax + 1)
        dn = np.ones_like(d)
        dn[:, 1:] = d[:, 1:] / np.maximum(cm, 1e-12)
        for k, f in enumerate(idx):
            row = dn[k]
            t = tmin
            found = -1
            while t < tmax:
                if row[t] < thresh:
                    while t + 1 < tmax and row[t + 1] < row[t]:
                        t += 1
                    found = t
                    break
                t += 1
            if found < 0:
                continue
            a, b, g = row[found - 1], row[found], row[found + 1]
            den = a - 2 * b + g
            tt = found + (0.5 * (a - g) / den if abs(den) > 1e-12 else 0.0)
            f0 = sr / tt
            if fmin * 0.9 <= f0 <= fmax * 1.1:
                out[f, 1], out[f, 2] = f0, 1.0
    return out


# ---------------------------------------------------------------- avatar

def sha16(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


class Avatar:
    """A VRM body for the retargeter: the body model, its name (the
    rigs/<name>.body.json flow.train.hidden loads the parser from), the
    finger bones the .vrm has."""

    def __init__(self, vrm=None, body=None, log=log):
        self.vrm_path = vrm
        self.body_path = None
        if body:
            bm = fb.load(body)
            self.body_path = body
        else:
            bm = None
            if not vrm:
                raise SystemExit("beat2: --vrm (and --body)")
            found = []
            for cand in sorted(glob.glob(os.path.join(ROOT, "rigs",
                                                      "*.body.json"))):
                try:
                    b = fb.load(cand)
                except Exception:
                    continue
                if getattr(b, "kind", None) == "vrm" and \
                        b.vrm["file"] == os.path.basename(vrm):
                    found.append((b, cand))
            if len(found) > 1:
                raise SystemExit(
                    f"beat2: {len(found)} body files in rigs/ are for "
                    f"{os.path.basename(vrm)}: "
                    f"{[os.path.basename(c) for _b, c in found]}; pass "
                    "--body (its name is the avatar's name in the export)")
            if found:
                bm, self.body_path = found[0]
            if bm is None:
                log(f"beat2: no rigs/*.body.json for {os.path.basename(vrm)};"
                    " building one in memory. flow.train.hidden needs it on "
                    "disk: python -m flow.vrmbody --vrm "
                    f"{vrm} --out rigs/<name>.body.json")
                blob = vb.build(vrm, log=lambda s: None)
                bm = vb.VRMBody(blob)
        if getattr(bm, "kind", None) != "vrm":
            raise SystemExit(f"{body}: not a VRM body file")
        self.bm = bm
        stem = os.path.basename(self.body_path or "")
        self.name = stem[:-len(".body.json")] if stem.endswith(".body.json") \
            else (bm.blob.get("rig") or os.path.splitext(
                os.path.basename(vrm or "avatar"))[0])
        self.fingers = self._fingers()

    def _fingers(self):
        """[(smplx joint index, three-vrm bone)] of the finger bones the .vrm
        has, [] when there is no file."""
        if not self.vrm_path or not os.path.exists(self.vrm_path):
            return []
        try:
            from .gltf import GLTF, vrm_info
        except ImportError:
            from flow.gltf import GLTF, vrm_info
        have = set(vrm_info(GLTF.load(self.vrm_path))["humanoid"])
        v0 = self.bm.vrm.get("version") in ("0", 0)
        out = []
        for side in ("left", "right"):
            for f in HAND:
                for k in range(3):
                    knuckle = VRM_KNUCKLE.get(f, VRM_KNUCKLE_DEFAULT)[k]
                    bone = f"{side}{VRM_FINGER[f]}{knuckle}"
                    in_file = FINGER_V0.get(
                        f"{VRM_FINGER[f]}{knuckle}", None) if v0 else None
                    name = (side + in_file) if in_file else bone
                    if name in have:
                        out.append((J[f"{side}_{f}{k + 1}"], bone))
        return out


# -------------------------------------------------------------- retarget

def world_rotations(pose, up=None):
    """[N, 55, 3, 3] world rotations of axis-angle locals [N, 55, 3], the
    root turned by `up` (a 3x3 fixing the clip's up axis)."""
    R = aa_to_mat(pose)
    G = np.empty_like(R)
    for j in range(55):
        p = PARENT[j]
        if p < 0:
            G[:, j] = R[:, j] if up is None else up @ R[:, j]
        else:
            G[:, j] = G[:, p] @ R[:, j]
    return G


def find_up(G):
    """(3x3 turning the clip to y up, name): the median of the spine's up
    vector decides among y up, z up and -z up. SystemExit when none is
    plausible (the clip's frame is not what this module assumes)."""
    up = np.median(G[:, J["spine3"], :, 1], axis=0)       # local y in world
    cands = {"y-up": np.eye(3), "z-up": rot(0, [-90.0])[0],
             "-z-up": rot(0, [90.0])[0]}
    best = max(cands, key=lambda k: float((cands[k] @ up)[1]))
    if float((cands[best] @ up)[1]) < 0.6:
        raise ValueError(f"the spine's up vector {np.round(up, 2).tolist()} "
                         "is not along y, z or -z: the clip's frame is not "
                         "SMPL-X's")
    return cands[best], best


def facing_yaw(G, fps, window):
    """[N] degrees: the torso's heading (spine3's front and left, about y),
    its circular moving mean over `window` s (0: the whole clip's)."""
    S = G[:, J["spine3"]]
    z = (S[:, 2, 2] + 1j * S[:, 0, 2]) + (S[:, 0, 0] - 1j * S[:, 2, 0])
    u = np.stack([z.real, z.imag], 1)
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    if window and window * fps > 1:
        m = moving_mean(u, int(round(window * fps)) | 1)
    else:
        m = np.broadcast_to(u.mean(0), u.shape)
    return np.degrees(np.arctan2(m[:, 1], m[:, 0]))


def _split(bm, pairs):
    """vrmbody.local_rotations' split: the bones present and their shares."""
    have = [(b, w) for b, w in pairs if b in bm.bones]
    tot = sum(w for _b, w in pairs)
    got = sum(w for _b, w in have)
    return [(b, w * tot / got) for b, w in have] if got else []


def chain(bm, pairs, yaw, pitch, tilt):
    """The product of a split's bones for these angles (n,): what the model
    makes of a head / torso angle."""
    out = np.broadcast_to(np.eye(3), (len(yaw), 3, 3))
    for _b, w in _split(bm, pairs):
        out = out @ euler_batch(w * yaw, w * pitch, w * tilt)
    return out


def solve_chain(bm, pairs, R, steps=4):
    """Angles (yaw, pitch, tilt) whose split chain is R (n, 3, 3): the
    model's spine / chest / upperChest (or neck / head) product is not one
    euler rotation, so the single-rotation angles are corrected by what the
    chain falls short, a few times. Returns (angles, first guess' error in
    degrees)."""
    want = np.stack(euler_angles(R))
    th = want.copy()
    first = None
    for _ in range(steps + 1):
        got = np.stack(euler_angles(chain(bm, pairs, *th)))
        if first is None:
            first = angle_between(chain(bm, pairs, *th), R)
        th = th + (want - got)
    return th, first


def fit(bm, G, fingers=()):
    """Fit the body's parameters to the SMPL-X world rotations G [N, 55, 3,
    3] (facing already off), top down (module docstring, retarget). Returns
    {raw: {param: [N]} (before the limits), val: {param: [N]} (clamped),
    fingers: [N, J, 4] or None, torso_split: [N] deg (what the model's
    spine / chest / upperChest split would lose against one rotation,
    before solve_chain corrects the angles for it),
    torso_clamp, head_clamp: [N] deg}."""
    N = len(G)
    lim = {k: (float(bm.params[k]["min"]), float(bm.params[k]["max"]))
           for k in vb.JOINTS}
    raw, val = {}, {}

    def put(name, x):
        raw[name] = np.asarray(x, float)
        val[name] = np.clip(raw[name], *lim[name])
        return val[name]
    top = G[:, J["spine3"]]
    (y, p, t), split_loss = solve_chain(bm, vb.TORSO_SPLIT, top)
    ty, tp, tt = (put(f"torso.{k}", v) for k, v in
                  (("yaw", y), ("pitch", p), ("tilt", t)))
    UC = chain(bm, vb.TORSO_SPLIT, ty, tp, tt)
    torso_clamp = angle_between(UC, top)
    UCt = np.swapaxes(UC, 1, 2)
    (y, p, t), _ = solve_chain(bm, vb.HEAD_SPLIT, UCt @ G[:, J["head"]])
    hy, hp, ht = (put(f"head.{k}", v) for k, v in
                  (("yaw", y), ("pitch", p), ("tilt", t)))
    head_clamp = angle_between(UC @ chain(bm, vb.HEAD_SPLIT, hy, hp, ht),
                               G[:, J["head"]])
    # Shoulders: the collar's turn about z, in the torso's frame.
    up = {}
    for s in "LR":
        sg = vb.SIGMA_SIDE[s]
        rel = UCt @ G[:, J[f"{vb.SIDE[s].lower()}_collar"]]
        up[s] = sg * np.degrees(np.arctan2(rel[:, 1, 0], rel[:, 0, 0])) \
            / vb.SHOULDER_DEG
    mean = put("shoulders.up", (up["L"] + up["R"]) / 2)
    sh_world, hand_world, wrist_in = {}, {}, {}
    ext = np.zeros(N)
    for s in "LR":
        sg = vb.SIGMA_SIDE[s]
        side = vb.SIDE[s].lower()
        put(f"shoulder.{s}.up", up[s] - (up["L"] + up["R"]) / 2)
        eff = mean + val[f"shoulder.{s}.up"]
        Wsh = UC @ rot(2, sg * vb.SHOULDER_DEG * eff)
        Wt = np.swapaxes(Wsh, 1, 2)
        # Upper arm: direction in the shoulder's frame (SMPL-X's own axial
        # turn only where the elbow is straight).
        A = Wt @ G[:, J[f"{side}_shoulder"]]
        e, pl, tw_s = vb._arm_angles_batch(s, A)
        ke, kp, kt, kb = vb.ARM[s]
        e = put(ke, e)
        pl = put(kp, pl)
        a = np.stack([sg * np.sin(np.radians(e)) * np.cos(np.radians(pl)),
                      -np.cos(np.radians(e)),
                      np.sin(np.radians(e)) * np.sin(np.radians(pl))], 1)
        S0 = down_to(a) @ vb.rz(-sg * 90.0)
        f_t = np.einsum("nji,nj->ni", Wsh, sg * G[:, J[f"{side}_elbow"], :, 0])
        v = np.einsum("nji,nj->ni", S0, f_t)
        bend = np.degrees(np.arctan2(np.hypot(v[:, 1], v[:, 2]), sg * v[:, 0]))
        tw_f = np.degrees(np.arctan2(v[:, 1], v[:, 2]))
        w = np.clip((bend - 10.0) / 15.0, 0.0, 1.0)
        tw = tw_s + w * wrap180(tw_f - tw_s)
        b = put(kb, bend)
        tw = put(kt, tw)
        # (inside the box, what arm_viol still counts is the extension limit)
        ext = np.maximum(ext, bm._viol_batch(s, val[ke], val[kp], tw))
        WLA = Wsh @ S0 @ rot(0, -tw) @ rot(1, -sg * b)
        wr = np.swapaxes(WLA, 1, 2) @ G[:, J[f"{side}_wrist"]]
        roll, flex, dev = vb._wrist_angles_batch(s, wr)
        kr, kf, kd = vb.WRIST[s]
        roll, flex, dev = put(kr, roll), put(kf, flex), put(kd, dev)
        sh_world[s] = Wsh
        hand_world[s] = WLA @ vb._wrist_rotation_batch(s, roll, flex, dev)
    out = {"raw": raw, "val": val, "torso_split": split_loss,
           "torso_clamp": torso_clamp, "head_clamp": head_clamp,
           "ext": ext, "fingers": None}
    if fingers:
        q = np.zeros((N, len(fingers), 4))
        for k, (j, _bone) in enumerate(fingers):
            s = "L" if JOINTS55[j].startswith("left") else "R"
            a_f = bm.align[s][1]
            pj = PARENT[j]
            Wp = hand_world[s] if JOINTS55[pj].endswith("wrist") else G[:, pj]
            L = a_f.T @ (np.swapaxes(Wp, 1, 2) @ G[:, j]) @ a_f
            q[:, k] = mat_to_quat(L)
        out["fingers"] = q
    return out


def hips_of(trans, yaw, hu, fps, window, xmax, ymax):
    """(hips.x, hips.y) in hu: the translation in the facing frame (yaw
    off), its moving mean off, clipped."""
    p = np.einsum("nij,nj->ni", rot(1, -yaw), trans)
    win = int(round(window * fps)) | 1 if window else 0
    if win > 1:
        p = p - moving_mean(p, win)
    else:
        p = p - p.mean(0)
    return (np.clip(p[:, 0] / hu, -xmax, xmax),
            np.clip(p[:, 1] / hu, -ymax, ymax))


def column_spec(cols):
    """How each trajectory column comes from the fit: ("param", name),
    ("palm", side index, axis), ("hips", name) or ("nan",)."""
    out = []
    for c in cols:
        m = re.fullmatch(r"hand\.([LR])\.([xyz])", c)
        if c.startswith(NAN_PREFIX):
            out.append(("nan",))
        elif m:
            out.append(("palm", "LR".index(m.group(1)), "xyz".index(m.group(2))))
        elif c in ("hips.x", "hips.y"):
            out.append(("hips", c))
        elif c in vb.JOINTS:
            out.append(("param", c))
        else:
            out.append(("nan",))
    return out


def penetration(bm, caps):
    """[N] hu: the deepest overlap per frame of an arm with the trunk / head
    or with the other arm -- VRMBody.penetration's measure (the same
    capsules and signed distance), over all frames at once. caps[side] is
    one arm_caps list per frame."""
    N = len(caps["L"])
    pen = np.zeros(N)
    A = {}
    for s in "LR":
        A[s] = [(np.array([c[j][1] for c in caps[s]]),
                 np.array([c[j][2] for c in caps[s]]),
                 caps[s][0][j][3], caps[s][0][j][0])
                for j in range(len(caps[s][0]))]
        for a, b, r, _n in A[s]:
            u = np.linspace(0.0, 1.0, 9)[None, :, None]
            pts = (a[:, None] + u * (b - a)[:, None]).reshape(-1, 3)
            sd = bm.body_sdf(pts).reshape(N, 9)
            pen = np.maximum(pen, (r - sd).max(1))
    for a1, b1, r1, n1 in A["L"]:
        for a2, b2_, r2, n2 in A["R"]:
            if n1 == "upper" and n2 == "upper":
                continue
            p, q = vb._closest_batch(a1, b1, a2, b2_)
            pen = np.maximum(pen, r1 + r2 - np.linalg.norm(p - q, axis=1))
    return pen


def measure(bm, val, hips, cols, fps):
    """Per frame, from the fitted (clamped) values: {traj [N, V] float32
    (NaN where the column is the face's), palms [N, 2, 3] hu, pen [N]
    (the deepest arm-in-body or arm-in-arm overlap, hu), speed [N] (the
    faster palm, hu/s)}. The body's own forward model; tests compare it
    with VRMBody.perceive."""
    spec = column_spec(cols)
    N = len(next(iter(val.values())))
    base = bm.default_values()
    traj = np.full((N, len(cols)), np.nan, np.float32)
    palms = np.zeros((N, 2, 3))
    names = list(vb.JOINTS)
    cols_arr = {k: np.asarray(val[k], float) for k in names}
    caps = {s: [] for s in "LR"}
    for i in range(N):
        v = dict(base)
        for k in names:
            v[k] = float(cols_arr[k][i])
        W, P = bm.fk(v)
        pal = bm.palms(v, W, P)
        palms[i, 0], palms[i, 1] = pal["L"], pal["R"]
        if bm.coll:
            for s in "LR":
                caps[s].append(bm.arm_caps(s, v, W, P))
        for k, sp in enumerate(spec):
            if sp[0] == "param":
                traj[i, k] = v[sp[1]]
            elif sp[0] == "palm":
                traj[i, k] = palms[i, sp[1], sp[2]]
            elif sp[0] == "hips":
                traj[i, k] = hips[sp[1]][i]
    pen = penetration(bm, caps) if bm.coll and N else np.zeros(N)
    d = np.linalg.norm(np.diff(palms, axis=0), axis=2) * fps     # [N-1, 2]
    speed = np.zeros(N)
    if N > 1:
        sp = d.max(1)
        speed[1:] = sp
        speed[:-1] = np.maximum(speed[:-1], sp)
    return {"traj": traj, "palms": palms, "pen": pen, "speed": speed}


def limit_excess(fit_, bm):
    """[N] degrees: the most any joint is past its limit before the clamp
    (shoulders' units are 12 deg), and the shoulder's extension."""
    ex = np.zeros(len(fit_["ext"]))
    for k, x in fit_["raw"].items():
        lo, hi = float(bm.params[k]["min"]), float(bm.params[k]["max"])
        d = np.maximum(np.maximum(lo - x, x - hi), 0.0)
        if k.startswith("shoulder"):
            d = d * vb.SHOULDER_DEG
        ex = np.maximum(ex, d)
    return np.maximum(ex, fit_["ext"])


def retarget_clip(av, clip, cfg):
    """Everything per frame for one clip: {fit, meas, hips, facing, up,
    cols}. The clip is read (read_clip) but not yet cut."""
    bm = av.bm
    G = world_rotations(clip["poses"])
    up, up_name = find_up(G)
    if up_name != "y-up":
        G = world_rotations(clip["poses"], up)
        clip = dict(clip, trans=clip["trans"] @ up.T)
    yaw = facing_yaw(G, clip["fps"], cfg["facing_window"])
    Fr = rot(1, -yaw)
    G = Fr[:, None] @ G
    fit_ = fit(bm, G, av.fingers if cfg["fingers"] else ())
    hx, hy = hips_of(clip["trans"], yaw, bm.hu, clip["fps"],
                     cfg["facing_window"], cfg["hips_x_max"],
                     cfg["hips_y_max"])
    try:
        from . import dataset as ds
    except ImportError:
        from flow import dataset as ds
    cols = ds.traj_vars(bm)
    meas = measure(bm, fit_["val"], {"hips.x": hx, "hips.y": hy}, cols,
                   clip["fps"])
    return {"fit": fit_, "meas": meas, "cols": cols, "up": up_name,
            "yaw": yaw, "hips": (hx, hy),
            "limit": limit_excess(fit_, bm)}


# ------------------------------------------------------ filter / segments

def utterances(words, gap=0.35, min_len=3.0, max_len=15.0, join=1.5):
    """Groups of [(start, end, word)] cut at silences >= gap, over-long ones
    split at the longest silence near the middle (else at the word nearest
    it), short ones merged forward (or back) with a neighbour within `join`
    s when the sum fits."""
    if not words:
        return []
    groups, cur = [], [words[0]]
    for w in words[1:]:
        if w[0] - cur[-1][1] >= gap:
            groups.append(cur)
            cur = []
        cur.append(w)
    groups.append(cur)

    def dur(g):
        return g[-1][1] - g[0][0]

    def cut(g):
        if dur(g) <= max_len or len(g) < 2:
            return [g]
        mid = (g[0][0] + g[-1][1]) / 2
        best, bs = 1, None
        for i in range(1, len(g)):
            sil = g[i][0] - g[i - 1][1]
            ok = dur(g[:i]) >= min_len / 2 and dur(g[i:]) >= min_len / 2
            score = (ok, sil - 0.02 * abs(g[i][0] - mid))
            if bs is None or score > bs:
                best, bs = i, score
        return cut(g[:best]) + cut(g[best:])
    groups = [p for g in groups for p in cut(g)]
    out = []
    for g in groups:
        if out and dur(out[-1]) < min_len and \
                g[0][0] - out[-1][-1][1] <= join and \
                g[-1][1] - out[-1][0][0] <= max_len:
            out[-1] = out[-1] + g
        else:
            out.append(g)
    if len(out) > 1 and dur(out[-1]) < min_len and \
            out[-1][0][0] - out[-2][-1][1] <= join and \
            out[-1][-1][1] - out[-2][0][0] <= max_len:
        out[-2] = out[-2] + out.pop()
    return out


def runs(mask):
    """[(a, b)] half-open index ranges where mask is True."""
    m = np.concatenate([[0], np.asarray(mask, np.int8), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def bad_frames(meas, limit, fps, cfg):
    """(bad [N] bool, {cause: [N] bool}): the frames past a threshold, grown
    by cfg bad_pad s."""
    why = {"pen": meas["pen"] > cfg["max_pen"],
           "speed": meas["speed"] > cfg["max_speed"],
           "limits": limit > cfg["max_limit"]}
    bad = why["pen"] | why["speed"] | why["limits"]
    k = int(round(cfg["bad_pad"] * fps))
    if k and bad.any():
        bad = moving_mean(bad.astype(float), 2 * k + 1) > 0
    return bad, why


def plan_segments(words, n_frames, fps, bad, cfg):
    """[{u, piece, t0, t1, f0, f1, words, status, reason}]: utterances, cut
    at bad runs, the pieces under min_seg dropped (reason short / cut /
    audio)."""
    groups = utterances(words, cfg["gap"], cfg["min_seg"], cfg["max_seg"])
    end = n_frames / fps
    out = []
    for u, g in enumerate(groups):
        before = g[0][0] - (groups[u - 1][-1][1] if u else 0.0)
        after = (groups[u + 1][0][0] if u + 1 < len(groups) else
                 g[-1][1] + 2 * cfg["pad"]) - g[-1][1]
        t0 = max(g[0][0] - min(cfg["pad"], before / 2), 0.0)
        t1 = min(g[-1][1] + min(cfg["pad"], after / 2), end)
        f0, f1 = int(math.floor(t0 * fps + 1e-6)), int(math.ceil(t1 * fps - 1e-6))
        f1 = min(f1, n_frames)
        seg = {"u": u, "t0": f0 / fps, "t1": f1 / fps, "f0": f0, "f1": f1}
        if g[-1][1] > end + 1e-6:
            out.append(dict(seg, piece=0, words=[w for w in g if w[1] <= end],
                            status="dropped", reason="audio"))
            continue
        good = ~bad[f0:f1]
        if good.all():
            pieces = [(f0, f1, g, False)]
        else:
            pieces = []
            for a, b in runs(good):
                ws = [w for w in g if w[0] >= (f0 + a) / fps - 1e-6
                      and w[1] <= (f0 + b) / fps + 1e-6]
                if ws:
                    pieces.append((f0 + a, f0 + b, ws, True))
        if not pieces:
            out.append(dict(seg, piece=0, words=g, status="dropped",
                            reason="bad"))
        multi = len(pieces) > 1
        for k, (a, b, ws, cutp) in enumerate(pieces):
            if cutp:
                t0p = max(ws[0][0] - cfg["pad"], a / fps)
                t1p = min(ws[-1][1] + cfg["pad"], b / fps)
                a, b = int(math.ceil(t0p * fps - 1e-6)), int(
                    math.floor(t1p * fps + 1e-6))
            p = dict(seg, piece=k if multi or cutp else 0, f0=a, f1=b,
                     t0=a / fps, t1=b / fps, words=ws)
            if (b - a) / fps < cfg["min_seg"]:
                p.update(status="dropped",
                         reason="cut" if cutp else "short")
            else:
                p.update(status="kept", reason="")
            out.append(p)
    return out


def segment_id(clip, seg):
    s = f"s{seg['u']:02d}"
    if seg["piece"]:
        s += chr(ord("a") + seg["piece"])
    return f"{SOURCE}.{clip}/{s}"


def key_of(sid):
    return sid.replace("/", "__")


def stat(x):
    x = np.asarray(x, float)
    if not len(x):
        return {"max": 0.0, "p99": 0.0, "p95": 0.0, "mean": 0.0}
    return {"max": round(float(x.max()), 4),
            "p99": round(float(np.percentile(x, 99)), 4),
            "p95": round(float(np.percentile(x, 95)), 4),
            "mean": round(float(x.mean()), 4)}


# ---------------------------------------------------------------- export

def process_clip(av, info, cfg, parser=None, log=log):
    """One clip -> (report dict, rows, arrays) -- rows and arrays as
    flow.dataexport action's, plus the extras of the module docstring."""
    stem = info["stem"]
    bm = av.bm
    clip = read_clip(info["npz"])
    rows_tg = read_textgrid(info["textgrid"])
    words = words_of(rows_tg)
    wav, sr = read_wav(info["wav"])
    fps = clip["fps"]
    N = len(clip["poses"])
    rt = retarget_clip(av, clip, cfg)
    meas, fit_ = rt["meas"], rt["fit"]
    bad, why = bad_frames(meas, rt["limit"], fps, cfg)
    segs = plan_segments(words, N, fps, bad, cfg)
    sid_, name = speaker_of(stem)
    rep = {"clip": stem, "speaker": {"id": sid_, "name": name},
           "avatar": av.name, "frames": N, "seconds": round(N / fps, 2),
           "src_fps": clip["src_fps"], "fps_source": clip["fps_source"],
           "notes": clip["notes"], "up_axis": rt["up"],
           "audio_seconds": round(len(wav) / sr, 2),
           "audio_minus_motion": round(len(wav) / sr - N / fps, 3),
           "words": len(words),
           "facing_deg": {"mean": round(float(np.degrees(np.arctan2(
               np.sin(np.radians(rt["yaw"])).mean(),
               np.cos(np.radians(rt["yaw"])).mean()))), 1),
               "std": round(float(np.std(rt["yaw"])), 1),
               "window_s": cfg["facing_window"]},
           "torso_split_deg": stat(fit_["torso_split"]),
           "torso_clamp_deg": stat(fit_["torso_clamp"]),
           "head_clamp_deg": stat(fit_["head_clamp"]),
           "penetration_hu": dict(stat(meas["pen"]), over_tol=round(float(
               (meas["pen"] > vb.PEN_TOL).mean()), 4)),
           "palm_speed_hu_s": stat(meas["speed"]),
           "limit_excess_deg": stat(rt["limit"]),
           "limit_worst": {k: round(float(max(
               np.max(np.maximum(bm.params[k]["min"] - x, 0)),
               np.max(np.maximum(x - bm.params[k]["max"], 0)))), 1)
               for k, x in fit_["raw"].items()
               if np.max(np.maximum(bm.params[k]["min"] - x, 0)) > 0
               or np.max(np.maximum(x - bm.params[k]["max"], 0)) > 0},
           "bad_frames": {"any": int(bad.sum()),
                          **{k: int(v.sum()) for k, v in why.items()}}}
    pros = prosody(wav, sr, fps, N)
    if len(wav) / sr + 0.5 < N / fps:
        rep["notes"].append("audio shorter than the motion")
    rows, arrays, segrep = [], {}, []
    body_sha = sha16(av.body_path) if av.body_path else ""
    for sg in segs:
        sid = segment_id(stem, sg)
        a, b = sg["f0"], sg["f1"]
        ent = {"id": sid, "t0": round(sg["t0"], 3), "t1": round(sg["t1"], 3),
               "frames": b - a, "words": len(sg["words"]),
               "status": sg["status"], "reason": sg["reason"]}
        if b > a:
            ent.update(pen_max=round(float(meas["pen"][a:b].max()), 4),
                       speed_max=round(float(meas["speed"][a:b].max()), 2),
                       limit_max=round(float(rt["limit"][a:b].max()), 1),
                       bad_frac=round(float(bad[a:b].mean()), 3))
        if sg["status"] == "kept":
            spoken, chars = spoken_text(sg["words"], a / fps)
            if cfg["check_text"] and parser is not None:
                p = parser()
                for ch in spoken:
                    p.feed(ch)
                p.close()
                if p.spoken != spoken:
                    ent.update(status="dropped", reason="text")
            if ent["status"] == "kept":
                key = key_of(sid)
                arrays[key + ".traj"] = meas["traj"][a:b].astype(np.float16)
                arrays[key + ".chars"] = chars.astype(np.float16)
                arrays[key + ".joints"] = np.stack(
                    [fit_["val"][k][a:b] for k in vb.JOINTS], 1
                ).astype(np.float16)
                arrays[key + ".prosody"] = pros[a:b].astype(np.float16)
                if fit_["fingers"] is not None:
                    arrays[key + ".fingers"] = fit_["fingers"][a:b].astype(
                        np.float16)
                rows.append({
                    "id": sid, "key": key, "avatar": av.name, "kind": "vrm",
                    "body_sha": body_sha, "vars": rt["cols"], "fps": fps,
                    "frames": b - a, "traj": SOURCE, "user": "",
                    "spoken": spoken, "reply_raw": spoken, "markers": [],
                    "speech_seconds": round(sg["words"][-1][1] - a / fps, 3),
                    "lang": {"asked": None, "reply": _lang(spoken)},
                    "clip": stem, "speaker": name or sid_,
                    "t0": round(a / fps, 3), "t1": round(b / fps, 3),
                    "prosody": list(PROSODY_COLS), "joints": list(vb.JOINTS)})
        segrep.append(ent)
    rep["segments"] = segrep
    rep["kept_seconds"] = round(sum(s["frames"] for s in segrep
                                    if s["status"] == "kept") / fps, 2)
    rep["dropped"] = dict(Counter(s["reason"] for s in segrep
                                  if s["status"] != "kept"))
    return rep, rows, arrays


def _lang(text):
    han = sum("一" <= c <= "鿿" for c in text)
    kana = sum("぀" <= c <= "ヿ" for c in text)
    if kana:
        return "ja"
    return "zh" if han > len(text) / 4 else "en"


def write_npz(path, arrays):
    """np.savez_compressed, one array at a time (no copy of the lot)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for k, a in arrays.items():
            with z.open(k + ".npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(a),
                                          allow_pickle=False)


def write_part(out, stem, rows, arrays):
    d = os.path.join(out, "parts")
    os.makedirs(d, exist_ok=True)
    write_npz(os.path.join(d, stem + ".npz"), arrays)
    with open(os.path.join(d, stem + ".jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge(out, log=log) -> int:
    """<out>/parts -> <out>/action.jsonl and action.npz (the stem
    flow.dataexport action's --out names); returns the number of rows."""
    parts = sorted(glob.glob(os.path.join(out, "parts", "*.jsonl")))
    n = 0
    tmp = os.path.join(out, "action.npz.tmp")
    with open(os.path.join(out, "action.jsonl"), "w", encoding="utf-8") as fo, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED,
                            allowZip64=True) as zo:
        for p in parts:
            with open(p, encoding="utf-8") as fh:
                lines = [ln for ln in fh if ln.strip()]
            if not lines:
                continue
            with zipfile.ZipFile(p[:-6] + ".npz") as zi:
                for name in zi.namelist():
                    with zi.open(name) as src, zo.open(
                            name, "w", force_zip64=True) as dst:
                        while True:
                            buf = src.read(1 << 20)
                            if not buf:
                                break
                            dst.write(buf)
            fo.writelines(lines)
            n += len(lines)
    os.replace(tmp, os.path.join(out, "action.npz"))
    log(f"beat2: {n} segments -> {out}/action.jsonl, action.npz")
    return n


# ------------------------------------------------------------------- cli

_W = {}


def _init(vrm, body, cfg):
    _W["av"] = Avatar(vrm, body, log=lambda s: None)
    _W["cfg"] = cfg
    try:
        from . import pose as pz
    except ImportError:
        from flow import pose as pz
    bm = _W["av"].bm
    _W["parser"] = lambda: pz.parser_for(bm, gestures={}, moves=True)


def _job(info_out):
    info, out = info_out
    av, cfg = _W["av"], _W["cfg"]
    try:
        rep, rows, arrays = process_clip(av, info, cfg, _W["parser"])
    except (ValueError, KeyError, OSError, IndexError) as e:
        return info["stem"], {"clip": info["stem"], "error": str(e),
                              "segments": []}
    write_part(out, info["stem"], rows, arrays)
    rd = os.path.join(out, "reports")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, info["stem"] + ".json"), "w",
              encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    return info["stem"], rep


def check_out(out, allow=False):
    """--out must not be a place git would take: inside this repo only
    under data/ (gitignored)."""
    a = os.path.abspath(out)
    if a.startswith(ROOT + os.sep) and not allow and not a.startswith(
            os.path.join(ROOT, "data") + os.sep):
        raise SystemExit(f"beat2: {out} is inside the repository but not under "
                         "data/: retargeted BEAT2 must never be committed "
                         "(licence). Use data/beat2/<avatar>, or "
                         "--allow-outside-data")


def retarget(a, log=log) -> dict:
    check_out(a.out, a.allow_outside_data)
    clips = list_clips(a.beat2, a.speakers, a.limit)
    cfg = {**CFG, **{k: getattr(a, k) for k in CFG if hasattr(a, k)}}
    av = Avatar(a.vrm, a.body)
    os.makedirs(os.path.join(a.out, "reports"), exist_ok=True)
    meta = {"avatar": av.name, "vrm": a.vrm, "body": av.body_path,
            "body_sha": sha16(av.body_path) if av.body_path else None,
            "cfg": cfg, "beat2": os.path.abspath(a.beat2),
            "speakers": a.speakers, "source": SOURCE}
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    todo = [(c, a.out) for c in clips if a.force or not os.path.exists(
        os.path.join(a.out, "reports", c["stem"] + ".json"))]
    log(f"beat2: {len(clips)} clips ({len(clips) - len(todo)} done), avatar "
        f"{av.name}, {len(av.fingers)} finger bones"
        f"{'' if cfg['fingers'] else ' (not stored: --fingers)'}")
    jobs = a.jobs or 1
    res = []

    def note(i, stem, rep):
        if "error" in rep:
            log(f"beat2: {i}/{len(todo)} {stem}: ERROR {rep['error']}")
        else:
            log(f"beat2: {i}/{len(todo)} {stem}: kept {rep['kept_seconds']} s of "
                f"{rep['seconds']} s, pen max "
                f"{rep['penetration_hu']['max']} hu")
    if jobs > 1 and len(todo) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(jobs, _init, (a.vrm, a.body,
                                                        cfg)) as pool:
            for i, (stem, rep) in enumerate(pool.imap_unordered(_job, todo), 1):
                res.append(rep)
                note(i, stem, rep)
    else:
        _init(a.vrm, a.body, cfg)
        for i, t in enumerate(todo, 1):
            stem, rep = _job(t)
            res.append(rep)
            note(i, stem, rep)
    n = merge(a.out, log)
    s = summary(a.out)
    log(summary_text(s))
    s["rows"] = n
    return s


def load_reports(out):
    reps = []
    for p in sorted(glob.glob(os.path.join(out, "reports", "*.json"))):
        with open(p, encoding="utf-8") as fh:
            reps.append(json.load(fh))
    return reps


def summary(out) -> dict:
    if not os.path.isdir(out):
        raise SystemExit(f"beat2: {out} does not exist: run retarget first")
    reps = [r for r in load_reports(out) if "error" not in r]
    segs = [s for r in reps for s in r["segments"]]
    kept = [s for s in segs if s["status"] == "kept"]
    fps = FPS

    def q(xs, p):
        return round(float(np.percentile(xs, p)), 4) if len(xs) else None
    s = {"clips": len(reps), "segments": len(segs), "kept": len(kept),
         "seconds": round(sum(r["seconds"] for r in reps), 1),
         "kept_seconds": round(sum(x["frames"] for x in kept) / fps, 1),
         "dropped": dict(Counter(x["reason"] for x in segs
                                 if x["status"] != "kept")),
         "up_axis": dict(Counter(r["up_axis"] for r in reps)),
         "kept_pen_max": {p: q([x["pen_max"] for x in kept], p)
                          for p in (50, 90, 99, 100)},
         "all_pen_max": {p: q([r["penetration_hu"]["max"] for r in reps], p)
                         for p in (50, 90, 99, 100)},
         "speed_p99": {p: q([r["palm_speed_hu_s"]["p99"] for r in reps], p)
                       for p in (50, 90, 99, 100)},
         "limit_p99": {p: q([r["limit_excess_deg"]["p99"] for r in reps], p)
                       for p in (50, 90, 99, 100)},
         "worst_clips": sorted(
             ({"clip": r["clip"], "pen_max": r["penetration_hu"]["max"],
               "kept_s": r["kept_seconds"], "of_s": r["seconds"]}
              for r in reps), key=lambda x: -x["pen_max"])[:8],
         "errors": {r["clip"]: r["error"] for r in load_reports(out)
                    if "error" in r}}
    by = Counter()
    for r in reps:
        by[r["speaker"]["name"] or r["speaker"]["id"] or "?"] += r["kept_seconds"]
    s["by_speaker"] = {k: round(v, 1) for k, v in by.most_common()}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)
    return s


def summary_text(s) -> str:
    L = [f"{s['clips']} clips, {s['seconds'] / 3600:.2f} h; {s['kept']}/"
         f"{s['segments']} segments kept, {s['kept_seconds'] / 3600:.2f} h"]
    if s["dropped"]:
        L.append("dropped: " + ", ".join(f"{k} {v}" for k, v in
                                         s["dropped"].items()))
    L.append("up axis: " + ", ".join(f"{k} {v}" for k, v in
                                     s["up_axis"].items()))
    for k, lab in (("all_pen_max", "per-clip max penetration (hu)"),
                   ("kept_pen_max", "kept segments' max penetration (hu)"),
                   ("speed_p99", "per-clip p99 palm speed (hu/s)"),
                   ("limit_p99", "per-clip p99 limit excess (deg)")):
        L.append(f"{lab} at p50 / p90 / p99 / max: " + " / ".join(
            str(s[k][p]) for p in (50, 90, 99, 100)))
    if s["by_speaker"]:
        L.append("kept s by speaker: " + ", ".join(
            f"{k} {v}" for k, v in list(s["by_speaker"].items())[:12]))
    if s["worst_clips"]:
        L.append("worst by penetration: " + ", ".join(
            f"{w['clip']} {w['pen_max']} (kept {w['kept_s']}/{w['of_s']} s)"
            for w in s["worst_clips"][:5]))
    for k, e in list(s["errors"].items())[:5]:
        L.append(f"error {k}: {e}")
    return "\n".join(L)


def states_of(av, joints, hips, pros=None, fingers=None, fingers_bones=()):
    """[state] of fitted joints [F, 23] (vrmbody.JOINTS) for the renderer:
    VRMBody.state, the mouth following the audio's RMS, fingers added."""
    bm = av.bm
    out = []
    ref = 1.0
    if pros is not None and len(pros):
        ref = max(float(np.percentile(pros[:, 0], 95)), 1e-4)
    for i in range(len(joints)):
        v = {k: float(joints[i, c]) for c, k in enumerate(vb.JOINTS)}
        v["hips.x"], v["hips.y"] = float(hips[i, 0]), float(hips[i, 1])
        if pros is not None:
            v["mouth.open"] = float(np.clip(pros[i, 0] / ref, 0, 1)) * 0.8
        st = bm.state(v)
        if fingers is not None:
            for k, bone in enumerate(fingers_bones):
                st["bones"][bone] = [round(float(x), 5) for x in fingers[i, k]]
        out.append(st)
    return out


def segment_states(av, z, row):
    """(states, hips [F, 2]) of a stored segment's arrays."""
    key = row["key"]
    joints = z[key + ".joints"].astype(np.float32)
    traj = z[key + ".traj"].astype(np.float32)
    hips = np.zeros((len(joints), 2), np.float32)
    for c, v in enumerate(row["vars"]):
        if v in ("hips.x", "hips.y"):
            hips[:, ("hips.x", "hips.y").index(v)] = np.nan_to_num(traj[:, c])
    pros = z[key + ".prosody"].astype(np.float32) if key + ".prosody" in z.files \
        else None
    fing = z[key + ".fingers"].astype(np.float32) if key + ".fingers" in z.files \
        else None
    return states_of(av, joints, hips, pros, fing,
                     [b for _j, b in av.fingers]), hips


def preview(a, log=log) -> dict:
    """Render one stored segment (module docstring, preview)."""
    rows = []
    with open(os.path.join(a.out, "action.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    row = next((r for r in rows if r["id"] == a.id), None)
    if row is None:
        raise SystemExit(f"beat2: {a.id} not in {a.out}/action.jsonl "
                         f"({len(rows)} rows, e.g. {[r['id'] for r in rows[:3]]})")
    with open(os.path.join(a.out, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    av = Avatar(a.vrm or meta["vrm"], a.body or meta["body"])
    d = os.path.join(a.out, "preview", key_of(a.id))
    os.makedirs(d, exist_ok=True)
    with np.load(os.path.join(a.out, "action.npz")) as z:
        states, _h = segment_states(av, z, row)
    try:
        from . import vrm as fv
    except ImportError:
        from flow import vrm as fv
    sp = fv.save_states(os.path.join(d, "states.json"), states, row["fps"],
                        av.body_path, av.vrm_path,
                        extra={"source": a.id, "spoken": row["spoken"]})
    res = {"states": sp, "dir": d}
    wav = None
    if a.beat2:
        info = next((c for c in list_clips(a.beat2)
                     if c["stem"] == row["clip"]), None)
        if info:
            x, sr = read_wav(info["wav"])
            s0, s1 = int(row["t0"] * sr), int(row["t1"] * sr)
            wav = os.path.join(d, "speech.wav")
            write_wav(wav, x[s0:s1], sr)
    log(f"beat2: {len(states)} states -> {sp}")
    if a.no_render:
        return res
    try:
        from . import vrmrender as vr
    except ImportError:
        from flow import vrmrender as vr
    mp4 = os.path.join(d, "retarget.mp4")
    vr.render(states, av.bm, mp4, row["fps"], a.size, av.vrm_path)
    res["mp4"] = mp4
    if a.also_vrm:
        if not a.beat2:
            raise SystemExit("beat2: --also-vrm re-retargets the clip: "
                             "--beat2 DIR")
        av2 = Avatar(a.also_vrm, a.also_body)
        cfg = {**CFG, **meta.get("cfg", {}), "fingers": False}
        info = next(c for c in list_clips(a.beat2)
                    if c["stem"] == row["clip"])
        rt = retarget_clip(av2, read_clip(info["npz"]), cfg)
        f0, f1 = int(round(row["t0"] * row["fps"])), int(round(
            row["t1"] * row["fps"]))
        joints = np.stack([rt["fit"]["val"][k][f0:f1] for k in vb.JOINTS], 1)
        hips = np.stack([rt["hips"][0][f0:f1], rt["hips"][1][f0:f1]], 1)
        with np.load(os.path.join(a.out, "action.npz")) as z:
            pros = z[row["key"] + ".prosody"].astype(np.float32)
        st2 = states_of(av2, joints, hips, pros)
        mp4b = os.path.join(d, "also.mp4")
        vr.render(st2, av2.bm, mp4b, row["fps"], a.size, av2.vrm_path)
        side = os.path.join(d, "side.mp4")
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-i",
                        mp4b, "-filter_complex", "hstack=inputs=2", "-c:v",
                        "libx264", "-pix_fmt", "yuv420p", side], check=True)
        mp4 = side
        res["mp4"] = side
    if wav:
        try:
            from .director import mux_cmd
        except ImportError:
            from flow.director import mux_cmd
        dest = os.path.join(d, "demo.mp4")
        cmd = mux_cmd(mp4, wav, "", dest)
        if cmd:
            import subprocess
            subprocess.run(cmd, check=True, capture_output=True)
            res["demo"] = dest
    log("beat2: " + json.dumps(res))
    return res


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retarget", help="BEAT2 -> <out>/action.{jsonl,npz}")
    r.add_argument("--beat2", required=True, help="the BEAT2 folder")
    r.add_argument("--vrm", required=True)
    r.add_argument("--body", default=None, help="rigs/<name>.body.json "
                   "(default: the rigs/ file for this .vrm)")
    r.add_argument("--out", required=True, help="e.g. data/beat2/<avatar>")
    r.add_argument("--speakers", default=None, help="ids or names: 1,scott")
    r.add_argument("--limit", type=int, default=0, help="first N clips")
    r.add_argument("--jobs", type=int, default=1, help="processes")
    r.add_argument("--force", action="store_true", help="redo clips done")
    r.add_argument("--allow-outside-data", action="store_true")
    r.add_argument("--fingers", action="store_true",
                   help="store the fingers' rotations too (preview)")
    for k, h in (("facing_window", "s of the facing's moving mean (0: the "
                  "clip's mean)"), ("max_pen", "hu: a frame deeper is bad"),
                 ("max_speed", "hu/s: a palm faster is bad"),
                 ("max_limit", "deg: a joint further past its limit is bad"),
                 ("bad_pad", "s of bad around a bad frame"),
                 ("min_seg", "s: shortest segment kept"),
                 ("max_seg", "s: longest segment"),
                 ("gap", "s of silence between words that ends an utterance"),
                 ("pad", "s of motion either side of an utterance")):
        r.add_argument("--" + k.replace("_", "-"), dest=k, type=float,
                       default=CFG[k], help=f"{h} [{CFG[k]}]")
    p = sub.add_parser("report", help="summarise <out>/reports")
    p.add_argument("--out", required=True)
    m = sub.add_parser("merge", help="rebuild action.{jsonl,npz} from parts/")
    m.add_argument("--out", required=True)
    v = sub.add_parser("preview", help="render one segment")
    v.add_argument("--out", required=True)
    v.add_argument("--id", required=True, help="beat2.<clip>/sNN")
    v.add_argument("--beat2", default=None, help="for the audio and --also-vrm")
    v.add_argument("--vrm", default=None)
    v.add_argument("--body", default=None)
    v.add_argument("--also-vrm", default=None)
    v.add_argument("--also-body", default=None)
    v.add_argument("--size", type=int, default=384)
    v.add_argument("--no-render", action="store_true", help="states.json only")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(list(sys.argv[1:] if argv is None else argv))
    if a.cmd == "retarget":
        retarget(a)
    elif a.cmd == "report":
        print(summary_text(summary(a.out)))
    elif a.cmd == "merge":
        merge(a.out)
    else:
        preview(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
