"""
What each of a rig's gestures shows, for the director's prompt.

The director offers the rig's motion files as [gesture:NAME] with NAME the
file's basename. On Azur Lane rigs those names mean something (touch_head,
login, wedding), but the Cubism samples' do not (Hiyori: hiyori_m01..m10,
Natori: mtn_00..07): in a DeepSeek chat with Hiyori it announced "a big wave
just for you!" with [gesture:hiyori_m02] and "I'll dance for you!" with
[gesture:hiyori_m07], guessing blindly. So each offered gesture gets a short
line measured from the motion itself:

  hiyori_m02  both arms move widely; hands swing back and forth; 5.9 s, lively

Only what a directed gesture actually shows is described. The runtime's
GesturePlayer writes a motion's expressive parameters minus the bound role
columns (head, body, gaze and brows stay with the event layer, so a motion's
own nod is never seen), minus physics outputs, blink/breath and mouth
(runtime.base_keep), plus its part opacities. Hiyori's motions all nod and
sway, but in a directed gesture only her arms, hands, shoulders, legs and
cheek move: describing the nod would promise what never plays.

Parameters map to body parts by role first (a standard-id role column the
binding left to the gesture layer, events.drop_switch_roles), then by name
hints that cover Cubism ids and Azur Lane's pinyin-English ones (ParamArmLA,
PARAM_Arm_L_hand, PARAM_Hand_R3), then by the rig file's group (arm, head,
face). Sides come from the id (L/R, left/right) and are the character's own
as the rigger named them. Nothing is said about which way an arm goes: the
sign of an arm parameter is the rigger's choice (Hiyori's ParamArmLA rests at
its minimum, Natori's arm variants are part swaps), so "rises" would be a
guess. What is measured: how far each part moves from its rest (fraction of
the parameter's range), whether it goes back and forth (direction reversals
of at least OSC_STEP), part swaps against the rest pose, the duration as
played (capped at director.GESTURE_MAX) and the energy relative to the rig's
other motions. Where nothing visible moves the line says so, and parts the
hints cannot place are reported as "unclear".

Absolute lines were useless on Azur Lane rigs (runs/demo/gestures_al.txt):
their clips loop with the arms and hands swaying throughout and run to the
8 s cap, so 11 of aidang_2's 14 lines were the same "both arms swing back
and forth; hands swing back and forth; a part appears; 8.0 s", and lafei
ships one clip under six labels. So, per rig (describe_rig):

  - motions that play the same (identical files, or curves within DUP_TOL
    frame by frame) form one group; the prompt gives the group one line
    listing all its labels (director.gesture_lines), any of which parses;
  - each line keeps only what sets the gesture apart from the rig's typical
    one over its distinct motions (distinctive): features most of them show
    are dropped, a part moving notably more or less than usual, a rarer
    part, switch, effect or shown part, an unusual length and the energy
    stay; nothing left means the label alone (touch_head says more than a
    measured line could), except for numbered labels (hiyori_m01), which
    keep their full line;
  - switch parameters (rig_switches: eye open/close, pose swaps) are never
    called motion: "switches arm pose", "eyes change", "cheeks blush".

An optional per-rig file overrides or hides lines; see load_overrides.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import numpy as np

try:
    from . import events as ev
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import events as ev

# A part counts as moving when some parameter of it travels at least this
# far, in normalised units ([-1, 1] over the parameter's range: 0.15 is 7.5%
# of it). Hiyori's ParamLeg twitch in m10 is 0.10 (-0.1 of [-1, 1] range
# halved); every arm move in her motions is >= 0.29.
MOVE_MIN = 0.15
# Amplitude words, on the largest excursion from rest (normalised units):
# under SMALL "slightly", from WIDE "widely". Hiyori's arm swings span 0.29
# (m05's left arm, an idle drift) to 2.0 (her hands in m02/m03/m06).
SMALL = 0.35
WIDE = 1.0
# Back and forth: a direction reversal counts once the curve has come back
# at least OSC_STEP (normalised) from its last extreme, so key-frame
# overshoot and easing do not count. OSC_MIN reversals (a full there and
# back plus one more) make "back and forth"; a single raise-and-return has
# one.
OSC_STEP = 0.3
OSC_MIN = 3
# Energy words: mean per-frame speed (events.gesture_library's vel over the
# played parameters) against the median of the rig's own motions, the same
# split events.calm_motions makes by energy. Rig-relative on purpose:
# Natori's motions run 0.004-0.07 and Hiyori's 0.03-0.20 (runtime
# calm_library logs), so one absolute cut would call one rig all calm.
CALM_X = 0.67
LIVELY_X = 1.5
# At most this many part phrases per line, the most telling first, plus one
# for part swaps: ~10 words. Slight moves are left out when anything larger
# is there (Hiyori m07's brows at 0.27 next to both arms at 1.0).
MAX_PHRASES = 2

# Distinctive, not absolute. Azur Lane clips loop with the arms and hands
# swaying the whole time and a part shown, so on aidang_2 11 of 14 lines
# read "both arms swing back and forth; hands swing back and forth; a part
# appears; 8.0 s" (runs/demo/gestures_al.txt). With at least MIN_RIG
# distinct motions a line keeps only what sets the gesture apart from the
# rig's typical one: a feature at least COMMON of them show is dropped
# (aidang_2's arm sway, hand sway and shown part are in 11, 11 and 12 of
# its 14 old lines). On the Cubism samples (rig_stats) Hiyori's arms move
# in 9 of 10 and her shoulders in 7, Natori's right-arm switch is thrown in
# all 8 and her shoulders move in 7: those go, so what is said is how the
# arms move, a blush, brows, a leg or hand switch. Her hands (4 of 10) and
# Natori's body (4 of 8) stay.
MIN_RIG = 3
COMMON = 0.7
# A commonly moving part still counts when this gesture moves it at least
# BIG_X (or at most SMALL_X) times the rig's median amplitude for it, and
# by at least REL_MIN normalised units.
BIG_X = 1.5
SMALL_X = 0.5
REL_MIN = 0.3
# Length against the rig's median over its distinct motions: lafei's main_1
# (2.2 s) and aidang_2's touch_head (2.9 s) against clips that run to the
# 8 s cap; Hiyori's m07-m09 (1.6-2.1 s) against her 4.3 s median.
SHORT_X = 0.6
LONG_X = 1.6
# Two motions play the same when their frame counts differ by at most
# DUP_FRAMES and every played parameter stays within DUP_TOL (normalised,
# [-1, 1] over its range: 0.05 is 2.5% of it) and every part opacity
# within DUP_PART, frame by frame. Identical files always do. The closest
# distinct clips on the Cubism samples are 1.0 apart (Natori mtn_05/07) and
# 1.15 (Hiyori m01/m09) over the frames both play, 20x the tolerance, which
# leaves room for re-exported copies with rounded keys.
DUP_FRAMES = 2
DUP_TOL = 0.05
DUP_PART = 0.1

_SIDE_RE = re.compile(r"(?:^|[_a-z0-9])([LR])(?=[A-Z0-9_]|$)")
_SIDE_WORD = re.compile(r"(left|right)", re.I)
# Natori-style limb variants: Arm + variant letter + side (ParamArmAL01,
# ParamArmBRHand01, ParamArmCLHandRoll1).
_LIMB_SIDE = re.compile(r"(?:Arm|Hand|ARM|HAND)[A-Z]?([LR])(?=[A-Z0-9_]|$)")
_LOWER_SIDE = re.compile(r"(?:^|_)([lr])(?:_|\d|$)")

# Name hints, first match wins: (substrings of the lower-case id, part).
HINTS = (
    (("shoulder",), "shoulders"),
    (("hand", "finger", "wrist"), "hand"),
    (("arm", "elbow"), "arm"),
    (("leg", "foot", "knee"), "legs"),
    (("cheek", "blush"), "cheeks"),
    (("hair", "ahoge"), "hair"),
    (("body", "bust", "waist", "hip", "chest"), "body"),
    (("glass",), "glasses"),
    (("tear",), "tears"),
    (("brow",), "brows"),
    (("smile",), "eyes"),
    (("eye",), "eyes"),
    (("head", "angle"), "head"),
    (("fx", "effect"), "effect"),
)
# Parts left out of the line: hair (and the rig file's hair group) is
# secondary motion that sways in every motion -- Hiyori's ParamHairAhoge
# reverses 2-12 times in each of her ten -- so it tells motions apart by
# nothing, and it would dominate the energy (the largest per-frame change).
IGNORED = {"hair"}
# Rig file groups (rigs/<name>.npz "group") that name a part outright.
GROUP_PART = {"hair": "hair", "arm": "arm", "head": "head", "body": "body", "face": "face",
              "brow": "brows", "gaze": "eyes"}
# A role column the gesture layer plays (a switch-like standard id): its part.
ROLE_PART = {"gaze": "eyes", "head": "head", "body": "body", "brow": "brows"}
# Parts with sides: phrase per side ("left arm", "both hands").
SIDED = {"arm": "arms", "hand": "hands"}


def side_of(pid):
    """"L", "R" or "" from a parameter id: ParamArmLA, PARAM_Hand_R3,
    PARAM_Arm_L_hand, PARAM_arm_LFX, ParamLeftShoulderUp. An L or R counts
    only after a lower-case letter, digit or underscore and before a capital,
    digit, underscore or the end, so PARAM_ALL_moveZ and ParamRibbon have
    none."""
    m = _SIDE_WORD.search(pid)
    if m:
        return m.group(1)[0].upper()
    m = _SIDE_RE.search(pid) or _LIMB_SIDE.search(pid)
    if m:
        return m.group(1)
    m = _LOWER_SIDE.search(pid.lower())
    return m.group(1).upper() if m else ""


def part_of(pid, group="", role_of=None):
    """Body part a played parameter belongs to, or "" when unclear."""
    role = (role_of or {}).get(pid)
    if role:
        return ROLE_PART.get(ev.GROUP_OF.get(role, ""), "")
    low = pid.lower()
    for keys, part in HINTS:
        if any(k in low for k in keys):
            return part
    return GROUP_PART.get(group, "")


def reversals(x, step=OSC_STEP):
    """Direction reversals of x with hysteresis `step`: a turn counts once x
    has come back `step` from its extreme since the last turn."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return 0
    n, d = 0, 0
    hi = lo = x[0]
    for v in x[1:]:
        hi, lo = max(hi, v), min(lo, v)
        if d >= 0 and hi - v >= step:
            n += d > 0
            d, lo = -1, v
            hi = v
        elif d <= 0 and v - lo >= step:
            n += d < 0
            d, hi = 1, v
            lo = v
    return n


def _norm(v, p, lo, hi):
    """Values of parameter p in normalised units ([-1, 1] over its range)."""
    s = (hi[p] - lo[p]) / 2.0
    return (np.asarray(v, float) - lo[p]) / s - 1.0


def measure(m, keep, lo, hi, rest, base_parts=None, groups=None,
            role_of=None, switches=None):
    """What one gesture plays: (moves, swaps, flips).

    m: authored_motion dict. keep(pid): runtime's gesture keep. lo/hi/rest:
    {pid: value} in the rig's units (rest: the base pose, else default).
    switches: {pid: (a, b)} the two authored states of switch parameters
    (rig_switches); those are never measured as motion.

    moves: {(part, side): (amplitude, reversals)} over the continuous
    parameters. swaps: {part id: +1 shown / -1 hidden} against the rest
    opacities. flips: {(part, side): +1 / -1} for switches the gesture
    throws away from their rest state (+1 toward the higher state).
    """
    switches = switches or {}
    out, flips = {}, {}
    for p, v in sorted(m["params"].items()):
        if not keep(p) or p not in lo or hi[p] <= lo[p]:
            continue
        part = part_of(p, (groups or {}).get(p, ""), role_of)
        if part in IGNORED:
            continue
        side = side_of(p) if part in SIDED else ""
        if p in switches:
            a, b = sorted(switches[p])
            v = np.asarray(v, float)
            r = rest.get(p, v[0])
            up = np.abs(v - b) < np.abs(v - a)
            r_up = abs(r - b) < abs(r - a)
            if np.any(up != r_up):
                flips[(part, side)] = -1 if r_up else 1
            continue
        n = _norm(v, p, lo, hi)
        if np.ptp(n) < MOVE_MIN:
            continue
        r = _norm(rest.get(p, v[0]), p, lo, hi)
        amp = float(np.max(np.abs(n - r)))
        amp = max(amp, float(np.ptp(n)) / 2.0)
        if part == "cheeks":
            # Only a rise reads as a blush.
            amp = float(np.max(n - r))
            if amp < MOVE_MIN:
                continue
        key = (part, side)
        a0, k0 = out.get(key, (0.0, 0))
        out[key] = (max(a0, amp), max(k0, reversals(n)))
    swaps = {}
    for p, v in sorted(m.get("parts", {}).items()):
        v = np.asarray(v, float)
        r = float((base_parts or {}).get(p, v[0]))
        if np.max(v) - r >= 0.5:
            swaps[p] = 1
        elif r - np.min(v) >= 0.5:
            swaps[p] = -1
    return out, swaps, flips


def _amp_word(a):
    return " slightly" if a < SMALL else (" widely" if a >= WIDE else "")


PLURAL = {"glasses", "arms", "hands", "shoulders", "legs", "brows", "eyes", "cheeks",
          "tears"}
POSE_PARTS = {"arm", "hand", "legs"}


def atoms(meas, swaps, flips):
    """{atom: (amplitude, reversals)}: the features one gesture shows, the
    unit a rig's prevalence is counted in. ("m", part, side) a continuous
    part moves; ("o", part, side) it goes back and forth; ("s", part, side)
    a switch is thrown (an arm or hand part swap folds into the arm's
    switch, so "switches arm pose" never comes twice); ("appear",) /
    ("hide",) another part swap."""
    out = {}
    for (part, side), (a, k) in meas.items():
        out[("m", part, side)] = (a, k)
        if k >= OSC_MIN:
            out[("o", part, side)] = (a, k)
    for (part, side), d in flips.items():
        if part == "cheeks" and d < 0:
            continue                      # a blush fading is not shown
        out[("s", part, side)] = (1.0, 0)
    for p, d in swaps.items():
        low = p.lower()
        if "arm" in low or "hand" in low:
            out[("s", "arm", side_of(p))] = (1.0, 0)
        else:
            out[("appear",) if d > 0 else ("hide",)] = (1.0, 0)
    return out


def _subject(part, sides):
    """("both ", "arms") for a sided part's sides, ("", part) otherwise."""
    if part not in SIDED:
        return "", part
    if {"L", "R"} <= sides:
        return "both ", SIDED[part]
    if sides in ({"L"}, {"R"}):
        return ("left " if "L" in sides else "right "), part
    return "", SIDED[part]


# Phrase order: the most telling first. Back and forth, then a part moving
# more than the rig's other gestures, then plain moves, then parts the
# others move that this one keeps still.
_RANK = {"osc": 0, "big": 1, "move": 2, "small": 3, "still": 4}


def phrases(entries, switch_atoms):
    """Phrases for one gesture. entries: [(kind, part, side, amp, k)] with
    kind in _RANK; switch_atoms: the atoms() keys of thrown switches and
    part swaps. At most MAX_PHRASES part phrases, then switch phrases, then
    parts kept still while room is left, MAX_PHRASES + 1 in all."""
    grouped = {}
    for kind, part, side, a, k in entries:
        g = grouped.setdefault((kind, part), [set(), 0.0, 0])
        g[0].add(side)
        g[1], g[2] = max(g[1], a), max(g[2], k)
    subj = {}
    for (kind, part), (sides, a, k) in sorted(grouped.items()):
        subj[(kind, part)] = _subject(part, sides) + (a, k)
    # Arms and hands on the same side(s) are one phrase: "both arms and
    # hands swing back and forth" (Hiyori m03), not two.
    for kind in _RANK:
        ar, hd = subj.get((kind, "arm")), subj.get((kind, "hand"))
        if ar and hd and ar[0] == hd[0]:
            del subj[(kind, "hand")]
            subj[(kind, "arm")] = (ar[0], f"{ar[1]} and {hd[1]}",
                                   max(ar[2], hd[2]), max(ar[3], hd[3]))
    items = sorted(((_RANK[kind], -round(a, 3), pre + word, kind, part)
                    for (kind, part), (pre, word, a, k) in subj.items()))
    # Slight moves are left out when anything larger is there (Hiyori m07's
    # brows at 0.27 next to both arms at 1.0).
    if any(kind != "move" or -na >= SMALL for _, na, _, kind, _ in items):
        items = [x for x in items if x[3] != "move" or -x[1] >= SMALL]
    out, still, unclear = [], [], False
    for _, na, s, kind, part in items:
        a = -na
        if not part:
            unclear = unclear or kind in ("osc", "move", "big")
            continue
        plural = " and " in s or s.split()[-1] in PLURAL
        if part == "cheeks" and kind in ("osc", "move", "big"):
            out.append("cheeks blush")
        elif part == "effect" and kind in ("osc", "move", "big"):
            out.append("an effect plays")
        elif kind == "osc":
            out.append(f"{s} {'swing' if plural else 'swings'} back and forth")
        elif kind == "move":
            out.append(f"{s} {'move' if plural else 'moves'}" + _amp_word(a))
        elif kind == "big":
            out.append(f"{s} {'move' if plural else 'moves'} more than "
                       "in the other gestures")
        elif kind == "small":
            # "barely" only where the move is slight outright: Hiyori m01's
            # arms (0.35-1.0, half her median) read as near still.
            out.append(f"{s} barely {'move' if plural else 'moves'}"
                       if a < SMALL else
                       f"{s} {'move' if plural else 'moves'} less than in "
                       "the other gestures")
        else:
            still.append(f"{s} {'stay' if plural else 'stays'} still")
    if unclear:
        out.append("unclear parts move")
    out = out[:MAX_PHRASES]
    sw = []
    sides = {}
    for at in sorted(switch_atoms):
        if at[0] == "s":
            sides.setdefault(at[1], set()).add(at[2])
    if "arm" in sides and "hand" in sides:
        sides["arm"] |= sides.pop("hand")
    for part in sorted(sides, key=lambda p: (p not in POSE_PARTS, p)):
        if part in POSE_PARTS:
            pre, _ = _subject(part, sides[part])
            pre = "" if pre == "both " else pre
            word = {"legs": "leg"}.get(part, part)
            sw.append(f"switches {pre}{word} pose")
        elif part == "cheeks":
            sw.append("cheeks blush")
        elif part == "effect":
            sw.append("an effect plays")
        elif part:
            pl = part in PLURAL
            sw.append(f"{part} {'change' if pl else 'changes'}")
        else:
            sw.append("a part switches")
    if ("appear",) in switch_atoms:
        sw.append("a part appears")
    if ("hide",) in switch_atoms:
        sw.append("a part hides")
    # A part kept still is the weakest thing to say: only where room is
    # left (Hiyori m05, Natori mtn_00: nothing else sets them apart).
    return (out + sw + still[:max(MAX_PHRASES - len(out) - len(sw), 0)]
            )[:MAX_PHRASES + 1]


def _cut(motions, fps, max_seconds):
    """The motions as a directed gesture plays them: cut at max_seconds
    (director.GESTURE_MAX)."""
    if max_seconds is None:
        return list(motions)
    T = max(int(round(max_seconds * fps)), 2)
    return [(n, {"params": {p: v[:T] for p, v in m["params"].items()},
                 "parts": {p: v[:T] for p, v in m["parts"].items()},
                 "frames": min(int(m["frames"]), T)})
            for n, m in motions]


def _same(a, b, keep, lo, hi, rest, base_parts):
    """Whether two (cut) motions play the same: frame counts within
    DUP_FRAMES and every played parameter within DUP_TOL (normalised) and
    every part opacity within DUP_PART, frame by frame. A parameter one of
    them does not key is compared against its rest value. No time
    stretching: a slower copy is a different gesture."""
    if abs(int(a["frames"]) - int(b["frames"])) > DUP_FRAMES:
        return False
    T = min(int(a["frames"]), int(b["frames"]))
    for p in sorted(set(a["params"]) | set(b["params"])):
        if not keep(p) or p not in lo or hi[p] <= lo[p]:
            continue
        r = rest.get(p, lo[p])
        va = np.asarray(a["params"].get(p, np.full(T, r)), float)[:T]
        vb = np.asarray(b["params"].get(p, np.full(T, r)), float)[:T]
        if np.max(np.abs(_norm(va, p, lo, hi) - _norm(vb, p, lo, hi))) > DUP_TOL:
            return False
    for p in sorted(set(a["parts"]) | set(b["parts"])):
        r = (base_parts or {}).get(p)
        if r is None and (p not in a["parts"] or p not in b["parts"]):
            continue
        va = np.asarray(a["parts"].get(p, np.full(T, r)), float)[:T]
        vb = np.asarray(b["parts"].get(p, np.full(T, r)), float)[:T]
        if np.max(np.abs(va - vb)) > DUP_PART:
            return False
    return True


def same_motions(motions, keep, lo, hi, rest, base_parts=None, hashes=None,
                 fps=30.0, max_seconds=None):
    """{motion name: the name of its group}: motions that play the same
    gesture, by identical file (hashes: {name: digest}) or by _same over
    what plays (cut at max_seconds). The group is named by its smallest
    member, so the result is the same whatever order the motions come in.
    lafei ships one 6.5 s clip under complete, home, mail, mission,
    mission_complete and touch_head; offered as six gestures, the model
    would think it is choosing between them."""
    ms = dict(_cut(motions, fps, max_seconds))
    names = sorted(ms)
    parent = {n: n for n in names}

    def root(n):
        while parent[n] != n:
            n = parent[n]
        return n

    for i, x in enumerate(names):
        for y in names[i + 1:]:
            if root(x) == root(y):
                continue
            hx, hy = (hashes or {}).get(x), (hashes or {}).get(y)
            if (hx is not None and hx == hy) or _same(
                    ms[x], ms[y], keep, lo, hi, rest, base_parts):
                rx, ry = root(x), root(y)
                parent[max(rx, ry)] = min(rx, ry)
    return {n: root(n) for n in names}


def label_of(name):
    """The director's gesture label for a motion name (director.
    gesture_labels): basename without .motion3.json, lowercased."""
    b = os.path.basename(name)
    return re.sub(r"\.motion3\.json$", "", b, flags=re.I).lower()


def opaque_labels(names):
    """Labels that say nothing about their motion: numbered series (the
    letters left once digits and separators go are shared with another
    label: hiyori_m01..m10, mtn_00..07, lafei's main_1..3) or three letters
    or fewer. For these a line is never left empty."""
    stems = {}
    for n in names:
        lab = label_of(n)
        stems.setdefault(re.sub(r"[\d_\-\s.]+", "", lab), []).append(lab)
    out = set()
    for stem, labs in stems.items():
        if len(labs) > 1 or len(stem) <= 3:
            out.update(labs)
    return out


def describe_rig(motions, keep, lo, hi, rest, base_parts=None, groups=None,
                 role_of=None, fps=30.0, max_seconds=None, switches=None,
                 hashes=None):
    """({motion name: description}, {motion name: group name}).

    Measured over what plays: a directed gesture is cut at max_seconds
    (director.GESTURE_MAX), so the frames after it are dropped first.
    Motions that play the same are grouped (same_motions) and each group
    counts once in the rig's statistics. With at least MIN_RIG distinct
    motions a line says what sets the gesture apart from the rig's others
    (distinctive()); an empty line ("") means nothing does, and the label
    alone is the better description -- except for opaque labels
    (opaque_labels), which keep their full measured line. Deterministic:
    the same motions and rig give the same lines, whatever their order."""
    switches = switches or {}
    motions = sorted(_cut(motions, fps, max_seconds), key=lambda x: x[0])
    same = same_motions(motions, keep, lo, hi, rest, base_parts, hashes)
    reps = sorted(set(same.values()))

    def ekeep(p):
        return (keep(p) and p not in switches
                and part_of(p, (groups or {}).get(p, ""), role_of)
                not in IGNORED)

    lib = ev.gesture_library(motions, ekeep, lo, hi, {}, fps)
    energy = {g["name"]: float(np.mean(g["vel"])) for g in lib}
    vals = sorted(energy[n] for n in reps if energy.get(n, 0) > 0)
    med_e = float(np.median(vals)) if vals else 0.0
    secs = {n: m["frames"] / fps for n, m in motions}
    med_s = float(np.median([secs[n] for n in reps])) if reps else 0.0
    feats = {}
    for name, m in motions:
        meas, swaps, flips = measure(m, keep, lo, hi, rest, base_parts,
                                     groups, role_of, switches)
        feats[name] = (meas, atoms(meas, swaps, flips))
    rig = rig_stats([feats[n][1] for n in reps])
    opaque = opaque_labels([n for n, _ in motions])
    out = {}
    for name, _m in motions:
        meas, at = feats[name]
        sec = secs[name]
        if not at:
            out[name] = f"little visible motion; {sec:.1f} s"
            continue
        e = energy.get(name, 0.0)
        mood = ""
        if med_e > 0:
            if e <= CALM_X * med_e:
                mood = ", calm"
            elif e >= LIVELY_X * med_e:
                mood = ", lively"
        full = phrases(*absolute(at))
        if len(reps) < MIN_RIG:
            out[name] = "; ".join(full + [f"{sec:.1f} s{mood}"])
            continue
        entries, sw = distinctive(at, rig)
        bits = phrases(entries, sw)
        length = ""
        if med_s > 0 and sec <= SHORT_X * med_s:
            length = ", short"
        elif med_s > 0 and sec >= LONG_X * med_s:
            length = ", long"
        # A numbered label says nothing, so its line must say what moves:
        # where only parts kept still or moved less are distinctive, the
        # measured phrases lead (Hiyori m01: "shoulders move; both arms
        # move", not "both arms move less than in the other gestures").
        positive = sw or any(e[0] in ("osc", "big", "move") for e in entries)
        # Parts kept still are absent from it, so they cannot contradict
        # it; "moves less" can (Hiyori m05: "both arms move widely; left
        # arm barely moves"), so it goes.
        if label_of(name) in opaque and not positive and full:
            rest_ = phrases([e for e in entries if e[0] == "still"], set())
            lead = full[:MAX_PHRASES]
            bits = (lead + [b for b in rest_ if b not in lead]
                    )[:MAX_PHRASES + 1]
        if bits or length or mood:
            out[name] = "; ".join(bits + [f"{sec:.1f} s{length}{mood}"])
        else:
            out[name] = ""
    return out, same


def describe_motions(motions, keep, lo, hi, rest, base_parts=None,
                     groups=None, role_of=None, fps=30.0, max_seconds=None,
                     switches=None, hashes=None):
    """{motion name: description}: describe_rig without the groups."""
    return describe_rig(motions, keep, lo, hi, rest, base_parts, groups,
                        role_of, fps, max_seconds, switches, hashes)[0]


def rig_stats(atom_sets):
    """The rig's typical gesture over its distinct motions: {atom: fraction
    of them that show it} and {("m", part, side): median amplitude where
    it moves}."""
    n = max(len(atom_sets), 1)
    count, amps = {}, {}
    for at in atom_sets:
        for k, (a, _) in at.items():
            count[k] = count.get(k, 0) + 1
            if k[0] == "m":
                amps.setdefault(k, []).append(a)
    return ({k: c / n for k, c in count.items()},
            {k: float(np.median(v)) for k, v in amps.items()})


def absolute(at):
    """(entries, switch atoms) for phrases(): everything the gesture shows."""
    entries = []
    for k, (a, r) in sorted(at.items()):
        if k[0] == "m":
            kind = "osc" if ("o",) + k[1:] in at else "move"
            entries.append((kind, k[1], k[2], a, r))
    return entries, {k for k in at if k[0] in ("s", "appear", "hide")}


def distinctive(at, rig):
    """(entries, switch atoms) for phrases(): only what sets the gesture
    apart from the rig's typical one (rig_stats). A feature at least
    COMMON of the rig's distinct motions show is dropped -- aidang_2's arm
    sway, hand sway and shown part are in 10-12 of its 14 clips -- unless
    this gesture moves that part notably more (BIG_X) or less (SMALL_X)
    than the rig's median; a part the rig commonly moves and this gesture
    does not is said to stay still."""
    prev, med = rig
    entries = []
    for k, (a, r) in sorted(at.items()):
        if k[0] != "m":
            continue
        _, part, side = k
        o = ("o", part, side)
        common = prev.get(k, 0) >= COMMON
        if o in at and (not common or prev.get(o, 0) < COMMON):
            entries.append(("osc", part, side, a, r))
        elif not common:
            entries.append(("move", part, side, a, r))
        elif a >= BIG_X * med[k] and a - med[k] >= REL_MIN:
            entries.append(("big", part, side, a, r))
        elif a <= SMALL_X * med[k] and med[k] - a >= REL_MIN:
            entries.append(("small", part, side, a, r))
    for k, f in sorted(prev.items()):
        if (k[0] == "m" and f >= COMMON and k not in at
                and k[1] not in ("", "effect", "cheeks")):
            entries.append(("still", k[1], k[2], med[k], 0))
    sw = {k for k in at if k[0] in ("s", "appear", "hide")
          and prev.get(k, 0) < COMMON}
    return entries, sw


# ---------------------------------------------------------------- overrides

def override_path(rig):
    """Where a rig's override file is looked for when --gestures is not
    given: next to the rig file, rigs/<name>.gestures.json for
    rigs/<name>.npz (the rig file is the per-rig thing every run names;
    the model directory is often a read-only download). The repository's
    .gitignore ignores both /rigs and *.json: commit one with git add -f."""
    return rig.removesuffix(".npz") + ".gestures.json"


def load_overrides(path):
    """{label: {"desc": str or None, "hide": bool}} from a JSON object
    mapping gesture label (case-insensitive, as offered) to a description
    string, or to {"desc": "...", "hide": true/false}. Missing file: {}."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    if not isinstance(blob, dict):
        raise ValueError(f"{path}: expected a JSON object label -> text")
    out = {}
    for k, v in blob.items():
        if isinstance(v, str):
            out[str(k).lower()] = {"desc": v.strip(), "hide": False}
        elif isinstance(v, dict):
            d = v.get("desc", v.get("description"))
            out[str(k).lower()] = {"desc": (str(d).strip() if d else None),
                                   "hide": bool(v.get("hide", False))}
        else:
            raise ValueError(f"{path}: {k!r}: expected text or an object")
    return out


def apply_overrides(labels, measured, overrides):
    """(labels, {label: description}) with overrides applied: a hidden
    label is dropped from both, an override text wins over the measured
    one. labels: {label: motion name}; measured: {motion name: text}."""
    keep = {}
    desc = {}
    for lab, name in labels.items():
        o = overrides.get(lab, {})
        if o.get("hide"):
            continue
        keep[lab] = name
        d = o.get("desc") or measured.get(name)
        if d:
            desc[lab] = d
    return keep, desc


def rig_descriptions(model_json, rig, roles, clips=None, fps=30.0,
                     max_seconds=None, probe_dir="out", log=print):
    """({motion name: description}, {motion name: group name}) for a rig
    on disk (describe_rig), measured on what the
    runtime's GesturePlayer will play: its expressive columns (rig_columns,
    following --clips as the runtime does) minus the bound `roles`
    ({role: (pid, sign, source)}, runtime.event_roles), through base_keep,
    minus the whole-model transforms it holds still (runtime.held_params:
    PARAM_bodyS, PARAM_moveX/Y on aidang_2, which would read as "body
    moves"), from the base pose base_state takes (the idle's first
    frame). Switches are rig_switches'; identical motion files are one
    group whatever their curves."""
    try:
        from .runtime import (authored_motions, base_keep, base_state,
                              held_params, rig_columns)
        from ..l2dprobe.motions import motion_files
    except ImportError:
        from flow.runtime import (authored_motions, base_keep, base_state,
                                  held_params, rig_columns)
        from l2dprobe.motions import motion_files
    col = rig_columns(rig, model_json, clips, log=lambda s: None)
    pids = col["pids"]
    expressive = {pids[i] for i in col["expressive"]}
    held = held_params(rig, probe_dir, pids, col["groups"],
                       col["expressive"], warn=False)
    role_ids = {v[0] for v in roles.values()} | held
    bk = base_keep(role_ids, col["phys"], dict(zip(pids, col["bands"])))

    def keep(p):
        return p in expressive and bk(p)

    lo = dict(zip(pids, col["lo"].astype(float).tolist()))
    hi = dict(zip(pids, col["hi"].astype(float).tolist()))
    rest = dict(zip(pids, col["default"].astype(float).tolist()))
    motions = authored_motions(model_json, fps)
    _, base_params, base_parts = base_state(motions, keep)
    rest.update(base_params)
    groups = dict(zip(pids, col["groups"]))
    # Standard-id columns left unbound (switch-like, drop_switch_roles)
    # are played by the gesture layer: name their part by role.
    std = {sid.lower(): r for r, sid in ev.ALL_ROLES.items()}
    role_of = {p: std[p.lower().replace("_", "")] for p in pids
               if p.lower().replace("_", "") in std}
    switches = rig_switches(model_json, pids, col["groups"], fps,
                            role_of)
    hashes = {}
    for path in motion_files(model_json):
        with open(path, "rb") as fh:
            hashes[os.path.basename(path)] = hashlib.sha1(fh.read()).hexdigest()
    return describe_rig(motions, keep, lo, hi, rest, base_parts, groups,
                        role_of, fps, max_seconds, switches, hashes)


def rig_switches(model_json, pids, groups, fps=30.0, role_of=None):
    """{pid: (a, b)}: the parameters a gesture throws between two held
    states rather than moves, which a line must never call motion. The
    runtime's switch test (motions.toggle_params) and its snap rule
    (runtime.snappable: arm and hand rotations an animator parks stay
    continuous), plus the eyes: the runtime never snaps gaze (snapping froze
    aidang_2's eyes), but on Azur Lane rigs the eyes are open/close and
    look switches held at their extremes, which read as "eyes swing back
    and forth" in 7 of aidang_2's 14 lines."""
    try:
        from .runtime import snappable
        from ..l2dprobe.motions import toggle_params
    except ImportError:
        from flow.runtime import snappable
        from l2dprobe.motions import toggle_params
    found = toggle_params(model_json, fps)
    group = dict(zip(pids, groups))
    out = {}
    for p in pids:
        if p not in found:
            continue
        if (snappable(p, group.get(p, ""))
                or part_of(p, group.get(p, ""), role_of) == "eyes"):
            out[p] = (found[p][1], found[p][2])
    return out
