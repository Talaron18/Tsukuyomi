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

An optional per-rig file overrides or hides lines; see load_overrides.
"""

from __future__ import annotations

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


def measure(m, keep, lo, hi, rest, base_parts=None, groups=None,
            role_of=None):
    """{part key: (amplitude, reversals)} over the parameters a gesture
    plays, plus part swaps against the rest opacities.

    m: authored_motion dict. keep(pid): runtime's gesture keep. lo/hi/rest:
    {pid: value} in the rig's units (rest: the base pose, else default).
    Part keys are (part, side). Swaps come back as ("swap", part id) with
    +1 (shown) or -1 (hidden).
    """
    out = {}
    for p, v in m["params"].items():
        if not keep(p) or p not in lo or hi[p] <= lo[p]:
            continue
        v = np.asarray(v, float)
        s = (hi[p] - lo[p]) / 2.0
        n = (v - lo[p]) / s - 1.0
        if np.ptp(n) < MOVE_MIN:
            continue
        r = (rest.get(p, v[0]) - lo[p]) / s - 1.0
        amp = float(np.max(np.abs(n - r)))
        amp = max(amp, float(np.ptp(n)) / 2.0)
        part = part_of(p, (groups or {}).get(p, ""), role_of)
        if part in IGNORED:
            continue
        if part == "cheeks":
            # Only a rise reads as a blush.
            amp = float(np.max(n - r))
            if amp < MOVE_MIN:
                continue
        key = (part, side_of(p) if part in SIDED else "")
        a0, k0 = out.get(key, (0.0, 0))
        out[key] = (max(a0, amp), max(k0, reversals(n)))
    swaps = {}
    for p, v in m.get("parts", {}).items():
        v = np.asarray(v, float)
        r = float((base_parts or {}).get(p, v[0]))
        if np.max(v) - r >= 0.5:
            swaps[p] = 1
        elif r - np.min(v) >= 0.5:
            swaps[p] = -1
    return out, swaps


def _amp_word(a):
    return " slightly" if a < SMALL else (" widely" if a >= WIDE else "")


PLURAL = {"glasses", "arms", "hands", "shoulders", "legs", "brows", "eyes", "cheeks",
          "tears"}


def phrase(meas, swaps):
    """Body-part phrases for one motion (measure()'s output), the largest
    movement first, at most MAX_PHRASES plus one for part swaps."""
    items, sided = [], {}
    for (part, side), ak in meas.items():
        if part in SIDED:
            sided.setdefault(part, {})[side] = ak
        else:
            items.append((part, part, ak[0], ak[1]))
    subj = {}
    for part, by in sided.items():
        a = max(x[0] for x in by.values())
        k = max(x[1] for x in by.values())
        if {"L", "R"} <= set(by):
            pre, word = "both ", SIDED[part]
        elif set(by) in ({"L"}, {"R"}):
            pre, word = ("left " if "L" in by else "right "), part
        else:
            pre, word = "", SIDED[part]
        subj[part] = (pre, word, a, k)
    # Arms and hands on the same side(s) are one phrase: "both arms and
    # hands swing back and forth" (Hiyori m03), not two.
    if "arm" in subj and "hand" in subj and subj["arm"][0] == subj["hand"][0]:
        pa, wa, aa, ka = subj.pop("arm")
        _, wh, ah, kh = subj.pop("hand")
        subj["arm"] = (pa, f"{wa} and {wh}", max(aa, ah), max(ka, kh))
    for part, (pre, word, a, k) in subj.items():
        items.append((part, pre + word, a, k))
    # Back and forth first (the most telling), then by amplitude; ties by
    # name so the line is byte-stable.
    items.sort(key=lambda x: (x[3] < OSC_MIN, -round(x[2], 3), x[1]))
    if any(a >= SMALL or k >= OSC_MIN for _, _, a, k in items):
        items = [x for x in items if x[2] >= SMALL or x[3] >= OSC_MIN]
    out, unclear = [], 0
    for part, subj, a, k in items:
        if not part:
            unclear += 1
            continue
        if part == "cheeks":
            out.append("cheeks blush")
        elif part == "effect":
            out.append("an effect plays")
        else:
            plural = " and " in subj or subj.split()[-1] in PLURAL
            if k >= OSC_MIN:
                out.append(f"{subj} {'swing' if plural else 'swings'} "
                           "back and forth")
            else:
                out.append(f"{subj} {'move' if plural else 'moves'}"
                           + _amp_word(a))
    if unclear:
        out.append("unclear parts move")
    out = out[:MAX_PHRASES]
    if swaps:
        ids = " ".join(sorted(swaps)).lower()
        if "arm" in ids or "hand" in ids:
            out.append("switches arm pose")
        elif any(s > 0 for s in swaps.values()):
            out.append("a part appears")
        else:
            out.append("a part hides")
    return out


def describe_motions(motions, keep, lo, hi, rest, base_parts=None,
                     groups=None, role_of=None, fps=30.0, max_seconds=None):
    """{motion name: description} for (name, authored_motion) pairs.

    Measured over what plays: a directed gesture is cut at max_seconds
    (director.GESTURE_MAX), so the frames after it are dropped first.
    Deterministic: the same motions and rig give the same lines, whatever
    their order."""
    if max_seconds is not None:
        T = max(int(round(max_seconds * fps)), 2)
        motions = [(n, {"params": {p: v[:T] for p, v in m["params"].items()},
                        "parts": {p: v[:T] for p, v in m["parts"].items()},
                        "frames": min(int(m["frames"]), T)})
                   for n, m in motions]
    energy = {}
    def ekeep(p):
        return keep(p) and part_of(p, (groups or {}).get(p, ""),
                                   role_of) not in IGNORED

    lib = ev.gesture_library(motions, ekeep, lo, hi, {}, fps)
    for g in lib:
        energy[g["name"]] = float(np.mean(g["vel"]))
    vals = sorted(e for e in energy.values() if e > 0)
    med = float(np.median(vals)) if vals else 0.0
    out = {}
    for name, m in motions:
        meas, swaps = measure(m, keep, lo, hi, rest, base_parts, groups,
                              role_of)
        bits = phrase(meas, swaps)
        sec = m["frames"] / fps
        e = energy.get(name, 0.0)
        if not bits:
            out[name] = f"little visible motion; {sec:.1f} s"
            continue
        mood = ""
        if med > 0:
            if e <= CALM_X * med:
                mood = ", calm"
            elif e >= LIVELY_X * med:
                mood = ", lively"
        out[name] = "; ".join(bits) + f"; {sec:.1f} s{mood}"
    return out


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
    """{motion name: description} for a rig on disk, measured on what the
    runtime's GesturePlayer will play: its expressive columns (rig_columns,
    following --clips as the runtime does) minus the bound `roles`
    ({role: (pid, sign, source)}, runtime.event_roles), through base_keep,
    minus the whole-model transforms it holds still (runtime.held_params:
    PARAM_bodyS, PARAM_moveX/Y on aidang_2, which would read as "body
    moves"), from the base pose base_state takes (the idle's first
    frame)."""
    try:
        from .runtime import (authored_motions, base_keep, base_state,
                              held_params, rig_columns)
    except ImportError:
        from flow.runtime import (authored_motions, base_keep, base_state,
                                  held_params, rig_columns)
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
    return describe_motions(motions, keep, lo, hi, rest, base_parts, groups,
                            role_of, fps, max_seconds)
