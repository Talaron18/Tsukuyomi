#!/usr/bin/env python3
"""
Director: a language model writes the character's spoken reply with inline
action tags; the tags become event tokens (flow.events' format), merged into
the fitted idle stream and rendered by flow.runtime.

Stage 1 of the VLA-style character. The language model never sees parameters
or numbers: it picks from a small fixed vocabulary of tags and places them in
its own sentence, which is the only timing it is good at. Everything metric --
how far a look goes, how long a move takes, how long a pose is held -- comes
from the rig's fitted statistics (the same `--events` file the idle uses), so
a directed glance moves like the idle glances around it.

  reply      "Oh, [brow:up] really? [look:away] I hadn't noticed. [nod]"
  parse      TagParser, incremental: fed streamed chunks, a tag may split
             across them. Unknown or malformed tags are dropped with a
             warning and never spoken.
  time       timeline(): a tag's time is where it sits in the spoken text at
             a speaking rate (characters per second, separate for CJK and
             Latin script), plus punctuation and [pause:x] silences. The
             same clock gives the subtitles.
  offer      offer(): only tags that would move something on this rig are
             in the prompt (no [lean:in] where the body already sits at
             that side, no [brow:*] without brow roles), from the roles
             and adapted statistics the runtime binds (runtime.event_roles).
             A vocabulary tag not offered is dropped by the parser.
             Each gesture gets a line saying what it shows, measured from
             the motion as the runtime plays it (flow.gestures), or taken
             from rigs/<rig>.gestures.json / --gestures, which can also
             hide one (the parser then drops it too).
  realize    realize(): each tag -> one or more tokens on the roles the rig
             binds (aidang_2 binds no gaze -- its eyeballs are switches --
             so [look:*] turns the head only). Amplitudes are fractions of
             the role's authored range on that side (the fitted p2/p98),
             move durations the fitted main sequence (events._move_dur),
             holds the fitted hold medians.
  merge      merge(): the fitted scheduler keeps producing idle moves; a
             directed move takes over the role group it touches for its
             move plus hold, and idle moves overlapping that window are
             dropped (events.render never retargets mid-flight, so an idle
             head turn left in would delay a directed nod, not mix with it).
             Blinks and untouched groups pass through unchanged.

  speak      --tts edge|siliconflow|voicevox|file|auto (flow.speech), the
             backend and voice per reply language from rigs/<rig>.voice.json
             or --voice-config (flags win): the spoken text is
             synthesised (split at [pause] tags, the pauses inserted as
             silence) into speech.wav, and the times above come from the
             audio instead of the rate: word boundaries where the service
             sends them (edge), the moras (VOICEVOX), else the phrases
             matched to the voiced stretches an energy detector finds. The
             render runs the audio plus --tail; a per-frame mouth curve from
             the speech level -- under --lipsync auto on the parameters and
             ranges the rig's own talking motions use (flow.mouth) -- goes
             to actions.json as "mouth", which the runtime turns into a
             "mouth" token, so --tokens-in replays it too. demo.mp4 gets the
             audio and the subtitles. --tts none [default] is the rate model,
             unchanged.

The runtime does realize + merge itself (`--direct actions.json`), because
the adapted statistics, bound roles and start pose live there; the merged
stream it writes with `--tokens` replays bit for bit with `--tokens-in`.

Usage (see --help):
    python -m flow.director --model-json M --rig rigs/R.npz --events ev.json \\
        --say "hello" --backend anthropic --out runs/demo/R [--render]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import unicodedata

import numpy as np

try:
    from . import events as ev
    from . import gestures as gz
    from . import speech as sp
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import events as ev
    from flow import gestures as gz
    from flow import speech as sp

# The only place model names appear: one default per backend.
DEFAULT_MODEL = {"anthropic": "claude-opus-5-5", "ollama": "gemma4:e4b",
                 "siliconflow": "Qwen/Qwen3.6-35B-A3B",
                 "deepseek": "deepseek-chat", "hf": "Qwen/Qwen3-8B"}
# OpenAI-compatible chat/completions services: base URL and the environment
# variable holding the key. SiliconFlow's mainland site is api.siliconflow.cn
# with its own keys; pass --api-base for it.
OPENAI_COMPAT = {"siliconflow": ("https://api.siliconflow.com/v1",
                                 "SILICONFLOW_API_KEY"),
                 "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY")}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# ------------------------------------------------------------- vocabulary

# tag -> allowed arguments (None: takes none). gesture's are the rig's own
# motion labels, pause's a number of seconds.
VOCAB = {
    "look": ("left", "right", "up", "down", "viewer", "away"),
    "nod": None, "shake": None,
    "tilt": ("left", "right"),
    "brow": ("up", "down", "furrow"),
    "blink": None,
    "lean": ("in", "back"),
    "gesture": "motion",
    "pause": "seconds",
}
MAX_PAUSE = 3.0
MAX_TAG = 48            # an unclosed '[' longer than this is not a tag

# Direction conventions, normalised role units (after the binding's sign).
# Cubism's standard ids: ParamAngleX/EyeBallX + turn toward screen right,
# AngleY/EyeBallY + up. AngleZ and BodyAngleY signs vary more between
# riggers; UNVERIFIED on the sample rigs -- flip here if the first render
# tilts or leans the wrong way.
SIGN = {"right": +1.0, "up": +1.0, "tilt_right": +1.0, "lean_in": +1.0}

# How far a directed move goes, as a fraction of the role's authored range on
# that side (centre to the fitted p98 or p2). A deliberate look sits near the
# edge of where riggers put the eyes; the head goes about half as far, as an
# authored head trails the eyes with a smaller turn (the fitted head-follow
# gain is ~0 -- heads rest ~5% of the time, the pairing is near chance -- so
# it cannot set this).
FRAC = {"look_gaze": 0.8, "look_head": 0.5, "away_gaze": 0.7, "away_down": 0.3,
        "nod": 0.5, "shake": 0.4, "tilt": 0.6, "lean": 0.7, "brow_form": 0.8}
# Brow expression value in fitted std units, loaded onto every brow column as
# events._brow does: 2 std on the heights is the fitted p98 (0.43 vs 0.42).
BROW_E = {"up": 2.0, "down": -2.0, "furrow": -1.5}
# Nod and shake strokes are conversational head movements, faster than any
# idle head move (fitted head move median 1.3 s): speech-linked nods run
# 0.2-0.5 s a stroke (Hadar et al. 1983). Hand-set, not fitted.
NOD_STROKE = 0.3
SHAKE_STROKE = 0.25
# A directed pose is held at least this long, else it reads as a twitch;
# otherwise for the group's fitted hold median.
MIN_HOLD = 1.0
GESTURE_MAX = 8.0       # seconds of a directed motion at most
# After a directed gesture the arms go back to the idle loop, as Azur Lane
# returns to its idle after a touch motion: the settle motion (runtime's
# calm_library, the idle the base state comes from) from its first frame for
# this long. Without it the player held the gesture's last frame until the
# next idle gesture -- mid-pose for a motion cut at GESTURE_MAX (aidang_2's
# touch_special runs past 9.8 s), and under the calm idle profile the next
# one can be 15-40 s away.
SETTLE = 2.0

# Speaking rate. Japanese TTS speaks ~7-8 morae/s, Mandarin ~4-5 syllables/s:
# 6 CJK characters/s sits between. English at ~150 words/min is ~14
# characters/s counting spaces. Punctuation is silent but pauses.
CJK_RATE = 6.0
LATIN_RATE = 14.0
SENT_PAUSE = 0.35
CLAUSE_PAUSE = 0.15
SENT_END = set(".!?。！？…\n")
CLAUSE = set(",;:、，；：—")
# What may follow a sentence end and still belong to it.
SENT_TAIL = SENT_END - {"\n"} | set("\"'”’」』)）")
SUB_MAX = {"cjk": 22, "latin": 48}   # characters per subtitle before a clause split

_FULLWIDTH = str.maketrans({"［": "[", "］": "]", "：": ":", "【": "[", "】": "]"})
_TAG_RE = re.compile(r"^\s*([A-Za-z_]+)\s*(?::\s*([^\s\]]+))?\s*$")


def gesture_labels(motion_names):
    """{label: motion name} for the rig's motion files: the basename without
    .motion3.json, lowercased (GesturePlayer looks motions up by basename).
    The first file wins a duplicate label."""
    out = {}
    for n in motion_names:
        b = os.path.basename(n)
        lab = re.sub(r"\.motion3\.json$", "", b, flags=re.I).lower()
        out.setdefault(lab, b)
    return out


# ----------------------------------------------------------------- parser

class TagParser:
    """Incremental tag parser: feed() streamed chunks, close() at the end.

    `spoken` is the text with tags removed (runs of spaces left by a removed
    tag collapsed), `tags` the accepted tags as {"tag", "arg", "at"} with
    `at` the character offset in `spoken` where they sit, `warnings` what was
    dropped. feed()/close() also return the tags completed by that call, the
    hook for acting on them as they arrive. `offered` (offer()'s tag keys,
    None for all): a tag in the vocabulary that this rig was not offered is
    dropped like an unknown one -- a model may still write one it saw in an
    earlier turn or in its training.
    """

    def __init__(self, gestures=(), offered=None):
        self.gestures = {g.lower() for g in gestures}
        self.offered = None if offered is None else set(offered)
        self.spoken = ""
        self.tags = []
        self.warnings = []
        self._buf = None          # text after an open '[', or None

    def _text(self, s):
        # Markdown emphasis is never spoken: Qwen3.6 without thinking wrote
        # "You ate *my* cake?" despite the prompt's no-markup rule.
        s = s.replace("*", "")
        if not s:
            return
        if not self.spoken:
            s = s.lstrip()
        elif self.spoken.endswith((" ", "\n")):
            s = s.lstrip(" ")
        self.spoken += s

    def _tag(self, inner):
        m = _TAG_RE.match(inner)
        if not m:
            self.warnings.append(f"malformed tag [{inner}] dropped")
            return None
        name, arg = m.group(1).lower(), m.group(2)
        arg = arg.lower() if arg is not None else None
        allowed = VOCAB.get(name, "?")
        if allowed == "?":
            self.warnings.append(f"unknown tag [{inner}] dropped")
            return None
        if allowed is None:
            if arg is not None:
                self.warnings.append(f"[{name}] takes no argument; "
                                     f"'{arg}' ignored")
            arg = None
        elif allowed == "seconds":
            try:
                v = float(arg)
            except (TypeError, ValueError):
                self.warnings.append(f"[{inner}]: pause needs seconds; dropped")
                return None
            if not math.isfinite(v) or v <= 0:
                self.warnings.append(f"[{inner}]: pause must be > 0; dropped")
                return None
            arg = str(round(min(v, MAX_PAUSE), 3))
        elif allowed == "motion":
            if arg not in self.gestures:
                self.warnings.append(f"[{inner}]: the rig has no motion "
                                     f"'{arg}'; dropped")
                return None
        elif arg not in allowed:
            self.warnings.append(f"[{inner}]: '{name}' takes "
                                 f"{'/'.join(allowed)}; dropped")
            return None
        if (self.offered is not None
                and _tag_key(name, arg) not in self.offered):
            self.warnings.append(f"[{inner}] is not offered on this rig; "
                                 "dropped")
            return None
        tag = {"tag": name, "arg": arg, "at": len(self.spoken)}
        self.tags.append(tag)
        return tag

    def feed(self, chunk):
        done = []
        chunk = chunk.translate(_FULLWIDTH)
        i = 0
        while i < len(chunk):
            if self._buf is None:
                j = chunk.find("[", i)
                if j < 0:
                    self._text(chunk[i:])
                    break
                self._text(chunk[i:j])
                self._buf = ""
                i = j + 1
                continue
            j = chunk.find("]", i)
            k = chunk.find("[", i)
            if k >= 0 and (j < 0 or k < j):
                # A new '[' before this one closed: the first was not a tag.
                self.warnings.append(f"unclosed '[{self._buf + chunk[i:k]}' "
                                     "dropped")
                self._buf = None
                i = k
                continue
            if j < 0:
                self._buf += chunk[i:]
                if len(self._buf) > MAX_TAG:
                    # Too long to be a tag: speak it, minus the bracket.
                    self.warnings.append(f"'[' never closed within {MAX_TAG} "
                                         "characters; kept as text")
                    rest, self._buf = self._buf, None
                    self._text(rest)
                break
            t = self._tag(self._buf + chunk[i:j])
            if t:
                done.append(t)
            self._buf = None
            i = j + 1
        return done

    def close(self):
        if self._buf is not None:
            self.warnings.append(f"unterminated tag '[{self._buf}' at the end "
                                 "dropped")
            self._buf = None
        self.spoken = self.spoken.strip()
        n = len(self.spoken)
        for t in self.tags:
            t["at"] = min(t["at"], n)
        return []


def parse(text, gestures=(), offered=None):
    p = TagParser(gestures, offered)
    p.feed(text)
    p.close()
    return p


# ----------------------------------------------------------------- timing

def _is_cjk(ch):
    return (unicodedata.east_asian_width(ch) in ("W", "F")
            and unicodedata.category(ch)[0] in "LN")


def timeline(spoken, tags, cjk_rate=CJK_RATE, latin_rate=LATIN_RATE):
    """(actions, spans, end): every non-pause tag with its time `t` in
    seconds, (start, end) of each spoken character -- the silence after
    punctuation and [pause] excluded -- and when speech ends.

    Tags at one offset act in their written order: a [pause] there delays
    the tags after it, not the ones before."""
    by_at = {}
    for t in tags:
        by_at.setdefault(t["at"], []).append(t)
    clock, acts, spans = 0.0, [], []
    for i in range(len(spoken) + 1):
        for t in by_at.get(i, ()):
            if t["tag"] == "pause":
                clock += float(t["arg"])
            else:
                acts.append({"t": round(clock, 4), "tag": t["tag"],
                             "arg": t["arg"], "at": i})
        if i == len(spoken):
            break
        ch = spoken[i]
        if ch in SENT_END or ch in CLAUSE or unicodedata.category(ch)[0] == "P":
            spans.append((clock, clock))
            clock += (SENT_PAUSE if ch in SENT_END
                      else CLAUSE_PAUSE if ch in CLAUSE else 0.0)
        else:
            d = 1.0 / (cjk_rate if _is_cjk(ch) else latin_rate)
            spans.append((clock, clock + d))
            clock += d
    return acts, spans, clock


def subtitles(spoken, spans):
    """[{"start", "end", "text"}]: sentences, split again at a clause break
    when longer than SUB_MAX characters. A segment runs from its first
    character to the end of its last (silences after it excluded), and is
    shown at least 0.5 s."""
    if not spoken:
        return []
    segs, a = [], 0
    cjk = sum(_is_cjk(c) for c in spoken) > len(spoken) / 3
    cap = SUB_MAX["cjk" if cjk else "latin"]
    run = False
    for i, ch in enumerate(spoken):
        last = i == len(spoken) - 1
        # A run of sentence ends ("...", "?!", "。。。") and a closing quote
        # stay with their phrase: cutting at its first '.' left ".", "."
        # as segments of their own (Hiyori's edge TTS run, "Well, maybe a
        # little... I kept checking").
        run = ch in SENT_END or (run and ch in SENT_TAIL)
        if run and not last and spoken[i + 1] in SENT_TAIL:
            continue
        if run or last or (ch in CLAUSE and i + 1 - a >= cap):
            while a < i and spoken[a].isspace():
                a += 1
            text = spoken[a:i + 1].strip()
            if text and not any(c.isalnum() for c in text):
                # Punctuation only: onto the phrase it ends, or, leading
                # ("...well"), into the one it starts. Never on its own.
                if not segs:
                    continue
                segs[-1]["text"] += text
                segs[-1]["end"] = round(max(segs[-1]["end"], spans[i][1]), 3)
            elif text:
                t0, t1 = spans[a][0], spans[i][1]
                segs.append({"start": round(t0, 3),
                             "end": round(max(t1, t0 + 0.5), 3),
                             "text": text})
            a = i + 1
    return segs


def srt(segs):
    def ts(s):
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:" \
               f"{ms // 1000 % 60:02d},{ms % 1000:03d}"
    return "".join(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n\n"
                   for i, s in enumerate(segs, 1))


# ---------------------------------------------------------------- realize

def _toward(st, r, side, frac):
    """frac of the role's authored range from its centre toward side (+1:
    the fitted p98, -1: the p2), within [-1, 1]."""
    R = st["roles"][r]
    edge = R["hi"] if side > 0 else R["lo"]
    return float(np.clip(R["center"] + frac * (edge - R["center"]), -1, 1))


def _hold(st, g):
    return max(float(st["groups"][g]["hold"][0]), MIN_HOLD)


# A lean is offered only when its target lies at least this far from the
# body_y centre, toward that side, in normalised role units ([-1, 1] over
# the authored range: 0.1 is 5% of it, 1 degree on the usual +-10 degree
# ParamBodyAngleY). aidang_2's own motions keep BodyAngleY near its max:
# adapted centre 0.932 (its start pose in the server's --no-idle plan),
# p98 0.976, so lean:in went 0.932 -> 0.963, +0.031, and showed nothing
# although every model tried it. Hiyori's lean:in reaches +0.65 and
# Natori's, the smallest of the sample rigs, +0.22 (adapted p98 0.32),
# twice this. Signed: adapt() blends centre and p98 separately, and a p98
# blended below the centre must not pass as room.
LEAN_MIN_REACH = 0.1


def _usable(st, roles, r):
    """Bound, and with an authored range to move in: Natori's BrowLX/RX are
    bound but fitted with lo = hi = 0, so nothing directed on them shows."""
    R = st["roles"].get(r)
    return r in roles and R is not None and R["hi"] - R["lo"] > 1e-6


def offer(st, roles):
    """(offered, left_out): the tag keys (all_tags()) realize() would move
    something visible for on this rig, and [(key, reason)] for the rest,
    both in VOCAB order.

    st: the run's statistics, adapted as the runtime adapts them
    (runtime.event_roles); roles: the bound role names. Blink, pause and
    gesture are always offered (gesture's motions are checked per tag by the
    parser).
    """
    u = {r: _usable(st, roles, r) for r in ev.ALL_ROLES}
    need = {
        "look:left": ("gaze_x", "head_x"), "look:right": ("gaze_x", "head_x"),
        "look:up": ("gaze_y", "head_y"), "look:down": ("gaze_y", "head_y"),
        "look:viewer": ("gaze_x", "gaze_y", "head_x", "head_y"),
        "look:away": ("gaze_x", "head_x"),
        "nod": ("head_y",), "shake": ("head_x",),
        "tilt:left": ("head_z",), "tilt:right": ("head_z",),
        "brow:up": ev.GROUPS["brow"], "brow:down": ev.GROUPS["brow"],
        "brow:furrow": ev.GROUPS["brow"],
        "lean:in": ("body_y",), "lean:back": ("body_y",),
    }
    why = {"look:left": "no gaze_x or head_x role",
           "look:up": "no gaze_y or head_y role",
           "look:viewer": "no gaze or head roles",
           "nod": "no head_y role", "shake": "no head_x role",
           "tilt:left": "no head_z role", "brow:up": "no brow roles",
           "lean:in": "no body_y role"}
    why.update({"look:right": why["look:left"], "look:down": why["look:up"],
                "look:away": why["look:left"], "tilt:right": why["tilt:left"],
                "brow:down": why["brow:up"], "brow:furrow": why["brow:up"],
                "lean:back": why["lean:in"]})
    if not any(u[r] for r in need["look:viewer"]):
        # One reason for all of them, so the log says look:* once.
        why.update({k: why["look:viewer"] for k in need if k[:5] == "look:"})
    offered, out = [], []
    for k in all_tags():
        if k not in need:
            offered.append(k)
            continue
        if not any(u[r] for r in need[k]):
            out.append((k, why[k]))
            continue
        if k.startswith("lean:"):
            s = SIGN["lean_in"] * (1 if k == "lean:in" else -1)
            c = st["roles"]["body_y"]["center"]
            reach = s * (_toward(st, "body_y", s, FRAC["lean"]) - c)
            if reach < LEAN_MIN_REACH:
                out.append((k, f"body_y centre {c:.2f}, no room: the lean "
                               f"moves {reach:+.3f}"))
                continue
        offered.append(k)
    return tuple(offered), out


def left_out_text(out):
    """"lean:in (body_y centre 0.98, no room ...), brow:* (no brow roles)":
    a tag whose arguments are all left out for one reason is named once."""
    parts, seen = [], set()
    for k, why in out:
        name = k.split(":")[0]
        if name in seen:
            continue
        args = VOCAB[name]
        same = [x for x in out if x[0].split(":")[0] == name]
        if (isinstance(args, tuple) and len(same) == len(args) > 1
                and len({w for _, w in same}) == 1):
            parts.append(f"{name}:* ({why})")
            seen.add(name)
        else:
            parts.append(f"{k} ({why})")
    return ", ".join(parts)


def realize(actions, st, roles, motions, rng, start=None, settle=None):
    """Event tokens for timed actions (timeline()), on the bound `roles` only.

    st: the run's statistics (load_stats, adapted). motions: {motion name:
    seconds}. Each directed token carries "src": "director" and, for moves,
    "hold": how long its target is kept before the idle may move the group
    again. Within a group the directed moves run one after another, as
    events.render would play them anyway. settle: a motion name; each
    directed gesture is then followed by SETTLE seconds of it from its first
    frame (also "src": "director"), unless another directed gesture starts
    first. No rng is drawn for it.
    """
    bound = set(roles)
    have = {g: [r for r in rs if r in bound] for g, rs in ev.GROUPS.items()}
    pos = {r: (start or {}).get(r, st["roles"][r]["center"]) for r in bound
           if r in ev.ALL_ROLES}
    labels = gesture_labels(motions)
    busy = {}
    out = []
    hf = st["head_follow"]
    typ = {"gaze": "saccade", "head": "head_turn", "body": "body_shift",
           "brow": "brow"}

    def move(g, t, target, hold, dur=None):
        target = {r: v for r, v in target.items() if r in bound}
        if not target:
            return None
        t = max(t, busy.get(g, 0.0))
        amp = float(np.sqrt(sum((v - pos[r]) ** 2 for r, v in target.items())))
        d = dur if dur is not None else ev._move_dur(st, g, amp, rng)
        tok = {"t": round(t, 4), "type": typ[g], "role": g,
               "target": {r: round(v, 5) for r, v in target.items()},
               "duration": round(d, 4), "hold": round(max(hold, 0.0), 4),
               "src": "director"}
        if g == "head":
            tok["follows"] = None
        out.append(tok)
        pos.update(target)
        busy[g] = t + d
        return tok, amp

    def centre(cols):
        return {r: st["roles"][r]["center"] for r in cols}

    for a in sorted(actions, key=lambda a: float(a["t"])):
        t, tag, arg = float(a["t"]), a["tag"], a.get("arg")
        if tag == "look":
            gz, hd = centre(have["gaze"]), centre(
                [r for r in have["head"] if r != "head_z"])
            if arg in ("left", "right"):
                s = SIGN["right"] * (1 if arg == "right" else -1)
                gz.update(gaze_x=_toward(st, "gaze_x", s, FRAC["look_gaze"]))
                hd.update(head_x=_toward(st, "head_x", s, FRAC["look_head"]))
            elif arg in ("up", "down"):
                s = SIGN["up"] * (1 if arg == "up" else -1)
                gz.update(gaze_y=_toward(st, "gaze_y", s, FRAC["look_gaze"]))
                hd.update(head_y=_toward(st, "head_y", s, FRAC["look_head"]))
            elif arg == "away":
                s = SIGN["right"] * (1 if rng.random() < 0.5 else -1)
                dn = -SIGN["up"]
                gz.update(gaze_x=_toward(st, "gaze_x", s, FRAC["away_gaze"]),
                          gaze_y=_toward(st, "gaze_y", dn, FRAC["away_down"]))
                hd.update(head_x=_toward(st, "head_x", s, FRAC["look_head"]),
                          head_y=_toward(st, "head_y", dn, FRAC["away_down"]))
            hold = _hold(st, "gaze")
            got = move("gaze", t, gz, hold)
            lag = 0.0
            if got:
                tok, amp = got
                if hold > 0:
                    out.append({"t": round(tok["t"] + tok["duration"], 4),
                                "type": "fixate", "role": "gaze",
                                "duration": tok["hold"], "src": "director"})
                if amp > ev.BLINK_AMP and rng.random() < ev.P_BLINK:
                    out.append({"t": tok["t"], "type": "blink", "role": "eyes",
                                "src": "director"})
                lag = hf["lag"]
            move("head", t + lag, hd, hold - lag)
        elif tag == "nod" and "head_y" in bound:
            c = st["roles"]["head_y"]["center"]
            dn = _toward(st, "head_y", -SIGN["up"], FRAC["nod"])
            move("head", t, {"head_y": dn}, 0.0, NOD_STROKE)
            move("head", t, {"head_y": c}, 0.0, 1.3 * NOD_STROKE)
        elif tag == "shake" and "head_x" in bound:
            c = st["roles"]["head_x"]["center"]
            s = 1 if rng.random() < 0.5 else -1
            for k, v in enumerate((s, -s)):
                move("head", t, {"head_x": _toward(st, "head_x", v,
                                                   FRAC["shake"])},
                     0.0, SHAKE_STROKE * (1.0 if k == 0 else 1.6))
            move("head", t, {"head_x": c}, 0.0, 1.3 * SHAKE_STROKE)
        elif tag == "tilt":
            s = SIGN["tilt_right"] * (1 if arg == "right" else -1)
            move("head", t, {"head_z": _toward(st, "head_z", s, FRAC["tilt"])},
                 _hold(st, "head"))
        elif tag == "lean":
            s = SIGN["lean_in"] * (1 if arg == "in" else -1)
            move("body", t, {"body_y": _toward(st, "body_y", s, FRAC["lean"])},
                 _hold(st, "body"))
        elif tag == "brow":
            e = BROW_E[arg]
            tgt = {}
            for r in have["brow"]:
                R = st["roles"][r]
                if arg == "furrow" and r in ("brow_lf", "brow_rf"):
                    # Cubism's BrowForm runs angry (-1) to smile (+1).
                    tgt[r] = _toward(st, r, -1, FRAC["brow_form"])
                    continue
                k = float(np.clip(st["brow_loading"].get(r, 1.0), -1, 1))
                tgt[r] = float(np.clip(R["center"] + R["std"] * k * e,
                                       max(R["lo"], -1), min(R["hi"], 1)))
            move("brow", t, tgt, _hold(st, "brow"))
        elif tag == "blink":
            out.append({"t": round(t, 4), "type": "blink", "role": "eyes",
                        "src": "director"})
        elif tag == "gesture":
            name = labels.get(str(arg).lower())
            if name is None:
                continue
            t = max(t, busy.get("arms", 0.0))
            d = min(float(motions[name]), GESTURE_MAX)
            out.append({"t": round(t, 4), "type": "gesture", "role": "arms",
                        "motion": name, "start": 0.0, "duration": round(d, 4),
                        "fade": st["gesture"]["fade"], "src": "director"})
            busy["arms"] = t + d
    if settle in motions:
        ges = [k for k in out if k["type"] == "gesture"]
        for k, nxt in zip(ges, ges[1:] + [None]):
            end = round(float(k["t"]) + float(k["duration"]), 4)
            if nxt is not None and float(nxt["t"]) < end + SETTLE:
                continue
            out.append({"t": end, "type": "gesture", "role": "arms",
                        "motion": settle, "start": 0.0,
                        "duration": round(min(float(motions[settle]),
                                              SETTLE), 4),
                        "fade": st["gesture"]["fade"], "src": "director",
                        "settle": True})
    return out


# ------------------------------------------------------------------ merge

MOVES = ("saccade", "head_turn", "body_shift", "brow")


def windows(directed):
    """{group: [(start, end)]} that directed tokens take over: a move for its
    duration plus hold, a gesture for its duration, a pose (flow.pose) the
    arms for its move plus hold when it moves them. A pose takes no other
    group: the idle's head, gaze, brow and body moves keep running under it
    (flow.pose.PoseLayer writes only what the pose plays, and lets the idle
    move a held target at reduced amplitude). Dropping them froze the whole
    group for the pose: a head.tilt pose stopped the idle's yaw and pitch
    too, and Hiyori's head and gaze were frozen in 58% of the E2 hold
    frames."""
    win = {}
    for k in directed:
        if k["type"] in MOVES:
            win.setdefault(k["role"], []).append(
                (float(k["t"]), float(k["t"]) + float(k["duration"])
                 + float(k.get("hold", 0.0))))
        elif k["type"] == "gesture":
            win.setdefault("arms", []).append(
                (float(k["t"]), float(k["t"]) + float(k["duration"])))
        elif k["type"] == "move" and "arms" in k.get("groups", ()):
            # flow.moves: an arm figure (clap, cheer...) takes the arms for
            # its whole play, ease in and out included.
            win.setdefault("arms", []).append(
                (float(k["t"]), float(k["t"]) + float(k["duration"])))
        elif k["type"] == "pose" and "arms" in k.get("groups", ()):
            # flow.pose: an idle gesture starting while a pose holds the
            # arms would fight it for the arm drawing.
            end = float(k["t"]) + float(k["duration"]) + float(k.get("hold",
                                                                     0.0))
            win.setdefault("arms", []).append((float(k["t"]), end))
    return win


def merge(idle, directed):
    """One token stream: `idle` minus what overlaps a directed window on the
    same group, plus `directed`, stably sorted by onset (events.render plays
    a group's moves in list order).

    An idle move is dropped when its [t, t + duration] meets the window: one
    that started before and is still moving would hold the directed move
    back. Its fixation goes with it. An idle gesture starting inside a
    directed one is dropped; one already playing gives way when the directed
    one starts (GesturePlayer plays the latest). Blinks stay. Head turns'
    `follows` indices are renumbered (null when their saccade was dropped).
    """
    win = windows(directed)

    def hit(g, a, b):
        return any(a <= w1 and b >= w0 for w0, w1 in win.get(g, ()))

    kept, gone_sacc = [], set()
    for i, k in enumerate(idle):
        typ = k["type"]
        if typ in MOVES:
            a = float(k["t"])
            if hit(k["role"], a, a + float(k["duration"])):
                if typ == "saccade":
                    gone_sacc.add(round(a + float(k["duration"]), 4))
                continue
        elif typ == "fixate":
            a = float(k["t"])
            if (round(a, 4) in gone_sacc
                    or hit("gaze", a, a + float(k.get("duration", 0.0)))):
                continue
        elif typ == "gesture":
            a = float(k["t"])
            if any(w0 <= a < w1 for w0, w1 in win.get("arms", ())):
                continue
        kept.append((i, dict(k)))
    allk = kept + [(None, dict(k)) for k in directed]
    allk.sort(key=lambda x: (float(x[1]["t"]), x[1]["type"]))
    new_of = {i: j for j, (i, _) in enumerate(allk) if i is not None}
    out = []
    for _, k in allk:
        if k["type"] == "head_turn" and k.get("follows") is not None:
            k["follows"] = new_of.get(k["follows"])
        out.append(k)
    return out


# --------------------------------------------------------------- prompting

SYSTEM_TEMPLATE = """\
You are {persona}

You are a Live2D character talking with the user face to face. Reply with \
exactly what you say aloud: one to three short spoken sentences, in the \
user's language. Act with your face and body by placing action tags inline, \
in square brackets, at the moment in your speech where the action happens. \
Tags are silent; they are never read out.

Action tags (use only these, exactly as written):
{tags}{gestures}
Rules:
- Use one or two tags per sentence, chosen for what that sentence means. \
Put a tag right before the words it goes with, often mid-sentence, not only \
at the start.
- {vary}Do not open every reply the same way.
- No other markup: no asterisk actions, no parentheses stage directions, no \
emoji, no lists, no narration.
- Never explain or mention the tags.

Examples of placement (not of content; never copy them):
{examples}"""

# One prompt line per entry: the tags it shows (those the rig offers of
# them) and what they mean. In VOCAB order, so the prompt is byte-stable.
TAG_LINES = (
    (("look:left", "look:right", "look:up", "look:down"),
     "glance that way (screen directions)"),
    (("look:viewer",), "look back at the user"),
    (("look:away",), "look aside: shy, thinking, evasive"),
    (("nod",), "one nod: yes, agreement, listening"),
    (("shake",), "shake the head: no, disbelief"),
    (("tilt:left", "tilt:right"), "tilt the head: curious, playful, puzzled"),
    (("brow:up",), "raise the brows: surprise, interest"),
    (("brow:down",), "lower the brows: serious, sad"),
    (("brow:furrow",), "knit the brows: annoyed, doubtful"),
    (("blink",), "a deliberate blink"),
    (("lean:in",), "lean toward the user: eager, intimate"),
    (("lean:back",), "lean away: surprised, wary"),
)
# "Vary them" suggestions, each kept only when one of its tags is offered:
# a rule naming a lean on a rig that cannot lean asks for a tag that is
# then dropped.
VARY = (
    (("nod",), "a nod on agreement"),
    (("brow:up", "brow:down", "brow:furrow"), "brows on surprise or doubt"),
    (("look:away",), "a glance away when shy or thinking"),
    (("lean:in", "lean:back"), "a lean when it gets personal"),
)
# Examples are kept only when every tag in them is offered. The language
# models copy the examples' tags most of all (every model used [lean:in] in
# 5-6 of 6 replies), so an example may only show what the rig can do. The
# last one uses tags every rig has and stands in when none of the others
# survive.
EXAMPLES = (
    "Hmm, [look:up] let me think... [nod] Yes, that works for me.",
    "[brow:up] Really? You did all that [lean:in] just for me?",
    "That's not [shake] what I meant at all. [look:away] Forget it.",
)
FALLBACK_EXAMPLE = "Oh... [pause:0.5] [blink] I didn't expect that."
_EX_TAG = re.compile(r"\[([a-z]+(?::[a-z]+)?)")


def all_tags():
    """Every tag key in VOCAB order: "name:arg", or "name" for a tag with
    no argument or a free one (gesture, pause)."""
    out = []
    for name, allowed in VOCAB.items():
        if isinstance(allowed, tuple):
            out += [f"{name}:{x}" for x in allowed]
        else:
            out.append(name)
    return tuple(out)


def _tag_key(name, arg):
    return f"{name}:{arg}" if isinstance(VOCAB.get(name), tuple) else name


def gesture_lines(labels, descriptions=None, same=None):
    """The prompt's [gesture:NAME] entry. With descriptions ({label: text},
    flow.gestures): one line per gesture, sorted, so the model picks by what
    moves rather than by an opaque file name (DeepSeek announced "a big
    wave" with Hiyori's hiyori_m02, one wide arm-and-hand move). An empty
    text leaves the bare label: nothing measured sets that gesture apart,
    and its name says more (Azur Lane's touch_head, wedding). `same`
    ({label: group}, gestures.same_motions): labels of one motion with the
    same text share a line, "complete / home / mail: same motion; ...", so
    the model does not think it is choosing between them (lafei ships one
    clip under six names); any of them still parses. Without descriptions:
    the labels inline, as before."""
    if not labels:
        return ""
    if descriptions is None:
        return ("[gesture:NAME]  play one of your own full-body motions, "
                "NAME one of: " + ", ".join(sorted(labels)) + ". Each lasts "
                "several seconds; use at most one per reply.\n")
    # "motions", not "full-body motions": on the Cubism samples a directed
    # gesture moves the arms only (the head and body stay with the tags).
    out = ("[gesture:NAME]  play one of your own motions, at most one per "
           "reply. NAME one of these (after it, what sets that one apart "
           "from the others; where nothing is said, go by the name; "
           "\"a / b: same motion\" means any of those names plays it):\n")
    lines = {}
    for lab in sorted(labels):
        d = (descriptions or {}).get(lab) or ""
        key = ((same or {}).get(lab, ("", lab)), d)
        lines.setdefault(key, []).append(lab)
    for (_, d), labs in sorted(lines.items(), key=lambda x: x[1][0]):
        if len(labs) > 1:
            d = "same motion" + (f"; {d}" if d else "")
        out += "  " + " / ".join(labs) + (f": {d}" if d else "") + "\n"
    return out


def system_prompt(persona, labels, offered=None, descriptions=None,
                  same=None):
    """Byte-stable for a given persona and rig (no clock, no turn count), so
    the prompt cache hits on every turn. `offered`: the tag keys this rig
    can show (offer()); None offers them all. `descriptions`: {label:
    text} for the gesture lines, `same` ({label: group}) to merge labels
    of one motion (gesture_lines)."""
    ok = set(all_tags() if offered is None else offered)
    tags = ""
    for keys, what in TAG_LINES:
        have = [k for k in keys if k in ok]
        if have:
            tags += " ".join(f"[{k}]" for k in have) + f"  {what}\n"
    tags += "[pause:0.5]  stay silent for that many seconds (0.2 to 3)\n"
    g = gesture_lines(labels, descriptions, same)
    vary = [what for keys, what in VARY if any(k in ok for k in keys)]
    vary = ("Vary them: " + ", ".join(vary) + ". ") if vary else "Vary them. "
    ex = [e for e in EXAMPLES if all(k in ok for k in _EX_TAG.findall(e))]
    examples = "\n".join(ex or [FALLBACK_EXAMPLE]) + "\n"
    persona = (persona or "a friendly, expressive character.").strip()
    return SYSTEM_TEMPLATE.format(persona=persona, tags=tags, gestures=g,
                                  vary=vary, examples=examples)


# ---------------------------------------------------------------- backends

class TextBackend:
    """A pre-written tagged reply (file or stdin), streamed in small chunks
    so the incremental parser runs exactly as it does on a live model."""

    def __init__(self, text, chunk=7):
        self.text, self.chunk = text, chunk
        self.stop_reason, self.usage = "end_turn", {}

    def stream(self, system, messages):
        for i in range(0, len(self.text), self.chunk):
            yield self.text[i:i + self.chunk]


def _unexpected_kwarg(exc):
    m = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
    return m.group(1) if m else None


class AnthropicBackend:
    def __init__(self, model, effort="low", max_tokens=2048, log=print):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.log = log
        self.stop_reason, self.usage = None, {}
        self._client = None

    def _open(self, kw):
        """client.beta.messages.stream(**kw), moving any keyword this SDK
        version does not know into extra_body (the SDK forwards it as is)."""
        extra = {}
        for _ in range(4):
            try:
                args = dict(kw, extra_body=extra) if extra else kw
                return self._client.beta.messages.stream(**args)
            except TypeError as exc:
                k = _unexpected_kwarg(exc)
                if not k or k not in kw:
                    raise
                extra[k] = kw.pop(k)
                self.log(f"anthropic: SDK has no '{k}=' argument; sending it "
                         "in extra_body")
        raise RuntimeError("anthropic: could not build the request")

    def stream(self, system, messages):
        try:
            import anthropic                  # lazy: only this backend needs it
        except ImportError as exc:
            raise SystemExit(f"--backend anthropic needs the SDK "
                             f"(pip install anthropic): {exc}") from exc
        if self._client is None:
            self._client = anthropic.Anthropic()
        kw = dict(model=self.model, max_tokens=self.max_tokens,
                  system=[{"type": "text", "text": system,
                           "cache_control": {"type": "ephemeral"}}],
                  messages=messages,
                  output_config={"effort": self.effort},
                  betas=[FALLBACK_BETA], fallbacks="default")
        with self._open(kw) as s:
            for text in s.text_stream:
                yield text
            msg = s.get_final_message()
        self.stop_reason = msg.stop_reason
        u = getattr(msg, "usage", None)
        self.usage = {k: getattr(u, k, None) for k in
                      ("input_tokens", "output_tokens",
                       "cache_creation_input_tokens",
                       "cache_read_input_tokens")} if u is not None else {}
        self.log(f"anthropic: stop {self.stop_reason}; usage "
                 + ", ".join(f"{k} {v}" for k, v in self.usage.items()))


class OllamaBackend:
    """Ollama's /api/chat, streamed (one JSON object per line), stdlib only."""

    def __init__(self, model, url="http://localhost:11434", seed=0,
                 timeout=300, log=print, num_ctx=None):
        self.model, self.url, self.seed = model, url.rstrip("/"), seed
        self.timeout, self.log = timeout, log
        # The model's context unless given: Ollama's default is shorter
        # than the body prompt plus a history (flow.train.chat.modelfile
        # sets it for a tuned model).
        self.num_ctx = num_ctx
        self.stop_reason, self.usage = None, {}

    def stream(self, system, messages):
        import urllib.request
        body = {"model": self.model, "stream": True,
                "messages": [{"role": "system", "content": system}] + messages,
                "options": {"seed": int(self.seed)}}
        if self.num_ctx:
            body["options"]["num_ctx"] = int(self.num_ctx)
        req = urllib.request.Request(
            self.url + "/api/chat", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        # A proxy from the environment (http_proxy) answered localhost with
        # 502 Bad Gateway on the server; Ollama is local, so go direct.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=self.timeout) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("error"):
                    raise RuntimeError(f"ollama: {d['error']}")
                text = (d.get("message") or {}).get("content") or ""
                if text:
                    yield text
                if d.get("done"):
                    self.stop_reason = d.get("done_reason", "stop")
                    self.usage = {k: d.get(k) for k in
                                  ("prompt_eval_count", "eval_count")}
        self.log(f"ollama: stop {self.stop_reason}; usage {self.usage}")


class OpenAICompatBackend:
    """chat/completions with stream=True (server-sent events), stdlib only.

    Follows RegionFlow's siliconflow_judge (bench/edit_quality_vlm.py): the
    response body is surfaced on errors, since that is where SiliconFlow says
    what was wrong; 429 and 5xx are retried with backoff, other 4xx are fatal.
    No enable_thinking / thinking_budget: on SiliconFlow those started 400ing
    on every model there. A reasoning model's trace arrives as
    delta.reasoning_content and is skipped -- only delta.content is spoken.
    """

    def __init__(self, name, model, api_base=None, key_env=None,
                 max_tokens=2048, timeout=120, retries=4, log=print,
                 thinking=None):
        base, env = OPENAI_COMPAT[name]
        self.name, self.model, self.max_tokens = name, model, max_tokens
        self.thinking = thinking
        self.url = (api_base or base).rstrip("/") + "/chat/completions"
        self.key_env = key_env or env
        self.timeout, self.retries, self.log = timeout, retries, log
        self.stop_reason, self.usage = None, {}

    def _open(self, body):
        key = os.environ.get(self.key_env)
        if not key:
            raise SystemExit(f"--backend {self.name} needs {self.key_env} "
                             "in the environment")
        return sp.post(self.url, body, key, self.name, self.timeout,
                       self.retries)

    def stream(self, system, messages):
        body = {"model": self.model, "stream": True,
                "max_tokens": self.max_tokens,
                # Usage arrives in a last chunk with empty choices; without
                # this most streamed replies carried none (flow.dataset's
                # budget then falls back to character-count estimates).
                "stream_options": {"include_usage": True},
                "messages": [{"role": "system", "content": system}] + messages}
        if self.thinking == "off":
            # Opt-in only: on SiliconFlow the reasoning models waited 15-19 s
            # before their first spoken word (DeepSeek-chat 0.6 s), but
            # GLM-5.3-Flash 400s on this ("model always thinks") and the
            # thinking parameters once broke every request there (RegionFlow).
            body["enable_thinking"] = False
        self.stop_reason, self.usage = None, {}
        with self._open(body) as fh:
            for line in fh:
                line = line.decode("utf-8", "replace").strip() \
                    if isinstance(line, bytes) else line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("usage"):
                    self.usage = {k: d["usage"].get(k) for k in
                                  ("prompt_tokens", "completion_tokens",
                                   "prompt_cache_hit_tokens")
                                  if k in d["usage"]}
                for ch in d.get("choices") or ():
                    text = (ch.get("delta") or {}).get("content") or ""
                    if text:
                        yield text
                    fr = ch.get("finish_reason")
                    if fr:
                        # A filtered reply goes down the refusal path: no
                        # motion from a half-spoken answer.
                        self.stop_reason = ("refusal" if fr == "content_filter"
                                            else fr)
        self.log(f"{self.name}: stop {self.stop_reason}; usage {self.usage}")


def make_backend(a, log=print):
    model = a.model or DEFAULT_MODEL.get(a.backend)
    if a.max_tokens is None:
        a.max_tokens = 8192 if a.backend in OPENAI_COMPAT else 2048
    if a.backend == "anthropic":
        return AnthropicBackend(model, a.effort, a.max_tokens, log)
    if a.backend == "ollama":
        return OllamaBackend(model, a.ollama_url, a.seed, log=log,
                             num_ctx=getattr(a, "ollama_ctx", None))
    if a.backend in OPENAI_COMPAT:
        return OpenAICompatBackend(a.backend, model, a.api_base, a.api_key_env,
                                   a.max_tokens, log=log, thinking=a.thinking)
    if a.backend == "hf":
        # A local transformers model (flow.train.hf; the tuned director):
        # torch and transformers load only here.
        try:
            from .train.hf import HFBackend
        except ImportError:
            from flow.train.hf import HFBackend
        return HFBackend(model, getattr(a, "hf_adapter", None),
                         getattr(a, "hf_template", None),
                         getattr(a, "hf_4bit", False), a.max_tokens,
                         a.seed, getattr(a, "hf_greedy", False), log=log)
    if a.backend == "file":
        if not a.reply_file:
            raise SystemExit("--backend file needs --reply-file")
        with open(a.reply_file, encoding="utf-8") as fh:
            return TextBackend(fh.read())
    if a.backend == "stdin":
        return TextBackend(sys.stdin.read())
    raise SystemExit(f"unknown backend {a.backend}")


# ------------------------------------------------------------------- turns

def ask(backend, system, messages, gestures, log=print, offered=None,
        parser=None):
    """Stream one reply through the parser. Returns a dict: raw tagged text,
    spoken text, tags, warnings, stop_reason, usage, refused. A refusal
    yields no spoken text and no tags, whatever streamed before it.
    `offered`: the tags the prompt offered (TagParser). `parser`: another
    parser to use instead (body mode: flow.pose.PoseParser)."""
    import time
    p = parser if parser is not None else TagParser(gestures, offered)
    raw = []
    t0 = time.monotonic()
    first = None
    for chunk in backend.stream(system, messages):
        if first is None:
            first = time.monotonic() - t0
        raw.append(chunk)
        p.feed(chunk)
    p.close()
    total = time.monotonic() - t0
    # Time to the first spoken chunk is what a live character waits on; a
    # reasoning model spends it thinking (Qwen3.6 once used all 2048 tokens).
    log(f"director: model {total:.1f}s, first text after "
        + (f"{first:.1f}s" if first is not None else "never"))
    refused = backend.stop_reason == "refusal"
    for w in p.warnings:
        log(f"director: {w}")
    if backend.stop_reason in ("max_tokens", "length"):
        log("director: WARNING reply cut off at the token limit")
    return {"raw": "".join(raw), "spoken": "" if refused else p.spoken,
            "tags": [] if refused else p.tags, "warnings": p.warnings,
            "stop_reason": backend.stop_reason, "usage": backend.usage,
            "refused": refused, "model_seconds": round(total, 3),
            "first_text_seconds": None if first is None else round(first, 3)}


def runtime_args(a, seconds):
    out = ["--model", a.model_json, "--rig", a.rig, "--events", a.events,
           "--probe-dir", a.probe_dir, "--fps", str(a.fps),
           "--seed", str(a.seed), "--seconds", f"{seconds:.2f}"]
    if a.clips:
        out += ["--clips", a.clips]
    if a.binding:
        out += ["--binding", a.binding]
    if a.events_pool_only:
        out.append("--events-pool-only")
    if a.no_snap:
        out.append("--no-snap")
    if a.no_idle:
        out.append("--no-idle")
    if a.idle_profile != "calm":
        out += ["--idle-profile", a.idle_profile]
    out += ["--breath-gain", f"{breath_gain(a):g}"]
    # The offline render's drift, pose ease and camera, as flow.live's.
    if getattr(a, "head_drift", False):
        out += ["--head-drift-scale",
                f"{float(getattr(a, 'head_drift_scale', 1.0)):g}"]
    if getattr(a, "pose_ease", False):
        out.append("--pose-ease")
    if not getattr(a, "camera_follow", True):
        out.append("--no-camera-follow")
    out += ["--frame", getattr(a, "frame", "full")]
    return out


def breath_gain(a, log=print):
    """The run's breath gain (runtime.breath_setting), resolved once with
    the voice config this run reads, and passed to the runtime as a number
    so that the printed render command reproduces it."""
    got = getattr(a, "_breath", None)
    if got is None:
        try:
            from .runtime import breath_setting
        except ImportError:
            from flow.runtime import breath_setting
        got = breath_setting(a.rig, getattr(a, "breath_gain", None),
                             voice_config_path(a), log)
        log(f"director: breath gain {got[0]:g} ({got[1]})")
        a._breath = got
    return got[0]


def length(acts, end, motions, tail):
    """Seconds to render: speech, any directed motion still playing after
    it, and a tail so the last move lands."""
    last = end
    labels = gesture_labels(motions)
    for x in acts:
        d = 2.0
        if x["tag"] == "gesture" and labels.get(x["arg"]) in motions:
            d = min(motions[labels[x["arg"]]], GESTURE_MAX)
        elif x["tag"] == "frame":
            d = 0.6         # flow.framing.VIEW_EASE
        last = max(last, x["t"] + d)
    return last + tail


def tts_kind(a):
    """--tts as given; unset is none (silent, as always), or auto when a
    --voice-config is named. A rigs/<rig>.voice.json on disk alone never
    switches speech on."""
    kind = getattr(a, "tts", None)
    if kind:
        return kind
    return "auto" if getattr(a, "voice_config", None) else "none"


def voice_config_path(a):
    """--voice-config, else <rig without .npz>.voice.json (as the gesture
    overrides); the latter is read only when it exists."""
    if getattr(a, "voice_config", None):
        return a.voice_config
    rig = getattr(a, "rig", None)
    if rig:
        return os.path.splitext(rig)[0] + ".voice.json"
    # --avatar vrm has no --rig: rigs/<name>.voice.json beside the body file
    # (rigs/<name>.body.json, or the one built for --vrm vrm/<name>.vrm).
    # Without this the server's oguri_cap/toukai_teio configs were never
    # read and every reply spoke in the default voice.
    vb = getattr(a, "vrm_body", None)
    if vb:
        base = os.path.basename(vb)
        stem = base[:-len(".body.json")] if base.endswith(".body.json") \
            else os.path.splitext(base)[0]
        return os.path.join(os.path.dirname(vb), stem + ".voice.json")
    vrm = getattr(a, "vrm", None)
    if vrm:
        stem = os.path.splitext(os.path.basename(vrm))[0]
        return os.path.join("rigs", stem + ".voice.json")
    return None


def make_tts(a, log=print):
    """A speech.VoiceRouter for the --tts backend (or the voice config's
    per language, --tts auto), or None for --tts none (the rate model)."""
    kind = tts_kind(a)
    if kind == "none":
        return None
    if kind not in ("auto",) + sp.TTS_KINDS:
        raise SystemExit(f"unknown --tts {kind}")
    g = lambda k: getattr(a, k, None)          # noqa: E731
    path = voice_config_path(a)
    if g("voice_config") and not os.path.exists(g("voice_config")):
        raise SystemExit(f"--voice-config {g('voice_config')}: no such file")
    cfg = sp.load_voice_config(path, log)
    if cfg:
        log(f"speech: voice config {path}: "
            + "; ".join(f"{k} {v.get('tts', 'edge')}"
                        + (f" {v['voice']}" if v.get("voice") else "")
                        for k, v in cfg.items()))
    cli = {"voice": g("voice"), "rate": g("tts_rate"), "model": g("tts_model"),
           "speed": g("tts_speed"), "pitch": g("tts_pitch")}

    def moods(st):
        # --mood off speaks every sentence in the base voice.
        return False if g("mood") == "off" else st.get("moods")

    def make(kind, st):
        if kind == "edge":
            return sp.EdgeTTS(st.get("voice"), st.get("rate") or "+0%",
                              log=log, moods=moods(st))
        if kind == "siliconflow":
            base, env = OPENAI_COMPAT["siliconflow"]
            # The chat --api-base/--api-key-env are SiliconFlow's only when
            # the chat backend is SiliconFlow: a DeepSeek base must not be
            # used here.
            if g("backend") == "siliconflow":
                base, env = g("api_base") or base, g("api_key_env") or env
            return sp.SiliconFlowTTS(g("tts_api_base") or base,
                                     g("tts_key_env") or env, st.get("model"),
                                     st.get("voice"), g("tts_format") or "wav",
                                     g("tts_sample_rate"), log=log)
        if kind == "voicevox":
            return sp.VoicevoxTTS(g("voicevox_url") or sp.VOICEVOX_URL,
                                  st.get("voice"),
                                  {k: st.get(k) for k in sp.VOICEVOX_KNOBS},
                                  log=log, moods=moods(st))
        if kind == "file":
            if not g("tts_file"):
                raise SystemExit("--tts file needs --tts-file")
            return sp.FileTTS(g("tts_file"))
        raise SystemExit(f"unknown tts {kind}")

    return sp.VoiceRouter(kind, cli, cfg, make, log)


def micro_plan(a, body, said, acts, spoken, log=print):
    """The speech-driven micro-expression tokens (flow.micro) of a spoken
    reply, for body mode with --micro; [] otherwise. The model's pose
    markers (acts) keep them off the same variables."""
    if body is None:
        return []
    try:
        from . import micro as mc
    except ImportError:
        from flow import micro as mc
    m = mc.planner_for(a, body["body"], log)
    if m is None:
        return []
    toks = m.plan(said["audio"], said["sr"], acts, 0.0, 0,
                  mc.sentence_ends(spoken, said["spans"]) or None)
    return toks


def plan(a, reply, out_dir, motions, log=print, tts=None, body=None):
    """Write reply.txt, spoken.txt, actions.json, subs.srt/json, then plan
    tokens.json in-process (runtime --direct --dump-only). Returns (render
    argv, seconds) or None for a refusal.

    tts (make_tts): speak the reply too. speech.wav is written, every tag's
    time and the subtitles come from the audio, the render runs as long as
    the audio plus the tail, and actions.json carries the mouth curve the
    runtime plays (speech.mouth_track).

    body (prepare()'s run in --mode body): the pose markers are solved
    (flow.pose.solve_poses, checked on the Core when one loaded), each pose
    action carries its solve as "solved" for the runtime to play, and
    actions.json gets the reach summary; the render runs until the last
    pose has relaxed."""
    os.makedirs(out_dir, exist_ok=True)
    extra = {}

    def body_solve(acts, seconds):
        if body is None:
            return seconds
        try:
            from . import pose as pz
        except ImportError:
            from flow import pose as pz
        bm = body["body"]
        end_m = 0.0
        if any(x.get("tag") == "move" for x in acts):
            # Motion primitives (flow.moves): built on the body, an idle
            # gesture under an arm figure dropped, each carried on its
            # action as "solved" for the runtime.
            try:
                from . import moves as mv
            except ImportError:
                from flow import moves as mv
            mvs = mv.solve_moves(bm, acts, log=log,
                                 seed=getattr(a, "seed", 0))
            mv.drop_clashing_gestures(acts, mvs, motions, log)
            for rec in mvs:
                acts[rec["act"]]["solved"] = rec
            for line in mv.report_lines(mvs, reply["spoken"]):
                log("move: " + line)
            if mvs:
                extra.update(mode="body", moves=len(mvs))
            end_m = mv.end_time(mvs) + a.tail
        poses = pz.solve_poses(bm, acts, core=body.get("core"), log=log)
        arms = body.setdefault("arm_params", pz.arm_params(bm))
        for rec in poses:
            rec["arm_params"] = sorted(set(rec["params"]) & arms)
        pz.drop_clashing_gestures(acts, poses, motions, log)
        for rec in poses:
            acts[rec["act"]]["solved"] = rec
        for line in pz.report_lines(poses, reply["spoken"]):
            log("pose: " + line)
        rs = pz.reach_summary(poses)
        log(f"pose: {rs['markers']} markers, {rs['ok']}/{rs['targets']} "
            "targets reached"
            + (f", {rs['exact_ok']}/{rs['exact_n']} on the Core"
               if rs["exact_n"] else "") + f" (solve {rs['solve_seconds']}s)")
        extra.update(mode="body", reach=rs)
        return max(seconds, pz.end_time(poses, getattr(a, "pose_ease", False))
                   + a.tail, end_m)
    if reply["refused"]:
        with open(os.path.join(out_dir, "refusal.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(f"stop_reason: {reply['stop_reason']}\n")
        log("director: the model refused this turn; no tokens written")
        return None
    with open(os.path.join(out_dir, "reply.txt"), "w", encoding="utf-8") as fh:
        fh.write(reply["raw"])
    with open(os.path.join(out_dir, "spoken.txt"), "w", encoding="utf-8") as fh:
        fh.write(reply["spoken"] + "\n")
    if tts is not None and reply["spoken"]:
        moods = None
        if getattr(a, "mood", None) != "off":
            moods = sp.sentence_moods(reply["spoken"], reply["tags"])
        said = sp.speak(reply["spoken"], reply["tags"], tts,
                        lambda ch: 1.0 / (a.cjk_rate if _is_cjk(ch)
                                          else a.latin_rate), log=log,
                        moods=moods)
        acts, spans, end = said["acts"], said["spans"], said["end"]
        sp.write_wav(os.path.join(out_dir, "speech.wav"), said["audio"],
                     said["sr"])
        seconds = body_solve(acts, length(acts, end, motions, a.tail))
        micro_toks = micro_plan(a, body, said, acts, reply["spoken"], log)
        if micro_toks:
            extra["micro"] = micro_toks
        extra.update({"tts": {"backend": getattr(tts, "kind", None) or a.tts,
                         "voice": tts.last_voice,
                         "timing": said["method"],
                         "voiced_end": round(said["speech_end"], 3)}})
        credit = getattr(tts, "credit", None)
        if credit:
            extra["tts"]["credit"] = credit
        if any(m != "neutral" for _, m in said["moods"]):
            extra["tts"]["moods"] = said["moods"]
            log("speech: moods " + ", ".join(
                f"{m} from {reply['spoken'][at:at + 12]!r}"
                + (f" ({v})" if (v := getattr(tts, "mood_voice", lambda m:
                                               None)(m)) else "")
                for at, m in said["moods"]))
        lipsync = getattr(a, "lipsync", None) or "auto"
        if lipsync != "off":
            # Sampled at the run's own frame times (runtime n_frames is
            # int(seconds * fps) of the --seconds it is given, rounded to
            # 2 decimals there); the runtime pads or crops to its count.
            cal = getattr(a, "mouth_cal", None) if lipsync == "auto" else None
            tr = sp.mouth_track(
                said["audio"], said["sr"], a.fps,
                int(round(seconds, 2) * a.fps) + 1, lipsync, a.mouth_gain,
                cal=cal, moras=said.get("moras"))
            # auto without a calibrated map is open, token and all.
            mode = lipsync if lipsync != "auto" or "map" in tr else "open"
            extra["mouth"] = dict(tr, mode=mode)
            if "gamma" in tr:
                log(f"mouth: openness ** {tr['gamma']:g} toward the rig's "
                    f"talking level {cal.get('level')}")
        log(f"speech: {end:.2f}s audio ({tts.last_voice}), tag times from "
            f"{said['method']} -> speech.wav")
        if credit:
            log(f"speech: credit \"{credit}\" wherever this audio is "
                "published (VOICEVOX voice library terms)")
    else:
        # A speech.wav left by an earlier run into this folder would be
        # muxed onto this silent one.
        stale = os.path.join(out_dir, "speech.wav")
        if os.path.exists(stale):
            os.remove(stale)
        acts, spans, end = timeline(reply["spoken"], reply["tags"],
                                    a.cjk_rate, a.latin_rate)
        seconds = body_solve(acts, length(acts, end, motions, a.tail))
    segs = subtitles(reply["spoken"], spans)
    with open(os.path.join(out_dir, "actions.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"speech_seconds": round(end, 3), "seconds": round(seconds, 3),
                   "actions": acts, "warnings": reply["warnings"],
                   "model_seconds": reply.get("model_seconds"),
                   "first_text_seconds": reply.get("first_text_seconds"),
                   "stop_reason": reply.get("stop_reason"),
                   "usage": reply.get("usage"), **extra}, fh,
                  indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "subs.srt"), "w", encoding="utf-8") as fh:
        fh.write(srt(segs))
    with open(os.path.join(out_dir, "subs.json"), "w", encoding="utf-8") as fh:
        json.dump(segs, fh, indent=1, ensure_ascii=False)
    log(f"director: {len(reply['spoken'])} spoken characters, "
        f"{end:.2f}s speech, {len(acts)} actions, {len(segs)} subtitles; "
        f"rendering {seconds:.2f}s")
    try:
        from .runtime import main as runtime_main
    except ImportError:
        from flow.runtime import main as runtime_main
    tok = os.path.join(out_dir, "tokens.json")
    base = runtime_args(a, seconds)
    rc = runtime_main(base + ["--direct", os.path.join(out_dir, "actions.json"),
                              "--tokens", tok, "--dump-only", "--dump",
                              os.path.join(out_dir, "plan.npz")])
    if rc:
        raise SystemExit(f"runtime planning failed ({rc})")
    cmd = [sys.executable, "-m", "flow.runtime"] + base + [
        "--tokens-in", tok, "--size", str(a.size),
        "--out", os.path.join(out_dir, "render")]
    return cmd, seconds


def render(cmd, out_dir, log=print):
    """Run the render in its own process (one GL backend per process), then
    mux the subtitles in as a soft track when ffmpeg is there."""
    log("director: " + shlex.join(cmd))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rc = subprocess.call(cmd, cwd=root)
    if rc:
        log(f"director: render exited {rc}")
        return rc
    vid = os.path.join(out_dir, "render", "preview.mp4")
    cmd = mux_cmd(vid, os.path.join(out_dir, "speech.wav"),
                  os.path.join(out_dir, "subs.srt"),
                  os.path.join(out_dir, "demo.mp4"))
    if shutil.which("ffmpeg") and os.path.exists(vid) and cmd:
        r = subprocess.run(cmd, capture_output=True)
        what = " and ".join(w for w, k in (("speech", "1:a"),
                                           ("subtitles", ":s"))
                            if any(x.endswith(k) for x in cmd))
        log(f"director: video with {what} -> {cmd[-1]}" if r.returncode == 0
            else f"director: ffmpeg could not mux {what}; preview.mp4 kept: "
            + r.stderr.decode("utf-8", "replace")[-300:])
    return 0


def mux_cmd(vid, wav, subs, dest):
    """ffmpeg argv putting speech (AAC) and soft subtitles (mov_text) on the
    rendered video, whichever exist; None when neither does. No -shortest:
    the video runs the tail past the audio. The PNGs are encoded at the run
    fps (runtime._encode -framerate), so frame f is at f / fps and the audio,
    which starts at 0 too, stays in step."""
    ins, maps, codecs = ["-i", vid], ["-map", "0:v"], ["-c:v", "copy"]
    if os.path.exists(wav):
        ins += ["-i", wav]
        maps += ["-map", f"{len(ins) // 2 - 1}:a"]
        codecs += ["-c:a", "aac", "-b:a", "128k"]
    if os.path.exists(subs) and os.path.getsize(subs):
        ins += ["-i", subs]
        maps += ["-map", f"{len(ins) // 2 - 1}:s"]
        codecs += ["-c:s", "mov_text"]
    if len(ins) == 2:
        return None
    return ["ffmpeg", "-y"] + ins + maps + codecs + [dest]


def add_args(ap):
    """The arguments flow.director and flow.live share: the rig, the
    language model, speech and lip sync."""
    ap.add_argument("--model-json", default=None,
                    help="the rig's .model3.json (required for a Live2D "
                         "avatar)")
    ap.add_argument("--rig", default=None, help="rigs/<name>.npz (required "
                    "for a Live2D avatar)")
    ap.add_argument("--avatar", default="live2d",
                    choices=("live2d", "vrm", "both"),
                    help="live2d [default]: the Live2D rig; vrm: a VRM "
                         "avatar (--vrm), drawn by the browser from body "
                         "states (flow.vrm; docs/embodiment.md E3); both "
                         "(flow.live only): the Live2D rig and the VRM side "
                         "by side, one reply driving both")
    ap.add_argument("--vrm", default=None, metavar="AVATAR.vrm",
                    help="--avatar vrm/both: the VRM file (its body file "
                         "rigs/<name>.body.json is used when it names this "
                         "file, else built from it, flow.vrmbody)")
    ap.add_argument("--vrm-body", default=None, metavar="BODY.json",
                    help="--avatar vrm/both: a VRM body file (flow.vrmbody "
                         "--out)")
    ap.add_argument("--clips", default=None, help="clips/<name>.npz (column "
                    "alignment, as flow.runtime)")
    ap.add_argument("--events", default="default", metavar="STATS.json",
                    help="fitted event statistics (flow.events --fit)")
    ap.add_argument("--probe-dir", default="out")
    ap.add_argument("--binding", default=None)
    ap.add_argument("--events-pool-only", action="store_true")
    ap.add_argument("--no-snap", action="store_true")
    ap.add_argument("--no-idle", action="store_true",
                    help="blinks only besides the tags: for checking what "
                         "each tag does")
    ap.add_argument("--head-drift", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="a slow idle wander of the head, a few degrees "
                         "(flow.stream.HeadDrift) [default on]; "
                         "--no-head-drift: the head holds still between the "
                         "idle's fixations")
    ap.add_argument("--head-drift-scale", type=float, default=1.0,
                    metavar="X", help="the drift's amplitude, x its "
                    "defaults (2.5 deg yaw, 1.5 pitch, 2 tilt; the body "
                    "sways 0.8 / 0.6) [1]; on in the offline render too")
    ap.add_argument("--pose-ease", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="organic pose playback: head / torso targets take "
                         "at least 0.35 s to arrive and 0.7 s to relax, "
                         "face targets 0.12 s (flow.pose.ANGLE_DUR); what "
                         "the model asked for and when is unchanged "
                         "[default on]; --no-pose-ease: the beats as "
                         "written")
    ap.add_argument("--camera-follow", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="in bust, the camera eases to full ~0.3 s before a "
                         "whole-body move (dance, twirl, shimmy, bounce, "
                         "cheer) and back ~0.5 s after it, unless a [frame] "
                         "marker of the model says otherwise there "
                         "(flow.framing.follow_tokens) [default on]; "
                         "--no-camera-follow: the camera stays")
    try:
        from . import micro as mc
    except ImportError:
        from flow import micro as mc
    mc.add_args(ap)
    ap.add_argument("--idle-profile", default="calm",
                    choices=("calm", "authored"),
                    help="the idle between and under the tags (flow.runtime "
                         "--idle-profile); authored: as before the calm "
                         "profile")
    ap.add_argument("--persona", default=None, help="who the character is "
                    "(first line of the system prompt)")
    ap.add_argument("--backend", default="anthropic",
                    choices=("anthropic", "siliconflow", "deepseek", "ollama",
                             "hf", "file", "stdin"))
    ap.add_argument("--model", default=None, help="language model (default "
                    + ", ".join(f"{k}: {v}" for k, v in DEFAULT_MODEL.items())
                    + ")")
    ap.add_argument("--effort", default="low", help="anthropic effort")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="reply token cap, thinking included [2048; 8192 "
                         "for siliconflow/deepseek, whose reasoning models "
                         "count their thinking: Qwen3.6 spent 2048 on it "
                         "and said nothing]")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-ctx", type=int, default=None,
                    help="ollama: num_ctx per request [the model's own: "
                         "a stock model's default may cut the body prompt]")
    ap.add_argument("--hf-adapter", default=None, metavar="DIR",
                    help="--backend hf: a LoRA adapter on --model "
                         "(flow.train.sft's <run>/adapter)")
    ap.add_argument("--hf-template", default=None,
                    choices=("qwen3", "chatml", "gemma3", "hf"),
                    help="--backend hf: the chat template family [the "
                         "adapter's train.json, else from --model]")
    ap.add_argument("--hf-4bit", action="store_true",
                    help="--backend hf: the base in 4 bit (bitsandbytes)")
    ap.add_argument("--hf-greedy", action="store_true",
                    help="--backend hf: greedy decoding (no sampling)")
    ap.add_argument("--thinking", default=None, choices=("off",),
                    help="siliconflow/deepseek: send enable_thinking=false "
                         "(cuts the wait before the first word on models "
                         "that support it; some reject it with a 400)")
    ap.add_argument("--api-base", default=None, help="siliconflow/deepseek: "
                    "override the base URL (e.g. https://api.siliconflow.cn/v1)")
    ap.add_argument("--api-key-env", default=None, help="siliconflow/deepseek: "
                    "environment variable holding the key (default "
                    "SILICONFLOW_API_KEY / DEEPSEEK_API_KEY)")
    ap.add_argument("--reply-file", default=None,
                    help="--backend file: a tagged reply to use as is")
    ap.add_argument("--cjk-rate", type=float, default=CJK_RATE,
                    help="CJK characters spoken per second")
    ap.add_argument("--latin-rate", type=float, default=LATIN_RATE,
                    help="other characters (spaces included) per second")
    ap.add_argument("--tts", default=None,
                    choices=("none", "auto", "edge", "siliconflow",
                             "voicevox", "file"),
                    help="speak the reply: tag times, subtitles and the "
                         "mouth then follow the audio (speech.wav). none "
                         "[default; auto with --voice-config]: silent, times "
                         "from --cjk-rate/--latin-rate; auto: per language "
                         "from the voice config, edge where it names none; "
                         "edge: edge-tts (pip install edge-tts); "
                         "siliconflow: its /audio/speech "
                         "(SILICONFLOW_API_KEY); voicevox: a VOICEVOX engine "
                         "at --voicevox-url (Japanese; other languages go "
                         "to edge); file: --tts-file")
    ap.add_argument("--voice-config", default=None, metavar="FILE.json",
                    help="per-language voices: {\"ja\": {\"tts\": "
                         "\"voicevox\", \"voice\": \"name/style\", "
                         "\"speed\": 1.0}, \"en\": {\"tts\": \"edge\", "
                         "\"voice\": \"en-US-AnaNeural\", \"rate\": "
                         "\"+0%%\"}}; keys tts, voice, rate (edge), model "
                         "(siliconflow), speed/pitch/intonation/volume "
                         "(voicevox). Flags win. Default with --tts: <rig "
                         "without .npz>.voice.json when it exists")
    ap.add_argument("--breath-gain", default="auto",
                    help="share of the breath parameter's range the breath "
                         "swings over; auto [default]: the voice config's "
                         "top-level \"breath_gain\", else calibrated from "
                         "<rig>.body.json to the same small displacement on "
                         "every rig (flow.runtime.breath_setting)")
    ap.add_argument("--voicevox-url", default=sp.VOICEVOX_URL,
                    help="VOICEVOX engine base URL (no proxy for localhost)")
    ap.add_argument("--tts-speed", type=float, default=None,
                    help="voicevox speedScale")
    ap.add_argument("--tts-pitch", type=float, default=None,
                    help="voicevox pitchScale")
    ap.add_argument("--voice", default=None,
                    help="TTS voice. voicevox: a style id, "
                         "\"character/style\" or a character (GET /speakers "
                         "lists them); edge default by the reply's script: "
                         + ", ".join(f"{k} {v}" for k, v in
                                     sp.EDGE_VOICES.items())
                         + f"; siliconflow default {sp.SF_TTS_VOICE} (a name "
                         "without ':' is prefixed '<tts-model>:')")
    ap.add_argument("--tts-model", default=None, help="siliconflow TTS "
                    f"model (default {sp.SF_TTS_MODEL}; unverified name)")
    ap.add_argument("--tts-format", default="wav", choices=("wav", "mp3"),
                    help="siliconflow response_format")
    ap.add_argument("--tts-sample-rate", type=int, default=None,
                    help="siliconflow sample_rate (not sent by default)")
    ap.add_argument("--tts-api-base", default=None, help="siliconflow TTS "
                    "base URL (default --api-base when --backend siliconflow, "
                    "else https://api.siliconflow.com/v1)")
    ap.add_argument("--tts-key-env", default=None, help="siliconflow TTS key "
                    "variable (default SILICONFLOW_API_KEY)")
    ap.add_argument("--tts-rate", default=None,
                    help="edge speaking rate, e.g. +10%% [+0%%]")
    ap.add_argument("--tts-file", default=None,
                    help="--tts file: a wav/mp3 of the reply")
    ap.add_argument("--lipsync", default=None,
                    choices=("auto", "open", "vowels", "off"),
                    help="with --tts: auto [default] drives the mouth "
                         "parameters the rig's own talking motions use, over "
                         "their authored speaking range (flow.mouth; open "
                         "where the motions have no lip flaps); open drives "
                         "the mouth-open parameter from the speech level, "
                         "rest to maximum; vowels also drives ParamA/I/U/E/O "
                         "where the rig has them (VOICEVOX: from the moras; "
                         "else a crude formant estimate); off: still mouth")
    ap.add_argument("--mouth-gain", type=float, default=1.0,
                    help="scale on the mouth opening (1: the utterance's "
                         "loud level opens it fully)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--frame", default="bust", choices=("bust", "full"),
                    help="Live2D renders: bust = head and chest (flow."
                         "framing), full = the whole canvas [bust]; the "
                         "home framing, every turn starts in it; a [frame] "
                         "marker of the beats prompt switches it until the "
                         "end of that reply")
    ap.add_argument("--gestures", default=None, metavar="FILE.json",
                    help="per-rig gesture descriptions: a JSON object label "
                         "-> text, or label -> {\"desc\": text, \"hide\": "
                         "true}; wins over the measured lines (flow."
                         "gestures). Default: <rig without .npz>.gestures."
                         "json when it exists")
    ap.add_argument("--mode", default="tags", choices=("tags", "body"),
                    help="tags [default]: the action tag vocabulary; body: "
                         "the model gets its measured body (flow.body, "
                         "--body) and writes [pose {...}] targets that the "
                         "solver plays (flow.pose; docs/embodiment.md E2)")
    ap.add_argument("--body", default=None, metavar="BODY.json",
                    help="--mode body: the rig's body file (default <rig "
                         "without .npz>.body.json)")
    ap.add_argument("--body-gestures", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="--mode body: offer the rig's own motions "
                         "([gesture:NAME], as tag mode offers them) next to "
                         "the poses [default on]; --no-body-gestures: pure "
                         "body targets, as E2 was first measured")
    ap.add_argument("--body-moves", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="--mode body: offer motion primitives ([move "
                         "{\"name\": \"dance\"}]: sway, bounce, clap, cheer, "
                         "... as this body can do them; flow.moves) [default "
                         "on]; --no-body-moves: the prompt and parser of "
                         "before them")
    ap.add_argument("--body-prompt", default="v1",
                    choices=("v1", "beats", "beats2"),
                    help="--mode body: v1 [default]: the prompt the first "
                         "students were taught with; beats: the expression "
                         "beats (1-3 face beats a sentence, each with a small "
                         "head move, a hand at the moments that call for one, "
                         "the camera line and [frame] markers; flow.pose "
                         "_beats_rule); beats2: beats with the first dense "
                         "example of several, [frame full] before whole-body "
                         "moves and the \"(reply in: ...)\" line on each "
                         "user turn. A student is served with the prompt it "
                         "was trained on")
    ap.add_argument("--body-image", action="store_true",
                    help="--mode body: send the model a picture of itself "
                         "with each line (anthropic; siliconflow and ollama "
                         "with a vision model; deepseek chat is text only "
                         "and gets none). Offline it is rendered in a "
                         "subprocess, so it needs GL")
    ap.add_argument("--mood", default="auto", choices=("auto", "off"),
                    help="with --tts: speak each sentence in its mood "
                         "(speech.mood_of: from its brow/nod/shake tags, "
                         "words and punctuation) -- a VOICEVOX style of the "
                         "same character found by name, or edge rate/pitch/"
                         "volume offsets; a voice config's \"moods\" names "
                         "them. off: the base voice always")


def body_path(a):
    """--body, else <rig without .npz>.body.json."""
    return getattr(a, "body", None) or (os.path.splitext(a.rig)[0]
                                        + ".body.json")


def move_menu(a, bm):
    """(prompt lines, names) of the motion primitives offered on this body
    (flow.moves) with --body-moves; ("", []) without."""
    if not getattr(a, "body_moves", True):
        return "", []
    try:
        from . import moves as mv
    except ImportError:
        from flow import moves as mv
    return mv.prompt_lines(bm), mv.offered(bm)


def prepare_body(a, info=sys.stdout):
    """prepare() for --mode body: the body file, the body-mode system
    prompt (flow.pose), a parser factory and, when the model file loads,
    the Core for the exact check of every solve."""
    try:
        from . import body as fb
        from . import pose as pz
        from .runtime import authored_motions
    except ImportError:
        from flow import body as fb
        from flow import pose as pz
        from flow.runtime import authored_motions
    path = body_path(a)
    if not os.path.exists(path):
        raise SystemExit(f"--mode body needs the body file {path} (python -m "
                         "flow.body --measure)")
    bm = fb.load(path)
    images = bool(getattr(a, "body_image", False))
    if images and a.backend not in pz.IMAGE_BACKENDS:
        print(f"director: --body-image: {a.backend} takes no images "
              f"(only {', '.join(pz.IMAGE_BACKENDS)}); text only",
              file=info)
        images = False
    core = None
    try:
        try:
            from ..l2dprobe import core as l2dcore
        except ImportError:
            from l2dprobe import core as l2dcore
        with _quiet_stdout(info):
            core = l2dcore.load(a.model_json)
    except Exception as exc:  # noqa: BLE001 - the check is optional
        print(f"director: no exact Core check ({exc.__class__.__name__}: "
              f"{exc})", file=info)
    motions = {n: m["frames"] / a.fps
               for n, m in authored_motions(a.model_json, a.fps)}
    labels, descriptions, glines = {}, {}, ""
    if getattr(a, "body_gestures", True):
        # The rig's own motions next to the poses, offered as tag mode
        # offers them: in the E2 A/B lafei's tag replies won with an
        # authored gesture (login) where her body replies, which had none,
        # moved a hand twice in 12 poses.
        try:
            from ..l2dprobe.motions import motion_files
            from .runtime import event_roles
        except ImportError:
            from l2dprobe.motions import motion_files
            from flow.runtime import event_roles
        names = [os.path.basename(p) for p in motion_files(a.model_json)]
        _st, roles, _ = event_roles(a.model_json, a.rig, a.events, a.clips,
                                    a.binding, a.fps, a.no_snap,
                                    a.events_pool_only, log=lambda s: None)
        labels, descriptions, same = offer_gestures(
            a, roles, gesture_labels(names), info)
        glines = gesture_lines(labels, descriptions, same)
    mlines, mnames = move_menu(a, bm)
    print(f"director: body mode, {path}: {len(pz.offered(bm))} variables, "
          f"{len(bm.variants)} arm variants"
          + (f", {len(labels)} gestures" if labels else "")
          + (f", motions {' '.join(mnames)}" if mnames else "")
          + (", images" if images else ""), file=info)
    return {"labels": labels, "offered": None, "descriptions": descriptions,
            "system": pz.system_prompt(bm, a.persona, images, glines,
                                       mlines, getattr(a, "body_prompt",
                                                       "v1")),
            "system_for": lambda k: pz.system_prompt(
                bm, a.persona, images, glines, mlines,
                getattr(a, "body_prompt", "v1"), k),
            "motions": motions, "stats": None, "roles": None,
            "mode": "body", "body": bm, "body_path": path, "core": core,
            "images": images, "max_tag": pz.MAX_MARKER,
            "parser": lambda: pz.parser_for(
                bm, gestures=labels, moves=bool(mnames),
                frames=pz.beats_like(getattr(a, "body_prompt", "v1")))}


def _quiet_stdout(info):
    """The Core prints its version banner from C; keep it off stdout when
    stdout is the prompt (--print-system)."""
    import contextlib
    if info is sys.stderr:
        try:
            from .runtime import _stdout_to_stderr
        except ImportError:
            from flow.runtime import _stdout_to_stderr
        return _stdout_to_stderr()
    return contextlib.nullcontext()


def snapshot(a, run, png, values=None, parts=None, log=print):
    """Render the body at these parameter values (default its rest) to
    png in a subprocess (one GL backend per process); the PNG bytes, or
    None when it could not be rendered."""
    cmd = [sys.executable, "-m", "flow.pose", "snapshot", "--body",
           run["body_path"], "--model-json", a.model_json, "--out", png,
           "--size", str(min(int(a.size), 512))]
    if values:
        cmd += ["--current", json.dumps(values)]
    if parts:
        cmd += ["--parts", json.dumps(parts)]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if r.returncode or not os.path.exists(png):
        log("director: snapshot failed, no image this turn: "
            + (r.stderr or r.stdout)[-300:])
        return None
    with open(png, "rb") as fh:
        return fh.read()


def offer_gestures(a, roles, labels, info=sys.stdout):
    """(labels, descriptions, same) of the gestures a prompt offers, from
    the rig's motion labels: measured on what the runtime will play with
    these roles bound (flow.gestures), then the rig's override file
    (hidden ones left out). Prints what each one shows to `info`. Tag mode
    and body mode with --body-gestures offer the same."""
    # What each gesture shows, measured on what the runtime will play with
    # these roles bound (flow.gestures), then the rig's override file.
    if a.gestures and not os.path.exists(a.gestures):
        raise SystemExit(f"--gestures {a.gestures}: no such file")
    gpath = a.gestures or gz.override_path(a.rig)
    over = gz.load_overrides(gpath)
    measured, groups = gz.rig_descriptions(a.model_json, a.rig, roles,
                                           a.clips, a.fps, GESTURE_MAX,
                                           a.probe_dir)
    all_labels = labels
    labels, descriptions = gz.apply_overrides(labels, measured, over)
    same = {lab: groups.get(n, n) for lab, n in labels.items()}
    if over:
        print(f"director: gesture overrides from {gpath}", file=info)
    for lab in sorted(all_labels):
        o = over.get(lab, {})
        src = ("hidden" if o.get("hide") else
               "override" if o.get("desc") else "measured")
        g = groups.get(all_labels[lab], all_labels[lab])
        mates = sorted(x for x, n in all_labels.items()
                       if x != lab and groups.get(n, n) == g)
        if mates:
            src += ", same motion as " + " / ".join(mates)
        text = (descriptions.get(lab) if lab in labels
                else measured.get(all_labels[lab]))
        print(f"director: gesture {lab} [{src}]: "
              + (text or "(label only)"), file=info)
    return labels, descriptions, same


def prepare(a, info=sys.stdout):
    """What a director run needs before its first turn, from add_args'
    arguments: {"labels", "offered", "system", "motions", "stats",
    "roles"}. Prints what it offers and what each gesture shows to
    `info`; flow.live calls it too. --mode body: prepare_body."""
    if getattr(a, "mode", "tags") == "body":
        return prepare_body(a, info)
    try:
        from ..l2dprobe.motions import motion_files
    except ImportError:
        from l2dprobe.motions import motion_files
    try:
        from .runtime import authored_motions, event_roles
    except ImportError:
        from flow.runtime import authored_motions, event_roles
    names = [os.path.basename(p) for p in motion_files(a.model_json)]
    labels = gesture_labels(names)
    # The roles and adapted statistics the runtime will plan with, bound the
    # same way (its log lines come again when plan() runs it), so the prompt
    # offers only tags that move something on this rig.
    st, roles, _ = event_roles(a.model_json, a.rig, a.events, a.clips,
                               a.binding, a.fps, a.no_snap,
                               a.events_pool_only, log=lambda s: None)
    offered, out = offer(st, roles)
    if out:
        print("director: not offered on this rig: " + left_out_text(out),
              file=info)
    labels, descriptions, same = offer_gestures(a, roles, labels, info)
    system = system_prompt(a.persona, labels, offered, descriptions, same)
    motions = {n: m["frames"] / a.fps
               for n, m in authored_motions(a.model_json, a.fps)}
    print(f"director: {len(labels)} gesture labels: "
          + (", ".join(sorted(labels)) or "none"), file=info)
    return {"labels": labels, "offered": offered, "system": system,
            "motions": motions, "stats": st, "roles": roles,
            "descriptions": descriptions}


def make_voice(a, log=print):
    """make_tts, plus the rig's mouth calibration in a.mouth_cal under
    --lipsync auto (flow.mouth). Returns the TTS or None."""
    tts = make_tts(a, log)
    if tts is not None and (a.lipsync or "auto") == "auto":
        try:
            from .mouth import calibrate, describe
        except ImportError:
            from flow.mouth import calibrate, describe
        try:
            from .runtime import authored_motions
        except ImportError:
            from flow.runtime import authored_motions
        a.mouth_cal = calibrate(a.model_json, a.rig, a.fps,
                                authored_motions(a.model_json, a.fps))
        log(describe(a.mouth_cal))
    return tts


def self_check(a, body, backend, system, messages, reply, out_dir, motions,
               log=print):
    """--body-check: one round of self-correction, offline. The model is
    shown what each of its poses reached (the Core's numbers where it
    loaded, the miss hints) and, with images, a picture of the last pose;
    an "OK" ends it, anything else is parsed as the corrected reply and
    planned (without speech: reach is what is judged) into out_dir/check.
    check.json: both replies, reach before and after. The exchange is not
    kept in the conversation."""
    try:
        from . import pose as pz
    except ImportError:
        from flow import pose as pz
    with open(os.path.join(out_dir, "actions.json"), encoding="utf-8") as fh:
        before = json.load(fh)
    poses = [x["solved"] for x in before["actions"] if x.get("solved")]
    rs0 = before.get("reach") or pz.reach_summary(poses)
    out = {"reply": reply["raw"], "before": rs0}
    if not poses:
        out.update(model_ok=None, after=rs0, note="no pose markers")
    else:
        img = None
        if body["images"]:
            last = poses[-1]
            vals = {}
            for q in poses:
                vals.update(q["params"])
            img = snapshot(a, body, os.path.join(out_dir, "check_seen.png"),
                           vals, last["parts"] or None, log)
        msg = pz.check_message(poses, reply["spoken"], img is not None)
        out["message"] = msg
        conv = list(messages) + [pz.user_message(msg, a.backend, img)]
        rep2 = ask(backend, system, conv, {}, log=log,
                   parser=body["parser"]())
        out["correction"] = rep2["raw"]
        out["model_ok"] = pz.is_ok_reply(rep2["raw"])
        log(f"director: self-check reply: {rep2['raw'].strip()}")
        if out["model_ok"] or rep2["refused"] or not rep2["spoken"]:
            out["after"] = rs0
        else:
            d = os.path.join(out_dir, "check")
            plan(a, rep2, d, motions, log=log, tts=None, body=body)
            with open(os.path.join(d, "actions.json"), encoding="utf-8") as fh:
                out["after"] = json.load(fh).get("reach") or rs0
    os.makedirs(os.path.join(out_dir, "check"), exist_ok=True)
    with open(os.path.join(out_dir, "check", "check.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    log(f"director: self-check: reach {rs0['ok']}/{rs0['targets']} -> "
        f"{out['after']['ok']}/{out['after']['targets']}"
        + (" (model said OK)" if out.get("model_ok") else ""))
    return out


def add_gs_args(ap):
    """--gesture-search and its knobs (flow.gesture_search; VRM avatars).
    All off by default: without --gesture-search nothing of it is
    imported or run."""
    g = ap.add_argument_group("gesture search (--avatar vrm)")
    g.add_argument("--gesture-search", default=None, metavar="DIR",
                   help="play real co-speech motion retrieved from this "
                        "library (python -m flow.gesture_search build) "
                        "under the markers; needs --tts")
    g.add_argument("--gs-encoders", default=None, metavar="DIR",
                   help="the probe's saved encoders (flow.probe match "
                        "--save)")
    g.add_argument("--gs-model", default=None, metavar="PATH",
                   help="the local model the hidden states came from "
                        "[the hidden folder's meta.json]")
    g.add_argument("--gs-hidden", default=None, metavar="DIR",
                   help="the hidden-state folder (meta.json, proj.npy) "
                        "[the encoders' record]")
    g.add_argument("--gs-variant", default="text+prosody")
    g.add_argument("--gs-amp", type=float, default=1.0,
                   help="arm amplitude about the rest pose [1]")
    g.add_argument("--gs-seed", type=int, default=0)
    g.add_argument("--gs-parts", default="hands", choices=("hands", "full"),
                   help="hands: the arms; full: also head and torso [hands]")
    g.add_argument("--gs-speaker", default=None,
                   help="only this library speaker (comma-separated)")
    g.add_argument("--gs-temperature", type=float, default=0.05,
                   help="0: always the best window [0.05]")
    g.add_argument("--gs-topk", type=int, default=6)
    g.add_argument("--gs-4bit", action="store_true",
                   help="load the text model in 4 bit (use the precision "
                        "the hidden states were extracted at)")
    g.add_argument("--gs-device", default="auto",
                   help="auto | cuda | cpu: the text model and encoders")
    g.add_argument("--gs-baseline", action="store_true",
                   help="also write states_markers.json: the same turn "
                        "without the retrieved motion")


def gs_setup(a, ap, bm, log=print):
    """The GestureSearch of the --gesture-search flags (None without)."""
    if not a.gesture_search:
        return None
    if not a.gs_encoders:
        ap.error("--gesture-search needs --gs-encoders")
    try:
        from . import gesture_search as gsm
    except ImportError:
        from flow import gesture_search as gsm
    return gsm.GestureSearch.from_args(a, bm, log)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_args(ap)
    add_gs_args(ap)
    ap.add_argument("--say", default=None, help="the user's message")
    ap.add_argument("--tail", type=float, default=1.5,
                    help="seconds rendered after the last speech or motion")
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", action="store_true",
                    help="render after planning (needs GL); otherwise print "
                         "the render command")
    ap.add_argument("--chat", action="store_true",
                    help="keep talking: read user lines from stdin, one turn "
                         "directory each (turn01, ...)")
    ap.add_argument("--print-system", action="store_true",
                    help="print the system prompt and stop")
    ap.add_argument("--body-targets", action="store_true",
                    help="= --mode body: the reply's actions are body "
                         "targets, solved and played over the idle")
    ap.add_argument("--body-check", action="store_true",
                    help="--mode body, an offline evaluation: after the "
                         "turn, show the model what each pose reached (and "
                         "its picture with --body-image), let it correct "
                         "the markers once, and re-plan the correction "
                         "into <out>/check (check.json: reach before and "
                         "after)")
    a = ap.parse_args(argv)
    if a.body_targets:
        a.mode = "body"
    if a.body_check and a.mode != "body":
        ap.error("--body-check needs --mode body")
    if a.avatar == "both":
        ap.error("--avatar both is for flow.live; offline, run each avatar")
    if a.avatar == "vrm":
        if a.body_check:
            ap.error("--body-check is Live2D only")
        return vrm_main(a, ap)
    check_live2d_args(a, ap)
    if a.gesture_search:
        print("director: --gesture-search is for VRM avatars (Live2D arms "
              "are drawings, not joints); ignored", file=sys.stderr)
    if a.gestures and not os.path.exists(a.gestures):
        ap.error(f"--gestures {a.gestures}: no such file")
    info = sys.stderr if a.print_system else sys.stdout
    run = prepare(a, info)
    if a.print_system:
        print(run["system"])
        return 0
    labels, offered, system = run["labels"], run["offered"], run["system"]
    motions = run["motions"]
    backend = make_backend(a)
    tts = make_voice(a)
    messages = []

    home_frame = a.frame
    body = run if run.get("mode") == "body" else None
    if body is not None:
        try:
            from . import pose as pz
        except ImportError:
            from flow import pose as pz

    def turn(text, out_dir):
        if body is None:
            messages.append({"role": "user", "content": text})
            parser = None
        else:
            # Each offline turn starts from the idle's rest: the pose the
            # body file measured. The numbers (and the picture) go here,
            # never in the system prompt, which stays byte-stable.
            img = None
            if body["images"]:
                os.makedirs(out_dir, exist_ok=True)
                img = snapshot(a, body, os.path.join(out_dir, "seen.png"))
            # The beats prompt also shows the framing the turn starts in
            # (a.frame, carried from the last [frame] marker).
            tail = pz.turn_tail(body["body"], a.frame, text, a.body_prompt)
            messages.append(pz.user_message(
                text + "\n" + pz.pose_now(body["body"]) + tail,
                a.backend, img))
            parser = body["parser"]()
        reply = ask(backend, system, messages, labels, offered=offered,
                    parser=parser)
        print(f"director: reply: {reply['raw'].strip()}")
        res = plan(a, reply, out_dir, motions, tts=tts, body=body)
        if body is not None:
            messages[-1] = pz.text_only(messages[-1])
            if pz.beats_like(a.body_prompt) and not reply["refused"]:
                # A switch lasts for its reply: the next turn starts, and is
                # shown, back in the home framing (--frame).
                a.frame = home_frame
        if res is None:
            messages.pop()            # a refused turn is not history
            return 3
        # The tagged reply is what the model wrote; keep it as its turn so
        # the next turn sees the same tags it is asked to use.
        messages.append({"role": "assistant", "content": reply["raw"]})
        if a.body_check:
            self_check(a, body, backend, system, messages, reply, out_dir,
                       motions)
        cmd, _ = res
        if a.render:
            return render(cmd, out_dir)
        print("render with:\n  " + shlex.join(cmd))
        return 0

    if not a.chat:
        if a.say is None and a.backend not in ("file", "stdin"):
            ap.error("--say is required (or --chat)")
        return turn(a.say or "", a.out)
    if a.backend in ("file", "stdin"):
        ap.error("--chat needs a live backend")
    n = 0
    rc = 0
    first = a.say
    while True:
        if first is not None:
            line, first = first, None
        else:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
        if not line:
            continue
        n += 1
        rc = turn(line, os.path.join(a.out, f"turn{n:02d}"))
    return 0 if rc in (0, 3) else rc


def check_live2d_args(a, ap):
    """A Live2D avatar needs its model and rig (no longer argparse-required
    since --avatar vrm needs neither)."""
    miss = [f for f, v in (("--model-json", a.model_json), ("--rig", a.rig))
            if not v]
    if miss:
        ap.error("the following arguments are required: " + ", ".join(miss)
                 + " (or --avatar vrm --vrm PATH)")


def vrm_main(a, ap) -> int:
    """main() for --avatar vrm (flow.vrm): plan each turn into states.json
    beside speech.wav and subs.srt; --render draws them in headless Chromium
    (flow.vrmrender) to render/preview.mp4 and demo.mp4."""
    try:
        from . import vrm as fv
        from . import pose as pz
    except ImportError:
        from flow import vrm as fv
        from flow import pose as pz
    if a.vrm is None and a.vrm_body is None:
        ap.error("--avatar vrm needs --vrm PATH (or --vrm-body FILE)")
    if (a.lipsync or "auto") == "auto":
        # A VRM has the five visemes; the Live2D mouth calibration has no
        # rig to read here.
        a.lipsync = "vowels"
    info = sys.stderr if a.print_system else sys.stdout
    bm, bpath = fv.body_for(a.vrm, a.vrm_body,
                            log=lambda s: print(s, file=info))
    run = fv.prepare(a, bm, bpath, info)
    if a.print_system:
        print(run["system"])
        return 0
    backend = make_backend(a)
    tts = make_voice(a)
    messages = []
    body = run if run.get("mode") == "body" else None
    gs = gs_setup(a, ap, bm)
    if gs is not None:
        run["gsearch"] = gs
        if tts is None:
            print("director: --gesture-search needs the spoken audio "
                  "(--tts edge|voicevox|...); turns play without it",
                  file=sys.stderr)

    def turn(text, out_dir):
        if body is None:
            messages.append({"role": "user", "content": text})
            parser = None
        else:
            # A VRM is drawn at one camera: always full (beats prompt).
            tail = pz.turn_tail(bm, "full", text, a.body_prompt)
            messages.append(pz.user_message(
                text + "\n" + pz.pose_now(bm) + tail, a.backend, None))
            parser = body["parser"]()
        reply = ask(backend, run["system"], messages, {},
                    offered=run["offered"], parser=parser)
        print(f"director: reply: {reply['raw'].strip()}")
        res = fv.plan(a, reply, out_dir, run, tts=tts)
        if res is None:
            messages.pop()
            return 3
        messages.append({"role": "assistant", "content": reply["raw"]})
        cmd = [sys.executable, "-m", "flow.vrmrender", "--run", out_dir]
        if a.render:
            try:
                from . import vrmrender as vr
            except ImportError:
                from flow import vrmrender as vr
            return vr.main(["--run", out_dir])
        print("render with:\n  " + shlex.join(cmd))
        return 0

    if not a.chat:
        if a.say is None and a.backend not in ("file", "stdin"):
            ap.error("--say is required (or --chat)")
        return turn(a.say or "", a.out)
    n, rc, first = 0, 0, a.say
    while True:
        if first is not None:
            line, first = first, None
        else:
            try:
                line = input("you> ").strip()
            except EOFError:
                break
        if not line:
            continue
        n += 1
        rc = turn(line, os.path.join(a.out, f"turn{n:02d}"))
    return 0 if rc in (0, 3) else rc


if __name__ == "__main__":
    raise SystemExit(main())
