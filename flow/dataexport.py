#!/usr/bin/env python3
"""
Quality filters, stats and exporters for flow.dataset's turn records.

  flags      per turn (flag()): hard reasons drop it -- refused, empty
             (nothing spoken), cutoff (the token limit), parse (a parser
             warning other than an unplayed marker), no_body (not one
             marker), miss (MISS share of its pose targets above
             --max-miss), dup (a near-identical user line earlier in the
             same avatar x persona, dedupe()), lite (a Live2D avatar
             generated without its model: its prompt has no gestures and its
             trajectory no idle; kept only with --keep-lite). Soft ones are
             counted only: no_pose, unplayed (markers parsed away, e.g. an
             invented [fPose ...] or a newer marker kind: kept out of the SFT
             exports all the same, SFT_HARD), planned (play_warnings: a gesture dropped
             under a pose or move, a move the body lacks), usage_estimated.
  correct    --correct-misses (default on for sft and stats): a pose target
             the solver missed is written as the value it reached. Per pose
             marker in a reply, the report of its pose record says which
             targets missed (ok false) and what the rig did reach (got); the
             marker's JSON gets that number in place of the missed value (an
             anchor such as "@shoulder.y" included): 2 decimals for hu and
             unit variables, whole numbers for deg and pct, whatever the
             unit's tolerance (body.TOL) is far beyond. Reached targets,
             dur / hold, the spoken text and every other byte of the reply
             are as written. The marker is found in reply_raw by the pose
             parser's own bracket scan and tied to its pose record by order,
             char offset (`at`) and targets; where the two lists do not
             agree the turn is left as it was and counted, as are records
             from before the report existed and markers whose key cannot be
             found in the text (a composite target, a missed got of null).
             A turn whose misses are all corrected no longer counts as a
             MISS turn for the export (flag miss_fixed, soft; "miss" stays in
             the flags and the stats), so a session is not cut there. The
             action export keeps what was played. stats counts markers and
             targets corrected, per avatar, and the skipped.
  weights    --avatar-weight lafei=2 (repeatable, or lafei=2,Hiyori=3): the
             SFT export writes that avatar's examples N times in a row, in
             the same order every run; the trainer's session split keeps
             copies together. Default: each once.
  dedupe     user lines normalised (case, punctuation, spaces) and compared
             by character-trigram Jaccard (>= --dup-jaccard) within an
             avatar x persona; the first one is kept.
  manners    --manners (default on for sft and stats; --no-manners for the
             output of before; --max-poses N, default 2): the teacher writes
             about 2.3 pose markers a reply and a third of them move a hand,
             and the student copies it -- arms up "for no reason" (lafei's
             can only swing out sideways) -- where streamers move the head,
             the face, the gaze and a little torso, and the hands rarely and
             on purpose. Per kept turn it is decided whether the user asked
             for a body action (the topic is pose / show / invite, or the
             line has a word of ASK_WORDS in en / ja / zh) and recorded for
             `stats`. Not asked: each [pose {...}] keeps head, gaze, brow,
             eye, mouth and cheek targets, torso yaw / pitch / tilt within
             TORSO_SMALL degrees, dur and hold; hands, arms, shoulders, a
             bigger torso move and composites go, a marker left with no
             target goes whole; [move] markers other than the head-only ones
             (HEAD_MOVES) and every [gesture:NAME] go; [blink] and [pause]
             stay. Asked: the body markers stay. Either way at most
             --max-poses pose markers stay, the first ones, but those with
             body targets first on an asked turn. It runs on the corrected
             reply (after --correct-misses), on history turns too, and the
             spoken text is checked with the pose parser: a turn whose
             spoken text would change is left as it was and counted.
  beats      the r2 teacher prompt (flow.pose "beats", flow.dataset
             --teacher-prompt) writes 1-3 expression beats a sentence: a
             pose marker whose targets are the face (brow, eye, mouth, cheek,
             gaze) with optional head angles. Counted by `stats` (beats per
             sentence, the face and the hand share of all pose targets), on
             the exported text, before and after manners. --max-beats N
             [default: N = BEATS_MAX when the dataset's avatars.json holds
             the beats prompt, else 0]: with N > 0 the beat markers are
             capped at N a reply on their own and --max-poses counts the
             other pose markers only; 0 is manners as before, a beat a pose
             marker like any, so a reply of beats keeps 2.
  stats      counts by avatar / persona / language / topic / marker tag,
             reach and MISS rates, a histogram of the targeted variables,
             tokens and cost from the usage fields, speech seconds, latency,
             flags by reason (`python -m flow.dataexport stats OUT`).
  sft        chat-format JSONL for a LoRA fine-tune: {"messages": [system,
             user, assistant, ...]} -- the OpenAI messages format that TRL's
             SFTTrainer, Unsloth and LLaMA-Factory (formatting: sharegpt
             with role/content tags, or its "openai" alias) read. The
             system prompt is the avatar's body-mode prompt byte for byte
             (avatars.json), the user turns are what the model saw (line +
             pose_now), the assistant turns the raw marked replies.
             --per session [default]: one example per session, cut before
             its first dropped turn (a dup alone does not cut: the reply
             is new); every assistant turn is a target, as these trainers
             train all of them. --per turn: one example per
             kept turn with its history; earlier assistant turns carry
             "weight": 0 (OpenAI's fine-tuning format; elsewhere use the
             trainer's last-turn-only option, e.g. LLaMA-Factory
             mask_history). --meta adds a "meta" key (ids, avatar, reach).
  action     (text, timed canonical trajectory) pairs for an action head:
             <out>.jsonl (one line per kept turn: id, avatar, vars, fps,
             spoken, reply_raw, markers with times, user line, the npz key)
             and <out>.npz with per key: traj [frames, vars] float16
             (--traj directed [default]: what the reply asks for; played:
             with the idle), chars [len(spoken), 2] each character's start,
             end, variant [frames].

Usage:
    python -m flow.dataexport stats data/datagen/pilot
    python -m flow.dataexport sft data/datagen/pilot --out sft.jsonl
    python -m flow.dataexport sft data/datagen/pilot --out sft.jsonl \\
        --avatar-weight lafei=2 --no-correct-misses --no-manners
    python -m flow.dataexport action data/datagen/pilot --out action
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np

try:
    from . import dataset as ds
    from . import pose as pz
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import dataset as ds
    from flow import pose as pz

HARD = ("refused", "empty", "cutoff", "parse", "no_body", "miss", "dup",
        "lite")
SOFT = ("no_pose", "unplayed", "planned", "usage_estimated", "miss_fixed")
# Soft flags that still keep a turn out of the SFT exports: a marker the
# parser threw away (an invented [fPose ...]) would be learned; the turn's
# trajectory is fine for the action head.
SFT_HARD = ("unplayed",)
MAX_MISS = 0.34
DUP_JACCARD = 0.85


def norm_line(s) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = "".join(ch if unicodedata.category(ch)[0] in "LN" else " "
                for ch in s)
    return re.sub(r"\s+", " ", s).strip()


def _grams(s):
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def dedupe(records, threshold=DUP_JACCARD) -> set:
    """Ids of turns whose user line nearly repeats an earlier one (by
    session and turn order) of the same avatar x persona."""
    out = set()
    groups = defaultdict(list)
    for r in sorted(records, key=lambda r: (r["session"], r["turn"])):
        groups[(r["avatar"]["name"], r["persona"])].append(r)
    for recs in groups.values():
        exact, kept = set(), []
        index = defaultdict(set)          # trigram -> kept positions
        for r in recs:
            n = norm_line(r.get("user"))
            if not n:
                continue
            if n in exact:
                out.add(r["id"])
                continue
            g = _grams(n)
            cand = Counter()
            for x in g:
                for j in index.get(x, ()):
                    cand[j] += 1
            dup = False
            for j, inter in cand.most_common(20):
                gj = kept[j]
                if inter / (len(g) + len(gj) - inter) >= threshold:
                    dup = True
                    break
            if dup:
                out.add(r["id"])
                continue
            exact.add(n)
            kept.append(g)
            for x in g:
                index[x].add(len(kept) - 1)
    return out


def flag(r, max_miss=MAX_MISS) -> list[str]:
    """The turn's flags (HARD and SOFT names), dedupe aside."""
    f = []
    if r.get("refused"):
        f.append("refused")
    if not (r.get("spoken") or "").strip():
        f.append("empty")
    if r.get("stop_reason") in ("max_tokens", "length"):
        f.append("cutoff")
    unplayed = r.get("unplayed") or {}
    # Planning lines (a gesture giving way to a pose or move, a body that
    # lacks a move) are the pipeline's choices, not the model's syntax:
    # counted, not dropped, or every dance turn would leave the SFT data.
    planned = set(r.get("play_warnings") or ())
    parse = [w for w in r.get("warnings") or ()
             if w not in planned and not (re.match(r"\s*\[(\w+)", w) and "dropped" in w
                     and "not a body-mode marker" in w)]
    if parse:
        f.append("parse")
    marks = [m for m in r.get("markers") or () if m["tag"] != "pause"]
    if not marks and not r.get("refused"):
        f.append("no_body")
    reach = r.get("reach") or {}
    if reach.get("targets") and reach["miss"] / reach["targets"] > max_miss:
        f.append("miss")
    if (r.get("avatar") or {}).get("kind") == "live2d-lite":
        f.append("lite")
    if not any(m["tag"] == "pose" for m in marks):
        f.append("no_pose")
    if unplayed:
        f.append("unplayed")
    if planned:
        f.append("planned")
    llm = r.get("llm") or {}
    if any((llm.get(k) or {}).get("usage_estimated") for k in
           ("character", "user")):
        f.append("usage_estimated")
    return f


def hard_flags(f, keep_lite=False) -> list[str]:
    """The flags that drop a turn: HARD ones, a miss its corrections
    cleared (miss_fixed) and lite when kept aside."""
    return [x for x in f if x in HARD and not (keep_lite and x == "lite")
            and not (x == "miss" and "miss_fixed" in f)]


def annotate(records, max_miss=MAX_MISS, dup=DUP_JACCARD, keep_lite=False,
             fix=None):
    """{id: (flags, kept)} for every record. `fix`: correct_misses' first
    result; a turn flagged miss whose remaining misses (those not
    corrected) are within max_miss gets the soft flag miss_fixed and is
    kept on that count ("miss" stays in its flags)."""
    dups = dedupe(records, dup)
    out = {}
    for r in records:
        f = flag(r, max_miss)
        if "miss" in f and r["id"] in (fix or ()):
            reach = r["reach"]
            left = reach["miss"] - fix[r["id"]]["targets"]
            if left / reach["targets"] <= max_miss:
                f.append("miss_fixed")
        if r["id"] in dups:
            f.append("dup")
        out[r["id"]] = (f, not hard_flags(f, keep_lite))
    return out


# --------------------------------------------------------------- stats

def stats(records, notes, prices=None, counts=None, mcounts=None,
          bcounts=None, bbefore=None) -> dict:
    s = {"turns": len(records), "kept": sum(k for _, k in notes.values()),
         "sessions": len({r["session"] for r in records})}
    c = defaultdict(Counter)
    var_hist = Counter()
    reach = Counter()
    tok = Counter()
    cost = 0.0
    speech, lat = [], []
    for r in records:
        f, kept = notes[r["id"]]
        c["avatar"][r["avatar"]["name"]] += 1
        c["persona"][r["persona"]] += 1
        c["topic"][r.get("topic")] += 1
        c["lang_asked"][(r.get("lang") or {}).get("asked")] += 1
        c["lang_reply"][(r.get("lang") or {}).get("reply")] += 1
        for x in f:
            c["flags"][x] += 1
        for m in r.get("markers") or ():
            c["markers"][m["tag"]] += 1
            if m.get("played") is not True and m["tag"] != "pause":
                c["not_played"][f"{m['tag']}:{m.get('played')}"] += 1
        for t, n in (r.get("unplayed") or {}).items():
            c["unplayed"][t] += n
        for p in r.get("poses") or ():
            if p.get("via"):
                continue
            for v in p.get("targets") or {}:
                var_hist[v] += 1
        rs = r.get("reach") or {}
        for k in ("markers", "targets", "ok", "miss", "exact_ok", "exact_n"):
            reach[k] += int(rs.get(k) or 0)
        for role in ("character", "user"):
            t = ((r.get("llm") or {}).get(role) or {}).get("tokens") or {}
            for k in ("in", "in_hit", "out"):
                tok[f"{role}.{k}"] += int(t.get(k) or 0)
        cost += float(r.get("cost_usd") or 0)
        if r.get("timing"):
            speech.append(r["timing"]["speech_seconds"])
        sec = ((r.get("llm") or {}).get("character") or {}).get("seconds")
        if sec is not None:
            lat.append(sec)
    st = session_turns(records, notes)
    s["sft_sessions"] = len(st)
    s["sft_session_turns"] = sum(len(v) for v in st.values())
    s["by"] = {k: dict(v.most_common()) for k, v in c.items()}
    s["variables"] = dict(var_hist.most_common())
    s["reach"] = dict(reach, rate=round(reach["ok"] / reach["targets"], 4)
                      if reach["targets"] else None)
    s["tokens"] = dict(tok)
    s["cost_usd"] = round(cost, 6)
    s["speech_seconds"] = round(sum(speech), 1)
    if lat:
        s["character_seconds"] = {"median": round(float(np.median(lat)), 2),
                                  "p90": round(float(np.percentile(lat, 90)),
                                               2)}
    if counts is not None:
        s["corrections"] = counts
    if mcounts is not None:
        s["manners"] = mcounts
    if bcounts is not None:
        s["beats"] = dict(bcounts, before=bbefore)
    if s["turns"]:
        s["cost_per_turn_usd"] = round(cost / s["turns"], 7)
    return s


def stats_text(s) -> str:
    out = [f"{s['turns']} turns in {s['sessions']} sessions, {s['kept']} kept "
           f"({s['kept'] / max(1, s['turns']):.0%}); session SFT export: "
           f"{s['sft_sessions']} sessions, {s['sft_session_turns']} turns"]
    for k in ("avatar", "persona", "lang_asked", "lang_reply", "topic",
              "markers", "not_played", "unplayed", "flags"):
        v = s["by"].get(k)
        if v:
            out.append(f"{k}: " + ", ".join(f"{a} {n}" for a, n in v.items()))
    r = s["reach"]
    if r["targets"]:
        out.append(f"reach: {r['ok']}/{r['targets']} targets ({r['rate']:.1%}), "
                   f"{r['miss']} MISS, {r['markers']} pose markers"
                   + (f"; Core {r['exact_ok']}/{r['exact_n']}"
                      if r["exact_n"] else ""))
    c = s.get("corrections")
    if c is not None:
        out.append(correction_line(c))
        for a, v in c["by_avatar"].items():
            out.append(f"  {a:14s} {v['markers']} markers, {v['targets']} "
                       f"targets corrected, {v['skipped']} skipped")
    if s.get("manners") is not None:
        out += manners_lines(s["manners"])
    if s.get("beats") is not None:
        b = dict(s["beats"])
        out += beats_lines(b, b.pop("before", None))
    if s["variables"]:
        top = max(s["variables"].values())
        out.append("targeted variables:")
        for v, n in s["variables"].items():
            out.append(f"  {v:20s} {n:6d} " + "#" * max(1, round(30 * n / top)))
    t = s["tokens"]
    out.append("tokens: character in {0} ({1} cached) out {2}; user in {3} "
               "({4} cached) out {5}".format(
                   t.get("character.in", 0), t.get("character.in_hit", 0),
                   t.get("character.out", 0), t.get("user.in", 0),
                   t.get("user.in_hit", 0), t.get("user.out", 0)))
    out.append(f"cost: ${s['cost_usd']:.4f}"
               + (f" (${s['cost_per_turn_usd'] * 1000:.3f} per 1000 turns)"
                  if s.get("cost_per_turn_usd") else "")
               + f"; {s['speech_seconds']} s of speech")
    if s.get("character_seconds"):
        out.append("character latency: median "
                   f"{s['character_seconds']['median']} s, p90 "
                   f"{s['character_seconds']['p90']} s")
    return "\n".join(out)


# ----------------------------------------------------------------- sft

def load_registry(root) -> dict:
    with open(os.path.join(root, "avatars.json"), encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------ miss corrections

_VALUE = (r'"[^"]*"|\'[^\']*\'|@[\w.]+(?:[+-][\d.]+)?|'
          r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def pose_spans(text):
    """[(start, end, at, targets)] for the pose markers of a raw reply:
    text[start:end] is the marker with its brackets, `at` the spoken offset
    and `targets` what the parser took from it (any variable, any landmark:
    no body at hand). The pose parser is fed one character at a time and
    its bracket state watched, so a marker is found where the parser found
    it (a '[' that never closed, a marker restarted by a new '[')."""
    p = pz.PoseParser()
    out, start = [], 0
    for i, ch in enumerate(text):
        c = ch.translate(pz.dr._FULLWIDTH)
        was = p._buf is not None
        restart = was and c == "[" and p._depth == 0 and not p._instr
        done = p.feed(ch)
        if (not was or restart) and p._buf is not None:
            start = i
        for t in done:
            if t["tag"] == pz.POSE:
                out.append((start, i + 1, t["at"], t["arg"]["targets"]))
    return out


def _same_targets(a, b) -> bool:
    """Every target in `a` is in `b` with the same value (numbers to a
    thousandth: the records are rounded, the parser's floats are not)."""
    for k, v in a.items():
        w = b.get(k)
        if isinstance(v, str) or isinstance(w, str) or w is None:
            if v != w:
                return False
        elif abs(float(v) - float(w)) > 1e-3:
            return False
    return True


def _got_text(got, unit) -> str:
    """The reached value as a model writes it: whole numbers for degrees
    and percents, up to 2 decimals for hu and unit variables."""
    s = f"{got:.0f}" if unit in ("deg", "pct") else f"{got:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _rewrite(marker, values):
    """The marker text with {key: new number} put in for those keys' values;
    None when a key is not found exactly once."""
    for key, new in values.items():
        pat = re.compile(r'([{,]\s*)(["\']?)' + re.escape(key)
                         + r'\2(\s*:\s*)(' + _VALUE + ")")
        hits = list(pat.finditer(marker))
        if len(hits) != 1:
            return None
        m = hits[0]
        marker = marker[:m.start(4)] + new + marker[m.end(4):]
    return marker


def correct_reply(r):
    """(text, markers, targets, skipped) for one record: its reply_raw with
    each missed target of each pose marker set to the value reached
    (module docstring). skipped: [] or reasons, "no_fields" / "mismatch"
    (the turn, left as it was) and "unmappable" (one marker)."""
    text = r.get("reply_raw") or ""
    marks = [m for m in r.get("markers") or () if m.get("tag") == pz.POSE]
    poses = [p for p in r.get("poses") or () if not p.get("via")]
    if any(m.get("at") is None or not isinstance(m.get("arg"), dict)
           for m in marks) or any("report" not in p or p.get("at") is None
                                  for p in poses):
        return text, 0, 0, ["no_fields"]
    spans = pose_spans(text)
    if len(spans) != len(marks) or any(
            sp[2] != m["at"] or not _same_targets(
                m["arg"].get("targets") or {}, sp[3])
            for sp, m in zip(spans, marks)):
        return text, 0, 0, ["mismatch"]
    # Marker -> pose record: same place and targets, each record once. A
    # marker with no record (the solver dropped it) has nothing to correct;
    # a record with no marker means the two lists disagree.
    pool, tied = list(poses), []
    for sp, m in zip(spans, marks):
        k = next((j for j, p in enumerate(pool) if p["at"] == m["at"]
                  and _same_targets(p.get("targets") or {},
                                    m["arg"].get("targets") or {})), None)
        tied.append(pool.pop(k) if k is not None else None)
    if pool:
        return text, 0, 0, ["mismatch"]
    edits, nm, nt, skipped = [], 0, 0, []
    for sp, p in zip(spans, tied):
        if p is None:
            continue
        miss = {v: e for v, e in p["report"].items() if not e.get("ok")}
        if not miss:
            continue
        vals = {}
        for v, e in miss.items():
            if e.get("got") is None or v not in (p.get("targets") or {}):
                break
            vals[v] = _got_text(float(e["got"]), e.get("unit"))
        else:
            new = _rewrite(text[sp[0]:sp[1]], vals)
            if new is not None:
                edits.append((sp[0], sp[1], new))
                nm += 1
                nt += len(vals)
                continue
        skipped.append("unmappable")
    for a, b, new in reversed(edits):
        text = text[:a] + new + text[b:]
    return text, nm, nt, skipped


def correct_misses(records):
    """({id: {"reply", "markers", "targets"}}, counts): the corrected
    replies of the turns that had a corrected marker, and the counters
    (markers, targets, turns corrected, miss_turns: turns with a MISS,
    skipped {reason: n}, by_avatar {name: {markers, targets, skipped}})."""
    fix = {}
    c = {"markers": 0, "targets": 0, "turns": 0, "miss_turns": 0,
         "skipped": Counter(), "by_avatar": {}}
    for r in records:
        if r.get("refused") or not (r.get("reach") or {}).get("miss"):
            continue
        a = c["by_avatar"].setdefault(
            r["avatar"]["name"], {"markers": 0, "targets": 0, "skipped": 0})
        c["miss_turns"] += 1
        text, nm, nt, skipped = correct_reply(r)
        for why in skipped:
            c["skipped"][why] += 1
            a["skipped"] += 1
        if nm:
            fix[r["id"]] = {"reply": text, "markers": nm, "targets": nt}
            c["turns"] += 1
            c["markers"] += nm
            c["targets"] += nt
            a["markers"] += nm
            a["targets"] += nt
    c["skipped"] = dict(c["skipped"])
    return fix, c


def correction_line(c) -> str:
    return (f"miss corrections: {c['markers']} markers, {c['targets']} "
            f"targets in {c['turns']} of {c['miss_turns']} turns with a miss"
            + (", skipped " + ", ".join(f"{k} {n}" for k, n in
                                        c["skipped"].items())
               if c["skipped"] else ""))


# ------------------------------------------------------ streamer manners

# Whether the user asked for a body action. A turn is "asked" when its topic
# is one of ASK_TOPICS (flow/datagen.personas.json: show = point, wave, raise
# a hand; pose = a pose, a bow, a dance; invite = come along) or its user line
# matches a pattern of ASK_WORDS. Heuristic and meant to be extended: the
# patterns are regular expressions (case-insensitive), all three languages are
# tried on every line (users mix them), and a line is first stripped of the
# ASK_NOT phrases, words that contain a trigger without asking for a movement
# (ja 相手 "partner" has 手; zh 举例 "for example" has 举; en "on the other
# hand"). "look" and "turn" count only where they are an instruction ("look
# at", "turn around"): "you look cute" asks for nothing. The head and the eyes
# are kept on every turn, asked or not, so a request to look somewhere is no
# reason to keep a hand.
ASK_TOPICS = ("pose", "show", "invite")
ASK_WORDS = {
    "en": [r"\bwav(?:e|es|ed|ing)\b", r"\brais(?:e|es|ed|ing)\b",
           r"\bhands?\b", r"\barms?\b", r"\bfingers?\b",
           r"\bpoint(?:s|ed|ing)? (?:to|at|out|over|where|it)\b",
           r"\bpointing\b", r"\bclap", r"\bdanc(?:e|es|ed|ing)\b",
           r"\bcheer(?:s|ing)?\b(?! (?:me )?up)", r"\bhug", r"\bcross(?:ed)?\b",
           r"\bstretch", r"\bbow(?:s|ed|ing)?\b", r"\bnod(?:s|ded|ding)?\b",
           r"\blook (?:at|over|there|here|left|right|up|down|around|behind"
           r"|this way|that way|back)\b",
           r"\bturn (?:around|left|right|your|back|over|away)\b",
           r"\bshow (?:me|us|how)\b", r"\bstrike\b", r"\bpose\b",
           r"\bspin", r"\bjump", r"\bpeace sign\b", r"\bthumbs?\b",
           r"\bhigh[- ]?five\b", r"\bshrug", r"\bsalut", r"\bflex",
           r"\btouch (?:your|the)\b", r"\bhold up\b", r"\bmimic me\b"],
    "ja": ["手", "振っ", "振る", "振り向", "上げ", "指", "踊", "拍手", "腕",
           "ポーズ", "ハグ", "抱", "見せ", "ピース", "万歳", "ばんざい",
           "お辞儀", "おじぎ", "伸び", "ストレッチ", "ジャンプ", "うなず",
           "頷", "敬礼", "回って", "くるっと", "ぐるっと", "こっち見", "こっちを見",
           "あっち", "あそこ見", "向いて", "ふりふり"],
    "zh": ["手", "挥", "举", "指", "跳舞", "跳个", "跳一", "跳起", "拍手", "拍照",
           "拍拍", "拍一", "抱", "鞠躬", "点头", "伸懒腰", "伸个懒腰", "敬礼",
           "耸肩", "叉腰", "比心", "比耶", "姿势", "姿勢", "摆个", "招手",
           "转身", "转圈", "转一圈", "转个", "转过", "转头", "转向", "转一下",
           "看我", "看这", "看那", "看镜头", "看左", "看右", "看上", "看下",
           "看向", "看一下", "看看"],
}
ASK_NOT = [
    r"on the other hand", r"second[- ]hand", r"hand(?:s)? down", r"by hand",
    r"upper hand", r"out of hand", r"\bin hand\b", r"hands[- ]on",
    r"相手", r"上手", r"下手", r"助手", r"苦手", r"歌手", r"選手", r"手伝",
    r"手紙", r"手作り", r"手間", r"手術", r"手に入", r"手前", r"手軽", r"得手",
    r"運転手", r"入手", r"投手", r"打手", r"騎手", r"手帳", r"手段", r"手続",
    r"申し上げ", r"取り上げ", r"盛り上げ", r"仕上げ", r"買い上げ", r"染め上げ",
    r"目指", r"指導", r"指示", r"指摘", r"指定", r"指名", r"指針", r"指令",
    r"指揮", r"指標", r"腕前", r"腕時計", r"抱負", r"振り返", r"抱え込",
    r"手机", r"歌手", r"对手", r"手段", r"助手", r"高手", r"手艺", r"手续",
    r"手术", r"手表", r"手套", r"选手", r"新手", r"手头", r"举例", r"举办",
    r"举行", r"举报", r"举止", r"指导", r"指示", r"指出", r"指定", r"指望",
    r"指标", r"指南", r"指责", r"指挥", r"抱歉", r"抱怨", r"抱负",
    r"看看你", r"看看这个", r"看看吧"]
_ASK = [re.compile(p, re.I) for ps in ASK_WORDS.values() for p in ps]
_NOT = [re.compile(p, re.I) for p in ASK_NOT]

# What a streamer keeps on a turn nobody asked for a gesture on: the head, the
# eyes, the face, and a small torso lean. Everything else in a pose marker
# (hands, arms, shoulders, a bigger torso move, hips, composites such as
# "hands": "@together") is a body gesture and goes. TORSO_SMALL: the largest
# |value| (degrees) of torso.yaw / pitch / tilt kept; torso.forward and an
# "@anchor" torso value go.
KEEP_PREFIX = ("head.", "gaze.", "brow.", "eye.", "mouth.", "cheek.")
TORSO_SMALL = 6.0
# Moves whose figure is the head alone (flow.moves PRIMS kind "head"; the
# aliases nod, nod_along, bop, look resolve to them). The others sway, bounce,
# shimmy, clap, cheer, wave, twirl and dance move the trunk or the arms. A
# [gesture:NAME] plays the rig's own motion, which on the Cubism samples and
# the Azur Lane rigs is the arms and legs (flow.gestures describes the arms
# first); records do not say which, so an unasked gesture is always dropped.
HEAD_MOVES = ("head_bob", "look_around")
MAX_POSES = 2
# Expression beats (see the module docstring): the pose markers of the
# "beats" prompt are the face, with the head along. gaze counts as face (a
# glance is a beat); BEATS_MAX: markers kept per reply when the dataset was
# written with that prompt (3 sentences x 3 beats).
FACE_PREFIX = ("brow.", "eye.", "mouth.", "cheek.", "gaze.")
BEATS_MAX = 9
BEATS_MARK = "expression beats"      # in the prompt text of that version
_SPEC = ("dur", "duration", "hold")
_ARM_KEYS = ("hand", "hands", "arm", "forearm", "elbow", "wrist")


def asked_line(line) -> bool:
    """True when a user line matches ASK_WORDS (after ASK_NOT is cut out)."""
    s = unicodedata.normalize("NFKC", line or "")
    for p in _NOT:
        s = p.sub(" ", s)
    return any(p.search(s) for p in _ASK)


def asked_for(r):
    """(asked, by) for a record: by is "topic", "keyword", "topic+keyword"
    or None."""
    by = []
    if r.get("topic") in ASK_TOPICS:
        by.append("topic")
    if asked_line(r.get("user")):
        by.append("keyword")
    return bool(by), "+".join(by) or None


def keep_key(key, value) -> bool:
    """True when a pose target survives on a turn with no gesture asked."""
    if key.startswith(KEEP_PREFIX):
        return True
    if key in ("torso.yaw", "torso.pitch", "torso.tilt"):
        try:
            return abs(float(value)) <= TORSO_SMALL
        except (TypeError, ValueError):
            return False
    return False


def is_hand(key) -> bool:
    return key.split(".")[0] in _ARM_KEYS


def is_face(key) -> bool:
    return key.startswith(FACE_PREFIX)


def is_beat(keys) -> bool:
    """True for a pose marker with these target keys that is an expression
    beat: some face variable, the rest head angles (nothing for the body)."""
    keys = list(keys)
    return any(is_face(k) for k in keys) and all(
        is_face(k) or k.startswith("head.") for k in keys)


class _AnyGesture(set):
    """Every [gesture:NAME] is a marker here: the labels are the rig's."""

    def __contains__(self, x):
        return True

    def __bool__(self):
        return True


def _parser():
    try:
        from . import moves as mv
    except ImportError:
        from flow import moves as mv
    p = pz.PoseParser(moves=[*mv.PRIMS, "dance"])
    p.gestures = _AnyGesture()
    return p


def marker_spans(text):
    """[(start, end, tag)] for every marker the body-mode parser takes from
    a raw reply (pose, move, gesture, blink, pause): text[start:end] with its
    brackets, tag the parser's dict. Found as pose_spans finds them."""
    p = _parser()
    out, start = [], 0
    for i, ch in enumerate(text):
        c = ch.translate(pz.dr._FULLWIDTH)
        was = p._buf is not None
        restart = was and c == "[" and p._depth == 0 and not p._instr
        done = p.feed(ch)
        if (not was or restart) and p._buf is not None:
            start = i
        for t in done:
            out.append((start, i + 1, t))
    return out


def spoken_of(text) -> str:
    p = _parser()
    p.feed(text)
    p.close()
    return p.spoken


def _pose_dict(text, a, b, targets):
    """The marker's JSON as an ordered {stripped key: value} (the values as
    written), or the parser's targets when it cannot be read again."""
    seg = text[a:b].translate(pz.dr._FULLWIDTH)[1:-1].strip()
    m = re.match(r"(?is)pose\b\s*:?\s*(.*)$", seg)
    try:
        d, _ = pz.loads_lenient(m.group(1) if m else seg)
    except ValueError:
        return dict(targets)
    return {str(k).strip(): v for k, v in d.items()}


def _cut(s, a, b, repl=""):
    """s with s[a:b] replaced; a removal takes the spaces after it too when
    spaces (or the start) are before, so no double space is left."""
    if not repl and (a == 0 or s[a - 1] in " \t\n") and b < len(s) \
            and s[b] in " \t":
        while b < len(s) and s[b] in " \t":
            b += 1
    return s[:a] + repl + s[b:]


def manners_reply(text, asked, max_poses=MAX_POSES, max_beats=0):
    """The reply with a streamer's manners (module docstring): dict with
    "reply" (the text; as it was when nothing changed), "skipped" (None, or
    why the text was left alone: "spoken" / "reparse") and counters: poses
    and targets and hand targets before / after, markers_emptied (pose
    markers left with no target), markers_capped, targets_removed, moves /
    gestures removed, body_before / body_after (a pose, move or gesture left
    in the reply). max_beats > 0: the expression beats (is_beat) are capped
    on their own, max_poses counts the other pose markers."""
    spans = marker_spans(text)
    poses, other = [], []
    for a, b, t in spans:
        if t["tag"] == pz.POSE:
            tg = t["arg"]["targets"]
            d = _pose_dict(text, a, b, tg)
            keys = [k for k in tg]
            if asked:
                kept = keys
            else:
                kept = [k for k in keys if keep_key(k, tg[k])]
            poses.append({"a": a, "b": b, "d": d, "keys": keys, "kept": kept,
                          "body": any(not keep_key(k, tg[k]) for k in keys),
                          "beat": bool(max_beats) and is_beat(kept)})
        else:
            other.append((a, b, t))
    c = Counter(poses_before=len(poses),
                targets_before=sum(len(p["keys"]) for p in poses),
                hand_before=sum(is_hand(k) for p in poses for k in p["keys"]),
                body_before=int(bool(poses or any(
                    t["tag"] in ("move", "gesture") for _, _, t in other))))
    gone = []                               # (a, b) markers to cut out
    live = []
    for p in poses:
        if p["kept"]:
            live.append(p)
        else:
            gone.append(p)
            c["markers_emptied"] += 1
    drop = set()
    for beat, cap in ((False, max_poses), (True, max_beats)):
        idx = [i for i, p in enumerate(live) if p["beat"] == beat]
        if len(idx) > cap:
            # The markers asking for the body come first, then the early
            # ones.
            rank = sorted(idx, key=lambda i: (
                not (asked and live[i]["body"]), i))
            drop |= set(rank[cap:])
    if drop:
        for i in sorted(drop):
            gone.append(live[i])
            c["markers_capped"] += 1
        live = [p for i, p in enumerate(live) if i not in drop]
    edits = [(p["a"], p["b"], "") for p in gone]
    for p in live:
        if len(p["kept"]) != len(p["keys"]):
            d = {k: v for k, v in p["d"].items()
                 if k in p["kept"] or k in _SPEC}
            edits.append((p["a"], p["b"], "[pose "
                          + json.dumps(d, ensure_ascii=False) + "]"))
    if not asked:
        for a, b, t in other:
            name = t["tag"]
            if name == "gesture" or (name == "move" and t["arg"].get("name")
                                     not in HEAD_MOVES):
                edits.append((a, b, ""))
                c["moves_removed" if name == "move" else
                  "gestures_removed"] += 1
    c["poses_after"] = len(live)
    c["targets_after"] = sum(len(p["kept"]) for p in live)
    c["hand_after"] = sum(is_hand(k) for p in live for k in p["kept"])
    c["targets_removed"] = c["targets_before"] - c["targets_after"]
    kept_moves = sum(1 for _, _, t in other if t["tag"] in ("move", "gesture")
                     ) - c["moves_removed"] - c["gestures_removed"]
    c["body_after"] = int(bool(live or kept_moves))
    res = {"reply": text, "skipped": None, "c": c}
    if not edits:
        return res
    new = text
    for a, b, repl in sorted(edits, reverse=True):
        new = _cut(new, a, b, repl)
    want = spoken_of(text)
    if spoken_of(new) != want:
        # Spaces cut a different way, or not at all.
        new = text
        for a, b, repl in sorted(edits, reverse=True):
            new = new[:a] + repl + new[b:]
        if spoken_of(new) != want:
            res["skipped"] = "spoken"
            return res
    after = [t for _, _, t in marker_spans(new)]
    if sum(t["tag"] == pz.POSE for t in after) != len(live):
        res["skipped"] = "reparse"
        return res
    res["reply"] = new
    return res


_MCOUNT = ("poses_before", "poses_after", "targets_before", "targets_after",
           "targets_removed", "hand_before", "hand_after", "markers_emptied",
           "markers_capped", "moves_removed", "gestures_removed",
           "body_before", "body_after")


def manners(records, fix=None, max_poses=MAX_POSES, count=None,
            max_beats=0):
    """({id: {"reply"}}, counts): `--manners` for the records' replies
    (corrected ones when `fix`, correct_misses' first result, has them), only
    the turns whose reply changed in the first. counts: per avatar
    {turns, asked, asked_topic, asked_keyword, unasked, changed, skipped, and
    the _MCOUNT counters} summed over the turns in `count` (a set of ids;
    None: all), "by" {id: how the turn was judged asked, or None}, "skipped"
    {reason: n}. Refused turns are left out. max_beats: manners_reply's."""
    out = {}
    c = {"by_avatar": {}, "by": {}, "skipped": Counter(), "max_poses":
         max_poses, "max_beats": max_beats}
    for r in records:
        if r.get("refused"):
            continue
        text = (fix or {}).get(r["id"], {}).get("reply",
                                                r.get("reply_raw") or "")
        asked, by = asked_for(r)
        res = manners_reply(text, asked, max_poses, max_beats)
        if res["reply"] != text:
            out[r["id"]] = {"reply": res["reply"]}
        if count is not None and r["id"] not in count:
            continue
        a = c["by_avatar"].setdefault(r["avatar"]["name"], Counter())
        a["turns"] += 1
        a["asked" if asked else "unasked"] += 1
        a["asked_topic"] += bool(by and "topic" in by)
        a["asked_keyword"] += bool(by and "keyword" in by)
        c["by"][r["id"]] = by
        if res["skipped"]:
            a["skipped"] += 1
            c["skipped"][res["skipped"]] += 1
            continue
        a["changed"] += res["reply"] != text
        for k in _MCOUNT:
            a[k] += res["c"].get(k, 0)
    c["by_avatar"] = {k: dict(v) for k, v in sorted(c["by_avatar"].items())}
    c["skipped"] = dict(c["skipped"])
    tot = Counter()
    for v in c["by_avatar"].values():
        tot.update(v)
    c["total"] = dict(tot)
    return out, c


def manners_lines(c) -> list:
    """The stats text for manners' counts."""
    def row(v):
        n = max(1, v.get("turns", 0))
        hb = v.get("hand_before", 0) / max(1, v.get("targets_before", 0))
        ha = v.get("hand_after", 0) / max(1, v.get("targets_after", 0))
        return (f"{v.get('asked', 0)} asked ({v.get('asked_topic', 0)} by "
                f"topic, {v.get('asked_keyword', 0)} by words) / "
                f"{v.get('unasked', 0)} not; poses per reply "
                f"{v.get('poses_before', 0) / n:.2f} -> "
                f"{v.get('poses_after', 0) / n:.2f}; hand share of targets "
                f"{hb:.0%} -> {ha:.0%}; removed {v.get('targets_removed', 0)}"
                f" targets, {v.get('markers_emptied', 0)} emptied + "
                f"{v.get('markers_capped', 0)} capped markers, "
                f"{v.get('moves_removed', 0)} moves, "
                f"{v.get('gestures_removed', 0)} gestures; replies with no "
                f"body marker {n - v.get('body_before', 0)} -> "
                f"{n - v.get('body_after', 0)} of {n}"
                + (f"; {v['skipped']} left as they were"
                   if v.get("skipped") else ""))
    out = [f"manners (at most {c['max_poses']} poses"
           + (f", {c['max_beats']} beats" if c.get("max_beats") else "")
           + "): " + row(c["total"])]
    for a, v in c["by_avatar"].items():
        out.append(f"  {a:14s} " + row(v))
    return out


_SENTENCE = re.compile(r"[^.!?\u3002\uff01\uff1f\u2026\n]*[^\s.!?\u3002"
                       r"\uff01\uff1f\u2026][^.!?\u3002\uff01\uff1f\u2026\n]*")
_BCOUNT = ("turns", "sentences", "poses", "beats", "beats_head", "targets",
           "face_targets", "hand_targets", "turns_with_beats")


def sentence_count(spoken) -> int:
    """Sentences in a spoken text: stretches with a letter between the
    sentence marks (. ! ? and their CJK forms, the ellipsis, a newline)."""
    return len(_SENTENCE.findall(spoken or ""))


def beat_counts(text) -> Counter:
    """Counters over one reply (_BCOUNT): sentences, pose markers, beats
    (is_beat) and those with a head angle, pose targets, the face and hand
    ones among them. A composite ("hands") target is a hand; a pose marker
    that cannot be read counts as a pose with its parsed targets."""
    c = Counter()
    c["sentences"] = sentence_count(spoken_of(text))
    for _a, _b, t in marker_spans(text):
        if t["tag"] != pz.POSE:
            continue
        keys = list(t["arg"]["targets"])
        c["poses"] += 1
        c["targets"] += len(keys)
        c["face_targets"] += sum(is_face(k) for k in keys)
        c["hand_targets"] += sum(is_hand(k) for k in keys)
        if is_beat(keys):
            c["beats"] += 1
            c["beats_head"] += any(k.startswith("head.") for k in keys)
    c["turns_with_beats"] = int(c["beats"] > 0)
    return c


def beats(records, texts=None, count=None) -> dict:
    """The expression-beat counts of the records' replies (the corrected /
    manners ones in `texts` {id: {"reply"}} when given), per avatar and
    "total", over the turns in `count` (a set of ids; None: all). Refused
    turns are left out."""
    by = {}
    for r in records:
        if r.get("refused") or (count is not None and r["id"] not in count):
            continue
        text = (texts or {}).get(r["id"], {}).get("reply",
                                                  r.get("reply_raw") or "")
        a = by.setdefault(r["avatar"]["name"], Counter())
        a["turns"] += 1
        a.update(beat_counts(text))
    by = {k: dict(v) for k, v in sorted(by.items())}
    tot = Counter()
    for v in by.values():
        tot.update(v)
    return {"by_avatar": by, "total": dict(tot)}


def beats_lines(c, before=None) -> list:
    """The stats text for beats' counts; `before` (the same, from the texts
    before manners) puts "before -> after" on the rates."""
    def rate(v, num, den):
        return v.get(num, 0) / max(1, v.get(den, 0))

    def pair(v, w, num, den, pct=False):
        f = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:.2f}")
        a = f(rate(v, num, den))
        return a if w is None else f"{f(rate(w, num, den))} -> {a}"

    def row(v, w):
        n = max(1, v.get("turns", 0))
        return (f"{v.get('beats', 0)} beats in {v.get('sentences', 0)} "
                f"sentences, per sentence {pair(v, w, 'beats', 'sentences')}"
                f"; face share of targets "
                f"{pair(v, w, 'face_targets', 'targets', True)}, hand share "
                f"{pair(v, w, 'hand_targets', 'targets', True)}; beats with "
                f"a head move {pair(v, None, 'beats_head', 'beats', True)}; "
                f"pose markers per reply "
                f"{pair(v, w, 'poses', 'turns')}; replies with a beat "
                f"{v.get('turns_with_beats', 0) / n:.0%} of {n}")
    pre = (before or {}).get("by_avatar", {})
    out = ["expression beats: " + row(c["total"],
                                     (before or {}).get("total"))]
    for a, v in c["by_avatar"].items():
        out.append(f"  {a:14s} " + row(v, pre.get(a)))
    return out


def default_max_beats(root) -> int:
    """--max-beats when not given: BEATS_MAX for a dataset whose prompts
    (avatars.json) carry the beats teacher prompt, else 0."""
    try:
        prompts = load_registry(root).get("prompts") or {}
    except (OSError, ValueError):
        return 0
    return BEATS_MAX if any(BEATS_MARK in (t or "")
                            for t in prompts.values()) else 0


def parse_weights(specs, avatars=None) -> dict:
    """{avatar: n} from --avatar-weight values ("lafei=2" or
    "lafei=2,Hiyori=3"); an avatar the data lacks (a typo) or a weight
    below 1 is an error."""
    out = {}
    for spec in specs or ():
        for part in str(spec).split(","):
            if not part.strip():
                continue
            name, _, n = part.partition("=")
            name = name.strip()
            try:
                n = int(n)
            except ValueError:
                raise SystemExit(f"--avatar-weight {part!r}: want NAME=N")
            if n < 1:
                raise SystemExit(f"--avatar-weight {part!r}: N is at least 1")
            if avatars is not None and name not in avatars:
                raise SystemExit(f"--avatar-weight: no avatar {name!r} in "
                                 f"the data (have {', '.join(sorted(avatars))})")
            out[name] = n
    return out


def _turn_msgs(r, fix=None):
    reply = (fix or {}).get(r["id"], {}).get("reply", r["reply_raw"])
    return [{"role": "user", "content": r["user_content"]},
            {"role": "assistant", "content": reply}]


def session_ok(note) -> bool:
    """A turn a session-mode example keeps: kept, or dropped only as a
    duplicate user line (its reply is new; a repeated greeting at turn 1
    must not cost the session's other turns)."""
    flags, kept = note
    if any(x in flags for x in SFT_HARD):
        return False
    return kept or hard_flags(flags) == ["dup"]


def session_turns(records, notes) -> dict:
    """{session: [records]} as --per session exports them: in turn order,
    refused turns skipped (never history), cut before the first turn
    session_ok rejects; sessions left empty are absent."""
    by_s = defaultdict(list)
    for r in records:
        by_s[r["session"]].append(r)
    out = {}
    for sid, recs in by_s.items():
        keep = []
        for r in sorted(recs, key=lambda r: r["turn"]):
            if r.get("refused"):
                continue
            if not session_ok(notes[r["id"]]):
                break
            keep.append(r)
        if keep:
            out[sid] = keep
    return out


def _repeat(exs, n):
    return [copy.deepcopy(x) for x in exs for _ in range(n)]


def sft_examples(root, records, notes, per="session", meta=False, fix=None,
                 weights=None):
    """The SFT examples (dicts with "messages"). `fix`: correct_misses'
    first result, with manners' over it when both are on (the replies they
    rewrote stand in for reply_raw, history included); `weights` {avatar: n}: that avatar's examples n times each,
    one after the other."""
    reg = load_registry(root)
    prompts = reg["prompts"]
    by_s = defaultdict(list)
    for r in records:
        by_s[r["session"]].append(r)
    sessions = session_turns(records, notes) if per == "session" else {}
    out = []
    for sid in sorted(by_s):
        recs = sorted(by_s[sid], key=lambda r: r["turn"])
        system = prompts.get(recs[0]["avatar"]["prompt_sha"])
        if system is None:
            continue
        base = [{"role": "system", "content": system}]
        times = (weights or {}).get(recs[0]["avatar"]["name"], 1)
        start = len(out)
        # What the history shows of a corrected turn is its corrected reply.
        fixed = {r["reply_raw"]: fix[r["id"]]["reply"] for r in recs
                 if r["id"] in (fix or ())}
        if per == "session":
            msgs = list(base)
            ids = []
            for r in sessions.get(sid, ()):
                msgs += _turn_msgs(r, fix)
                ids.append(r["id"])
            if ids:
                ex = {"messages": msgs}
                if meta:
                    ex["meta"] = {"session": sid, "turns": ids,
                                  "avatar": recs[0]["avatar"]["name"],
                                  "persona": recs[0]["persona"]}
                out.append(ex)
            out[start:] = _repeat(out[start:], times)
            continue
        for r in recs:
            f, kept = notes[r["id"]]
            if not kept or any(x in f for x in SFT_HARD):
                continue
            msgs = list(base)
            hist = r.get("history") or ()
            # The history is the earlier turns of the session that were not
            # refused: tied to their records by order when the replies
            # agree (two turns may repeat a reply and differ in the fix),
            # else by the reply text.
            prev = [x for x in recs if x["turn"] < r["turn"]
                    and not x.get("refused")]
            tied = len(prev) == len(hist) and all(
                x["reply_raw"] == h["reply"] for x, h in zip(prev, hist))
            for i, h in enumerate(hist):
                reply = (fix[prev[i]["id"]]["reply"]
                         if tied and prev[i]["id"] in (fix or ())
                         else fixed.get(h["reply"], h["reply"]))
                msgs.append({"role": "user", "content": h["user_content"]})
                msgs.append({"role": "assistant", "content": reply,
                             "weight": 0})
            msgs += _turn_msgs(r, fix)
            ex = {"messages": msgs}
            if meta:
                ex["meta"] = {"id": r["id"], "avatar": r["avatar"]["name"],
                              "persona": r["persona"], "reach": r["reach"]}
            out.append(ex)
        out[start:] = _repeat(out[start:], times)
    return out


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    return len(rows)


# -------------------------------------------------------------- action

def action_pairs(root, records, notes, out, which="directed"):
    """Write out.jsonl and out.npz; returns the number of pairs."""
    suffix = ".directed" if which == "directed" else ""
    rows, arrays = [], {}
    for r in sorted(records, key=lambda r: r["id"]):
        if not notes[r["id"]][1] or not r.get("traj"):
            continue
        tr = ds.load_traj(root, r, suffix)
        if tr is None:
            continue
        key = r["id"].replace("/", "__")
        arrays[key + ".traj"] = tr.astype(np.float16)
        ch = ds.load_traj(root, r, ".chars")
        if ch is not None:
            arrays[key + ".chars"] = ch.astype(np.float16)
        vv = ds.load_traj(root, r, ".variant")
        if vv is not None:
            arrays[key + ".variant"] = vv
        rows.append({"id": r["id"], "key": key, "avatar": r["avatar"]["name"],
                     "kind": r["avatar"]["kind"],
                     "body_sha": r["avatar"]["body_sha"],
                     "vars": r["traj"]["vars"], "fps": r["timing"]["fps"],
                     "frames": int(tr.shape[0]), "traj": which,
                     "user": r["user"], "spoken": r["spoken"],
                     "reply_raw": r["reply_raw"],
                     "markers": [{k: m.get(k) for k in
                                  ("tag", "arg", "at", "t", "played")}
                                 for m in r["markers"]],
                     "speech_seconds": r["timing"]["speech_seconds"],
                     "lang": r["lang"]})
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    write_jsonl(out + ".jsonl", rows)
    np.savez_compressed(out + ".npz", **arrays)
    return len(rows)


# ----------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stats", "sft", "action"):
        p = sub.add_parser(name)
        p.add_argument("root", help="flow.dataset's --out")
        p.add_argument("--max-miss", type=float, default=MAX_MISS)
        p.add_argument("--dup-jaccard", type=float, default=DUP_JACCARD)
        p.add_argument("--keep-lite", action="store_true")
        if name != "action":
            p.add_argument("--correct-misses", default=True,
                           action=argparse.BooleanOptionalAction,
                           help="write each missed pose target as the "
                                "value reached [on]")
            p.add_argument("--manners", default=True,
                           action=argparse.BooleanOptionalAction,
                           help="streamer manners: on a turn that asked for "
                                "no body action, no hand / arm / shoulder "
                                "targets, arm moves or gestures, and at most "
                                "--max-poses pose markers per reply [on]")
            p.add_argument("--max-poses", type=int, default=MAX_POSES,
                           help=f"manners: pose markers kept per reply "
                                f"[{MAX_POSES}]")
            p.add_argument("--max-beats", type=int, default=None,
                           metavar="N",
                           help="manners: expression beats (face markers "
                                "with optional head angles) kept per reply, "
                                "on top of --max-poses; 0: they count as "
                                f"poses [{BEATS_MAX} for a dataset written "
                                "with the beats teacher prompt, else 0]")
        if name == "stats":
            p.add_argument("--json", default=None)
        if name == "sft":
            p.add_argument("--out", required=True)
            p.add_argument("--per", default="session",
                           choices=("session", "turn"))
            p.add_argument("--meta", action="store_true")
            p.add_argument("--avatar-weight", action="append", default=[],
                           metavar="NAME=N",
                           help="write that avatar's examples N times "
                                "(repeatable, or NAME=N,NAME=N)")
        if name == "action":
            p.add_argument("--out", required=True,
                           help="path stem: <out>.jsonl and <out>.npz")
            p.add_argument("--traj", default="directed",
                           choices=("directed", "played"))
    a = ap.parse_args(argv)
    recs = ds.iter_records(a.root)
    if not recs:
        print(f"no records under {a.root}/shards", file=sys.stderr)
        return 1
    fix, counts = (correct_misses(recs) if a.cmd != "action"
                   and a.correct_misses else ({}, None))
    notes = annotate(recs, a.max_miss, a.dup_jaccard, a.keep_lite, fix)
    mfix, mcounts = ({}, None)
    kept = {i for i, (_, k) in notes.items() if k}
    if a.cmd != "action" and a.manners:
        mb = (default_max_beats(a.root) if a.max_beats is None
              else a.max_beats)
        mfix, mcounts = manners(recs, fix, a.max_poses, kept, mb)
    if a.cmd == "stats":
        bafter = beats(recs, {**fix, **mfix}, kept)
        s = stats(recs, notes, counts=counts, mcounts=mcounts,
                  bcounts=bafter,
                  bbefore=beats(recs, fix, kept) if mcounts else None)
        print(stats_text(s))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump(s, fh, ensure_ascii=False, indent=1)
        return 0
    if a.cmd == "sft":
        w = parse_weights(a.avatar_weight,
                          {r["avatar"]["name"] for r in recs})
        n = write_jsonl(a.out, sft_examples(a.root, recs, notes, a.per,
                                            a.meta, {**fix, **mfix}, w))
        print(f"{n} examples -> {a.out}")
        if counts:
            print(correction_line(counts))
        if mcounts:
            print("\n".join(manners_lines(mcounts)))
        print("\n".join(beats_lines(
            beats(recs, {**fix, **mfix}, kept),
            beats(recs, fix, kept) if mcounts else None)))
        return 0
    n = action_pairs(a.root, recs, notes, a.out, a.traj)
    print(f"{n} pairs -> {a.out}.jsonl, {a.out}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
