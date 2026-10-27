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
        --avatar-weight lafei=2 --no-correct-misses
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

def stats(records, notes, prices=None, counts=None) -> dict:
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
    first result (the replies it rewrote stand in for reply_raw, history
    included); `weights` {avatar: n}: that avatar's examples n times each,
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
            for h in r.get("history") or ():
                msgs.append({"role": "user", "content": h["user_content"]})
                msgs.append({"role": "assistant",
                             "content": fixed.get(h["reply"], h["reply"]),
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
    if a.cmd == "stats":
        s = stats(recs, notes, counts=counts)
        print(stats_text(s))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump(s, fh, ensure_ascii=False, indent=1)
        return 0
    if a.cmd == "sft":
        w = parse_weights(a.avatar_weight,
                          {r["avatar"]["name"] for r in recs})
        n = write_jsonl(a.out, sft_examples(a.root, recs, notes, a.per,
                                            a.meta, fix, w))
        print(f"{n} examples -> {a.out}")
        if counts:
            print(correction_line(counts))
        return 0
    n = action_pairs(a.root, recs, notes, a.out, a.traj)
    print(f"{n} pairs -> {a.out}.jsonl, {a.out}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
