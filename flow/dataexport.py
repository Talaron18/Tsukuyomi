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
             counted only: no_pose, unplayed (markers parsed away, e.g. a
             newer marker kind), usage_estimated.
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
    python -m flow.dataexport action data/datagen/pilot --out action
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np

try:
    from . import dataset as ds
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import dataset as ds

HARD = ("refused", "empty", "cutoff", "parse", "no_body", "miss", "dup",
        "lite")
SOFT = ("no_pose", "unplayed", "usage_estimated")
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
    parse = [w for w in r.get("warnings") or ()
             if not (re.match(r"\s*\[(\w+)", w) and "dropped" in w
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
    llm = r.get("llm") or {}
    if any((llm.get(k) or {}).get("usage_estimated") for k in
           ("character", "user")):
        f.append("usage_estimated")
    return f


def annotate(records, max_miss=MAX_MISS, dup=DUP_JACCARD, keep_lite=False):
    """{id: (flags, kept)} for every record."""
    dups = dedupe(records, dup)
    out = {}
    for r in records:
        f = flag(r, max_miss)
        if r["id"] in dups:
            f.append("dup")
        hard = [x for x in f if x in HARD and not (keep_lite and x == "lite")]
        out[r["id"]] = (f, not hard)
    return out


# --------------------------------------------------------------- stats

def stats(records, notes, prices=None) -> dict:
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


def _turn_msgs(r):
    return [{"role": "user", "content": r["user_content"]},
            {"role": "assistant", "content": r["reply_raw"]}]


def session_ok(note) -> bool:
    """A turn a session-mode example keeps: kept, or dropped only as a
    duplicate user line (its reply is new; a repeated greeting at turn 1
    must not cost the session's other turns)."""
    flags, kept = note
    return kept or [x for x in flags if x in HARD] == ["dup"]


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


def sft_examples(root, records, notes, per="session", meta=False):
    """The SFT examples (dicts with "messages")."""
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
        if per == "session":
            msgs = list(base)
            ids = []
            for r in sessions.get(sid, ()):
                msgs += _turn_msgs(r)
                ids.append(r["id"])
            if ids:
                ex = {"messages": msgs}
                if meta:
                    ex["meta"] = {"session": sid, "turns": ids,
                                  "avatar": recs[0]["avatar"]["name"],
                                  "persona": recs[0]["persona"]}
                out.append(ex)
            continue
        for r in recs:
            if not notes[r["id"]][1]:
                continue
            msgs = list(base)
            for h in r.get("history") or ():
                msgs.append({"role": "user", "content": h["user_content"]})
                msgs.append({"role": "assistant", "content": h["reply"],
                             "weight": 0})
            msgs += _turn_msgs(r)
            ex = {"messages": msgs}
            if meta:
                ex["meta"] = {"id": r["id"], "avatar": r["avatar"]["name"],
                              "persona": r["persona"], "reach": r["reach"]}
            out.append(ex)
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
        if name == "stats":
            p.add_argument("--json", default=None)
        if name == "sft":
            p.add_argument("--out", required=True)
            p.add_argument("--per", default="session",
                           choices=("session", "turn"))
            p.add_argument("--meta", action="store_true")
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
    notes = annotate(recs, a.max_miss, a.dup_jaccard, a.keep_lite)
    if a.cmd == "stats":
        s = stats(recs, notes)
        print(stats_text(s))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump(s, fh, ensure_ascii=False, indent=1)
        return 0
    if a.cmd == "sft":
        n = write_jsonl(a.out, sft_examples(a.root, recs, notes, a.per,
                                            a.meta))
        print(f"{n} examples -> {a.out}")
        return 0
    n = action_pairs(a.root, recs, notes, a.out, a.traj)
    print(f"{n} pairs -> {a.out}.jsonl, {a.out}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
