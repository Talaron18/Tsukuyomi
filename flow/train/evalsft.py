#!/usr/bin/env python3
"""
The tuned director against its teacher on held-out sessions (flow.train.
sft's split.json): the same system prompt (the registry's bytes), the same
user lines with the same pose lines, turn by turn, each model keeping its
own replies as history (as the director does live). Every reply goes
through the avatar's parser and is played the way the data engine plays it
(flow.dataset.play: moves built, poses solved, gestures dropped), not
rendered.

  run       one model (--name) over the held-out sessions -> <out>/<name>.
            jsonl (a line per turn) and <name>.summary.json. Any director
            backend: ollama (the GGUF the Modelfile serves: the default
            for the student, it is what flow.live will run), hf
            (transformers + the adapter), deepseek (the teacher, live), or
            "recorded": the dataset's own teacher replies (no calls; no
            latency).
  compare   the summaries side by side.
  ab        a blind A/B pack in scripts/body_vs_tags.sh's layout: the first
            --n held-out first turns per avatar, each model's reply
            rendered by flow.director (--backend file: the reply as
            written), <out>/ab/<avatar>/<name>/<i>/ (reply.txt, actions.
            json, demo.mp4), then flow.legacy.abtest make <out>/ab/page.

Per turn (and summed): parse (a parser warning other than an unplayed
marker: flow.dataexport's "parse" flag), refused (the backend said so)
and refusal_text (an apology / "as an AI" / "I can't" opening, or nothing
spoken: local backends never report a refusal), cutoff (the token limit),
markers per tag, unplayed (markers parsed away), not_played (false,
"unsolved", "dropped"), reach targets / ok / MISS from the pose solve,
first_text_seconds, model_seconds, out tokens and tokens/s after the first
token.

Usage (scripts/eval_sft.sh runs all of it):
    python -m flow.train.evalsft run --data data/datagen/r1 \\
        --split runs/sft/q8/split.json --out runs/sft/q8/eval \\
        --name student --backend ollama --model tsk-q8
    python -m flow.train.evalsft run ... --name teacher --backend deepseek
    python -m flow.train.evalsft compare runs/sft/q8/eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

try:
    from .. import dataexport as dx
    from .. import dataset as ds
    from .. import director as dr
    from .. import pose as pz
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from flow import dataexport as dx
    from flow import dataset as ds
    from flow import director as dr
    from flow import pose as pz

REFUSAL = re.compile(
    r"^\W*(i'?m sorry,? but|sorry,? (but )?i can'?t|i can(no|')t (help|do|"
    r"assist|comply)|i am unable|i'?m unable|as an ai|as a language model|"
    r"i'?m an ai|抱歉|对不起，?我(不能|无法)|我无法|申し訳(ありません|ございません)"
    r"|できません)", re.I)


def refusal_text(spoken) -> bool:
    s = (spoken or "").strip()
    return not s or bool(REFUSAL.search(s))


class Recorded:
    """The dataset's own replies as a backend: turn by turn, as recorded."""

    def __init__(self, replies):
        self.replies, self.k = list(replies), 0
        self.model, self.stop_reason, self.usage = "recorded", None, {}

    def stream(self, system, messages):
        text = self.replies[self.k]
        self.k += 1
        self.stop_reason = "stop"
        for i in range(0, len(text), 7):
            yield text[i:i + 7]


def held_out_sessions(data, split_path, limit=None):
    """{session: [records in turn order]} of the split's eval sessions that
    are in the data (refused turns kept: the user line still comes)."""
    with open(split_path, encoding="utf-8") as fh:
        want = set(json.load(fh)["eval_sessions"])
    by_s = defaultdict(list)
    for r in ds.iter_records(data):
        if r["session"] in want:
            by_s[r["session"]].append(r)
    out = {s: sorted(v, key=lambda r: r["turn"]) for s, v in
           sorted(by_s.items())}
    if limit:
        out = dict(list(out.items())[:limit])
    return out


def _avatars(data, names, opts, log):
    reg = dx.load_registry(data)
    out = {}
    for n in names:
        spec = dict(reg["avatars"][n]["spec"])
        # The spec holds the generating machine's paths; on another one
        # (the GPU box) the event stats and clips may be absent: the
        # defaults score the same markers (reach is the body file's).
        ev = spec.get("events")
        if ev and ev != "default" and not os.path.exists(ev):
            log(f"eval: {n}: no {ev} here; default event stats")
            spec["events"] = "default"
        if spec.get("clips") and not os.path.exists(spec["clips"]):
            log(f"eval: {n}: no {spec['clips']} here; no clips")
            spec["clips"] = None
        out[n] = ds.Avatar(spec, opts, {"eval": None}, log)
    return out, reg


def turn_metrics(av, reply, seed):
    """One reply's numbers (module docstring)."""
    warn = list(reply.get("warnings") or ())
    parse = [w for w in warn if not (re.match(r"\s*\[(\w+)", w)
                                     and "dropped" in w
                                     and "not a body-mode marker" in w)]
    unplayed = Counter()
    for w in warn:
        m = re.match(r"\s*\[(\w+)", w)
        if m and "dropped" in w:
            unplayed[m.group(1)] += 1
    m = {"parse": len(parse), "parse_warnings": parse,
         "unplayed": dict(unplayed),
         "refused": bool(reply.get("refused")),
         "refusal_text": refusal_text(reply.get("spoken")),
         "cutoff": reply.get("stop_reason") in ("max_tokens", "length"),
         "first_text_seconds": reply.get("first_text_seconds"),
         "model_seconds": reply.get("model_seconds")}
    tok = ds.usage_tokens(reply.get("usage")) or {
        "out": ds.est_tokens(reply.get("raw")), "in": 0, "in_hit": 0}
    m["out_tokens"] = tok["out"]
    f, ms = m["first_text_seconds"], m["model_seconds"]
    m["tokens_per_second"] = round((tok["out"] - 1) / (ms - f), 2) \
        if f is not None and ms and ms > f and tok["out"] > 1 else None
    tags = Counter(t["tag"] for t in reply.get("tags") or ())
    m["markers"] = dict(tags)
    m["reach"] = pz.reach_summary([])
    m["not_played"] = {}
    if reply.get("refused") or not reply.get("spoken"):
        return m
    with av.lock:
        fields, _arr = ds.play(av, reply, seed)
    m["reach"] = fields["reach"]
    m["not_played"] = dict(Counter(
        f"{x['tag']}:{x['played']}" for x in fields["markers"]
        if x["played"] is not True))
    m["moves"] = len(fields["moves"])
    return m


def run(o, log=print) -> dict:
    sess = held_out_sessions(o.data, o.split, o.sessions)
    if not sess:
        raise SystemExit(f"no held-out session of {o.split} in {o.data}")
    opts = ds.run_opts(o.data)
    opts.tts = "none"
    names = sorted({v[0]["avatar"]["name"] for v in sess.values()})
    avatars, reg = _avatars(o.data, names, opts, log)
    os.makedirs(o.out, exist_ok=True)
    path = os.path.join(o.out, f"{o.name}.jsonl")
    rows = []
    t0 = time.monotonic()
    for sid, recs in sess.items():
        av = avatars[recs[0]["avatar"]["name"]]
        system = reg["prompts"][recs[0]["avatar"]["prompt_sha"]]
        if o.backend == "recorded":
            be = Recorded([r["reply_raw"] for r in recs])
        else:
            be = dr.make_backend(o, log=lambda s: None)
        history = []
        for r in recs[:o.turns or None]:
            msgs = list(history) + [{"role": "user",
                                     "content": r["user_content"]}]
            reply = dr.ask(be, system, msgs, av.labels("eval"),
                           log=lambda s: None, parser=av.parser("eval"))
            m = turn_metrics(av, reply, r["seed"])
            row = {"id": r["id"], "session": sid, "turn": r["turn"],
                   "avatar": av.name, "persona": r["persona"],
                   "name": o.name, "backend": o.backend,
                   "model": getattr(be, "model", None), "user": r["user"],
                   "reply_raw": reply["raw"], "spoken": reply["spoken"],
                   **m}
            rows.append(row)
            if not reply["refused"]:
                history += [{"role": "user", "content": r["user_content"]},
                            {"role": "assistant", "content": reply["raw"]}]
            log(f"evalsft: {o.name} {r['id']}: {sum(m['markers'].values())} "
                f"markers, reach {m['reach']['ok']}/{m['reach']['targets']}, "
                f"parse {m['parse']}"
                + (f", first text {m['first_text_seconds']}s"
                   if m["first_text_seconds"] is not None else ""))
    with open(path, "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    s = summarize(rows)
    s.update(name=o.name, backend=o.backend, model=o.model,
             sessions=len(sess), seconds=round(time.monotonic() - t0, 1))
    with open(os.path.join(o.out, f"{o.name}.summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(s, fh, indent=1, ensure_ascii=False)
    log(summary_text([s]))
    return s


def _q(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 3)


def summarize(rows) -> dict:
    n = len(rows)
    tags = Counter()
    np_ = Counter()
    unpl = Counter()
    reach = Counter()
    for r in rows:
        tags.update(r["markers"])
        np_.update(r["not_played"])
        unpl.update(r["unplayed"])
        for k in ("targets", "ok", "miss"):
            reach[k] += int(r["reach"].get(k) or 0)

    def rate(k):
        return round(sum(bool(r[k]) for r in rows) / n, 4) if n else None
    return {"turns": n,
            "parse_rate": rate("parse"), "refused_rate": rate("refused"),
            "refusal_text_rate": rate("refusal_text"),
            "cutoff_rate": rate("cutoff"),
            "markers_per_turn": {t: round(c / n, 3) for t, c in
                                 sorted(tags.items())} if n else {},
            "body_markers_per_turn": round(sum(
                c for t, c in tags.items() if t not in ("pause", "frame"))
                / n, 3)
            if n else None,
            "unplayed_per_turn": round(sum(unpl.values()) / n, 3) if n
            else None,
            "unplayed": dict(unpl), "not_played": dict(np_),
            "reach": dict(reach, rate=round(reach["ok"] / reach["targets"], 4)
                          if reach["targets"] else None),
            "miss_per_turn": round(reach["miss"] / n, 3) if n else None,
            "first_text_seconds": {"p50": _q([r["first_text_seconds"]
                                              for r in rows], 0.5),
                                   "p90": _q([r["first_text_seconds"]
                                              for r in rows], 0.9)},
            "tokens_per_second": {"p50": _q([r["tokens_per_second"]
                                             for r in rows], 0.5)},
            "out_tokens_per_turn": round(sum(r["out_tokens"] for r in rows)
                                         / n, 1) if n else None}


def summary_text(sums) -> str:
    cols = [("turns", lambda s: s["turns"]),
            ("parse", lambda s: s["parse_rate"]),
            ("refused", lambda s: s["refused_rate"]),
            ("refusal~", lambda s: s["refusal_text_rate"]),
            ("cutoff", lambda s: s["cutoff_rate"]),
            ("body/turn", lambda s: s["body_markers_per_turn"]),
            ("pose/turn", lambda s: s["markers_per_turn"].get("pose", 0)),
            ("move/turn", lambda s: s["markers_per_turn"].get("move", 0)),
            ("frame/turn", lambda s: s["markers_per_turn"].get("frame", 0)),
            ("unplayed/turn", lambda s: s["unplayed_per_turn"]),
            ("reach", lambda s: s["reach"]["rate"]),
            ("MISS/turn", lambda s: s["miss_per_turn"]),
            ("first s p50", lambda s: s["first_text_seconds"]["p50"]),
            ("tok/s p50", lambda s: s["tokens_per_second"]["p50"]),
            ("out tok", lambda s: s["out_tokens_per_turn"])]
    head = f"{'':14s}" + "".join(f"{c:>14s}" for c, _ in cols)
    out = [head]
    for s in sums:
        cells = []
        for _c, f in cols:
            v = f(s)
            cells.append(f"{'-' if v is None else v!s:>14s}")
        out.append(f"{s.get('name', '?')[:14]:14s}" + "".join(cells))
    return "\n".join(out)


def compare(out) -> str:
    sums = []
    for f in sorted(os.listdir(out)):
        if f.endswith(".summary.json"):
            with open(os.path.join(out, f), encoding="utf-8") as fh:
                sums.append(json.load(fh))
    text = summary_text(sums)
    print(text)
    return text


# ------------------------------------------------------------------ A/B

def director_argv(spec, reply_file, out_dir, seed=0, tts="auto",
                  render=True):
    """flow.director playing a written reply on this avatar (body mode)."""
    argv = [sys.executable, "-m", "flow.director", "--mode", "body",
            "--backend", "file", "--reply-file", reply_file, "--out",
            out_dir, "--seed", str(seed), "--tts", tts]
    if spec["kind"] == "vrm":
        argv += ["--avatar", "vrm", "--vrm-body", spec["body"], "--events",
                 spec.get("events") or "default"]
        if spec.get("vrm"):
            argv += ["--vrm", spec["vrm"]]
    else:
        argv += ["--model-json", spec["model"], "--rig", spec["rig"],
                 "--events", spec.get("events") or "default", "--probe-dir",
                 spec.get("probe_dir") or "out", "--body", spec["body"]]
        if spec.get("clips"):
            argv += ["--clips", spec["clips"]]
    if render:
        argv.append("--render")
    return argv


def ab(o, log=print) -> list[str]:
    """The A/B pack (module docstring); returns abtest's items."""
    reg = dx.load_registry(o.data)
    first = defaultdict(list)          # avatar -> [(id, {name: reply})]
    rows = {}
    for name in o.names:
        with open(os.path.join(o.out, f"{name}.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["turn"] == 1 and r["spoken"]:
                    rows.setdefault(r["id"], {})[name] = r
    for rid in sorted(rows):
        by = rows[rid]
        if len(by) == len(o.names):
            av = next(iter(by.values()))["avatar"]
            if len(first[av]) < o.n:
                first[av].append((rid, by))
    items = []
    root = os.path.join(o.out, "ab")
    for av, lst in sorted(first.items()):
        spec = reg["avatars"][av]["spec"]
        if spec["kind"] == "live2d-lite":
            log(f"evalsft: ab: {av} has no model to render; skipped")
            continue
        for i, (_rid, by) in enumerate(lst):
            for name, r in by.items():
                d = os.path.join(root, av, name, str(i))
                os.makedirs(d, exist_ok=True)
                rf = os.path.join(d, "reply.in.txt")
                with open(rf, "w", encoding="utf-8") as fh:
                    fh.write(r["reply_raw"])
                argv = director_argv(spec, rf, d, o.seed, o.tts,
                                     not o.no_render)
                if not o.no_render and o.xvfb:
                    argv = ["xvfb-run", "-a", "-s",
                            "-screen 0 1024x1024x24"] + argv
                log("evalsft: ab: " + " ".join(argv))
                with open(d + ".log", "w", encoding="utf-8") as lf:
                    rc = subprocess.run(argv, stdout=lf, stderr=lf).returncode
                if rc:
                    log(f"evalsft: ab: {d} failed (see {d}.log)")
                    continue
                for v in ("demo.mp4", os.path.join("render", "preview.mp4")):
                    if os.path.exists(os.path.join(d, v)):
                        items.append(f"{av}-{i}/{name}={os.path.join(d, v)}")
                        break
    if len(items) >= 2 and not o.no_render:
        try:
            from ..legacy import abtest
        except ImportError:
            from flow.legacy import abtest
        abtest.make(os.path.join(root, "page"), items, o.seed)
        log(f"evalsft: ab: {len(items)} clips -> {root}/page/index.html "
            "(judge, then python -m flow.legacy.abtest tally "
            f"{root}/page)")
    return items


# ----------------------------------------------------------------- cli

def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    dr.add_args(p)
    p.add_argument("--data", required=True, help="the flow.dataset --out")
    p.add_argument("--split", required=True, help="flow.train.sft's "
                   "split.json")
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True, help="student, teacher, ...")
    p.add_argument("--sessions", type=int, default=None,
                   help="the first N held-out sessions only")
    p.add_argument("--turns", type=int, default=None,
                   help="the first N turns of each")
    p.set_defaults(backend="ollama")
    p.add_argument("--recorded", action="store_true",
                   help="the dataset's own replies (--backend recorded)")
    p = sub.add_parser("compare")
    p.add_argument("out")
    p = sub.add_parser("ab")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="the run's --out")
    p.add_argument("--names", default="teacher,student")
    p.add_argument("--n", type=int, default=6,
                   help="first turns per avatar")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tts", default="auto")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--no-xvfb", dest="xvfb", action="store_false")
    o = ap.parse_args(argv)
    if o.cmd == "run" and o.recorded:
        o.backend = "recorded"
    if o.cmd == "ab":
        o.names = [x for x in o.names.split(",") if x]
    return o


def main(argv=None) -> int:
    o = parse(list(sys.argv[1:] if argv is None else argv))
    if o.cmd == "run":
        run(o)
    elif o.cmd == "compare":
        compare(o.out)
    else:
        ab(o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
