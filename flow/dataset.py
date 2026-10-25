#!/usr/bin/env python3
"""
The Stage 2 data engine: conversations with timed body actions, generated
unattended, as training data for a local character model.

docs/embodiment.md's baseline is an API model directing the body (tags, or
E2's body mode: [pose {...}] markers solved on the measured body). This
module runs that pipeline at scale, with a second language model playing
the user, and records each turn with everything a student needs:

  (a) distillation: a small local LLM fine-tuned (LoRA) to write the same
      marked replies itself, given the same system prompt and user turns
      (flow.dataexport sft);
  (b) an action head on a local model's hidden states, predicting the
      canonical body-state trajectory a reply plays (flow.dataexport
      action): dense per-frame targets, aligned to the spoken characters.

  session    one avatar x one persona x one simulated user, N turns with
             history. The user simulator (DeepSeek by default, any
             director backend) gets the persona, a user type, a topic and a
             language per turn (flow/datagen.personas.json: extendable) and
             the conversation so far as spoken text, markers stripped; its
             line is cleaned (no "User:", quotes, brackets, at most
             USER_MAX characters).
  character  the real body-mode director: the system prompt and parser of
             director.prepare (Live2D: the rig's measured body plus its own
             gestures; VRM: flow.vrm.prepare), the user line plus
             pose.pose_now (each turn from the rest, as the offline director
             does), history as the director keeps it (the raw marked reply
             as the assistant turn; a refused turn is not history).
  play       not rendered: director.timeline (--tts none: the rate model)
             or speech.speak (a real TTS: word / mora timings), pose.
             solve_poses (with the Core's exact check on Live2D when the
             model loads), director.realize + pose.tokens into a
             stream.MotionStream on the avatar's rig (stream.EventRig on
             Live2D, vrm.VRMRig on VRM), and every frame through the body's
             perceive into canonical variables. Two trajectories: "played"
             (the idle, blinks and breath under the directed motion, seeded:
             what the viewer sees) and "directed" (idle and autonomic off,
             the same tokens: what the reply alone asks for, the learnable
             target).

Record format (tsukuyomi.turns/1), one JSON object per line in
<out>/shards/turns-NNNNN.jsonl, one per turn (refused and flagged turns
included; flow.dataexport filters):

  format      "tsukuyomi.turns/1"
  id          "<session>/tNN"; session, turn (1-based), n_turns, run, seed
  avatar      {name, kind: live2d | vrm | live2d-lite, body (path),
               body_sha (sha256 of the body file, 16 hex), prompt_sha}
  persona, user_type, topic
  lang        {asked (en/ja/zh), user, reply: detected from the script, or
               null}
  history     [{user, user_content, reply}] the earlier turns as the model
              saw them: user_content is the user message (line + "\\n" +
              pose_now), reply the raw marked text
  user, user_content
  reply_raw   the model's text, markers in it; spoken: what is said
  refused, stop_reason, warnings (parser and planning)
  markers     [{tag, arg, at (char offset in spoken), t (seconds; null for
              pause), text (as written), played: true | false | "dropped"
              (a gesture giving way to an arm pose) | "unsolved"}]
  unplayed    {tag: count} markers the parser dropped (e.g. tags of another
              mode, or markers newer than this pipeline)
  poses       pose.solve_poses records, rounded: t, start, dur, hold,
              release, targets (as written), resolved, params (as played),
              parts, variant_name, report {var: {target, got, error, unit,
              ok}}, exact / exact_ok (Live2D with the Core), ok, via
  reach       pose.reach_summary {markers, targets, ok, miss, exact_ok,
              exact_n, solve_seconds}
  timing      {speech_seconds, seconds, frames, fps, method: rate | the
              TTS's (edge words, voicevox moras ...), tts}
  traj        {file: "traj/<session>.npz", key: "tNN", vars: [canonical
              variable per column], idle: true} or null (refused)
  llm         {character, user}: {backend, model, usage (as the backend
              logged it), tokens {in, in_hit, out}, usage_estimated,
              seconds, first_text_seconds, attempts}
  cost_usd    both calls at the run's prices

  <out>/traj/<session>.npz (np.savez_compressed), per turn key k = "tNN":
    k             float16 [frames, vars] played trajectory (NaN: absent)
    k.directed    float16 [frames, vars] directed only
    k.variant     int8 [frames] arm variant index (Live2D drawings; 0 VRM)
    k.chars       float16 [len(spoken), 2] each spoken character's start,
                  end in seconds (silent punctuation: start == end)
    k.replay      JSON string: {acts (timed), tokens (directed tokens as
                  streamed)} -- with the seed, enough to replay the stream
                  after pose / stream change
    vars          JSON string: the variable list
  <out>/avatars.json   tsukuyomi.avatars/1: per avatar kind, body, body_sha,
                  pose_now, traj_vars, variants, prompts {persona: sha};
                  "prompts" {sha: system prompt text} verbatim (the SFT
                  export uses these bytes)
  <out>/sessions.jsonl  one line per finished session: {session, status
                  ok | failed, turns, tokens, cost_usd, error, run, at}; a
                  session is written whole (shard lines, npz, then this
                  line), so --resume skips "ok" ones and reruns the rest;
                  readers dedupe turns by id (last wins)
  <out>/runs.jsonl  one line per run: argv, git head and dirty flag, prices,
                  start time; the end totals as a second line

Values are rounded (ROUND decimals) and trajectories float16: a 6-turn
session at 30 fps is ~100 KB of npz.

Usage:
    python -m flow.dataset gen --avatars Hiyori,seed-san,lafei \\
        --personas cheerful,sleepy,tsundere --sessions 20 --turns 6 \\
        --out data/datagen/pilot
    python -m flow.dataset estimate ...       # the cost before starting
    scripts/datagen.sh                         # the unattended runner
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import zlib

import numpy as np

try:
    from . import body as fb
    from . import director as dr
    from . import pose as pz
    from . import speech as sp
    from . import stream as fs
    from . import vrm as fv
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import body as fb
    from flow import director as dr
    from flow import pose as pz
    from flow import speech as sp
    from flow import stream as fs
    from flow import vrm as fv

FORMAT = "tsukuyomi.turns/1"
AVATARS_FORMAT = "tsukuyomi.avatars/1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "flow", "datagen.personas.json")
ROUND = 4
USER_MAX = 300          # characters of a simulated user line at most
SHARD_SIZE = 5000       # turn records per shard
# Prices in US$ per million tokens: DeepSeek's deepseek-chat list prices
# as last checked (2025-09: input 0.28 on a cache miss, 0.028 on a hit,
# output 0.42). Check https://api-docs.deepseek.com/quick_start/pricing
# before trusting an estimate; --price-* override them.
PRICES = {"in": 0.28, "in_hit": 0.028, "out": 0.42}
LANG_NAMES = {"en": "English", "ja": "Japanese", "zh": "Chinese (Simplified)",
              "ko": "Korean"}
FATAL = re.compile(r"\b(401|402|403) from ")


class Fatal(Exception):
    """A failure no retry fixes (a missing or rejected key, no balance):
    the run stops with the manifest saved."""


def _r(x, nd=ROUND):
    """x with every float rounded (dicts, lists, tuples, numpy scalars)."""
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else round(v, nd)
    return x


def sha(data, n=16) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:n]


def file_sha(path) -> str:
    with open(path, "rb") as fh:
        return sha(fh.read())


def git_state(root=ROOT) -> dict:
    def run(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=root, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance, not required
            return ""
    return {"head": run("rev-parse", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


# ------------------------------------------------------------ config

def load_config(path=None) -> dict:
    """The persona / user / topic lists (JSON, or YAML when PyYAML is
    installed and the file ends in .yaml / .yml)."""
    path = path or CONFIG
    with open(path, encoding="utf-8") as fh:
        if path.endswith((".yaml", ".yml")):
            import yaml
            cfg = yaml.safe_load(fh)
        else:
            cfg = json.load(fh)
    for key in ("personas", "users", "topics"):
        if not cfg.get(key):
            raise SystemExit(f"{path}: no {key}")
    ids = [p["id"] for p in cfg["personas"]]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate persona ids")
    return cfg


def persona_text(cfg, persona, avatar) -> str:
    """The director's --persona for this avatar: the avatar's name from the
    config's "names" (else its own), then the persona's description, then
    the config's language note."""
    name = (cfg.get("names") or {}).get(avatar, avatar)
    text = persona["persona"].strip()
    note = cfg.get("persona_suffix", "")
    return f"{name}, {text}" + (f" {note}" if note else "")


def _pick(rng, weights: dict):
    keys = sorted(weights)
    w = [float(weights[k]) for k in keys]
    return rng.choices(keys, weights=w)[0]


# ------------------------------------------------------------- tokens

def est_tokens(text) -> int:
    """A rough token count (DeepSeek's tokenizer: ~0.6 per CJK character,
    ~0.3 per other character): for estimates and for a usage the backend
    did not report."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if ord(ch) >= 0x2E80)
    return int(math.ceil(cjk * 0.6 + (len(text) - cjk) * 0.3))


def _content_text(c):
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c or ""


def usage_tokens(usage) -> dict | None:
    """{in, in_hit, out} from a backend's usage fields (OpenAI-compatible,
    Anthropic or Ollama), or None when it reported none."""
    u = usage or {}
    if u.get("prompt_tokens") is not None or u.get("completion_tokens") \
            is not None:
        return {"in": int(u.get("prompt_tokens") or 0),
                "in_hit": int(u.get("prompt_cache_hit_tokens") or 0),
                "out": int(u.get("completion_tokens") or 0)}
    if u.get("input_tokens") is not None:
        hit = int(u.get("cache_read_input_tokens") or 0)
        return {"in": int(u["input_tokens"]) + hit
                + int(u.get("cache_creation_input_tokens") or 0),
                "in_hit": hit, "out": int(u.get("output_tokens") or 0)}
    if u.get("prompt_eval_count") is not None:
        return {"in": int(u.get("prompt_eval_count") or 0), "in_hit": 0,
                "out": int(u.get("eval_count") or 0)}
    return None


def cost(tokens, prices=PRICES) -> float:
    if not tokens:
        return 0.0
    hit = min(tokens["in_hit"], tokens["in"])
    return ((tokens["in"] - hit) * prices["in"] + hit * prices["in_hit"]
            + tokens["out"] * prices["out"]) / 1e6


def call_tokens(usage, system, messages, out_text):
    """(tokens, estimated): the usage's tokens, else an estimate from the
    characters sent and received."""
    t = usage_tokens(usage)
    if t is not None:
        return t, False
    sent = (system or "") + "".join(_content_text(m.get("content"))
                                    for m in messages)
    return {"in": est_tokens(sent), "in_hit": 0,
            "out": est_tokens(out_text)}, True


# ------------------------------------------------------------ avatars

def _find_model(name, roots):
    for r in roots:
        if not os.path.exists(r):
            continue
        for dp, _dn, fn in os.walk(r, followlinks=True):
            if f"{name}.model3.json" in fn:
                return os.path.join(dp, f"{name}.model3.json")
    return None


def avatar_spec(item, models=("live2d-models", "CubismWebSamples"),
                rigs="rigs", events=None, lite=False) -> dict:
    """An --avatars entry: NAME, or NAME:key=value,... (model, rig, body,
    clips, events, probe_dir, vrm). A body file of kind "vrm" (rigs/NAME.
    body.json) is a VRM avatar; anything else is a Live2D rig, which needs
    its model3.json (found under `models`, as scripts/body_vs_tags.sh
    finds it) unless `lite` (no rig, no idle: body file only)."""
    name, _, rest = item.partition(":")
    kv = dict(p.split("=", 1) for p in rest.split(",") if "=" in p)
    body = kv.get("body") or os.path.join(rigs, f"{name}.body.json")
    if not os.path.exists(body):
        raise SystemExit(f"avatar {name}: no body file {body}")
    with open(body, encoding="utf-8") as fh:
        kind = json.load(fh).get("kind", "live2d")
    spec = {"name": name, "body": body, "kind": kind}
    if kind == "vrm":
        spec["vrm"] = kv.get("vrm")
        spec["events"] = kv.get("events") or events or "default"
        return spec
    model = kv.get("model") or _find_model(name, models)
    rig = kv.get("rig") or os.path.join(rigs, f"{name}.npz")
    if model is None or not os.path.exists(rig):
        if not lite:
            raise SystemExit(
                f"avatar {name}: Live2D needs its model3.json (searched "
                f"{', '.join(models)}) and {rig}; pass {name}:model=PATH,"
                "rig=PATH, or --live2d-lite to generate from the body file "
                "alone (no gestures, no idle: flagged, kept out of the SFT "
                "export by default)")
        spec["kind"] = "live2d-lite"
        return spec
    clips = kv.get("clips") or os.path.join("clips2", f"{name}.npz")
    ev_path = kv.get("events") or events
    if ev_path is None:
        ev_path = "ev.json" if os.path.exists("ev.json") else "default"
    spec.update(model=model, rig=rig,
                clips=clips if os.path.exists(clips) else None,
                events=ev_path, probe_dir=kv.get("probe_dir", "out"))
    return spec


def director_args(spec, opts, persona=None):
    """The director's arguments for this avatar in body mode, parsed by
    director.add_args (so every default is the director's own)."""
    ap = argparse.ArgumentParser()
    dr.add_args(ap)
    argv = ["--mode", "body", "--backend", opts.backend,
            "--fps", str(opts.fps), "--seed", "0"]
    if opts.model:
        argv += ["--model", opts.model]
    if opts.api_base:
        argv += ["--api-base", opts.api_base]
    if spec["kind"] == "vrm":
        argv += ["--avatar", "vrm", "--vrm-body", spec["body"],
                 "--events", spec["events"]]
        if spec.get("vrm"):
            argv += ["--vrm", spec["vrm"]]
    elif spec["kind"] == "live2d":
        argv += ["--model-json", spec["model"], "--rig", spec["rig"],
                 "--events", spec["events"], "--probe-dir",
                 spec["probe_dir"], "--body", spec["body"]]
        if spec.get("clips"):
            argv += ["--clips", spec["clips"]]
    if persona:
        argv += ["--persona", persona]
    if opts.tts and opts.tts != "none":
        argv += ["--tts", opts.tts]
    if not opts.gestures:
        argv.append("--no-body-gestures")
    a = ap.parse_args(argv)
    a.tail = opts.tail
    if opts.max_tokens:
        a.max_tokens = opts.max_tokens
    return a


class Avatar:
    """One avatar, prepared once: its body, rig (the stream's), Core, TTS,
    and one director run per persona (system prompt, parser, gestures).
    `lock` serialises the CPU play (the Core, the rig's gesture player and
    the body's caches are not thread-safe); the model calls stay
    concurrent."""

    def __init__(self, spec, opts, personas, log=print):
        self.spec, self.name, self.kind = spec, spec["name"], spec["kind"]
        self.opts = opts
        self.lock = threading.Lock()
        self.body_path = spec["body"]
        self.body_sha = file_sha(self.body_path)
        quiet = io.StringIO()
        self.runs = {}
        self.a = director_args(spec, opts)
        self.core, self.rig, self.motions, self.mouth_cal = None, None, {}, None
        if self.kind == "vrm":
            self.bm, _ = fv.body_for(None, self.body_path, log=lambda s: None)
            self.rig = fv.VRMRig(self.bm, spec["events"], opts.fps,
                                 self.a.idle_profile, log=lambda s: None)
        elif self.kind == "live2d":
            self.rig = fs.EventRig(
                spec["model"], spec["rig"], spec["events"], spec.get("clips"),
                self.a.binding, opts.fps, spec["probe_dir"],
                self.a.idle_profile, self.a.no_snap, self.a.events_pool_only,
                log=lambda s: None, breath_gain=self.a.breath_gain,
                voice_config=dr.voice_config_path(self.a))
            self.motions = dict(self.rig.motion_seconds)
        else:
            self.bm = fb.load(self.body_path)
        for pid, ptext in personas.items():
            a = director_args(spec, opts, ptext)
            if self.kind == "vrm":
                run = fv.prepare(a, self.bm, self.body_path, quiet)
            elif self.kind == "live2d":
                run = dr.prepare(a, quiet)
                self.bm = run["body"]
                if self.core is None:
                    self.core = run.get("core")
            else:
                bm = self.bm
                run = {"system": pz.system_prompt(bm, ptext, False, ""),
                       "labels": {}, "body": bm,
                       "parser": lambda bm=bm: pz.parser_for(bm)}
            self.runs[pid] = run
        self.pose_now = pz.pose_now(self.bm)
        self.traj_vars = traj_vars(self.bm, opts.traj_vars)
        self.variant_names = [v["name"] for v in self.bm.variants]
        self.arms = pz.arm_params(self.bm)
        self.tts = None
        if opts.tts and opts.tts != "none":
            if self.kind == "vrm":
                self.a.lipsync = "vowels"
            self.tts = dr.make_voice(self.a, log=lambda s: None)
            self.mouth_cal = getattr(self.a, "mouth_cal", None)
        log(f"dataset: avatar {self.name} ({self.kind}, {self.body_path} "
            f"{self.body_sha}): {len(self.traj_vars)} trajectory variables, "
            f"{len(self.variant_names)} arm variants, "
            + ", ".join(f"{p} prompt {len(r['system'])} chars"
                        for p, r in self.runs.items()))

    def system(self, persona):
        return self.runs[persona]["system"]

    def parser(self, persona):
        return self.runs[persona]["parser"]()

    def labels(self, persona):
        return self.runs[persona].get("labels") or {}

    def registry(self) -> dict:
        return {"kind": self.kind, "body": self.body_path,
                "body_sha": self.body_sha, "pose_now": self.pose_now,
                "traj_vars": self.traj_vars, "variants": self.variant_names,
                "fps": float(self.opts.fps),
                "gestures": sorted(self.labels(next(iter(self.runs))))
                if self.runs else [],
                "spec": {k: v for k, v in self.spec.items()},
                "prompts": {p: sha(r["system"]) for p, r in self.runs.items()}}


def traj_vars(bm, which="offered") -> list[str]:
    """The trajectory's columns: the variables the prompt offers (pose.
    offered), the mouth, and the hands' depth on a 3D body; "all": every
    variable some variant of the body has (pose.available + mouth)."""
    if which == "all":
        out = list(pz.available(bm))
    else:
        out = list(pz.offered(bm))
        out += [v for v in ("hand.L.z", "hand.R.z") if v in bm.fidx]
    if "mouth.open" in bm.fidx and "mouth.open" not in out:
        out.append("mouth.open")
    return out


# --------------------------------------------------------------- play

def _quiet(_s):
    pass


def _frames(av, acts, poses, seed, n, idle, mouth=None):
    """Run the avatar's stream n frames with this turn's directed tokens:
    ([n, vars] canonical, [n] variant index, directed tokens)."""
    bm, fps = av.bm, float(av.opts.fps)
    cols = av.traj_vars
    vidx = {name: k for k, name in enumerate(av.variant_names)}
    out = np.full((n, len(cols)), np.nan, np.float32)
    var = np.zeros(n, np.int8)
    if av.kind == "live2d-lite":
        toks = pz.tokens(poses, {}, av.arms)
        lay = pz.layer_for(toks, fps, bm.default_values(),
                           bm.blob.get("rest_parts"))
        rest = bm.default_values()
        rest_parts = dict(bm.blob.get("rest_parts", {}))
        for f in range(n):
            over, parts = dict(rest), dict(rest_parts)
            if lay is not None:
                over, parts = lay.apply(f, over, parts)
            st = bm.perceive(over, parts=parts)
            out[f] = [st.get(c, np.nan) for c in cols]
            var[f] = vidx.get(st.get("arm.variant"), 0)
        return out, var, toks
    rig = av.rig
    cal = av.mouth_cal or {}
    stream = fs.MotionStream(rig, seed=seed, idle=idle, log=_quiet,
                             autonomic=idle,
                             mouth_map=cal.get("map") if av.kind == "live2d"
                             else None)
    rng = np.random.default_rng([int(seed), 1])
    directed = dr.realize([dict(x) for x in acts], rig.stats,
                          list(rig.roles), av.motions, rng, stream.pose(),
                          settle=getattr(rig, "settle", None))
    if poses:
        directed += pz.tokens(poses, rig.roles, av.arms)
    stream.direct(copy.deepcopy(directed))
    if mouth is not None:
        stream.add_mouth(0, mouth, len(mouth["open"]))
    for f in range(n):
        over, parts = stream.frame()
        st = bm.perceive(over, parts=parts) if av.kind == "live2d" \
            else bm.perceive(over)
        out[f] = [st.get(c, np.nan) for c in cols]
        var[f] = vidx.get(st.get("arm.variant"), 0)
    return out, var, directed


def play(av, reply, seed):
    """A reply on the avatar, not rendered: (record fields, arrays). Must
    run under av.lock."""
    opts, a = av.opts, av.a
    spoken, tags = reply["spoken"], reply["tags"]
    fps = float(opts.fps)
    warnings = []

    def log(s):
        if "WARNING" in s or "dropped" in s or "failed" in s:
            warnings.append(s)
    tts_info, mouth, method = None, None, "rate"
    if av.tts is not None and spoken:
        moods = sp.sentence_moods(spoken, tags)
        said = sp.speak(spoken, tags, av.tts,
                        lambda ch: 1.0 / (a.cjk_rate if dr._is_cjk(ch)
                                          else a.latin_rate), log=_quiet,
                        moods=moods)
        acts, spans, end = said["acts"], said["spans"], said["end"]
        method = said["method"]
        tts_info = {"backend": getattr(av.tts, "kind", None) or opts.tts,
                    "voice": getattr(av.tts, "last_voice", None)}
        if av.kind != "live2d-lite":
            nf = int(math.ceil(end * fps)) + 2
            lip = "vowels" if av.kind == "vrm" else "auto"
            cal = av.mouth_cal if av.kind == "live2d" else None
            mouth = sp.mouth_track(said["audio"], said["sr"], fps, nf,
                                   lip if cal or lip == "vowels" else "open",
                                   a.mouth_gain, cal=cal,
                                   moras=said.get("moras"))
    else:
        acts, spans, end = dr.timeline(spoken, tags, a.cjk_rate,
                                       a.latin_rate)
    orig = list(acts)
    poses, gone = [], []
    if any(x.get("tag") == pz.POSE for x in acts):
        poses = pz.solve_poses(av.bm, acts, core=av.core, log=log)
        for rec in poses:
            rec["arm_params"] = sorted(set(rec["params"]) & av.arms)
        gone = pz.drop_clashing_gestures(acts, poses, av.motions, log)
    seconds = dr.length(acts, end, av.motions, opts.tail)
    if poses:
        seconds = max(seconds, pz.end_time(poses) + opts.tail)
    n = int(round(seconds, 2) * fps) + 1
    gone_ids = {id(x) for x in gone}
    solved = {id(acts[p["act"]]): p for p in poses}
    markers = []
    by_at = {}
    for x in orig:
        by_at.setdefault(x["at"], []).append(x)
    used = set()
    for t in tags:
        if t["tag"] == "pause":
            markers.append({"tag": "pause", "arg": t["arg"], "at": t["at"],
                            "t": None, "text": pz.tag_text(t),
                            "played": True})
            continue
        x = next((y for y in by_at.get(t["at"], ()) if id(y) not in used
                  and y["tag"] == t["tag"]), None)
        if x is None:
            continue
        used.add(id(x))
        if id(x) in gone_ids:
            played = "dropped"
        elif x["tag"] == pz.POSE:
            rec = solved.get(id(x))
            played = True if rec and (rec["params"] or rec["parts"]) \
                else "unsolved"
        else:
            played = x["tag"] in ("blink", "gesture")
        markers.append({"tag": x["tag"], "arg": x["arg"], "at": x["at"],
                        "t": x["t"], "text": pz.tag_text(x),
                        "played": played})
    played_traj, var, directed = _frames(av, acts, poses, seed, n, True,
                                         mouth)
    if av.kind == "live2d-lite":
        directed_traj = played_traj
    else:
        directed_traj, _v, _d = _frames(av, acts, poses, seed, n, False,
                                        mouth)
    chars = np.asarray(spans, np.float32).reshape(-1, 2)
    keep = ("t", "start", "dur", "hold", "release", "targets", "resolved",
            "params", "parts", "variant_name", "report", "exact", "exact_ok",
            "ok", "via", "hints", "at")
    fields = {
        "markers": _r(markers),
        "poses": [_r({k: p[k] for k in keep if k in p}) for p in poses],
        "reach": pz.reach_summary(poses),
        "timing": {"speech_seconds": round(float(end), 3),
                   "seconds": round(float(seconds), 3), "frames": n,
                   "fps": fps, "method": method, "tts": tts_info},
        "play_warnings": warnings,
    }
    arrays = {"": played_traj.astype(np.float16),
              ".directed": directed_traj.astype(np.float16),
              ".variant": var, ".chars": chars.astype(np.float16),
              ".replay": json.dumps(_r({"acts": acts, "tokens": directed}),
                                    ensure_ascii=False)}
    return fields, arrays


# -------------------------------------------------------- user simulator

USER_SYSTEM = """\
You write the next message of a USER chatting with an animated character, \
for a dataset of natural conversations.

The user: {user}
The character: {character}

Write ONLY the user's next message, as the user would type it in a chat: \
one to three short sentences. No name or "User:" prefix, no quotation \
marks, no narration, no stage directions, no brackets, no emotes like \
*waves*. Stay in the conversation: react to what the character just said \
when it fits, but move on to the topic you are given. Do not repeat \
earlier messages or open the same way twice."""


def user_prompt(cfg, persona_txt, user, topic, lang, mix, transcript,
                char_name):
    lines = []
    for who, text in transcript:
        lines.append(f"{'User' if who == 'user' else char_name}: {text}")
    conv = "\n".join(lines) if lines else "(nothing yet: this is the first " \
        "message of the conversation)"
    ask = (f"Conversation so far:\n{conv}\n\nWrite the user's next message. "
           f"What it does: {topic['hint']}. Language: {LANG_NAMES.get(lang, lang)}")
    if mix:
        ask += f", with a few words of {LANG_NAMES.get(mix, mix)} mixed in"
    return ask + "."


_PREFIX = re.compile(r"^\s*(user|you|me|human|ユーザー|用户|我)\s*[:：]\s*", re.I)
_QUOTES = "\"'“”‘’「」『』"


def clean_user_line(text) -> str:
    """The simulator's text as one user line, or "" when nothing usable
    is left: first paragraph, no speaker prefix or surrounding quotes, no
    [bracketed] or *emote* spans, whitespace collapsed, at most USER_MAX
    characters."""
    t = (text or "").strip()
    t = re.split(r"\n\s*\n", t)[0]
    t = _PREFIX.sub("", t).strip()
    for _ in range(2):
        if len(t) >= 2 and t[0] in _QUOTES and t[-1] in _QUOTES:
            t = t[1:-1].strip()
    t = re.sub(r"\[[^\]]*\]|\*[^*\n]{1,60}\*|（[^）]{0,40}）", " ", t)
    t = t.replace("[", " ").replace("]", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > USER_MAX:
        cut = max(t.rfind(c, 0, USER_MAX) for c in ".!?。！？")
        t = t[:cut + 1] if cut > USER_MAX // 3 else t[:USER_MAX]
    return t


def detect_lang(text):
    lang = sp.script_language(text or "")
    return {"han": "zh"}.get(lang, lang)


def collect(backend, system, messages):
    return "".join(backend.stream(system, messages))


# ------------------------------------------------------------ sessions

def plan_sessions(avatars, personas, cfg, n, turns, seed=0) -> list[dict]:
    """n sessions in all, spread round-robin over avatar x persona (a
    persona with an "avatars" list only on those). Ids are stable for a
    cell index, so a longer run resumes a shorter one."""
    cells = []
    by_id = {p["id"]: p for p in cfg["personas"]}
    for av in avatars:
        for p in personas:
            allow = by_id[p].get("avatars")
            if not allow or av in allow:
                cells.append((av, p))
    if not cells:
        raise SystemExit("no avatar x persona pairs (see the personas' "
                         "\"avatars\" lists)")
    out = []
    for i in range(n):
        av, p = cells[i % len(cells)]
        k = i // len(cells)
        sid = f"{av}.{p}.{k:04d}"
        out.append({"id": sid, "avatar": av, "persona": p, "turns": turns,
                    "seed": zlib.crc32(f"{seed}:{sid}".encode()) & 0x7FFFFFFF})
    return out


def _sleep(stop, seconds):
    if stop is not None:
        return stop.wait(seconds)
    time.sleep(seconds)
    return False


def with_retries(fn, retries, backoff, stop=None, log=print, what="call"):
    """fn() with retries on transient errors (network, 429, 5xx, a broken
    stream), exponential backoff with jitter; (result, attempts). A
    missing key (SystemExit) or a 401 / 402 / 403 raises Fatal."""
    for k in range(retries + 1):
        try:
            return fn(), k + 1
        except SystemExit as exc:
            raise Fatal(str(exc)) from None
        except Fatal:
            raise
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            if FATAL.search(str(exc)):
                raise Fatal(str(exc)) from None
            if k == retries:
                raise
            d = min(backoff * 2 ** k, 300.0) * (1 + 0.25 * random.random())
            log(f"dataset: {what} failed ({exc.__class__.__name__}: "
                f"{str(exc)[:200]}); retry {k + 1}/{retries} in {d:.0f}s")
            if _sleep(stop, d):
                raise
    raise RuntimeError("unreachable")


def _model_of(backend):
    return getattr(backend, "model", None) or backend.__class__.__name__


def run_session(sess, av, cfg, opts, char_backend, user_backend, run_id,
                stop=None, log=print, prices=PRICES):
    """One session: (records, {npz key: array}, totals). Raises when a call
    fails past its retries (the session is then rerun on resume)."""
    rng = random.Random(sess["seed"])
    persona = next(p for p in cfg["personas"] if p["id"] == sess["persona"])
    ptext = persona_text(cfg, persona, sess["avatar"])
    user = rng.choice(cfg["users"])
    char_name = (cfg.get("names") or {}).get(sess["avatar"], sess["avatar"])
    langs = persona.get("languages") or {"en": 1.0}
    lang0 = _pick(rng, langs)
    usys = USER_SYSTEM.format(user=user["style"], character=ptext)
    system = av.system(sess["persona"])
    psha = sha(system)
    history, transcript, records, arrays = [], [], [], {}
    tot = {"in": 0, "in_hit": 0, "out": 0, "cost_usd": 0.0}
    openers = [t for t in cfg["topics"] if t.get("opener")]
    for turn in range(1, sess["turns"] + 1):
        if stop is not None and stop.is_set():
            raise InterruptedError("stopped")
        if turn == 1 and openers and rng.random() < 0.7:
            topic = rng.choice(openers)
        else:
            topic = rng.choices(cfg["topics"],
                                [float(t.get("weight", 1)) for t in
                                 cfg["topics"]])[0]
        lang = lang0 if rng.random() > float(persona.get("switch", 0.15)) \
            else _pick(rng, langs)
        others = [k for k in langs if k != lang]
        mix = rng.choice(others) if others and rng.random() < float(
            persona.get("mix", 0.1)) else None
        umsg = [{"role": "user", "content": user_prompt(
            cfg, ptext, user, topic, lang, mix, transcript, char_name)}]
        line, u_att, u_text, u_sec = "", 0, "", 0.0
        for _ in range(3):
            t0 = time.monotonic()
            u_text, k = with_retries(lambda: collect(user_backend, usys, umsg),
                                     opts.retries, opts.backoff, stop, log,
                                     "user simulator")
            u_sec += time.monotonic() - t0
            u_att += k
            line = clean_user_line(u_text)
            if line:
                break
        if not line:
            line = rng.choice(topic.get("examples") or ["Hi!"])
        u_tok, u_est = call_tokens(getattr(user_backend, "usage", None), usys,
                                   umsg, u_text)
        content = line + "\n" + av.pose_now
        msgs = []
        for h in history:
            msgs.append({"role": "user", "content": h["user_content"]})
            msgs.append({"role": "assistant", "content": h["reply"]})
        msgs.append(pz.user_message(content, opts.backend, None))
        reply, c_att = with_retries(
            lambda: dr.ask(char_backend, system, msgs,
                           av.labels(sess["persona"]), log=_quiet,
                           parser=av.parser(sess["persona"])),
            opts.retries, opts.backoff, stop, log, "character")
        c_tok, c_est = call_tokens(reply.get("usage"), system, msgs,
                                   reply["raw"])
        rec = {"format": FORMAT, "id": f"{sess['id']}/t{turn:02d}",
               "session": sess["id"], "turn": turn,
               "n_turns": sess["turns"], "run": run_id,
               "seed": sess["seed"] + turn,
               "avatar": {"name": av.name, "kind": av.kind,
                          "body": av.body_path, "body_sha": av.body_sha,
                          "prompt_sha": psha},
               "persona": sess["persona"], "user_type": user["id"],
               "topic": topic["id"],
               "lang": {"asked": lang, "mix": mix,
                        "user": detect_lang(line),
                        "reply": detect_lang(reply["spoken"])},
               "history": [dict(h) for h in history],
               "user": line, "user_content": content,
               "reply_raw": reply["raw"], "spoken": reply["spoken"],
               "refused": bool(reply["refused"]),
               "stop_reason": reply.get("stop_reason"),
               "warnings": list(reply.get("warnings") or [])}
        unplayed = {}
        for w in rec["warnings"]:
            m = re.match(r"\s*\[(\w+)", w)
            if m and "dropped" in w:
                unplayed[m.group(1)] = unplayed.get(m.group(1), 0) + 1
        rec["unplayed"] = unplayed
        key = f"t{turn:02d}"
        if reply["refused"]:
            rec.update(markers=[], poses=[], reach=pz.reach_summary([]),
                       timing=None, traj=None)
        else:
            with av.lock:
                fields, arr = play(av, reply, sess["seed"] + turn)
            rec["warnings"] += fields.pop("play_warnings")
            rec.update(fields)
            rec["traj"] = {"file": f"traj/{sess['id']}.npz", "key": key,
                           "vars": av.traj_vars, "idle": True}
            for suf, v in arr.items():
                arrays[key + suf] = v
            history.append({"user": line, "user_content": content,
                            "reply": reply["raw"]})
        transcript.append(("user", line))
        if reply["spoken"]:
            transcript.append(("char", reply["spoken"]))
        cc, uc = cost(c_tok, prices), cost(u_tok, prices)
        rec["llm"] = {
            "character": {"backend": opts.backend,
                          "model": _model_of(char_backend),
                          "usage": reply.get("usage") or {}, "tokens": c_tok,
                          "usage_estimated": c_est,
                          "seconds": reply.get("model_seconds"),
                          "first_text_seconds":
                              reply.get("first_text_seconds"),
                          "attempts": c_att},
            "user": {"backend": opts.user_backend,
                     "model": _model_of(user_backend),
                     "usage": getattr(user_backend, "usage", None) or {},
                     "tokens": u_tok, "usage_estimated": u_est,
                     "seconds": round(u_sec, 3), "attempts": u_att,
                     "raw": u_text if u_text.strip() != line else None}}
        rec["cost_usd"] = round(cc + uc, 7)
        for t_ in (c_tok, u_tok):
            for k_ in ("in", "in_hit", "out"):
                tot[k_] += t_[k_]
        tot["cost_usd"] += cc + uc
        records.append(rec)
    if arrays:
        arrays["vars"] = json.dumps(av.traj_vars)
    return records, arrays, tot


# --------------------------------------------------------------- store

class Store:
    """The output folder: shards, trajectories, the session manifest and
    the avatar registry. Thread-safe; a session is written whole."""

    def __init__(self, out, shard_size=SHARD_SIZE):
        self.out, self.shard_size = out, int(shard_size)
        self.lock = threading.Lock()
        for d in ("shards", "traj"):
            os.makedirs(os.path.join(out, d), exist_ok=True)
        self.manifest = os.path.join(out, "sessions.jsonl")
        self.done, self.spent = set(), {"in": 0, "in_hit": 0, "out": 0,
                                        "cost_usd": 0.0}
        self.failed = set()
        if os.path.exists(self.manifest):
            with open(self.manifest, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        m = json.loads(line)
                    except ValueError:
                        continue          # a line cut by a crash
                    if m.get("status") == "ok":
                        self.done.add(m["session"])
                        self.failed.discard(m["session"])
                    else:
                        self.failed.add(m["session"])
                    for k in ("in", "in_hit", "out"):
                        self.spent[k] += int((m.get("tokens") or {}).get(k, 0))
                    self.spent["cost_usd"] += float(m.get("cost_usd") or 0)
        shards = sorted(f for f in os.listdir(os.path.join(out, "shards"))
                        if f.startswith("turns-") and f.endswith(".jsonl"))
        self.shard = int(shards[-1][6:11]) if shards else 0
        self.lines = 0
        if shards:
            with open(self._shard_path(), encoding="utf-8") as fh:
                self.lines = sum(1 for _ in fh)

    def _shard_path(self):
        return os.path.join(self.out, "shards", f"turns-{self.shard:05d}.jsonl")

    def write_session(self, sid, records, arrays, tot, run_id):
        with self.lock:
            if arrays:
                path = os.path.join(self.out, "traj", f"{sid}.npz")
                tmp = path + ".tmp.npz"
                np.savez_compressed(tmp, **{k: (np.array(v) if isinstance(
                    v, str) else v) for k, v in arrays.items()})
                os.replace(tmp, path)
            if self.lines >= self.shard_size:
                self.shard += 1
                self.lines = 0
            with open(self._shard_path(), "a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False,
                                        separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.lines += len(records)
            self._manifest({"session": sid, "status": "ok",
                            "turns": len(records),
                            "tokens": {k: tot[k] for k in
                                       ("in", "in_hit", "out")},
                            "cost_usd": round(tot["cost_usd"], 7)}, run_id)
            self.done.add(sid)

    def fail_session(self, sid, error, tot, run_id):
        with self.lock:
            self._manifest({"session": sid, "status": "failed",
                            "error": str(error)[:500],
                            "tokens": {k: tot.get(k, 0) for k in
                                       ("in", "in_hit", "out")},
                            "cost_usd": round(tot.get("cost_usd", 0.0), 7)},
                           run_id)
            self.failed.add(sid)

    def _manifest(self, m, run_id):
        m.update(run=run_id, at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        for k in ("in", "in_hit", "out"):
            self.spent[k] += int(m["tokens"][k])
        self.spent["cost_usd"] += float(m["cost_usd"])
        with open(self.manifest, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def register(self, avatars):
        """avatars.json: merged with what an earlier run wrote; a prompt
        whose bytes changed is kept under its new sha (old records keep
        theirs)."""
        path = os.path.join(self.out, "avatars.json")
        reg = {"format": AVATARS_FORMAT, "avatars": {}, "prompts": {}}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                reg = json.load(fh)
        changed = []
        for av in avatars.values():
            old = reg["avatars"].get(av.name)
            new = av.registry()
            if old and (old.get("body_sha") != new["body_sha"]
                        or old.get("prompts") != new["prompts"]):
                changed.append(av.name)
                new["earlier"] = old.get("earlier", []) + [
                    {k: old.get(k) for k in ("body_sha", "prompts")}]
            reg["avatars"][av.name] = new
            for r in av.runs.values():
                reg["prompts"][sha(r["system"])] = r["system"]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(reg, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return changed

    def log_run(self, entry):
        with open(os.path.join(self.out, "runs.jsonl"), "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def iter_records(out):
    """Every turn record under out/shards, deduped by id (the last written
    wins: a session rerun after a crash between its shard lines and its
    manifest line)."""
    d = os.path.join(out, "shards")
    seen = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else ():
        if not (f.startswith("turns-") and f.endswith(".jsonl")):
            continue
        with open(os.path.join(d, f), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                seen[r["id"]] = r
    return list(seen.values())


def load_traj(out, rec, which=""):
    """A record's trajectory ([frames, vars] float32), or None; which: ""
    (played), ".directed", ".variant", ".chars", ".replay"."""
    t = rec.get("traj")
    if not t:
        return None
    with np.load(os.path.join(out, t["file"])) as z:
        k = t["key"] + which
        if k not in z:
            return None
        v = z[k]
    if which == ".replay":
        return json.loads(str(v))
    return v.astype(np.float32) if v.dtype == np.float16 else v


# -------------------------------------------------------------- budget

class Budget:
    """Spent tokens and dollars (an earlier run's included): a new session
    starts only while both caps leave room for one more like the average
    so far (the estimate before any finished)."""

    def __init__(self, spent, cap_usd=None, cap_tokens=None, per_session=0.0,
                 per_session_tokens=0):
        self.lock = threading.Lock()
        self.cost = float(spent.get("cost_usd", 0.0))
        self.tokens = int(spent.get("in", 0)) + int(spent.get("out", 0))
        self.cap_usd, self.cap_tokens = cap_usd, cap_tokens
        self.est, self.est_tok = float(per_session), int(per_session_tokens)
        self.n, self.run_cost, self.run_tok = 0, 0.0, 0
        self.inflight = 0

    def add(self, tot):
        with self.lock:
            self.cost += tot["cost_usd"]
            t = tot["in"] + tot["out"]
            self.tokens += t
            self.run_cost += tot["cost_usd"]
            self.run_tok += t
            self.n += 1

    def per_session(self):
        if self.n:
            return self.run_cost / self.n, self.run_tok / self.n
        return self.est, self.est_tok

    def room(self) -> bool:
        c, t = self.per_session()
        with self.lock:
            k = self.inflight + 1
            if self.cap_usd is not None and self.cost + k * c > self.cap_usd:
                return False
            if self.cap_tokens is not None and \
                    self.tokens + k * t > self.cap_tokens:
                return False
        return True


# ------------------------------------------------------------ estimate

def estimate(avatars, sessions, opts, prices=PRICES) -> dict:
    """Tokens, dollars and hours for these sessions before any call, from
    the real system prompts and pose lines and typical line lengths
    (--est-reply-tokens, --est-user-tokens). Cached input: DeepSeek's
    context cache matches a repeated prefix (the system prompt, the
    history), so after a cell's first call most input is a hit; the
    upper bound prices everything as a miss."""
    R, U = opts.est_reply_tokens, opts.est_user_tokens
    SIM = est_tokens(USER_SYSTEM) + 120         # its prompt around the transcript
    tok = {"in": 0, "in_hit": 0, "out": 0}
    seen_cells = set()
    for s in sessions:
        av = avatars[s["avatar"]]
        sys_t = est_tokens(av.system(s["persona"]))
        pose_t = est_tokens(av.pose_now)
        cell = (s["avatar"], s["persona"])
        for t in range(s["turns"]):
            hist = t * (U + pose_t + R)
            tok["in"] += sys_t + hist + U + pose_t
            tok["in_hit"] += (sys_t if cell in seen_cells or t else 0) + hist
            seen_cells.add(cell)
            tok["out"] += R
            tok["in"] += SIM + t * (U + int(R * 0.6))
            tok["out"] += U
    hi = cost(dict(tok, in_hit=0), prices)
    lo = cost(tok, prices)
    turns = sum(s["turns"] for s in sessions)
    sec = turns * (opts.est_turn_seconds) / max(1, opts.concurrency)
    return {"sessions": len(sessions), "turns": turns, "tokens": tok,
            "cost_usd": round(lo, 4), "cost_usd_no_cache": round(hi, 4),
            "hours": round(sec / 3600, 2)}


def estimate_text(e, prices=PRICES) -> str:
    t = e["tokens"]
    return (f"dataset: estimate for {e['sessions']} sessions, {e['turns']} "
            f"turns: ~{t['in'] / 1e6:.2f}M input tokens ({t['in_hit'] / 1e6:.2f}M "
            f"cache hits), ~{t['out'] / 1e6:.2f}M output; ~${e['cost_usd']:.2f} "
            f"(${e['cost_usd_no_cache']:.2f} with no cache hits) at "
            f"${prices['in']}/{prices['in_hit']}/{prices['out']} per M "
            f"(in miss/hit/out: check the provider's current prices); "
            f"~{e['hours']:.1f} h at the assumed latency")


# ------------------------------------------------------------- generate

def default_backend_factory(opts):
    """(role, avatar) -> a fresh backend: one per session and role, since
    a backend keeps its last call's usage and stop reason on itself."""
    def make(role, av):
        if role == "character":
            a = copy.copy(av.a)
        else:
            ap = argparse.ArgumentParser()
            dr.add_args(ap)
            argv = ["--backend", opts.user_backend]
            if opts.user_model:
                argv += ["--model", opts.user_model]
            if opts.user_api_base:
                argv += ["--api-base", opts.user_api_base]
            a = ap.parse_args(argv)
            a.max_tokens = 256
        return dr.make_backend(a, log=_quiet)
    return make


def check_keys(opts):
    """The API keys the backends need, before anything starts."""
    miss = []
    for be in {opts.backend, opts.user_backend}:
        env = {"anthropic": "ANTHROPIC_API_KEY"}.get(be) or \
            (dr.OPENAI_COMPAT.get(be, (None, None))[1])
        if env and not os.environ.get(env):
            miss.append(f"{be} needs {env}")
    if miss:
        raise SystemExit("dataset: " + "; ".join(miss))


def generate(opts, backend_factory=None, log=print, argv=None) -> dict:
    """Generate (or resume) the sessions of opts into opts.out. Returns the
    run's totals."""
    cfg = load_config(opts.config)
    by_id = {p["id"]: p for p in cfg["personas"]}
    personas = opts.personas or [p["id"] for p in cfg["personas"]]
    for p in personas:
        if p not in by_id:
            raise SystemExit(f"no persona {p!r} in {opts.config or CONFIG} "
                             f"(have {', '.join(by_id)})")
    prices = {"in": opts.price_in, "in_hit": opts.price_in_hit,
              "out": opts.price_out}
    roots = [r for r in (opts.models or "").split(":") if r]
    specs = [avatar_spec(x, roots, opts.rigs, opts.events, opts.live2d_lite)
             for x in opts.avatars]
    sessions = plan_sessions([s["name"] for s in specs], personas, cfg,
                             opts.sessions, opts.turns, opts.seed)
    store = Store(opts.out, opts.shard_size)
    todo = [s for s in sessions if s["id"] not in store.done]
    if store.done:
        log(f"dataset: resuming {opts.out}: {len(store.done)} sessions done "
            f"(${store.spent['cost_usd']:.4f} spent), {len(todo)} to go")
    avatars = {}
    for s in specs:
        ptexts = {p: persona_text(cfg, by_id[p], s["name"]) for p in personas}
        avatars[s["name"]] = Avatar(s, opts, ptexts, log)
    changed = store.register(avatars)
    if changed:
        log("dataset: WARNING the body file or prompt changed since an "
            f"earlier run for {', '.join(changed)}: new records carry the new "
            "prompt_sha (avatars.json keeps both)")
    est = estimate(avatars, todo, opts, prices)
    log(estimate_text(est, prices))
    if opts.dry_run or not todo:
        return {"estimate": est, "sessions": 0, "cost_usd": 0.0}
    if backend_factory is None:
        check_keys(opts)
        backend_factory = default_backend_factory(opts)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    store.log_run({"run": run_id, "started": time.strftime(
        "%Y-%m-%dT%H:%M:%S"), "argv": argv, "git": git_state(),
        "prices": prices, "estimate": est,
        "opts": {k: v for k, v in vars(opts).items()
                 if isinstance(v, (str, int, float, bool, list, type(None)))}})
    budget = Budget(store.spent, opts.budget_usd, opts.budget_tokens,
                    est["cost_usd"] / max(1, len(todo)),
                    (est["tokens"]["in"] + est["tokens"]["out"])
                    / max(1, len(todo)))
    stop = threading.Event()
    state = {"fails": 0, "fatal": None, "ok": 0, "failed": 0,
             "estimated": 0, "reason": None}
    slock = threading.Lock()
    t_start = time.monotonic()

    def work(sess):
        av = avatars[sess["avatar"]]
        tot = {"in": 0, "in_hit": 0, "out": 0, "cost_usd": 0.0}
        try:
            cb, ub = backend_factory("character", av), \
                backend_factory("user", av)
            recs, arrs, tot = run_session(sess, av, cfg, opts, cb, ub, run_id,
                                          stop, log, prices)
        except InterruptedError:
            return
        except Fatal as exc:
            with slock:
                state["fatal"] = str(exc)
            stop.set()
            log(f"dataset: FATAL {exc}; stopping (resume with the same "
                "command)")
            return
        except Exception as exc:  # noqa: BLE001 - a session, not the run
            store.fail_session(sess["id"], f"{exc.__class__.__name__}: {exc}",
                               tot, run_id)
            with slock:
                state["fails"] += 1
                state["failed"] += 1
                if state["fails"] >= opts.max_consecutive_failures:
                    state["reason"] = (f"{state['fails']} sessions failed in "
                                       "a row")
                    stop.set()
            log(f"dataset: session {sess['id']} failed: "
                f"{exc.__class__.__name__}: {str(exc)[:300]}")
            return
        store.write_session(sess["id"], recs, arrs, tot, run_id)
        budget.add(tot)
        est_n = sum(r["llm"]["character"]["usage_estimated"]
                    + r["llm"]["user"]["usage_estimated"] for r in recs)
        with slock:
            state["fails"] = 0
            state["ok"] += 1
            first_est = state["ok"] == 1 and est_n
            state["estimated"] += est_n
            ok = state["ok"]
        if first_est:
            log("dataset: WARNING the backend reported no usage for "
                f"{est_n} calls of the first session: tokens and the budget "
                "come from character counts (usage_estimated)")
        miss = sum(r["reach"]["miss"] for r in recs)
        tgt = sum(r["reach"]["targets"] for r in recs)
        el = time.monotonic() - t_start
        left = len(todo) - ok
        c, _t = budget.per_session()
        log(f"dataset: {sess['id']} ok ({len(recs)} turns, "
            f"{tgt - miss}/{tgt} targets reached, ${tot['cost_usd']:.4f}); "
            f"{ok}/{len(todo)} done, ${budget.cost:.4f} spent, "
            f"~${c * left:.3f} and ~{el / ok * left / 3600:.1f} h to go")

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    pending = list(todo)
    running = set()
    try:
        with ThreadPoolExecutor(max_workers=max(1, opts.concurrency)) as ex:
            while (pending or running) and not (stop.is_set() and not running):
                while pending and len(running) < opts.concurrency \
                        and not stop.is_set():
                    if not budget.room():
                        state["reason"] = (f"budget: ${budget.cost:.4f} spent "
                                           f"of {opts.budget_usd}, "
                                           f"{budget.tokens} tokens of "
                                           f"{opts.budget_tokens}")
                        pending = []
                        break
                    with budget.lock:
                        budget.inflight += 1
                    fut = ex.submit(work, pending.pop(0))
                    running.add(fut)
                if not running:
                    break
                done, _ = wait(running, timeout=1.0,
                               return_when=FIRST_COMPLETED)
                for f in done:
                    running.discard(f)
                    with budget.lock:
                        budget.inflight -= 1
                    f.result()
    except KeyboardInterrupt:
        stop.set()
        state["reason"] = "interrupted"
        log("dataset: interrupted; sessions in flight are dropped (resume "
            "with the same command)")
    reason = state["fatal"] or state["reason"] or "done"
    out = {"run": run_id, "ok": state["ok"], "failed": state["failed"],
           "spent_usd": round(budget.cost, 6),
           "run_usd": round(budget.run_cost, 6), "run_tokens": budget.run_tok,
           "estimated_calls": state["estimated"], "stopped": reason,
           "hours": round((time.monotonic() - t_start) / 3600, 3)}
    store.log_run({"run": run_id, "ended": time.strftime(
        "%Y-%m-%dT%H:%M:%S"), **out})
    log(f"dataset: {reason}: {state['ok']} sessions written, "
        f"{state['failed']} failed; this run ${budget.run_cost:.4f}, "
        f"{budget.run_tok} tokens; all runs ${budget.cost:.4f} -> {opts.out}")
    if state["fatal"]:
        out["exit"] = 2
    return out


# ----------------------------------------------------------------- cli

def add_args(ap):
    ap.add_argument("--avatars", default="Hiyori,seed-san,lafei",
                    help="comma-separated: NAME (rigs/NAME.body.json; a VRM "
                         "body, or a Live2D rig with its model under "
                         "--models) or NAME:model=P,rig=P,body=P,clips=P")
    ap.add_argument("--personas", default=None,
                    help="comma-separated persona ids [all in --config]")
    ap.add_argument("--config", default=None,
                    help=f"persona / user / topic lists [{CONFIG}]")
    ap.add_argument("--sessions", type=int, default=20,
                    help="sessions in all, round-robin over avatar x persona")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--out", default="data/datagen/run")
    ap.add_argument("--backend", default="deepseek",
                    help="the character's backend (director backends)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--user-backend", default="deepseek")
    ap.add_argument("--user-model", default=None)
    ap.add_argument("--user-api-base", default=None)
    ap.add_argument("--tts", default="none",
                    help="none [default]: the rate model; auto / edge / "
                         "voicevox / siliconflow: real speech for word "
                         "timings (not saved)")
    ap.add_argument("--gestures", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="offer the Live2D rig's own motions next to poses "
                         "(the director's default)")
    ap.add_argument("--models", default="live2d-models:CubismWebSamples",
                    help="colon-separated folders holding .model3.json files")
    ap.add_argument("--rigs", default="rigs")
    ap.add_argument("--events", default=None,
                    help="event stats [ev.json when present, else default]")
    ap.add_argument("--live2d-lite", action="store_true",
                    help="a Live2D avatar without its model: body file only "
                         "(no gestures, no idle; flagged)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--tail", type=float, default=0.5,
                    help="seconds of trajectory after speech and motion")
    ap.add_argument("--traj-vars", default="offered",
                    choices=("offered", "all"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--budget-tokens", type=int, default=None)
    ap.add_argument("--price-in", type=float, default=PRICES["in"],
                    help="US$ per M input tokens, cache miss")
    ap.add_argument("--price-in-hit", type=float, default=PRICES["in_hit"])
    ap.add_argument("--price-out", type=float, default=PRICES["out"])
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=4.0,
                    help="seconds before the first retry (doubling)")
    ap.add_argument("--max-consecutive-failures", type=int, default=5)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--est-reply-tokens", type=int, default=160)
    ap.add_argument("--est-user-tokens", type=int, default=30)
    ap.add_argument("--est-turn-seconds", type=float, default=12.0,
                    help="assumed wall seconds per turn (both calls)")
    ap.add_argument("--dry-run", action="store_true",
                    help="prepare, print the estimate, stop")


def split_avatars(text) -> list[str]:
    """--avatars: commas part the avatars and also the key=value pairs of
    one ("x:model=P,rig=P,y"): a piece with "=" and no ":" before it
    belongs to the avatar before."""
    out = []
    for x in (p for p in text.split(",") if p):
        if out and "=" in x and ":" not in x.split("=", 1)[0] and ":" in out[-1]:
            out[-1] += "," + x
        else:
            out.append(x)
    return out


def parse_opts(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "estimate"):
        add_args(sub.add_parser(name))
    a = ap.parse_args(argv)
    a.avatars = split_avatars(a.avatars)
    a.personas = [x for x in a.personas.split(",") if x] if a.personas \
        else None
    if a.cmd == "estimate":
        a.dry_run = True
    return a


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = parse_opts(argv)
    out = generate(opts, argv=argv)
    return int(out.get("exit", 0))


if __name__ == "__main__":
    raise SystemExit(main())
