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
             or speech.speak (a real TTS: word / mora timings), moves.
             solve_moves (seeded by the turn's seed and each marker's
             place, as director.plan builds them) and pose.solve_poses
             (with the Core's exact check on Live2D when the model loads),
             a gesture under an arm move or pose dropped, director.realize
             + pose.tokens + moves.tokens into a
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
  quiet       true on a quiet-chat turn (beats2; absent otherwise): user is
              flow.pose.QUIET_CHAT, no simulator was called, topic is
              "quiet" and lang.asked / lang.user are the session's current
              language, as the "(reply in: ...)" line is
  reply_raw   the model's text, markers in it; spoken: what is said
  refused, stop_reason, warnings (parser and planning)
  markers     [{tag, arg, at (char offset in spoken), t (seconds; null for
              pause), text (as written), played: true | false | "dropped"
              (a gesture giving way to an arm pose or move) | "unsolved" (a
              pose that moved nothing, a move this body cannot do)}]
  unplayed    {tag: count} markers the parser dropped (e.g. tags of another
              mode, or markers newer than this pipeline)
  poses       pose.solve_poses records, rounded: t, start, dur, hold,
              release, targets (as written), resolved, params (as played),
              parts, variant_name, report {var: {target, got, error, unit,
              ok}}, exact / exact_ok (Live2D with the Core), ok, via, at
              (the marker's place in spoken: flow.dataexport --correct-misses
              ties a marker in reply_raw to its pose by order, `at` and
              targets, and writes a missed target as report[var].got)
  moves       flow.moves.solve_moves records, rounded: name, t, at, start,
              seconds, beats, tempo, amp, segments, solves, solve_seconds
  framing     beats prompt only: {start, home, end, switches}, the framing
              the turn started in (shown to the model as "(camera: ...)"
              after the pose line in user_content), the session's home, the
              one the reply ended in, and the [frame] markers that changed
              it. A session's home is bust or full per --bust-share
              (flow.framing.start_frame); every turn starts in it, a
              switch lasts until the end of its reply
  reach       pose.reach_summary {markers, targets, ok, miss, exact_ok,
              exact_n, solve_seconds}
  play_warnings  the planning lines among warnings (pose: / move:)
  timing      {speech_seconds, seconds, frames, fps, method: rate | the
              TTS's (edge words, voicevox moras ...), tts}
  traj        {file: "traj/<session>.npz", key: "tNN", vars: [canonical
              variable per column], idle: true, version: TRAJ_VERSION
              (absent: 1, before moves played); a replayed one also
              body_sha (the body it was streamed on), replayed (when)} or
              null (refused)
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
    k.replay      JSON string, version 2: {v, acts (timed, as the parser
                  and timing gave them: before any solve or drop), tags
                  (the parser's, pauses included), end (speech seconds),
                  seed, mouth (the voice's mouth token or null), method,
                  tts, tokens (directed tokens as streamed)} -- enough to
                  stream the turn again after a pose / moves / stream
                  change (`replay`). Version 1 (no "v"): {acts after the
                  gesture drops, tokens}.
    vars          JSON string: the variable list
  <out>/avatars.json   tsukuyomi.avatars/1: per avatar kind, body, body_sha,
                  pose_now, traj_vars, variants, prompts {persona: sha},
                  budget (flow.framing.budget: the motion budget measured
                  from the body: reach shares, face controls, tier, hand
                  moves a reply per framing, the bust window), landmarks;
                  "prompts" {sha: system prompt text} verbatim (the SFT
                  export uses these bytes)
  <out>/sessions.jsonl  one line per finished session: {session, status
                  ok | failed, turns, tokens, cost_usd, error, run, at}; a
                  session is written whole (shard lines, npz, then this
                  line), so --resume skips "ok" ones and reruns the rest;
                  readers dedupe turns by id (last wins)
  <out>/runs.jsonl  one line per run: argv, git head and dirty flag, prices,
                  start time; the end totals as a second line

Values are rounded (ROUND decimals) and trajectories float16: 3 turns of
~5 s at 30 fps on Seed-san were 17 KB of npz (both trajectories), the turn
records ~6 KB each (the history grows with the turn).

replay (`python -m flow.dataset replay OUT [--force] [--sessions a,b]`):
the records whose traj.version is older than TRAJ_VERSION (all with
--force) streamed again from their replay records, no model called; the
arrays to traj/v<TRAJ_VERSION>/<session>.npz (the old file kept), the
updated records (markers, poses, moves, reach, timing, warnings, traj) to
a new shard. A version 2 replay gives back the arrays it was made with.

Usage:
    python -m flow.dataset gen --avatars Hiyori,seed-san,lafei \\
        --personas cheerful,sleepy,tsundere --sessions 20 --turns 6 \\
        --out data/datagen/pilot
    python -m flow.dataset estimate ...       # the cost before starting
    python -m flow.dataset replay data/datagen/pilot
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
    from . import framing as fr
    from . import moves as mv
    from . import pose as pz
    from . import speech as sp
    from . import stream as fs
    from . import vrm as fv
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import body as fb
    from flow import director as dr
    from flow import framing as fr
    from flow import moves as mv
    from flow import pose as pz
    from flow import speech as sp
    from flow import stream as fs
    from flow import vrm as fv

FORMAT = "tsukuyomi.turns/1"
AVATARS_FORMAT = "tsukuyomi.avatars/1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "flow", "datagen.personas.json")
ROUND = 4
# The trajectories' version (a record's traj.version; 1 when absent): 2 =
# [move] markers played (flow.moves). `replay` brings older ones up to it.
TRAJ_VERSION = 2
# The .replay record's layout: 1 = {acts (after the gesture drops), tokens};
# 2 adds what a replay needs on its own (module docstring).
REPLAY_VERSION = 2
# Trajectory columns that are 0, not absent, when nothing writes them: a
# VRM's whole-body offset (flow.moves writes it only while a move plays).
ZERO = {"hips.x": 0.0, "hips.y": 0.0}
USER_MAX = 300          # characters of a simulated user line at most
SHARD_SIZE = 5000       # turn records per shard
# The character's system prompt (flow.pose.PROMPTS): "beats2" from r3 on,
# "beats" regenerates r2 and "v1" r1.
TEACHER_PROMPT = "beats2"
# Within-session language switches of the beats2 prompt (--lang-switch): the
# personas' own "switch" (0.1-0.15) carried an earlier turn's language over
# into the answer to a user who had moved on; r2 data keeps theirs.
LANG_SWITCH = 0.03
# Quiet-chat turns of the beats2 prompt (--quiet-share): the share of turns
# (never the first) on which the simulated user sends exactly what flow.live
# sends after a while without a message, flow.pose.QUIET_CHAT, and the
# character keeps talking by itself. Drawn from its own generator, so
# nothing else of a session moves.
QUIET_SHARE = 0.10
# Prices in US$ per million tokens: DeepSeek's deepseek-chat list prices
# as last checked (2025-09: input 0.28 on a cache miss, 0.028 on a hit,
# output 0.42). Check https://api-docs.deepseek.com/quick_start/pricing
# before trusting an estimate; --price-* override them.
PRICES = {"in": 0.28, "in_hit": 0.028, "out": 0.42}
LANG_NAMES = pz.LANG_NAMES
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
        raise SystemExit(
            f"avatar {name}: no body file {body} (it is not in the repo; "
            f"copy it from the machine that has it, or build it: python -m "
            f"flow.body --model-json <{name}.model3.json> --rig "
            f"rigs/{name}.npz --measure, or pass {name}:body=PATH)")
    try:
        with open(body, encoding="utf-8") as fh:
            kind = json.load(fh).get("kind", "live2d")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"avatar {name}: cannot read body file {body}: "
                         f"{exc}")
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
    argv += ["--body-prompt", getattr(opts, "teacher_prompt", TEACHER_PROMPT)]
    a = ap.parse_args(argv)
    a.micro = False         # no speech-driven micro-expressions in the data
    a.head_drift = False    # nor drift, pose ease or camera: the data is the
    a.pose_ease = False     # trajectory as the model wrote it
    a.head_carry = False    # (no carried angles, no resting expression)
    a.baseline_scale = 0.0
    a.camera_follow = False
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
                # The director's body-mode prompt without the rig: no
                # gestures, the motion primitives this body can do.
                bm = self.bm
                mlines, mnames = dr.move_menu(a, bm)
                run = {"system": pz.system_prompt(bm, ptext, False, "",
                                                  mlines, a.body_prompt),
                       "system_for": lambda k, bm=bm, ptext=ptext,
                       mlines=mlines: pz.system_prompt(
                           bm, ptext, False, "", mlines, a.body_prompt, k),
                       "labels": {}, "body": bm,
                       "parser": lambda bm=bm, m=bool(mnames),
                       f=pz.beats_like(a.body_prompt):
                           pz.parser_for(bm, moves=m, frames=f)}
            self.runs[pid] = run
            # beats2: one system prompt per dense example variant (the
            # session picks one, flow.pose.variant_for); the key of
            # variant 0 is the persona's, as for the older prompts.
            if a.body_prompt == "beats2" and run.get("system_for"):
                for k in range(1, pz.variant_count(self.bm)):
                    self.runs[self.key(pid, k)] = {
                        **run, "system": run["system_for"](k)}
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

    @staticmethod
    def key(persona, variant=0):
        return persona if not variant else f"{persona}#{variant}"

    def system(self, persona, variant=0):
        return self.runs[self.key(persona, variant)]["system"]

    def parser(self, persona, variant=0):
        return self.runs[self.key(persona, variant)]["parser"]()

    def labels(self, persona, variant=0):
        return self.runs[self.key(persona, variant)].get("labels") or {}

    def registry(self) -> dict:
        return {"kind": self.kind, "body": self.body_path,
                "body_sha": self.body_sha, "pose_now": self.pose_now,
                "traj_vars": self.traj_vars, "variants": self.variant_names,
                "fps": float(self.opts.fps),
                "gestures": sorted(self.labels(next(iter(self.runs))))
                if self.runs else [],
                "spec": {k: v for k, v in self.spec.items()},
                "prompts": {p: sha(r["system"]) for p, r in self.runs.items()},
                "budget": fr.budget(self.bm),
                "landmarks": {k: v for k, v in self.bm.landmarks.items()}}


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
    if getattr(bm, "kind", None) == "vrm":
        # The VRM's mouth is its visemes (the voice's vowel shares), which
        # are parameters, not canonical features: taken as played.
        out += [p for p in (f"viseme.{v}" for v in "aiueo")
                if p in bm.params]
        # The sway and bob of the whole avatar (flow.moves.HIPS), 0 when no
        # move plays.
        out += list(mv.HIPS)
    return out


# --------------------------------------------------------------- play

def _quiet(_s):
    pass


def _frames(av, acts, poses, mvs, seed, n, idle, mouth=None):
    """Run the avatar's stream n frames with this turn's directed tokens:
    ([n, vars] canonical, [n] variant index, directed tokens)."""
    bm, fps = av.bm, float(av.opts.fps)
    cols = av.traj_vars
    vidx = {name: k for k, name in enumerate(av.variant_names)}
    out = np.full((n, len(cols)), np.nan, np.float32)
    var = np.zeros(n, np.int8)
    if av.kind == "live2d-lite":
        toks = mv.tokens(mvs) + pz.tokens(poses, {}, av.arms)
        rest = bm.default_values()
        rest_parts = dict(bm.blob.get("rest_parts", {}))
        lay = pz.layer_for(toks, fps, rest, rest_parts)
        mlay = mv.layer_for(toks, fps, rest, rest_parts)
        for f in range(n):
            over, parts = dict(rest), dict(rest_parts)
            still = ()
            if mlay is not None and mlay.busy():
                over, parts = mlay.apply(f, over, parts)
                still = mlay.holding
            if lay is not None:
                over, parts = lay.apply(f, over, parts, still)
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
    if mvs:
        directed += mv.tokens(mvs)
    stream.direct(copy.deepcopy(directed))
    if mouth is not None:
        stream.add_mouth(0, mouth, len(mouth["open"]))
    for f in range(n):
        over, parts = stream.frame()
        st = bm.perceive(over, parts=parts) if av.kind == "live2d" \
            else bm.perceive(over)
        out[f] = [st[c] if c in st else over.get(c, ZERO.get(c, np.nan))
                  for c in cols]
        var[f] = vidx.get(st.get("arm.variant"), 0)
    return out, var, directed


def timed(av, reply):
    """A reply's timing on the avatar: {acts (every non-pause marker with
    its time t, rounded as the replay record keeps it), spans (each spoken
    character's start, end), end (speech seconds), mouth (the voice's
    mouth token or None), method, tts}."""
    opts, a = av.opts, av.a
    spoken, tags = reply["spoken"], reply["tags"]
    fps = float(opts.fps)
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
    # Rounded before anything is solved: a replay from the stored acts
    # must see the same times (ceil(t * fps) flips on the 5th decimal).
    return {"acts": [_r(dict(x)) for x in acts], "spans": spans,
            "end": round(float(end), ROUND), "mouth": mouth, "method": method,
            "tts": tts_info}


def play(av, reply, seed):
    """A reply on the avatar, not rendered: (record fields, arrays). Must
    run under av.lock."""
    return play_timed(av, reply["tags"], timed(av, reply), seed)


def play_timed(av, tags, tm, seed):
    """play() from a timed reply (timed(), or a replay record): moves built
    on the body (flow.moves, seeded by the turn's seed and where each
    marker is), poses solved, the gestures they clash with dropped, as
    director.plan does; both trajectories streamed."""
    opts = av.opts
    fps = float(opts.fps)
    warnings = []

    def log(s):
        if "WARNING" in s or "dropped" in s or "failed" in s:
            warnings.append(s)
    pre = copy.deepcopy(tm["acts"])
    acts = [dict(x) for x in tm["acts"]]
    end, spans, mouth = tm["end"], tm["spans"], tm.get("mouth")
    orig = list(acts)
    mvs, moved, gone = [], {}, []
    if any(x.get("tag") == mv.MOVE for x in acts):
        # Built first, before any act is dropped (a move's seed hashes its
        # index in acts), as director.plan builds them.
        mvs = mv.solve_moves(av.bm, acts, log=log, seed=seed)
        moved = {id(acts[m["act"]]) for m in mvs}
        gone += mv.drop_clashing_gestures(acts, mvs, av.motions, log)
    poses = []
    if any(x.get("tag") == pz.POSE for x in acts):
        poses = pz.solve_poses(av.bm, acts, core=av.core, log=log)
        for rec in poses:
            rec["arm_params"] = sorted(set(rec["params"]) & av.arms)
        gone += pz.drop_clashing_gestures(acts, poses, av.motions, log)
    seconds = dr.length(acts, end, av.motions, opts.tail)
    if poses:
        seconds = max(seconds, pz.end_time(poses) + opts.tail)
    if mvs:
        seconds = max(seconds, mv.end_time(mvs) + opts.tail)
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
        elif x["tag"] == mv.MOVE:
            # Not built: a primitive this body cannot do (logged).
            played = True if id(x) in moved else "unsolved"
        else:
            played = x["tag"] in ("blink", "gesture", pz.FRAME)
        markers.append({"tag": x["tag"], "arg": x["arg"], "at": x["at"],
                        "t": x["t"], "text": pz.tag_text(x),
                        "played": played})
    played_traj, var, directed = _frames(av, acts, poses, mvs, seed, n,
                                         True, mouth)
    if av.kind == "live2d-lite":
        directed_traj = played_traj
    else:
        directed_traj, _v, _d = _frames(av, acts, poses, mvs, seed, n,
                                        False, mouth)
    chars = np.asarray(spans, np.float32).reshape(-1, 2)
    keep = ("t", "start", "dur", "hold", "release", "targets", "resolved",
            "params", "parts", "variant_name", "report", "exact", "exact_ok",
            "ok", "via", "hints", "at")
    mkeep = ("name", "t", "at", "start", "seconds", "beats", "tempo", "amp",
             "segments", "solves", "solve_seconds")
    fields = {
        "markers": _r(markers),
        "poses": [_r({k: p[k] for k in keep if k in p}) for p in poses],
        "moves": [_r({k: m[k] for k in mkeep if k in m}) for m in mvs],
        "reach": pz.reach_summary(poses),
        "timing": {"speech_seconds": round(float(end), 3),
                   "seconds": round(float(seconds), 3), "frames": n,
                   "fps": fps, "method": tm.get("method", "rate"),
                   "tts": tm.get("tts")},
        "play_warnings": warnings,
    }
    replay = {"v": REPLAY_VERSION, "acts": pre,
              "tags": [{k: t.get(k) for k in ("tag", "arg", "at")}
                       for t in tags],
              "end": float(end), "seed": int(seed), "mouth": mouth,
              "method": tm.get("method", "rate"), "tts": tm.get("tts"),
              "tokens": directed}
    arrays = {"": played_traj.astype(np.float16),
              ".directed": directed_traj.astype(np.float16),
              ".variant": var, ".chars": chars.astype(np.float16),
              ".replay": json.dumps(_r(replay), ensure_ascii=False)}
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
    """The language of a line (flow.pose.detect_lang): the record's
    lang.user / lang.reply and, under beats2, the user turn's
    "(reply in: ...)" line."""
    return pz.detect_lang(text)


def switch_prob(persona, opts) -> float:
    """The chance a turn's language is drawn afresh (not the session's
    first): --lang-switch when given, else LANG_SWITCH for the beats2
    prompt and the persona's own "switch" for the older ones (r1 / r2
    regenerate as they were)."""
    v = getattr(opts, "lang_switch", None)
    if v is None and getattr(opts, "teacher_prompt",
                             TEACHER_PROMPT) == "beats2":
        v = LANG_SWITCH
    return float(persona.get("switch", 0.15) if v is None else v)


def quiet_turn(sess, turn, opts) -> bool:
    """Whether turn `turn` of the session is a quiet-chat turn: beats2 only,
    never the first, with chance --quiet-share, from the session's seed and
    the turn alone (a resumed session decides the same)."""
    share = getattr(opts, "quiet_share", None)
    share = QUIET_SHARE if share is None else float(share)
    if (getattr(opts, "teacher_prompt", TEACHER_PROMPT) != "beats2"
            or turn <= 1 or share <= 0):
        return False
    r = random.Random(zlib.crc32(f"{sess['seed']}:{turn}:quiet".encode()))
    return r.random() < share


def collect(backend, system, messages):
    return "".join(backend.stream(system, messages))


# ------------------------------------------------------------ sessions

def plan_sessions(avatars, personas, cfg, n, turns, seed=0,
                  bust_share=None) -> list[dict]:
    """n sessions in all, spread round-robin over avatar x persona (a
    persona with an "avatars" list only on those). Ids are stable for a
    cell index, so a longer run resumes a shorter one. bust_share (the
    beats prompt): each session also gets the "frame" it starts in
    (flow.framing.start_frame: bust with that probability, from the run
    seed and the id alone)."""
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
        if bust_share is not None:
            out[-1]["frame"] = fr.start_frame(seed, sid, bust_share)
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
                stop=None, log=print, prices=PRICES, tot=None):
    """One session: (records, {npz key: array}, totals). Raises when a call
    fails past its retries (the session is then rerun on resume); `tot`
    (a dict the caller passes) holds what was spent up to then, call by
    call, so a failed session's spend still counts."""
    rng = random.Random(sess["seed"])
    persona = next(p for p in cfg["personas"] if p["id"] == sess["persona"])
    ptext = persona_text(cfg, persona, sess["avatar"])
    user = rng.choice(cfg["users"])
    char_name = (cfg.get("names") or {}).get(sess["avatar"], sess["avatar"])
    langs = persona.get("languages") or {"en": 1.0}
    lang0 = _pick(rng, langs)
    usys = USER_SYSTEM.format(user=user["style"], character=ptext)
    prompt = getattr(opts, "teacher_prompt", TEACHER_PROMPT)
    # beats2: the dense example this session's prompt shows (from its seed).
    variant = pz.variant_for(sess["seed"], av.bm) if prompt == "beats2" \
        else 0
    system = av.system(sess["persona"], variant)
    psha = sha(system)
    history, transcript, records, arrays = [], [], [], {}
    if tot is None:
        tot = {}
    for k_ in ("in", "in_hit", "out"):
        tot.setdefault(k_, 0)
    tot.setdefault("cost_usd", 0.0)

    def spend(t):
        for k_ in ("in", "in_hit", "out"):
            tot[k_] += t[k_]
        c = cost(t, prices)
        tot["cost_usd"] += c
        return c
    # A topic with "prompts" is only drawn under those teacher prompts (the
    # look and wave requests of r3): v1 and beats see the same list, and
    # draw the same topics, as before.
    topics = [t for t in cfg["topics"]
              if prompt in t.get("prompts", (prompt,))]
    openers = [t for t in topics if t.get("opener")]
    # The framing the character is in (flow.framing), shown with each line
    # under the beats prompt and moved by its [frame] markers; a body whose
    # framing cannot switch (a VRM) is always full. None: v1, not shown.
    # The session's home framing is the one every turn starts in: a [frame]
    # switch lasts until the end of its reply.
    framing = home = None
    if pz.beats_like(prompt):
        framing = home = sess.get("frame", "bust") if fr.can_frame(av.bm) \
            else "full"
    last_line, cur_lang = None, lang0     # the last real line, its language
    for turn in range(1, sess["turns"] + 1):
        if stop is not None and stop.is_set():
            raise InterruptedError("stopped")
        if turn == 1 and openers and rng.random() < 0.7:
            topic = rng.choice(openers)
        else:
            topic = rng.choices(topics,
                                [float(t.get("weight", 1)) for t in
                                 topics])[0]
        lang = lang0 if rng.random() > switch_prob(persona, opts) \
            else _pick(rng, langs)
        others = [k for k in langs if k != lang]
        mix = rng.choice(others) if others and rng.random() < float(
            persona.get("mix", 0.1)) else None
        umsg = [{"role": "user", "content": user_prompt(
            cfg, ptext, user, topic, lang, mix, transcript, char_name)}]
        line, u_att, u_text, u_sec = "", 0, "", 0.0
        u_tok, u_est, uc = {"in": 0, "in_hit": 0, "out": 0}, False, 0.0
        quiet = last_line is not None and quiet_turn(sess, turn, opts)
        if quiet:
            # Chat is quiet: the line flow.live sends, no simulator, the
            # session's current language.
            topic = {"id": "quiet"}
            lang, mix = cur_lang, None
            line = pz.quiet_user(last_line)[0]
        for _ in range(0 if quiet else 3):
            t0 = time.monotonic()
            u_text, k = with_retries(lambda: collect(user_backend, usys, umsg),
                                     opts.retries, opts.backoff, stop, log,
                                     "user simulator")
            u_sec += time.monotonic() - t0
            u_att += k
            t1, e1 = call_tokens(getattr(user_backend, "usage", None), usys,
                                 umsg, u_text)
            uc += spend(t1)
            u_tok = {k_: u_tok[k_] + t1[k_] for k_ in u_tok}
            u_est = u_est or e1
            line = clean_user_line(u_text)
            if line:
                break
        if not line:
            line = rng.choice(topic.get("examples") or ["Hi!"])
        content = line + "\n" + av.pose_now
        if framing is not None:
            framing = home
            content += pz.turn_tail(av.bm, framing, line, prompt,
                                    lang_text=pz.quiet_user(last_line)[1]
                                    if quiet else None)
        start_frame = framing
        msgs = []
        for h in history:
            msgs.append({"role": "user", "content": h["user_content"]})
            msgs.append({"role": "assistant", "content": h["reply"]})
        msgs.append(pz.user_message(content, opts.backend, None))
        reply, c_att = with_retries(
            lambda: dr.ask(char_backend, system, msgs,
                           av.labels(sess["persona"], variant), log=_quiet,
                           parser=av.parser(sess["persona"], variant)),
            opts.retries, opts.backoff, stop, log, "character")
        c_tok, c_est = call_tokens(reply.get("usage"), system, msgs,
                                   reply["raw"])
        cc = spend(c_tok)
        rec = {"format": FORMAT, "id": f"{sess['id']}/t{turn:02d}",
               "session": sess["id"], "turn": turn,
               "n_turns": sess["turns"], "run": run_id,
               "seed": sess["seed"] + turn,
               "avatar": {"name": av.name, "kind": av.kind,
                          "body": av.body_path, "body_sha": av.body_sha,
                          "prompt_sha": psha,
                          **({"variant": variant} if prompt == "beats2"
                             else {})},
               "persona": sess["persona"], "user_type": user["id"],
               "topic": topic["id"],
               "lang": {"asked": lang, "mix": mix,
                        "user": detect_lang(last_line if quiet else line),
                        "reply": detect_lang(reply["spoken"])},
               "history": [dict(h) for h in history],
               "user": line, "user_content": content,
               **({"quiet": True} if quiet else {}),
               "reply_raw": reply["raw"], "spoken": reply["spoken"],
               "refused": bool(reply["refused"]),
               "stop_reason": reply.get("stop_reason"),
               "warnings": list(reply.get("warnings") or [])}
        if framing is not None:
            end = start_frame if reply["refused"] else fr.last_frame(
                reply["tags"], start_frame)
            rec["framing"] = {"start": start_frame, "home": home, "end": end,
                              "switches": fr.switches(reply["tags"],
                                                      start_frame)}
        unplayed = {}
        for w in rec["warnings"]:
            m = re.match(r"\s*\[(\w+)", w)
            if m and "dropped" in w:
                unplayed[m.group(1)] = unplayed.get(m.group(1), 0) + 1
        rec["unplayed"] = unplayed
        key = f"t{turn:02d}"
        if reply["refused"]:
            rec.update(markers=[], poses=[], moves=[],
                       reach=pz.reach_summary([]), timing=None, traj=None,
                       play_warnings=[])
        else:
            with av.lock:
                fields, arr = play(av, reply, sess["seed"] + turn)
            rec["warnings"] += fields["play_warnings"]
            rec.update(fields)
            rec["traj"] = {"file": f"traj/{sess['id']}.npz", "key": key,
                           "vars": av.traj_vars, "idle": True,
                           "version": TRAJ_VERSION}
            for suf, v in arr.items():
                arrays[key + suf] = v
            history.append({"user": line, "user_content": content,
                            "reply": reply["raw"]})
        if quiet:
            pass                    # nobody wrote: no user line to show
        else:
            transcript.append(("user", line))
            last_line, cur_lang = line, lang
        if reply["spoken"]:
            transcript.append(("char", reply["spoken"]))
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

    def write_records(self, traj_file, records, arrays):
        """Arrays (traj_file under out) then records, as write_session does,
        with no manifest line: flow.dataset replay."""
        with self.lock:
            path = os.path.join(self.out, traj_file)
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **{k: (np.array(v) if isinstance(
                v, str) else v) for k, v in arrays.items()})
            os.replace(tmp, path)
            with open(self._shard_path(), "a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False,
                                        separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.lines += len(records)

    def fail_session(self, sid, error, tot, run_id, status="failed"):
        """A session not written (failed past its retries, or stopped
        mid-way: status "interrupted"), with what it spent; rerun on
        resume."""
        with self.lock:
            self._manifest({"session": sid, "status": status,
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


# -------------------------------------------------------------- replay

def run_opts(out):
    """The options of the latest run into out (runs.jsonl), over gen's
    defaults: what the avatars were prepared with."""
    opts = parse_opts(["gen"])
    path = os.path.join(out, "runs.jsonl")
    last = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("opts"):
                    last = e["opts"]
    # Runs from before the option existed taught with "beats": the default
    # moved on to beats2 since.
    opts.teacher_prompt = "beats"
    for k, v in last.items():
        if hasattr(opts, k):
            setattr(opts, k, v)
    return opts


def replay_timed(rec, rep):
    """(tags, timed) for play_timed from a record and its .replay. Version
    2 holds them as played. Version 1 (records before REPLAY_VERSION 2)
    kept the acts after the gesture drops and no tags or mouth: the tags
    come from the markers, a dropped gesture's act from its marker (the
    drops are decided again), the speech end from timing; a real TTS's
    mouth is gone (None: the stream's mouth stays at rest)."""
    chars = None
    tags = rep.get("tags")
    if rep.get("v", 1) >= 2:
        return tags, {"acts": rep["acts"], "end": float(rep["end"]),
                      "mouth": rep.get("mouth"), "method": rep.get("method"),
                      "tts": rep.get("tts"), "spans": chars}
    keep = ("t", "tag", "arg", "at")
    acts = [{k: x[k] for k in keep if k in x} for x in rep["acts"]]
    for m in rec.get("markers") or ():
        if m.get("played") == "dropped" and m.get("t") is not None:
            acts.append({"t": m["t"], "tag": m["tag"], "arg": m["arg"],
                         "at": m["at"]})
    acts.sort(key=lambda x: (float(x["t"]), int(x.get("at") or 0)))
    tags = [{"tag": m["tag"], "arg": m["arg"], "at": m["at"]}
            for m in rec.get("markers") or ()]
    tm = rec.get("timing") or {}
    return tags, {"acts": acts, "end": float(tm.get("speech_seconds") or 0),
                  "mouth": None, "method": tm.get("method", "rate"),
                  "tts": tm.get("tts"), "spans": chars}


def replay(out, force=False, sessions=None, log=print) -> dict:
    """Re-stream every record whose trajectory is older than TRAJ_VERSION
    (all with force) from its .replay record: the moves built and the poses
    solved again with the current code and body, both trajectories
    streamed again; no model is called. Per session the new arrays go to
    traj/v<TRAJ_VERSION>/<session>.npz (the old file is kept) and the
    updated records to a new shard (readers keep the last record of an
    id), traj.version set. Resumable: a session written is not redone."""
    reg_path = os.path.join(out, "avatars.json")
    with open(reg_path, encoding="utf-8") as fh:
        reg = json.load(fh)
    recs = iter_records(out)
    by_s = {}
    for r in recs:
        if r.get("traj") and (force or int(r["traj"].get("version", 1))
                              < TRAJ_VERSION):
            by_s.setdefault(r["session"], []).append(r)
    if sessions:
        by_s = {k: v for k, v in by_s.items() if k in set(sessions)}
    if not by_s:
        log(f"dataset: replay: nothing older than trajectory version "
            f"{TRAJ_VERSION} under {out}")
        return {"sessions": 0, "turns": 0}
    opts = run_opts(out)
    avatars = {}
    store = Store(out)
    store.shard += 1            # a shard of its own: these lines win
    store.lines = 0
    vdir = f"traj/v{TRAJ_VERSION}"
    os.makedirs(os.path.join(out, vdir), exist_ok=True)
    n_turns, moved = 0, 0
    t0 = time.monotonic()
    for sid in sorted(by_s):
        rs = sorted(by_s[sid], key=lambda r: r["turn"])
        name = rs[0]["avatar"]["name"]
        if name not in avatars:
            entry = reg["avatars"].get(name)
            if entry is None:
                raise SystemExit(f"replay: {name} is not in {reg_path}")
            spec = dict(entry["spec"])
            avatars[name] = Avatar(spec, opts, {"replay": None}, log)
        av = avatars[name]
        arrays = {}
        new = []
        for r in rs:
            rep = load_traj(out, r, ".replay")
            if rep is None:
                log(f"dataset: replay: {r['id']} has no replay record; kept")
                continue
            if r["avatar"].get("body_sha") != av.body_sha:
                log(f"dataset: replay: {r['id']}: the body file changed "
                    "since it was generated; replayed on the current one")
            tags, tm = replay_timed(r, rep)
            tm["spans"] = load_traj(out, r, ".chars")
            if tm["spans"] is None:
                tm["spans"] = np.zeros((0, 2), np.float32)
            with av.lock:
                fields, arr = play_timed(av, tags, tm, int(r["seed"]))
            r = copy.deepcopy(r)
            old = r.get("play_warnings")
            if old is None:
                # Version 1 kept them mixed in: the play's lines name
                # their layer.
                said = [w for w in r.get("warnings") or ()
                        if not w.startswith(("pose:", "move:"))]
            else:
                said = [w for w in r.get("warnings") or () if w not in old]
            timing = dict(fields.pop("timing"))
            timing["speech_seconds"] = (r.get("timing") or {}).get(
                "speech_seconds", timing["speech_seconds"])
            r.update(fields)
            r["timing"] = timing
            r["warnings"] = said + fields["play_warnings"]
            key = r["traj"]["key"]
            r["traj"] = {"file": f"{vdir}/{sid}.npz", "key": key,
                         "vars": av.traj_vars, "idle": True,
                         "version": TRAJ_VERSION, "body_sha": av.body_sha,
                         "replayed": time.strftime("%Y-%m-%dT%H:%M:%S")}
            for suf, v in arr.items():
                arrays[key + suf] = v
            new.append(r)
            moved += sum(m["tag"] == mv.MOVE and m["played"] is True
                         for m in r["markers"])
        if not new:
            continue
        arrays["vars"] = json.dumps(av.traj_vars)
        store.write_records(f"{vdir}/{sid}.npz", new, arrays)
        n_turns += len(new)
    log(f"dataset: replay: {n_turns} turns in {len(by_s)} sessions "
        f"re-streamed ({moved} moves played) in "
        f"{time.monotonic() - t0:.1f}s -> {vdir}/, shard "
        f"{os.path.basename(store._shard_path())}")
    return {"sessions": len(by_s), "turns": n_turns, "moves": moved}


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

    def add(self, tot, finished=True):
        """A session's spend; finished=False (failed, interrupted) counts
        toward the caps but not the per-session average."""
        with self.lock:
            if not finished:
                self.cost += tot.get("cost_usd", 0.0)
                self.tokens += tot.get("in", 0) + tot.get("out", 0)
                self.run_cost += tot.get("cost_usd", 0.0)
                self.run_tok += tot.get("in", 0) + tot.get("out", 0)
                return
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
                             opts.sessions, opts.turns, opts.seed,
                             getattr(opts, "bust_share", fr.BUST_SHARE)
                             if pz.beats_like(getattr(
                                 opts, "teacher_prompt", TEACHER_PROMPT))
                             else None)
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
                                          stop, log, prices, tot)
        except InterruptedError:
            if tot["in"] or tot["out"]:
                store.fail_session(sess["id"], "interrupted", tot, run_id,
                                   "interrupted")
                budget.add(tot, finished=False)
            return
        except Fatal as exc:
            if tot["in"] or tot["out"]:
                store.fail_session(sess["id"], f"fatal: {exc}", tot, run_id)
                budget.add(tot, finished=False)
            with slock:
                state["fatal"] = str(exc)
            stop.set()
            log(f"dataset: FATAL {exc}; stopping (resume with the same "
                "command)")
            return
        except Exception as exc:  # noqa: BLE001 - a session, not the run
            store.fail_session(sess["id"], f"{exc.__class__.__name__}: {exc}",
                               tot, run_id)
            budget.add(tot, finished=False)
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
    ap.add_argument("--teacher-prompt", default=TEACHER_PROMPT,
                    choices=pz.PROMPTS,
                    help="the character's system prompt: beats2 [default]: "
                         "the r3 prompt (beats with six dense examples, one "
                         "per session; [frame full] before whole-body moves; "
                         "a \"(reply in: ...)\" line on every user turn; "
                         "fewer language switches); beats: 1-3 expression "
                         "beats a sentence (face, with a small head move in "
                         "the same marker; a hand at the moments that call "
                         "for one; the camera line and [frame] markers), the "
                         "prompt of r2, to regenerate it byte for byte; v1: "
                         "the prompt of the first run (r1)")
    ap.add_argument("--lang-switch", type=float, default=None, metavar="P",
                    help="the chance a turn's language is drawn afresh "
                         "instead of the session's [beats2: "
                         f"{LANG_SWITCH:g}; older prompts: each persona's "
                         "own \"switch\" in --config, 0.1-0.15]")
    ap.add_argument("--quiet-share", type=float, default=None, metavar="P",
                    help="beats2: the share of turns (never the first) on "
                         "which the simulated user sends the quiet-chat "
                         "line flow.live sends after a while without a "
                         "message (flow.pose.QUIET_CHAT) and the character "
                         f"keeps talking [{QUIET_SHARE:g}]; 0 is none")
    ap.add_argument("--bust-share", type=float, default=fr.BUST_SHARE,
                    metavar="P",
                    help="beats prompt: the share of sessions that start in "
                         "the bust framing (the rest start full; the "
                         "character's [frame] markers move it from there) "
                         f"[{fr.BUST_SHARE:g}]; a VRM is always full")
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
    if argv[:1] == ["replay"]:
        ap = argparse.ArgumentParser(
            prog="flow.dataset replay",
            description="re-stream old records' trajectories from their "
                        "replay records (no model calls)")
        ap.add_argument("out", help="a flow.dataset --out")
        ap.add_argument("--force", action="store_true",
                        help="every record, not only those older than "
                             f"trajectory version {TRAJ_VERSION}")
        ap.add_argument("--sessions", default=None,
                        help="comma-separated session ids only")
        a = ap.parse_args(argv[1:])
        replay(a.out, a.force, [x for x in (a.sessions or "").split(",")
                                if x] or None)
        return 0
    opts = parse_opts(argv)
    out = generate(opts, argv=argv)
    return int(out.get("exit", 0))


if __name__ == "__main__":
    raise SystemExit(main())
