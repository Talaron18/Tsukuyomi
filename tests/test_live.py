"""
flow.stream and flow.live: the live director. No GL, no network: the
language model, the TTS and the renderer are faked.

The stream must draw what the offline runtime draws for the same tokens,
frame for frame; the idle must run on across window seams and directed
tokens must land in its future without a jump; sentences must be cut from
streamed chunks however they are split; each sentence's audio, mouth and
tokens must sit at one server time; the voice must follow the mood; and
the web server must serve the page, take a line and stream frames and
audio.
"""

from __future__ import annotations

import json
import os
import shlex

import numpy as np
import pytest

from flow import director as dr
from flow import events as ev
from flow import runtime as rt
from flow import stream as fs
from tests.test_speech import two_phrases, wav_bytes

FPS = 30.0


def _plan(tmp_path, extra=()):
    """A director --tts file run on the test rig with a mouth: (model, rig,
    tokens.json blob, plan.npz from a --tokens-in --dump-only replay)."""
    from tests.test_speech import _mouth_rig
    model, rig = _mouth_rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text("Hello there. [nod] How are you? [gesture:m]",
                     encoding="utf-8")
    audio = tmp_path / "said.wav"
    audio.write_bytes(wav_bytes(two_phrases()))
    out = tmp_path / "run"
    assert dr.main(["--model-json", str(model), "--rig", str(rig),
                    "--backend", "file", "--reply-file", str(reply),
                    "--out", str(out), "--probe-dir", str(tmp_path),
                    "--seed", "3", "--tts", "file", "--tts-file", str(audio),
                    "--tail", "6", *extra]) == 0
    blob = json.loads((out / "tokens.json").read_text())
    plan = np.load(out / "plan.npz", allow_pickle=True)
    return model, rig, blob, plan


def _stream_plan(rig, blob, n):
    """The stream's frames as runtime --dump records them: the expressive
    columns normalised, the mouth, part opacities."""
    st = fs.MotionStream(rig, tokens=blob["tokens"], autonomic=False,
                         log=lambda s: None)
    mt = next((k for k in blob["tokens"] if k["type"] == "mouth"), None)
    if mt is not None:
        st.add_mouth(0, mt, n)
    ex, pids, lo = rig.expressive, rig.pids, rig.lo
    span = np.maximum(rig.hi[ex] - lo[ex], 1e-6)
    traj = np.empty((n, len(ex)), np.float32)
    frames = []
    for f in range(n):
        over, parts = st.frame()
        for c, i in enumerate(ex):
            traj[f, c] = ((over[pids[i]] - lo[i]) / span[c] * 2 - 1
                          if pids[i] in over else rig.default_n[c])
        frames.append((over, parts))
    np.clip(traj, -1, 1, out=traj)
    return traj, frames


def test_stream_draws_what_the_runtime_draws(tmp_path, capsys):
    """Fed the runtime's own merged tokens (idle, directed, underlay,
    mouth), the stream's per-frame writes equal runtime --tokens-in
    --dump-only's plan: the live render is the offline render, a frame at a
    time."""
    model, rig_path, blob, plan = _plan(tmp_path)
    capsys.readouterr()
    rig = fs.EventRig(str(model), str(rig_path), "default", None, None, FPS,
                      str(tmp_path), log=lambda s: None)
    n = len(plan["traj"])
    traj, frames = _stream_plan(rig, blob, n)
    np.testing.assert_array_equal(traj, plan["traj"])
    mo = np.array([[fr[0][p] for p in plan["mouth_pid"]] for fr in frames],
                  np.float32)
    np.testing.assert_array_equal(mo, plan["mouth"])
    parts = list(plan["part_id"])
    if parts:
        op = np.array([[fr[1].get(p, np.nan) for p in parts]
                       for fr in frames], np.float32)
        np.testing.assert_array_equal(op, plan["part_opacity"])


def test_stream_matches_runtime_on_hiyori(tmp_path):
    """The same on a real rig (switches, part swaps, the calm underlay),
    when the Cubism samples are there: the idle the runtime schedules for a
    seed, replayed through the stream."""
    base = os.environ.get("TSUKUYOMI_SAMPLES")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = None
    for cand in ("Samples/Resources", "Resources", ""):
        p = os.path.join(base or "/nonexistent", cand, "Hiyori",
                         "Hiyori.model3.json")
        if os.path.exists(p):
            model = p
            break
    rig_path = os.path.join(root, "rigs", "Hiyori.npz")
    clips = os.path.join(root, "clips2", "Hiyori.npz")
    if model is None or not os.path.exists(rig_path):
        pytest.skip("TSUKUYOMI_SAMPLES with Hiyori and rigs/Hiyori.npz needed")
    tok, dump = tmp_path / "t.json", tmp_path / "p.npz"
    args = ["--model", model, "--rig", rig_path, "--clips", clips,
            "--events", "default", "--probe-dir", str(tmp_path), "--seed",
            "4", "--seconds", "40"]
    assert rt.main(args + ["--tokens", str(tok), "--dump-only", "--dump",
                           str(tmp_path / "a.npz")]) == 0
    assert rt.main(args + ["--tokens-in", str(tok), "--dump-only", "--dump",
                           str(dump)]) == 0
    plan = np.load(dump, allow_pickle=True)
    blob = json.loads(tok.read_text())
    rig = fs.EventRig(model, rig_path, "default", clips, None, FPS,
                      str(tmp_path), log=lambda s: None)
    traj, frames = _stream_plan(rig, blob, len(plan["traj"]))
    np.testing.assert_array_equal(traj, plan["traj"])
    parts = list(plan["part_id"])
    op = np.array([[fr[1].get(p, np.nan) for p in parts] for fr in frames],
                  np.float32)
    np.testing.assert_array_equal(op, plan["part_opacity"])


def _rig(tmp_path):
    from tests.test_speech import _mouth_rig
    model, rig = _mouth_rig(tmp_path / "x")
    return fs.EventRig(str(model), str(rig), "default", None, None, FPS,
                       str(tmp_path), log=lambda s: None)


def _roles(frames, rig):
    ex, pids, lo = rig.expressive, rig.pids, rig.lo
    cols = {r: rig.col_of[p] for r, (p, _, _) in rig.roles.items()}
    out = {}
    for r, c in cols.items():
        i = ex[c]
        out[r] = np.array([(fr[0][pids[i]] - lo[i]) / rig.span_e[c] * 2 - 1
                           for fr in frames])
    return out


def test_streamed_idle_is_the_one_shot_idle(tmp_path):
    """The idle generated in 20 s windows (refilled 10 s ahead) renders
    exactly what the same schedulers planned in one call render: the
    window seams do not exist in the motion."""
    rig = _rig(tmp_path)
    n = int(70 * FPS)
    live = fs.MotionStream(rig, seed=7, autonomic=False, log=lambda s: None)
    got = [live.frame() for _ in range(n)]
    sc = ev.IdleScheduler(rig.stats, list(rig.roles),
                          np.random.default_rng(7), rig.start)
    toks = sc.advance(80.0)
    toks += ev.GestureScheduler(sc.st, rig.gest,
                                np.random.default_rng([7, 2])).advance(80.0)
    toks.append(ev.underlay_token(rig.stats, rig.settle, 1e9))
    for k in toks:
        if "follows" in k:
            k["follows"] = None
    once = fs.MotionStream(rig, tokens=toks, autonomic=False,
                           log=lambda s: None)
    want = [once.frame() for _ in range(n)]
    for (o1, p1), (o2, p2) in zip(got, want):
        assert o1 == o2 and p1 == p2
    assert live.ahead >= 70 and sum(k["type"] == "gesture" for k in toks) > 0


def test_directed_tokens_merge_into_the_future_without_a_jump(tmp_path):
    """A reply's tokens, realized at a time a little ahead, displace the
    idle moves they overlap on their group and play from where the role is:
    no frame-to-frame jump, the nod reaches its target, and nothing
    already shown changes."""
    rig = _rig(tmp_path)
    logs = []
    s = fs.MotionStream(rig, seed=1, autonomic=False, log=logs.append)
    ref = fs.MotionStream(rig, seed=1, autonomic=False, log=lambda x: None)
    frames, base = [], []
    rng = np.random.default_rng([1, 1])
    for f in range(int(30 * FPS)):
        if f == int(10 * FPS):
            acts = [{"t": 0.4, "tag": "nod", "arg": None},
                    {"t": 1.5, "tag": "look", "arg": "left"},
                    {"t": 2.0, "tag": "brow", "arg": "up"}]
            t0 = (f + 6) / FPS
            toks = dr.realize([dict(a, t=a["t"] + t0) for a in acts],
                              rig.stats, list(rig.roles),
                              rig.motion_seconds, rng, s.pose(),
                              settle=rig.settle)
            s.direct(toks)
        frames.append(s.frame())
        base.append(ref.frame())
    # The past is untouched: up to the merge both streams are one.
    for (a, _), (b, _) in zip(frames[:300], base[:300]):
        assert a == b
    tr, br = _roles(frames, rig), _roles(base, rig)
    for r, y in tr.items():
        assert np.abs(np.diff(y)).max() < 0.2, r
    assert any(np.abs(tr[r] - br[r]).max() > 0.05 for r in tr)
    y = tr["head_y"][300:420]
    nod = dr._toward(rig.stats, "head_y", -dr.SIGN["up"], dr.FRAC["nod"])
    assert y.min() == pytest.approx(nod, abs=0.02) or \
        any("late" in m for m in logs)
    # The directed window owns the head: no idle head move starts in it.
    assert not [k for k in s.pending if k["type"] == "head_turn"
                and k.get("src") != "director" and 10.2 < k["t"] < 11.0]
