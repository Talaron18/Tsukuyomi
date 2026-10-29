"""
The dense worked example of the beats prompt (flow.pose) and the speech-
driven micro-expressions (flow.micro). No network, no TTS: the audio is
synthesised, and one test reads the demo speech.wav files when they exist.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np
import pytest

from flow import body as fb
from flow import dataexport as dx
from flow import micro as mc
from flow import pose as pz
from flow import stream as fs
from tests.test_beats import FACES
from tests.test_dataexport import _run
from tests.test_dataset import ROOT

RIG_NAMES = ("Hiyori", "lafei", "Natori", "seed-san", "three-vrm-girl",
             "VRM1_Constraint_Twist_Sample")


def _bm(name):
    return fb.load(os.path.join(ROOT, "rigs", f"{name}.body.json"))


def _marks(text):
    return [json.loads(m) for m in re.findall(r"\[pose (\{[^}]*\})\]", text)]


# ----------------------------------------------------- the dense example

@pytest.mark.parametrize("name", RIG_NAMES)
def test_dense_example_uses_only_variables_the_body_offers(name):
    bm = _bm(name)
    have = set(pz.offered(bm))
    ex = pz._beat_examples(bm)
    assert ex, name
    marks = _marks(ex[0])
    assert len(marks) >= 6
    nonface = 0
    for m in marks:
        keys = [k for k in m if k not in ("dur", "hold")]
        assert set(keys) <= have, (name, set(keys) - have)
        assert dx.is_beat(keys)                  # face, head along, no hand
        assert any(k.startswith("head.") for k in keys) or \
            not (have & {"head.tilt", "head.yaw", "head.pitch"})
        nonface += not dx.is_beat(keys)
    assert nonface <= 1
    assert "hand." not in ex[0]
    # Four sentences, a beat at nearly every clause.
    plain = re.sub(r"\[pose \{[^}]*\}\]", "", ex[0])
    assert len(re.findall(r"[.?!]", plain)) == 4
    sp = pz.system_prompt(bm, "X.", prompt="beats")
    assert ex[0] in sp and dx.BEATS_MARK in sp


def test_dense_example_varies_the_face_and_follows_the_body():
    hiyori = pz._beat_examples(_bm("Hiyori"))[0]
    lafei = pz._beat_examples(_bm("lafei"))[0]
    for text in (hiyori, lafei):
        for want in ("brow.L.lift", "mouth.form", "gaze.x", "cheek.blush"):
            assert want in text
    assert "eye.L.smile" in hiyori
    # lafei has no smiling eye: a squint and a closed laugh with eye.*.open.
    assert "eye.L.smile" not in lafei
    assert '"eye.L.open": 0.45' in lafei and '"eye.L.open": 0,' in lafei


# Digests of the v1 prompt (system_prompt(bm, "X.")) before the dense example.
V1_DIGEST = {"Hiyori": "44842b09fe417e26", "lafei": "7367f7386be57589",
             "seed-san": "719fee22fd233b9c", "Natori": "bba679326c22bcd5",
             "three-vrm-girl": "4c485b2bf22333aa"}


@pytest.mark.parametrize("name", sorted(V1_DIGEST))
def test_v1_prompt_is_byte_identical(name):
    p = pz.system_prompt(_bm(name), "X.")
    assert hashlib.sha256(p.encode()).hexdigest()[:16] == V1_DIGEST[name]
    assert dx.BEATS_MARK not in p


def test_export_caps_keep_the_dense_example_whole():
    assert dx.BEATS_MAX == 16
    for name in ("Hiyori", "lafei"):
        ex = pz._beat_examples(_bm(name))[0]
        n = ex.count("[pose")
        res = dx.manners_reply(ex, False, max_poses=2, max_beats=dx.BEATS_MAX)
        assert res["reply"].count("[pose") == n >= 8
    # A long reply written the same way (two examples, 16 beats) still fits.
    ex = pz._beat_examples(_bm("Hiyori"))[0]
    res = dx.manners_reply(ex + " " + ex, False, max_poses=2,
                           max_beats=dx.BEATS_MAX)
    assert res["reply"].count("[pose") == 2 * ex.count("[pose")


# ------------------------------------------------------------ the audio

SR = 16000


def _speech(dur=8.0, seed=0):
    """Syllable bursts (with a few strong ones, a few long pauses) on a
    wandering pitch: enough structure for stress peaks and pauses."""
    r = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    env = np.zeros_like(t)
    pos = 0.2
    while pos < dur - 0.3:
        L = r.uniform(0.15, 0.3)
        i = (t >= pos) & (t < pos + L)
        env[i] = r.choice([0.2, 0.2, 0.5, 0.9]) * np.hanning(i.sum())
        pos += L + r.uniform(0.03, 0.1) + (0.5 if r.random() < 0.15 else 0)
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.4 * t) + 40 * (env > 0.4)
    return (env * np.sin(2 * np.pi * np.cumsum(f0) / SR) * 0.3).astype(
        np.float32)


def _face(toks):
    return [k for k in toks if k["type"] == "micro"]


def test_features_find_stress_pauses_and_the_end():
    x = _speech()
    F = mc.features(x, SR)
    assert F["has_f0"] and F["voiced"].any()
    pk = mc.stress_peaks(F)
    assert pk and all(0 <= t <= 8.0 for t, _ in pk)
    assert all(b >= mc.PEAK_Z for _, b in pk)
    assert 7.0 < mc.voiced_end(F) <= 8.0
    assert mc.pauses(F)
    assert mc.features(np.zeros(100, np.float32), SR)["voiced"].sum() == 0


def test_micro_is_seeded_and_deterministic():
    x = _speech()
    bm = _bm("Hiyori")
    a = mc.Micro(bm, seed=3, log=lambda s: None).plan(x, SR, [], 5.0, 1)
    b = mc.Micro(bm, seed=3, log=lambda s: None).plan(x, SR, [], 5.0, 1)
    c = mc.Micro(bm, seed=4, log=lambda s: None).plan(x, SR, [], 5.0, 1)
    assert a == b and a
    assert _face(a) != _face(c) or a != c
    assert all(k["t"] >= 5.0 for k in a)             # absolute stream time


def test_micro_sizes_and_times():
    toks = mc.Micro(_bm("Hiyori"), seed=1, log=lambda s: None).plan(
        _speech(), SR, [], 0.0, 0)
    brows = [k for k in toks if k.get("kind") == "brow"]
    smiles = [k for k in toks if k.get("kind") == "smile"]
    assert brows and smiles
    for k in brows:
        assert 0.25 <= k["rise"] + k["hold"] + k["fall"] <= 0.4 + 1e-9
        for d in k["params"].values():
            assert 0.2 - 1e-3 <= abs(d) <= 0.3 + 1e-3     # of the range
    for k in smiles:
        assert set(k["params"]) == {"ParamEyeLSmile", "ParamEyeRSmile"}
        for d in k["params"].values():
            assert 0.25 - 1e-3 <= abs(d) <= 0.35 + 1e-3
    assert max(smiles, key=lambda k: k["t"])["t"] > 6.5   # eases in at the end


def test_minimum_spacing_within_and_across_sentences():
    m = mc.Micro(_bm("lafei"), seed=0, log=lambda s: None)
    x = _speech(10.0, seed=2)
    ts = []
    for i in range(3):
        ts += [k["t"] for k in _face(m.plan(x, SR, [], 10.0 * i + 0.1, i))]
    ts.sort()
    assert len(ts) >= 4
    assert min(np.diff(ts)) >= mc.MIN_SPACING - 1e-6


def test_model_beats_suppress_micro_on_the_same_variables():
    x = _speech(10.0, seed=2)
    bm = _bm("Hiyori")
    free = mc.Micro(bm, seed=0, log=lambda s: None).plan(x, SR, [], 0.0, 0)
    t_brow = [k["t"] for k in free if k.get("kind") == "brow"]
    assert t_brow
    # A model beat on the brows right where the first micro lift would be.
    beat = {"t": t_brow[0] + 0.1, "tag": "pose", "at": 0, "arg": {
        "targets": {"brow.L.lift": 0.7, "brow.R.lift": 0.7}, "dur": 0.2,
        "hold": 0.4}}
    cut = mc.Micro(bm, seed=0, log=lambda s: None).plan(x, SR, [beat], 0.0, 0)
    for k in cut:
        if k.get("kind") == "brow":
            assert not (beat["t"] - mc.SUPPRESS <= k["t"]
                        <= beat["t"] + 0.6 + mc.SUPPRESS)
    assert len([k for k in cut if k.get("kind") == "brow"]) \
        < len(t_brow) + 1
    # The window is of the same variables: a beat on the mouth leaves brows.
    mouth = dict(beat, arg={"targets": {"mouth.form": 0.5}})
    keep = mc.Micro(bm, seed=0, log=lambda s: None).plan(
        x, SR, [mouth], 0.0, 0)
    assert [k["t"] for k in keep if k.get("kind") == "brow"] == t_brow


def test_variables_a_body_lacks_are_skipped():
    smile = mc.Micro(_bm("lafei"), seed=0, log=lambda s: None).unit["smile"]
    assert set(smile) == {"ParamMouthForm"}          # no smiling eye
    hi = mc.Micro(_bm("Hiyori"), seed=0, log=lambda s: None).unit
    assert "ParamMouthForm" not in hi["smile"]       # her form rests at max
    # Natori's solve also moves arm drawings: none of those are kept.
    nat = mc.Micro(_bm("Natori"), seed=0, log=lambda s: None).unit
    assert all("Arm" not in p for g in nat.values() for p in g)
    # A body without brows: no brow event.
    bm = _bm("Hiyori")
    m = mc.Micro(bm, seed=0, log=lambda s: None)
    m.unit["brow"] = {}
    assert not [k for k in m.plan(_speech(), SR, [], 0, 0)
                if k.get("kind") == "brow"]


def test_blinks_at_pauses_are_spaced():
    toks = mc.Micro(_bm("Hiyori"), seed=0, log=lambda s: None).plan(
        _speech(), SR, [], 0.0, 0)
    bl = [k["t"] for k in toks if k["type"] == "blink"]
    assert bl and min(np.diff(bl), default=9) >= mc.BLINK_SPACING - 1e-6


def test_off_switches():
    x = _speech()
    assert mc.Micro(_bm("Hiyori"), scale=0.0).plan(x, SR, [], 0, 0) == []
    ap = argparse.ArgumentParser()
    from flow import director as dr
    dr.add_args(ap)
    a = ap.parse_args([])
    assert a.micro is True and a.micro_scale == 1.0
    assert mc.planner_for(a, _bm("Hiyori")) is not None
    assert mc.planner_for(a, None) is None                  # no body mode
    off = ap.parse_args(["--no-micro"])
    assert mc.planner_for(off, _bm("Hiyori")) is None
    assert mc.planner_for(ap.parse_args(["--micro-scale", "0"]),
                          _bm("Hiyori")) is None
    half = mc.planner_for(ap.parse_args(["--micro-scale", "0.5"]),
                          _bm("Hiyori"))
    full = mc.Micro(_bm("Hiyori"), seed=0, log=lambda s: None)
    h = half.plan(x, SR, [], 0, 0)
    f = full.plan(x, SR, [], 0, 0)
    d_h = max(abs(d) for k in _face(h) for d in k["params"].values())
    d_f = max(abs(d) for k in _face(f) for d in k["params"].values())
    assert d_h < d_f
    assert mc.planner_for(argparse.Namespace(), _bm("Hiyori")) is None


def test_plan_logs_one_summary_line():
    lines = []
    m = mc.Micro(_bm("Hiyori"), seed=0, log=lines.append)
    m.plan(_speech(), SR, [], 0, 0)
    assert len(lines) == 1 and lines[0].startswith("micro: ") \
        and "brow" in lines[0] and "blink" in lines[0]
    assert m.summary().startswith("micro: ")


# ----------------------------------------------------------- the layer

def test_layer_eases_adds_and_clips():
    lay = mc.layer_for([{"type": "micro", "t": 1.0, "rise": 0.1,
                         "hold": 0.1, "fall": 0.2, "params": {"p": 0.5}}])
    assert lay is not None and mc.layer_for([]) is None
    b = {"p": (-1.0, 1.0)}
    assert lay.apply(0.5, {"p": 0.2}, b)["p"] == 0.2
    assert lay.apply(1.15, {"p": 0.2}, b)["p"] == pytest.approx(0.7)
    assert lay.apply(1.2, {"p": 0.9}, b)["p"] == 1.0         # clipped
    mid = lay.apply(1.3, {"p": 0.2}, b)["p"]
    assert 0.2 < mid < 0.7
    assert lay.apply(1.6, {"p": 0.2}, b)["p"] == 0.2
    assert not lay.busy() or lay.apply(2.0, {"p": 0.2}, b)["p"] == 0.2


def test_stream_plays_micro_and_a_model_pose_wins():
    from flow import vrm as fv
    bm = _bm("seed-san")
    rig = fv.VRMRig(bm, fps=30.0, log=lambda s: None)
    toks = mc.Micro(bm, seed=0, log=lambda s: None).plan(
        _speech(), SR, [], 0.0, 0)
    assert _face(toks)

    def run(tokens, n=240):
        st = fs.MotionStream(rig, seed=5, idle=False, autonomic=False,
                             log=lambda s: None)
        if tokens:
            st.direct(list(tokens))
        return [st.frame()[0] for _ in range(n)]
    base, on = run([]), run(toks)
    brow = [k for k in _face(toks) if k["kind"] == "brow"][0]
    f = int((brow["t"] + brow["rise"]) * 30)
    assert on[f]["brow.L.lift"] > base[f]["brow.L.lift"] + 0.1
    assert run(toks)[f]["brow.L.lift"] == on[f]["brow.L.lift"]   # replay
    # No micro token: the stream is what it was.
    assert all(a == b for a, b in zip(base, run([])))
    # A model pose holding the brows: they sit at its target.
    poses = pz.solve_poses(bm, [{"t": brow["t"] - 0.5, "tag": "pose", "at": 0,
                                 "arg": {"targets": {"brow.L.lift": 0.9},
                                         "dur": 0.2, "hold": 2.0}}],
                           log=lambda s: None)
    held = run(pz.tokens(poses, rig.roles, pz.arm_params(bm)) + [brow])
    assert held[f]["brow.L.lift"] > 0.8


def test_demo_speech_gets_a_reasonable_rate():
    from flow import speech as sp
    wavs = sorted(glob.glob(os.path.join(ROOT, "runs", "demo_q8m", "*",
                                         "speech.wav")))[:6]
    if not wavs:
        pytest.skip("no demo speech")
    n = dur = 0
    for w in wavs:
        x, sr = sp.read_audio(w)
        n += len(_face(mc.Micro(_bm("Hiyori"), seed=0,
                                log=lambda s: None).plan(x, sr, [], 0, 0)))
        dur += len(x) / sr
    assert 0.1 < n / dur < 0.8                      # events a second


# ------------------------------------------------- the data stays as was

# Digests of the dataset trajectories (traj/*.npz) of FACES on two avatars,
# taken before the micro layer existed.
TRAJ_DIGEST = {"seed-san": "4d815fcfb2a45c8c", "Hiyori": "afcb01e730c5fcfc"}


def _traj_digest(root):
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(root, "traj", "**", "*.npz"),
                              recursive=True)):
        z = np.load(f, allow_pickle=True)
        for k in sorted(z.files):
            if z[k].dtype == object:
                continue
            h.update(k.encode())
            h.update(np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("avatar", sorted(TRAJ_DIGEST))
def test_dataset_trajectories_are_unchanged(tmp_path, avatar):
    out, recs = _run(tmp_path, FACES, avatars=avatar, sessions=1, turns=2,
                     prompt="beats")
    assert _traj_digest(out) == TRAJ_DIGEST[avatar]
    for r in recs:
        assert "micro" not in json.dumps(r.get("tokens", ""))


def test_dataset_director_args_turn_micro_off():
    from flow import dataset as ds
    spec = {"kind": "vrm", "name": "x", "body": "rigs/seed-san.body.json",
            "events": "default"}
    opts = ds.parse_opts(["gen", "--avatars", "seed-san", "--live2d-lite"])
    assert ds.director_args(spec, opts).micro is False


# ------------------------------------------- the offline render, end to end

def test_offline_render_plays_the_micro_tokens(tmp_path):
    """actions.json's "micro" tokens reach the render's tokens, move the
    parameters where they say, and the live stream draws the same frames."""
    from flow.runtime import main as runtime_main
    from tests.test_live import _stream_plan
    from tests.test_pose import _body_rig
    model, rig, _bm_ = _body_rig(tmp_path / "x")
    acts = tmp_path / "actions.json"
    acts.write_text(json.dumps({"actions": [], "micro": [
        {"t": 1.0, "type": "micro", "role": "micro", "kind": "brow",
         "src": "director", "params": {"ParamAllX": 0.4}, "rise": 0.1,
         "hold": 0.2, "fall": 0.2},
        {"t": 2.0, "type": "blink", "role": "eyes", "src": "director",
         "micro": True}]}))
    base = ["--model", str(model), "--rig", str(rig), "--events", "default",
            "--probe-dir", str(tmp_path), "--seed", "2", "--seconds", "4",
            "--fps", "30"]
    out = {}
    for name, extra in (("on", ["--direct", str(acts)]),
                        ("off", [])):
        tok = tmp_path / f"{name}.json"
        assert runtime_main(base + extra + ["--tokens", str(tok),
                                            "--dump-only", "--dump",
                                            str(tmp_path / f"{name}.npz")]
                            ) == 0
        out[name] = (json.loads(tok.read_text()),
                     np.load(tmp_path / f"{name}.npz", allow_pickle=True))
    blob, plan = out["on"]
    assert sum(k["type"] == "micro" for k in blob["tokens"]) == 1
    assert not any(k["type"] == "micro" for k in out["off"][0]["tokens"])
    pid = list(plan["pid"])
    c = pid.index("ParamAllX")
    d = plan["traj"][:, c] - out["off"][1]["traj"][:, c]
    assert np.abs(d).max() > 0.01 and np.abs(d[:25]).max() == 0
    assert np.abs(d[60:]).max() == 0                 # back after its fall
    er = fs.EventRig(str(model), str(rig), "default", None, None, 30.0,
                     str(tmp_path), log=lambda s: None)
    traj, _frames = _stream_plan(er, blob, len(plan["traj"]))
    np.testing.assert_array_equal(traj, plan["traj"])
    # --tokens-in replays the same frames.
    assert runtime_main(base + ["--tokens-in", str(tmp_path / "on.json"),
                                "--dump-only", "--dump",
                                str(tmp_path / "re.npz")]) == 0
    np.testing.assert_array_equal(
        np.load(tmp_path / "re.npz", allow_pickle=True)["traj"], plan["traj"])


def test_director_micro_plan_needs_body_mode_audio_and_the_flag():
    from flow import director as dr
    ap = argparse.ArgumentParser()
    dr.add_args(ap)
    a = ap.parse_args([])
    bm = _bm("Hiyori")
    said = {"audio": _speech(), "sr": SR,
            "spans": [(i * 0.1, i * 0.1 + 0.1) for i in range(60)]}
    spoken = "Oh really. " * 5
    got = dr.micro_plan(a, {"body": bm}, said, [], spoken, lambda s: None)
    assert got and all(k["type"] in ("micro", "blink") for k in got)
    assert dr.micro_plan(a, None, said, [], spoken) == []
    a.micro = False
    assert dr.micro_plan(a, {"body": bm}, said, [], spoken) == []
