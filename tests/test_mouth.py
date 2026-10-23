"""
flow.mouth: the lip sync calibrated from a rig's own talking motions. Fake
rigs, each with one voice-line motion keyed the way a rigger would:

  vowel rig   lip flaps on ParamA/I/O (to 0.8), ParamMouthOpenY never moves
              -> drive the vowels, OpenY left alone;
  open rig    flaps on ParamMouthOpenY to ~0.7, ParamMouthForm narrowing
              with every opening -> drive OpenY to its p95 and Form along;
  held rig    OpenY held open for seconds (an expression, Hiyori m05 style)
              -> no talking: the open-only lip sync of before.

A planned run on the vowel rig must put the speech on the vowels and
replay bit for bit; on the held rig --lipsync auto must equal --lipsync
open exactly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from flow import director as dr
from flow import mouth as mo
from tests.test_speech import _replay, two_phrases, wav_bytes

FPS = 30.0
MOUTH = [("ParamMouthOpenY", 0.0, 1.0, 0.0), ("ParamMouthForm", -1.0, 1.0, 0.0),
         ("ParamA", 0.0, 1.0, 0.0), ("ParamI", 0.0, 1.0, 0.0),
         ("ParamU", 0.0, 1.0, 0.0), ("ParamE", 0.0, 1.0, 0.0),
         ("ParamO", 0.0, 1.0, 0.0)]


def linear(points):
    """motion3 Segments through (t, v) points, linear segments."""
    seg = [points[0][0], points[0][1]]
    for t, v in points[1:]:
        seg += [0, t, v]
    return seg


def _talk_rig(root, curves, dur=6.0):
    """test_speech's mouth rig plus the vowel parameters, and its two
    motions replaced by one voice line with `curves` {pid: points}."""
    from l2dprobe.analyze import group_of
    from tests.test_director import _rig
    model, rig = _rig(root)
    z = dict(np.load(rig, allow_pickle=True))
    z["pid"] = np.array(list(z["pid"]) + [p for p, *_ in MOUTH], dtype=object)
    z["kind"] = np.array(list(z["kind"]) + ["continuous"] * len(MOUTH),
                         dtype=object)
    z["group"] = np.array(list(z["group"]) + [group_of(p) for p, *_ in MOUTH],
                          dtype=object)
    for k, j in (("minimum", 1), ("maximum", 2), ("default", 3)):
        z[k] = np.concatenate([z[k], np.array([m[j] for m in MOUTH],
                                              np.float32)])
    z["descriptor"] = np.zeros((len(z["pid"]), 4), np.float32)
    np.savez(rig, **z)
    arm = [0, 0, 1, 2.0, 1, 4.0, 0, 6.0, 0]
    talk = {"Version": 3, "Meta": {"Duration": dur, "Fps": 30},
            "Curves": [{"Target": "Parameter", "Id": "ParamArmL",
                        "Segments": arm}]
            + [{"Target": "Parameter", "Id": p, "Segments": linear(v)}
               for p, v in curves.items()]}
    (root / "m.motion3.json").write_text(json.dumps(talk))
    return model, rig


def vowel_rig(root):
    # ParamA, I and O take turns: each opening is one vowel.
    a, i, o = ([(0.0, 0.0), (1.0, 0.0)] for _ in range(3))
    for k in range(12):
        s = 1.0 + k * 0.25
        for c, tgt in zip((a, i, o), (0, 1, 2)):
            c += [(s + 0.1, 0.8 if k % 3 == tgt else 0.0), (s + 0.25, 0.0)]
    return _talk_rig(root, {"ParamA": a, "ParamI": i, "ParamO": o,
                            "ParamMouthOpenY": [(0.0, 0.0), (6.0, 0.0)]})


def open_rig(root):
    rng = np.random.default_rng(0)
    pts = [(0.0, 0.0), (1.0, 0.0)]
    form = [(0.0, 0.0), (1.0, 0.0)]
    for k in range(14):
        s = 1.0 + k * 0.28
        pk = float(rng.uniform(0.55, 0.72))
        pts += [(s + 0.12, pk), (s + 0.28, 0.02)]
        form += [(s + 0.12, -0.6 * pk), (s + 0.28, 0.0)]
    return _talk_rig(root, {"ParamMouthOpenY": pts, "ParamMouthForm": form})


def held_rig(root):
    return _talk_rig(root, {"ParamMouthOpenY": [(0.0, 0.0), (0.5, 1.0),
                                                (4.0, 1.0), (4.5, 0.0),
                                                (6.0, 0.0)],
                            "ParamMouthForm": [(0.0, 1.0), (6.0, 1.0)]})


# ------------------------------------------------------------ calibration

def test_vowel_rig_drives_vowels(tmp_path):
    model, rig = vowel_rig(tmp_path / "v")
    cal = mo.calibrate(str(model), str(rig), FPS)
    assert cal["drive"] == "vowels" and cal["motions"] == ["m.motion3.json"]
    assert set(cal["map"]) == {"ParamA", "ParamI", "ParamO"}
    assert {m["curve"] for m in cal["map"].values()} == {"a", "i", "o"}
    # Keyed to 0.8 with linear ramps: frames straddle the peak.
    assert all(0.72 < m["hi"] <= 0.8 for m in cal["map"].values())
    ps = cal["params"]
    assert not ps["ParamMouthOpenY"]["used"] and not ps["ParamU"]["used"]
    line = mo.describe(cal)
    assert "talks with ParamA" in line and "ParamMouthOpenY" in line
    assert "unused" in line and "driving vowels" in line


def test_open_rig_drives_open_to_its_range_and_form_follows(tmp_path):
    model, rig = open_rig(tmp_path / "o")
    cal = mo.calibrate(str(model), str(rig), FPS)
    assert cal["drive"] == "open"
    m = cal["map"]["ParamMouthOpenY"]
    assert m["curve"] == "open" and m["lo"] == 0.0
    assert 0.6 < m["hi"] < 0.73
    f = cal["map"]["ParamMouthForm"]
    assert f["curve"] == "open" and f["lo"] == 0.0 and f["hi"] < -0.3
    assert cal["params"]["ParamMouthForm"]["follow"]
    assert "ParamA" not in cal["map"]
    assert 0.3 < cal["level"] < 1
    assert "follows: ParamMouthForm" in mo.describe(cal)


def test_held_mouth_is_not_talking(tmp_path):
    model, rig = held_rig(tmp_path / "h")
    cal = mo.calibrate(str(model), str(rig), FPS)
    assert cal["drive"] is None and cal["map"] == {}
    assert "no talking curves" in mo.describe(cal)
    assert "as before" in mo.describe(cal)


def test_talking_frames_need_a_run_of_quick_flaps():
    t = np.arange(0, 6, 1 / FPS)
    quick = (np.sin(2 * np.pi * 4 * t) > 0) * (t > 1) * (t < 3)
    assert mo.talking_frames(quick.astype(float), FPS).sum() > 50
    slow = (np.sin(2 * np.pi * 0.8 * t) > 0).astype(float)    # panting
    assert not mo.talking_frames(slow, FPS).any()
    two = quick * (t < 1.5)                                      # two flaps
    assert not mo.talking_frames(two.astype(float), FPS).any()


def test_level_gamma_bounds():
    op = np.array([0, 0.5, 0.6, 0.7, 0])
    assert mo.level_gamma(op, None) == 1.0
    g = mo.level_gamma(op, 0.7)
    assert np.median(op[op > 0] ** g) == pytest.approx(0.7)
    assert mo.level_gamma(op, 0.999) == mo.GAMMA_LO
    assert mo.level_gamma(op, 0.01) == mo.GAMMA_HI


# --------------------------------------------------------------- planning

def _plan(tmp_path, model, rig, name, *extra):
    reply = tmp_path / "reply.txt"
    reply.write_text("Hello there. [nod] How are you?", encoding="utf-8")
    audio = tmp_path / "said.wav"
    audio.write_bytes(wav_bytes(two_phrases()))
    out = tmp_path / name
    assert dr.main(["--model-json", str(model), "--rig", str(rig),
                    "--backend", "file", "--reply-file", str(reply),
                    "--out", str(out), "--probe-dir", str(tmp_path),
                    "--seed", "2", "--tts", "file", "--tts-file", str(audio),
                    *extra]) == 0
    return out


def test_vowel_rig_plans_and_replays_bit_for_bit(tmp_path, capsys):
    model, rig = vowel_rig(tmp_path / "v")
    out = _plan(tmp_path, model, rig, "run")
    log = capsys.readouterr().out
    assert "mouth: x talks with ParamA" in log
    toks = json.loads((out / "tokens.json").read_text())["tokens"]
    (mt,) = [k for k in toks if k["type"] == "mouth"]
    assert mt["mode"] == "auto" and set(mt["map"]) == {"ParamA", "ParamI",
                                                       "ParamO"}
    assert set(mt["vowels"]) == set("aiueo")
    plan = np.load(out / "plan.npz", allow_pickle=True)
    assert sorted(plan["mouth_pid"]) == ["ParamA", "ParamI", "ParamO"]
    assert plan["mouth"].max() <= 0.8 + 1e-6 and plan["mouth"].max() > 0.2
    rp = _replay(log, tmp_path)
    np.testing.assert_array_equal(plan["mouth"], rp["mouth"])
    np.testing.assert_array_equal(plan["traj"], rp["traj"])


def test_open_rig_plans_open_and_form(tmp_path, capsys):
    model, rig = open_rig(tmp_path / "o")
    out = _plan(tmp_path, model, rig, "run")
    plan = np.load(out / "plan.npz", allow_pickle=True)
    pid = list(plan["mouth_pid"])
    assert sorted(pid) == ["ParamMouthForm", "ParamMouthOpenY"]
    op = plan["mouth"][:, pid.index("ParamMouthOpenY")]
    fm = plan["mouth"][:, pid.index("ParamMouthForm")]
    assert op.max() <= 0.73 and op.max() > 0.5
    assert fm.min() < -0.3 and (fm[op == 0] == 0).all()


def test_auto_without_talking_curves_is_open_exactly(tmp_path, capsys):
    model, rig = held_rig(tmp_path / "h")
    a = _plan(tmp_path, model, rig, "auto")
    b = _plan(tmp_path, model, rig, "open", "--lipsync", "open")
    assert "no talking curves" in capsys.readouterr().out
    assert (a / "tokens.json").read_text() == (b / "tokens.json").read_text()
    pa = np.load(a / "plan.npz", allow_pickle=True)
    pb = np.load(b / "plan.npz", allow_pickle=True)
    np.testing.assert_array_equal(pa["mouth"], pb["mouth"])
    assert list(pa["mouth_pid"]) == ["ParamMouthOpenY"]


def test_lipsync_open_keeps_the_old_drive_on_a_talking_rig(tmp_path):
    model, rig = vowel_rig(tmp_path / "v")
    out = _plan(tmp_path, model, rig, "run", "--lipsync", "open")
    toks = json.loads((out / "tokens.json").read_text())["tokens"]
    (mt,) = [k for k in toks if k["type"] == "mouth"]
    assert mt["mode"] == "open" and "map" not in mt
    plan = np.load(out / "plan.npz", allow_pickle=True)
    assert list(plan["mouth_pid"]) == ["ParamMouthOpenY"]


def test_mapped_mouth_is_clamped_to_the_rig():
    from flow.runtime import mouth_frames
    pids = ["ParamMouthOpenY", "ParamA"]
    tok = {"fps": 30.0, "open": [0, 1.0, 0.5], "vowels": {"a": [0, 1, 1]},
           "map": {"ParamA": {"curve": "a", "lo": 0.0, "hi": 1.5},
                   "Gone": {"curve": "open", "lo": 0, "hi": 1}}}
    m = mouth_frames(tok, pids, ["mouth"] * 2, np.array([1.0, 1.0]),
                     np.zeros(2), 30.0, 4, lo=np.zeros(2))
    assert set(m) == {"ParamA"}
    np.testing.assert_allclose(m["ParamA"], [0, 1, 1, 0])
