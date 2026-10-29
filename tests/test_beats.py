"""
Expression beats: the r2 teacher prompt (flow.pose "beats"), the beat counts
and the beat cap of flow.dataexport, and the idle head drift of
flow.stream. No network: the dataset runs are mocked models.
"""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

from flow import body as fb
from flow import dataexport as dx
from flow import dataset as ds
from flow import pose as pz
from flow import stream as fs
from tests.test_dataexport import _run
from tests.test_dataset import ROOT

SEED_VRM = os.path.join(ROOT, "assets", "vrm",
                        "VRM1_Constraint_Twist_Sample.vrm")
RIGS = [os.path.join(ROOT, "rigs", f"{n}.body.json")
        for n in ("Hiyori", "lafei", "seed-san", "Natori")]

BEATS = ('[pose {"brow.L.lift": 0.7, "brow.R.lift": 0.7, "head.pitch": 4, '
         '"dur": 0.2}] Wait, you did what? [pose {"eye.L.smile": 0.8, '
         '"eye.R.smile": 0.8, "mouth.form": 0.8, "head.tilt": 6}] That is '
         'funny. [pose {"gaze.x": -0.4}] Hm. '
         '[pose {"hand.R.y": 1.2, "head.tilt": 5}] Look!')


FACES = ('[pose {"brow.L.lift": 0.5, "head.yaw": 3}] Oh? [pose {"mouth.form": '
         '0.6, "head.tilt": 5}] Nice. [pose {"eye.L.smile": 0.8, "eye.R.smile"'
         ': 0.8, "head.tilt": -4}] Really funny.')


# ---------------------------------------------------------------- prompt

def test_v1_prompt_is_the_default_and_the_beats_prompt_differs():
    for path in RIGS:
        bm = fb.load(path)
        v1 = pz.system_prompt(bm, "X.")
        assert v1 == pz.system_prompt(bm, "X.", prompt="v1")
        assert dx.BEATS_MARK not in v1
        b = pz.system_prompt(bm, "X.", prompt="beats")
        assert dx.BEATS_MARK in b and b != v1
        assert b == pz.system_prompt(fb.load(path), "X.", prompt="beats")
    with pytest.raises(ValueError):
        pz.system_prompt(fb.load(RIGS[0]), "X.", prompt="v9")


def test_beats_prompt_names_only_variables_the_body_has():
    for path in RIGS:
        bm = fb.load(path)
        have = set(pz.offered(bm))
        b = pz.system_prompt(bm, "X.", prompt="beats")
        text = b[b.index("How to move"):]
        rules = text[:text.index("Examples of placement")]
        named = set(re.findall(r'"?\b((?:head|eye|brow|mouth|cheek|gaze|'
                               r'hand|torso)\.[A-Za-z.]+)\b', text))
        assert named <= have | {"hand.R.y", "hand.L.y"}, (path,
                                                         named - have)
        assert ("eye.L.smile" in rules) == ("eye.L.smile" in have)
    lafei = pz.system_prompt(fb.load(RIGS[1]), "X.", prompt="beats")
    lafei = lafei[lafei.index("How to move"):]
    assert "eye.L.smile" not in lafei and '"eye.L.open": 0' in lafei
    assert "brow.L.form" not in lafei


def test_beats_prompt_examples_carry_the_head_in_the_marker():
    for path in RIGS:
        b = pz.system_prompt(fb.load(path), "X.", prompt="beats")
        ex = b[b.index("Examples of placement"):]
        marks = re.findall(r"\[pose (\{[^}]*\})\]", ex)
        assert len(marks) >= 3
        for m in marks:
            keys = list(json.loads(m))
            assert dx.is_beat(k for k in keys if k not in ("dur", "hold"))
        assert "hand." not in ex


def test_dataset_teaches_with_beats_unless_asked_for_v1(tmp_path):
    assert ds.parse_opts(["gen"]).teacher_prompt == "beats"
    for prompt, want in (("beats", True), ("v1", False)):
        out, _recs = _run(tmp_path / prompt, FACES, avatars="seed-san",
                          sessions=1, turns=1, prompt=prompt)
        reg = dx.load_registry(out)
        assert any(dx.BEATS_MARK in t for t in reg["prompts"].values()) \
            is want
        assert dx.default_max_beats(out) == (dx.BEATS_MAX if want else 0)


# --------------------------------------------------------------- counting

def test_a_beat_is_a_face_marker_with_optional_head():
    assert dx.is_beat(["eye.L.smile", "head.tilt"])
    assert dx.is_beat(["gaze.x"])
    assert not dx.is_beat(["head.tilt"])                    # a nod
    assert not dx.is_beat(["mouth.form", "hand.R.y"])
    assert not dx.is_beat(["brow.L.lift", "torso.tilt"])
    assert not dx.is_beat([])


def test_sentences_are_counted_in_en_and_ja():
    assert dx.sentence_count("Hi there. How are you? Fine!") == 3
    assert dx.sentence_count("こんにちは。元気ですか？") == 2
    assert dx.sentence_count("Well... maybe.") == 2
    assert dx.sentence_count("") == 0


def test_beat_counts_of_a_reply():
    c = dx.beat_counts(BEATS)
    assert c["poses"] == 4 and c["beats"] == 3 and c["beats_head"] == 2
    assert c["sentences"] == 4
    assert c["targets"] == 10
    assert c["hand_targets"] == 1
    assert c["face_targets"] == 2 + 3 + 1
    assert dx.beat_counts("No markers here.")["beats"] == 0


def _rec(i, text, avatar="A"):
    return {"id": i, "avatar": {"name": avatar}, "reply_raw": text,
            "session": "s" + i, "turn": 0, "persona": "p", "lang": {},
            "topic": "small_talk", "user": "hello"}


def test_beats_over_records_and_lines():
    recs = [_rec("1", BEATS), _rec("2", "Plain."), _rec("3", BEATS, "B")]
    b = dx.beats(recs)
    assert b["total"]["turns"] == 3 and b["total"]["beats"] == 6
    assert set(b["by_avatar"]) == {"A", "B"}
    assert dx.beats(recs, count={"1"})["total"]["turns"] == 1
    assert dx.beats(recs, {"1": {"reply": "Plain."}})["total"]["beats"] == 3
    lines = dx.beats_lines(b)
    assert lines[0].startswith("expression beats: 6 beats in 9 sentences")
    assert "hand share" in lines[0] and len(lines) == 3
    mid = dx.beats(recs, {"1": {"reply": "Plain."}})
    assert "->" in dx.beats_lines(mid, b)[0]


def test_stats_carry_the_beats():
    recs = [_rec("1", BEATS)]
    notes = {"1": ([], True)}
    s = dx.stats(recs, notes, bcounts=dx.beats(recs),
                 bbefore=dx.beats(recs))
    assert s["beats"]["total"]["beats"] == 3
    assert "expression beats:" in dx.stats_text(s)
    json.dumps(s)
    assert "expression beats" not in dx.stats_text(dx.stats(recs, notes))


# ---------------------------------------------------------------- manners

def test_max_beats_caps_beats_apart_from_poses():
    # Unasked: the hand goes, the four face markers stay as beats.
    old = dx.manners_reply(BEATS, False, max_poses=2)
    assert dx.beat_counts(old["reply"])["poses"] == 2
    new = dx.manners_reply(BEATS, False, max_poses=2, max_beats=9)
    assert new["skipped"] is None
    c = dx.beat_counts(new["reply"])
    assert c["beats"] == 3 and c["poses"] == 4
    assert "hand." not in new["reply"] and '"head.tilt": 5' in new["reply"]
    assert dx.spoken_of(new["reply"]) == dx.spoken_of(BEATS)
    two = dx.manners_reply(BEATS, False, max_poses=2, max_beats=2)
    assert dx.beat_counts(two["reply"])["beats"] == 2
    assert two["c"]["markers_capped"] == 1
    # The first beats stay.
    assert '"brow.L.lift": 0.7' in two["reply"]
    assert '"gaze.x"' not in two["reply"]


def test_zero_max_beats_is_manners_as_before():
    for asked in (False, True):
        a = dx.manners_reply(BEATS, asked, max_poses=2)
        b = dx.manners_reply(BEATS, asked, max_poses=2, max_beats=0)
        assert a["reply"] == b["reply"] and dict(a["c"]) == dict(b["c"])


def test_manners_counts_name_the_beat_cap():
    recs = [_rec("1", BEATS)]
    _m, c = dx.manners(recs, None, 2, max_beats=9)
    assert c["max_beats"] == 9
    assert "9 beats" in dx.manners_lines(c)[0]
    _m, c0 = dx.manners(recs, None, 2)
    assert "beats" not in dx.manners_lines(c0)[0]


def test_export_keeps_beats_of_a_beats_dataset_and_not_of_r1(tmp_path):
    rows = {}
    for prompt in ("beats", "v1"):
        out, _ = _run(tmp_path / prompt, FACES, avatars="seed-san",
                      sessions=1, turns=1, prompt=prompt)
        dst = tmp_path / f"{prompt}.jsonl"
        assert dx.main(["sft", out, "--out", str(dst)]) == 0
        row = json.loads(dst.read_text().splitlines()[0])
        rows[prompt] = row["messages"][-1]["content"]
    assert dx.beat_counts(rows["beats"])["beats"] == 3
    assert dx.beat_counts(rows["v1"])["poses"] == 2


def test_stats_command_prints_the_beats_block(tmp_path, capsys):
    out, _ = _run(tmp_path, FACES, avatars="seed-san", sessions=1, turns=1,
                  prompt="beats")
    js = tmp_path / "s.json"
    assert dx.main(["stats", out, "--json", str(js)]) == 0
    text = capsys.readouterr().out
    assert "expression beats:" in text and "manners (at most 2 poses, 12 " \
        "beats, 1 hand-moving markers)" in text
    s = json.loads(js.read_text())
    assert s["beats"]["total"]["beats"] > 0
    assert s["beats"]["before"]["total"]["turns"] == s["beats"]["total"][
        "turns"]
    assert dx.main(["stats", out, "--no-manners"]) == 0
    assert "->" not in capsys.readouterr().out.split("expression beats:")[1] \
        .split("\n")[0]


# ------------------------------------------------------------- head drift

def _rig(lo=-30.0, hi=30.0):
    pids = ["ParamAngleX", "ParamAngleY", "ParamAngleZ", "other"]
    return SimpleNamespace(
        pids=pids, expressive=[0, 1, 2, 3],
        lo=np.array([lo, lo, lo, -1.0]), hi=np.array([hi, hi, hi, 1.0]),
        role_cols=[("head_x", 0, 1.0), ("head_y", 1, 1.0),
                   ("head_z", 2, 1.0), ("gaze_x", 3, 1.0)])


def _trace(d, seconds=120, fps=30):
    ts = np.arange(0, seconds, 1 / fps)
    return {p: np.array([d.offsets(t)[p] for t in ts])
            for p in ("ParamAngleX", "ParamAngleY", "ParamAngleZ")}, ts


def test_drift_is_slow_small_smooth_and_seeded():
    tr, ts = _trace(fs.HeadDrift(_rig(), seed=1))
    for p, peak in (("ParamAngleX", 2.5), ("ParamAngleY", 1.5),
                    ("ParamAngleZ", 2.0)):
        x = tr[p]
        assert np.abs(x).max() <= peak + 1e-9
        assert x.std() > 0.15 * peak                    # it does move
        # Several seconds a swing: the fastest component is 5 s.
        assert np.abs(np.diff(x)).max() * 30 <= peak * 2 * np.pi / 5 + 1e-9
    again, _ = _trace(fs.HeadDrift(_rig(), seed=1))
    other, _ = _trace(fs.HeadDrift(_rig(), seed=2))
    assert all(np.array_equal(tr[p], again[p]) for p in tr)
    assert not np.array_equal(tr["ParamAngleX"], other["ParamAngleX"])
    # The gaze role is not the head's.
    assert set(fs.HeadDrift(_rig(), 0).offsets(1.0)) == set(tr)


def test_drift_scale_zero_and_normalised_rigs():
    over = {"ParamAngleX": 5.0, "ParamAngleY": 0.0}
    assert fs.HeadDrift(_rig(), 0, 0.0).apply(3.0, dict(over)) == over
    big = fs.HeadDrift(_rig(), 0, 2.0)
    assert max(abs(v) for v in big.offsets(7.0).values()) > 0
    # +-1 parameters: 2.5 degrees of a nominal 30 is 1/12 of the half range.
    tr, _ = _trace(fs.HeadDrift(_rig(-1.0, 1.0), seed=1))
    assert np.abs(tr["ParamAngleX"]).max() <= 2.5 / 30 + 1e-9
    # Kept inside the range.
    d = fs.HeadDrift(_rig(-1.0, 1.0), 3, 100.0)
    for t in np.arange(0, 30, 0.5):
        v = d.apply(t, {"ParamAngleX": 0.9})["ParamAngleX"]
        assert -1.0 <= v <= 1.0


def test_drift_args_default_on_and_switch_off():
    from flow import director as dr
    import argparse
    ap = argparse.ArgumentParser()
    dr.add_args(ap)
    a = ap.parse_args([])
    assert a.head_drift is True and fs.drift_from_args(a) == 1.0
    a = ap.parse_args(["--no-head-drift"])
    assert fs.drift_from_args(a) is None
    a = ap.parse_args(["--head-drift-scale", "0.5"])
    assert fs.drift_from_args(a) == 0.5
    assert fs.drift_from_args(argparse.Namespace()) is None


@pytest.mark.skipif(not os.path.exists(SEED_VRM), reason="no VRM sample")
def test_stream_drifts_at_rest_and_a_head_pose_takes_over():
    from flow import vrm as fv
    from flow import vrmbody as vb
    bm = vb.VRMBody(vb.build(SEED_VRM, log=lambda s: None))
    rig = fv.VRMRig(bm, fps=30.0, log=lambda s: None)

    def run(drift, tokens=(), n=240):
        st = fs.MotionStream(rig, seed=5, idle=False, autonomic=False,
                             head_drift=drift, log=lambda s: None)
        if tokens:
            st.direct(list(tokens))
        return [st.frame()[0] for _ in range(n)]

    off, on = run(None), run(1.0)
    # Without it a quiet idle holds the head; with it the head wanders by
    # a few degrees and no more.
    yaw0 = np.array([o["head.yaw"] for o in off])
    yaw1 = np.array([o["head.yaw"] for o in on])
    assert yaw0.std() < 1e-6 and 0.2 < yaw1.std() < 2.5
    assert np.abs(yaw1 - yaw0).max() <= 2.5 + 1e-9
    assert run(1.0)[100]["head.yaw"] == yaw1[100]       # seeded
    # A pose on the head parameter wins while it holds: the drift is under
    # it, not on top.
    poses = pz.solve_poses(bm, [{"t": 1.0, "tag": "pose", "arg": {
        "targets": {"head.yaw": 20.0}, "dur": 0.3, "hold": 3.0}, "at": 0}],
        log=lambda s: None)
    held = run(1.0, pz.tokens(poses, rig.roles, pz.arm_params(bm)))
    mid = [o["head.yaw"] for o in held[60:100]]
    assert np.mean(mid) > 17.0 and max(mid) - min(mid) < 4.0
