"""
Motion primitives (flow.moves): [move {...}] parsing, the curves, what each
body is offered, the layer over the idle and under the poses, collisions on
a VRM, the Live2D degrade path, and outputs unchanged without [move].

No network, no GL: the body files under rigs/ and test_pose's fake rig.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from flow import body as fb
from flow import director as dr
from flow import moves as mv
from flow import pose as pz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RIGS = os.path.join(ROOT, "rigs")
FPS = 30.0


def _load(name):
    path = os.path.join(RIGS, f"{name}.body.json")
    if not os.path.exists(path):
        pytest.skip(f"{path} missing")
    return fb.load(path)


@pytest.fixture(scope="module")
def seed():
    return _load("seed-san")


@pytest.fixture(scope="module")
def hiyori():
    return _load("Hiyori")


@pytest.fixture(scope="module")
def lafei():
    return _load("lafei")


# ----------------------------------------------------------------- parse

def test_move_markers_parse_whole_or_split_and_tolerate_bad_json(seed):
    reply = ('Let\'s dance! [move {"name": "dance", "beats": 8, tempo: 120,}]'
             ' yay [clap] [move:cheer {"side": "left"}] [move nod_along] '
             '[move {"name": "moonwalk"}] [move {"name": "sway", "beats": '
             '999, "amp": 3}] [move {"name": }] end.')
    whole = pz.parser_for(seed, moves=True)
    whole.feed(reply)
    whole.close()
    for cut in (5, 17, 31, 40, 63):
        p = pz.parser_for(seed, moves=True)
        for i in range(0, len(reply), cut):
            p.feed(reply[i:i + cut])
        p.close()
        assert p.tags == whole.tags and p.spoken == whole.spoken
    assert whole.spoken == "Let's dance! yay end."
    args = [t["arg"] for t in whole.tags]
    assert args[0] == {"name": "dance", "beats": 8.0, "tempo": 120.0}
    assert args[1] == {"name": "clap"}
    assert args[2] == {"name": "cheer", "side": "L"}
    # An alias: head_bob at half tempo, smaller.
    assert args[3]["name"] == "head_bob" and args[3]["tempo_scale"] == 0.5
    # Clamped into range.
    assert args[4] == {"name": "sway", "beats": 32.0, "amp": 1.0}
    w = " | ".join(whole.warnings)
    assert "JSON repaired" in w and "no motion 'moonwalk'" in w
    assert "bad JSON" in w
    assert all(t["tag"] == "move" for t in whole.tags)
    assert pz.tag_text(whole.tags[0]) == \
        '[move {"name": "dance", "beats": 8.0, "tempo": 120.0}]'


def test_without_moves_the_parser_and_prompt_are_those_of_before(seed,
                                                                 hiyori):
    for bm in (seed, hiyori):
        p0 = pz.parser_for(bm)
        p0.feed('Hi [move {"name": "sway"}] there [dance].')
        p0.close()
        assert p0.tags == [] and "move" not in p0.spoken \
            and "dance" not in p0.spoken
        assert "is not a body-mode marker" in p0.warnings[0]
        assert pz.system_prompt(bm, "X.") == pz.system_prompt(
            bm, "X.", False, "", "")
        on = pz.system_prompt(bm, "X.", moves=mv.prompt_lines(bm))
        assert "[move {" in on and "[move {" not in pz.system_prompt(bm)
        assert mv.PROMPT_RULE.strip() in on
    # The VRM prompt sha1 of test_vrm is checked there; here the off path
    # of the director's option.
    import argparse
    a = argparse.Namespace(body_moves=False)
    assert dr.move_menu(a, seed) == ("", [])


# ---------------------------------------------------------------- curves

def test_curves_follow_the_tempo_stay_in_bounds_and_ease(seed):
    rec = mv.build(seed, {"name": "sway", "tempo": 120, "beats": 8,
                          "amp": 0.5}, log=lambda s: None)
    spb = 0.5
    assert rec["seconds"] == pytest.approx(8 * spb)
    tilt = np.array(rec["add"]["torso.tilt"])
    t = np.arange(len(tilt)) * rec["dt"]
    # Period two beats: a sine of 1 s.
    ref = np.sin(2 * math.pi * t / (2 * spb))
    assert np.corrcoef(tilt, ref)[0, 1] > 0.999
    assert np.abs(tilt).max() <= 9.0 * 0.5 + 1e-6
    # The envelope: zero at both ends with zero slope, one inside.
    d, ri, ro = rec["seconds"], rec["ramp_in"], rec["ramp_out"]
    assert mv.envelope(0.0, d, ri, ro) == 0 and mv.envelope(d, d, ri, ro) == 0
    assert mv.envelope(d / 2, d, ri, ro) == 1.0
    assert mv.envelope(0.01, d, ri, ro) < 1e-3
    # One beat, longer for a clip that moves an arm (a VRM's hanging arms
    # under a sway).
    assert 0.3 <= ri <= mv.ARM_RAMP
    # pulse: 1 on the beat, 0 between.
    assert mv.pulse(3.0) == pytest.approx(1) and mv.pulse(2.5) == \
        pytest.approx(0)
    # PCHIP: never past its keyframes (a clap's contact stays a contact).
    Y = np.array([[0.0, 1.0, 0.0, 1.0, 1.0, 0.0]])
    M = mv.pchip_slopes(Y)
    u = np.linspace(0, 5, 501)
    v = np.array([mv.pchip_at(Y, M, x)[0] for x in u])
    assert v.min() >= -1e-12 and v.max() <= 1 + 1e-12


def test_a_dance_mixes_figures_and_varies_with_the_seed(seed):
    a = mv.build(seed, {"name": "dance", "beats": 16}, log=lambda s: None,
                 seed=1)
    b = mv.build(seed, {"name": "dance", "beats": 16}, log=lambda s: None,
                 seed=2)
    a2 = mv.build(seed, {"name": "dance", "beats": 16}, log=lambda s: None,
                  seed=1)
    segs = [(s["figures"], s["arms"]) for s in a["segments"]]
    assert segs == [(s["figures"], s["arms"]) for s in a2["segments"]]
    assert segs != [(s["figures"], s["arms"]) for s in b["segments"]]
    assert len(segs) >= 4 and sum(s[1] is not None for s in segs) >= 2
    assert all(x != y for x, y in zip(segs, segs[1:]))
    assert a["segments"][-1]["beats"][1] == 16


# ------------------------------------------------------------ capability

def test_each_body_is_offered_what_it_can_do(seed, hiyori, lafei):
    assert mv.offered(seed) == list(mv.ORDER)
    # lafei's left hand never crosses to the middle: no clap; neither
    # hand gets above her shoulder with the arm alone: no cheer.
    lo = mv.offered(lafei)
    assert "clap" not in lo and "cheer" not in lo and "sway" in lo
    txt = mv.prompt_lines(lafei)
    assert "clap:" not in txt and "cheer:" not in txt and "dance:" in txt
    # Hiyori claps in her raised drawing; her arm alone lifts no fist.
    hc = mv.caps(hiyori)
    assert hc["clap"]["ok"] and hc["clap"]["variant"]["both"] == 1
    assert not hc["cheer"]["ok"]
    assert "cheer:" not in mv.prompt_lines(hiyori)
    assert "knees" not in mv.prompt_lines(hiyori)
    assert "knees" in mv.prompt_lines(seed)
    p = pz.parser_for(lafei, moves=True)
    p.feed('[move {"name": "clap"}] Hi.')
    p.close()
    assert p.tags == [] and "not a motion this body can do" in p.warnings[0]


def test_live2d_degrades_a_side_and_skips_what_it_cannot_do(lafei, hiyori):
    logs = []
    assert mv.build(lafei, {"name": "clap"}, log=logs.append) is None
    assert "not on this body" in logs[0]
    logs = []
    w = mv.build(hiyori, {"name": "wave", "side": "both"},
                 log=logs.append)
    assert w["parts"] == {"PartArmA": 0.0, "PartArmB": 1.0}
    assert w["parts_from"] == {"PartArmA": 1.0, "PartArmB": 0.0}
    assert set(w["abs"]) <= mv._arm_set(hiyori)
    # A Live2D arm channel never jumps: within L2D_RATE spans a second.
    for p_, arr in w["abs"].items():
        e = hiyori.params[p_]
        step = np.abs(np.diff(arr)).max()
        assert step <= mv.L2D_RATE * (e["max"] - e["min"]) * w["dt"] + 1e-6


# ------------------------------------------------------------------ play

def _tok(add=None, abs_=None, t=0.0, dur=2.0, ramp=0.5, dt=0.1, **kw):
    n = int(round(dur / dt)) + 1
    return {"t": t, "type": "move", "duration": dur, "ramp_in": ramp,
            "ramp_out": ramp, "dt": dt,
            "add": {p: [float(f(i * dt)) for i in range(n)]
                    for p, f in (add or {}).items()},
            "abs": {p: [float(f(i * dt)) for i in range(n)]
                    for p, f in (abs_ or {}).items()},
            "step": [], "parts": {}, "parts_from": {}, **kw}


def test_layer_adds_over_the_idle_blends_absolutes_and_hands_over():
    lay = mv.MoveLayer(FPS, {"a": 0.0, "b": 0.0, "c": 5.0})
    lay.commit(_tok(add={"a": lambda t: 2.0}, abs_={"b": lambda t: 10.0}))
    lay.commit(_tok(abs_={"b": lambda t: -10.0}, t=1.0, dur=1.5))
    vals = []
    for f in range(int(3.0 * FPS)):
        idle = math.sin(f / 7.0)            # the idle under it, moving
        over, _ = lay.apply(f, {"a": idle, "b": idle, "c": 5.0}, None)
        vals.append((idle, over["a"] if "a" in over else idle, over["b"],
                     over["c"]))
    v = np.array(vals)
    # c untouched; a: the idle plus the offset (the idle keeps moving).
    assert (v[:, 3] == 5.0).all()
    mid = int(0.75 * FPS)
    assert v[mid, 1] == pytest.approx(v[mid, 0] + 2.0)
    assert np.std(v[int(0.6 * FPS):int(0.9 * FPS), 1]) > 0.05
    # b: blended in from the idle, then the later token takes it over.
    assert v[0, 2] == pytest.approx(v[0, 0], abs=1e-6)
    assert v[mid, 2] == pytest.approx(10.0)
    assert v[int(1.75 * FPS), 2] == pytest.approx(-10.0)
    # Nothing jumps: the largest step is the ease's (min-jerk peak, 1.875
    # x 20 over 15 frames), not a cut.
    assert np.abs(np.diff(v[:, 2])).max() < 1.875 * 20 / 15 + 0.1
    # After the last token, the idle again, and the layer is idle.
    assert v[-1, 2] == pytest.approx(v[-1, 0])
    assert not lay.busy()


def test_a_pose_during_a_move_takes_its_parts_and_gives_them_back():
    from flow.pose import PoseLayer
    mvl = mv.MoveLayer(FPS, {"x": 0.0})
    pl = PoseLayer(FPS, {"x": 0.0})
    mvl.commit(_tok(abs_={"x": lambda t: 4.0 * math.sin(2 * math.pi * t)},
                    dur=6.0, dt=0.05))
    pl.commit({"t": 2.0, "type": "pose", "params": {"x": 20.0}, "parts": {},
               "step": [], "duration": 0.3, "hold": 1.0, "release": 0.5})
    out = []
    for f in range(int(6 * FPS)):
        over, parts = mvl.apply(f, {"x": 0.0}, None)
        under = over["x"]
        over, parts = pl.apply(f, over, parts, mvl.holding)
        out.append((under, over["x"]))
    o = np.array(out)
    held = o[int(2.4 * FPS):int(3.2 * FPS)]
    # Held: no life from the move's joint under it, only the hold's
    # settle (SETTLE of the way toward what is under it, eased).
    assert (np.abs(held[:, 1] - 20.0) <= pz.SETTLE * np.abs(
        held[:, 0] - 20.0) + 1e-6).all()
    # After the release the move's own curve again.
    after = o[int(4.0 * FPS):int(5.0 * FPS)]
    assert np.allclose(after[:, 0], after[:, 1])
    assert np.ptp(after[:, 1]) > 4.0


def test_vrm_hands_never_pass_through_during_clap_twirl_or_a_dance(seed):
    """Every played frame (keyframes, interpolation, envelope) of a clap,
    a twirl, a sway (the hanging arms) and a dance on Seed-san stays within
    the collision model's tolerance."""
    from flow import vrmbody as vb
    for spec in ({"name": "clap"}, {"name": "twirl"}, {"name": "sway"},
                 {"name": "dance", "beats": 12, "tempo": 120}):
        rec = mv.build(seed, spec, log=lambda s: None, seed=5)
        tok = mv.tokens([dict(rec, start=0.0)])[0]
        lay = mv.MoveLayer(FPS, seed.default_values())
        lay.commit(tok)
        worst = 0.0
        for f in range(0, int(rec["seconds"] * FPS), 2):
            over, _ = lay.apply(f, dict(seed.default_values()), None)
            worst = max(worst, max(seed.penetration(over).values()))
        assert worst <= vb.PEN_TOL + 0.005, (spec, worst)
    # The clap's hands do meet: the palms within a hand's width.
    rec = mv.build(seed, {"name": "clap"}, log=lambda s: None)
    i = int(round(2 * 60 / rec["tempo"] / rec["dt"]))   # on beat 2
    v = {**seed.default_values(), **{p: a[i] for p, a in rec["abs"].items()}}
    pal = seed.palms(v)
    assert np.linalg.norm(pal["L"] - pal["R"]) < 0.35


def test_vrm_state_carries_the_hips_only_while_a_move_writes_them(seed):
    st = seed.state(seed.default_values())
    assert "hips" not in st
    st2 = seed.state({**seed.default_values(), "hips.y": -0.05})
    assert st2["hips"][1] == pytest.approx(-0.05 * seed.hu, abs=1e-5)
    assert {k: v for k, v in st2.items() if k != "hips"} == st


# ---------------------------------------------------------- the pipeline

def test_director_runtime_and_stream_play_a_move_alike(tmp_path, capsys):
    """Body mode with a [move]: actions carry the solved clip, the runtime
    replays the printed command bit for bit, the stream draws the same
    frames from the same tokens; the same reply without the marker plans
    the bytes it did with moves off."""
    from flow.runtime import main as runtime_main
    from tests.test_live import _stream_plan
    from tests.test_pose import _body_rig
    from flow import stream as fs
    model, rig, bm = _body_rig(tmp_path / "x")
    names = mv.offered(bm)
    assert "sway" in names
    arm = next((n for n in ("wave", "clap", "twirl") if n in names), None)
    mark = '[move {"name": "sway", "beats": 4}]' + (
        f'[move {{"name": "{arm}", "beats": 4}}]' if arm else "")
    reply = tmp_path / "reply.txt"
    reply.write_text(f"Let's go! {mark} Woo, nice.", encoding="utf-8")
    out = tmp_path / "run"
    args = ["--model-json", str(model), "--rig", str(rig), "--backend",
            "file", "--reply-file", str(reply), "--out", str(out),
            "--probe-dir", str(tmp_path), "--seed", "2", "--body-targets"]
    assert dr.main(args) == 0
    log = capsys.readouterr().out
    assert "move: sway at" in log
    a = json.loads((out / "actions.json").read_text())
    mvs = [x["solved"] for x in a["actions"] if x["tag"] == "move"]
    assert mvs and a["seconds"] >= mv.end_time(mvs)
    blob = json.loads((out / "tokens.json").read_text())
    mt = [k for k in blob["tokens"] if k["type"] == "move"]
    assert len(mt) == len(mvs)
    plan = np.load(out / "plan.npz", allow_pickle=True)
    pid = list(plan["pid"])
    tilt = plan["traj"][:, pid.index("ParamBodyAngleZ")]
    assert np.ptp(tilt) > 0.2
    import shlex
    cmd = [ln for ln in log.splitlines() if "--tokens-in" in ln][0]
    argv = shlex.split(cmd.strip())
    argv = argv[argv.index("flow.runtime") + 1:]
    k = argv.index("--out")
    argv = argv[:k] + argv[k + 2:]
    assert runtime_main(argv + ["--dump-only", "--dump",
                                str(tmp_path / "re.npz")]) == 0
    re_ = np.load(tmp_path / "re.npz", allow_pickle=True)
    np.testing.assert_array_equal(re_["traj"], plan["traj"])
    er = fs.EventRig(str(model), str(rig), "default", None, None, FPS,
                     str(tmp_path), log=lambda s: None)
    traj, frames = _stream_plan(er, blob, len(plan["traj"]))
    np.testing.assert_array_equal(traj, plan["traj"])
    # Without the marker: moves on or off, the same bytes.
    reply.write_text("Let's go! Woo, nice.", encoding="utf-8")
    outs = []
    for flag in ([], ["--no-body-moves"]):
        o = tmp_path / f"plain{len(flag)}"
        assert dr.main(args[:-6] + [str(o)] + args[-5:] + flag) == 0
        outs.append(o)
    capsys.readouterr()
    for f in ("tokens.json", "subs.json"):
        assert (outs[0] / f).read_bytes() == (outs[1] / f).read_bytes()
    p0 = np.load(outs[0] / "plan.npz", allow_pickle=True)
    p1 = np.load(outs[1] / "plan.npz", allow_pickle=True)
    for k_ in ("traj", "mask", "part_opacity"):
        np.testing.assert_array_equal(p0[k_], p1[k_])


def test_vrm_follower_plays_the_live2d_sides_moves_on_its_own_body(seed):
    import argparse
    from flow import stream as fs
    from flow import vrm as fv
    rig = fv.VRMRig(seed, log=lambda s: None)
    stream = fs.MotionStream(rig, seed=0, log=lambda s: None)
    a = argparse.Namespace(seed=0, lipsync="off", mouth_gain=1.0)
    logs = []
    fol = fv.Follower(a, seed, rig, stream, log=logs.append)
    acts = [{"t": 0.2, "tag": "move", "arg": {"name": "clap", "beats": 4},
             "at": 3}]
    fol.sentence(acts, 1.0, 30, "Yay!", None, 1.0)
    mt = [k for k in stream.pending if k["type"] == "move"]
    assert len(mt) == 1 and mt[0]["t"] == pytest.approx(1.2)
    assert "arms" in mt[0]["groups"]
    assert any("clap" in s for s in logs)
    for _ in range(int(2.5 * FPS)):
        over, _p = stream.frame()
    assert stream.moves.holding


def test_live_performer_puts_a_move_on_the_stream(tmp_path):
    import argparse
    from flow import live as lv
    from flow import stream as fs
    from tests.test_live import FakeClock, SpeakingTTS
    from tests.test_pose import _body_rig
    model, rig_path, bm = _body_rig(tmp_path / "x")
    rig = fs.EventRig(str(model), str(rig_path), "default", None, None, FPS,
                      str(tmp_path), log=lambda s: None)
    clock = FakeClock(2.0)
    hub = lv.Hub(clock)
    stream = fs.MotionStream(rig, seed=0, autonomic=False,
                             log=lambda s: None)
    a = argparse.Namespace(seed=0, lead=0.25, gap=0.12, lipsync="off",
                           mouth_gain=1.0, cjk_rate=dr.CJK_RATE,
                           latin_rate=dr.LATIN_RATE, mood="auto",
                           backend="deepseek")
    run = {"system": "", "labels": {}, "offered": None, "mode": "body",
           "body": bm, "core": None, "images": False,
           "max_tag": pz.MAX_MARKER,
           "parser": lambda: pz.parser_for(bm, moves=True)}
    reply = ('Let\'s dance! [move {"name": "dance", "beats": 8}] Come on. '
             '[move {"name": "dance", "beats": 8}] Again!')
    logs = []
    p = lv.Performer(a, run, rig, stream, clock, hub, dr.TextBackend(reply),
                     SpeakingTTS(clock), log=logs.append)
    p.turn("dance for me")
    mt = [k for k in stream.pending if k["type"] == "move"]
    assert len(mt) == 2
    au = [m for m in hub.history if m["type"] == "audio"]
    assert mt[0]["t"] >= au[0]["t"] - 1e-6
    # Two sentences, two different dances (each marker its own seed).
    assert mt[0]["add"] != mt[1]["add"]
    rep = [m for m in hub.history if m["type"] == "reply"][0]
    assert any("moves" in s for s in rep["sentences"])
    assert any("dance 8 beats" in s for s in logs)
    tilt = []
    for _ in range(int((mt[0]["t"] + 3.0) * FPS)):
        over, _p = stream.frame()
        tilt.append(over.get("ParamBodyAngleZ", 0.0))
    assert np.ptp(tilt) > 1.0


def test_avatar_both_starts_the_vrms_move_with_the_live2d_one(tmp_path,
                                                             seed):
    """--avatar both: the follower builds its moves before the sentence's
    time is taken, as the Live2D side does, so a dance at the start of a
    sentence starts on both bodies at once, the same dance."""
    import argparse
    from flow import live as lv
    from flow import stream as fs
    from flow import vrm as fv
    from tests.test_live import FakeClock, SpeakingTTS
    from tests.test_pose import _body_rig
    model, rig_path, bm = _body_rig(tmp_path / "x")
    rig = fs.EventRig(str(model), str(rig_path), "default", None, None, FPS,
                      str(tmp_path), log=lambda s: None)
    clock = FakeClock(2.0)
    hub = lv.Hub(clock)
    stream = fs.MotionStream(rig, seed=0, autonomic=False,
                             log=lambda s: None)
    a = argparse.Namespace(seed=0, lead=0.25, gap=0.12, lipsync="off",
                           mouth_gain=1.0, cjk_rate=dr.CJK_RATE,
                           latin_rate=dr.LATIN_RATE, mood="auto",
                           backend="deepseek")
    run = {"system": "", "labels": {}, "offered": None, "mode": "body",
           "body": bm, "core": None, "images": False,
           "max_tag": pz.MAX_MARKER,
           "parser": lambda: pz.parser_for(bm, moves=True)}
    p = lv.Performer(a, run, rig, stream, clock, hub,
                     dr.TextBackend('[move {"name": "dance", "beats": 8}] '
                                    "Let's dance!"),
                     SpeakingTTS(clock), log=lambda s: None)
    vrig = fv.VRMRig(seed, log=lambda s: None)
    vstream = fs.MotionStream(vrig, seed=0, log=lambda s: None)
    p.followers.append(fv.Follower(a, seed, vrig, vstream,
                                   log=lambda s: None))
    p.turn("dance!")
    m2 = [k for k in stream.pending if k["type"] == "move"]
    mv_ = [k for k in vstream.pending if k["type"] == "move"]
    assert len(m2) == 1 and len(mv_) == 1
    assert mv_[0]["t"] == pytest.approx(m2[0]["t"])
    au = [m for m in hub.history if m["type"] == "audio"]
    # At the first word (the clip's quiet lead-in before it).
    assert au[0]["t"] <= m2[0]["t"] <= au[0]["t"] + 0.25
    # The same dance: the same figures on both bodies.
    assert mv_[0]["name"] == m2[0]["name"] == "dance"
