"""
flow.gesture_search on synthetic data: no BEAT2, no model but a tiny random
one, no GPU.

The selection is numpy: a planted window with the query's embedding is
chosen; of two equally similar windows the one starting near the pose the
track is in; a window a recent pick overlaps is out, and a pick from a
recording already used pays; a speaker filter and the temperature (0: the
best, else a draw from the seed). The layer: cross-faded windows never step
by more than a minimum-jerk fade allows, a track eases in and out, a pose
that plays an arm takes it from the gesture (ducked before it starts, the
gesture back after, nothing of it in a held pose) while the other arm goes
on, head and torso ride on the idle's. The body: values are clamped and
scaled about the rest pose, a window that goes through the trunk is solved
again by the collision-aware IK or not played. The default path: a turn
planned without the flag is the same bytes with the module made
unimportable, and a stream with a layer that has nothing to play is the
stream without one.

With torch (the tiny encoders are trained a few seconds on a synthetic
export where the text states name the gesture class of each second and the
class is the frequency of the arm's swing): what the library stores is
what the probe's own batch pipeline embeds (the same normalisation, the
same tokens), the held-out clips are not in it, a query whose text is class
c retrieves windows that swing at c Hz, the eval reports retrieval far
above chance, and the text states of a sentence are those flow.train.hidden
stores (with transformers: the tiny random model of the other tests).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import pytest

from flow import beat2 as b2
from flow import body as fb
from flow import gesture_search as gs
from flow import probe as pr
from flow import stream as fs
from flow import vrm as fv
from flow import vrmbody as vb
from flow.train import hidden as fh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_BODY = os.path.join(ROOT, "rigs", "seed-san.body.json")
FPS = 30
W = 90
J = vb.JOINTS


@pytest.fixture(scope="module")
def bm():
    return fb.load(SEED_BODY)


def rest_joints(bm):
    return np.array([bm.params[k]["default"] for k in J], float)


# ------------------------------------------------- a library by hand

def toy_library(bm, n=40, dim=8, seed=0, spk=2, clips=4):
    """n windows of 90 frames each in their own segment: embeddings random
    unit vectors, the right elbow bend a constant per window (so a window's
    pose is known), speakers / clips round robin."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    base = rest_joints(bm)
    joints = np.tile(base, (n * W, 1))
    for i in range(n):
        joints[i * W:(i + 1) * W, J.index("forearm.R.bend")] = 10 + 3 * i
    seg_off = np.arange(n + 1) * W
    meta = {"format": gs.FORMAT, "joints": J, "window": W, "fps": 30.0,
            "hop": 15, "dim": dim, "speakers": [f"s{i}" for i in range(spk)],
            "clips": [f"c{i}" for i in range(clips)], "langs": ["en"],
            "seg_ids": [f"seg{i}" for i in range(n)],
            "seg_keys": [f"seg{i}" for i in range(n)],
            "rest": [float(x) for x in base], "rms_ref": 0.05}
    arrays = {"emb": emb, "win_seg": np.arange(n), "win_start": np.zeros(n),
              "start_pose": joints[::W], "end_pose": joints[W - 1::W],
              "joints": joints.astype(np.float16), "seg_off": seg_off,
              "seg_clip": np.arange(n) % clips, "seg_spk": np.arange(n) % spk,
              "seg_lang": np.zeros(n)}
    return gs.Library(arrays, meta)


def chooser_for(lib, bm, **kw):
    p = gs.Params(**{"temperature": 0.0, "topk": 4, "pool": 40, **kw})
    rt = gs.Retarget(bm, lib.rest, p.parts, p.amp, p.nudge_pen)
    return gs.Chooser(lib, rt, p, np.random.default_rng(1)), rt


def test_query_windows_cover_the_stretch():
    assert gs.window_starts(60, 90, 60, 77) == [0]          # shorter: one
    assert gs.window_starts(97, 90, 60, 77) == [0]          # held for 7
    assert gs.window_starts(150, 90, 60, 77) == [0, 60]
    s = gs.window_starts(400, 90, 60, 77)
    assert s[0] == 0 and s[-1] == 310 and max(np.diff(s)) <= 77
    assert min(np.diff(s)) >= 30
    s = gs.window_starts(300, 90, 120, 77)                # gap cap wins
    assert max(np.diff(s)) <= 77 and s[-1] == 210


def test_the_planted_window_is_chosen(bm):
    lib = toy_library(bm)
    ch, _ = chooser_for(lib, bm, cont_w=0.0)
    for w in (3, 17, 31):
        idx, score, sim, _c = ch.rank(lib.emb[w])
        assert idx[0] == w and sim[0] == pytest.approx(1.0, abs=1e-5)
        assert (np.diff(score) <= 1e-9).all()


def test_continuity_prefers_the_pose_near_the_current_one(bm):
    lib = toy_library(bm, n=6)
    # two windows with the same embedding: elbow bend 13 (window 1) and
    # 25 (window 5); the track is at bend 14
    lib.emb[5] = lib.emb[1]
    ch, rt = chooser_for(lib, bm, cont_w=1.0, topk=6)
    cur = rt.rest_values()
    cur[rt.keys.index("forearm.R.bend")] = 14.0
    idx, score, sim, cost = ch.rank(lib.emb[1], 0, cur, rt.palms(cur))
    assert idx[0] == 1 and sim[0] == pytest.approx(sim[1] if idx[1] == 5
                                                   else sim[0])
    k5 = list(idx).index(5)
    assert cost[0] < cost[k5] and score[0] > score[k5]
    cur[rt.keys.index("forearm.R.bend")] = 26.0
    idx, *_ = ch.rank(lib.emb[1], 0, cur, rt.palms(cur))
    assert idx[0] == 5
    # without the cost the order is the cosine's (a tie either way)
    ch0, _ = chooser_for(lib, bm, cont_w=0.0)
    s0 = ch0.rank(lib.emb[1])[2]
    assert s0[0] == pytest.approx(s0[1], abs=1e-6)


def test_no_repeat_memory(bm):
    lib = toy_library(bm, n=30, clips=3)
    ch, _ = chooser_for(lib, bm, cont_w=0.0, memory=3, clip_pen=0.0)
    q = lib.emb[7]
    first = int(ch.rank(q)[0][0])
    ch.remember(first)
    # the same recording's window again would overlap it: out
    lib.win_seg[0] = lib.win_seg[first]       # another window of that segment
    lib.win_start[0] = 30
    idx = ch.rank(q)[0]
    assert first not in idx and 0 not in idx
    # a window of a clip already picked pays; the others do not
    ch2, _ = chooser_for(lib, bm, cont_w=0.0, memory=3, clip_pen=0.5)
    ch2.remember(7)
    idx2, score2, sim2, _c = ch2.rank(lib.emb[7 + 3])
    same = lib.win_clip[idx2] == lib.win_clip[7]
    assert same.any() and (score2[same] < sim2[same] - 0.4).all()
    # the memory forgets
    ch3, _ = chooser_for(lib, bm, memory=2)
    for w in (1, 2, 3):
        ch3.remember(w)
    assert list(ch3.recent) == [2, 3]
    ch3.forget()
    assert not ch3.recent


def test_filters_and_temperature(bm):
    lib = toy_library(bm, n=30, spk=3)
    ch, _ = chooser_for(lib, bm, speakers=("S1",), cont_w=0.0)
    idx = ch.rank(lib.emb[0])[0]
    assert (lib.win_spk[idx] == 1).all()
    with pytest.raises(SystemExit):
        chooser_for(lib, bm, speakers=("nobody",))
    # temperature 0: always the best; else a draw, from the seed
    ch0, _ = chooser_for(lib, bm, cont_w=0.0)
    score = np.array([0.9, 0.85, 0.5])
    assert {ch0.draw(score) for _ in range(20)} == {0}
    picks = []
    for seed in (1, 2):
        c, _ = chooser_for(lib, bm, temperature=0.05)
        c.rng = np.random.default_rng(seed)
        picks.append([c.draw(score) for _ in range(40)])
    assert picks[0] != picks[1]
    assert set(picks[0]) <= {0, 1, 2} and picks[0].count(2) < 5
    again, _ = chooser_for(lib, bm, temperature=0.05)
    again.rng = np.random.default_rng(1)
    assert [again.draw(score) for _ in range(40)] == picks[0]


# ------------------------------------------------------------ the body

def test_values_are_scaled_about_the_rest_and_clamped(bm):
    rest = rest_joints(bm)
    rt = gs.Retarget(bm, rest, "hands", amp=0.5)
    g = rest.copy()
    g[J.index("upper_arm.L.elev")] = 108.0
    v = rt.apply(g)
    c = rt.keys.index("upper_arm.L.elev")
    d = bm.params["upper_arm.L.elev"]["default"]
    assert v[c] == pytest.approx(d + 0.5 * (108.0 - d))
    big = gs.Retarget(bm, rest, "hands", amp=5.0)
    g[J.index("upper_arm.L.elev")] = 160.0
    g[J.index("forearm.R.bend")] = 140.0
    v = big.apply(g)
    assert (v >= big.lo - 1e-9).all() and (v <= big.hi + 1e-9).all()
    assert v[big.keys.index("upper_arm.L.elev")] == 170.0
    # head and torso: the deviation from the library's median pose
    full = gs.Retarget(bm, rest, "full", amp=1.0)
    g = rest.copy()
    g[J.index("head.yaw")] = rest[J.index("head.yaw")] + 12.0
    v = full.apply(g)
    assert v[full.keys.index("head.yaw")] == pytest.approx(12.0)
    assert full.kinds[full.keys.index("head.yaw")] == "add"
    assert "head.yaw" not in gs.Retarget(bm, rest, "hands").keys


def test_a_window_plays_on_another_vrm_body():
    """The joints are the same names on every VRM body: the girl's body
    (other proportions, other limits) takes the seed-san window, clamped to
    its own, and its collision model is the one that is asked."""
    girl = fb.load(os.path.join(ROOT, "rigs", "three-vrm-girl.body.json"))
    seed = fb.load(SEED_BODY)
    lib = toy_library(seed, n=3)
    lib.joints[:W, J.index("upper_arm.L.elev")] = 175.0      # past any limit
    g = gs.GestureSearch(lib, None, girl, None, gs.Params(amp=1.0))
    V, info = g.prepare(0)
    c = g.rt.keys.index("upper_arm.L.elev")
    assert V[:, c].max() == girl.params["upper_arm.L.elev"]["max"]
    assert (V >= g.rt.lo - 1e-9).all() and (V <= g.rt.hi + 1e-9).all()
    pal = g.rt.palms(V[0])
    assert pal.shape == (2, 3) and np.isfinite(pal).all()
    assert g.rt.palms(V[0])[0, 1] != gs.Retarget(
        seed, lib.rest).palms(V[0])[0, 1]               # its own proportions


def through_the_trunk(bm):
    """Both forearms across the chest, the hands through each other and
    the trunk: found, then checked with the body's own collision model."""
    best = None
    for plane in (60, 90, 120, 130):
        for bend in (60, 100, 140):
            for el in (30, 60, 90):
                v = bm.default_values()
                for s in "LR":
                    v.update({f"upper_arm.{s}.elev": el,
                              f"upper_arm.{s}.plane": plane,
                              f"forearm.{s}.bend": bend})
                pen = bm.penetration(v)
                m = max(pen.values())
                if best is None or m > best[0]:
                    best = (m, v)
    return best


def test_a_window_through_the_body_is_solved_again_or_not_played(bm):
    m, v = through_the_trunk(bm)
    assert m > 0.15, m
    lib = toy_library(bm, n=2)
    for i in range(2):
        for k in vb.JOINTS:
            lib.joints[i * W:(i + 1) * W, J.index(k)] = v[k]
    lib.start_pose = lib.joints[::W].astype(np.float32)
    p = gs.Params(temperature=0.0)
    g = gs.GestureSearch(lib, None, bm, None, p)
    got = g.prepare(0)
    # solved again (the contact fix) so that it clears the skip threshold,
    # or refused: never played as it was
    if got is not None:
        V, info = got
        assert info["pen"] > p.nudge_pen and info["fixed"] > 0
        assert g.rt.pen(V) <= p.skip_pen + 1e-6
    else:
        assert g.stats["skipped"] == 1
    clean = gs.GestureSearch(toy_library(bm, n=2), None, bm, None, p)
    V, info = clean.prepare(0)
    assert info["fixed"] == 0 and clean.rt.pen(V) <= p.nudge_pen
    # a track blended from clean windows is checked again
    assert clean.rt.pen(np.vstack([V, V])) <= p.nudge_pen


# ------------------------------------------------------------ the layer

def flat_window(rt, n=W, **vals):
    V = np.tile(rt.rest_values(), (n, 1))
    for k, x in vals.items():
        V[:, rt.keys.index(k)] = x
    return V


def play_layer(layer, key, n, defaults):
    out = []
    for f in range(n):
        over, _p = layer.apply(f, dict(defaults), None)
        out.append(over[key])
    return np.array(out)


def test_windows_cross_fade_without_jumps(bm):
    rt = gs.Retarget(bm, rest_joints(bm), "hands")
    p = gs.Params()
    k = "upper_arm.R.elev"
    items = [(0, flat_window(rt, **{k: 30.0})),
             (60, flat_window(rt, **{k: 90.0})),
             (120, flat_window(rt, **{k: 40.0}))]
    V = gs.blend_items(items, W, p.fade * FPS, 0, 220)
    c = rt.keys.index(k)
    assert V[0, c] == 30.0 and V[-1, c] == 40.0
    # the fades are centred between the windows (frames 75 and 135), a
    # minimum-jerk fade of 12 frames: peak step 1.875 x swing / 12
    assert 30.0 < V[70, c] < 40.0 or V[70, c] == pytest.approx(30.0, abs=3)
    assert V[75, c] == pytest.approx(60.0, abs=1.0)
    step = np.abs(np.diff(V[:, c])).max()
    assert step <= 1.875 * 60.0 / 12 + 0.5, step
    lay = gs.GestureLayer(FPS, {k: 8.0}, rt.limits())
    lay.commit(gs.token_of(rt, V, 0, FPS, p, []))
    y = play_layer(lay, k, 230, {k: 8.0})
    assert y[0] == 8.0 and y[-1] == 8.0 and not lay.busy()
    assert np.abs(np.diff(y)).max() <= 1.875 * 60.0 / 12 + 0.5
    # eased in and out: the envelope is a minimum-jerk ramp over 0.4 s
    ramp = int(p.ramp * FPS)
    assert abs(y[ramp // 2] - 8.0) < 0.6 * abs(30.0 - 8.0) + 1
    assert y[ramp + 2] == pytest.approx(30.0, abs=0.7)
    assert np.abs(np.diff(y[:ramp + 1])).max() < 1.875 * 22.0 / ramp + 0.5


def test_a_chain_of_tracks_takes_over_without_a_jump(bm):
    rt = gs.Retarget(bm, rest_joints(bm), "hands")
    p = gs.Params()
    k = "forearm.L.bend"
    a = gs.token_of(rt, flat_window(rt, 90, **{k: 100.0}), 0, FPS, p, [])
    b = gs.token_of(rt, flat_window(rt, 90, **{k: 20.0}), 40, FPS, p, [])
    lay = gs.GestureLayer(FPS, {k: 14.0}, rt.limits())
    lay.commit(a)
    lay.commit(b)
    y = play_layer(lay, k, 150, {k: 14.0})
    # the later token takes over as it eases in: no dip toward the idle
    assert np.abs(np.diff(y)).max() < 1.875 * 86.0 / (p.ramp * FPS) + 0.5
    assert y[30] == pytest.approx(100.0, abs=0.5)
    assert y[40:55].min() > 19.9 and y[60] == pytest.approx(20.0, abs=0.5)
    assert y[-1] == 14.0


def test_head_and_torso_ride_on_the_idle(bm):
    rt = gs.Retarget(bm, rest_joints(bm), "full")
    k = "head.yaw"
    V = flat_window(rt, 60, **{k: 10.0})
    lay = gs.GestureLayer(FPS, {}, rt.limits())
    lay.commit(gs.token_of(rt, V, 0, FPS, gs.Params(), []))
    over, _ = lay.apply(30, {k: 7.0}, None)
    assert over[k] == pytest.approx(17.0)          # the idle's 7 + the 10
    lay2 = gs.GestureLayer(FPS, {}, rt.limits())
    lay2.commit(gs.token_of(rt, flat_window(rt, 60, **{k: 80.0}), 0, FPS,
                            gs.Params(), []))
    assert lay2.apply(30, {k: 7.0}, None)[0][k] == bm.params[k]["max"]


def seed_stream(bm, seed=1):
    rig = fv.VRMRig(bm, log=lambda s: None)
    return rig, fs.MotionStream(rig, seed=seed, idle=True,
                                log=lambda s: None)


def pose_token(t, params, dur=0.5, hold=1.0, release=0.5):
    return {"t": t, "type": "pose", "role": "pose", "params": params,
            "parts": {}, "parts_from": {}, "step": [], "duration": dur,
            "hold": hold, "release": release, "groups": ["arms"],
            "variant": "", "src": "director"}


def frames_of(stream, n):
    return [stream.frame()[0] for _ in range(n)]


def test_a_pose_takes_the_arm_and_the_gesture_resumes(bm):
    rt = gs.Retarget(bm, rest_joints(bm), "hands")
    p = gs.Params()
    R, L = "upper_arm.R.elev", "upper_arm.L.elev"
    V = flat_window(rt, 400, **{R: 60.0, L: 70.0})
    tok = gs.token_of(rt, V, 0, FPS, p, [])
    pose = pose_token(4.0, {R: 120.0})
    n = 300
    runs = {}
    for name, toks, layer in (("both", [tok, pose], True),
                              ("pose", [pose], False),
                              ("gesture", [tok], True)):
        rig, st = seed_stream(bm)
        if layer:
            st.gsearch = gs.GestureLayer(FPS, dict(zip(rig.pids,
                                                       rig.dflt.tolist())),
                                         rt.limits())
        st.direct(toks)
        runs[name] = frames_of(st, n)
    f_hold = int(5.0 * FPS)             # in the middle of the hold
    f_after = int(8.0 * FPS)            # long after the release
    r = lambda name, key, f: runs[name][f][key]          # noqa: E731
    # gesturing before the marker, the arm out of the rest pose
    assert abs(r("both", R, 90) - 60.0) < 1.0
    # in the hold: exactly the pose, as without any gesture (no leak)
    assert r("both", R, f_hold) == pytest.approx(r("pose", R, f_hold),
                                                 abs=1e-6)
    assert 100.0 < r("both", R, f_hold) <= 120.0      # (it settles a little)
    # the other arm never stops
    assert abs(r("both", L, f_hold) - 70.0) < 1.0
    # and after the release the right arm is back in the gesture
    assert abs(r("both", R, f_after) - 60.0) < 1.0
    assert r("pose", R, f_after) < 20.0
    # nothing jumps on the way: outside the pose's own stroke (4.0-4.5 s,
    # 112 deg in 15 frames) no step is more than a 0.4 s minimum-jerk ease
    # of the 52 deg between the rest and the gesture (8.1 deg / frame)
    y = np.array([f[R] for f in runs["both"]])
    d = np.abs(np.diff(y))
    d[int(4.0 * FPS) - 1:int(4.5 * FPS) + 1] = 0
    assert d.max() < 9.0, d.max()
    # the gesture ducked before the marker started (not at its start)
    assert r("both", R, int(3.9 * FPS)) < r("both", R, 90) - 20


def test_a_move_that_plays_the_hands_ducks_the_gesture(bm):
    rt = gs.Retarget(bm, rest_joints(bm), "hands")
    rig, st = seed_stream(bm)
    st.gsearch = gs.GestureLayer(FPS, dict(zip(rig.pids, rig.dflt.tolist())),
                                 rt.limits())
    lay = st.gsearch
    lay.note([{"type": "move", "t": 2.0, "duration": 4.0, "ramp_in": 0.5,
               "ramp_out": 0.5, "abs": {"upper_arm.L.elev": [1, 2]},
               "add": {"head.pitch": [0, 1]}}])
    assert lay.gate("upper_arm.L.elev", 3.0) == 0.0
    assert lay.gate("upper_arm.L.elev", 1.0) == 1.0
    assert 0.0 < lay.gate("upper_arm.L.elev", 2.3) < 1.0
    assert 0.0 < lay.gate("upper_arm.L.elev", 5.8) < 1.0
    assert lay.gate("upper_arm.L.elev", 8.0) == 1.0
    assert lay.gate("head.pitch", 3.0) == 1.0         # additive: rides on


# ------------------------------------------------------ the default path

def vrm_turn(tmp, reply, hide=False, extra=()):
    from flow import director as dr
    rp = tmp / "reply.txt"
    rp.write_text(reply, encoding="utf-8")
    out = tmp / "run"
    argv = ["--avatar", "vrm", "--vrm-body", SEED_BODY, "--mode", "body",
            "--backend", "file", "--reply-file", str(rp), "--seed", "3",
            "--out", str(out), *extra]
    if hide:
        saved = sys.modules.get("flow.gesture_search")
        sys.modules["flow.gesture_search"] = None      # import fails
        try:
            assert dr.main(argv) == 0
        finally:
            if saved is not None:
                sys.modules["flow.gesture_search"] = saved
            else:
                del sys.modules["flow.gesture_search"]
    else:
        assert dr.main(argv) == 0
    res = {}
    for name in ("tokens.json", "states.json", "subs.json"):
        res[name] = (out / name).read_bytes()
    acts = json.loads((out / "actions.json").read_text(encoding="utf-8"))
    res["actions.json"] = json.dumps(no_clock(acts), sort_keys=True).encode()
    return res


def no_clock(x):
    """x without the wall-clock fields (how long a solve or the model
    took)."""
    if isinstance(x, dict):
        return {k: no_clock(v) for k, v in x.items()
                if not k.endswith("_seconds") or k in ("speech_seconds",
                                                      "seconds")}
    if isinstance(x, list):
        return [no_clock(v) for v in x]
    return x


REPLY = ('Hello there! [pose {"head.tilt": 12}] I was just thinking about '
         'you. [move {"name": "sway", "beats": 4}] Shall we start?')


def test_without_the_flag_the_turn_is_what_it_was(tmp_path_factory):
    a = vrm_turn(tmp_path_factory.mktemp("d1"), REPLY)
    b = vrm_turn(tmp_path_factory.mktemp("d2"), REPLY, hide=True)
    assert a.keys() == b.keys()
    for k in a:
        assert hashlib.sha256(a[k]).digest() == hashlib.sha256(b[k]).digest(), k
    assert b"gsearch" not in a["tokens.json"]
    assert b"gesture_search" not in a["actions.json"]


def test_a_layer_with_nothing_to_play_changes_nothing(bm):
    rt = gs.Retarget(bm, rest_joints(bm), "hands")
    out = []
    for attach in (False, True):
        rig, st = seed_stream(bm, seed=5)
        if attach:
            st.gsearch = gs.GestureLayer(FPS, dict(zip(
                rig.pids, rig.dflt.tolist())), rt.limits())
        st.direct([pose_token(1.0, {"upper_arm.R.elev": 100.0})])
        out.append(json.dumps(frames_of(st, 120), sort_keys=True))
    assert out[0] == out[1]


def test_the_flag_on_a_live2d_avatar_is_ignored(tmp_path, capsys):
    from flow import director as dr
    with pytest.raises(SystemExit):
        dr.main(["--backend", "file", "--out", str(tmp_path),
                 "--gesture-search", str(tmp_path)])      # no rig: usual error
    assert dr.add_gs_args is not None


# ------------------------------------- the encoders (torch), a world

def slot_freqs(x):
    """The dominant frequency (Hz, 1-4) of each 1 s slot of a 30 fps
    signal."""
    out = []
    for a in range(0, len(x) - 29, 30):
        s = x[a:a + 30] - x[a:a + 30].mean()
        out.append(int(np.argmax(np.abs(np.fft.rfft(s))[1:6])) + 1)
    return out


def add_joints(stem, bm):
    """`.joints` for a probe fixture: the left arm swings with the hand's
    x (so a window's class, its frequency, is in the joints too)."""
    rows, npz = fh.read_export(stem)
    z = dict(np.load(npz))
    base = rest_joints(bm).astype(np.float32)
    for r in rows:
        y = z[r["key"] + ".traj"].astype(np.float32)
        hx = np.clip(y[:, pr.CANON.index("hand.L.x")], -1.3, 1.3)
        Jn = np.tile(base, (len(y), 1))
        Jn[:, J.index("forearm.L.bend")] = 50 + 30 * hx
        Jn[:, J.index("upper_arm.L.elev")] = 45 + 20 * hx
        z[r["key"] + ".joints"] = Jn.astype(np.float16)
    np.savez(npz, **z)


@pytest.fixture(scope="module")
def world(tmp_path_factory, bm):
    pytest.importorskip("torch")
    from tests.test_probe import _match_opts, synth_match
    tmp = tmp_path_factory.mktemp("gsw")
    synth_match(str(tmp / "ex"), str(tmp / "hid"), "content")
    add_joints(str(tmp / "ex"), bm)
    o = _match_opts(tmp, variants="text+prosody", save=str(tmp / "enc"),
                    steps=400)
    rep = pr.match_probe(o, log=lambda s: None)
    lib = gs.build(str(tmp / "ex"), str(tmp / "enc"), str(tmp / "lib"),
                   hop=30, device="cpu", log=lambda s: None)
    return {"tmp": tmp, "rep": rep, "lib": lib, "o": o}


def test_the_probe_saves_what_the_library_needs(world):
    d = world["tmp"] / "enc" / "text+prosody"
    cfg = json.loads((d / "config.json").read_text())
    assert (d / "model.pt").exists()
    assert cfg["cols"] == pr.CANON and cfg["window"] == 90
    assert cfg["sel"] == [0] and cfg["D"] == 12 and cfg["L"] == 1
    assert len(cfg["norm"]["mean"]) == 12 == len(cfg["norm"]["sd"])
    assert cfg["hidden"]["meta"]["dim"] == 12
    held = {f"{k}_sp{k}_{c}_1_1" for k in range(6) for c in (3, 4)}
    assert set(cfg["eval_clips"]) == held and held.isdisjoint(
        cfg["train_clips"]) and len(cfg["train_clips"]) == 18
    enc = pr.load_encoders(str(world["tmp"] / "enc"), device="cpu")
    assert enc.W == 90 and enc.cols == pr.CANON
    with pytest.raises(SystemExit) as e:
        pr.load_encoders(str(world["tmp"] / "enc"), "text", device="cpu")
    assert "text+prosody" in str(e.value)       # says what was saved


def test_saving_changes_nothing_else(world, tmp_path):
    """The report with --save is the report without it."""
    from tests.test_probe import _match_opts
    o = _match_opts(world["tmp"], variants="text+prosody", steps=400)
    again = pr.match_probe(o, log=lambda s: None)
    a, b = world["rep"], again
    assert a["variants"][0]["eval"] == b["variants"][0]["eval"]
    assert a["info"] == b["info"]


def held_corpus(world):
    cfg = json.loads((world["tmp"] / "enc" / "text+prosody" /
                      "config.json").read_text())
    stem = str(world["tmp"] / "ex")
    hid = str(world["tmp"] / "hid")
    held = set(cfg["eval_clips"])
    rows, npz = pr.read_rows(stem, hidden=hid)
    rows = [r for r in rows if pr.clip_of(r) in held]
    o = argparse_ns(cfg)
    c = pr.Corpus(rows, npz, hid, o, log=lambda s: None)
    return cfg, c, o, rows, npz


def argparse_ns(cfg):
    import argparse
    return argparse.Namespace(
        window=cfg["window"], layer=cfg["layer_arg"], hop=30,
        center=cfg["center"], lookback=cfg["lookback"],
        max_tokens=cfg["max_tokens"], variants_list=["text+prosody"],
        active_speed=0.3)


def test_the_queries_are_embedded_as_the_probe_embeds(world):
    """Motion windows and conditioning built here (the library's and the
    director's path) are the probe's own batch pipeline's."""
    enc = pr.load_encoders(str(world["tmp"] / "enc"), device="cpu")
    cfg, c, o, rows, npz = held_corpus(world)
    win = c.windows(set(c.clips))
    zm, zc = gs.embed_windows(enc, c, win, o)
    z = np.load(npz)
    by_id = {r["id"]: r for r in rows}
    for i in (0, 3, 11, len(zm) - 1):
        s = c.segs[win["seg"][i]]
        a = int(win["start"][i])
        m = enc.embed_motion(s["y"][a:a + 90][None])
        np.testing.assert_allclose(m[0], zm[i], atol=1e-5)
        r = by_id[s["id"]]
        q = gs.Query(0.0, len(s["y"]), s["h"], s["tt"],
                     z[r["key"] + ".prosody"].astype(np.float32), (0.0, 1.0))
        e = gs.embed_query(enc, q, [a])
        np.testing.assert_allclose(e[0], zc[i], atol=1e-4)


def test_the_library_is_the_training_clips_windows(world):
    lib = world["lib"]
    cfg = json.loads((world["tmp"] / "enc" / "text+prosody" /
                      "config.json").read_text())
    held = set(cfg["eval_clips"])
    assert set(lib.meta["clips"]) == set(cfg["train_clips"])
    assert not (set(lib.meta["clips"]) & held)
    assert lib.meta["held_out_clips"] == sorted(held)
    assert lib.meta["skipped"]["held-out clip"] == 12 * 2
    n_seg = 18 * 2
    per = len(range(0, 360 - 90 + 1, 30))
    assert lib.meta["segments"] == n_seg and len(lib) == n_seg * per
    assert lib.emb.shape == (n_seg * per, 8 * 4) or lib.emb.shape[1] == 32
    assert np.allclose(np.linalg.norm(lib.emb, axis=1), 1, atol=1e-3)
    # what a window stores is the segment's frames
    rows, npz = fh.read_export(str(world["tmp"] / "ex"))
    z = np.load(npz)
    for w in (0, 5, len(lib) - 1):
        sid = lib.meta["seg_ids"][int(lib.win_seg[w])]
        r = next(x for x in rows if x["id"] == sid)
        a = int(lib.win_start[w])
        src = z[r["key"] + ".joints"].astype(np.float32)[a:a + 90]
        np.testing.assert_array_equal(lib.joints_of(w), src)
        np.testing.assert_array_equal(lib.start_pose[w], src[0])
        np.testing.assert_array_equal(lib.end_pose[w], src[-1])
        d = lib.describe(w)
        assert d["segment"] == sid and d["speaker"].startswith("sp")
    # the embedding is the encoder's
    enc = pr.load_encoders(str(world["tmp"] / "enc"), device="cpu")
    w = 7
    r = next(x for x in rows if x["id"] == lib.meta["seg_ids"][
        int(lib.win_seg[w])])
    cm = pr.columns(r["vars"], pr.CANON)
    y = z[r["key"] + ".traj"].astype(np.float32)[:, [cm[k] for k in pr.CANON]]
    a = int(lib.win_start[w])
    np.testing.assert_allclose(enc.embed_motion(y[a:a + 90][None])[0],
                               lib.emb[w], atol=2e-3)       # (stored fp16)
    assert lib.rms_ref == pytest.approx(0.05, rel=0.5)
    # round trip
    back = gs.Library.load(str(world["tmp"] / "lib"))
    assert len(back) == len(lib) and back.meta["seg_ids"] == lib.meta["seg_ids"]
    np.testing.assert_array_equal(back.joints_of(3), lib.joints_of(3))
    with pytest.raises(SystemExit):
        gs.Library.load(str(world["tmp"] / "nothing"))


def held_queries(world, bm, n=24):
    """Queries of the held-out segments (their stored text states and
    recorded prosody) and the ground-truth class of each second."""
    cfg, c, o, rows, npz = held_corpus(world)
    z = np.load(npz)
    idx = fh.load_index(str(world["tmp"] / "hid"))
    out = []
    for r in rows[:n]:
        q = gs.segment_query(r, z, idx, str(world["tmp"] / "hid"),
                             cfg["sel"])
        y = z[r["key"] + ".traj"].astype(np.float32)
        out.append((q, slot_freqs(y[:, pr.CANON.index("hand.L.x")])))
    return out


def test_text_that_names_the_gesture_retrieves_it(world, bm):
    """The held-out query's text says, per second, how fast the arm swings;
    the windows retrieved swing like that (chance: one second in four)."""
    enc = pr.load_encoders(str(world["tmp"] / "enc"), device="cpu")
    lib = world["lib"]
    p = gs.Params(temperature=0.0, cont_w=0.0, memory=0)
    g = gs.GestureSearch(lib, enc, bm, None, p)
    hit = tot = 0
    for q, truth in held_queries(world, bm):
        emb = gs.embed_query(enc, q, [0])
        idx = g.chooser.rank(emb[0])[0]
        got = slot_freqs(lib.joints_of(int(idx[0]))[:, J.index(
            "forearm.L.bend")])
        hit += sum(a == b for a, b in zip(got, truth[:3]))
        tot += 3
    assert tot > 60 and hit / tot > 0.55, (hit, tot)


def test_eval_reports_retrieval_and_statistics(world, tmp_path):
    rep = gs.evaluate(str(world["tmp"] / "ex"), str(world["tmp"] / "enc"),
                      str(world["tmp"] / "lib"), out=str(tmp_path), hop=30,
                      k=16, boot=200, device="cpu", log=lambda s: None)
    d = rep["retrieved_closer_best"]
    assert rep["queries"] > 100 and d["lo"] > 0.55 and d["share"] > 0.65
    assert d["mean_cos_retrieved"] > d["mean_cos_random"]
    t = rep["truth_top1_of_16"]
    assert t["acc"] > 3 * t["chance"] and t["lo"] > t["chance"]
    m = rep["motion"]
    assert set(m) == set(gs.STATS)
    for stat in ("hand_std", "hand_v"):
        # the arm's swing is the class: the retrieved windows' statistic
        # follows the truth's, a random window's does not
        r = m[stat]["retrieved"]["r_with_truth"]["r"]
        rr = m[stat]["random"]["r_with_truth"]["r"]
        assert r > rr + 0.1, (stat, r, rr)
    assert (tmp_path / "report.json").exists()
    assert "retrieved window vs a random" in (tmp_path / "report.txt"
                                              ).read_text()


def test_cli_build_and_eval(world, tmp_path):
    out = tmp_path / "lib2"
    assert gs.main(["build", "--export", str(world["tmp"] / "ex"),
                    "--encoders", str(world["tmp"] / "enc"), "--out",
                    str(out), "--hop", "60", "--device", "cpu"]) == 0
    assert len(gs.Library.load(str(out))) < len(world["lib"])
    assert gs.main(["eval", "--export", str(world["tmp"] / "ex"),
                    "--encoders", str(world["tmp"] / "enc"), "--lib",
                    str(out), "--boot", "50", "--device", "cpu", "--out",
                    str(tmp_path / "ev"), "--k", "8"]) == 0
    assert (tmp_path / "ev" / "report.json").exists()
    # the library must not land where git would take it
    with pytest.raises(SystemExit) as e:
        gs.build(str(world["tmp"] / "ex"), str(world["tmp"] / "enc"),
                 os.path.join(ROOT, "gesture_lib_test"), device="cpu",
                 log=lambda s: None)
    assert "data/" in str(e.value)
    assert not os.path.exists(os.path.join(ROOT, "gesture_lib_test"))


def test_plan_tokens_for_a_held_out_segment(world, bm):
    enc = pr.load_encoders(str(world["tmp"] / "enc"), device="cpu")
    lib = world["lib"]
    g = gs.GestureSearch(lib, enc, bm, None, gs.Params(seed=2))
    q, _truth = held_queries(world, bm, 1)[0]
    toks, rep = g.plan([q])
    assert len(toks) == 1 and toks[0]["type"] == "gsearch"
    t = toks[0]
    V = np.array(t["values"])
    assert V.shape == (t["duration"] * FPS, len(t["keys"])) or \
        abs(len(V) - t["duration"] * FPS) < 1
    assert len(rep[0]["windows"]) >= 4 and (V >= np.array(t["lo"]) - 1e-3).all()
    assert (V[:, :len(gs.ARM_KEYS)] <= np.array(t["hi"][:len(gs.ARM_KEYS)])
            + 1e-3).all()
    # the windows are about 2 s apart on the sentence's clock (never so far
    # that a fade has nothing to fade from), none repeated
    fr = [w["frame"] for w in rep[0]["windows"]]
    assert fr[0] == 0 and all(0 < b - a <= 90 - 12 for a, b in zip(fr, fr[1:]))
    keys = {(w["segment"], w["start"]) for w in rep[0]["windows"]}
    assert len(keys) == len(rep[0]["windows"])
    # no randomness lost to the cache: the same seed plays the same
    g2 = gs.GestureSearch(lib, enc, bm, None, gs.Params(seed=2))
    assert g2.plan([q])[0][0]["values"] == t["values"]
    g3 = gs.GestureSearch(lib, enc, bm, None, gs.Params(seed=3))
    assert g3.plan([q])[0][0]["values"] != t["values"]
    # the random baseline is not the search
    rnd = gs.GestureSearch(lib, enc, bm, None, gs.Params(seed=2)).plan(
        [q], random=True)[0][0]
    assert rnd["values"] != t["values"]
    # played through the stream the track shows up on the body's arm
    rig, st = seed_stream(bm)
    st.gsearch = g.layer(dict(zip(rig.pids, rig.dflt.tolist())))
    st.direct(toks)
    fr_ = frames_of(st, int(t["duration"] * FPS) + 5)
    arm = np.array([f["forearm.L.bend"] for f in fr_])
    assert arm.max() - arm.min() > 10


# ------------------------------------------------- a reply, as spoken

def fake_text(text):
    """Token states like the synthetic fixture's: a one-hot class per
    word, the words' character spans."""
    import re
    ms = list(re.finditer(r"\S+", text))
    h = np.zeros((len(ms), 1, 12), np.float16)
    for i in range(len(ms)):
        h[i, 0, i % 4] = 3.0
    return h, np.array([(m.start(), m.end()) for m in ms], np.int32)


def said_of(sentences, gap=0.3, cps=12.0, sr=16000):
    """A reply as flow.speech.speak returns it: sentences' characters at
    `cps` characters a second with `gap` s between sentences, a tone burst
    per word as the audio."""
    spoken, spans, t = "", [], 0.3
    for k, s in enumerate(sentences):
        if k:
            spoken += " "
            spans.append((round(t, 4), round(t, 4)))
        for ch in s:
            d = 0.0 if ch == " " else 1.0 / cps
            spans.append((round(t, 4), round(t + d, 4)))
            t += d
        spoken += s
        t += gap
    x = np.zeros(int((t + 0.5) * sr))
    tt = np.arange(len(x)) / sr
    for ch, (a, b) in zip(spoken, spans):
        if ch != " " and b > a:
            i, j = int(a * sr), int(b * sr)
            x[i:j] = 0.2 * np.sin(2 * np.pi * 140 * tt[i:j])
    return spoken, {"spans": spans, "audio": x, "sr": sr, "end": t}


def test_a_reply_is_planned_sentence_by_sentence(world, bm):
    enc = pr.load_encoders(str(world["tmp"] / "enc"), device="cpu")
    lib = world["lib"]
    g = gs.GestureSearch(lib, enc, bm, fake_text, gs.Params(seed=1))
    s1 = "one two three four five six seven eight nine ten eleven twelve."
    s2 = "nine ten eleven twelve."
    spoken, said = said_of([s1, s2], gap=0.4)
    toks, rep = g.plan_reply(spoken, said)
    # sentences closer than a second chain into one track
    assert len(toks) == 1 and toks[0]["type"] == "gsearch"
    assert toks[0]["t"] < 0.3 and toks[0]["t"] * FPS == pytest.approx(
        round(toks[0]["t"] * FPS))
    last = said["spans"][-1][1]
    assert toks[0]["t"] + toks[0]["duration"] == pytest.approx(
        last + 0.2, abs=0.2)
    # windows every 2 s over the first sentence, then the second's
    fr = [w["frame"] for w in rep[0]["windows"]]
    assert fr == sorted(fr) and len(fr) >= 3
    spoken, said = said_of([s1, s2], gap=2.5)
    g2 = gs.GestureSearch(lib, enc, bm, fake_text, gs.Params(seed=1))
    toks2, rep2 = g2.plan_reply(spoken, said)
    assert len(toks2) == 2 and toks2[1]["t"] > toks2[0]["t"] + \
        toks2[0]["duration"]
    # a sentence needs the text model
    with pytest.raises(SystemExit):
        gs.GestureSearch(lib, enc, bm, None, gs.Params()).plan_reply(
            spoken, said)
    # the audio is brought to the library's loudness (rms_ref 0.05-ish)
    p = gs.reply_prosody(said["audio"], said["sr"], 30.0, lib.rms_ref)
    v = p[:, 2] > 0.5
    assert np.median(p[v, 0]) == pytest.approx(lib.rms_ref, rel=1e-3)
    assert lib.rms_ref and gs.reply_prosody(said["audio"], said["sr"],
                                            30.0)[v, 0].min() > 0


def test_the_director_plays_the_search_under_the_markers(world, bm,
                                                         monkeypatch):
    from flow import director as dr
    tmp = world["tmp"]
    sr = 16000
    spoken_x = np.zeros(int(7 * sr))
    t = np.arange(len(spoken_x)) / sr
    for a in np.arange(0.4, 6.4, 0.5):
        i, j = int(a * sr), int((a + 0.35) * sr)
        spoken_x[i:j] = 0.2 * np.sin(2 * np.pi * 150 * t[i:j])
    wav = str(tmp / "said.wav")
    b2.write_wav(wav, spoken_x, sr)
    made = []

    class Fake:
        def __init__(self, model, hidden, sel, load_4bit, device, log):
            made.append((model, hidden, sel))

        def __call__(self, text):
            return fake_text(text)
    monkeypatch.setattr(gs, "TextStates", Fake)
    reply = tmp / "reply.txt"
    reply.write_text('Hello there my friend, how are you doing today? '
                     '[pose {"head.tilt": 12}] I was just thinking about '
                     'you and the weather we had. Shall we start now?',
                     encoding="utf-8")
    common = ["--avatar", "vrm", "--vrm-body", SEED_BODY, "--mode", "body",
              "--backend", "file", "--reply-file", str(reply), "--seed", "3",
              "--tts", "file", "--tts-file", wav]
    base = tmp / "dir_base"
    assert dr.main(common + ["--out", str(base)]) == 0
    out = tmp / "dir_gs"
    assert dr.main(common + [
        "--out", str(out), "--gesture-search", str(tmp / "lib"),
        "--gs-encoders", str(tmp / "enc"), "--gs-model", "some/model",
        "--gs-device", "cpu", "--gs-amp", "0.8", "--gs-seed", "4",
        "--gs-baseline"]) == 0
    assert made and made[0][0] == "some/model" and made[0][2] == [0]
    toks = json.loads((out / "tokens.json").read_text())
    gts = [k for k in toks if k["type"] == "gsearch"]
    assert gts and any(k["type"] == "pose" for k in toks)
    acts = json.loads((out / "actions.json").read_text())
    rep = acts["gesture_search"]
    assert rep["amp"] == 0.8 and rep["seed"] == 4 and rep["tracks"]
    # the other panel is the turn as it is without the flag: the
    # markers-only states are the flag-absent run's
    sa = json.loads((base / "states.json").read_text())["states"]
    sb = json.loads((out / "states_markers.json").read_text())["states"]
    sc = json.loads((out / "states.json").read_text())["states"]
    assert sa == sb and len(sc) == len(sa) and sc != sa
    # the arm moves with the search where no marker holds it
    arm = lambda sts: np.array([s["bones"]["leftLowerArm"][3]    # noqa: E731
                                for s in sts])
    assert np.ptp(arm(sc)) > np.ptp(arm(sa)) + 1e-3
    # without --tts: said so, played without
    nowav = tmp / "dir_nowav"
    assert dr.main(["--avatar", "vrm", "--vrm-body", SEED_BODY, "--mode",
                    "body", "--backend", "file", "--reply-file", str(reply),
                    "--seed", "3", "--out", str(nowav), "--gesture-search",
                    str(tmp / "lib"), "--gs-encoders", str(tmp / "enc"),
                    "--gs-device", "cpu"]) == 0
    assert b"gsearch" not in (nowav / "tokens.json").read_bytes()
    with pytest.raises(SystemExit):
        dr.main(common + ["--out", str(tmp / "x"), "--gesture-search",
                          str(tmp / "lib")])            # needs the encoders


def test_ids_lists_the_held_out_segments(world):
    lines = []
    got = gs.held_out_ids(str(world["tmp"] / "ex"), str(world["tmp"] / "enc"),
                          5, log=lines.append)
    assert len(got) == 24 and all(i.startswith("beat2.") for i in got)
    assert all(("_3_1_1" in i or "_4_1_1" in i) for i in got)
    assert len(lines) == 6 and "24 held-out" in lines[-1]


def test_demo_of_a_held_out_segment(world, bm, tmp_path):
    seg = "beat2.3_sp3_3_1_1/s00"
    a = gs.parse(["demo", "--id", seg, "--export", str(world["tmp"] / "ex"),
                  "--encoders", str(world["tmp"] / "enc"), "--lib",
                  str(world["tmp"] / "lib"), "--out", str(tmp_path), "--vrm",
                  os.path.join(ROOT, "assets", "vrm",
                               "VRM1_Constraint_Twist_Sample.vrm"),
                  "--body", SEED_BODY, "--device", "cpu", "--no-render",
                  "--seconds", "6"])[0]
    res = gs.demo_segment(a, log=lambda s: None)
    assert res["panels"] == ["idle", "random", "retrieved", "truth"]
    d = tmp_path / "beat2.3_sp3_3_1_1__s00"
    lens, arms = set(), {}
    for name in res["panels"]:
        rec = json.loads((d / f"{name}.states.json").read_text())
        lens.add(len(rec["states"]))
        arms[name] = np.array([s["bones"]["leftLowerArm"][3]
                               for s in rec["states"]])
    assert lens == {180}
    assert np.ptp(arms["idle"]) < 1e-9        # the arms rest in the idle
    for name in ("random", "retrieved", "truth"):
        assert np.ptp(arms[name]) > 0.01, name
    assert not np.allclose(arms["random"], arms["retrieved"])
    rep = json.loads((d / "report.json").read_text())
    assert rep["retrieved"][0]["windows"] and rep["spoken"]
    # a segment of the library's clips is said not to be held out
    msgs = []
    a.id = "beat2.0_sp0_0_1_1/s00"
    gs.demo_segment(a, log=msgs.append)
    assert any("not a held-out clip" in m for m in msgs)
    with pytest.raises(SystemExit):
        a.id = "nope"
        gs.demo_segment(a, log=lambda s: None)
    # another avatar of yours: its own body file, not the export's
    a.id = seg
    a.vrm = os.path.join(ROOT, "assets", "vrm", "three-vrm-girl.vrm")
    a.body = None
    a.out = str(tmp_path / "girl")
    gs.demo_segment(a, log=lambda s: None)
    g = json.loads((tmp_path / "girl" / "beat2.3_sp3_3_1_1__s00" /
                    "truth.states.json").read_text())
    assert g["body"].endswith("three-vrm-girl.body.json")
    assert g["vrm"].endswith("three-vrm-girl.vrm") and len(g["states"]) == 180


# ------------------------------------------------ the text states (torch)

from tests.test_beat2 import exported  # noqa: E402,F401  (fixture)


def test_text_states_are_those_flow_train_hidden_stores(exported, tmp_path):
    """The in-process states of a sentence are, to the bit, what
    flow.train.hidden --context none --text spoken --proj stores for it:
    the same tokens, layers, projection, fp16 rounding, and the probe's
    layer selection on top."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tests import tiny_lm
    root, _a, _s = exported
    stem = str(root / "out" / "action")
    rows, _ = fh.read_export(stem)
    lm = tiny_lm.build(str(tmp_path / "tiny-qwen3"),
                       [r["spoken"] for r in rows] * 5)
    o = fh.parse(["--export", stem, "--model", lm, "--layers=-1,1",
                  "--context", "none", "--text", "spoken", "--proj", "8",
                  "--out", str(tmp_path / "h")])
    meta = fh.extract(o, log=lambda s: None)
    assert meta["layers"] == [2, 1]
    idx = fh.load_index(str(tmp_path / "h"))
    ts = gs.TextStates(lm, str(tmp_path / "h"), [1], device="cpu",
                       log=lambda s: None)
    for r in rows[:2]:
        e = idx[r["id"]]
        with np.load(tmp_path / "h" / e["shard"]) as z:
            h = z[e["key"] + ".h"]
            spans = z[e["key"] + ".spoken"]
        got, tok = ts(r["spoken"])
        assert got.dtype == np.float16 and got.shape == (len(h), 1, 8)
        np.testing.assert_array_equal(got[:, 0], h[:, 1])
        np.testing.assert_array_equal(tok, spans)
        assert ts(r["spoken"])[0] is got              # cached
    # a hidden folder from another context is said to differ
    msgs = []
    meta = json.loads((tmp_path / "h" / "meta.json").read_text())
    meta["context"], meta["proj"] = "turn", 0
    os.makedirs(tmp_path / "h2")
    (tmp_path / "h2" / "meta.json").write_text(json.dumps(meta))
    gs.TextStates(lm, str(tmp_path / "h2"), [0], log=msgs.append)
    assert any("--context turn" in m for m in msgs)


def test_a_sentence_is_made_to_look_like_the_librarys_texts():
    """BEAT2's words are what the TextGrid says (no punctuation, lower
    case, if the export's texts are): a TTS sentence is brought to that
    before the text model reads it, its characters' times kept."""
    form = gs.text_form([{"spoken": "hello there how are you"},
                         {"spoken": "nice to meet you i am here"}])
    assert form == {"lower": 1.0, "punct": 0.0}
    mixed = gs.text_form([{"spoken": "Hello, there. How are you?"}])
    assert mixed["lower"] < 0.95 and mixed["punct"] > 0.1
    assert gs.text_form([{"spoken": "你好吗"}]) == {"lower": 0.0,
                                                    "punct": 0.0}
    t, keep = gs.normalise_text("Hello,  World! It's 5-6 \u2026", form)
    assert t == "hello world it's 5-6"
    assert list("Hello,  World! It's 5-6 \u2026"[i] for i in keep[:5]) == \
        list("Hello")
    assert "W" == "Hello,  World! It's 5-6 \u2026"[keep[6]]
    assert gs.normalise_text("Hi, You!", mixed)[0] == "Hi, You!"
    assert gs.normalise_text("Hi, You!", None)[0] == "Hi, You!"
    t, keep = gs.normalise_text("\uff28\uff49 there?", form)       # NFKC
    assert t == "hi there" and len(keep) == len(t)
    # through the queries: the model gets the normalised text, and a token
    # is timed from where its word is spoken in the original
    spoken, said = said_of(["Hello, there World!"], cps=10.0)
    seen = []

    def spy(text):
        seen.append(text)
        return fake_text(text)
    q, = gs.sentence_queries(spoken, said["spans"], said["audio"],
                             said["sr"], spy, form=form)
    assert seen == ["hello there world"]
    t0 = q.t0
    word = spoken.index("World")
    assert q.tt[2] == pytest.approx(said["spans"][word][0] - t0, abs=1e-3)
    assert q.tt[1] == pytest.approx(said["spans"][spoken.index("there")][0]
                                    - t0, abs=1e-3)
    seen.clear()
    gs.sentence_queries(spoken, said["spans"], said["audio"], said["sr"],
                        spy)
    assert seen == ["Hello, there World!"]


def test_sentences_become_queries_on_the_audio_clock():
    spoken, said = said_of(["one two three.", "four five six seven."],
                           gap=0.5)
    qs = gs.sentence_queries(spoken, said["spans"], said["audio"],
                             said["sr"], fake_text, 30.0, 0.3, 0.05)
    assert len(qs) == 2
    q1, q2 = qs
    first = said["spans"][0][0]
    assert q1.t0 == pytest.approx(math.floor((first - 0.3) * 30) / 30)
    assert q1.t0 * 30 == pytest.approx(round(q1.t0 * 30))
    assert q2.t0 > q1.t0 + q1.n / 30 - 0.7 and q2.lang == "en"
    for q in qs:
        assert q.pros.shape == (q.n, 3) and q.h.shape[1:] == (1, 12)
        assert (np.diff(q.tt) >= -1e-6).all()
        assert 0.25 < q.speech[0] < 0.4 or q.speech[0] >= 0.0
        assert q.tt.max() <= q.speech[1] + 1e-3
    # voiced where the tone is, silent in the pad
    assert q1.pros[:5, 2].max() == 0 and q1.pros[:, 2].max() == 1
    # punctuation alone is not a sentence
    qs = gs.sentence_queries("...", [(0, 0.1)] * 3, said["audio"],
                             said["sr"], fake_text)
    assert qs == []


def test_demo_say_renders_the_markers_beside_the_search(world, tmp_path,
                                                        monkeypatch):
    """A free sentence through the director: one plan, two videos side by
    side (Chromium), the audio on it."""
    from tests.test_vrm import _chromium
    _chromium()
    import shutil
    import subprocess
    sr = 16000
    x = np.zeros(int(2.4 * sr))
    t = np.arange(len(x)) / sr
    for a in np.arange(0.2, 2.0, 0.4):
        i, j = int(a * sr), int((a + 0.25) * sr)
        x[i:j] = 0.2 * np.sin(2 * np.pi * 150 * t[i:j])
    wav = str(tmp_path / "said.wav")
    b2.write_wav(wav, x, sr)
    reply = tmp_path / "reply.txt"
    reply.write_text("Hello there, how are you doing today.",
                     encoding="utf-8")

    class Fake:
        def __init__(self, *a, **k):
            pass

        def __call__(self, text):
            return fake_text(text)
    monkeypatch.setattr(gs, "TextStates", Fake)
    tmp = world["tmp"]
    a, rest = gs.parse(["demo", "--say", "hi", "--export", str(tmp / "ex"),
                        "--encoders", str(tmp / "enc"), "--lib",
                        str(tmp / "lib"), "--out", str(tmp_path / "out"),
                        "--device", "cpu", "--size", "64", "--backend",
                        "file", "--reply-file", str(reply), "--vrm-body",
                        SEED_BODY, "--tts", "file", "--tts-file", wav,
                        "--tail", "0.2", "--mode", "body"])
    assert "--mode" in rest and a.gs_model is None      # not --model
    assert gs.demo_say(a, rest, log=lambda s: None) == 0
    d = tmp_path / "out" / "say"
    assert (d / "states_markers.json").exists()
    demo = d / "demo.mp4"
    assert demo.exists() and demo.stat().st_size > 1000
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width",
         "-of", "csv=p=0", str(demo)], capture_output=True, text=True).stdout \
        if shutil.which("ffprobe") else "video,128\naudio"
    assert "video,128" in out and "audio" in out         # 2 x 64 px wide
