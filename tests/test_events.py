"""
flow.events: the fitted event scheduler. Pure numpy, no GL, no sample rigs.

Extraction must call a hold that runs into a clip end censored and a clip
that never moves a group unanimated; the censored fit must undo the bias of
a naive median; the scheduler must not be periodic and must really hold
(zero velocity) between moves; and the token stream must be the whole
interface -- rendering a dumped stream reproduces the run bit for bit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from flow import events as ev
from flow.behaviors import min_jerk
from flow.score import periodic_peak

FPS = 30.0


def _move(a, b, n):
    return a + (b - a) * min_jerk(np.linspace(0, 1, n))


def test_segment_holds_moves_and_censoring():
    x = np.r_[np.zeros(30), _move(0, 0.5, 12), np.full(45, 0.5),
              _move(0.5, -0.2, 15), np.full(20, -0.2)]
    holds, moves = ev.segment(x, FPS)
    # First and last holds touch a clip end: lower bounds only.
    assert [h[2] for h in holds] == [True, False, True]
    assert holds[1][1] == pytest.approx(45 / FPS, abs=3 / FPS)
    complete = [m for m in moves if not m[2]]
    assert len(complete) == 2
    assert complete[0][3][0] == pytest.approx(0.5, abs=1e-6)
    assert complete[1][3][0] == pytest.approx(-0.7, abs=1e-6)
    # A clip that never moves is one hold spanning it, censored both ends.
    holds, moves = ev.segment(np.full(90, 0.3), FPS)
    assert len(holds) == 1 and holds[0][2] and not moves
    assert holds[0][1] == pytest.approx(89 / FPS)


def test_segment_splits_back_to_back_movements():
    # Two movements with no pause between them (a head that swings and
    # swings back) are two moves, not one move of net amplitude ~0.
    x = np.r_[np.zeros(10), _move(0, 0.6, 20)[:-1], _move(0.6, 0.05, 20),
              np.full(10, 0.05)]
    _, moves = ev.segment(x, FPS)
    big = [m for m in moves if abs(m[3][0]) > 0.3]
    assert len(big) == 2 and big[0][5]           # first one chains on
    assert not ev.segment(x, FPS, split=False)[1][0][5]


def test_censored_lognormal_undoes_truncation_bias():
    rng = np.random.default_rng(0)
    true_med, sigma = 2.0, 0.6
    d = rng.lognormal(np.log(true_med), sigma, 400)
    window = rng.uniform(1.0, 5.0, 400)          # clips end where they end
    complete = d[d <= window]
    censored = window[d > window]
    naive = float(np.median(complete))
    med, s = ev.censored_lognormal(complete, censored)
    assert naive < 0.8 * true_med                # the bias being fixed
    assert med == pytest.approx(true_med, rel=0.2)
    assert s == pytest.approx(sigma, rel=0.3)
    assert ev.censored_lognormal([], [3.0, 4.0]) is None
    km = ev.kaplan_meier_median(complete, censored)
    assert km == pytest.approx(true_med, rel=0.25)


def _clip_file(path, clips, pids):
    blob = {"pid": np.array(pids, dtype=object), "fps": np.float32(FPS),
            "n_clips": np.int32(len(clips))}
    for i, c in enumerate(clips):
        blob[f"clip{i}"] = c.astype(np.float32)
        blob[f"mask{i}"] = np.ones(len(pids), bool)
    np.savez(path, **blob)


def test_fit_pools_per_rig_and_skips_holdout(tmp_path):
    pids = ["ParamEyeBallX", "ParamAngleX"]
    rng = np.random.default_rng(1)

    def clip(hold_s):
        seq, v = [], 0.0
        while sum(len(s) for s in seq) < 240:
            nv = float(rng.uniform(-0.6, 0.6))
            seq += [np.full(int(hold_s * FPS), v), _move(v, nv, 9)]
            v = nv
        g = np.concatenate(seq)[:240]
        return np.stack([g, 0.5 * g], 1)

    for name, hold in (("A", 1.0), ("B", 1.4), ("C", 1.2), ("Held", 8.0)):
        _clip_file(tmp_path / f"{name}.npz", [clip(hold) for _ in range(4)]
                   + [np.zeros((200, 2))], pids)
    files = sorted(str(p) for p in tmp_path.glob("*.npz"))
    st = ev.fit(files, holdout={"Held"})
    assert st["rigs"] == ["A", "B", "C"]
    med = st["groups"]["gaze"]["hold"][0]
    assert 0.8 < med < 1.8                       # Held's 8 s never got in
    assert st["per_rig"]["A"]["groups"]["gaze"]["n_unanimated"] == 1
    json.dumps(st)


def test_schedule_is_aperiodic_and_holds_are_still():
    st = ev.load_stats("default")
    roles = list(ev.ALL_ROLES)
    for seed in (0, 1, 2):
        traj, blinks, toks = ev.generate(st, roles, 60.0, FPS,
                                         np.random.default_rng(seed))
        for r, v in traj.items():
            assert len(v) == 1800 and np.abs(v).max() <= 1
            peak, _ = periodic_peak(v.astype(np.float64), FPS)
            assert peak <= 0.6, (seed, r, peak)
        # Outside every gaze move the gaze does not move at all.
        moving = np.zeros(1800, bool)
        for k in toks:
            if k["type"] == "saccade":
                a = int(np.floor(k["t"] * FPS))
                moving[a:int(np.ceil((k["t"] + k["duration"]) * FPS)) + 1] = True
        d = np.abs(np.diff(traj["gaze_x"]))
        held = ~moving[1:]
        assert held.sum() > 600
        assert d[held].max() == 0.0
        assert all(0 <= t < 60 for t in blinks)
        assert {k["type"] for k in toks} >= {"saccade", "head_turn",
                                             "body_shift", "brow"}
    a = ev.generate(st, ["gaze_x"], 20, FPS, np.random.default_rng(0))[0]
    b = ev.generate(st, ["gaze_x"], 20, FPS, np.random.default_rng(1))[0]
    assert np.abs(a["gaze_x"] - b["gaze_x"]).mean() > 1e-3


def test_render_is_minimum_jerk_and_never_retargets_mid_flight():
    toks = [{"t": 0.5, "type": "saccade", "role": "gaze",
             "target": {"gaze_x": 0.6}, "duration": 0.5},
            # arrives mid-flight: must wait for the first move to end
            {"t": 0.7, "type": "saccade", "role": "gaze",
             "target": {"gaze_x": -0.2}, "duration": 0.4}]
    y, _ = ev.render(toks, ["gaze_x"], 2.0, FPS)
    y = y["gaze_x"]
    assert y[:15].max() == 0.0
    assert y[30] == pytest.approx(0.6, abs=1e-6)   # first move completes
    assert y[45:] == pytest.approx(np.full(len(y) - 45, -0.2), abs=1e-6)
    v = np.diff(y)
    assert abs(v[15]) < 0.01 and abs(v[29]) < 0.01  # starts and ends at rest
    assert np.all(v[15:30] >= -1e-7)                # monotone, no kink


def test_gesture_player_fades_and_marks_unkeyed():
    motions = {"a": {"frames": 60, "params": {"ParamArmL": np.linspace(0, 1, 60)},
                     "parts": {"PartA": np.ones(60)}},
               "b": {"frames": 60, "params": {"ParamArmR": np.full(60, 0.5)},
                     "parts": {"PartA": np.zeros(60)}}}
    gp = ev.GesturePlayer(motions, lambda p: True, {"ParamArmL": 0.2}, {},
                          {"ParamArmR": 0.0}, FPS)
    toks = [{"t": 0.0, "type": "gesture", "motion": "a", "start": 0.0,
             "duration": 1.0, "fade": 0.0},
            {"t": 1.0, "type": "gesture", "motion": "b", "start": 0.0,
             "duration": 1.0, "fade": 0.5}]
    fr = gp.frames(toks, 90)
    assert fr[10][0]["ParamArmL"] == pytest.approx(10 / 59)
    # mid-fade: both motions keyed, values between old and new
    assert 0.0 < fr[37][1]["PartA"] < 1.0
    assert gp.keyed[37] == {"ParamArmL", "ParamArmR"}
    # after the fade ArmL is unkeyed and back at its base value
    assert fr[60][0]["ParamArmL"] == pytest.approx(0.2)
    assert gp.keyed[60] == {"ParamArmR"}


def test_gesture_schedule_prefers_no_flips():
    st = ev.load_stats("default")
    st["gesture"]["flip_cost"] = 5.0
    lib = [{"name": n, "seconds": 3.0, "fps": FPS,
            "states": np.full((90, 1), s, np.int8), "vel": np.zeros(90)}
           for n, s in (("a0", 0), ("b0", 0), ("c1", 1))]
    toks = ev.schedule_gestures(st, lib, 120.0, np.random.default_rng(0))
    names = [k["motion"] for k in toks]
    flips = sum((a[-1] == "1") != (b[-1] == "1") for a, b in zip(names, names[1:]))
    assert len(names) > 20 and flips <= 3
    assert all(a != b for a, b in zip(names, names[1:]))


def test_binding_has_no_descriptor_fallback():
    b = ev.bind(["PARAM_ANGLE_X", "ParamBrowLAngle", "ParamFoo"])
    assert b["head_x"][0] == "PARAM_ANGLE_X" and b["brow_la"][2] == "id"
    assert "gaze_x" not in b
    b = ev.bind(["ParamFoo"], override={"gaze_x": ["ParamFoo", -1]})
    assert b["gaze_x"] == ("ParamFoo", -1.0, "override")


def test_events_dump_and_token_replay_are_identical(tmp_path, capsys):
    """--events end to end without GL; --tokens-in reproduces --tokens."""
    from flow.runtime import main
    from l2dprobe.analyze import group_of

    arm = [0, 0, 1, 2.0, 1, 4.0, 0, 6.0, 0]
    swap = [0, 0, 0, 2.9, 0, 0, 3.0, 1, 0, 6.0, 1]
    # A body sway, so that the run adapts the body to the rig's own motion.
    body = [0, 0, 0, 1.0, 5, 0, 2.0, 5, 0, 3.0, -4, 0, 4.5, -4, 0, 6.0, 0]
    motion = {"Version": 3, "Meta": {"Duration": 6.0, "Fps": 30},
              "Curves": [
                  {"Target": "Parameter", "Id": "ParamArmL", "Segments": arm},
                  {"Target": "Parameter", "Id": "ParamBodyAngleX",
                   "Segments": body},
                  {"Target": "Parameter", "Id": "ParamArmSwap",
                   "Segments": swap},
                  {"Target": "PartOpacity", "Id": "PartArmB", "Segments": swap}]}
    (tmp_path / "m.motion3.json").write_text(json.dumps(motion))
    model = tmp_path / "x.model3.json"
    model.write_text(json.dumps({"Version": 3, "FileReferences": {
        "Moc": "x.moc3", "Motions": {"Idle": [{"File": "m.motion3.json"}]}}}))
    pids = ["ParamAngleX", "ParamEyeBallX", "ParamBrowLY", "ParamArmL",
            "ParamArmSwap", "ParamEyeLOpen", "ParamBodyAngleX"]
    n = len(pids)
    rig = tmp_path / "x.npz"
    np.savez(rig, pid=np.array(pids, dtype=object),
             kind=np.array(["continuous"] * n, dtype=object),
             group=np.array([group_of(p) for p in pids], dtype=object),
             minimum=np.array([-30, -1, -1, 0, 0, 0, -10], np.float32),
             maximum=np.array([30, 1, 1, 1, 1, 1, 10], np.float32),
             default=np.array([0, 0, 0, 0, 0, 1, 0], np.float32),
             descriptor=np.zeros((n, 4), np.float32))
    base = ["--model", str(model), "--rig", str(rig), "--events", "default",
            "--seconds", "20", "--seed", "3", "--dump-only",
            "--probe-dir", str(tmp_path)]
    tok = tmp_path / "t.json"
    assert main(base + ["--dump", str(tmp_path / "a.npz"),
                        "--tokens", str(tok)]) == 0
    assert "body adapted to the rig's own motions" in capsys.readouterr().out
    assert main(base + ["--dump", str(tmp_path / "b.npz"),
                        "--tokens-in", str(tok)]) == 0
    a = np.load(tmp_path / "a.npz", allow_pickle=True)
    b = np.load(tmp_path / "b.npz", allow_pickle=True)
    assert str(a["mode"]).startswith("events")
    np.testing.assert_array_equal(a["traj"], b["traj"])
    np.testing.assert_array_equal(a["part_opacity"], b["part_opacity"])
    cols = list(a["pid"])
    gaze = a["traj"][:, cols.index("ParamEyeBallX")]
    assert (np.diff(gaze) == 0).mean() > 0.3          # real holds
    armc = a["traj"][:, cols.index("ParamArmL")]
    assert armc.std() > 0.1                           # gestures move the arm
    toks = json.loads(tok.read_text())["tokens"]
    assert any(k["type"] == "gesture" for k in toks)


def test_switch_role_columns_go_to_the_gesture_layer():
    roles = ev.bind(["ParamBrowLY", "ParamBrowRY", "ParamEyeBallX"])
    kept, dropped = ev.drop_switch_roles(roles, {"ParamBrowLY": (0.0, 0, 1)})
    assert set(dropped) == {"brow_l"} and "brow_r" in kept
    # An explicit override is kept even on a switch.
    o = ev.bind(["ParamBrowLY"], override={"brow_l": "ParamBrowLY"})
    assert ev.drop_switch_roles(o, {"ParamBrowLY": (0.0, 0, 1)})[0] == o


def test_events_leave_switch_brows_to_the_rigs_motions(tmp_path):
    """A brow the rig's motions only ever jump between two states (xuefeng)
    is not event-driven: it plays from the motions, at their two states,
    instead of sitting pinned by the snapper (std 0)."""
    from flow.runtime import main
    from l2dprobe.analyze import group_of

    # Stepped segments (type 2): -1 until 2 s, +1 until 4 s, -1 to the end.
    brow = [0, -1, 2, 2.0, 1, 2, 4.0, -1, 2, 6.0, -1]
    arm = [0, 0, 1, 2.0, 1, 4.0, 0, 6.0, 0]
    motions = [{"Version": 3, "Meta": {"Duration": 6.0, "Fps": 30},
                "Curves": [{"Target": "Parameter", "Id": "ParamBrowLY",
                            "Segments": brow},
                           {"Target": "Parameter", "Id": "ParamArmL",
                            "Segments": arm}]},
               {"Version": 3, "Meta": {"Duration": 6.0, "Fps": 30},
                "Curves": [{"Target": "Parameter", "Id": "ParamArmL",
                            "Segments": arm}]}]
    refs = []
    for i, m in enumerate(motions):
        (tmp_path / f"m{i}.motion3.json").write_text(json.dumps(m))
        refs.append({"File": f"m{i}.motion3.json"})
    model = tmp_path / "x.model3.json"
    model.write_text(json.dumps({"Version": 3, "FileReferences": {
        "Moc": "x.moc3", "Motions": {"Idle": refs}}}))
    pids = ["ParamAngleX", "ParamEyeBallX", "ParamBrowLY", "ParamArmL"]
    n = len(pids)
    rig = tmp_path / "x.npz"
    np.savez(rig, pid=np.array(pids, dtype=object),
             kind=np.array(["continuous"] * n, dtype=object),
             group=np.array([group_of(p) for p in pids], dtype=object),
             minimum=np.array([-30, -1, -1, 0], np.float32),
             maximum=np.array([30, 1, 1, 1], np.float32),
             default=np.array([0, 0, 0, 0], np.float32),
             descriptor=np.zeros((n, 4), np.float32))
    out = tmp_path / "a.npz"
    tok = tmp_path / "t.json"
    assert main(["--model", str(model), "--rig", str(rig), "--events",
                 "default", "--seconds", "60", "--seed", "1", "--dump-only",
                 "--probe-dir", str(tmp_path), "--dump", str(out),
                 "--tokens", str(tok)]) == 0
    a = np.load(out, allow_pickle=True)
    b = a["traj"][:, list(a["pid"]).index("ParamBrowLY")]
    assert "brow_l" not in json.loads(tok.read_text())["roles"]
    assert not any(k["type"] == "brow" for k in
                   json.loads(tok.read_text())["tokens"])
    assert b.std() > 0.3                              # it does change...
    assert (np.abs(np.abs(b) - 1) < 0.1).mean() > 0.8  # ...between states


def _body_clip(rng, hold_s, n=300):
    """Body sways (head-free): moves of 0.6 s, holds of hold_s (0 = none)."""
    seq, v = [], 0.0
    while sum(len(s) for s in seq) < n:
        nv = float(rng.uniform(-0.6, 0.6))
        seq += [_move(v, nv, 18)]
        if hold_s:
            seq.append(np.full(int(hold_s * FPS), nv))
        v = nv
    x = np.concatenate(seq)[:n]
    return np.stack([x, 0.5 * x], 1)


def _still(x):
    """flow.score's still fraction of one (T, k) clip."""
    from flow.score import stillness
    return stillness([(x, np.ones_like(x, bool))], range(x.shape[1]), FPS)


def test_motion_clips_match_the_clips_file():
    """The runtime's own-motion clips are the clips' values, signed."""
    m = {"frames": 4, "params": {"ParamBodyAngleX": np.array([0., 5, 10, 5]),
                                 "ParamFoo": np.zeros(4)}, "parts": {}}
    roles = {"body_x": ("ParamBodyAngleX", -1.0, "override")}
    (fps, have), = ev.motion_clips([("a", m)], roles,
                                   {"ParamBodyAngleX": -10.0},
                                   {"ParamBodyAngleX": 10.0}, FPS)
    np.testing.assert_allclose(have["body_x"], [0.0, -0.5, -1.0, -0.5])
    assert fps == FPS and set(have) == {"body_x"}


def test_adapt_moves_the_body_toward_the_rigs_own_motions():
    rng = np.random.default_rng(0)
    pool = ev.load_stats("default")
    # A pool whose body never rests (the median training rig) ...
    pool["groups"]["body"].update(hold=[0.3, 0.5], move=[0.6, 0.3],
                                  p_chain=0.7, amp_rms=0.5)
    for r in ("body_x", "body_z"):
        pool["roles"][r].update(std=0.3, lo=-0.75, hi=0.75)
    # ... and a rig whose own motions shift and then rest for ~2 s.
    own = ev.fit_clips((FPS, {"body_x": c[:, 0], "body_z": c[:, 1]})
                       for c in [_body_clip(rng, 2.0) for _ in range(6)])
    st = ev.adapt(pool, own)
    assert pool["groups"]["body"]["hold"][0] == 0.3       # not mutated
    assert st["groups"]["body"]["hold"][0] > 1.0
    assert st["groups"]["body"]["p_chain"] < 0.3
    assert st["roles"]["body_x"]["std"] == pytest.approx(
        own["roles"]["body_x"]["std"], rel=0.2)
    for g in ("gaze", "head", "brow"):                   # pool kept
        assert st["groups"][g] == pool["groups"][g]
    assert st["roles"]["gaze_x"] == pool["roles"]["gaze_x"]
    assert set(st["adapted"]) == {"body"}

    def still(s):
        tr = [ev.generate(s, ["body_x", "body_z"], 60, FPS,
                          np.random.default_rng(k))[0] for k in range(3)]
        return np.mean([_still(np.stack([t["body_x"], t["body_z"]], 1))
                        for t in tr])
    authored = np.mean([_still(_body_clip(rng, 2.0, 600)) for _ in range(3)])
    assert still(pool) < 0.5 * authored <= still(st)
    # Nothing of its own (no motion keys the body): the pool as it was.
    none = ev.adapt(pool, ev.fit_clips(iter([])))
    assert none["groups"] == pool["groups"]
