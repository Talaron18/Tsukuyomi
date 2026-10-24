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
    fitted = json.loads(json.dumps(st))
    fitted["idle"]["profile"] = "authored"
    for seed in (0, 1, 2):
        for v in ev.generate(fitted, roles, 60.0, FPS,
                             np.random.default_rng(seed))[0].values():
            assert periodic_peak(v.astype(np.float64), FPS)[0] <= 0.6
    # The calm profile's few long holds make sparse step signals, on which
    # the judge's peak fires by chance (3 body, 2 head and 2 gaze roles of
    # 100 seeds x 16 roles; before the calm head 3 body, 4 head and the 8
    # brows of one seed): judged as a rate, not seed by seed.
    peaks = []
    for seed in range(20):
        for v in ev.generate(st, roles, 60.0, FPS,
                             np.random.default_rng(seed))[0].values():
            peaks.append(periodic_peak(v.astype(np.float64), FPS)[0])
    assert np.mean(np.array(peaks) > 0.6) <= 0.02
    for seed in (0, 1, 2):
        traj, blinks, toks = ev.generate(st, roles, 60.0, FPS,
                                         np.random.default_rng(seed))
        for v in traj.values():
            assert len(v) == 1800 and np.abs(v).max() <= 1
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
        for k in toks:            # "follows" indexes the sorted stream
            if k["type"] == "head_turn" and k.get("follows") is not None:
                assert toks[k["follows"]]["type"] == "saccade"
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
    # mid-fade: both motions keyed; parameters blend, but part visibility
    # only switches (old part before the midpoint, new after) -- a blended
    # part drew both arm variants half visible ("4 hands" in the A/B)
    fade = [fr[f][1]["PartA"] for f in range(30, 45)]
    assert set(fade) <= {0.0, 1.0}
    assert fade[0] == 1.0 and fade[-1] == 0.0
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
                 "--tokens", str(tok),
                 # every motion back to back, so m0 surely plays its brow
                 "--idle-profile", "authored"]) == 0
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
    # Adaptation as fitted: the calm idle profile would floor the holds.
    pool["idle"]["profile"] = "authored"
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


AL_NAMES = ("complete", "home", "idle", "login", "mail", "main_1", "main_2",
            "main_3", "mission_complete", "mission", "touch_body",
            "touch_head", "touch_special", "wedding")


def test_calm_motions_by_label_group_then_energy():
    al = [n + ".motion3.json" for n in AL_NAMES]
    names, how = ev.calm_motions(al, {n: "" for n in al})
    assert how == "label"
    assert sorted(ev.motion_label(n) for n in names) == [
        "home", "idle", "main", "main", "main"]
    cub = ["m01.motion3.json", "m02.motion3.json", "m04.motion3.json"]
    names, how = ev.calm_motions(cub, {"m01.motion3.json": "Idle",
                                       "m02.motion3.json": "Idle",
                                       "m04.motion3.json": "TapBody"})
    assert (names, how) == (cub[:2], "group")
    names, how = ev.calm_motions(cub, {}, {cub[0]: 0.3, cub[1]: 0.1,
                                           cub[2]: 0.2})
    assert (sorted(names), how) == (sorted(cub[1:]), "energy")
    assert ev.calm_motions(cub[:1]) == (cub[:1], "all")


def test_idle_view_calms_body_and_head_and_authored_is_untouched():
    st = ev.load_stats("default")
    st["groups"]["body"].update(hold=[0.16, 1.0], move=[1.07, 0.5],
                                p_chain=0.5)
    st["groups"]["head"].update(hold=[0.54, 1.01], p_chain=0.64)
    keep = json.loads(json.dumps(st))
    v = ev.idle_view(st)
    assert st == keep                                     # not mutated
    B = v["groups"]["body"]
    assert B["hold"] == [5.0, 0.6] and B["move"][0] == 1.5
    assert B["p_chain"] == 0.0 and B["rho_min"] == 0.0
    H = v["groups"]["head"]
    assert H["hold"] == [3.0, 0.8] and H["move"] == st["groups"]["head"]["move"]
    assert H["p_chain"] == 0.0 and H["rho_min"] == 0.0
    assert H["amp"][0] == pytest.approx(0.6 * st["groups"]["head"]["amp"][0])
    assert v["head_follow"]["min_amp"] == 0.75
    for g, f in (("body", 0.4), ("head", 0.6)):
        for r in ev.GROUPS[g]:
            R, R0 = v["roles"][r], st["roles"][r]
            assert R["center"] == R0["center"]
            assert R["std"] == pytest.approx(f * R0["std"])
            assert R["hi"] - R["lo"] == pytest.approx(
                f * (R0["hi"] - R0["lo"]))
    for g in ("gaze", "brow"):
        assert v["groups"][g] == st["groups"][g]
    for r in ev.GROUPS["gaze"]:
        assert v["roles"][r] == st["roles"][r]
    assert v["gesture"]["rest"] == [15.0, 0.5] and v["gesture"]["lead"]
    st["idle"]["profile"] = "authored"
    assert ev.idle_view(st) is st


def test_calm_idle_rests_and_plays_only_calm_motions():
    """aidang_2 on the server, 38 s: a body shift every 0.5-1.5 s and a
    gesture every ~3-5 s, touch_*/login/complete among them."""
    al = [n + ".motion3.json" for n in AL_NAMES]
    lib = [{"name": n, "seconds": 10.0} for n in al]
    calm, _ = ev.calm_motions(al)
    roles = ["head_x", "head_y", "head_z", "body_x", "body_y", "brow_l"]
    rates, gaps, motions, busy, before_rates = [], [], set(), [], []
    st = ev.load_stats("default")
    # The body as the server adapted it to aidang_2's own motions.
    st["groups"]["body"].update(hold=[0.16, 1.0], move=[1.07, 0.5],
                                p_chain=0.5)
    old = json.loads(json.dumps(st))
    old["idle"]["profile"] = "authored"
    for seed in range(8):
        toks = ev.schedule(st, roles, 120.0, np.random.default_rng(seed),
                           [g for g in lib if g["name"] in calm])
        ges = [k for k in toks if k["type"] == "gesture"]
        motions |= {k["motion"] for k in ges}
        gaps += list(np.diff([k["t"] for k in ges]))
        busy.append(sum(k["duration"] for k in ges) / 120.0)
        rates.append(sum(k["type"] == "body_shift" for k in toks) / 2.0)
        before = ev.schedule(old, roles, 120.0, np.random.default_rng(seed),
                             lib)
        before_rates.append(sum(k["type"] == "body_shift"
                                for k in before) / 2.0)
    assert motions and motions <= set(calm)
    assert 10.0 <= np.median(gaps) <= 40.0 and min(gaps) > 4.0
    assert np.mean(busy) < 0.4
    assert np.mean(rates) <= 12.0                       # shifts per minute
    assert np.mean(before_rates) > 3 * np.mean(rates)


def _head_track(toks, roles, start, seconds=60.0):
    tr, _ = ev.render(toks, roles, seconds, FPS, start)
    return np.stack([tr[r] for r in ev.GROUPS["head"]], 1)


def test_calm_head_fixates_and_follows_only_large_saccades():
    """Fitted, the head turned ~40 times a minute and was still in 2-5% of
    1 s windows on Hiyori, Natori and aidang_2: the busiest thing left."""
    from flow.score import periodicity, stillness
    st = ev.load_stats("default")
    st["groups"]["head"].update(hold=[0.54, 1.01], p_chain=0.64)
    old = json.loads(json.dumps(st))
    old["idle"]["profile"] = "authored"
    roles = list(ev.ALL_ROLES)
    start = {r: st["roles"][r]["center"] for r in roles}
    out = {"calm": [], "authored": []}
    for name, s in (("calm", st), ("authored", old)):
        for seed in range(12):
            toks = ev.schedule(s, roles, 60.0, np.random.default_rng(seed),
                               None, start)
            X = _head_track(toks, roles, start)
            seg = [(X, np.ones_like(X, bool))]
            turns = [k for k in toks if k["type"] == "head_turn"]
            out[name].append((
                len(turns), stillness(seg, range(3), FPS),
                periodicity(seg, range(3), FPS)[0],
                sum(k["type"] == "saccade" for k in toks),
                [k["follows"] for k in turns if k.get("follows") is not None],
                toks))
    calm, auth = out["calm"], out["authored"]
    m = lambda rs, i: float(np.mean([r[i] for r in rs]))  # noqa: E731
    assert 6 <= m(calm, 0) <= 20 and m(auth, 0) > 2 * m(calm, 0)
    assert 0.5 <= m(calm, 1) <= 0.8 and m(auth, 1) < 0.3
    assert m(calm, 2) < 0.3 and np.median([r[2] for r in calm]) < 0.3
    # the eyes stay as lively: same saccade rate
    assert m(calm, 3) == pytest.approx(m(auth, 3), rel=0.25)
    # a head turn that follows a saccade follows a large one
    for *_, follows, toks in calm:
        pos = {r: start[r] for r in ev.GROUPS["gaze"]}
        amp = {}
        for i, k in enumerate(toks):
            if k["type"] == "saccade":
                amp[i] = float(np.sqrt(sum((v - pos[r]) ** 2
                                           for r, v in k["target"].items())))
                pos.update(k["target"])
        assert all(amp[i] >= 0.75 for i in follows)


def test_calm_head_leaves_directed_moves_alone():
    from flow import director as dr
    st = ev.load_stats("default")
    old = json.loads(json.dumps(st))
    old["idle"]["profile"] = "authored"
    acts = [{"t": 1.0, "tag": "look", "arg": "left"},
            {"t": 3.0, "tag": "nod"}, {"t": 6.0, "tag": "shake"},
            {"t": 9.0, "tag": "lean", "arg": "in"}]
    roles = list(ev.ALL_ROLES)
    a = dr.realize(acts, st, roles, {}, np.random.default_rng(0))
    b = dr.realize(acts, old, roles, {}, np.random.default_rng(0))
    assert a == b and any(k["type"] == "head_turn" for k in a)


def _underlay_motions():
    n = 90
    k = np.arange(n)
    return {
        # the idle loop: seamless hair, a switch, an arm, a part curve
        "idle": {"frames": n, "params": {
            "ParamHairBack": np.sin(2 * np.pi * k / n) * 10,
            "ParamArmLA": np.linspace(-5, 5, n),
            "ParamSkirtT": np.where(k < 45, 0.0, 1.0),
            "ParamRibbon": np.linspace(0, 1, n)},          # not seamless
            "parts": {"PartArmA": np.where(k < 45, 1.0, 0.0)}},
        "main": {"frames": n, "params": {
            "ParamHairBack": np.full(n, 3.0), "ParamArmLA": np.full(n, 2.0),
            "ParamSkirtT": np.zeros(n), "ParamRibbon": np.full(n, 0.5)},
            "parts": {"PartArmA": np.ones(n)}},
    }


def test_underlay_keeps_loose_params_alive_during_rests_only():
    motions = _underlay_motions()
    base = {p: float(v[0]) for p, v in motions["idle"]["params"].items()}
    loose = ev.loose_params(
        ["ParamHairBack", "ParamArmLA", "ParamSkirtT", "ParamRibbon",
         "ParamHipX"], ["hair", "arm", "other", "other", "other"],
        switches={"ParamSkirtT"})
    assert loose == ["ParamHairBack", "ParamRibbon"]
    gest = [{"t": 4.0, "type": "gesture", "role": "arms", "motion": "main",
             "start": 0.0, "duration": 2.0, "fade": 0.5},
            {"t": 12.0, "type": "gesture", "role": "arms", "motion": "main",
             "start": 0.5, "duration": 2.0, "fade": 0.5}]
    st = ev.load_stats("default")
    und = ev.underlay_token(st, "idle", 20.0)
    assert und["gain"] == 0.5 and und["t"] == 0.0
    st["idle"]["profile"] = "authored"
    assert ev.underlay_token(st, "idle", 20.0) is None
    n = 600

    def play(toks):
        gp = ev.GesturePlayer(motions, lambda p: True, base,
                              {"PartArmA": 1.0}, {}, FPS, loose)
        return gp.frames(toks, n), gp.keyed

    plain, k_plain = play(gest)
    gp0 = ev.GesturePlayer(motions, lambda p: True, base, {"PartArmA": 1.0},
                           {}, FPS)
    assert gp0.frames(gest, n) == plain               # no token: as before
    under, k_under = play([und] + gest)
    col = lambda fr, p: np.array([f[0][p] for f in fr])  # noqa: E731
    # arms, switches and parts: bit for bit
    for p in ("ParamArmLA", "ParamSkirtT"):
        assert np.array_equal(col(plain, p), col(under, p))
    assert [f[1] for f in plain] == [f[1] for f in under]
    hair = col(under, "ParamHairBack")
    d = np.abs(np.diff(hair))
    playing = np.zeros(n - 1, bool)
    for g in gest:
        playing[int(g["t"] * FPS) + 1:int((g["t"] + g["duration"]) * FPS)] = True
    # held in the plain player during rests, moving under the underlay
    rest = ~playing
    rest[:int(1.5 * FPS)] = False                     # its ramp-in
    assert np.abs(np.diff(col(plain, "ParamHairBack")))[
        int(7 * FPS):int(11 * FPS)].max() == 0.0
    assert (d[rest] > 1e-3).mean() > 0.9
    # never more than the loop's own speed x gain, plus the ramp
    loop_v = np.abs(np.diff(motions["idle"]["params"]["ParamHairBack"])).max()
    assert d.max() <= loop_v * 1.5 + 1e-9
    # while a gesture plays (past its fade) the underlay is off
    a, b = int(4.6 * FPS), int(6.0 * FPS)
    assert np.array_equal(hair[a:b], col(plain, "ParamHairBack")[a:b])
    # the ribbon is not seamless: played forward and back, no seam jump
    rib = np.abs(np.diff(col(under, "ParamRibbon")))
    assert rib.max() <= 0.5 * (1 / 89) * 1.5 + 0.02
    # what moves is keyed (the scorer reads the dump's mask)
    assert "ParamHairBack" in k_under[int(2 * FPS)]   # before any gesture
    assert k_plain[int(2 * FPS)] == set()


def test_idle_streams_in_windows_without_a_seam():
    """flow.live generates the idle ahead in rolling windows. Advanced in
    windows of any size, the schedulers emit exactly the tokens one call
    emits: the state (each group's next onset, a head turn due after its
    saccade, the brow expression, the last gesture's switch states) carries
    over, so a window edge cannot show -- no group restarts its first move
    at 0.2-1 s past every seam as a fresh schedule() would."""
    st = ev.load_stats("default")
    roles = list(ev.ALL_ROLES)
    lib = [{"name": "idle.motion3.json", "seconds": 8.0},
           {"name": "home.motion3.json", "seconds": 5.0}]
    for prof in ("calm", "authored"):
        s = json.loads(json.dumps(st))
        s["idle"]["profile"] = prof
        one = ev.IdleScheduler(s, roles, np.random.default_rng(3))
        whole = one.advance(90.0)
        g_whole = ev.GestureScheduler(one.st, lib,
                                      np.random.default_rng(4)).advance(90.0)
        for win in (7.0, 10.0, 13.0):
            sc = ev.IdleScheduler(s, roles, np.random.default_rng(3))
            gs = ev.GestureScheduler(sc.st, lib, np.random.default_rng(4))
            got, g_got, t = [], [], 0.0
            while t < 90.0:
                t = min(t + win, 90.0)
                got += sc.advance(t)
                sc.toks.clear()            # a stream keeps no history
                g_got += gs.advance(t)
            assert got == whole, (prof, win)
            assert g_got == g_whole, (prof, win)
    # schedule() is the same machinery run once, then sorted.
    toks = ev.schedule(st, roles, 30.0, np.random.default_rng(0))
    raw = ev.IdleScheduler(st, roles, np.random.default_rng(0)).advance(30.0)
    assert [(k["t"], k["type"]) for k in toks] == sorted(
        (k["t"], k["type"]) for k in raw)
