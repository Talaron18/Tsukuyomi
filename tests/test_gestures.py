"""
flow.gestures: what each gesture shows, for the director's prompt. Fake
rigs and motions only.

Different motions must read differently (an arm raised and held, an arm
going back and forth, an unbound head nodding, a small idle body sway); a
nod on a bound role column is not played by a gesture, so it is not
described; the per-rig override file wins and can hide a gesture; and the
lines and the prompt are byte-stable whatever order the motions come in.
"""

from __future__ import annotations

import json

import numpy as np

from flow import director as dr
from flow import gestures as gz

FPS = 30.0
T = 120                                   # 4 s
t = np.arange(T) / FPS
LO = {"ParamArmL": 0.0, "ParamArmR": 0.0, "PARAM_head_nod": -30.0,
      "PARAM_body_sway": -10.0, "PARAM_xyz": 0.0}
HI = {"ParamArmL": 1.0, "ParamArmR": 1.0, "PARAM_head_nod": 30.0,
      "PARAM_body_sway": 10.0, "PARAM_xyz": 1.0}
REST = {"ParamArmL": 0.0, "ParamArmR": 0.0, "PARAM_head_nod": 0.0,
        "PARAM_body_sway": 0.0, "PARAM_xyz": 0.0}


def _m(params, parts=None):
    return {"params": {k: np.asarray(v, np.float32) for k, v in params.items()},
            "parts": parts or {}, "frames": T}


def _motions():
    raise_ = np.clip(t / 0.5, 0, 1)                       # up and held
    wave = 0.5 + 0.4 * np.sin(2 * np.pi * 1.5 * t)       # 6 swings
    nod = -12 * np.abs(np.sin(2 * np.pi * 1.0 * t))      # 8 dips
    sway = 1.2 * np.sin(2 * np.pi * 0.25 * t)            # one slow sway
    return [("raise.motion3.json", _m({"ParamArmL": raise_})),
            ("wave.motion3.json", _m({"ParamArmR": wave})),
            ("nod.motion3.json", _m({"PARAM_head_nod": nod})),
            ("sway.motion3.json", _m({"PARAM_body_sway": sway})),
            ("blob.motion3.json", _m({"PARAM_xyz": raise_}))]


def _describe(motions, keep=lambda p: True):
    return gz.describe_motions(motions, keep, LO, HI, REST, fps=FPS,
                               max_seconds=dr.GESTURE_MAX)


def test_sides_and_parts_from_ids():
    assert gz.side_of("ParamArmLA") == "L"
    assert gz.side_of("PARAM_Hand_R3") == "R"
    assert gz.side_of("PARAM_Arm_L_hand") == "L"
    assert gz.side_of("ParamArmBRHand01") == "R"
    assert gz.side_of("ParamLeftShoulderUp") == "L"
    assert gz.side_of("PARAM_ALL_moveZ") == ""
    assert gz.side_of("ParamSideupRibbon") == ""
    assert gz.part_of("PARAM_Arm_L_hand") == "hand"
    assert gz.part_of("ParamArmLA") == "arm"
    assert gz.part_of("ParamWaistAngleZ") == "body"
    assert gz.part_of("ParamHairAhoge") == "hair"
    assert gz.part_of("PARAM_xyz", "other") == ""
    assert gz.part_of("ParamAngleY", "", {"ParamAngleY": "head_y"}) == "head"


def test_reversals_ignore_small_wiggles():
    x = np.sin(np.linspace(0, 6 * np.pi, 300))
    assert gz.reversals(x) == 6          # 3 peaks, 3 troughs
    assert gz.reversals(0.05 * x) == 0
    assert gz.reversals(np.linspace(-1, 1, 50)) == 0


def test_different_motions_read_differently():
    d = _describe(_motions())
    assert d["raise.motion3.json"].startswith("left arm moves widely;")
    assert "back and forth" not in d["raise.motion3.json"]
    assert d["wave.motion3.json"].startswith(
        "right arm swings back and forth;")
    assert d["nod.motion3.json"].startswith("head swings back and forth;")
    assert d["sway.motion3.json"].startswith("body moves slightly;")
    # Nothing the hints can place: said so, not invented.
    assert d["blob.motion3.json"].startswith("unclear parts move;")
    assert len(set(d.values())) == len(d)
    assert all(v.split("; ")[-1].startswith("4.0 s") for v in d.values())
    # Rig-relative energy: the slow sway is the calm one.
    assert d["sway.motion3.json"].endswith("calm")
    # A part the gesture does not play is not described.
    d2 = _describe(_motions(), keep=lambda p: p != "ParamArmR")
    assert d2["wave.motion3.json"].startswith("little visible motion")


def test_part_swaps_against_the_rest_pose_and_the_cut():
    m = _m({"ParamArmL": np.zeros(T)},
           {"PartArmB": np.ones(T, np.float32)})
    d = gz.describe_motions([("x", m)], lambda p: True, LO, HI, REST,
                            {"PartArmB": 0.0}, fps=FPS)
    assert d["x"] == "switches arm pose; 4.0 s"
    long = _m({"ParamArmL": np.r_[np.zeros(T - 30), np.ones(30)]})
    d = gz.describe_motions([("x", long)], lambda p: True, LO, HI, REST,
                            fps=FPS, max_seconds=3.0)
    # The move after the cut never plays.
    assert d["x"] == "little visible motion; 3.0 s"


def test_lines_are_stable_whatever_the_order():
    ms = _motions()
    a = _describe(ms)
    b = _describe(list(reversed(ms)))
    assert a == b
    labels = dr.gesture_labels([n for n, _ in ms])
    desc = {k: a[v] for k, v in labels.items()}
    p1 = dr.system_prompt("Hiyori.", labels, None, desc)
    p2 = dr.system_prompt("Hiyori.", dict(reversed(labels.items())), None,
                          dict(reversed(desc.items())))
    assert p1 == p2
    lines = [ln for ln in p1.splitlines() if ln.startswith("  ")]
    assert [ln.split(":")[0].strip() for ln in lines] == sorted(labels)


def test_overrides_win_and_hide(tmp_path):
    f = tmp_path / "r.gestures.json"
    f.write_text(json.dumps({"Wave": "waves hello",
                             "nod": {"hide": True},
                             "raise": {"desc": "raises her hand"}}))
    over = gz.load_overrides(str(f))
    labels = {"wave": "wave.motion3.json", "nod": "nod.motion3.json",
              "raise": "raise.motion3.json", "sway": "sway.motion3.json"}
    measured = _describe(_motions())
    keep, desc = gz.apply_overrides(labels, measured, over)
    assert sorted(keep) == ["raise", "sway", "wave"]
    assert desc["wave"] == "waves hello"
    assert desc["raise"] == "raises her hand"
    assert desc["sway"] == measured["sway.motion3.json"]
    assert gz.load_overrides(str(tmp_path / "none.json")) == {}
    assert gz.override_path("rigs/Hiyori.npz") == "rigs/Hiyori.gestures.json"


def _rig(root):
    """A rig whose two motions nod the head on the standard ParamAngleY
    (bound to head_y, so the event layer owns it) and wave the arm."""
    from l2dprobe.analyze import group_of
    root.mkdir(parents=True)
    k = []
    for i in range(9):
        k += [0, i * 0.5, (15.0 if i % 2 else -15.0)]
    nod = [0.0, -15.0] + k[3:]
    wave = [0.0, 0.0] + [x if j % 3 != 2 else (1.0 if x > 0 else 0.0)
                          for j, x in enumerate(k[3:])]
    for name, pid, seg in (("nod", "ParamAngleY", nod),
                           ("wave", "ParamArmL", wave)):
        (root / f"{name}.motion3.json").write_text(json.dumps({
            "Version": 3, "Meta": {"Duration": 4.0, "Fps": 30},
            "Curves": [{"Target": "Parameter", "Id": pid,
                        "Segments": seg}]}))
    model = root / "x.model3.json"
    model.write_text(json.dumps({"Version": 3, "FileReferences": {
        "Moc": "x.moc3", "Motions": {"": [{"File": "nod.motion3.json"},
                                          {"File": "wave.motion3.json"}]}}}))
    pids = ["ParamAngleX", "ParamAngleY", "ParamEyeBallX", "ParamArmL"]
    n = len(pids)
    rig = root / "x.npz"
    np.savez(rig, pid=np.array(pids, dtype=object),
             kind=np.array(["continuous"] * n, dtype=object),
             group=np.array([group_of(p) for p in pids], dtype=object),
             minimum=np.array([-30, -30, -1, 0], np.float32),
             maximum=np.array([30, 30, 1, 1], np.float32),
             default=np.array([0, 0, 0, 0], np.float32),
             descriptor=np.zeros((n, 4), np.float32))
    return model, rig


def test_main_prompt_describes_what_plays_and_honours_the_file(tmp_path,
                                                               capsys):
    model, rig = _rig(tmp_path / "x")
    args = ["--model-json", str(model), "--rig", str(rig), "--out",
            str(tmp_path / "o"), "--probe-dir", str(tmp_path),
            "--print-system"]
    assert dr.main(args) == 0
    cap = capsys.readouterr()
    # The nod is on the bound head_y column: a gesture never plays it.
    assert "  nod: little visible motion; 4.0 s" in cap.out
    assert "  wave: left arm swings back and forth; 4.0 s" in cap.out
    assert "director: gesture wave [measured]: left arm swings" in cap.err
    first = cap.out
    assert dr.main(args) == 0
    assert capsys.readouterr().out == first             # byte-stable
    # The default override file sits next to the rig file.
    (tmp_path / "x" / "x.gestures.json").write_text(json.dumps(
        {"wave": "waves hello", "nod": {"hide": True}}))
    assert dr.main(args) == 0
    cap = capsys.readouterr()
    assert "  wave: waves hello\n" in cap.out
    assert "nod" not in cap.out.split("[gesture:NAME]")[1].split("Rules")[0]
    assert "director: gesture nod [hidden]" in cap.err
    # An explicit --gestures file instead.
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"nod": "nods"}))
    assert dr.main(args + ["--gestures", str(g)]) == 0
    out = capsys.readouterr().out
    assert "  nod: nods\n" in out and "  wave: left arm swings" in out


def test_hidden_gesture_is_not_parsed():
    labels = {"wave": "wave.motion3.json"}
    p = dr.parse("Hi [gesture:nod] there [gesture:wave]", gestures=labels)
    assert [x["arg"] for x in p.tags] == ["wave"]
