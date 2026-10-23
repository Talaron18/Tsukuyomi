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


# ------------------------------------------------- Azur Lane-like rigs
# Clips that loop with the arms swaying throughout, run to the 8 s cap and
# show a part (runs/demo/gestures_al.txt): a line must say what sets one
# apart, merge clips that are the same, and never call a switch motion.

AL_LO = {"PARAM_Arm_L": -30.0, "PARAM_Arm_R": -30.0, "PARAM_Hand_L": -30.0,
         "PARAM_Hand_R": -30.0, "PARAM_eye_close": 0.0, "PARAM_fx": 0.0}
AL_HI = {k: (1.0 if k in ("PARAM_eye_close", "PARAM_fx") else 30.0)
         for k in AL_LO}
AL_REST = {k: 0.0 for k in AL_LO}
LONG = 300                                # 10 s, cut to GESTURE_MAX
tl = np.arange(LONG) / FPS


def _al(params, frames=LONG, shown=True):
    return {"params": {k: np.asarray(v, np.float32)[:frames]
                       for k, v in params.items()},
            "parts": ({"PartBag": np.ones(frames, np.float32)} if shown
                      else {}),
            "frames": frames}


def _sway(amp=12.0, f=0.5, ph=0.0):
    return {"PARAM_Arm_L": amp * np.sin(2 * np.pi * f * tl + ph),
            "PARAM_Arm_R": amp * np.sin(2 * np.pi * f * tl + ph + 1),
            "PARAM_Hand_L": amp * np.sin(2 * np.pi * f * tl + ph + 2),
            "PARAM_Hand_R": amp * np.sin(2 * np.pi * f * tl + ph + 3)}


EYES = ((tl // 1.0) % 2 == 1).astype(float)      # held open/shut, jumping


def _al_rig():
    return [("home.motion3.json", _al(_sway(ph=0.0))),
            ("login.motion3.json", _al(_sway(ph=1.0))),
            ("mail.motion3.json", _al(_sway(ph=2.0))),
            ("mission.motion3.json", _al(_sway(ph=3.0))),
            ("touch_body.motion3.json", _al({**_sway(), "PARAM_Hand_L":
                                             28 * np.sin(np.pi * tl),
                                             "PARAM_Hand_R":
                                             28 * np.sin(np.pi * tl + 1)})),
            ("touch_head.motion3.json", _al(_sway(f=0.8), frames=87)),
            ("touch_special.motion3.json",
             _al({**_sway(ph=4.0), "PARAM_fx": np.clip(tl - 1, 0, 1)})),
            ("idle.motion3.json",
             _al({"PARAM_eye_close": EYES}, shown=False))]


def _al_describe(motions, **kw):
    return gz.describe_rig(motions, lambda p: True, AL_LO, AL_HI, AL_REST,
                           {"PartBag": 0.0}, fps=FPS,
                           max_seconds=dr.GESTURE_MAX, **kw)


def test_common_features_dropped_distinctive_kept():
    sw = {"PARAM_eye_close": (0.0, 1.0)}
    d, same = _al_describe(_al_rig(), switches=sw)
    assert len(set(same.values())) == len(same)          # all different
    # Arm and hand sway and the shown part are in 7 of 8: never said.
    for v in d.values():
        assert "swing" not in v and "a part appears" not in v
    # Nothing sets these apart: the label alone.
    for n in ("home", "login", "mail", "mission"):
        assert d[f"{n}.motion3.json"] == ""
    assert d["touch_body.motion3.json"].startswith(
        "both hands move more than in the other gestures;")
    assert "short" in d["touch_head.motion3.json"]
    assert "2.9 s" in d["touch_head.motion3.json"]
    assert d["touch_special.motion3.json"].startswith("an effect plays;")
    # The idle alone keeps its arms still and switches its eyes.
    idle = d["idle.motion3.json"]
    assert "eyes change" in idle and "stay still" in idle
    # The same clips described one by one (too few to compare) say it all.
    one, _ = _al_describe(_al_rig()[:2], switches=sw)
    assert one["home.motion3.json"].startswith(
        "both arms and hands swing back and forth; a part appears; 8.0 s")


def test_switch_params_are_not_motion():
    ms = _al_rig()
    sw = {"PARAM_eye_close": (0.0, 1.0)}
    # Taken for motion, the eye switch "swings" (aidang_2's old lines).
    raw, _ = _al_describe(ms)
    assert "eyes swing back and forth" in raw["idle.motion3.json"]
    d, _ = _al_describe(ms, switches=sw)
    assert all("eyes swing" not in v and "eyes move" not in v
               for v in d.values())
    # It counts in no energy either: the idle is the calm one either way,
    # but its jumps no longer make it the rig's liveliest motion.
    assert d["idle.motion3.json"].endswith("calm")
    # A pose switch is said as one, with its side.
    m = _al({"PARAM_Arm_L": np.r_[np.zeros(60), np.full(LONG - 60, 30.0)]},
            shown=False)
    d, _ = _al_describe(ms[:4] + [("pose.motion3.json", m)],
                        switches={**sw, "PARAM_Arm_L": (0.0, 30.0)})
    assert d["pose.motion3.json"].startswith("switches left arm pose;")


def test_same_motions_merge_into_one_parseable_line():
    ms = _al_rig()
    base = ms[4][1]
    copy = {"params": dict(base["params"]), "parts": dict(base["parts"]),
            "frames": base["frames"]}
    rng = np.random.default_rng(0)
    near = {"params": {k: v + rng.normal(0, 0.2, len(v)).astype(np.float32)
                       * (AL_HI[k] - AL_LO[k]) / 100
                       for k, v in base["params"].items()},
            "parts": dict(base["parts"]), "frames": base["frames"]}
    ms = ms + [("pat.motion3.json", copy), ("poke.motion3.json", near)]
    d, same = _al_describe(ms, switches={"PARAM_eye_close": (0.0, 1.0)})
    assert same["pat.motion3.json"] == same["poke.motion3.json"] \
        == same["touch_body.motion3.json"] == "pat.motion3.json"
    assert d["poke.motion3.json"] == d["touch_body.motion3.json"]
    # The group counts once: touch_body's wide hands are still unusual.
    assert d["pat.motion3.json"].startswith("both hands move more")
    # Twice the tolerance apart is a different gesture.
    far = {"params": {k: v + np.float32(0.1 * (AL_HI[k] - AL_LO[k]) / 2)
                      for k, v in base["params"].items()},
           "parts": dict(base["parts"]), "frames": base["frames"]}
    _, s2 = _al_describe(ms[:-2] + [("far.motion3.json", far)])
    assert s2["far.motion3.json"] == "far.motion3.json"
    # Identical files are one group whatever the curves.
    _, s3 = _al_describe(ms[:2], hashes={"home.motion3.json": "x",
                                         "login.motion3.json": "x"})
    assert s3["login.motion3.json"] == "home.motion3.json"
    # One prompt line names every label; each of them parses.
    labels = dr.gesture_labels([n for n, _ in ms])
    desc = {k: d[v] for k, v in labels.items() if d[v]}
    grp = {k: same[v] for k, v in labels.items()}
    p = dr.system_prompt("A.", labels, None, desc, grp)
    line = [ln for ln in p.splitlines() if ln.startswith("  pat")]
    assert line == ["  pat / poke / touch_body: same motion; "
                    + d["pat.motion3.json"]]
    assert "  home\n" in p                      # nothing distinctive
    assert not any(ln.startswith("  touch_body") for ln in p.splitlines())
    for lab in ("pat", "poke", "touch_body"):
        r = dr.parse(f"Hey [gesture:{lab}] you", gestures=labels)
        assert [x["arg"] for x in r.tags] == [lab]
    # Byte-stable whatever the order.
    assert p == dr.system_prompt("A.", dict(reversed(labels.items())), None,
                                 dict(reversed(desc.items())), grp)
    d2, same2 = _al_describe(list(reversed(ms)),
                             switches={"PARAM_eye_close": (0.0, 1.0)})
    assert (d2, same2) == (d, same)


def test_overrides_win_over_merged_lines(tmp_path):
    labels = {"a": "a.motion3.json", "b": "b.motion3.json",
              "c": "c.motion3.json"}
    measured = {n: "right arm moves; 6.5 s" for n in labels.values()}
    grp = {k: "a.motion3.json" for k in labels}
    f = tmp_path / "r.gestures.json"
    f.write_text(json.dumps({"b": "waves hello", "c": {"hide": True}}))
    keep, desc = gz.apply_overrides(labels, measured,
                                    gz.load_overrides(str(f)))
    g = dr.gesture_lines(keep, desc, grp)
    # The overridden label gets its own line, the hidden one is gone.
    assert "  a: right arm moves; 6.5 s\n" in g
    assert "  b: waves hello\n" in g
    assert g.splitlines()[1:] == ["  a: right arm moves; 6.5 s",
                                  "  b: waves hello"]
    p = dr.parse("[gesture:c] hi [gesture:b]", gestures=keep)
    assert [x["arg"] for x in p.tags] == ["b"]


def test_main_merges_identical_files(tmp_path, capsys):
    model, rig = _rig(tmp_path / "x")
    root = tmp_path / "x"
    (root / "wave2.motion3.json").write_bytes(
        (root / "wave.motion3.json").read_bytes())
    blob = json.loads(model.read_text())
    blob["FileReferences"]["Motions"][""].append(
        {"File": "wave2.motion3.json"})
    model.write_text(json.dumps(blob))
    args = ["--model-json", str(model), "--rig", str(rig), "--out",
            str(tmp_path / "o"), "--probe-dir", str(tmp_path),
            "--print-system"]
    assert dr.main(args) == 0
    cap = capsys.readouterr()
    assert ("  wave / wave2: same motion; left arm swings back and forth; "
            "4.0 s\n") in cap.out
    assert "director: gesture wave2 [measured, same motion as wave]" \
        in cap.err


def test_unbound_gaze_held_at_extremes_is_a_switch(tmp_path):
    """aidang_2 binds no gaze (its eyeballs are switches), so the gesture
    layer plays ParamEyeBallX, named "eyes" through its standard role. Held
    at its extremes and jumping, it is a switch, never "eyes swing"."""
    root = tmp_path / "g"
    root.mkdir()
    seg = [0.0, -1.0]
    for i in range(1, 9):                 # jump every 0.5 s, then hold
        v = 1.0 if i % 2 else -1.0
        seg += [0, i * 0.5 - 0.02, -v, 0, i * 0.5, v]
    (root / "look.motion3.json").write_text(json.dumps({
        "Version": 3, "Meta": {"Duration": 4.0, "Fps": 30},
        "Curves": [{"Target": "Parameter", "Id": "ParamEyeBallX",
                    "Segments": seg}]}))
    model = root / "g.model3.json"
    model.write_text(json.dumps({"Version": 3, "FileReferences": {
        "Moc": "g.moc3", "Motions": {"": [{"File": "look.motion3.json"}]}}}))
    role_of = {"ParamEyeBallX": "gaze_x"}
    sw = gz.rig_switches(str(model), ["ParamEyeBallX", "ParamArmLAngle"],
                         ["gaze", "arm"], FPS, role_of)
    assert list(sw) == ["ParamEyeBallX"]
    assert np.allclose(sw["ParamEyeBallX"], (-1.0, 1.0), atol=1e-4)
    from flow.runtime import authored_motions
    ms = authored_motions(str(model), FPS)
    lo, hi = {"ParamEyeBallX": -1.0}, {"ParamEyeBallX": 1.0}
    rest = {"ParamEyeBallX": -1.0}
    raw = gz.describe_motions(ms, lambda p: True, lo, hi, rest,
                              role_of=role_of, fps=FPS)
    assert raw["look.motion3.json"].startswith("eyes swing back and forth")
    d = gz.describe_motions(ms, lambda p: True, lo, hi, rest,
                            role_of=role_of, fps=FPS, switches=sw)
    assert d["look.motion3.json"].startswith("eyes change;")
