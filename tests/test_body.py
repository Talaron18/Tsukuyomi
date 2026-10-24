"""
flow.body without GL: region names, the forward model on a synthetic rig
whose geometry is known in closed form, solver round trips, the file format.

The synthetic rig (FakeCore) has the Core model's interface and a body like
Hiyori's: a face that rolls about the chin with ParamAngleZ (a third of the
declared degrees, as Hiyori's does), a torso, two hanging arms in PartArmA
that rotate about the shoulders, a left hand that turns at the wrist, a
raised left arm in PartArmB (a pose-group swap) and a whole-model shift.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from flow import body


def _box(cx, cy, w, h, n=4):
    xs = np.linspace(cx - w / 2, cx + w / 2, n)
    ys = np.linspace(cy - h / 2, cy + h / 2, n)
    return np.array([(x, y) for x in xs for y in ys], float)


def _rot(P, c, deg):
    t = math.radians(deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (P - c) @ R.T + c


class FakeCore:
    """l2dprobe.core.CoreModel's interface over closed-form geometry."""

    SHOULDER = {"L": np.array([0.4, 0.8]), "R": np.array([-0.4, 0.8])}
    WRIST_L = np.array([0.45, 0.3])

    def __init__(self):
        self.part_ids = ["PartCore", "PartFace", "PartBody", "PartArmA",
                         "PartArmB"]
        self.part_parent = np.array([-1, 0, 0, 0, 0])
        # (id, part, rest vertices)
        D = [
            ("ArtMesh0", 1, _box(0.0, 1.2, 0.5, 0.6)),       # face
            ("ArtMesh1", 2, _box(0.0, 0.4, 0.8, 0.8)),       # torso
            ("ArtMesh2", 3, _box(0.42, 0.55, 0.1, 0.5)),     # upper arm L
            ("ArtMesh3", 3, _box(0.45, 0.2, 0.12, 0.16)),    # hand L
            ("ArtMesh4", 3, _box(-0.42, 0.55, 0.1, 0.5)),    # upper arm R
            ("ArtMesh5", 3, _box(-0.45, 0.2, 0.12, 0.16)),   # hand R
            ("ArtMesh6", 4, _box(0.55, 0.9, 0.3, 0.1)),      # raised arm L
            ("ArtMesh7", 4, _box(0.8, 1.05, 0.12, 0.16)),    # raised hand L
        ]
        self.drawable_ids = [d[0] for d in D]
        self.drawable_parent = np.array([d[1] for d in D])
        self._rest = [d[2] for d in D]
        self.vertex_counts = np.array([len(v) for v in self._rest])
        self.param_ids = ["ParamAngleX", "ParamAngleZ", "ParamBodyAngleZ",
                          "ParamArmLA", "ParamArmRA", "ParamHandL",
                          "ParamArmLB", "ParamAllX"]
        self.param_min = np.array([-30, -30, -10, -10, -10, -1, -10, -1.0])
        self.param_max = np.array([30, 30, 10, 10, 10, 1, 10, 1.0])
        self.param_default = np.zeros(8)
        self._part_op = np.ones(len(self.part_ids))
        self.set()

    @property
    def offsets(self):
        return np.concatenate([[0], np.cumsum(self.vertex_counts)])

    def part_chain(self, part):
        out = []
        while part >= 0:
            out.append(self.part_ids[part])
            part = int(self.part_parent[part])
        return out

    def set(self, values=None, parts=None, reset=True):
        v = dict(zip(self.param_ids, self.param_default))
        v.update({k: float(np.clip(x, self.param_min[self.param_ids.index(k)],
                                   self.param_max[self.param_ids.index(k)]))
                  for k, x in (values or {}).items() if k in v})
        if reset:
            self._part_op = np.ones(len(self.part_ids))
        for p, x in (parts or {}).items():
            if p in self.part_ids:
                self._part_op[self.part_ids.index(p)] = x
        chin = np.array([0.0, 0.9])
        out = []
        for i, P in enumerate(self._rest):
            P = P.copy()
            if i == 0:
                P = _rot(P, chin, -v["ParamAngleZ"] / 3.0)
                P[:, 0] += 0.002 * v["ParamAngleX"]
            if i == 3:
                P = _rot(P, self.WRIST_L, 30.0 * v["ParamHandL"])
            if i in (2, 3):
                P = _rot(P, self.SHOULDER["L"], 3.0 * v["ParamArmLA"])
            if i in (4, 5):
                P = _rot(P, self.SHOULDER["R"], -3.0 * v["ParamArmRA"])
            if i in (6, 7):
                P = _rot(P, self.SHOULDER["L"], 3.0 * v["ParamArmLB"])
            if i != 1:
                # The body bends above the hips; the torso box stays put
                # below them (feet do not move).
                P = _rot(P, np.array([0.0, 0.0]), 0.5 * v["ParamBodyAngleZ"])
            P[:, 0] += 0.2 * v["ParamAllX"]
            out.append(P)
        self._V = out
        return None

    def vertices(self, i):
        return self._V[i].copy()

    def all_vertices(self):
        return np.concatenate(self._V)

    def opacities(self):
        return np.array([self._part_op[p] * self._part_op[0]
                         for p in self.drawable_parent])

    def render_orders(self):
        return np.arange(len(self.drawable_ids))


def _info():
    return {"part_names": {"PartArmA": "腕 A", "PartArmB": "腕 B"},
            "param_names": {"ParamHandL": "左手", "ParamArmLA": "左腕 A",
                            "ParamArmRA": "右腕 A", "ParamArmLB": "左腕 B"},
            "pose_groups": [["PartArmA", "PartArmB"]],
            "bands": {}, "groups": {}, "toggles": {}, "held": set(),
            "physics": set(), "motion_frames": [],
            "band_of": lambda pid: "expressive"}


@pytest.fixture(scope="module")
def measured():
    core = FakeCore()
    blob = body.measure(core, _info(), log=lambda *a: None)
    return core, blob


# ------------------------------------------------------------- names

@pytest.mark.parametrize("texts,leaf", [
    (["ArtMesh12", "PartArmALFore", "左腕A 前腕"], "forearm"),
    (["ArtMesh21", "PartEyeBlow", "まゆ毛"], "brow"),
    (["D_ARM_L_01", "PartArmL"], "arm"),
    (["ArtMesh131", "ArtMesh55_Skinning", "前髪左(スキニング)"], "hair"),
    (["ArtMesh115", "ArtMesh61_Skinning", "サイドアップ右(スキニング)"], "hair"),
    (["x", "PartHand11", "手袋_1"], "hand"),
    (["shou_L", ""], "hand"),            # pinyin hand
    (["PARAM_shoulder", ""], ""),       # not a hand, and no region word
    (["tou", ""], "face"),               # pinyin head -> its face region
    (["yan_L", ""], "eye"),
    (["ArtMesh9", "PartNeck", "首"], "neck"),
    (["ArtMesh", "PartSketch", "[ 下絵 ]"], "background"),
    (["ArtMesh99"], ""),
])
def test_name_leaf(texts, leaf):
    assert body.name_leaf(texts)[0] == leaf


def test_text_side():
    assert body.text_side(["ArtMesh1", "PartArmAL"]) == "L"
    assert body.text_side(["ArtMesh1", "PartEyeR", "右目"]) == "R"
    assert body.text_side(["ArtMesh1", "", "左腕A"]) == "L"
    assert body.text_side(["ArtMesh1", "PartArmA", "腕 A"]) == ""


def test_assign_regions_splits_arm_by_motion():
    """An arm named only as an arm: the drawable the wrist param moves is
    the hand, the side comes from the sided arm param, a nameless drawable
    the head params move is unclear with a head guess."""
    params = [body.param_meta("ParamAngleX"),
              body.param_meta("ParamArmLA", "左腕 A", "arm"),
              body.param_meta("ParamHandL", "左手", "arm")]
    texts = [["A", "PartArmA", "腕 A"], ["B", "PartArmA", "腕 A"],
             ["C", "PartFace", "顔"], ["D"]]
    rest = [[0.4, 0.5], [0.45, 0.2], [0.0, 1.2], [0.0, 1.3]]
    resp = np.array([[0.0, 1.0, 0.0],      # upper arm: arm param only
                     [0.0, 2.0, 1.0],      # hand: wrist moves it
                     [1.0, 0.0, 0.0],
                     [0.8, 0.0, 0.0]])
    out = body.assign_regions(texts, rest, resp, params, face_x=0.0)
    assert [o["region"] for o in out] == ["upper_arm.L", "hand.L", "face",
                                          "unclear"]
    assert out[3]["guess"] == "head"
    assert out[1]["confidence"] == "high"


def test_canon_spec_units():
    spec = body.canon_spec()
    assert spec["head.yaw"]["source"] == "declared"
    assert spec["hand.R.x"] == {**spec["hand.R.x"], "unit": "hu",
                                "source": "measured"}
    assert body.feature_unit("head.roll") == "deg"
    assert body.feature_unit("mouth.open") == "unit"


# ------------------------------------------------------ forward model

def test_frame_regions_and_roles(measured):
    _core, blob = measured
    regions = [d["region"] for d in blob["drawables"]]
    # The right arm and the raised left arm have no wrist parameter: they
    # stay whole ("arm") and their hands are the distal vertices.
    assert regions == ["face", "torso", "upper_arm.L", "hand.L",
                       "arm.R", "arm.R", "arm.L", "arm.L"]
    assert blob["frame"]["unit"] == pytest.approx(0.6)
    assert blob["frame"]["origin"] == pytest.approx([0.0, 0.9])
    assert blob["params"]["ParamAllX"]["role"] == "held"
    assert blob["params"]["ParamArmLA"]["role"] == "solvable"
    assert [v["name"] for v in blob["variants"]] == ["PartArmA", "PartArmB"]
    # The raised variant has no right arm.
    k = blob["features"].index("hand.R.x")
    assert blob["variants"][0]["present"][k]
    assert not blob["variants"][1]["present"][k]


def test_forward_matches_geometry(measured, tmp_path):
    """At sweep points and off-grid, single and paired, the saved and
    reloaded model reproduces the synthetic rig's exact features."""
    core, blob = measured
    path = tmp_path / "fake.body.json"
    body.save(blob, str(path))
    bm = body.load(str(path))
    for vals in ({"ParamArmLA": 5.0}, {"ParamAngleZ": 18.0},
                 {"ParamArmLA": -3.7, "ParamHandL": 0.6},
                 {"ParamArmLA": 7.0, "ParamBodyAngleZ": -6.0},
                 {"ParamAngleX": 11.0, "ParamBodyAngleZ": 4.0}):
        ex = body.exact_state(core, bm, vals, 0)
        got = bm.perceive(vals, 0)
        for f in ("hand.L.x", "hand.L.y", "hand.L.roll", "head.roll",
                  "head.x", "hand.R.y", "head.yaw"):
            tol = 0.5 if f.endswith("roll") else 0.01
            assert got[f] == pytest.approx(ex[f], abs=tol), (vals, f)
    # Closed form: ParamAngleZ +30 rolls the face -10 degrees.
    assert bm.perceive({"ParamAngleZ": 30.0}, 0)["head.roll"] == \
        pytest.approx(-10.0, abs=0.05)
    assert bm.perceive({}, 0)["head.yaw"] == 0.0


def test_save_load_round_trip(measured, tmp_path):
    _core, blob = measured
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    body.save(blob, str(p1))
    bm = body.load(str(p1))
    body.save(bm.to_dict(), str(p2))
    assert p1.read_text() == p2.read_text()
    with open(p1, encoding="utf-8") as fh:
        head = fh.read(200)
    assert body.FORMAT in head
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "other"}')
    with pytest.raises(ValueError):
        body.load(str(bad))


# ------------------------------------------------------------ solver

@pytest.fixture(scope="module")
def model(measured, tmp_path_factory):
    _core, blob = measured
    path = tmp_path_factory.mktemp("b") / "fake.body.json"
    body.save(blob, str(path))
    return body.load(str(path))


@pytest.mark.parametrize("vals", [
    {"ParamArmLA": 4.0, "ParamAngleZ": -12.0},
    {"ParamArmLA": -6.0, "ParamHandL": 0.5, "ParamArmRA": 3.0},
    {"ParamAngleX": 20.0, "ParamArmRA": -8.0, "ParamBodyAngleZ": 3.0},
])
def test_solve_round_trip(model, vals):
    """targets -> params -> forward -> targets, within tolerance, bounds
    respected, and perceive reads the same state back."""
    state = model.perceive(vals, 0)
    targets = {f: state[f] for f in ("hand.L.x", "hand.L.y", "hand.R.y",
                                     "head.roll", "head.yaw")}
    res = model.solve(targets)
    assert res["variant"] == 0
    for t, e in res["report"].items():
        assert e["ok"], (t, e)
    for p, x in res["params"].items():
        e = model.params[p]
        assert e["min"] - 1e-9 <= x <= e["max"] + 1e-9
    back = model.perceive(res["params"], parts=res["parts"])
    for t, y in targets.items():
        assert back[t] == pytest.approx(y, abs=body.TOL[
            body.feature_unit(t)])


def test_solve_respects_limits(model):
    """An unreachable target ends at the bound, reported as a miss."""
    res = model.solve({"head.yaw": 90.0})
    assert res["params"]["ParamAngleX"] == pytest.approx(30.0)
    assert not res["report"]["head.yaw"]["ok"]
    # Head roll: 10 degrees from ParamAngleZ plus 5 from ParamBodyAngleZ.
    lo, hi = model.limits("head.roll", 0)
    assert lo == pytest.approx(-15.0, abs=0.2)
    assert hi == pytest.approx(15.0, abs=0.2)


def test_solve_picks_switch_variant(model):
    """A left hand above the shoulder only the raised arm (PartArmB) can
    reach: the solver swaps the pose part; the held whole-model shift is
    never used."""
    y = model.landmarks["shoulder.y"] + 0.3
    res = model.solve({"hand.L.y": y})
    assert res["variant"] == 1
    assert res["parts"] == {"PartArmA": 0.0, "PartArmB": 1.0}
    assert res["report"]["hand.L.y"]["ok"]
    assert "ParamAllX" not in res["params"]
    # And with the variant pinned to the hanging arm it cannot.
    pinned = model.solve({"hand.L.y": y}, variant=0)
    assert not pinned["report"]["hand.L.y"]["ok"]


def test_landmark_targets(model):
    t = model.resolve({"hand.L.y": "@shoulder.y", "hand.L.x": "@chest.x+0.1",
                       "head.y": "@head.y-0.05"})
    assert t["hand.L.y"] == pytest.approx(model.landmarks["shoulder.y"])
    assert t["hand.L.x"] == pytest.approx(model.landmarks["chest"][0] + 0.1)
    assert t["head.y"] == pytest.approx(model.landmarks["head"][1] - 0.05)
    with pytest.raises(ValueError):
        model.resolve({"hand.L.y": "@elbow.y"})


def test_whole_model_detector():
    V0 = _box(0, 0, 1, 2, 6)
    assert body.whole_model(V0, V0 + [0.1, 0.0])
    assert body.whole_model(V0, _rot(V0, np.array([0.0, -1.0]), 5.0))
    bent = V0.copy()
    bent[V0[:, 1] > 0] = _rot(V0[V0[:, 1] > 0], np.array([0.0, 0.0]), 10.0)
    assert not body.whole_model(V0, bent)


def test_variants_without_pose3():
    """Azur Lane rigs ship no pose3: their motions switch arm parts with
    PartOpacity curves. The variants come from those curves, and the rest
    one (the idle motion's first frame) shows one arm, not both."""
    info = _info()
    info["pose_groups"] = []
    n = 20
    on, off = np.ones(n), np.zeros(n)
    info["motion_frames"] = [
        {("Meta", "name"): "idle.motion3.json",
         ("PartOpacity", "PartArmA"): on, ("PartOpacity", "PartArmB"): off},
        {("Meta", "name"): "wave.motion3.json",
         ("PartOpacity", "PartArmA"): np.r_[on[:5], off[5:]],
         ("PartOpacity", "PartArmB"): np.r_[off[:5], on[5:]]},
    ]
    blob = body.measure(FakeCore(), info, log=lambda *a: None)
    assert len(blob["variants"]) == 2
    v0, v1 = blob["variants"]
    assert v0["parts"] == {"PartArmA": 1.0, "PartArmB": 0.0}
    assert v1["parts"] == {"PartArmA": 0.0, "PartArmB": 1.0}
    assert blob["rest_parts"] == {"PartArmA": 1.0, "PartArmB": 0.0}
    # Variant 0 measures the hanging arm alone: its hand is where the
    # hanging hand is, not averaged with the raised one.
    k = blob["features"].index("hand.L.y")
    assert v0["rest"][k] < 0 < v1["rest"][k]


class GloveCore(FakeCore):
    """FakeCore whose raised hand has a glove shown by ParamGloveShow, a
    switch under PartArmB, which is hidden at rest."""

    def __init__(self):
        super().__init__()
        self.param_ids = self.param_ids + ["ParamGloveShow"]
        self.param_min = np.r_[self.param_min, 0.0]
        self.param_max = np.r_[self.param_max, 1.0]
        self.param_default = np.r_[self.param_default, 0.0]
        self.set()

    def opacities(self):
        op = super().opacities()
        if hasattr(self, "_glove"):
            op[7] *= self._glove
        return op

    def set(self, values=None, parts=None, reset=True):
        self._glove = float((values or {}).get("ParamGloveShow", 0.0) > 0.5)
        return super().set({k: v for k, v in (values or {}).items()
                            if k != "ParamGloveShow"}, parts, reset)


def test_switch_under_hidden_switched_part():
    """A show/hide switch under a part the motions hide at rest is still
    found: the opacity pass shows every switched part."""
    info = _info()
    info["pose_groups"] = []
    info["param_names"]["ParamGloveShow"] = "手袋 表示"
    n = 20
    on, off = np.ones(n), np.zeros(n)
    info["motion_frames"] = [
        {("Meta", "name"): "idle.motion3.json",
         ("PartOpacity", "PartArmA"): on, ("PartOpacity", "PartArmB"): off},
        {("Meta", "name"): "wave.motion3.json",
         ("PartOpacity", "PartArmA"): off, ("PartOpacity", "PartArmB"): on,
         ("Parameter", "ParamGloveShow"): on},
    ]
    blob = body.measure(GloveCore(), info, log=lambda *a: None)
    assert blob["params"]["ParamGloveShow"]["role"] == "variant"
    assert any(v["switches"].get("ParamGloveShow") == 1.0
               for v in blob["variants"])


def test_declared_signs_from_geometry(measured, model):
    """ParamAngleZ + rolls the fake face clockwise (like Hiyori's), so
    head.tilt is the negated parameter and agrees with head.roll."""
    _core, blob = measured
    assert blob["declared"]["head.tilt"][1] < 0
    assert blob["declared"]["head.yaw"][1] > 0
    st = model.perceive({"ParamAngleZ": -30.0}, 0)
    assert st["head.tilt"] > 0 and st["head.roll"] > 0
    assert "head.tilt" in blob["meaning"]


def test_autonomic_not_used_for_pose(measured):
    """Blink and breath are the runtime's; a pose never leans on them."""
    _core, blob = measured
    blob = dict(blob)
    blob["params"] = {p: dict(e) for p, e in blob["params"].items()}
    blob["params"]["ParamBodyAngleZ"]["band"] = "autonomic"
    bm = body.BodyModel(blob)
    res = bm.solve({"head.roll": 12.0})
    assert "ParamBodyAngleZ" not in res["params"]


# ------------------------------------------------------ real Core (opt)

def test_core_reads_hiyori(samples):
    """The Cubism Core through ctypes: needs live2d-py, no GL."""
    from l2dprobe import core
    if not core.available():
        pytest.skip("no Cubism Core (live2d-py) importable")
    from tests.conftest import sample_model
    m = core.load(sample_model(samples, "Hiyori"))
    m.set()
    V0 = m.all_vertices()
    m.set({"ParamArmLA": 10.0})
    moved = np.linalg.norm(m.all_vertices() - V0, axis=1) > 1e-4
    # The character's left arm is at +x.
    assert moved.sum() > 50 and V0[moved, 0].mean() > 0.05
    m.set(parts={"PartArmA": 0.0, "PartArmB": 1.0})
    parts = [m.part_ids[p] for p in m.drawable_parent]
    op = m.opacities()
    assert max(op[i] for i, p in enumerate(parts) if p == "PartArmA") == 0
    assert min(op[i] for i, p in enumerate(parts) if p == "PartArmB") == 1


# ------------------------------------------- side naming and effects

MIRROR = {"ParamArmLA": "ParamArmRA", "ParamArmRA": "ParamArmLA",
          "ParamHandL": "ParamHandR", "ParamArmLB": "ParamArmRB"}


class MirroredCore(FakeCore):
    """FakeCore with its parameters named by screen side (the Azur Lane
    way): the arm at +x, the character's left, is ParamArmRA. With
    sided_drawables the drawables are named by screen side too."""

    def __init__(self, sided_drawables=False):
        super().__init__()
        self.param_ids = [MIRROR.get(p, p) for p in self.param_ids]
        if sided_drawables:
            self.drawable_ids = ["D_FACE", "D_BODY", "D_ARM_R_01",
                                 "D_ARM_R_02", "D_ARM_L_01", "D_ARM_L_02",
                                 "D_ARM_R_03", "D_ARM_R_04"]

    def set(self, values=None, parts=None, reset=True):
        inv = {v: k for k, v in MIRROR.items()}
        shown, self.param_ids = self.param_ids, [inv.get(p, p) for p in
                                                 self.param_ids]
        try:
            return super().set({inv.get(k, k): x for k, x in
                                (values or {}).items()}, parts, reset)
        finally:
            self.param_ids = shown


def _mirrored_info():
    info = _info()
    info["param_names"] = {"ParamHandR": "右手", "ParamArmRA": "右腕 A",
                           "ParamArmLA": "左腕 A", "ParamArmRB": "右腕 B"}
    return info


def test_detect_naming_character(measured):
    _core, blob = measured
    assert blob["side_naming"]["params"] == "character"
    assert blob["side_naming"]["votes"]["params"][1] == 0


def test_screen_named_params_give_the_same_body(measured):
    """Parameters named by screen side: detected, and the body (regions,
    sides, frame, rest features) is the one the character-named rig has;
    a hand target on the character's left moves the rig's ...R param."""
    _core, ref = measured
    blob = body.measure(MirroredCore(), _mirrored_info(), log=lambda *a: None)
    assert blob["side_naming"]["params"] == "screen"
    assert blob["side_naming"]["drawables"] == "character"   # no votes
    assert [d["region"] for d in blob["drawables"]] == \
        [d["region"] for d in ref["drawables"]]
    assert blob["landmarks"] == ref["landmarks"]
    np.testing.assert_allclose(blob["variants"][0]["rest"],
                               ref["variants"][0]["rest"])
    bm = body.BodyModel(blob)
    y0 = blob["landmarks"]["hand.L"][1]
    res = bm.solve({"hand.L.y": y0 + 0.1}, variant=0)
    assert res["report"]["hand.L.y"]["ok"]
    assert abs(res["params"].get("ParamArmRA", 0.0)) > 0.5
    assert abs(res["params"].get("ParamArmLA", 0.0)) < 0.5


def test_screen_named_drawables_detected():
    """Drawables named by screen side too: both conventions detected, and
    every arm drawable's side is its rest position's."""
    core = MirroredCore(sided_drawables=True)
    blob = body.measure(core, _mirrored_info(), log=lambda *a: None)
    assert blob["side_naming"]["drawables"] == "screen"
    assert blob["side_naming"]["params"] == "screen"
    for d, P in zip(blob["drawables"], core._rest):
        if d["side"]:
            assert d["side"] == ("L" if P[:, 0].mean() > 0 else "R"), d
    assert blob["landmarks"]["hand.L"][0] > 0 > \
        blob["landmarks"]["hand.R"][0]


def test_mirror_id():
    assert body.mirror_id("ParamEyeLOpen") == "ParamEyeROpen"
    assert body.mirror_id("ParamBrowRAngle") == "ParamBrowLAngle"
    assert body.mirror_id("ParamLeftShoulderUp") == "ParamRightShoulderUp"
    assert body.mirror_id("ParamAngleX") == "ParamAngleX"


class FxCore(FakeCore):
    """FakeCore plus an effect: a fist blur named hand_fx in the arm part,
    over the face, shown and moved by PARAM_FXhand, and a hit-area mesh
    over the head."""

    def __init__(self):
        super().__init__()
        self.part_ids.append("PartTouch")
        self.part_parent = np.append(self.part_parent, 0)
        self.drawable_ids += ["hand_fx", "TouchHead"]
        self.drawable_parent = np.append(self.drawable_parent, [3, 5])
        self._rest += [_box(0.1, 1.1, 0.4, 0.4), _box(0.0, 1.3, 0.8, 1.2)]
        self.vertex_counts = np.array([len(v) for v in self._rest])
        self.param_ids.append("PARAM_FXhand")
        self.param_min = np.append(self.param_min, 0.0)
        self.param_max = np.append(self.param_max, 1.0)
        self.param_default = np.append(self.param_default, 0.0)
        self._part_op = np.ones(len(self.part_ids))
        self.set()

    def set(self, values=None, parts=None, reset=True):
        values = dict(values or {})
        self._fx = float(values.get("PARAM_FXhand", 0.0))
        super().set(values, parts, reset)
        if len(self._V) > 8:        # FakeCore.__init__ sets before us
            self._V[8] = self._V[8] + np.array([0.1 * self._fx, 0.0])

    def opacities(self):
        op = super().opacities()
        op[8] *= float(self._fx > 0.5)
        return op


def test_effects_never_variants_or_solved():
    info = _info()
    info["toggles"] = {"PARAM_FXhand": [0.0, 1.0]}
    core = FxCore()
    blob = body.measure(core, info, log=lambda *a: None)
    regions = {d["id"]: d["region"] for d in blob["drawables"]}
    assert regions["hand_fx"] == "effect"
    assert regions["TouchHead"] == "hitarea"
    assert blob["params"]["PARAM_FXhand"]["role"] == "effect"
    assert "PARAM_FXhand" in blob["effects"]["params"]
    assert "hand_fx" in blob["effects"]["drawables"]
    for v in blob["variants"]:
        assert "PARAM_FXhand" not in v["switches"]
        assert "PARAM_FXhand" not in v["sweeps"]
    # The hit area does not set the frame: the face alone does.
    assert blob["frame"]["unit"] == pytest.approx(0.6)
    bm = body.BodyModel(blob)
    res = bm.solve({"hand.L.y": blob["landmarks"]["hand.L"][1] + 0.1})
    assert "PARAM_FXhand" not in res["params"]


def test_miss_reported_with_reach(model, tmp_path, capsys):
    """An unreachable hand target: MISS per target, the reachable range
    alone, the other side's range, and the render still runs (here: no
    GL, so only the report)."""
    res = model.solve({"hand.R.y": 5.0})
    assert not res["report"]["hand.R.y"]["ok"]
    h = model.miss_hints(res)["hand.R.y"]
    lo, hi = h["reach"]
    assert lo < res["report"]["hand.R.y"]["got"] + 1e-6 <= hi + 1e-6
    assert h["mirror"][0] == "hand.L.y" and h["mirror"][1] is not None
    path = tmp_path / "m.body.json"
    body.save(model.to_dict(), str(path))
    body.main(["--body", str(path), "--solve", '{"hand.R.y": 5.0}'])
    out = capsys.readouterr().out
    assert "MISS" in out and "alone it reaches" in out
    assert "hand.L.y reaches" in out
    assert "missed jointly" not in out
    # Reachable alone, not together with an impossible partner.
    y = model.landmarks["hand.R"][1]
    body.main(["--body", str(path), "--solve",
               json.dumps({"hand.R.y": y + 0.05, "hand.R.x": 5.0}),
               "--current", "{}"])
    out = capsys.readouterr().out
    assert "MISS" in out


def test_v1_file_upgrades(measured, tmp_path):
    """A format-1 file of a character-named rig loads unchanged but for
    the new fields."""
    _core, blob = measured
    old = {k: v for k, v in blob.items() if k not in ("side_naming",
                                                      "effects")}
    old["format"] = body.FORMAT_V1
    path = tmp_path / "v1.body.json"
    body.save(old, str(path))
    bm = body.load(str(path))
    assert bm.blob["format"] == body.FORMAT
    assert bm.blob["side_naming"]["params"] == "character"
    assert bm.features == blob["features"]
    assert bm.landmarks == blob["landmarks"]
    np.testing.assert_allclose(bm.variants[0]["rest"],
                               blob["variants"][0]["rest"], rtol=1e-4)


RIGS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "rigs")


@pytest.mark.parametrize("rig,drawables,params", [
    ("lafei", "screen", "screen"), ("xuefeng", "character", "screen")])
def test_azur_lane_v1_files(rig, drawables, params):
    """The measured Azur Lane files (format 1): naming detected, the
    character's right hand on -x moved by the rig's ...LT param, effects
    out of the variants."""
    path = os.path.join(RIGS, f"{rig}.body.json")
    if not os.path.exists(path):
        pytest.skip(f"no {path}")
    bm = body.load(path)
    sn = bm.blob["side_naming"]
    assert (sn["drawables"], sn["params"]) == (drawables, params)
    assert bm.landmarks["hand.R"][0] < 0 < bm.landmarks["hand.L"][0]
    # The rig's screen-named L arm (upper ...LT_Angle, or ...LB_Agle once a
    # re-measure solves it as the continuous deformer it is: xuefeng moves
    # 9 drawables up to 7.7 model units with it, no opacity change).
    assert bm.movers("hand.R", 0, top=1)[0][0].startswith("PARAM_Arm_L")
    assert bm.blob["declared"]["eye.L.open"][0] == "ParamEyeROpen"
    for v in bm.variants:
        assert not any("FX" in k for k in v["switches"]), v["name"]
    res = bm.solve({"hand.R.y": "@shoulder.y"})
    assert res["report"]["hand.R.y"]["ok"]
    assert not any(bm.params[p]["role"] == "effect" for p in res["params"])
