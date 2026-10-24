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
