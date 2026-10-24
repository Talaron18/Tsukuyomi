"""
Continuous deformers the probe called discrete (xuefeng's ParamEyeL/ROpen:
categorical from pixels, so banded discrete and role "switch", so never
blinked procedurally nor solved): the Core geometry check
(l2dprobe.core.continuous_deformer), the runtime's re-banding and snap filter
(flow.runtime.reband, snap_set), and the body roles, on a synthetic face
whose eyelid is a pure deformer and whose other eye parameter swaps art by
opacity. Without a Core the bands are what they were.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from flow import body
from flow import runtime as rt
from l2dprobe import core
from l2dprobe.motions import band_of

from test_body import FakeCore, _info


class FaceCore:
    """CoreModel's interface: an upper lid that slides down linearly as
    ParamEyeLOpen closes, an eye that swaps open / closed drawings by opacity
    (ParamEyeROpen), a mouth that opens by deformation, a smile drawn as a
    deformer and one drawn as a swap, a parameter whose mesh jumps half way,
    and one that moves nothing."""

    def __init__(self):
        self.param_ids = ["ParamEyeLOpen", "ParamEyeROpen", "ParamMouthOpenY",
                          "ParamEyeLSmile", "ParamEyeRSmile", "ParamArmLSwitch",
                          "ParamEyeJump", "ParamEyeDead"]
        n = len(self.param_ids)
        self.param_min = np.zeros(n, np.float32)
        self.param_max = np.ones(n, np.float32)
        self.param_default = np.array([1, 1, 0, 0, 0, 0, 0, 0], np.float32)
        self.part_ids = ["PartFace"]
        self.drawable_ids = ["lid_L", "eye_R_open", "eye_R_closed", "mouth",
                             "smile_L", "smile_R_a", "smile_R_b", "arm"]
        self._rest = [np.array([[0.0, 1.0], [0.2, 1.0], [0.1, 1.1]])
                      + [i, 0] for i in range(len(self.drawable_ids))]
        self.vertex_counts = np.array([len(v) for v in self._rest])
        self.set()

    @property
    def offsets(self):
        return np.concatenate([[0], np.cumsum(self.vertex_counts)])

    def set(self, values=None, parts=None, reset=True):
        v = dict(zip(self.param_ids, self.param_default.tolist()))
        v.update(values or {})
        V = [P.copy() for P in self._rest]
        V[0][:, 1] -= 0.76 * (1 - v["ParamEyeLOpen"])        # the lid closes
        V[3][:, 1] -= 0.1 * v["ParamMouthOpenY"]
        V[4][:, 0] += 0.05 * v["ParamEyeLSmile"]
        V[7][:, 0] += 0.3 * v["ParamArmLSwitch"]             # moves, no swap
        V[5][:, 1] += 0.4 * (v["ParamEyeJump"] > 0.5)        # a step
        self._V = V
        op = np.ones(len(self.drawable_ids))
        r = v["ParamEyeROpen"]
        op[1], op[2] = float(r > 0.5), float(r <= 0.5)
        s = v["ParamEyeRSmile"]
        op[5], op[6] = 1 - s, s
        self._op = op

    def vertices(self, i):
        return self._V[i].copy()

    def all_vertices(self):
        return np.concatenate(self._V)

    def opacities(self):
        return self._op.copy()


# ---------------------------------------------------------- core check

@pytest.mark.parametrize("pid,expect,why", [
    ("ParamEyeLOpen", True, "no opacity change"),
    ("ParamMouthOpenY", True, "monotonic"),
    ("ParamEyeLSmile", True, "1 drawables move"),
    ("ParamEyeROpen", False, "opacity changes by 1.000 (eye_R_open)"),
    ("ParamEyeRSmile", False, "opacity changes"),
    ("ParamEyeJump", False, "vertices jump"),
    ("ParamEyeDead", False, "moves no vertex"),
    ("ParamMissing", False, "not in the moc"),
])
def test_continuous_deformer(pid, expect, why):
    ok, reason = core.continuous_deformer(FaceCore(), pid)
    assert ok is expect and why in reason, reason


def test_sweep_reads_linear_lid():
    """The finding on xuefeng: the lid moves 0.076 per 0.1 step, linearly,
    opacity 1 throughout."""
    sw = core.sweep(FaceCore(), "ParamEyeLOpen")
    assert sw["move"][:, 0] == pytest.approx(np.linspace(0.76, 0, 11))
    assert sw["step"] == pytest.approx(np.full(10, 0.076))
    assert (sw["opacity"][:, 0] == 1).all()


# ------------------------------------------------------------ runtime

@pytest.fixture
def face(tmp_path, monkeypatch):
    """A model json on disk (its mtime keys the cache), the fake Core, and
    the probe's bands with every eye and mouth parameter categorical."""
    mj = tmp_path / "face.model3.json"
    mj.write_text(json.dumps({"FileReferences": {"Moc": "face.moc3"}}))
    rt._CONTINUOUS_CACHE.clear()
    monkeypatch.setattr(rt, "_load_core", lambda m: FaceCore())
    m = FaceCore()
    pids = list(m.param_ids)
    groups = ["eyelid", "eyelid", "mouth", "eye", "eye", "arm", "eye", "eye"]
    kinds = ["categorical"] * 3 + ["continuous"] * 3 + ["categorical"] * 2
    bands = [band_of(p, k, g, set()) for p, k, g in zip(pids, kinds, groups)]
    assert bands[:3] == ["discrete"] * 3
    yield str(mj), pids, kinds, groups, bands
    rt._CONTINUOUS_CACHE.clear()


def test_reband_deformer_not_swap(face):
    mj, pids, kinds, groups, bands = face
    new, cont = rt.reband(pids, kinds, groups, bands, mj)
    by = dict(zip(pids, new))
    # The lid is a deformer: blinked procedurally (autonomic), so smooth.
    assert by["ParamEyeLOpen"] == "autonomic"
    # The opacity swap stays discrete; so do the jump and the dead one.
    assert by["ParamEyeROpen"] == "discrete"
    assert by["ParamEyeJump"] == by["ParamEyeDead"] == "discrete"
    assert by["ParamMouthOpenY"] == "mouth"
    # Nothing moves into the expressive band (the clips' columns).
    assert [b == "expressive" for b in new] == \
        [b == "expressive" for b in bands]
    # Evidence for every face deformer; the arm is not judged.
    assert set(cont) == {"ParamEyeLOpen", "ParamMouthOpenY",
                         "ParamEyeLSmile"}
    keep = rt.base_keep(set(), set(), by)
    assert not keep("ParamEyeLOpen") and keep("ParamEyeROpen")


def test_snap_set_spares_face_deformers(face):
    """Motions that key a smile 0/1 make it switch-like; the deformer still
    eases, the art swap and the arm toggle are snapped as before."""
    mj, pids, kinds, groups, bands = face
    new, cont = rt.reband(pids, kinds, groups, bands, mj)
    found = {p: (0.0, 0.0, 1.0) for p in pids}
    rows = range(len(pids))
    snapped = rt.snap_set(found, pids, groups, new, rows, cont)
    assert set(snapped) == {"ParamEyeRSmile", "ParamArmLSwitch"}
    # Without the evidence (no Core) the deformer is snapped, as before.
    assert "ParamEyeLSmile" in rt.snap_set(found, pids, groups, new, rows)


def test_no_core_keeps_bands(face, monkeypatch):
    mj, pids, kinds, groups, bands = face
    monkeypatch.setattr(rt, "_load_core", lambda m: None)
    new, cont = rt.reband(pids, kinds, groups, bands, mj)
    assert new == bands and cont == {}
    # A model file that is not there: the same.
    rt._CONTINUOUS_CACHE.clear()
    assert rt.reband(pids, kinds, groups, bands, mj + ".gone") == (bands, {})


def test_rig_columns_rebands(face, tmp_path):
    """End to end through the rig file: the eyelid leaves the discrete band
    and rig_columns says so."""
    mj, pids, kinds, groups, bands = face
    n = len(pids)
    rig = tmp_path / "face.npz"
    np.savez(rig, pid=np.array(pids), kind=np.array(kinds),
             group=np.array(groups), minimum=np.zeros(n),
             maximum=np.ones(n), default=np.ones(n),
             descriptor=np.zeros((n, 4)))
    logs = []
    col = rt.rig_columns(str(rig), mj, log=logs.append)
    assert dict(zip(pids, col["bands"]))["ParamEyeLOpen"] == "autonomic"
    assert col["rebanded"]["ParamEyeLOpen"] == ("discrete", "autonomic")
    assert "ParamEyeLOpen" in col["continuous"]
    assert any("ParamEyeLOpen discrete->autonomic" in s for s in logs)
    # The runtime's blink targets (main, stream.EventRig) follow the band.
    eye = [p for p, b in zip(pids, col["bands"])
           if b == "autonomic" and "open" in p.lower()]
    assert eye == ["ParamEyeLOpen"]


# --------------------------------------------------------------- body

class SwapCore(FakeCore):
    """test_body's rig plus ParamSwap, which hides the raised hand by
    opacity and moves nothing."""

    def __init__(self):
        self._swap = 0.0
        super().__init__()
        self.param_ids = self.param_ids + ["ParamSwap"]
        self.param_min = np.append(self.param_min, 0.0)
        self.param_max = np.append(self.param_max, 1.0)
        self.param_default = np.append(self.param_default, 0.0)
        self.set()

    def set(self, values=None, parts=None, reset=True):
        values = dict(values or {})
        self._swap = float(values.pop("ParamSwap", 0.0))
        return super().set(values, parts, reset)

    def opacities(self):
        o = super().opacities()
        if self._swap > 0.5:
            o[7] = 0.0
        return o


def test_body_discrete_deformer_is_solvable():
    """A parameter banded discrete that the Core shows is a deformer
    (ParamHandL here) is solved and says why; a discrete opacity swap stays
    a switch; nothing else gains the keys."""
    info = _info()
    info["band_of"] = lambda pid: ("discrete" if pid in
                                   ("ParamHandL", "ParamSwap")
                                   else "expressive")
    blob = body.measure(SwapCore(), info, log=lambda *a: None)
    P = blob["params"]
    assert P["ParamHandL"]["role"] == "solvable"
    assert P["ParamHandL"]["continuous"] is True
    assert "no opacity change" in P["ParamHandL"]["continuous_why"]
    assert P["ParamSwap"]["role"] == "switch"
    assert "continuous" not in P["ParamSwap"]
    assert not any("continuous" in e for p, e in P.items()
                   if p != "ParamHandL")
    bm = body.BodyModel(blob)
    assert "ParamHandL" in bm.solvable(0)
    assert "continuous deformers in the Core" in body.describe(bm)


def test_body_carries_runtime_evidence():
    """A parameter rig_columns re-banded (discrete -> autonomic) is solvable
    through its new band and carries the runtime's evidence."""
    info = _info()
    info["band_of"] = lambda pid: ("autonomic" if pid == "ParamHandL"
                                   else "expressive")
    info["continuous"] = {"ParamHandL": "lid moves", "ParamArmLA": "x"}
    info["rebanded"] = {"ParamHandL": ("discrete", "autonomic")}
    P = body.measure(FakeCore(), info, log=lambda *a: None)["params"]
    assert P["ParamHandL"]["role"] == "solvable"
    assert P["ParamHandL"]["continuous_why"] == "lid moves"
    assert "continuous" not in P["ParamArmLA"]
