"""
flow.gltf, flow.vrmbody, flow.vrm, flow.vrmrender: a VRM avatar as a body
(docs/embodiment.md E3). No network.

The body file must put a VRM in flow.body's frame (origin chin, unit head
height, +x the character's own left, VRM 0 turned round to face +Z like
1.0), the IK solver must reach what is reachable inside the joint limits
and say what it misses, the idle roles must land on the canonical
variables in the calm profile's units, the stream must emit body states on
the frame clock, and the server must serve the VRM page and its states.
With Chromium (Playwright), Python's forward kinematics must match
three-vrm's to a millimetre on the shipped VRM 1.0 and VRM 0 samples, and a
short clip must render.
"""

from __future__ import annotations

import json
import math
import os
import struct

import numpy as np
import pytest

from flow import body as fb
from flow import gltf as gl
from flow import pose as pz
from flow import stream as fs
from flow import vrm as fv
from flow import vrmbody as vb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = os.path.join(ROOT, "assets", "vrm", "VRM1_Constraint_Twist_Sample.vrm")
GIRL0 = os.path.join(ROOT, "assets", "vrm", "three-vrm-girl.vrm")
FPS = 30.0

# A tiny humanoid (metres, VRM 1.0 frame: faces +Z, left = +X): bone ->
# (parent, world position).
SKELETON = {
    "hips": (None, (0, 0.90, 0)), "spine": ("hips", (0, 1.00, 0)),
    "chest": ("spine", (0, 1.10, 0)), "upperChest": ("chest", (0, 1.20, 0)),
    "neck": ("upperChest", (0, 1.33, 0)), "head": ("neck", (0, 1.40, 0)),
    "leftShoulder": ("upperChest", (0.02, 1.30, 0)),
    "leftUpperArm": ("leftShoulder", (0.10, 1.30, 0)),
    "leftLowerArm": ("leftUpperArm", (0.32, 1.30, 0)),
    "leftHand": ("leftLowerArm", (0.53, 1.30, 0)),
    "rightShoulder": ("upperChest", (-0.02, 1.30, 0)),
    "rightUpperArm": ("rightShoulder", (-0.10, 1.30, 0)),
    "rightLowerArm": ("rightUpperArm", (-0.32, 1.30, 0)),
    "rightHand": ("rightLowerArm", (-0.53, 1.30, 0)),
}
CHIN_Y, TOP_Y = 1.36, 1.60          # the head box below: hu = 0.24


def make_vrm(version="1"):
    """A minimal .vrm (GLB bytes): the skeleton, one skinned mesh (a head
    box on the head bone, a torso box on the chest) and a face mesh whose
    morph names are VRoid's, with blink / viseme / emotion expressions.
    Version "0" is written as Unity exports VRM 0: facing -Z."""
    flip = np.array([-1.0, 1.0, -1.0]) if version == "0" else np.ones(3)
    names = list(SKELETON)
    idx = {n: i for i, n in enumerate(names)}
    world = {n: np.array(p, float) * flip for n, (_par, p) in SKELETON.items()}
    nodes = []
    for n in names:
        par = SKELETON[n][0]
        t = world[n] - (world[par] if par else 0)
        nodes.append({"name": n, "translation": [float(x) for x in t],
                      "children": [idx[c] for c in names
                                   if SKELETON[c][0] == n]})
    blob = bytearray()
    accessors, views = [], []

    def add(arr, ctype, typ):
        a = np.ascontiguousarray(arr)
        off = len(blob)
        blob.extend(a.tobytes())
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": off,
                      "byteLength": a.nbytes})
        acc = {"bufferView": len(views) - 1, "componentType": ctype,
               "count": int(a.shape[0]), "type": typ}
        if typ == "VEC3" and ctype == 5126:
            acc["min"] = a.min(0).tolist()
            acc["max"] = a.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    def box(x0, x1, y0, y1, z0, z1):
        return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1)
                         for z in (z0, z1)], np.float32)
    head = box(-0.09, 0.09, CHIN_Y, TOP_Y, -0.1, 0.09)
    # The chin: the lowest point in front of the head bone.
    head = np.vstack([head, [[0.0, CHIN_Y - 0.0, 0.08]]]).astype(np.float32)
    torso = box(-0.14, 0.14, 0.95, 1.30, -0.1, 0.12)
    pos = np.vstack([head, torso]) * flip.astype(np.float32)
    joints = np.zeros((len(pos), 4), np.uint16)
    joints[:len(head), 0] = names.index("head")
    joints[len(head):, 0] = names.index("chest")
    weights = np.zeros((len(pos), 4), np.float32)
    weights[:, 0] = 1
    ibm = np.stack([np.eye(4, dtype=np.float32) for _ in names])
    for i, n in enumerate(names):
        ibm[i, :3, 3] = -world[n]
    ibm_col = np.ascontiguousarray(ibm.transpose(0, 2, 1)).reshape(-1, 16)
    a_pos = add(pos, 5126, "VEC3")
    a_j = add(joints, 5123, "VEC4")
    a_w = add(weights, 5126, "VEC4")
    a_ibm = add(ibm_col.astype(np.float32), 5126, "MAT4")
    face_pos = add(np.array([[0, 1.45, 0.08], [0.01, 1.45, 0.08],
                             [0, 1.46, 0.08]], np.float32) * flip.astype(
                                 np.float32), 5126, "VEC3")
    morphs = ["Face.Fcl_EYE_Close", "Face.Fcl_EYE_Close_L",
              "Face.Fcl_EYE_Close_R", "Face.Fcl_MTH_A", "Face.Fcl_MTH_I",
              "Face.Fcl_MTH_U", "Face.Fcl_MTH_E", "Face.Fcl_MTH_O",
              "Face.Fcl_ALL_Joy", "Face.Fcl_BRW_Surprised",
              "Face.Fcl_BRW_Angry", "Face.Fcl_BRW_Fun", "Face.Fcl_EYE_Joy_L",
              "Face.Fcl_EYE_Joy_R", "Face.Fcl_MTH_Fun", "Face.Fcl_MTH_Sorrow"]
    meshes = [{"name": "Body", "primitives": [{"attributes": {
        "POSITION": a_pos, "JOINTS_0": a_j, "WEIGHTS_0": a_w}}]},
        {"name": "Face", "primitives": [{"attributes": {"POSITION": face_pos}}],
         "extras": {"targetNames": morphs}}]
    nodes.append({"name": "Body", "mesh": 0, "skin": 0})
    nodes.append({"name": "Face", "mesh": 1})
    body_node, face_node = len(nodes) - 2, len(nodes) - 1
    js = {"asset": {"version": "2.0"}, "scene": 0,
          "scenes": [{"nodes": [idx["hips"], body_node, face_node]}],
          "nodes": nodes, "meshes": meshes,
          "skins": [{"joints": list(range(len(names))),
                     "inverseBindMatrices": a_ibm}],
          "accessors": accessors, "bufferViews": views,
          "buffers": [{"byteLength": len(blob)}]}
    presets = {"blink": 0, "blinkLeft": 1, "blinkRight": 2, "aa": 3,
               "ih": 4, "ou": 5, "ee": 6, "oh": 7, "happy": 8}
    if version == "1":
        js["extensionsUsed"] = ["VRMC_vrm"]
        js["extensions"] = {"VRMC_vrm": {
            "specVersion": "1.0",
            "meta": {"name": "test", "authors": ["tests"],
                     "licenseUrl": "https://vrm.dev/licenses/1.0/",
                     "allowRedistribution": True,
                     "modification": "allowModificationRedistribution",
                     "creditNotation": "unnecessary"},
            "humanoid": {"humanBones": {n: {"node": idx[n]} for n in names}},
            "expressions": {"preset": {
                k: {"morphTargetBinds": [{"node": face_node, "index": i,
                                          "weight": 1.0}]}
                for k, i in presets.items()}},
            "lookAt": {"type": "bone", "offsetFromHeadBone": [0, 0.06, 0],
                       **{k: {"inputMaxValue": 90, "outputScale": 10}
                          for k in ("rangeMapHorizontalInner",
                                    "rangeMapHorizontalOuter",
                                    "rangeMapVerticalDown",
                                    "rangeMapVerticalUp")}}}}
    else:
        v0 = {"blink": "blink", "blinkLeft": "blink_l",
              "blinkRight": "blink_r", "aa": "a", "ih": "i", "ou": "u",
              "ee": "e", "oh": "o", "happy": "joy"}
        js["extensionsUsed"] = ["VRM"]
        js["extensions"] = {"VRM": {
            "specVersion": "0.0",
            "meta": {"title": "test", "author": "tests",
                     "licenseName": "CC0"},
            "humanoid": {"humanBones": [{"bone": n, "node": idx[n]}
                                        for n in names]},
            "blendShapeMaster": {"blendShapeGroups": [
                {"name": k, "presetName": v0[k],
                 "binds": [{"mesh": 1, "index": i, "weight": 100}]}
                for k, i in presets.items()] + [
                {"name": "Surprised", "presetName": "unknown",
                 "binds": [{"mesh": 1, "index": 9, "weight": 100}]}]},
            "firstPerson": {"lookAtTypeName": "Bone",
                            "lookAtHorizontalOuter": {"xRange": 90},
                            "lookAtVerticalUp": {"xRange": 90},
                            "lookAtVerticalDown": {"xRange": 90}}}}
    jb = json.dumps(js).encode()
    jb += b" " * (-len(jb) % 4)
    out = struct.pack("<III", gl.GLB_MAGIC, 2, 12 + 8 + len(jb) + 8
                      + len(blob))
    out += struct.pack("<II", len(jb), gl.CHUNK_JSON) + jb
    out += struct.pack("<II", len(blob), gl.CHUNK_BIN) + bytes(blob)
    return out


@pytest.fixture(params=["1", "0"])
def tiny(request, tmp_path):
    p = tmp_path / f"tiny{request.param}.vrm"
    p.write_bytes(make_vrm(request.param))
    return str(p), request.param


@pytest.fixture(scope="module")
def seed_body():
    return vb.VRMBody(vb.build(SEED, log=lambda s: None))


# ------------------------------------------------------------- parsing

def test_vrm_0_and_1_read_into_one_vocabulary(tiny):
    path, ver = tiny
    info = gl.vrm_info(gl.GLTF.load(path))
    assert info["version"] == ver
    assert info["faces"] == ("+z" if ver == "1" else "-z")
    assert set(info["humanoid"]) == set(SKELETON)
    # VRM 0 presets under their 1.0 names (a -> aa, joy -> happy, ...).
    for name in ("blink", "blinkLeft", "aa", "oh", "happy"):
        assert info["expressions"][name]["binds"], name
    if ver == "0":
        assert "Surprised" in info["expressions"]          # a custom one
        assert info["expressions"]["aa"]["binds"][0][2] == 1.0  # 100 -> 1
    assert info["license"]["redistribution"]


def test_skinned_rest_vertices_come_back_in_place(tiny):
    path, ver = tiny
    g = gl.GLTF.load(path)
    (name, X, dom, _), *_ = g.skinned_vertices()
    flip = np.array([-1, 1, -1]) if ver == "0" else np.ones(3)
    assert name == "Body"
    assert X[:, 1].max() == pytest.approx(TOP_Y, abs=1e-6)
    hb = gl.vrm_info(g)["humanoid"]
    assert (dom[:9] == hb["head"]).all() and (dom[9:] == hb["chest"]).all()
    assert np.allclose(g.node_position(hb["leftHand"]),
                       np.array(SKELETON["leftHand"][1]) * flip)


def test_shipped_samples_say_they_may_be_redistributed():
    for p, ver in ((SEED, "1"), (GIRL0, "0")):
        info = gl.vrm_info(gl.GLTF.load(p))
        assert info["version"] == ver
        assert info["license"]["redistribution"] is True
        assert info["license"]["modification"] in (
            "allowModificationRedistribution", "allow")


# ------------------------------------------------------------ body file

def test_body_frame_and_sides_on_both_versions(tiny):
    path, ver = tiny
    blob = vb.build(path, log=lambda s: None)
    hu = blob["frame"]["unit"]
    assert hu == pytest.approx(TOP_Y - CHIN_Y, abs=1e-4)
    assert blob["frame"]["origin"][1] == pytest.approx(CHIN_Y, abs=1e-6)
    L = blob["landmarks"]
    # The character's own left is +x, VRM 0 (facing -Z) turned round.
    assert L["shoulder.L"][0] == pytest.approx(0.10 / hu, abs=1e-3)
    assert L["shoulder.R"][0] == pytest.approx(-0.10 / hu, abs=1e-3)
    assert L["shoulder.y"] == pytest.approx((1.30 - CHIN_Y) / hu, abs=1e-3)
    assert L["hand.L"][0] > 0 > L["hand.R"][0]
    assert L["hand.L"][1] < L["chest"][1] < L["shoulder.y"] < 0
    assert L["head"][1] == pytest.approx(0.5, abs=0.01)
    assert blob["vrm"]["bones"]["leftHand"]["rest"][0] > 0
    assert blob["depth"]["chest"] > 0          # the chest is in front
    # The file is what flow.pose reads: it loads as a body and prompts.
    p = os.path.join(os.path.dirname(path), "b.body.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(blob, fh)
    bm = fb.load(p)
    assert isinstance(bm, vb.VRMBody)
    text = pz.system_prompt(bm)
    assert "You are a 3D (VRM) character" in text
    assert "left hand:" in text and "right hand:" in text
    assert "cheek.blush" in text            # listed as not on this body


def test_reach_is_two_bone_ik_over_the_limits(seed_body):
    bm = seed_body
    sh = bm.landmarks["shoulder.L"]
    up = bm.limits("hand.L.y")[1]
    # Straight up from the shoulder: upper arm + forearm + half a hand.
    arm = (np.linalg.norm(np.array(bm.vrm["bones"]["leftLowerArm"]["rest"])
                          - bm.vrm["bones"]["leftUpperArm"]["rest"])
           + np.linalg.norm(np.array(bm.vrm["palm"]["L"])
                            - bm.vrm["bones"]["leftLowerArm"]["rest"])) / bm.hu
    assert up == pytest.approx(sh[1] + arm, abs=0.08)
    assert up > bm.landmarks["head"][1] * 2              # above the head
    assert bm.limits("hand.L.x")[1] == pytest.approx(sh[0] + arm, abs=0.05)
    # Symmetric body, symmetric reach.
    lx, rx = bm.limits("hand.L.x"), bm.limits("hand.R.x")
    assert lx[0] == pytest.approx(-rx[1], abs=1e-4)   # rest rounded to um


def test_shipped_body_file_is_what_build_writes():
    with open(os.path.join(ROOT, "rigs", "seed-san.body.json"),
              encoding="utf-8") as fh:
        shipped = json.load(fh)
    fresh = json.loads(json.dumps(vb.build(SEED, log=lambda s: None)))
    assert fresh == shipped


# ---------------------------------------------------------------- solver

def test_arm_angles_invert_arm_rotation():
    rng = np.random.default_rng(0)
    for s in "LR":
        for _ in range(50):
            e, p, w = rng.uniform(5, 165), rng.uniform(-90, 130), \
                rng.uniform(-100, 90)
            got = vb.arm_angles(s, vb.arm_rotation(s, e, p, w))
            assert got == pytest.approx((e, p, w), abs=1e-6)
    # The meaning of the angles: elevation 90 in plane 0 is straight out
    # to the character's own side, plane 90 straight forward.
    assert vb.arm_rotation("L", 90, 0, 0) @ [1, 0, 0] == pytest.approx(
        [1, 0, 0])
    assert vb.arm_rotation("R", 90, 90, 0) @ [-1, 0, 0] == pytest.approx(
        [0, 0, 1])
    assert vb.arm_rotation("L", 0, 37, 0) @ [1, 0, 0] == pytest.approx(
        [0, -1, 0])


def test_ik_round_trips_reachable_hands_within_the_limits(seed_body):
    bm = seed_body
    rng = np.random.default_rng(1)
    rest = bm.default_values()
    n = tried = 0
    while tried < 80:
        s = "LR"[n % 2]
        n += 1
        v = dict(rest)
        for k in vb.ARM[s]:
            lo, hi = bm.params[k]["min"], bm.params[k]["max"]
            v[k] = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
        # Reachable: within the shoulder's coupled limits and out of the
        # body (a hand placed inside the chest is moved out, not reached).
        if bm.arm_viol(s, *(v[k] for k in vb.ARM[s][:3])) > 0 or \
                max(bm.penetration(v).values()) > vb.PEN_TOL:
            continue
        tried += 1
        target = bm.palms(v)[s]
        res = bm.solve({f"hand.{s}.{q}": float(target[i])
                        for i, q in enumerate("xyz")})
        assert all(e["ok"] for e in res["report"].values()), res["report"]
        for k, x in res["params"].items():
            assert bm.params[k]["min"] - 1e-9 <= x <= bm.params[k]["max"] \
                + 1e-9
        # perceive (forward kinematics) reads back what the report says.
        got = bm.perceive({**rest, **res["params"]})
        for var, e in res["report"].items():
            assert got[var] == pytest.approx(e["got"], abs=1e-9)


def test_hand_targets_in_x_and_y_get_a_depth_in_front_of_the_body(seed_body):
    bm = seed_body
    for t in ({"hand.R.x": "@chest.x", "hand.R.y": "@chest.y"},
              {"hand.L.x": 0.0, "hand.L.y": "@head.y"}):
        res = bm.solve(t)
        assert all(e["ok"] for e in res["report"].values()), res["report"]
        s = next(iter(t))[5]
        z = bm.palms({**bm.default_values(), **res["params"]})[s][2]
        front = bm.depth["chest"] if "chest" in str(t) else bm.depth["face"]
        assert z >= front + vb.HAND_CLEAR - 1e-3       # depth rounded


def test_unreachable_targets_stretch_toward_them_and_say_so(seed_body):
    bm = seed_body
    res = bm.solve({"hand.R.x": -4.0, "hand.R.y": 0.0})
    rep = res["report"]
    assert not rep["hand.R.x"]["ok"]
    assert res["params"]["forearm.R.bend"] == pytest.approx(0.0, abs=1e-6)
    assert rep["hand.R.x"]["got"] == pytest.approx(bm.limits("hand.R.x")[0],
                                                   abs=0.05)
    # Joint targets are clamped to the limits and reported.
    res = bm.solve({"head.yaw": 80})
    assert res["params"]["head.yaw"] == 50 and not res["report"][
        "head.yaw"]["ok"]
    hints = pz.hints_text(bm, res, res["report"])
    assert "alone it reaches -50.00 to 50.00" in hints["head.yaw"]


def test_pose_markers_solve_through_flow_pose(seed_body):
    bm = seed_body
    acts = [{"t": 0.5, "tag": "pose", "arg": {"targets": {
        "hand.R.y": "@head.y", "hand.R.x": -0.9}}, "at": 0},
        {"t": 2.0, "tag": "pose", "arg": {"targets": {
            "head.pitch": -12, "brow.L.slide": 0.5}, "dur": 0.25,
            "hold": 0}, "at": 3}]
    poses = pz.solve_poses(bm, acts, log=lambda s: None)
    assert [p["ok"] for p in poses] == [True, True]
    assert set(poses[0]["params"]) <= set(vb.ARM["R"])
    assert poses[1]["params"] == {"head.pitch": -12.0}
    assert pz.arm_params(bm) >= set(vb.ARM["L"])
    toks = pz.tokens(poses, fv.VRMRig(bm, log=lambda s: None).roles,
                     pz.arm_params(bm))
    assert toks[0]["groups"] == ["arms"] and toks[1]["groups"] == ["head"]


# ------------------------------------------------------------ role map

def test_idle_roles_land_on_canonical_variables_in_standard_units(
        seed_body):
    rig = fv.VRMRig(seed_body, log=lambda s: None)
    assert rig.roles["head_x"][0] == "head.yaw"
    assert rig.roles["head_z"][:2] == ("head.tilt", -1.0)
    assert "brow_lx" not in rig.roles and "brow_la" not in rig.roles
    # A role at normalised 1 is the Cubism standard parameter's end:
    # ParamAngleX 30 = head.yaw 30 deg, ParamBodyAngleX 10 = torso 10 deg.
    from flow import runtime as rt
    for role, want in (("head_x", 30.0), ("body_x", 10.0), ("gaze_x", 1.0),
                       ("head_z", -30.0)):
        pid, sg, _ = rig.roles[role]
        c = rig.col_of[pid]
        i = rig.expressive[c]
        v = rt.denorm(np.array([sg * 1.0]), rig.lo[[i]], rig.hi[[i]])[0]
        assert v == pytest.approx(want)
    assert "eye.L.open" not in rig.base_params      # the blink writes it
    assert rig.base_params["upper_arm.L.elev"] == vb.LIMITS[9][4]


def test_stream_runs_the_idle_poses_blinks_and_mouth_on_vrm(seed_body):
    bm = seed_body
    rig = fv.VRMRig(bm, fps=FPS, log=lambda s: None)
    st = fs.MotionStream(rig, seed=3, log=lambda s: None)
    poses = pz.solve_poses(bm, [{"t": 1.0, "tag": "pose", "arg": {
        "targets": {"hand.R.y": "@head.y"}, "hold": 1.0}, "at": 0}],
        log=lambda s: None)
    st.direct(pz.tokens(poses, rig.roles, pz.arm_params(bm)))
    n = 150
    mouth = {"fps": FPS, "open": [0.0] * 60 + [0.8] * 30 + [0.0] * 10,
             "vowels": {"a": [0.0] * 60 + [0.6] * 30, "o": [0.0] * 60
                        + [0.2] * 30}}
    st.add_mouth(0, mouth, 100)
    frames = [st.frame()[0] for _ in range(n)]
    for o in frames:
        assert set(o) <= set(bm.params)
    eye = [o["eye.L.open"] for o in frames]
    assert min(eye) < 0.5 and max(eye) == 1.0 or min(eye) == 1.0
    assert frames[70]["viseme.a"] == pytest.approx(0.6)
    assert frames[70]["viseme.o"] == pytest.approx(0.2)
    assert frames[95]["viseme.a"] == 0.0
    head = [o["head.yaw"] for o in frames]
    assert np.ptp(head) > 0                  # the idle moves the head
    # The pose holds the hand at head height after its move.
    k = poses[0]
    f_hold = int((k["start"] + k["dur"] + 0.3) * FPS)
    y = bm.palms(frames[f_hold])["R"][1]
    assert y == pytest.approx(bm.landmarks["head"][1], abs=0.08)
    assert 0 < frames[40]["breath"] or 0 < frames[20]["breath"]


# ----------------------------------------------------- state and stream

def test_state_is_unit_quaternions_look_and_expressions(seed_body):
    bm = seed_body
    st = bm.state({"head.yaw": 20, "gaze.x": 0.5, "gaze.y": 0.5,
                   "eye.L.open": 0.0, "mouth.form": 1.0, "viseme.a": 0.7,
                   "brow.L.lift": 1.0, "brow.R.lift": 1.0})
    for b, q in st["bones"].items():
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-4)
    # Head yaw 20 = neck 8 + head 12 about +Y.
    assert st["bones"]["head"][1] == pytest.approx(
        math.sin(math.radians(6)), abs=1e-4)
    la = bm.vrm["look_at"]
    assert st["look"][0] == pytest.approx(0.5 * la["h"])
    assert st["look"][1] < 0                # three-vrm: + pitch is down
    assert st["expr"]["blinkLeft"] == 1.0 and "blinkRight" not in st["expr"]
    assert st["expr"]["aa"] == 0.7
    morph = {k.rsplit(".", 1)[-1]: v for k, v in st["morph"].items()}
    assert morph["Fcl_MTH_Fun"] == 1.0 and morph["Fcl_BRW_Surprised"] == 1.0
    assert json.loads(json.dumps(st)) == st


def test_state_loop_stamps_frames_on_the_clock(seed_body, tmp_path):
    from flow import live as lv
    bm = seed_body
    rig = fv.VRMRig(bm, fps=FPS, log=lambda s: None)
    clock = lv.Clock()
    hub = lv.Hub(clock)
    q = hub.subscribe("states")
    rec = fv.StateLog(str(tmp_path), FPS, None, SEED)
    loop = fv.StateLoop(fs.MotionStream(rig, seed=0, log=lambda s: None),
                        bm, clock, hub, FPS, record=rec, report=0)
    for _ in range(5):
        loop.step()
    got = [q.get_nowait() for _ in range(5)]
    assert [m["f"] for m in got] == list(range(5))
    assert [m["t"] for m in got] == pytest.approx([f / FPS for f in range(5)],
                                               abs=1e-4)
    path = rec.close()
    with open(path, encoding="utf-8") as fh:
        saved = json.load(fh)
    assert len(saved["states"]) == 5 and saved["fps"] == FPS
    assert saved["states"][0]["bones"] == {k: got[0][k] for k in ("bones",)
                                           }["bones"]


def test_server_serves_the_vrm_page_assets_and_states(seed_body, tmp_path):
    import threading
    import urllib.request
    from flow import live as lv
    from flow import vrmrender as vr
    bm = seed_body
    rig = fv.VRMRig(bm, fps=FPS, log=lambda s: None)
    clock = lv.Clock()
    hub = lv.Hub(clock)
    loop = fv.StateLoop(fs.MotionStream(rig, seed=0, log=lambda s: None),
                        bm, clock, hub, FPS, log=lambda s: None, report=0)
    app = lv.App(clock, hub, FPS, None, loop)
    app.vrm = {"path": SEED, "config": dict(vr.page_config(bm),
                                            avatar="vrm", fps=FPS)}
    srv = lv.Server(app, "127.0.0.1", 0)
    stop = threading.Event()
    th = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(path):
        return opener.open(f"http://127.0.0.1:{srv.server_address[1]}"
                           + path, timeout=5)
    try:
        srv.start()
        th.start()
        page = get("/").read().decode("utf-8")
        assert "three-vrm" in page and "/states" in page
        cfg = json.loads(get("/avatar.json").read())
        assert cfg["camera"] == bm.vrm["camera"] and cfg["avatar"] == "vrm"
        assert get("/avatar.vrm").read()[:4] == b"glTF"
        js = get("/web/vrm.js").read().decode("utf-8")
        assert "export function applyState" in js
        assert get("/web/vendor/three-vrm/three-vrm.module.min.js").read(
            200).startswith(b"/*! (c) 2019-2026 pixiv")
        with pytest.raises(urllib.error.HTTPError):
            get("/web/../live.py")
        s = get("/states")
        msg = None
        for line in s:
            line = line.decode("utf-8").strip()
            if line.startswith("data:"):
                msg = json.loads(line[5:])
                break
        s.close()
        assert msg["t"] == pytest.approx(msg["f"] / FPS, abs=1e-4)
        assert "bones" in msg and "look" in msg and "expr" in msg
        st = json.loads(get("/state").read())
        assert st["frame"] > 0
    finally:
        stop.set()
        srv.stop()
        th.join(timeout=5)


def test_director_plans_a_vrm_turn_into_states(tmp_path):
    from flow import director as dr
    reply = tmp_path / "reply.txt"
    reply.write_text('Hi! [pose {"hand.R.y": "@head.y", "hand.R.x": -0.9}] '
                     'Over there, [pose {"hand.L.x": 2.0, "hand.L.y": '
                     '"@shoulder.y", "head.yaw": 25, "gaze.x": 0.7}] see? '
                     '[pose {"head.pitch": -12, "dur": 0.25, "hold": 0}]'
                     '[pose {"head.pitch": 0, "dur": 0.3, "hold": 0}] Yes.',
                     encoding="utf-8")
    out = tmp_path / "run"
    rc = dr.main(["--avatar", "vrm", "--vrm-body",
                  os.path.join(ROOT, "rigs", "seed-san.body.json"),
                  "--mode", "body", "--backend", "file", "--reply-file",
                  str(reply), "--out", str(out)])
    assert rc == 0
    rec = json.loads((out / "states.json").read_text(encoding="utf-8"))
    acts = json.loads((out / "actions.json").read_text(encoding="utf-8"))
    assert acts["reach"]["ok"] == acts["reach"]["targets"] == 8
    assert len(rec["states"]) == int(round(acts["seconds"], 2) * FPS) + 1
    assert (out / "subs.srt").exists() and (out / "tokens.json").exists()


def test_live2d_arguments_still_required_for_a_live2d_avatar(tmp_path):
    from flow import director as dr
    with pytest.raises(SystemExit):
        dr.main(["--backend", "file", "--out", str(tmp_path)])


# ------------------------------------------------------------ Chromium

def _chromium():
    from flow import vrmrender as vr
    why = vr.have_chromium()
    if why:
        pytest.skip(why)
    # The browsers may be anywhere (~/.cache/ms-playwright after
    # `playwright install chromium`, or PLAYWRIGHT_BROWSERS_PATH): ask.
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            exe = p.chromium.executable_path
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Playwright cannot start: {exc}")
    if not exe or not os.path.exists(exe):
        pytest.skip("no Playwright Chromium (playwright install chromium)")
    return vr


@pytest.mark.parametrize("avatar", [SEED, GIRL0])
def test_three_vrm_draws_what_python_computes(avatar):
    """The browser's normalized bones end where Python's forward
    kinematics puts them (to 0.01 mm on both samples when measured), the
    character's left hand at +x, and gaze.x + turns the eyes to +x."""
    vr = _chromium()
    bm = vb.VRMBody(vb.build(avatar, log=lambda s: None))
    names = ["head", "leftUpperArm", "leftLowerArm", "leftHand",
             "rightLowerArm", "rightHand"]
    poses = [{}, bm.solve({"hand.R.y": 0.6, "hand.R.x": -0.9})["params"],
             {"head.yaw": 20, "torso.yaw": 15, "torso.tilt": 8,
              "upper_arm.L.elev": 90, "upper_arm.L.plane": 90,
              "forearm.L.bend": 30, "upper_arm.L.twist": 40},
             bm.solve({"hands": "@behind_back"})["params"]]
    with vr.serve(avatar, vr.page_config(bm)) as base:
        pg = vr.Page(base, 128)
        try:
            pg.eval("() => TSK.eyeDir()")
            knuckles = ["leftMiddleProximal", "rightMiddleProximal"]
            for v in poses:
                pg.frame(bm.state(v), 0.0, 0)
                got = pg.eval("(n) => TSK.positions(n)", names + knuckles)
                W, P = bm.fk(v)
                for n in names:
                    assert np.array(got[n]) == pytest.approx(P[n], abs=1e-4)
                # The hand's own rotation (the wrist: VRM 0's x / z
                # negation included) puts the knuckles where Python does
                # (the palm is halfway from the wrist to them).
                for s, n in zip("LR", knuckles):
                    h = f"{vb.SIDE[s]}Hand"
                    want = P[h] + W[h] @ (2 * (bm.palm_rest[s] - bm.rest[h]))
                    assert np.array(got[n]) == pytest.approx(want, abs=1e-4)
            assert got["leftHand"][0] > bm.origin[0]
            pg.frame(bm.state({"gaze.x": 1.0}), 0.0, 0)
            assert pg.eval("() => TSK.eyeDir()")[0] > 0.05
            pg.frame(bm.state({"gaze.y": 1.0}), 0.0, 0)
            assert pg.eval("() => TSK.eyeDir()")[1] > 0.05
            pg.frame(bm.state({"gaze.y": -1.0}), 0.0, 0)
            assert pg.eval("() => TSK.eyeDir()")[1] < -0.05
            # At rest the eyes (three-vrm's own eye bones, between them)
            # look straight into the page's camera.
            pg.frame(bm.state({}), 0.0, 0)
            d = np.array(pg.eval("() => TSK.eyeDir()"))
            eyes = pg.eval("(n) => TSK.positions(n)", ["leftEye", "rightEye"])
            e = 0.5 * (np.array(eyes["raw:leftEye"])
                       + np.array(eyes["raw:rightEye"]))
            tc = np.array(bm.vrm["camera"]["position"]) - e
            off = math.degrees(math.acos(d @ tc / np.linalg.norm(tc)))
            assert off < 1.5
            assert not pg.errors
        finally:
            pg.close()


def test_offline_render_writes_a_clip(seed_body, tmp_path):
    vr = _chromium()
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    bm = seed_body
    states = [bm.state({"head.yaw": 10.0 * i}) for i in range(3)]
    out = str(tmp_path / "clip.mp4")
    n = vr.render(states, bm, out, fps=FPS, size=128, vrm=SEED,
                  log=lambda s: None, warmup=0.1)
    assert n == 3 and os.path.getsize(out) > 1000


# ---------------------------------------------- gaze, camera and the idle

@pytest.fixture(scope="module")
def girl_body():
    return vb.VRMBody(vb.build(GIRL0, log=lambda s: None))


@pytest.mark.parametrize("which", ["seed", "girl"])
def test_rest_gaze_meets_the_camera(which, seed_body, girl_body):
    """The camera is at the eyes' height, so the eyes at rest (straight
    ahead, head at rest) look into it -- 3.7 / 3.2 deg over it before."""
    bm = seed_body if which == "seed" else girl_body
    cam = bm.vrm["camera"]
    eye = bm.to_world(bm.vrm["eyes"])
    to_cam = np.array(cam["position"]) - eye
    off = math.degrees(math.acos(to_cam[2] / np.linalg.norm(to_cam)))
    assert off < 1.5
    assert cam["position"][1] > cam["target"][1]      # looking down a bit


def test_gaze_signs_and_eye_range(seed_body):
    bm = seed_body
    assert bm.state({})["look"] == [0.0, 0.0]
    assert bm.state({"gaze.x": 1.0})["look"][0] > 0     # toward +x
    assert bm.state({"gaze.y": 1.0})["look"][1] < 0     # up = - pitch
    assert bm.state({"gaze.y": -1.0})["look"][1] > 0
    # Seed-san's own range: 90 deg of input = 10 deg of eye at gaze 1.
    assert bm.eye_degrees() == pytest.approx((10.0, 10.0, 10.0))
    # An export whose range map turns the eyes 90 deg at full input is
    # capped at EYE_MAX_DEG.
    with open(os.path.join(ROOT, "rigs", "seed-san.body.json"),
              encoding="utf-8") as fh:
        blob = json.load(fh)
    blob["vrm"]["look_at"].update(h_out=90.0, h_in=90.0, v_up_out=60.0,
                                  v_down_out=90.0)
    wild = vb.VRMBody(blob)
    assert max(wild.eye_degrees()) == pytest.approx(vb.EYE_MAX_DEG)
    assert abs(wild.state({"gaze.x": 1.0})["look"][0]) == pytest.approx(
        90.0 * vb.EYE_MAX_DEG / 90.0)


def test_idle_gaze_and_head_stay_near_the_viewer(seed_body):
    """The idle on the VRM: head pitch 15 deg per head_y unit (Live2D
    draws its pitch ~0.48 of its yaw), eyes within a few degrees."""
    bm = seed_body
    assert fv.ROLE_MAP["head_y"][2] == 15.0
    rig = fv.VRMRig(bm, fps=FPS, log=lambda s: None)
    pitch, eye_y, eye_x = [], [], []
    deg = bm.eye_degrees()
    for seed in range(3):
        st = fs.MotionStream(rig, seed=seed, log=lambda s: None)
        for _ in range(int(20 * FPS)):
            o = st.frame()[0]
            pitch.append(o["head.pitch"])
            eye_y.append(o["gaze.y"] * deg[1])
            eye_x.append(o["gaze.x"] * deg[0])
    pitch, eye_y, eye_x = map(np.array, (pitch, eye_y, eye_x))
    assert np.percentile(np.abs(pitch), 98) < 8.0
    assert abs(pitch.mean()) < 2.0
    assert np.percentile(np.abs(eye_y), 98) < 8.0
    assert np.percentile(np.abs(eye_x), 98) < 10.0
    assert np.std(eye_x) > 0.5                  # it does look about


def test_follower_converts_live2d_angles():
    acts = [{"t": 0.0, "tag": "pose", "arg": {"targets": {
        "head.pitch": -20, "head.yaw": 12, "hand.R.y": "@head.y"}}},
        {"t": 0.5, "tag": "blink"}]
    got = fv.from_live2d(acts)
    assert got[0]["arg"]["targets"] == {"head.pitch": -10.0, "head.yaw": 12,
                                        "hand.R.y": "@head.y"}
    assert acts[0]["arg"]["targets"]["head.pitch"] == -20     # not mutated
    assert got[1] is acts[1]


# ------------------------------------------------------------ collisions

POSES = {
    "together": {"hand.L.x": 0.0, "hand.L.y": "@chest.y",
                 "hand.R.x": 0.0, "hand.R.y": "@chest.y"},
    "behind (z)": {"hand.L.x": 0.15, "hand.L.y": "@hips.y", "hand.L.z": -0.6,
                   "hand.R.x": -0.15, "hand.R.y": "@hips.y",
                   "hand.R.z": -0.6},
    "crossed": {"hand.L.x": -0.4, "hand.L.y": "@chest.y",
                "hand.R.x": 0.4, "hand.R.y": "@chest.y"},
    "R to L shoulder": {"hand.R.x": "@shoulder.L.x",
                        "hand.R.y": "@shoulder.y"},
    "@together": {"hands": "@together"},
    "@clasped": {"hands": "@clasped"},
    "@behind_back": {"hands": "@behind_back"},
    "@chest": {"hand.R": "@chest"},
    "@hip": {"hand.L": "@hip"},
    "@back": {"hand.R": "@back"},
}


@pytest.mark.parametrize("which", ["seed", "girl"])
def test_solved_poses_do_not_penetrate(which, seed_body, girl_body):
    """No arm in the trunk or head and no arm through the other, in the
    rest pose, in every solved pose and on the way there (the path the
    pose layer plays, through the solver's detour when it gives one). The
    old solver: together 0.08 / 0.09 hu (body / hands), behind the back
    0.32 into the hips, crossed forearms 0.20 through each other."""
    bm = seed_body if which == "seed" else girl_body
    rest = bm.default_values()
    assert max(bm.penetration(rest).values()) == 0.0
    tol = vb.PEN_TOL + 1e-6
    for name, t in POSES.items():
        res = bm.solve(t)
        v = {**rest, **res["params"]}
        pen = bm.penetration(v)
        assert max(pen.values()) <= tol, (name, pen)
        a = {k: rest[k] for k in v}
        if res.get("via"):
            V = {**v, **res["via"]}
            path = max(bm.path_pen(a, V), bm.path_pen(V, v))
        else:
            path = bm.path_pen(a, v)
        assert path <= 2 * tol, (name, path)
    # The composites reach what they name.
    res = bm.solve({"hands": "@together"})
    assert all(e["ok"] for e in res["report"].values()), res["report"]
    pal = bm.palms({**rest, **res["params"]})
    assert np.linalg.norm(pal["L"] - pal["R"]) < 0.3        # palm to palm
    assert pal["L"][2] > bm.depth["chest"]                   # in front
    res = bm.solve({"hands": "@behind_back"})
    assert all(e["ok"] for e in res["report"].values()), res["report"]
    pal = bm.palms({**rest, **res["params"]})
    for s in "LR":
        assert pal[s][2] < -0.5                 # behind the back
        assert abs(pal[s][1] - bm.landmarks["hips"][1]) < 0.5
    res = bm.solve({"hand.R": "@chest"})
    assert all(e["ok"] for e in res["report"].values()), res["report"]


def test_two_hands_at_one_place_meet_and_say_so(seed_body):
    """What --avatar both sends for "hold your hands together" (the
    Live2D prompt's plain x, y): the hands meet side by side, reported."""
    bm = seed_body
    res = bm.solve(POSES["together"])
    pal = bm.palms({**bm.default_values(), **res["params"]})
    assert pal["L"][0] > pal["R"][0]
    assert not res["report"]["hand.L.x"]["ok"]
    assert "meet" in res["why"]["hand.L.x"]
    hints = pz.hints_text(bm, res, res["report"])
    assert "meet" in hints["hand.L.x"]


def test_targets_inside_the_body_miss_with_a_reason(seed_body):
    bm = seed_body
    chest = bm.landmarks["chest"]
    res = bm.solve({"hand.L.x": 0.1, "hand.L.y": chest[1], "hand.L.z": 0.0})
    v = {**bm.default_values(), **res["params"]}
    assert max(bm.penetration(v).values()) <= vb.PEN_TOL + 1e-6
    assert not res["report"]["hand.L.z"]["ok"]
    assert res["why"]["hand.L.z"]
    # The palm ends on the chest's front, not inside it.
    assert bm.palms(v)["L"][2] > bm.depth["chest"] - 0.1
    # flow.pose reports it as a miss with the reason.
    poses = pz.solve_poses(bm, [{"t": 0.0, "tag": "pose", "arg": {
        "targets": {"hand.L.x": 0.1, "hand.L.y": "@chest.y",
                    "hand.L.z": 0.0}}, "at": 0}], log=lambda s: None)
    assert not poses[0]["ok"] and "body" in " ".join(poses[0]["hints"].values())


def test_composites_parse_and_play_with_a_detour(seed_body):
    bm = seed_body
    p = pz.parser_for(bm)
    tags = p.feed('Here [pose {"hands": "@behind_back"}] and '
                  '[pose {"hand.R": "@chest.front"}] [pose {"hands": '
                  '"@nowhere"}] ok.')
    p.close()
    assert [t["arg"]["targets"] for t in tags] == [
        {"hands": "@behind_back"}, {"hand.R": "@chest.front"}]
    assert any("@nowhere" in w for w in p.warnings)
    acts = [{"t": 0.0, **tags[0]}]
    poses = pz.solve_poses(bm, acts, log=lambda s: None)
    # In by a detour, held, out by it again: three tokens, one report.
    assert [q.get("via") for q in poses] == ["in", None, "out"]
    assert pz.reach_summary(poses)["markers"] == 1
    assert poses[1]["start"] == pytest.approx(poses[0]["start"]
                                              + poses[0]["dur"])
    assert len(pz.report_lines(poses)) == 1
    text = pz.system_prompt(bm)
    assert '"hands": "@together"' in text and '"@behind_back"' in text


def test_live2d_bodies_take_no_composites_and_are_unchanged():
    """The Live2D side of --avatar both: its prompt and parser are the
    bytes they were before the VRM composites (sha1 of the prompt and the
    parse of a reply with composites, recorded before the change)."""
    import hashlib
    bm = fb.load(os.path.join(ROOT, "rigs", "Hiyori.body.json"))
    text = pz.system_prompt(bm)
    assert hashlib.sha1(text.encode()).hexdigest() == \
        "8a81f66495839c7db92eb5de1a88401bd4586d0c"
    assert "@together" not in text
    p = pz.parser_for(bm)
    tags = p.feed('Yes [pose {"hands": "@together", "head.tilt": 8}] ok.')
    p.close()
    assert tags[0]["arg"]["targets"] == {"head.tilt": 8.0}
    assert any("'hands'" in w for w in p.warnings)


def test_a_stale_body_file_is_rebuilt(tmp_path):
    """A body file an older flow.vrmbody wrote (revision 1: no collision
    model) next to its .vrm is built again when the avatar is loaded."""
    import shutil
    vrm = tmp_path / "zz_stale_sample.vrm"
    shutil.copy(SEED, vrm)
    out = os.path.join(ROOT, "rigs", "zz_stale_sample.body.json")
    try:
        blob = vb.build(str(vrm), log=lambda s: None)
        del blob["vrm"]["collision"], blob["vrm"]["rev"]
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        said = []
        bm, path = fv.body_for(str(vrm), None, log=said.append)
        assert path == out and bm.coll and bm.vrm["rev"] == vb.BODY_REV
        assert any("rebuilding" in s for s in said)
        with open(out, encoding="utf-8") as fh:
            assert json.load(fh)["vrm"]["rev"] == vb.BODY_REV
    finally:
        if os.path.exists(out):
            os.remove(out)


@pytest.mark.parametrize("which", ["seed", "girl"])
def test_played_poses_stay_out_of_the_body(which, seed_body, girl_body):
    """Every frame the stream plays -- move, hold, release -- not only the
    solved pose: the hold's settle (drifting each angle toward the rest)
    put the hands 0.32-0.43 hu into the hips by the end of a hold behind
    the back until a pose with a detour holds still."""
    bm = seed_body if which == "seed" else girl_body
    rig = fv.VRMRig(bm, fps=FPS, log=lambda s: None)
    for tg in ({"hands": "@behind_back"}, {"hand.R": "@back"},
               {"hands": "@together"}):
        st = fs.MotionStream(rig, seed=1, log=lambda s: None)
        poses = pz.solve_poses(bm, [{"t": 0.5, "tag": "pose", "arg": {
            "targets": tg, "hold": 2.0}, "at": 0}], log=lambda s: None)
        st.direct(pz.tokens(poses, rig.roles, pz.arm_params(bm)))
        worst = max(max(bm.penetration(st.frame()[0]).values())
                    for _ in range(int(7 * FPS)))
        assert worst <= 2 * vb.PEN_TOL, (tg, worst)


def test_follower_plays_live2d_markers_in_vrm_units(seed_body):
    """--avatar both end to end: Hiyori's prompt's nod (head.pitch -12,
    ParamAngleY units) is a 6 deg nod on the VRM, and a Live2D gaze marker
    turns the VRM's eyes the same way (look-at pitch - up, + down)."""
    import argparse
    bm = seed_body
    rig = fv.VRMRig(bm, fps=FPS, log=lambda s: None)
    st = fs.MotionStream(rig, seed=1, idle=False, log=lambda s: None)
    a = argparse.Namespace(seed=1, lipsync="off", mouth_gain=1.0)
    fol = fv.Follower(a, bm, rig, st, log=lambda s: None)
    acts = [{"t": 0.2, "tag": "pose", "arg": {"targets": {
        "head.pitch": -12, "gaze.y": 0.5}, "dur": 0.25, "hold": 1.0},
        "at": 0},
        {"t": 2.5, "tag": "pose", "arg": {"targets": {"gaze.y": -0.5},
                                          "dur": 0.25, "hold": 1.0},
         "at": 5}]
    fol.sentence(acts, 0.0, 0, "Hmm, yes.", None, 3.0)
    frames = [st.frame()[0] for _ in range(int(4 * FPS))]
    held = frames[int(0.8 * FPS)]
    assert held["head.pitch"] == pytest.approx(-6.0, abs=0.6)
    assert bm.state(held)["look"][1] < 0          # gaze up: - pitch
    later = frames[int(3.1 * FPS)]
    assert later["gaze.y"] < -0.3
    assert bm.state(later)["look"][1] > 0          # gaze down: + pitch


def test_voice_config_found_beside_a_vrm_body():
    """--avatar vrm has no --rig: the voice config is rigs/<name>.voice.json
    (the server's oguri_cap/toukai_teio configs were never read)."""
    import types
    from flow import director as dr
    ns = types.SimpleNamespace
    assert dr.voice_config_path(ns(rig=None, vrm="vrm/oguri_cap.vrm")) \
        == os.path.join("rigs", "oguri_cap.voice.json")
    assert dr.voice_config_path(ns(rig=None, vrm=None,
                                   vrm_body="x/teio.body.json")) \
        == os.path.join("x", "teio.voice.json")
    assert dr.voice_config_path(ns(rig="rigs/Hiyori.npz", vrm="a.vrm")) \
        == "rigs/Hiyori.voice.json"
    assert dr.voice_config_path(ns(rig=None, voice_config="v.json")) == "v.json"
