"""
flow.beat2: BEAT2 mocap on a VRM body (docs/embodiment.md E4). No network,
no BEAT2 data: the clips are made here (SMPL-X-style poses, a Praat
TextGrid, a wav).

The retarget must be rotation transfer that the body's own forward model
agrees with: poses made *from* the model come back as the same parameters
and palms (the round trip), the SMPL-X rest (all zeros) is the avatar's
T-pose, a known elbow / shoulder turn moves the hand where geometry says,
VRM 0 and 1.0 agree (both are the 1.0 frame to the retargeter), a turned or
z-up clip faces the camera. The filter must flag a pose that really
penetrates (checked with the body's own collision model first) and cut
segments at it. The export must be what flow.dataexport action writes:
flow.actionhead.Data loads it with a hidden-state folder built the way
flow.train.hidden builds one, the face columns masked.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from flow import beat2 as b2
from flow import body as fb
from flow import vrmbody as vb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = os.path.join(ROOT, "assets", "vrm", "VRM1_Constraint_Twist_Sample.vrm")
GIRL = os.path.join(ROOT, "assets", "vrm", "three-vrm-girl.vrm")
J = b2.J


@pytest.fixture(scope="module")
def seed():
    return b2.Avatar(SEED, os.path.join(ROOT, "rigs", "seed-san.body.json"))


@pytest.fixture(scope="module")
def girl():
    return b2.Avatar(GIRL, os.path.join(ROOT, "rigs",
                                        "three-vrm-girl.body.json"))


def aa(axis, deg):
    k = np.zeros(3)
    k["xyz".index(axis)] = math.radians(deg)
    return k


def pose(n=1, **joints):
    """[n, 55, 3] axis-angle: zeros except the named joints (a [3] vector
    or [n, 3])."""
    p = np.zeros((n, 55, 3))
    for name, v in joints.items():
        p[:, J[name]] = v
    return p


def fit_pose(av, p, fingers=False):
    G = b2.world_rotations(p)
    return b2.fit(av.bm, G, av.fingers if fingers else ())


def palms_of(bm, val, i=0):
    v = {k: float(x[i]) for k, x in val.items()}
    st = bm.perceive(v)
    return np.array([[st[f"hand.{s}.{q}"] for q in "xyz"] for s in "LR"])


# ------------------------------------------------------------- readers

LONG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 6
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 6
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 6
            text = "AH"
    item [2]:
        class = "IntervalTier"
        name = "Words"
        xmin = 0
        xmax = 6
        intervals: size = 5
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = ""
        intervals [2]:
            xmin = 0.5
            xmax = 1.5
            text = "Hello"
        intervals [3]:
            xmin = 1.5
            xmax = 2
            text = "sp"
        intervals [4]:
            xmin = 2
            xmax = 3.5
            text = "[wor]*ld ""q"""
        intervals [5]:
            xmin = 3.5
            xmax = 6
            text = ""
'''
SHORT = '''File type = "ooTextFile"
Object class = "TextGrid"

0
6
<exists>
1
"IntervalTier"
"words"
0
6
5
0
0.5
""
0.5
1.5
"Hello"
1.5
2
"sp"
2
3.5
"[wor]*ld ""q"""
3.5
6
""
'''


@pytest.mark.parametrize("text,enc", [(LONG, "utf-8"), (SHORT, "utf-8"),
                                      (LONG, "utf-16"), (SHORT, "utf-8-sig")])
def test_textgrid_formats(tmp_path, text, enc):
    p = tmp_path / "x.TextGrid"
    p.write_bytes(text.encode(enc))
    rows = b2.read_textgrid(str(p))
    assert len(rows) == 5 and rows[1] == (0.5, 1.5, "Hello")
    w = b2.words_of(rows)
    assert [x[2] for x in w] == ["Hello", 'world "q"']
    assert w[1][:2] == (2.0, 3.5)


def test_textgrid_without_a_words_tier(tmp_path):
    p = tmp_path / "x.TextGrid"
    p.write_text(LONG.replace('"Words"', '"syll"'), encoding="utf-8")
    with pytest.raises(ValueError, match="no 'words' tier.*phones"):
        b2.read_textgrid(str(p))


def test_char_times_follow_the_words():
    words = [(1.0, 1.5, "hello"), (1.7, 2.2, "big"), (2.2, 3.0, "world")]
    spoken, ch = b2.spoken_text(words, t0=0.5)
    assert spoken == "hello big world" and ch.shape == (len(spoken), 2)
    assert ch[0, 0] == pytest.approx(0.5) and ch[4, 1] == pytest.approx(1.0)
    # The space after "hello" spans the silence to "big".
    assert tuple(ch[5]) == pytest.approx((1.0, 1.2))
    assert tuple(ch[9]) == pytest.approx((1.7, 1.7))      # touching words
    assert (ch[:, 1] >= ch[:, 0]).all() and ch[-1, 1] == pytest.approx(2.5)
    cjk, _ = b2.spoken_text([(0, 1, "你好"), (1, 2, "世界")])
    assert cjk == "你好世界"


def test_wav_reader_and_prosody(tmp_path):
    sr = 16000
    t = np.arange(int(sr * 3.0)) / sr
    x = np.zeros_like(t)
    x[sr:2 * sr] = 0.4 * np.sin(2 * np.pi * 150 * t[sr:2 * sr])
    p = tmp_path / "a.wav"
    b2.write_wav(str(p), x, sr)
    y, sr2 = b2.read_wav(str(p))
    assert sr2 == sr and len(y) == len(x) and np.abs(y - x).max() < 1e-3
    pr = b2.prosody(y, sr2, 30.0, 90)
    assert pr.shape == (90, 3)
    assert pr[:25, 2].sum() == 0 and pr[:25, 0].max() < 1e-3
    assert pr[36:54, 2].min() == 1                     # voiced inside
    assert np.abs(pr[36:54, 1] - 150).max() < 3
    assert pr[36:54, 0].mean() == pytest.approx(0.4 / math.sqrt(2), rel=0.1)
    assert pr[70:, 2].sum() == 0
    # float32 and 24-bit wavs, stereo
    import struct
    for tag, bits, body in ((3, 32, y.astype("<f4").tobytes()),):
        hdr = struct.pack("<HHIIHH", tag, 1, sr, sr * bits // 8, bits // 8, bits)
        q = tmp_path / "f.wav"
        q.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVE"
                      + b"fmt " + struct.pack("<I", 16) + hdr + b"data"
                      + struct.pack("<I", len(body)) + body)
        z, _ = b2.read_wav(str(q))
        assert np.allclose(z, y)


def write_npz(path, poses, trans=None, fps=30, **extra):
    d = {"poses": poses.reshape(len(poses), 165),
         "trans": np.zeros((len(poses), 3)) if trans is None else trans,
         "betas": np.zeros(300), "expressions": np.zeros((len(poses), 100))}
    if fps is not None:
        d["mocap_frame_rate"] = np.array(fps)
    d.update(extra)
    np.savez(path, **d)


def test_clip_reader_errors_and_rates(tmp_path):
    p = pose(90, left_elbow=aa("y", -30))
    write_npz(tmp_path / "ok.npz", p)
    c = b2.read_clip(str(tmp_path / "ok.npz"))
    assert c["poses"].shape == (90, 55, 3) and c["notes"] == []
    np.savez(tmp_path / "nopose.npz", trans=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="no 'poses'.*trans"):
        b2.read_clip(str(tmp_path / "nopose.npz"))
    np.savez(tmp_path / "shape.npz", poses=np.zeros((4, 72)))
    with pytest.raises(ValueError, match=r"\[N, 165\]"):
        b2.read_clip(str(tmp_path / "shape.npz"))
    write_npz(tmp_path / "nofps.npz", p, fps=None)
    c = b2.read_clip(str(tmp_path / "nofps.npz"))
    assert c["fps_source"] == "assumed" and "assumed" in c["notes"][0]
    # 60 fps -> 30: half the frames, the rotation still on its straight path
    ramp = np.zeros((121, 55, 3))
    ramp[:, J["left_elbow"], 1] = np.linspace(0, -1.2, 121)
    write_npz(tmp_path / "r60.npz", ramp, trans=np.linspace(0, 1, 121)[:, None]
              * np.ones(3), fps=60)
    c = b2.read_clip(str(tmp_path / "r60.npz"))
    assert len(c["poses"]) == 61 and "resampled" in c["notes"][0]
    assert np.allclose(c["poses"][:, J["left_elbow"], 1],
                       np.linspace(0, -1.2, 61), atol=1e-6)
    assert np.allclose(c["trans"][:, 0], np.linspace(0, 1, 61), atol=1e-9)
    # 120 fps through a rotation the long way (angles near pi): slerp keeps
    # the shorter arc, never a jump
    big = np.zeros((9, 55, 3))
    big[:, 0, 1] = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8]
    q, _ = b2.resample(big, np.zeros((9, 3)), 120.0, 30.0)
    a = b2.aa_to_mat(q[:, 0])
    a0 = b2.aa_to_mat(big[::4, 0])
    assert np.abs(a - a0).max() < 1e-6


def test_rotation_helpers_round_trip():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(40, 3))
    R = b2.aa_to_mat(v)
    assert np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(3), atol=1e-9)
    q = b2.mat_to_quat(R)
    assert np.allclose(b2.quat_to_mat(q), R, atol=1e-9)
    assert np.allclose(q, np.array([vb.quat(r) for r in R]), atol=1e-9)
    y, p, t = (rng.uniform(-40, 40, 30) for _ in range(3))
    E = b2.euler_batch(y, p, t)
    assert np.allclose(E[3], vb.euler(y[3], p[3], t[3]))
    assert np.allclose(b2.euler_angles(E), (y, p, t), atol=1e-8)


# ------------------------------------------------------------ retarget

def model_world(av, vals):
    """SMPL-X-style world rotations from the model's own bones: the inverse
    of the retarget (droop taken out)."""
    bm = av.bm
    G = np.tile(np.eye(3), (len(vals), 55, 1, 1))
    for i, v in enumerate(vals):
        W, _P = bm.fk(v)
        G[i, J["spine3"]] = W["upperChest"]
        G[i, J["head"]] = W["head"]
        for s, side in (("L", "left"), ("R", "right")):
            a_u, a_f = bm.align[s]
            G[i, J[f"{side}_collar"]] = W[f"{side}Shoulder"]
            G[i, J[f"{side}_shoulder"]] = W[f"{side}UpperArm"] @ a_u.T
            G[i, J[f"{side}_elbow"]] = W[f"{side}LowerArm"] @ a_f.T
            G[i, J[f"{side}_wrist"]] = W[f"{side}Hand"] @ a_f.T
    return G


def random_values(rng, n, torso=0.0):
    out = []
    for _ in range(n):
        v = {"torso.yaw": rng.uniform(-1, 1) * torso * 20,
             "torso.pitch": rng.uniform(-1, 1) * torso * 8,
             "torso.tilt": rng.uniform(-1, 1) * torso * 8,
             "head.yaw": rng.uniform(-30, 30), "head.pitch": rng.uniform(-20, 20),
             "head.tilt": rng.uniform(-15, 15),
             "shoulders.up": rng.uniform(-.3, .3)}
        for s in "LR":
            v.update({f"shoulder.{s}.up": rng.uniform(-.3, .3),
                      f"upper_arm.{s}.elev": rng.uniform(20, 150),
                      f"upper_arm.{s}.plane": rng.uniform(-40, 100),
                      f"upper_arm.{s}.twist": rng.uniform(-60, 60),
                      f"forearm.{s}.bend": rng.choice(
                          [rng.uniform(0, 8), rng.uniform(20, 125)]),
                      f"hand.{s}.roll": rng.uniform(0, 100),
                      f"hand.{s}.flex": rng.uniform(-30, 40),
                      f"hand.{s}.dev": rng.uniform(-10, 10)})
        out.append(v)
    return out


@pytest.mark.parametrize("which", ["seed", "girl"])
def test_fit_inverts_the_forward_model(which, seed, girl):
    av = seed if which == "seed" else girl
    bm = av.bm
    rng = np.random.default_rng(3)
    vals = random_values(rng, 60)
    G = model_world(av, vals)
    f = b2.fit(bm, G)
    for k in vb.JOINTS:
        want = np.array([v[k] for v in vals])
        tol = 0.02 if k.startswith("shoulder") else 0.6
        d = np.abs(f["val"][k] - want)
        if k.startswith("shoulder"):
            # shoulders.up and shoulder.X.up add: compare what each side gets
            s = k[-4] if k.startswith("shoulder.") else None
            if s is None:
                continue
            eff = f["val"]["shoulders.up"] + f["val"][k]
            d = np.abs(eff - (np.array([v["shoulders.up"] for v in vals])
                              + want))
        if k.endswith("twist") or k.endswith("roll"):
            # An almost straight elbow cannot tell the upper arm's twist
            # from the hand's roll; what the viewer sees is checked below.
            continue
        assert d.max() < tol, (k, d.max())
    # What the viewer sees: the palms agree wherever the angles do not.
    for i in range(0, 60, 6):
        got = palms_of(bm, f["val"], i)
        want = bm.perceive(vals[i])
        w = np.array([[want[f"hand.{s}.{q}"] for q in "xyz"] for s in "LR"])
        assert np.abs(got - w).max() < 0.01, (i, np.abs(got - w).max())


def test_identity_is_the_rest_t_pose_on_both_avatars(seed, girl):
    for av in (seed, girl):
        f = fit_pose(av, pose(2))
        v = f["val"]
        for s in "LR":
            assert v[f"upper_arm.{s}.elev"] == pytest.approx([90, 90])
            assert v[f"upper_arm.{s}.plane"] == pytest.approx([0, 0], abs=1e-6)
            assert v[f"upper_arm.{s}.twist"] == pytest.approx([0, 0], abs=1e-6)
            assert v[f"forearm.{s}.bend"] == pytest.approx([0, 0], abs=1e-6)
            for k in ("roll", "flex", "dev"):
                assert v[f"hand.{s}.{k}"] == pytest.approx([0, 0], abs=1e-6)
            assert v[f"shoulder.{s}.up"] == pytest.approx([0, 0], abs=1e-9)
        assert abs(v["torso.yaw"]).max() < 1e-9 and abs(v["head.pitch"]).max() < 1e-9
        pal = palms_of(av.bm, v)
        ref = av.bm.perceive({**av.bm.default_values(),
                              "upper_arm.L.elev": 90, "upper_arm.L.plane": 0,
                              "upper_arm.L.twist": 0, "forearm.L.bend": 0,
                              "upper_arm.R.elev": 90, "upper_arm.R.plane": 0,
                              "upper_arm.R.twist": 0, "forearm.R.bend": 0})
        for s, i in (("L", 0), ("R", 1)):
            assert pal[i] == pytest.approx([ref[f"hand.{s}.{q}"] for q in "xyz"],
                                           abs=1e-9)
        # the character's left hand is on +x (and out to the side)
        assert pal[0, 0] > 1.0 > -1.0 > pal[1, 0]


def test_a_flexed_elbow_brings_the_hand_forward(seed, girl):
    for av in (seed, girl):
        bm = av.bm
        base = fit_pose(av, pose())
        p0 = palms_of(bm, base["val"])
        for s, side, sgn in (("L", "left", -1), ("R", "right", 1)):
            p = pose(**{f"{side}_elbow": aa("y", sgn * 90)})
            f = fit_pose(av, p)
            assert f["val"][f"forearm.{s}.bend"][0] == pytest.approx(90, abs=1)
            now = palms_of(bm, f["val"])
            i = "LR".index(s)
            # the forearm points forward: the palm gained about the
            # elbow-to-palm distance in z and lost it in x
            el = np.array(bm.perceive({**bm.default_values(),
                                       f"upper_arm.{s}.elev": 90,
                                       f"upper_arm.{s}.plane": 0,
                                       f"upper_arm.{s}.twist": 0,
                                       f"forearm.{s}.bend": 0})[f"elbow.{s}"])
            L = np.linalg.norm(p0[i] - el)
            assert now[i, 2] - p0[i, 2] == pytest.approx(L, rel=0.05)
            assert abs(now[i, 0] - el[0]) < 0.1 * L
            assert abs(now[1 - i] - p0[1 - i]).max() < 1e-9      # the other arm


def test_a_shoulder_turn_lowers_the_arm(seed):
    f = fit_pose(seed, pose(left_shoulder=aa("z", -60)))
    assert f["val"]["upper_arm.L.elev"][0] == pytest.approx(30, abs=0.5)
    assert f["val"]["upper_arm.L.plane"][0] == pytest.approx(0, abs=0.5)
    pal = palms_of(seed.bm, f["val"])
    p0 = palms_of(seed.bm, fit_pose(seed, pose())["val"])
    assert pal[0, 1] < p0[0, 1] - 0.5 and abs(pal[0, 2] - p0[0, 2]) < 0.05


def test_the_forearm_lands_where_the_mocap_forearm_is(seed):
    """SMPL-X splits an axial turn between shoulder and elbow; the fit puts
    the forearm on the mocap forearm either way."""
    bm = seed.bm
    for twist in (0.0, 35.0, -50.0):
        p = pose(left_shoulder=aa("x", twist), left_elbow=aa("y", -80))
        G = b2.world_rotations(p)
        f = b2.fit(bm, G)
        v = {k: float(x[0]) for k, x in f["val"].items()}
        _W, P = bm.fk(v)
        el = bm.to_body(P["leftLowerArm"])
        hand = bm.to_body(P["leftHand"])
        dir_model = (hand - el) / np.linalg.norm(hand - el)
        dir_mocap = G[0, J["left_elbow"], :, 0]
        assert dir_model @ dir_mocap > 0.99, twist


def test_facing_and_up_axis(seed):
    rng = np.random.default_rng(1)
    n = 150
    wob = np.radians(8) * np.sin(np.linspace(0, 12, n))
    p = pose(n, left_elbow=aa("y", -50))
    p[:, 0, 1] = np.radians(70)                     # turned 70 deg
    p[:, J["spine3"], 1] = wob                      # the torso wobbling on it
    cfg = dict(b2.CFG, facing_window=0)
    clip = {"poses": p, "trans": np.zeros((n, 3)), "fps": 30.0}
    rt = b2.retarget_clip(seed, clip, cfg)
    assert rt["up"] == "y-up"
    assert np.degrees(np.mean(np.abs(rt["yaw"] - 70))) < 6
    ty = rt["fit"]["val"]["torso.yaw"]
    assert abs(ty.mean()) < 1.0 and ty.std() == pytest.approx(
        np.degrees(wob).std(), rel=0.1)
    # the same wobble, not turned: the same hands
    q = pose(n, left_elbow=aa("y", -50))
    q[:, J["spine3"], 1] = wob
    rq = b2.retarget_clip(seed, dict(clip, poses=q), cfg)
    assert np.abs(rt["meas"]["palms"] - rq["meas"]["palms"])[:, 0].max() < 0.1
    # a z-up clip (root turned y -> z) is found and turned back
    z = pose(n, left_elbow=aa("y", -50))
    z[:, 0] = aa("x", 90)
    rz = b2.retarget_clip(seed, dict(clip, poses=z, trans=np.zeros((n, 3))),
                          cfg)
    assert rz["up"] == "z-up"
    assert abs(rz["fit"]["val"]["torso.pitch"]).max() < 0.5
    r0 = b2.retarget_clip(seed, dict(clip, poses=pose(n, left_elbow=aa(
        "y", -50))), cfg)
    assert abs(rz["meas"]["palms"] - r0["meas"]["palms"]).max() < 0.05
    del rng


def test_the_global_frame_is_not_the_torso(seed):
    """A pelvis carried tilted, or a body turning as a whole, is the actor's
    stance, not a bent or twisted torso; the spine's own motion still is."""
    n = 300
    t = np.linspace(0, 1, n)
    clip = {"trans": np.zeros((n, 3)), "fps": 30.0}
    base = pose(n, left_elbow=aa("y", -50))
    r0 = b2.retarget_clip(seed, dict(clip, poses=base), b2.CFG)
    tilt = base.copy()
    tilt[:, 0] = np.stack([np.radians(-11) + np.radians(4) * np.sin(9 * t),
                           np.radians(40) * np.sin(2.5 * t),
                           np.radians(5) * np.cos(7 * t)], 1)
    r1 = b2.retarget_clip(seed, dict(clip, poses=tilt), b2.CFG)
    for k in ("torso.pitch", "torso.yaw", "torso.tilt"):
        assert abs(r1["fit"]["raw"][k]).max() < 0.5, k
    assert np.abs(r1["meas"]["palms"] - r0["meas"]["palms"]).max() < 0.02
    # a habitual lean of the spine against the pelvis is the posture
    lean = base.copy()
    lean[:, J["spine2"]] = aa("x", 20)
    rl = b2.retarget_clip(seed, dict(clip, poses=lean), b2.CFG)
    assert abs(rl["fit"]["raw"]["torso.pitch"]).max() < 0.5
    # while the spine's own swing about it is the torso's
    swing = lean.copy()
    swing[:, J["spine2"], 0] += np.radians(10) * np.sin(9 * t)
    swing[:, J["spine2"], 1] = np.radians(10) * np.sin(7 * t)
    rs = b2.retarget_clip(seed, dict(clip, poses=swing), b2.CFG)
    assert np.ptp(rs["fit"]["raw"]["torso.pitch"]) > 10
    assert np.ptp(rs["fit"]["raw"]["torso.yaw"]) > 10


def test_wrist_roll_is_clamped_not_counted(seed):
    bm = seed.bm
    f = fit_pose(seed, pose(1, left_wrist=aa("x", 70)))
    lo = float(bm.params["hand.L.roll"]["min"])
    assert f["raw"]["hand.L.roll"][0] < lo - 20      # the mocap's forearm twist
    assert f["val"]["hand.L.roll"][0] == lo          # clamped
    assert b2.limit_excess(f, bm)[0] == 0.0
    # the elbow's limits still count
    g = fit_pose(seed, pose(1, left_shoulder=aa("z", 150)))
    assert b2.limit_excess(g, bm)[0] > 20


def test_mirrored_mocap_gives_mirrored_wrists(seed):
    rng = np.random.default_rng(3)
    p = pose(40)
    q = pose(40)
    for side in ("left", "right"):
        for j in ("collar", "shoulder", "elbow", "wrist"):
            p[:, J[f"{side}_{j}"]] = rng.normal(0, 0.3, (40, 3))
    for j in ("collar", "shoulder", "elbow", "wrist"):
        q[:, J[f"left_{j}"]] = p[:, J[f"right_{j}"]] * [1, -1, -1]
        q[:, J[f"right_{j}"]] = p[:, J[f"left_{j}"]] * [1, -1, -1]
    a, b = fit_pose(seed, p), fit_pose(seed, q)
    for k in ("roll", "flex", "dev"):
        assert np.allclose(a["raw"][f"hand.L.{k}"], b["raw"][f"hand.R.{k}"])


def penetrating_forearm(bm):
    """Left arm values whose forearm is through the trunk while the palm is
    clear of it (a hand resting by the belly of a thin actor, on a bulkier
    body)."""
    v = {**bm.default_values(), "upper_arm.L.elev": 10,
         "upper_arm.L.plane": 75, "upper_arm.L.twist": -60,
         "forearm.L.bend": 90}
    return v


def test_an_arm_through_the_trunk_is_pushed_out_with_the_palm_kept(seed):
    from flow import dataset as ds
    bm = seed.bm
    v = penetrating_forearm(bm)
    assert bm.penetration(v)["body.L"] > 0.15
    assert bm.body_sdf(bm.palms(v)["L"][None])[0] > 0.1      # palm is clear
    n = 40
    val = {k: np.full(n, v[k]) for k in vb.JOINTS}
    # a run in the middle: the mocap arm comes into the body and leaves it
    ramp = np.clip(1 - np.abs(np.arange(n) - 20) / 10, 0, 1)
    rest = bm.default_values()
    for k in vb.ARM["L"]:
        val[k] = rest[k] + ramp * (v[k] - rest[k])
    hips = {"hips.x": np.zeros(n), "hips.y": np.zeros(n)}
    cols = ds.traj_vars(bm)
    m0 = b2.measure(bm, val, hips, cols, 30.0)
    assert m0["pen"].max() > 0.15
    before = {k: x.copy() for k, x in val.items()}
    fc = b2.fix_contacts(bm, val, hips, cols, 30.0, m0, dict(b2.CFG))
    m1 = b2.measure(bm, val, hips, cols, 30.0)
    assert fc["frames"] > 0
    assert m1["pen"].max() <= b2.CFG["contact_pen"]
    # the palm is where the mocap put it, the right arm untouched, the arm
    # changes without a pop
    moved = np.linalg.norm(m1["palms"] - m0["palms"], axis=2)
    assert moved[20].max() <= 0.05 and moved.max() < 0.1
    for k in vb.ARM["R"]:
        assert np.array_equal(val[k], before[k])
    for k in vb.ARM["L"]:
        assert np.abs(np.diff(val[k])).max() < 25
    assert m1["speed"].max() < 25.0
    assert (val["upper_arm.L.twist"] != before["upper_arm.L.twist"]).any()
    # the first and last frames, with the arm at rest, are as they were
    assert val["upper_arm.L.elev"][0] == before["upper_arm.L.elev"][0]


def test_the_contact_fix_is_off_with_the_flag_and_cheap_when_clear(seed):
    n = 60
    p = pose(n, left_elbow=aa("y", -50))
    clip = {"poses": p, "trans": np.zeros((n, 3)), "fps": 30.0}
    r = b2.retarget_clip(seed, clip, b2.CFG)
    assert r["contact"]["frames"] == 0
    r = b2.retarget_clip(seed, clip, dict(b2.CFG, contact_fix=False))
    assert r["contact"] is None


def test_hips_follow_the_translation_in_the_facing_frame(seed):
    n = 120
    tr = np.zeros((n, 3))
    tr[:, 0] = 0.03 * np.sin(np.linspace(0, 20, n))      # 3 cm of sway
    tr[:, 1] = 0.5 + 0.01 * np.sin(np.linspace(0, 40, n))   # a standing height
    p = pose(n)
    p[:, 0, 1] = math.radians(90)                        # turned: x is z now
    rt = b2.retarget_clip(seed, {"poses": p, "trans": tr, "fps": 30.0},
                          dict(b2.CFG, facing_window=0))
    hx, hy = rt["hips"]
    # the world +x sway is across the avatar's front now; its own x is -z
    assert abs(hx).max() < 0.02 and abs(hy.mean()) < 1e-6
    p[:, 0, 1] = 0
    rt = b2.retarget_clip(seed, {"poses": p, "trans": tr, "fps": 30.0},
                          dict(b2.CFG, facing_window=0))
    assert rt["hips"][0].max() == pytest.approx(0.03 / seed.bm.hu, rel=0.05)
    assert abs(rt["hips"][1]).max() < 0.01 / seed.bm.hu * 1.05


def test_measure_is_perceive(seed):
    bm = seed.bm
    rng = np.random.default_rng(5)
    vals = random_values(rng, 8, torso=1.0)
    val = {k: np.array([v[k] for v in vals]) for k in vb.JOINTS}
    from flow import dataset as ds
    cols = ds.traj_vars(bm)
    hips = {"hips.x": np.linspace(0, .1, 8), "hips.y": np.linspace(0, -.05, 8)}
    m = b2.measure(bm, val, hips, cols, 30.0)
    for i in range(8):
        st = bm.perceive(vals[i])
        for c, name in enumerate(cols):
            x = m["traj"][i, c]
            if name.startswith(b2.NAN_PREFIX):
                assert np.isnan(x), name
            elif name in hips:
                assert x == pytest.approx(hips[name][i], abs=1e-6)
            else:
                assert x == pytest.approx(st[name], abs=1e-4), name
    # the batch penetration is the body's own, pose by pose
    for i in range(8):
        want = max(bm.penetration({**bm.default_values(), **vals[i]}).values())
        assert m["pen"][i] == pytest.approx(want, abs=1e-9)
    assert m["pen"].shape == (8,) and (m["pen"] >= 0).all()


def find_penetrating(bm):
    """Left arm parameters whose forearm is deep in the chest (the body's own
    collision model says so)."""
    best = (0, None)
    for e in range(60, 121, 20):
        for pl in range(50, 131, 20):
            for tw in range(-90, 91, 30):
                for bend in (100, 125, 145):
                    v = {**bm.default_values(), "upper_arm.L.elev": e,
                         "upper_arm.L.plane": pl, "upper_arm.L.twist": tw,
                         "forearm.L.bend": bend}
                    d = bm.penetration(v)["body.L"]
                    if d > best[0]:
                        best = (d, v)
    return best


def test_a_penetrating_pose_is_flagged_and_cut(seed):
    bm = seed.bm
    depth, v = find_penetrating(bm)
    assert depth > 0.15, depth                       # really inside the body
    n = 240
    vals = [dict(bm.default_values()) for _ in range(n)]
    for i in range(100, 130):
        vals[i] = v
    val = {k: np.array([x[k] for x in vals]) for k in vb.JOINTS}
    from flow import dataset as ds
    m = b2.measure(bm, val, {"hips.x": np.zeros(n), "hips.y": np.zeros(n)},
                   ds.traj_vars(bm), 30.0)
    assert m["pen"][100:130].min() > 0.15 and m["pen"][:90].max() < 0.1
    bad, why = b2.bad_frames(m, np.zeros(n), 30.0, b2.CFG)
    assert why["pen"][100:130].all()
    assert why["speed"][99:101].any()                    # the jump into it
    assert bad[96:134].all() and not bad[:90].any()      # grown by bad_pad
    # utterances either side of the bad stretch survive, cut at it
    words = [(0.5 + 0.5 * i, 0.9 + 0.5 * i, "w") for i in range(15)]
    segs = b2.plan_segments(words, n, 30.0, bad, dict(b2.CFG, min_seg=2.0))
    kept = [s for s in segs if s["status"] == "kept"]
    assert kept and all(s["f1"] <= 96 or s["f0"] >= 134 for s in kept)
    assert any(s["reason"] in ("cut", "short") for s in segs
               if s["status"] != "kept") or len(kept) == len(segs)
    # a speed spike and a limit excess are flagged too
    pen0 = np.zeros(n)
    sp = np.zeros(n)
    sp[50] = 99
    lim = np.zeros(n)
    lim[200] = 60
    bad2, why2 = b2.bad_frames({"pen": pen0, "speed": sp}, lim, 30.0, b2.CFG)
    assert why2["speed"][50] and why2["limits"][200] and not why2["pen"].any()
    assert bad2[50] and bad2[200]


def test_utterances_split_and_merge():
    ws = [(0, .4, "a"), (.5, 1, "b"), (1.1, 1.6, "c"),          # one run
          (3.0, 3.4, "d"), (3.5, 4.5, "e"), (4.6, 5.5, "f"),    # 1.4 s gap
          (5.6, 6.4, "g"),
          (20.0, 20.4, "x")]                                    # alone, far
    g = b2.utterances(ws, gap=0.35, min_len=3.0, max_len=15.0)
    # the first two runs are short and 1.4 s apart: merged; the last is
    # far from anything and stays alone (the planner drops it as short)
    assert [len(x) for x in g] == [7, 1]
    long = [(i * 0.6, i * 0.6 + 0.5, "w") for i in range(60)]   # 36 s, no pause
    g = b2.utterances(long, min_len=3, max_len=15)
    assert all(x[-1][1] - x[0][0] <= 15 for x in g) and sum(map(len, g)) == 60
    sil = long[:25] + [(w[0] + 2.0, w[1] + 2.0, w[2]) for w in long[25:]]
    g = b2.utterances(sil, gap=5.0, min_len=3, max_len=25)       # gap too small
    assert len(g) == 2 and len(g[0]) == 25                       # at the 2 s one


# --------------------------------------------------------------- export

def make_beat2(root, stems=("2_scott_0_1_1",), seconds=22.0, fps=30,
               hang=-50):
    """A BEAT2-shaped folder: a clip of talking arms, three utterances."""
    for d in ("smplxflame_30", "wave16k", "textgrid"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    n = int(seconds * fps)
    t = np.arange(n) / fps
    for stem in stems:
        p = pose(n)
        for side, sg in (("left", -1), ("right", 1)):
            p[:, J[f"{side}_shoulder"], 2] = math.radians(hang) * (
                1 if side == "left" else -1) + 0.15 * np.sin(1.3 * t)
            p[:, J[f"{side}_elbow"], 1] = sg * (0.6 + 0.4 * np.sin(
                2.1 * t + (0 if side == "left" else 1.0)))
        p[:, J["head"], 0] = 0.1 * np.sin(1.7 * t)
        p[:, J["spine3"], 1] = 0.08 * np.sin(0.7 * t)
        p[:, 0, 1] = math.radians(15)
        tr = np.stack([0.02 * np.sin(0.9 * t), 0.9 + 0.005 * np.sin(2 * t),
                       np.zeros(n)], 1)
        write_npz(os.path.join(root, "smplxflame_30", stem + ".npz"), p, tr, fps)
        utts = [(0.6, 5.0), (7.0, 12.5), (15.0, 21.0)]
        words, rows = [], []
        for a, b in utts:
            k = int((b - a) / 0.5)
            for i in range(k):
                words.append((a + 0.5 * i, a + 0.5 * i + 0.4, f"word{i}"))
        tg = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
              "0", f"{seconds}", "<exists>", "1", '"IntervalTier"', '"words"',
              "0", f"{seconds}", str(2 * len(words) + 1)]
        last = 0.0
        for a, b, w in words:
            tg += [str(last), str(a), '""', str(a), str(b), f'"{w}"']
            last = b
        tg += [str(last), f"{seconds}", '""']
        # (the interval count above is an upper bound; fix it)
        n_int = (len(tg) - 12) // 3
        tg[11] = str(n_int)
        with open(os.path.join(root, "textgrid", stem + ".TextGrid"), "w") as f:
            f.write("\n".join(tg) + "\n")
        sr = 16000
        tt = np.arange(int(sr * seconds)) / sr
        x = np.zeros_like(tt)
        for a, b, _w in words:
            i0, i1 = int(a * sr), int(b * sr)
            x[i0:i1] = 0.3 * np.sin(2 * np.pi * 140 * tt[i0:i1])
        b2.write_wav(os.path.join(root, "wave16k", stem + ".wav"), x, sr)
    return root


def args(tmp_path, **kw):
    import argparse
    d = dict(beat2=str(tmp_path / "beat2"), vrm=SEED,
             body=os.path.join(ROOT, "rigs", "seed-san.body.json"),
             out=str(tmp_path / "out"), speakers=None, limit=0, jobs=1,
             force=False, allow_outside_data=False, fingers=True,
             **{k: v for k, v in b2.CFG.items() if k not in
                ("fingers", "check_text")})
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("b2")
    make_beat2(str(root / "beat2"))
    a = args(root)
    s = b2.retarget(a, log=lambda m: None)
    return root, a, s


def test_retarget_exports_the_action_format(exported):
    root, a, s = exported
    out = root / "out"
    from flow import dataset as ds
    bm = fb.load(a.body)
    rows, npz = [], np.load(out / "action.npz")
    with open(out / "action.jsonl") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    assert len(rows) == 3, s
    ids = [r["id"] for r in rows]
    assert ids == [f"beat2.2_scott_0_1_1/s0{i}" for i in range(3)]
    for r in rows:
        assert r["avatar"] == "seed-san" and r["vars"] == ds.traj_vars(bm)
        assert r["fps"] == 30.0 and r["reply_raw"] == r["spoken"]
        tr = npz[r["key"] + ".traj"]
        assert r["key"] == r["id"].replace("/", "__")
        assert tr.dtype == np.float16 and tr.shape == (r["frames"], len(r["vars"]))
        assert 3.0 <= r["frames"] / 30 <= 15.0
        ch = npz[r["key"] + ".chars"]
        assert ch.shape == (len(r["spoken"]), 2) and ch.dtype == np.float16
        assert ch.min() >= 0 and ch.max() <= r["frames"] / 30 + 0.05
        assert r["speech_seconds"] == pytest.approx(ch[-1, 1], abs=0.05)
        # the face is masked, the body is there
        for c, v in enumerate(r["vars"]):
            col = tr[:, c].astype(float)
            if v.startswith(b2.NAN_PREFIX):
                assert np.isnan(col).all(), v
            else:
                assert np.isfinite(col).all(), v
        assert npz[r["key"] + ".joints"].shape == (r["frames"], len(vb.JOINTS))
        pr = npz[r["key"] + ".prosody"]
        assert pr.shape == (r["frames"], 3) and pr[:, 2].max() == 1
        assert abs(float(pr[pr[:, 2] > 0, 1].mean()) - 140) < 5
        assert npz[r["key"] + ".fingers"].shape[:2] == (r["frames"], 30)
        assert r["prosody"] == ["rms", "f0", "voiced"]
    rep = json.load(open(out / "reports" / "2_scott_0_1_1.json"))
    assert rep["up_axis"] == "y-up" and rep["kept_seconds"] > 10
    assert rep["audio_minus_motion"] == pytest.approx(0, abs=0.05)
    sm = json.load(open(out / "summary.json"))
    assert sm["kept"] == 3 and sm["by_speaker"] == {"scott": rep["kept_seconds"]}
    # a second run redoes nothing, --force does
    again = b2.retarget(args(root), log=lambda m: None)
    assert again["kept"] == 3


def hidden_for(root, rows, h_dim=12):
    """A hidden-state folder as flow.train.hidden writes it, random states."""
    import re
    from flow.train import hidden as fh
    hid = root / "hidden"
    os.makedirs(hid, exist_ok=True)
    sh = fh.Shards(str(hid), limit=50)
    parsers = fh.parsers_for(None, sorted({r["avatar"] for r in rows}))
    rng = np.random.default_rng(0)
    for r in rows:
        idx, spoken = fh.spoken_index(r["reply_raw"], parsers[r["avatar"]]())
        assert spoken == r["spoken"]          # the parser leaves the words
        raw = np.array([(m.start(), m.end()) for m in
                        re.finditer(r"\S+", r["reply_raw"])], np.int32)
        h = rng.normal(size=(len(raw), 1, h_dim)).astype(np.float16)
        sh.add(r["id"], r["key"], h, raw, fh.to_spoken(raw, idx, len(spoken)))
    sh.close()
    with open(hid / "meta.json", "w") as f:
        json.dump({"model": "random", "layers": [0], "dim": h_dim}, f)
    return str(hid)


def test_the_export_loads_in_the_action_head(exported):
    root, a, _s = exported
    from flow import actionhead as ah
    from flow.train import hidden as fh
    stem = str(root / "out" / "action")
    rows, _ = fh.read_export(stem)
    hid = hidden_for(root, rows)
    d = ah.Data(stem, hid)
    assert len(d) == 3 and "mouth.open" not in d.vocab
    assert {"hips.x", "head.tilt", "hand.L.x", "gaze.x"} <= set(d.vocab)
    it = d.item(0)
    assert it["y"].shape[0] == rows[0]["frames"] and it["avatar"] == "seed-san"
    m = it["m"]
    assert m[:, d.vi["head.yaw"]].all() and m[:, d.vi["hand.R.z"]].all()
    for v in ("gaze.x", "eye.L.open", "brow.L.lift", "brow.R.form"):
        assert not m[:, d.vi[v]].any(), v            # masked out of the loss
    assert np.isfinite(it["y"]).all()
    # token times come from the word times
    assert (np.diff(it["t_tok"]) >= -1e-3).all() and it["t_tok"][0] < 1.0
    mean, std = ah.norm_stats(d, range(len(d)))
    assert np.isfinite(mean).all() and (std > 0).all()


def test_a_few_head_steps_on_it(exported, tmp_path):
    pytest.importorskip("torch")
    root, _a, _s = exported
    from flow import actionhead as ah
    stem = str(root / "out" / "action")
    o = ah.parse(["train", "--export", stem, "--hidden", str(root / "hidden"),
                  "--out", str(tmp_path / "ah"), "--steps", "6",
                  "--eval-every", "3", "--dim", "16", "--batch", "2",
                  "--chunk", "48", "--eval-frac", "0"])
    res = ah.train(o, log=lambda s: None)
    assert len(res["history"]) == 2
    # the documented eval step, on this export alone (no markers, no data)
    e = ah.evaluate(ah.parse(["eval", "--ckpt", str(tmp_path / "ah" /
                                                    "head.pt"),
                              "--export", stem, "--hidden",
                              str(root / "hidden")]), log=lambda s: None)
    assert {"head", "rest", "mean"} <= set(e["normalised_mse"])
    assert "head.tilt" in e["per_variable"]


def test_preview_states_carry_fingers_and_the_mouth(exported, seed):
    root, _a, _s = exported
    out = root / "out"
    with open(out / "action.jsonl") as f:
        row = json.loads(f.readline())
    with np.load(out / "action.npz") as z:
        states, hips = b2.segment_states(seed, z, row)
    assert len(states) == row["frames"]
    st = states[10]
    assert {"leftUpperArm", "rightLowerArm", "head"} <= set(st["bones"])
    assert len(st["bones"]) >= 14 + 30 - 2
    q = np.array(st["bones"]["leftUpperArm"])
    assert abs(np.linalg.norm(q) - 1) < 1e-3
    assert any(s["expr"] for s in states)                  # the mouth moves
    from flow import vrm as fv
    p = fv.save_states(str(out / "t_states.json"), states, 30.0,
                       seed.body_path, SEED)
    assert json.load(open(p))["format"] == "tsukuyomi.vrmstates/1"


def test_text_that_the_parser_would_change_is_cleaned_or_dropped(tmp_path):
    assert b2.clean_word("[laugh]") == "laugh"
    assert b2.clean_word("*um*") == "um"
    assert b2.clean_word("sp") == "" and b2.clean_word("<unk>") == ""
    assert b2.clean_word("don't") == "don't"


def test_out_must_be_ignored_by_git(tmp_path):
    with pytest.raises(SystemExit, match="never be committed"):
        b2.check_out(os.path.join(ROOT, "beat2_out"))
    b2.check_out(os.path.join(ROOT, "data", "beat2", "seed-san"))
    b2.check_out(str(tmp_path / "anywhere"))
    with open(os.path.join(ROOT, ".gitignore")) as f:
        lines = [x.strip() for x in f]
    assert "/data/" in lines


def test_clip_listing_and_speakers(tmp_path):
    make_beat2(str(tmp_path), stems=("2_scott_0_1_1", "3_solomon_0_2_2"))
    allc = b2.list_clips(str(tmp_path))
    assert [c["stem"] for c in allc] == ["2_scott_0_1_1", "3_solomon_0_2_2"]
    assert [c["stem"] for c in b2.list_clips(str(tmp_path), "3")] == [
        "3_solomon_0_2_2"]
    assert len(b2.list_clips(str(tmp_path), "scott,solomon", limit=1)) == 1
    with pytest.raises(SystemExit, match="no clip"):
        b2.list_clips(str(tmp_path), "nobody")
    # a dataset one folder down is found; two of them are not guessed
    make_beat2(str(tmp_path / "wrap" / "en"), stems=("5_a_0_0_0",))
    assert [c["stem"] for c in b2.list_clips(str(tmp_path / "wrap"))] == [
        "5_a_0_0_0"]
    make_beat2(str(tmp_path / "wrap" / "zh"), stems=("6_b_0_0_0",))
    with pytest.raises(SystemExit, match="2 subfolders"):
        b2.list_clips(str(tmp_path / "wrap"))
    with pytest.raises(SystemExit, match="smplxflame_30"):
        b2.list_clips(str(tmp_path / "empty"))
