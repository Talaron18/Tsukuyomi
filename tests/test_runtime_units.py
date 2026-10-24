"""
Runtime pieces that need neither GL nor torch.

Switches: snapping has hysteresis (a value hovering near the midpoint must not
flicker the limb) and a ramp (a change of state takes `ramp` frames, like an
authored swap, not one).

base_state: the idle frame it holds must not include autonomic or mouth
parameters. It once did, which froze blink and breath at the idle's first
frame and switched both off for every rig with an idle motion.

Breath: its gain is calibrated from the body file (lafei and xuefeng down
to about a tenth, Hiyori unchanged), a flag or the voice config overrides
it, and it scales the breath without touching the blinks.
"""

from __future__ import annotations

import numpy as np
import pytest

from flow.runtime import Switches, base_keep, base_state
from l2dprobe.analyze import group_of
from l2dprobe.motions import band_of


def test_switch_hysteresis_holds_near_midpoint():
    sw = Switches({"p": (0.0, 1.0)}, ramp=4)
    assert sw("p", 0.2) == 0.0              # starts on the nearer side
    # Above the midpoint but short of `enter`: no change of state.
    for v in (0.55, 0.65, 0.69, 0.6):
        assert sw("p", v) == 0.0


def test_switch_ramps_over_ramp_frames():
    sw = Switches({"p": (0.0, 1.0)}, ramp=4)
    sw("p", 0.0)
    out = [sw("p", 0.8) for _ in range(5)]
    assert out == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.0])
    # Once on the high side, values down to 1 - enter keep it there.
    assert sw("p", 0.35) == 1.0
    down = [sw("p", 0.1) for _ in range(4)]
    assert down == pytest.approx([0.75, 0.5, 0.25, 0.0])


def test_switch_uses_rig_units_and_reverses_mid_ramp():
    sw = Switches({"arm": (-30.0, 30.0)}, ramp=3)
    assert sw("arm", -25.0) == -30.0
    assert sw("arm", 25.0) == pytest.approx(-10.0)
    # Reversal mid-ramp walks back from where it is, never jumps.
    assert sw("arm", -25.0) == pytest.approx(-30.0)
    assert sw("arm", 0.0) == -30.0


def _motion(n=10):
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return {"params": {p: ramp + i for i, p in enumerate(
                ("ParamAngleX", "ParamEyeLOpen", "ParamEyeLSmile",
                 "ParamMouthOpenY", "ParamBreath", "ParamHairFront",
                 "ParamArmLA"))},
            "parts": {"PartArmA": np.ones(n, np.float32),
                      "PartArmB": np.zeros(n, np.float32)},
            "frames": n}


def test_base_state_excludes_autonomic_mouth_physics_and_driven():
    pids = list(_motion()["params"])
    phys = {"ParamHairFront"}
    bands = {p: band_of(p, "continuous", group_of(p), phys) for p in pids}
    assert bands["ParamEyeLOpen"] == "autonomic"
    assert bands["ParamMouthOpenY"] == "mouth"
    assert bands["ParamBreath"] == "autonomic"
    keep = base_keep({"ParamAngleX"}, phys, bands)
    other = _motion()
    other["params"]["ParamArmLA"] = other["params"]["ParamArmLA"] + 100
    motions = [("mtn_01.motion3.json", other),
               ("idle_02.motion3.json", _motion())]
    name, params, parts = base_state(motions, keep)
    assert name == "idle_02.motion3.json"           # idle preferred
    assert set(params) == {"ParamArmLA", "ParamEyeLSmile"}
    assert params["ParamArmLA"] == pytest.approx(6.0)   # its first frame
    assert parts == {"PartArmA": 1.0, "PartArmB": 0.0}


def test_base_state_without_motions():
    assert base_state([], lambda p: True) == (None, {}, {})


def test_static_dump_replays_motion_curve_for_curve(tmp_path):
    """--static writes the authored curves unchanged, snapping included.

    ParamArmSwap is switch-like (toggle_params flags it), and snapping used to
    rewrite it during authored playback; --static is the reference everything
    else is judged against, so its values must be the curve's own.
    """
    import json
    from flow.runtime import main
    from l2dprobe.motions import eval_segments

    swap = [0, 0, 0, 2.9, 0, 0, 3.0, 1, 0, 6.0, 1]
    turn = [0, -30, 1, 2, -30, 4, 30, 6, 30]
    motion = {"Version": 3, "Meta": {"Duration": 6.0, "Fps": 30},
              "Curves": [
                  {"Target": "Parameter", "Id": "ParamArmSwap", "Segments": swap},
                  {"Target": "Parameter", "Id": "ParamAngleX", "Segments": turn},
                  {"Target": "PartOpacity", "Id": "PartArmB", "Segments": swap}]}
    (tmp_path / "m.motion3.json").write_text(json.dumps(motion))
    model = tmp_path / "x.model3.json"
    model.write_text(json.dumps({"Version": 3, "FileReferences": {
        "Moc": "x.moc3", "Motions": {"Idle": [{"File": "m.motion3.json"}]}}}))
    pids = ["ParamAngleX", "ParamArmSwap", "ParamEyeLOpen"]
    rig = tmp_path / "x.npz"
    np.savez(rig, pid=np.array(pids, dtype=object),
             kind=np.array(["continuous"] * 3, dtype=object),
             group=np.array([group_of(p) for p in pids], dtype=object),
             minimum=np.array([-30, 0, 0], np.float32),
             maximum=np.array([30, 1, 1], np.float32),
             default=np.array([0, 0, 1], np.float32),
             descriptor=np.zeros((3, 4), np.float32))
    dump = tmp_path / "d.npz"
    assert main(["--model", str(model), "--rig", str(rig), "--static",
                 "--seconds", "6", "--dump", str(dump), "--dump-only"]) == 0
    z = np.load(dump, allow_pickle=True)
    assert list(z["pid"]) == ["ParamAngleX", "ParamArmSwap"]
    t = np.arange(180) / 30.0
    assert z["traj"][:, 0] == pytest.approx(eval_segments(turn, t) / 30, abs=1e-5)
    assert z["traj"][:, 1] == pytest.approx(eval_segments(swap, t) * 2 - 1,
                                            abs=1e-5)
    assert list(z["part_id"]) == ["PartArmB"]
    assert z["part_opacity"][:, 0] == pytest.approx(eval_segments(swap, t),
                                                    abs=1e-5)


def test_gaze_head_body_are_never_snapped():
    # aidang_2's animators hold the eyes at extremes, so toggle_params flagged
    # ParamEyeBallX/Y as switches and gaze froze. Continuous groups are exempt
    # whether the rig file or the id says so.
    from flow.runtime import snappable
    for pid in ("ParamEyeBallX", "ParamEyeBallY", "ParamAngleX",
                "ParamBodyAngleZ"):
        assert not snappable(pid, "other"), pid
    assert not snappable("Param14", "gaze")
    for pid in ("ParamHandChangeR", "ParamArmL2Hand1T", "ParamSphereOn"):
        assert snappable(pid, "arm" if "Arm" in pid or "Hand" in pid else "other"), pid


def test_whole_model_hold_never_takes_head_body_gaze():
    # On a bust-up rig a body angle can reach the top and bottom of the box
    # and look like a whole-model transform. ParamAllRotate sits in the body
    # group but is a whole-model rotation and must stay holdable.
    from flow.runtime import protected_from_hold
    for pid in ("ParamBodyAngleX", "ParamAngleZ", "ParamEyeBallX"):
        assert protected_from_hold(pid, "other"), pid
    assert protected_from_hold("Param14", "gaze")
    for pid in ("ParamAllX", "ParamAllRotate", "zongtiZ", "PARAM_moveY"):
        assert not protected_from_hold(pid, "body"), pid


def test_smooth_causal_uses_only_the_past_and_cuts_jitter():
    from flow.runtime import smooth_causal
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((300, 2)) * 0.01, 0)
    x = x + rng.standard_normal(x.shape) * 0.05          # high-freq wobble
    y = smooth_causal(x.astype(np.float32), 0.08, 30.0)
    jerk = lambda z: np.abs(np.diff(z, 3, axis=0)).mean()  # noqa: E731
    assert jerk(y) < jerk(x) / 5
    x2 = x.copy()
    x2[200:] += 1.0                                        # change the future
    y2 = smooth_causal(x2.astype(np.float32), 0.08, 30.0)
    assert np.array_equal(y[:200], y2[:200])               # past unaffected
    assert smooth_causal(x, 0, 30.0) is x


def test_arm_rotations_are_not_snapped_but_arm_swaps_are():
    from flow.runtime import snappable
    for pid in ("ParamArmForeLDAngel", "ParamArmUpperLDAngel",
                "ParamHandRAngle", "ParamArmRotate"):
        assert not snappable(pid, "arm"), pid
    for pid in ("ParamArmL2Hand1T", "ParamHandLDChangePa"):
        assert snappable(pid, "arm"), pid


# ---------------------------------------------------------------- breath

def _rig_path(name):
    import os
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "rigs", f"{name}.npz")


def test_breath_gain_is_calibrated_from_the_body_file():
    """The Azur Lane rigs' breath bobs the whole figure (lafei's torso
    0.069 hu over ParamBreath's range); their gain brings it to Hiyori's
    0.0073 hu. Hiyori's own stays at 1: her breath is the target."""
    import json
    from flow.runtime import BREATH_TARGET_HU, breath_reach, breath_setting
    got = {}
    for r in ("Hiyori", "Natori", "lafei", "xuefeng"):
        g, src = breath_setting(_rig_path(r), log=lambda s: None)
        with open(_rig_path(r)[:-4] + ".body.json", encoding="utf-8") as fh:
            pid, hu = breath_reach(json.load(fh))
        assert pid == "ParamBreath" and "body.json" in src
        got[r] = (g, hu)
        # Calibrated: what the breath moves now is the target, or less when
        # the rig's whole range is already below it.
        assert g * hu == pytest.approx(min(hu, BREATH_TARGET_HU), rel=0.01)
    assert got["Hiyori"][0] == 1.0
    assert got["Hiyori"][1] == pytest.approx(0.0073, abs=5e-4)
    for r in ("lafei", "xuefeng"):
        assert got[r][1] > 0.06 and got[r][0] < 0.15
    assert 0.3 < got["Natori"][0] < 0.6


def test_breath_gain_flag_and_voice_config_win(tmp_path):
    import json
    import shutil
    from flow import speech as sp
    from flow.runtime import breath_setting
    rig = tmp_path / "lafei.npz"
    shutil.copy(_rig_path("lafei")[:-4] + ".body.json",
                tmp_path / "lafei.body.json")
    assert breath_setting(str(rig), log=lambda s: None)[0] < 0.15
    assert breath_setting(str(rig), "0.5") == (0.5, "--breath-gain")
    vc = tmp_path / "lafei.voice.json"
    vc.write_text(json.dumps({"breath_gain": 0.3, "en": {"tts": "edge"}}))
    assert breath_setting(str(rig), "auto", log=lambda s: None)[0] == 0.3
    assert breath_setting(str(rig), "0.5")[0] == 0.5
    # The voice config's languages read as before, the key without a line.
    logs = []
    assert sp.load_voice_config(str(vc), logs.append) == {
        "en": {"tts": "edge"}}
    assert logs == []
    # No body file and no override: the full range, said once.
    logs = []
    assert breath_setting(str(tmp_path / "x.npz"), log=logs.append) == (
        1.0, "uncalibrated")
    assert len(logs) == 1 and "flow.body --measure" in logs[0]


def test_breath_gain_scales_the_breath_and_leaves_blinks_alone():
    from flow.runtime import Autonomic
    a = Autonomic(30.0, np.random.default_rng(4))
    b = Autonomic(30.0, np.random.default_rng(4), breath_gain=0.1)
    old, t = [], 0.0
    for f in range(300):
        ea, ba = a.step()
        eb, bb = b.step()
        assert ea == eb                    # the same blinks
        assert bb == pytest.approx(0.1 * ba)
        # Gain 1 is the breath of before, bit for bit.
        t += 1.0 / 30.0                    # as Autonomic keeps time
        assert ba == float(0.5 * (1 + np.sin(2 * np.pi * 0.25 * t)))
        old.append(ba)
    assert min(old) < 0.01 and max(old) > 0.99
