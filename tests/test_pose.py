"""
flow.pose (body mode, docs/embodiment.md E2) without GL or network: marker
parsing (split across stream chunks, bad JSON), timing from the words, the
solve on the measured Hiyori and lafei bodies, the pose layer's continuity
(no jumps, arm variants swapped at the midpoint, the idle untouched
elsewhere), its merge with the idle, the prompt's byte stability, reach
reports, the director end to end in body mode on a synthetic rig, the
stream drawing what the runtime draws with poses in it, the live
performer, and gesture sides through the body file's side naming.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from flow import body as fb
from flow import director as dr
from flow import events as ev
from flow import gestures as gz
from flow import pose as pz
from flow import speech as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIYORI = os.path.join(ROOT, "rigs", "Hiyori.body.json")
LAFEI = os.path.join(ROOT, "rigs", "lafei.body.json")
FPS = 30.0


@pytest.fixture(scope="module")
def hiyori():
    return fb.load(HIYORI)


@pytest.fixture(scope="module")
def lafei():
    return fb.load(LAFEI)


def _chunks(text, n):
    return [text[i:i + n] for i in range(0, len(text), n)]


def _parse(bm, chunks):
    p = pz.parser_for(bm)
    for c in chunks:
        p.feed(c)
    p.close()
    return p


# ---------------------------------------------------------------- parse

REPLY = ('Hi there! [pose {"hand.R.y": "@shoulder.y", "head.tilt": 8}] So '
         'good to see you. [pose {"head.pitch": -12, "dur": 0.25, "hold": 0}]'
         '[pose {"head.pitch": 0, "dur": 0.3, "hold": 0}] Yes. [blink] '
         '[pause:0.4] Look [pose {"gaze.x": -0.5, "note": "]"}] here.')


def test_markers_parse_the_same_however_the_stream_splits_them(hiyori):
    whole = _parse(hiyori, [REPLY])
    assert whole.spoken == "Hi there! So good to see you. Yes. Look here."
    assert [t["tag"] for t in whole.tags] == ["pose", "pose", "pose",
                                              "blink", "pause", "pose"]
    assert whole.tags[0]["arg"] == {"targets": {"hand.R.y": "@shoulder.y",
                                                "head.tilt": 8.0}}
    assert whole.tags[1]["arg"] == {"targets": {"head.pitch": -12.0},
                                    "dur": 0.25, "hold": 0.0}
    # A "]" inside a JSON string does not end the marker; the unknown key
    # is reported and the marker kept.
    assert whole.tags[-1]["arg"]["targets"] == {"gaze.x": -0.5}
    assert any("'note'" in w for w in whole.warnings)
    for n in (1, 2, 3, 7, 13):
        p = _parse(hiyori, _chunks(REPLY, n))
        assert p.spoken == whole.spoken and p.tags == whole.tags, n


def test_bad_markers_are_dropped_never_spoken(hiyori):
    text = ('One [pose {"head.yaw": 10,,}] two [pose {head.yaw: 10, '
            '\'gaze.y\': 0.5,}] three [pose {"mouth.open": 1, "x.y": 2}] '
            'four [nod] five [pose {"hand.L.y": "@elbow.y"}] six [pose '
            '{"dur": 9, "hold": -1, "head.yaw": "abc"}] seven [pose '
            '{"head.yaw": 5')
    p = _parse(hiyori, [text])
    assert p.spoken == "One two three four five six seven"
    assert "{" not in p.spoken and "pose" not in p.spoken
    # Only the repairable marker survives: bare keys, single quotes,
    # trailing comma.
    assert [t["arg"]["targets"] for t in p.tags] == [{"head.yaw": 10.0,
                                                      "gaze.y": 0.5}]
    w = "\n".join(p.warnings)
    for frag in ("bad JSON", "JSON repaired", "the voice drives the mouth",
                 "no variable 'x.y'", "[nod] is not a body-mode marker",
                 "no landmark '@elbow.y'", "not a number or @landmark",
                 "unterminated marker"):
        assert frag in w, frag
    # A JSON body that never closes is dropped whole, not read out.
    p = _parse(hiyori, ["Hi [pose {\"head.yaw\": " + "1" * 700 + " ok."])
    assert p.spoken == "Hi"
    # dur and hold are clipped to their ranges.
    p = _parse(hiyori, ['[pose {"head.yaw": 5, "dur": 9, "hold": -2}] x'])
    assert p.tags[0]["arg"] == {"targets": {"head.yaw": 5.0}, "dur": 3.0,
                                "hold": 0.0}


def test_marker_spellings(hiyori):
    p = _parse(hiyori, ['A [pose: {"head.yaw": 5}] b [{"head.yaw": 6}] c '
                        '[POSE{"head.yaw": 7}] d.'])
    assert [t["arg"]["targets"]["head.yaw"] for t in p.tags] == [5, 6, 7]
    assert p.spoken == "A b c d."


def test_sentence_cutter_keeps_a_marker_whole():
    from flow.live import SentenceCutter
    text = ('Hi. [pose {"hand.R.y": -0.2, "head.tilt": 8.5, "dur": 0.6, '
            '"hold": 1.5}] Nice to meet you, really. Bye.')
    for n in (1, 5, 11):
        c = SentenceCutter(max_tag=pz.MAX_MARKER)
        out = []
        for ch in _chunks(text, n):
            out += c.feed(ch)
        out += c.close()
        assert out[0] == "Hi."
        assert out[1].strip().startswith('[pose {"hand.R.y"')
        assert out[1].strip().endswith("really.")


# ---------------------------------------------------------------- timing

def test_pose_time_is_its_place_in_the_speech(hiyori):
    p = _parse(hiyori, ['Oh, [pose {"head.tilt": 8}] hello '
                        '[pose {"head.yaw": 10}] world.'])
    acts, _spans, _end = dr.timeline(p.spoken, p.tags)
    assert [a["tag"] for a in acts] == ["pose", "pose"]
    assert acts[0]["t"] == pytest.approx(3 / dr.LATIN_RATE + dr.CLAUSE_PAUSE,
                                       abs=1e-4)
    # From the TTS's word boundaries, exactly as a tag's.
    from tests.test_speech import FakeTTS, quiet, w_rate
    words = [("Oh", 0.10, 0.30), ("hello", 0.60, 1.00), ("world", 1.20,
                                                         1.70)]
    said = sp.speak(p.spoken, p.tags, FakeTTS([(quiet(2.0, -20), words)]),
                    w_rate, log=lambda s: None)
    assert [a["t"] for a in said["acts"]] == [0.6, 1.2]
    assert said["acts"][0]["arg"] == {"targets": {"head.tilt": 8.0}}

    # And from the moras, as VOICEVOX times them (its words are its mora
    # runs; speak() reports the method by the TTS's "timing").
    class MoraTTS(FakeTTS):
        timing = "moras"

        def synth(self, text):
            x, sr, w = super().synth(text)
            moras = [{"text": c, "start": a, "vstart": a, "end": b}
                     for c, a, b in (("o", 0.1, 0.3), ("ha", 0.65, 1.0),
                                     ("wa", 1.25, 1.7))]
            return x, sr, w, moras

    words = [("Oh", 0.10, 0.30), ("hello", 0.65, 1.00), ("world", 1.25,
                                                         1.70)]
    said = sp.speak(p.spoken, p.tags, MoraTTS([(quiet(2.0, -20), words)]),
                    w_rate, log=lambda s: None)
    assert said["method"] == "moras" and said["moras"]
    assert [a["t"] for a in said["acts"]] == [0.65, 1.25]


# ----------------------------------------------------------------- solve

def test_solve_reports_reach_on_what_is_played(hiyori):
    acts = [{"t": 0.5, "tag": "pose", "at": 0,
             "arg": {"targets": {"hand.R.y": "@shoulder.y",
                                 "head.tilt": 8}}},
            {"t": 0.6, "tag": "pose", "at": 0,
             "arg": {"targets": {"head.pitch": -12}, "dur": 0.25,
                     "hold": 0}},
            {"t": 0.6, "tag": "pose", "at": 0,
             "arg": {"targets": {"head.pitch": 0}, "dur": 0.3, "hold": 0}},
            {"t": 6.0, "tag": "pose", "at": 0,
             "arg": {"targets": {"hand.L.y": -1.0}}}]
    poses = pz.solve_poses(hiyori, acts, log=lambda s: None)
    a, n1, n2, miss = poses
    # The raised hand needs the other drawn arm pose: parts swap, both
    # targets met.
    assert a["variant_name"] == "PartArmB" and a["ok"]
    assert a["parts"] == {"PartArmA": 0.0, "PartArmB": 1.0}
    assert a["params"]["ParamAngleZ"] == pytest.approx(-8.0, abs=0.1)
    # Tiny prior nudges (BodyAngleX 0.15 % of its span) stay with the idle.
    assert "ParamBodyAngleX" not in a["params"]
    # A nod: the second stroke waits for the first. The head-only poses
    # keep the arm pose the body is in, and carry it: the first pose's
    # release must not swap the arms back under them.
    assert n1["start"] == pytest.approx(0.6)
    assert n2["start"] == pytest.approx(0.85)
    assert n1["variant_name"] == "PartArmB" and n1["parts"] == a["parts"]
    assert n1["parts_from"] == a["parts"]          # no swap: already there
    # Between the hanging and the raised arm there is nothing: reported,
    # with where it does reach.
    assert not miss["ok"] and not miss["report"]["hand.L.y"]["ok"]
    # An arm pose carries its drawing even without a swap: an idle gesture
    # may be playing the other one.
    assert miss["parts"] == {"PartArmA": 1.0, "PartArmB": 0.0}
    assert miss["parts_from"] == miss["parts"]
    assert "alone it reaches" in miss["hints"]["hand.L.y"]
    rs = pz.reach_summary(poses)
    assert rs["markers"] == 4 and rs["targets"] == 5 and rs["miss"] == 1
    lines = pz.report_lines(poses)
    assert "MISS" in lines[-1] and lines[0].endswith("ok")


def test_lafei_raises_the_right_arm_through_a_discrete_joint(lafei):
    """lafei's raised arm is PARAM_Arm_LT_Angle (screen-named, banded
    discrete by the probe, a joint by the body file): solved and played
    as solved."""
    poses = pz.solve_poses(lafei, [{"t": 0.0, "tag": "pose", "at": 0,
                                    "arg": {"targets": {
                                        "hand.R.y": "@shoulder.y"}}}])
    (p,) = poses
    assert p["ok"]
    assert "PARAM_Arm_LT_Angle" in p["params"]
    assert lafei.params["PARAM_Arm_LT_Angle"]["band"] == "discrete"


# ------------------------------------------------------------------ play

def _tok(t, params, dur=0.5, hold=1.0, release=0.5, parts=None, step=()):
    return {"t": t, "type": "pose", "role": "pose", "params": params,
            "parts": parts or {}, "step": list(step), "duration": dur,
            "hold": hold, "release": release, "groups": [],
            "src": "director"}


def _play(toks, n, under, **kw):
    """[(over, parts)] of a PoseLayer over `under(f) -> (over, parts)`."""
    lay = pz.PoseLayer(FPS, {"ParamA": 0.0, "ParamB": 0.0},
                       {"PartArmA": 1.0, "PartArmB": 0.0}, **kw)
    for k in toks:
        lay.commit(k)
    out = []
    for f in range(n):
        o, p = under(f)
        out.append(lay.apply(f, o, p))
    return out


def _idle(f):
    t = f / FPS
    return ({"ParamA": 0.3 * np.sin(1.3 * t), "ParamC": np.cos(t)},
            {"PartArmA": 1.0, "PartArmB": 0.0})


def test_layer_moves_holds_and_releases_without_a_jump():
    toks = [_tok(1.0, {"ParamA": 0.8, "ParamB": -0.5}),
            # Replaces A mid-hold, from where it is on screen.
            _tok(1.8, {"ParamA": -0.6}, dur=0.4, hold=0.5)]
    # Frozen holds (life = settle = 0): exactly at the targets.
    out = _play(toks, 150, _idle, life=0.0, settle=0.0)
    a = np.array([o["ParamA"] for o, _ in out])
    idle_a = np.array([_idle(f)[0]["ParamA"] for f in range(150)])
    # Untouched before the first pose and after the last release.
    np.testing.assert_array_equal(a[:30], idle_a[:30])
    assert np.allclose(a[110:], idle_a[110:])
    # At the targets during the holds.
    assert a[int(1.6 * FPS)] == pytest.approx(0.8)
    assert a[int(2.5 * FPS)] == pytest.approx(-0.6)
    # No step anywhere larger than the fastest minimum-jerk move here:
    # peak speed 1.875 x distance / duration.
    v_max = 1.875 * (0.8 + 0.6) / 0.4 / FPS
    assert np.abs(np.diff(a)).max() <= v_max * 1.05
    # ParamB appears, holds and goes again; ParamC (the idle's) never
    # touched.
    b = [o.get("ParamB") for o, _ in out]
    assert b[29] is None and b[int(1.6 * FPS)] == pytest.approx(-0.5)
    assert b[-1] is None
    assert all(o["ParamC"] == _idle(f)[0]["ParamC"]
               for f, (o, _) in enumerate(out))


def test_held_pose_keeps_the_idles_life_within_bounds_and_never_jumps():
    """Defaults: a held target moves with LIFE of the idle's motion since
    the pose began and settles SETTLE of the way toward the idle over the
    hold; it stays near its target, never jumps, and is never frozen."""
    t0, dur, hold, rel = 1.0, 0.4, 2.0, 0.5
    toks = [_tok(t0, {"ParamA": 0.8, "ParamB": -0.5}, dur=dur, hold=hold,
                 release=rel)]
    n = int((t0 + dur + hold + rel + 1.0) * FPS)
    out = _play(toks, n, _idle)
    a = np.array([o["ParamA"] for o, _ in out])
    b = np.array([o.get("ParamB", np.nan) for o, _ in out])
    idle_a = np.array([_idle(f)[0]["ParamA"] for f in range(n)])
    h0 = int(np.ceil((t0 + dur) * FPS)) + 1
    h1 = int((t0 + dur + hold) * FPS)
    u0 = idle_a[int(np.ceil(t0 * FPS))]
    held = a[h0:h1]
    # Alive: it moves on every frame of the hold.
    assert (np.abs(np.diff(held)) > 1e-6).all()
    # Within bounds: life adds at most LIFE x the idle's excursion, settle
    # at most SETTLE of the way to the idle.
    exc = np.abs(idle_a[h0:h1] - u0).max()
    gap = np.abs(idle_a[h0:h1] - 0.8).max()
    assert np.abs(held - 0.8).max() <= pz.LIFE * exc + pz.SETTLE * gap + 1e-9
    # Settled toward the idle by the end, but still clearly posed.
    assert abs(a[h1 - 1] - idle_a[h1 - 1]) < abs(0.8 - idle_a[h1 - 1])
    assert abs(a[h1 - 1] - 0.8) < 0.5 * abs(0.8 - idle_a[h1 - 1])
    # A parameter with no idle under it (B) settles too: never frozen.
    hb = b[h0:h1]
    assert (np.abs(np.diff(hb)) > 1e-7).all()
    assert hb[-1] == pytest.approx(-0.5 * (1 - pz.SETTLE), abs=1e-3)
    # No jumps: no step above the minimum-jerk move's peak plus the idle's
    # own speed, and back on the idle after the release.
    v_max = 1.875 * 0.8 / dur / FPS + np.abs(np.diff(idle_a)).max()
    assert np.abs(np.diff(a)).max() <= v_max * 1.05
    np.testing.assert_array_equal(a[:int(t0 * FPS)], idle_a[:int(t0 * FPS)])
    end = int(np.ceil((t0 + dur + hold + rel) * FPS)) + 1
    np.testing.assert_allclose(a[end:], idle_a[end:])
    assert np.isnan(b[end:]).all()


def test_variant_switch_swaps_parts_at_the_midpoint_never_two_arms():
    toks = [_tok(1.0, {"ParamB": 1.0, "ParamSwitch": 1.0}, dur=0.6,
                 hold=1.0, release=0.6,
                 parts={"PartArmA": 0.0, "PartArmB": 1.0},
                 step=["ParamSwitch"])]
    out = _play(toks, 120, _idle)
    pa = np.array([p["PartArmA"] for _, p in out])
    pb = np.array([p["PartArmB"] for _, p in out])
    assert set(pa) | set(pb) <= {0.0, 1.0}
    assert (pa + pb == 1.0).all()                 # exactly one arm drawn
    f_on = int(np.argmax(pb > 0))
    assert abs(f_on - (1.0 + 0.3) * FPS) <= 1
    f_off = f_on + int(np.argmax(pb[f_on:] == 0))
    assert abs(f_off - (1.0 + 0.6 + 1.0 + 0.3) * FPS) <= 1
    sw = [o.get("ParamSwitch", 0.0) for o, _ in out]
    assert set(sw) <= {0.0, 1.0}
    assert sw[f_on] == 1.0 and sw[f_on - 1] == 0.0
    # The part keeps being written after the release (the backend keeps a
    # part's last opacity).
    assert out[-1][1]["PartArmB"] == 0.0


def test_pose_merge_takes_the_arms_only_and_keeps_the_idle_moving():
    idle = [{"t": 0.5, "type": "head_turn", "role": "head",
             "target": {"head_x": 0.2}, "duration": 1.0, "follows": None},
            {"t": 0.6, "type": "brow", "role": "brow",
             "target": {"brow_l": 0.2}, "duration": 0.5},
            {"t": 1.5, "type": "gesture", "role": "arms", "motion": "m",
             "start": 0, "duration": 3, "fade": 0.5},
            {"t": 4.5, "type": "head_turn", "role": "head",
             "target": {"head_x": 0.1}, "duration": 1.0, "follows": None},
            {"t": 2.0, "type": "blink", "role": "eyes"}]
    poses = [{"start": 1.0, "dur": 0.5, "hold": 1.5, "release": 0.5,
              "params": {"ParamAngleX": 10.0, "ParamArmLA": 3.0},
              "parts": {}, "step": [], "variant_name": "A"}]
    roles = {"head_x": ("ParamAngleX", 1.0, "id")}
    toks = pz.tokens(poses, roles, {"ParamArmLA"})
    assert toks[0]["groups"] == ["arms", "head"]
    out = dr.merge(idle, toks)
    kinds = [(k["type"], k["t"]) for k in out]
    # The idle's head moves run under the pose (the layer lets them show
    # at LIFE on the head parameters it holds, fully on the others).
    assert ("head_turn", 0.5) in kinds and ("head_turn", 4.5) in kinds
    assert ("gesture", 1.5) not in kinds          # starts inside it
    assert ("brow", 0.6) in kinds
    assert ("blink", 2.0) in kinds and ("pose", 1.0) in kinds
    # A pose that moves no arm takes nothing: the idle gesture stays.
    head_only = pz.tokens([dict(poses[0], params={"ParamAngleX": 10.0})],
                          roles, {"ParamArmLA"})
    assert head_only[0]["groups"] == ["head"]
    assert ("gesture", 1.5) in [(k["type"], k["t"])
                                for k in dr.merge(idle, head_only)]


# ---------------------------------------------------------------- prompt

def test_prompt_is_byte_stable_and_says_what_the_body_can_do(hiyori, lafei):
    a = pz.system_prompt(hiyori, "Hiyori.")
    assert a == pz.system_prompt(fb.load(HIYORI), "Hiyori.")
    import re
    assert not re.search(r"Param|PARAM|Part[A-Z]", a)   # no rig ids
    assert "hand.L is your left hand" in a
    assert "cannot stop between heights" in a     # the two drawn arms
    assert "(your body now: ...)" in a and "picture of how" not in a
    assert "picture of how you look" in pz.system_prompt(hiyori, "H", True)
    b = pz.system_prompt(lafei, "Laffey.")
    assert b == pz.system_prompt(fb.load(LAFEI), "Laffey.")
    assert "your left hand cannot reach shoulder height" in b
    assert "no separate hands" in b
    assert "eye.L.smile" in b.split("do not exist on this body:")[1]
    # Every example marker parses on its own body without a warning.
    for bm, text in ((hiyori, a), (lafei, b)):
        ex = text.split("never copy them):\n")[1]
        p = _parse(bm, [ex])
        assert p.tags and not p.warnings, p.warnings
    # The pose numbers go to the user message.
    assert pz.pose_now(hiyori) == ("(your body now: at rest; hand.L (0.95, "
                                   "-2.00), hand.R (-0.78, -2.09))")
    up = pz.pose_now(hiyori, {"ParamArmLB": 0.0},
                     {"PartArmA": 0.0, "PartArmB": 1.0})
    assert up.startswith("(your body now: arms in another drawing, ")
    now = pz.pose_now(hiyori, {"ParamAngleZ": -8.0})
    assert "head.tilt 8" in now and "at rest" not in now


def test_prompt_asks_for_brief_varied_beats_and_the_whole_body(hiyori,
                                                                lafei):
    """E2's body replies were long static holds on the face; the prompt
    now asks for brief beats, short holds, hands where they reach, and
    variety -- and keeps every constraint line."""
    import re
    for bm, hand in ((hiyori, "your left hand reaches up to shoulder "
                      "height"),
                     (lafei, "your right hand reaches up to face height")):
        p = pz.system_prompt(bm, "X.")
        assert hand in p
        for line in ("Move in brief beats", "Keep holds short",
                     "Hold 2 seconds or more only for the one moment",
                     "Vary your moves from reply to reply",
                     "held eyes stop blinking",
                     # the constraints stay
                     "What you cannot do:", "Only the variables listed "
                     "above, inside their ranges. Valid JSON.",
                     "No other tags or markup", "Never explain or mention"):
            assert line in p, line
        assert "never while you pose a hand" not in p    # no gestures
        ex = p.split("never copy them):\n")[1]
        # No example holds long (they are copied most of all), one glances
        # away and back.
        holds = [float(h) for h in re.findall(r'"hold": ([0-9.]+)', ex)]
        assert holds and max(holds) <= 0.5
        assert '"gaze.x": -0.6' in ex and '"gaze.x": 0' in ex
    g = pz.system_prompt(hiyori, "X.", gestures="[gesture:NAME]  x\n")
    assert "[gesture:NAME]  x\n[blink]" in g
    assert "never while you pose a hand" in g


def test_body_gestures_parse_and_give_way_to_an_arm_pose(hiyori):
    p = pz.parser_for(hiyori, gestures=("wave", "bow"))
    p.feed('Hi! [gesture:wave] Look [pose {"hand.L.y": "@shoulder.y"}] '
           'here. [gesture:dance] [nod]')
    p.close()
    assert [t["tag"] for t in p.tags] == ["gesture", "pose"]
    assert any("no motion 'dance'" in w for w in p.warnings)
    assert any("[nod] is not a body-mode marker (pose, blink, pause, "
               "gesture)" in w for w in p.warnings)
    q = pz.parser_for(hiyori)                      # --no-body-gestures
    q.feed("Hi! [gesture:wave] there.")
    q.close()
    assert not q.tags and "(pose, blink, pause)" in q.warnings[0]
    # A gesture while a pose moves an arm gives way; one beside a head
    # pose stays.
    motions = {"Wave.motion3.json": 3.0, "Bow.motion3.json": 2.0}
    acts = [{"t": 0.2, "tag": "gesture", "arg": "wave", "at": 0},
            {"t": 1.0, "tag": "pose", "arg": {}, "at": 5},
            {"t": 9.0, "tag": "gesture", "arg": "bow", "at": 9},
            {"t": 9.5, "tag": "pose", "arg": {}, "at": 12}]
    poses = [{"act": 1, "start": 1.0, "dur": 0.5, "hold": 1.0,
              "release": 0.5, "arm_params": ["ParamArmLB"], "parts": {}},
             {"act": 3, "start": 9.5, "dur": 0.5, "hold": 1.0,
              "release": 0.5, "arm_params": [], "parts": {}}]
    logs = []
    gone = pz.drop_clashing_gestures(acts, poses, motions, logs.append)
    assert [x["arg"] for x in gone] == ["wave"]
    assert [x["tag"] for x in acts] == ["pose", "gesture", "pose"]
    assert [acts[p["act"]]["at"] for p in poses] == [5, 12]
    assert "[gesture:wave] at 0.20s dropped" in logs[0]


def test_images_only_for_backends_that_take_them():
    png = b"\x89PNG fake"
    assert pz.user_message("hi", "deepseek", png) == {"role": "user",
                                                      "content": "hi"}
    m = pz.user_message("hi", "anthropic", png)
    assert m["content"][0]["type"] == "image"
    assert m["content"][0]["source"]["media_type"] == "image/png"
    m = pz.user_message("hi", "siliconflow", png)
    assert m["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,")
    assert pz.user_message("hi", "ollama", png)["images"]
    assert pz.text_only(m) == {"role": "user", "content": "hi"}


def test_self_check_message_and_ok(hiyori):
    poses = pz.solve_poses(hiyori, [{"t": 0.0, "tag": "pose", "at": 4,
                                     "arg": {"targets": {"hand.L.y": -1.0}}}])
    msg = pz.check_message(poses, "Hey, look here.")
    assert "hand.L.y -1.00 -> -1.66 MISS" in msg and "'look here.'" in msg
    assert pz.is_ok_reply("OK") and pz.is_ok_reply(" ok. ")
    assert not pz.is_ok_reply("OK, [pose {}] fixed")


# ------------------------------------------------- end to end on a fake rig

def _body_rig(root):
    """A rig whose parameters are test_body's FakeCore's (hanging arms in
    PartArmA, a raised left arm in PartArmB), its measured body file next
    to it, and one arm motion."""
    from l2dprobe.analyze import group_of
    from tests.test_body import FakeCore, _info
    root.mkdir(parents=True)
    core = FakeCore()
    blob = fb.measure(core, _info(), log=lambda *a: None)
    blob["rig"], blob["model_json"] = "x", "x.model3.json"
    fb.save(blob, str(root / "x.body.json"))
    arm = [0, 0, 1, 2.0, 3, 4.0, 0, 6.0, 0]
    motion = {"Version": 3, "Meta": {"Duration": 6.0, "Fps": 30},
              "Curves": [{"Target": "Parameter", "Id": "ParamArmRA",
                          "Segments": arm}]}
    (root / "home.motion3.json").write_text(json.dumps(motion))
    model = root / "x.model3.json"
    model.write_text(json.dumps({"Version": 3, "FileReferences": {
        "Moc": "x.moc3", "Motions": {"": [{"File": "home.motion3.json"}]}}}))
    pids = list(core.param_ids) + ["ParamEyeLOpen"]
    n = len(pids)
    lo = list(core.param_min) + [0]
    hi = list(core.param_max) + [1]
    df = list(core.param_default) + [1]
    rig = root / "x.npz"
    np.savez(rig, pid=np.array(pids, dtype=object),
             kind=np.array(["continuous"] * n, dtype=object),
             group=np.array([group_of(p) for p in pids], dtype=object),
             minimum=np.array(lo, np.float32),
             maximum=np.array(hi, np.float32),
             default=np.array(df, np.float32),
             descriptor=np.zeros((n, 4), np.float32))
    return model, rig, fb.load(str(root / "x.body.json"))


def test_director_body_mode_end_to_end(tmp_path, capsys):
    from flow.runtime import main as runtime_main
    model, rig, bm = _body_rig(tmp_path / "x")
    y = bm.landmarks["shoulder.y"] + 0.3
    reply = tmp_path / "reply.txt"
    reply.write_text(f'Hi! [pose {{"hand.L.y": {y:.3f}}}] Look at this. '
                     '[pose {"head.yaw": 15, "hold": 0.5}] Nice. [nod]',
                     encoding="utf-8")
    out = tmp_path / "run"
    args = ["--model-json", str(model), "--rig", str(rig), "--backend",
            "file", "--reply-file", str(reply), "--out", str(out),
            "--probe-dir", str(tmp_path), "--seed", "2", "--body-targets"]
    assert dr.main(args) == 0
    log = capsys.readouterr().out
    assert "[nod] is not a body-mode marker" in log
    assert "pose: 2 markers, 2/2 targets reached" in log
    a = json.loads((out / "actions.json").read_text())
    assert a["mode"] == "body" and a["reach"]["ok"] == 2
    solved = [x["solved"] for x in a["actions"] if x["tag"] == "pose"]
    assert solved[0]["parts"] == {"PartArmA": 0.0, "PartArmB": 1.0}
    # The render runs until the last pose has relaxed.
    assert a["seconds"] >= pz.end_time(solved)
    blob = json.loads((out / "tokens.json").read_text())
    pt = [k for k in blob["tokens"] if k["type"] == "pose"]
    # The head pose comes while the raised arm is held, and holds that
    # drawing too.
    assert "arms" in pt[0]["groups"] and pt[1]["groups"] == ["arms", "head"]
    plan = np.load(out / "plan.npz", allow_pickle=True)
    pid = list(plan["pid"])
    lb = plan["traj"][:, pid.index("ParamArmLB")]
    assert lb.max() - lb.min() > 0.2               # the raised arm moves
    assert plan["mask"][:, pid.index("ParamArmLB")].any()
    parts = list(plan["part_id"])
    op = plan["part_opacity"]
    pa, pb = op[:, parts.index("PartArmA")], op[:, parts.index("PartArmB")]
    # The fake rig's motions set no parts (no base parts): untouched until
    # the first pose, then exactly one arm drawn.
    w = ~np.isnan(pa)
    assert w.any() and ((pa + pb)[w] == 1.0).all() and np.nanmax(pb) == 1.0
    # The printed render command replays it bit for bit.
    import shlex
    cmd = [ln for ln in log.splitlines() if "--tokens-in" in ln][0]
    argv = shlex.split(cmd.strip())
    argv = argv[argv.index("flow.runtime") + 1:]
    k = argv.index("--out")
    argv = argv[:k] + argv[k + 2:]
    assert runtime_main(argv + ["--dump-only", "--dump",
                                str(tmp_path / "re.npz")]) == 0
    re_ = np.load(tmp_path / "re.npz", allow_pickle=True)
    np.testing.assert_array_equal(re_["traj"], plan["traj"])
    np.testing.assert_array_equal(re_["part_opacity"], plan["part_opacity"])
    # And the stream draws the same frames from the same tokens.
    from flow import stream as fs
    from tests.test_live import _stream_plan
    er = fs.EventRig(str(model), str(rig), "default", None, None, FPS,
                     str(tmp_path), log=lambda s: None)
    traj, frames = _stream_plan(er, blob, len(plan["traj"]))
    np.testing.assert_array_equal(traj, plan["traj"])
    sop = np.array([[fr[1].get(p, np.nan) for p in parts] for fr in frames],
                   np.float32)
    np.testing.assert_array_equal(sop, plan["part_opacity"])
    # The body-mode prompt is what --print-system prints: with the rig's
    # own motions offered by default, without them pure body.
    capsys.readouterr()
    assert dr.main(args + ["--print-system", "--no-body-gestures",
                           "--no-body-moves"]) == 0
    assert capsys.readouterr().out.strip() == pz.system_prompt(bm).strip()
    assert dr.main(args + ["--print-system"]) == 0
    sysp = capsys.readouterr().out
    # Motion primitives are offered by default (flow.moves).
    assert "[move {" in sysp and "[move {" not in pz.system_prompt(bm)
    assert "[gesture:NAME]" in sysp and "  home: " in sysp
    assert "never while you pose a hand" in sysp
    assert "[gesture:NAME]" not in pz.system_prompt(bm)


def test_live_performer_plays_body_targets(tmp_path):
    import argparse
    from flow import live as lv
    from flow import stream as fs
    from tests.test_live import FakeClock, SpeakingTTS
    model, rig_path, bm = _body_rig(tmp_path / "x")
    rig = fs.EventRig(str(model), str(rig_path), "default", None, None, FPS,
                      str(tmp_path), log=lambda s: None)
    clock = FakeClock(2.0)
    hub = lv.Hub(clock)
    stream = fs.MotionStream(rig, seed=0, autonomic=False,
                             log=lambda s: None)
    a = argparse.Namespace(seed=0, lead=0.25, gap=0.12, lipsync="off",
                           mouth_gain=1.0, cjk_rate=dr.CJK_RATE,
                           latin_rate=dr.LATIN_RATE, mood="auto",
                           backend="deepseek")
    run = {"system": pz.system_prompt(bm), "labels": {}, "offered": None,
           "mode": "body", "body": bm, "core": None, "images": False,
           "max_tag": pz.MAX_MARKER, "parser": lambda: pz.parser_for(bm)}
    reply = ('Hello there. [pose {"head.yaw": 15, "hold": 0.5}] Nice to '
             'see you. [pose {"hand.L.y": "@shoulder.y+0.3"}] Bye!')
    logs = []
    p = lv.Performer(a, run, rig, stream, clock, hub, dr.TextBackend(reply),
                     SpeakingTTS(clock), log=logs.append)
    p.turn("hi")
    mine = [k for k in stream.pending if k["type"] == "pose"]
    assert len(mine) == 2
    au = [m for m in hub.history if m["type"] == "audio"]
    assert mine[0]["t"] >= au[1]["t"] - 1e-6        # in its sentence
    assert any("head.yaw 15 -> 15 ok" in s for s in logs)
    # The user turn carried the body's numbers; history keeps text only.
    assert "(your body now:" in p.messages[0]["content"]
    rep = [m for m in hub.history if m["type"] == "reply"][0]
    assert any(s.get("reach") for s in rep["sentences"])
    # Played: the raised arm comes in, the parts swap once.
    pb = []
    for f in range(int((mine[1]["t"] + 3.0) * FPS)):
        over, parts = stream.frame()
        pb.append(parts.get("PartArmB", 0.0))
    assert max(pb) == 1.0 and set(pb) <= {0.0, 1.0}


# -------------------------------------------------------------- gestures

def test_gesture_sides_follow_the_body_files_side_naming(tmp_path):
    assert gz.character_side("PARAM_Arm_LT_Angle") == "L"
    assert gz.character_side("PARAM_Arm_LT_Angle", "screen") == "R"
    assert gz.character_side("PARAM_ALL_moveZ", "screen") == ""
    assert gz.side_naming(os.path.join(ROOT, "rigs", "lafei.npz")) == {
        "drawables": "screen", "params": "screen",
        "votes": {"drawables": [1, 11], "params": [0, 10]}}
    assert gz.side_naming(os.path.join(ROOT, "rigs", "Hiyori.npz"))[
        "params"] == "character"
    assert gz.side_naming(str(tmp_path / "none.npz")) is None
    # A motion raising lafei's PARAM_Arm_LT_Angle moves her right arm.
    m = {"params": {"PARAM_Arm_LT_Angle": np.linspace(0, 1, 30)},
         "parts": {}, "frames": 30}
    lo, hi, rest = ({"PARAM_Arm_LT_Angle": 0.0}, {"PARAM_Arm_LT_Angle": 1.0},
                    {"PARAM_Arm_LT_Angle": 0.0})
    keep = lambda p: True                      # noqa: E731
    sn = {"params": "screen", "drawables": "screen"}
    named, _, _ = gz.measure(m, keep, lo, hi, rest)
    seen, _, _ = gz.measure(m, keep, lo, hi, rest, naming=sn)
    assert list(named) == [("arm", "L")] and list(seen) == [("arm", "R")]
    at = gz.atoms({}, {"PartArmL": 1}, {}, sn)
    assert ("s", "arm", "R") in at


def test_live_gesture_from_an_earlier_sentence_gives_way_to_an_arm_pose(
        tmp_path):
    import argparse
    from flow import live as lv
    from flow import stream as fs
    from tests.test_live import FakeClock, SpeakingTTS
    model, rig_path, bm = _body_rig(tmp_path / "x")
    rig = fs.EventRig(str(model), str(rig_path), "default", None, None, FPS,
                      str(tmp_path), log=lambda s: None)
    labels = dr.gesture_labels(rig.motion_seconds)
    lab = sorted(labels)[0]
    clock = FakeClock(2.0)
    stream = fs.MotionStream(rig, seed=0, autonomic=False,
                             log=lambda s: None)
    a = argparse.Namespace(seed=0, lead=0.25, gap=0.12, lipsync="off",
                           mouth_gain=1.0, cjk_rate=dr.CJK_RATE,
                           latin_rate=dr.LATIN_RATE, mood="auto",
                           backend="deepseek")
    run = {"system": "", "labels": labels, "offered": None,
           "mode": "body", "body": bm, "core": None, "images": False,
           "max_tag": pz.MAX_MARKER,
           "parser": lambda: pz.parser_for(bm, gestures=labels)}

    def directed_gestures():
        return [k for k in stream.pending if k["type"] == "gesture"
                and k.get("src") == "director" and not k.get("settle")]

    logs = []
    p = lv.Performer(a, run, rig, stream, clock, lv.Hub(clock),
                     dr.TextBackend(f"Hello! [gesture:{lab}] Hi."),
                     SpeakingTTS(clock), log=logs.append)
    p.turn("hi")
    assert len(directed_gestures()) == 1
    # The next turn raises a hand while that gesture is still to play.
    p.backend = dr.TextBackend('Look [pose {"hand.L.y": "@shoulder.y+0.3"}] '
                               'here.')
    p.turn("where?")
    assert directed_gestures() == []
    assert any("gives way to an arm pose" in s for s in logs)
    assert [k for k in stream.pending if k["type"] == "pose"]


def test_tag_mode_plans_do_not_depend_on_the_breath_gain(tmp_path, capsys):
    """Tag mode changes only in the rendered breath: the gain is written
    in the render loop after the dump, so tokens and plan are the same at
    any gain; the render command carries it."""
    model, rig, _bm = _body_rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text("Oh, [brow:up] really? [nod] [look:away] Fine.",
                     encoding="utf-8")
    out = {}
    for g in ("1", "0.1"):
        d = tmp_path / f"g{g}"
        assert dr.main(["--model-json", str(model), "--rig", str(rig),
                        "--backend", "file", "--reply-file", str(reply),
                        "--out", str(d), "--probe-dir", str(tmp_path),
                        "--breath-gain", g]) == 0
        log = capsys.readouterr().out
        assert f"--breath-gain {g}" in log
        out[g] = d
    assert ((out["1"] / "tokens.json").read_bytes()
            == (out["0.1"] / "tokens.json").read_bytes())
    p1 = np.load(out["1"] / "plan.npz", allow_pickle=True)
    p2 = np.load(out["0.1"] / "plan.npz", allow_pickle=True)
    for k in ("traj", "mask", "part_opacity"):
        np.testing.assert_array_equal(p1[k], p2[k])
    # Without pose tokens the merge's windows are the moves' and gestures'
    # as before (poses take only the arms now).
    toks = [{"t": 1.0, "type": "head_turn", "role": "head", "duration": 0.5,
             "hold": 1.0}, {"t": 2.0, "type": "gesture", "role": "arms",
                            "duration": 3.0}]
    assert dr.windows(toks) == {"head": [(1.0, 2.5)], "arms": [(2.0, 5.0)]}
