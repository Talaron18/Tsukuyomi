"""
Framing and motion budget: the [frame bust] / [frame full] marker (parser,
unplayed accounting), the camera line and the budget line of the beats
prompt, the camera's ease between the two set_view states (and its use by
the live performer / render loop), the framing state of generated data and
the hand cap of manners. No network, no GL: mocked models, FakeBackend.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pytest

from flow import body as fb
from flow import dataexport as dx
from flow import dataset as ds
from flow import director as dr
from flow import framing as fr
from flow import pose as pz
from flow import stream as fs
from flow import vrm as fv
from tests.test_dataset import CharLLM, ROOT, UserLLM
from tests.test_pose import FPS, _body_rig

RIG = {n: os.path.join(ROOT, "rigs", f"{n}.body.json")
       for n in ("Hiyori", "lafei", "Natori", "seed-san")}


def _bm(name):
    if name == "seed-san":
        return fv.body_for(None, RIG[name], log=lambda s: None)[0]
    return fb.load(RIG[name])


# ---------------------------------------------------------------- parser

def _parse(text, frames=True):
    p = pz.PoseParser(frames=frames)
    p.feed(text)
    p.close()
    return p


def test_frame_marker_parses_in_both_spellings():
    p = _parse("Hi [frame full] there. [frame:bust] [Frame \"full\"] Bye.")
    assert [(t["tag"], t["arg"], t["at"]) for t in p.tags] == [
        ("frame", "full", 3), ("frame", "bust", 10), ("frame", "full", 10)]
    assert p.spoken == "Hi there. Bye." and not p.warnings
    assert pz.tag_text(p.tags[0]) == "[frame full]"


def test_frame_marker_with_a_bad_view_is_dropped_and_counted_unplayed():
    p = _parse("Hi [frame sideways] there.")
    assert not p.tags and p.spoken == "Hi there."
    (w,) = p.warnings
    assert "bust/full" in w and "dropped" in w
    # The dataset's unplayed accounting reads the tag name off the warning.
    assert re.match(r"\s*\[(\w+)", w).group(1) == "frame"


def test_frame_marker_is_off_unless_the_parser_takes_it():
    p = _parse("Hi [frame full] there.", frames=False)
    assert not p.tags and "not a body-mode marker" in p.warnings[0]
    assert "frame" not in p.warnings[0].split("(")[1]
    bm = fb.load(RIG["Hiyori"])
    on = pz.parser_for(bm, frames=True)
    on.feed("[frame full]")
    assert on.tags and on.tags[0]["tag"] == "frame"
    # A VRM is drawn at one camera: it cannot switch.
    vrm = pz.parser_for(_bm("seed-san"), frames=True)
    vrm.feed("[frame full]")
    assert not vrm.tags


def test_exports_keep_frame_markers_and_count_them():
    text = "Hi. [frame full] [pose {\"head.tilt\": 3}] Look."
    spans = dx.marker_spans(text)
    assert [t["tag"] for _a, _b, t in spans] == ["frame", "pose"]
    assert dx.spoken_of(text) == "Hi. Look."
    r = {"refused": False, "spoken": "Hi.", "markers": [
        {"tag": "frame", "arg": "full", "played": True}], "reach": {}}
    assert "no_body" in dx.flag(r)          # a frame alone is no body marker


# ------------------------------------------------------------ the prompt

def test_v1_prompt_has_no_camera():
    for n in RIG:
        v1 = pz.system_prompt(_bm(n), "X.")
        assert "amera" not in v1 and "[frame" not in v1
        assert "Motion budget" not in v1


def test_camera_line_is_worded_from_the_window_and_the_hands():
    w = fr.window()
    assert (w["top"], w["bottom"]) == (fr.ABOVE, -fr.BELOW)
    hiyori = pz.camera_rule(_bm("Hiyori"))
    # Her hands rest below the bust picture; a hand shows once raised.
    assert "1.5 hu above your chin to 1.6 hu below" in hiyori
    assert "rest at y -2.00 and -2.09, below the picture" in hiyori
    assert "above y -1.60" in hiyori and "not seen" in hiyori
    assert "skip it and let the face carry the moment" in hiyori
    assert "instead of switching to full" in hiyori
    assert "a switch lasts for this reply, and the next one starts back in " \
        "your home" in hiyori
    assert "In full your whole body is in the picture" in hiyori
    lafei = pz.camera_rule(_bm("lafei"))
    # lafei's hands rest at -1.10 / -1.15: inside the picture, not hidden.
    assert "in the lower part of the picture" in lafei
    assert "not seen" not in lafei and "All your hand moves stay" in lafei
    seed = pz.camera_rule(_bm("seed-san"))
    assert "filmed full" in seed and "bust (head" not in seed


def test_beats_rule_asks_for_a_beat_in_every_sentence_and_variety():
    sp = pz.system_prompt(_bm("Hiyori"), "X.", prompt="beats")
    assert "every sentence gets at least one expression beat" in sp
    assert "two or three" in sp and "every 1.5 to 2 seconds" in sp
    assert "Vary them, not the same smile each time" in sp
    assert "Make expression beats in every sentence: at least one" in sp
    assert "One to three" not in sp
    assert dx.BEATS_MARK in sp
    assert "A beat is a quick change of" in sp       # the definition
    # The budget line keeps a hand move from being a reason to go full.
    assert "never for a single hand move" in pz.budget_line(_bm("Hiyori"))
    assert "never for a single hand move" not in pz.budget_line(
        _bm("seed-san"))
    # r1's prompt is untouched.
    assert "every sentence gets" not in pz.system_prompt(_bm("Hiyori"), "X.")


def test_beats_cap_does_not_bind_for_a_short_reply_of_two_beats_each():
    assert dx.BEATS_MAX == 12
    four = " ".join(
        '[pose {"mouth.form": 0.4, "head.tilt": 3}] Yes. '
        '[pose {"brow.L.lift": 0.4, "brow.R.lift": 0.4}] Really?'
        for _ in range(4))
    res = dx.manners_reply(four, False, max_poses=2, max_beats=dx.BEATS_MAX)
    assert res["reply"].count("[pose") == 8


def test_user_turn_camera_line_follows_the_framing():
    bm = _bm("Hiyori")
    bust = pz.camera_now(bm, "bust")
    assert bust.startswith("(camera: bust") and "head and chest" in bust
    assert "below y -1.60 are out of frame" in bust
    full = pz.camera_now(bm, "full")
    assert full.startswith("(camera: full") and "-1.60" not in full
    # A body that cannot switch is only ever shown full.
    assert pz.camera_now(_bm("seed-san"), "bust").startswith("(camera: full")


def test_beats_prompt_carries_the_marker_entry_and_the_lines():
    sp = pz.system_prompt(_bm("Hiyori"), "X.", prompt="beats")
    assert "[frame bust] or [frame full]" in sp and "0.6 seconds" in sp
    # A switch lasts for the reply and is rarer: a whole-body moment only.
    assert "At most once in a reply" in sp and "at most twice" not in sp
    assert "whole-body moment" in sp and "not for a greeting wave" in sp
    assert "lasts until the end of this reply" in sp
    assert "starts back in your home" in sp
    assert "- Camera: " in sp and "- Motion budget: " in sp
    assert dx.BEATS_MARK in sp
    vrm = pz.system_prompt(_bm("seed-san"), "X.", prompt="beats")
    assert "[frame bust]" not in vrm and "- Camera: you are filmed full" in vrm
    # Byte-stable per body: the framing goes in the user turn.
    assert sp == pz.system_prompt(_bm("Hiyori"), "X.", prompt="beats")


# ---------------------------------------------------------------- budget

def test_budget_is_measured_from_the_body():
    b = {n: fr.budget(_bm(n)) for n in RIG}
    assert b["Hiyori"]["tier"] == "limited" and b["lafei"]["tier"] == "limited"
    assert b["Natori"]["tier"] == "moderate"
    assert b["seed-san"]["tier"] == "capable"
    assert b["seed-san"]["reach"]["all"] == 1.0
    assert b["Hiyori"]["reach"]["all"] < fr.LIMITED
    assert b["Natori"]["reach"]["drawings"] == 12
    # Face controls come from pose.offered.
    for n, bm in ((n, _bm(n)) for n in RIG):
        off = pz.offered(bm)
        assert b[n]["offered"] == len(off)
        assert b[n]["face"] == sum(v.startswith(fr.FACE_PREFIX) for v in off)
    assert b["seed-san"]["kind"] == "vrm" and b["Hiyori"]["kind"] == "live2d"
    assert b["Hiyori"]["rest_visible"] == {"L": False, "R": False}
    assert b["lafei"]["rest_visible"] == {"L": True, "R": True}
    assert b["Hiyori"]["hand_moves"] == {"bust": 1, "full": 1}
    assert b["seed-san"]["hand_moves"]["bust"] == 0
    assert b["seed-san"]["hand_moves"]["full"] > b["Hiyori"]["hand_moves"][
        "full"]
    assert b["Hiyori"]["frames"] == ["bust", "full"]
    assert b["seed-san"]["frames"] == ["full"]
    json.dumps(b)                       # avatars.json takes it as it is


def test_budget_line_says_what_the_body_can_do():
    hiyori = pz.budget_line(_bm("Hiyori"))
    assert "arms are limited" in hiyori and "rarely" in hiyori
    assert "14% of the places" in hiyori and "2 arm drawings" in hiyori
    assert "16 face controls" in hiyori
    natori = pz.budget_line(_bm("Natori"))
    assert "part of the way" in natori and "up to 2 hand moves" in natori
    seed = pz.budget_line(_bm("seed-san"))
    assert "almost anywhere" in seed and "in full use your body" in seed
    assert "3 hand moves" in seed and "in bust" not in seed   # always full


# ------------------------------------------------------------ the camera

VIEWS = {"bust": (2.0, 0.1, -0.3), "full": (1.0, 0.0, 0.0)}


def test_camera_eases_between_the_two_views():
    cam = fr.Camera(VIEWS, "bust")
    assert cam.at(0.0) == VIEWS["bust"] and cam.at(5.0) == VIEWS["bust"]
    assert cam.commit(2.0, "full") and not cam.commit(2.0, "wide")
    assert cam.at(1.99) == VIEWS["bust"]
    assert cam.at(2.0) == VIEWS["bust"]                 # starts at rest
    assert cam.at(2.0 + fr.VIEW_EASE) == VIEWS["full"]
    assert cam.at(9.0) == VIEWS["full"] and cam.target() == "full"
    mid = cam.at(2.0 + fr.VIEW_EASE / 2)
    assert mid == pytest.approx((1.5, 0.05, -0.15))     # all three move
    s = [cam.at(2.0 + i * 0.05)[0] for i in range(13)]
    assert all(a >= b for a, b in zip(s, s[1:]))        # monotone
    assert (s[0] - s[1]) < (s[5] - s[6])                # slow at the ends


def test_a_switch_back_mid_ease_turns_around_from_where_it_was():
    cam = fr.Camera(VIEWS, "bust")
    cam.commit(1.0, "full")
    here = cam.at(1.3)
    cam.commit(1.3, "bust")
    assert cam.at(1.3) == pytest.approx(here)
    assert cam.at(1.3 + fr.VIEW_EASE + 0.01) == VIEWS["bust"]
    assert 1.0 < cam.at(1.5)[0] < 2.0
    # Pure in t: any order of asks agrees.
    assert cam.at(1.5) == cam.at(1.5)


def test_frame_acts_become_tokens_the_camera_reads():
    acts = [{"t": 1.0, "tag": "frame", "arg": "full"},
            {"t": 0.5, "tag": "pose", "arg": {"targets": {}}},
            {"t": 4.0, "tag": "frame", "arg": "bust"}]
    toks = fr.frame_tokens(acts, 10.0)
    assert [(k["t"], k["view"], k["type"]) for k in toks] == [
        (11.0, "full", "frame"), (14.0, "bust", "frame")]
    cam = fr.Camera(VIEWS, "bust")
    assert fr.view_tokens(toks[::-1], cam) == 2          # sorted by time
    assert cam.at(12.0) == VIEWS["full"] and cam.at(15.0) == VIEWS["bust"]
    assert fr.last_frame(acts, "bust") == "bust"
    assert fr.last_frame([], "full") == "full"
    assert fr.switches(acts, "bust") == 2 and fr.switches(acts, "full") == 1


def test_render_loop_sets_the_view_each_frame_it_changes():
    import threading
    from flow import live as lv

    class Stream:
        def __init__(self):
            self.camera = fr.Camera(VIEWS, "bust")
            self.camera.commit(0.2, "full")
            self.f = 0

        def frame(self):
            self.f += 1
            return {}, {}

    class Enc:
        dropped = 0

        def submit(self, *a):
            pass

    be = lv.FakeBackend(32)
    loop = lv.RenderLoop(Stream(), be, lv.Clock(), Enc(), 30,
                         log=lambda s: None, report=0)
    loop.run(threading.Event(), max_frames=30)
    assert be.views[0] == VIEWS["bust"] and be.views[-1] == VIEWS["full"]
    n_ease = int(fr.VIEW_EASE * 30)
    assert 2 <= len(be.views) <= n_ease + 4              # only on change
    scales = [v[0] for v in be.views]
    assert scales == sorted(scales, reverse=True)


def test_stream_commits_frame_tokens_to_its_camera(tmp_path):
    model, rig_path, bm = _body_rig(tmp_path / "x")
    rig = fs.EventRig(str(model), str(rig_path), "default", None, None, FPS,
                      str(tmp_path), log=lambda s: None)
    stream = fs.MotionStream(rig, seed=0, autonomic=False, log=lambda s: None)
    stream.camera = fr.Camera(VIEWS, "bust")
    stream.direct(fr.frame_tokens([{"t": 0.5, "tag": "frame",
                                    "arg": "full"}]))
    for _ in range(int(2 * FPS)):
        stream.frame()
    assert stream.camera.target() == "full"
    assert stream.camera.at(2.0) == VIEWS["full"]
    assert stream.camera.at(0.4) == VIEWS["bust"]


def test_live_performer_switches_the_camera_and_shows_the_framing(tmp_path):
    from flow import live as lv
    from tests.test_live import FakeClock, SpeakingTTS
    model, rig_path, bm = _body_rig(tmp_path / "x")
    rig = fs.EventRig(str(model), str(rig_path), "default", None, None, FPS,
                      str(tmp_path), log=lambda s: None)
    clock = FakeClock(2.0)
    hub = lv.Hub(clock)
    stream = fs.MotionStream(rig, seed=0, autonomic=False, log=lambda s: None)
    stream.camera = fr.Camera(VIEWS, "bust")
    a = argparse.Namespace(seed=0, lead=0.25, gap=0.12, lipsync="off",
                           mouth_gain=1.0, cjk_rate=dr.CJK_RATE,
                           latin_rate=dr.LATIN_RATE, mood="auto",
                           backend="deepseek", body_prompt="beats",
                           frame="bust")
    run = {"system": "sys", "labels": {}, "offered": None, "mode": "body",
           "body": bm, "core": None, "images": False,
           "max_tag": pz.MAX_MARKER,
           "parser": lambda: pz.parser_for(bm, frames=True)}
    reply = ('Hello there. Let me show you. [frame full] [pose {"head.yaw": '
             '15}] Ta-da! [frame bust] Back to you.')
    logs = []
    p = lv.Performer(a, run, rig, stream, clock, hub, dr.TextBackend(reply),
                     SpeakingTTS(clock), log=logs.append)
    assert p.framing == "bust"
    p.turn("hi")
    assert "(camera: bust" in p.messages[0]["content"]
    toks = [k for k in stream.pending if k["type"] == "frame"]
    assert [k["view"] for k in toks] == ["full", "bust"]
    assert toks[0]["t"] < toks[1]["t"] and toks[0]["duration"] == fr.VIEW_EASE
    assert p.framing == "bust"                    # it switched back
    assert any("camera -> full" in s for s in logs)
    # A turn that ends away from home goes back to it, and shows home to
    # the next turn.
    sent = []                     # the frame tokens as they are directed
    direct = stream.direct
    stream.direct = lambda toks: (
        sent.extend(k for k in toks if k["type"] == "frame"), direct(toks))[1]
    p2 = lv.Performer(a, run, rig, stream, FakeClock(2.0), lv.Hub(clock),
                      dr.TextBackend("One. [frame full] Two."),
                      SpeakingTTS(clock), log=lambda s: None)
    p2.turn("hi")
    p2.messages.append({"role": "assistant", "content": "x"})
    p2.turn("again")
    assert "(camera: bust" in p2.messages[-2]["content"]
    assert p2.framing == "bust"
    assert [k["view"] for k in sent] == ["full", "bust", "full", "bust"]
    assert sent[2]["t"] < sent[3]["t"]            # the return, after the speech
    # Full as home: a switch to bust returns to full.
    a.frame = "full"
    p4 = lv.Performer(a, run, rig, stream, FakeClock(2.0), lv.Hub(clock),
                      dr.TextBackend("One. [frame bust] Two."),
                      SpeakingTTS(clock), log=lambda s: None)
    assert p4.framing == p4.home == "full"
    p4.turn("hi")
    assert p4.framing == "full"
    assert [k["view"] for k in sent[4:]] == ["bust", "full"]
    stream.direct = direct
    a.frame = "bust"
    # Not the beats prompt: no line, no framing.
    a.body_prompt = "v1"
    p3 = lv.Performer(a, run, rig, stream, FakeClock(2.0), lv.Hub(clock),
                      dr.TextBackend("One."), SpeakingTTS(clock),
                      log=lambda s: None)
    assert p3.framing is None


# ------------------------------------------------------- generated data

def test_start_framing_is_seeded_and_configurable():
    ids = [f"Hiyori.cheerful.{k:04d}" for k in range(400)]
    a = [fr.start_frame(0, i) for i in ids]
    assert a == [fr.start_frame(0, i) for i in ids]           # stable
    assert a != [fr.start_frame(1, i) for i in ids]           # by seed
    share = a.count("bust") / len(a)
    assert 0.62 < share < 0.78
    assert {fr.start_frame(0, i, 1.0) for i in ids} == {"bust"}
    assert {fr.start_frame(0, i, 0.0) for i in ids} == {"full"}


def test_plan_sessions_adds_the_start_framing_only_for_beats():
    cfg = ds.load_config()
    base = ds.plan_sessions(["Hiyori"], ["cheerful"], cfg, 5, 2, 3)
    withf = ds.plan_sessions(["Hiyori"], ["cheerful"], cfg, 5, 2, 3, 0.5)
    assert all("frame" not in s for s in base)
    assert [{k: v for k, v in s.items() if k != "frame"} for s in withf] \
        == base                       # ids and seeds untouched
    assert {s["frame"] for s in withf} <= {"bust", "full"}


SWITCH = ('Hi! [pose {"brow.L.lift": 0.5, "head.tilt": 3}] Watch this. '
          '[frame full] [pose {"hand.R.y": 0.2}] Ta-da! [frame bust] Done.')
STAY = '[pose {"mouth.form": 0.5}] Oh? Nice.'


def _gen(tmp_path, replies, *extra, avatars="Hiyori,seed-san", sessions=4,
         turns=3):
    out = tmp_path / "run"
    ds.generate(
        ds.parse_opts(["gen", "--avatars", avatars, "--live2d-lite",
                       "--sessions", str(sessions), "--turns", str(turns),
                       "--out", str(out), "--personas", "cheerful",
                       "--backoff", "0.01", *extra]),
        backend_factory=lambda role, av: CharLLM(replies) if role ==
        "character" else UserLLM(), log=lambda s: None)
    return str(out), ds.iter_records(str(out))


def test_dataset_carries_the_framing_across_turns(tmp_path):
    out, recs = _gen(tmp_path, [SWITCH, STAY, STAY], "--bust-share", "1.0")
    reg = dx.load_registry(out)
    hi = [r for r in recs if r["avatar"]["name"] == "Hiyori"]
    assert hi
    for r in hi:
        if r["turn"] == 1:           # bust -> full -> bust, by the markers
            assert r["framing"] == {"start": "bust", "home": "bust",
                                    "end": "bust", "switches": 2}
    by_s = {}
    for r in sorted(hi, key=lambda r: (r["session"], r["turn"])):
        by_s.setdefault(r["session"], []).append(r)
    for rs in by_s.values():
        for r0, r1 in zip(rs, rs[1:]):
            assert r1["framing"]["start"] == r0["framing"]["home"]
        for r in rs:
            assert r["user_content"].endswith(
                pz.camera_now(_bm("Hiyori"), r["framing"]["start"]))
    # The frame markers are played (not "unplayed"), the registry has the
    # budget and what the manners need.
    assert all(m["played"] is True for r in hi for m in r["markers"]
               if m["tag"] == "frame")
    assert not any(r["unplayed"] for r in hi)
    av = reg["avatars"]["Hiyori"]
    assert av["budget"]["tier"] == "limited" and av["landmarks"]["chest"]
    assert reg["avatars"]["seed-san"]["budget"]["tier"] == "capable"
    # A VRM is full whatever the share, and never takes the marker.
    vrm = [r for r in recs if r["avatar"]["name"] == "seed-san"]
    assert vrm and all(r["framing"]["start"] == "full" == r["framing"]["end"]
                       for r in vrm)
    assert all(m["tag"] != "frame" for r in vrm for m in r["markers"])
    assert all(r["unplayed"].get("frame") for r in vrm if "frame" in
               r["reply_raw"])


def test_dataset_state_follows_the_markers_from_a_full_start(tmp_path):
    out, recs = _gen(tmp_path, ["One. [frame bust] Two.", STAY, STAY],
                     "--bust-share", "0.0", avatars="Hiyori", sessions=2)
    for sid in {r["session"] for r in recs}:
        rs = sorted((r for r in recs if r["session"] == sid),
                    key=lambda r: r["turn"])
        assert rs[0]["framing"] == {"start": "full", "home": "full",
                                    "end": "bust", "switches": 1}
        assert "(camera: full" in rs[0]["user_content"]
        # The switch lasted for its reply: the next turn is home again.
        assert "(camera: full" in rs[1]["user_content"]
        assert rs[1]["framing"]["start"] == "full"


def test_dataset_turns_start_at_home_after_a_switch_away(tmp_path):
    for share, home, other in (("1.0", "bust", "full"),
                               ("0.0", "full", "bust")):
        out, recs = _gen(tmp_path / share, [f"One. [frame {other}] Two.",
                                            STAY, STAY],
                         "--bust-share", share, avatars="Hiyori",
                         sessions=2, turns=3)
        for sid in {r["session"] for r in recs}:
            rs = sorted((r for r in recs if r["session"] == sid),
                        key=lambda r: r["turn"])
            assert [r["framing"]["start"] for r in rs] == [home] * 3
            assert rs[0]["framing"]["end"] == other
            assert rs[0]["framing"]["home"] == home
            assert rs[1]["framing"]["end"] == home        # no marker: home
            for r in rs:
                assert r["user_content"].endswith(
                    pz.camera_now(_bm("Hiyori"), home))


def test_v1_data_has_no_framing_and_no_camera_line(tmp_path):
    out, recs = _gen(tmp_path, [STAY], "--teacher-prompt", "v1",
                     avatars="Hiyori", sessions=1, turns=2)
    assert all("framing" not in r and "(camera" not in r["user_content"]
               for r in recs)
    assert dx.default_max_hands(out) is None


# ------------------------------------------------------------- manners

HANDS = ('Hi! [pose {"hand.R.y": 0.2, "head.tilt": 5}] Look at this. '
         '[pose {"hand.L.y": 0.1, "mouth.form": 0.5}] And that. '
         '[pose {"hand.R.y": 0.3, "shoulders.up": 1}] And this. Bye.')


def _hands(c):
    return c.get("hand_after", 0)


def test_manners_without_max_hands_strip_unasked_hands_as_before():
    new = dx.manners_reply(HANDS, False, 5, 0)
    assert "hand." not in new["reply"]
    assert new["reply"] == dx.manners_reply(HANDS, False, 5, 0, None, "bust",
                                            None)["reply"]


def test_manners_keep_hands_up_to_the_cap_when_not_asked():
    r = dx.manners_reply(HANDS, False, 5, 0, max_hands=1)
    assert r["skipped"] is None
    text = r["reply"]
    assert text.count('"hand.') == 1 and '"hand.R.y": 0.2' in text
    assert "head.tilt" in text and "mouth.form" in text    # faces stay
    assert "shoulders" not in text                          # still stripped
    assert r["c"]["hands_capped"] == 2
    two = dx.manners_reply(HANDS, False, 5, 0, max_hands=2)["reply"]
    assert two.count('"hand.') == 2
    # A capped marker left with only face targets is a beat again: it is
    # counted with the beats, not against the pose cap.
    mixed = ('[pose {"hand.R.y": 0.2, "mouth.form": 0.5}] a [pose '
             '{"hand.L.y": 0.1, "mouth.form": 0.4}] b [pose {"brow.L.lift": '
             '0.5}] c [pose {"eye.L.smile": 0.8}] d')
    kept = dx.manners_reply(mixed, False, 2, 9, max_hands=1)["reply"]
    assert dx.beat_counts(kept)["beats"] == 3 and '"hand.L' not in kept
    assert kept.count("[pose") == 4
    none = dx.manners_reply(HANDS, False, 5, 0, max_hands=0)["reply"]
    assert '"hand.' not in none
    # Asked: nothing is capped.
    asked = dx.manners_reply(HANDS, True, 5, 0, max_hands=1)["reply"]
    assert asked.count('"hand.') == 3


def test_manners_cap_counts_moves_and_gestures_as_hand_moves():
    text = ('Yay! [move {"name": "clap", "beats": 2}] And [pose {"hand.R.y"'
            ': 0.2}] this [move {"name": "head_bob"}] ok.')
    r = dx.manners_reply(text, False, 5, 0, max_hands=1)["reply"]
    assert '"clap"' in r and '"hand.R.y"' not in r and "head_bob" in r


def _view(name="Hiyori"):
    bm = _bm(name)
    return {"landmarks": dict(bm.landmarks), "window": fr.window()}


def test_manners_in_bust_strip_only_hand_targets_out_of_the_picture():
    low = '[pose {"hand.R.y": -2.0, "hand.R.x": -0.8, "head.tilt": 4}] Hm.'
    high = '[pose {"hand.R.y": "@shoulder.y", "head.tilt": 4}] Hm.'
    v = _view()
    bust_low = dx.manners_reply(low, False, 5, 0, 1, "bust", v)
    assert '"hand.' not in bust_low["reply"] and "head.tilt" in \
        bust_low["reply"] and bust_low["c"]["hands_hidden"] == 2
    assert '"hand.R.y"' in dx.manners_reply(high, False, 5, 0, 1, "bust",
                                            v)["reply"]
    assert '"hand.R.y"' in dx.manners_reply(low, False, 5, 0, 1, "full",
                                            v)["reply"]
    # Asked or not, a hand nobody sees is not learned in bust.
    assert '"hand.' not in dx.manners_reply(low, True, 5, 0, 1, "bust",
                                            v)["reply"]
    # A [frame full] earlier in the reply makes the same hand visible.
    seq = '[frame full] ' + low
    assert '"hand.R.y"' in dx.manners_reply(seq, False, 5, 0, 1, "bust",
                                            v)["reply"]
    seq = '[frame bust] ' + low
    assert '"hand.R.y"' not in dx.manners_reply(seq, False, 5, 0, 1, "full",
                                                v)["reply"]
    # A hand left out of the marker is where it rests: out of the picture.
    only_x = '[pose {"hand.R.x": -0.5, "head.tilt": 2}] Hm.'
    assert '"hand.R.x"' not in dx.manners_reply(only_x, False, 5, 0, 1,
                                                "bust", v)["reply"]


def test_hidden_sides_use_the_marker_the_landmarks_and_the_window():
    v = _view()
    assert dx.hidden_sides({"hand.R.y": -2.0}, v) == {"R"}
    assert dx.hidden_sides({"hand.R.y": -0.5, "hand.L.y": -1.9}, v) == {"L"}
    assert dx.hidden_sides({"hand.R.y": "@nowhere"}, v) == set()
    assert dx.hidden_sides({"head.tilt": 3}, v) == set()
    assert dx.hidden_sides({"hand.R.y": -2.0}, {}) == set()


def test_beats_data_defaults_to_the_cap_and_exports_it(tmp_path):
    out, recs = _gen(tmp_path, [HANDS], "--bust-share", "0.0",
                     avatars="Hiyori", sessions=1, turns=1)
    assert dx.default_max_hands(out) == dx.MAX_HANDS == 1
    views = dx.views_of(out)
    assert set(views["Hiyori"]) == {"landmarks", "window"}
    notes = dx.annotate(recs)
    _m, c = dx.manners(recs, None, 5, None, 9, 1, views)
    assert c["max_hands"] == 1
    assert "1 hand-moving markers" in dx.manners_lines(c)[0]
    assert "capped at 1" in dx.manners_lines(c)[1]
    # r1-style: no cap, same as before.
    _m0, c0 = dx.manners(recs, None, 5, None, 9)
    assert c0["max_hands"] is None and "hand-moving" not in \
        "".join(dx.manners_lines(c0))


# ----------------------------------------------------------------- stats

def test_stats_report_framing_and_budget(tmp_path, capsys):
    out, recs = _gen(tmp_path, [SWITCH, STAY, STAY], "--bust-share", "0.5",
                     avatars="Hiyori,seed-san", sessions=6, turns=3)
    js = tmp_path / "s.json"
    assert dx.main(["stats", out, "--keep-lite", "--json", str(js)]) == 0
    text = capsys.readouterr().out
    assert "framing (by the framing a turn starts in):" in text
    assert "motion budget" in text and "limited" in text and "capable" in text
    s = json.loads(js.read_text())
    f = s["framing"]
    assert set(f["by_avatar"]["Hiyori"]) <= {"bust", "full"}
    tot = f["total"]
    n = sum(v["turns"] for v in tot.values())
    assert 0 < n <= len(recs)           # the kept turns (dups drop)
    assert sum(v.get("frame_markers", 0) for v in tot.values()) > 0
    assert sum(v.get("switches", 0) for v in tot.values()) > 0
    assert s["budget"]["Hiyori"]["tier"] == "limited"
    # seed-san never frames: always full.
    assert set(f["by_avatar"]["seed-san"]) == {"full"}


def test_frame_counts_against_the_home_framing():
    # Home bust: away once, back once; the turn was spent (partly) away.
    c = dx.frame_counts("A. [frame full] B. [frame bust] C.", "bust", "bust")
    assert (c["away"], c["returns"], c["turns_away"], c["switches"]) \
        == (1, 1, 1, 2)
    c = dx.frame_counts("A. [frame full] B.", "bust", "bust")
    assert (c["away"], c["returns"], c["turns_away"]) == (1, 0, 1)
    c = dx.frame_counts("A. B.", "bust", "bust")
    assert (c["away"], c["returns"], c["turns_away"]) == (0, 0, 0)
    # An old record that started away from home (the framing was carried).
    c = dx.frame_counts("A. [frame bust] B.", "full", "bust")
    assert (c["away"], c["returns"], c["turns_away"]) == (0, 1, 1)
    c = dx.frame_counts("A. B.", "full", "bust")
    assert c["turns_away"] == 1
    # Without a home it is the start (home framing data).
    assert dx.frame_counts("A. [frame full] B.", "bust")["away"] == 1


def _frec(i, turn, start, text, sec, home="bust"):
    r = {"id": f"s/t{turn:02d}", "session": "s", "turn": turn,
         "avatar": {"name": "A"}, "reply_raw": text, "refused": False,
         "timing": {"speech_seconds": sec},
         "framing": {"start": start, "end": start, "switches": 0}}
    if home:
        r["framing"]["home"] = home
    return r


def test_framing_stats_have_home_counts_and_beats_per_second():
    b = '[pose {"mouth.form": 0.5}] Hi. [pose {"brow.L.lift": 0.5}] There.'
    recs = [_frec(0, 1, "bust", b + " [frame full] Yes.", 4.0),
            _frec(0, 2, "bust", b + " [frame full] Yes. [frame bust] Back.",
                  6.0),
            _frec(0, 3, "bust", b, 2.0)]
    f = dx.framing(recs)["total"]["bust"]
    assert (f["turns"], f["away"], f["returns"], f["turns_away"]) \
        == (3, 2, 1, 2)
    assert f["speech_seconds"] == 12.0 and f["timed_beats"] == 6
    text = "\n".join(dx.framing_lines(dx.framing(recs)))
    assert "2 away from home, 1 back" in text
    assert "2 turns away from home" in text and "beats/s 0.50" in text
    # Records from before the home framing: the first turn's start is home.
    old = [_frec(0, 1, "bust", b + " [frame full] Yes.", 4.0, home=None),
           _frec(0, 2, "full", b, 3.0, home=None)]
    fo = dx.framing(old)["total"]
    assert fo["bust"]["away"] == 1 and fo["bust"]["turns_away"] == 1
    assert fo["full"]["turns_away"] == 1 and fo["full"]["away"] == 0


def test_r1_stats_read_as_before(tmp_path, capsys):
    out, recs = _gen(tmp_path, [STAY], "--teacher-prompt", "v1",
                     avatars="Hiyori", sessions=1, turns=2)
    assert dx.main(["stats", out]) == 0
    text = capsys.readouterr().out
    assert "framing (by" not in text and "hand-moving" not in text


# ------------------------------------------------------- offline director

def _director(tmp_path, reply_text, *extra):
    model, rig, bm = _body_rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text(reply_text, encoding="utf-8")
    out = tmp_path / "run"
    args = ["--model-json", str(model), "--rig", str(rig), "--backend",
            "file", "--reply-file", str(reply), "--out", str(out),
            "--probe-dir", str(tmp_path), "--seed", "2", "--body-targets",
            *extra]
    return args, out


def test_director_plans_frame_markers_into_tokens(tmp_path, capsys):
    args, out = _director(
        tmp_path, 'Hi! [pose {"head.yaw": 10}] Watch this. [frame full] Ta-da.'
        ' [frame bust] Done.', "--body-prompt", "beats", "--frame", "bust")
    assert dr.main(args) == 0
    log = capsys.readouterr().out
    acts = json.loads((out / "actions.json").read_text())["actions"]
    assert [x["arg"] for x in acts if x["tag"] == "frame"] == ["full", "bust"]
    blob = json.loads((out / "tokens.json").read_text())
    ft = [k for k in blob["tokens"] if k["type"] == "frame"]
    assert [k["view"] for k in ft] == ["full", "bust"]
    assert 0 < ft[0]["t"] < ft[1]["t"] and ft[0]["duration"] == fr.VIEW_EASE
    # The render replays them: it starts in the framing asked for.
    cmd = [ln for ln in log.splitlines() if "--tokens-in" in ln][0]
    assert "--frame bust" in cmd
    # A v1 run drops the marker like any unknown tag.
    args, out = _director(tmp_path / "v1", 'Hi! [frame full] Ta-da.')
    assert dr.main(args) == 0
    assert "not a body-mode marker" in capsys.readouterr().out
    assert not [k for k in json.loads((out / "tokens.json").read_text())[
        "tokens"] if k["type"] == "frame"]


def test_director_chat_starts_every_turn_in_the_home_framing(tmp_path,
                                                            monkeypatch):
    seen = []

    class Say(dr.TextBackend):
        replies = ["One. [frame full] Two.", "Three.", "Four. [frame bust]"]

        def stream(self, system, messages):
            seen.append(messages[-1]["content"])
            self.text = self.replies[len(seen) - 1]
            return super().stream(system, messages)
    monkeypatch.setattr(dr, "make_backend", lambda a, log=print: Say(""))
    lines = iter(["again", "and again"])

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None
    monkeypatch.setattr("builtins.input", fake_input)
    model, rig, bm = _body_rig(tmp_path / "x")
    args = ["--model-json", str(model), "--rig", str(rig), "--backend",
            "deepseek", "--out", str(tmp_path / "run"), "--probe-dir",
            str(tmp_path), "--seed", "2", "--body-targets", "--body-prompt",
            "beats", "--chat", "--say", "hi", "--frame", "bust"]
    assert dr.main(args) == 0
    tail = [c.splitlines()[-1] for c in seen]
    assert tail[0].startswith("(camera: bust")
    assert tail[1].startswith("(camera: bust")        # home again after full
    assert tail[2].startswith("(camera: bust")
