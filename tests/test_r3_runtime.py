"""
The live runtime's fixes after the first trial of tsk-q8r2 and the r3 data
that goes with them: the history window, the head staying where a beat put
it (and a short reaction yaw not staying), the expression baseline, talking
while the chat is quiet (live and the dataset share the line), and the
beats2 sentences about it. No GL, no network, a fake clock.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np

from flow import dataexport as dx
from flow import dataset as ds
from flow import director as dr
from flow import pose as pz
from flow import stream as fs
from tests import test_dataset as td
from tests.test_live import _events, _rig

FPS = 30.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------- the layer

def _tok(t, params, dur=0.25, hold=0.5, release=0.5, **kw):
    return {"t": t, "type": "pose", "role": "pose", "params": params,
            "parts": {}, "parts_from": {}, "step": [], "duration": dur,
            "hold": hold, "release": release, "groups": [],
            "variant": "v", "src": "director", **kw}


def _run(toks, secs, baseline=0.0, under=None, ease=True):
    lay = pz.PoseLayer(FPS, {"H": 0.0, "T": 0.0, "F": 0.0}, ease=ease,
                       angles={"H", "T"}, baseline=baseline, life=0.0,
                       settle=0.0)
    for k in toks:
        lay.commit(k)
    rows = []
    for f in range(int(secs * FPS)):
        over = dict(under(f) if under else {"H": 0.0, "T": 0.0, "F": 0.0})
        if lay.has_base():
            over = lay.base_apply(f, over)
        over, _ = lay.apply(f, over, None)
        rows.append((over["H"], over["T"], over["F"]))
    return np.array(rows)


def at(rows, t):
    return rows[int(round(t * FPS))]


def test_a_tilt_stays_until_the_next_one_and_blends_straight_across():
    toks = [_tok(1.0, {"T": 7.0}, carry=["T"]),
            _tok(4.0, {"T": -6.0}, carry=["T"]),
            pz.release_token(8.0)]
    rows = _run(toks, 12)
    # held far past its 0.5 s hold, not back at centre between the beats
    assert abs(at(rows, 3.5)[1] - 7.0) < 1e-6
    # blends from +7 to -6 (never through the centre and back)
    seg = rows[int(4.0 * FPS):int(4.8 * FPS), 1]
    assert (np.diff(seg) <= 1e-9).all() and seg.min() > -6.01
    assert abs(at(rows, 7.0)[1] + 6.0) < 1e-6
    # the end of the reply: relaxed slowly (over at least a second)
    assert at(rows, 8.5)[1] < -3.0
    assert abs(at(rows, 11.0)[1]) < 1e-6


def test_without_the_carry_field_a_beat_releases_as_before():
    rows = _run([_tok(1.0, {"T": 7.0})], 6)
    assert abs(at(rows, 4.0)[1]) < 1e-6


def test_no_end_token_the_head_relaxes_after_a_long_silence():
    rows = _run([_tok(1.0, {"T": 7.0}, carry=["T"])], 30)
    assert abs(at(rows, 15.0)[1] - 7.0) < 1e-6
    assert abs(at(rows, 1.0 + pz.CARRY_MAX + 3.0)[1]) < 1e-6


def test_the_end_token_releases_over_at_least_a_second():
    rel = pz.release_token(5.0, release=0.1)
    assert rel["release"] >= 1.0 and rel["end"] and not rel["params"]
    rows = _run([_tok(1.0, {"H": 20.0}, hold=3.0, carry=["H"]), rel], 9)
    v = rows[:, 0]
    assert abs(at(rows, 4.9)[0] - 20.0) < 1e-6
    assert np.abs(np.diff(v[int(5.0 * FPS):])).max() < 20.0 / (
        1.0 * FPS) * 2.5                       # no snap
    assert at(rows, 5.5)[0] > 5.0 and abs(at(rows, 7.0)[0]) < 1e-6


def _rec(start, res, hold=0.5, dur=0.25, **kw):
    P = {"head.yaw": "H", "head.pitch": "P", "head.tilt": "T",
         "gaze.x": "G", "brow.L.lift": "B", "mouth.form": "M"}
    return {"start": start, "dur": dur, "hold": hold, "release": 0.5,
            "params": {P[k]: v for k, v in res.items()}, "parts": {},
            "parts_from": {}, "step": [], "variant_name": "v",
            "resolved": res, **kw}


ROLES = {"head_x": ("H", 1.0, ""), "head_y": ("P", 1.0, ""),
         "head_z": ("T", 1.0, ""), "gaze_x": ("G", 1.0, ""),
         "brow_l": ("B", 1.0, "")}


def test_a_short_reaction_yaw_relaxes_and_a_look_persists():
    tk = lambda *a, **k: pz.tokens([_rec(*a, **k)], ROLES, (),  # noqa: E731
                                   carry=True)[0]
    # the copied opening: yaw -5, held 0.5 -> tilt/pitch carry, yaw does not
    t = tk(0.4, {"head.yaw": -5, "head.pitch": 5, "head.tilt": 3})
    assert t["carry"] == ["P", "T"]
    # edge: 6 degrees and 0.6 s still a reaction; more of either is a look
    assert "H" not in tk(0.4, {"head.yaw": 6}, hold=0.6).get("carry", ())
    assert tk(0.4, {"head.yaw": 7}, hold=0.5)["carry"] == ["H"]
    assert tk(0.4, {"head.yaw": 4}, hold=0.7)["carry"] == ["H"]
    assert tk(0.4, {"head.yaw": -20}, hold=3.0)["carry"] == ["H"]
    # gaze: a glance relaxes, a look stays
    assert "carry" not in tk(0.4, {"gaze.x": 0.3}, hold=0.5)
    assert tk(0.4, {"gaze.x": 0.8}, hold=2.0)["carry"] == ["G"]
    # face params are not angles; no fields without the flags
    assert "carry" not in tk(0.4, {"brow.L.lift": 0.7})
    plain = pz.tokens([_rec(0.4, {"head.yaw": 20}, hold=2)], ROLES, ())[0]
    assert "carry" not in plain and "base" not in plain


def test_reaction_relaxes_in_playback_while_a_look_stays_to_the_end():
    react = pz.tokens([_rec(0.5, {"head.yaw": -5, "head.tilt": 4})], ROLES,
                      (), carry=True)
    rows = _run(react + [pz.release_token(8.0)], 10)
    assert abs(at(rows, 4.0)[0]) < 1e-6          # yaw relaxed after its hold
    assert abs(at(rows, 4.0)[1] - 4.0) < 1e-6    # the tilt stayed
    look = pz.tokens([_rec(0.5, {"head.yaw": 25}, hold=3.0, dur=0.4)], ROLES,
                     (), carry=True)
    rows = _run(look + [pz.release_token(8.0)], 11)
    assert abs(at(rows, 6.0)[0] - 25.0) < 1e-6   # past the 3 s hold
    assert abs(at(rows, 10.5)[0]) < 1e-6         # relaxed after the reply


def test_the_dataset_and_the_action_head_play_what_they_did():
    recs = [_rec(0.5, {"head.yaw": 20, "head.tilt": 5, "mouth.form": 0.8},
                 hold=2.0)]
    assert pz.tokens(recs, ROLES, ()) == pz.tokens(recs, ROLES, (),
                                                   carry=False,
                                                   baseline=False)
    lay = pz.layer_for(pz.tokens(recs, ROLES, ()), FPS, {"H": 0.0, "T": 0.0})
    assert lay.baseline == 0 and not lay.carried and not lay.off


# ------------------------------------------------------------ baseline

def _face(valence=0.0, **kw):
    return _tok(1.0, {"F": 0.8}, dur=0.2, hold=0.5, base=["F"],
                valence=valence, **kw)


def test_a_face_beat_leaves_a_resting_share_that_decays():
    rows = _run([_face(), pz.release_token(6.0)], 20, baseline=1.0)
    f = rows[:, 2]
    assert abs(f[int(1.4 * FPS)] - 0.8) < 0.1
    # after the beat's release the face keeps about 37.5 % of it ...
    rest = f[int(3.0 * FPS)]
    assert 0.2 < rest < 0.4 and abs(rest - 0.8 * pz.BASE_FRAC) < 0.03
    # ... until the reply ends, then fades over BASE_DECAY seconds
    assert abs(f[int(5.9 * FPS)] - rest) < 0.02
    assert 0 < f[int((6.0 + pz.BASE_DECAY / 2) * FPS)] < rest
    assert abs(f[int((6.0 + pz.BASE_DECAY + 0.2) * FPS)]) < 1e-6
    # no end token: held BASE_HOLD_MAX, then faded
    rows = _run([_face()], 30, baseline=1.0)
    assert rows[int(1.0 * FPS + 3 * FPS), 2] > 0.2
    assert abs(rows[int((1.5 + pz.BASE_HOLD_MAX + pz.BASE_DECAY + 1) * FPS),
                    2]) < 1e-6


def test_baseline_scale_zero_is_off_and_scales_the_rest():
    base = _run([_face()], 12)
    off = _run([_face()], 12, baseline=0.0)
    np.testing.assert_array_equal(base, off)
    assert abs(base[int(3.0 * FPS), 2]) < 1e-6        # blank between beats
    half = _run([_face()], 12, baseline=0.5)[int(3.0 * FPS), 2]
    full = _run([_face()], 12, baseline=1.0)[int(3.0 * FPS), 2]
    assert abs(half - full / 2) < 0.03


def test_a_new_beat_replaces_the_baseline_from_where_it_is():
    toks = [_face(), _tok(4.0, {"F": -0.5}, hold=0.5, base=["F"],
                          valence=-0.4)]
    rows = _run(toks, 12, baseline=1.0)
    f = rows[:, 2]
    k = int(4.0 * FPS)
    assert abs(f[k + 1] - f[k - 1]) < 0.1            # starts from the rest
    assert 0.2 < f[k - 1] < 0.4
    assert f[int(6.0 * FPS)] < 0 < f[int(3.0 * FPS)]


def test_a_beat_that_agrees_with_the_replys_mood_keeps_more():
    def rest(v_first, v_second):
        toks = [_tok(1.0, {"F": 0.8}, hold=0.4, base=["F"],
                     valence=v_first),
                _tok(3.0, {"F": 0.8}, hold=0.4, base=["F"],
                     valence=v_second)]
        return _run(toks, 12, baseline=1.0)[int(6.0 * FPS), 2]
    assert rest(0.8, 0.8) > rest(0.8, -0.8)


def test_a_face_beat_marks_the_face_and_not_angles_hands_or_openings():
    roles = dict(ROLES, body_x=("BX", 1.0, ""))
    rec = _rec(1.0, {"head.yaw": 3, "brow.L.lift": 0.7, "mouth.form": 0.5})
    rec["params"].update({"BX": 2.0, "ParamArmLA": 1.0, "ParamMouthOpenY": 1,
                          "ParamEyeLOpen": 0.0})
    rec["step"] = []
    t = pz.tokens([rec], roles, {"ParamArmLA"}, baseline=True)[0]
    assert t["base"] == ["B", "M"] and -1 <= t["valence"] <= 1
    assert pz.valence(_rec(0, {"mouth.form": 0.8, "brow.L.lift": 0.5})) > 0
    assert pz.valence(_rec(0, {"mouth.form": -0.6, "brow.L.lift": -0.6})) < 0
    assert pz.valence(_rec(0, {"head.tilt": 4})) == 0


def test_the_stream_with_baseline_off_is_the_stream_of_before(tmp_path):
    rig = _rig(tmp_path)
    pids = [rig.pids[i] for i in rig.expressive]
    brow = next(p for p in pids if "brow" in p.lower() or "Brow" in p)
    toks = [_tok(1.0, {brow: 0.8}, base=[brow], valence=0.5),
            pz.release_token(3.0)]

    def frames(**kw):
        st = fs.MotionStream(rig, seed=1, autonomic=False,
                             log=lambda s: None, **kw)
        st.direct(copy.deepcopy(toks))
        return [st.frame()[0].get(brow) for _ in range(300)]
    a, b = frames(), frames(baseline_scale=0.0)
    assert a == b
    c = frames(baseline_scale=1.0)
    assert c != a and abs(c[int(2.5 * FPS)] - a[int(2.5 * FPS)]) > 0.05


# ------------------------------------------------------ the performer

class Recorder:
    """A backend that records the messages of every request."""

    stop_reason, usage = "end_turn", {}

    def __init__(self, reply="Okay. Right."):
        self.reply, self.sent = reply, []

    def stream(self, system, messages):
        self.sent.append(copy.deepcopy(messages))
        yield self.reply


def _perf(tmp_path, **kw):
    p, hub, stream, clock, tts = td_perf(tmp_path)
    for k, v in kw.items():
        setattr(p, k, v)
    return p, hub, stream, clock, tts


def td_perf(tmp_path):
    from tests.test_live import _performer
    p, hub, stream, clock, tts = _performer(tmp_path, "Okay. Right.")
    p.backend = Recorder()
    p.log = lambda s: None
    return p, hub, stream, clock, tts


def test_the_model_sees_only_the_last_k_exchanges(tmp_path):
    p, hub, *_ = td_perf(tmp_path)
    p.history_turns = 2
    for i in range(8):
        p.turn(f"line {i}")
    sent = p.backend.sent
    assert [len(m) for m in sent] == [1, 3, 5, 5, 5, 5, 5, 5]
    for i, m in enumerate(sent):
        assert m[0]["role"] == "user" and m[-1]["content"] == f"line {i}"
        assert all(a["role"] != b["role"] for a, b in zip(m, m[1:]))
    assert sent[7][0]["content"] == "line 5"
    logs = [e["text"] for e in _events(hub, "log")
            if "history trimmed" in e["text"]]
    assert len(logs) == 5 and "last 2 exchanges" in logs[0]


def test_history_turns_five_is_a_request_of_at_most_six_turns(tmp_path):
    p, *_ = td_perf(tmp_path)
    p.history_turns = 5
    for i in range(12):
        p.turn(f"line {i}")
    assert max(len(m) for m in p.backend.sent) == 11
    p2, *_ = td_perf(tmp_path / "b")
    p2.history_turns = 0                             # the whole conversation
    for i in range(12):
        p2.turn(f"line {i}")
    assert [len(m) for m in p2.backend.sent][-1] == 23


# ------------------------------------------------------------- quiet

def _quiet_perf(tmp_path, after=20.0, cap=3):
    p, hub, stream, clock, tts = td_perf(tmp_path)
    p.quiet_after, p.quiet_max = after, cap
    return p, hub, clock


def test_a_quiet_turn_is_due_after_the_silence_and_not_before(tmp_path):
    p, hub, clock = _quiet_perf(tmp_path)
    assert not p.quiet_due()                         # nothing said yet
    p.turn("Hello there")
    p.quiet_t0 = p.next_frame / FPS                  # its speech ends
    end = p.quiet_t0
    clock.t = end + 19.0
    assert not p.quiet_due()
    clock.t = end + 20.0
    assert p.quiet_due()
    p.say("hi")                                      # a message waits
    assert not p.quiet_due()
    p.inbox.get()
    p.quiet_n = 0
    p.busy = True
    assert not p.quiet_due()                         # one is being spoken
    p.busy = False
    p.quiet_after = 0
    assert not p.quiet_due()                         # --quiet-after 0: off


def test_quiet_turns_are_capped_and_a_real_message_resets_the_count(tmp_path):
    p, hub, clock = _quiet_perf(tmp_path)
    p.turn("Hello there")
    sent = []
    for k in range(5):
        p.quiet_t0 = p.next_frame / FPS
        clock.t = p.quiet_t0 + 21.0
        if not p.quiet_due():
            break
        text, quiet = p.next_turn(poll=0.01)
        assert quiet and text == pz.QUIET_CHAT
        p.turn(text, True)
        sent.append(p.backend.sent[-1][-1]["content"])
    assert k == 3 and len(sent) == 3 and p.quiet_n == 3
    assert not p.quiet_due()
    # waiting on the inbox now: a real message comes through, resets
    p.say("I am back")
    assert p.quiet_n == 0
    text, quiet = p.next_turn(poll=0.01)
    assert (text, quiet) == ("I am back", False)
    p.turn(text)
    p.quiet_t0 = p.next_frame / FPS
    clock.t = p.quiet_t0 + 25
    assert p.quiet_due()


def test_a_quiet_turn_keeps_the_conversation_language_and_logs(tmp_path):
    p, hub, clock = _quiet_perf(tmp_path)
    p.turn("こんにちは、元気？")
    assert p.lang == "ja"
    p.quiet_t0 = p.next_frame / FPS
    clock.t = p.quiet_t0 + 30
    p.turn(pz.QUIET_CHAT, True)
    assert p.lang == "ja" and p.last_user == "こんにちは、元気？"
    assert any("chat quiet" in e["text"] for e in _events(hub, "log"))
    assert _events(hub, "user")[-1].get("quiet") is True
    assert _events(hub, "reply")[-1]["quiet"] is True


def test_the_quiet_turn_in_body_mode_has_the_tail_in_the_chats_language():
    bm = td_body()
    for prompt in ("beats", "beats2"):
        line, lang_text = pz.quiet_user("こんにちは、元気？")
        assert line == pz.QUIET_CHAT
        tail = pz.turn_tail(bm, "bust", line, prompt, lang_text=lang_text)
        assert tail.startswith("\n(camera: ")
        assert tail.endswith("\n(reply in: Japanese)") == (prompt == "beats2")
    # the real turn is as it was, "OK!" settles nothing, the default holds
    assert pz.turn_tail(bm, "bust", "Hello there!", "beats2").endswith(
        "(reply in: English)")
    assert pz.turn_tail(bm, "bust", "OK!", "beats2", lang_text=None) == \
        pz.turn_tail(bm, "bust", "OK!", "beats2")
    assert pz.turn_tail(bm, "bust", "x", "v1", lang_text="y") == ""


def test_live_and_the_dataset_send_the_same_quiet_user_content(tmp_path):
    out = tmp_path / "q"
    made = {}
    res = ds.generate(
        td._opts(out, "--quiet-share", "1", sessions=2, turns=4,
                 avatars="seed-san"),
        backend_factory=td._factory(made), log=lambda s: None)
    assert res["ok"] == 2
    recs = ds.iter_records(str(out))
    by_s = {}
    for r in recs:
        by_s.setdefault(r["session"], []).append(r)
    for rs in by_s.values():
        rs.sort(key=lambda r: r["turn"])
        assert "quiet" not in rs[0] and rs[0]["user"] != pz.QUIET_CHAT
        for prev, r in zip(rs, rs[1:]):
            assert r["quiet"] is True and r["topic"] == "quiet"
            assert r["user"] == pz.QUIET_CHAT
            assert r["user_content"].startswith(pz.QUIET_CHAT + "\n")
            # the language of the last real line, the tail live writes
            assert r["lang"]["user"] == ds.detect_lang(rs[0]["user"])
            assert r["lang"]["mix"] is None
            assert r["lang"]["asked"] == prev["lang"]["asked"]
    # the user simulator was only called for the first turn of each session
    assert sum(len(u.sent) for u in made["user"]) == 2
    # the history the model saw carries the quiet turns as they were
    r = by_s[sorted(by_s)[0]][-1]
    assert r["history"][-1]["user"] == pz.QUIET_CHAT


def test_dataset_quiet_share_is_from_its_own_generator():
    sess = {"seed": 123, "id": "x"}
    opts = argparse.Namespace(teacher_prompt="beats2", quiet_share=None)
    hits = sum(ds.quiet_turn(dict(sess, seed=s), t, opts)
               for s in range(400) for t in range(2, 6))
    assert 0.06 < hits / 1600 < 0.14                 # 10 %
    assert not any(ds.quiet_turn(dict(sess, seed=s), 1,
                                 argparse.Namespace(teacher_prompt="beats2",
                                                    quiet_share=1.0))
                   for s in range(50))              # never the first turn
    for p in ("beats", "v1"):                        # r1 / r2 regenerate
        assert not ds.quiet_turn(sess, 3, argparse.Namespace(
            teacher_prompt=p, quiet_share=1.0))
    assert ds.quiet_turn(sess, 3, argparse.Namespace(
        teacher_prompt="beats2", quiet_share=1.0))
    assert not ds.quiet_turn(sess, 3, argparse.Namespace(
        teacher_prompt="beats2", quiet_share=0.0))


def test_a_beats_run_draws_what_it_drew(tmp_path):
    # --quiet-share changes nothing for the prompts of r1 and r2
    res = []
    for share in ("0", "1"):
        out = tmp_path / share
        made = {}
        ds.generate(td._opts(out, "--teacher-prompt", "beats",
                             "--quiet-share", share, sessions=2, turns=3,
                             avatars="seed-san"),
                    backend_factory=td._factory(made), log=lambda s: None)
        res.append(sorted((r["id"], r["user"], r["topic"]) for r in
                          ds.iter_records(str(out))))
    assert res[0] == res[1] and all("quiet" not in t for _, _, t in res[0])


def test_export_keeps_quiet_turns(tmp_path):
    out = tmp_path / "q"
    made = {}
    ds.generate(td._opts(out, "--quiet-share", "1", sessions=2, turns=4,
                         avatars="seed-san"),
                backend_factory=td._factory(made), log=lambda s: None)
    recs = ds.iter_records(str(out))
    quiet = [r for r in recs if r.get("quiet")]
    assert len(quiet) == 6
    assert dx.dedupe(recs) == set() or all(
        not r.get("quiet") or r["id"] not in dx.dedupe(recs) for r in recs)
    for r in quiet:
        assert dx.asked_for(r) == (False, None)
    notes = dx.annotate(recs, lang_filter=True)
    for r in quiet:
        flags, kept = notes[r["id"]]
        assert "dup" not in flags and "language" not in flags
    assert dx.dedupe(quiet) == set()                 # identical lines, kept


# ----------------------------------------------------------- the prompt

def test_beats2_says_what_quiet_chat_and_a_look_mean_and_beats_does_not():
    from tests.test_facevocab import PINNED, _digest
    bm = td_body()
    b2 = pz.system_prompt(bm, prompt="beats2")
    assert f'"{pz.QUIET_CHAT}"' in b2
    assert "keep talking as a streamer does when the chat is quiet" in b2
    assert "ask the viewers a question" in b2
    assert "never say that the chat is quiet" in b2
    assert "Your head stays at the angle a marker gave it until your next " \
        "marker sets that angle again" in b2
    assert "asks you to look or turn somewhere" in b2
    assert '"hold" 2 to 4' in b2
    b1 = pz.system_prompt(bm, prompt="beats")
    assert pz.QUIET_CHAT not in b1 and "asks you to look" not in b1
    assert "head stays at the angle" not in b1
    assert (_digest(bm, "v1"), _digest(bm, "beats")) == PINNED["Hiyori"]


def td_body():
    from flow import body as fb
    return fb.load(td.HIYORI)


# ------------------------------------------- the body-mode performer

def _body_performer(tmp_path, reply, **opts):
    from flow import live as lv
    from tests.test_live import FakeClock, SpeakingTTS
    from tests.test_pose import _body_rig
    model, rig_path, bm = _body_rig(tmp_path / "x")
    rig = fs.EventRig(str(model), str(rig_path), "default", None, None, FPS,
                      str(tmp_path), log=lambda s: None)
    clock = FakeClock(2.0)
    stream = fs.MotionStream(rig, seed=0, autonomic=False,
                             baseline_scale=opts.get("baseline_scale", 0.0),
                             log=lambda s: None)
    a = argparse.Namespace(seed=0, lead=0.25, gap=0.12, lipsync="off",
                           mouth_gain=1.0, cjk_rate=dr.CJK_RATE,
                           latin_rate=dr.LATIN_RATE, mood="auto",
                           backend="deepseek", **opts)
    run = {"system": pz.system_prompt(bm), "labels": {}, "offered": None,
           "mode": "body", "body": bm, "core": None, "images": False,
           "max_tag": pz.MAX_MARKER, "parser": lambda: pz.parser_for(bm)}
    p = lv.Performer(a, run, rig, stream, clock, lv.Hub(clock),
                     dr.TextBackend(reply), SpeakingTTS(clock),
                     log=lambda s: None)
    return p, stream, rig


def test_live_carries_the_head_and_releases_it_when_the_speech_ends(tmp_path):
    reply = ('Hello there. [pose {"head.tilt": 7, "dur": 0.25, "hold": 0.4}]'
             ' Nice to see you. [pose {"head.tilt": -6, "head.yaw": 3, '
             '"dur": 0.25, "hold": 0.4}] Bye now!')
    p, stream, rig = _body_performer(tmp_path, reply, head_carry=True,
                                     baseline_scale=1.0)
    p.turn("hi")
    poses = [k for k in stream.pending if k["type"] == "pose"]
    beats = [k for k in poses if not k.get("end")]
    ends = [k for k in poses if k.get("end")]
    tilt = rig.roles["head_z"][0]
    yaw = rig.roles["head_x"][0]
    assert len(beats) == 2 and len(ends) == 1
    assert all(tilt in k["carry"] for k in beats)
    assert yaw not in beats[1]["carry"]            # 3 degrees: a reaction
    assert ends[0]["t"] >= beats[1]["t"] + beats[1]["hold"]
    assert abs(ends[0]["t"] - p.next_frame / FPS) < 1e-3
    assert ends[0]["release"] >= 1.0
    # played: still tilted well past the second beat's hold, relaxed after
    ys = []
    for f in range(int((ends[0]["t"] + 4.0) * FPS)):
        over, _ = stream.frame()
        ys.append(over[tilt])
    first = int((beats[0]["t"] + 1.0) * FPS)     # past a 0.4 s hold
    mid = int((beats[1]["t"] + 0.6) * FPS)
    assert abs(ys[first]) > 5.0 and abs(ys[mid]) > 4.0
    assert ys[first] * ys[mid] < 0                # the two beats' tilts
    idle = fs.MotionStream(rig, seed=0, autonomic=False,
                           log=lambda s: None)
    iy = [idle.frame()[0][tilt] for _ in ys]
    assert abs(ys[-1] - iy[-1]) < 0.3             # relaxed: the idle's own

    # --no-head-carry: the tokens of before, and no end token
    p2, stream2, _ = _body_performer(tmp_path / "b", reply, head_carry=False)
    p2.turn("hi")
    poses2 = [k for k in stream2.pending if k["type"] == "pose"]
    assert len(poses2) == 2 and not any("carry" in k or "end" in k
                                        for k in poses2)


def test_the_offline_director_plays_carry_and_baseline_as_the_stream_does(
        tmp_path, capsys):
    """The director's render and flow.live's stream draw the same frames
    from the same tokens (the end-of-reply token among them), and the
    offline flags turn each feature off."""
    from tests.test_live import _stream_plan
    from tests.test_pose import _body_rig
    model, rig, bm = _body_rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text('Hi there! [pose {"brow.L.lift": 0.8, "head.tilt": 6, '
                     '"dur": 0.2, "hold": 0.4}] Look over there. '
                     '[pose {"head.yaw": 20, "hold": 1}] Nice. Bye! I really '
                     'mean '
                     'it, so see you very soon and take good care.',
                     encoding="utf-8")
    plans = {}
    for tag, extra in (("on", []), ("nocarry", ["--no-head-carry"]),
                       ("nobase", ["--baseline-scale", "0"])):
        out = tmp_path / tag
        assert dr.main(["--model-json", str(model), "--rig", str(rig),
                        "--backend", "file", "--reply-file", str(reply),
                        "--out", str(out), "--probe-dir", str(tmp_path),
                        "--seed", "2", "--body-targets", *extra]) == 0
        capsys.readouterr()
        blob = json.loads((out / "tokens.json").read_text())
        plans[tag] = (blob, np.load(out / "plan.npz", allow_pickle=True))
    on = [k for k in plans["on"][0]["tokens"] if k["type"] == "pose"]
    assert on[-1].get("end") and [k["carry"] for k in on[:-1]] == [
        [k for k in on[0]["carry"]], ["ParamAngleX"]]
    nc = [k for k in plans["nocarry"][0]["tokens"] if k["type"] == "pose"]
    assert not any("carry" in k for k in nc) and nc[-1].get("end")
    er = fs.EventRig(str(model), str(rig), "default", None, None, FPS,
                     str(tmp_path), log=lambda s: None)
    for tag in ("on", "nocarry", "nobase"):
        blob, plan = plans[tag]
        traj, _ = _stream_plan(er, blob, len(plan["traj"]), seed=2,
                               head_drift=1.0, pose_ease=True,
                               baseline_scale=1.0)
        np.testing.assert_array_equal(traj, plan["traj"])
    # The runtime's replay with a resting face on (a parameter of the fake
    # rig stands in for a face's): the stream draws it frame for frame.
    from flow.runtime import main as runtime_main
    blob, plan = plans["on"]
    face = {"t": 1.0, "type": "pose", "role": "pose", "duration": 0.2,
            "hold": 0.4, "release": 0.5, "groups": [], "src": "director",
            "params": {"ParamHandL": 0.8}, "parts": {}, "parts_from": {},
            "step": [], "variant": "v", "base": ["ParamHandL"],
            "valence": 0.5}
    blob = dict(blob, tokens=sorted(blob["tokens"] + [face],
                                    key=lambda k: (k["t"], k["type"])))
    (tmp_path / "face.json").write_text(json.dumps(blob))
    n = len(plan["traj"])
    for scale in (0.0, 1.0):
        argv = ["--model", str(model), "--rig", str(rig), "--events", "default",
                "--probe-dir",
                str(tmp_path), "--fps", "30", "--seed", "2", "--seconds",
                f"{(n - 1) / FPS:.2f}", "--head-drift-scale", "1",
                "--pose-ease", "--baseline-scale", f"{scale:g}",
                "--tokens-in", str(tmp_path / "face.json"), "--dump-only",
                "--dump", str(tmp_path / f"face{scale:g}.npz")]
        assert runtime_main(argv) == 0
        got = np.load(tmp_path / f"face{scale:g}.npz", allow_pickle=True)
        traj, _ = _stream_plan(er, blob, len(got["traj"]), seed=2,
                               head_drift=1.0, pose_ease=True,
                               baseline_scale=scale)
        np.testing.assert_array_equal(traj, got["traj"])
        if scale:
            hand = list(got["pid"]).index("ParamHandL")
            rest = got["traj"][int(3.0 * FPS), hand]
            blank = np.load(tmp_path / "face0.npz",
                            allow_pickle=True)["traj"][int(3.0 * FPS), hand]
            assert abs(rest - blank) > 0.05
    # carried: the yaw look is still on well past its 2 s hold, and the
    # no-carry render has released it
    pid = list(plans["on"][1]["pid"])
    t_look = [k for k in on if "ParamAngleX" in k["params"]][0]["t"]
    f = int((t_look + 0.35 + 1.0 + 1.0) * FPS)
    on_x, off_x = (plans[t][1]["traj"][:, plan_col(plans[t][1], "ParamAngleX")]
                   for t in ("on", "nocarry"))
    assert abs(on_x[f]) > 0.4 and abs(off_x[f]) < 0.2


def plan_col(plan, p):
    """The column of parameter p in a plan's expressive trajectory."""
    return list(plan["pid"]).index(p)
