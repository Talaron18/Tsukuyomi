"""
flow.dataexport's miss corrections and avatar weights, without network: a
run of flow.dataset with mocked models whose replies ask for what the
bodies cannot reach (seed-san, a VRM, and Hiyori's body file), records
edited in memory for the mismatches.
"""

from __future__ import annotations

import copy
import json
import os

import pytest

from flow import body as fb
from flow import dataexport as dx
from flow import dataset as ds
from flow import pose as pz
from flow import vrm as fv
from tests.test_dataset import CharLLM, UserLLM, HIYORI, SEED_SAN, ROOT

LAFEI = os.path.join(ROOT, "rigs", "lafei.body.json")

# One marker with a missed hand and an anchor on the other hand, one with
# two missed hands, one that is reached.
MISSED = ('Hi! [pose {"hand.R.x": 2.5, "hand.R.y": "@shoulder.y", '
          '"head.tilt": 8, "dur": 0.5}] ok. '
          '[pose {"hand.L.y": 5, "hand.L.x": -0.3}] Look [pose {"gaze.x": '
          '-0.5}] there.')
REACHED = 'Hello! [pose {"head.tilt": 8, "hold": 0.5}] Nice to see you.'


def _run(tmp_path, reply, avatars="seed-san,Hiyori,lafei", sessions=3, turns=2,
         prompt="v1"):
    out = tmp_path / "run"

    def make(role, av):
        return CharLLM([reply]) if role == "character" else UserLLM()
    opts = ds.parse_opts(["gen", "--avatars", avatars, "--live2d-lite",
                          "--sessions", str(sessions), "--turns", str(turns),
                          "--out", str(out), "--personas", "cheerful",
                          "--teacher-prompt", prompt])
    ds.generate(opts, backend_factory=make, log=lambda s: None)
    return str(out), ds.iter_records(str(out))


@pytest.fixture(scope="module")
def missed(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("miss"), MISSED)


def _by(recs, avatar):
    return [r for r in recs if r["avatar"]["name"] == avatar]


def _num(x):
    return dx._got_text(x, "hu")


def test_missed_hand_gets_the_reached_value(missed):
    _out, recs = missed
    fix, c = dx.correct_misses(recs)
    r = _by(recs, "Hiyori")[0]
    rep = r["poses"][1]["report"]
    assert not rep["hand.L.y"]["ok"] and not rep["hand.L.x"]["ok"]
    exp = MISSED.replace(
        '"hand.R.x": 2.5', '"hand.R.x": ' + _num(r["poses"][0]["report"]
                                                   ["hand.R.x"]["got"])
    ).replace('"hand.L.y": 5,', '"hand.L.y": ' + _num(rep["hand.L.y"]["got"])
              + ",").replace('"hand.L.x": -0.3', '"hand.L.x": '
                             + _num(rep["hand.L.x"]["got"]))
    assert fix[r["id"]]["reply"] == exp
    assert fix[r["id"]]["markers"] == 2 and fix[r["id"]]["targets"] == 3
    # Reached: the anchor (Hiyori reaches it), dur, the gaze marker, the
    # spoken text and the original record.
    assert '"@shoulder.y"' in exp and '"dur": 0.5' in exp
    assert '[pose {"gaze.x": -0.5}]' in exp
    assert r["reply_raw"] == MISSED
    assert c["skipped"] == {}
    assert c["by_avatar"]["Hiyori"]["targets"] == 3 * 2        # 2 turns


def test_an_anchor_that_missed_becomes_a_number(missed):
    _out, recs = missed
    fix, _c = dx.correct_misses(recs)
    r = _by(recs, "seed-san")[0]
    e = r["poses"][0]["report"]["hand.R.y"]
    assert r["poses"][0]["targets"]["hand.R.y"] == "@shoulder.y" and not e["ok"]
    text = fix[r["id"]]["reply"]
    assert '"hand.R.y": ' + _num(e["got"]) + "," in text
    assert "@shoulder" not in text
    # Only the numbers moved: the text with every number blanked is the same.
    blank = lambda s: __import__("re").sub(r"-?\d+(\.\d+)?|\"@shoulder.y\"",
                                            "N", s)
    assert blank(text) == blank(r["reply_raw"])


def test_a_reached_pose_is_untouched(tmp_path):
    _out, recs = _run(tmp_path, REACHED)
    assert all(r["reach"]["miss"] == 0 for r in recs)
    fix, c = dx.correct_misses(recs)
    assert fix == {} and c["markers"] == 0 and c["miss_turns"] == 0


def test_markers_and_poses_that_disagree_are_skipped(missed):
    _out, recs = missed
    r = copy.deepcopy(_by(recs, "Hiyori")[0])
    for edit in (lambda r: r["markers"].pop(0),
                 lambda r: r["markers"][1].__setitem__("at", 3),
                 lambda r: r["poses"].append(dict(r["poses"][0])),
                 lambda r: r["poses"][0]["targets"].__setitem__("hand.R.x",
                                                                1.0),
                 lambda r: r.__setitem__("reply_raw",
                                         r["reply_raw"].replace("Hi!",
                                                                "Hi [pose "
                                                                '{"head.yaw": 3}]'))):
        x = copy.deepcopy(r)
        edit(x)
        fix, c = dx.correct_misses([x])
        assert fix == {} and c["skipped"] == {"mismatch": 1}, c
        assert c["by_avatar"]["Hiyori"]["skipped"] == 1


def test_records_without_the_report_are_skipped(missed):
    _out, recs = missed
    r = copy.deepcopy(_by(recs, "Hiyori")[0])
    for p in r["poses"]:
        del p["report"]
    fix, c = dx.correct_misses([r])
    assert fix == {} and c["skipped"] == {"no_fields": 1}
    r = copy.deepcopy(_by(recs, "Hiyori")[0])
    del r["markers"][0]["at"]
    assert dx.correct_misses([r])[1]["skipped"] == {"no_fields": 1}


def test_a_target_that_cannot_be_found_skips_only_its_marker(missed):
    _out, recs = missed
    r = copy.deepcopy(_by(recs, "Hiyori")[0])
    r["poses"][1]["report"]["hand.L.y"]["got"] = None     # no value reached
    fix, c = dx.correct_misses([r])
    assert c["skipped"] == {"unmappable": 1} and c["markers"] == 1
    text = fix[r["id"]]["reply"]
    assert '"hand.L.y": 5, "hand.L.x": -0.3' in text and '"hand.R.x": 2.5' \
        not in text
    # Not all corrected: 2 of 6 targets still miss, above a 0.2 share.
    notes = dx.annotate([r], max_miss=0.2, fix=fix)
    assert "miss" in notes[r["id"]][0] and "miss_fixed" not in notes[r["id"]][0]


def test_lenient_json_is_rewritten_in_place():
    reply = "Hi [pose: {'hand.R.x': 2.5, hand.R.y: @shoulder.y+0.1}] bye"
    spans = dx.pose_spans(reply)
    assert len(spans) == 1 and spans[0][2] == 3
    new = dx._rewrite(reply[spans[0][0]:spans[0][1]],
                      {"hand.R.x": "0.5", "hand.R.y": "-0.2"})
    assert new == "[pose: {'hand.R.x': 0.5, hand.R.y: -0.2}]"
    assert dx._rewrite("[pose {\"a\": 1}]", {"b": "2"}) is None
    assert dx._got_text(-0.004, "hu") == "0" and dx._got_text(7.6, "deg") == "8"
    assert dx._got_text(0.5, "unit") == "0.5" and dx._got_text(-0.2, "hu") \
        == "-0.2"


def test_a_session_cut_at_a_miss_turn_continues(missed):
    out, recs = missed
    plain = dx.annotate(recs, keep_lite=True)
    assert all("miss" in f and not k for f, k in plain.values())
    assert dx.sft_examples(out, recs, plain, per="session") == []
    fix, _c = dx.correct_misses(recs)
    notes = dx.annotate(recs, keep_lite=True, fix=fix)
    for f, k in notes.values():
        assert "miss" in f and "miss_fixed" in f and k
    ex = dx.sft_examples(out, recs, notes, per="session", meta=True, fix=fix)
    assert len(ex) == 3 and {len(e["meta"]["turns"]) for e in ex} == {2}
    assert not any("[pose {\"hand.R.x\": 2.5" in m["content"]
                   for e in ex for m in e["messages"])
    # The miss is kept in the stats, with the counters.
    s = dx.stats(recs, notes, counts=_c)
    assert s["by"]["flags"]["miss"] == s["by"]["flags"]["miss_fixed"] == 6
    assert s["corrections"]["markers"] == _c["markers"] >= 12
    assert f"miss corrections: {_c['markers']} markers" in dx.stats_text(s)
    # Per turn: the history shows the corrected earlier reply.
    ex = dx.sft_examples(out, recs, notes, per="turn", meta=True, fix=fix)
    last = next(e for e in ex if e["meta"]["id"].endswith("/t02"))
    hist = [m for m in last["messages"] if m["role"] == "assistant"]
    assert hist[0]["content"] == hist[1]["content"] != MISSED


def test_avatar_weights_repeat_sessions(missed):
    out, recs = missed
    fix, _c = dx.correct_misses(recs)
    notes = dx.annotate(recs, keep_lite=True, fix=fix)
    one = dx.sft_examples(out, recs, notes, meta=True, fix=fix)
    w = dx.parse_weights(["Hiyori=3"], {"Hiyori", "seed-san"})
    many = dx.sft_examples(out, recs, notes, meta=True, fix=fix, weights=w)
    n = {a: sum(e["meta"]["avatar"] == a for e in one)
         for a in ("Hiyori", "seed-san", "lafei")}
    assert n == {"Hiyori": 1, "seed-san": 1, "lafei": 1}
    assert len(many) == 5
    assert many == dx.sft_examples(out, recs, notes, meta=True, fix=fix,
                                   weights=w)                   # same order
    sids = [e["meta"]["session"] for e in many]
    assert [sids.count(x) for x in dict.fromkeys(sids)] == [3, 1, 1] or \
        sorted(sids.count(x) for x in set(sids)) == [1, 1, 3]
    h = [e for e in many if e["meta"]["avatar"] == "Hiyori"]
    assert h[0] == h[1] == h[2] and h[0] is not h[1]
    assert dx.parse_weights(["Hiyori=2,seed-san=1", "x=4"]) == \
        {"Hiyori": 2, "seed-san": 1, "x": 4}
    for bad in ("Hiyori", "Hiyori=0", "Hiyori=x", "nobody=2"):
        with pytest.raises(SystemExit):
            dx.parse_weights([bad], {"Hiyori"})


def _bodies():
    return {"Hiyori": fb.load(HIYORI),
            "lafei": fb.load(LAFEI),
            "seed-san": fv.body_for(None, SEED_SAN, log=lambda s: None)[0]}


def test_corrected_text_parses_and_solves_to_ok(missed):
    out, recs = missed
    fix, _c = dx.correct_misses(recs)
    bodies = _bodies()
    for r in recs:
        bm = bodies[r["avatar"]["name"]]
        text = fix[r["id"]]["reply"]
        parser = pz.parser_for(bm)
        tags = parser.feed(text) + parser.close()
        assert parser.spoken == r["spoken"]          # the speech is the same
        assert not [w for w in parser.warnings if "repaired" in w]
        acts = [dict(t, t=0.5 * i + 0.1) for i, t in enumerate(tags)
                if t["tag"] == pz.POSE]
        assert len(acts) == 3
        # Each marker solved on its own from the rest: the reach the
        # student is taught is the one the body has. The Live2D bodies
        # reach every corrected value; seed-san's hands that would pass
        # through the body land near the value, not on it (the solver
        # shifts them again: ~85% of such markers re-solve ok), so there
        # the error only has to shrink.
        for a, p in zip(acts, [p for p in r["poses"] if not p.get("via")]):
            rec = pz.solve_poses(bm, [a], log=lambda s: None)[0]
            if r["avatar"]["name"] != "seed-san":
                assert rec["ok"], (r["avatar"]["name"], a, rec["report"])
                continue
            for v, e in p["report"].items():
                assert abs(rec["report"][v]["error"]) <= abs(e["error"]) + 0.01


def test_cli_and_trainer_loading(missed, tmp_path, capsys):
    out, _recs = missed
    dst = tmp_path / "sft.jsonl"
    argv = ["sft", out, "--out", str(dst), "--meta", "--keep-lite",
            "--avatar-weight", "Hiyori=2"]
    assert dx.main(argv) == 0
    text = capsys.readouterr().out
    assert "4 examples" in text and "miss corrections: " in text
    rows = [json.loads(x) for x in dst.read_text().splitlines()]
    assert not any("2.5" in m["content"] for e in rows
                   for m in e["messages"] if m["role"] == "assistant")
    # Off: the replies as the teacher wrote them, and the sessions cut at
    # the first MISS turn (here all of them).
    assert dx.main(argv + ["--no-correct-misses"]) == 0
    assert "0 examples" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        dx.main(["sft", out, "--out", str(dst), "--avatar-weight", "x=2"])
    from flow.train import sft
    exs = sft.load_examples(out, "session", keep_lite=True,
                            avatar_weight=["lafei=2"])
    assert [e["meta"]["avatar"] for e in exs].count("lafei") == 2
    assert len(exs) == 4
    assert sft.load_examples(out, "session", keep_lite=True,
                             correct_misses=False) == []
    ns = sft.parse(["stats", "--data", out])
    assert ns.correct_misses is True and ns.avatar_weight == []
    ns = sft.parse(["stats", "--data", out, "--no-correct-misses",
                    "--avatar-weight", "lafei=2"])
    assert ns.correct_misses is False and ns.avatar_weight == ["lafei=2"]


# ------------------------------------------------------------ manners

BUSY = ('Hi! [pose {"hand.R.y": 1.2, "head.tilt": 8, "dur": 0.5}] ok. '
        '[pose {"hand.L.x": 0.4, "shoulders.up": 0.5}] Look '
        '[pose {"gaze.x": -0.5, "torso.tilt": 4, "torso.yaw": 15}] there. '
        '[move {"name": "wave"}][gesture:home] [move {"name": "head_bob"}] '
        'Yes [blink] [pause:0.3] done.')


def _kinds(text):
    return [(t["tag"], t["arg"] if t["tag"] != "pose"
             else sorted(t["arg"]["targets"]))
            for _a, _b, t in dx.marker_spans(text)]


def test_unasked_turn_loses_hands_keeps_head_and_face():
    res = dx.manners_reply(BUSY, False, max_poses=5)
    out = res["reply"]
    assert res["skipped"] is None
    assert '[pose {"head.tilt": 8, "dur": 0.5}] ok.' in out
    assert '[pose {"gaze.x": -0.5, "torso.tilt": 4}]' in out
    assert "hand." not in out and "shoulders" not in out
    assert "torso.yaw" not in out                    # 15 deg: not small
    c = res["c"]
    assert c["hand_before"] == 2 and c["hand_after"] == 0
    assert c["markers_emptied"] == 1 and c["targets_removed"] == 4
    # The empty marker is gone, with no double space where it stood.
    assert "ok. Look" in out and "  " not in out
    assert dx.spoken_of(out) == dx.spoken_of(BUSY)


def test_arm_moves_and_gestures_go_head_moves_and_blinks_stay():
    out = dx.manners_reply(BUSY, False, max_poses=5)["reply"]
    assert [k for k, _ in _kinds(out) if k != "pose"] == [
        "move", "blink", "pause"]
    assert '[move {"name": "head_bob"}]' in out
    assert "wave" not in out and "gesture" not in out
    c = dx.manners_reply(BUSY, False, max_poses=5)["c"]
    assert c["moves_removed"] == 1 and c["gestures_removed"] == 1
    # An alias of a head move is a head move.
    assert '[move {"name": "nod"}]' in dx.manners_reply(
        'Yes [move {"name": "nod"}] ok', False)["reply"]


def test_asked_turn_keeps_the_body_but_still_caps():
    res = dx.manners_reply(BUSY, True, max_poses=5)
    assert res["reply"] == BUSY and res["c"]["targets_removed"] == 0
    res = dx.manners_reply(BUSY, True, max_poses=2)
    out = res["reply"]
    # The two markers with hand targets stay; the gaze one is capped.
    assert out.count("[pose") == 2 and "gaze.x" not in out
    assert "hand.R.y" in out and "hand.L.x" in out
    assert "[move" in out and "[gesture:home]" in out
    assert res["c"]["markers_capped"] == 1
    assert dx.spoken_of(out) == dx.spoken_of(BUSY)


def test_cap_keeps_the_first_markers_when_unasked():
    text = ('A [pose {"head.yaw": 5}] b [pose {"head.yaw": 6}] c '
            '[pose {"head.yaw": 7}] d [pose {"head.yaw": 8}] e')
    out = dx.manners_reply(text, False, max_poses=2)["reply"]
    assert out == 'A [pose {"head.yaw": 5}] b [pose {"head.yaw": 6}] c d e'
    assert dx.manners_reply(text, False, max_poses=9)["reply"] == text


def test_marker_forms_and_the_untouched_ones():
    # [pose: {...}], a bare [{...}] and fullwidth brackets are markers too; a
    # marker with nothing to drop is not rewritten.
    text = 'Hi [pose: {"hand.R.y": 1, "head.yaw": 3}] you［pose {"head.pitch": 2}］ hey'
    out = dx.manners_reply(text, False)["reply"]
    assert out == 'Hi [pose {"head.yaw": 3}] you［pose {"head.pitch": 2}］ hey'
    # Hands only, at the start and the end of the reply.
    assert dx.manners_reply('[pose {"hand.R.y": 1}] Hello there. '
                            '[pose {"hands": "@together"}]', False)[
        "reply"] == "Hello there. "
    # Text with no markers, and a marker the parser drops, stay as written.
    for t in ("Just words.", 'Hm [pose {"hand.R.y": }] ok', "[wink] ok"):
        assert dx.manners_reply(t, False)["reply"] == t


def test_the_ask_signals():
    asked = ["Can you wave to my friend?", "Raise your right hand!",
             "Strike a cute pose", "Give me a hug", "Please bow",
             "look at the door", "turn around for me", "show me how big",
             "Can you point to the door?", "Let's dance together",
             "手を振って！", "ちょっと踊ってみて！", "腕を上げて", "指さして",
             "拍手して", "ハグして", "请挥挥手", "举起你的手", "跳舞给我看",
             "转个圈", "抱抱我", "看这边", "指一下门在哪"]
    quiet = ["What did you have for lunch today?", "You look really cute",
             "It's my turn to pay", "What's the point?", "On the other hand...",
             "cheer me up", "I ate the cake you were saving. Sorry.",
             "相手の気持ちが分からない", "上手だね！", "おはよう！今日も元気？",
             "嗨，好久不见！", "比如说，举例来说", "对不起，我很抱歉", "手机没电了",
             "I got promoted today!", "Want to go to the beach with me?"]
    for t in asked:
        assert dx.asked_line(t), t
    for t in quiet:
        assert not dx.asked_line(t), t
    # By topic, with or without the words.
    for topic in ("pose", "show", "invite"):
        assert dx.asked_for({"topic": topic, "user": "ok"}) == (True, "topic")
    assert dx.asked_for({"topic": "pose", "user": "wave!"}) == (
        True, "topic+keyword")
    assert dx.asked_for({"topic": "greeting", "user": "wave!"}) == (
        True, "keyword")
    assert dx.asked_for({"topic": "emotion", "user": "sad"}) == (False, None)


def _first(recs, avatar="Hiyori"):
    return [r for r in recs if r["avatar"]["name"] == avatar][0]


def test_manners_on_records(missed):
    _out, recs = missed
    recs = copy.deepcopy(recs)
    for r in recs:
        r["topic"], r["user"] = "small_talk", "Hello there"
    one = _first(recs)
    two = copy.deepcopy(one)
    two["id"], two["topic"] = one["id"] + "x", "pose"
    fix, _c = dx.correct_misses(recs)
    mfix, c = dx.manners(recs + [two], fix, 2)
    # It starts from the corrected reply, and the original is untouched.
    assert "hand." not in mfix[one["id"]]["reply"]
    assert "head.tilt" in mfix[one["id"]]["reply"]
    assert "hand.R.x" in mfix[two["id"]]["reply"]
    assert one["reply_raw"] == MISSED
    assert dx.spoken_of(mfix[one["id"]]["reply"]) == one["spoken"]
    a = c["by_avatar"]["Hiyori"]
    assert a["asked"] == 1 and a["asked_topic"] == 1
    assert a["unasked"] == a["turns"] - 1 and a["hand_after"] > 0
    assert c["by"][one["id"]] is None and c["by"][two["id"]] == "topic"
    assert c["skipped"] == {}
    # The count set limits the statistics, not the rewriting.
    mfix2, c2 = dx.manners(recs, fix, 2, count={one["id"]})
    assert mfix2 == {k: v for k, v in mfix.items() if k != two["id"]}
    assert c2["total"]["turns"] == 1
    lines = dx.manners_lines(c)
    assert "hand share of targets" in lines[0] and len(lines) > 1


def test_stats_carry_the_manners_counts(missed):
    _out, recs = missed
    notes = dx.annotate(recs)
    _m, c = dx.manners(recs, None, 2)
    s = dx.stats(recs, notes, mcounts=c)
    assert s["manners"]["total"]["turns"] == len(recs)
    assert "manners (at most 2 poses)" in dx.stats_text(s)
    json.dumps(s)


def test_a_turn_whose_spoken_text_would_change_is_left(monkeypatch):
    monkeypatch.setattr(dx, "spoken_of", lambda t: str(len(t)))
    res = dx.manners_reply(BUSY, False)
    assert res["reply"] == BUSY and res["skipped"] == "spoken"


def test_history_turns_are_transformed_per_turn(missed):
    out, recs = missed
    recs = copy.deepcopy(recs)
    lines = ["Hello there", "Tell me about your day", "Which tea is best?",
             "Nice weather, isn't it", "I like rainy mornings", "Any plans?"]
    for r, line in zip(recs, lines):
        r["topic"], r["user"] = "small_talk", line
    fix, _ = dx.correct_misses(recs)
    notes = dx.annotate(recs, keep_lite=True, fix=fix)
    mfix, _ = dx.manners(recs, fix, 2)
    ex = dx.sft_examples(out, recs, notes, per="turn", fix={**fix, **mfix})
    texts = [m["content"] for e in ex for m in e["messages"]
             if m["role"] == "assistant"]
    assert texts and not any("hand." in t for t in texts)
    hist = [m for e in ex for m in e["messages"] if m.get("weight") == 0]
    assert hist
    ses = dx.sft_examples(out, recs, notes, per="session", fix={**fix, **mfix})
    assert not any("hand." in m["content"] for e in ses
                   for m in e["messages"] if m["role"] == "assistant")


def test_flags_off_reproduce_the_old_export(missed, tmp_path, capsys):
    out, recs = missed
    for per in ("session", "turn"):
        old = tmp_path / f"old-{per}.jsonl"
        new = tmp_path / f"new-{per}.jsonl"
        on = tmp_path / f"on-{per}.jsonl"
        assert dx.main(["sft", out, "--out", str(new), "--per", per,
                        "--no-manners"]) == 0
        notes = dx.annotate(recs, fix=dx.correct_misses(recs)[0])
        rows = dx.sft_examples(out, recs, notes, per, False,
                               dx.correct_misses(recs)[0], {})
        dx.write_jsonl(str(old), rows)
        assert old.read_bytes() == new.read_bytes()
        assert dx.main(["sft", out, "--out", str(on), "--per", per]) == 0
        assert on.read_bytes() != new.read_bytes()
        assert dx.main(["sft", out, "--out", str(on), "--per", per,
                        "--max-poses", "9", "--no-manners"]) == 0
        assert on.read_bytes() == new.read_bytes()
    capsys.readouterr()
    assert dx.main(["stats", out]) == 0
    assert "manners (at most 2 poses)" in capsys.readouterr().out
    assert dx.main(["stats", out, "--no-manners"]) == 0
    assert "manners" not in capsys.readouterr().out


def test_the_trainer_passes_manners_through_and_the_hash_changes(missed):
    from flow.train import sft
    out, _recs = missed
    on = sft.load_examples(out, "session", keep_lite=True)
    off = sft.load_examples(out, "session", keep_lite=True, manners=False)
    capped = sft.load_examples(out, "session", keep_lite=True, max_poses=1)
    # The data hash is of the token ids: other text, other hash.
    dump = lambda e: json.dumps(e, sort_keys=True)
    assert dump(on) != dump(off) and dump(on) != dump(capped)
    assert dump(off) == dump(
        sft.load_examples(out, "session", keep_lite=True, manners=False))
    p = sft.argparse.ArgumentParser()
    sft.add_data_args(p)
    o = p.parse_args(["--data", out])
    assert o.manners is True and o.max_poses is None
    assert p.parse_args(["--data", out, "--no-manners"]).manners is False
