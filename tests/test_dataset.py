"""
flow.dataset / flow.dataexport (the Stage 2 data engine) without network:
the generation loop with mocked user and character models on a VRM body,
a Live2D lite body and the synthetic Live2D rig of test_pose (the full
EventRig path), the record schema, what the character model saw against
the SFT export, user-line cleaning, filters and dedupe, the exporters,
resume after an interruption, the budget cap, retries and fatal errors.
Invariants only (shapes, variable lists, times, reach counts): the stream's
floats belong to flow.stream and flow.pose.
"""

from __future__ import annotations

import copy
import json
import os
import re

import numpy as np
import pytest

from flow import dataexport as dx
from flow import dataset as ds
from flow import pose as pz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_SAN = os.path.join(ROOT, "rigs", "seed-san.body.json")
HIYORI = os.path.join(ROOT, "rigs", "Hiyori.body.json")

REPLY = ('Hello! [pose {"hand.R.y": "@shoulder.y", "head.tilt": 8}] Nice to '
         'see you. [blink] Look [pose {"gaze.x": -0.5}] there. [nod]')


class CharLLM:
    """A character model: records what it was sent, streams REPLY (or a
    reply per call), reports DeepSeek-style usage."""

    def __init__(self, replies=None, usage=True, fail=0, exc=None):
        self.replies, self.usage_on = replies, usage
        self.sent, self.usage, self.stop_reason = [], {}, None
        self.model, self.fail, self.exc = "mock-char", fail, exc

    def stream(self, system, messages):
        if self.fail:
            self.fail -= 1
            raise self.exc or RuntimeError("503 from deepseek: busy")
        self.sent.append((system, copy.deepcopy(messages)))
        text = REPLY if self.replies is None else \
            self.replies[(len(self.sent) - 1) % len(self.replies)]
        for i in range(0, len(text), 9):
            yield text[i:i + 9]
        self.stop_reason = "stop"
        self.usage = {"prompt_tokens": 2000, "completion_tokens": 80,
                      "prompt_cache_hit_tokens": 1500} if self.usage_on else {}


class UserLLM:
    LINES = ['User: "Hi there, how was your day?"', "Can you wave to me?",
             "わあ、すごい！もう一回やって？", "你今天想做什么？",
             "Point at the window [points] please.", "Tell me a secret!"]

    def __init__(self):
        self.k, self.usage, self.stop_reason = 0, {}, None
        self.model, self.sent = "mock-user", []

    def stream(self, system, messages):
        self.sent.append(messages[-1]["content"])
        line = self.LINES[self.k % len(self.LINES)]
        self.k += 1
        self.usage = {"prompt_tokens": 300, "completion_tokens": 20}
        yield line


def _factory(store):
    def make(role, av):
        b = CharLLM() if role == "character" else UserLLM()
        store.setdefault(role, []).append(b)
        return b
    return make


def _opts(out, *extra, avatars="seed-san,Hiyori", sessions=4, turns=3):
    return ds.parse_opts(["gen", "--avatars", avatars, "--live2d-lite",
                          "--sessions", str(sessions), "--turns", str(turns),
                          "--out", str(out), "--concurrency", "2",
                          "--personas", "cheerful,sleepy", "--backoff",
                          "0.01", *extra])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen") / "run"
    made = {}
    res = ds.generate(_opts(out), backend_factory=_factory(made),
                      log=lambda s: None)
    return out, res, made


# ------------------------------------------------------------ generation

def test_generation_writes_whole_sessions(run):
    out, res, made = run
    assert res["ok"] == 4 and res["failed"] == 0 and res["stopped"] == "done"
    recs = ds.iter_records(str(out))
    assert len(recs) == 12
    man = [json.loads(x) for x in open(out / "sessions.jsonl")]
    assert sorted(m["session"] for m in man) == sorted(
        {r["session"] for r in recs})
    assert all(m["status"] == "ok" and m["turns"] == 3 for m in man)
    # One fresh backend per session and role (usage lives on the backend).
    assert len(made["character"]) == 4 and len(made["user"]) == 4
    assert {r["avatar"]["kind"] for r in recs} == {"vrm", "live2d-lite"}
    runs = [json.loads(x) for x in open(out / "runs.jsonl")]
    assert "git" in runs[0] and runs[-1]["ok"] == 4


def test_record_schema(run):
    out, _res, _ = run
    reg = dx.load_registry(str(out))
    assert reg["format"] == ds.AVATARS_FORMAT
    for r in ds.iter_records(str(out)):
        assert r["format"] == ds.FORMAT
        assert r["id"] == f"{r['session']}/t{r['turn']:02d}"
        av = reg["avatars"][r["avatar"]["name"]]
        assert r["avatar"]["body_sha"] == av["body_sha"] == ds.file_sha(
            r["avatar"]["body"])
        assert r["avatar"]["prompt_sha"] in reg["prompts"]
        assert len(r["history"]) == r["turn"] - 1
        want = r["user"] + "\n" + av["pose_now"]
        if r.get("framing"):             # the beats prompt: the camera line
            assert r["user_content"].startswith(want + "\n(camera: ")
        else:
            assert r["user_content"] == want
        assert "[" not in r["user"] and not r["user"].startswith("User")
        assert r["spoken"] == "Hello! Nice to see you. Look there."
        # Markers: char offsets into the spoken text, times in order.
        tags = [m["tag"] for m in r["markers"]]
        assert tags == ["pose", "blink", "pose"]
        assert [m["at"] for m in r["markers"]] == [7, 24, 29]
        ts = [m["t"] for m in r["markers"]]
        assert ts == sorted(ts) and ts[0] > 0
        assert all(m["played"] is True for m in r["markers"])
        assert r["unplayed"] == {"nod": 1}
        assert r["reach"]["markers"] == 2 and r["reach"]["targets"] == 3
        assert r["poses"][0]["targets"]["head.tilt"] == 8
        tm = r["timing"]
        assert tm["method"] == "rate" and tm["fps"] == 30.0
        assert tm["seconds"] > tm["speech_seconds"] > 0
        assert r["llm"]["character"]["tokens"] == {"in": 2000, "in_hit": 1500,
                                                   "out": 80}
        assert not r["llm"]["character"]["usage_estimated"]
        want = ds.cost({"in": 2000, "in_hit": 1500, "out": 80}) + \
            ds.cost({"in": 300, "in_hit": 0, "out": 20})
        assert r["cost_usd"] == pytest.approx(want, rel=1e-4)
        assert r["lang"]["asked"] in ("en", "ja", "zh")


def test_trajectories(run):
    out, _res, _ = run
    for r in ds.iter_records(str(out)):
        av = dx.load_registry(str(out))["avatars"][r["avatar"]["name"]]
        tr = ds.load_traj(str(out), r)
        d = ds.load_traj(str(out), r, ".directed")
        assert tr.shape == d.shape == (r["timing"]["frames"],
                                       len(r["traj"]["vars"]))
        assert r["traj"]["vars"] == av["traj_vars"]
        k = r["traj"]["vars"].index("head.tilt")
        # The pose's head tilt (8 deg) is reached and released.
        assert np.nanmax(d[:, k]) == pytest.approx(8.0, abs=0.1)
        assert abs(d[-1, k]) < 0.5
        f0 = int(r["poses"][0]["start"] * 30)
        assert abs(d[max(0, f0 - 1), k]) < 0.5
        ch = ds.load_traj(str(out), r, ".chars")
        assert ch.shape == (len(r["spoken"]), 2)
        assert (np.diff(ch[:, 0]) >= -1e-3).all()
        rep = ds.load_traj(str(out), r, ".replay")
        assert [a["tag"] for a in rep["acts"]] == ["pose", "blink", "pose"]
        if r["avatar"]["kind"] == "vrm":
            # The played trajectory has the idle under it.
            assert not np.allclose(np.nan_to_num(tr), np.nan_to_num(d))
            assert {"hand.R.z", "mouth.open"} <= set(r["traj"]["vars"])


def test_character_saw_what_sft_exports(run):
    """The exporter's messages are byte for byte what the model got."""
    out, _res, made = run
    recs = ds.iter_records(str(out))
    notes = {r["id"]: ([], True) for r in recs}
    ex = dx.sft_examples(str(out), recs, notes, per="session")
    assert len(ex) == 4
    sent = {}
    for b in made["character"]:
        system, msgs = b.sent[-1]
        sent[msgs[0]["content"], system] = (system, msgs)
    for e in ex:
        m = e["messages"]
        assert m[0]["role"] == "system"
        system, msgs = next(v for k, v in sent.items()
                            if k == (m[1]["content"], m[0]["content"]))
        assert m[0]["content"] == system
        assert m[1:-1] == msgs              # the last call's input...
        assert m[-1] == {"role": "assistant", "content": REPLY}   # ...and reply
    # Per turn: the history is not trained.
    ex = dx.sft_examples(str(out), recs, notes, per="turn", meta=True)
    assert len(ex) == 12
    for e in ex:
        assert all(x.get("weight") == 0 for x in e["messages"][2:-1]
                   if x["role"] == "assistant")
        assert "weight" not in e["messages"][-1] and "meta" in e


def test_user_simulator_gets_spoken_text_only(run):
    _out, _res, made = run
    for b in made["user"]:
        for c in b.sent:
            assert "[pose" not in c and "(your body now" not in c
        assert "Hello! Nice to see you. Look there." in b.sent[-1]


def test_action_export(run, tmp_path):
    out, _res, _ = run
    recs = ds.iter_records(str(out))
    notes = dx.annotate(recs, keep_lite=True)
    n = dx.action_pairs(str(out), recs, notes, str(tmp_path / "act"))
    kept = sum(k for _, k in notes.values())
    assert n == kept > 0
    rows = [json.loads(x) for x in open(tmp_path / "act.jsonl")]
    with np.load(tmp_path / "act.npz") as z:
        for row in rows:
            tr = z[row["key"] + ".traj"]
            assert tr.dtype == np.float16
            assert tr.shape == (row["frames"], len(row["vars"]))
            assert z[row["key"] + ".chars"].shape[0] == len(row["spoken"])
            assert row["markers"][0]["t"] > 0


# ------------------------------------------------------------- full Live2D

def test_live2d_rig_end_to_end(tmp_path):
    """A Live2D rig with its model (test_pose's synthetic one): the
    director's own prepare (gestures offered) and EventRig stream."""
    from tests.test_pose import _body_rig
    model, rig, bm = _body_rig(tmp_path / "x")
    y = bm.landmarks["shoulder.y"] + 0.3
    reply = (f'Hi! [pose {{"hand.L.y": {y:.3f}}}] Look at this. '
             '[gesture:home] Nice. [pose {"head.yaw": 15, "hold": 0.5}]')
    made = {}

    def make(role, av):
        b = CharLLM([reply]) if role == "character" else UserLLM()
        made.setdefault(role, []).append(b)
        return b
    opts = ds.parse_opts(["gen", "--avatars",
                          f"x:model={model},rig={rig},body={tmp_path}/x/"
                          "x.body.json,probe_dir=" + str(tmp_path),
                          "--sessions", "1", "--turns", "2",
                          "--personas", "cheerful",
                          "--out", str(tmp_path / "run")])
    res = ds.generate(opts, backend_factory=make, log=lambda s: None)
    assert res["ok"] == 1
    recs = sorted(ds.iter_records(str(tmp_path / "run")),
                  key=lambda r: r["turn"])
    r = recs[0]
    assert r["avatar"]["kind"] == "live2d"
    system = made["character"][0].sent[0][0]
    assert "[gesture:NAME]" in system            # the director's prompt
    g = [m for m in r["markers"] if m["tag"] == "gesture"]
    # The gesture overlaps the raised arm: it gives way to the pose.
    assert g and g[0]["played"] == "dropped"
    assert r["reach"]["ok"] == 2
    d = ds.load_traj(str(tmp_path / "run"), r, ".directed")
    v = ds.load_traj(str(tmp_path / "run"), r, ".variant")
    k = r["traj"]["vars"].index("hand.L.y")
    # The raised arm's drawing while the pose plays, the hand near its
    # target (the pose layer's own floats aside), then back down.
    assert set(np.unique(v)) == {0, 1}
    assert float(np.median(d[v == 1, k])) == pytest.approx(y, abs=0.05)
    assert d[-1, k] < y - 0.5


def test_live2d_without_model_fails_unless_lite(tmp_path):
    with pytest.raises(SystemExit, match="live2d-lite"):
        ds.avatar_spec("Hiyori", [str(tmp_path)])
    assert ds.avatar_spec("Hiyori", [str(tmp_path)],
                          lite=True)["kind"] == "live2d-lite"
    assert ds.avatar_spec("seed-san")["kind"] == "vrm"


# ------------------------------------------------------------ resume etc.

def test_resume_after_interruption(tmp_path):
    out = tmp_path / "run"
    made = {}
    calls = {"n": 0}

    def flaky(role, av):
        calls["n"] += 1
        if role == "character" and calls["n"] > 4:
            return CharLLM(fail=99, exc=RuntimeError(
                "401 from deepseek: bad key"))
        return _factory(made)(role, av)
    res = ds.generate(_opts(out, "--concurrency", "1"),
                      backend_factory=flaky, log=lambda s: None)
    assert res["stopped"].startswith("401") and res["exit"] == 2
    first = {r["session"] for r in ds.iter_records(str(out))}
    assert len(first) == 2
    spent = ds.Store(str(out)).spent["cost_usd"]
    assert spent > 0
    made2 = {}
    res = ds.generate(_opts(out), backend_factory=_factory(made2),
                      log=lambda s: None)
    assert res["ok"] == 2 and len(made2["character"]) == 2
    recs = ds.iter_records(str(out))
    assert len(recs) == 12 and len({r["session"] for r in recs}) == 4
    man = [json.loads(x) for x in open(out / "sessions.jsonl")]
    ok = [m["cost_usd"] for m in man if m["status"] == "ok"]
    assert len(ok) == 4 and max(ok) == pytest.approx(min(ok))
    # The session the 401 stopped had paid its user line: counted too.
    total = sum(m["cost_usd"] for m in man)
    assert total > sum(ok)
    assert ds.Store(str(out)).spent["cost_usd"] == pytest.approx(total)
    # Nothing left: a third run calls nothing.
    res = ds.generate(_opts(out), backend_factory=_factory({}),
                      log=lambda s: None)
    assert res["sessions"] == 0


def test_duplicate_lines_from_a_rerun_dedupe_by_id(tmp_path, run):
    out, _res, _ = run
    recs = ds.iter_records(str(out))
    st = ds.Store(str(tmp_path / "copy"))
    st.write_session("a", recs[:3], {}, {"in": 0, "in_hit": 0, "out": 0,
                                        "cost_usd": 0.0}, "r")
    st.write_session("a", recs[:3], {}, {"in": 0, "in_hit": 0, "out": 0,
                                        "cost_usd": 0.0}, "r")
    assert len(ds.iter_records(str(tmp_path / "copy"))) == 3


def test_budget_cap_stops_new_sessions(tmp_path):
    cap = 0.0025         # each session costs ~0.00093 (3 turns)
    res = ds.generate(_opts(tmp_path / "run", "--concurrency", "1",
                            "--budget-usd", str(cap), sessions=8),
                      backend_factory=_factory({}), log=lambda s: None)
    assert res["stopped"].startswith("budget")
    assert res["ok"] == 2 and res["spent_usd"] <= cap


def test_transient_errors_retry_with_a_fresh_parser(tmp_path):
    made = {}

    def make(role, av):
        b = CharLLM(fail=2) if role == "character" else UserLLM()
        made.setdefault(role, []).append(b)
        return b
    res = ds.generate(_opts(tmp_path / "run", sessions=1, turns=2),
                      backend_factory=make, log=lambda s: None)
    assert res["ok"] == 1
    r = sorted(ds.iter_records(str(tmp_path / "run")),
               key=lambda r: r["turn"])
    assert r[0]["llm"]["character"]["attempts"] == 3
    assert r[0]["spoken"] == "Hello! Nice to see you. Look there."


def test_failing_sessions_trip_the_breaker(tmp_path):
    def make(role, av):
        return CharLLM(fail=99) if role == "character" else UserLLM()
    res = ds.generate(_opts(tmp_path / "run", "--retries", "0",
                            "--max-consecutive-failures", "2",
                            "--concurrency", "1", sessions=6),
                      backend_factory=make, log=lambda s: None)
    assert res["failed"] == 2 and "in a row" in res["stopped"]
    man = [json.loads(x) for x in open(tmp_path / "run" / "sessions.jsonl")]
    assert [m["status"] for m in man] == ["failed", "failed"]


def test_missing_usage_is_estimated(tmp_path):
    def make(role, av):
        return CharLLM(usage=False) if role == "character" else UserLLM()
    ds.generate(_opts(tmp_path / "run", sessions=1, turns=1),
                backend_factory=make, log=lambda s: None)
    r = ds.iter_records(str(tmp_path / "run"))[0]
    c = r["llm"]["character"]
    assert c["usage_estimated"] and c["tokens"]["in"] > 1000
    assert "usage_estimated" in dx.flag(r)


def test_estimate_before_starting(tmp_path):
    msgs = []
    res = ds.generate(_opts(tmp_path / "run", "--dry-run", sessions=6,
                            turns=6), log=msgs.append)
    e = res["estimate"]
    assert e["sessions"] == 6 and e["turns"] == 36
    assert 0 < e["cost_usd"] < e["cost_usd_no_cache"]
    assert any("estimate for 6 sessions" in m for m in msgs)
    assert not os.path.exists(tmp_path / "run" / "sessions.jsonl")


def test_refused_turn_is_not_history(tmp_path):
    class Refuse(CharLLM):
        def stream(self, system, messages):
            yield from super().stream(system, messages)
            if len(self.sent) == 1:
                self.stop_reason = "refusal"
    made = {}

    def make(role, av):
        b = Refuse() if role == "character" else UserLLM()
        made.setdefault(role, []).append(b)
        return b
    ds.generate(_opts(tmp_path / "run", avatars="seed-san", sessions=1,
                      turns=2), backend_factory=make, log=lambda s: None)
    r = sorted(ds.iter_records(str(tmp_path / "run")), key=lambda r: r["turn"])
    assert r[0]["refused"] and r[0]["traj"] is None
    assert r[1]["history"] == []
    assert len(made["character"][0].sent[1][1]) == 1
    assert "refused" in dx.flag(r[0])


# -------------------------------------------------------------- filters

def test_clean_user_line():
    c = ds.clean_user_line
    assert c('User: "Hi there!"') == "Hi there!"
    assert c("「こんにちは！」") == "こんにちは！"
    assert c("Wave at me *smiles* [waves] please") == "Wave at me please"
    assert c("First line.\n\nSecond paragraph.") == "First line."
    assert c("[pose]") == ""
    assert len(c("word " * 200)) <= ds.USER_MAX


def _rec(i, user, persona="p", **kw):
    r = {"id": f"s{i}/t01", "session": f"s{i}", "turn": 1, "user": user,
         "persona": persona, "avatar": {"name": "a", "kind": "vrm"},
         "spoken": "Hi.", "refused": False, "stop_reason": "stop",
         "warnings": [], "unplayed": {},
         "markers": [{"tag": "pose", "at": 0, "t": 0.1, "played": True}],
         "reach": {"targets": 3, "miss": 0}, "llm": {}}
    r.update(kw)
    return r


def test_dedupe_near_identical_lines():
    recs = [_rec(0, "Can you wave to my friend?"),
            _rec(1, "can you wave to my friend!!"),
            _rec(2, "Can you wave to my friend, please?"),
            _rec(3, "What's your favourite food?"),
            _rec(4, "Can you wave to my friend?", persona="q")]
    d = dx.dedupe(recs, 0.85)
    assert d == {"s1/t01", "s2/t01"} or d == {"s1/t01"}
    assert "s1/t01" in d and "s3/t01" not in d and "s4/t01" not in d


def test_flags():
    assert dx.flag(_rec(0, "x")) == []
    assert "miss" in dx.flag(_rec(0, "x", reach={"targets": 3, "miss": 2}))
    assert "no_body" in dx.flag(_rec(0, "x", markers=[]))
    assert "empty" in dx.flag(_rec(0, "x", spoken=""))
    assert "cutoff" in dx.flag(_rec(0, "x", stop_reason="length"))
    assert "parse" in dx.flag(_rec(0, "x", warnings=["bad JSON in marker"]))
    f = dx.flag(_rec(0, "x", warnings=[
        "[nod] is not a body-mode marker (pose, blink, pause); dropped"],
        unplayed={"nod": 1}))
    assert "parse" not in f and "unplayed" in f
    notes = dx.annotate([_rec(0, "a b c"), _rec(1, "a b c"),
                         _rec(2, "z", avatar={"name": "a",
                                              "kind": "live2d-lite"})])
    assert notes["s1/t01"] == (["dup"], False)
    assert notes["s2/t01"][1] is False
    assert dx.annotate([_rec(2, "z", avatar={"name": "a", "kind":
                                             "live2d-lite"})],
                       keep_lite=True)["s2/t01"][1]


def test_session_export_cuts_at_the_first_dropped_turn(run):
    out, _res, _ = run
    recs = ds.iter_records(str(out))
    sid = recs[0]["session"]
    notes = {r["id"]: ([], not (r["session"] == sid and r["turn"] == 2))
             for r in recs}
    ex = dx.sft_examples(str(out), recs, notes, per="session", meta=True)
    mine = next(e for e in ex if e["meta"]["session"] == sid)
    assert len(mine["meta"]["turns"]) == 1 and len(mine["messages"]) == 3


def test_stats(run):
    out, _res, _ = run
    recs = ds.iter_records(str(out))
    notes = dx.annotate(recs, keep_lite=True)
    s = dx.stats(recs, notes)
    assert s["turns"] == 12 and s["sessions"] == 4
    assert s["by"]["markers"] == {"pose": 24, "blink": 12}
    assert s["by"]["unplayed"] == {"nod": 12}
    assert s["reach"]["targets"] == 36 and s["reach"]["rate"] == 1.0
    assert s["variables"] == {"hand.R.y": 12, "head.tilt": 12, "gaze.x": 12}
    assert s["tokens"]["character.in"] == 24000
    assert s["cost_usd"] == pytest.approx(sum(r["cost_usd"] for r in recs))
    text = dx.stats_text(s)
    assert "reach: 36/36" in text and re.search(r"hand\.R\.y\s+12 #", text)
    assert dx.main(["stats", str(out)]) == 0


def test_usage_tokens_formats():
    assert ds.usage_tokens({"prompt_tokens": 10, "completion_tokens": 2,
                            "prompt_cache_hit_tokens": 4}) == \
        {"in": 10, "in_hit": 4, "out": 2}
    assert ds.usage_tokens({"input_tokens": 10, "output_tokens": 3,
                            "cache_read_input_tokens": 50,
                            "cache_creation_input_tokens": 5}) == \
        {"in": 65, "in_hit": 50, "out": 3}
    assert ds.usage_tokens({"prompt_eval_count": 7, "eval_count": 1}) == \
        {"in": 7, "in_hit": 0, "out": 1}
    assert ds.usage_tokens({}) is None
    assert ds.cost({"in": 1_000_000, "in_hit": 0, "out": 0}) == \
        ds.PRICES["in"]


def test_config_and_session_plan():
    cfg = ds.load_config()
    s = ds.plan_sessions(["a", "b"], ["cheerful", "sleepy"], cfg, 6, 3)
    assert [x["id"] for x in s[:5]] == [
        "a.cheerful.0000", "a.sleepy.0000", "b.cheerful.0000",
        "b.sleepy.0000", "a.cheerful.0001"]
    # Stable ids and seeds: a longer plan starts with the shorter one.
    assert ds.plan_sessions(["a", "b"], ["cheerful", "sleepy"], cfg, 9,
                            3)[:6] == s
    p = next(x for x in cfg["personas"] if x["id"] == "cheerful")
    assert ds.persona_text(cfg, p, "lafei").startswith("Laffey, a cheerful")
    assert pz.system_prompt  # the body prompt the avatars use


def test_real_tts_timing_drives_markers_and_mouth():
    """--tts: speech.speak times the markers from the audio, and the voice's
    mouth track moves mouth.open on the trajectory."""
    from flow import director as dr
    from tests.test_live import SpeakingTTS
    opts = ds.parse_opts(["gen", "--avatars", "seed-san"])
    av = ds.Avatar(ds.avatar_spec("seed-san"), opts, {"p": "Seed-san."},
                   log=lambda s: None)
    reply = dr.ask(CharLLM(), av.system("p"), [{"role": "user",
                                                "content": "hi"}],
                   {}, log=lambda s: None, parser=av.parser("p"))
    rate, _ = ds.play(av, reply, 1)
    av.tts = SpeakingTTS()
    fields, arr = ds.play(av, reply, 1)
    assert fields["timing"]["method"] != "rate"
    assert fields["timing"]["tts"]["voice"] == "fake"
    assert av.tts.calls                         # the TTS was asked
    t_rate = [m["t"] for m in rate["markers"] if m["t"] is not None]
    t_tts = [m["t"] for m in fields["markers"] if m["t"] is not None]
    assert t_tts != t_rate and t_tts == sorted(t_tts) and t_tts[0] > 0.2
    cols = av.traj_vars
    k = cols.index("viseme.a")
    played = arr[""].astype(np.float32)
    assert np.nanmax(played[:, k]) - np.nanmin(played[:, k]) > 0.1
    assert np.nanmax(arr[".directed"].astype(np.float32)[:, k]) > 0.1
    assert arr[".chars"].shape == (len(reply["spoken"]), 2)


def test_failed_session_spend_counts(tmp_path):
    """A session dying at its third call still paid for the first two: the
    manifest and the resumed budget carry it."""
    class Late(CharLLM):
        def stream(self, system, messages):
            if len(self.sent) >= 2:
                raise RuntimeError("503 from deepseek: busy")
            yield from super().stream(system, messages)
    out = tmp_path / "run"
    res = ds.generate(_opts(out, "--retries", "0", "--concurrency", "1",
                            sessions=1, turns=3),
                      backend_factory=lambda role, av: Late() if
                      role == "character" else UserLLM(), log=lambda s: None)
    assert res["failed"] == 1 and res["run_usd"] > 0
    man = [json.loads(x) for x in open(out / "sessions.jsonl")]
    assert man[0]["status"] == "failed" and man[0]["cost_usd"] > 0
    assert man[0]["tokens"]["in"] == 2 * 2000 + 3 * 300
    assert ds.Store(str(out)).spent["cost_usd"] == pytest.approx(
        man[0]["cost_usd"])


def test_duplicate_greeting_does_not_cut_a_session(run):
    out, _res, _ = run
    recs = ds.iter_records(str(out))
    sid = recs[0]["session"]
    notes = {r["id"]: ((["dup"], False) if r["session"] == sid
                       and r["turn"] == 1 else ([], True)) for r in recs}
    ex = dx.sft_examples(str(out), recs, notes, per="session", meta=True)
    mine = next(e for e in ex if e["meta"]["session"] == sid)
    assert len(mine["meta"]["turns"]) == 3
    # Per turn, a dup is dropped.
    ids = [e["meta"]["id"] for e in dx.sft_examples(str(out), recs, notes,
                                                    per="turn", meta=True)]
    assert f"{sid}/t01" not in ids and len(ids) == 11


# ------------------------------------------------------- moves and replay

MOVE_REPLY = ('Yay! [move {"name": "clap", "beats": 2, "tempo": 120}] Let\'s '
              'go, clap along. [pose {"head.tilt": 8}] Done!')


@pytest.fixture(scope="module")
def moved(tmp_path_factory):
    """A run whose replies clap (seed-san and the Hiyori lite body)."""
    out = tmp_path_factory.mktemp("gen") / "moves"

    def make(role, av):
        return CharLLM([MOVE_REPLY]) if role == "character" else UserLLM()
    res = ds.generate(_opts(out, sessions=4, turns=1), backend_factory=make,
                      log=lambda s: None)
    assert res["ok"] == 4
    return out


def test_move_markers_play_in_both_trajectories(moved):
    recs = ds.iter_records(str(moved))
    assert {r["avatar"]["kind"] for r in recs} == {"vrm", "live2d-lite"}
    for r in recs:
        m = [x for x in r["markers"] if x["tag"] == "move"]
        assert len(m) == 1 and m[0]["played"] is True
        assert m[0]["arg"] == {"name": "clap", "beats": 2.0, "tempo": 120.0}
        assert "move" not in r["unplayed"]
        assert r["traj"]["version"] == ds.TRAJ_VERSION
        mv_ = r["moves"][0]
        assert mv_["name"] == "clap" and mv_["start"] == pytest.approx(
            m[0]["t"])
        # The clip ends inside the trajectory.
        assert r["timing"]["seconds"] >= mv_["start"] + mv_["seconds"]
        cols = r["traj"]["vars"]
        f0 = int(mv_["start"] * 30)
        f1 = int((mv_["start"] + mv_["seconds"]) * 30)
        for which in ("", ".directed"):
            tr = ds.load_traj(str(moved), r, which)
            # Two claps: the hands travel toward each other and back.
            gap = tr[f0:f1, cols.index("hand.L.x")] - \
                tr[f0:f1, cols.index("hand.R.x")]
            assert np.nanmax(gap) - np.nanmin(gap) > 0.15
            # Before the move the hands are where the idle keeps them.
            pre = tr[:max(1, f0 - 1), cols.index("hand.L.x")]
            assert np.nanmax(pre) - np.nanmin(pre) < 0.05
        if r["avatar"]["kind"] == "vrm":
            assert {"hips.x", "hips.y"} <= set(cols)
            assert not np.isnan(ds.load_traj(str(moved), r)[
                :, cols.index("hips.x")]).any()
        rep = ds.load_traj(str(moved), r, ".replay")
        assert rep["v"] == ds.REPLAY_VERSION and rep["seed"] == r["seed"]
        assert [t["tag"] for t in rep["tags"]] == ["move", "pose"]
        assert any(k["type"] == "move" for k in rep["tokens"])


def _no_clock(x):
    if isinstance(x, dict):
        return {k: _no_clock(v) for k, v in x.items()
                if k != "solve_seconds"}
    if isinstance(x, list):
        return [_no_clock(v) for v in x]
    return x


def _arrays(out, r):
    return {w: ds.load_traj(str(out), r, w)
            for w in ("", ".directed", ".variant", ".chars")}


def test_replay_reproduces_the_trajectories(moved, tmp_path):
    """The replay record is enough: re-streamed with no model, every array
    comes back the same and the records are the same but for traj."""
    import shutil
    out = tmp_path / "copy"
    shutil.copytree(moved, out)
    before = {r["id"]: r for r in ds.iter_records(str(out))}
    msgs = []
    assert ds.replay(str(out), log=msgs.append)["turns"] == 0  # up to date
    res = ds.replay(str(out), force=True, log=msgs.append)
    assert res["turns"] == len(before) and res["moves"] == len(before)
    after = {r["id"]: r for r in ds.iter_records(str(out))}
    assert after.keys() == before.keys()
    for i, r in after.items():
        b = before[i]
        assert r["traj"]["file"].startswith(f"traj/v{ds.TRAJ_VERSION}/")
        assert r["traj"]["version"] == ds.TRAJ_VERSION
        for k in ("markers", "poses", "moves", "reach", "timing", "warnings",
                  "unplayed", "reply_raw"):
            assert _no_clock(r[k]) == _no_clock(b[k]), k
        x, y = _arrays(out, b), _arrays(out, r)
        for w in x:
            assert np.array_equal(x[w], y[w], equal_nan=True), (i, w)
    # The old files are kept; the exports read the new ones.
    assert (out / "traj" / f"{b['session']}.npz").exists()
    notes = dx.annotate(list(after.values()), keep_lite=True)
    assert dx.action_pairs(str(out), list(after.values()), notes,
                           str(tmp_path / "act")) == len(after)


def test_replay_brings_version_1_records_up(moved, tmp_path):
    """A record from before moves played (replay record version 1, move
    marker unplayed, no traj.version): replayed, its move plays."""
    import shutil
    out = tmp_path / "old"
    shutil.copytree(moved, out)
    recs = ds.iter_records(str(out))
    for f in os.listdir(out / "traj"):
        if not f.endswith(".npz"):
            continue
        with np.load(out / "traj" / f) as z:
            arr = {k: z[k] for k in z.files}
        for k in [k for k in arr if k.endswith(".replay")]:
            rep = json.loads(str(arr[k]))
            arr[k] = np.array(json.dumps({"acts": rep["acts"],
                                          "tokens": []}))
        np.savez_compressed(out / "traj" / f, **arr)
    for f in os.listdir(out / "shards"):
        os.remove(out / "shards" / f)
    old = []
    for r in recs:
        r = copy.deepcopy(r)
        for k in ("moves", "play_warnings"):
            r.pop(k)
        r["traj"].pop("version")
        for m in r["markers"]:
            if m["tag"] == "move":
                m["played"] = False
        old.append(r)
    with open(out / "shards" / "turns-00000.jsonl", "w") as fh:
        for r in old:
            fh.write(json.dumps(r) + "\n")
    res = ds.replay(str(out), log=lambda s: None)
    assert res["turns"] == len(old) and res["moves"] == len(old)
    for r in ds.iter_records(str(out)):
        assert r["traj"]["version"] == ds.TRAJ_VERSION
        m = next(x for x in r["markers"] if x["tag"] == "move")
        assert m["played"] is True and r["moves"][0]["name"] == "clap"
        d = ds.load_traj(str(out), r, ".directed")
        assert d.shape == (r["timing"]["frames"], len(r["traj"]["vars"]))
        assert ds.load_traj(str(out), r, ".replay")["v"] == ds.REPLAY_VERSION


def test_replay_cli(moved, tmp_path):
    import shutil
    out = tmp_path / "cli"
    shutil.copytree(moved, out)
    assert ds.main(["replay", str(out), "--force", "--sessions",
                    "seed-san.cheerful.0000"]) == 0
    recs = ds.iter_records(str(out))
    v = {r["session"]: r["traj"]["file"] for r in recs}
    assert v["seed-san.cheerful.0000"].startswith("traj/v2/")
    assert not v["Hiyori.sleepy.0000"].startswith("traj/v2/")


def test_replay_on_a_live2d_rig(tmp_path):
    """The full EventRig path (gestures, the idle, the arm drawings): a
    replay of every turn gives the arrays back, a dropped gesture stays
    dropped."""
    from tests.test_pose import _body_rig
    model, rig, bm = _body_rig(tmp_path / "x")
    y = bm.landmarks["shoulder.y"] + 0.3
    reply = (f'Hi! [pose {{"hand.L.y": {y:.3f}}}] Look at this. '
             '[gesture:home] Nice. [move {"name": "sway", "beats": 2}] Ok.')
    opts = ds.parse_opts(["gen", "--avatars",
                          f"x:model={model},rig={rig},body={tmp_path}/x/"
                          "x.body.json,probe_dir=" + str(tmp_path),
                          "--sessions", "1", "--turns", "2",
                          "--personas", "cheerful",
                          "--out", str(tmp_path / "run")])
    res = ds.generate(opts, backend_factory=lambda role, av: CharLLM(
        [reply]) if role == "character" else UserLLM(), log=lambda s: None)
    assert res["ok"] == 1
    out = tmp_path / "run"
    before = {r["id"]: r for r in ds.iter_records(str(out))}
    for r in before.values():
        tags = {m["tag"]: m["played"] for m in r["markers"]}
        assert tags["gesture"] == "dropped" and tags["pose"] is True
        assert tags["move"] in (True, "unsolved")
    assert ds.replay(str(out), force=True, log=lambda s: None)["turns"] == 2
    for r in ds.iter_records(str(out)):
        b = before[r["id"]]
        assert _no_clock(r["markers"]) == _no_clock(b["markers"])
        x, y_ = _arrays(out, b), _arrays(out, r)
        for w in x:
            assert np.array_equal(x[w], y_[w], equal_nan=True), w


def test_planning_warnings_are_soft():
    from flow import dataexport as dx
    w = "gesture wave dropped: arms taken by move clap"
    r = {"spoken": "hi", "markers": [{"tag": "move"}], "warnings": [w],
         "play_warnings": [w]}
    f = dx.flag(r)
    assert "parse" not in f and "planned" in f
    r["warnings"].append("[pose {]: bad JSON (x); dropped")
    assert "parse" in dx.flag(r)


def test_unplayed_markers_stay_out_of_sft():
    from flow import dataexport as dx
    assert not dx.session_ok((["unplayed"], True))
    assert dx.session_ok((["no_pose"], True))
    assert dx.session_ok((["dup"], False))
