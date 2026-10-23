"""
flow.director: tagged replies -> event tokens. No network, no GL: language
models are faked (a fake `anthropic` module, a fake Ollama stream, fake
SiliconFlow/DeepSeek server-sent events, the file
backend).

The parser must give the same result however the reply is chunked and never
speak or crash on a bad tag; a tag's time is its place in the speech; a
directed move must own its role group for its duration and hold while the
idle stream stays exactly what the same seed schedules elsewhere; a refusal
writes no tokens; and the planned stream must replay bit for bit through
runtime --tokens-in.
"""

from __future__ import annotations

import io
import json
import sys
import types

import numpy as np
import pytest

from flow import director as dr
from flow import events as ev

FPS = 30.0
REPLY = ("[look:viewer] Oh, you're back! [brow:up] I was wondering "
         "[look:away] where you'd gone. [pause:0.5][nod] Yes. [gesture:m]")


def _chunks(text, sizes):
    out, i, k = [], 0, 0
    while i < len(text):
        n = sizes[k % len(sizes)]
        out.append(text[i:i + n])
        i += n
        k += 1
    return out


def test_parser_is_chunking_invariant():
    whole = dr.parse(REPLY, gestures=["m"])
    assert whole.spoken == ("Oh, you're back! I was wondering where you'd "
                            "gone. Yes.")
    assert [t["tag"] for t in whole.tags] == ["look", "brow", "look", "pause",
                                              "nod", "gesture"]
    assert whole.tags[1]["at"] == whole.spoken.index("I was")
    rng = np.random.default_rng(0)
    for sizes in ([1], [2, 5], [7], list(rng.integers(1, 12, 20))):
        p = dr.TagParser(["m"])
        got = []
        for c in _chunks(REPLY, sizes):
            got += p.feed(c)
        p.close()
        assert p.spoken == whole.spoken and p.tags == whole.tags, sizes
        assert got == p.tags          # each tag reported once, as it closes
        assert not p.warnings


def test_bad_tags_are_dropped_never_spoken():
    p = dr.parse("Hi [wave] there [look:sideways] you [nod:twice] "
                 "[gesture:dance] [pause:abc] [ look : LEFT ] ok ［tilt：left］ "
                 "end [shake", gestures=["idle"])
    assert p.spoken == "Hi there you ok end"
    assert [(t["tag"], t["arg"]) for t in p.tags] == [
        ("nod", None), ("look", "left"), ("tilt", "left")]
    w = " ".join(p.warnings)
    for s in ("[wave]", "sideways", "twice", "dance", "abc", "unterminated"):
        assert s in w
    # A bracket that never closes is not a tag: spoken, minus the bracket.
    p = dr.parse("a [" + "x" * 60 + " b")
    assert p.spoken.startswith("a xxx") and p.spoken.endswith(" b")
    # A second '[' before the first closed: the first is dropped.
    p = dr.parse("a [oops [nod] b")
    assert p.spoken == "a b" and p.tags[0]["tag"] == "nod"
    # Pauses are clipped, never negative.
    assert dr.parse("[pause:99]x").tags[0]["arg"] == str(dr.MAX_PAUSE)
    assert not dr.parse("[pause:-1]x").tags


def test_tag_time_is_its_place_in_the_speech():
    p = dr.parse("Hello world [nod] again[pause:1][blink][look:up] end")
    acts, spans, end = dr.timeline(p.spoken, p.tags, latin_rate=10.0)
    t = {a["tag"]: a["t"] for a in acts}
    assert t["nod"] == pytest.approx(len("Hello world ") / 10)
    # The pause delays the tags after it, and the speech.
    assert t["blink"] == pytest.approx(len("Hello world again") / 10 + 1.0)
    assert t["look"] == t["blink"]
    assert end == pytest.approx(len(p.spoken) / 10 + 1.0)
    assert "pause" not in t
    # CJK has its own rate; punctuation is silent but pauses.
    p = dr.parse("こんにちは。[nod]元気？")
    acts, spans, end = dr.timeline(p.spoken, p.tags, cjk_rate=5.0)
    assert acts[0]["t"] == pytest.approx(5 / 5 + dr.SENT_PAUSE)
    assert end == pytest.approx(7 / 5 + 2 * dr.SENT_PAUSE)
    segs = dr.subtitles(p.spoken, spans)
    assert [s["text"] for s in segs] == ["こんにちは。", "元気？"]
    assert segs[0]["end"] <= segs[1]["start"]


def test_srt_format():
    s = dr.srt([{"start": 0.0, "end": 1.25, "text": "Hi."},
                {"start": 61.5, "end": 3725.001, "text": "Bye."}])
    assert s == ("1\n00:00:00,000 --> 00:00:01,250\nHi.\n\n"
                 "2\n00:01:01,500 --> 01:02:05,001\nBye.\n\n")


def _st():
    return ev.load_stats("default")


def test_look_falls_back_to_the_head_when_gaze_is_unbound():
    """aidang_2's eyeballs are switches and not event roles: a look turns
    the head and never emits a saccade (nor crashes)."""
    st = _st()
    roles = ["head_x", "head_y", "head_z", "brow_l", "brow_r"]
    toks = dr.realize([{"t": 1.0, "tag": "look", "arg": "left"},
                       {"t": 3.0, "tag": "lean", "arg": "in"},
                       {"t": 3.5, "tag": "brow", "arg": "furrow"}],
                      st, roles, {}, np.random.default_rng(0))
    assert not any(k["type"] in ("saccade", "fixate", "body_shift")
                   for k in toks)
    (h,) = [k for k in toks if k["type"] == "head_turn"]
    assert h["t"] == 1.0 and h["target"]["head_x"] < 0
    assert set(h["target"]) == {"head_x", "head_y"}
    (b,) = [k for k in toks if k["type"] == "brow"]
    assert set(b["target"]) == {"brow_l", "brow_r"}
    assert all(v < 0 for v in b["target"].values())
    toks = dr.realize([{"t": 1.0, "tag": "look", "arg": "right"}], st,
                      list(ev.ALL_ROLES), {}, np.random.default_rng(0))
    s = [k for k in toks if k["type"] == "saccade"][0]
    R = st["roles"]["gaze_x"]
    assert s["target"]["gaze_x"] == pytest.approx(
        R["center"] + dr.FRAC["look_gaze"] * (R["hi"] - R["center"]))
    h = [k for k in toks if k["type"] == "head_turn"][0]
    assert h["t"] == pytest.approx(1.0 + st["head_follow"]["lag"])


def test_merge_overrides_touched_groups_only():
    st = _st()
    roles = list(ev.ALL_ROLES)
    idle = ev.schedule(st, roles, 30.0, np.random.default_rng(4))
    directed = dr.realize([{"t": 10.0, "tag": "nod", "arg": None},
                           {"t": 20.0, "tag": "brow", "arg": "up"}],
                          st, roles, {}, np.random.default_rng(1))
    merged = dr.merge(idle, directed)
    win = dr.windows(directed)
    assert set(win) == {"head", "brow"}
    for k in merged:
        if k.get("src") == "director" or k["type"] not in dr.MOVES:
            continue
        a, b = k["t"], k["t"] + k["duration"]
        assert not any(a <= w1 and b >= w0 for w0, w1 in win.get(k["role"], ()))
    # Untouched groups, blinks: exactly the idle stream of the same seed.
    for typ in ("saccade", "body_shift", "blink", "fixate"):
        assert ([k for k in merged if k["type"] == typ]
                == [k for k in idle if k["type"] == typ])
    assert merged == sorted(merged, key=lambda k: k["t"])
    for k in merged:              # follows points at a saccade, or nowhere
        if k["type"] == "head_turn" and k.get("follows") is not None:
            assert merged[k["follows"]]["type"] == "saccade"
    # Rendered: the nod goes down to its target and back, on time.
    y, _ = ev.render(merged, ["head_y"], 30.0, FPS)
    n0 = [k for k in directed if k["type"] == "head_turn"][0]
    f = int(round((10.0 + n0["duration"]) * FPS))
    assert y["head_y"][f] == pytest.approx(n0["target"]["head_y"], abs=1e-4)
    # Deterministic.
    again = dr.merge(idle, dr.realize(
        [{"t": 10.0, "tag": "nod", "arg": None},
         {"t": 20.0, "tag": "brow", "arg": "up"}], st, roles, {},
        np.random.default_rng(1)))
    assert again == merged


def test_directed_gesture_takes_the_arms():
    st = _st()
    idle = ev.schedule_gestures(st, [("a.motion3.json", 3.0),
                                     ("b.motion3.json", 3.0)], 20.0,
                                np.random.default_rng(0))
    directed = dr.realize([{"t": 5.0, "tag": "gesture", "arg": "A"}], st, [],
                          {"a.motion3.json": 4.0, "b.motion3.json": 3.0},
                          np.random.default_rng(0))
    assert directed[0]["motion"] == "a.motion3.json"
    merged = dr.merge(idle, directed)
    ges = [k for k in merged if k["type"] == "gesture"]
    assert not [k for k in ges if 5.0 <= k["t"] < 9.0 and "src" not in k]
    assert any(k["t"] >= 9.0 for k in ges)              # idle resumes after


class _FakeStream:
    def __init__(self, chunks, stop):
        self.chunks, self.stop = chunks, stop

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def text_stream(self):
        return iter(self.chunks)

    def get_final_message(self):
        u = types.SimpleNamespace(input_tokens=900, output_tokens=40,
                                  cache_creation_input_tokens=0,
                                  cache_read_input_tokens=850)
        return types.SimpleNamespace(stop_reason=self.stop, usage=u)


def _fake_anthropic(monkeypatch, chunks, stop="end_turn", reject=()):
    calls = []

    class Messages:
        def stream(self, **kw):
            for k in reject:
                if k in kw:
                    raise TypeError(f"stream() got an unexpected keyword "
                                    f"argument '{k}'")
            calls.append(kw)
            return _FakeStream(chunks, stop)

    class Anthropic:
        def __init__(self):
            self.beta = types.SimpleNamespace(messages=Messages())

    monkeypatch.setitem(sys.modules, "anthropic",
                        types.SimpleNamespace(Anthropic=Anthropic))
    return calls


def test_anthropic_request_shape_and_kwarg_fallback(monkeypatch):
    calls = _fake_anthropic(monkeypatch, ["[no", "d] Hel", "lo."],
                            reject=("fallbacks",))
    logs = []
    be = dr.AnthropicBackend(dr.DEFAULT_MODEL["anthropic"], "low",
                             log=logs.append)
    sysp = dr.system_prompt("Atago.", dr.gesture_labels(["home.motion3.json"]))
    msgs = [{"role": "user", "content": "hi"}]
    r = dr.ask(be, sysp, msgs, ["home"], log=logs.append)
    assert r["spoken"] == "Hello." and r["tags"][0]["tag"] == "nod"
    (kw,) = calls
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["system"][0]["text"] == sysp
    assert kw["output_config"] == {"effort": "low"}
    assert kw["betas"] == [dr.FALLBACK_BETA]
    assert kw["extra_body"] == {"fallbacks": "default"}
    assert "fallbacks" not in kw
    for k in ("thinking", "temperature", "top_p", "budget_tokens"):
        assert k not in kw
    assert kw["messages"][-1]["role"] == "user"          # no prefill
    assert r["usage"]["cache_read_input_tokens"] == 850
    assert any("cache_read_input_tokens 850" in s for s in logs)


def test_system_prompt_is_byte_stable_and_lists_the_rigs_motions():
    labels = dr.gesture_labels(["Home.motion3.json", "login.motion3.json"])
    a = dr.system_prompt("Atago.", labels)
    assert a == dr.system_prompt("Atago.", dict(reversed(labels.items())))
    assert "[gesture:NAME]" in a and "home, login" in a
    assert "[gesture:" not in dr.system_prompt("Atago.", {})


def test_refusal_writes_no_tokens(monkeypatch, tmp_path):
    _fake_anthropic(monkeypatch, ["[nod] Sure, here is"], stop="refusal")
    be = dr.AnthropicBackend("m", log=lambda s: None)
    r = dr.ask(be, "sys", [{"role": "user", "content": "x"}], [],
               log=lambda s: None)
    assert r["refused"] and r["spoken"] == "" and r["tags"] == []
    a = types.SimpleNamespace()
    assert dr.plan(a, r, str(tmp_path / "t"), {}, log=lambda s: None) is None
    assert sorted(p.name for p in (tmp_path / "t").iterdir()) == ["refusal.txt"]


def test_ollama_stream(monkeypatch):
    import urllib.request
    lines = [{"message": {"role": "assistant", "content": "[tilt:"}},
             {"message": {"role": "assistant", "content": "left] Hm?"}},
             {"message": {"content": ""}, "done": True, "done_reason": "stop",
              "eval_count": 5}]
    seen = {}

    def urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        return io.BytesIO(b"".join(json.dumps(x).encode() + b"\n"
                                   for x in lines))

    # Ollama goes through its own proxy-less opener.
    monkeypatch.setattr(urllib.request, "build_opener",
                        lambda *h: types.SimpleNamespace(open=urlopen))
    be = dr.OllamaBackend("m", "http://h:1/", seed=3, log=lambda s: None)
    r = dr.ask(be, "SYS", [{"role": "user", "content": "q"}], [],
               log=lambda s: None)
    assert r["spoken"] == "Hm?" and r["tags"][0]["arg"] == "left"
    assert seen["url"] == "http://h:1/api/chat"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["body"]["stream"] is True and seen["body"]["options"]["seed"] == 3


def _sse(events):
    return io.BytesIO(b"".join(b"data: " + json.dumps(e).encode() + b"\n\n"
                               for e in events) + b"data: [DONE]\n\n")


def test_openai_compat_stream_skips_reasoning(monkeypatch):
    import urllib.request
    ev_ = [{"choices": [{"delta": {"reasoning_content": "[nod] thinking"}}]},
           {"choices": [{"delta": {"content": "[look:"}}]},
           {"choices": [{"delta": {"content": "up] Oh?"},
                         "finish_reason": "stop"}]},
           {"choices": [], "usage": {"prompt_tokens": 9,
                                     "completion_tokens": 4}}]
    seen = {}

    def urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return _sse(ev_)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "k")
    be = dr.OpenAICompatBackend("siliconflow", "m", log=lambda s: None)
    r = dr.ask(be, "SYS", [{"role": "user", "content": "q"}], [],
               log=lambda s: None)
    assert r["spoken"] == "Oh?" and [t["arg"] for t in r["tags"]] == ["up"]
    assert seen["url"] == "https://api.siliconflow.com/v1/chat/completions"
    assert seen["auth"] == "Bearer k" and seen["body"]["stream"] is True
    assert seen["body"]["messages"][0] == {"role": "system", "content": "SYS"}
    assert be.stop_reason == "stop" and be.usage["completion_tokens"] == 4


def test_openai_compat_content_filter_is_a_refusal(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None:
                        _sse([{"choices": [{"delta": {"content": "Hel"},
                                            "finish_reason": "content_filter"}]}]))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    be = dr.OpenAICompatBackend("deepseek", "m", log=lambda s: None)
    r = dr.ask(be, "SYS", [{"role": "user", "content": "q"}], [],
               log=lambda s: None)
    assert r["refused"] and r["spoken"] == ""


def test_openai_compat_4xx_is_fatal_and_shows_body(monkeypatch):
    import urllib.error
    import urllib.request
    calls = []

    def urlopen(req, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(b'{"code":20015,"message":"bad"}'))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    be = dr.OpenAICompatBackend("deepseek", "m", log=lambda s: None)
    with pytest.raises(RuntimeError, match="20015"):
        list(be.stream("SYS", []))
    assert len(calls) == 1


def _rig(root):
    """A tiny rig in a folder named like the Azur Lane collection (spaces,
    brackets), with one motion that moves the arm."""
    from l2dprobe.analyze import group_of
    root.mkdir(parents=True)
    arm = [0, 0, 1, 2.0, 1, 4.0, 0, 6.0, 0]
    motion = {"Version": 3, "Meta": {"Duration": 6.0, "Fps": 30},
              "Curves": [{"Target": "Parameter", "Id": "ParamArmL",
                          "Segments": arm}]}
    (root / "m.motion3.json").write_text(json.dumps(motion))
    (root / "home.motion3.json").write_text(json.dumps(motion))
    model = root / "x.model3.json"
    model.write_text(json.dumps({"Version": 3, "FileReferences": {
        "Moc": "x.moc3", "Motions": {"": [{"File": "m.motion3.json"},
                                          {"File": "home.motion3.json"}]}}}))
    pids = ["ParamAngleX", "ParamAngleY", "ParamEyeBallX", "ParamBrowLY",
            "ParamArmL", "ParamEyeLOpen", "ParamBodyAngleY"]
    n = len(pids)
    rig = root / "x.npz"
    np.savez(rig, pid=np.array(pids, dtype=object),
             kind=np.array(["continuous"] * n, dtype=object),
             group=np.array([group_of(p) for p in pids], dtype=object),
             minimum=np.array([-30, -30, -1, -1, 0, 0, -10], np.float32),
             maximum=np.array([30, 30, 1, 1, 1, 1, 10], np.float32),
             default=np.array([0, 0, 0, 0, 0, 1, 0], np.float32),
             descriptor=np.zeros((n, 4), np.float32))
    return model, rig


def test_file_backend_end_to_end_replays_bit_for_bit(tmp_path, capsys):
    from flow.runtime import main as runtime_main
    model, rig = _rig(tmp_path / "Azue Lane(JP)" / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text(REPLY.replace("[gesture:m]", "[gesture:home] [bogus]"),
                     encoding="utf-8")
    out = tmp_path / "run"
    args = ["--model-json", str(model), "--rig", str(rig), "--backend", "file",
            "--reply-file", str(reply), "--out", str(out), "--seed", "5",
            "--probe-dir", str(tmp_path)]
    assert dr.main(args) == 0
    log = capsys.readouterr().out
    assert "unknown tag [bogus] dropped" in log
    for f in ("reply.txt", "spoken.txt", "actions.json", "subs.srt",
              "subs.json", "tokens.json", "plan.npz"):
        assert (out / f).exists(), f
    blob = json.loads((out / "tokens.json").read_text())
    toks = blob["tokens"]
    mine = [k for k in toks if k.get("src") == "director"]
    assert {k["type"] for k in mine} >= {"saccade", "head_turn", "brow",
                                         "gesture"}
    assert [k["motion"] for k in mine if k["type"] == "gesture"] == [
        "home.motion3.json"]
    assert any(k.get("src") is None for k in toks)      # the idle is there
    # The printed render command, as a dump-only replay.
    cmd = [ln for ln in log.splitlines() if "--tokens-in" in ln][0]
    import shlex
    argv = shlex.split(cmd.strip())
    argv = argv[argv.index("flow.runtime") + 1:]
    k = argv.index("--out")
    argv = argv[:k] + argv[k + 2:]
    assert runtime_main(argv + ["--dump-only", "--dump",
                                str(tmp_path / "replay.npz")]) == 0
    a = np.load(out / "plan.npz", allow_pickle=True)
    b = np.load(tmp_path / "replay.npz", allow_pickle=True)
    np.testing.assert_array_equal(a["traj"], b["traj"])
    np.testing.assert_array_equal(a["part_opacity"], b["part_opacity"])
    # Same seed, same reply: the same tokens.
    assert dr.main(args[:-6] + ["--out", str(tmp_path / "again"), "--seed",
                                "5", "--probe-dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "again" / "tokens.json").read_text()) == blob
    srt = (out / "subs.srt").read_text(encoding="utf-8")
    assert srt.startswith("1\n00:00:00,000 --> ") and "Oh, you're back!" in srt


def test_no_idle_keeps_only_blinks_and_the_tags(tmp_path):
    model, rig = _rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text(REPLY.replace("[gesture:m]", "[gesture:home]"),
                     encoding="utf-8")
    out = tmp_path / "run"
    assert dr.main(["--model-json", str(model), "--rig", str(rig),
                    "--backend", "file", "--reply-file", str(reply),
                    "--out", str(out), "--probe-dir", str(tmp_path),
                    "--no-idle"]) == 0
    toks = json.loads((out / "tokens.json").read_text())["tokens"]
    assert any(k.get("src") == "director" for k in toks)
    assert all(k.get("src") == "director" or k["type"] == "blink"
               for k in toks)


def _stats_like(body_y=None):
    st = _st()
    if body_y is not None:
        st["roles"]["body_y"].update(body_y)
    return st


def test_offer_leaves_out_tags_the_rig_cannot_show():
    st = _st()
    everything, out = dr.offer(st, list(ev.ALL_ROLES))
    assert everything == dr.all_tags() and out == []
    labels = {"home": "home.motion3.json"}
    assert dr.system_prompt("A.", labels, everything) == dr.system_prompt(
        "A.", labels)
    # aidang_2 (server run): body_y adapted centre 0.932, p98 0.976 --
    # lean:in went to 0.963, +0.031. lean:back still reaches -0.36. Gaze is
    # a switch there: looks turn the head.
    aidang = _stats_like({"center": 0.932, "lo": -0.912, "hi": 0.976})
    roles = [r for r in ev.ALL_ROLES if not r.startswith("gaze")]
    offered, out = dr.offer(aidang, roles)
    assert [k for k, _ in out] == ["lean:in"]
    assert {"lean:back", "look:left", "look:away", "nod"} <= set(offered)
    msg = dr.left_out_text(out)
    assert msg.startswith("lean:in (body_y centre 0.93, no room")
    assert "+0.031" in msg
    # A p98 blended below the centre is no room either (the check is signed).
    below = _stats_like({"center": 0.9, "lo": -0.9, "hi": 0.5})
    assert "lean:in" not in dr.offer(below, roles)[0]
    # Natori: the smallest lean of the sample rigs (adapted p98 0.32) stays.
    natori = _stats_like({"center": 0.003, "lo": -0.398, "hi": 0.323})
    assert "lean:in" in dr.offer(natori, list(ev.ALL_ROLES))[0]
    # lafei: no body_y bound. xuefeng: no brows either (switches).
    lafei = [r for r in ev.ALL_ROLES if r != "body_y"]
    assert dr.left_out_text(dr.offer(st, lafei)[1]) == \
        "lean:* (no body_y role)"
    xuefeng = [r for r in lafei if ev.GROUP_OF[r] != "brow"]
    offered, out = dr.offer(st, xuefeng)
    assert dr.left_out_text(out) == "brow:* (no brow roles), " \
        "lean:* (no body_y role)"
    # A bound brow with no fitted range (Natori's BrowLX/RX) is no brow.
    flat = _st()
    flat["roles"]["brow_lx"].update(lo=0.0, hi=0.0, std=0.0)
    assert "brow:up" not in dr.offer(flat, ["head_x", "brow_lx"])[0]
    # The prompt never shows a tag that is left out, in the list, the rules
    # or the examples; it is byte-stable whatever order the keys come in.
    p = dr.system_prompt("A.", labels, offered)
    assert p == dr.system_prompt("A.", labels, tuple(reversed(offered)))
    for k, _ in out:
        assert f"[{k}]" not in p
    assert "brow" not in p and "lean" not in p
    assert "[look:away] Forget it." in p and "[nod]" in p
    # Only blink and pause left: the stand-in example.
    bare, out = dr.offer(st, [])
    assert set(bare) == {"blink", "pause", "gesture"}
    p = dr.system_prompt("A.", {}, bare)
    assert dr.FALLBACK_EXAMPLE in p and "[nod]" not in p
    assert "look:*" in dr.left_out_text(out)


def test_parser_drops_tags_not_offered():
    offered, _ = dr.offer(_st(), [r for r in ev.ALL_ROLES if r != "body_y"])
    p = dr.parse("[lean:in] Hi [nod] there [pause:0.3][lean:back] you "
                 "[gesture:m]", gestures=["m"], offered=offered)
    assert p.spoken == "Hi there you"
    assert [t["tag"] for t in p.tags] == ["nod", "pause", "gesture"]
    assert sum("not offered" in w for w in p.warnings) == 2
    # Without `offered` the whole vocabulary parses, as before.
    assert len(dr.parse("[lean:in] Hi").tags) == 1


def test_main_prompt_follows_the_rigs_bound_roles(tmp_path, capsys):
    """The fake rig has no ParamAngleZ or ParamEyeBallY: no tilt; looks up
    and down still turn the head."""
    model, rig = _rig(tmp_path / "x")
    assert dr.main(["--model-json", str(model), "--rig", str(rig),
                    "--out", str(tmp_path / "o"), "--probe-dir",
                    str(tmp_path), "--print-system"]) == 0
    cap = capsys.readouterr()
    assert "not offered on this rig: tilt:* (no head_z role)" in cap.err
    assert "[tilt:" not in cap.out and "[look:up]" in cap.out
    assert cap.out.startswith("You are a friendly")


def test_openai_compat_thinking_off_is_opt_in(monkeypatch):
    import urllib.request
    bodies = []

    def urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        return _sse([{"choices": [{"delta": {"content": "Hi."},
                                   "finish_reason": "stop"}]}])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "k")
    for thinking in (None, "off"):
        be = dr.OpenAICompatBackend("siliconflow", "m", log=lambda s: None,
                                    thinking=thinking)
        list(be.stream("SYS", []))
    assert "enable_thinking" not in bodies[0]
    assert bodies[1]["enable_thinking"] is False
