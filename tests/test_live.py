"""
flow.stream and flow.live: the live director. No GL, no network: the
language model, the TTS and the renderer are faked.

The stream must draw what the offline runtime draws for the same tokens,
frame for frame; the idle must run on across window seams and directed
tokens must land in its future without a jump; sentences must be cut from
streamed chunks however they are split; each sentence's audio, mouth and
tokens must sit at one server time; the voice must follow the mood; and
the web server must serve the page, take a line and stream frames and
audio.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from flow import director as dr
from flow import events as ev
from flow import runtime as rt
from flow import stream as fs
from tests.test_speech import two_phrases, wav_bytes

FPS = 30.0


def _plan(tmp_path, extra=()):
    """A director --tts file run on the test rig with a mouth: (model, rig,
    tokens.json blob, plan.npz from a --tokens-in --dump-only replay)."""
    from tests.test_speech import _mouth_rig
    model, rig = _mouth_rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text("Hello there. [nod] How are you? [gesture:m]",
                     encoding="utf-8")
    audio = tmp_path / "said.wav"
    audio.write_bytes(wav_bytes(two_phrases()))
    out = tmp_path / "run"
    assert dr.main(["--model-json", str(model), "--rig", str(rig),
                    "--backend", "file", "--reply-file", str(reply),
                    "--out", str(out), "--probe-dir", str(tmp_path),
                    "--seed", "3", "--tts", "file", "--tts-file", str(audio),
                    "--tail", "6", *extra]) == 0
    blob = json.loads((out / "tokens.json").read_text())
    plan = np.load(out / "plan.npz", allow_pickle=True)
    return model, rig, blob, plan


def _stream_plan(rig, blob, n):
    """The stream's frames as runtime --dump records them: the expressive
    columns normalised, the mouth, part opacities."""
    st = fs.MotionStream(rig, tokens=blob["tokens"], autonomic=False,
                         log=lambda s: None)
    mt = next((k for k in blob["tokens"] if k["type"] == "mouth"), None)
    if mt is not None:
        st.add_mouth(0, mt, n)
    ex, pids, lo = rig.expressive, rig.pids, rig.lo
    span = np.maximum(rig.hi[ex] - lo[ex], 1e-6)
    traj = np.empty((n, len(ex)), np.float32)
    frames = []
    for f in range(n):
        over, parts = st.frame()
        for c, i in enumerate(ex):
            traj[f, c] = ((over[pids[i]] - lo[i]) / span[c] * 2 - 1
                          if pids[i] in over else rig.default_n[c])
        frames.append((over, parts))
    np.clip(traj, -1, 1, out=traj)
    return traj, frames


def test_stream_draws_what_the_runtime_draws(tmp_path, capsys):
    """Fed the runtime's own merged tokens (idle, directed, underlay,
    mouth), the stream's per-frame writes equal runtime --tokens-in
    --dump-only's plan: the live render is the offline render, a frame at a
    time."""
    model, rig_path, blob, plan = _plan(tmp_path)
    capsys.readouterr()
    rig = fs.EventRig(str(model), str(rig_path), "default", None, None, FPS,
                      str(tmp_path), log=lambda s: None)
    n = len(plan["traj"])
    traj, frames = _stream_plan(rig, blob, n)
    np.testing.assert_array_equal(traj, plan["traj"])
    mo = np.array([[fr[0][p] for p in plan["mouth_pid"]] for fr in frames],
                  np.float32)
    np.testing.assert_array_equal(mo, plan["mouth"])
    parts = list(plan["part_id"])
    if parts:
        op = np.array([[fr[1].get(p, np.nan) for p in parts]
                       for fr in frames], np.float32)
        np.testing.assert_array_equal(op, plan["part_opacity"])


def test_stream_matches_runtime_on_hiyori(tmp_path):
    """The same on a real rig (switches, part swaps, the calm underlay),
    when the Cubism samples are there: the idle the runtime schedules for a
    seed, replayed through the stream."""
    base = os.environ.get("TSUKUYOMI_SAMPLES")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = None
    for cand in ("Samples/Resources", "Resources", ""):
        p = os.path.join(base or "/nonexistent", cand, "Hiyori",
                         "Hiyori.model3.json")
        if os.path.exists(p):
            model = p
            break
    rig_path = os.path.join(root, "rigs", "Hiyori.npz")
    clips = os.path.join(root, "clips2", "Hiyori.npz")
    if model is None or not os.path.exists(rig_path):
        pytest.skip("TSUKUYOMI_SAMPLES with Hiyori and rigs/Hiyori.npz needed")
    tok, dump = tmp_path / "t.json", tmp_path / "p.npz"
    args = ["--model", model, "--rig", rig_path, "--clips", clips,
            "--events", "default", "--probe-dir", str(tmp_path), "--seed",
            "4", "--seconds", "40"]
    assert rt.main(args + ["--tokens", str(tok), "--dump-only", "--dump",
                           str(tmp_path / "a.npz")]) == 0
    assert rt.main(args + ["--tokens-in", str(tok), "--dump-only", "--dump",
                           str(dump)]) == 0
    plan = np.load(dump, allow_pickle=True)
    blob = json.loads(tok.read_text())
    rig = fs.EventRig(model, rig_path, "default", clips, None, FPS,
                      str(tmp_path), log=lambda s: None)
    traj, frames = _stream_plan(rig, blob, len(plan["traj"]))
    np.testing.assert_array_equal(traj, plan["traj"])
    parts = list(plan["part_id"])
    op = np.array([[fr[1].get(p, np.nan) for p in parts] for fr in frames],
                  np.float32)
    np.testing.assert_array_equal(op, plan["part_opacity"])


def _rig(tmp_path):
    from tests.test_speech import _mouth_rig
    model, rig = _mouth_rig(tmp_path / "x")
    return fs.EventRig(str(model), str(rig), "default", None, None, FPS,
                       str(tmp_path), log=lambda s: None)


def _roles(frames, rig):
    ex, pids, lo = rig.expressive, rig.pids, rig.lo
    cols = {r: rig.col_of[p] for r, (p, _, _) in rig.roles.items()}
    out = {}
    for r, c in cols.items():
        i = ex[c]
        out[r] = np.array([(fr[0][pids[i]] - lo[i]) / rig.span_e[c] * 2 - 1
                           for fr in frames])
    return out


def test_streamed_idle_is_the_one_shot_idle(tmp_path):
    """The idle generated in 20 s windows (refilled 10 s ahead) renders
    exactly what the same schedulers planned in one call render: the
    window seams do not exist in the motion."""
    rig = _rig(tmp_path)
    n = int(70 * FPS)
    live = fs.MotionStream(rig, seed=7, autonomic=False, log=lambda s: None)
    got = [live.frame() for _ in range(n)]
    sc = ev.IdleScheduler(rig.stats, list(rig.roles),
                          np.random.default_rng(7), rig.start)
    toks = sc.advance(80.0)
    toks += ev.GestureScheduler(sc.st, rig.gest,
                                np.random.default_rng([7, 2])).advance(80.0)
    toks.append(ev.underlay_token(rig.stats, rig.settle, 1e9))
    for k in toks:
        if "follows" in k:
            k["follows"] = None
    once = fs.MotionStream(rig, tokens=toks, autonomic=False,
                           log=lambda s: None)
    want = [once.frame() for _ in range(n)]
    for (o1, p1), (o2, p2) in zip(got, want):
        assert o1 == o2 and p1 == p2
    assert live.ahead >= 70 and sum(k["type"] == "gesture" for k in toks) > 0


def test_directed_tokens_merge_into_the_future_without_a_jump(tmp_path):
    """A reply's tokens, realized at a time a little ahead, displace the
    idle moves they overlap on their group and play from where the role is:
    no frame-to-frame jump, the nod reaches its target, and nothing
    already shown changes."""
    rig = _rig(tmp_path)
    logs = []
    s = fs.MotionStream(rig, seed=1, autonomic=False, log=logs.append)
    ref = fs.MotionStream(rig, seed=1, autonomic=False, log=lambda x: None)
    frames, base = [], []
    rng = np.random.default_rng([1, 1])
    for f in range(int(30 * FPS)):
        if f == int(10 * FPS):
            acts = [{"t": 0.4, "tag": "nod", "arg": None},
                    {"t": 1.5, "tag": "look", "arg": "left"},
                    {"t": 2.0, "tag": "brow", "arg": "up"}]
            t0 = (f + 6) / FPS
            toks = dr.realize([dict(a, t=a["t"] + t0) for a in acts],
                              rig.stats, list(rig.roles),
                              rig.motion_seconds, rng, s.pose(),
                              settle=rig.settle)
            s.direct(toks)
        frames.append(s.frame())
        base.append(ref.frame())
    # The past is untouched: up to the merge both streams are one.
    for (a, _), (b, _) in zip(frames[:300], base[:300]):
        assert a == b
    tr, br = _roles(frames, rig), _roles(base, rig)
    for r, y in tr.items():
        assert np.abs(np.diff(y)).max() < 0.2, r
    assert any(np.abs(tr[r] - br[r]).max() > 0.05 for r in tr)
    y = tr["head_y"][300:420]
    nod = dr._toward(rig.stats, "head_y", -dr.SIGN["up"], dr.FRAC["nod"])
    assert y.min() == pytest.approx(nod, abs=0.02) or \
        any("late" in m for m in logs)
    # The directed window owns the head: no idle head move starts in it.
    assert not [k for k in s.pending if k["type"] == "head_turn"
                and k.get("src") != "director" and 10.2 < k["t"] < 11.0]


# ------------------------------------------------------------------ mood

from flow import speech as sp                                 # noqa: E402

TOBARI = [(118, "夜語トバリ", "ノーマル"), (119, "夜語トバリ", "明るい"),
          (120, "夜語トバリ", "哀しみ"), (121, "夜語トバリ", "呆れ")]
METAN = [(2, "四国めたん", "ノーマル"), (0, "四国めたん", "あまあま"),
         (6, "四国めたん", "ツンツン"), (4, "四国めたん", "セクシー"),
         (36, "四国めたん", "ささやき"), (37, "四国めたん", "ヒソヒソ")]
ZUNDA = [(3, "ずんだもん", "ノーマル"), (1, "ずんだもん", "あまあま"),
         (7, "ずんだもん", "ツンツン"), (22, "ずんだもん", "ささやき"),
         (75, "ずんだもん", "なみだめ"), (76, "ずんだもん", "ヘロヘロ")]


def test_mood_of_reads_the_tags_words_and_punctuation():
    def m(text):
        p = dr.parse(text)
        return sp.mood_of(p.spoken, p.tags)[0]
    assert m("[brow:down] I see. That's a shame.") == "sad"
    assert m("[brow:furrow] You did what?") == "annoyed"
    assert m("[brow:up] Really? You came back!") == "surprised"
    assert m("[look:away] Um... that's embarrassing.") == "shy"
    assert m("ふぁ……ねむい……") == "sleepy"
    assert m("えっ、まさか！") == "surprised"
    assert m("Yay, thank you!") == "happy"
    assert m("I'm so sorry...") == "sad"
    assert m("Hmph! Whatever!") == "annoyed"
    assert m("[nod] Okay.") == "neutral"          # a nod alone tips nothing
    assert m("It is three o'clock.") == "neutral"
    assert m("ごめんね…") == "sad"
    assert m("べつに、あなたのためじゃないんだから！") == "annoyed"
    assert m("ふんわりしたパンが好き。") == "happy"       # not "ふんっ"
    assert m("ねえ、内緒だよ。[lean:in]") == "whisper"
    p = dr.parse("Oh no. [brow:down] I lost it... [brow:up] Wait, found "
                 "it! Yay!")
    assert sp.sentence_moods(p.spoken, p.tags) == [
        (0, "neutral"), (7, "sad"), (p.spoken.index("Wait"), "surprised"),
        (p.spoken.index("Yay"), "happy")]


def test_voicevox_mood_styles_by_name_per_character():
    ms = sp.mood_style
    assert [ms(TOBARI, "夜語トバリ", m, 118) for m in
            ("neutral", "happy", "sad", "annoyed", "whisper")] \
        == [118, 119, 120, 121, 118]
    assert [ms(METAN, "四国めたん", m, 2) for m in
            ("happy", "sad", "annoyed", "whisper")] == [0, 2, 6, 36]
    # Only the character's own styles: ずんだもん's are not めたん's.
    assert ms(METAN + ZUNDA, "ずんだもん", "whisper", 3) == 22
    assert ms(METAN + ZUNDA, "ずんだもん", "annoyed", 3) == 7


USAGI = [(61, "中国うさぎ", "ノーマル"), (62, "中国うさぎ", "おどろき"),
         (63, "中国うさぎ", "こわがり"), (64, "中国うさぎ", "へろへろ")]
MIKO = [(43, "櫻歌ミコ", "ノーマル"), (44, "櫻歌ミコ", "第二形態"),
        (45, "櫻歌ミコ", "ロリ")]


def test_server_voices_lafei_and_xuefeng():
    """The voices picked by ear on the server: lafei 中国うさぎ/ノーマル
    (61), whose styles fit Laffey's moods; xuefeng 櫻歌ミコ/第二形態
    (44), which has no emotional style, so every mood keeps 44 and goes
    to the AudioQuery offsets instead."""
    ms = sp.mood_style
    assert [ms(USAGI, "中国うさぎ", m, 61) for m in
            ("surprised", "shy", "sleepy", "happy", "sad")] \
        == [62, 63, 64, 61, 61]
    assert {ms(MIKO, "櫻歌ミコ", m, 44) for m in sp.MOODS} == {44}
    q = sp.voicevox_mood_query({"speedScale": 1.1, "pitchScale": 0.0,
                                "intonationScale": 1.0, "volumeScale": 1.0},
                               sp.VOICEVOX_MOODS["sad"])
    assert q == pytest.approx({"speedScale": 0.99, "pitchScale": -0.03,
                               "intonationScale": 0.8, "volumeScale": 0.9})
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rig, voice in (("lafei", "中国うさぎ/ノーマル"),
                       ("xuefeng", "櫻歌ミコ/第二形態")):
        cfg = sp.load_voice_config(os.path.join(root, "rigs",
                                                f"{rig}.voice.json"))
        assert cfg["ja"]["tts"] == "voicevox" and cfg["ja"]["voice"] == voice
        assert cfg["en"]["tts"] == "edge"


def test_voicevox_speaks_each_mood_in_its_style():
    from tests.test_voicevox import SPEAKERS, Engine
    eng = Engine()
    speakers = SPEAKERS + [{"name": "夜語トバリ", "styles": [
        {"name": n, "id": i, "type": "talk"} for i, _, n in TOBARI]}]
    import tests.test_voicevox as tv
    old = tv.SPEAKERS
    tv.SPEAKERS = speakers
    try:
        logs = []
        t = sp.VoicevoxTTS("http://127.0.0.1:50021", "夜語トバリ/ノーマル",
                           log=logs.append, opener=eng)
        t.synth("こんにちは、今日は良い天気ですね。", mood="sad")
        t.synth("こんにちは、今日は良い天気ですね。", mood="whisper")
        t.synth("こんにちは、今日は良い天気ですね。")
        ids = [(c[1], c[2]["speaker"]) for c in eng.calls
               if c[1] in ("/audio_query", "/synthesis")]
        assert ids == [("/audio_query", "120"), ("/synthesis", "120"),
                       ("/audio_query", "118"), ("/synthesis", "118"),
                       ("/audio_query", "118"), ("/synthesis", "118")]
        assert t.speaker == 118 and t.credit == "VOICEVOX:夜語トバリ"
        assert any("mood sad -> 夜語トバリ/哀しみ (120)" in s for s in logs)
        # The voice config names a style; false turns moods off.
        t2 = sp.VoicevoxTTS("http://x:1", "夜語トバリ/ノーマル",
                            moods={"happy": "呆れ"}, opener=Engine(),
                            log=lambda s: None)
        assert t2.mood_id("happy") == 121 and t2.mood_id("sad") == 120
        t3 = sp.VoicevoxTTS("http://x:1", "夜語トバリ/ノーマル", moods=False,
                            opener=Engine(), log=lambda s: None)
        assert t3.mood_id("sad") == 118
        # No style for the mood: the base one with the query offsets.
        t4 = sp.VoicevoxTTS("http://x:1", "夜語トバリ/ノーマル",
                            moods={"sleepy": {"speed": 0.5}}, opener=eng,
                            log=lambda s: None)
        assert t4.mood_id("sleepy") == 118
        t4.synth("こんにちは、今日は良い天気ですね。", mood="sleepy")
        last = eng.calls[-1]
        assert last[1] == "/synthesis" and last[2]["speaker"] == "118"
        assert json.loads(last[3])["speedScale"] == 0.5
    finally:
        tv.SPEAKERS = old


def test_edge_mood_offsets_are_what_edge_tts_validates(monkeypatch):
    import re
    assert sp.edge_prosody("+0%", None) == ("+0%", "+0Hz", "+0%")
    assert sp.edge_prosody("+10%", "sad") == ("+0%", "-10Hz", "-10%")
    assert sp.edge_prosody("-5%", "happy") == ("+1%", "+12Hz", "+0%")
    assert sp.edge_prosody(None, "happy", {"happy": {"rate": "+20%",
                                                     "pitch": "+3Hz"}}) \
        == ("+20%", "+3Hz", "+0%")
    for m in sp.MOODS:
        r, p, v = sp.edge_prosody("+3%", m)
        assert re.match(r"^[+-]\d+%$", r) and re.match(r"^[+-]\d+Hz$", p)
        assert re.match(r"^[+-]\d+%$", v)
    seen = []

    class Communicate:
        def __init__(self, text, voice, *, rate="+0%", volume="+0%",
                     pitch="+0Hz", boundary="SentenceBoundary"):
            seen.append(dict(rate=rate, pitch=pitch, volume=volume))

        def stream_sync(self):
            yield {"type": "audio", "data": wav_bytes(two_phrases())}

    import sys
    import types
    fake = types.ModuleType("edge_tts")
    fake.Communicate = Communicate
    monkeypatch.setitem(sys.modules, "edge_tts", fake)
    tts = sp.EdgeTTS("en-US-AnaNeural", "+0%", log=lambda s: None)
    p = dr.parse("Hello there. [brow:down] I miss you.")
    said = sp.speak(p.spoken, p.tags, tts, lambda ch: 1 / 14,
                    log=lambda s: None,
                    moods=sp.sentence_moods(p.spoken, p.tags))
    assert seen == [dict(rate="+0%", pitch="+0Hz", volume="+0%"),
                    dict(rate="-10%", pitch="-10Hz", volume="-10%")]
    assert said["moods"] == [(0, "neutral"), (13, "sad")]
    # The [brow:down] opens the second piece: its time is that piece's.
    assert said["acts"][0]["t"] >= len(two_phrases()) / 24000 - 1e-6


def test_one_neutral_mood_makes_the_calls_of_before():
    from tests.test_speech import FakeTTS
    p = dr.parse("Hello there. [nod] How are you? [pause:0.4] Fine.")
    calls = {}
    for moods in (None, sp.sentence_moods(p.spoken, p.tags)):
        f = FakeTTS([(two_phrases(), None)] * 3)
        said = sp.speak(p.spoken, p.tags, f, lambda ch: 1 / 14,
                        log=lambda s: None, moods=moods)
        calls[moods is None] = (f.texts, said["acts"], said["end"])
    assert calls[True] == calls[False]
    assert calls[True][0] == ["Hello there. How are you?", "Fine."]


# ------------------------------------------------------------- sentences

from flow import live as lv                                   # noqa: E402


def _cut(text, sizes):
    from tests.test_director import _chunks
    c = lv.SentenceCutter()
    out = []
    for ch in _chunks(text, sizes):
        out += c.feed(ch)
    return out + c.close()


CUTS = [
    ("Hello there. [nod] How are you? I'm fine... Really!",
     ["Hello there.", " [nod] How are you?", " I'm fine...", " Really!"]),
    ("It costs 3.14 dollars, e.g.this. [pause:0.5] OK",
     ["It costs 3.14 dollars, e.g.this.", " [pause:0.5] OK"]),
    ("こんにちは。[nod]元気？うん……そうだね！[gesture:m]",
     ["こんにちは。", "[nod]元気？", "うん……", "そうだね！", "[gesture:m]"]),
    ("He said \"no.\" Then left?! [shake]",
     ["He said \"no.\"", " Then left?!", " [shake]"]),
    ("「本当？」って聞いた。Yes.", ["「本当？」", "って聞いた。", "Yes."]),
    ("Line one\nline two", ["Line one\n", "line two"]),
]


@pytest.mark.parametrize("text,want", CUTS)
def test_sentences_are_cut_from_streamed_chunks(text, want):
    """However the reply is chunked -- a tag, a "..." or a "?!" split
    across chunks -- the cuts are the same, and nothing is lost."""
    rng = np.random.default_rng(len(text))
    for sizes in ([1], [2, 5], [7], [len(text)],
                  list(rng.integers(1, 9, 30))):
        got = _cut(text, sizes)
        assert got == want, sizes
        assert "".join(got) == text


def test_a_long_first_sentence_is_cut_at_a_clause():
    text = ("Well, I was thinking about what you said to me yesterday "
            "evening at the station, and honestly I think you were right.")
    got = _cut(text, [3])
    assert len(got) == 2 and got[0].endswith("station,")
    # A later sentence gets the longer cap.
    got = _cut("Hi. " + text, [3])
    assert got == ["Hi.", " " + text]
    jp = "あのね、きのうのことなんだけど、ずっと考えていたの、やっぱりあなたが正しかったって。"
    got = _cut(jp, [2])
    assert len(got) == 2 and got[0].endswith("、")


# ---------------------------------------------------------- the pipeline

class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def now(self):
        return self.t


class SpeakingTTS:
    """A TTS whose clip for a text is 0.2 s quiet, 0.05 s per character of
    voice, 0.2 s quiet; records texts and moods."""

    whole = False
    last_voice = "fake"

    def __init__(self, clock=None, cost=0.4):
        self.calls, self.clock, self.cost = [], clock, cost

    def synth(self, text, mood=None):
        from tests.test_speech import quiet, vowel
        self.calls.append((text, mood))
        if self.clock is not None:
            self.clock.t += self.cost          # synthesis takes a while
        x = np.concatenate([quiet(0.2), vowel(0.05 * len(text)),
                            quiet(0.2, seed=1)]).astype(np.float32)
        from tests.test_speech import SR
        return x, SR, None

    def mood_voice(self, mood):
        return None


def _performer(tmp_path, reply, clock=None, tts=None):
    import argparse
    rig = _rig(tmp_path)
    clock = clock or FakeClock(5.0)
    hub = lv.Hub(clock)
    stream = fs.MotionStream(rig, seed=0, autonomic=False,
                             log=lambda s: None)
    a = argparse.Namespace(seed=0, lead=0.25, gap=0.12, lipsync="open",
                           mouth_gain=1.0, cjk_rate=dr.CJK_RATE,
                           latin_rate=dr.LATIN_RATE, mood="auto")
    labels = dr.gesture_labels(["m.motion3.json", "home.motion3.json"])
    run = {"system": "sys", "labels": labels, "offered": None}
    tts = tts if tts is not None else SpeakingTTS(clock)
    p = lv.Performer(a, run, rig, stream, clock, hub,
                     dr.TextBackend(reply), tts, log=lambda s: None)
    return p, hub, stream, clock, tts


def _events(hub, kind):
    return [m for m in hub.history if m["type"] == kind]


def test_each_sentence_is_spoken_and_staged_at_one_server_time(tmp_path):
    """Sentence by sentence: audio at now + lead on the frame grid, the
    next after the previous one's audio plus the gap; the tags realized at
    their place in that audio; the mouth curve starting on the audio's
    frame; subtitles on the same clock; the mood picked per sentence."""
    reply = ("Hello there. [brow:down] I'm so sorry... [nod] "
             "[gesture:home] Thank you!")
    p, hub, stream, clock, tts = _performer(tmp_path, reply)
    p.turn("hi")
    au = _events(hub, "audio")
    assert [m["text"] for m in au] == ["Hello there.", "I'm so sorry...",
                                       "Thank you!"]
    assert [c[1] for c in tts.calls] == [None, "sad", "happy"]
    # The first clip waits on its own TTS only (0.4 s here) plus the lead.
    assert au[0]["t"] == pytest.approx(5.0 + 0.4 + 0.25, abs=1 / FPS)
    for m in au:
        assert m["t"] * FPS == pytest.approx(round(m["t"] * FPS), abs=1e-3)
    for m0, m1 in zip(au, au[1:]):
        assert m1["t"] >= m0["t"] + m0["dur"] + 0.12 - 1e-6
    # Directed tokens sit at their sentence's audio time.
    mine = [k for k in stream.pending if k.get("src") == "director"]
    brow = [k for k in mine if k["type"] == "brow"]
    assert brow and brow[0]["t"] == pytest.approx(au[1]["t"] + 0.2, abs=0.05)
    ges = [k for k in mine if k["type"] == "gesture" and not k.get("settle")]
    assert ges and ges[0]["t"] >= au[2]["t"] - 1e-6
    # Mouth curves start on each clip's frame.
    assert [m[0] for m in stream.mouths] == [round(m["t"] * FPS) for m in au]
    subs = _events(hub, "sub")
    assert subs[0]["t"] == pytest.approx(au[0]["t"] + 0.2, abs=0.05)
    rep = _events(hub, "reply")[0]
    assert rep["first_audio_after"] == pytest.approx(0.65, abs=1 / FPS)
    assert p.messages[-1] == {"role": "assistant", "content": reply}


def test_lip_sync_follows_the_audio_on_the_frame_clock(tmp_path):
    """A/V on one clock: the mouth is open on exactly the frames whose
    time falls in the voiced part of the clip that plays then."""
    p, hub, stream, clock, tts = _performer(tmp_path, "Hello there, friend.")
    clock.t = 0.0
    tts.cost = 0.0
    p.turn("hi")
    (au,) = _events(hub, "audio")
    t0 = au["t"]
    n = int((t0 + au["dur"] + 1.0) * FPS)
    opened = []
    for f in range(n):
        over, _ = stream.frame()
        opened.append(over.get("ParamMouthOpenY", 0.0) > 0.3)
    t = np.arange(n) / FPS - t0
    voice = (t > 0.2 + 0.03) & (t < 0.2 + 0.05 * len("Hello there, friend.")
                                - 0.03)
    silent = (t < 0.2 - 0.04) | (t > au["dur"] - 0.2 + 0.06)
    assert np.mean(np.array(opened)[voice]) > 0.9
    assert not np.array(opened)[silent].any()


def test_tags_after_the_last_words_land_at_their_end(tmp_path):
    p, hub, stream, clock, tts = _performer(tmp_path, "Okay. [nod]")
    p.turn("hi")
    (au,) = _events(hub, "audio")
    nods = [k for k in stream.pending if k.get("src") == "director"
            and k["type"] == "head_turn"]
    assert nods and nods[0]["t"] == pytest.approx(au["t"] + au["dur"],
                                                  abs=2 / FPS)


# ---------------------------------------------------------------- server

def _get(url, timeout=5):
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(url, timeout=timeout)


def test_server_page_line_frames_and_audio(tmp_path):
    """On an ephemeral port, with a fake renderer: the page, the clock, a
    posted line spoken (an audio event on /events, the clip on /audio), and
    timestamped frames on /video and /mjpeg."""
    import threading
    import urllib.request
    clock = lv.Clock()
    p, hub, stream, _, tts = _performer(tmp_path, "Hello there. [nod] Bye.",
                                        clock=clock)
    p.hub = hub = lv.Hub(clock)
    tts.clock = None
    enc = lv.Encoder(hub, 70)
    loop = lv.RenderLoop(stream, lv.FakeBackend(64), clock, enc, FPS,
                         log=lambda s: None, report=0)
    srv = lv.Server(lv.App(clock, hub, FPS, p, loop), "127.0.0.1", 0)
    stop = threading.Event()
    th = threading.Thread(target=loop.run, args=(stop,), daemon=True)
    try:
        srv.start()
        th.start()
        p.start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        page = _get(base + "/").read().decode("utf-8")
        assert "<title>Tsukuyomi live</title>" in page and "/video" in page
        t = json.loads(_get(base + "/clock").read())["t"]
        assert abs(t - clock.now()) < 0.5
        ev_stream = _get(base + "/events")
        req = urllib.request.Request(
            base + "/say", data=json.dumps({"text": "hi"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        r = opener.open(req, timeout=5)
        assert r.status == 202 and json.loads(r.read())["ok"]
        audio = None
        for line in ev_stream:
            line = line.decode("utf-8").strip()
            if line.startswith("data:"):
                m = json.loads(line[5:])
                if m["type"] == "audio":
                    audio = m
                    break
        assert audio is not None and audio["text"] == "Hello there."
        assert audio["t"] > t
        wav = _get(base + audio["url"]).read()
        assert wav[:4] == b"RIFF" and len(wav) > 1000
        v = _get(base + "/video")
        head = v.read(lv.FRAME_HEAD.size)
        magic, f, ts, n = lv.FRAME_HEAD.unpack(head)
        jpg = v.read(n)
        assert magic == lv.FRAME_MAGIC and jpg[:2] == b"\xff\xd8"
        assert ts == pytest.approx(f / FPS)
        mj = _get(base + "/mjpeg")
        assert mj.headers["Content-Type"].startswith("multipart/x-mixed")
        assert mj.read(9) == b"--frame\r\n"
        st = json.loads(_get(base + "/state").read())
        assert st["rendered"] > 0
        assert _get(base + "/frame.jpg").read()[:2] == b"\xff\xd8"
        for h in (ev_stream, v, mj):
            h.close()
    finally:
        stop.set()
        srv.stop()
        th.join(timeout=5)
        enc.close()


def test_render_loop_skips_late_frames_but_steps_them(tmp_path):
    """A renderer slower than the frame rate: frames are rendered fewer
    than the clock's, every frame's state is still stepped, the frame
    numbers keep the clock, and dt covers the skipped time."""
    import threading
    rig = _rig(tmp_path)
    stream = fs.MotionStream(rig, seed=0, autonomic=False,
                             log=lambda s: None)
    hub = lv.Hub(lv.Clock())
    got = []
    hub.frame = lambda f, t, jpeg: got.append((f, t))

    class Slow(lv.FakeBackend):
        dts = []

        def render(self, overrides=None, parts=None):
            self.dts.append(self.dt)
            return super().render(overrides, parts)

    be = Slow(32, cost=0.07)                # ~14 fps of 30
    enc = lv.Encoder(hub, 60)
    loop = lv.RenderLoop(stream, be, lv.Clock(), enc, FPS,
                         log=lambda s: None, report=0)
    loop.run(threading.Event(), max_frames=45)
    enc.close()
    assert stream.f == 45 and loop.f == 45
    assert loop.skipped > 10 and loop.rendered + loop.skipped == 45
    assert max(be.dts) > 1.5 / FPS
    assert [t for f, t in got] == [f / FPS for f, _ in got]


def test_a_japanese_turn_keeps_its_voice_on_kana_less_sentences(tmp_path):
    """Sentence by sentence, "本当？" has no kana and "OK！" no Japanese at
    all; the turn's language (settled by うん。) picks the voice for them
    too, as the whole reply does offline."""
    picked = []

    class Router(SpeakingTTS):
        def begin(self, spoken, lang=None):
            picked.append(lang)
    p, hub, *_ = _performer(tmp_path, "うん。本当？大丈夫。OK！",
                            tts=Router())
    p.turn("ねえ、聞いて")
    assert picked == ["ja"] * 4
    picked.clear()
    p.backend = dr.TextBackend("本当？それはいいね。")
    p.turn("hello")                     # Han first, then kana
    assert picked == ["zh", "ja"]
    picked.clear()
    p.backend = dr.TextBackend("Well. 好的。")
    p.turn("hi")                        # too short to settle; then Han
    assert picked == ["en", "zh"]


def test_mood_words_are_whole_words():
    assert sp.mood_of("This function returns a crystal saddle napkin.")[0] \
        == "neutral"
    assert sp.mood_of("Thanks so much, that was fun.")[0] == "happy"
    assert sp.mood_of("困ったな。")[0] == "neutral"


def test_recording_keeps_the_clock_over_dropped_frames(tmp_path):
    import io as _io
    import types
    r = lv.Recorder(str(tmp_path / "rec"), FPS, log=lambda s: None)
    if r.proc is not None:
        r.proc.kill()
    sink = _io.BytesIO()
    r.proc = types.SimpleNamespace(stdin=sink)
    for f in (0, 1, 4, 5):              # 2 and 3 skipped or dropped
        r.frame(f, bytes([f]))
    assert sink.getvalue() == bytes([0, 1, 4, 4, 4, 5]) and r.frames == 6
    r.proc = None
    r.audio(0.5, np.ones(800, np.float32) * 0.1, 16000)
    r.close()
    from flow.speech import read_audio
    x, sr = read_audio(str(tmp_path / "rec" / "speech.wav"))
    assert abs(len(x) / sr - 0.55) < 0.01 and abs(x[:7000]).max() < 1e-3
