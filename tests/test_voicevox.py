"""
VOICEVOX (flow.speech.VoicevoxTTS), the mora timing and mouth track, and
the per-language voice config. No engine and no network: a fake opener
answers /speakers, /audio_query and /synthesis the way a VOICEVOX engine is
remembered to (UNVERIFIED against a live one), with an AudioQuery shaped
like the engine's for "こんにちは、今日は良い天気ですね。" (devoiced ス as
"U", ん as "N", a pause mora at the comma) and a wav whose stretches follow
the query's moras.

A tag's time must be its character's mora; pauses and ッ/ん must close the
mouth; an error body must reach the user; a non-Japanese reply must go to
edge; the voice config must lose to a flag.
"""

from __future__ import annotations

import io
import json
import sys
import types
import urllib.error
import urllib.parse

import numpy as np
import pytest

from flow import director as dr
from flow import speech as sp
from tests.test_speech import quiet, vowel, wav_bytes

SR = 24000
FPS = 30.0
TEXT = "こんにちは、今日は良い天気ですね。"
SPEAKERS = [
    {"name": "四国めたん", "speaker_uuid": "u1", "version": "0.14.0",
     "styles": [{"name": "ノーマル", "id": 2, "type": "talk"},
                {"name": "あまあま", "id": 0, "type": "talk"}]},
    {"name": "ずんだもん", "speaker_uuid": "u2", "version": "0.14.0",
     "styles": [{"name": "ノーマル", "id": 3}, {"name": "ハミング",
                                                  "id": 3001,
                                                  "type": "humming"}]},
    {"name": "冥鳴ひまり", "speaker_uuid": "u3", "version": "0.14.0",
     "styles": [{"name": "ノーマル", "id": 14}]},
]


def _m(text, cons, cl, vow, vl, pitch=5.6):
    return {"text": text, "consonant": cons, "consonant_length": cl,
            "vowel": vow, "vowel_length": vl, "pitch": pitch}


def query():
    pause = {"text": "、", "consonant": None, "consonant_length": None,
             "vowel": "pau", "vowel_length": 0.3, "pitch": 0.0}
    phrases = [
        [_m("コ", "k", 0.06, "o", 0.08), _m("ン", None, None, "N", 0.07),
         _m("ニ", "n", 0.04, "i", 0.07), _m("チ", "ch", 0.07, "i", 0.06),
         _m("ワ", "w", 0.05, "a", 0.10)],
        [_m("キョ", "ky", 0.07, "o", 0.09), _m("ウ", None, None, "o", 0.08),
         _m("ワ", "w", 0.05, "a", 0.09)],
        [_m("ヨ", "y", 0.05, "o", 0.09), _m("イ", None, None, "i", 0.09)],
        [_m("テ", "t", 0.05, "e", 0.08), _m("ン", None, None, "N", 0.07),
         _m("キ", "k", 0.06, "i", 0.06), _m("デ", "d", 0.04, "e", 0.08),
         _m("ス", "s", 0.08, "U", 0.06, 0.0), _m("ネ", "n", 0.05, "e", 0.15)],
    ]
    return {"accent_phrases": [
        {"moras": ms, "accent": 1, "pause_mora": pause if k == 0 else None,
         "is_interrogative": False} for k, ms in enumerate(phrases)],
        "speedScale": 1.0, "pitchScale": 0.0, "intonationScale": 1.0,
        "volumeScale": 1.0, "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.1, "outputSamplingRate": SR,
        "outputStereo": False,
        "kana": "コンニチワ'、キョ'ウワ/ヨ'イ/テ'ンキデスネ"}


FORMANTS = {"a": (800, 1300), "i": (320, 2500), "u": (350, 1400),
            "e": (500, 2100), "o": (500, 900)}


def render_query(q):
    """A wav as the engine would send for `q`: each mora's vowel as a
    steady synthetic vowel (consonant included), pauses and the pre/post
    silence quiet, ん a quiet hum; lengths divided by speedScale."""
    sp_ = q["speedScale"]
    parts = [quiet(q["prePhonemeLength"] / sp_, sr=SR)]
    for ap in q["accent_phrases"]:
        for m in ap["moras"]:
            d = ((m["consonant_length"] or 0) + m["vowel_length"]) / sp_
            v = m["vowel"].lower()
            if v in FORMANTS:
                parts.append(vowel(d, *FORMANTS[v], sr=SR))
            else:
                parts.append(vowel(d, 300, 1200, amp=0.05, sr=SR))
        if ap["pause_mora"]:
            parts.append(quiet(ap["pause_mora"]["vowel_length"] / sp_,
                               seed=3, sr=SR))
    parts.append(quiet(q["postPhonemeLength"] / sp_, seed=4, sr=SR))
    return wav_bytes(np.concatenate(parts), SR)


class Engine:
    """A fake urllib opener speaking VOICEVOX."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def open(self, req, timeout=None):
        u = urllib.parse.urlsplit(req.full_url)
        qs = dict(urllib.parse.parse_qsl(u.query))
        self.calls.append((req.get_method(), u.path, qs, req.data))
        if self.fail and u.path == self.fail[0]:
            code, body = self.fail[1:]
            if code is None:
                raise urllib.error.URLError(ConnectionRefusedError(111,
                                                                   "refused"))
            raise urllib.error.HTTPError(req.full_url, code, "err", {},
                                         io.BytesIO(body))
        if u.path == "/speakers":
            return io.BytesIO(json.dumps(SPEAKERS).encode())
        if u.path == "/audio_query":
            assert req.get_method() == "POST" and req.data == b""
            return io.BytesIO(json.dumps(query(), ensure_ascii=False)
                              .encode())
        if u.path == "/synthesis":
            assert req.get_method() == "POST"
            return io.BytesIO(render_query(json.loads(req.data)))
        raise AssertionError(u.path)


def tts(voice="四国めたん/ノーマル", knobs=None, eng=None, log=None):
    return sp.VoicevoxTTS("http://127.0.0.1:50021/", voice, knobs,
                          log=log or (lambda s: None), opener=eng or Engine())


# ---------------------------------------------------------------- engine

def test_speakers_resolve_by_name_style_character_and_id():
    t = tts("四国めたん/ノーマル")
    assert t.resolve() == 2 and t.credit == "VOICEVOX:四国めたん"
    assert t.last_voice == "四国めたん/ノーマル (2)"
    assert tts("ずんだもん").resolve() == 3          # first talk style
    t = tts("14")
    assert t.resolve() == 14 and t.credit == "VOICEVOX:冥鳴ひまり"
    seen = []
    assert tts(None, log=seen.append).resolve() == 2
    assert "no voice given" in seen[0]
    with pytest.raises(RuntimeError) as e:
        tts("ずんだもん/ハミング").resolve()           # not a talk style
    assert "冥鳴ひまり/ノーマル (14)" in str(e.value)


def test_query_and_synthesis_flow():
    eng = Engine()
    t = tts(knobs={"speed": 1.25, "pitch": None}, eng=eng)
    x, sr, words, moras = t.synth(TEXT)
    (m0, p0, *_), (m1, p1, q1, d1), (m2, p2, q2, d2) = eng.calls
    assert (m0, p0) == ("GET", "/speakers")
    assert (m1, p1, q1) == ("POST", "/audio_query",
                            {"text": TEXT, "speaker": "2"})
    assert (m2, p2, q2["speaker"]) == ("POST", "/synthesis", "2")
    sent = json.loads(d2)
    assert sent["speedScale"] == 1.25 and sent["pitchScale"] == 0.0
    assert sr == SR
    # The engine's speed shortens the audio; the mora times follow it.
    total = (0.1 + 0.1 + 0.3 + sum(
        (m["consonant_length"] or 0) + m["vowel_length"]
        for ap in query()["accent_phrases"] for m in ap["moras"])) / 1.25
    assert abs(len(x) / sr - total) < 0.01
    assert abs(moras[-1]["end"] - (total - 0.1 / 1.25)) < 0.01
    assert [w for w, *_ in words] == ["こ", "ん", "に", "ち", "は", "今日",
                                      "は", "良", "い", "天気", "で", "す",
                                      "ね"]


def test_error_body_and_unreachable_engine_are_surfaced():
    eng = Engine(fail=("/audio_query", 422,
                       b'{"detail":[{"msg":"value is not a valid integer"}]}'))
    with pytest.raises(RuntimeError, match="422 from voicevox POST "
                       "/audio_query.*not a valid integer"):
        tts(eng=eng).synth(TEXT)
    with pytest.raises(RuntimeError, match="unreachable.*docker run"):
        tts(eng=Engine(fail=("/speakers", None, b""))).synth(TEXT)


def test_query_moras_pause_scale_and_rescale():
    q = query()
    q["speedScale"], q["pauseLengthScale"] = 2.0, 0.5
    ms = sp.query_moras(q)
    pause = [m for m in ms if m["vowel"] == "pau"]
    assert len(pause) == 1
    assert pause[0]["end"] - pause[0]["start"] == pytest.approx(0.3 * 0.5 / 2)
    assert ms[0]["start"] == pytest.approx(0.05)
    assert ms[0]["vstart"] == pytest.approx(0.05 + 0.03)
    seen = []
    ms2 = sp.query_moras(q, dur=2 * (ms[-1]["end"] + 0.05), log=seen.append)
    assert ms2[-1]["end"] == pytest.approx(2 * ms[-1]["end"])
    assert "rescaled" in seen[0]


def test_alignment_survives_a_reading_mismatch():
    ms = sp.query_moras(query())
    # The reply wrote いい where the engine read ヨイ: one mismatch, the
    # rest still lands on its own moras.
    w = dict((a, (s, e)) for a, s, e in
             sp.align_moras("こんにちは、今日はいい天気ですね。", ms))
    te = next(m for m in ms if m["text"] == "テ")
    assert w["天気"][0] == pytest.approx(te["start"])
    assert w["ね"][1] == pytest.approx(ms[-1]["end"])


# ---------------------------------------------------------- timing, mouth

def test_mora_timing_places_tags_and_subtitles():
    p = dr.TagParser([])
    p.feed("こんにちは、[nod]今日は良い天気[brow:up]ですね。")
    p.close()
    t = tts()
    said = sp.speak(p.spoken, p.tags, t, lambda ch: 1 / 7,
                    log=lambda s: None)
    assert said["method"] == "moras"
    ms = sp.query_moras(query())
    kyo = next(m for m in ms if m["text"] == "キョ")
    de = next(m for m in ms if m["text"] == "デ")
    nod, brow = said["acts"]
    assert nod["t"] == pytest.approx(kyo["start"], abs=0.01)
    assert brow["t"] == pytest.approx(de["start"], abs=0.01)
    subs = dr.subtitles(p.spoken, said["spans"])
    assert subs[0]["start"] == pytest.approx(ms[0]["start"], abs=0.01)
    assert subs[-1]["end"] >= ms[-1]["end"] - 0.01
    assert said["moras"][0]["text"] == "コ"


def test_mora_track_shapes_and_closes():
    ms = sp.query_moras(query())
    n = int(2.6 * FPS)
    op = np.full(n, 0.9)
    o2, shapes, vis = sp.mora_track(ms, FPS, op)
    t = (np.arange(n) + 0.5) / FPS
    pau = next(m for m in ms if m["vowel"] == "pau")
    inside = (t > pau["start"] + 0.05) & (t < pau["end"] - 0.05)
    assert (o2[inside] == 0).all() and all(vis[f] == "pau"
                                           for f in np.where(inside)[0])
    assert o2[t < 0.05].max() == 0                  # pre-phoneme silence
    ne = next(m for m in ms if m["text"] == "ネ")
    f = int((ne["vstart"] + ne["end"]) / 2 * FPS)
    assert vis[f] == "e" and shapes["e"][f] == pytest.approx(o2[f])
    # Shapes sum to the openness wherever the mouth is open.
    tot = sum(shapes.values())
    np.testing.assert_allclose(tot, o2, atol=1e-6)
    # ん: nearly shut, shaped as u.
    nn = [m for m in ms if m["vowel"] == "N"][0]
    f = int((nn["start"] + nn["end"]) / 2 * FPS)
    assert vis[f] == "N" and o2[f] < 0.5 * 0.9
    # The devoiced U of す reads as u.
    su = next(m for m in ms if m["text"] == "ス")
    f = int((su["vstart"] + su["end"]) / 2 * FPS)
    assert vis[f] in ("u", "e")


def test_mouth_track_auto_with_moras_drives_mapped_vowels():
    t = tts()
    x, sr, _, ms = t.synth(TEXT)
    cal = {"map": {"ParamA": {"curve": "a", "lo": 0.0, "hi": 0.9},
                   "ParamO": {"curve": "o", "lo": 0.0, "hi": 0.8}},
           "level": 0.9}
    n = int(len(x) / sr * FPS) + 1
    tr = sp.mouth_track(x, sr, FPS, n, "auto", cal=cal, moras=ms)
    assert set(tr) >= {"open", "vowels", "visemes", "map", "gamma"}
    assert len(tr["visemes"]) == n and set(tr["visemes"]) <= set(sp.VISEMES)
    op = np.array(tr["open"])
    assert 0.5 <= tr["gamma"] <= 2.0
    assert np.median(op[op > 0]) == pytest.approx(0.9, abs=0.02)
    # Vowel curves are the openness split by the moras' shapes.
    tot = sum(np.array(c) for c in tr["vowels"].values())
    np.testing.assert_allclose(tot, op, atol=6e-4)
    assert max(tr["vowels"]["a"]) > 0.5 and max(tr["vowels"]["e"]) > 0.5
    # Without a calibrated map, auto is open exactly.
    plain = sp.mouth_track(x, sr, FPS, n, "open")
    assert sp.mouth_track(x, sr, FPS, n, "auto", cal={"map": {}}) == plain


# ------------------------------------------------------------ voice config

def test_voice_settings_precedence():
    cfg = {"ja": {"tts": "voicevox", "voice": "冥鳴ひまり/ノーマル",
                  "speed": 1.1},
           "en": {"tts": "edge", "voice": "en-US-AnaNeural", "rate": "+5%"}}
    none = {"voice": None, "rate": None, "speed": None}
    log = []
    assert sp.voice_settings("ja", "auto", none, cfg) == (
        "voicevox", {"voice": "冥鳴ひまり/ノーマル", "speed": 1.1})
    assert sp.voice_settings("en", "auto", none, cfg) == (
        "edge", {"voice": "en-US-AnaNeural", "rate": "+5%"})
    # A flag wins over the config ...
    assert sp.voice_settings("ja", "auto", dict(none, speed=0.9), cfg)[1] \
        == {"voice": "冥鳴ひまり/ノーマル", "speed": 0.9}
    # ... and an explicit --tts drops entries for another backend.
    assert sp.voice_settings("ja", "edge", none, cfg) == ("edge", {})
    # VOICEVOX on English goes to edge with the config's edge entry; the
    # --voice flag named a VOICEVOX voice and stays out.
    k, st = sp.voice_settings("en", "voicevox", dict(none, voice="2"), cfg,
                              log.append)
    assert (k, st) == ("edge", {"voice": "en-US-AnaNeural", "rate": "+5%"})
    assert "Japanese only" in log[0]
    assert sp.voice_settings("zh", "auto", none, cfg) == ("edge", {})
    assert sp.voice_settings("ko", "auto", none,
                             {"default": {"tts": "edge", "voice": "v"}}) \
        == ("edge", {"voice": "v"})


def test_load_voice_config(tmp_path):
    p = tmp_path / "x.voice.json"
    p.write_text(json.dumps({"_comment": "hi", "ja": {"tts": "voicevox",
                                                      "bogus": 1},
                             "fr": {"tts": "edge"}}))
    seen = []
    assert sp.load_voice_config(str(p), seen.append) == {
        "ja": {"tts": "voicevox"}}
    assert len(seen) == 2
    assert sp.load_voice_config(str(tmp_path / "none.json")) == {}
    p.write_text(json.dumps({"ja": {"tts": "say"}}))
    with pytest.raises(SystemExit, match="unknown tts"):
        sp.load_voice_config(str(p))


def _args(**kw):
    base = dict(tts=None, voice_config=None, rig=None, voice=None,
                tts_rate=None, tts_model=None, tts_speed=None, tts_pitch=None,
                voicevox_url=None, backend="file")
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_make_tts_routes_by_language(tmp_path, monkeypatch):
    rig = tmp_path / "aidang_2.npz"
    (tmp_path / "aidang_2.voice.json").write_text(json.dumps(
        {"ja": {"tts": "voicevox", "voice": "冥鳴ひまり/ノーマル"},
         "en": {"tts": "edge", "voice": "en-US-AnaNeural"}},
        ensure_ascii=False), encoding="utf-8")
    # The rig's file alone never switches speech on.
    assert dr.make_tts(_args(rig=str(rig))) is None
    seen = []
    r = dr.make_tts(_args(rig=str(rig), tts="auto"), log=seen.append)
    assert "aidang_2.voice.json" in seen[0]
    vv = r.pick("ja")
    assert isinstance(vv, sp.VoicevoxTTS) and vv.voice == "冥鳴ひまり/ノーマル"
    assert vv.url == sp.VOICEVOX_URL and r.kind == "voicevox"
    ed = r.pick("en")
    assert isinstance(ed, sp.EdgeTTS) and ed.voice == "en-US-AnaNeural"
    assert ed.rate == "+0%"
    assert r.pick("en") is ed                         # cached
    # --voice-config alone means auto; a flag wins over the file.
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"en": {"tts": "edge", "voice": "a",
                                      "rate": "+5%"}}))
    r = dr.make_tts(_args(voice_config=str(cfg), tts_rate="-5%"))
    assert r.pick("en").rate == "-5%" and r.pick("en").voice == "a"
    with pytest.raises(SystemExit, match="no such file"):
        dr.make_tts(_args(voice_config=str(tmp_path / "nope.json")))


def test_voicevox_router_falls_back_to_edge_for_english(monkeypatch):
    fake = types.ModuleType("edge_tts")
    used = {}

    class Communicate:
        def __init__(self, text, voice, rate="+0%", boundary=None):
            used["voice"] = voice

        def stream_sync(self):
            yield {"type": "audio", "data": wav_bytes(vowel(0.5), 16000)}
    fake.Communicate = Communicate
    monkeypatch.setitem(sys.modules, "edge_tts", fake)
    seen = []
    r = dr.make_tts(_args(tts="voicevox"), log=seen.append)
    p = dr.TagParser([])
    p.feed("Hello there.")
    p.close()
    said = sp.speak(p.spoken, p.tags, r, lambda ch: 1 / 14,
                    log=lambda s: None)
    assert r.kind == "edge" and used["voice"] == "en-US-AnaNeural"
    assert any("Japanese only" in s for s in seen)
    assert said["moras"] is None and r.credit is None
