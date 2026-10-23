"""
flow.speech and the director's --tts: speech audio, tag timing from it, lip
sync. No network and no ears: the audio is numpy-made (harmonic "vowels"
with formant envelopes, silence, noise), edge-tts is a fake module and
SiliconFlow a fake urlopen.

The mouth must open on voice and shut in silence without flapping on
noise; a tag's time must follow the word boundaries when the service sends
them and the voiced stretches when it does not; a [pause] must be heard;
--tts none must leave every output as it was; and a planned run with a mouth
must replay bit for bit through runtime --tokens-in.
"""

from __future__ import annotations

import io
import json
import shlex
import shutil
import subprocess
import sys
import types
import wave

import numpy as np
import pytest

from flow import director as dr
from flow import speech as sp

SR = 16000
FPS = 30.0


def vowel(dur, f1=800, f2=1300, f0=150.0, amp=0.3, sr=SR):
    """A steady synthetic vowel: harmonics of f0 under two formant peaks."""
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for k in range(1, int(4000 / f0)):
        f = k * f0
        g = (np.exp(-((f - f1) / 120) ** 2) + 0.7 * np.exp(-((f - f2) / 150) ** 2)
             + 0.02)
        x += g * np.sin(2 * np.pi * f * t)
    x *= amp / np.abs(x).max()
    ramp = min(int(0.01 * sr), len(x) // 2)
    x[:ramp] *= np.linspace(0, 1, ramp)
    x[len(x) - ramp:] *= np.linspace(1, 0, ramp)
    return x.astype(np.float32)


def quiet(dur, db=-70.0, seed=0, sr=SR):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(dur * sr)) * 10 ** (db / 20)).astype(
        np.float32)


def wav_bytes(x, sr=SR):
    buf = io.BytesIO()
    pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2")
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def two_phrases():
    """0.3 s quiet, 0.8 s voice, 0.5 s quiet, 0.6 s voice, 0.4 s quiet."""
    x = np.concatenate([quiet(0.3), vowel(0.8), quiet(0.5, seed=1),
                        vowel(0.6), quiet(0.4, seed=2)])
    return x


def w_rate(ch):
    return 1.0 / 14.0


# ----------------------------------------------------------------- mouth

def test_mouth_opens_on_voice_and_shuts_in_silence():
    x = two_phrases()
    n = int(len(x) / SR * FPS)
    m = sp.mouth_open(x, SR, FPS, n)
    t = np.arange(n) / FPS

    def at(a, b):
        return m[(t >= a) & (t < b)]
    assert (at(0.0, 0.28) == 0).all()                 # before the voice
    assert (at(0.4, 1.0) > 0.7).all()                 # first phrase, open
    # Closed again within RELEASE-ish time of the phrase end (1.1 s): the
    # release keeps 27% a frame, under SHUT after 3 frames (0.1 s).
    assert (at(1.32, 1.58) == 0).all()
    assert (at(1.7, 2.1) > 0.7).all()
    assert (at(2.42, 2.6) == 0).all()
    assert m.max() <= 1.0


def test_mouth_dips_between_syllables():
    """Two syllables with a 70 ms closure between them (aidang_2's edge
    run dips to -39/-42 dBFS for 2 frames between syllables): the mouth
    must visibly close there, not hang half open (0.41/0.23 with the old
    50 ms window and 60 ms release)."""
    x = np.concatenate([quiet(0.2), vowel(0.3), quiet(0.07, -60.0, seed=4),
                        vowel(0.3), quiet(0.3, seed=5)])
    n = int(len(x) / SR * FPS)
    m = sp.mouth_open(x, SR, FPS, n)
    t = np.arange(n) / FPS
    gap = m[(t > 0.5) & (t < 0.58)]
    assert gap.min() < 0.3, m.round(2)
    assert m[(t > 0.3) & (t < 0.45)].min() > 0.7
    assert m[(t > 0.62) & (t < 0.8)].max() > 0.7


def test_mouth_does_not_flap_on_noise():
    # Noise only, at -70 dBFS: under the absolute floor, never opens.
    n = 60
    assert (sp.mouth_open(quiet(2.0, -70.0), SR, FPS, n) == 0).all()
    # Speech with a noise bed 40 dB under it: the bed stays shut.
    x = np.concatenate([vowel(0.5), quiet(1.0, -50.0, seed=3) + 0,
                        vowel(0.5)])
    m = sp.mouth_open(x, SR, FPS, int(2.0 * FPS))
    t = np.arange(len(m)) / FPS
    assert (m[(t > 0.75) & (t < 1.45)] == 0).all()


def test_vowel_weights_tell_a_i_o_apart():
    for f1, f2, want in ((800, 1300, "a"), (320, 2500, "i"), (500, 900, "o")):
        x = vowel(0.6, f1, f2, f0=200.0)
        n = int(0.6 * FPS)
        op = sp.mouth_open(x, SR, FPS, n)
        vw = sp.vowels(x, SR, FPS, op)
        mid = n // 2
        best = max(vw, key=lambda v: vw[v][mid])
        assert best == want, (want, {v: float(c[mid]) for v, c in vw.items()})
        np.testing.assert_allclose(sum(c[mid] for c in vw.values()), op[mid],
                                   rtol=1e-5)


# -------------------------------------------------------------- alignment

class FakeTTS:
    """Returns given audio (and words) for each call; records the texts."""

    whole = False

    def __init__(self, outs):
        self.outs, self.texts = list(outs), []
        self.last_voice = "fake"

    def synth(self, text):
        self.texts.append(text)
        x, words = self.outs.pop(0)
        return x, SR, words


def test_vad_alignment_puts_tags_on_the_voiced_stretches():
    spoken = "Hello there. [nod] How are you?"
    p = dr.parse(spoken)
    said = sp.speak(p.spoken, p.tags, FakeTTS([(two_phrases(), None)]),
                    w_rate, log=lambda s: None)
    assert said["method"] == "vad"
    (nod,) = said["acts"]
    # "How" starts the second phrase, at 1.6 s (VAD hop 10 ms).
    assert abs(nod["t"] - 1.6) < 0.03, nod
    i_dot = p.spoken.index(".")
    assert abs(said["spans"][i_dot][0] - 1.1) < 0.03     # first phrase end
    assert abs(said["spans"][0][0] - 0.3) < 0.03         # "H" at its onset
    segs = dr.subtitles(p.spoken, said["spans"])
    assert [s["text"] for s in segs] == ["Hello there.", "How are you?"]
    assert abs(segs[1]["start"] - 1.6) < 0.03
    assert abs(segs[1]["end"] - 2.2) < 0.03
    assert said["end"] == pytest.approx(len(two_phrases()) / SR)


def test_vad_joins_stop_closures_inside_a_word():
    # An 80 ms gap (a stop closure) is not a phrase break; 300 ms is.
    x = np.concatenate([vowel(0.3), quiet(0.08), vowel(0.3), quiet(0.3),
                        vowel(0.3)])
    segs = sp.voiced(x, SR)
    assert len(segs) == 2, segs


def test_word_boundaries_set_tag_times():
    spoken = "Oh, [look:up] hello [nod] world."
    p = dr.parse(spoken)
    words = [("Oh", 0.10, 0.30), ("hello", 0.60, 1.00), ("world", 1.20, 1.70)]
    said = sp.speak(p.spoken, p.tags, FakeTTS([(quiet(2.0, -20), words)]),
                    w_rate, log=lambda s: None)
    assert said["method"] == "words"
    assert [a["t"] for a in said["acts"]] == [0.6, 1.2]
    sp_ = said["spans"]
    assert sp_[p.spoken.index(",")] == (0.3, 0.3)       # punctuation: silent
    assert sp_[p.spoken.index("d")][1] == pytest.approx(1.7)
    segs = dr.subtitles(p.spoken, sp_)
    assert segs == [{"start": 0.1, "end": 1.7, "text": "Oh, hello world."}]


def test_ellipsis_stays_with_its_phrase():
    """Hiyori's edge run had segments ".", "." cut out of "little...":
    a run of sentence ends and a closing quote belong to the phrase, by
    the rate model and by word boundaries alike."""
    p = dr.parse("Well, maybe a little... [nod] I kept checking. "
                 "そうね。。。でも「いいわ。」 Hmm… okay?!")
    want = ["Well, maybe a little...", "I kept checking.", "そうね。。。",
            "でも「いいわ。」", "Hmm…", "okay?!"]
    _, spans, _ = dr.timeline(p.spoken, p.tags)
    rate = dr.subtitles(p.spoken, spans)
    assert [s["text"] for s in rate] == want
    words = [("Well", 0.1, 0.3), ("maybe", 0.4, 0.6), ("a", 0.6, 0.7),
             ("little", 0.7, 1.0), ("I", 1.5, 1.6), ("kept", 1.6, 1.8),
             ("checking", 1.8, 2.2)]
    q = dr.parse("Well, maybe a little... [nod] I kept checking.")
    said = sp.speak(q.spoken, q.tags, FakeTTS([(quiet(2.5, -20), words)]),
                    w_rate, log=lambda s: None)
    assert said["method"] == "words"
    segs = dr.subtitles(q.spoken, said["spans"])
    assert [s["text"] for s in segs] == want[:2]
    assert segs[0]["end"] == pytest.approx(1.0)
    # Punctuation alone is never a segment, leading or after a clause.
    sp1 = [(0.1 * i, 0.1 * i + 0.1) for i in range(20)]
    assert [s["text"] for s in dr.subtitles("...hi. A, ... b.", sp1)] == \
        ["...hi.", "A, ...", "b."]


def test_poor_word_match_falls_back_to_energy():
    p = dr.parse("Hello there. [nod] How are you?")
    words = [("bonjour", 0.0, 0.5), ("xyz", 0.5, 1.0)]
    log = []
    said = sp.speak(p.spoken, p.tags, FakeTTS([(two_phrases(), words)]),
                    w_rate, log=log.append)
    assert said["method"] == "vad" and "cover 0%" in log[0]
    assert abs(said["acts"][0]["t"] - 1.6) < 0.03


def test_pause_is_synthesised_as_silence():
    p = dr.parse("One. [nod] [pause:0.5] [look:up] Two.")
    a, b = vowel(0.4), vowel(0.3)
    tts = FakeTTS([(a, [("One", 0.0, 0.4)]), (b, [("Two", 0.0, 0.3)])])
    said = sp.speak(p.spoken, p.tags, tts, w_rate, log=lambda s: None)
    assert tts.texts == ["One.", "Two."]
    assert len(said["audio"]) == len(a) + int(0.5 * SR) + len(b)
    nod, look = said["acts"]
    assert nod["tag"] == "nod" and nod["t"] == pytest.approx(0.4)   # before
    assert look["t"] == pytest.approx(0.9)                          # after
    assert np.abs(said["audio"][len(a):len(a) + int(0.5 * SR)]).max() == 0


# --------------------------------------------------------------- backends

def _fake_edge(monkeypatch, audio, words, boundary_kw=True):
    seen = {}

    class Communicate:
        if boundary_kw:
            def __init__(self, text, voice, *, rate="+0%",
                         boundary="SentenceBoundary"):
                seen.update(text=text, voice=voice, boundary=boundary,
                            rate=rate)
        else:
            def __init__(self, text, voice, *, rate="+0%"):
                seen.update(text=text, voice=voice, boundary=None, rate=rate)

        def stream_sync(self):
            yield {"type": "audio", "data": audio[:100]}
            for w, s, e in words:
                yield {"type": "WordBoundary", "offset": int(s * 1e7),
                       "duration": int((e - s) * 1e7), "text": w}
            yield {"type": "audio", "data": audio[100:]}

    fake = types.ModuleType("edge_tts")
    fake.Communicate = Communicate
    # edge-tts may really be installed; the fake must win.
    monkeypatch.setitem(sys.modules, "edge_tts", fake)
    return seen


@pytest.mark.parametrize("boundary_kw", [True, False])
def test_edge_tts_words_and_voice_by_script(monkeypatch, boundary_kw):
    seen = _fake_edge(monkeypatch, wav_bytes(vowel(1.0)),
                      [("こんにちは", 0.1, 0.6), ("世界", 0.7, 0.95)],
                      boundary_kw)
    tts = sp.EdgeTTS(log=lambda s: None)
    p = dr.parse("こんにちは、[nod]世界。")
    said = sp.speak(p.spoken, p.tags, tts, lambda ch: 1 / 6,
                    log=lambda s: None)
    assert seen["voice"] == "ja-JP-NanamiNeural" == tts.last_voice
    assert seen["boundary"] == ("WordBoundary" if boundary_kw else None)
    assert said["method"] == "words"
    assert said["acts"][0]["t"] == pytest.approx(0.7)
    assert sp.EDGE_VOICES[sp.language("你好世界")] == "zh-CN-XiaoxiaoNeural"
    assert sp.language("Hello") == "en"


def test_edge_tts_missing_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "edge_tts", None)
    with pytest.raises(SystemExit, match="pip install edge-tts"):
        sp.EdgeTTS().synth("hi")


def test_siliconflow_tts_request_and_errors(monkeypatch):
    import urllib.error
    import urllib.request
    seen = {}
    body = wav_bytes(vowel(0.5))

    def urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "k")
    tts = sp.SiliconFlowTTS("https://h/v1/", "SILICONFLOW_API_KEY")
    x, sr, words = tts.synth("Hi there.")
    assert sr == SR and len(x) == int(0.5 * SR) and words is None
    assert seen["url"] == "https://h/v1/audio/speech"
    assert seen["auth"] == "Bearer k"
    assert seen["body"] == {"model": sp.SF_TTS_MODEL, "input": "Hi there.",
                            "voice": f"{sp.SF_TTS_MODEL}:anna",
                            "response_format": "wav", "stream": False}
    body = b'{"code":20012,"message":"voice not found"}'
    with pytest.raises(RuntimeError, match="voice not found"):
        tts.synth("Hi")

    def bad(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {},
                                     io.BytesIO(b'{"message":"bad model"}'))
    monkeypatch.setattr(urllib.request, "urlopen", bad)
    with pytest.raises(RuntimeError, match="400 .*bad model"):
        tts.synth("Hi")


def test_siliconflow_tts_retries_a_cut_off_download(monkeypatch):
    import http.client
    import urllib.request
    body = wav_bytes(vowel(0.5))
    calls = []

    class Cut(io.BytesIO):
        def read(self, *a):
            raise http.client.IncompleteRead(b"x" * 10, 36430)

    def urlopen(req, timeout=None):
        calls.append(1)
        return Cut() if len(calls) == 1 else io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("SILICONFLOW_API_KEY", "k")
    tts = sp.SiliconFlowTTS("https://h/v1/", "SILICONFLOW_API_KEY",
                            log=lambda s: None)
    x, sr, _ = tts.synth("Hi there.")
    assert len(calls) == 2 and len(x) == int(0.5 * SR)


def test_siliconflow_tts_base_follows_the_chat_backend_only():
    a = types.SimpleNamespace(tts="siliconflow", backend="deepseek",
                              api_base="https://api.deepseek.com",
                              api_key_env=None, tts_api_base=None,
                              tts_key_env=None, tts_model=None, voice="x:y",
                              tts_format="wav", tts_sample_rate=None)
    t = dr.make_tts(a)
    assert t.url == "https://api.siliconflow.com/v1/audio/speech"
    assert t.voice == "x:y"
    a.backend, a.api_base = "siliconflow", "https://api.siliconflow.cn/v1"
    assert dr.make_tts(a).url == "https://api.siliconflow.cn/v1/audio/speech"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_mp3_decodes_through_ffmpeg(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(wav_bytes(np.concatenate([vowel(0.5), quiet(0.5)])))
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        str(tmp_path / "a.mp3")], capture_output=True)
    if r.returncode:
        pytest.skip("ffmpeg has no mp3 encoder")
    x, sr = sp.read_audio(str(tmp_path / "a.mp3"))
    assert sr == sp.DECODE_SR
    assert abs(len(x) / sr - 1.0) < 0.08          # mp3 pads a frame or two
    m = sp.mouth_open(x, sr, FPS, 30)
    assert m[5:12].min() > 0.5 and m[24:].max() == 0


# ------------------------------------------------------------- end to end

def _mouth_rig(root):
    """test_director's rig plus ParamMouthOpenY and ParamMouthForm."""
    from tests.test_director import _rig
    from l2dprobe.analyze import group_of
    model, rig = _rig(root)
    z = dict(np.load(rig, allow_pickle=True))
    add = [("ParamMouthOpenY", 0.0, 1.0, 0.0), ("ParamMouthForm", -2, 1, 1)]
    z["pid"] = np.array(list(z["pid"]) + [p for p, *_ in add], dtype=object)
    z["kind"] = np.array(list(z["kind"]) + ["continuous"] * 2, dtype=object)
    z["group"] = np.array(list(z["group"]) + [group_of(p) for p, *_ in add],
                          dtype=object)
    for k, j in (("minimum", 1), ("maximum", 2), ("default", 3)):
        z[k] = np.concatenate([z[k], np.array([a[j] for a in add],
                                              np.float32)])
    z["descriptor"] = np.zeros((len(z["pid"]), 4), np.float32)
    np.savez(rig, **z)
    return model, rig


def _replay(log, tmp_path):
    from flow.runtime import main as runtime_main
    cmd = [ln for ln in log.splitlines() if "--tokens-in" in ln][0]
    argv = shlex.split(cmd.strip())
    argv = argv[argv.index("flow.runtime") + 1:]
    k = argv.index("--out")
    argv = argv[:k] + argv[k + 2:]
    out = tmp_path / "replay.npz"
    assert runtime_main(argv + ["--dump-only", "--dump", str(out)]) == 0
    return np.load(out, allow_pickle=True)


def test_tts_file_end_to_end_with_mouth_replays_bit_for_bit(tmp_path, capsys):
    model, rig = _mouth_rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text("Hello there. [nod] How are you?", encoding="utf-8")
    audio = tmp_path / "said.wav"
    audio.write_bytes(wav_bytes(two_phrases()))
    out = tmp_path / "run"
    assert dr.main(["--model-json", str(model), "--rig", str(rig),
                    "--backend", "file", "--reply-file", str(reply),
                    "--out", str(out), "--probe-dir", str(tmp_path),
                    "--seed", "2", "--tts", "file", "--tts-file",
                    str(audio)]) == 0
    log = capsys.readouterr().out
    assert (out / "speech.wav").exists()
    acts = json.loads((out / "actions.json").read_text())
    assert acts["tts"]["timing"] == "vad"
    assert abs(acts["actions"][0]["t"] - 1.6) < 0.03
    assert acts["speech_seconds"] == pytest.approx(2.6, abs=1e-3)
    assert acts["seconds"] >= 2.6 + 1.5
    toks = json.loads((out / "tokens.json").read_text())["tokens"]
    (mt,) = [k for k in toks if k["type"] == "mouth"]
    assert mt["fps"] == FPS and mt["mode"] == "open"
    plan = np.load(out / "plan.npz", allow_pickle=True)
    assert list(plan["mouth_pid"]) == ["ParamMouthOpenY"]
    mo = plan["mouth"][:, 0]
    t = np.arange(len(mo)) / FPS
    assert (mo[(t > 0.4) & (t < 1.0)] > 0.7).all()
    assert (mo[(t > 1.32) & (t < 1.58)] == 0).all()
    assert (mo[t > 2.45] == 0).all()                 # the tail is shut
    rp = _replay(log, tmp_path)
    np.testing.assert_array_equal(plan["traj"], rp["traj"])
    np.testing.assert_array_equal(plan["mouth"], rp["mouth"])
    np.testing.assert_array_equal(plan["part_opacity"], rp["part_opacity"])
    subs = json.loads((out / "subs.json").read_text(encoding="utf-8"))
    assert [abs(x["start"] - y) < 0.03 for x, y in zip(subs, (0.3, 1.6))] \
        == [True, True]


def test_no_idle_keeps_the_mouth(tmp_path):
    model, rig = _mouth_rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text("Hello there. [nod] How are you?", encoding="utf-8")
    audio = tmp_path / "said.wav"
    audio.write_bytes(wav_bytes(two_phrases()))
    out = tmp_path / "run"
    assert dr.main(["--model-json", str(model), "--rig", str(rig),
                    "--backend", "file", "--reply-file", str(reply),
                    "--out", str(out), "--probe-dir", str(tmp_path),
                    "--no-idle", "--tts", "file", "--tts-file",
                    str(audio)]) == 0
    toks = json.loads((out / "tokens.json").read_text())["tokens"]
    assert sum(k["type"] == "mouth" for k in toks) == 1


def test_tts_none_writes_nothing_new(tmp_path):
    model, rig = _mouth_rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text("Hello there. [nod] How are you?", encoding="utf-8")
    out = tmp_path / "run"
    assert dr.main(["--model-json", str(model), "--rig", str(rig),
                    "--backend", "file", "--reply-file", str(reply),
                    "--out", str(out), "--probe-dir", str(tmp_path)]) == 0
    assert not (out / "speech.wav").exists()
    acts = json.loads((out / "actions.json").read_text())
    assert "mouth" not in acts and "tts" not in acts
    toks = json.loads((out / "tokens.json").read_text())["tokens"]
    assert not any(k["type"] == "mouth" for k in toks)
    plan = np.load(out / "plan.npz", allow_pickle=True)
    assert "mouth" not in plan.files and "mouth_pid" not in plan.files


def test_lipsync_off_writes_no_mouth(tmp_path):
    """--lipsync off: speech and its timing, but no mouth anywhere -- the
    mouth tuning cannot change such a run."""
    model, rig = _mouth_rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text("Hello there. [nod] How are you?", encoding="utf-8")
    audio = tmp_path / "said.wav"
    audio.write_bytes(wav_bytes(two_phrases()))
    runs = []
    for k in range(2):
        out = tmp_path / f"run{k}"
        assert dr.main(["--model-json", str(model), "--rig", str(rig),
                        "--backend", "file", "--reply-file", str(reply),
                        "--out", str(out), "--probe-dir", str(tmp_path),
                        "--tts", "file", "--tts-file", str(audio),
                        "--lipsync", "off"]) == 0
        acts = json.loads((out / "actions.json").read_text())
        assert "mouth" not in acts and acts["tts"]["timing"] == "vad"
        toks = json.loads((out / "tokens.json").read_text())["tokens"]
        assert not any(t["type"] == "mouth" for t in toks)
        plan = np.load(out / "plan.npz", allow_pickle=True)
        assert "mouth" not in plan.files
        runs.append((toks, plan["traj"]))
    assert runs[0][0] == runs[1][0]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


def test_mouth_frames_resample_and_map_to_the_rig():
    from flow.runtime import mouth_frames, mouth_targets
    pids = ["ParamMouthForm", "ParamMouthOpenY", "ParamA", "ParamI"]
    bands = ["mouth"] * 4
    assert mouth_targets(pids, bands) == (1, {"a": 2, "i": 3})
    hi = np.array([1, 1, 1, 1.0])
    dflt = np.array([1, 0, 0, 0.0])
    tok = {"t": 0.0, "fps": 10.0, "open": [0.0, 1.0, 0.0],
           "vowels": {"a": [0, 0.5, 0], "i": [0, 0.25, 0]}}
    m = mouth_frames(tok, pids, bands, hi, dflt, 20.0, 8)
    assert set(m) == {"ParamMouthOpenY", "ParamA", "ParamI"}
    np.testing.assert_allclose(m["ParamMouthOpenY"],
                               [0, 0.5, 1, 0.5, 0, 0, 0, 0])
    np.testing.assert_allclose(m["ParamA"][2], 0.5)
    # No open parameter: the openness goes on ParamA.
    m = mouth_frames({"open": [0.2, 0.4]}, ["ParamA"], ["mouth"], hi[:1],
                     dflt[2:3], 30.0, 3)
    np.testing.assert_allclose(m["ParamA"], [0.2, 0.4, 0])


def test_mux_command(tmp_path):
    vid, wav, subs = (tmp_path / n for n in ("v.mp4", "s.wav", "s.srt"))
    dest = str(tmp_path / "d.mp4")
    assert dr.mux_cmd(str(vid), str(wav), str(subs), dest) is None
    wav.write_bytes(b"x")
    cmd = dr.mux_cmd(str(vid), str(wav), str(subs), dest)
    assert cmd[cmd.index("-map") + 3] == "1:a" and "-shortest" not in cmd
    subs.write_text("1\n")
    cmd = dr.mux_cmd(str(vid), str(wav), str(subs), dest)
    assert "2:s" in cmd and "mov_text" in cmd and "aac" in cmd


def test_vad_phrases_land_in_their_own_stretches():
    # Spoken slower than its share: "Ok." takes 0.6 s of 1.9 s voiced, which
    # proportional spreading (2 of 24 letters) would put inside the next.
    text = "Ok. Absolutely wonderful. Yes."
    w = [0.0 if (c.isspace() or not c.isalnum()) else 1.0 for c in text]
    spans = sp.vad_spans(text, w, [(0.0, 0.6), (0.9, 1.9), (2.2, 2.5)], 2.6)
    assert spans[text.index("A")][0] == pytest.approx(0.9)
    assert spans[text.index("Y")][0] == pytest.approx(2.2)
    assert spans[text.index(".")] == (pytest.approx(0.6),) * 2
    # An apostrophe is no phrase break.
    assert sp._chunks("you're fine. ok", [1] * 15) == [(0, 13), (13, 15)]


def test_edge_voice_is_chosen_once_per_reply(monkeypatch):
    voices = []
    _fake_edge(monkeypatch, wav_bytes(vowel(0.3)), [])
    Comm = sys.modules["edge_tts"].Communicate

    class Rec(Comm):
        def __init__(self, text, voice, **kw):
            voices.append(voice)
            super().__init__(text, voice, **kw)
    sys.modules["edge_tts"].Communicate = Rec
    p = dr.parse("えっ、[pause:0.5] 本当？")
    sp.speak(p.spoken, p.tags, sp.EdgeTTS(log=lambda s: None),
             lambda ch: 1 / 6, log=lambda s: None)
    assert voices == ["ja-JP-NanamiNeural"] * 2


def test_silent_rerun_drops_an_old_speech_wav(tmp_path):
    model, rig = _mouth_rig(tmp_path / "x")
    reply = tmp_path / "reply.txt"
    reply.write_text("Hello there. [nod] How are you?", encoding="utf-8")
    audio = tmp_path / "said.wav"
    audio.write_bytes(wav_bytes(two_phrases()))
    out = tmp_path / "run"
    base = ["--model-json", str(model), "--rig", str(rig), "--backend",
            "file", "--reply-file", str(reply), "--out", str(out),
            "--probe-dir", str(tmp_path)]
    assert dr.main(base + ["--tts", "file", "--tts-file", str(audio)]) == 0
    assert (out / "speech.wav").exists()
    assert dr.main(base) == 0
    assert not (out / "speech.wav").exists()
