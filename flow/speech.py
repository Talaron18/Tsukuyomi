#!/usr/bin/env python3
"""
Speech for the director: text-to-speech, tag timing from the audio, and a
lip-sync curve. numpy and the standard library only; ffmpeg decodes mp3.

  synth      a TTS backend turns one stretch of spoken text into mono audio
             and, when the service reports them, word boundaries:
               edge         edge-tts (Microsoft neural voices, free). Streams
                            WordBoundary events, offset/duration in 100 ns
                            ticks. Checked against edge-tts 7.3.1's source
                            (Communicate(text, voice, boundary=...),
                            stream_sync()); no live call was possible where
                            this was written.
               siliconflow  OpenAI-compatible POST {base}/audio/speech. No
                            word timings.
               voicevox     a VOICEVOX engine over HTTP (Japanese only: other
                            languages go to edge, logged). Sends its AudioQuery's
                            moras, timed by their phoneme lengths. The API is
                            from memory (GET /speakers, POST /audio_query,
                            POST /synthesis), UNVERIFIED against a live engine
                            where this was written; error bodies are surfaced.
               file         a recorded wav/mp3, used as is.
             VoiceRouter picks the backend per reply from its language and
             the rig's voice config (rigs/<rig>.voice.json), flags winning.
  speak      the reply is split at its [pause:x] tags, each piece synthesised
             and joined with that much silence, so a pause is heard. Every
             spoken character gets a (start, end) in the audio:
               words        from the word boundaries, matched in order
                            against the text;
               moras        VOICEVOX: the reply's kana and kanji runs aligned
                            to the moras (align_moras), then as words;
               vad          otherwise the characters' weights (the rate
                            model's seconds per character, punctuation and
                            spaces 0) are spread over the voiced stretches an
                            energy detector finds, silences skipped.
             A tag's time is its character's start, as in the rate model.
  mouth      per video frame: RMS level over a 1.5-frame window, in dB below
             the utterance's own loud level (95th percentile), gated with
             hysteresis, attack/release smoothed, snapped shut below 0.05.
             Optional vowel weights (a/i/u/e/o) from the first two spectral
             envelope peaks, or with VOICEVOX from the moras (mora_track:
             a/i/u/e/o/N/pau per frame, pauses and ッ closing the mouth).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import unicodedata
import wave

import numpy as np

DECODE_SR = 24000           # edge-tts sends 24 kHz mp3; lip sync needs less
TICKS = 1e7                 # edge-tts offsets are in 100 ns units

# Default edge voices by the script of the reply. UNVERIFIED live here: the
# names are from Microsoft's published voice list; --voice overrides.
EDGE_VOICES = {"ja": "ja-JP-NanamiNeural", "zh": "zh-CN-XiaoxiaoNeural",
               "ko": "ko-KR-SunHiNeural", "en": "en-US-AnaNeural"}
# SiliconFlow speech: UNVERIFIED names (from memory of its docs); knobs.
SF_TTS_MODEL = "FunAudioLLM/CosyVoice2-0.5B"
SF_TTS_VOICE = "anna"       # prefixed with the model: "<model>:anna"

# Voice activity, for alignment. Speech is gated at 30 dB under its loud
# level (and never under -50 dBFS, so a quiet file's noise is not speech).
# Gaps shorter than MIN_GAP are joined: stop closures inside a word ("ap-pt")
# are 50-120 ms of near silence, and a break there would push every later
# character into the wrong stretch. A real phrase break from the TTS voices
# is 200 ms or more.
VAD_HOP = 0.010
VAD_WIN = 0.030
VAD_RANGE_DB = 30.0
VAD_FLOOR_DB = -50.0
MIN_GAP = 0.18
MIN_VOICED = 0.05
# Word boundaries are used only if they cover at least this share of the
# spoken characters' weight; otherwise the text did not match what the
# service read (numbers read out, a voice that only marks sentences) and the
# energy alignment is the better guess.
MIN_COVER = 0.5

# Lip sync. Level maps linearly over MOUTH_SPAN dB under the loud level, so a
# vowel at the loud level opens fully, a consonant 10 dB down ~0.6 and
# anything 24 dB down (~27 dB under the peak of an edge voice) is shut. The
# gate is separate from the VAD's and does not join gaps: the mouth should
# close between phrases and dip between syllables. The level is read over
# MOUTH_WIN and smoothed with a fast attack and a jaw-like release.
#
# Measured on the server's edge runs (runs/demo/tts/{aidang_2,Hiyori}/edge,
# 30 fps; each frame's level over its own 33 ms; silent = under -40 dBFS;
# a syllable dip = a level minimum 8 dB under the maxima either side within
# 0.27 s). With a 50 ms window (1.5 frames) about the frame start and a
# 60 ms release, aidang_2's mouth was 0.43 in the -56 dBFS gap at 1.07 s
# and 0.41 / 0.23 at the dip at 1.47-1.50 s; open > 0.1 in 13% / 18% of
# silent frames, median 0.42 / 0.46 at dips (14% / 31% of them under 0.3),
# correlation with the level 0.86 / 0.87. With 25 ms over the frame's own
# stretch and a 25 ms release: 0.06 and 0.11 / 0.00 there, 4% / 5%, median
# 0.12 / 0.07 (79% / 77% under 0.3; the rest are shallow dips, 16-22 dB
# under the peak, which the level map itself leaves 0.3-0.5 open),
# correlation 0.95 / 0.96; mean open while voiced 0.77 -> 0.75 / 0.80 ->
# 0.78. The span is unchanged: the map was already linear in dB; the long
# window and release hid the dips. A shorter release does little more at
# 30 fps: one frame already decays to exp(-33/25) = 27%.
MOUTH_SPAN = 24.0
MOUTH_WIN = 0.025
MOUTH_GATE_DB = 30.0
MOUTH_HYST_DB = 3.0
MOUTH_FLOOR_DB = -55.0
ATTACK = 0.015
RELEASE = 0.025
SHUT = 0.05

# Vowel prototypes (F1, F2) in Hz, for an adult female voice -- Japanese
# five-vowel values (u unrounded, F2 ~1400). Distances in octaves.
VOWEL_F = {"a": (800, 1300), "i": (320, 2500), "u": (350, 1400),
           "e": (500, 2100), "o": (500, 900)}
VOWEL_OCT = 0.35


# ------------------------------------------------------------------ audio

def _from_wav(data):
    with wave.open(io.BytesIO(data)) as w:
        ch, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"{width * 8}-bit wav")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def decode(data):
    """(mono float32 in [-1, 1], sample rate) from wav or any format ffmpeg
    reads. PCM wav goes through the wave module; the rest (mp3, float wav)
    through ffmpeg at DECODE_SR."""
    if data[:4] == b"RIFF":
        try:
            return _from_wav(data)
        except (wave.Error, ValueError, EOFError):
            pass
    if not shutil.which("ffmpeg"):
        raise SystemExit("decoding this audio needs ffmpeg on PATH")
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le",
                        "-ac", "1", "-ar", str(DECODE_SR), "pipe:1"],
                       input=data, capture_output=True)
    if r.returncode:
        raise RuntimeError("ffmpeg could not decode the audio: "
                           + r.stderr.decode("utf-8", "replace")[-500:])
    return (np.frombuffer(r.stdout, "<i2").astype(np.float32) / 32768.0,
            DECODE_SR)


def read_audio(path):
    with open(path, "rb") as fh:
        return decode(fh.read())


def write_wav(path, x, sr):
    pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def _resample(x, sr, to):
    if sr == to or not len(x):
        return x
    n = int(round(len(x) * to / sr))
    return np.interp(np.arange(n) * sr / to, np.arange(len(x)),
                     x).astype(np.float32)


def _db(x, sr, centres, win):
    """RMS level in dBFS of a window of `win` seconds around each centre."""
    h = max(int(win * sr / 2), 1)
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    i = np.clip((np.asarray(centres) * sr).astype(int), 0, len(x))
    a, b = np.clip(i - h, 0, len(x)), np.clip(i + h, 0, len(x))
    ms = (c[b] - c[a]) / np.maximum(b - a, 1)
    return 10 * np.log10(ms + 1e-12)


def _loud(db):
    live = db[db > -70]
    return float(np.percentile(live, 95)) if len(live) else None


# -------------------------------------------------------------- alignment

def voiced(x, sr):
    """[(start, end)] seconds of speech, short gaps joined."""
    dur = len(x) / sr
    t = np.arange(0.0, dur, VAD_HOP)
    if not len(t):
        return []
    db = _db(x, sr, t, VAD_WIN)
    ref = _loud(db)
    if ref is None:
        return []
    on = max(ref - VAD_RANGE_DB, VAD_FLOOR_DB)
    off = on - 4.0
    segs, s, state = [], None, False
    for ti, d in zip(t, db):
        if not state and d > on:
            state, s = True, ti
        elif state and d < off:
            state = False
            segs.append([s, ti])
    if state:
        segs.append([s, dur])
    out = []
    for a, b in segs:
        if out and a - out[-1][1] < MIN_GAP:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(round(a, 4), round(b, 4)) for a, b in out if b - a >= MIN_VOICED]


def _warp(weights, segs):
    """(start, end) per character: cumulative weight spread linearly over
    the voiced time of `segs`, silences skipped. A weighted character starts
    in the stretch its weight begins in (at a boundary, the next one); a
    weightless one (punctuation, space) sits where the speech before it
    stopped."""
    lens = np.array([b - a for a, b in segs])
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    V = cum[-1]
    w = np.asarray(weights, float)
    W = w.sum()

    def right(c):          # the stretch with cum[k] <= c
        k = int(np.searchsorted(cum, c, side="right") - 1)
        k = min(max(k, 0), len(segs) - 1)
        return segs[k][0] + min(c - cum[k], lens[k])

    def left(c):           # the stretch with cum[k] < c
        k = int(np.searchsorted(cum, c, side="left") - 1)
        k = min(max(k, 0), len(segs) - 1)
        return segs[k][0] + max(min(c - cum[k], lens[k]), 0.0)

    if W <= 0:
        return [(segs[0][0], segs[0][0])] * len(w)
    out, acc = [], 0.0
    for wi in w:
        c0, c1 = acc / W * V, (acc + wi) / W * V
        if wi > 0:
            out.append((right(c0), left(c1)))
        else:
            out.append((left(c0) if c0 > 0 else segs[0][0],) * 2)
        acc += wi
    return out


# Alignment costs: squared log ratio of a group's voiced time to what its
# weight predicts, plus these per extra member. A phrase the voice reads
# with a breath inside it, or two clauses it runs together without a gap,
# both happen, but less often than the voice pausing at the punctuation.
SPLIT_COST = 0.3
JOIN_COST = 0.2


def _chunks(text, weights):
    """[(start, end)] character ranges ending after a phrase break: a
    punctuation mark followed by a space or the end, or a full-width one
    (。、！？). An apostrophe or hyphen inside a word is no break."""
    out, a, n = [], 0, len(text)
    for i, ch in enumerate(text):
        if unicodedata.category(ch)[0] != "P":
            continue
        nxt = text[i + 1] if i + 1 < n else " "
        if (nxt.isspace() or unicodedata.category(nxt)[0] == "P"
                or unicodedata.east_asian_width(ch) in ("W", "F")):
            if unicodedata.category(nxt)[0] == "P" and i + 1 < n:
                continue                       # "?!" breaks after the last
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if sum(weights[a:j]) > 0:
                out.append((a, j))
                a = j
    if a < n:
        if out and sum(weights[a:n]) <= 0:
            out[-1] = (out[-1][0], n)
        else:
            out.append((a, n))
    return out


def vad_spans(text, weights, segs, dur):
    """(start, end) per character from voiced stretches alone. The text's
    phrases (_chunks) are matched to the stretches by dynamic programming --
    each group is one phrase over one or more stretches, or several phrases
    over one -- so a sentence end lands in a silence rather than wherever
    its share of the characters falls; inside a group the weight is spread
    linearly (_warp). Proportional spreading alone put the second of two
    sentences 0.5 s early when the first was spoken slower than its share."""
    if not segs:
        segs = [(0.0, dur)]
    w = [float(x) for x in weights]
    ch = _chunks(text, w)
    if not ch:
        return _warp(w, segs)
    cw = [sum(w[a:b]) for a, b in ch]
    sl = [b - a for a, b in segs]
    m, k = len(ch), len(segs)
    rate = sum(sl) / max(sum(cw), 1e-9)

    def cost(i0, i1, j0, j1):      # chunks [i0, i1), stretches [j0, j1)
        e = sum(cw[i0:i1]) * rate
        d = sum(sl[j0:j1])
        return (np.log((d + 1e-3) / (e + 1e-3)) ** 2
                + SPLIT_COST * (j1 - j0 - 1) + JOIN_COST * (i1 - i0 - 1))

    INF = float("inf")
    best = [[INF] * (k + 1) for _ in range(m + 1)]
    back = [[None] * (k + 1) for _ in range(m + 1)]
    best[0][0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            cands = [(i - 1, jj) for jj in range(j)]           # 1 : many
            cands += [(ii, j - 1) for ii in range(i - 1)]      # many : 1
            for ii, jj in cands:
                if best[ii][jj] == INF:
                    continue
                c = best[ii][jj] + cost(ii, i, jj, j)
                if c < best[i][j]:
                    best[i][j], back[i][j] = c, (ii, jj)
    out = [None] * len(text)
    i, j = m, k
    while i > 0:
        ii, jj = back[i][j]
        a, b = ch[ii][0], ch[i - 1][1]
        out[a:b] = _warp(w[a:b], segs[jj:j])
        i, j = ii, jj
    return out


def word_spans(text, weights, words, dur):
    """(spans, covered weight share) from word boundaries [(word, start,
    end)], matched in order against the text (case-insensitive, at most 40
    characters skipped). Inside a word the time is spread by weight; an
    unmatched stretch between words takes the time between them the same
    way, and a weightless one sits at the end of the word before it."""
    low = text.lower()
    w = np.asarray(weights, float)
    n = len(text)
    owner = [None] * n
    marks, cur = [], 0
    for word, s, e in words:
        key = word.strip().lower()
        if not key:
            continue
        j = low.find(key, cur)
        if j < 0 or j - cur > 40:
            continue
        marks.append((j, j + len(key), float(s), float(e)))
        cur = j + len(key)
    spans = [None] * n

    def spread(a, b, t0, t1):
        tot = w[a:b].sum()
        acc = 0.0
        for j in range(a, b):
            if tot > 0:
                spans[j] = (t0 + (t1 - t0) * acc / tot,
                            t0 + (t1 - t0) * (acc + w[j]) / tot)
                if w[j] <= 0:
                    spans[j] = (spans[j][0],) * 2
            else:
                spans[j] = (t0, t0)
            acc += w[j]

    covered = 0.0
    prev_c, prev_t = 0, None
    for c0, c1, s, e in marks:
        if c0 > prev_c:
            gap = w[prev_c:c0].sum()
            t0 = prev_t if prev_t is not None else (0.0 if gap > 0 else s)
            spread(prev_c, c0, t0, s if gap > 0 else t0)
        spread(c0, c1, s, e)
        covered += w[c0:c1].sum()
        prev_c, prev_t = c1, e
    if prev_c < n:
        t0 = prev_t if prev_t is not None else 0.0
        spread(prev_c, n, t0, dur if w[prev_c:].sum() > 0 else t0)
    share = covered / w.sum() if w.sum() > 0 else 0.0
    return [(round(a, 4), round(b, 4)) for a, b in spans], share


def speech_end(spans, segs, dur):
    ends = [b for _, b in spans]
    if segs:
        ends.append(segs[-1][1])
    return min(max(ends) if ends else 0.0, dur)


# ------------------------------------------------------------------- TTS

def language(text):
    """'ja' with any kana, 'ko' with Hangul, 'zh' with other CJK
    ideographs, else 'en'."""
    han = False
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:
            return "ja"
        if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            return "ko"
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            han = True
    return "zh" if han else "en"


def post(url, body, key, name, timeout=120, retries=4):
    """POST JSON with a bearer key; the open response. The response body is
    surfaced on errors, since that is where SiliconFlow says what was wrong;
    429 and 5xx are retried with backoff, other 4xx are fatal."""
    import time
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    last = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:1000].decode("utf-8", "replace")
            last = RuntimeError(f"{exc.code} from {name}: {detail}")
            if exc.code != 429 and exc.code < 500:
                raise last from None
        except (urllib.error.URLError, TimeoutError) as exc:
            last = RuntimeError(f"{name}: {exc}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise last


class EdgeTTS:
    """edge-tts. Word boundaries need boundary="WordBoundary" from 7.0 on
    (the default became SentenceBoundary); older versions have no such
    argument and send word boundaries anyway."""

    whole = False

    def __init__(self, voice=None, rate="+0%", log=print):
        self.voice, self.rate, self.log = voice, rate, log
        self.last_voice = None
        self._now = None

    def begin(self, spoken):
        """Pick the voice from the whole reply: a piece after a [pause]
        ("本当？") has no kana and would otherwise be read in Chinese."""
        self._now = self.voice or EDGE_VOICES[language(spoken)]

    def synth(self, text):
        try:
            import edge_tts               # lazy: only this backend needs it
        except ImportError as exc:
            raise SystemExit(f"--tts edge needs edge-tts (pip install "
                             f"edge-tts): {exc}") from exc
        voice = self._now or self.voice or EDGE_VOICES[language(text)]
        self.last_voice = voice
        try:
            c = edge_tts.Communicate(text, voice, rate=self.rate,
                                     boundary="WordBoundary")
        except TypeError:
            c = edge_tts.Communicate(text, voice, rate=self.rate)
        if hasattr(c, "stream_sync"):
            items = c.stream_sync()
        else:
            import asyncio

            async def collect():
                return [m async for m in c.stream()]
            items = asyncio.run(collect())
        audio, words = [], []
        for m in items:
            if m["type"] == "audio":
                audio.append(m["data"])
            elif m["type"] == "WordBoundary":
                s = float(m["offset"]) / TICKS
                words.append((m["text"], s, s + float(m["duration"]) / TICKS))
        if not audio:
            raise RuntimeError(f"edge-tts sent no audio for voice {voice}")
        x, sr = decode(b"".join(audio))
        return x, sr, words


class SiliconFlowTTS:
    """OpenAI-compatible /audio/speech. Model and voice names UNVERIFIED;
    a voice without ':' is prefixed with the model, as SiliconFlow's system
    voices are named "<model>:<voice>"."""

    whole = False

    def __init__(self, base, key_env, model=None, voice=None, fmt="wav",
                 sample_rate=None, log=print):
        self.url = base.rstrip("/") + "/audio/speech"
        self.key_env, self.fmt, self.sr = key_env, fmt, sample_rate
        self.model = model or SF_TTS_MODEL
        v = voice or SF_TTS_VOICE
        self.voice = v if ":" in v else f"{self.model}:{v}"
        self.last_voice, self.log = self.voice, log

    def synth(self, text):
        key = os.environ.get(self.key_env)
        if not key:
            raise SystemExit(f"--tts siliconflow needs {self.key_env} in the "
                             "environment")
        body = {"model": self.model, "input": text, "voice": self.voice,
                "response_format": self.fmt, "stream": False}
        if self.sr:
            body["sample_rate"] = int(self.sr)
        import http.client
        for attempt in range(3):
            try:
                with post(self.url, body, key, "siliconflow tts") as fh:
                    data = fh.read()
                break
            except (http.client.IncompleteRead, ConnectionError) as exc:
                # The connection dropped mid-download on the server (Hiyori:
                # 226784 bytes read, 36430 more expected); synthesis is
                # stateless, so ask again.
                if attempt == 2:
                    raise RuntimeError(f"siliconflow tts: download cut off "
                                       f"three times ({exc!r})") from None
                self.log(f"siliconflow tts: download cut off ({exc!r}); "
                         "retrying")
        if data[:1] in (b"{", b"["):
            raise RuntimeError("siliconflow tts answered with JSON, not audio: "
                               + data[:1000].decode("utf-8", "replace"))
        x, sr = decode(data)
        return x, sr, None


class FileTTS:
    """A recorded reply. The whole file is the speech: [pause] tags are not
    inserted, the energy alignment finds the silences that are there."""

    whole = True

    def __init__(self, path):
        self.path = path
        self.last_voice = os.path.basename(path)

    def synth(self, text):
        x, sr = read_audio(self.path)
        return x, sr, None


# -------------------------------------------------------------- VOICEVOX

VOICEVOX_URL = "http://127.0.0.1:50021"
# AudioQuery knobs a voice config or flag may set: our name -> its field.
VOICEVOX_KNOBS = {"speed": "speedScale", "pitch": "pitchScale",
                  "intonation": "intonationScale", "volume": "volumeScale"}
SMALL_KANA = set("ァィゥェォャュョヮぁぃぅぇぉゃゅょゎ")
# Kana the engine reads as another mora: particles は/へ/を and the
# yotsugana. Matched at a small cost, not as a mismatch.
KANA_READ = {("ハ", "ワ"), ("ヘ", "エ"), ("ヲ", "オ"), ("ヅ", "ズ"),
             ("ヂ", "ジ")}
# Alignment costs (align_moras). A kana that is not the mora there costs
# MISMATCH, less than dropping it and the mora (DROP each), so a one-off
# reading difference does not shift the rest. A kanji/Latin/digit run takes
# 1 .. RUN_MORAS * its length moras at RUN_COST each: cheap, so the kana
# around it decide where it ends.
MISMATCH = 0.6
DROP = 1.0
RUN_MORAS = 4
RUN_MAX = 16
RUN_COST = 0.05
# Rescaling the summed phoneme lengths to the decoded audio: a ratio off by
# more than this is logged (the engine rounds each phoneme to its frame
# rate, ~93.75 Hz from memory, so a few percent is expected).
RATIO_WARN = 0.05


def _kata(ch):
    o = ord(ch)
    return chr(o + 0x60) if 0x3041 <= o <= 0x3096 else ch


def _is_kana(ch):
    o = ord(ch)
    return (0x3041 <= o <= 0x309F or 0x30A1 <= o <= 0x30FA or ch == "ー"
            or 0x31F0 <= o <= 0x31FF)


def kana_units(text):
    """[(kind, reading, start, end)] over `text`: "k" one kana mora (a kana
    with the small ゃ/ゅ/ょ... after it, in katakana; ー alone), "x" a run
    of anything else spoken (kanji, Latin, digits), "p" punctuation or a
    symbol, "s" white space."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        cat = unicodedata.category(ch)
        if ch.isspace():
            out.append(("s", "", i, i + 1))
            i += 1
        elif _is_kana(ch) and ch not in SMALL_KANA:
            j = i + 1
            while j < n and text[j] in SMALL_KANA:
                j += 1
            out.append(("k", "".join(_kata(c) for c in text[i:j]), i, j))
            i = j
        elif cat[0] in "PS" or ch in SMALL_KANA:
            out.append(("p", "", i, i + 1))
            i += 1
        else:
            j = i + 1
            while (j < n and not text[j].isspace() and not _is_kana(text[j])
                   and unicodedata.category(text[j])[0] not in "PS"):
                j += 1
            out.append(("x", text[i:j], i, j))
            i = j
    return out


def query_moras(query, dur=None, log=print):
    """[{"text", "vowel", "start", "vstart", "end"}] seconds from an
    AudioQuery: prePhonemeLength, then each accent phrase's moras
    (consonant_length + vowel_length) and its pause_mora, all divided by
    speedScale (the engine adds the pre/post silences as phonemes before
    applying the speed, from memory). A pause mora's vowel is "pau";
    pauseLength replaces and pauseLengthScale scales its length where the
    engine has them. With `dur` (the decoded audio's seconds) the times are
    rescaled so the total, postPhonemeLength included, is the audio's: the
    query's lengths are what was synthesised, the engine's frame rounding
    and any unknown in the above are absorbed by one ratio, logged when it
    is far from 1."""
    speed = float(query.get("speedScale") or 1.0)
    p_len = query.get("pauseLength")
    p_scale = float(query.get("pauseLengthScale") or 1.0)
    t = float(query.get("prePhonemeLength") or 0.0) / speed
    out = []
    for ap in query.get("accent_phrases") or ():
        for mo in ap.get("moras") or ():
            c = float(mo.get("consonant_length") or 0.0) / speed
            v = float(mo.get("vowel_length") or 0.0) / speed
            out.append({"text": mo.get("text") or "",
                        "vowel": mo.get("vowel") or "pau",
                        "consonant": mo.get("consonant"),
                        "start": t, "vstart": t + c, "end": t + c + v})
            t += c + v
        pm = ap.get("pause_mora")
        if pm:
            v = float(pm.get("vowel_length") or 0.0)
            if p_len is not None:
                v = float(p_len)
            v = v * p_scale / speed
            out.append({"text": pm.get("text") or "、", "vowel": "pau",
                        "consonant": None, "start": t, "vstart": t,
                        "end": t + v})
            t += v
    total = t + float(query.get("postPhonemeLength") or 0.0) / speed
    if dur and total > 0:
        r = dur / total
        if abs(r - 1) > RATIO_WARN:
            log(f"voicevox: audio {dur:.2f}s vs query {total:.2f}s (x{r:.3f})"
                "; mora times rescaled to the audio")
        for m in out:
            for k in ("start", "vstart", "end"):
                m[k] *= r
    return out


def _kana_eq(u, m):
    if u == m["text"]:
        return 0.0
    if (u, m["text"]) in KANA_READ:
        return 0.1
    if u == "ー" and not m.get("consonant") and m["vowel"] != "pau":
        return 0.1                     # a long vowel the engine spelt out
    return MISMATCH


def align_moras(text, moras):
    """[(unit text, start, end)] for the spoken units of `text` that got
    moras, in order. Dynamic programming over (units, moras): a kana unit
    takes one mora (free when it is that mora), a kanji/Latin/digit run
    1..RUN_MORAS per character, punctuation and spaces none; pause moras
    are skipped free, other moras and units dropped at DROP.

    Why not spread by phrase: the reply is mixed kanji and kana, and the
    engine's reading of a kanji is unknown here, but every kana is a fixed
    point -- in typical replies most characters are kana -- so a kanji run
    lands between the kana around it and only its inside is a guess (its
    moras' span, start to end). A unit that matched nothing is left to
    word_spans, which spreads the time between its neighbours."""
    units = kana_units(text)
    U, M = len(units), len(moras)
    INF = float("inf")
    best = [[INF] * (M + 1) for _ in range(U + 1)]
    back = [[None] * (M + 1) for _ in range(U + 1)]
    best[0][0] = 0.0
    pau = [m["vowel"] in ("pau",) for m in moras]

    def relax(i, j, c, frm):
        if c < best[i][j]:
            best[i][j], back[i][j] = c, frm

    for i in range(U + 1):
        for j in range(M + 1):
            c0 = best[i][j]
            if c0 == INF:
                continue
            if j < M:                          # skip a mora
                relax(i, j + 1, c0 + (0.0 if pau[j] else DROP), (i, j, 0))
            if i == U:
                continue
            kind, rd, a, b = units[i]
            if kind in "ps":
                relax(i + 1, j, c0, (i, j, 0))
                continue
            relax(i + 1, j, c0 + DROP * (len(rd) if kind == "x" else 1),
                  (i, j, 0))                   # drop the unit
            if kind == "k":
                if j < M and not pau[j]:
                    relax(i + 1, j + 1, c0 + _kana_eq(rd, moras[j]),
                          (i, j, 1))
                continue
            for m in range(1, min(RUN_MORAS * len(rd), RUN_MAX) + 1):
                if j + m > M or pau[j + m - 1]:
                    break
                relax(i + 1, j + m, c0 + RUN_COST * m, (i, j, m))
    out = []
    i, j = U, M
    while i > 0 or j > 0:
        pi, pj, m = back[i][j]
        if pi < i and m > 0:
            kind, rd, a, b = units[pi]
            out.append((text[a:b], moras[pj]["start"],
                        moras[pj + m - 1]["end"]))
        i, j = pi, pj
    return out[::-1]


def _open_url(url):
    """An opener: no proxy for a local engine (an http_proxy from the
    environment answered localhost with 502 for Ollama on the server)."""
    import urllib.parse
    import urllib.request
    host = urllib.parse.urlsplit(url).hostname or ""
    if host in ("localhost", "127.0.0.1", "::1") or host.startswith("127."):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


class VoicevoxTTS:
    """A VOICEVOX engine. `voice`: a style id, "character/style" or a
    character (its first style), resolved through GET /speakers; None takes
    the first talk style listed. Knobs (VOICEVOX_KNOBS) are written into
    the AudioQuery before synthesis. The voice libraries' terms ask for a
    credit, "VOICEVOX:<character>", wherever the audio is published:
    `credit` has it."""

    whole = False
    timing = "moras"

    def __init__(self, url=VOICEVOX_URL, voice=None, knobs=None, log=print,
                 timeout=120, opener=None):
        self.url = (url or VOICEVOX_URL).rstrip("/")
        self.voice, self.knobs = voice, dict(knobs or {})
        self.log, self.timeout = log, timeout
        self.opener = opener or _open_url(self.url)
        self._speakers = None
        self.speaker = self.character = self.last_voice = None

    @property
    def credit(self):
        return f"VOICEVOX:{self.character}" if self.character else "VOICEVOX"

    def _req(self, method, path, params=None, body=None):
        import urllib.error
        import urllib.parse
        import urllib.request
        url = self.url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        # POST /audio_query has no body; urllib would send a GET without
        # data, so an empty one.
        data = (json.dumps(body, ensure_ascii=False).encode("utf-8")
                if body is not None else (b"" if method == "POST" else None))
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"} if body is not None
            else {})
        try:
            with self.opener.open(req, timeout=self.timeout) as fh:
                return fh.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:1000].decode("utf-8", "replace")
            raise RuntimeError(f"{exc.code} from voicevox {method} {path}: "
                               f"{detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise RuntimeError(
                f"voicevox at {self.url} unreachable ({exc}); is the engine "
                "running? e.g. docker run --rm -p 127.0.0.1:50021:50021 "
                "voicevox/voicevox_engine:cpu-latest (image name unverified)"
            ) from None

    def speakers(self, wait=20.0):
        """GET /speakers. An empty list is asked again for up to `wait`
        seconds: on the server one run got [] from an engine that listed 45
        characters when curl asked moments later (most likely still loading
        its voice libraries)."""
        import time
        if self._speakers is None:
            t0 = time.monotonic()
            while True:
                raw = self._req("GET", "/speakers")
                try:
                    got = json.loads(raw)
                except ValueError:
                    raise RuntimeError("voicevox /speakers is not JSON: "
                                       + raw[:500].decode("utf-8", "replace"))
                if got or time.monotonic() - t0 >= wait:
                    break
                self.log("voicevox: /speakers is empty; waiting for the "
                         "engine to load its voices")
                time.sleep(2.0)
            self._speakers, self._speakers_raw = got, raw[:300]
        return self._speakers

    def styles(self):
        """[(id, character, style)] of the talk styles."""
        out = []
        for sp_ in self.speakers():
            for st in sp_.get("styles") or ():
                if st.get("type", "talk") in (None, "talk"):
                    out.append((int(st["id"]), str(sp_.get("name")),
                                str(st.get("name"))))
        return out

    def resolve(self):
        if self.speaker is not None:
            return self.speaker
        st = self.styles()
        v = None if self.voice is None else str(self.voice).strip()
        hit = None
        if v is None:
            if not st:
                raise RuntimeError("voicevox lists no talk styles")
            hit = st[0]
            self.log(f"voicevox: no voice given; using {hit[1]}/{hit[2]} "
                     f"(id {hit[0]})")
        elif v.isdigit():
            hit = next((s for s in st if s[0] == int(v)), None)
            if hit is None:
                hit = (int(v), f"speaker {v}", "?")
                self.log(f"voicevox: style id {v} is not in /speakers; "
                         "using it anyway")
        else:
            name, _, style = v.rpartition("/") if "/" in v else (v, "", "")
            for exact in (True, False):
                def eq(x, y):
                    return x == y if exact else x.casefold() == y.casefold()
                hit = next((s for s in st if eq(s[1], name)
                            and (not style or eq(s[2], style))), None)
                if hit:
                    break
            if hit is None and not st:
                # Seen on the server: the engine answered /speakers with no
                # voices at all, and the message ended in an empty list.
                raise RuntimeError(
                    f"voicevox: the engine at {self.url} lists no voices "
                    f"(GET /speakers returned {len(self.speakers())} "
                    f"characters: {getattr(self, '_speakers_raw', b'')!r}); "
                    "its voice models may still be loading -- check "
                    f"`curl --noproxy '*' {self.url}/speakers` and the "
                    "engine's log")
            if hit is None:
                raise RuntimeError(
                    f"voicevox: no voice {v!r}; available: "
                    + ", ".join(f"{c}/{s} ({i})" for i, c, s in st))
        self.speaker, self.character = hit[0], hit[1]
        self.last_voice = f"{hit[1]}/{hit[2]} ({hit[0]})"
        return self.speaker

    def query(self, text):
        sid = self.resolve()
        raw = self._req("POST", "/audio_query",
                        {"text": text, "speaker": sid})
        try:
            q = json.loads(raw)
        except ValueError:
            raise RuntimeError("voicevox /audio_query is not JSON: "
                               + raw[:500].decode("utf-8", "replace"))
        for k, v in self.knobs.items():
            if v is not None and k in VOICEVOX_KNOBS:
                q[VOICEVOX_KNOBS[k]] = float(v)
        return q

    def synth(self, text):
        """(audio, sr, words, moras): words from align_moras over the mora
        times, which are rescaled to the audio's length."""
        q = self.query(text)
        data = self._req("POST", "/synthesis", {"speaker": self.speaker},
                         body=q)
        if data[:4] != b"RIFF":
            raise RuntimeError("voicevox /synthesis did not send a wav: "
                               + data[:500].decode("utf-8", "replace"))
        x, sr = decode(data)
        moras = query_moras(q, len(x) / sr, self.log)
        words = align_moras(text, moras)
        return x, sr, words, moras


# ---------------------------------------------------------- voice config

# Per-language voice settings, from rigs/<rig>.voice.json or --voice-config:
#   {"ja": {"tts": "voicevox", "voice": "冥鳴ひまり/ノーマル", "speed": 1.0},
#    "en": {"tts": "edge", "voice": "en-US-AnaNeural", "rate": "+0%"},
#    "default": {...}}
# Languages are language()'s: ja, zh, ko, en; "default" covers the rest.
VOICE_KEYS = {"tts", "voice", "rate", "model", *VOICEVOX_KNOBS}
VOICE_LANGS = ("ja", "zh", "ko", "en", "default")
TTS_KINDS = ("edge", "siliconflow", "voicevox", "file")


def load_voice_config(path, log=print):
    """{language: {key: value}} from a voice config file; {} when there is
    none. Unknown languages and keys are dropped with a line each."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    if not isinstance(blob, dict):
        raise SystemExit(f"{path}: a voice config is a JSON object "
                         "language -> settings")
    out = {}
    for lang, entry in blob.items():
        if lang.startswith("_"):
            continue                       # "_comment" and the like
        if lang not in VOICE_LANGS or not isinstance(entry, dict):
            log(f"voice config {path}: {lang!r} ignored (languages: "
                + ", ".join(VOICE_LANGS) + ")")
            continue
        bad = set(entry) - VOICE_KEYS
        if bad:
            log(f"voice config {path}: {lang}: unknown keys "
                + ", ".join(sorted(bad)) + " ignored")
        if entry.get("tts") and entry["tts"] not in TTS_KINDS:
            raise SystemExit(f"{path}: {lang}: unknown tts {entry['tts']!r}")
        out[lang] = {k: v for k, v in entry.items() if k in VOICE_KEYS}
    return out


def voice_settings(lang, tts, cli, cfg, log=print):
    """(backend kind, settings) for replies in `lang`.

    `tts`: the --tts flag ("auto": the config's choice, edge where it names
    none). `cli`: flag values (voice, rate, speed, ...; None = not given).
    The config entry for the language (else "default") applies when it
    names the same backend or none; a flag wins over it. VOICEVOX speaks
    Japanese only: another language falls back to edge, with that
    language's edge entry if the config has one and no voice flag (the
    flag named a VOICEVOX voice)."""
    entry = cfg.get(lang) or cfg.get("default") or {}
    kind = entry.get("tts", "edge") if tts == "auto" else tts
    use = entry if entry.get("tts", kind) == kind else {}
    out = {k: v for k, v in use.items() if k != "tts"}
    out.update({k: v for k, v in cli.items() if v is not None})
    if kind == "voicevox" and lang != "ja":
        log(f"speech: VOICEVOX speaks Japanese only; this {lang} reply goes "
            "to edge")
        kind = "edge"
        out = {k: v for k, v in (entry if entry.get("tts", "edge") == "edge"
                                 else {}).items() if k != "tts"}
        if cli.get("rate") is not None:
            out["rate"] = cli["rate"]
    return kind, out


class VoiceRouter:
    """The TTS for a reply, chosen in begin() from its language: `make(kind,
    settings)` builds a backend, cached per (kind, settings). speak() reads
    `whole` after begin(), so the choice is made there."""

    def __init__(self, tts, cli, cfg, make, log=print):
        self.tts, self.cli, self.cfg = tts, dict(cli), cfg
        self.make, self.log = make, log
        self._made = {}
        self.cur = self.kind = None

    def pick(self, lang):
        kind, st = voice_settings(lang, self.tts, self.cli, self.cfg,
                                  self.log)
        key = (kind, tuple(sorted((k, str(v)) for k, v in st.items())))
        if key not in self._made:
            self._made[key] = self.make(kind, st)
        self.kind = kind
        return self._made[key]

    def begin(self, spoken):
        self.cur = self.pick(language(spoken))
        if hasattr(self.cur, "begin"):
            self.cur.begin(spoken)

    @property
    def whole(self):
        return bool(getattr(self.cur, "whole", False))

    @property
    def timing(self):
        return getattr(self.cur, "timing", None)

    @property
    def last_voice(self):
        return getattr(self.cur, "last_voice", None)

    @property
    def credit(self):
        return getattr(self.cur, "credit", None)

    def synth(self, text):
        if self.cur is None:
            self.begin(text)
        return self.cur.synth(text)


# ----------------------------------------------------------------- speak

def speak(spoken, tags, tts, weight, log=print):
    """Synthesize `spoken` and time it. Returns {audio, sr, spans (per
    character, seconds), acts (non-pause tags with t, as timeline()), end
    (audio seconds), speech_end, method, moras (VOICEVOX: query_moras of
    every piece on the joined audio's clock, else None)}.

    `weight(ch)`: seconds a character takes in the rate model, 0 for one
    that is silent. Pause tags split the text; each piece is synthesised
    alone and followed by the pause's silence (FileTTS: not split)."""
    n = len(spoken)
    if hasattr(tts, "begin"):
        tts.begin(spoken)
    pauses = [(t["at"], float(t["arg"])) for t in tags if t["tag"] == "pause"]
    if tts.whole and pauses:
        log(f"speech: {len(pauses)} [pause] tags taken from the recording's "
            "own silences")
        pauses = []
    cuts = [0] + [at for at, _ in pauses] + [n]
    w_all = [0.0 if (ch.isspace() or unicodedata.category(ch)[0] == "P")
             else weight(ch) for ch in spoken]
    pieces, spans, seg_end, methods = [], [None] * n, [], []
    moras = None
    clock, sr = 0.0, None
    for k in range(len(cuts) - 1):
        if k > 0:
            pieces.append(("sil", pauses[k - 1][1]))
            clock += pauses[k - 1][1]
        a, b = cuts[k], cuts[k + 1]
        text = spoken[a:b]
        body = text.strip()
        if not body:
            for j in range(a, b):
                spans[j] = (clock, clock)
            seg_end.append(clock)
            continue
        lead = len(text) - len(text.lstrip())
        res = tts.synth(body)
        x, xsr, words = res[:3]
        if len(res) > 3 and res[3] is not None:
            moras = (moras or []) + [
                {**m, **{k: clock + m[k] for k in ("start", "vstart", "end")}}
                for m in res[3]]
        if sr is None:
            sr = xsr
        x = _resample(x, xsr, sr)
        dur = len(x) / sr
        wts = w_all[a + lead:a + lead + len(body)]
        segs = voiced(x, sr)
        loc, method = None, "vad"
        if words:
            loc, share = word_spans(body, wts, words, dur)
            if share >= MIN_COVER:
                method = getattr(tts, "timing", None) or "words"
            else:
                log(f"speech: word boundaries cover {share:.0%} of the text; "
                    "aligning by voiced energy instead")
                loc = None
        if loc is None:
            loc = vad_spans(body, wts, segs, dur)
        end_loc = speech_end(loc, segs if method == "vad" else [], dur)
        methods.append(method)
        first = loc[0][0] if loc else 0.0
        for j in range(a, b):
            r = j - a - lead
            if r < 0:
                spans[j] = (clock + first,) * 2
            elif r >= len(body):
                spans[j] = (clock + end_loc,) * 2
            else:
                spans[j] = (clock + loc[r][0], clock + loc[r][1])
        seg_end.append(clock + end_loc)
        pieces.append(("pcm", x))
        clock += dur
    sr = sr or DECODE_SR
    audio = np.concatenate(
        [np.zeros(int(round(v * sr)), np.float32) if kind == "sil" else v
         for kind, v in pieces] or [np.zeros(0, np.float32)])
    end = len(audio) / sr
    acts, k = [], 0
    for t in tags:
        if t["tag"] == "pause":
            k += 1 if not tts.whole else 0
            continue
        i = t["at"]
        kk = min(k, len(seg_end) - 1)
        if i < cuts[kk + 1] and i < n:
            tt = spans[i][0]
        else:
            tt = seg_end[kk] if seg_end else 0.0
        acts.append({"t": round(float(tt), 4), "tag": t["tag"],
                     "arg": t["arg"], "at": i})
    return {"audio": audio, "sr": sr,
            "spans": [(round(float(a), 4), round(float(b), 4))
                      for a, b in spans],
            "acts": acts, "end": end,
            "speech_end": max(seg_end) if seg_end else 0.0,
            "method": "+".join(sorted(set(methods))) or "none",
            "moras": moras}


# -------------------------------------------------------------- lip sync

def mouth_open(x, sr, fps, n_frames):
    """Mouth openness in [0, 1] per frame (frame f at f / fps)."""
    t = np.arange(n_frames) / fps
    out = np.zeros(n_frames, np.float32)
    if not len(x) or not n_frames:
        return out
    # Frame f is on screen from f / fps to (f + 1) / fps: read the level
    # over that stretch (centred half a frame on), not around its start.
    # The mouth then leads its sound by at most half a frame, the side
    # viewers tolerate (audio late by up to ~125 ms passes, early by
    # ~45 ms shows: ITU-R BT.1359).
    db = _db(x, sr, t + 0.5 / fps, MOUTH_WIN)
    db[t * sr >= len(x)] = -120.0
    ref = _loud(db)
    if ref is None:
        return out
    on = max(ref - MOUTH_GATE_DB, MOUTH_FLOOR_DB)
    off = on - MOUTH_HYST_DB
    lvl = np.clip((db - (ref - MOUTH_SPAN)) / MOUTH_SPAN, 0, 1)
    dt = 1.0 / fps
    ka, kr = 1 - np.exp(-dt / ATTACK), 1 - np.exp(-dt / RELEASE)
    y, gate = 0.0, False
    for f in range(n_frames):
        gate = db[f] > (off if gate else on)
        target = lvl[f] if gate else 0.0
        y += (ka if target > y else kr) * (target - y)
        out[f] = 0.0 if y < SHUT else y
    return out


def _envelope_peaks(x, sr, c, win=0.04):
    h = int(win * sr / 2)
    i = int(c * sr)
    seg = x[max(i - h, 0):i + h]
    if len(seg) < 16:
        return None
    seg = seg * np.hanning(len(seg))
    nfft = 4096
    mag = np.abs(np.fft.rfft(seg, nfft))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    # Smooth over ~300 Hz so the harmonics (f0 150-300 Hz) merge into the
    # formant envelope.
    k = max(int(300 / (f[1] - f[0])), 1)
    env = np.convolve(mag, np.hanning(k) / np.hanning(k).sum(), "same")

    def peak(lo, hi):
        m = (f >= lo) & (f <= hi)
        return float(f[m][np.argmax(env[m])]) if m.any() else None
    f1 = peak(200, 1000)
    f2 = peak(max(f1 + 250, 800), 3000) if f1 else None
    return (f1, f2) if f1 and f2 else None


def vowels(x, sr, fps, open_):
    """{"a": [...], ...}: soft vowel weights per frame times the openness
    (they sum to it), nearest of VOWEL_F in octaves."""
    n = len(open_)
    out = {v: np.zeros(n, np.float32) for v in VOWEL_F}
    for fr in range(n):
        if open_[fr] <= 0:
            continue
        pk = _envelope_peaks(x, sr, fr / fps)
        if pk is None:
            out["a"][fr] = open_[fr]
            continue
        d = {v: ((np.log2(pk[0] / p1)) ** 2 + (np.log2(pk[1] / p2)) ** 2)
             / VOWEL_OCT ** 2 for v, (p1, p2) in VOWEL_F.items()}
        w = {v: np.exp(-dv / 2) for v, dv in d.items()}
        s = sum(w.values())
        for v in out:
            out[v][fr] = open_[fr] * (w[v] / s if s > 1e-9 else 0.2)
    return out


# Mora-driven mouth (VOICEVOX). Each frame reads the moras over its own
# stretch widened by MORA_XFADE either side, sampled MORA_SUB times, so a
# vowel change cross-fades over ~2 frames instead of switching. A consonant
# shows its mora's vowel (the lips anticipate it). ん keeps the mouth at
# most N_OPEN open, shaped as u; ッ, pauses and anything outside the moras
# close it. Hand-set: ん is a nasal with the lips near closed, ッ a held
# closure.
MORA_XFADE = 0.02
MORA_SUB = 7
N_OPEN = 0.2
VISEMES = ("a", "i", "u", "e", "o", "N", "pau")


def mora_weights(moras, fps, n):
    """(gate, {vowel: share}, visemes) per frame from the mora timeline
    (speak's "moras"): how much of the level-based openness may show (0 in
    pauses, N_OPEN in ん), the share of each vowel shape (summing to 1 where
    any vowel or ん is heard; ん counts as u) and the frame's main viseme.
    Devoiced vowels (upper case) keep their shape."""
    starts = np.array([m["start"] for m in moras]) if moras else np.zeros(0)
    labs = []
    for m in moras or ():
        v = m["vowel"]
        labs.append("N" if v == "N" else v.lower() if v.lower() in "aiueo"
                    and len(v) == 1 else "pau")
    gate = np.zeros(n)
    share = {v: np.zeros(n) for v in VOWEL_F}
    vis = []
    off = np.linspace(-MORA_XFADE, 1.0 / fps + MORA_XFADE, MORA_SUB)
    for f in range(n):
        c = dict.fromkeys(VISEMES, 0)
        for t in f / fps + off:
            k = int(np.searchsorted(starts, t, side="right") - 1)
            c[labs[k] if 0 <= k and t < moras[k]["end"] else "pau"] += 1
        vis.append(max(VISEMES, key=lambda v: c[v]))
        gate[f] = ((MORA_SUB - c["pau"]) / MORA_SUB
                   * (1 - (1 - N_OPEN) * c["N"] / MORA_SUB))
        tot = sum(c[v] for v in "aiueo") + c["N"]
        if tot:
            for v in "aiueo":
                share[v][f] = (c[v] + (c["N"] if v == "u" else 0)) / tot
    return gate, share, vis


def mora_track(moras, fps, open_):
    """(openness, {vowel: weight * openness}, visemes) per frame: the
    level-based openness says how far the mouth opens, the moras which
    shape it takes and when it must be shut (mora_weights)."""
    gate, share, vis = mora_weights(moras, fps, len(open_))
    op = np.asarray(open_, np.float64) * gate
    return op, {v: (op * w).astype(np.float32) for v, w in share.items()}, vis


def mouth_track(x, sr, fps, n_frames, mode="open", gain=1.0, cal=None,
                moras=None):
    """The mouth token's payload: {"fps", "open": [...]} and, mode
    "vowels", {"vowels": {"a": [...], ...}}, rounded to 4 decimals so the
    plan and a --tokens-in replay read the same numbers.

    Mode "auto" with a rig calibration (flow.mouth.calibrate) that found
    talking curves: the openness is shaped to the rig's authored speaking
    level (cal["gamma"] from flow.mouth.level_gamma), and the payload
    carries the calibration's "map" (parameter -> curve and its speaking
    range) for the runtime; vowel curves are added when the map drives two
    or more vowels. Without such a calibration "auto" is "open" exactly.
    With `moras` (VOICEVOX) the vowels come from the moras, else from the
    formant estimate."""
    op = np.clip(mouth_open(x, sr, fps, n_frames) * gain, 0, 1)
    tr = {"fps": float(fps)}
    share = None
    if moras is not None:
        gate, share, tr["visemes"] = mora_weights(moras, fps, n_frames)
        op = op * gate
    if mode == "auto" and cal and cal.get("map"):
        try:
            from .mouth import level_gamma
        except ImportError:
            from flow.mouth import level_gamma
        g = level_gamma(op, cal.get("level"))
        op = op ** g
        tr["gamma"] = round(float(g), 4)
        need = sum(m["curve"] in VOWEL_F for m in cal["map"].values()) >= 2
        mode = "vowels" if need else "open"
        tr["map"] = cal["map"]
    elif mode == "auto":
        mode = "open"
    vow = None
    if mode == "vowels":
        vow = ({v: op * w for v, w in share.items()} if share is not None
               else vowels(x, sr, fps, op))
    tr["open"] = [round(float(v), 4) for v in op]
    if vow is not None:
        tr["vowels"] = {v: [round(float(u), 4) for u in c]
                        for v, c in vow.items()}
    return tr
