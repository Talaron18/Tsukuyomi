#!/usr/bin/env python3
"""
Live director: talk to the character in a browser and watch it answer, the
idle running on between turns.

flow.director plans a turn offline (reply -> tags -> TTS -> runtime plan ->
render subprocess -> demo.mp4). This runs the same pieces continuously in
one process on the server:

  render     one Live2DBackend on the main thread (GL contexts are thread
             bound), fixed timestep, frame f on screen at f / fps of one
             server clock (time.monotonic from start). flow.stream gives
             each frame's parameters: the idle generated ahead in rolling
             windows, a reply's tokens merged into its future. When the CPU
             cannot keep up, the GL render of late frames is skipped (their
             parameter state is still stepped: switches, blinks and token
             commits are stateful) and the next render advances physics by
             the time that passed (backend.dt), so the motion never drifts
             behind the clock. Frames go to JPEG (Pillow) on an encoder
             thread. The achieved fps is logged every 10 s.
  turn       a line from the page goes to the language model, streamed;
             SentenceCutter cuts its text at sentence ends as it arrives.
             Each sentence, as soon as it is complete: tags parsed
             (director.TagParser), its mood read (speech.mood_of), spoken
             (speech.speak through the VoiceRouter), its tags timed by the
             words / moras, its mouth curve computed (speech.mouth_track),
             then everything put at one server time t: the audio starts at
             t, the mouth curve on frame t * fps, the tags realized
             (director.realize) at t + their offset and merged into the
             stream. t is the later of now + --lead and the end of the
             previous sentence's audio + --gap, snapped to a frame. The
             model streams on while a sentence is spoken; the TTS of the
             next sentence runs while this one plays. Latencies per stage
             are logged and sent to the page.
  viewer     a stdlib HTTP server (127.0.0.1 by default; open it through an
             SSH tunnel). The page reads frames from /video, a
             close-delimited stream of [magic, frame, server time, length,
             JPEG], events from /events (server-sent: audio, subtitles,
             log, reply), audio clips from /audio/<id>.wav, and posts
             lines to /say. /mjpeg is a plain multipart stream for any
             player, with no sync.
  sync       everything carries a server-clock time. The page estimates
             its offset to that clock from /clock pings (NTP style: the
             sample with the smallest round trip, offset = server time -
             midpoint of the ping), then plays both media at server time
             + a fixed playout delay D (?delay=, default 0.3 s): a frame is
             drawn when its time + D comes, an audio clip is started by
             WebAudio at its time + D mapped through getOutputTimestamp()
             (which also takes the output latency out). Audio is pushed
             --lead ahead of its time, so it is always there first; frames
             arrive after render + encode + network, which D must cover
             (the page shows how late frames arrive). Both media on one
             delay keep lip sync whatever the network delay, as long as it
             stays under D.
  record     --record DIR: video.mp4 (every frame on the clock, a skipped
             or dropped frame filled with the next), speech.wav (the clips
             mixed at their times), subs.srt, turns.jsonl (lines, replies,
             tags, latencies) and session.mp4 with all of them muxed,
             written at exit.

  body       --mode body (docs/embodiment.md E2, flow.pose): the model
             gets its measured body in the system prompt and, with each
             line, where its body is now (the last frame plus the poses
             still to come) and, with --body-image on a backend that takes
             images, the frame on screen. It writes [pose {...}] markers;
             each sentence's markers are solved with the prior toward that
             pose and merged into the stream as pose tokens beside the
             tags' tokens; the Core check of each solve runs after they are
             on the stream, and its per-target report goes to the log, the
             page and turns.jsonl.

Usage (see --help; the rig, model and TTS arguments are flow.director's):
    xvfb-run -a -s "-screen 0 1024x1024x24" python -m flow.live \\
        --model-json M --rig rigs/R.npz --clips clips2/R.npz --events ev.json \\
        --backend deepseek --tts auto --persona "..." --port 8765 \\
        [--record runs/live/R]
    then on your machine: ssh -L 8765:127.0.0.1:8765 server, and open
    http://127.0.0.1:8765/ (add --mode body [--body-image] for body
    targets instead of tags)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import wave
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

try:
    from . import director as dr
    from . import moves as mv
    from . import pose as pz
    from . import speech as sp
    from . import stream as fs
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import director as dr
    from flow import moves as mv
    from flow import pose as pz
    from flow import speech as sp
    from flow import stream as fs

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live.html")
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
VRM_PAGE = os.path.join(WEB, "vrm.html")
FRAME_MAGIC = b"TSKF"
# <magic 4s, frame u32, server time f64, JPEG length u32>, little endian.
FRAME_HEAD = struct.Struct("<4sIdI")


class Clock:
    """The one server clock: seconds since start, monotonic."""

    def __init__(self, t0=None):
        self.t0 = time.monotonic() if t0 is None else t0

    def now(self):
        return time.monotonic() - self.t0


# ------------------------------------------------------------ sentences

LATIN_END = set(".!?")
# Characters of spoken text before a sentence is cut at a clause mark
# anyway: the first sentence sooner, since the first audio waits on it. A
# Latin character is ~1/14 s spoken, a CJK one ~1/6 s (director's rates):
# 60 Latin ~ 4 s, 24 CJK ~ 4 s of speech.
FIRST_CAP = {"latin": 60, "cjk": 24}
CAP = {"latin": 160, "cjk": 64}


class SentenceCutter:
    """Cut a streamed tagged reply into sentences as its chunks arrive.

    feed(chunk) returns the sentences completed by it, close() the rest
    (raw text, tags and all; director.TagParser parses each). A sentence
    ends after a run of sentence ends (".", "...", "?!", "。", "…", a
    closing quote or bracket after them) once a character that is not part
    of the run arrives. A Latin . ! or ? ends one only before whitespace
    (or a tag), so "3.14", "e.g.x" and "Mr.Smith" do not; 。！？… end one
    whatever follows. Nothing inside an open [tag] is looked at, and a tag
    written after a full stop starts the next sentence, where it sits in
    the speech. A sentence running past FIRST_CAP / CAP spoken characters
    is cut at its next clause mark (, 、 ; ...), to bound the wait for the
    first audio.
    """

    def __init__(self, first_cap=None, cap=None, max_tag=None):
        self.first_cap = first_cap or FIRST_CAP
        self.cap = cap or CAP
        # Body mode's pose markers hold JSON (flow.pose.MAX_MARKER).
        self.max_tag = max_tag or dr.MAX_TAG
        self.buf = ""
        self.i = 0                  # next character of buf to look at
        self.count = 0              # spoken characters since the last cut
        self.cjk = 0
        self.n = 0                  # sentences cut so far

    def _limit(self):
        caps = self.first_cap if self.n == 0 else self.cap
        return caps["cjk" if self.cjk * 2 > self.count else "latin"]

    def _cut(self, j, out):
        s = self.buf[:j]
        self.buf, self.i = self.buf[j:], 0
        self.count = self.cjk = 0
        if s.strip():
            out.append(s)
            self.n += 1

    def feed(self, chunk):
        self.buf += chunk.translate(dr._FULLWIDTH)
        out = []
        while self.i < len(self.buf):
            b, i = self.buf, self.i
            c = b[i]
            if c == "[":
                k = b.find("]", i + 1, i + self.max_tag + 2)
                if k >= 0:
                    self.i = k + 1
                    continue
                if len(b) - i <= self.max_tag + 1:
                    break                      # the tag may still close
                # Too long to be a tag: text (the parser keeps it too).
            if c in sp.SENT_END:
                j = i + 1
                while j < len(b) and b[j] in sp.SENT_TAIL:
                    j += 1
                if j >= len(b):
                    self.i = i
                    break                      # the run may go on
                if (c in LATIN_END and all(x in LATIN_END | sp.SENT_TAIL
                                           for x in b[i:j])
                        and not (b[j].isspace() or b[j] == "[")):
                    self.i = j
                    continue
                self._cut(j, out)
                continue
            if not c.isspace():
                self.count += 1
                self.cjk += dr._is_cjk(c)
            if c in dr.CLAUSE and self.count >= self._limit():
                self._cut(i + 1, out)
                continue
            self.i = i + 1
        return out

    def close(self):
        out = []
        if self.buf.strip():
            out.append(self.buf)
            self.n += 1
        self.buf, self.i = "", 0
        return out


# ------------------------------------------------------------------ hub

def wav_bytes(x, sr):
    buf = io.BytesIO()
    pcm = (np.clip(np.asarray(x, np.float32), -1, 1) * 32767).round()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


class Hub:
    """What the server hands out: the latest frames to each video client
    (a short queue each; a slow client loses its oldest frames, never
    holds the render back), events to each event client, audio clips by
    id. New event clients get the recent events still to come (audio and
    subtitles not yet over) and the last log lines."""

    def __init__(self, clock, max_audio=64):
        self.clock = clock
        self.lock = threading.Lock()
        self.frame_subs, self.event_subs = set(), set()
        # VRM body states (flow.vrm), one per frame, for /states.
        self.state_subs = set()
        self.last_state = None
        self.audio = OrderedDict()
        self.max_audio = max_audio
        self.history = deque(maxlen=200)
        self.last_frame = None
        self.n_audio = 0
        self.stop = threading.Event()

    def frame(self, f, t, jpeg):
        item = (f, t, jpeg)
        with self.lock:
            self.last_frame = item
            subs = list(self.frame_subs)
        for q in subs:
            _put_latest(q, item)

    def state(self, f, t, st, avatar="vrm"):
        """A VRM body state for frame f at server time t."""
        item = {"f": f, "t": round(t, 4), "avatar": avatar, **st}
        with self.lock:
            self.last_state = item
            subs = list(self.state_subs)
        for q in subs:
            _put_latest(q, item)

    def event(self, kind, **data):
        msg = dict(data, type=kind)
        with self.lock:
            self.history.append(msg)
            subs = list(self.event_subs)
        for q in subs:
            _put_latest(q, msg)
        return msg

    def add_audio(self, x, sr, t, text=""):
        with self.lock:
            self.n_audio += 1
            aid = f"a{self.n_audio}"
            self.audio[aid] = wav_bytes(x, sr)
            while len(self.audio) > self.max_audio:
                self.audio.popitem(last=False)
        self.event("audio", id=aid, t=round(t, 4),
                   dur=round(len(x) / sr, 4), url=f"/audio/{aid}.wav",
                   text=text)
        return aid

    def subscribe(self, kind, maxsize=None):
        q = queue.Queue(maxsize=maxsize or (90 if kind in ("frames", "states")
                                            else 500))
        with self.lock:
            (self.frame_subs if kind == "frames" else self.state_subs
             if kind == "states" else self.event_subs).add(q)
            if kind == "events":
                now = self.clock.now()
                for m in self.history:
                    end = m.get("t", 0) + m.get("dur", 0) \
                        if m["type"] == "audio" else m.get("end")
                    if m["type"] in ("audio", "sub") and end is not None \
                            and end < now:
                        continue
                    _put_latest(q, m)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.frame_subs.discard(q)
            self.event_subs.discard(q)
            self.state_subs.discard(q)


def _put_latest(q, item):
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


# --------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0: a streamed body ends when the connection closes, which a
    # browser's fetch() reads as a stream.
    protocol_version = "HTTP/1.0"
    server_version = "TsukuyomiLive/1"

    def log_message(self, fmt, *args):      # quiet: the app logs itself
        pass

    @property
    def app(self):
        return self.server.app

    def _send(self, code, body, ctype="application/json", extra=()):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _stream_head(self, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        hub = self.app.hub
        try:
            if path in ("/", "/index.html"):
                # A VRM avatar (alone or beside the Live2D rig) is drawn by
                # the browser: flow/web/vrm.html.
                vrm = getattr(self.app, "vrm", None)
                with open(PAGE if vrm is None else VRM_PAGE,
                          encoding="utf-8") as fh:
                    page = fh.read()
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/favicon.ico":
                # The browser asks on its own; a 404 here was the page's
                # one console error.
                self.send_response(204)
                self.end_headers()
                return None
            if path.startswith("/web/"):
                return self._static(path[5:])
            if path == "/avatar.vrm" and getattr(self.app, "vrm", None):
                with open(self.app.vrm["path"], "rb") as fh:
                    return self._send(200, fh.read(),
                                      "application/octet-stream")
            if path == "/avatar.json" and getattr(self.app, "vrm", None):
                return self._send(200, self.app.vrm["config"])
            if path == "/states":
                return self._states()
            if path == "/clock":
                return self._send(200, {"t": self.app.clock.now()})
            if path == "/state":
                return self._send(200, self.app.state())
            if path == "/frame.jpg":
                lf = hub.last_frame
                if lf is None:
                    return self._send(404, {"error": "no frame yet"})
                return self._send(200, lf[2], "image/jpeg")
            if path.startswith("/audio/") and path.endswith(".wav"):
                data = hub.audio.get(path[7:-4])
                if data is None:
                    return self._send(404, {"error": "no such clip"})
                return self._send(200, data, "audio/wav")
            if path == "/events":
                return self._events()
            if path == "/video":
                return self._frames(raw=True)
            if path == "/mjpeg":
                return self._frames(raw=False)
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError):
            return None

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if path != "/say":
            return self._send(404, {"error": "not found"})
        try:
            text = json.loads(raw.decode("utf-8") or "{}").get("text", "")
        except ValueError:
            text = raw.decode("utf-8", "replace")
        text = str(text).strip()
        if not text:
            return self._send(400, {"error": "empty line"})
        waiting = self.app.say(text)
        return self._send(202, {"ok": True, "queued": waiting})

    def _events(self):
        hub = self.app.hub
        q = hub.subscribe("events")
        self._stream_head("text/event-stream; charset=utf-8")
        try:
            hello = {"type": "hello", "t": self.app.clock.now(),
                     "fps": self.app.fps}
            self.wfile.write(f"data: {json.dumps(hello)}\n\n".encode())
            self.wfile.flush()
            while not hub.stop.is_set():
                try:
                    m = q.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")     # finds a gone client
                    self.wfile.flush()
                    continue
                line = json.dumps(m, ensure_ascii=False)
                self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unsubscribe(q)

    def _static(self, rel):
        """flow/web files (the VRM page's modules and vendored libraries);
        nothing outside that directory."""
        full = os.path.realpath(os.path.join(WEB, rel))
        if not full.startswith(os.path.realpath(WEB) + os.sep) or \
                not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ctype = {".js": "text/javascript", ".html": "text/html; "
                 "charset=utf-8", ".json": "application/json"}.get(
                     os.path.splitext(full)[1], "application/octet-stream")
        with open(full, "rb") as fh:
            return self._send(200, fh.read(), ctype)

    def _states(self):
        """Server-sent VRM body states, one per frame (Hub.state)."""
        hub = self.app.hub
        q = hub.subscribe("states")
        self._stream_head("text/event-stream; charset=utf-8")
        try:
            while not hub.stop.is_set():
                try:
                    m = q.get(timeout=1.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                line = json.dumps(m, separators=(",", ":"))
                self.wfile.write(f"data: {line}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unsubscribe(q)

    def _frames(self, raw):
        hub = self.app.hub
        q = hub.subscribe("frames")
        self._stream_head("application/octet-stream" if raw else
                          "multipart/x-mixed-replace; boundary=frame")
        try:
            while not hub.stop.is_set():
                try:
                    f, t, jpeg = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if raw:
                    self.wfile.write(FRAME_HEAD.pack(FRAME_MAGIC, f, t,
                                                     len(jpeg)) + jpeg)
                else:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unsubscribe(q)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, app, host="127.0.0.1", port=8765):
        self.app = app
        super().__init__((host, port), Handler)

    def start(self):
        th = threading.Thread(target=self.serve_forever, name="http",
                              daemon=True)
        th.start()
        return th

    def stop(self):
        self.app.hub.stop.set()
        self.shutdown()
        self.server_close()


# ---------------------------------------------------------------- turns

class Performer:
    """Turns, one at a time, on a worker thread: the language model's
    reply streamed and cut into sentences, each spoken and put on the
    stream as soon as it is complete (module docstring, "turn")."""

    def __init__(self, a, run, rig, stream, clock, hub, backend, tts,
                 log=print, recorder=None):
        self.a, self.run, self.rig, self.stream = a, run, rig, stream
        self.clock, self.hub, self.backend, self.tts = clock, hub, backend, tts
        self.log, self.recorder = log, recorder
        self.fps = rig.fps
        self.messages = []
        self.inbox = queue.Queue()
        self.rng = np.random.default_rng([int(a.seed), 1])
        self.next_frame = 0               # the first frame after the speech
        self.turns = 0
        self.busy = False
        self.thread = None
        self.lang, self.lang_set = "en", False
        # Body mode (--mode body): the run from director.prepare_body, and
        # a callable giving the latest rendered frame (RGBA array) for the
        # picture a vision backend gets.
        self.body = run if run.get("mode") == "body" else None
        self.snap = None
        # --avatar both: the VRM follows every sentence (VRMFollower).
        self.followers = []

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="turns",
                                       daemon=True)
        self.thread.start()

    def say(self, text):
        self.inbox.put(text)
        return self.inbox.qsize() + (1 if self.busy else 0)

    def _loop(self):
        while True:
            text = self.inbox.get()
            if text is None:
                return
            self.busy = True
            try:
                self.turn(text)
            except Exception as exc:  # noqa: BLE001 - one bad turn, not the show
                self.log(f"live: turn failed: {exc!r}")
                self.hub.event("log", text=f"turn failed: {exc}")
            finally:
                self.busy = False

    def emit(self, text):
        self.log(text)
        self.hub.event("log", text=text)

    # One turn ---------------------------------------------------------

    def turn(self, text):
        clock = self.clock
        self.turns += 1
        n = self.turns
        t_in = clock.now()
        self.hub.event("user", text=text, t=t_in)
        # The reply's language, for the voice: the prompt asks for the
        # user's language; the first sentence whose script settles it wins
        # (lang_of), and the rest of the turn keeps it.
        self.lang, self.lang_set = sp.language(text), False
        if self.body is None:
            self.messages.append({"role": "user", "content": text})
        else:
            self.messages.append(self.body_message(text))
        q = queue.Queue()
        raw, info = [], {"first": None, "stop": None, "error": None}

        def produce():
            cut = SentenceCutter(max_tag=self.run.get("max_tag"))
            try:
                for chunk in self.backend.stream(self.run["system"],
                                                 self.messages):
                    if info["first"] is None:
                        info["first"] = clock.now() - t_in
                    raw.append(chunk)
                    for s in cut.feed(chunk):
                        q.put((s, clock.now() - t_in))
                for s in cut.close():
                    q.put((s, clock.now() - t_in))
            except Exception as exc:  # noqa: BLE001
                info["error"] = exc
            finally:
                info["stop"] = getattr(self.backend, "stop_reason", None)
                q.put(None)

        th = threading.Thread(target=produce, name="llm", daemon=True)
        th.start()
        lat, k = [], 0
        while True:
            item = q.get()
            if item is None:
                break
            s, t_cut = item
            k += 1
            res = self.sentence(s, n, k, t_in, t_cut)
            if res:
                lat.append(res)
        th.join()
        reply = "".join(raw)
        if self.body is not None:
            self.messages[-1] = pz.text_only(self.messages[-1])
        if info["error"] is not None:
            self.emit(f"live: turn {n}: model failed: {info['error']!r}")
        refused = info["stop"] == "refusal"
        if refused or not reply.strip():
            self.messages.pop()                 # not history, as director
            self.emit(f"live: turn {n}: "
                      + ("refused" if refused else "no reply"))
        else:
            self.messages.append({"role": "assistant", "content": reply})
        first = info["first"]
        summary = {"turn": n, "user": text, "reply": reply,
                   "model_first_text": None if first is None
                   else round(first, 3), "sentences": lat,
                   "stop_reason": info["stop"]}
        if lat:
            summary["first_audio_after"] = lat[0]["audio_at"]
        self.hub.event("reply", **summary)
        if self.recorder is not None:
            self.recorder.turn(summary)
        self.emit(f"live: turn {n}: model first text "
                  + (f"{first:.2f}s" if first is not None else "never")
                  + (f", first audio {lat[0]['audio_at']:.2f}s after enter"
                     if lat else "") + f"; reply: {reply.strip()}")

    def sentence(self, raw, n, k, t_in, t_cut):
        """Plan one sentence and put it on the stream. Returns its
        latencies (seconds after the line was entered)."""
        a, fps = self.a, self.fps
        p = (dr.TagParser(self.run["labels"], self.run["offered"])
             if self.body is None else self.body["parser"]())
        p.feed(raw)
        p.close()
        for w in p.warnings:
            self.emit(f"director: {w}")
        spoken, tags = p.spoken, p.tags
        t0 = self.clock.now()
        mood = "neutral"
        audio = None
        if spoken and self.tts is not None:
            if getattr(a, "mood", "auto") != "off":
                mood = sp.mood_of(spoken, tags)[0]
            said = sp.speak(spoken, tags, self.tts, self._weight,
                            log=self.log, moods=[(0, mood)],
                            lang=self.lang_of(spoken))
            acts, spans, end = said["acts"], said["spans"], said["end"]
            audio, sr, moras = said["audio"], said["sr"], said.get("moras")
        elif spoken:
            acts, spans, end = dr.timeline(spoken, tags, a.cjk_rate,
                                           a.latin_rate)
        else:
            acts = [{"t": 0.0, "tag": t["tag"], "arg": t["arg"], "at": 0}
                    for t in tags if t["tag"] != "pause"]
            spans, end = [], 0.0
        t_tts = self.clock.now() - t0
        mouth = None
        if audio is not None and (a.lipsync or "auto") != "off":
            nf = int(math.ceil(end * fps)) + 2
            lip = a.lipsync or "auto"
            cal = getattr(a, "mouth_cal", None) if lip == "auto" else None
            mouth = sp.mouth_track(audio, sr, fps, nf, lip, a.mouth_gain,
                                   cal=cal, moras=moras)
        mvs = []
        if self.body is not None and any(x["tag"] == "move" for x in acts):
            # Motion primitives (flow.moves), built from where the body is
            # heading, before the sentence's time is taken (a 16-beat dance
            # is 0.6-1 s of solves; after it, the audio would start late).
            # Each marker's variation is seeded by the run's seed, the turn
            # and the sentence, written on the marker: the --avatar both
            # follower draws the same, and the realize generator is not
            # touched.
            t_m = self.clock.now()
            base = mv.move_seed(a, 1000 * n + k)
            for i, x in enumerate(acts):
                if x["tag"] == "move":
                    x["arg"] = dict(x["arg"], seed=mv.hash_seed(
                        base, x.get("at"), i))
            cur, cur_parts = self.stream.body_now()
            mvs = mv.solve_moves(self.body["body"], acts, cur, cur_parts,
                                 log=self.log, seed=base)
            mv.drop_clashing_gestures(acts, mvs, self.rig.motion_seconds,
                                      self.log)
            lat_move = self.clock.now() - t_m
        # When: as soon as possible, never over the previous sentence.
        now = self.clock.now()
        lead_f = int(math.ceil((now + a.lead) * fps))
        if spoken:
            f0 = max(lead_f, self.next_frame)
        else:
            # Tags after the last words: at the end of that speech.
            f0 = max(lead_f, self.next_frame - int(round(a.gap * fps)))
        t_start = f0 / fps
        late = (f0 - self.next_frame) / fps if spoken and k > 1 else 0.0
        if spoken:
            self.next_frame = f0 + int(math.ceil((end + a.gap) * fps))
        poses = []
        if self.body is not None and any(x["tag"] == pz.POSE for x in acts):
            bm = self.body["body"]
            t_s = self.clock.now()
            cur, cur_parts = self.stream.body_now()
            poses = pz.solve_poses(bm, acts, cur, cur_parts, log=self.log,
                                   keep_shown=True)
            arms = self.body.setdefault("arm_params", pz.arm_params(bm))
            for rec in poses:
                rec["arm_params"] = sorted(set(rec["params"]) & arms)
            # Solved first: a gesture under an arm pose gives way to it
            # before anything is realized (--body-gestures).
            pz.drop_clashing_gestures(acts, poses, self.rig.motion_seconds,
                                      self.log)
        shifted = [dict(x, t=round(float(x["t"]) + t_start, 4)) for x in acts]
        directed = dr.realize(shifted, self.rig.stats, list(self.rig.roles),
                              self.rig.motion_seconds, self.rng,
                              self.stream.pose(), settle=self.rig.settle)
        if poses:
            for k_ in pz.tokens(poses, self.rig.roles, arms):
                k_["t"] = round(float(k_["t"]) + t_start, 4)
                directed.append(k_)
            for rec in poses:                  # stream time, for the log
                rec["start"] = round(rec["start"] + t_start, 4)
            # A gesture an earlier sentence directed gives way too.
            arm_w = [(p_["start"], p_["start"] + p_["dur"] + p_["hold"]
                      + p_["release"]) for p_ in poses
                     if p_.get("arm_params") or p_.get("parts")]
            if arm_w and self.stream.drop_gestures(arm_w):
                self.emit(f"live: turn {n}.{k} an earlier gesture gives way "
                          "to an arm pose")
            lat_solve = self.clock.now() - t_s
        if mvs:
            for k_ in mv.tokens(mvs):
                k_["t"] = round(float(k_["t"]) + t_start, 4)
                directed.append(k_)
            for rec in mvs:
                rec["start"] = round(rec["start"] + t_start, 4)
            if self.stream.drop_gestures(mv.windows(mvs)):
                self.emit(f"live: turn {n}.{k} an earlier gesture gives way "
                          "to a motion")
            for line in mv.report_lines(mvs, spoken):
                self.emit(f"live: turn {n}.{k} {line}")
        gone = self.stream.direct(directed) if directed else 0
        if mouth is not None:
            self.stream.add_mouth(f0, mouth, len(mouth["open"]))
        for fol in self.followers:
            fol.sentence(acts, t_start, f0, spoken,
                         None if audio is None else (audio, sr, moras),
                         end)
        text = spoken
        if audio is not None:
            self.hub.add_audio(audio, sr, t_start, text)
            if self.recorder is not None:
                self.recorder.audio(t_start, audio, sr)
        for s in dr.subtitles(spoken, spans) if spoken else ():
            sub = {"start": round(s["start"] + t_start, 3),
                   "end": round(s["end"] + t_start, 3), "text": s["text"]}
            self.hub.event("sub", t=sub["start"], end=sub["end"],
                           text=s["text"])
            if self.recorder is not None:
                self.recorder.sub(sub)
        voice = (self.tts.mood_voice(mood)
                 if self.tts is not None and mood != "neutral"
                 and hasattr(self.tts, "mood_voice") else None)
        tags_s = " ".join(pz.tag_text(x) for x in acts)
        lat = {"k": k, "cut_at": round(t_cut, 3),
               "tts": round(t_tts, 3),
               "audio_at": round(t_start - t_in, 3),
               "speech": round(end, 3), "mood": mood,
               "gap_late": round(max(late, 0.0), 3)}
        self.emit(f"live: turn {n}.{k} cut {t_cut:.2f}s, "
                  f"{'tts' if audio is not None else 'plan'} {t_tts:.2f}s, "
                  f"audio at {t_start - t_in:.2f}s ({end:.2f}s), mood {mood}"
                  + (f" ({voice})" if voice else "")
                  + (f", {late:.2f}s after the previous" if late > 0.05
                     else "")
                  + (f"; {tags_s}" if tags_s else "")
                  + (f"; {gone} idle tokens displaced" if gone else "")
                  + f": {spoken}")
        if mvs:
            lat["moves"] = round(lat_move, 3)
        if poses:
            # The exact check after the poses are on the stream: the Core's
            # geometry is a report, not something the motion waits for.
            pz.add_exact(self.body["body"], self.body.get("core"), poses,
                         self.log)
            for line in pz.report_lines(poses, spoken):
                self.emit(f"live: turn {n}.{k} {line}")
            lat["solve"] = round(lat_solve, 3)
            lat["reach"] = pz.reach_summary(poses)
        return lat

    def body_message(self, text):
        """The user line with where the body is now and, for a backend that
        takes images (--body-image), the frame on screen."""
        bm = self.body["body"]
        vals, parts = self.stream.body_now()
        img = None
        if self.body.get("images") and self.snap is not None:
            frame = self.snap()
            if frame is not None:
                from PIL import Image
                buf = io.BytesIO()
                Image.fromarray(np.asarray(frame)).convert("RGB").save(
                    buf, "PNG")
                img = buf.getvalue()
        return pz.user_message(text + "\n" + pz.pose_now(bm, vals, parts),
                               self.a.backend, img)

    def lang_of(self, spoken):
        """The language to speak this sentence in. Offline, the voice is
        picked from the whole reply; here a sentence comes alone, and
        "本当？" (no kana) reads as Chinese, "OK！" as English. Kana or
        Hangul settle it whenever they appear; enough Latin letters, or
        Han alone (Chinese unless the turn is Japanese), settle it only
        while nothing has; anything else keeps the turn's language."""
        s = sp.script_language(spoken)
        if s in ("ja", "ko"):
            self.lang, self.lang_set = s, True
        elif not self.lang_set and s == "en":
            self.lang, self.lang_set = "en", True
        elif not self.lang_set and s == "han":
            self.lang = "ja" if self.lang == "ja" else "zh"
            self.lang_set = True
        return self.lang

    def _weight(self, ch):
        return 1.0 / (self.a.cjk_rate if dr._is_cjk(ch) else self.a.latin_rate)


# ------------------------------------------------------- render, encode

class Encoder:
    """JPEG-encodes rendered frames on its own thread (Pillow releases the
    GIL while it encodes) and hands them to the hub and the recorder. A
    full queue drops the frame rather than hold the render loop."""

    def __init__(self, hub, quality=80, recorder=None, log=print):
        self.hub, self.quality, self.recorder, self.log = hub, quality, \
            recorder, log
        self.q = queue.Queue(maxsize=4)
        self.dropped = 0
        self.errors = 0
        self.encode_s = deque(maxlen=300)
        self.thread = threading.Thread(target=self._loop, name="jpeg",
                                       daemon=True)
        self.thread.start()

    def submit(self, f, t, img):
        try:
            self.q.put_nowait((f, t, img))
        except queue.Full:
            self.dropped += 1

    def close(self):
        self.q.put(None)
        self.thread.join(timeout=10)

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            f, t, img = item
            try:
                t0 = time.perf_counter()
                jpeg = encode_jpeg(img, self.quality)
                self.encode_s.append(time.perf_counter() - t0)
                self.hub.frame(f, t, jpeg)
                if self.recorder is not None:
                    self.recorder.frame(f, jpeg)
            except Exception as exc:  # noqa: BLE001 - say so, keep going
                self.errors += 1
                if self.errors <= 3:
                    self.log(f"live: frame {f} not encoded: {exc!r}")


def encode_jpeg(img, quality=80):
    from PIL import Image
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(a), "RGB").save(
        buf, "JPEG", quality=int(quality))
    return buf.getvalue()


class FakeBackend:
    """A renderer with no GL: a flat image whose colour follows the head
    and mouth, so the pipeline (and the page) can run anywhere. Used by the
    tests and --renderer none."""

    def __init__(self, size=128, cost=0.0):
        self.size, self.cost = size, cost
        self.dt = None
        self.n = 0

    def render(self, overrides=None, parts=None):
        if self.cost:
            time.sleep(self.cost)
        self.n += 1
        o = overrides or {}
        img = np.zeros((self.size, self.size, 4), np.uint8)
        img[..., 3] = 255
        img[..., 0] = int(127 + 4 * float(o.get("ParamAngleX", 0.0))) % 256
        img[..., 1] = int(255 * float(o.get("ParamMouthOpenY", 0.0))) % 256
        img[..., 2] = (self.n * 3) % 256
        return img


class RenderLoop:
    """Frame f at f / fps on the clock: compute its parameters, render,
    hand to the encoder. Late frames are stepped but not rendered; the next
    render's dt covers them."""

    def __init__(self, stream, backend, clock, encoder, fps, log=print,
                 absent=(), report=10.0):
        self.stream, self.backend, self.clock = stream, backend, clock
        self.encoder, self.fps, self.log = encoder, float(fps), log
        self.absent = set(absent)
        self.report = report
        self.rendered = self.skipped = 0
        self.last_img = None                   # body mode's self-image
        self.f = 0
        self.cost = deque(maxlen=300)          # seconds per rendered frame

    def run(self, stop, max_frames=None):
        dt0 = 1.0 / self.fps
        t_rep, n_rep, s_rep = self.clock.now(), 0, 0
        while not stop.is_set():
            if max_frames is not None and self.f >= max_frames:
                return
            now = self.clock.now()
            due = int(now * self.fps)
            if self.f > due:
                time.sleep(min(self.f / self.fps - now, 0.05))
                continue
            skip = due - self.f
            if max_frames is not None:
                skip = min(skip, max_frames - 1 - self.f)
            for _ in range(skip):
                self.stream.frame()             # stateful: never skipped
            self.f += skip
            self.skipped += skip
            t0 = time.perf_counter()
            over, parts = self.stream.frame()
            if self.absent:
                over = {p: v for p, v in over.items() if p not in self.absent}
            self.backend.dt = dt0 * (skip + 1)
            img = self.backend.render(over, parts=parts)
            self.last_img = img
            self.backend.dt = dt0
            self.cost.append(time.perf_counter() - t0)
            self.encoder.submit(self.f, self.f / self.fps, img)
            self.rendered += 1
            self.f += 1
            now = self.clock.now()
            if self.report and now - t_rep >= self.report:
                fps = (self.rendered - n_rep) / (now - t_rep)
                self.log(f"live: {fps:.1f} fps rendered of {self.fps:g} "
                         f"({self.skipped - s_rep} frames skipped, "
                         f"{1000 * np.median(self.cost):.1f} ms per frame "
                         f"to render, {self.encoder.dropped} encoder "
                         "drops)")
                t_rep, n_rep, s_rep = now, self.rendered, self.skipped


# ------------------------------------------------------------- recording

class Recorder:
    """--record DIR (module docstring). Frames are the encoder's JPEGs piped
    to ffmpeg (image2pipe -> H.264). Keyed on frame numbers: a frame the
    render skipped or the encoder dropped is filled with the next frame
    that arrives, so the video keeps the clock the speech is mixed on."""

    def __init__(self, out_dir, fps, log=print, video=True):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.fps, self.log = out_dir, float(fps), log
        self.lock = threading.Lock()
        self.clips, self.subs = [], []
        self.frames = 0
        self.proc = None
        self.turns = open(os.path.join(out_dir, "turns.jsonl"), "a",
                          encoding="utf-8")
        if not video:
            pass            # a VRM session records its states (flow.vrm)
        elif shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe",
                 "-c:v", "mjpeg", "-framerate", f"{self.fps:g}", "-i", "-",
                 "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt",
                 "yuv420p", os.path.join(out_dir, "video.mp4")],
                stdin=subprocess.PIPE)
        else:
            log("live: no ffmpeg; --record keeps audio, subtitles and turns "
                "only")

    def frame(self, f, jpeg):
        if self.proc is None:
            return
        try:
            # Any frames missing before this one (skipped, dropped) are
            # this one again.
            for _ in range(max(f + 1 - self.frames, 0)):
                self.proc.stdin.write(jpeg)
            self.frames = max(self.frames, f + 1)
        except (BrokenPipeError, OSError):
            self.proc = None

    def audio(self, t, x, sr):
        with self.lock:
            self.clips.append((t, np.asarray(x, np.float32), sr))

    def sub(self, s):
        with self.lock:
            self.subs.append(s)

    def turn(self, summary):
        self.turns.write(json.dumps(summary, ensure_ascii=False) + "\n")
        self.turns.flush()

    def close(self):
        self.turns.close()
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        with self.lock:
            clips, subs = list(self.clips), list(self.subs)
        wav = os.path.join(self.dir, "speech.wav")
        if clips:
            sr = clips[0][2]
            end = max(t + len(x) / s for t, x, s in clips)
            n = int(math.ceil(max(end, self.frames / self.fps) * sr))
            mix = np.zeros(n, np.float32)
            for t, x, s in clips:
                y = sp._resample(x, s, sr)
                i = int(round(t * sr))
                mix[i:i + len(y)] += y[:max(n - i, 0)]
            sp.write_wav(wav, np.clip(mix, -1, 1), sr)
        srt = os.path.join(self.dir, "subs.srt")
        with open(srt, "w", encoding="utf-8") as fh:
            fh.write(dr.srt(sorted(subs, key=lambda s: s["start"])))
        vid = os.path.join(self.dir, "video.mp4")
        cmd = dr.mux_cmd(vid, wav, srt, os.path.join(self.dir,
                                                     "session.mp4"))
        if cmd and os.path.exists(vid) and shutil.which("ffmpeg"):
            r = subprocess.run(cmd, capture_output=True)
            self.log("live: session -> " + cmd[-1] if r.returncode == 0 else
                     "live: ffmpeg could not mux the session: "
                     + r.stderr.decode("utf-8", "replace")[-300:])


# ------------------------------------------------------------------ app

class App:
    """The pieces the server talks to."""

    def __init__(self, clock, hub, fps, performer=None, loop=None):
        self.clock, self.hub, self.fps = clock, hub, fps
        self.performer, self.loop = performer, loop
        # --avatar vrm / both: {"path": the .vrm, "config": the page's
        # avatar.json}; None serves the Live2D page.
        self.vrm = None

    def say(self, text):
        return self.performer.say(text)

    def state(self):
        out = {"t": self.clock.now(), "fps": self.fps}
        if self.loop is not None:
            out.update(frame=self.loop.f,
                       rendered=getattr(self.loop, "rendered", self.loop.f),
                       skipped=getattr(self.loop, "skipped", 0))
        if self.performer is not None:
            out.update(turns=self.performer.turns,
                       busy=self.performer.busy)
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    dr.add_args(ap)
    ap.set_defaults(backend="deepseek")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to serve on (127.0.0.1: reach it through "
                         "ssh -L 8765:127.0.0.1:8765 server)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="record the session: video.mp4, speech.wav, "
                         "subs.srt, turns.jsonl, session.mp4")
    ap.add_argument("--say", default=None, help="a first line, said at start")
    ap.add_argument("--lead", type=float, default=0.25,
                    help="seconds between a sentence being ready and its "
                         "audio starting: time to merge its tokens before "
                         "the render reaches them and to get the audio to "
                         "the page")
    ap.add_argument("--gap", type=float, default=0.12,
                    help="silence between sentences (each TTS clip already "
                         "has its own lead-in and tail)")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality")
    ap.add_argument("--renderer", default="gl", choices=("gl", "none"),
                    help="none: no GL, a flat test image (the pipeline, page "
                         "and audio still run)")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="seconds of physics on the first pose before the "
                         "clock starts")
    ap.add_argument("--bg", default="0.12,0.12,0.14",
                    help="background colour, r,g,b in 0..1")
    a = ap.parse_args(argv)
    if a.avatar == "vrm":
        return vrm_live(a, ap)
    dr.check_live2d_args(a, ap)
    import importlib.util
    if importlib.util.find_spec("PIL") is None:     # frames are JPEG via PIL
        ap.error("flow.live needs Pillow for the frames: pip install Pillow")
    if a.renderer == "gl":
        _, ensure_display = fs.rt._backend()
        ensure_display(["-m", f"{__package__ or 'flow'}.live"]
                       + (argv if argv is not None else sys.argv[1:]))
    run = dr.prepare(a)
    rig = fs.EventRig(a.model_json, a.rig, a.events, a.clips, a.binding,
                      a.fps, a.probe_dir, a.idle_profile, a.no_snap,
                      a.events_pool_only, breath_gain=a.breath_gain,
                      voice_config=dr.voice_config_path(a))
    backend = dr.make_backend(a)
    if a.tts is None:
        a.tts = "auto"           # live is for talking: speak by default
    tts = dr.make_voice(a)
    cal = getattr(a, "mouth_cal", None) or {}
    clock = Clock()
    hub = Hub(clock)
    stream = fs.MotionStream(rig, seed=a.seed, idle=not a.no_idle,
                             mouth_map=cal.get("map")
                             if (a.lipsync or "auto") == "auto" else None,
                             log=print)
    recorder = Recorder(a.record, a.fps) if a.record else None
    perf = Performer(a, run, rig, stream, clock, hub, backend, tts,
                     recorder=recorder)
    vrm_side = None
    if a.avatar == "both":
        vrm_side = both_setup(a, perf, clock, hub)
    encoder = Encoder(hub, a.quality, recorder)
    stop = threading.Event()
    # SIGTERM (timeout, systemd, a closed ssh session's kill) stops like
    # Ctrl-C, so --record still writes its files.
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    bg = tuple(float(x) for x in a.bg.split(",")) + (1.0,)

    def serve(be, absent=()):
        loop = RenderLoop(stream, be, clock, encoder, a.fps, absent=absent)
        perf.snap = lambda: loop.last_img
        app = App(clock, hub, a.fps, perf, loop)
        if vrm_side is not None:
            # The VRM's states on their own thread, on the same clock.
            app.vrm = vrm_side["page"]
            threading.Thread(target=vrm_side["loop"].run, args=(stop,),
                             name="vrm", daemon=True).start()
        srv = Server(app, a.host, a.port)
        srv.start()
        print(f"live: serving http://{a.host}:{srv.server_address[1]}/ "
              f"(ssh -L {srv.server_address[1]}:127.0.0.1:"
              f"{srv.server_address[1]} <server>, then open that address)")
        perf.start()
        if a.say:
            perf.say(a.say)
        try:
            loop.run(stop)
        except KeyboardInterrupt:
            print("live: stopping")
        finally:
            stop.set()
            srv.stop()
            encoder.close()
            if recorder is not None:
                recorder.close()
            if vrm_side is not None and vrm_side["log"] is not None:
                print("live: vrm states -> " + vrm_side["log"].close())

    if a.renderer == "none":
        serve(FakeBackend(a.size))
        return 0
    Live2DBackend, _ = fs.rt._backend()
    with Live2DBackend(a.model_json, size=a.size, bg=bg,
                       strip_dynamics=False, fps=a.fps) as be:
        be.calibrate()
        print("backend: " + ", ".join(
            f"{k}={be._api_report[k]}" for k in ("order", "timestep", "parts")
            if k in be._api_report))
        by_id = {s.pid for s in be.params}
        absent = rig.expr_ids - by_id
        # Physics settles on the first pose before the clock starts; the
        # stream's own first frame is computed again from frame 0.
        warm = fs.MotionStream(rig, seed=a.seed, idle=False, autonomic=False,
                               log=lambda s: None)
        over, parts = warm.frame()
        for _ in range(int(a.warmup * a.fps)):
            be.render({p: v for p, v in over.items() if p not in absent},
                      parts=parts)
        clock.t0 = time.monotonic()
        serve(be, absent)
    return 0


def both_setup(a, perf, clock, hub):
    """--avatar both: a VRM beside the Live2D rig. Its own stream (idle,
    blinks, breath) follows every sentence the Live2D rig plays (flow.vrm.
    Follower); the prompt stays the Live2D body's (the more constrained
    one: a target it reaches the VRM reaches too)."""
    try:
        from . import vrm as fv
    except ImportError:
        from flow import vrm as fv
    bm, bpath, rig, page = fv.live_setup(a)
    stream = fs.MotionStream(rig, seed=a.seed, idle=not a.no_idle, log=print)
    perf.followers.append(fv.Follower(a, bm, rig, stream, log=perf.emit))
    slog = (fv.StateLog(a.record, a.fps, bpath, page["path"]) if a.record
            else None)
    loop = fv.StateLoop(stream, bm, clock, hub, a.fps, record=slog)
    return {"page": page, "loop": loop, "log": slog}


def vrm_live(a, ap) -> int:
    """main() for --avatar vrm: the same turns (flow.live's Performer) on a
    VRM body (flow.vrm); the page draws the states the server streams on
    /states. No GL, no Pillow."""
    try:
        from . import vrm as fv
    except ImportError:
        from flow import vrm as fv
    if a.vrm is None and a.vrm_body is None:
        ap.error("--avatar vrm needs --vrm PATH (or --vrm-body FILE)")
    if a.body_image:
        print("live: --body-image: the server cannot draw a VRM (the browser "
              "does); text only")
        a.body_image = False
    if (a.lipsync or "auto") == "auto":
        a.lipsync = "vowels"
    bm, bpath, rig, page = fv.live_setup(a)
    run = fv.prepare(a, bm, bpath)
    backend = dr.make_backend(a)
    if a.tts is None:
        a.tts = "auto"
    tts = dr.make_voice(a)
    clock = Clock()
    hub = Hub(clock)
    stream = fs.MotionStream(rig, seed=a.seed, idle=not a.no_idle, log=print)
    recorder = (Recorder(a.record, a.fps, video=False) if a.record
                else None)
    slog = (fv.StateLog(a.record, a.fps, bpath, page["path"]) if a.record
            else None)
    perf = Performer(a, run, rig, stream, clock, hub, backend, tts,
                     recorder=recorder)
    loop = fv.StateLoop(stream, bm, clock, hub, a.fps, record=slog)
    app = App(clock, hub, a.fps, perf, loop)
    app.vrm = page
    stop = threading.Event()
    import signal
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    srv = Server(app, a.host, a.port)
    srv.start()
    print(f"live: serving http://{a.host}:{srv.server_address[1]}/ "
          f"(VRM {os.path.basename(page['path'])}; ssh -L "
          f"{srv.server_address[1]}:127.0.0.1:{srv.server_address[1]} "
          "<server>, then open that address)")
    perf.start()
    if a.say:
        perf.say(a.say)
    try:
        loop.run(stop)
    except KeyboardInterrupt:
        print("live: stopping")
    finally:
        stop.set()
        srv.stop()
        if recorder is not None:
            recorder.close()
        if slog is not None:
            print("live: vrm states -> " + slog.close() + " (render: python "
                  f"-m flow.vrmrender --run {a.record})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
