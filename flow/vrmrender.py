#!/usr/bin/env python3
"""
Offline renderer for VRM runs: a recorded state stream -> frames -> mp4.

The server has no GPU and no 3D renderer of its own; the browser draws the
avatar with three.js + three-vrm (flow/web/vrm.js). For review and A/B
videos this drives the same page in headless Chromium through Playwright:
it serves flow/web/ and the avatar on a 127.0.0.1 port, loads render.html,
and for every frame of the state stream calls TSK.frame(state, 1 / fps),
which applies the state, steps three-vrm (spring bones, look-at,
expressions, node constraints) by exactly one frame and returns the canvas
as JPEG. Frames are piped to ffmpeg; the speech is muxed after, so the
video is frame-exact against the audio the director planned on (no clock).

WebGL in headless Chromium runs on SwiftShader (CPU); 512 px frames of the
sample avatar take ~0.1-0.3 s each here. Before the first frame the rest
state is stepped WARMUP seconds so the hair has settled (Live2D's live
loop warms its physics the same way).

Needs: pip install playwright && playwright install chromium (this
container has the browsers under PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers),
and ffmpeg.

Usage:
    python -m flow.vrmrender --run runs/e3/seed/0     # states.json -> render/
    python -m flow.vrmrender --states S.json --body rigs/seed-san.body.json \\
        --out render/preview.mp4 [--audio speech.wav] [--size 512]
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import functools
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARMUP = 1.0
DEFAULT_BROWSERS = "/opt/pw-browsers"
# SwiftShader WebGL: the only GL a headless CPU box has.
CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader",
                 "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def vrm_path(bm) -> str:
    """The .vrm a body file was built from: next to the file name it
    records, under assets/vrm/, or as given."""
    f = bm.vrm["file"]
    for cand in (f, os.path.join(ROOT, "assets", "vrm", f),
                 os.path.join(ROOT, f)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"{f}: the body file's avatar is not under "
                            "assets/vrm/; pass --vrm")


def page_config(bm, bg=(0.12, 0.12, 0.14)) -> dict:
    """What the pages need from the body file: the camera and background."""
    return {"camera": bm.vrm["camera"], "bg": list(bg),
            "version": bm.vrm["version"]}


class _Handler(SimpleHTTPRequestHandler):
    extra = {}

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        got = self.extra.get(path)
        if got is not None:
            body, ctype = got() if callable(got) else got
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def guess_type(self, path):
        if str(path).endswith(".js"):
            return "text/javascript"
        return super().guess_type(path)


@contextlib.contextmanager
def serve(vrm, cfg, host="127.0.0.1"):
    """flow/web at /, the avatar at /avatar.vrm, cfg at /avatar.json, on an
    ephemeral port; yields the base URL."""
    with open(vrm, "rb") as fh:
        data = fh.read()
    extra = {"/avatar.vrm": (data, "application/octet-stream"),
             "/avatar.json": (json.dumps(cfg).encode(), "application/json")}
    handler = type("H", (_Handler,), {"extra": extra})
    srv = ThreadingHTTPServer((host, 0), functools.partial(handler,
                                                           directory=WEB))
    srv.daemon_threads = True
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://{host}:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def have_chromium() -> str | None:
    """None when Playwright and its Chromium are usable, else why not."""
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return "no playwright (pip install playwright && playwright " \
               "install chromium)"
    if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and \
            os.path.isdir(DEFAULT_BROWSERS):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = DEFAULT_BROWSERS
    return None


class Page:
    """render.html in headless Chromium: frame(state, dt) -> JPEG bytes."""

    def __init__(self, base, size=512, timeout=120.0):
        why = have_chromium()
        if why:
            raise RuntimeError(why)
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(args=CHROMIUM_ARGS)
        self.page = self.browser.new_page(viewport={"width": size,
                                                    "height": size})
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("console", lambda m: self.errors.append(m.text)
                     if m.type == "error" else None)
        self.page.goto(f"{base}/render.html?vrm=/avatar.vrm&cfg=/avatar.json"
                       f"&size={size}")
        self.page.wait_for_function(
            "() => window.TSK && (window.TSK.frame || window.TSK.error)",
            timeout=timeout * 1000)
        err = self.page.evaluate("() => window.TSK.error")
        if err:
            raise RuntimeError("render.html: " + err)

    def frame(self, state, dt, quality=0.92) -> bytes | None:
        url = self.page.evaluate("([s, dt, q]) => TSK.frame(s, dt, q)",
                                 [state, dt, quality])
        if url is None:
            return None
        return base64.b64decode(url.split(",", 1)[1])

    def eval(self, js, arg=None):
        return self.page.evaluate(js, arg)

    def close(self):
        with contextlib.suppress(Exception):
            self.browser.close()
        with contextlib.suppress(Exception):
            self._pw.stop()


def render(states, bm, out, fps=30.0, size=512, vrm=None, log=print,
           warmup=WARMUP):
    """states: [state] one per frame (VRMBody.state) -> out, a silent
    H.264 mp4 at fps (frame f at f / fps, as the audio is planned). Returns
    the frame count."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is needed for the video")
    vrm = vrm or vrm_path(bm)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v",
         "mjpeg", "-framerate", f"{fps:g}", "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE)
    import time
    t0 = time.monotonic()
    with serve(vrm, page_config(bm)) as base:
        pg = Page(base, size)
        try:
            dt = 1.0 / fps
            if states:
                for _ in range(int(round(warmup * fps))):
                    pg.frame(states[0], dt, 0)
            for i, st in enumerate(states):
                proc.stdin.write(pg.frame(st, dt))
                if log and i and i % 100 == 0:
                    log(f"vrmrender: {i}/{len(states)} frames, "
                        f"{(time.monotonic() - t0) / i:.3f} s each")
        finally:
            pg.close()
            proc.stdin.close()
            proc.wait()
    if pg.errors and log:
        log("vrmrender: page said: " + "; ".join(pg.errors[:5]))
    if log:
        log(f"vrmrender: {len(states)} frames -> {out} "
            f"({time.monotonic() - t0:.1f} s)")
    return len(states)


def contact_sheet(video, out, n=12, cols=4, width=256, times=None):
    """Frames of a video tiled into one PNG (ffmpeg's tile filter), with
    each frame's time; for a quick look at a render."""
    import tempfile
    from PIL import Image, ImageDraw
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", video],
                       capture_output=True, text=True)
    dur = float(r.stdout.strip() or 0)
    times = times or [dur * (i + 0.5) / n for i in range(n)]
    tiles = []
    with tempfile.TemporaryDirectory() as td:
        for i, t in enumerate(times):
            p = os.path.join(td, f"{i}.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss",
                            f"{t:.3f}", "-i", video, "-frames:v", "1",
                            "-vf", f"scale={width}:-1", p], check=True)
            im = Image.open(p).convert("RGB")
            ImageDraw.Draw(im).text((6, 4), f"{t:.2f}s", fill=(255, 255, 0))
            tiles.append(im)
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", help="a VRM director run directory "
                    "(states.json, speech.wav, subs.srt): renders "
                    "render/preview.mp4 and demo.mp4")
    ap.add_argument("--states", help="states.json (flow.vrm record)")
    ap.add_argument("--body", help="the VRM body file (default: the one "
                    "states.json names)")
    ap.add_argument("--vrm", help="the .vrm (default: from the body file)")
    ap.add_argument("--out", help="mp4 to write")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--sheet", default=None, metavar="PNG",
                    help="also write a contact sheet of the video")
    a = ap.parse_args(argv)
    try:
        from . import body as fb
    except ImportError:
        sys.path.insert(0, ROOT)
        from flow import body as fb
    states_path = a.states or (os.path.join(a.run, "states.json")
                               if a.run else None)
    if not states_path:
        ap.error("--run or --states")
    with open(states_path, encoding="utf-8") as fh:
        rec = json.load(fh)
    bm = fb.load(a.body or rec["body"])
    out = a.out or os.path.join(a.run or ".", "render", "preview.mp4")
    vrm = a.vrm or (rec.get("vrm") if rec.get("vrm") and os.path.exists(
        rec["vrm"]) else None)
    render(rec["states"], bm, out, rec.get("fps", 30.0), a.size, vrm)
    if a.run or a.audio:
        try:
            from .director import mux_cmd
        except ImportError:
            from flow.director import mux_cmd
        wav = a.audio or os.path.join(a.run, "speech.wav")
        srt = os.path.join(a.run, "subs.srt") if a.run else ""
        dest = (os.path.join(a.run, "demo.mp4") if a.run
                else out[:-4] + ".av.mp4")
        cmd = mux_cmd(out, wav, srt, dest)
        if cmd:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"vrmrender: {dest}")
    if a.sheet:
        print(f"vrmrender: sheet {contact_sheet(out, a.sheet)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
