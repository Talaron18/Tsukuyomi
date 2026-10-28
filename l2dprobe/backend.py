"""
Headless Live2D Cubism rendering backend for parameter probing.

Requires:
    pip install live2d-py PyOpenGL glfw numpy

Headless display notes:
    The Cubism renderer needs a real GL context. Easiest reliable path on a
    server is Xvfb:

        xvfb-run -s "-screen 0 1024x1024x24" python run_probe.py ...

    With no X server at all, pass osmesa=True: GLFW's Null platform plus an
    OSMesa software context. Note that GLFW has no "EGL platform" -- EGL is a
    context-creation API, and on the X11 platform it still needs a display.

Design notes:
    - We render into an offscreen FBO, not the window backbuffer. Hidden-window
      backbuffer contents are undefined on several drivers.
    - The FBO carries a DEPTH24_STENCIL8 renderbuffer. Live2D clipping masks
      use the stencil buffer; without it, any model with masks renders wrong.
    - live2d-py's surface API has drifted between versions, so parameter access
      goes through a shim that probes for whichever spelling exists and fails
      loudly (printing dir(model)) rather than silently doing nothing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

from .sanitize import ProbeModel

try:
    import glfw
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        f"could not import glfw: {exc}. If glfw is installed, this is a "
        "loader failure, not a missing package. Run: python -m l2dprobe.doctor"
    ) from exc

# PyOpenGL must be told to load OSMesa *before* OpenGL.GL is imported, so
# run_probe sets PYOPENGL_PLATFORM early. Left here as documentation of the
# ordering constraint: importing this module first would lock in libGL.
try:
    from OpenGL.GL import (
        GL_COLOR_ATTACHMENT0,
        GL_DEPTH24_STENCIL8,
        GL_DEPTH_STENCIL_ATTACHMENT,
        GL_FRAMEBUFFER,
        GL_FRAMEBUFFER_COMPLETE,
        GL_LINEAR,
        GL_RENDERBUFFER,
        GL_RGBA,
        GL_RGBA8,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_UNSIGNED_BYTE,
        glBindFramebuffer,
        glBindRenderbuffer,
        glBindTexture,
        glCheckFramebufferStatus,
        glFramebufferRenderbuffer,
        glFramebufferTexture2D,
        glGenFramebuffers,
        glGenRenderbuffers,
        glGenTextures,
        glReadPixels,
        glRenderbufferStorage,
        glTexImage2D,
        glTexParameteri,
        glViewport,
    )
except AttributeError as exc:  # pragma: no cover
    # PyOpenGL's platform module imported but exposes no GL handle, so the
    # underlying system library is missing. Under PYOPENGL_PLATFORM=osmesa that
    # means libOSMesa.so is absent; otherwise libGL.so.1 is.
    _plat = os.environ.get("PYOPENGL_PLATFORM", "(default/glx)")
    _hint = ("sudo apt install libosmesa6" if _plat == "osmesa"
             else "sudo apt install libgl1")
    raise ImportError(
        f"PyOpenGL selected platform {_plat!r} but found no GL library "
        f"({exc}).\n"
        f"Install it with:  {_hint}\n"
        "Or use a virtual display instead:\n"
        "  sudo apt install xvfb && xvfb-run -s '-screen 0 1024x1024x24' ...\n"
        "Diagnostics: python -m l2dprobe.doctor"
    ) from exc


# Do not swallow this. `live2d` can import (it prints a banner) while
# `live2d.v3` fails for an entirely different reason: the native extension
# not loading, a missing shared library, or an ABI mismatch. Reporting
# "pip install live2d-py" for those sends you the wrong way.
try:
    import live2d.v3 as live2d
except Exception as exc:  # pragma: no cover  # noqa: BLE001
    raise ImportError(
        f"live2d.v3 failed to import ({type(exc).__name__}: {exc}).\n"
        "The package is likely installed but its native extension will not "
        "load. Run:  python -m l2dprobe.doctor"
    ) from exc


def ensure_display(argv=None) -> None:
    """Re-exec under xvfb-run when there is no display.

    Every GL entry point needs a display, and forgetting the wrapper produces a
    glfw error several layers down rather than an obvious one. Re-execing costs
    one process and removes a whole class of "why did this fail" round trips.
    Does nothing if DISPLAY is set, if xvfb-run is absent, or if we are already
    inside the re-exec.
    """
    import shutil
    import subprocess
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    if os.environ.get("_L2D_XVFB"):
        return                       # already re-execed; do not loop
    if shutil.which("xvfb-run") is None:
        return                       # let the caller fail with its own message
    env = dict(os.environ, _L2D_XVFB="1")
    cmd = ["xvfb-run", "-a", "-s", "-screen 0 1024x1024x24",
           sys.executable] + (argv if argv is not None else sys.argv)
    print("no DISPLAY; re-running under xvfb-run", flush=True)
    raise SystemExit(subprocess.call(cmd, env=env))


# One GL context and one Cubism framework per process, shared by every backend
# opened in it. Tearing them down per rig (live2d.dispose + glfw.terminate) and
# re-initialising left the next model drawing with no textures: correct
# silhouette, every pixel black. Any process that opened two backends --
# render_dataset's pool workers, run_probe --batch, contact_sheet -- produced
# black renders for every rig after its first.
_SHARED: dict = {}


class IncompatibleRig(RuntimeError):
    """The Cubism Core cannot load this rig. Skip it rather than crash."""


@dataclass(frozen=True)
class ParamSpec:
    """Static description of one rig parameter, straight from the model."""

    index: int
    pid: str
    minimum: float
    maximum: float
    default: float

    @property
    def span(self) -> float:
        return float(self.maximum - self.minimum)

    def sample(self, t: float) -> float:
        """t in [0, 1] -> value in [min, max]."""
        return float(self.minimum + t * self.span)


def _first_attr(obj, names):
    for name in names:
        fn = getattr(obj, name, None)
        if fn is not None:
            return fn, name
    return None, None


class Live2DBackend:
    """Loads one .model3.json and renders it at arbitrary parameter values."""

    # Where to apply parameter writes relative to model.Update().
    # live2d-py's LAppModel.Update() does LoadParameters -> motion -> SaveParameters,
    # so writes made before it can be restored away. Which order works depends on
    # the version, so we detect it empirically in calibrate().
    ORDERS = ("after", "before", "sandwich")
    # With physics and pose live, "after" is wrong however well it scores: the
    # per-frame reset lands after Update() has run the solver, so every physics
    # output is put back to its default and hair, cloth and ribbons never move.
    # Calibration cannot see this -- it probes head parameters, which "after"
    # applies fine -- so it picked "after" on some runs and silently froze all
    # secondary motion. Dynamic renders only consider orders that leave the
    # solver's output alone.
    DYNAMIC_ORDERS = ("before", "sandwich")

    def __init__(self, model_json: str, size: int = 512, bg=(0.0, 0.0, 0.0, 0.0),
                 osmesa: bool = False, order: str = "after",
                 strip_dynamics: bool = True,
                 gl_version: tuple[int, int] | None = None,
                 fps: float | None = None):
        if not os.path.exists(model_json):
            raise FileNotFoundError(
                f"{model_json}\n"
                "In CubismWebSamples the models sit under Samples/Resources/, "
                "not Resources/. To list what is actually present:\n"
                "  python -m l2dprobe.run_probe --find ."
            )
        self.model_json = model_json
        self.size = size
        self.bg = bg
        self.osmesa = osmesa
        self.order = order
        self.strip_dynamics = strip_dynamics
        self.gl_version = gl_version
        # Seconds of simulated time per render. LAppModel.Update() measures
        # wall-clock time, so offline rendering at hundreds of frames a second
        # advanced physics and pose fades by ~2-3 ms per video frame instead of
        # 33 ms: springs barely moved and a 0.5 s pose fade took several
        # seconds of video. A fixed step ties dynamics to the output frame rate.
        self.dt = 1.0 / fps if fps else None
        self._update = None
        self._probe_model = None
        self._window = None
        self._fbo = None
        self.model = None
        self.params: list[ParamSpec] = []
        self._setter = None
        self._api_report: dict[str, str] = {}

    # ---------------------------------------------------------------- context

    def _init_context(self):
        # GLFW separates *platform* (X11, Wayland, Null) from *context creation
        # API* (GLX, EGL, OSMesa). There is no EGL platform. Fully headless with
        # no X server means the Null platform plus an OSMesa (software) context.
        if self.osmesa:
            plat = getattr(glfw, "PLATFORM", 0x00050003)
            plat_null = getattr(glfw, "PLATFORM_NULL", 0x00060005)
            glfw.init_hint(plat, plat_null)

        if not glfw.init():
            raise RuntimeError(
                "glfw.init() failed. With DISPLAY unset you need either:\n"
                "  xvfb-run -s '-screen 0 1024x1024x24' python -m l2dprobe...\n"
                "or --osmesa for software rendering with no X server at all."
            )

        if self.osmesa:
            api = getattr(glfw, "CONTEXT_CREATION_API", 0x0002200B)
            osmesa_api = getattr(glfw, "OSMESA_CONTEXT_API", 0x00036003)
            glfw.window_hint(api, osmesa_api)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        # Do NOT request 3.3 by default. Cubism's renderer is
        # CubismRenderer_OpenGLES2 and its shaders use legacy GLSL
        # (attribute/varying), which will not compile in a 3.3 core context.
        # The failure is silent: shaders fail, Draw() becomes a no-op, and the
        # canvas stays black. GLFW's default (1.0) means "any context the
        # driver likes", which is what the live2d-py examples rely on.
        if self.gl_version is not None:
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, self.gl_version[0])
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, self.gl_version[1])
        self._window = glfw.create_window(self.size, self.size, "probe", None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window failed")
        glfw.make_context_current(self._window)

    def _init_fbo(self):
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.size, self.size, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.size, self.size)

        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL_RENDERBUFFER, rbo)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("offscreen framebuffer incomplete")
        self._fbo = fbo
        glViewport(0, 0, self.size, self.size)

    # ------------------------------------------------------------------ model

    def open(self):
        shared = _SHARED.get("ctx")
        if shared is not None:
            if shared["size"] != self.size or shared["osmesa"] != self.osmesa:
                raise RuntimeError(
                    "a Live2D backend was already opened in this process with "
                    f"size={shared['size']} osmesa={shared['osmesa']}; the GL "
                    "context is shared and cannot change. Use one size per "
                    "process.")
            self._window, self._fbo = shared["window"], shared["fbo"]
            glfw.make_context_current(self._window)
        else:
            self._init_context()
            live2d.init()
            live2d.glInit()
            self._init_fbo()
            _SHARED["ctx"] = {"window": self._window, "fbo": self._fbo,
                              "size": self.size, "osmesa": self.osmesa}

        # Physics/pose/motions are time-integrated and would make renders
        # history-dependent. Strip them from the definition before loading.
        self._probe_model = ProbeModel(self.model_json,
                                       strip=self.strip_dynamics).__enter__()
        if self._probe_model.removed:
            self._api_report["stripped"] = ",".join(self._probe_model.removed)

        self.model = live2d.LAppModel()

        # moc3 version 6 files segfault against a Core that only supports 5.
        # Check first; a clean skip beats taking the whole batch down.
        moc = self._find_moc(self._probe_model.path)
        if moc is not None:
            try:
                if not self.model.HasMocConsistencyFromFile(moc):
                    raise IncompatibleRig(
                        f"{os.path.basename(moc)} failed the Core consistency "
                        "check (usually a moc3 version newer than this "
                        "Cubism Core supports)")
            except IncompatibleRig:
                raise
            except Exception:  # noqa: BLE001
                pass  # check unavailable in this build; fall through and try

        self.model.LoadModelJson(self._probe_model.path)
        self.model.Resize(self.size, self.size)

        self._disable_autonomics()
        self._bind_param_api()
        self._bind_update()
        self._bind_parts()
        if not self.strip_dynamics and self.order not in self.DYNAMIC_ORDERS:
            self.order = self.DYNAMIC_ORDERS[0]
        return self

    def _bind_parts(self):
        """Map part ids to the indices SetPartOpacity takes.

        Part opacity is how most rigs pick an arm or hand variant. Live2D's
        Native motion player applies PartOpacity curves only through a pose3
        file; a rig shipped without one (Azur Lane's, whose Unity runtime
        applies the curves directly) draws every variant at once unless we
        set the opacities ourselves.
        """
        ids_fn = getattr(self.model, "GetPartIds", None)
        self._set_part = getattr(self.model, "SetPartOpacity", None)
        self._pose = False
        self.part_index: dict[str, int] = {}
        if ids_fn is None or self._set_part is None:
            self._api_report["parts"] = "MISSING (no part opacity control)"
            return
        self.part_index = {str(p): i for i, p in enumerate(ids_fn())}
        self._api_report["parts"] = f"{len(self.part_index)}"
        # With a pose3 file the pose decides which part of a group is shown,
        # and it reads that from a parameter named after each part -- the
        # slot Live2D's own motion player writes PartOpacity curves into --
        # not from the opacity we set. Writing only the opacity, the pose kept
        # fading toward its default part while we forced the authored one:
        # Natori drew a translucent second arm behind every gesture.
        self._pose = False
        if not self.strip_dynamics:
            try:
                import json as _json
                with open(self.model_json, encoding="utf-8") as fh:
                    refs = _json.load(fh).get("FileReferences", {})
                self._pose = bool(refs.get("Pose"))
            except Exception:  # noqa: BLE001
                pass
        if self._pose:
            self._api_report["parts"] += " (pose-fed)"

    def _apply_parts(self, parts):
        if not parts or self._set_part is None:
            return
        for pid, value in parts.items():
            i = self.part_index.get(pid)
            if i is not None:
                self._set_part(i, float(value))
                if self._pose:
                    self._setter(pid, float(value), 1.0)

    def _bind_update(self):
        """Pick an update entry point that honours a fixed timestep.

        live2d-py >= 0.8 wraps a native model whose Update takes the delta
        explicitly; the Python wrapper's Update() feeds it wall-clock time.
        Older builds expose the native Update directly, with or without a delta
        argument. Without a usable fixed step the render still works, but
        physics timing is wrong, so that is recorded rather than hidden.
        """
        if self.dt is None:
            self._update = self.model.Update
            self._api_report["timestep"] = "wallclock"
            return
        inner = getattr(self.model, "_model", None)
        for target, label in ((inner, "native"), (self.model, "direct")):
            fn = getattr(target, "Update", None) if target is not None else None
            if fn is None:
                continue
            try:
                fn(0.0)
            except TypeError:
                continue
            # self.dt read at each call, not bound here: flow.live skips the
            # GL render of a frame it cannot keep up with and sets dt to the
            # frames that passed, so physics and pose fades still advance
            # with the clock.
            self._update = lambda: fn(self.dt)
            self._api_report["timestep"] = f"fixed {self.dt:.4f}s ({label})"
            return
        self._update = self.model.Update
        self._api_report["timestep"] = "wallclock (no fixed-step API found)"
        print("WARNING: this live2d-py build has no fixed-timestep Update; "
              "physics will run at render speed, not video speed",
              file=sys.stderr)

    @staticmethod
    def _find_moc(model_json: str) -> str | None:
        try:
            import json as _json
            with open(model_json, encoding="utf-8") as fh:
                refs = _json.load(fh).get("FileReferences", {})
            moc = refs.get("Moc")
            if not moc:
                return None
            return os.path.join(os.path.dirname(model_json), moc)
        except Exception:  # noqa: BLE001
            return None

    def _disable_autonomics(self):
        """Blink/breath/motion must be off or every render is time-dependent."""
        for name, arg in (
            ("SetAutoBreathEnable", False),
            ("SetAutoBlinkEnable", False),
        ):
            fn = getattr(self.model, name, None)
            if fn is not None:
                fn(arg)
                self._api_report[name] = "ok"
            else:
                self._api_report[name] = "MISSING"
        stop, name = _first_attr(self.model, ["StopAllMotions", "StopAllMotion"])
        if stop is not None:
            try:
                stop()
                self._api_report[name] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._api_report[name] = f"raised {exc!r}"

    def _bind_param_api(self):
        count_fn, count_name = _first_attr(self.model, ["GetParameterCount"])
        get_fn, get_name = _first_attr(self.model, ["GetParameter"])
        set_fn, set_name = _first_attr(
            self.model, ["SetParameterValue", "SetParameterValueById"]
        )

        if count_fn is None or get_fn is None or set_fn is None:
            raise RuntimeError(
                "could not locate the parameter API on LAppModel.\n"
                "This means live2d-py changed its surface. Available members:\n  "
                + "\n  ".join(sorted(n for n in dir(self.model) if not n.startswith("_")))
            )

        self._api_report.update(
            {"count": count_name, "get": get_name, "set": set_name}
        )
        self._setter = set_fn

        specs = []
        for i in range(count_fn()):
            p = get_fn(i)
            specs.append(
                ParamSpec(
                    index=i,
                    pid=str(getattr(p, "id", getattr(p, "pid", f"param{i}"))),
                    minimum=float(getattr(p, "min", getattr(p, "minimum", 0.0))),
                    maximum=float(getattr(p, "max", getattr(p, "maximum", 1.0))),
                    default=float(getattr(p, "default", 0.0)),
                )
            )
        self.params = specs

    # ----------------------------------------------------------------- render

    def _apply(self, overrides, reset: bool = True, parts=None):
        # Reset every frame so probes never leak into each other.
        if reset:
            for spec in self.params:
                self._setter(spec.pid, spec.default, 1.0)
        if overrides:
            for pid, value in overrides.items():
                self._setter(pid, float(value), 1.0)
        self._apply_parts(parts)

    def set_view(self, scale: float = 1.0, offset_x: float = 0.0,
                 offset_y: float = 0.0) -> None:
        """Camera only (flow.framing): live2d-py's model scale about the
        canvas centre, then a shift in half-canvas units (+x right, +y
        up). Parameters and measurements are untouched."""
        self.model.SetScale(float(scale))
        self.model.SetOffset(float(offset_x), float(offset_y))

    def render(self, overrides: dict[str, float] | None = None,
               order: str | None = None,
               parts: dict[str, float] | None = None) -> np.ndarray:
        """Render with all params at default except `overrides`. Returns HxWx4 uint8.

        `parts` sets part opacities by part id. Parts not named keep whatever
        opacity they last had, which is also how a motion without a pose file
        leaves them.
        """
        order = order or self.order
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, self.size, self.size)
        live2d.clearBuffer(*self.bg)

        update = self._update or self.model.Update
        if order == "before":
            self._apply(overrides, parts=parts)
            update()
        elif order == "after":
            update()
            self._apply(overrides, parts=parts)
        elif order == "sandwich":
            self._apply(overrides, parts=parts)
            update()
            # Re-assert only what we drive. A second full reset would erase
            # the physics and pose results Update() just produced.
            self._apply(overrides, reset=self.strip_dynamics, parts=parts)
        else:
            raise ValueError(f"unknown parameter order {order!r}")

        self.model.Draw()
        return self._read()

    def render_native(self) -> np.ndarray:
        """Render whatever the model's own update produces, writing nothing.

        This is Live2D Native's view of authored playback: its motion player
        applies parameter curves, then physics and pose. It is not the whole
        truth -- without a pose3 file it ignores PartOpacity curves, which
        engines such as Unity apply directly -- so a rig that switches parts
        that way shows every variant here as well.
        """
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, self.size, self.size)
        live2d.clearBuffer(*self.bg)
        (self._update or self.model.Update)()
        self.model.Draw()
        return self._read()

    def _read(self) -> np.ndarray:
        buf = glReadPixels(0, 0, self.size, self.size, GL_RGBA, GL_UNSIGNED_BYTE)
        # PyOpenGL returns bytes on some builds and an ndarray on others.
        # np.frombuffer raises on the latter, so normalise first.
        if isinstance(buf, np.ndarray):
            img = buf.astype(np.uint8, copy=False).reshape(self.size, self.size, 4)
        else:
            img = np.frombuffer(buf, dtype=np.uint8).reshape(self.size, self.size, 4)
        return np.flipud(img).copy()  # GL origin is bottom-left

    def gl_info(self) -> dict:
        """Report what context we actually got, not what we asked for."""
        from OpenGL.GL import (GL_RENDERER, GL_SHADING_LANGUAGE_VERSION,
                               GL_VENDOR, GL_VERSION, glGetString)
        out = {}
        for name, enum in (("version", GL_VERSION), ("glsl", GL_SHADING_LANGUAGE_VERSION),
                           ("renderer", GL_RENDERER), ("vendor", GL_VENDOR)):
            try:
                raw = glGetString(enum)
                out[name] = raw.decode() if isinstance(raw, bytes) else str(raw)
            except Exception as exc:  # noqa: BLE001
                out[name] = f"<{exc}>"
        return out

    def canary(self) -> tuple[bool, tuple[int, int, int, int]]:
        """Clear to opaque red and read it back.

        This isolates the plumbing from Live2D entirely. If red does not come
        back, the context, FBO or readback is broken and no amount of Live2D
        debugging will help. If red does come back but the model does not
        appear, the renderer is the problem.
        """
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, self.size, self.size)
        live2d.clearBuffer(1.0, 0.0, 0.0, 1.0)
        buf = glReadPixels(0, 0, self.size, self.size, GL_RGBA, GL_UNSIGNED_BYTE)
        arr = (buf.astype(np.uint8, copy=False) if isinstance(buf, np.ndarray)
               else np.frombuffer(buf, dtype=np.uint8))
        px = arr.reshape(self.size, self.size, 4)[self.size // 2, self.size // 2]
        got = tuple(int(v) for v in px)
        return got[0] > 200 and got[3] > 200, got

    def calibrate(self, verbose: bool = False) -> str:
        """Pick the parameter-write order that actually affects the render.

        Uses a few wide-range parameters as test probes and scores each ordering
        by how much it moves pixels. Guessing this wrong yields a clean run where
        every parameter looks dead, so we measure instead of assuming.
        """
        # Candidates, in order of how likely they are to actually do
        # something. Ranking purely by widest range fails on rigs carrying
        # hundreds of parameters where the widest slots are unused customs:
        # all six probes move nothing and calibrate wrongly concludes that
        # parameter writes do not work. Standard IDs are tried first because
        # they are almost always bound to real deformers.
        STANDARD_FIRST = ("ParamAngleX", "ParamAngleY", "ParamAngleZ",
                          "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
                          "ParamEyeBallX", "ParamEyeLOpen", "ParamMouthOpenY")
        live = [s for s in self.params if s.span > 0]
        if not live:
            raise RuntimeError("model exposes no parameters with a usable range")
        by_id = {s.pid: s for s in live}
        probes = [by_id[p] for p in STANDARD_FIRST if p in by_id]
        probes += [s for s in sorted(live, key=lambda s: -s.span)
                   if s.pid not in {x.pid for x in probes}]
        probes = probes[:24]

        scores: dict[str, float] = {}
        orders = self.ORDERS if self.strip_dynamics else self.DYNAMIC_ORDERS
        for order in orders:
            base = self.render(order=order).astype(np.int16)
            total = 0.0
            for spec in probes:
                far = (spec.maximum
                       if abs(spec.maximum - spec.default) >= abs(spec.default - spec.minimum)
                       else spec.minimum)
                img = self.render({spec.pid: far}, order=order).astype(np.int16)
                total += float(np.abs(img - base).mean()) / 255.0
                # One clearly responsive parameter settles this ordering; no
                # need to render the remaining candidates.
                if total > 1e-3:
                    break
            scores[order] = total
            if verbose:
                print(f"  order={order:<9} response={total:.6f}")

        best = max(scores, key=scores.get)
        if scores[best] <= 1e-5:
            # Distinguish "parameters do not apply" from "nothing renders".
            # Comparing renders to each other cannot tell these apart: a blank
            # canvas differenced against a blank canvas is also zero.
            probe_img = self.render()
            drawn = float((probe_img[..., 3] > 0).mean())
            ink = float(probe_img[..., :3].max())
            if drawn < 1e-4 and ink < 1.0:
                raise RuntimeError(
                    "nothing is being rendered: the canvas is completely "
                    f"empty (alpha coverage {drawn:.6f}, max rgb {ink:.0f}).\n"
                    "This is a drawing problem, not a parameter problem. The "
                    "model may not be drawing into our FBO, or the projection "
                    "may place it off-canvas.\n"
                    "Investigate with:  python -m l2dprobe.inspect_tool "
                    f"{self.model_json!r} --dump dbg/"
                )
            raise RuntimeError(
                "the model renders "
                f"(alpha coverage {drawn:.4f}) but no parameter ordering "
                "changes the image. Parameter writes are not reaching the "
                "mesh; LAppModel likely needs an update entry point other "
                "than Update().\n"
                "Dump the API surface with:  python -m l2dprobe.inspect_tool "
                f"{self.model_json!r} --dump dbg/\n"
                f"scores: {scores}"
            )
        self.order = best
        self._api_report["order"] = best
        return best

    def close(self):
        # The context and framework stay up for the next backend in this
        # process; see _SHARED. The OS reclaims them at exit.
        if self._probe_model is not None:
            self._probe_model.__exit__(None, None, None)
            self._probe_model = None
        self.model = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------ self-checks

    def check_determinism(self, trials: int = 3) -> tuple[bool, float]:
        """Same input must give identical pixels.

        If this fails, something time-varying is still running (physics, breath,
        an idle motion, a random pose). Every descriptor downstream would be
        contaminated by it, so we refuse to continue.
        """
        ref = self.render()
        worst = 0.0
        for _ in range(trials - 1):
            cur = self.render()
            worst = max(worst, float(np.abs(cur.astype(np.int16) - ref).max()))
        return worst == 0.0, worst

    def check_state_leak(self, pid: str | None = None) -> dict:
        """Detect parameter writes landing one frame late.

        If SetParameterValue lands after csmUpdateModel has already run, the
        value does not reach the mesh this frame but persists into the next
        one. Every descriptor would then describe the PREVIOUS parameter
        probed, and check_determinism would still pass because the lag is
        perfectly deterministic.

        The tell: rendering the same override twice in a row gives a different
        image the second time. A correct implementation is order-independent
        and gives identical images.
        """
        spec = next((s for s in self.params if s.pid == pid), None)
        if spec is None:
            spec = max((s for s in self.params if s.span > 0),
                       key=lambda s: s.span)
        far = spec.maximum

        a = self.render()                    # neutral
        b = self.render({spec.pid: far})     # override, preceded by neutral
        c = self.render({spec.pid: far})     # same override, preceded by itself
        d = self.render()                    # neutral, preceded by override
        e = self.render()                    # neutral, preceded by neutral

        def diff(x, y):
            return float(np.abs(x.astype(np.int16) - y.astype(np.int16)).mean())

        out = {
            "param": spec.pid,
            "effect_size": diff(a, b),      # want clearly > 0
            "repeat_drift": diff(b, c),     # want 0; nonzero means one-frame lag
            "neutral_drift": diff(d, e),    # want 0; nonzero means leakage
            "settled_effect": diff(c, e),   # effect once state has settled
        }
        out["lagged"] = out["repeat_drift"] > 0.01 or out["neutral_drift"] > 0.01
        return out

    def check_responsiveness(self, threshold: float = 1e-4) -> list[str]:
        """Return params that visibly move, as a sanity check that sets apply.

        If this comes back empty, SetParameterValue is being overwritten by
        model.Update() and the call order needs to change.
        """
        base = self.render().astype(np.int16)
        alive = []
        for spec in self.params:
            if spec.span == 0:
                continue
            far = spec.maximum if abs(spec.maximum - spec.default) > abs(
                spec.default - spec.minimum) else spec.minimum
            img = self.render({spec.pid: far}).astype(np.int16)
            if np.abs(img - base).mean() / 255.0 > threshold:
                alive.append(spec.pid)
        return alive
