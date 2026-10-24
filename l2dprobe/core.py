"""
Mesh geometry straight from the Cubism Core, with no GL and no renderer.

Neither live2d-py surface gives what the body measurement needs. 0.8.x's
live2d.v3 binds no vertex accessor at all; 1.0.0 adds GetDrawableVertices but
still no parent parts, opacities or render orders. Both wheels, however,
link the Cubism Core statically and export its whole C API (nm -D lists 490
csm* symbols in live2d/v3/_v3cpp.so 0.8.1 and live2d/_live2d.so 1.0.0), and
the Core is the part that deforms meshes: csmUpdateModel computes every
drawable's vertex positions on the CPU from the parameter and part-opacity
arrays. So this module loads the .moc3 itself through ctypes and reads those
arrays. Measured on Hiyori (Core 6.0.1, moc v3): load + 134 drawables, one
update and a full vertex read in well under a millisecond.

What this does not run: physics, pose fades, motions, breath, blink -- the
framework layer. Geometry here is the rest pose plus the parameter values
and part opacities written, which is what a forward model wants.

Coordinates are Cubism model units as the Core reports them: origin at the
canvas centre-ish (the rig's own origin), y up. Sides are the rig's own; see
flow.body for the convention measured from ParamEyeLOpen / ParamArmLA.
"""

from __future__ import annotations

import ctypes as C
import importlib.util
import os

import numpy as np

# Candidate extension modules that carry the Core, newest first; L2D_CORE
# overrides with an explicit path (a libLive2DCubismCore.so works too).
_MODULES = ("live2d._live2d", "live2d.v3._v3cpp")


class IncompatibleRig(RuntimeError):
    """The Core cannot load this moc (usually a newer moc3 version)."""


def core_path() -> str | None:
    env = os.environ.get("L2D_CORE")
    if env:
        return env
    for name in _MODULES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            return spec.origin
    return None


_LIB = None


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = core_path()
    if path is None:
        raise ImportError("no Cubism Core found: install live2d-py (any of "
                          "0.8.x / 1.0) or set L2D_CORE to a .so exporting "
                          "csmReviveMocInPlace")
    lib = C.CDLL(path)
    if not hasattr(lib, "csmReviveMocInPlace"):
        raise ImportError(f"{path} does not export the Cubism Core C API")
    _LIB = lib
    return lib


def available() -> bool:
    try:
        _lib()
        return True
    except (ImportError, OSError):
        return False


def _fn(name, restype, *argtypes, alt=()):
    lib = _lib()
    for n in (name,) + tuple(alt):
        f = getattr(lib, n, None)
        if f is not None:
            f.restype = restype
            f.argtypes = list(argtypes)
            return f
    return None


def _aligned(data: bytes, align: int):
    """A ctypes buffer holding data at an `align`-byte boundary, and that
    address. The Core requires 64 for the moc and 16 for the model."""
    buf = C.create_string_buffer(len(data) + align)
    addr = (C.addressof(buf) + align - 1) & ~(align - 1)
    if data:
        C.memmove(addr, data, len(data))
    return buf, addr


def _strings(ptr, n) -> list[str]:
    return [ptr[i].decode("utf-8", "replace") for i in range(n)]


class CoreModel:
    """One .moc3 instantiated in the Core. Parameters and part opacities are
    written by id; update() deforms; vertices(i) reads drawable i."""

    def __init__(self, moc_path: str):
        with open(moc_path, "rb") as fh:
            data = fh.read()
        vp = C.c_void_p
        # Both buffers stay referenced on self: the Core keeps pointers into
        # them, and a collected buffer is a segfault on the next update.
        self._moc_buf, maddr = _aligned(data, 64)
        consistent = _fn("csmHasMocConsistency", C.c_int, vp, C.c_uint)
        if consistent is not None and not consistent(maddr, len(data)):
            raise IncompatibleRig(f"{os.path.basename(moc_path)} fails the "
                                  "Core's moc consistency check")
        moc = _fn("csmReviveMocInPlace", vp, vp, C.c_uint)(maddr, len(data))
        if not moc:
            raise IncompatibleRig(f"csmReviveMocInPlace refused {moc_path}")
        size = _fn("csmGetSizeofModel", C.c_uint, vp)(moc)
        self._model_buf = C.create_string_buffer(size + 16)
        aaddr = (C.addressof(self._model_buf) + 15) & ~15
        m = _fn("csmInitializeModelInPlace", vp, vp, vp, C.c_uint)(
            moc, aaddr, size)
        if not m:
            raise IncompatibleRig("csmInitializeModelInPlace failed")
        self._m = m
        self._update = _fn("csmUpdateModel", None, vp)

        def get(name, res, alt=()):
            f = _fn(name, res, vp, alt=alt)
            return None if f is None else f(m)

        P = C.POINTER
        n_par = get("csmGetParameterCount", C.c_int)
        n_part = get("csmGetPartCount", C.c_int)
        n_draw = get("csmGetDrawableCount", C.c_int)
        self.param_ids = _strings(get("csmGetParameterIds", P(C.c_char_p)),
                                  n_par)
        self.part_ids = _strings(get("csmGetPartIds", P(C.c_char_p)), n_part)
        self.drawable_ids = _strings(get("csmGetDrawableIds", P(C.c_char_p)),
                                     n_draw)

        def arr(name, ctype, n, alt=()):
            p = get(name, P(ctype), alt)
            return None if not p else np.ctypeslib.as_array(p, (n,))

        # Writable views straight onto the Core's own arrays.
        self._values = arr("csmGetParameterValues", C.c_float, n_par)
        self._part_op = arr("csmGetPartOpacities", C.c_float, n_part)
        self.param_min = arr("csmGetParameterMinimumValues", C.c_float,
                             n_par).copy()
        self.param_max = arr("csmGetParameterMaximumValues", C.c_float,
                             n_par).copy()
        self.param_default = arr("csmGetParameterDefaultValues", C.c_float,
                                 n_par).copy()
        self.part_parent = arr("csmGetPartParentPartIndices", C.c_int,
                               n_part).copy()
        self.drawable_parent = arr("csmGetDrawableParentPartIndices",
                                   C.c_int, n_draw).copy()
        self.vertex_counts = arr("csmGetDrawableVertexCounts", C.c_int,
                                 n_draw).copy()
        self._opac = arr("csmGetDrawableOpacities", C.c_float, n_draw)
        # Core 6 renamed csmGetDrawableRenderOrders to csmGetRenderOrders
        # (which also covers offscreens, listed after the drawables).
        self._orders = arr("csmGetRenderOrders", C.c_int, n_draw,
                           alt=("csmGetDrawableRenderOrders",))
        self._pos = get("csmGetDrawableVertexPositions", P(P(C.c_float)))
        self._pindex = {p: i for i, p in enumerate(self.param_ids)}
        self._partindex = {p: i for i, p in enumerate(self.part_ids)}
        self._rest_parts = self._part_op.copy()
        self.reset()

    # ------------------------------------------------------------- writes

    def reset(self, parts: bool = True) -> None:
        self._values[:] = self.param_default
        if parts:
            self._part_op[:] = self._rest_parts

    def set(self, values: dict | None = None, parts: dict | None = None,
            reset: bool = True) -> None:
        """Write parameters (clamped to their range) and part opacities by
        id, then update. Unknown ids are ignored, as SetParameterValueById
        does."""
        if reset:
            self.reset()
        for pid, v in (values or {}).items():
            i = self._pindex.get(pid)
            if i is not None:
                self._values[i] = np.clip(v, self.param_min[i],
                                          self.param_max[i])
        for pid, v in (parts or {}).items():
            i = self._partindex.get(pid)
            if i is not None:
                self._part_op[i] = v
        self._update(self._m)

    # -------------------------------------------------------------- reads

    def vertices(self, i: int) -> np.ndarray:
        n = int(self.vertex_counts[i])
        return np.ctypeslib.as_array(self._pos[i], (n * 2,)).reshape(n, 2) \
            .copy()

    def all_vertices(self) -> np.ndarray:
        """Every drawable's vertices stacked, (sum of counts, 2); slices by
        self.offsets."""
        return np.concatenate([self.vertices(i)
                               for i in range(len(self.drawable_ids))])

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.vertex_counts)])

    def opacities(self) -> np.ndarray:
        return self._opac.copy()

    def render_orders(self) -> np.ndarray | None:
        return None if self._orders is None else self._orders.copy()

    def part_chain(self, part: int) -> list[str]:
        """Part ids from `part` up to the root."""
        out = []
        while part >= 0 and len(out) < 64:
            out.append(self.part_ids[part])
            part = int(self.part_parent[part])
        return out


def moc_of(model_json: str) -> str:
    import json
    with open(model_json, encoding="utf-8") as fh:
        moc = json.load(fh)["FileReferences"]["Moc"]
    return os.path.join(os.path.dirname(model_json), moc)


def load(model_json: str) -> CoreModel:
    return CoreModel(moc_of(model_json))


# ------------------------------------------------------------- sweeps

SWEEP_STEPS = 11
JUMP_RATIO = 0.5      # one step carrying more of the path than this: a jump
MOVE_EPS = 1e-4       # of the drawn extent: below this nothing moves
OPACITY_EPS = 1e-3    # a drawable opacity change below this is no change


def sweep(m: CoreModel, pid: str, steps: int = SWEEP_STEPS,
          parts: dict | None = None) -> dict:
    """Sweep one parameter over its range with everything else at default.

    Returns {"values" (S,), "move" (S, D): max vertex shift of each drawable
    from its rest (the parameter at default), "opacity" (S, D), "rest_opacity"
    (D,), "step" (S-1,): the largest vertex shift between consecutive steps,
    "extent": the rest mesh's larger side}. `parts` are part opacities written
    for every step (all 1.0 shows drawables a hidden pose part would zero).
    Part opacities are Core inputs, not outputs: no parameter changes them."""
    i = m.param_ids.index(pid)
    lo, hi = float(m.param_min[i]), float(m.param_max[i])
    m.set({}, parts)
    rest_v = m.all_vertices()
    rest_o = m.opacities()
    off = m.offsets
    vals = np.linspace(lo, hi, steps)
    move = np.zeros((steps, len(m.drawable_ids)))
    opac = np.zeros((steps, len(m.drawable_ids)))
    step = np.zeros(max(steps - 1, 0))
    prev = None
    for s, v in enumerate(vals):
        m.set({pid: float(v)}, parts)
        V = m.all_vertices()
        d = np.abs(V - rest_v).max(1) if len(V) else np.zeros(0)
        move[s] = [d[off[k]:off[k + 1]].max() if off[k + 1] > off[k]
                   else 0.0 for k in range(len(m.drawable_ids))]
        opac[s] = m.opacities()
        if prev is not None:
            step[s - 1] = float(np.abs(V - prev).max()) if len(V) else 0.0
        prev = V
    m.set({}, parts)
    ext = float(np.ptp(rest_v, 0).max()) if len(rest_v) else 0.0
    return {"values": vals, "move": move, "opacity": opac,
            "rest_opacity": rest_o, "step": step, "extent": ext}


def continuous_deformer(m: CoreModel, pid: str, steps: int = SWEEP_STEPS,
                        parts: dict | None = None,
                        jump: float = JUMP_RATIO) -> tuple[bool, str]:
    """(is it, why): whether `pid` is a continuous deformer in the Core.

    It is when sweeping it moves vertices, smoothly (no step between sweep
    points carries more than `jump` of the whole vertex path, the probe's
    stepiness test on exact geometry), and changes no drawable's opacity
    anywhere in the range. An art swap fails the opacity test; a parameter
    that only fades or moves nothing is no deformer. Pixel probes can call an
    eyelid categorical (the lid covers the iris, which reads as an alpha
    change); vertices cannot be fooled that way."""
    if pid not in m.param_ids:
        return False, "not in the moc"
    i = m.param_ids.index(pid)
    if not m.param_max[i] > m.param_min[i]:
        return False, "empty range"
    if parts is None:
        parts = {p: 1.0 for p in m.part_ids}
    sw = sweep(m, pid, steps, parts)
    dop = float(np.abs(sw["opacity"] - sw["rest_opacity"]).max()) \
        if sw["opacity"].size else 0.0
    if dop > OPACITY_EPS:
        k = int(np.abs(sw["opacity"] - sw["rest_opacity"]).max(0).argmax())
        return False, (f"opacity changes by {dop:.3f} "
                       f"({m.drawable_ids[k]})")
    path = float(sw["step"].sum())
    if path <= MOVE_EPS * max(sw["extent"], 1e-9):
        return False, "moves no vertex"
    ratio = float(sw["step"].max()) / path
    if ratio > jump:
        return False, f"vertices jump ({ratio:.2f} of the path in one step)"
    peak = sw["move"].max(1)
    mono = bool(np.all(np.diff(peak[:int(peak.argmin()) + 1]) <= 1e-6)
                and np.all(np.diff(peak[int(peak.argmin()):]) >= -1e-6))
    hit = int((sw["move"].max(0) > 1e-6).sum())
    return True, (f"{hit} drawables move up to {peak.max():.4g} model units, "
                  f"largest step {ratio:.2f} of the path"
                  + (", monotonic" if mono else "")
                  + ", no opacity change")
