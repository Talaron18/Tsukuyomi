"""
Turn a Live2D rig into a set of grounded, rig-agnostic parameter descriptors.

The point: we cannot trust parameter names. Real rigs ship IDs like `Param14`,
`パラメータ29`, or `ParamHairFuwa`. So instead of parsing names, we *observe*
what each parameter does to the rendered image, and describe it by its effect.

For each parameter we sweep K values across [min, max], diff each render against
the default pose, and extract:

    support     32x32 heatmap of where on the canvas the parameter has effect
    area        fraction of the canvas it touches
    centroid    normalised (x, y) of its effect
    bbox        normalised extent
    response    magnitude at each sweep step (shape of the response curve)
    linearity   correlation of response with the swept value
    stepiness   how much of the response arrives in one discontinuous jump
    alpha_ratio how much of the change is opacity vs deformation
    flow        dominant displacement direction, from centroid drift

Downstream, `kind` is the important field. A parameter that switches a part on
and off is categorical; a flow-matching motion model must not interpolate
through it. Everything else is continuous.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

try:
    from .backend import Live2DBackend, ParamSpec
except ImportError:  # running as loose scripts rather than a package
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from yachiyo.l2dprobe.backend import Live2DBackend, ParamSpec

GRID = 32
SGRID = 16   # signed maps are coarser; they encode direction, not shape
DEAD_AREA = 2e-4      # below this fraction of canvas touched -> unused param
STEP_RATIO = 0.55     # response arriving in one jump -> categorical
ALPHA_DOMINANT = 0.7  # change is mostly opacity -> part toggle


@dataclass
class ParamDescriptor:
    pid: str
    index: int
    minimum: float
    maximum: float
    default: float

    kind: str = "continuous"        # continuous | categorical | dead
    area: float = 0.0
    centroid: tuple[float, float] = (0.0, 0.0)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    linearity: float = 0.0
    stepiness: float = 0.0
    alpha_ratio: float = 0.0
    flow: tuple[float, float] = (0.0, 0.0)
    response: list[float] = field(default_factory=list)
    sweep: list[float] = field(default_factory=list)
    mirror_of: str | None = None
    support: np.ndarray | None = None    # (GRID, GRID) magnitude, |delta|
    flow_lo: np.ndarray | None = None    # (SGRID, SGRID, 2) displacement at min
    flow_hi: np.ndarray | None = None    # (SGRID, SGRID, 2) displacement at max
    curl: float = 0.0
    divergence: float = 0.0
    travel_lo: float = 0.0   # how far min sits from default, as a fraction of span
    travel_hi: float = 0.0   # how far max sits from default

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("support", "flow_lo", "flow_hi"):
            d.pop(k)
        return d


def content_bbox(base: np.ndarray, thresh: int = 8) -> tuple[int, int, int, int]:
    """Bounding box of the character in the base render, from its alpha.

    Rigs are framed differently: one may fill the canvas, another sit small and
    offset. Descriptors expressed in raw canvas coordinates would therefore put
    the same semantic parameter in different places on different rigs, and the
    cross-rig comparison would fail for a framing reason rather than a real
    one. Everything spatial is normalised into this box instead.
    """
    alpha = base[..., 3]
    ys, xs = np.nonzero(alpha > thresh)
    if len(xs) == 0:
        h, w = alpha.shape
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _crop_resize(mask: np.ndarray, bbox, grid: int = GRID) -> np.ndarray:
    """Crop to the character box, then resample to grid x grid.

    Nearest-neighbour on purpose: no scipy dependency, and the support map is
    a coarse occupancy field where interpolation buys nothing.
    """
    x0, y0, x1, y1 = bbox
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((grid, grid), np.float32)
    h, w = crop.shape
    yi = np.clip((np.arange(grid) + 0.5) * h / grid, 0, h - 1).astype(int)
    xi = np.clip((np.arange(grid) + 0.5) * w / grid, 0, w - 1).astype(int)
    return crop[np.ix_(yi, xi)].astype(np.float32)


def _downsample(mask: np.ndarray, grid: int = GRID) -> np.ndarray:
    h, w = mask.shape
    ph, pw = h // grid, w // grid
    m = mask[: ph * grid, : pw * grid]
    return m.reshape(grid, ph, grid, pw).mean(axis=(1, 3)).astype(np.float32)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Composite RGBA over mid-grey and return float32 luminance.

    Compositing matters: leaving alpha out makes the transparent border a hard
    edge that generates spurious flow along the whole silhouette.
    """
    a = (img[..., 3:4].astype(np.float32) / 255.0)
    rgb = img[..., :3].astype(np.float32)
    comp = rgb * a + 128.0 * (1.0 - a)
    return comp.mean(axis=2)


def _flow_fft(a: np.ndarray, b: np.ndarray, grid: int) -> np.ndarray:
    """Block-wise phase correlation. Dependency-free fallback for optical flow.

    Each block's dominant translation is the argmax of the inverse FFT of the
    normalised cross-power spectrum. Crude next to Farneback but it recovers
    the sign of motion, which is the entire point.
    """
    h, w = a.shape
    bh, bw = h // grid, w // grid
    out = np.zeros((grid, grid, 2), np.float32)
    if bh < 4 or bw < 4:
        return out
    win = (np.hanning(bh)[:, None] * np.hanning(bw)[None, :]).astype(np.float32)
    for i in range(grid):
        for j in range(grid):
            pa = a[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] * win
            pb = b[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] * win
            if pa.std() < 1e-3 and pb.std() < 1e-3:
                continue
            fa, fb = np.fft.rfft2(pa), np.fft.rfft2(pb)
            cross = fa * np.conj(fb)
            mag = np.abs(cross)
            cross = np.where(mag > 1e-9, cross / (mag + 1e-9), 0)
            corr = np.fft.irfft2(cross, s=pa.shape)
            dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
            if dy > bh // 2:
                dy -= bh
            if dx > bw // 2:
                dx -= bw
            out[i, j] = (dx / bw, dy / bh)
    return out


def _flow_field(base: np.ndarray, img: np.ndarray, bbox, grid: int = SGRID
                ) -> np.ndarray:
    """Dense displacement field, character-normalised, as (grid, grid, 2).

    Displacement is appearance-invariant in a way signed intensity is not:
    moving right is +dx whether the character's hair is dark or light. That
    invariance is exactly what cross-rig matching needs, and it is what
    separates head yaw from head roll, since rotation shows up as curl while
    translation does not.
    """
    x0, y0, x1, y1 = bbox
    a = _to_gray(base)[y0:y1, x0:x1]
    b = _to_gray(img)[y0:y1, x0:x1]
    if a.size == 0:
        return np.zeros((grid, grid, 2), np.float32)
    try:
        import cv2
        flow = cv2.calcOpticalFlowFarneback(
            a.astype(np.uint8), b.astype(np.uint8), None,
            0.5, 3, 21, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        bh, bw = max(h // grid, 1), max(w // grid, 1)
        flow = flow[:bh * grid, :bw * grid]
        out = flow.reshape(grid, bh, grid, bw, 2).mean(axis=(1, 3))
        out[..., 0] /= max(w, 1)
        out[..., 1] /= max(h, 1)
        return out.astype(np.float32)
    except ImportError:
        return _flow_fft(a, b, grid)


def _curl_div(flow: np.ndarray) -> tuple[float, float]:
    """Summary statistics that name the kind of motion.

    Curl is large for rotation (head roll), divergence for scaling, and the
    mean vector for translation (yaw, pitch). These three separate the head
    parameters that magnitude-only descriptors collapse together.
    """
    dx, dy = flow[..., 0], flow[..., 1]
    ddx_dy = np.gradient(dx, axis=0)
    ddy_dx = np.gradient(dy, axis=1)
    ddx_dx = np.gradient(dx, axis=1)
    ddy_dy = np.gradient(dy, axis=0)
    return float((ddy_dx - ddx_dy).mean()), float((ddx_dx + ddy_dy).mean())


def _moments(mask: np.ndarray):
    total = mask.sum()
    if total <= 0:
        return (0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
    ys, xs = np.nonzero(mask > mask.max() * 0.1)
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx = float((xx * mask).sum() / total / w)
    cy = float((yy * mask).sum() / total / h)
    if len(xs) == 0:
        return (cx, cy), (0.0, 0.0, 0.0, 0.0)
    bbox = (float(xs.min() / w), float(ys.min() / h),
            float(xs.max() / w), float(ys.max() / h))
    return (cx, cy), bbox


def probe_parameter(be: Live2DBackend, spec: ParamSpec, base: np.ndarray,
                    steps: int = 5, bbox=None) -> ParamDescriptor:
    d = ParamDescriptor(pid=spec.pid, index=spec.index, minimum=spec.minimum,
                        maximum=spec.maximum, default=spec.default)
    if spec.span == 0:
        d.kind = "dead"
        d.support = np.zeros((GRID, GRID), np.float32)
        return d

    ts = np.linspace(0.0, 1.0, steps)
    values = [spec.sample(float(t)) for t in ts]
    d.sweep = [float(v) for v in values]

    base_f = base.astype(np.float32) / 255.0
    accum = np.zeros(base.shape[:2], np.float32)
    response, alpha_share, centroids = [], [], []
    raw_lo = raw_hi = None

    for k, value in enumerate(values):
        raw = be.render({spec.pid: value})
        if k == 0:
            raw_lo = raw
        if k == len(values) - 1:
            raw_hi = raw
        img = raw.astype(np.float32) / 255.0
        diff = np.abs(img - base_f)
        rgb = diff[..., :3].sum(axis=2)
        alpha = diff[..., 3]
        mag = rgb + alpha
        accum = np.maximum(accum, mag)
        response.append(float(mag.mean()))
        denom = rgb.sum() + alpha.sum() + 1e-9
        alpha_share.append(float(alpha.sum() / denom))
        c, _ = _moments(mag)
        centroids.append(c)

    d.response = response
    peak = max(response)
    if peak <= 0:
        d.kind = "dead"
        d.support = np.zeros((GRID, GRID), np.float32)
        return d

    mask = accum / (accum.max() + 1e-9)
    # Spatial fields live in character-relative coordinates so they are
    # comparable across rigs with different framing.
    if bbox is not None:
        norm = _crop_resize(mask, bbox, GRID * 4)
        d.support = _downsample(norm, GRID)
        d.area = float((norm > 0.05).mean())
        d.centroid, d.bbox = _moments(norm)
    else:
        d.support = _downsample(mask)
        d.area = float((mask > 0.05).mean())
        d.centroid, d.bbox = _moments(mask)
    d.alpha_ratio = float(np.mean(alpha_share))
    if bbox is not None:
        # How much room the parameter actually has on each side of its default.
        # ParamEyeLOpen is [0, 1.2] with default 1.0: driving it to max moves
        # almost nothing, and all the signal is toward min. A flow field
        # measured across that dead 17% is numerical noise, and unit-normalising
        # it later would amplify that noise to full magnitude. Zero the dead
        # side explicitly instead, and scale both sides together so their
        # relative magnitude survives.
        span = spec.span + 1e-9
        d.travel_lo = float(abs(spec.default - spec.minimum) / span)
        d.travel_hi = float(abs(spec.maximum - spec.default) / span)

        lo = (_flow_field(base, raw_lo, bbox) if d.travel_lo >= 0.05
              else np.zeros((SGRID, SGRID, 2), np.float32))
        hi = (_flow_field(base, raw_hi, bbox) if d.travel_hi >= 0.05
              else np.zeros((SGRID, SGRID, 2), np.float32))

        scale = max(float(np.abs(lo).max()), float(np.abs(hi).max()))
        if scale > 0:
            lo, hi = lo / scale, hi / scale
        d.flow_lo, d.flow_hi = lo.astype(np.float32), hi.astype(np.float32)

        # Curl/divergence from whichever side actually travels.
        d.curl, d.divergence = _curl_div(hi if d.travel_hi >= d.travel_lo else lo)

    r = np.asarray(response, np.float32)
    if r.std() > 1e-9:
        d.linearity = float(abs(np.corrcoef(r, ts)[0, 1]))
    diffs = np.abs(np.diff(r))
    d.stepiness = float(diffs.max() / (diffs.sum() + 1e-9))

    lo, hi = centroids[0], centroids[-1]
    d.flow = (float(hi[0] - lo[0]), float(hi[1] - lo[1]))

    if d.area < DEAD_AREA:
        d.kind = "dead"
    elif d.stepiness > STEP_RATIO or d.alpha_ratio > ALPHA_DOMINANT:
        d.kind = "categorical"
    else:
        d.kind = "continuous"
    return d


def find_mirror_pairs(descs: list[ParamDescriptor], iou_min: float = 0.55) -> None:
    """Match L/R parameter pairs by horizontally flipped support overlap.

    Fills `mirror_of` in place. Purely geometric, so it works on rigs whose
    naming gives no hint about laterality.
    """
    live = [d for d in descs if d.kind != "dead" and d.support is not None]
    used: set[str] = set()
    for i, a in enumerate(live):
        if a.pid in used or a.mirror_of:
            continue
        flipped = np.fliplr(a.support)
        best, best_score = None, 0.0
        for b in live[i + 1:]:
            if b.pid in used:
                continue
            inter = np.minimum(flipped, b.support).sum()
            union = np.maximum(flipped, b.support).sum() + 1e-9
            score = float(inter / union)
            # a param is not its own mirror unless it straddles the centre line
            if score > best_score and abs(a.centroid[0] - b.centroid[0]) > 0.03:
                best, best_score = b, score
        if best is not None and best_score >= iou_min:
            a.mirror_of, best.mirror_of = best.pid, a.pid
            used.update({a.pid, best.pid})


def probe_rig(model_json: str, size: int = 512, steps: int = 5,
              strict: bool = True, **backend_kwargs,
              ) -> tuple[list[ParamDescriptor], dict]:
    """Probe one rig.

    Any extra keyword (egl, osmesa, bg, ...) is forwarded straight to
    Live2DBackend, so adding a new context backend never requires editing
    this function or run_probe's call site.
    """
    with Live2DBackend(model_json, size=size, **backend_kwargs) as be:
        be.calibrate()
        ok, worst = be.check_determinism()
        if not ok and strict:
            raise RuntimeError(
                f"renders are not deterministic (max pixel delta {worst}). "
                "Something time-varying is still active: physics, auto-breath, "
                "auto-blink, or an idle motion. Fix that before probing, or "
                "pass strict=False to proceed anyway."
            )

        base = be.render()
        bbox = content_bbox(base)
        descs = [probe_parameter(be, s, base, steps=steps, bbox=bbox)
                 for s in be.params]
        meta = {
            "model": model_json,
            "size": size,
            "steps": steps,
            "deterministic": ok,
            "worst_pixel_delta": worst,
            "api": dict(be._api_report),
            "n_params": len(descs),
            "content_bbox": [int(v) for v in bbox],
        }

    find_mirror_pairs(descs)
    meta["n_dead"] = sum(d.kind == "dead" for d in descs)
    meta["n_categorical"] = sum(d.kind == "categorical" for d in descs)
    meta["n_continuous"] = sum(d.kind == "continuous" for d in descs)
    meta["n_mirrored"] = sum(d.mirror_of is not None for d in descs)
    return descs, meta


def save(descs: list[ParamDescriptor], meta: dict, out_prefix: str) -> None:
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "params": [d.to_json() for d in descs]},
                  fh, indent=2, ensure_ascii=False)
    def stack(attr, shape):
        return np.stack([
            getattr(d, attr) if getattr(d, attr) is not None
            else np.zeros(shape, np.float32) for d in descs
        ])

    np.savez_compressed(
        f"{out_prefix}.npz",
        support=stack("support", (GRID, GRID)),
        flow_lo=stack("flow_lo", (SGRID, SGRID, 2)),
        flow_hi=stack("flow_hi", (SGRID, SGRID, 2)),
        pids=np.array([d.pid for d in descs], dtype=object))
