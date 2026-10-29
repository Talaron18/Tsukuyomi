#!/usr/bin/env python3
"""
Speech-driven micro-expressions: small, short face changes placed from the
spoken audio, between the model's own expression beats.

A streamer's face does more than the script says: the brows twitch up on a
stressed word, a smile eases in as a sentence lands, the eyes blink in a
pause. The teacher's beats (flow.pose) are the big changes, about one every
3 s of speech; this layer adds the small ones from what the TTS audio shows:

  brow      a lift of 0.2-0.3 of the brow's range (0.25-0.4 s in all) at a
            stress peak. Stress is the loudness envelope (RMS in dB over
            40 ms, smoothed over 120 ms, against the sentence's own median)
            plus, when flow.beat2.prosody finds a pitch (YIN), the pitch's
            rise over its local median in semitones; local maxima over
            +-0.2 s above a threshold, the strongest first, at most
            MAX_RATE a second of voiced speech.
  smile     0.25-0.35 of the range of the eyes' smile (else the mouth's
            form, up), easing in over 0.35 s so that it peaks as the
            sentence ends and fading over 0.4 s. Not at a question (a final
            pitch rise of 2 semitones or more).
  blink     an ordinary blink token at a pause of 0.35 s or more inside the
            speech and at the end of the audio, 2.5 s apart at the closest
            (the stream's Autonomic moves its next blink to the request).

Rules. Seeded and deterministic (a pure function of the audio, the seed, a
key and the model's markers). MIN_SPACING between two face events (brow or
smile), also across sentences (the state of a Micro). A face event is
dropped when a model beat on the same variables is within SUPPRESS seconds
of it (before its onset, or after its hold): the model's beats win. `scale`
multiplies the amplitudes (0 is off).

Playing. A micro event is a token {"type": "micro", "params": {parameter:
offset}, "rise", "hold", "fall"}: parameter offsets from the body model (the
canonical variable solved once at full range, then scaled), face parameters
only, a variable the body lacks skipped. MicroLayer adds the eased offset to
the frame before the moves and poses are applied (as flow.stream.HeadDrift
adds the head's), so a pose that targets the parameter takes it over.
Tokens, not state, so a --tokens-in replay and flow.stream play the same.

Not used for the dataset's trajectories or the action head's data: only
flow.director's render and flow.live build a Micro.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

try:
    from .behaviors import min_jerk
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow.behaviors import min_jerk

MIN_SPACING = 0.8         # s between two face events (onsets)
SUPPRESS = 0.6            # s around a model beat on the same variables
BLINK_SPACING = 2.5       # s between two micro blinks
BLINK_PAUSE = 0.35        # s of silence inside the speech that brings a blink
MAX_RATE = 0.7            # brow lifts a second of voiced speech, at most
HOP = 0.02                # s, the feature frame
BROW_FRAC = (0.20, 0.30)  # of the brow's range
SMILE_FRAC = (0.25, 0.35)
BROW_T = ((0.08, 0.12), (0.04, 0.08), (0.13, 0.20))      # rise, hold, fall
SMILE_T = (0.35, 0.50, 0.40)
PEAK_Z = 0.7              # stress score a peak must clear
QUESTION_ST = 2.0         # semitones of final rise that make a question
GROUPS = {"brow": ("brow.",), "smile": ("eye.", "mouth.", "cheek.")}


# --------------------------------------------------------------- features

def _smooth(x, n):
    if n <= 1 or not len(x):
        return x
    k = np.ones(n) / n
    pad = np.concatenate([np.full(n // 2, x[0]), x, np.full(n // 2, x[-1])])
    return np.convolve(pad, k, "valid")[:len(x)]


def features(x, sr, hop=HOP):
    """Frame features of a sentence's audio at `hop` seconds: {t, db (dB,
    smoothed), voiced (bool), st (semitones of f0 over its local median, 0
    where unvoiced or without pitch), f0 (Hz or 0), has_f0}."""
    x = np.asarray(x, np.float64)
    n = int(len(x) / sr / hop)
    t = np.arange(n) * hop
    if n < 3:
        return {"t": t, "db": np.zeros(n), "voiced": np.zeros(n, bool),
                "st": np.zeros(n), "f0": np.zeros(n), "has_f0": False}
    h = max(int(0.02 * sr), 1)
    c = np.concatenate([[0.0], np.cumsum(x ** 2)])
    i = np.clip((t * sr).astype(int), 0, len(x))
    a, b = np.clip(i - h, 0, len(x)), np.clip(i + h, 0, len(x))
    ms = (c[b] - c[a]) / np.maximum(b - a, 1)
    db = 10 * np.log10(ms + 1e-12)
    ref = float(np.percentile(db, 95))
    voiced = db > max(ref - 30.0, -75.0)
    f0 = np.zeros(n)
    has = False
    try:
        try:
            from .beat2 import prosody
        except ImportError:
            from flow.beat2 import prosody
        p = prosody(x, sr, 1.0 / hop, n)
        f0 = np.where(p[:, 2] > 0.5, p[:, 1], 0.0)
        has = int((f0 > 0).sum()) >= 8
    except Exception:                       # no pitch: loudness alone
        f0 = np.zeros(n)
    st = np.zeros(n)
    if has:
        v = f0 > 0
        semi = np.zeros(n)
        semi[v] = 12 * np.log2(f0[v] / 100.0)
        # Local median over ~1 s, voiced frames only, for the rise.
        w = int(0.5 / hop)
        for j in np.flatnonzero(v):
            seg = semi[max(j - w, 0):j + w + 1]
            seg = seg[seg != 0]
            st[j] = semi[j] - (np.median(seg) if len(seg) else semi[j])
        st = _smooth(st, max(int(0.12 / hop), 1))
    return {"t": t, "db": _smooth(db, max(int(0.12 / hop), 1)),
            "voiced": voiced, "st": st, "f0": f0, "has_f0": has}


def stress_peaks(F, rate=MAX_RATE):
    """[(time, score)] of the stress peaks of a sentence, strongest first
    (at most `rate` a second of voiced speech, and one)."""
    t, v = F["t"], F["voiced"]
    if v.sum() < 8:
        return []
    db = F["db"]
    z = (db - np.median(db[v])) / 6.0
    score = z + (0.7 * F["st"] / 3.0 if F["has_f0"] else 0.0)
    score = np.where(v, score, -9.0)
    w = int(0.2 / HOP)
    out = []
    for j in range(len(t)):
        lo, hi = max(j - w, 0), min(j + w + 1, len(t))
        if score[j] >= PEAK_Z and score[j] >= score[lo:hi].max() - 1e-12:
            if not out or t[j] - out[-1][0] > 0.2:
                out.append((float(t[j]), float(score[j])))
    out.sort(key=lambda p: -p[1])
    k = max(1, int(round(rate * v.sum() * HOP)))
    return out[:k]


def pauses(F, min_len=BLINK_PAUSE):
    """[start] of the silences of min_len or more with speech on both sides."""
    v, t = F["voiced"], F["t"]
    idx = np.flatnonzero(v)
    if not len(idx):
        return []
    out, j, first, last = [], 0, idx[0], idx[-1]
    n = int(round(min_len / HOP))
    j = first
    while j < last:
        if v[j]:
            j += 1
            continue
        k = j
        while k < last and not v[k]:
            k += 1
        if k - j >= n:
            out.append(float(t[j]))
        j = k
    return out


def voiced_end(F):
    idx = np.flatnonzero(F["voiced"])
    return float(F["t"][idx[-1]]) + HOP if len(idx) else 0.0


def final_rise(F, end, win=0.3):
    """Semitones the pitch gains over the last `win` seconds before `end`
    (0 without pitch)."""
    if not F["has_f0"]:
        return 0.0
    t, f0 = F["t"], F["f0"]
    m = (t >= end - win) & (t <= end) & (f0 > 0)
    idx = np.flatnonzero(m)
    if len(idx) < 4:
        return 0.0
    h = len(idx) // 2
    a, b = np.median(f0[idx[:h]]), np.median(f0[idx[h:]])
    return float(12 * math.log2(b / a)) if a > 0 and b > 0 else 0.0


# ----------------------------------------------------------------- layer

class MicroLayer:
    """The micro tokens played a frame at a time: apply(tf, over, bounds,
    defaults) adds each live event's eased offset to the parameters. The
    offset rises over `rise` (minimum jerk), holds, falls over `fall`."""

    def __init__(self):
        self.events = []
        self.written = set()

    def commit(self, tok):
        self.events.append(tok)

    def busy(self):
        return bool(self.events)

    @staticmethod
    def envelope(tok, tf):
        t0 = float(tok["t"])
        r, h, f = float(tok["rise"]), float(tok["hold"]), float(tok["fall"])
        if tf < t0 or tf >= t0 + r + h + f:
            return 0.0
        if tf < t0 + r:
            return float(min_jerk((tf - t0) / max(r, 1e-6)))
        if tf < t0 + r + h:
            return 1.0
        return 1.0 - float(min_jerk((tf - t0 - r - h) / max(f, 1e-6)))

    def apply(self, tf, over, bounds=None, defaults=None):
        """over with the offsets added (bounds {pid: (lo, hi)} clip them;
        defaults {pid: value} stand in for a parameter over lacks)."""
        self.written = set()
        if not self.events:
            return over
        self.events = [k for k in self.events
                       if tf < float(k["t"]) + float(k["rise"])
                       + float(k["hold"]) + float(k["fall"])]
        add = {}
        for k in self.events:
            e = self.envelope(k, tf)
            if e <= 0.0:
                continue
            for p, d in k["params"].items():
                add[p] = add.get(p, 0.0) + e * d
        if not add:
            return over
        over = dict(over)
        for p, d in add.items():
            base = over.get(p, (defaults or {}).get(p))
            if base is None:
                continue
            v = base + d
            if bounds and p in bounds:
                v = min(max(v, bounds[p][0]), bounds[p][1])
            over[p] = float(v)
            self.written.add(p)
        return over


def layer_for(tokens):
    """A MicroLayer with the micro tokens among `tokens`, or None."""
    toks = [k for k in tokens if k.get("type") == "micro"]
    if not toks:
        return None
    lay = MicroLayer()
    for k in sorted(toks, key=lambda k: float(k["t"])):
        lay.commit(dict(k))
    return lay


def bounds_of(pids, lo, hi):
    return {p: (float(a), float(b)) for p, a, b in zip(pids, lo, hi)}


# --------------------------------------------------------------- planner

def sentence_ends(spoken, spans):
    """[(end second, is_question)] of the sentences of `spoken`, from the
    per-character spans (speech.speak)."""
    out = []
    end_marks = ".!?。！？…"
    n = len(spoken)
    for i, ch in enumerate(spoken):
        if ch in end_marks and i < len(spans) \
                and (i + 1 >= n or spoken[i + 1] not in end_marks):
            out.append((float(spans[i][1]), ch in "?？"))
    return out


class Micro:
    """The micro-expression planner of one body. plan() returns the tokens
    of a sentence's (or reply's) audio; the Micro keeps the spacing across
    calls (the last face event, the last blink, the model beats' windows),
    so a live stream's sentences join up. Not thread-safe: one per stream."""

    def __init__(self, bm, seed=0, scale=1.0, log=print):
        self.bm, self.seed, self.scale = bm, int(seed), float(scale)
        self.log = log
        self.unit = {}
        self.last_face = -1e9
        self.last_blink = -1e9
        self.windows = []                   # (group, a, b) absolute
        self.totals = {"brow": 0, "smile": 0, "blink": 0, "suppressed": 0}
        if self.scale > 0:
            self.unit = {g: self._unit(g) for g in ("brow", "smile")}

    # ---- the body's parameter offsets for a group at its full range
    def _unit(self, group):
        bm = self.bm
        have = set(self._offered())
        if group == "brow":
            want = [v for v in ("brow.L.lift", "brow.R.lift") if v in have]
        else:
            want = [v for v in ("eye.L.smile", "eye.R.smile") if v in have]
            if "mouth.form" in have and not want:
                want = ["mouth.form"]
            elif "mouth.form" in have:
                want.append("mouth.form")
        targets = {}
        for v in want:
            lim = bm.limits(v, 0)
            if lim is None:
                continue
            rest = float(bm.variants[0]["rest"][bm.fidx[v]])
            hi = float(lim[1])
            if hi - rest > 0.05:
                targets[v] = hi
        if not targets:
            return {}
        try:
            from . import pose as pz
            from . import runtime as rt
        except ImportError:
            from flow import pose as pz
            from flow import runtime as rt
        res = bm.solve(targets, current=None)
        if res.get("variant") is None:
            return {}
        base = bm.default_values()
        arms = pz.arm_params(bm)
        out = {}
        for p, x in res["params"].items():
            e = bm.params[p]
            if e.get("role") in ("switch", "variant") or p in arms \
                    or rt.voice_owns(p):
                continue
            d = float(x) - float(base.get(p, e["default"]))
            if abs(d) > 1e-6:
                out[p] = d
        return out

    def _offered(self):
        try:
            from . import pose as pz
        except ImportError:
            from flow import pose as pz
        return pz.offered(self.bm)

    # ---- the model's beats: where they touch the same variables
    def _model_windows(self, acts, t0):
        bm = self.bm
        out = []
        for x in acts or ():
            if x.get("tag") != "pose" or x.get("micro"):
                continue
            spec = x.get("arg") or {}
            try:
                keys = list(bm.resolve(dict(spec.get("targets") or {})))
            except (ValueError, KeyError):
                keys = list(spec.get("targets") or {})
            dur = float(spec.get("dur") or 0.5)
            hold = float(spec["hold"]) if spec.get("hold") is not None \
                else 1.0
            a = t0 + float(x["t"])
            for g, prefixes in GROUPS.items():
                if any(k.startswith(prefixes) for k in keys):
                    out.append((g, a - SUPPRESS, a + dur + hold + SUPPRESS))
        return out

    def plan(self, audio, sr, acts=(), t0=0.0, key=0, ends=None):
        """Micro tokens for audio that starts at stream time t0. acts: the
        sentence's actions (local times), whose pose markers keep the
        micro events off the same variables. ends: [(seconds, question)]
        sentence ends; default: where the audio's voice ends."""
        if self.scale <= 0 or audio is None or not len(audio):
            return []
        F = features(audio, sr)
        if not F["voiced"].any():
            return []
        rng = np.random.default_rng([self.seed, 5, int(key)])
        self.windows += self._model_windows(acts, t0)
        self.windows = [w for w in self.windows if w[2] > t0 - 5.0]
        cands = []
        if self.unit.get("smile"):
            for e, q in (ends if ends is not None
                         else [(voiced_end(F), False)]):
                q = q or final_rise(F, e) >= QUESTION_ST
                if q or e < 0.6:
                    continue
                cands.append((10.0, "smile", max(e - SMILE_T[0], 0.0)))
        if self.unit.get("brow"):
            for tp, sc in stress_peaks(F):
                cands.append((sc, "brow", max(tp - 0.08, 0.0)))
        toks, n_sup = [], 0
        # Strongest first, then back in time order; one pass keeps it
        # independent of the order the features were found in.
        cands.sort(key=lambda c: (-c[0], c[2]))
        taken = []
        for _pri, kind, tl in cands:
            t = t0 + tl
            if any(abs(t - u) < MIN_SPACING for u in taken) \
                    or abs(t - self.last_face) < MIN_SPACING:
                continue
            if any(g == kind and a <= t <= b for g, a, b in self.windows):
                n_sup += 1
                continue
            taken.append(t)
            toks.append(self._token(kind, t, rng))
        if taken:
            self.last_face = max(self.last_face, max(taken))
        # Blinks at the pauses inside the speech and at its end.
        blinks = [t0 + p + 0.1 for p in pauses(F)] + [t0 + voiced_end(F)
                                                      + 0.1]
        nb = 0
        for tb in sorted(blinks):
            if tb - self.last_blink >= BLINK_SPACING:
                toks.append({"t": round(tb, 4), "type": "blink",
                             "role": "eyes", "src": "director",
                             "micro": True})
                self.last_blink = tb
                nb += 1
        toks.sort(key=lambda k: (float(k["t"]), k["type"]))
        nbrow = sum(1 for k in toks if k.get("kind") == "brow")
        nsm = sum(1 for k in toks if k.get("kind") == "smile")
        self.totals["brow"] += nbrow
        self.totals["smile"] += nsm
        self.totals["blink"] += nb
        self.totals["suppressed"] += n_sup
        dur = float(F["voiced"].sum() * HOP)
        self.log(f"micro: {len(toks)} events ({nbrow} brow, {nsm} smile, "
                 f"{nb} blink) over {dur:.1f}s of voice"
                 + (f"; {n_sup} dropped for a model beat nearby"
                    if n_sup else "")
                 + (", pitch + loudness" if F["has_f0"]
                    else ", loudness"))
        return toks

    def _token(self, kind, t, rng):
        if kind == "brow":
            frac = rng.uniform(*BROW_FRAC)
            rise, hold, fall = (rng.uniform(*r) for r in BROW_T)
        else:
            frac = rng.uniform(*SMILE_FRAC)
            rise, hold, fall = SMILE_T
        frac = min(frac * self.scale, 1.0)
        return {"t": round(float(t), 4), "type": "micro", "role": "micro",
                "kind": kind, "src": "director",
                "params": {p: round(d * frac, 5)
                           for p, d in self.unit[kind].items()},
                "rise": round(float(rise), 4), "hold": round(float(hold), 4),
                "fall": round(float(fall), 4)}

    def summary(self):
        t = self.totals
        return (f"micro: {t['brow']} brow, {t['smile']} smile, {t['blink']} "
                f"blink events placed" + (f", {t['suppressed']} dropped for "
                                          "a model beat nearby"
                                          if t["suppressed"] else ""))


def add_args(ap):
    """--micro / --no-micro / --micro-scale (the director's parser, which
    flow.live shares)."""
    import argparse
    ap.add_argument("--micro", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="speech-driven micro-expressions (flow.micro): "
                         "small brow lifts on stressed syllables, a smile "
                         "easing in at sentence ends, blinks at pauses, "
                         "between the model's own beats; body mode with "
                         "--tts [default on]; --no-micro: only the model's "
                         "beats")
    ap.add_argument("--micro-scale", type=float, default=1.0, metavar="X",
                    help="the micro-expressions' amplitude, x their "
                         "defaults (0.2-0.3 of a range) [1]; 0 is off")


def planner_for(a, bm, log=print):
    """A Micro for parsed arguments, or None (off, or no body)."""
    if bm is None or not getattr(a, "micro", False):
        return None
    s = float(getattr(a, "micro_scale", 1.0))
    if s <= 0:
        return None
    return Micro(bm, getattr(a, "seed", 0), s, log)
