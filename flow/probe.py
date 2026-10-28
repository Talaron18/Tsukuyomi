#!/usr/bin/env python3
"""
Probes of why the action head does not beat a constant on BEAT2
(docs/embodiment.md, "Action head ablations on BEAT2"): no controller, only
questions put to the data, each with a control that would show an answer
that is really speaker or recording identity.

  data      the export flow.beat2 writes (<stem>.jsonl, <stem>.npz: per
            segment the canonical trajectory `traj`, `chars` the spoken
            characters' times, `prosody` [F, 3] rms, f0 Hz, voiced at the
            frame rate; the row's clip, speaker, lang, t0) and, for match,
            flow.train.hidden's shards (token states and spans, read as
            flow.actionhead.Data reads them: a token's time is its first
            spoken character's start, flow.actionhead.token_times).
            Columns used (those the body has): hands hand.L/R.x/y/z (hu),
            head head.yaw/pitch/tilt (deg), torso torso.yaw/pitch/tilt (deg).
            --limit N keeps whole clips until N segments are in (a quick
            run still has several segments per clip).
  prosody   Does the voice predict the motion in the ground truth (no
            model)? Per segment:
              voice   loud      mean log RMS over voiced frames
                      loud_rng  its range, p90 - p10
                      voiced    the voiced fraction of frames
                      f0        mean f0 (Hz) over voiced frames
                      f0_rng    its range, p90 - p10
                      rate      spoken units per voiced second (units:
                                CJK / kana / hangul characters for zh and
                                ja, plus any other space-separated token;
                                the space-separated words for en, es;
                                needs --min-voiced s of voiced speech)
                      rate_span units per second of the speech span (first
                                character's start to last one's end)
                      pauses    count and total seconds of silences longer
                                than --pause s between spoken characters
                                (a separator's own span and a gap between
                                words)
              motion  hand_std  amplitude: the std of the hand positions
                                (hu) around the segment mean, the rms over
                                columns; hand_rng: p95 - p5 (mean over
                                columns); head_std, head_rng: the same of
                                the head angles (deg)
                      hand_v, hand_v90  mean and p90 speed of the hands
                                (hu/s: the norm of one hand's position
                                step, averaged over the hands); head_v,
                                head_v90 (deg/s)
                      hand_act  the fraction of frames with the hands
                                faster than --active-speed hu/s
            Windowed (--win s windows, --wstep s hop): the window's
            loudness against the window's amplitude (std around the
            window mean) and mean speed, the motion window shifted by lags
            -1 .. +1 s (positive: the motion follows the audio; windows
            whose shifted twin leaves the segment are left out). Frame
            level: the accent (|d rms| smoothed, as actionhead's beat
            alignment) against the hands' and head's speed: the per-segment
            correlation of the eval (flow.actionhead.beat_stats, the
            pooled-std speed, mean over segments: the 0.03 of the grid) and
            the pooled one in natural units. Every number is Pearson and
            Spearman with a 95% bootstrap CI that resamples whole clips (a
            segment's neighbours, and a window's, are not independent):
              overall          as it is (speaker style drives it)
              within speaker   x and y z-scored per speaker (ranks
                               first for Spearman), then pooled: the
                               answer that matters
              within, no dur   within speaker, with the log duration
                               regressed out of both (a longer segment
                               has more spread by itself)
              by language      within speaker, per language
            and the paired difference of two correlations (same resamples)
            for the two hypotheses: loudness moves the amplitude more than
            the speed; the speaking rate moves the speed more than the
            amplitude. Amplitude and speed are not independent measures
            (a faster oscillation of the same size is faster): read the
            difference, and the tables of all features.
  match     Does the text, or the prosody, carry information about the
            motion? A contrastive probe. Windows of --window frames (90: 3
            s) every --hop frames of the segments; per window the motion
            (the canonical columns and their frame differences, normalised
            on the training frames; --center subtracts the window mean,
            leaving the gesture without the posture), the tokens whose time
            is in [start - --lookback s, end] (with their time relative
            to the start; a learned null token is always there) and the
            prosody (flow.actionhead.pros_feat). Encoders to --dim: a
            small 1-D conv for the motion and the prosody; for the text a
            LayerNorm per stored layer, a learned softmax mix of the layers
            (--layer: an index into the stored layers, a layer number from
            meta.json, or all), a projection and time features, and
            --queries attention queries pooling the tokens. text+prosody:
            the two embeddings through one linear layer. Symmetric InfoNCE
            between the motion and the conditioning embedding, learned
            temperature. Variants, each trained separately for --steps x
            --batch (the same budget, the same seed):
              text, prosody, text+prosody   batches of one speaker's
                                windows (the negatives share the speaker)
              text_shuffled   the tokens of another window, a fixed
                              permutation over the windows of a split
                              (train and eval alike): the pipeline's
                              control, chance whatever the model
              text_random     batches of random windows: speaker and
                              language are enough to match, the leak
                              the speaker batches remove
              text_clip       batches from one recording, windows that do
                              not overlap in time: the hardest
            Held out: whole clips (--split clip, the same hash split as
            flow.actionhead.split_ids on clip ids; --eval-clips picks
            them), or whole speakers (--split speaker). Every variant is
            scored under all three ways of drawing the K - 1 negatives
            (--k, default 32), query = the conditioning of a held-out
            window, candidates = motions:
              random    any window of another clip
              speaker   a window of the same speaker, another clip
              clip      a window of the same clip not overlapping it (the
                        speaker, the recording and the language are all
                        the same: what is left is what is said and heard
                        against what is done)
            as top-1 / top-5 retrieval with a clip-bootstrap CI against
            chance 1/K and 5/K; a query with fewer than K - 1 candidates
            is skipped (counted). The curves (loss, top-1 on a training
            subset and on the held-out windows, under the training
            scheme) are in the JSON and the report, so that overfitting
            shows. The reported model is the last step (none is selected
            on the held-out windows).
  sem       --sem DIR: BEAT2's semantic annotations, found as
            DIR/**/<clip>.(txt|csv|tsv|json); the format is not known
            here, so each line is read as numbers and words: the first
            two numbers are start and end (clip seconds); a word that is
            not beat / none / neutral / idle marks a semantic gesture,
            else a third number above 0 does, else any interval does. The
            match accuracy is then reported apart for the windows that
            overlap such an interval and those that do not. Nothing found
            or parsed: said, and skipped.
  output    <out>/report.txt and <out>/report.json (prosody and match
            under their names; `all` runs both).

Usage:
    python -m flow.probe prosody --export data/beat2/seed-san/action \\
        --out ~/tsk-runs/probe
    python -m flow.probe match --export data/beat2/seed-san/action \\
        --hidden ~/tsk-runs/hidden/beat2 --out ~/tsk-runs/probe --layer 1
    python -m flow.probe all --export ... --hidden ... --out ... --limit 2000
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata

import numpy as np

try:
    from . import actionhead as ah
    from .train import hidden as fh
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import actionhead as ah
    from flow.train import hidden as fh

HANDS = [f"hand.{s}.{a}" for s in "LR" for a in "xyz"]
HEAD = [f"head.{a}" for a in ("yaw", "pitch", "tilt")]
TORSO = [f"torso.{a}" for a in ("yaw", "pitch", "tilt")]
CANON = HANDS + HEAD + TORSO
FPS = 30.0

FEATURES = ["loud", "loud_rng", "voiced", "f0", "f0_rng", "rate", "rate_span",
            "pauses", "pause_s"]
AMPLITUDE = ["hand_std", "hand_rng", "head_std", "head_rng"]
SPEED = ["hand_v", "hand_v90", "head_v", "head_v90"]
TARGETS = AMPLITUDE + SPEED + ["hand_act"]


def log(s):
    print(s, flush=True)


# ---------------------------------------------------------------- data

def _cjk(ch):
    return unicodedata.name(ch, "").startswith(("CJK", "HIRAGANA", "KATAKANA",
                                                 "HANGUL"))


def lang_of(r):
    v = r.get("lang")
    if isinstance(v, dict):
        v = v.get("reply") or v.get("asked")
    return str(v or "?")


def clip_of(r):
    return str(r.get("clip") or r["id"].rsplit("/", 1)[0])


def speaker_name(r):
    return str(r.get("speaker") or r.get("avatar") or "?")


def read_rows(export, limit=0, seed=0, hidden=None):
    """(rows, npz path): the export's rows (only those with hidden states
    when `hidden` is a folder's index), cut to whole clips holding about
    `limit` segments (clips in a seeded random order)."""
    rows, npz = fh.read_export(export)
    if hidden is not None:
        idx = fh.load_index(hidden)
        rows = [r for r in rows if r["id"] in idx]
    if limit and len(rows) > limit:
        by = {}
        for r in rows:
            by.setdefault(clip_of(r), []).append(r)
        names = sorted(by)
        np.random.default_rng(seed).shuffle(names)
        keep, n = [], 0
        for c in names:
            keep += by[c]
            n += len(by[c])
            if n >= limit:
                break
        ks = {r["id"] for r in keep}
        rows = [r for r in rows if r["id"] in ks]
    return rows, npz


def columns(vars_, names):
    """{name: column} of the canonical names a row has."""
    pos = {v: i for i, v in enumerate(vars_)}
    return {n: pos[n] for n in names if n in pos}


def split_hash(name, frac, seed=0):
    h = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 2 ** 32 < frac


def split_names(names, frac, seed=0, force=()):
    """(train, eval) name lists by hash (flow.train.sft's rule); `force`
    names the eval ones instead. Both are non-empty if there are two
    names."""
    names = sorted(set(names))
    if force:
        ev = [n for n in names if n in set(force)]
    else:
        ev = [n for n in names if split_hash(n, frac, seed)]
        if not ev and len(names) > 1:
            ev = [names[-1]]
    tr = [n for n in names if n not in set(ev)]
    if not tr and len(ev) > 1:
        tr, ev = ev[:-1], ev[-1:]
    return tr, ev


# ------------------------------------------------------- correlations

def rank(x):
    """Average ranks (1-based) of x."""
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    c = np.cumsum(cnt)
    return (c - (cnt - 1) / 2.0)[inv]


def within(x, spk, ranks=False, min_n=5):
    """x z-scored per speaker id (ranks first with `ranks`); a speaker with
    fewer than min_n values, or no spread, gives NaN."""
    out = np.full(len(x), np.nan)
    order = np.argsort(spk, kind="stable")
    s = spk[order]
    cut = np.flatnonzero(np.diff(s)) + 1
    for g in np.split(order, cut):
        if len(g) < min_n:
            continue
        v = x[g].astype(np.float64)
        if ranks:
            v = rank(v)
        sd = v.std()
        if sd > 1e-12:
            out[g] = (v - v.mean()) / sd
    return out


def _cluster_stats(x, y, g, n):
    f = lambda v: np.bincount(g, v, n)
    return np.stack([np.bincount(g, minlength=n).astype(np.float64), f(x),
                     f(y), f(x * x), f(y * y), f(x * y)], 1)


def _pearson(S):
    n, sx, sy, sxx, syy, sxy = (S[..., i] for i in range(6))
    with np.errstate(all="ignore"):
        cov = sxy / n - sx * sy / n ** 2
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        return cov / np.sqrt(vx * vy)


def _counts(n_clusters, B, seed):
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_clusters, size=(B, n_clusters))
    c = np.zeros((B, n_clusters))
    for b in range(B):
        c[b] = np.bincount(pick[b], minlength=n_clusters)
    return c


def corr(x, y, spk, clip, mode="within", ctrl=None, B=1000, seed=0,
         universe=None):
    """Pearson and Spearman of x and y with 95% bootstrap CIs over clips.
    spk, clip: ids per value. mode raw: as it is (Spearman on global ranks);
    within: each z-scored per speaker (Spearman: per-speaker ranks), pooled.
    ctrl: a covariate (e.g. log duration) regressed out of both after the
    within-speaker z-scoring. universe: the clip ids all the corr calls
    share (their resamples are then the same: differences are paired).
    Returns {n, clips, r, r_lo, r_hi, rho, rho_lo, rho_hi, boot_r,
    boot_rho} (the last two arrays, dropped before the JSON)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ctrl is not None:
        ok &= np.isfinite(ctrl)
    out = {"n": int(ok.sum()), "clips": 0, "r": None, "r_lo": None,
           "r_hi": None, "rho": None, "rho_lo": None, "rho_hi": None}
    if ok.sum() < 10:
        return out
    x, y, spk, clip = x[ok], y[ok], np.asarray(spk)[ok], np.asarray(clip)[ok]
    c = None if ctrl is None else np.asarray(ctrl, np.float64)[ok]
    uni = np.unique(clip) if universe is None else np.asarray(universe)
    gi = np.searchsorted(uni, clip)
    out["clips"] = int(len(np.unique(clip)))
    counts = _counts(len(uni), B, seed)
    for name, rk in (("r", False), ("rho", True)):
        if mode == "within":
            a, b = within(x, spk, rk), within(y, spk, rk)
            cc = None if c is None else within(c, spk, rk)
            if cc is not None and np.isfinite(cc).sum() >= 10:
                m = np.isfinite(cc) & np.isfinite(a) & np.isfinite(b)
                d2 = max((cc[m] ** 2).sum(), 1e-12)
                a = np.where(m, a - (a * cc)[m].sum() / d2 * cc, np.nan)
                b = np.where(m, b - (b * cc)[m].sum() / d2 * cc, np.nan)
            keep = np.isfinite(a) & np.isfinite(b)
            a, b, g = a[keep], b[keep], gi[keep]
        else:
            a, b, g = (rank(x), rank(y), gi) if rk else (x, y, gi)
        if len(a) < 10:
            continue
        S = _cluster_stats(a, b, g, len(uni))
        r = float(_pearson(S.sum(0)))
        if not math.isfinite(r):                    # a constant
            continue
        bs = _pearson(counts @ S)
        bs = bs[np.isfinite(bs)]
        out[name] = r
        if len(bs) >= 10:
            out[name + "_lo"], out[name + "_hi"] = (
                float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
            out["boot_" + name] = _pearson(counts @ S)
    return out


def clean(d):
    """The corr result without its bootstrap arrays, floats rounded."""
    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in d.items() if not k.startswith("boot_")}


def diff_ci(a, b, key="boot_r", strength=False):
    """(difference, lo, hi) of two corr results' point estimates and their
    paired bootstrap (same resamples), or None. strength: of |r| (each
    signed as its point estimate is), the strength of the relation."""
    if key not in a or key not in b:
        return None
    k = key[5:]
    sa = sb = 1.0
    if strength:
        sa, sb = math.copysign(1, a[k]), math.copysign(1, b[k])
    bs = sa * a[key] - sb * b[key]
    bs = bs[np.isfinite(bs)]
    if len(bs) < 10:
        return None
    return (float(sa * a[k] - sb * b[k]), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


# ----------------------------------------------------- prosody features

def units_of(spoken, lang):
    """Spoken units (module docstring): CJK / kana / hangul characters plus
    other space-separated tokens for zh and ja, space-separated words
    otherwise."""
    if lang in ("zh", "ja"):
        n = sum(1 for ch in spoken if _cjk(ch))
        n += sum(1 for w in spoken.split()
                 if not any(_cjk(ch) for ch in w))
        return n
    return len(spoken.split())


def pauses(spoken, chars, min_s):
    """(count, seconds) of the silences longer than min_s between spoken
    characters: a separator character's span, and the gap between one
    character's end and the next one's start."""
    ch = np.asarray(chars, np.float64).reshape(-1, 2)
    if len(ch) < 2:
        return 0, 0.0
    gaps = list(np.maximum(ch[1:, 0] - ch[:-1, 1], 0))
    if len(spoken) == len(ch):
        gaps += [ch[i, 1] - ch[i, 0] for i, c in enumerate(spoken)
                 if c.isspace()]
    g = np.array([x for x in gaps if x > min_s])
    return len(g), float(g.sum())


def voice_features(pros, spoken, chars, lang, o):
    """{feature: value or NaN} of one segment (module docstring)."""
    nan = float("nan")
    out = {k: nan for k in FEATURES}
    if pros is None or len(pros) == 0:
        return out
    rms, f0, vo = pros[:, 0], pros[:, 1], pros[:, 2] > 0.5
    out["voiced"] = float(vo.mean())
    if vo.sum() >= 3:
        lr = np.log(rms[vo] + ah.PROS_RMS0)
        out["loud"] = float(lr.mean())
        out["loud_rng"] = float(np.percentile(lr, 90) - np.percentile(lr, 10))
        ff = f0[vo][f0[vo] > 0]
        if len(ff) >= 3:
            out["f0"] = float(ff.mean())
            out["f0_rng"] = float(np.percentile(ff, 90) - np.percentile(ff, 10))
    u = units_of(spoken, lang)
    vs = vo.sum() / FPS
    if vs >= o.min_voiced:
        out["rate"] = u / vs
    ch = np.asarray(chars, np.float64).reshape(-1, 2)
    if len(ch) and ch[-1, 1] - ch[0, 0] > 0.5:
        out["rate_span"] = u / (ch[-1, 1] - ch[0, 0])
    n, s = pauses(spoken, chars, o.pause)
    out["pauses"], out["pause_s"] = float(n), s
    return out


def side_groups(cols):
    """[[columns]] per hand with the axes the row has."""
    return [[cols[f"hand.{s}.{a}"] for a in "xyz" if f"hand.{s}.{a}" in cols]
            for s in "LR" if any(f"hand.{s}.{a}" in cols for a in "xyz")]


def speed_of(y, groups, fps):
    """[F-1] mean over groups of the norm of the step of the group's
    columns, per second."""
    if not groups:
        return None
    sp = [np.sqrt((np.diff(y[:, g], axis=0) ** 2).sum(1)) * fps
          for g in groups]
    return np.mean(sp, 0)


def motion_features(y, cols, fps, o):
    """({target: value}, hand speed [F-1] or None, head speed [F-1] or None)
    of a trajectory y [F, V]."""
    nan = float("nan")
    out = {k: nan for k in TARGETS}
    hc = [cols[n] for n in HANDS if n in cols]
    kc = [cols[n] for n in HEAD if n in cols]
    spd = {}
    for tag, cs, groups in (("hand", hc, side_groups(cols)),
                            ("head", kc, [kc] if kc else [])):
        if not cs or len(y) < 10:
            spd[tag] = None
            continue
        z = y[:, cs]
        if not np.isfinite(z).all():
            spd[tag] = None
            continue
        out[tag + "_std"] = float(np.sqrt((z.std(0) ** 2).mean()))
        out[tag + "_rng"] = float((np.percentile(z, 95, 0)
                                   - np.percentile(z, 5, 0)).mean())
        sp = speed_of(y, groups, fps)
        spd[tag] = sp
        out[tag + "_v"] = float(sp.mean())
        out[tag + "_v90"] = float(np.percentile(sp, 90))
        if tag == "hand":
            out["hand_act"] = float((sp > o.active_speed).mean())
    return out, spd["hand"], spd["head"]


def _cum(a):
    return np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, 0)], 0)


class WinSeg:
    """Cumulative sums of one segment, for window statistics at any start."""

    def __init__(self, y, cols, pros, fps, sp_hand, sp_head):
        self.F = len(y)
        hc = [cols[n] for n in HANDS if n in cols]
        kc = [cols[n] for n in HEAD if n in cols]
        self.ok = {}
        self.s1, self.s2 = {}, {}
        for tag, cs in (("hand", hc), ("head", kc)):
            if cs and np.isfinite(y[:, cs]).all():
                z = y[:, cs].astype(np.float64)
                self.s1[tag], self.s2[tag] = _cum(z), _cum(z * z)
        self.sp = {"hand": None if sp_hand is None else _cum(sp_hand),
                   "head": None if sp_head is None else _cum(sp_head)}
        vo = (pros[:, 2] > 0.5).astype(np.float64)
        lr = np.log(pros[:, 0] + ah.PROS_RMS0) * vo
        self.cv, self.cl = _cum(vo), _cum(lr)

    def loud(self, s, w):
        nv = self.cv[s + w] - self.cv[s]
        with np.errstate(all="ignore"):
            v = (self.cl[s + w] - self.cl[s]) / nv
        return np.where(nv >= 0.25 * w, v, np.nan)

    def amp(self, tag, s, w):
        if tag not in self.s1:
            return np.full(len(s), np.nan)
        m = (self.s1[tag][s + w] - self.s1[tag][s]) / w
        v = (self.s2[tag][s + w] - self.s2[tag][s]) / w - m * m
        return np.sqrt(np.maximum(v, 0).mean(1))

    def speed(self, tag, s, w):
        if self.sp[tag] is None:
            return np.full(len(s), np.nan)
        return (self.sp[tag][s + w - 1] - self.sp[tag][s]) / (w - 1)


WIN_TARGETS = ["hand_amp", "head_amp", "hand_speed", "head_speed"]


def analyse_segments(rows, npz_path, o, log=log):
    """The per-segment arrays and window arrays of the prosody probe."""
    z = np.load(npz_path)
    nrow = len(rows)
    # The eval-style speed divides each column by its pooled std: taken
    # over about 2000 segments.
    acc_ = {c: np.zeros(3) for c in CANON}
    for r in rows[::max(nrow // 2000, 1)]:
        k = r["key"] + ".traj"
        if k not in z.files:
            continue
        t = z[k].astype(np.float64)
        for c, j in columns(r["vars"], CANON).items():
            v = t[:, j][np.isfinite(t[:, j])]
            acc_[c] += (len(v), v.sum(), (v * v).sum())
    sd = {}
    for c, (n_, s1_, s2_) in acc_.items():
        v = max(s2_ / n_ - (s1_ / n_) ** 2, 0) if n_ else 0.0
        sd[c] = math.sqrt(v) if v > 1e-8 else 1.0
    lags = np.round(np.arange(-o.lag_max, o.lag_max + 1e-9, o.lag_step), 4) + 0.0
    W, H = int(round(o.win * FPS)), int(round(o.wstep * FPS))
    seg = {k: [] for k in ("speaker", "clip", "lang", "dur")}
    segf = {k: [] for k in FEATURES + TARGETS}
    win = {"speaker": [], "clip": [], "lang": [], "loud": [],
           **{t: [] for t in WIN_TARGETS}}
    fr = {"speaker": [], "clip": [], "acc": [], "hand": [], "head": []}
    beat = {"hands": [], "head": [], "spk": [], "clip": []}
    skip = int(round(o.skip * FPS))
    used = 0
    spk_id, clip_id, lang_id = {}, {}, {}
    gid = lambda d, k: d.setdefault(k, len(d))
    for i, r in enumerate(rows):
        k = r["key"]
        if k + ".traj" not in z.files or k + ".prosody" not in z.files:
            continue
        y = z[k + ".traj"].astype(np.float32)
        pros = z[k + ".prosody"].astype(np.float32)
        if len(pros) != len(y) or len(y) < 30:
            continue
        cols = columns(r["vars"], CANON)
        fps = float(r.get("fps") or FPS)
        lang = lang_of(r)
        chars = z[k + ".chars"] if k + ".chars" in z.files else np.zeros((0, 2))
        sp_, cl_, lg_ = gid(spk_id, speaker_name(r)), gid(
            clip_id, clip_of(r)), gid(lang_id, lang)
        vf = voice_features(pros, r.get("spoken", ""), chars, lang, o)
        mf, sph, spk_ = motion_features(y, cols, fps, o)
        if sph is None and spk_ is None:
            continue
        used += 1
        seg["speaker"].append(sp_)
        seg["clip"].append(cl_)
        seg["lang"].append(lg_)
        seg["dur"].append(len(y) / fps)
        for q in FEATURES:
            segf[q].append(vf[q])
        for q in TARGETS:
            segf[q].append(mf[q])
        # eval-style beat alignment: pooled-std speed, mean per segment
        if len(y) - skip >= 5:
            yy = y[skip:]
            m = np.isfinite(yy)
            names = {c: j for j, c in enumerate(r["vars"])}
            sdv = np.array([sd.get(c, 1.0) for c in r["vars"]], np.float32)
            for tag, grp in (("hands", HANDS), ("head", HEAD)):
                cs = [names[c] for c in grp if c in names]
                sp = ah.motion_speed(np.nan_to_num(yy), m, cs, sdv, fps) \
                    if cs else None
                if sp is not None and len(sp) >= 5:
                    st = ah.beat_stats(sp, pros[skip:, 0][1:], fps)
                    beat[tag].append(st)
                else:
                    beat[tag].append(None)
            beat["spk"].append(sp_)
            beat["clip"].append(cl_)
        # frame level, natural units: accent against speed
        acc = ah._box(np.abs(np.diff(pros[:, 0], prepend=pros[:1, 0])))[1:]
        for tag, sp in (("hand", sph), ("head", spk_)):
            fr[tag].append(ah._box(sp)[skip:] if sp is not None
                           else np.full(len(acc) - skip, np.nan))
        fr["acc"].append(acc[skip:])
        fr["speaker"].append(np.full(len(acc) - skip, sp_))
        fr["clip"].append(np.full(len(acc) - skip, cl_))
        # windows
        ws = WinSeg(y, cols, pros, fps, sph, spk_)
        starts = np.arange(0, len(y) - W + 1, H)
        if not len(starts):
            continue
        ld = ws.loud(starts, W)
        cols_out = {t: np.full((len(starts), len(lags)), np.nan)
                    for t in WIN_TARGETS}
        for j, lag in enumerate(lags):
            sh = starts + int(round(lag * fps))
            v = (sh >= 0) & (sh + W <= len(y))
            if not v.any():
                continue
            s_ = sh[v]
            cols_out["hand_amp"][v, j] = ws.amp("hand", s_, W)
            cols_out["head_amp"][v, j] = ws.amp("head", s_, W)
            cols_out["hand_speed"][v, j] = ws.speed("hand", s_, W)
            cols_out["head_speed"][v, j] = ws.speed("head", s_, W)
        win["speaker"].append(np.full(len(starts), sp_))
        win["clip"].append(np.full(len(starts), cl_))
        win["lang"].append(np.full(len(starts), lg_))
        win["loud"].append(ld)
        for t in WIN_TARGETS:
            win[t].append(cols_out[t])
        if used % 2000 == 0:
            log(f"probe prosody: {used}/{nrow} segments")
    inv = lambda d: [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]
    S = {k: np.asarray(v, np.float64) for k, v in segf.items()}
    S.update({k: np.asarray(v) for k, v in seg.items()})
    cat = lambda v: np.concatenate(v) if v else np.zeros(0)
    Wn = {k: (np.concatenate(v, 0) if v else np.zeros((0, len(lags))))
          for k, v in win.items() if k in WIN_TARGETS}
    Wn.update({k: cat(win[k]) for k in ("speaker", "clip", "lang", "loud")})
    Fr = {k: cat(v) for k, v in fr.items()}
    return {"seg": S, "win": Wn, "frame": Fr, "beat": beat, "lags": lags,
            "names": {"speaker": inv(spk_id), "clip": inv(clip_id),
                      "lang": inv(lang_id)}, "used": used, "sd": sd}


# ---------------------------------------------------------- the report

def _cell(c):
    if c is None or c.get("r") is None:
        return "   -  "
    if c.get("r_lo") is None:
        return f"{c['r']:+.2f} "
    star = "*" if (c["r_lo"] > 0 or c["r_hi"] < 0) else " "
    return f"{c['r']:+.2f}{star}"


def feature_table(A, mode, o, ctrl=None, lang=None, seed=0):
    """{feature: {target: corr}} at segment level."""
    S = A["seg"]
    sel = np.ones(len(S["dur"]), bool) if lang is None else \
        S["lang"] == A["names"]["lang"].index(lang)
    uni = np.unique(S["clip"][sel])
    out = {}
    for f in FEATURES:
        out[f] = {}
        for t in TARGETS:
            out[f][t] = corr(S[f][sel], S[t][sel], S["speaker"][sel],
                             S["clip"][sel], mode,
                             None if ctrl is None else ctrl[sel],
                             o.boot, seed, uni)
    return out


def _ci(c):
    return "[-]" if c.get("r_lo") is None else \
        f"[{c['r_lo']:+.3f}, {c['r_hi']:+.3f}]"


def table_lines(tab, title):
    L = [title, "  " + "feature".ljust(10) + " ".join(
        t.rjust(8) for t in TARGETS) + "   |r| amp  |r| spd"]
    for f in FEATURES:
        cs = tab[f]
        a = [abs(cs[t]["r"]) for t in AMPLITUDE if cs[t].get("r") is not None]
        s = [abs(cs[t]["r"]) for t in SPEED if cs[t].get("r") is not None]
        L.append("  " + f.ljust(10) + " ".join(_cell(cs[t]).rjust(8)
                                                 for t in TARGETS)
                 + f"   {np.mean(a):.2f}     {np.mean(s):.2f}"
                 if a and s else "  " + f.ljust(10) + "  (no data)")
    return L


def prosody_probe(o, log=log) -> dict:
    rows, npz = read_rows(o.export, o.limit, o.seed)
    log(f"probe prosody: {len(rows)} segments")
    A = analyse_segments(rows, npz, o, log)
    S, nm = A["seg"], A["names"]
    n = A["used"]
    rep = {"segments": n, "hours": round(float(S["dur"].sum()) / 3600, 2)
           if n else 0, "speakers": len(nm["speaker"]),
           "clips": len(nm["clip"]),
           "languages": {l: int((S["lang"] == i).sum())
                         for i, l in enumerate(nm["lang"])},
           "windows": int(len(A["win"]["loud"]))}
    L = [f"prosody probe: {rep['segments']} segments, {rep['hours']} h, "
         f"{rep['speakers']} speakers, {rep['clips']} clips, languages "
         f"{rep['languages']}, {rep['windows']} windows "
         f"({o.win:g} s / {o.wstep:g} s)",
         "r = Pearson, * = the 95% clip-bootstrap CI excludes 0; hand_* in "
         "hu, head_* in deg"]
    if n < 20:
        L.append("too few segments for correlations")
        rep["text"] = L
        return rep
    logd = np.log(np.maximum(S["dur"], 1e-3))
    uni = np.unique(S["clip"])
    seg = {}
    for key, mode, ctrl in (("overall", "raw", None),
                            ("within_speaker", "within", None),
                            ("within_speaker_no_duration", "within", logd)):
        seg[key] = feature_table(A, mode, o, ctrl, None, o.seed)
    seg["by_language"] = {l: feature_table(A, "within", o, None, l, o.seed)
                          for l in nm["lang"] if (S["lang"] == nm["lang"].index(l)).sum() >= 30}
    rep["segment_level"] = {
        k: ({f: {t: clean(c) for t, c in d.items()} for f, d in v.items()}
            if k != "by_language" else
            {l: {f: {t: clean(c) for t, c in d.items()}
                 for f, d in tab.items()} for l, tab in v.items()})
        for k, v in seg.items()}
    L.append("")
    L += table_lines(seg["within_speaker"],
                     "SEGMENT LEVEL, within speaker (z-scored per speaker, "
                     "pooled):")
    L.append("")
    L += table_lines(seg["within_speaker_no_duration"],
                     "same, log duration regressed out:")
    L.append("")
    L += table_lines(seg["overall"], "overall (speaker style included):")
    for l, tab in seg["by_language"].items():
        L.append("")
        L += table_lines(tab, f"language {l}, within speaker:")
    # windowed
    Wd, lags = A["win"], A["lags"]
    win = {"lags": [float(x) for x in lags], "within_speaker": {},
           "overall": {}, "by_language": {}}
    uniw = np.unique(Wd["clip"])
    best = {}
    for t in WIN_TARGETS:
        cs = [corr(Wd["loud"], Wd[t][:, j], Wd["speaker"], Wd["clip"],
                   "within", None, o.boot, o.seed, uniw)
              for j in range(len(lags))]
        win["within_speaker"][t] = [clean(c) for c in cs]
        ok = [j for j, c in enumerate(cs) if c.get("r") is not None]
        best[t] = max(ok, key=lambda j: abs(cs[j]["r"])) if ok else None
        win["overall"][t] = clean(corr(
            Wd["loud"], Wd[t][:, int(np.argmin(abs(lags)))], Wd["speaker"],
            Wd["clip"], "raw", None, o.boot, o.seed, uniw))
        win["_c_" + t] = cs
    j0 = int(np.argmin(abs(lags)))
    for l in nm["lang"]:
        m = Wd["lang"] == nm["lang"].index(l)
        if m.sum() < 100:
            continue
        win["by_language"][l] = {t: clean(corr(
            Wd["loud"][m], Wd[t][m, j0], Wd["speaker"][m], Wd["clip"][m],
            "within", None, o.boot, o.seed, uniw)) for t in WIN_TARGETS}
    L += ["", f"WINDOWED, loudness vs motion in the window, within speaker "
          f"(r by lag s; + = motion follows the audio):",
          "  lag    " + " ".join(t.rjust(10) for t in WIN_TARGETS)]
    for j, lg in enumerate(lags):
        L.append(f"  {lg:+.1f}   " + " ".join(
            (_cell(win["_c_" + t][j]).rjust(10)) for t in WIN_TARGETS))
    L.append("  best lag " + " ".join(
        (f"{lags[best[t]]:+.1f}" if best[t] is not None else "-").rjust(10)
        for t in WIN_TARGETS))
    L.append("  lag 0, overall (no speaker control): " + ", ".join(
        f"{t} {_cell(win['overall'][t]).strip()}" for t in WIN_TARGETS))
    for l, d in win["by_language"].items():
        L.append(f"  lag 0, {l}, within speaker: " + ", ".join(
            f"{t} {_cell(d[t]).strip()}" for t in WIN_TARGETS))
    cdict = {t: win.pop("_c_" + t) for t in WIN_TARGETS}
    rep["windowed"] = win
    # frame level
    Fr = A["frame"]
    fl = {}
    if len(Fr["acc"]):
        unif = np.unique(Fr["clip"])
        for tag in ("hand", "head"):
            fl[tag] = {
                "within_speaker": clean(corr(
                    Fr["acc"], Fr[tag], Fr["speaker"], Fr["clip"], "within",
                    None, min(o.boot, 300), o.seed, unif)),
                "overall": clean(corr(
                    Fr["acc"], Fr[tag], Fr["speaker"], Fr["clip"], "raw",
                    None, min(o.boot, 300), o.seed, unif))}
    bs = A["beat"]
    ev = {}
    for tag in ("hands", "head"):
        cs = np.array([np.nan if (b is None or b["corr"] is None)
                       else b["corr"] for b in bs[tag]])
        cl = np.array(bs["clip"])
        okc = np.isfinite(cs)
        if okc.sum() >= 5:
            uc, gi = np.unique(cl[okc], return_inverse=True)
            sums = np.bincount(gi, cs[okc], len(uc))
            cnt = np.bincount(gi, minlength=len(uc)).astype(float)
            bc = _counts(len(uc), min(o.boot, 1000), o.seed)
            bm = (bc @ sums) / np.maximum(bc @ cnt, 1)
            ev[tag] = {"corr": round(float(cs[okc].mean()), 4),
                       "lo": round(float(np.percentile(bm, 2.5)), 4),
                       "hi": round(float(np.percentile(bm, 97.5)), 4),
                       "segments": int(okc.sum())}
    rep["frame_level"] = {"natural_units": fl, "eval_style": ev}
    L += ["", "FRAME LEVEL, accent (|d rms|, smoothed) against speed, "
          f"first {o.skip:g} s of each segment left out:"]
    for tag in ("hand", "head"):
        if tag in fl:
            L.append(f"  {tag} speed, pooled frames, natural units: within "
                     f"speaker {_cell(fl[tag]['within_speaker']).strip()} "
                     f"{_ci(fl[tag]['within_speaker'])}, overall "
                     f"{_cell(fl[tag]['overall']).strip()}"
                     if fl[tag]["within_speaker"].get("r") is not None else
                     f"  {tag}: no data")
    for tag, e in ev.items():
        L.append(f"  {tag}, eval style (pooled-std speed, mean of the "
                 f"per-segment corr over {e['segments']} segments): "
                 f"{e['corr']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]  "
                 "(the grid's ground truth: 0.03)")
    # hypotheses
    hyp = {}
    uniq = uni

    def seg_corr(f, t):
        return corr(S[f], S[t], S["speaker"], S["clip"], "within", None,
                    o.boot, o.seed, uniq)

    def pair(f, a, b):
        ca, cb = seg_corr(f, a), seg_corr(f, b)
        d = diff_ci(ca, cb, strength=True)
        return {a: clean(ca), b: clean(cb), "diff": None if d is None else {
            "abs_diff": round(d[0], 4), "lo": round(d[1], 4),
            "hi": round(d[2], 4)}}

    def verdict(p, a, b, want):
        d = p["diff"]
        if d is None:
            return "no data"
        if d["lo"] > 0 or d["hi"] < 0:
            got = a if d["abs_diff"] > 0 else b
            return (f"{got} is the stronger (diff CI excludes 0)"
                    + (", as hypothesised" if got == want
                       else ", NOT as hypothesised"))
        return "no difference between the two (diff CI includes 0)"
    L += ["", "HYPOTHESES (segment level, within speaker; |r| of the two "
          "paired by the same clip resamples):"]
    for name, f, a, b, want in (
            ("loudness: amplitude > speed", "loud", "hand_std", "hand_v",
             "hand_std"),
            ("loudness (head): amplitude > speed", "loud", "head_std",
             "head_v", "head_std"),
            ("rate: speed > amplitude", "rate", "hand_v", "hand_std",
             "hand_v"),
            ("rate (head): speed > amplitude", "rate", "head_v", "head_std",
             "head_v")):
        p = pair(f, a, b)
        p["verdict"] = verdict(p, a, b, want)
        hyp[name] = p
        ra, rb = p[a], p[b]
        if ra.get("r") is None or rb.get("r") is None:
            L.append(f"  {name}: no data")
            continue
        d = p["diff"]
        L.append(f"  {name}: {f} -> {a} {ra['r']:+.3f} "
                 f"{_ci(ra)}, {b} {rb['r']:+.3f} {_ci(rb)}; "
                 + (f"|r| diff {d['abs_diff']:+.3f} [{d['lo']:+.3f}, "
                    f"{d['hi']:+.3f}]: " if d else "") + p["verdict"])
    rep["hypotheses"] = hyp
    rep["text"] = L
    return rep


# ------------------------------------------------------------- match

def need_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("flow.probe match needs torch")


class Corpus:
    """The windows of a set of segments with everything the encoders read."""

    def __init__(self, rows, npz_path, hidden, o, log=log):
        self.o = o
        self.W = int(o.window)
        z = np.load(npz_path)
        idx = fh.load_index(hidden)
        with open(os.path.join(hidden, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        layers = self.meta.get("layers") or []
        self.sel = self._layers(o.layer, layers, idx)
        self.cols = [c for c in CANON]
        need_pros = any("prosody" in v for v in o.variants_list)
        self.segs = []
        by_shard = {}
        for r in rows:
            by_shard.setdefault(idx[r["id"]]["shard"], []).append(r)
        for si, (shard, rs) in enumerate(sorted(by_shard.items())):
            sh = np.load(os.path.join(hidden, shard))
            for r in rs:
                e = idx[r["id"]]
                k = r["key"]
                if k + ".traj" not in z.files:
                    continue
                if need_pros and k + ".prosody" not in z.files:
                    continue
                cmap = columns(r["vars"], CANON)
                if len(cmap) < len(CANON):
                    continue
                tr = z[k + ".traj"].astype(np.float32)
                y = tr[:, [cmap[c] for c in CANON]]
                pros = z[k + ".prosody"].astype(np.float32) \
                    if k + ".prosody" in z.files else np.zeros(
                        (len(tr), 3), np.float32)
                if len(pros) != len(tr) or len(tr) < self.W:
                    continue
                h = sh[e["key"] + ".h"]
                h = h[:, self.sel] if h.ndim == 3 else h[:, None]
                chars = z[k + ".chars"] if k + ".chars" in z.files \
                    else np.zeros((0, 2), np.float32)
                tt = ah.token_times(sh[e["key"] + ".spoken"], chars,
                                    r.get("speech_seconds") or 0.0)
                self.segs.append({
                    "id": r["id"], "clip": clip_of(r),
                    "speaker": speaker_name(r), "lang": lang_of(r),
                    "fps": float(r.get("fps") or FPS),
                    "t0": float(r.get("t0") or 0.0), "y": y,
                    "pros": ah.pros_feat(pros), "h": np.ascontiguousarray(
                        h.astype(np.float16)), "tt": tt.astype(np.float32)})
            log(f"probe match: loaded shard {si + 1}/{len(by_shard)}, "
                f"{len(self.segs)} segments")
        if not self.segs:
            raise SystemExit("flow.probe match: no segment has the canonical "
                             "columns, hidden states and (for prosody "
                             "variants) prosody")
        self.D = self.segs[0]["h"].shape[-1]
        self.L = self.segs[0]["h"].shape[1]
        self.clips = sorted({s["clip"] for s in self.segs})
        self.speakers = sorted({s["speaker"] for s in self.segs})

    @staticmethod
    def _layers(spec, layers, idx):
        spec = str(spec)
        n = len(layers) or 1
        if spec == "all":
            return list(range(n))
        v = int(spec)
        if v in layers:                  # a layer number from meta.json
            return [layers.index(v)]
        if 0 <= v < n:                   # an index into the stored layers
            return [v]
        raise SystemExit(f"--layer {spec}: the stored layers are {layers}")

    def windows(self, keep_clips):
        """Window table of the segments in keep_clips: arrays seg, start,
        clip (index into self.clips), spk, tclip (clip seconds)."""
        W, hop = self.W, int(self.o.hop)
        sg, st, cl, sp, tc = [], [], [], [], []
        ci = {c: i for i, c in enumerate(self.clips)}
        si = {s: i for i, s in enumerate(self.speakers)}
        for i, s in enumerate(self.segs):
            if s["clip"] not in keep_clips:
                continue
            for a in range(0, len(s["y"]) - W + 1, hop):
                if not np.isfinite(s["y"][a:a + W]).all():
                    continue
                sg.append(i)
                st.append(a)
                cl.append(ci[s["clip"]])
                sp.append(si[s["speaker"]])
                tc.append(s["t0"] + a / s["fps"])
        return {"seg": np.array(sg, np.int64), "start": np.array(st, np.int64),
                "clip": np.array(cl, np.int64), "spk": np.array(sp, np.int64),
                "tclip": np.array(tc, np.float64)}


def motion_stats(corpus, win):
    """(mean, std) [C] of the frames of the windows' segments."""
    s = s2 = 0
    n = 0
    for i in np.unique(win["seg"]):
        y = corpus.segs[i]["y"]
        y = y[np.isfinite(y).all(1)]
        s = s + y.sum(0, dtype=np.float64)
        s2 = s2 + (y.astype(np.float64) ** 2).sum(0)
        n += len(y)
    m = s / max(n, 1)
    sd = np.sqrt(np.maximum(s2 / max(n, 1) - m * m, 0))
    return m.astype(np.float32), np.where(sd < 1e-4, 1.0, sd).astype(np.float32)


def shuffle_map(n, seed):
    """A fixed permutation of range(n) without fixed points (n > 1)."""
    p = np.random.default_rng(seed).permutation(n)
    for i in np.flatnonzero(p == np.arange(n)):
        j = (i + 1) % n
        p[i], p[j] = p[j], p[i]
    return p


def make_batch(corpus, win, widx, text_src, norm, o, torch, device):
    """Tensors of the windows widx (indices into win): motion [B, W, 2C],
    pros [B, W, 3], tokens h [B, T, L, D], rel time [B, T], mask [B, T]
    (the text of window text_src[w])."""
    W = corpus.W
    mean, sd = norm
    B = len(widx)
    mot = np.zeros((B, W, 2 * len(CANON)), np.float32)
    pr = np.zeros((B, W, 3), np.float32)
    toks, tms = [], []
    for b, w in enumerate(widx):
        s = corpus.segs[win["seg"][w]]
        a = int(win["start"][w])
        y = (s["y"][a:a + W] - mean) / sd
        if o.center:
            y = y - y.mean(0)
        mot[b, :, :len(CANON)] = y
        mot[b, 1:, len(CANON):] = np.diff(y, axis=0) * 5.0
        pr[b] = s["pros"][a:a + W]
        t = text_src[w]
        s2 = corpus.segs[win["seg"][t]]
        a2 = int(win["start"][t])
        fps = s2["fps"]
        lo, hi = a2 / fps - o.lookback, (a2 + W) / fps
        k = np.flatnonzero((s2["tt"] >= lo) & (s2["tt"] <= hi))[-o.max_tokens:]
        toks.append(s2["h"][k])
        tms.append((s2["tt"][k] - a2 / fps))
    T = max([len(x) for x in toks] + [1])
    h = np.zeros((B, T, corpus.L, corpus.D), np.float16)
    tm = np.zeros((B, T), np.float32)
    mk = np.zeros((B, T), bool)
    for b in range(B):
        n = len(toks[b])
        h[b, :n] = toks[b]
        tm[b, :n] = tms[b]
        mk[b, :n] = True
    f = lambda a, dt=None: torch.as_tensor(a, dtype=dt).to(device)
    return {"mot": f(mot), "pros": f(pr), "h": f(h).float(), "tm": f(tm),
            "mask": f(mk)}


def build_model(corpus, o, use_text, use_pros, torch):
    nn = torch.nn
    dim = o.dim
    C2 = 2 * len(CANON)

    def conv(cin):
        return nn.Sequential(
            nn.Conv1d(cin, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(384, dim))

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.ModuleList(nn.LayerNorm(corpus.D)
                                    for _ in range(corpus.L))
            self.mix = nn.Parameter(torch.zeros(corpus.L))
            self.proj = nn.Linear(corpus.D, 128)
            self.tproj = nn.Linear(8, 128)
            self.null = nn.Parameter(torch.randn(128) * 0.02)
            self.att = nn.Linear(128, o.queries)
            self.val = nn.Linear(128, 64)
            self.out = nn.Linear(64 * o.queries, dim)
            self.drop = nn.Dropout(0.1)

        def forward(self, b):
            h = b["h"]
            w = torch.softmax(self.mix, 0)
            x = sum(w[i] * self.ln[i](h[:, :, i]) for i in range(h.shape[2]))
            t = b["tm"][..., None] * torch.tensor(
                [0.5, 1.0, 2.0, 4.0], device=h.device)
            tf = torch.cat([torch.sin(t), torch.cos(t)], -1)
            x = torch.nn.functional.gelu(self.proj(self.drop(x))
                                         + self.tproj(tf))
            x = torch.cat([self.null.expand(x.shape[0], 1, -1), x], 1)
            m = torch.cat([torch.ones_like(b["mask"][:, :1]), b["mask"]], 1)
            a = self.att(x).masked_fill(~m[..., None], -1e9)
            a = torch.softmax(a, 1)                      # [B, T, Q]
            v = self.val(x)                              # [B, T, 64]
            p = torch.einsum("btq,btd->bqd", a, v).flatten(1)
            return self.out(p)

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.motion = conv(C2)
            self.text = Text() if use_text else None
            self.pros = conv(3) if use_pros else None
            self.fuse = nn.Sequential(nn.LayerNorm(2 * dim),
                                      nn.Linear(2 * dim, dim)) \
                if use_text and use_pros else None
            self.scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        def em(self, b):
            return self.motion(b["mot"].transpose(1, 2))

        def ec(self, b):
            parts = []
            if self.text is not None:
                parts.append(self.text(b))
            if self.pros is not None:
                parts.append(self.pros(b["pros"].transpose(1, 2)))
            return self.fuse(torch.cat(parts, -1)) if self.fuse is not None \
                else parts[0]

    return Probe()


VARIANTS = {            # name: (text, prosody, shuffle tokens, train scheme)
    "text": (True, False, False, "speaker"),
    "prosody": (False, True, False, "speaker"),
    "text+prosody": (True, True, False, "speaker"),
    "text_shuffled": (True, False, True, "speaker"),
    "text_random": (True, False, False, "random"),
    "text_clip": (True, False, False, "clip"),
}
SCHEMES = ("random", "speaker", "clip")


def candidates(win, pool, scheme, K, rng, limit=0, wsec=3.0):
    """(queries [Q], neg [Q, K-1], skipped): the pool's windows (indices
    into win) as queries with K - 1 negatives drawn per scheme (module
    docstring); queries with too few candidates are skipped; limit: at most
    that many queries (random)."""
    pool = np.asarray(pool)
    cl, sp, tc = win["clip"][pool], win["spk"][pool], win["tclip"][pool]
    order = rng.permutation(len(pool))
    q, neg, skipped = [], [], 0
    by_sp = {s: np.flatnonzero(sp == s) for s in np.unique(sp)}
    by_cl = {c: np.flatnonzero(cl == c) for c in np.unique(cl)}
    for i in order:
        if scheme == "random":
            c = np.flatnonzero(cl != cl[i])
        elif scheme == "speaker":
            c = by_sp[sp[i]]
            c = c[cl[c] != cl[i]]
        else:
            c = by_cl[cl[i]]
            c = c[np.abs(tc[c] - tc[i]) >= wsec - 1e-6]
        if len(c) < K - 1:
            skipped += 1
            continue
        q.append(i)
        neg.append(rng.choice(c, K - 1, replace=False))
        if limit and len(q) >= limit:
            break
    if not q:
        return pool[:0], np.zeros((0, K - 1), np.int64), skipped
    return pool[np.array(q)], pool[np.array(neg)], skipped


def embed_all(model, corpus, win, widx, text_src, norm, o, torch, device,
              bs=256):
    model.eval()
    zm, zc = [], []
    with torch.no_grad():
        for a in range(0, len(widx), bs):
            b = make_batch(corpus, win, widx[a:a + bs], text_src, norm, o,
                           torch, device)
            zm.append(torch.nn.functional.normalize(model.em(b), dim=-1).cpu())
            zc.append(torch.nn.functional.normalize(model.ec(b), dim=-1).cpu())
    model.train()
    return torch.cat(zm), torch.cat(zc)


def retrieval(zm, zc, pos, q, neg, torch):
    """Rank (0 = best) of each query's own motion among itself and its
    negatives. pos: {window index: row in zm}."""
    qi = torch.as_tensor([pos[i] for i in q])
    ni = torch.as_tensor([[pos[j] for j in r] for r in neg])
    cand = torch.cat([qi[:, None], ni], 1)
    sims = torch.einsum("qd,qkd->qk", zc[qi], zm[cand])
    return (sims[:, 1:] > sims[:, :1]).sum(1).numpy()


def acc_ci(hit, clip, B=1000, seed=0):
    """(mean, lo, hi) of a 0/1 array with a clip bootstrap."""
    hit = np.asarray(hit, np.float64)
    if not len(hit):
        return None, None, None
    uc, gi = np.unique(clip, return_inverse=True)
    s = np.bincount(gi, hit, len(uc))
    n = np.bincount(gi, minlength=len(uc)).astype(float)
    c = _counts(len(uc), B, seed)
    b = (c @ s) / np.maximum(c @ n, 1)
    return (float(hit.mean()), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)))


def read_sem(root, clips, log=log):
    """{clip: [(start, end)]} of the semantic intervals in root (module
    docstring), {} when nothing parses."""
    out, files, parsed = {}, 0, 0
    quiet = {"beat", "none", "neutral", "idle", "other", "nan", ""}
    for c in clips:
        for p in glob.glob(os.path.join(root, "**", c + ".*"), recursive=True):
            if not p.lower().endswith((".txt", ".csv", ".tsv", ".json")):
                continue
            files += 1
            iv = []
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    txt = f.read()
                if p.lower().endswith(".json"):
                    dat = json.loads(txt)
                    lines = [" ".join(str(v) for v in (
                        d.values() if isinstance(d, dict) else d))
                        for d in (dat if isinstance(dat, list) else [])]
                else:
                    lines = txt.splitlines()
            except (OSError, ValueError):
                continue
            for ln in lines:
                parts = [x for x in re.split(r"[\t,; ]+", ln.strip()) if x]
                nums, words = [], []
                for x in parts:
                    try:
                        nums.append(float(x))
                    except ValueError:
                        words.append(x.lower())
                if len(nums) < 2 or nums[1] <= nums[0]:
                    continue
                sem = any(w not in quiet for w in words[:2]) if words else True
                if len(nums) >= 3 and not words:
                    sem = nums[2] > 0
                if sem:
                    iv.append((nums[0], nums[1]))
            if iv:
                parsed += 1
                out.setdefault(c, []).extend(iv)
    log(f"probe match: --sem {root}: {files} candidate files, {parsed} with "
        f"semantic intervals")
    return out


def sem_flags(win, corpus, sem, W):
    """[Nwin] bool: the window overlaps a semantic interval of its clip."""
    f = np.zeros(len(win["seg"]), bool)
    for i in range(len(f)):
        iv = sem.get(corpus.clips[win["clip"][i]])
        if not iv:
            continue
        a = win["tclip"][i]
        b = a + W / corpus.segs[win["seg"][i]]["fps"]
        f[i] = any(s < b and e > a for s, e in iv)
    return f


def train_variant(name, corpus, win, tr, ev, norm, o, device, log=log):
    """(model, result dict) of one variant (module docstring)."""
    import torch
    use_text, use_pros, shuf, scheme = VARIANTS[name]
    torch.manual_seed(o.seed)
    rng = np.random.default_rng(o.seed)
    model = build_model(corpus, o, use_text, use_pros, torch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=o.lr, total_steps=o.steps, pct_start=0.1)
    src = np.arange(len(win["seg"]))
    if shuf:
        for pool in (tr, ev):
            if len(pool) > 1:
                src[pool] = pool[shuffle_map(len(pool), o.seed + 7)]
    groups = {"random": None,
              "speaker": {s: tr[win["spk"][tr] == s]
                          for s in np.unique(win["spk"][tr])},
              "clip": {c: tr[win["clip"][tr] == c]
                       for c in np.unique(win["clip"][tr])}}
    if scheme != "random":
        mn = min(8, o.batch)
        groups[scheme] = {k: v for k, v in groups[scheme].items()
                          if len(v) >= mn}
        if not groups[scheme]:
            raise SystemExit(f"{name}: no {scheme} has {mn} training windows")
        gk = list(groups[scheme])
        gw = np.array([len(groups[scheme][k]) for k in gk], float)
        gw /= gw.sum()
    Wf = corpus.W / corpus.segs[0]["fps"]

    def draw():
        if scheme == "random":
            return rng.choice(tr, min(o.batch, len(tr)), replace=False)
        g = groups[scheme][gk[rng.choice(len(gk), p=gw)]]
        if scheme == "speaker":
            return rng.choice(g, min(o.batch, len(g)), replace=False)
        out, ts = [], []
        for i in rng.permutation(len(g)):
            t = win["tclip"][g[i]]
            if all(abs(t - u) >= Wf - 1e-6 for u in ts):
                out.append(g[i])
                ts.append(t)
            if len(out) >= o.batch:
                break
        return np.array(out)

    # curve subsets: queries under the training scheme
    crng = np.random.default_rng(o.seed + 1)
    cq = {}
    for pn, pool in (("train", tr), ("eval", ev)):
        q, ng, _ = candidates(win, pool, scheme, o.k, crng, o.curve_n, Wf)
        cq[pn] = (q, ng)

    def top1(pn):
        q, ng = cq[pn]
        if not len(q):
            return None
        need = np.unique(np.concatenate([q, ng.ravel()]))
        zm, zc = embed_all(model, corpus, win, need, src, norm, o, torch,
                           device)
        pos = {int(w): i for i, w in enumerate(need)}
        return float((retrieval(zm, zc, pos, q, ng, torch) == 0).mean())

    curve, run, t0 = [], [], time.time()
    model.train()
    for step in range(1, o.steps + 1):
        wi = draw()
        if len(wi) < 4:
            continue
        b = make_batch(corpus, win, wi, src, norm, o, torch, device)
        zm = torch.nn.functional.normalize(model.em(b), dim=-1)
        zc = torch.nn.functional.normalize(model.ec(b), dim=-1)
        lg = model.scale.exp().clamp(max=100) * zc @ zm.T
        lab = torch.arange(len(wi), device=device)
        loss = (torch.nn.functional.cross_entropy(lg, lab)
                + torch.nn.functional.cross_entropy(lg.T, lab)) / 2
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % o.eval_every == 0 or step == o.steps:
            c = {"step": step, "loss": round(float(np.mean(run)), 4),
                 "train_top1": top1("train"), "eval_top1": top1("eval")}
            run = []
            curve.append(c)
            log(f"probe match: {name} step {step}/{o.steps} loss "
                f"{c['loss']:.3f} top1 train {c['train_top1']} eval "
                f"{c['eval_top1']}")
    res = {"variant": name, "text": use_text, "prosody": use_pros,
           "shuffled": shuf, "train_scheme": scheme, "steps": o.steps,
           "batch": o.batch, "curve": curve,
           "seconds": round(time.time() - t0, 1)}
    # final: all three schemes
    erng = np.random.default_rng(o.seed + 2)
    flags = getattr(o, "_sem_flags", None)
    res["eval"] = {}
    for sc in SCHEMES:
        q, ng, skipped = candidates(win, ev, sc, o.k, erng, o.eval_n, Wf)
        d = {"queries": int(len(q)), "skipped": int(skipped),
             "chance_top1": 1 / o.k, "chance_top5": min(5 / o.k, 1.0)}
        if len(q):
            need = np.unique(np.concatenate([q, ng.ravel()]))
            zm, zc = embed_all(model, corpus, win, need, src, norm, o, torch,
                               device)
            pos = {int(w): i for i, w in enumerate(need)}
            rk = retrieval(zm, zc, pos, q, ng, torch)
            cl = win["clip"][q]
            for key, hit in (("top1", rk == 0), ("top5", rk < 5)):
                m, lo, hi = acc_ci(hit, cl, o.boot, o.seed)
                d[key] = {"acc": round(m, 4), "lo": round(lo, 4),
                          "hi": round(hi, 4)}
            if flags is not None:
                d["semantic"] = {}
                for lab, sel in (("with", flags[q]), ("without", ~flags[q])):
                    if sel.sum() >= 10:
                        m, lo, hi = acc_ci(rk[sel] == 0, cl[sel], o.boot,
                                           o.seed)
                        d["semantic"][lab] = {"n": int(sel.sum()),
                                              "top1": round(m, 4),
                                              "lo": round(lo, 4),
                                              "hi": round(hi, 4)}
        res["eval"][sc] = d
    return model, res


def match_probe(o, log=log) -> dict:
    need_torch()
    import torch
    device = o.device or ("cuda" if torch.cuda.is_available() else "cpu")
    o.variants_list = [v.strip() for v in o.variants.split(",") if v.strip()]
    for v in o.variants_list:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; known: {sorted(VARIANTS)}")
    rows, npz = read_rows(o.export, o.limit, o.seed, o.hidden)
    log(f"probe match: {len(rows)} segments with hidden states, device "
        f"{device}")
    corpus = Corpus(rows, npz, o.hidden, o, log)
    if o.split == "speaker":
        spk_of = {}
        for s in corpus.segs:
            spk_of.setdefault(s["speaker"], set()).add(s["clip"])
        trs, evs = split_names(list(spk_of), o.eval_frac, o.seed)
        ev_clips = {c for s in evs for c in spk_of[s]}
        tr_clips = {c for s in trs for c in spk_of[s]}
    else:
        trc, evc = split_names(corpus.clips, o.eval_frac, o.seed,
                               [c for c in o.eval_clips.split(",") if c])
        tr_clips, ev_clips = set(trc), set(evc)
    wt = corpus.windows(tr_clips)
    we = corpus.windows(ev_clips)
    if not len(we["seg"]) or not len(wt["seg"]):
        raise SystemExit("flow.probe match: the train or the held-out split "
                         "has no windows")
    n0 = len(wt["seg"])
    win = {k: np.concatenate([wt[k], we[k]]) for k in wt}
    tr = np.arange(n0)
    ev = np.arange(n0, len(win["seg"]))
    norm = motion_stats(corpus, wt)
    info = {"segments": len(corpus.segs), "windows_train": int(len(tr)),
            "windows_eval": int(len(ev)), "split": o.split,
            "clips_train": len(tr_clips), "clips_eval": len(ev_clips),
            "speakers_eval": int(len(np.unique(win["spk"][ev]))),
            "layers": o.layer, "k": o.k, "device": device,
            "steps": o.steps, "batch": o.batch}
    log(f"probe match: {info}")
    if o.sem:
        sem = read_sem(o.sem, corpus.clips, log)
        if sem:
            o._sem_flags = sem_flags(win, corpus, sem, corpus.W)
            info["semantic_windows_eval"] = int(o._sem_flags[ev].sum())
        else:
            log("probe match: no semantic annotations found or parsed; "
                "--sem skipped")
            info["sem"] = "none found"
    results = []
    for v in o.variants_list:
        _, res = train_variant(v, corpus, win, tr, ev, norm, o, device, log)
        results.append(res)
    rep = {"info": info, "variants": results}
    rep["text"] = match_lines(rep, o)
    return rep


def match_lines(rep, o):
    i = rep["info"]
    L = [f"match probe: {i['windows_train']} train / {i['windows_eval']} "
         f"held-out windows ({i['split']} split: {i['clips_train']} / "
         f"{i['clips_eval']} clips, {i['speakers_eval']} held-out speakers), "
         f"K={i['k']}, {i['steps']} steps x {i['batch']}, device "
         f"{i['device']}, layer {i['layers']}",
         f"chance: top-1 {1 / i['k']:.3f}, top-5 {min(5 / i['k'], 1):.3f}; "
         "query = the conditioning of a held-out window, candidates = "
         "motions; [95% clip-bootstrap CI]",
         "", "held-out top-1 / top-5 by how the negatives are drawn "
         "(the train column: the scheme it was trained on):"]
    L.append("  " + "variant".ljust(15) + "train".ljust(8) + "".join(
        f"{s:^34}" for s in SCHEMES))
    for r in rep["variants"]:
        cells = []
        for s in SCHEMES:
            d = r["eval"][s]
            if "top1" not in d:
                cells.append("(no queries)".center(34))
                continue
            t1, t5 = d["top1"], d["top5"]
            cells.append(f"{t1['acc']:.3f} [{t1['lo']:.3f},{t1['hi']:.3f}] "
                         f"{t5['acc']:.3f}".center(34))
        L.append("  " + r["variant"].ljust(15) + r["train_scheme"].ljust(8)
                 + "".join(cells))
    L.append("  queries: " + ", ".join(
        f"{s} {rep['variants'][0]['eval'][s]['queries']} (skipped "
        f"{rep['variants'][0]['eval'][s]['skipped']})" for s in SCHEMES)
        if rep["variants"] else "")
    L += ["", "curves (loss; top-1 train subset / held-out, under the "
          "training scheme):"]
    for r in rep["variants"]:
        L.append(f"  {r['variant']} ({r['seconds']} s): " + "  ".join(
            f"{c['step']}: {c['loss']:.2f} "
            f"{'-' if c['train_top1'] is None else format(c['train_top1'], '.2f')}/"
            f"{'-' if c['eval_top1'] is None else format(c['eval_top1'], '.2f')}"
            for c in r["curve"]))
    sem = [r for r in rep["variants"]
           if any("semantic" in r["eval"][s] for s in SCHEMES)]
    if sem:
        L += ["", "semantic gestures (top-1, windows with / without):"]
        for r in sem:
            for s in SCHEMES:
                d = r["eval"][s].get("semantic")
                if d:
                    L.append(f"  {r['variant']} {s}: " + ", ".join(
                        f"{k} {v['top1']:.3f} [{v['lo']:.3f},{v['hi']:.3f}] "
                        f"n={v['n']}" for k, v in d.items()))
    return L


# ----------------------------------------------------------------- cli

def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("prosody", "match", "all"))
    ap.add_argument("--export", required=True,
                    help="flow.beat2 / flow.dataexport action's stem")
    ap.add_argument("--hidden", default=None,
                    help="flow.train.hidden's folder (match, all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="about this many segments, whole clips [all]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=1000,
                    help="bootstrap resamples [1000]")
    g = ap.add_argument_group("prosody")
    g.add_argument("--win", type=float, default=2.0)
    g.add_argument("--wstep", type=float, default=1.0)
    g.add_argument("--lag-max", type=float, default=1.0)
    g.add_argument("--lag-step", type=float, default=0.1)
    g.add_argument("--active-speed", type=float, default=0.3,
                   help="hu/s above which the hands are active [0.3]")
    g.add_argument("--pause", type=float, default=0.2,
                   help="s of silence that count as a pause [0.2]")
    g.add_argument("--min-voiced", type=float, default=1.0,
                   help="s of voiced speech for a rate [1.0]")
    g.add_argument("--skip", type=float, default=1.0,
                   help="s at the start left out of the frame level [1.0]")
    g = ap.add_argument_group("match")
    g.add_argument("--layer", default="1",
                   help="index into the stored layers, a layer number, or "
                        "all [1]")
    g.add_argument("--variants", default=",".join(VARIANTS))
    g.add_argument("--split", default="clip", choices=("clip", "speaker"))
    g.add_argument("--eval-frac", type=float, default=0.1)
    g.add_argument("--eval-clips", default="",
                   help="comma-separated clips to hold out")
    g.add_argument("--window", type=int, default=90, help="frames [90]")
    g.add_argument("--hop", type=int, default=60, help="frames [60]")
    g.add_argument("--lookback", type=float, default=1.0)
    g.add_argument("--max-tokens", type=int, default=48)
    g.add_argument("--k", type=int, default=32)
    g.add_argument("--steps", type=int, default=3000)
    g.add_argument("--batch", type=int, default=128)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--dim", type=int, default=128)
    g.add_argument("--queries", type=int, default=4)
    g.add_argument("--center", action="store_true")
    g.add_argument("--eval-n", type=int, default=4000,
                   help="queries per scheme at the end [4000]")
    g.add_argument("--eval-every", type=int, default=250)
    g.add_argument("--curve-n", type=int, default=200)
    g.add_argument("--device", default=None)
    g.add_argument("--sem", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    o = parse(argv)
    if o.cmd in ("match", "all") and not o.hidden:
        raise SystemExit("flow.probe: match needs --hidden")
    os.makedirs(o.out, exist_ok=True)
    rep, text = {}, []
    t0 = time.time()
    if o.cmd in ("prosody", "all"):
        r = prosody_probe(o)
        text += r.pop("text")
        rep["prosody"] = r
    if o.cmd in ("match", "all"):
        if text:
            text.append("")
        r = match_probe(o)
        text += r.pop("text")
        rep["match"] = r
    rep["seconds"] = round(time.time() - t0, 1)
    rep["command"] = o.cmd
    with open(os.path.join(o.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")
    with open(os.path.join(o.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print("\n".join(text))
    log(f"probe: {o.out}/report.txt, report.json ({rep['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
