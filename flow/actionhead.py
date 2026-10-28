#!/usr/bin/env python3
"""
An action head (docs/embodiment.md Stage 2 (b), the pi0 pattern at small
scale): a local language model's hidden states over a reply, on the
clock of the reply's speech, mapped to the directed canonical trajectory
the reply plays -- the body motion the markers ask for, continuous, per
avatar.

  data      flow.dataexport action (<stem>.jsonl, <stem>.npz: the
            directed trajectory per turn, its variables, each spoken
            character's start and end) and flow.train.hidden's shards
            (states per reply token, each token's spoken span). A token's
            time is its first spoken character's start (a marker's
            tokens: the next spoken character's, so a marker sits where it
            acts); the speech end after the last.
  targets   one variable vocabulary over all avatars (the union of their
            trajectory columns, the voice's mouth and visemes left out: the
            voice owns them), normalised per variable on the training
            frames; a variable an avatar lacks, or a NaN, is masked out of
            the loss. Conditioning on the body: an avatar embedding (index
            0 = unknown, used for 10% of training clips so a new body gets
            something) plus a projection of the body's variable mask.
  model     frames as queries (sinusoidal time, the body), a small
            transformer decoder cross-attending to the projected token
            states (with their times), each frame seeing the tokens up to
            --lookahead seconds after it (a streaming head cannot see the
            end of a reply it is still speaking) and a learned null token;
            a linear read-out per variable. Trained on random chunks of
            --chunk frames; predicted chunk by chunk with half overlap,
            cross-faded. --objective mse [default] regresses; fm is flow
            matching (as flow.legacy.flowprior: x_tau = (1 - tau) x0 + tau
            x1, the velocity x1 - x0 learned, Euler steps to sample), so
            several plausible motions are not averaged into a smaller one.
  history   --history SECONDS (0: off): the head also sees the previous
            frames of the trajectory, as memory tokens every frame of the
            chunk attends to (values and what is known of them, at times
            before the chunk), and predicts the next chunk. Trained
            teacher-forced on the clip's own frames, corrupted as in
            LiveGesture's autoregressive masking (noise up to --hist-noise,
            whole frames and whole variables dropped with --hist-drop, no
            history at all with --hist-none) so that the rollout, which
            feeds the head its own frames, does not drift. Before the
            clip's start the history is missing, not zeros. At eval /
            inference the head starts from the first `history` seconds of
            ground truth (--given N frames of it; 0: from nothing, the mean
            pose) and rolls out chunk by chunk, committing half a chunk
            each time (no cross-fade: the history joins the chunks).
            --target delta predicts offsets from the last known history
            frame (the anchor), the abs target the values.
  prosody   --prosody: `<key>.prosody` (flow.beat2: rms, f0 Hz, voiced at
            the frame rate) as three per-frame features (log RMS, log f0
            over 150 Hz, voiced; pros_feat) added to the frame queries. A
            row without it gets a learned "missing" embedding, and a
            share of training rows (--pros-drop) lose theirs so that
            embedding is trained even on data that always has it.
            --no-text drops the token states (history and / or prosody
            only): what the text adds is the difference to the same run
            with it. The prosody sees the whole chunk (offline, not
            causal).
  layers    the stored states are [n, L, D] flattened to L*D. --layer-mix
            normalises each layer and sums them with learned softmax
            weights (SmolVLA, ELLSA 2) instead of reading the
            concatenation; the weights are logged, saved and in the eval.
  norm      --norm pooled [default]: one mean / std per variable over the
            training frames; avatar: each avatar's own (the pooled ones
            for the unknown avatar and for a variable it has no frames of).
            The metrics always divide by the pooled std, so runs with
            different --norm are scored on one scale.
  eval      per variable (natural units: hu, degrees, the face's ranges)
            the mean squared error of the head, of the rest (the clip's
            first frame held: the directed trajectory has no idle, so it
            starts at rest) and of the training mean, and with something
            observed (--skip seconds, default the history; the first
            frames are not scored for anyone) of the hold (the last frame
            before the scored part held); with --markers (flow.train.
            evalsft's per-turn jsonl, e.g. the tuned model's own replies)
            also of those markers played on the avatar (flow.dataset.play):
            the baseline the head has to beat. On teacher text that
            baseline would be the target itself (zero error), which is why
            it takes another model's markers. A summary block, in the log
            and the JSON, for each of them, all on the pooled std:
              normalised MSE   absolute: posture included
              relative MSE     each trajectory minus its own mean over the
                               clip: the motion around the posture (the
                               baselines, constant, have the clip's
                               variance)
              velocity MSE     of the frame-to-frame differences
              std ratio        predicted over true motion std (within the
                               clip; 0: a constant, 1: as much motion;
                               well under 1: over-smoothed), and the same
                               of the velocity; null where the target does
                               not move
              beat alignment   for rows with prosody, head hands / head
                               motion speed against the audio's accents
                               (|d rms|, smoothed): corr (Pearson), and
                               BEAT's consistency (the share of accent
                               peaks with a speed peak within +-0.2 s) and
                               precision (the share of speed peaks within
                               0.2 s of an accent), for the ground truth
                               too: the ceiling. null where undefined.
            Also the per-variable table, and per variable the motion
            metrics (per_variable_motion in the JSON). `table` prints the
            summaries of several runs side by side.
  train     a head.best.pt at the best eval loss (with a history: the
            middle chunk of each eval turn after a clean history; else the
            first chunk, as always), head.pt the last step, train.json;
            --patience N stops after N evaluations without a better one.
            eval / export take a run folder (its head.best.pt) or a file.
            Checkpoints of before (no history, prosody, mix, best) load
            and evaluate unchanged.
  export    a trajectory (predicted, or any [frames, vars] of canonical
            values) back to parameters with the body's own solver (every
            --every frames, the previous solve as the prior, Live2D arm
            drawings smoothed so they do not flicker) and into move tokens
            (flow.moves: absolute keyframes, PCHIP between them) that the
            existing players take: tokens.json for flow.runtime --tokens-in
            (Live2D) and, on a VRM, states.json through flow.stream for
            flow.vrmrender. The directed trajectory goes through the same
            path, so the two clips differ only by the prediction.

Usage:
    python -m flow.actionhead train --export data/datagen/r1/export/action \\
        --hidden runs/hidden/q8 --out runs/ah/q8
    python -m flow.actionhead eval --ckpt runs/ah/q8 \\
        --export ... --hidden ... [--markers runs/sft/q8/eval/student.jsonl \\
        --data data/datagen/r1]
    python -m flow.actionhead train ... --history 1 --prosody --out runs/ah/b2
    python -m flow.actionhead table runs/ah/*/eval.json   # side by side
    python -m flow.actionhead export --ckpt runs/ah/q8 --export ... \\
        --hidden ... --data data/datagen/r1 --id seed-san.cheerful.0003/t02 \\
        --out runs/ah/q8/side
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

try:
    from . import body as fb
    from . import moves as mv
    from .train import hidden as fh
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from flow import body as fb
    from flow import moves as mv
    from flow.train import hidden as fh

EXCLUDE = ("mouth.", "viseme.")
HIPS = mv.HIPS
CHUNK = 96              # frames: 3.2 s at 30 fps
LOOKAHEAD = 1.0         # s of reply a frame may see past its own time
UNKNOWN_P = 0.1


def log(s):
    print(s, flush=True)


# ---------------------------------------------------------------- data

def token_times(spoken_spans, chars, end):
    """[n] seconds: each token's first spoken character's start (module
    docstring)."""
    s0 = np.asarray(spoken_spans, np.int64).reshape(-1, 2)[:, 0]
    ch = np.asarray(chars, np.float32).reshape(-1, 2)
    t = np.full(len(s0), float(end), np.float32)
    ok = s0 < len(ch)
    t[ok] = ch[s0[ok], 0]
    return t


class Data:
    """The (hidden states, trajectory) pairs of an export and a hidden-state
    folder, by turn id."""

    def __init__(self, export, hidden, vocab=None, avatars=None,
                 exclude=EXCLUDE, prosody=False):
        self.prosody = prosody
        rows, self.npz_path = fh.read_export(export)
        self.index = fh.load_index(hidden)
        self.hidden = hidden
        self.rows = [r for r in rows if r["id"] in self.index]
        self.vocab = list(vocab) if vocab else sorted(
            {v for r in self.rows for v in r["vars"]
             if not v.startswith(exclude)})
        self.vi = {v: i for i, v in enumerate(self.vocab)}
        names = sorted({r["avatar"] for r in self.rows})
        self.avatars = list(avatars) if avatars else ["?"] + names
        self.ai = {a: i for i, a in enumerate(self.avatars)}
        self._npz = None
        self._sh = {}
        with open(os.path.join(hidden, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.rows)

    def _z(self):
        if self._npz is None:
            self._npz = np.load(self.npz_path)
        return self._npz

    def _shard(self, name):
        if name not in self._sh:
            if len(self._sh) > 4:
                self._sh.clear()
            self._sh[name] = np.load(os.path.join(self.hidden, name))
        return self._sh[name]

    def item(self, i):
        """{id, avatar, h [n, L*D], t_tok [n], y [F, V] (0 where masked),
        m [F, V] bool, avail [V], fps, vars, pros [F, 3] (rms, f0 Hz, voiced:
        flow.beat2's `<key>.prosody`; None when the row has none or the data
        was opened without prosody)}."""
        r = self.rows[i]
        z = self._z()
        e = self.index[r["id"]]
        sh = self._shard(e["shard"])
        h = sh[e["key"] + ".h"].astype(np.float32)
        h = h.reshape(h.shape[0], -1)
        chars = z[r["key"] + ".chars"] if r["key"] + ".chars" in z.files \
            else np.zeros((0, 2), np.float32)
        tt = token_times(sh[e["key"] + ".spoken"], chars,
                         r.get("speech_seconds") or 0.0)
        tr = z[r["key"] + ".traj"].astype(np.float32)
        F, V = tr.shape[0], len(self.vocab)
        y = np.zeros((F, V), np.float32)
        m = np.zeros((F, V), bool)
        avail = np.zeros(V, bool)
        for c, v in enumerate(r["vars"]):
            k = self.vi.get(v)
            if k is None:
                continue
            col = tr[:, c]
            ok = ~np.isnan(col)
            y[ok, k] = col[ok]
            m[:, k] = ok
            avail[k] = True
        pros = None
        if self.prosody and r["key"] + ".prosody" in z.files:
            pros = z[r["key"] + ".prosody"].astype(np.float32)
            if len(pros) != F:
                pros = None
        return {"id": r["id"], "avatar": r["avatar"],
                "a": self.ai.get(r["avatar"], 0), "h": h, "t_tok": tt,
                "y": y, "m": m, "avail": avail, "fps": float(r["fps"]),
                "vars": r["vars"], "pros": pros}


def norm_stats(data, ids, per_avatar=False):
    """(mean, std): [V] over the training frames (pooled), or with
    per_avatar [A, V], row a the avatar's own (row 0, the unknown avatar,
    and a variable an avatar has no frames of: the pooled ones)."""
    V, A = len(data.vocab), len(data.avatars)
    s, s2, n = (np.zeros((A, V)) for _ in range(3))
    for i in ids:
        it = data.item(i)
        s[it["a"]] += (it["y"] * it["m"]).sum(0)
        s2[it["a"]] += (it["y"] ** 2 * it["m"]).sum(0)
        n[it["a"]] += it["m"].sum(0)

    def fin(s, s2, n):
        mean = s / np.maximum(n, 1)
        std = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0))
        return mean, np.where(std < 1e-4, 1.0, std)

    pm, ps = fin(s.sum(0), s2.sum(0), n.sum(0))
    if not per_avatar:
        return pm.astype(np.float32), ps.astype(np.float32)
    mean, std = fin(s, s2, n)
    have = n > 0
    mean = np.where(have, mean, pm)
    std = np.where(have, std, ps)
    mean[0], std[0] = pm, ps
    return mean.astype(np.float32), std.astype(np.float32)


def stats_for(mean, std, a):
    """(mean, std) [V] for avatar index a (a pooled [V] pair passes through;
    an index past the table, an avatar the checkpoint does not know, is the
    unknown avatar's row 0)."""
    if np.ndim(mean) == 1:
        return mean, std
    a = a if 0 <= a < len(mean) else 0
    return mean[a], std[a]


def pooled(mean, std, ck=None):
    """The pooled [V] statistics: what the metrics divide by, so that runs
    with different --norm are scored on one scale."""
    if ck is not None and ck.get("pooled_std") is not None:
        return ck["pooled_mean"], ck["pooled_std"]
    if np.ndim(mean) == 1:
        return mean, std
    return mean[0], std[0]


PROS_RMS0 = 1e-4        # floor of the log RMS
PROS_F0 = 150.0         # Hz, the centre of the log f0 feature


def pros_feat(p):
    """[F, 3] features of flow.beat2's prosody (rms, f0 Hz, voiced), at the
    frame rate: log RMS (about 0 in speech, -2 in silence), log f0 over
    150 Hz scaled by 0.4 (0 where unvoiced), voiced."""
    p = np.asarray(p, np.float32)
    v = (p[:, 2] > 0.5).astype(np.float32)
    f0 = np.log(np.maximum(p[:, 1], 1.0) / PROS_F0) / 0.4 * v
    return np.stack([(np.log(np.maximum(p[:, 0], 0) + PROS_RMS0) + 5) / 2,
                     f0, v], 1).astype(np.float32)


def last_valid(y, m):
    """[V] the last frame of y [H, V] where m [H, V] holds, per variable
    (0 for a variable with none): the anchor of --target delta."""
    out = np.zeros(y.shape[1], np.float32)
    for k in range(y.shape[1]):
        i = np.flatnonzero(m[:, k])
        if len(i):
            out[k] = y[i[-1], k]
    return out


def corrupt_history(y, m, rng, noise=0.1, drop_frame=0.1, drop_var=0.1,
                    drop_all=0.1):
    """(y, m) of a history [H, V] as the head will see it when it rolls out
    on its own frames (LiveGesture's autoregressive masking): with
    probability drop_all no history at all; else each variable dropped
    over the whole history with drop_var, each frame with drop_frame, and
    Gaussian noise of a random scale up to `noise` (normalised units) on
    what is left. m says what is kept."""
    m = m.copy()
    if rng.random() < drop_all:
        m[:] = False
    else:
        m &= rng.random(m.shape[1]) >= drop_var
        m &= (rng.random(len(m)) >= drop_frame)[:, None]
    y = y + rng.normal(size=y.shape) * (noise * rng.random())
    return (y * m).astype(np.float32), m


def split_ids(data, frac=0.1, seed=0):
    """Train / eval row indices by session (flow.train.sft's rule)."""
    try:
        from .train.sft import held_out
    except ImportError:
        from flow.train.sft import held_out
    tr, ev = [], []
    for i, r in enumerate(data.rows):
        (ev if held_out(r["id"].rsplit("/", 1)[0], frac, seed)
         else tr).append(i)
    if not ev and len(tr) > 1:
        ev.append(tr.pop())
    return tr, ev


# --------------------------------------------------------------- model

_CLS = None


def model_cls():
    """The torch module, defined on first use (torch imported here)."""
    global _CLS
    if _CLS is not None:
        return _CLS
    import torch
    from torch import nn

    def sinus(t, d):
        """[..., d] sinusoidal features of t in seconds (periods 0.05 s to
        ~60 s)."""
        half = d // 2
        f = torch.exp(torch.linspace(math.log(2 * math.pi / 0.05),
                                     math.log(2 * math.pi / 60.0), half,
                                     device=t.device))
        a = t[..., None] * f
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    class ActionHead(nn.Module):
        def __init__(self, d_in, n_vars, n_avatars, d=128, layers=2,
                     heads=4, lookahead=LOOKAHEAD, objective="mse",
                     dropout=0.1, history=0, hist_fps=30.0, prosody=False,
                     text=True, mix=0, target="abs"):
            super().__init__()
            self.cfg = dict(d_in=d_in, n_vars=n_vars, n_avatars=n_avatars,
                            d=d, layers=layers, heads=heads,
                            lookahead=lookahead, objective=objective,
                            dropout=dropout)
            # Options added later are written only when used, so a
            # checkpoint of the plain head keeps its original config.
            for k, v, dflt in (("history", history, 0),
                               ("hist_fps", hist_fps, 30.0),
                               ("prosody", prosody, False),
                               ("text", text, True), ("mix", mix, 0),
                               ("target", target, "abs")):
                if v != dflt and (k != "hist_fps" or history):
                    self.cfg[k] = v
            self.d, self.lookahead, self.objective = d, lookahead, objective
            self.hist, self.hist_fps = int(history), float(hist_fps)
            self.use_pros, self.use_text = bool(prosody), bool(text)
            self.mix, self.delta = int(mix), target == "delta"
            if self.delta and not self.hist:
                raise ValueError("--target delta needs a history")
            if self.use_text:
                if self.mix:
                    if d_in % self.mix:
                        raise ValueError(f"{d_in} states do not split into "
                                         f"{self.mix} layers")
                    self.mix_logits = nn.Parameter(torch.zeros(self.mix))
                    self.tok = nn.Sequential(nn.LayerNorm(d_in // self.mix),
                                             nn.Linear(d_in // self.mix, d))
                else:
                    self.tok = nn.Sequential(nn.LayerNorm(d_in),
                                             nn.Linear(d_in, d))
            self.null = nn.Parameter(torch.zeros(1, 1, d))
            self.avatar = nn.Embedding(n_avatars, d)
            self.avail = nn.Linear(n_vars, d)
            self.q_time = nn.Linear(d, d)
            self.k_time = nn.Linear(d, d)
            if self.hist:
                self.hist_in = nn.Linear(2 * n_vars, d)
                self.hist_type = nn.Parameter(torch.zeros(1, 1, d))
            if self.use_pros:
                self.pros_in = nn.Sequential(nn.Linear(3, d), nn.SiLU(),
                                             nn.Linear(d, d))
                self.pros_missing = nn.Parameter(torch.zeros(d))
            if objective == "fm":
                self.x_in = nn.Linear(n_vars, d)
                self.tau = nn.Linear(d, d)
            layer = nn.TransformerDecoderLayer(
                d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(layer, layers)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_vars))

        def layer_weights(self):
            """[L] the softmax weights of the layer mix (None: no mix)."""
            if not (self.use_text and self.mix):
                return None
            return torch.softmax(self.mix_logits, 0).detach().cpu().numpy()

        def embed_tokens(self, h):
            """[B, N, d] the token states, the layers mixed by their
            softmax weights when --layer-mix (each layer normalised first:
            depths differ in scale)."""
            if not self.mix:
                return self.tok(h)
            B, N, _ = h.shape
            x = h.reshape(B, N, self.mix, -1)
            x = torch.nn.functional.layer_norm(x, x.shape[-1:])
            w = torch.softmax(self.mix_logits, 0)
            return self.tok((x * w[:, None]).sum(2))

        def forward(self, h, t_tok, tok_pad, t_frame, frame_pad, a, avail,
                    x_tau=None, tau=None, hist=None, hist_m=None, pros=None,
                    pros_m=None):
            """h [B, N, d_in], t_tok [B, N], tok_pad [B, N] (True: padding),
            t_frame [B, F], frame_pad [B, F], a [B], avail [B, V] ->
            [B, F, V] (normalised values, or the velocity for fm).
            hist [B, H, V] (the H frames before the chunk, normalised) and
            hist_m [B, H, V] (what is known of them); pros [B, F, 3]
            (pros_feat) and pros_m [B] (False: this row has none)."""
            B = avail.shape[0]
            dev = avail.device
            mem = [self.null.expand(B, 1, -1)]
            pad = [torch.zeros(B, 1, dtype=torch.bool, device=dev)]
            if self.hist:
                hm = hist_m.float()
                rel = (torch.arange(self.hist, device=dev) - self.hist
                       ).float() / self.hist_fps
                mem.append(self.hist_in(torch.cat([hist * hm, hm], -1))
                           + self.k_time(sinus(rel, self.d))[None]
                           + self.hist_type)
                pad.append(~hist_m.any(-1))
            n_pre = sum(x.shape[1] for x in mem)
            mmask = None
            if self.use_text:
                mem.append(self.embed_tokens(h)
                           + self.k_time(sinus(t_tok, self.d)))
                pad.append(tok_pad)
                # Frame f sees tokens up to lookahead seconds after it, and
                # the null token and the history always.
                late = t_tok[:, None, :] > (t_frame[:, :, None]
                                            + self.lookahead)
                late = late | tok_pad[:, None, :]
                late = torch.cat([torch.zeros(B, late.shape[1], n_pre,
                                              dtype=torch.bool, device=dev),
                                  late], 2)
                heads = self.dec.layers[0].multihead_attn.num_heads
                mmask = late.repeat_interleave(heads, 0)
            mem = torch.cat(mem, 1)
            pad = torch.cat(pad, 1)
            q = self.q_time(sinus(t_frame, self.d)) + (
                self.avatar(a) + self.avail(avail.float()))[:, None]
            if self.use_pros:
                if pros is None:
                    q = q + self.pros_missing
                else:
                    q = q + torch.where(pros_m[:, None, None],
                                        self.pros_in(pros),
                                        self.pros_missing)
            if self.objective == "fm":
                q = q + self.x_in(x_tau) + self.tau(sinus(tau, self.d))[:, None]
            x = self.dec(q, mem, memory_mask=mmask,
                         tgt_key_padding_mask=frame_pad,
                         memory_key_padding_mask=pad)
            return self.out(x)

    _CLS = ActionHead
    return _CLS


def batch(items, mean, std, chunk, rng, unknown_p=0.0, train=True, hist=0,
          f0s=None, corrupt=None, pros=False, pros_drop=0.0):
    """Random chunks of the items, padded, as tensors. hist: frames of
    history before each chunk (hist, hist_m; missing before the clip's
    start), corrupted by `corrupt` (corrupt_history's options) when
    training; anchor [B, V] is the last known history frame. f0s: the
    chunks' first frames [random when training, else 0]. pros: pros_feat
    of the chunk where the row has prosody (pros_drop: the share of
    training rows that lose it, to teach the "missing" embedding)."""
    import torch
    hs, tts, ys, ms, tfs, aa, av = [], [], [], [], [], [], []
    hy, hm, an, ps, pm = [], [], [], [], []
    for k, it in enumerate(items):
        F = it["y"].shape[0]
        n = min(chunk, F)
        if f0s is not None:
            f0 = int(f0s[k])
        else:
            f0 = int(rng.integers(0, F - n + 1)) if train else 0
        mu, sd = stats_for(mean, std, it["a"])
        yn = (it["y"] - mu) / sd
        tfs.append(np.arange(f0, f0 + n, dtype=np.float32) / it["fps"])
        ys.append(yn[f0:f0 + n])
        ms.append(it["m"][f0:f0 + n])
        hs.append(it["h"])
        tts.append(it["t_tok"])
        a = it["a"]
        if train and rng.random() < unknown_p:
            a = 0
        aa.append(a)
        av.append(it["avail"])
        if hist:
            V = yn.shape[1]
            a0 = max(0, f0 - hist)
            y_, m_ = yn[a0:f0], it["m"][a0:f0]
            if train and corrupt is not None and len(y_):
                y_, m_ = corrupt_history(y_, m_, rng, **corrupt)
                an.append(last_valid(yn[a0:f0], m_))
            else:
                an.append(last_valid(y_, m_))
            Y_ = np.zeros((hist, V), np.float32)
            M_ = np.zeros((hist, V), bool)
            if len(y_):
                Y_[hist - len(y_):], M_[hist - len(y_):] = y_, m_
            hy.append(Y_)
            hm.append(M_)
        if pros:
            ok = it.get("pros") is not None and not (
                train and rng.random() < pros_drop)
            pm.append(ok)
            ps.append(pros_feat(it["pros"][f0:f0 + n]) if ok
                      else np.zeros((n, 3), np.float32))
    B = len(items)
    N = max(len(x) for x in tts)
    Fm = max(len(x) for x in tfs)
    D, V = hs[0].shape[1], ys[0].shape[1]
    H = np.zeros((B, max(N, 1), D), np.float32)
    T = np.zeros((B, max(N, 1)), np.float32)
    TP = np.ones((B, max(N, 1)), bool)
    Y = np.zeros((B, Fm, V), np.float32)
    M = np.zeros((B, Fm, V), bool)
    TF = np.zeros((B, Fm), np.float32)
    FP = np.ones((B, Fm), bool)
    P = np.zeros((B, Fm, 3), np.float32)
    for b in range(B):
        n = len(tts[b])
        H[b, :n], T[b, :n], TP[b, :n] = hs[b], tts[b], False
        f = len(tfs[b])
        Y[b, :f], M[b, :f], TF[b, :f], FP[b, :f] = ys[b], ms[b], tfs[b], False
        if pros:
            P[b, :f] = ps[b]
    t = torch.from_numpy
    out = {"h": t(H), "t_tok": t(T), "tok_pad": t(TP), "t_frame": t(TF),
           "frame_pad": t(FP), "a": torch.tensor(aa, dtype=torch.long),
           "avail": t(np.stack(av)), "y": t(Y), "m": t(M)}
    if hist:
        out.update(hist=t(np.stack(hy)), hist_m=t(np.stack(hm)),
                   anchor=t(np.stack(an).astype(np.float32)))
    if pros:
        out.update(pros=t(P), pros_m=torch.tensor(pm, dtype=torch.bool))
    return out


_KEYS = ("h", "t_tok", "tok_pad", "t_frame", "frame_pad", "a", "avail")


def inputs(b):
    """The model's keyword arguments in a batch."""
    kw = {k: b[k] for k in _KEYS}
    for k in ("hist", "hist_m", "pros", "pros_m"):
        if k in b:
            kw[k] = b[k]
    return kw


def loss_of(model, b, gen=None):
    import torch
    y = b["y"] - b["anchor"][:, None] if model.delta else b["y"]
    if model.objective == "fm":
        x1 = y
        x0 = torch.randn(x1.shape, generator=gen)
        tau = torch.rand(x1.shape[0], generator=gen)
        xt = (1 - tau[:, None, None]) * x0 + tau[:, None, None] * x1
        pred = model(**inputs(b), x_tau=xt, tau=tau)
        tgt = x1 - x0
    else:
        pred = model(**inputs(b))
        tgt = y
    m = b["m"].float()
    return ((pred - tgt) ** 2 * m).sum() / m.sum().clamp(min=1)


def sample(model, b, gen, steps):
    """[F, V] the model's chunk for a batch of one (normalised; the
    anchor added back under --target delta)."""
    import torch
    with torch.no_grad():
        if model.objective == "fm":
            x = torch.randn(b["y"].shape, generator=gen)
            for k in range(steps):
                tau = torch.full((1,), k / steps)
                x = x + model(**inputs(b), x_tau=x, tau=tau) / steps
            y = x[0].numpy()
        else:
            y = model(**inputs(b))[0].numpy()
    if model.delta:
        y = y + b["anchor"][0].numpy()
    return y


def predict(model, it, mean, std, chunk=CHUNK, steps=8, seed=0, given=None):
    """[F, V] in natural units. A head without history: chunks of `chunk`
    frames, half overlap, cross-faded. A head with history rolls out on its
    own frames from the first `given` frames of the clip (default: its
    history; they come back as they are, 0: from nothing, no history known)
    chunk by chunk, committing half a chunk each time."""
    import torch
    F, V = it["y"].shape
    mu, sd = stats_for(mean, std, it["a"])
    gen = torch.Generator().manual_seed(int(seed))
    model.eval()
    hop = max(1, chunk // 2)
    pros = model.use_pros and it.get("pros") is not None
    if model.hist:
        H = model.hist
        g = min(model.hist if given is None else int(given), F)
        yn = (it["y"] - mu) / sd
        buf = np.zeros((F, V), np.float32)
        ok = np.zeros((F, V), bool)
        buf[:g], ok[:g] = yn[:g], it["m"][:g]
        f = g
        while f < F:
            n = min(chunk, F - f)
            sub = dict(it, y=it["y"][f:f + n], m=it["m"][f:f + n],
                       pros=it["pros"][f:f + n] if pros else None)
            b = batch([sub], mean, std, chunk, None, train=False,
                      pros=model.use_pros)
            b["t_frame"] = b["t_frame"] + f / it["fps"]
            a0 = max(0, f - H)
            hy = np.zeros((H, V), np.float32)
            hm = np.zeros((H, V), bool)
            hy[H - (f - a0):], hm[H - (f - a0):] = buf[a0:f], ok[a0:f]
            b["hist"], b["hist_m"] = torch.from_numpy(hy)[None], \
                torch.from_numpy(hm)[None]
            b["anchor"] = torch.from_numpy(last_valid(hy, hm))[None]
            y = sample(model, b, gen, steps)
            c = n if f + n >= F else hop
            buf[f:f + c] = y[:c]
            ok[f:f + c] = it["avail"][None]
            f += c
        out = buf * sd + mu
        out[:g] = it["y"][:g]
        return out
    out = np.zeros((F, V), np.float32)
    wsum = np.zeros((F, 1), np.float32)
    starts = list(range(0, max(1, F - chunk + 1), hop))
    if starts[-1] + chunk < F:
        starts.append(F - chunk)
    for f0 in starts:
        n = min(chunk, F - f0)
        sub = dict(it, y=it["y"][f0:f0 + n], m=it["m"][f0:f0 + n],
                   pros=it["pros"][f0:f0 + n] if pros else None)
        b = batch([sub], mean, std, chunk, np.random.default_rng(0),
                  train=False, pros=model.use_pros)
        b["t_frame"] = b["t_frame"] + f0 / it["fps"]
        y = sample(model, b, gen, steps)
        w = np.minimum(1.0, np.minimum(np.arange(1, n + 1),
                                       np.arange(n, 0, -1)) / max(1, hop / 2))
        if f0 == 0:
            w[:hop] = 1.0
        if f0 + n == F:
            w[-hop:] = 1.0
        out[f0:f0 + n] += (y * sd + mu) * w[:, None]
        wsum[f0:f0 + n] += w[:, None]
    return out / np.maximum(wsum, 1e-6)


# ------------------------------------------------------------- training

def save(path, model, data, mean, std, extra=None):
    import torch
    torch.save({"state": model.state_dict(), "config": model.cfg,
                "vocab": data.vocab, "avatars": data.avatars,
                "mean": mean, "std": std, "meta": data.meta,
                **(extra or {})}, path)


def resolve(path):
    """A checkpoint file as given, or a run folder's head.best.pt (head.pt
    when the run has no best: runs from before)."""
    if os.path.isdir(path):
        for n in ("head.best.pt", "head.pt"):
            if os.path.exists(os.path.join(path, n)):
                return os.path.join(path, n)
        raise SystemExit(f"actionhead: no head.best.pt / head.pt in {path}")
    return path


def load(path):
    import torch
    ck = torch.load(resolve(path), map_location="cpu", weights_only=False)
    m = model_cls()(**ck["config"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def build(o, data, d_in, fps):
    """The model the options ask for."""
    n_layers = len(data.meta.get("layers") or [0])
    return model_cls()(
        d_in, len(data.vocab), len(data.avatars), o.dim, o.layers, o.heads,
        o.lookahead, o.objective, history=int(round(o.history * fps)),
        hist_fps=fps, prosody=o.prosody, text=not o.no_text,
        mix=n_layers if o.layer_mix else 0, target=o.target)


def train(o, log=log) -> dict:
    import torch
    torch.manual_seed(o.seed)
    data = Data(o.export, o.hidden, prosody=o.prosody)
    tr, ev = split_ids(data, o.eval_frac, o.seed)
    if not tr:
        raise SystemExit("actionhead: no training turns")
    pool = norm_stats(data, tr)
    mean, std = norm_stats(data, tr, True) if o.norm == "avatar" else pool
    first = data.item(0)
    d_in = first["h"].shape[1]
    model = build(o, data, d_in, first["fps"])
    H = model.hist
    corrupt = dict(noise=o.hist_noise, drop_frame=o.hist_drop,
                   drop_var=o.hist_drop, drop_all=o.hist_none) if H else None
    opt = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=0.01)
    rng = np.random.default_rng(o.seed)
    gen = torch.Generator().manual_seed(o.seed)
    os.makedirs(o.out, exist_ok=True)
    opts = {k: v for k, v in vars(o).items()
            if isinstance(v, (int, float, str, bool))}
    log(f"actionhead: {len(tr)} train / {len(ev)} eval turns, "
        f"{len(data.vocab)} variables, {len(data.avatars) - 1} avatars, "
        f"token states {d_in}, {sum(p.numel() for p in model.parameters())} "
        f"parameters, {o.objective}"
        + (f", history {H} frames ({o.target})" if H else "")
        + (", prosody" if o.prosody else "")
        + ("" if not o.no_text else ", no text")
        + (f", layer mix of {model.mix}" if model.mix else "")
        + (", per-avatar norm" if o.norm == "avatar" else ""))
    extra = {"pooled_mean": pool[0], "pooled_std": pool[1],
             "norm": o.norm, "opts": opts}

    def eval_loss():
        # With a history the chunk is the middle one, after a clean
        # history; without, the first (as it has always been).
        out = []
        for i in ev:
            it = data.item(i)
            f0 = [max(0, (len(it["y"]) - o.chunk) // 2)] if H else None
            out.append(float(loss_of(model, batch(
                [it], mean, std, o.chunk, rng, train=False, hist=H, f0s=f0,
                pros=o.prosody), gen)))
        return out

    best, hist, bad, last = math.inf, [], 0, {}
    t0 = time.monotonic()
    for step in range(1, o.steps + 1):
        model.train()
        pick = rng.choice(tr, size=min(o.batch, len(tr)), replace=False)
        b = batch([data.item(int(i)) for i in pick], mean, std, o.chunk, rng,
                  UNKNOWN_P, hist=H, corrupt=corrupt, pros=o.prosody,
                  pros_drop=o.pros_drop)
        loss = loss_of(model, b, gen)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % o.eval_every == 0 or step == o.steps:
            model.eval()
            with torch.no_grad():
                el = eval_loss() if ev else []
            e = float(np.mean(el)) if el else float(loss)
            w = model.layer_weights()
            hist.append({"step": step, "train": round(float(loss.detach()),
                                                      5),
                         "eval": round(e, 5)})
            if w is not None:
                hist[-1]["layer_mix"] = [round(float(x), 4) for x in w]
            log(f"actionhead: step {step} loss {float(loss.detach()):.4f} eval "
                f"{e:.4f} ({time.monotonic() - t0:.0f}s)"
                + ("" if w is None else " layer weights " + " ".join(
                    f"{x:.3f}" for x in w)))
            last = dict(extra, eval_ids=[data.rows[i]["id"] for i in ev],
                        history=hist, step=step)
            if e < best:
                best, bad = e, 0
                save(os.path.join(o.out, "head.best.pt"), model, data, mean,
                     std, dict(last, best=True))
            else:
                bad += 1
            if o.patience and bad >= o.patience:
                log(f"actionhead: no better eval loss in {bad} evaluations; "
                    f"stopping at step {step}")
                break
    # The last step's, whatever it was: head.best.pt is the one to use.
    save(os.path.join(o.out, "head.pt"), model, data, mean, std, last)
    w = model.layer_weights()
    with open(os.path.join(o.out, "train.json"), "w", encoding="utf-8") as f:
        json.dump({"best_eval": best, "history": hist,
                   "vocab": data.vocab, "avatars": data.avatars,
                   "layer_mix": None if w is None else [float(x) for x in w],
                   "opts": opts}, f, indent=1)
    return {"best_eval": best, "history": hist}


# ----------------------------------------------------------------- eval

SCORES = ("n", "se", "sen", "rel", "vel", "nv", "vp", "vt", "vvp", "vvt")


def motion_scores(x, y, m, sd):
    """Per-variable sums ([V] arrays, in units of sd, the normalising std,
    but `se` natural) of how a trajectory x [F, V] differs from the target
    y, where m [F, V] says y is known:
      n, se, sen   frames; squared error; squared error / sd^2
      rel          squared error of x and y each minus its own mean over
                   the known frames (the motion around the posture)
      vel, nv      squared error of the frame-to-frame differences (both
                   frames known), their count
      vp, vt       sum of squares of x and y minus their means (the motion
                   each has: their ratio is the predicted to true std)
      vvp, vvt     the same of the differences."""
    mf = m.astype(np.float64)
    xn, yn = x / sd, y / sd
    n = mf.sum(0)
    mu_x = (xn * mf).sum(0) / np.maximum(n, 1)
    mu_y = (yn * mf).sum(0) / np.maximum(n, 1)
    dx, dy = (xn - mu_x) * mf, (yn - mu_y) * mf
    mv = (m[1:] & m[:-1]).astype(np.float64)
    vx, vy = np.diff(xn, axis=0) * mv, np.diff(yn, axis=0) * mv
    return {"n": n, "se": ((x - y) ** 2 * mf).sum(0),
            "sen": ((xn - yn) ** 2 * mf).sum(0),
            "rel": ((dx - dy) ** 2).sum(0), "vel": ((vx - vy) ** 2).sum(0),
            "nv": mv.sum(0), "vp": (dx ** 2).sum(0), "vt": (dy ** 2).sum(0),
            "vvp": (vx ** 2).sum(0), "vvt": (vy ** 2).sum(0)}


def _ratio(p, t):
    """sqrt(p / t), 0 when p is 0, None (JSON null) when t is 0 too."""
    if t <= 1e-12:
        return None
    return float(math.sqrt(max(p, 0) / t))


def score_summary(acc):
    """Totals of a method's summed motion_scores: normalised MSE, relative
    MSE, velocity MSE (all over the variables' own variance, per known
    value) and the predicted / true std of the motion and of its velocity."""
    n = acc["n"].sum()
    nv = acc["nv"].sum()
    return {"normalised_mse": float(acc["sen"].sum() / max(n, 1)),
            "relative_mse": float(acc["rel"].sum() / max(n, 1)),
            "velocity_mse": float(acc["vel"].sum() / max(nv, 1)),
            "std_ratio": _ratio(acc["vp"].sum(), acc["vt"].sum()),
            "velocity_std_ratio": _ratio(acc["vvp"].sum(),
                                         acc["vvt"].sum())}


def _box(x, k=3):
    if k <= 1 or len(x) < 2:
        return np.asarray(x, np.float64)
    x = np.asarray(x, np.float64)
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"),
                       np.ones(k) / k, "valid")


def _peaks(x, thr):
    """Indices of the local maxima of x above thr."""
    if len(x) < 3:
        return np.zeros(0, int)
    c = x[1:-1]
    return np.flatnonzero((c >= x[:-2]) & (c > x[2:]) & (c > thr)) + 1


def beat_stats(speed, rms, fps, tol=0.2):
    """How a motion speed [F] lines up with the audio's accents (the
    RMS's changes, rising or falling: |d rms| smoothed, so a copy of the
    RMS has exactly the accents' speed): {corr, onsets, hit, peaks, near}.
      corr    Pearson correlation of the smoothed speed and accent strength
              (None when either is constant)
      onsets  accent peaks (above 30% of the strongest), hit: those with a
              speed peak within +-tol s (BEAT's beat consistency, hit /
              onsets)
      peaks   speed peaks (above the mean speed), near: those within +-tol
              s of an accent (near / peaks: how much of the motion's
              beat is the audio's)."""
    rms = np.asarray(rms, np.float64)
    acc = _box(np.abs(np.diff(rms, prepend=rms[:1])))
    sp = _box(speed)
    out = {"corr": None, "onsets": 0, "hit": 0, "peaks": 0, "near": 0}
    if acc.std() > 1e-9 and sp.std() > 1e-9:
        out["corr"] = float(np.corrcoef(sp, acc)[0, 1])
    ons = _peaks(acc, 0.3 * acc.max()) if acc.max() > 1e-9 else []
    pks = _peaks(sp, sp.mean()) if sp.max() > 1e-9 else []
    w = tol * fps + 1e-9
    out["onsets"], out["peaks"] = len(ons), len(pks)
    out["hit"] = int(sum(np.any(np.abs(pks - o) <= w) for o in ons)) \
        if len(pks) else 0
    out["near"] = int(sum(np.any(np.abs(ons - p) <= w) for p in pks)) \
        if len(ons) else 0
    return out


def motion_speed(y, m, cols, sd, fps):
    """[F-1] the speed of a group of variables (their columns), in
    normalised units per second (None: the group is not known all along)."""
    if not cols or not m[:, cols].all():
        return None
    d = np.diff(y[:, cols] / sd[cols], axis=0) * fps
    return np.sqrt((d ** 2).sum(1))


def beat_scores(series, groups, rms, fps, tol=0.2):
    """Pooled beat_stats over segments. series {name: [(y [F, V], m, sd [V])]
    per segment} (y natural, sd what it is divided by), groups {name:
    columns}, rms [segments][F]: for each series and group {corr (mean over
    segments where defined), consistency (hit / onsets), precision (near /
    peaks), segments}."""
    out = {}
    for name, segs in series.items():
        out[name] = {}
        for g, cols in groups.items():
            cs, tot = [], {"onsets": 0, "hit": 0, "peaks": 0, "near": 0}
            n = 0
            for (y, m, sd), r in zip(segs, rms):
                sp = motion_speed(y, m, cols, sd, fps)
                if sp is None or len(sp) < 5:
                    continue
                st = beat_stats(sp, r[1:], fps, tol)
                n += 1
                if st["corr"] is not None:
                    cs.append(st["corr"])
                for k in tot:
                    tot[k] += st[k]
            out[name][g] = {
                "corr": float(np.mean(cs)) if cs else None,
                "consistency": tot["hit"] / tot["onsets"] if tot["onsets"]
                else None,
                "precision": tot["near"] / tot["peaks"] if tot["peaks"]
                else None, "segments": n}
    return out


def evaluate(o, log=log) -> dict:
    ckpt = resolve(o.ckpt)
    model, ck = load(ckpt)
    data = Data(o.export, o.hidden, ck["vocab"], ck["avatars"], prosody=True)
    ids = set(o.ids) if o.ids else set(ck.get("eval_ids") or [])
    rows = [i for i, r in enumerate(data.rows) if not ids or r["id"] in ids]
    mean, std = ck["mean"], ck["std"]
    pmean, pstd = pooled(mean, std, ck)
    markers = {}
    if o.markers:
        markers = marker_trajs(o.markers, o.data, data, rows)
    V = len(data.vocab)
    skip_s = model.hist / model.hist_fps if o.skip is None else o.skip
    acc = {}
    series = {"gt": [], "head": []}
    rms = []
    nrows, fps = 0, 30.0
    for i in rows:
        it = data.item(i)
        fps = it["fps"]
        y, m = it["y"], it["m"]
        st = int(round(skip_s * it["fps"]))
        if len(y) - st < 5:
            continue
        nrows += 1
        p = predict(model, it, mean, std, o.chunk, o.steps, o.seed,
                    given=o.given)
        # Every score on the frames after `st` (the head's own history
        # given as ground truth is not scored), on the pooled std.
        meths = {"head": p, "rest": np.repeat(y[:1], len(y), 0),
                 "mean": np.repeat(pmean[None], len(y), 0)}
        if st:
            meths["hold"] = np.repeat(y[st - 1:st], len(y), 0)
        mk = markers.get(it["id"])
        if mk is not None:
            meths["markers"] = mk[:len(y)]
        mm = m[st:]
        for k, x in meths.items():
            sc = motion_scores(x[st:], y[st:], mm, pstd)
            a = acc.setdefault(k, {q: np.zeros(V) for q in SCORES})
            for q in SCORES:
                a[q] += sc[q]
        if it.get("pros") is not None:
            series["gt"].append((y[st:], mm, pstd))
            series["head"].append((p[st:], mm, pstd))
            rms.append(it["pros"][st:, 0])
    per = {}
    ext = {}
    for v, k in data.vi.items():
        n = acc["head"]["n"][k]
        if not n:
            continue
        per[v] = {c: round(float(a["se"][k] / n), 6) for c, a in
                  acc.items() if c != "markers"}
        ext[v] = {}
        for q, f in (("relative_mse", lambda a: a["rel"][k] / n),
                     ("velocity_mse", lambda a: a["vel"][k] / max(
                         a["nv"][k], 1)),
                     ("std_ratio", lambda a: _ratio(a["vp"][k], a["vt"][k])),
                     ("velocity_std_ratio", lambda a: _ratio(
                         a["vvp"][k], a["vvt"][k]))):
            ext[v][q] = {c: (None if f(a) is None else round(float(f(a)), 5))
                         for c, a in acc.items()}
        if "markers" in acc and acc["markers"]["n"][k]:
            per[v]["markers"] = round(float(
                acc["markers"]["se"][k] / acc["markers"]["n"][k]), 6)
    summ = {c: score_summary(a) for c, a in acc.items()}
    groups = {"hands": [k for v, k in data.vi.items()
                        if v.startswith("hand.")],
              "head": [k for v, k in data.vi.items()
                       if v.startswith("head.")]}
    beat = beat_scores(series, groups, rms, fps) if rms else {}
    tot = {c: round(s_["normalised_mse"], 5) for c, s_ in summ.items()}
    w = model.layer_weights()
    out = {"turns": nrows, "ckpt": ckpt, "skip_seconds": skip_s,
           "history_seconds": model.hist / model.hist_fps,
           "normalised_mse": tot, "per_variable": per,
           "relative_mse": {c: s_["relative_mse"] for c, s_ in summ.items()},
           "velocity_mse": {c: s_["velocity_mse"] for c, s_ in summ.items()},
           "std_ratio": {c: s_["std_ratio"] for c, s_ in summ.items()},
           "velocity_std_ratio": {c: s_["velocity_std_ratio"]
                                  for c, s_ in summ.items()},
           "per_variable_motion": ext, "beat": beat,
           "layer_mix": None if w is None else {
               "layers": data.meta.get("layers"),
               "weights": [round(float(x), 4) for x in w]},
           "options": ck.get("config", {}), "norm": ck.get("norm", "pooled")}
    log(f"actionhead: {nrows} turns; normalised MSE " + ", ".join(
        f"{k} {v}" for k, v in tot.items()))
    for v, d in per.items():
        log(f"  {v:16s} " + "  ".join(f"{k} {x:.5g}" for k, x in d.items()))
    for line in summary_lines(out):
        log(line)
    if o.json:
        with open(o.json, "w", encoding="utf-8") as fh_:
            json.dump(out, fh_, indent=1)
    return out


def _f(x, w=8):
    return f"{'-':>{w}}" if x is None else f"{x:{w}.4f}"


def summary_lines(r):
    """The compact block of an evaluate result."""
    cols = [c for c in ("head", "rest", "mean", "hold", "markers")
            if c in r["normalised_mse"]]
    out = [f"summary: {r['turns']} eval turns, first {r['skip_seconds']:g} s "
           f"not scored, checkpoint {r['ckpt']}",
           f"{'':26s}" + "".join(f"{c:>8s}" for c in cols)]
    for lab, key in (("normalised MSE", "normalised_mse"),
                     ("relative MSE (motion)", "relative_mse"),
                     ("velocity MSE", "velocity_mse"),
                     ("std ratio (pred/true)", "std_ratio"),
                     ("velocity std ratio", "velocity_std_ratio")):
        out.append(f"{lab:26s}" + "".join(_f(r[key].get(c)) for c in cols))
    b = r.get("beat") or {}
    for g in ("hands", "head"):
        for k, lab in (("corr", "corr"), ("consistency", "hit"),
                       ("precision", "prec")):
            if b.get("gt", {}).get(g):
                out.append(f"{'beat ' + g + ' ' + lab:26s}"
                           + _f(b["head"][g][k]) + "  gt "
                           + _f(b["gt"][g][k]).strip())
    if r.get("layer_mix"):
        out.append("layer mix " + " ".join(
            f"{l}: {w:.3f}" for l, w in zip(r["layer_mix"]["layers"] or [],
                                            r["layer_mix"]["weights"])))
    return out


def summary_row(r):
    """{name: value} of the numbers a grid compares."""
    row = {}
    for key, lab in (("normalised_mse", "nmse"), ("relative_mse", "rel"),
                     ("velocity_mse", "vel"), ("std_ratio", "std")):
        for c in ("head", "rest", "hold"):
            if c in r[key] and (c == "head" or key in ("normalised_mse",
                                                       "relative_mse")):
                row[f"{lab}.{c}"] = r[key][c]
    b = r.get("beat") or {}
    for g in ("hands", "head"):
        if b.get("gt", {}).get(g):
            row[f"beat.{g}.corr"] = b["head"][g]["corr"]
            row[f"beat.{g}.corr.gt"] = b["gt"][g]["corr"]
    return row


def table(paths, log=log):
    """Eval JSONs side by side: one column per run."""
    runs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            runs.append((os.path.basename(os.path.dirname(p)) or p,
                         summary_row(json.load(f))))
    keys = [k for k in runs[0][1]] if runs else []
    for _n, r in runs:
        keys += [k for k in r if k not in keys]
    w = max([len(n) for n, _ in runs] + [9]) + 1
    lines = [f"{'':22s}" + "".join(f"{n:>{w}s}" for n, _ in runs)]
    for k in keys:
        lines.append(f"{k:22s}" + "".join(
            _f(r.get(k), w) if r.get(k) is not None else f"{'-':>{w}s}"
            for _n, r in runs))
    for ln in lines:
        log(ln)
    return lines


def marker_trajs(path, data_root, data, rows):
    """{id: [F, V]} another model's replies (flow.train.evalsft's jsonl)
    played on the avatar: the directed trajectory of its markers."""
    try:
        from . import dataset as ds
        from . import director as dr
    except ImportError:
        from flow import dataset as ds
        from flow import director as dr
    want = {data.rows[i]["id"]: data.rows[i] for i in rows}
    replies = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in want:
                replies[r["id"]] = r["reply_raw"]
    if not replies:
        return {}
    recs = {r["id"]: r for r in ds.iter_records(data_root)}
    opts = ds.run_opts(data_root)
    opts.tts = "none"
    reg = json.load(open(os.path.join(data_root, "avatars.json"),
                         encoding="utf-8"))
    avs, out = {}, {}
    for rid, raw in replies.items():
        name = want[rid]["avatar"]
        if name not in avs:
            avs[name] = ds.Avatar(dict(reg["avatars"][name]["spec"]), opts,
                                  {"eval": None}, lambda s: None)
        av = avs[name]
        p = av.parser("eval")
        p.feed(raw)
        p.close()
        reply = {"raw": raw, "spoken": p.spoken, "tags": p.tags}
        with av.lock:
            _f, arr = ds.play(av, reply, recs[rid]["seed"] if rid in recs
                              else 0)
        tr = arr[".directed"].astype(np.float32)
        F = want[rid]["frames"]
        y = np.zeros((F, len(data.vocab)), np.float32)
        for c, v in enumerate(av.traj_vars):
            k = data.vi.get(v)
            if k is None:
                continue
            col = np.nan_to_num(tr[:F, c])
            y[:len(col), k] = col
            if len(col) < F:
                y[len(col):, k] = col[-1] if len(col) else 0
        out[rid] = y
    return out


# --------------------------------------------------------------- export

def _smooth_variants(var, win=15):
    """Arm drawings by a running majority (no flicker between drawings)."""
    var = np.asarray(var)
    out = var.copy()
    h = win // 2
    for i in range(len(var)):
        w = var[max(0, i - h):i + h + 1]
        vals, cnt = np.unique(w, return_counts=True)
        out[i] = vals[np.argmax(cnt)]
    return out


def to_tokens(bm, traj, vars_, fps, every=3, variant=None, log=log):
    """Move tokens (flow.moves) playing this canonical trajectory on bm:
    ([tokens], seconds). traj [F, len(vars_)]; NaN / absent variables are
    left to the solver's prior."""
    vrm = getattr(bm, "kind", None) == "vrm"
    F = len(traj)
    # Uniform keyframes (the token's dt); the last one holds the last frame.
    nk = int(math.ceil((F - 1) / every)) + 1 if F > 1 else 1
    keys = [min(k * every, F - 1) for k in range(nk)]
    cols = [(c, v) for c, v in enumerate(vars_)
            if not v.startswith(EXCLUDE) and v not in HIPS
            and v in bm.fidx]
    cur = bm.default_values()
    parts = None if vrm else dict(bm.blob.get("rest_parts", {}))
    sol = []
    t0 = time.monotonic()
    for f in keys:
        tg = {v: float(traj[f, c]) for c, v in cols
              if not np.isnan(traj[f, c])}
        kw = {"via": False} if vrm else {"restart": False}
        var_f = None if variant is None else int(variant[f])
        r = bm.solve(tg, current=cur, variant=var_f, parts=parts, **kw)
        cur = dict(cur, **r["params"])
        parts = r.get("parts") or parts
        sol.append(r)
    vs = np.array([r.get("variant") or 0 for r in sol])
    if not vrm and variant is None and len(set(vs.tolist())) > 1:
        sm = _smooth_variants(vs, max(3, int(round(0.5 * fps / every))))
        cur = bm.default_values()
        for i, f in enumerate(keys):
            if sm[i] != vs[i]:
                tg = {v: float(traj[f, c]) for c, v in cols
                      if not np.isnan(traj[f, c])}
                sol[i] = bm.solve(tg, current=cur, variant=int(sm[i]),
                                  restart=False)
            cur = dict(cur, **sol[i]["params"])
        vs = sm
    log(f"actionhead: {len(keys)} solves in {time.monotonic() - t0:.1f}s")
    hips = [(c, v) for c, v in enumerate(vars_) if v in HIPS] if vrm else []
    pids = sorted({p for r in sol for p in r["params"]})
    dt = every / fps
    toks = []
    # One token per run of one arm drawing (Live2D), cross-faded.
    runs, s = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or vs[i] != vs[s]:
            runs.append((s, i))
            s = i
    seconds = (F - 1) / fps
    for a, b in runs:
        lo, hi = max(0, a - 1), min(len(keys), b + 1)
        absd = {}
        for p in pids:
            col = []
            last = None
            for r in sol[lo:hi]:
                x = r["params"].get(p, last)
                last = x
                col.append(x)
            first = next((x for x in col if x is not None), None)
            if first is None:
                continue
            absd[p] = [round(float(first if x is None else x), 5)
                       for x in col]
        for c, v in hips:
            absd[v] = [round(float(np.nan_to_num(traj[keys[j], c])), 5)
                       for j in range(lo, hi)]
        t_start = lo * dt
        dur = (hi - 1 - lo) * dt + 1.0 / fps
        ramp = 1.0 / fps if (a == 0 and b == len(keys)) else dt
        toks.append({"t": round(t_start, 4), "type": mv.MOVE, "role": mv.MOVE,
                     "name": "actionhead", "duration": round(dur, 4),
                     "ramp_in": ramp if a else 1.0 / fps,
                     "ramp_out": ramp if b < len(keys) else 1.0 / fps,
                     "dt": dt, "add": {}, "abs": absd, "step": [],
                     "parts": {} if vrm else dict(sol[a].get("parts") or {}),
                     "parts_from": {}, "groups": ["arms"],
                     "src": "director"})
    return toks, seconds


def play_tokens(bm, toks, fps, n):
    """[n, ...] canonical states of the tokens played over the rest (no
    idle): the exporter's own check."""
    vrm = getattr(bm, "kind", None) == "vrm"
    rest = bm.default_values()
    rest_parts = {} if vrm else dict(bm.blob.get("rest_parts", {}))
    lay = mv.layer_for(toks, fps, rest, rest_parts)
    out = []
    for f in range(n):
        over, parts = dict(rest), dict(rest_parts)
        if lay is not None:
            over, parts = lay.apply(f, over, parts)
        out.append(bm.perceive(over) if vrm else
                   bm.perceive(over, parts=parts))
    return out


def write_clip(out, bm, body_path, traj, vars_, fps, every=3, variant=None,
               seconds=None, log=log):
    """tokens.json (and states.json on a VRM) for one trajectory."""
    os.makedirs(out, exist_ok=True)
    toks, sec = to_tokens(bm, traj, vars_, fps, every, variant, log)
    sec = max(sec, seconds or 0)
    with open(os.path.join(out, "tokens.json"), "w", encoding="utf-8") as f:
        json.dump({"seconds": round(sec, 3), "fps": fps, "tokens": toks}, f)
    if getattr(bm, "kind", None) == "vrm":
        try:
            from . import stream as fs
            from . import vrm as fv
        except ImportError:
            from flow import stream as fs
            from flow import vrm as fv
        rig = fv.VRMRig(bm, "default", fps, "calm", log=lambda s: None)
        st = fs.MotionStream(rig, seed=0, idle=False, autonomic=False,
                             log=lambda s: None)
        st.direct(toks)
        states = fv.stream_states(st, bm, int(round(sec * fps)) + 1)
        fv.save_states(os.path.join(out, "states.json"), states, fps,
                       body_path)
    return toks, sec


def export(o, log=log) -> dict:
    """The prediction and the directed trajectory of one turn, side by
    side, as clips for the renderers."""
    model, ck = load(o.ckpt)
    data = Data(o.export, o.hidden, ck["vocab"], ck["avatars"],
                prosody=True)
    i = next((k for k, r in enumerate(data.rows) if r["id"] == o.id), None)
    if i is None:
        raise SystemExit(f"actionhead: {o.id} not in the export / hidden")
    it = data.item(i)
    row = data.rows[i]
    pred = predict(model, it, ck["mean"], ck["std"], o.chunk, o.steps, o.seed)
    reg = json.load(open(os.path.join(o.data, "avatars.json"),
                         encoding="utf-8"))
    spec = reg["avatars"][row["avatar"]]["spec"]
    bm = fb.load(spec["body"])
    with np.load(data.npz_path) as z:
        directed = z[row["key"] + ".traj"].astype(np.float32)
        variant = z[row["key"] + ".variant"] if row["key"] + ".variant" \
            in z.files else None
    # The prediction in the body's own column order.
    P = np.full((len(pred), len(row["vars"])), np.nan, np.float32)
    for c, v in enumerate(row["vars"]):
        k = data.vi.get(v)
        if k is not None:
            P[:, c] = pred[:, k]
    sec = row["frames"] / row["fps"]
    res = {}
    for name, tr, var in (("pred", P, None), ("directed", directed, variant)):
        d = os.path.join(o.out, name)
        write_clip(d, bm, spec["body"], tr, row["vars"], row["fps"], o.every,
                   var, sec, log)
        res[name] = d
    cmds = render_commands(spec, res, sec)
    with open(os.path.join(o.out, "render.sh"), "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -e\n" + "\n".join(cmds) + "\n")
    log(f"actionhead: {o.id} -> {o.out} (pred/, directed/); render with "
        f"bash {os.path.join(o.out, 'render.sh')}")
    return res


def render_commands(spec, dirs, seconds):
    out = []
    for name, d in dirs.items():
        if spec["kind"] == "vrm":
            c = (f"python -m flow.vrmrender --states {d}/states.json --body "
                 f"{spec['body']} --out {d}/preview.mp4")
            if spec.get("vrm"):
                c += f" --vrm {spec['vrm']}"
        else:
            c = (f"xvfb-run -a -s '-screen 0 1024x1024x24' python -m "
                 f"flow.runtime --model {spec.get('model')} --rig "
                 f"{spec.get('rig')} --events {spec.get('events') or 'default'}"
                 f" --probe-dir {spec.get('probe_dir') or 'out'} --tokens-in "
                 f"{d}/tokens.json --seconds {seconds:.2f} --no-idle --out "
                 f"{d}/render")
        out.append(c)
    return out


# ----------------------------------------------------------------- cli

def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("table", help="eval JSONs side by side")
    pt.add_argument("jsons", nargs="+")
    for name in ("train", "eval", "export"):
        p = sub.add_parser(name)
        p.add_argument("--export", required=True,
                       help="flow.dataexport action's --out stem")
        p.add_argument("--hidden", required=True,
                       help="flow.train.hidden's --out")
        p.add_argument("--chunk", type=int, default=CHUNK)
        p.add_argument("--steps", type=int, default=8 if name != "train"
                       else 2000, help="train: optimizer steps; eval / "
                       "export: fm sampling steps")
        p.add_argument("--seed", type=int, default=0)
        if name == "train":
            p.add_argument("--out", required=True)
            p.add_argument("--dim", type=int, default=128)
            p.add_argument("--layers", type=int, default=2)
            p.add_argument("--heads", type=int, default=4)
            p.add_argument("--lookahead", type=float, default=LOOKAHEAD)
            p.add_argument("--objective", default="mse",
                           choices=("mse", "fm"))
            p.add_argument("--lr", type=float, default=3e-4)
            p.add_argument("--batch", type=int, default=16)
            p.add_argument("--eval-every", type=int, default=100)
            p.add_argument("--eval-frac", type=float, default=0.1)
            p.add_argument("--patience", type=int, default=0,
                           help="stop after this many evaluations without "
                                "a better eval loss [0: never]")
            p.add_argument("--history", type=float, default=0.0,
                           help="seconds of the trajectory's past the head "
                                "also sees (teacher-forced, corrupted); "
                                "predicts the next chunk [0: off]")
            p.add_argument("--target", default="abs",
                           choices=("abs", "delta"),
                           help="delta: predict offsets from the last "
                                "known history frame (needs --history)")
            p.add_argument("--hist-noise", type=float, default=0.1,
                           help="history corruption: up to this std of "
                                "noise (normalised units)")
            p.add_argument("--hist-drop", type=float, default=0.1,
                           help="... probability a history frame, and a "
                                "variable over all of it, is dropped")
            p.add_argument("--hist-none", type=float, default=0.1,
                           help="... probability of no history at all")
            p.add_argument("--prosody", action="store_true",
                           help="read <key>.prosody (flow.beat2) as "
                                "per-frame features; rows without it get "
                                "a learned 'missing' embedding")
            p.add_argument("--pros-drop", type=float, default=0.1,
                           help="share of training rows whose prosody is "
                                "hidden (trains the 'missing' embedding)")
            p.add_argument("--no-text", action="store_true",
                           help="ablation: no token states (history / "
                                "prosody only)")
            p.add_argument("--layer-mix", action="store_true",
                           help="softmax-weighted sum of the stored layers "
                                "instead of their concatenation")
            p.add_argument("--norm", default="pooled",
                           choices=("pooled", "avatar"),
                           help="per-variable statistics over all avatars, "
                                "or each avatar's own")
        else:
            p.add_argument("--ckpt", required=True,
                           help="a checkpoint, or a run folder [its "
                                "head.best.pt]")
        if name == "eval":
            p.add_argument("--ids", default=None,
                           help="comma-separated turn ids [the held-out "
                                "ones in the checkpoint]")
            p.add_argument("--markers", default=None,
                           help="flow.train.evalsft jsonl: those replies' "
                                "markers played, as a baseline")
            p.add_argument("--data", default=None,
                           help="the flow.dataset folder (--markers)")
            p.add_argument("--json", default=None)
            p.add_argument("--skip", type=float, default=None,
                           help="seconds at the start no score counts "
                                "[the checkpoint's history; the same for "
                                "every run of a comparison]")
            p.add_argument("--given", type=int, default=None,
                           help="frames of ground truth a history head "
                                "starts from [its history; 0: cold]")
        if name == "export":
            p.add_argument("--data", required=True)
            p.add_argument("--id", required=True)
            p.add_argument("--every", type=int, default=3,
                           help="frames between solved keyframes")
            p.add_argument("--out", required=True)
    o = ap.parse_args(argv)
    if o.cmd == "eval":
        o.ids = [x for x in (o.ids or "").split(",") if x]
        if o.markers and not o.data:
            ap.error("--markers needs --data")
    if o.cmd == "train":
        if o.target == "delta" and o.history <= 0:
            ap.error("--target delta needs --history")
        if o.no_text and not (o.history > 0 or o.prosody):
            ap.error("--no-text needs --history and / or --prosody")
    return o


def main(argv=None) -> int:
    o = parse(list(sys.argv[1:] if argv is None else argv))
    if o.cmd == "table":
        table(o.jsons)
    else:
        {"train": train, "eval": evaluate, "export": export}[o.cmd](o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
