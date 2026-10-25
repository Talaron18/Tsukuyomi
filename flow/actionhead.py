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
  eval      per variable (natural units: hu, degrees, the face's ranges)
            the mean squared error of the head, of the rest (the clip's
            first frame held: the directed trajectory has no idle, so it
            starts at rest) and of the training mean; with --markers (flow.
            train.evalsft's per-turn jsonl, e.g. the tuned model's own
            replies) also of those markers played on the avatar (flow.
            dataset.play): the baseline the head has to beat. On teacher
            text that baseline would be the target itself (zero error),
            which is why it takes another model's markers.
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
    python -m flow.actionhead eval --ckpt runs/ah/q8/head.pt \\
        --export ... --hidden ... [--markers runs/sft/q8/eval/student.jsonl \\
        --data data/datagen/r1]
    python -m flow.actionhead export --ckpt runs/ah/q8/head.pt --export ... \\
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
                 exclude=EXCLUDE):
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
        m [F, V] bool, avail [V], fps, vars}."""
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
        return {"id": r["id"], "avatar": r["avatar"],
                "a": self.ai.get(r["avatar"], 0), "h": h, "t_tok": tt,
                "y": y, "m": m, "avail": avail, "fps": float(r["fps"]),
                "vars": r["vars"]}


def norm_stats(data, ids):
    s = np.zeros(len(data.vocab))
    s2 = np.zeros(len(data.vocab))
    n = np.zeros(len(data.vocab))
    for i in ids:
        it = data.item(i)
        s += (it["y"] * it["m"]).sum(0)
        s2 += (it["y"] ** 2 * it["m"]).sum(0)
        n += it["m"].sum(0)
    mean = s / np.maximum(n, 1)
    std = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0))
    std = np.where(std < 1e-4, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


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
                     dropout=0.1):
            super().__init__()
            self.cfg = dict(d_in=d_in, n_vars=n_vars, n_avatars=n_avatars,
                            d=d, layers=layers, heads=heads,
                            lookahead=lookahead, objective=objective,
                            dropout=dropout)
            self.d, self.lookahead, self.objective = d, lookahead, objective
            self.tok = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
            self.null = nn.Parameter(torch.zeros(1, 1, d))
            self.avatar = nn.Embedding(n_avatars, d)
            self.avail = nn.Linear(n_vars, d)
            self.q_time = nn.Linear(d, d)
            self.k_time = nn.Linear(d, d)
            if objective == "fm":
                self.x_in = nn.Linear(n_vars, d)
                self.tau = nn.Linear(d, d)
            layer = nn.TransformerDecoderLayer(
                d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
            self.dec = nn.TransformerDecoder(layer, layers)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_vars))

        def forward(self, h, t_tok, tok_pad, t_frame, frame_pad, a, avail,
                    x_tau=None, tau=None):
            """h [B, N, d_in], t_tok [B, N], tok_pad [B, N] (True: padding),
            t_frame [B, F], frame_pad [B, F], a [B], avail [B, V] ->
            [B, F, V] (normalised values, or the velocity for fm)."""
            B = h.shape[0]
            mem = self.tok(h) + self.k_time(sinus(t_tok, self.d))
            mem = torch.cat([self.null.expand(B, 1, -1), mem], 1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool,
                                         device=h.device), tok_pad], 1)
            q = self.q_time(sinus(t_frame, self.d)) + (
                self.avatar(a) + self.avail(avail.float()))[:, None]
            if self.objective == "fm":
                q = q + self.x_in(x_tau) + self.tau(sinus(tau, self.d))[:, None]
            # Frame f sees tokens up to lookahead seconds after it, and the
            # null token always.
            late = t_tok[:, None, :] > (t_frame[:, :, None] + self.lookahead)
            late = late | tok_pad[:, None, :]
            late = torch.cat([torch.zeros(B, late.shape[1], 1,
                                          dtype=torch.bool,
                                          device=h.device), late], 2)
            heads = self.dec.layers[0].multihead_attn.num_heads
            mmask = late.repeat_interleave(heads, 0)
            x = self.dec(q, mem, memory_mask=mmask,
                         tgt_key_padding_mask=frame_pad,
                         memory_key_padding_mask=pad)
            return self.out(x)

    _CLS = ActionHead
    return _CLS


def batch(items, mean, std, chunk, rng, unknown_p=0.0, train=True):
    """Random chunks of the items, padded, as tensors."""
    import torch
    hs, tts, ys, ms, tfs, aa, av = [], [], [], [], [], [], []
    for it in items:
        F = it["y"].shape[0]
        n = min(chunk, F)
        f0 = int(rng.integers(0, F - n + 1)) if train else 0
        tfs.append(np.arange(f0, f0 + n, dtype=np.float32) / it["fps"])
        ys.append((it["y"][f0:f0 + n] - mean) / std)
        ms.append(it["m"][f0:f0 + n])
        hs.append(it["h"])
        tts.append(it["t_tok"])
        a = it["a"]
        if train and rng.random() < unknown_p:
            a = 0
        aa.append(a)
        av.append(it["avail"])
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
    for b in range(B):
        n = len(tts[b])
        H[b, :n], T[b, :n], TP[b, :n] = hs[b], tts[b], False
        f = len(tfs[b])
        Y[b, :f], M[b, :f], TF[b, :f], FP[b, :f] = ys[b], ms[b], tfs[b], False
    t = torch.from_numpy
    return {"h": t(H), "t_tok": t(T), "tok_pad": t(TP), "t_frame": t(TF),
            "frame_pad": t(FP), "a": torch.tensor(aa, dtype=torch.long),
            "avail": t(np.stack(av)), "y": t(Y), "m": t(M)}


def loss_of(model, b, gen=None):
    import torch
    keys = ("h", "t_tok", "tok_pad", "t_frame", "frame_pad", "a", "avail")
    if model.objective == "fm":
        x1 = b["y"]
        x0 = torch.randn(x1.shape, generator=gen)
        tau = torch.rand(x1.shape[0], generator=gen)
        xt = (1 - tau[:, None, None]) * x0 + tau[:, None, None] * x1
        pred = model(*[b[k] for k in keys], x_tau=xt, tau=tau)
        tgt = x1 - x0
    else:
        pred = model(*[b[k] for k in keys])
        tgt = b["y"]
    m = b["m"].float()
    return ((pred - tgt) ** 2 * m).sum() / m.sum().clamp(min=1)


def predict(model, it, mean, std, chunk=CHUNK, steps=8, seed=0):
    """[F, V] in natural units: chunks of `chunk` frames, half overlap,
    cross-faded."""
    import torch
    F, V = it["y"].shape
    out = np.zeros((F, V), np.float32)
    wsum = np.zeros((F, 1), np.float32)
    hop = max(1, chunk // 2)
    starts = list(range(0, max(1, F - chunk + 1), hop))
    if starts[-1] + chunk < F:
        starts.append(F - chunk)
    gen = torch.Generator().manual_seed(int(seed))
    model.eval()
    for f0 in starts:
        n = min(chunk, F - f0)
        sub = dict(it, y=it["y"][f0:f0 + n], m=it["m"][f0:f0 + n])
        b = batch([sub], mean, std, chunk, np.random.default_rng(0),
                  train=False)
        b["t_frame"] = b["t_frame"] + f0 / it["fps"]
        keys = ("h", "t_tok", "tok_pad", "t_frame", "frame_pad", "a",
                "avail")
        with torch.no_grad():
            if model.objective == "fm":
                x = torch.randn(b["y"].shape, generator=gen)
                for k in range(steps):
                    tau = torch.full((1,), k / steps)
                    x = x + model(*[b[q] for q in keys], x_tau=x,
                                  tau=tau) / steps
                y = x[0].numpy()
            else:
                y = model(*[b[q] for q in keys])[0].numpy()
        w = np.minimum(1.0, np.minimum(np.arange(1, n + 1),
                                       np.arange(n, 0, -1)) / max(1, hop / 2))
        if f0 == 0:
            w[:hop] = 1.0
        if f0 + n == F:
            w[-hop:] = 1.0
        out[f0:f0 + n] += (y * std + mean) * w[:, None]
        wsum[f0:f0 + n] += w[:, None]
    return out / np.maximum(wsum, 1e-6)


# ------------------------------------------------------------- training

def save(path, model, data, mean, std, extra=None):
    import torch
    torch.save({"state": model.state_dict(), "config": model.cfg,
                "vocab": data.vocab, "avatars": data.avatars,
                "mean": mean, "std": std, "meta": data.meta,
                **(extra or {})}, path)


def load(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = model_cls()(**ck["config"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def train(o, log=log) -> dict:
    import torch
    torch.manual_seed(o.seed)
    data = Data(o.export, o.hidden)
    tr, ev = split_ids(data, o.eval_frac, o.seed)
    if not tr:
        raise SystemExit("actionhead: no training turns")
    mean, std = norm_stats(data, tr)
    d_in = data.item(0)["h"].shape[1]
    model = model_cls()(d_in, len(data.vocab), len(data.avatars), o.dim,
                        o.layers, o.heads, o.lookahead, o.objective)
    opt = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=0.01)
    rng = np.random.default_rng(o.seed)
    gen = torch.Generator().manual_seed(o.seed)
    os.makedirs(o.out, exist_ok=True)
    log(f"actionhead: {len(tr)} train / {len(ev)} eval turns, "
        f"{len(data.vocab)} variables, {len(data.avatars) - 1} avatars, "
        f"token states {d_in}, {sum(p.numel() for p in model.parameters())} "
        f"parameters, {o.objective}")
    best, hist = math.inf, []
    t0 = time.monotonic()
    for step in range(1, o.steps + 1):
        model.train()
        pick = rng.choice(tr, size=min(o.batch, len(tr)), replace=False)
        b = batch([data.item(int(i)) for i in pick], mean, std, o.chunk, rng,
                  UNKNOWN_P)
        loss = loss_of(model, b, gen)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % o.eval_every == 0 or step == o.steps:
            model.eval()
            with torch.no_grad():
                el = [float(loss_of(model, batch([data.item(i)], mean, std,
                                                 o.chunk, rng, train=False),
                                    gen)) for i in ev] if ev else []
            e = float(np.mean(el)) if el else float(loss)
            hist.append({"step": step, "train": round(float(loss.detach()),
                                                      5),
                         "eval": round(e, 5)})
            log(f"actionhead: step {step} loss {float(loss.detach()):.4f} eval "
                f"{e:.4f} ({time.monotonic() - t0:.0f}s)")
            if e < best:
                best = e
                save(os.path.join(o.out, "head.pt"), model, data, mean, std,
                     {"eval_ids": [data.rows[i]["id"] for i in ev],
                      "history": hist})
    with open(os.path.join(o.out, "train.json"), "w", encoding="utf-8") as f:
        json.dump({"best_eval": best, "history": hist,
                   "vocab": data.vocab, "avatars": data.avatars,
                   "opts": {k: v for k, v in vars(o).items()
                            if isinstance(v, (int, float, str, bool))}}, f,
                  indent=1)
    return {"best_eval": best, "history": hist}


# ----------------------------------------------------------------- eval

def evaluate(o, log=log) -> dict:
    model, ck = load(o.ckpt)
    data = Data(o.export, o.hidden, ck["vocab"], ck["avatars"])
    ids = set(o.ids) if o.ids else set(ck.get("eval_ids") or [])
    rows = [i for i, r in enumerate(data.rows) if not ids or r["id"] in ids]
    mean, std = ck["mean"], ck["std"]
    markers = {}
    if o.markers:
        markers = marker_trajs(o.markers, o.data, data, rows)
    V = len(data.vocab)
    se = {k: np.zeros(V) for k in ("head", "rest", "mean", "markers")}
    n = np.zeros(V)
    nm = np.zeros(V)
    for i in rows:
        it = data.item(i)
        y, m = it["y"], it["m"]
        p = predict(model, it, mean, std, o.chunk, o.steps, o.seed)
        rest = np.repeat(y[:1], len(y), 0)
        for k, x in (("head", p), ("rest", rest),
                     ("mean", np.repeat(mean[None], len(y), 0))):
            se[k] += ((x - y) ** 2 * m).sum(0)
        n += m.sum(0)
        mk = markers.get(it["id"])
        if mk is not None:
            se["markers"] += ((mk[:len(y)] - y) ** 2 * m).sum(0)
            nm += m.sum(0)
    per = {}
    for v, k in data.vi.items():
        if not n[k]:
            continue
        per[v] = {"head": round(se["head"][k] / n[k], 6),
                  "rest": round(se["rest"][k] / n[k], 6),
                  "mean": round(se["mean"][k] / n[k], 6)}
        if nm[k]:
            per[v]["markers"] = round(se["markers"][k] / nm[k], 6)
    # Each variable's error over its own variance (units differ).
    var = std ** 2
    tot = {k: round(float(np.sum(se[k] / var) / max(1, np.sum(
        n if k != "markers" else nm))), 5) for k in se
        if k != "markers" or nm.sum()}
    out = {"turns": len(rows), "normalised_mse": tot, "per_variable": per}
    log(f"actionhead: {len(rows)} turns; normalised MSE " + ", ".join(
        f"{k} {v}" for k, v in tot.items()))
    for v, d in per.items():
        log(f"  {v:16s} " + "  ".join(f"{k} {x:.5g}" for k, x in d.items()))
    if o.json:
        with open(o.json, "w", encoding="utf-8") as fh_:
            json.dump(out, fh_, indent=1)
    return out


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
    data = Data(o.export, o.hidden, ck["vocab"], ck["avatars"])
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
        else:
            p.add_argument("--ckpt", required=True)
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
    return o


def main(argv=None) -> int:
    o = parse(list(sys.argv[1:] if argv is None else argv))
    {"train": train, "eval": evaluate, "export": export}[o.cmd](o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
