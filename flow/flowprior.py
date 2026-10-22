#!/usr/bin/env python3
"""
Chunked conditional flow-matching motion prior.

Replaces the deterministic next-frame model, which failed in two measurable
ways on held-out rigs:

  jitter    Free rollout per-frame delta reached 3.4x the real clips while
            amplitude stayed below it. Single-step teacher forcing gives no
            incentive to stay on the data manifold across 120 steps, so error
            compounds into noise.
  damping   Rollout std ran 0.26-0.99 of real. When one context admits several
            plausible continuations, squared error returns their average, and
            the average of several motions is a smaller motion.

Both follow from the objective, not the architecture, so the architecture is
kept and the objective changed.

Flow matching (Lipman et al. 2022 / rectified flow, Liu et al. 2022):

    x0 ~ N(0, I);  x1 = real chunk;  t ~ U(0,1)
    xt = (1-t)*x0 + t*x1
    loss = || v(xt, t, context) - (x1 - x0) ||^2

Sampling integrates v from noise in a handful of Euler steps. No Jacobians, no
invertibility, no likelihood -- none of which this task needs. Predicting a
whole chunk removes single-step compounding inside it, and sampling rather than
averaging preserves amplitude.

Normalizing-flow frameworks like BiFlow and TarFlow target ImageNet-scale
class-conditional synthesis and buy exact likelihood, which is machinery this
problem cannot feed with 70 minutes of motion and would never use.

Conditioning slot (for the language model's hidden state)
--------------------------------------------------------
With `--cond-dim D > 0` the model takes one optional context vector per chunk:
`cond` of shape (B, D), float, passed to forward()/sample() -- or to
FlowStream.next_chunk(cond) at runtime -- for the chunk about to be generated
(the 0.4 s that follows the current context). It is projected to one token
and prepended to the context; a LEARNED null token stands in when cond is None.
Training replaces cond with the null token with probability `--cond-drop`
(0.1), so the model stays usable unconditioned and classifier-free guidance
(v = v_null + w (v_cond - v_null)) is possible later. Training cond comes from
`cond<i>` arrays (T_i, D) in a clips file, read at the first frame of each
target chunk; clips without them train the null token only. With D = 0 no
module is created and the state_dict, forward and sampling are exactly the
unconditioned model's.

Usage:
    python -m flow.flowprior clips/ --rigs rigs/ \
        --holdout aidang_2,Natori,Hiyori,lafei,xuefeng \
        --epochs 40 --threads 36 --save flowprior.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

try:
    from .prior import load
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prior import load

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"needs pytorch: {exc}") from exc


def chunk_base(ctx_vals: torch.Tensor, chunk: int, mode: str = "hold"
               ) -> torch.Tensor:
    """Baseline continuation of the context over the next `chunk` frames.

    Flow matching transports N(0,I) to the target, so the target wants to be
    small and centred. Modelling the residual from a baseline achieves that --
    but only if the baseline is actually good over the whole chunk.

    Constant velocity is the right baseline at one frame (MSE 0.00022 against
    hold's 0.00147) and the wrong one at twelve. It runs the instantaneous
    velocity straight for 0.4s while real motion curves, so its error grows
    quadratically and the measured residual RMS came out at 0.266 -- larger
    than the total displacement real motion covers in that time. The model then
    has to undo an overshoot before it can add anything, and the leftover
    overshoot showed up as a rollout 1.7x the real amplitude.

    modes:
      hold      repeat the last context frame. Error is bounded by how far the
                character actually moves in 0.4s.
      damped    velocity that decays over the chunk: right at the first frame,
                degrading to hold by the last.
      velocity  straight extrapolation. Kept for comparison.
    """
    last = ctx_vals[:, -1]
    if mode == "hold":
        return last[:, None].expand(-1, chunk, -1)
    vel = ctx_vals[:, -1] - ctx_vals[:, -2]
    steps = torch.arange(1, chunk + 1, device=ctx_vals.device,
                         dtype=ctx_vals.dtype)
    if mode == "damped":
        tau = max(chunk / 3.0, 1.0)
        steps = tau * (1 - torch.exp(-steps / tau))
    return last[:, None] + vel[:, None] * steps[None, :, None]


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


HOLDOUT = "aidang_2,Natori,Hiyori,lafei,xuefeng"


class FlowPrior(nn.Module):
    def __init__(self, ddim: int, state: int = 128, layers: int = 4,
                 heads: int = 4, ctx: int = 32, chunk: int = 12,
                 cond_dim: int = 0):
        super().__init__()
        self.ctx, self.chunk, self.state = ctx, chunk, state
        self.cond_dim = cond_dim
        # Scale of the residual, measured from training data. Registered as a
        # buffer so sampling uses the same number training saw.
        self.register_buffer("res_scale", torch.tensor(1.0))
        self.base_mode = "hold"
        # Shared frame encoder, unchanged from the deterministic prior.
        self.val_in = nn.Linear(1 + ddim, state)
        self.pool_q = nn.Parameter(torch.randn(1, 1, state) * 0.02)
        self.pool = nn.MultiheadAttention(state, heads, batch_first=True)

        self.ctx_pos = nn.Parameter(torch.randn(1, ctx, state) * 0.02)
        self.chunk_pos = nn.Parameter(torch.randn(1, chunk, state) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(state, state), nn.SiLU(),
                                   nn.Linear(state, state))
        enc = nn.TransformerEncoderLayer(state, heads, state * 4, 0.1,
                                         batch_first=True, norm_first=True)
        self.backbone = nn.TransformerEncoder(enc, layers,
                                              enable_nested_tensor=False)
        # +2: this parameter's own noisy value and the flow time. Pooling
        # collapses every parameter into one state, so without its own value a
        # head cannot know what it is being asked to move -- the same omission
        # that made the deterministic model predict zero.
        self.head = nn.Sequential(
            nn.Linear(ddim + state + 2, state), nn.SiLU(),
            nn.Linear(state, state // 2), nn.SiLU(),
            nn.Linear(state // 2, 1))
        # Created last and only when used, so a cond_dim=0 model has the same
        # parameters, initialised from the same RNG draws, as before the slot.
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, state)
            self.cond_null = nn.Parameter(torch.randn(1, 1, state) * 0.02)

    def cond_token(self, cond, b, drop: float = 0.0):
        """(B,1,S) token for `cond` (B,D) or None; None -> learned null."""
        null = self.cond_null.expand(b, 1, -1)
        if cond is None:
            return null
        tok = self.cond_proj(cond.to(null.dtype))[:, None]
        if drop > 0:
            keep = (torch.rand(b, 1, 1, device=tok.device) >= drop)
            tok = torch.where(keep, tok, null)
        return tok

    def encode(self, vals, desc):
        b, t, p = vals.shape
        desc = desc.unsqueeze(0) if desc.dim() == 2 else desc
        d = desc.unsqueeze(1).expand(b, t, p, desc.shape[-1])
        x = self.val_in(torch.cat([vals.unsqueeze(-1), d], dim=-1))
        x = x.reshape(b * t, p, -1)
        q = self.pool_q.expand(b * t, 1, -1)
        pooled, _ = self.pool(q, x, x, need_weights=False)
        return pooled.reshape(b, t, -1)

    def forward(self, xt, t, ctx_vals, desc, cond=None, cond_drop: float = 0.0):
        """Velocity field at (xt, t) given real context. Returns (B,K,P).

        cond: (B, cond_dim) or None; ignored when cond_dim is 0."""
        b, k, p = xt.shape
        desc = desc.unsqueeze(0) if desc.dim() == 2 else desc
        h_ctx = self.encode(ctx_vals, desc) + self.ctx_pos[:, :ctx_vals.shape[1]]
        temb = self.t_mlp(time_embedding(t, self.state))
        h_chunk = self.encode(xt, desc) + self.chunk_pos[:, :k] + temb[:, None]
        # Chunk tokens attend to context and to each other: the chunk is
        # produced in one shot, so there is nothing causal to enforce inside it.
        seq = [h_ctx, h_chunk]
        if self.cond_dim > 0:
            seq.insert(0, self.cond_token(cond, b, cond_drop))
        h = self.backbone(torch.cat(seq, dim=1))[:, -k:]

        dd = desc.unsqueeze(1).expand(b, k, p, desc.shape[-1])
        hh = h.unsqueeze(2).expand(b, k, p, h.shape[-1])
        tt = t[:, None, None, None].expand(b, k, p, 1)
        loc = xt.unsqueeze(-1)
        return self.head(torch.cat([dd, hh, loc, tt], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, ctx_vals, desc, n_params, steps: int = 6, cond=None,
               generator=None):
        """Integrate from noise, then add back the baseline.

        `generator` (a torch.Generator) makes the noise reproducible per
        seed; without one, torch's global RNG is used and every process
        starts from the same state, so separate runs would be identical."""
        b = ctx_vals.shape[0]
        x = torch.randn(b, self.chunk, n_params, generator=generator,
                        device=generator.device if generator is not None
                        else ctx_vals.device).to(ctx_vals.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((b,), i * dt, device=ctx_vals.device)
            x = x + self(x, t, ctx_vals, desc, cond) * dt
        base = chunk_base(ctx_vals, self.chunk, self.base_mode)
        return (base + self.res_scale * x).clamp(-1, 1)


def batches(rigs, ctx, chunk, batch, steps, rng):
    for _ in range(steps):
        rig = rigs[rng.integers(len(rigs))]     # rig-balanced
        xs, ys, ms, cs = [], [], [], []
        need = ctx + chunk + 1
        usable = [i for i, c in enumerate(rig["clips"]) if len(c) >= need]
        if not usable:
            continue
        for _ in range(batch):
            ci = usable[rng.integers(len(usable))]
            c = rig["clips"][ci]
            s = rng.integers(0, len(c) - need)
            xs.append(c[s:s + ctx])
            ys.append(c[s + ctx:s + ctx + chunk])
            ms.append(rig["masks"][ci].astype(np.float32))
            if rig.get("cond"):
                cs.append(rig["cond"][ci][s + ctx])
        cond = (torch.from_numpy(np.stack(cs).astype(np.float32))
                if rig.get("cond") else None)
        yield (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
               torch.from_numpy(rig["desc"]), torch.from_numpy(np.stack(ms)),
               cond)


@torch.no_grad()
def evaluate(model, rig, ctx, chunk, device, n_samples=8, steps=6):
    """Chunk prediction quality, scored the way a stochastic model deserves.

    Mean error over samples punishes a model for producing a valid alternative
    continuation, which is exactly the behaviour we want. Best-of-N asks the
    fairer question: does the distribution contain the truth? Constant-velocity
    extrapolation over the same chunk is the reference.
    """
    model.eval()
    desc = torch.from_numpy(rig["desc"]).to(device)
    best, mean, vel, div = [], [], [], []
    hold, damped = [], []
    for c in rig["clips"]:
        need = ctx + chunk
        for s in range(0, len(c) - need, chunk):
            cv = torch.from_numpy(c[s:s + ctx][None]).to(device)
            true = c[s + ctx:s + ctx + chunk]
            reps = cv.expand(n_samples, -1, -1).contiguous()
            gen = model.sample(reps, desc, rig["P"], steps).cpu().numpy()
            errs = ((gen - true[None]) ** 2).mean(axis=(1, 2))
            best.append(errs.min())
            mean.append(errs.mean())
            div.append(gen.std(axis=0).mean())
            last = c[s + ctx - 1]
            v = last - c[s + ctx - 2]
            horizon = np.arange(1, chunk + 1)[:, None]
            vel.append(np.mean((last[None] + v[None] * horizon - true) ** 2))
            hold.append(np.mean((np.repeat(last[None], chunk, 0) - true) ** 2))
            tau = max(chunk / 3.0, 1.0)
            damp = tau * (1 - np.exp(-horizon / tau))
            damped.append(np.mean((last[None] + v[None] * damp - true) ** 2))
    return {k: float(np.mean(v)) for k, v in
            (("best", best), ("mean", mean), ("velocity", vel),
             ("hold", hold), ("damped", damped), ("diversity", div)) if v}


@torch.no_grad()
def rollout(model, rig, ctx, chunk, device, seconds=4.0, fps=30, steps=6):
    """Free-run by re-feeding generated chunks. This is the honest test."""
    model.eval()
    desc = torch.from_numpy(rig["desc"]).to(device)
    c = rig["clips"][0]
    buf = torch.from_numpy(c[:ctx][None]).to(device)
    out = []
    for _ in range(int(seconds * fps / chunk)):
        nxt = model.sample(buf, desc, rig["P"], steps)
        out.append(nxt.cpu().numpy()[0])
        buf = torch.cat([buf, nxt], dim=1)[:, -ctx:]
    out = np.concatenate(out)
    # Saturation split by whether this rig's clips ever drive the column. The
    # loss masks undriven columns, so the model gets no gradient for them; if
    # they are what pins at +-1, the fix is in the objective, not in training
    # length. aidang_2 on the server had 56% of all values pinned.
    driven = np.zeros(out.shape[1], bool)
    for m in rig["masks"]:
        driven |= np.asarray(m, bool)
    pinned = np.abs(out) > 0.98
    sat = lambda cols: float(pinned[:, cols].mean()) if cols.any() else None  # noqa: E731
    return {"std": float(out.std(axis=0).mean()),
            "frame_delta": float(np.abs(np.diff(out, axis=0)).mean()),
            "saturation": float(pinned.mean()),
            "saturation_driven": sat(driven),
            "saturation_undriven": sat(~driven),
            "undriven_cols": int((~driven).sum())}


CONFIG_KEYS = ("ddim", "state", "layers", "heads", "ctx", "chunk", "cond_dim",
               "base_mode", "res_scale", "solver_steps", "holdout")


def save_checkpoint(path, model, cfg):
    """state_dict plus everything needed to rebuild the model. The runtime
    tells this apart from a deterministic prior by the ctx/chunk keys."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **cfg}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = {k: ck[k] for k in CONFIG_KEYS if k in ck}
    cfg.setdefault("layers", 4)
    cfg.setdefault("heads", 4)
    cfg.setdefault("cond_dim", 0)
    model = FlowPrior(cfg["ddim"], cfg["state"], cfg["layers"], cfg["heads"],
                      ctx=cfg["ctx"], chunk=cfg["chunk"],
                      cond_dim=cfg["cond_dim"])
    model.load_state_dict(ck["state_dict"])
    model.base_mode = cfg.get("base_mode", "hold")
    model.to(device).eval()
    return model, cfg


class FlowStream:
    """Real-time generation: keep a context window, sample a chunk, append.

    This is the runtime interface a VLM-conditioned character would call,
    one chunk (model.chunk frames) at a time:

        stream = FlowStream(model, desc, seed_ctx, steps=6, seed=0)
        frames = stream.next_chunk(cond)   # (chunk, P) in [-1, 1]

    `cond` is (cond_dim,) or None for the chunk about to be generated.

    `keep` frames of each sampled chunk are committed and the rest discarded
    (receding horizon). Generated chunks start close to the context and
    accelerate toward their end, so committing whole chunks produced a pulse
    at the chunk rate: on aidang_2 the last frames moved 2.6x the first, every
    0.4 s. Keeping the first half removed the pulse and halved jerk in a local
    test, at twice the sampling cost.
    Diversity comes only from the noise, which a per-stream torch.Generator
    makes reproducible: the same seed repeats, different seeds differ.
    """

    def __init__(self, model, desc, seed_ctx, steps=6, seed=0, device="cpu",
                 keep=None):
        self.model, self.steps, self.device = model, steps, device
        self.keep = max(1, min(int(keep or model.chunk), model.chunk))
        self.desc = torch.as_tensor(desc).to(device)
        ctx = np.asarray(seed_ctx, np.float32)[-model.ctx:]
        if len(ctx) < model.ctx:
            ctx = np.concatenate([np.repeat(ctx[:1], model.ctx - len(ctx), 0),
                                  ctx])
        self.buf = torch.from_numpy(ctx[None].copy()).to(device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed))

    @torch.no_grad()
    def next_chunk(self, cond=None):
        if cond is not None:
            cond = torch.as_tensor(np.asarray(cond, np.float32)[None]
                                   ).to(self.device)
        nxt = self.model.sample(self.buf, self.desc, self.buf.shape[-1],
                                self.steps, cond=cond, generator=self.gen
                                )[:, :self.keep]
        self.buf = torch.cat([self.buf, nxt], dim=1)[:, -self.model.ctx:]
        return nxt[0].cpu().numpy()

    def generate(self, n_frames, cond_fn=None):
        """n_frames of free rollout; cond_fn(frame_index) -> cond or None."""
        out, f = [], 0
        while f < n_frames:
            out.append(self.next_chunk(cond_fn(f) if cond_fn else None))
            f += len(out[-1])
        return np.concatenate(out)[:n_frames].astype(np.float32)


def attach_cond(rigs, clip_dir, cond_dim):
    """Per-clip cond arrays from `cond<i>` keys, aligned with load()'s clips.

    load() drops clips shorter than ctx+chunk, so they are matched back by
    length and order. A rig without them gets none and trains the null token.
    """
    if cond_dim <= 0:
        return
    for r in rigs:
        z = np.load(os.path.join(clip_dir, f"{r['name']}.npz"),
                    allow_pickle=True)
        if "cond0" not in z:
            continue
        cands = [(z[f"clip{i}"].shape[0], z[f"cond{i}"].astype(np.float32))
                 for i in range(int(z["n_clips"])) if f"cond{i}" in z]
        conds, j = [], 0
        for c in r["clips"]:
            while j < len(cands) and cands[j][0] != len(c):
                j += 1
            if j == len(cands) or cands[j][1].shape[-1] != cond_dim:
                conds = None
                break
            conds.append(cands[j][1])
            j += 1
        if conds:
            r["cond"] = conds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("--rigs", default="rigs")
    ap.add_argument("--holdout", default=HOLDOUT,
                    help="comma-separated rig names never trained on; each "
                         "one present is evaluated separately")
    ap.add_argument("--ctx", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=12, help="frames; 12 = 0.4s")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--solver-steps", type=int, default=6)
    ap.add_argument("--base", default="damped",
                    choices=("hold", "damped", "velocity"),
                    help="baseline the residual is measured from; measured on "
                         "Hiyori as damped 0.0285 < hold 0.0402 < velocity 0.0604")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--state", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--cond-dim", type=int, default=0,
                    help="per-chunk context vector size (0 = no slot)")
    ap.add_argument("--cond-drop", type=float, default=0.1,
                    help="probability of training on the null token instead")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, metavar="PATH.pt",
                    help="write state_dict + config for flow.runtime --ckpt")
    ap.add_argument("--log", default=None, metavar="PATH.json",
                    help="training log: per-epoch loss and held-out "
                         "evaluation (default: next to --save, else "
                         "flowprior_log.json)")
    ap.add_argument("--rollout-seconds", type=float, default=20.0,
                    help="free-rollout length for the held-out evaluation; "
                         "saturation takes several seconds to build up")
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if a.device == "auto" else a.device)
    hold = [h.strip() for h in a.holdout.split(",") if h.strip()]
    rigs = load(a.clips, a.rigs, a.ctx + a.chunk)
    attach_cond(rigs, a.clips, a.cond_dim)
    train = [r for r in rigs if r["name"] not in hold]
    held = [r for r in rigs if r["name"] in hold]
    if not train:
        print("no training rigs left after the holdout", file=sys.stderr)
        return 1
    present = {os.path.basename(p)[:-4]
               for p in glob.glob(os.path.join(a.clips, "*.npz"))}

    ddim = rigs[0]["desc"].shape[1]
    print(f"{len(train)} training rigs ({', '.join(r['name'] for r in train)})"
          f"; held out {', '.join(hold)} -- present here: "
          f"{', '.join(h for h in hold if h in present) or 'none'}")
    if a.cond_dim:
        n_c = sum(1 for r in train if r.get("cond"))
        print(f"cond slot {a.cond_dim}-d: {n_c} training rigs carry cond "
              f"arrays, the rest train the null token")
    model = FlowPrior(ddim, a.state, a.layers, a.heads, ctx=a.ctx,
                      chunk=a.chunk, cond_dim=a.cond_dim).to(device)
    model.base_mode = a.base
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params"
          f"  chunk={a.chunk} frames ({a.chunk / 30:.2f}s)  "
          f"solver={a.solver_steps} steps  device={device}")

    # Measure the residual scale before training. Guessing it would put the
    # target at the wrong magnitude relative to N(0,I), which is the whole
    # reason the first attempt produced noise.
    probe_rng = np.random.default_rng(1)
    # Accumulate sum of squares rather than stacking: rigs have different
    # parameter counts, so the batches cannot be concatenated into one array.
    # Residuals are near zero mean, so RMS is the scale we want anyway.
    tot_sq, count = 0.0, 0
    for cv, x1, _d, _m, _c in batches(train, a.ctx, a.chunk, 32, 30,
                                      probe_rng):
        r = (x1 - chunk_base(cv, a.chunk, a.base)).numpy()
        tot_sq += float((r ** 2).sum())
        count += r.size
    scale = math.sqrt(tot_sq / count) if count else 1.0
    model.res_scale.fill_(max(scale, 1e-4))
    print(f"residual scale (rms of chunk minus '{a.base}' baseline): {scale:.5f}")
    log_path = a.log or (os.path.splitext(a.save)[0] + ".log.json" if a.save
                         else "flowprior_log.json")
    log = {"args": vars(a), "device": device, "res_scale": scale,
           "train_rigs": [r["name"] for r in train], "holdout": hold,
           "epochs": [], "heldout": {}}

    def write_log():
        # Rewritten after every epoch so a long run that is interrupted still
        # leaves its numbers behind -- the last server run left none.
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=1, default=float)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    rng = np.random.default_rng(a.seed)

    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for cv, x1, desc, mask, cond in batches(train, a.ctx, a.chunk,
                                                a.batch, a.steps, rng):
            cv, x1, desc = cv.to(device), x1.to(device), desc.to(device)
            m = mask.to(device).unsqueeze(1)
            # Residual from the baseline, scaled to roughly unit variance so
            # the transport from N(0,I) is short.
            base = chunk_base(cv, a.chunk, a.base)
            x1r = (x1 - base) / model.res_scale
            x0 = torch.randn_like(x1r)
            t = torch.rand(x1r.shape[0], device=device)
            xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1r
            target = x1r - x0                     # straight-line velocity
            if cond is not None:
                cond = cond.to(device)
            pred = model(xt, t, cv, desc, cond, a.cond_drop)
            loss = (((pred - target) ** 2) * m).sum() / (m.sum() * a.chunk + 1e-9)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sched.step()
        log["epochs"].append({"epoch": ep, "loss": tot / max(n, 1),
                              "seconds": time.time() - t0})
        write_log()
        if ep == 0:
            print(f"  epoch 0 {time.time() - t0:.0f}s -> "
                  f"~{(time.time() - t0) * a.epochs / 60:.0f} min")
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:>3}  loss {tot / max(n, 1):.5f}")

    if a.save:
        save_checkpoint(a.save, model, {
            "ddim": ddim, "state": a.state, "layers": a.layers,
            "heads": a.heads, "ctx": a.ctx, "chunk": a.chunk,
            "cond_dim": a.cond_dim, "base_mode": a.base,
            "res_scale": float(model.res_scale), "solver_steps": a.solver_steps,
            "holdout": hold})
        print(f"saved -> {a.save}")

    if not held:
        print("no held-out rig present in this clips dir; nothing evaluated")
    for h in held:
        e = evaluate(model, h, a.ctx, a.chunk, device, steps=a.solver_steps)
        r = rollout(model, h, a.ctx, a.chunk, device, steps=a.solver_steps,
                    seconds=a.rollout_seconds)
        real = np.concatenate(h["clips"])
        log["heldout"][h["name"]] = {
            "eval": e, "rollout": r,
            "real_std": float(real.std(axis=0).mean()),
            "real_frame_delta": float(np.abs(np.diff(real, axis=0)).mean())}
        write_log()
        print(f"\nHELD-OUT RIG {h['name']}")
        if not e:
            print("  no clip long enough to evaluate")
            continue
        ref = min(("hold", e["hold"]), ("damped", e["damped"]),
                  ("velocity", e["velocity"]), key=lambda kv: kv[1])
        print(f"  chunk MSE   best-of-8 {e['best']:.5f}   mean {e['mean']:.5f}")
        print(f"  baselines   hold {e['hold']:.5f}   damped {e['damped']:.5f}   "
              f"velocity {e['velocity']:.5f}   -> best is {ref[0]}")
        print(f"  best-of-8 vs {ref[0]}  {(1 - e['best'] / ref[1]) * 100:+.1f}%")
        print(f"  sample diversity  {e['diversity']:.4f}   "
              "(0 means it ignores the noise)")
        print(f"  free rollout  std={r['std']:.4f}  "
              f"per-frame delta={r['frame_delta']:.5f}")
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"  pinned at +-1 {r['saturation']:.2f}  (driven cols "
              f"{fmt(r['saturation_driven'])}, never-driven cols "
              f"{fmt(r['saturation_undriven'])}, {r['undriven_cols']} of them)")
        print(f"  real clips    std={real.std(axis=0).mean():.4f}  "
              f"per-frame delta={np.abs(np.diff(real, axis=0)).mean():.5f}")
    write_log()
    print(f"\nlog -> {log_path}")
    if held:
        print("\nRollout ratios near 1.0 are the goal. The deterministic model "
              "hit\n0.26-0.99 amplitude with up to 3.4x the frame-to-frame "
              "motion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
