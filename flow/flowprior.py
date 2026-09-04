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

Usage:
    python -m l2dprobe.flowprior clips/ --rigs rigs/ --holdout Hiyori \
        --epochs 40 --threads 36
"""

from __future__ import annotations

import argparse
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


class FlowPrior(nn.Module):
    def __init__(self, ddim: int, state: int = 128, layers: int = 4,
                 heads: int = 4, ctx: int = 32, chunk: int = 12):
        super().__init__()
        self.ctx, self.chunk, self.state = ctx, chunk, state
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
        self.backbone = nn.TransformerEncoder(enc, layers)
        # +2: this parameter's own noisy value and the flow time. Pooling
        # collapses every parameter into one state, so without its own value a
        # head cannot know what it is being asked to move -- the same omission
        # that made the deterministic model predict zero.
        self.head = nn.Sequential(
            nn.Linear(ddim + state + 2, state), nn.SiLU(),
            nn.Linear(state, state // 2), nn.SiLU(),
            nn.Linear(state // 2, 1))

    def encode(self, vals, desc):
        b, t, p = vals.shape
        desc = desc.unsqueeze(0) if desc.dim() == 2 else desc
        d = desc.unsqueeze(1).expand(b, t, p, desc.shape[-1])
        x = self.val_in(torch.cat([vals.unsqueeze(-1), d], dim=-1))
        x = x.reshape(b * t, p, -1)
        q = self.pool_q.expand(b * t, 1, -1)
        pooled, _ = self.pool(q, x, x, need_weights=False)
        return pooled.reshape(b, t, -1)

    def forward(self, xt, t, ctx_vals, desc):
        """Velocity field at (xt, t) given real context. Returns (B,K,P)."""
        b, k, p = xt.shape
        desc = desc.unsqueeze(0) if desc.dim() == 2 else desc
        h_ctx = self.encode(ctx_vals, desc) + self.ctx_pos[:, :ctx_vals.shape[1]]
        temb = self.t_mlp(time_embedding(t, self.state))
        h_chunk = self.encode(xt, desc) + self.chunk_pos[:, :k] + temb[:, None]
        # Chunk tokens attend to context and to each other: the chunk is
        # produced in one shot, so there is nothing causal to enforce inside it.
        h = self.backbone(torch.cat([h_ctx, h_chunk], dim=1))[:, -k:]

        dd = desc.unsqueeze(1).expand(b, k, p, desc.shape[-1])
        hh = h.unsqueeze(2).expand(b, k, p, h.shape[-1])
        tt = t[:, None, None, None].expand(b, k, p, 1)
        loc = xt.unsqueeze(-1)
        return self.head(torch.cat([dd, hh, loc, tt], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def sample(self, ctx_vals, desc, n_params, steps: int = 6):
        """Integrate from noise, then add back the constant-velocity baseline."""
        b = ctx_vals.shape[0]
        x = torch.randn(b, self.chunk, n_params, device=ctx_vals.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((b,), i * dt, device=ctx_vals.device)
            x = x + self(x, t, ctx_vals, desc) * dt
        base = chunk_base(ctx_vals, self.chunk, self.base_mode)
        return (base + self.res_scale * x).clamp(-1, 1)


def batches(rigs, ctx, chunk, batch, steps, rng):
    for _ in range(steps):
        rig = rigs[rng.integers(len(rigs))]     # rig-balanced
        xs, ys, ms = [], [], []
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
        yield (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
               torch.from_numpy(rig["desc"]), torch.from_numpy(np.stack(ms)))


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
    return {"std": float(out.std(axis=0).mean()),
            "frame_delta": float(np.abs(np.diff(out, axis=0)).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("--rigs", default="rigs")
    ap.add_argument("--holdout", default="Hiyori")
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
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    rigs = load(a.clips, a.rigs, a.ctx + a.chunk)
    train = [r for r in rigs if r["name"] != a.holdout]
    held = next((r for r in rigs if r["name"] == a.holdout), None)
    if held is None or not train:
        print("holdout or training rigs missing", file=sys.stderr)
        return 1

    ddim = rigs[0]["desc"].shape[1]
    print(f"{len(train)} training rigs, holdout {held['name']} "
          f"({held['P']} params, {len(held['clips'])} clips)")
    device = "cpu"
    model = FlowPrior(ddim, a.state, ctx=a.ctx, chunk=a.chunk).to(device)
    model.base_mode = a.base
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params"
          f"  chunk={a.chunk} frames ({a.chunk / 30:.2f}s)  "
          f"solver={a.solver_steps} steps")

    # Measure the residual scale before training. Guessing it would put the
    # target at the wrong magnitude relative to N(0,I), which is the whole
    # reason the first attempt produced noise.
    probe_rng = np.random.default_rng(1)
    # Accumulate sum of squares rather than stacking: rigs have different
    # parameter counts, so the batches cannot be concatenated into one array.
    # Residuals are near zero mean, so RMS is the scale we want anyway.
    tot_sq, count = 0.0, 0
    for cv, x1, _d, _m in batches(train, a.ctx, a.chunk, 32, 30, probe_rng):
        r = (x1 - chunk_base(cv, a.chunk, a.base)).numpy()
        tot_sq += float((r ** 2).sum())
        count += r.size
    scale = math.sqrt(tot_sq / count) if count else 1.0
    model.res_scale.fill_(max(scale, 1e-4))
    print(f"residual scale (rms of chunk minus '{a.base}' baseline): {scale:.5f}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    rng = np.random.default_rng(0)

    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for cv, x1, desc, mask in batches(train, a.ctx, a.chunk, a.batch,
                                          a.steps, rng):
            cv, x1, desc = cv.to(device), x1.to(device), desc.to(device)
            m = mask.to(device).unsqueeze(1)
            # Residual from the constant-velocity baseline, scaled to roughly
            # unit variance so the transport from N(0,I) is short.
            base = chunk_base(cv, a.chunk, a.base)
            x1r = (x1 - base) / model.res_scale
            x0 = torch.randn_like(x1r)
            t = torch.rand(x1r.shape[0], device=device)
            xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1r
            target = x1r - x0                     # straight-line velocity
            pred = model(xt, t, cv, desc)
            loss = (((pred - target) ** 2) * m).sum() / (m.sum() * a.chunk + 1e-9)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sched.step()
        if ep == 0:
            print(f"  epoch 0 {time.time() - t0:.0f}s -> "
                  f"~{(time.time() - t0) * a.epochs / 60:.0f} min")
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:>3}  loss {tot / max(n, 1):.5f}")

    e = evaluate(model, held, a.ctx, a.chunk, device, steps=a.solver_steps)
    r = rollout(model, held, a.ctx, a.chunk, device, steps=a.solver_steps)
    real = np.concatenate(held["clips"])
    print(f"\nHELD-OUT RIG {held['name']}")
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
    print(f"  real clips    std={real.std(axis=0).mean():.4f}  "
          f"per-frame delta={np.abs(np.diff(real, axis=0)).mean():.5f}")
    print("\nRollout ratios near 1.0 are the goal. The deterministic model hit")
    print("0.26-0.99 amplitude with up to 3.4x the frame-to-frame motion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
