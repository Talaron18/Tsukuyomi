#!/usr/bin/env python3
"""
Rig-agnostic motion prior.

The question Block 1 could not answer: retrieval showed descriptors CLUSTER
sensibly, not that a model can ACT through them. This is the test. Train on
authored animation from most rigs, then roll out on a rig never seen during
training, driving parameters it has never encountered.

Architecture (per the design decisions):

  encode    Each parameter contributes (value, descriptor) to a shared latent
            via attention pooling. Variable parameter counts collapse to one
            fixed-width state.
  temporal  Causal transformer over the state sequence. This is where motion
            dynamics live.
  decode    value_i = MLP([descriptor_i, state]). Independent projection, not
            cross-attention: if the state encodes "turning left while leaning
            back", each parameter decodes its own contribution from its own
            descriptor. O(P) instead of O(P^2), and unseen descriptors are
            handled identically to seen ones.

Predicts the DELTA from the current frame, not the absolute value. Absolute
prediction lets the model win by echoing its input, which produces a good loss
and a motionless character.

Baselines are the point of the evaluation. A prior that never moves scores well
on next-frame MSE because animation is mostly slow. Reported alongside:

  hold      next frame = current frame
  velocity  next frame = current + (current - previous)

Beating hold is trivial. Beating constant velocity means the model has learned
something about how Live2D characters actually move.

Usage:
    python -m l2dprobe.prior clips/ --rigs rigs/ --holdout Hiyori --epochs 40
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"needs pytorch: {exc}") from exc


def load(clip_dir: str, rig_dir: str, window: int = 48):
    """Pair each rig's clips with the descriptors of its expressive parameters."""
    rigs = []
    for path in sorted(glob.glob(os.path.join(clip_dir, "*.npz"))):
        name = os.path.basename(path).replace(".npz", "")
        rig_path = os.path.join(rig_dir, f"{name}.npz")
        if not os.path.exists(rig_path):
            continue
        cz = np.load(path, allow_pickle=True)
        rz = np.load(rig_path, allow_pickle=True)
        rows = cz["param_row"].astype(int)
        desc = rz["descriptor"][rows].astype(np.float32)

        clips, masks = [], []
        n = int(cz["n_clips"])
        for i in range(n):
            c = cz[f"clip{i}"].astype(np.float32)
            if len(c) < window + 2:
                continue
            clips.append(c)
            masks.append(cz[f"mask{i}"] if f"mask{i}" in cz
                         else np.ones(c.shape[1], bool))
        if clips:
            rigs.append({"name": name, "desc": desc, "clips": clips,
                         "masks": masks, "P": desc.shape[0]})
    return rigs


class Prior(nn.Module):
    def __init__(self, ddim: int = 64, state: int = 256, layers: int = 4,
                 heads: int = 4, window: int = 48):
        super().__init__()
        self.val_in = nn.Linear(1 + ddim, state)
        self.pool_q = nn.Parameter(torch.randn(1, 1, state) * 0.02)
        self.pool = nn.MultiheadAttention(state, heads, batch_first=True)
        self.pos = nn.Parameter(torch.randn(1, window, state) * 0.02)
        enc = nn.TransformerEncoderLayer(state, heads, state * 4, 0.1,
                                         batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(enc, layers)
        # +3 for the parameter's own local history. Without it each head must
        # recover its own velocity from a single pooled state shared by every
        # parameter -- 124 velocities through a 256-d bottleneck -- and the
        # model settles for predicting zero.
        self.head = nn.Sequential(
            nn.Linear(ddim + state + 3, state), nn.SiLU(),
            nn.Linear(state, state // 2), nn.SiLU(),
            nn.Linear(state // 2, 1))

    @staticmethod
    def _batched(desc):
        """Accept (P,D) or (B,P,D). A bare (P,D) broadcasts across the batch."""
        return desc.unsqueeze(0) if desc.dim() == 2 else desc

    def encode(self, vals, desc):
        """(B,T,P) values + (P,D) or (B,P,D) descriptors -> (B,T,S) state."""
        b, t, p = vals.shape
        desc = self._batched(desc)
        d = desc.unsqueeze(1).expand(b, t, p, desc.shape[-1])
        x = self.val_in(torch.cat([vals.unsqueeze(-1), d], dim=-1))
        x = x.reshape(b * t, p, -1)
        q = self.pool_q.expand(b * t, 1, -1)
        pooled, _ = self.pool(q, x, x, need_weights=False)
        return pooled.reshape(b, t, -1)

    @staticmethod
    def local_features(vals):
        """Per-parameter value, first and second difference. (B,T,P,3)"""
        vm1 = torch.cat([vals[:, :1], vals[:, :-1]], dim=1)
        vm2 = torch.cat([vals[:, :1], vm1[:, :-1]], dim=1)
        d1 = vals - vm1
        d2 = vals - 2 * vm1 + vm2
        return torch.stack([vals, d1, d2], dim=-1), d1

    def forward(self, vals, desc):
        """Correction to the constant-velocity extrapolation, per parameter.

        Predicting the raw delta means learning "keep moving as you were" from
        nothing, and at 30 fps the mean delta is ~0, so zero is a comfortable
        optimum. Predicting a correction makes the velocity baseline the model's
        zero point: outputting nothing reproduces it exactly, and anything
        learned is strict improvement.
        """
        b, t, p = vals.shape
        desc = self._batched(desc)
        h = self.encode(vals, desc) + self.pos[:, :t]
        mask = torch.triu(torch.ones(t, t, device=vals.device, dtype=torch.bool),
                          diagonal=1)
        h = self.temporal(h, mask=mask)
        hh = h.unsqueeze(2).expand(b, t, p, h.shape[-1])
        dd = desc.unsqueeze(1).expand(b, t, p, desc.shape[-1])
        loc, _ = self.local_features(vals)
        return self.head(torch.cat([dd, hh, loc], dim=-1)).squeeze(-1)

    def predict(self, vals, desc):
        """Next-frame absolute prediction: constant velocity plus correction."""
        _, d1 = self.local_features(vals)
        return vals + d1 + self(vals, desc)


def rollout_loss(model, x, y_future, desc, mask, k, tf_prob):
    """Loss over a k-step rollout instead of a single teacher-forced step.

    Single-step training never shows the model its own outputs, so at inference
    small errors compound and pull the trajectory toward the mean pose. That is
    the measured 0.47x amplitude, and it is a property of the training regime
    rather than of the architecture.

    Feeding predictions back during training -- scheduled sampling -- makes the
    model responsible for where its own errors lead. tf_prob keeps some real
    frames in the loop early on, since a model that has learned nothing yet
    diverges immediately and learns from garbage.
    """
    buf = x
    total = 0.0
    for i in range(k):
        pred = model.predict(buf, desc)[:, -1]
        true = y_future[:, i]
        total = total + (((pred - true) ** 2) * mask).sum() / (mask.sum() + 1e-9)
        use_true = torch.rand(pred.shape[0], 1, device=pred.device) < tf_prob
        nxt = torch.where(use_true, true, pred)
        buf = torch.cat([buf[:, 1:], nxt[:, None]], dim=1)
    return total / k


def stat_loss(model, x, y_future, desc, mask, k, sub=4):
    """Match the free rollout's motion statistics to the real segment's.

    Multi-step MSE cannot fix collapse, it causes it: the optimal k-step
    prediction under squared error is the conditional mean at horizon k, and
    that mean flattens toward the average pose as the future grows uncertain.
    Longer horizons therefore reward freezing, which is what the k=8 and k=24
    runs measured (+8.1% and -6.5% against velocity, rollout std falling 0.18 ->
    0.11 -> 0.07).

    This term compares the temporal standard deviation and mean frame-to-frame
    change of a free rollout against the real clip's. A model that freezes gets
    a large penalty no matter how plausible each individual frame is, and unlike
    MSE it cannot be satisfied by averaging. It constrains how much the
    character moves without dictating where it moves, which is the part 74
    minutes of data cannot teach.

    No teacher forcing here on purpose: the regime being constrained is the free
    rollout, so that is the regime it must be measured in.
    """
    # Unrolling k steps keeps k copies of the network's activations alive for
    # the backward pass, which is roughly k times a normal batch and was enough
    # to get the process OOM-killed. Two bounds:
    #   * run this term on a slice of the batch, since it constrains a
    #     population statistic and does not need every sample;
    #   * detach the rolling buffer, so each step's graph reaches back only to
    #     its own input rather than through the whole chain. The model still
    #     learns to produce the right spread from a given context; it just is
    #     not credited for how errors compound, which multi-step MSE tried to
    #     teach and which made collapse worse anyway.
    x = x[:sub]
    y_future = y_future[:sub]
    mask = mask[:sub]

    buf, outs = x, []
    for _ in range(k):
        pred = model.predict(buf, desc)[:, -1]
        outs.append(pred)
        buf = torch.cat([buf[:, 1:], pred.detach()[:, None]], dim=1)
    gen = torch.stack(outs, dim=1)
    real = y_future[:, :k]

    def stats(z):
        return z.std(dim=1), (z[:, 1:] - z[:, :-1]).abs().mean(dim=1)

    sg, dg = stats(gen)
    sr, dr = stats(real)
    err = (sg - sr).abs() + (dg - dr).abs()
    return (err * mask).sum() / (mask.sum() + 1e-9)


def batches(rigs, window, batch, steps, balance=True, rng=None, horizon=1):
    rng = rng or np.random.default_rng()
    for _ in range(steps):
        ri = rng.integers(len(rigs)) if balance else \
            rng.choice(len(rigs), p=np.array([len(r["clips"]) for r in rigs],
                                             float) / sum(len(r["clips"])
                                                          for r in rigs))
        rig = rigs[ri]
        xs, ys, ms = [], [], []
        need = window + max(horizon, 1) + 1
        usable = [i for i, c in enumerate(rig["clips"]) if len(c) >= need]
        if not usable:
            continue
        for _ in range(batch):
            ci = usable[rng.integers(len(usable))]
            c = rig["clips"][ci]
            s = rng.integers(0, len(c) - need)
            xs.append(c[s:s + window])
            ys.append(c[s + 1:s + window + max(horizon, 1)])
            # Per-sample mask: clips within a rig drive different parameter
            # subsets, so one clip's mask cannot stand in for the batch.
            ms.append(rig["masks"][ci].astype(np.float32))
        yield (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)),
               torch.from_numpy(rig["desc"]), torch.from_numpy(np.stack(ms)))


def evaluate(model, rig, window, device, rng):
    """Next-frame error against hold and constant-velocity baselines."""
    model.eval()
    errs = {"model": [], "hold": [], "velocity": []}
    desc = torch.from_numpy(rig["desc"]).to(device)
    with torch.no_grad():
        for c in rig["clips"]:
            for s in range(0, len(c) - window - 1, window):
                x = torch.from_numpy(c[s:s + window][None]).to(device)
                y = c[s + 1:s + window + 1]
                pred = model.predict(x, desc).cpu().numpy()[0]
                errs["model"].append(np.mean((pred - y) ** 2))
                cur = c[s:s + window]
                errs["hold"].append(np.mean((cur - y) ** 2))
                # prev[i] = c[s+i-1], clamped at the clip start. Slicing with
                # max(s-1,0) clamps only the lower bound and silently shortens
                # the window by one frame when s == 0.
                idx = np.clip(np.arange(s - 1, s + window - 1), 0, len(c) - 1)
                vel = cur + (cur - c[idx])
                errs["velocity"].append(np.mean((vel - y) ** 2))
    return {k: float(np.mean(v)) for k, v in errs.items() if v}


def rollout(model, rig, window, device, steps=120):
    """Free-run from a real seed. Divergence here is what MSE cannot show."""
    model.eval()
    desc = torch.from_numpy(rig["desc"]).to(device)
    c = rig["clips"][0]
    buf = torch.from_numpy(c[:window][None]).to(device)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            nxt = model.predict(buf, desc)[:, -1]
            nxt = nxt.clamp(-1, 1)
            out.append(nxt.cpu().numpy()[0])
            buf = torch.cat([buf[:, 1:], nxt[:, None]], dim=1)
    out = np.stack(out)
    # Temporal std per parameter, then averaged. A global std over the whole
    # array is dominated by differences BETWEEN parameters, so a completely
    # static pose still scores high -- which is what it reported last run.
    return {"std": float(out.std(axis=0).mean()),
            "range": float(out.max() - out.min()),
            "frame_delta": float(np.abs(np.diff(out, axis=0)).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("--rigs", default="rigs")
    ap.add_argument("--holdout", default="Hiyori")
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--state", type=int, default=256)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--rollout-k", type=int, default=0,
                    help="multi-step MSE horizon. Default 0: it makes collapse "
                         "worse, not better, since k-step MSE targets the "
                         "conditional mean at horizon k.")
    ap.add_argument("--rollout-weight", type=float, default=1.0)
    ap.add_argument("--stat-k", type=int, default=32,
                    help="free-rollout length for the motion-statistics term")
    ap.add_argument("--stat-weight", type=float, default=0.3,
                    help="fraction of the MSE term this contributes, after "
                         "automatic magnitude balancing")
    ap.add_argument("--stat-batch", type=int, default=4,
                    help="samples used for the statistics term; memory scales "
                         "with stat_k x stat_batch")
    ap.add_argument("--save", default=None,
                    help="write the trained model here for the runtime")
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    rigs = load(a.clips, a.rigs, a.window)
    if not rigs:
        print("no clips paired with rigs", file=sys.stderr)
        return 1
    train = [r for r in rigs if r["name"] != a.holdout]
    held = next((r for r in rigs if r["name"] == a.holdout), None)
    if held is None:
        print(f"holdout {a.holdout} not found", file=sys.stderr)
        return 1

    ddim = rigs[0]["desc"].shape[1]
    print(f"{len(train)} training rigs, holdout {held['name']} "
          f"({held['P']} params, {len(held['clips'])} clips), ddim={ddim}")

    device = "cpu"
    model = Prior(ddim, a.state, window=a.window).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    rng = np.random.default_rng(0)

    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        # Teacher forcing is annealed away: full at the start, mostly off by
        # the end, so the model finishes training on its own predictions --
        # which is the regime it actually runs in.
        tf_prob = max(0.0, 1.0 - ep / max(a.epochs * 0.6, 1))
        for x, y, desc, mask in batches(train, a.window, a.batch, a.steps,
                                        rng=rng,
                                        horizon=max(a.rollout_k, a.stat_k)):
            x, y, desc = x.to(device), y.to(device), desc.to(device)
            m = mask.to(device).unsqueeze(1)          # (B,1,P)
            pred = model.predict(x, desc)
            loss = (((pred - y[:, :x.shape[1]]) ** 2) * m).sum() \
                / (m.sum() * x.shape[1] + 1e-9)
            if a.rollout_k > 0 and a.rollout_weight > 0:
                fut = y[:, x.shape[1] - 1:x.shape[1] - 1 + a.rollout_k]
                if fut.shape[1] == a.rollout_k:
                    loss = loss + a.rollout_weight * rollout_loss(
                        model, x, fut, desc, m.squeeze(1), a.rollout_k, tf_prob)
            if a.stat_k > 1 and a.stat_weight > 0:
                fut = y[:, x.shape[1] - 1:x.shape[1] - 1 + a.stat_k]
                if fut.shape[1] == a.stat_k:
                    st = stat_loss(model, x, fut, desc, m.squeeze(1),
                                   a.stat_k, sub=a.stat_batch)
                    # Balance the two terms by magnitude instead of by a hand
                    # picked constant. The raw statistics term runs ~30x the
                    # MSE term, so a weight of 0.5 silently made this a
                    # statistics-only objective: prediction fell below the hold
                    # baseline and the rollout collapsed further than before.
                    # Scaling by the observed ratio makes stat_weight mean what
                    # it says -- the fraction of the loss this term contributes.
                    ratio = (loss.detach() / (st.detach() + 1e-9)).clamp(0, 1e3)
                    loss = loss + a.stat_weight * ratio * st
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        sched.step()
        if ep == 0:
            print(f"  epoch 0 {time.time() - t0:.0f}s -> "
                  f"~{(time.time() - t0) * a.epochs / 60:.0f} min total  "
                  f"(rollout_k={a.rollout_k})")
        if ep % 5 == 0 or ep == a.epochs - 1:
            e = evaluate(model, held, a.window, device, rng)
            print(f"  epoch {ep:>3} train {tot / max(n,1):.5f} | holdout "
                  f"model {e['model']:.5f}  hold {e['hold']:.5f}  "
                  f"vel {e['velocity']:.5f}")

    e = evaluate(model, held, a.window, device, rng)
    r = rollout(model, held, a.window, device)
    real = np.concatenate(held["clips"])
    ref = {"std": float(real.std(axis=0).mean()),
           "frame_delta": float(np.abs(np.diff(real, axis=0)).mean())}
    print(f"\nHELD-OUT RIG {held['name']}")
    print(f"  next-frame MSE   model {e['model']:.5f}   "
          f"hold {e['hold']:.5f}   velocity {e['velocity']:.5f}")
    print(f"  vs hold      {(1 - e['model'] / e['hold']) * 100:+.1f}%")
    print(f"  vs velocity  {(1 - e['model'] / e['velocity']) * 100:+.1f}%")
    print(f"  free rollout  std={r['std']:.4f}  "
          f"per-frame delta={r['frame_delta']:.5f}")
    print(f"  real clips    std={ref['std']:.4f}  "
          f"per-frame delta={ref['frame_delta']:.5f}   <- target scale")
    if a.save:
        # Config travels with the weights: the runtime must rebuild the same
        # architecture, and window/state are not recoverable from a state_dict
        # alone without guessing.
        torch.save({"state_dict": model.state_dict(),
                    "ddim": ddim, "state": a.state, "window": a.window,
                    "holdout": a.holdout}, a.save)
        print(f"\nsaved -> {a.save}")

    print("\nBeating hold is easy. Beating velocity means it learned dynamics.")
    print("A rollout std near zero means it collapsed to a still pose, which")
    print("next-frame MSE will happily reward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
