#!/usr/bin/env python3
"""
Learned parameter encoder.

What this replaces: the hand-designed descriptor pooled flow to 16x16 and then
to global scalars (curl, divergence, centroid). That destroyed laterality --
on a full-body rig the two eyes land 2-3 grid cells apart, and the blur used to
make near-misses match erased the difference. Mirror detection independently
confirmed it, finding zero L/R pairs on five of seven rigs.

What it keeps: flow as the input representation. Flow is appearance-invariant
and produced the largest single gain in the hand-designed descriptor. The
failure was the pooling, not the representation.

Objective
---------
Encode the flow field at sweep value t_src, then predict the flow field at a
DIFFERENT value t_dst from (embedding, t_src, t_dst) alone. A plain autoencoder
would let the embedding memorise one snapshot including rig-specific detail;
predicting a held-out value forces it to encode how the parameter responds
across its range, which is its function.

No identity labels and no contrastive loss. Self-supervision means training on
all ~400 usable parameters rather than the ~28 standard IDs shared between
rigs, which at this data scale is the difference between training and
overfitting.

Evaluation
----------
Leave-one-rig-out. Train on six rigs, evaluate retrieval on the seventh.
Training-set retrieval would look excellent and mean nothing, because with
seven characters the encoder can plausibly learn "this is Mao's flow
statistics" instead of "this is a head rotation".

Baseline to beat: restricted top1 41.7%, MRR 0.591.

Usage:
    python -m l2dprobe.encoder data/ --stats
    python -m l2dprobe.encoder data/ --holdout Hiyori --epochs 60
    python -m l2dprobe.encoder data/ --all-folds --epochs 60
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import time

import numpy as np

try:
    from .analyze import csls, group_of
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze import csls, group_of

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"needs pytorch: {exc}") from exc

# Pure numerical noise floor only. The real dead/alive decision comes from the
# probe's support-AREA classification, which is scale-free. Flow magnitude is
# normalised by the character box, so it is 30x larger on Wanko (a small dog
# filling its box) than on Mao (a tall full-body rig whose facial parameters
# move little relative to the whole figure). Any single absolute magnitude
# cutoff therefore keeps everything on one rig and guts another.
MIN_MAG = 1e-5

# Swept per fold. The optimum sits between "learned only" and "hand only":
# as the weight grows the fused descriptor converges back to hand-designed
# alone, so a peak in the middle is evidence the two are complementary.
FUSE_WEIGHTS = (1.0, 2.0, 3.0, 4.0, 6.0)


# ----------------------------------------------------------------- data

def live_pids_from_probe(pattern: str | None) -> dict[str, set]:
    """Map rig name -> set of parameter ids the probe did not call dead."""
    if not pattern:
        return {}
    import json
    out = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        name = os.path.basename(path)
        for prefix in ("rig_", ):
            if name.startswith(prefix):
                name = name[len(prefix):]
        name = name.split(".")[0]
        out[name] = {p["pid"] for p in blob["params"] if p["kind"] != "dead"}
    return out


def load_rigs(data_dir: str, min_mag: float = MIN_MAG, verbose: bool = True,
              probe_glob: str | None = None, only: set | None = None):
    """`only` restricts which rigs are loaded at all.

    Needed to attribute results: the 41-rig run also changed model width and
    decoder resolution, so its gain cannot be pinned on the extra data without
    running the same architecture on the smaller set.
    """
    live = live_pids_from_probe(probe_glob)
    rigs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        if only is not None and \
                os.path.basename(path).replace(".npz", "") not in only:
            continue
        z = np.load(path, allow_pickle=True)
        flow = z["flow"].astype(np.float32)         # (P, K, R, R, 2)
        pids = list(z["pids"])
        alpha = z["alpha"].astype(np.float32)
        tnorm = z["tnorm"].astype(np.float32)

        # Per-parameter magnitude, used both to drop noise-only parameters and
        # to normalise. The dataset dump kept everything because its guard was
        # max() > 0, which floating-point noise satisfies.
        name = os.path.basename(path).replace(".npz", "")
        mag = np.abs(flow).reshape(len(pids), -1).max(axis=1)
        keep = mag > min_mag
        src = "noise-floor only"
        if name in live:
            keep &= np.array([pid in live[name] for pid in pids])
            src = "probe area classification"
        if verbose:
            q = np.percentile(mag, [10, 50, 90])
            print(f"  {name:<10} {len(pids):>4} params -> {int(keep.sum()):>4} kept  "
                  f"({src})  mag pcts {np.array2string(q, precision=4)}")

        if keep.sum() == 0:
            continue
        flow, pids = flow[keep], [pids[i] for i in np.nonzero(keep)[0]]
        scale = np.abs(flow).reshape(len(pids), -1).max(axis=1)
        flow = flow / scale[:, None, None, None, None]

        rigs.append({
            "name": os.path.basename(path).replace(".npz", ""),
            "flow": flow, "pids": pids, "alpha": alpha,
            "tnorm": tnorm[keep], "log_scale": np.log10(scale + 1e-9),
        })
    return rigs


def make_input(flow: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """(R,R,2) flow -> (6,R,R) tensor: flow_x, flow_y, magnitude, alpha, x, y.

    The coordinate channels are not decoration. Without them the convolutional
    features are translation-equivariant and the final pooling discards absolute
    position, which is exactly the blindness that lost left-from-right.
    """
    r = flow.shape[0]
    mag = np.linalg.norm(flow, axis=-1)
    ys, xs = np.mgrid[0:r, 0:r]
    return np.stack([
        flow[..., 0], flow[..., 1], mag, alpha,
        (xs / (r - 1)) * 2 - 1, (ys / (r - 1)) * 2 - 1,
    ]).astype(np.float32)


def build_tensors(rigs, device):
    """Precompute encoder inputs compactly.

    Stored as 3 channels in float16, not 6 in float32. The magnitude channel is
    derivable from the flow, and the two coordinate channels are the same for
    every sample, so keeping all six materialised wasted two thirds of the
    memory traffic. With a 16 MiB L3 shared by 36 cores, every batch streams
    from RAM, so bandwidth is the binding constraint rather than FLOPs: this
    cuts the per-batch read by about 4x and the six-channel tensor is rebuilt
    in cache.

    Returns base (M,3,R,R) fp16, tgt (M,2,R,R) fp16, tn (N,K), owner (N,), K.
    Both are flattened over (parameter, step) so they are rank 4 and can be
    laid out channels_last once instead of per batch.
    """
    base, tgt, tn, owner = [], [], [], []
    for ri, rig in enumerate(rigs):
        p, k, r = rig["flow"].shape[:3]
        alpha = rig["alpha"][None].repeat(k, 0)
        for pi in range(p):
            f = rig["flow"][pi]                       # (K, R, R, 2)
            base.append(np.stack([f[..., 0], f[..., 1], alpha], axis=1))
            tgt.append(f.transpose(0, 3, 1, 2))
            tn.append(rig["tnorm"][pi])
            owner.append(ri)
    k = rigs[0]["flow"].shape[1]
    base = torch.from_numpy(np.concatenate(base)).to(torch.float16)
    tgt = torch.from_numpy(np.concatenate(tgt)).to(torch.float16)
    tn = torch.from_numpy(np.stack(tn)).to(torch.float32)
    return (base.to(device), tgt.to(device), tn.to(device),
            np.asarray(owner), k)


def coord_channels(r: int, device) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, r, device=device),
                            torch.linspace(-1, 1, r, device=device),
                            indexing="ij")
    return torch.stack([xs, ys])[None]


def make_batch(base: torch.Tensor, idx: torch.Tensor,
               coords: torch.Tensor) -> torch.Tensor:
    """Rebuild the 6-channel input for one batch: fx, fy, mag, alpha, x, y."""
    b = base[idx].float()
    mag = torch.linalg.vector_norm(b[:, :2], dim=1, keepdim=True)
    return torch.cat([b[:, :2], mag, b[:, 2:3],
                      coords.expand(b.shape[0], -1, -1, -1)], dim=1)


def flip_batch(x, y):
    """Mirror a whole batch at once instead of per sample.

    Horizontal displacement negates under mirroring, and so does the x
    coordinate channel -- miss that and the model gets taught that the same
    motion happens at two different places.
    """
    x = torch.flip(x, dims=[-1]).clone()
    x[:, 0] *= -1     # flow_x
    x[:, 4] *= -1     # coord_x
    y = torch.flip(y, dims=[-1]).clone()
    y[:, 0] *= -1
    return x, y


# ---------------------------------------------------------------- model

def _gn(c: int) -> nn.GroupNorm:
    import math as _m
    return nn.GroupNorm(max(1, _m.gcd(8, c)), c)


class Encoder(nn.Module):
    def __init__(self, dim: int = 64, width: float = 1.0):
        super().__init__()
        c = [6] + [max(8, int(w * width)) for w in (32, 64, 128, 128)]
        self.conv = nn.Sequential(*[
            layer for i in range(4) for layer in (
                nn.Conv2d(c[i], c[i + 1], 4, 2, 1), _gn(c[i + 1]), nn.SiLU())
        ])
        self.fc = nn.Linear(c[-1] * 4 * 4, dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        # Unit-norm embeddings: retrieval is cosine, so train on the sphere.
        return F.normalize(self.fc(h), dim=-1)


class Decoder(nn.Module):
    """Predicts flow at `out_res`, not necessarily the input resolution.

    The decoder is ~80% of training compute and is discarded afterwards. Its
    cost is dominated by the last upsampling stages, where feature maps are
    largest. Predicting at 32x32 instead of 64x64 removes the most expensive
    layer entirely and asks the embedding to encode the same field at slightly
    coarser output -- the embedding is a 64-vector either way, so what it must
    capture barely changes.
    """

    def __init__(self, dim: int = 64, out_res: int = 32, width: float = 1.0):
        super().__init__()
        import math as _m
        n_up = max(1, int(_m.log2(out_res // 4)))
        c0 = max(16, int(128 * width))
        chans = [c0] + [max(16, c0 // (2 ** (i + 1))) for i in range(n_up)]
        self.c0 = c0
        self.fc = nn.Linear(dim + 2, c0 * 4 * 4)
        layers = []
        for i in range(n_up):
            layers += [nn.ConvTranspose2d(chans[i], chans[i + 1], 4, 2, 1),
                       _gn(chans[i + 1]), nn.SiLU()]
        layers += [nn.Conv2d(chans[-1], 2, 3, 1, 1)]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z, t):
        h = self.fc(torch.cat([z, t], dim=-1)).view(-1, self.c0, 4, 4)
        return self.deconv(h)


def masked_loss(pred, target):
    if pred.shape[-1] != target.shape[-1]:
        # Match the decoder's output resolution by average pooling the target.
        target = F.adaptive_avg_pool2d(target, pred.shape[-2:])
    """MSE weighted toward where the target actually moves.

    Flow fields are overwhelmingly zero, so plain MSE is minimised by predicting
    nothing everywhere. Weighting by target magnitude keeps the gradient on the
    part of the field that carries the parameter's identity.
    """
    w = target.abs().amax(dim=1, keepdim=True)
    w = w / (w.amax(dim=(2, 3), keepdim=True) + 1e-6)
    # The 0.01 floor (was 0.1) still penalises hallucinating motion in still
    # regions, but no longer lets the mostly-zero background outvote the moving
    # part -- predicting nothing everywhere was too comfortable a local optimum.
    return (((pred - target) ** 2) * (0.01 + w)).mean()


# ------------------------------------------------------------- training

def retrieval(embs: dict, rigs, holdout: str, allowed=None) -> dict:
    """Same metric as analyze.py: restricted top1 and MRR, names for scoring.

    `allowed` fixes which rigs act as retrieval targets. Without it the metric
    silently changes whenever the dataset grows: going from 7 to 41 rigs took
    Hiyori's query count from 103 to 627 and dropped every score, including the
    hand-designed baseline, purely because there were more targets to confuse.
    Comparing across dataset sizes requires the yardstick to stay put.
    """
    by_name = {r["name"]: r for r in rigs}
    n = t1 = 0
    mrr = 0.0
    g1 = gn = 0
    for other in embs:
        if other == holdout:
            continue
        if allowed is not None and other not in allowed:
            continue
        pa, pb = by_name[holdout]["pids"], by_name[other]["pids"]
        shared = set(pa) & set(pb)
        cand = [j for j, q in enumerate(pb) if q in shared]
        if not shared or not cand:
            continue
        sim = csls(embs[holdout] @ embs[other][cand].T)
        for i, pid in enumerate(pa):
            if pid not in shared:
                continue
            ranked = [pb[cand[j]] for j in np.argsort(-sim[i])]
            n += 1
            if ranked[0] == pid:
                t1 += 1
            mrr += 1.0 / (ranked.index(pid) + 1)
            g = group_of(pid)
            if g != "other":
                gn += 1
                g1 += group_of(ranked[0]) == g
    if not n:
        return {}
    return {"n": n, "top1": t1 / n, "mrr": mrr / n,
            "group": g1 / gn if gn else float("nan")}


def baseline_fold(probe_glob: str, holdout: str, allowed=None) -> dict:
    """Hand-designed descriptor score for the SAME holdout rig.

    Comparing a single fold against the 41.7% all-pairs average was unfair to
    the baseline: that average includes easy and hard rigs alike, while a fold
    scores only the held-out rig's queries. Computed per fold, the two numbers
    are finally measuring the same thing.
    """
    try:
        from .analyze import featurize, fit_scalar_stats, load_rig
    except ImportError:
        from analyze import featurize, fit_scalar_stats, load_rig
    paths = sorted(glob.glob(probe_glob))
    if not paths:
        return {}
    hrigs = [load_rig(p) for p in paths]
    for r in hrigs:
        n = r["name"]
        r["name"] = n[4:].split(".")[0] if n.startswith("rig_") else n.split(".")[0]
    if holdout not in {r["name"] for r in hrigs}:
        return {}
    stats = fit_scalar_stats(hrigs)
    embs, shaped = {}, []
    for r in hrigs:
        f, pids, _ = featurize(r, stats)
        if len(f) == 0:
            continue
        embs[r["name"]] = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        shaped.append({"name": r["name"], "pids": pids})
    return retrieval(embs, shaped, holdout, allowed)


def hand_features(probe_glob: str) -> dict:
    """Hand-designed descriptors keyed by rig, as {pid: unit vector}."""
    try:
        from .analyze import featurize, fit_scalar_stats, load_rig
    except ImportError:
        from analyze import featurize, fit_scalar_stats, load_rig
    paths = sorted(glob.glob(probe_glob))
    if not paths:
        return {}
    hrigs = [load_rig(p) for p in paths]
    for r in hrigs:
        n = r["name"]
        r["name"] = n[4:].split(".")[0] if n.startswith("rig_") else n.split(".")[0]
    stats = fit_scalar_stats(hrigs)
    out = {}
    for r in hrigs:
        f, pids, _ = featurize(r, stats)
        if len(f) == 0:
            continue
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        out[r["name"]] = dict(zip(pids, f))
    return out


def fuse(embs: dict, hand: dict, rigs, weight: float = 1.0) -> dict:
    """Concatenate learned embeddings with hand-designed descriptors.

    The two encode different things: the hand features carry strong priors
    (log-area, curl, laterality) that a CNN would need far more than seven
    characters to discover, while the learned embedding sees spatial detail at
    a resolution the 32x32 pooling threw away. If they are complementary rather
    than redundant, the concatenation beats both -- and if it does not, that is
    itself worth knowing before spending effort on more rigs.
    """
    out = {}
    for rig in rigs:
        name = rig["name"]
        if name not in embs or name not in hand:
            continue
        rows = []
        for i, pid in enumerate(rig["pids"]):
            h = hand[name].get(pid)
            dim = len(next(iter(hand[name].values())))
            h = np.zeros(dim, np.float32) if h is None else h
            rows.append(np.concatenate([embs[name][i], weight * h]))
        m = np.stack(rows)
        out[name] = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    return out


def run_fold(rigs, holdout: str, epochs: int, device: str, dim: int = 64,
             batch: int = 256, lr: float = 2e-3, balance: bool = False,
             allowed=None, out_res: int = 32, width: float = 1.0,
             bf16: bool = False, channels_last: bool = False):
    base_t, tgt, tn, owner, k = build_tensors(rigs, device)
    hold_idx = next(i for i, r in enumerate(rigs) if r["name"] == holdout)
    train = np.nonzero(owner != hold_idx)[0]
    res = base_t.shape[-1]
    coords = coord_channels(res, device)
    if channels_last:
        # Done once on rank-4 tensors. Previously this ran per batch and copied
        # ~50 MB every step, which on a 16 MiB L3 costs more than it saves.
        base_t = base_t.contiguous(memory_format=torch.channels_last)
        tgt = tgt.contiguous(memory_format=torch.channels_last)

    enc = Encoder(dim, width).to(device)
    dec = Decoder(dim, out_res, width).to(device)
    if channels_last:
        # oneDNN's CPU convolutions prefer NHWC; the default NCHW forces a
        # layout conversion on every layer. Only the weights can be converted
        # here; the data tensors are laid out below, once, after being
        # flattened over (parameter, step) so they are rank 4.
        enc = enc.to(memory_format=torch.channels_last)
        dec = dec.to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in list(enc.parameters()) + list(dec.parameters()))
    print(f"    model: {nparam / 1e6:.2f}M params  out_res={out_res}  "
          f"width={width}  bf16={bf16}  channels_last={channels_last}")
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    t_start = time.time()
    for ep in range(epochs):
        enc.train(); dec.train()
        # One sample per (parameter, source step) with a randomly drawn target
        # step, rather than enumerating all 42 ordered pairs. Over many epochs
        # this covers the same pairs at a fraction of the per-epoch cost.
        if balance:
            # Sample rigs equally instead of parameters equally. Azur Lane rigs
            # carry 150-300 parameters each against 20-90 for the Cubism
            # samples, so parameter-uniform sampling would put ~90% of the
            # gradient on one studio's conventions and the encoder would
            # optimise for those rather than for deformation in general.
            groups = [train[owner[train] == ri] for ri in np.unique(owner[train])]
            per = int(np.median([len(g) for g in groups]))
            pick = np.concatenate([
                np.random.choice(g, per, replace=len(g) < per) for g in groups
            ])
            pi = np.repeat(pick, k)
        else:
            pi = np.repeat(train, k)
        si = np.tile(np.arange(k), len(pi) // k)
        ti = (si + np.random.randint(1, k, size=len(si))) % k
        perm = np.random.permutation(len(pi))
        pi, si, ti = pi[perm], si[perm], ti[perm]

        tot = steps = 0
        for b in range(0, len(pi), batch):
            bp = torch.from_numpy(pi[b:b + batch]).long()
            bs = torch.from_numpy(si[b:b + batch]).long()
            bt = torch.from_numpy(ti[b:b + batch]).long()
            flat_s = (bp * k + bs).to(base_t.device)
            flat_t = (bp * k + bt).to(base_t.device)
            x = make_batch(base_t, flat_s, coords)
            y = tgt[flat_t].float()
            t = torch.stack([tn[bp, bs], tn[bp, bt]], dim=-1)
            if np.random.rand() < 0.5:
                x, y = flip_batch(x, y)

            if bf16:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = masked_loss(dec(enc(x), t), y)
            else:
                loss = masked_loss(dec(enc(x), t), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); steps += 1
        sched.step()
        if ep == 0:
            per = time.time() - t_start
            print(f"    epoch 0 took {per:.1f}s -> ~{per * epochs / 60:.0f} min "
                  f"for {epochs} epochs")
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    epoch {ep:>3}  loss {tot / max(steps, 1):.5f}")

    enc.eval()
    embs = {}
    with torch.no_grad():
        for ri, rig in enumerate(rigs):
            idx = torch.from_numpy(np.nonzero(owner == ri)[0]).long()
            flat = torch.from_numpy(
                (idx.numpy()[:, None] * k + np.arange(k)[None]).ravel()).long()
            x = make_batch(base_t, flat, coords)
            z = enc(x).reshape(len(idx), k, -1)

            # Weight each sweep step by how much it actually moves. dataset.py
            # zeroes steps within 5% of the default, so a parameter with a
            # mid-range default contributes two or three all-zero inputs. Those
            # encode to nearly the same point for EVERY parameter, and a plain
            # mean drags all descriptors toward that shared hub, destroying the
            # fine distinctions retrieval depends on.
            w = base_t[flat][:, :2].float().abs().amax(dim=(1, 2, 3))
            w = w.reshape(len(idx), k)
            w = torch.where(w > 1e-6, w, torch.zeros_like(w))
            if float(w.sum()) == 0:
                w = torch.ones_like(w)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
            z = (z * w.unsqueeze(-1)).sum(dim=1)
            embs[rig["name"]] = F.normalize(z, dim=-1).cpu().numpy()
    return retrieval(embs, rigs, holdout, allowed), embs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--all-folds", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--balance", action="store_true",
                    help="sample rigs equally rather than parameters equally")
    ap.add_argument("--out-res", type=int, default=32,
                    help="decoder output resolution; 32 is ~3x cheaper than 64")
    ap.add_argument("--width", type=float, default=1.0,
                    help="channel multiplier; 0.5 is ~4x cheaper")
    ap.add_argument("--bf16", action="store_true",
                    help="bfloat16 autocast; big win on AVX512-BF16/AMX cores")
    ap.add_argument("--channels-last", action="store_true",
                    help="NHWC layout, which oneDNN prefers on CPU")
    ap.add_argument("--data-rigs", default=None,
                    help="comma-separated rigs to LOAD; restricts the training "
                         "set for ablations")
    ap.add_argument("--eval-rigs", default=None,
                    help="comma-separated rigs used as retrieval targets; "
                         "training still uses everything. Fixes the yardstick "
                         "so scores compare across dataset sizes.")
    ap.add_argument("--min-mag", type=float, default=MIN_MAG)
    ap.add_argument("--probe", default="out/rig_*.json",
                    help="probe JSONs, used for the dead/alive decision")
    ap.add_argument("--stats", action="store_true", help="inspect data and exit")
    a = ap.parse_args(argv)

    print("loading (magnitude percentiles per rig):")
    only = ({x.strip() for x in a.data_rigs.split(",") if x.strip()}
            if a.data_rigs else None)
    rigs = load_rigs(a.data, a.min_mag, probe_glob=a.probe, only=only)
    total = sum(r["flow"].shape[0] for r in rigs)
    print(f"\n{total} usable parameters across {len(rigs)} rigs")
    if a.stats:
        print("\n'kept' should now match the probe's continuous+categorical")
        print("counts. If a rig says 'noise-floor only', its probe JSON was")
        print("not found -- check --probe.")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # Data lives in memory and batching is manual, so there are no worker
        # processes to compete with: give every core to the math.
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(1)
    if a.bf16:
        try:
            flags = open("/proc/cpuinfo").read()
        except OSError:
            flags = ""
        if "avx512_bf16" not in flags and "amx_bf16" not in flags:
            print("  WARNING: no avx512_bf16/amx_bf16 on this CPU. bfloat16 "
                  "will be emulated in software and run SLOWER than fp32. "
                  "Dropping --bf16.")
            a.bf16 = False
    print(f"device: {device}  threads={torch.get_num_threads()}  "
          f"batch={a.batch}")

    allowed = None
    if a.eval_rigs:
        allowed = {x.strip() for x in a.eval_rigs.split(",") if x.strip()}
        have = {r["name"] for r in rigs}
        missing = allowed - have
        if missing:
            print(f"warning: --eval-rigs names not in data: {sorted(missing)}")
        print(f"evaluating against {len(allowed & have)} fixed target rigs")

    folds = ([r["name"] for r in rigs] if a.all_folds
             else [a.holdout or rigs[0]["name"]])
    results = {}
    for name in folds:
        print(f"\n=== holdout: {name} ===")
        base = baseline_fold(a.probe, name, allowed)
        if base:
            print(f"  baseline (hand-designed, same fold): n={base['n']} "
                  f"top1={base['top1']:.1%} mrr={base['mrr']:.3f} "
                  f"group={base['group']:.1%}")
        res, embs = run_fold(rigs, name, a.epochs, device, a.dim,
                             batch=a.batch, lr=a.lr, balance=a.balance,
                             allowed=allowed, out_res=a.out_res, width=a.width,
                             bf16=a.bf16, channels_last=a.channels_last)
        if res:
            results[name] = res
            delta = ("" if not base
                     else f"   (baseline {base['top1']:.1%} / {base['mrr']:.3f})")
            print(f"  LEARNED  n={res['n']}  top1={res['top1']:.1%}  "
                  f"mrr={res['mrr']:.3f}  group={res['group']:.1%}{delta}")
            hand = hand_features(a.probe)
            if hand:
                res["fused"] = {}
                for w in FUSE_WEIGHTS:
                    fr = retrieval(fuse(embs, hand, rigs, w), rigs, name, allowed)
                    if fr:
                        res["fused"][w] = fr
                        print(f"  FUSED w={w:<4} top1={fr['top1']:.1%}  "
                              f"mrr={fr['mrr']:.3f}  group={fr['group']:.1%}")
            if base:
                res["baseline"] = base

    if len(results) > 1:
        tot = sum(r["n"] for r in results.values())
        print("\n" + "=" * 68)
        print(f"MEAN over {len(results)} folds (weighted by n)")

        def wmean(get):
            return sum(get(r) * r["n"] for r in results.values()) / tot

        if all("baseline" in r for r in results.values()):
            print(f"  hand-designed  top1={wmean(lambda r: r['baseline']['top1']):.1%}"
                  f"  mrr={wmean(lambda r: r['baseline']['mrr']):.3f}"
                  f"  group={wmean(lambda r: r['baseline']['group']):.1%}")
        print(f"  learned        top1={wmean(lambda r: r['top1']):.1%}"
              f"  mrr={wmean(lambda r: r['mrr']):.3f}"
              f"  group={wmean(lambda r: r['group']):.1%}")
        for w in FUSE_WEIGHTS:
            if all(w in r.get("fused", {}) for r in results.values()):
                print(f"  fused w={w:<5}  top1={wmean(lambda r: r['fused'][w]['top1']):.1%}"
                      f"  mrr={wmean(lambda r: r['fused'][w]['mrr']):.3f}"
                      f"  group={wmean(lambda r: r['fused'][w]['group']):.1%}")

        print("\n  Pick the fuse weight from folds OTHER than the one being")
        print("  scored. Reading the best weight off the same fold you report")
        print("  is test-set tuning and inflates the result.")
    print("\nPer-fold baselines are printed above; the 41.7% all-pairs "
          "average is not a like-for-like comparison for a single fold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
