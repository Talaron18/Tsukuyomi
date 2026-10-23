"""
flow.flowprior: conditioning slot, seeding and checkpoints. Needs torch.

cond_dim=0 must be the unconditioned model exactly (same parameters, same
output), so old checkpoints load and nothing changes for callers that never
condition. Sampling must depend on the seed it is given and on nothing else:
torch's global RNG starts in the same state in every process, so without a
per-stream generator three "seeds" were three copies of one rollout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flow.flowprior import (FlowPrior, FlowStream, load_checkpoint,  # noqa: E402
                            save_checkpoint)

DDIM, P, CTX, CHUNK = 8, 5, 6, 4


def _model(cond_dim=0, seed=0):
    torch.manual_seed(seed)
    m = FlowPrior(DDIM, state=16, layers=1, heads=2, ctx=CTX, chunk=CHUNK,
                  cond_dim=cond_dim)
    return m.eval()


def test_cond_dim_zero_is_the_unconditioned_model():
    plain = _model(0)
    cond = _model(3)
    extra = set(cond.state_dict()) - set(plain.state_dict())
    assert extra == {"cond_proj.weight", "cond_proj.bias", "cond_null"}
    # Same draws for the shared parameters: the slot is created last.
    for k, v in plain.state_dict().items():
        assert torch.equal(v, cond.state_dict()[k]), k
    xt = torch.randn(2, CHUNK, P)
    t = torch.rand(2)
    ctx = torch.randn(2, CTX, P)
    desc = torch.randn(P, DDIM)
    with torch.no_grad():
        a = plain(xt, t, ctx, desc)
        b = plain(xt, t, ctx, desc, cond=torch.randn(2, 3))  # ignored
        c0 = cond(xt, t, ctx, desc)
        c1 = cond(xt, t, ctx, desc, cond=torch.randn(2, 3))
    assert torch.equal(a, b)
    assert not torch.allclose(c0, c1)            # the slot is live


def test_stream_seeds_repeat_and_differ(tmp_path):
    m = _model(2)
    path = tmp_path / "fp.pt"
    save_checkpoint(str(path), m, {
        "ddim": DDIM, "state": 16, "layers": 1, "heads": 2, "ctx": CTX,
        "chunk": CHUNK, "cond_dim": 2, "base_mode": "hold", "res_scale": 1.0,
        "solver_steps": 3, "holdout": ["Natori"]})
    m2, cfg = load_checkpoint(str(path))
    assert cfg["holdout"] == ["Natori"] and cfg["cond_dim"] == 2
    desc = np.random.default_rng(0).standard_normal((P, DDIM)).astype(
        np.float32)
    seed_ctx = np.zeros((CTX, P), np.float32)

    def run(seed, cond_fn=None):
        s = FlowStream(m2, desc, seed_ctx, steps=3, seed=seed)
        return s.generate(10, cond_fn)

    a, a2, b = run(0), run(0), run(1)
    assert a.shape == (10, P) and np.abs(a).max() <= 1
    assert np.array_equal(a, a2)
    assert np.abs(a - b).mean() > 1e-4
    c = run(0, lambda f: np.ones(2, np.float32))
    assert np.abs(a - c).mean() > 1e-6


def test_stream_keep_commits_part_of_each_chunk():
    # Receding horizon: with keep=k each sample commits k frames, so a stream
    # needs CHUNK/k times as many samples and only the first k frames of each
    # sampled chunk ever reach the output.
    m = _model(0)
    desc = np.random.default_rng(0).standard_normal((P, DDIM)).astype(
        np.float32)
    s = FlowStream(m, desc, np.zeros((CTX, P), np.float32), steps=2, seed=0,
                   keep=1)
    assert s.next_chunk().shape == (1, P)
    full = FlowStream(m, desc, np.zeros((CTX, P), np.float32), steps=2,
                      seed=0)
    assert full.next_chunk().shape == (CHUNK, P)
    assert FlowStream(m, desc, np.zeros((CTX, P), np.float32),
                      keep=99).keep == CHUNK


# --- smooth chunk basis and context noise --------------------------------

from flow import flowprior as fp  # noqa: E402
from flow.flowprior import chunk_basis, perturb_context  # noqa: E402


def test_basis_round_trip_and_low_frequency():
    chunk = 12
    n = torch.arange(1, chunk + 1, dtype=torch.float32)[:, None] / chunk
    for k in (3, 5):
        B = chunk_basis(chunk, k)
        assert B.shape == (chunk, k)
        assert torch.allclose(B.T @ B, torch.eye(k), atol=1e-5)
        # Anything in the span comes back exactly.
        c = torch.randn(k, 4)
        assert torch.allclose(B @ (B.T @ (B @ c)), B @ c, atol=1e-5)
        # Smooth motions starting from the anchor (a ramp, an ease-in, a
        # bump over the whole chunk) keep >= 97% of their energy at k = 3.
        sig = torch.cat([0.3 * n, 0.2 * n ** 2,
                         0.1 * torch.sin(math.pi * n)], dim=1)
        rec = B @ (B.T @ sig)
        rel = (rec - sig).pow(2).sum(0) / sig.pow(2).sum(0)
        assert (rel < 0.03).all(), rel
    # White noise projected onto the basis loses its high frequencies:
    # second differences (what jerk is made of) shrink by far more than
    # the signal does.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(chunk, 200, generator=g)
    B = chunk_basis(chunk, 3)
    y = B @ (B.T @ x)
    d2 = lambda v: (v[2:] - 2 * v[1:-1] + v[:-2]).pow(2).mean()  # noqa: E731
    assert d2(y) / y.pow(2).mean() < 0.05 * d2(x) / x.pow(2).mean()
    with pytest.raises(ValueError):
        chunk_basis(chunk, 0)


def test_basis_zero_matches_the_per_frame_model(tmp_path):
    # The basis adds no parameter, no persistent buffer and no RNG draw, so
    # a basis model initialises exactly like the per-frame one and an old
    # checkpoint (no "basis" key) loads as the per-frame model.
    plain = _model(0)
    torch.manual_seed(0)
    smooth = FlowPrior(DDIM, state=16, layers=1, heads=2, ctx=CTX,
                       chunk=CHUNK, basis=2).eval()
    assert plain.state_dict().keys() == smooth.state_dict().keys()
    for k, v in plain.state_dict().items():
        assert torch.equal(v, smooth.state_dict()[k]), k
    path = tmp_path / "old.pt"
    save_checkpoint(str(path), plain, {
        "ddim": DDIM, "state": 16, "layers": 1, "heads": 2, "ctx": CTX,
        "chunk": CHUNK, "base_mode": "hold", "res_scale": 1.0,
        "solver_steps": 3})
    m2, cfg = load_checkpoint(str(path))
    assert cfg["basis"] == 0 and m2.basis == 0 and m2.n_tokens == CHUNK
    desc = np.random.default_rng(0).standard_normal((P, DDIM)).astype(
        np.float32)
    ctx = np.zeros((CTX, P), np.float32)
    a = FlowStream(plain, desc, ctx, steps=3, seed=4).generate(9)
    b = FlowStream(m2, desc, ctx, steps=3, seed=4).generate(9)
    assert np.array_equal(a, b)


def test_ctx_noise_is_training_only(monkeypatch):
    cv = torch.zeros(3, CTX, P)
    x1 = torch.zeros(3, CHUNK, P)
    c2, y2 = perturb_context(cv, x1, 0.0, 0.1)
    assert c2 is cv and y2 is x1
    # Every inference path runs without it.
    def boom(*a, **k):
        raise AssertionError("perturb_context called outside training")
    monkeypatch.setattr(fp, "perturb_context", boom)
    m = _model(0)
    desc = np.random.default_rng(0).standard_normal((P, DDIM)).astype(
        np.float32)
    FlowStream(m, desc, np.zeros((CTX, P), np.float32), steps=2).generate(6)
    rig = {"desc": desc, "P": P,
           "clips": [np.zeros((CTX + 2 * CHUNK + 1, P), np.float32)],
           "masks": [np.ones(P, bool)]}
    fp.evaluate(m, rig, CTX, CHUNK, "cpu", n_samples=2, steps=2)
    fp.rollout(m, rig, CTX, CHUNK, "cpu", seconds=0.5, steps=2)


def test_ctx_noise_bends_back_continuously():
    g = torch.Generator().manual_seed(0)
    cv = torch.zeros(64, CTX, P)
    x1 = torch.zeros(64, 12, P)
    c2, y2 = perturb_context(cv, x1, 1.0, 0.1, "walk", frac=1.0, tail=4,
                             generator=g)
    # Only the tail moves; the target ends back on the true motion.
    assert torch.equal(c2[:, :-4], cv[:, :-4])
    assert (c2[:, -4:] != 0).any()
    assert y2[:, -1].abs().max() < 1e-6
    # No step at the seam: the first target frame continues the corrupted
    # context's last value and velocity, not the clean motion's.
    e_last, v_e = c2[:, -1], c2[:, -1] - c2[:, -2]
    w1 = 0.5 * (1 + math.cos(math.pi / 12))
    assert torch.allclose(y2[:, 0], (e_last + v_e) * w1, atol=1e-6)
    # Last-frame std is about sigma * scale.
    assert 0.07 < float(c2[:, -1].std()) < 0.13
    _, y3 = perturb_context(cv, x1, 1.0, 0.1, "walk", frac=0.0)
    assert torch.equal(y3, x1)


def test_stream_with_basis_checkpoint(tmp_path):
    torch.manual_seed(0)
    m = FlowPrior(DDIM, state=16, layers=1, heads=2, ctx=CTX, chunk=CHUNK,
                  basis=2).eval()
    path = tmp_path / "b.pt"
    save_checkpoint(str(path), m, {
        "ddim": DDIM, "state": 16, "layers": 1, "heads": 2, "ctx": CTX,
        "chunk": CHUNK, "base_mode": "hold", "res_scale": 1.0,
        "solver_steps": 3, "basis": 2})
    m2, cfg = load_checkpoint(str(path))
    assert cfg["basis"] == 2 and m2.n_tokens == 2
    xt = torch.randn(3, 2, P)
    v = m2(xt, torch.rand(3), torch.randn(3, CTX, P), torch.randn(P, DDIM))
    assert v.shape == (3, 2, P)
    desc = np.random.default_rng(0).standard_normal((P, DDIM)).astype(
        np.float32)
    ctx = np.zeros((CTX, P), np.float32)
    s = FlowStream(m2, desc, ctx, steps=2, seed=0, keep=1)
    assert s.next_chunk().shape == (1, P)
    full = FlowStream(m2, desc, ctx, steps=2, seed=0)
    out = full.next_chunk()
    assert out.shape == (CHUNK, P) and np.abs(out).max() <= 1
    # Every sampled chunk lies in the basis span around the hold baseline.
    B = chunk_basis(CHUNK, 2).numpy()
    r = out - ctx[-1]
    if np.abs(out).max() < 1:          # unclamped, so exactly in the span
        assert np.allclose(B @ (B.T @ r), r, atol=1e-5)
    assert FlowStream(m2, desc, ctx, steps=2).generate(7).shape == (7, P)
