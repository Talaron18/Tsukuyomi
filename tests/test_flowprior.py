"""
flow.flowprior: conditioning slot, seeding and checkpoints. Needs torch.

cond_dim=0 must be the unconditioned model exactly (same parameters, same
output), so old checkpoints load and nothing changes for callers that never
condition. Sampling must depend on the seed it is given and on nothing else:
torch's global RNG starts in the same state in every process, so without a
per-stream generator three "seeds" were three copies of one rollout.
"""

from __future__ import annotations

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
