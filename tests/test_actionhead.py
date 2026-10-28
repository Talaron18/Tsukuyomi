"""
flow.actionhead and flow.train.hidden on tiny data on CPU: a mocked
generation (test_dataset's clap and pose replies on seed-san and the
Hiyori lite body) exported for the action head, random token states with
real spoken offsets (the parser fed one character at a time), the head
trained a few steps (mse and flow matching), its evaluation against the
rest and mean baselines and against markers played again (the teacher's
own: zero error, the plumbing's check), and the exporter: a trajectory
solved back to parameters and played as move tokens comes back to
itself, and one turn's prediction and directed clip are written for the
renderers. The hidden-state extractor runs on the tiny random Qwen3.
Needs torch (the extractor also transformers and tokenizers).
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

from flow import dataexport as dx
from flow import dataset as ds
from flow.train import hidden as fh
from tests.test_dataset import MOVE_REPLY, REPLY, CharLLM, UserLLM, _opts

torch = pytest.importorskip("torch")

from flow import actionhead as ah  # noqa: E402


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    root = tmp_path_factory.mktemp("ah")
    out = root / "run"
    res = ds.generate(_opts(out, sessions=4, turns=2),
                      backend_factory=lambda role, av: CharLLM(
                          [MOVE_REPLY, REPLY]) if role == "character"
                      else UserLLM(), log=lambda s: None)
    assert res["ok"] == 4
    recs = ds.iter_records(str(out))
    notes = dx.annotate(recs, keep_lite=True)
    stem = str(root / "export" / "action")
    n = dx.action_pairs(str(out), recs, notes, stem)
    assert n == 8
    rows, _ = fh.read_export(stem)
    hid = root / "hidden"
    os.makedirs(hid)
    sh = fh.Shards(str(hid), limit=40)       # several shards
    parsers = fh.parsers_for(str(out), sorted({r["avatar"] for r in rows}))
    rng = np.random.default_rng(0)
    for r in rows:
        text = r["reply_raw"]
        idx, spoken = fh.spoken_index(text, parsers[r["avatar"]]())
        assert spoken == r["spoken"]
        raw = np.array([(m.start(), m.end()) for m in
                        re.finditer(r"\S+", text)], np.int32)
        h = rng.normal(size=(len(raw), 1, 12)).astype(np.float16)
        sh.add(r["id"], r["key"], h, raw, fh.to_spoken(raw, idx,
                                                       len(spoken)))
    sh.close()
    with open(hid / "meta.json", "w") as f:
        json.dump({"model": "random", "layers": [0], "dim": 12}, f)
    return out, stem, str(hid)


def test_token_times_follow_the_speech(made):
    out, stem, hid = made
    d = ah.Data(stem, hid)
    assert len(d) == 8 and "mouth.open" not in d.vocab
    assert {"hips.x", "head.tilt", "hand.L.x"} <= set(d.vocab)
    it = d.item(0)
    t = it["t_tok"]
    assert (np.diff(t) >= -1e-3).all() and t[0] >= 0
    # The marker's token sits where it acts: the [move ...] word's time is
    # the next spoken character's start.
    rec = next(r for r in ds.iter_records(str(out)) if r["id"] == it["id"])
    mk = next(m for m in rec["markers"] if m["tag"] == rec["markers"][0][
        "tag"])
    assert min(abs(t - mk["t"])) < 0.15
    # A variable the body lacks is masked.
    lite = next(i for i in range(len(d)) if d.rows[i]["avatar"] == "Hiyori")
    assert not d.item(lite)["avail"][d.vi["hips.x"]]


@pytest.mark.parametrize("objective", ["mse", "fm"])
def test_train_and_eval(made, tmp_path, objective):
    out, stem, hid = made
    o = ah.parse(["train", "--export", stem, "--hidden", hid, "--out",
                  str(tmp_path / "ah"), "--steps", "30", "--eval-every",
                  "10", "--dim", "32", "--batch", "4", "--chunk", "48",
                  "--objective", objective, "--eval-frac", "0.3"])
    res = ah.train(o, log=lambda s: None)
    h = res["history"]
    assert len(h) == 3 and h[-1]["train"] < h[0]["train"] * 1.5
    ck = str(tmp_path / "ah" / "head.pt")
    # The teacher's own markers played again are the target (zero error).
    evrows = tmp_path / "markers.jsonl"
    with open(evrows, "w") as f:
        for r in ds.iter_records(str(out)):
            f.write(json.dumps({"id": r["id"], "reply_raw": r["reply_raw"]})
                    + "\n")
    e = ah.evaluate(ah.parse(["eval", "--ckpt", ck, "--export", stem,
                              "--hidden", hid, "--markers", str(evrows),
                              "--data", str(out), "--steps", "2"]),
                    log=lambda s: None)
    nm = e["normalised_mse"]
    assert set(nm) == {"head", "rest", "mean", "markers"}
    assert nm["markers"] < 1e-4 and nm["rest"] > nm["markers"]
    assert "head.tilt" in e["per_variable"]


def test_exporter_round_trip_and_side_by_side(made, tmp_path):
    out, stem, hid = made
    from flow import body as fb
    rows, npz = fh.read_export(stem)
    r = next(x for x in rows if x["avatar"] == "seed-san")
    with np.load(npz) as z:
        tr = z[r["key"] + ".traj"].astype(np.float32)
    bm = fb.load("rigs/seed-san.body.json")
    toks, sec = ah.to_tokens(bm, tr, r["vars"], r["fps"], every=2,
                             log=lambda s: None)
    assert len(toks) == 1 and toks[0]["type"] == "move"
    back = ah.play_tokens(bm, toks, r["fps"], len(tr))
    for v, tol in (("head.tilt", 0.6), ("hand.L.x", 0.04), ("hand.R.y", 0.04)):
        c = r["vars"].index(v)
        got = np.array([s[v] for s in back])
        err = np.abs(got[2:-2] - tr[2:-2, c])
        assert np.nanmedian(err) < tol / 4, v
        # At a clap's contact the hands, given as positions only (the
        # trajectory has no palm orientation, the clip had @together's),
        # are parted by the collision model: 0.10 hu wider each.
        assert np.nanmax(err) < max(tol, 0.12), v
    # One turn's prediction and its directed clip, both for the renderers.
    o = ah.parse(["train", "--export", stem, "--hidden", hid, "--out",
                  str(tmp_path / "ah"), "--steps", "5", "--eval-every", "5",
                  "--dim", "16", "--batch", "2"])
    ah.train(o, log=lambda s: None)
    res = ah.export(ah.parse(["export", "--ckpt", str(tmp_path / "ah" /
                                                       "head.pt"),
                              "--export", stem, "--hidden", hid, "--data",
                              str(out), "--id", r["id"], "--every", "3",
                              "--out", str(tmp_path / "side")]),
                    log=lambda s: None)
    for d in res.values():
        blob = json.load(open(os.path.join(d, "tokens.json")))
        assert blob["tokens"] and blob["seconds"] > 0
        st = json.load(open(os.path.join(d, "states.json")))
        assert st["format"] == "tsukuyomi.vrmstates/1" and st["states"]
    assert "flow.vrmrender" in open(tmp_path / "side" / "render.sh").read()


def test_exporter_on_a_live2d_body(made):
    out, stem, _hid = made
    from flow import body as fb
    rows, npz = fh.read_export(stem)
    r = next(x for x in rows if x["avatar"] == "Hiyori")
    with np.load(npz) as z:
        tr = z[r["key"] + ".traj"].astype(np.float32)[:60]
        var = z[r["key"] + ".variant"][:60]
    bm = fb.load("rigs/Hiyori.body.json")
    toks, _ = ah.to_tokens(bm, tr, r["vars"], r["fps"], every=6,
                           variant=var, log=lambda s: None)
    back = ah.play_tokens(bm, toks, r["fps"], len(tr))
    c = r["vars"].index("head.tilt")
    got = np.array([s["head.tilt"] for s in back])
    assert np.nanmax(np.abs(got[6:-6] - tr[6:-6, c])) < 1.0


def test_extract_on_a_tiny_model(made, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from flow.train import chat
    from tests import tiny_lm
    out, stem, _hid = made
    texts = [r["reply_raw"] for r in ds.iter_records(str(out))] * 5
    lm = tiny_lm.build(str(tmp_path / "tiny-qwen3"), texts + [chat.render(
        [{"role": "user", "content": "hi"}], "qwen3")[0]])
    o = fh.parse(["--export", stem, "--data", str(out), "--model", lm,
                  "--layers=-1,1", "--out", str(tmp_path / "h"),
                  "--shard-tokens", "50", "--proj", "8"])
    meta = fh.extract(o, log=lambda s: None)
    assert meta["layers"] == [2, 1] and meta["dim"] == 8
    idx = fh.load_index(str(tmp_path / "h"))
    assert len(idx) == 8
    e = next(iter(idx.values()))
    with np.load(tmp_path / "h" / e["shard"]) as z:
        h = z[e["key"] + ".h"]
        sp = z[e["key"] + ".spoken"]
    assert h.shape == (e["tokens"], 2, 8) and h.dtype == np.float16
    assert sp.shape == (e["tokens"], 2) and (np.diff(sp[:, 0]) >= 0).all()
    d = ah.Data(stem, str(tmp_path / "h"))
    assert d.item(0)["h"].shape[1] == 16


def test_extract_prefix_cache_and_resume(made, tmp_path):
    """The system prompt's cached KV gives the states a whole forward
    gives; a run cut short continues where its shards end."""
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from flow.train import chat
    from tests import tiny_lm
    out, stem, _hid = made
    texts = [r["reply_raw"] for r in ds.iter_records(str(out))] * 5
    lm = tiny_lm.build(str(tmp_path / "tiny-qwen3"), texts + [chat.render(
        [{"role": "user", "content": "hi"}], "qwen3")[0]])
    logs = []

    def run(name, *extra):
        o = fh.parse(["--export", stem, "--data", str(out), "--model", lm,
                      "--layers=-1,1", "--out", str(tmp_path / name),
                      "--shard-tokens", "50", "--device", "cpu", *extra])
        fh.extract(o, log=logs.append)
        idx = fh.load_index(str(tmp_path / name))
        hs = {}
        for e in idx.values():
            with np.load(tmp_path / name / e["shard"]) as z:
                hs[e["id"]] = z[e["key"] + ".h"].astype(np.float32)
        return hs

    cached = run("cached")
    assert any("cached for 8 turns" in s for s in logs)
    whole = run("whole", "--no-prefix-cache")
    assert cached.keys() == whole.keys() and len(cached) == 8
    for k in cached:
        np.testing.assert_allclose(cached[k], whole[k], atol=2e-2)
    # Cut short: drop the last shard (and its index lines stay behind, as
    # after a crash); the same command finishes the run.
    d = tmp_path / "cached"
    last = sorted(f for f in os.listdir(d) if f.startswith("hidden-"))[-1]
    os.remove(d / last)
    logs.clear()
    again = run("cached")
    assert any("resuming" in s for s in logs)
    assert again.keys() == cached.keys()
    for k in again:
        np.testing.assert_allclose(again[k], cached[k], atol=2e-2)
    with pytest.raises(SystemExit):
        run("cached", "--proj", "8")


# ------------------------------------------- motion history, prosody, mix

def _train(made, out, *extra, steps=12, every=4):
    _out, stem, hid = made
    o = ah.parse(["train", "--export", stem, "--hidden", hid, "--out",
                  str(out), "--steps", str(steps), "--eval-every",
                  str(every), "--dim", "16", "--batch", "4", "--chunk", "48",
                  "--eval-frac", "0.3", *extra])
    return ah.train(o, log=lambda s: None)


def _eval(made, ckpt, *extra):
    _out, stem, hid = made
    return ah.evaluate(ah.parse(["eval", "--ckpt", str(ckpt), "--export",
                                 stem, "--hidden", hid, "--steps", "2",
                                 *extra]), log=lambda s: None)


def test_default_head_keeps_the_old_format(made, tmp_path):
    """No new option, no new parameter or config entry: checkpoints of the
    plain head load as before (also one without the entries added later),
    and a run folder resolves to its best checkpoint, or head.pt alone."""
    _train(made, tmp_path / "a")
    ck = torch.load(tmp_path / "a" / "head.pt", weights_only=False)
    assert set(ck["config"]) == {"d_in", "n_vars", "n_avatars", "d", "layers",
                                 "heads", "lookahead", "objective",
                                 "dropout"}
    assert {k.split(".")[0] for k in ck["state"]} == {
        "tok", "null", "avatar", "avail", "q_time", "k_time", "dec", "out"}
    assert ck["mean"].ndim == 1
    for k in ("pooled_mean", "pooled_std", "norm", "opts", "step", "best"):
        ck.pop(k, None)
    torch.save(ck, tmp_path / "a" / "head.pt")
    os.remove(tmp_path / "a" / "head.best.pt")
    assert ah.resolve(str(tmp_path / "a")).endswith("head.pt")
    e = _eval(made, tmp_path / "a")
    assert set(e["normalised_mse"]) == {"head", "rest", "mean"}
    assert e["skip_seconds"] == 0 and e["norm"] == "pooled"
    assert e["layer_mix"] is None


def test_best_checkpoint_and_patience(made, tmp_path):
    res = _train(made, tmp_path / "b", steps=24, every=4)
    h = res["history"]
    assert os.path.exists(tmp_path / "b" / "head.pt")
    best = torch.load(tmp_path / "b" / "head.best.pt", weights_only=False)
    last = torch.load(tmp_path / "b" / "head.pt", weights_only=False)
    top = min(h, key=lambda x: x["eval"])
    assert best["step"] == top["step"] and last["step"] == 24
    assert res["best_eval"] == pytest.approx(top["eval"], abs=1e-4)
    assert ah.resolve(str(tmp_path / "b")).endswith("head.best.pt")
    assert ah.resolve(str(tmp_path / "b" / "head.pt")).endswith("head.pt")
    # patience 1: stops after the first evaluation that does not improve
    r2 = _train(made, tmp_path / "c", "--patience", "1", "--lr", "0.5",
                steps=400, every=2)
    assert len(r2["history"]) < 200


def test_history_is_teacher_forced_and_rolled_out(made, tmp_path):
    out, stem, hid = made
    d = ah.Data(stem, hid)
    mean, std = ah.norm_stats(d, range(len(d)))
    it = d.item(0)
    F = len(it["y"])
    rng = np.random.default_rng(0)
    b = ah.batch([it], mean, std, 48, rng, hist=15, f0s=[30], train=False)
    yn = (it["y"] - mean) / std
    np.testing.assert_allclose(b["hist"][0].numpy(), yn[15:30], atol=1e-5)
    assert b["hist_m"][0].numpy().shape == (15, len(d.vocab))
    np.testing.assert_allclose(b["anchor"][0].numpy(),
                               ah.last_valid(yn[15:30], it["m"][15:30]))
    # before the clip's start the history is missing, not zeros known
    b0 = ah.batch([it], mean, std, 48, rng, hist=15, f0s=[5], train=False)
    assert not b0["hist_m"][0, :10].any() and b0["hist_m"][0, 10:].any()
    # training corrupts it
    bc = ah.batch([it], mean, std, 48, rng, hist=15, f0s=[30],
                  corrupt=dict(noise=0.5, drop_frame=0.3, drop_var=0.3,
                               drop_all=0.0))
    assert not np.allclose(bc["hist"][0].numpy(), b["hist"][0].numpy())

    res = _train(made, tmp_path / "h", "--history", "0.5", steps=8, every=4)
    assert len(res["history"]) == 2
    model, ck = ah.load(str(tmp_path / "h"))
    assert model.hist == 15 and ck["config"]["history"] == 15
    p = ah.predict(model, it, ck["mean"], ck["std"], 48, 2)
    assert p.shape == it["y"].shape and np.isfinite(p).all()
    np.testing.assert_array_equal(p[:15], it["y"][:15])    # given, as they are
    assert not np.allclose(p[15:], it["y"][15:])
    cold = ah.predict(model, it, ck["mean"], ck["std"], 48, 2, given=0)
    assert cold.shape == p.shape and not np.allclose(cold[:15], it["y"][:15])
    # The rollout depends on the history it is given.
    shifted = dict(it, y=it["y"] + 5 * (np.arange(F)[:, None] < 15))
    assert not np.allclose(ah.predict(model, shifted, ck["mean"], ck["std"],
                                      48, 2)[20:], p[20:])
    e = _eval(made, tmp_path / "h")
    assert e["skip_seconds"] == 0.5 and e["history_seconds"] == 0.5
    nm = e["normalised_mse"]
    assert set(nm) == {"head", "rest", "mean", "hold"}
    for k in ("relative_mse", "velocity_mse", "std_ratio"):
        assert set(e[k]) == set(nm), k
    assert e["std_ratio"]["rest"] == 0.0
    assert "std_ratio" in e["per_variable_motion"]["head.tilt"]
    json.dumps(e)                                  # no NaN, JSON-safe
    assert "NaN" not in json.dumps(e)


def test_delta_target_is_an_offset_from_the_last_frame(made, tmp_path):
    out, stem, hid = made
    d = ah.Data(stem, hid)
    mean, std = ah.norm_stats(d, range(len(d)))
    with pytest.raises(SystemExit):
        ah.parse(["train", "--export", stem, "--hidden", hid, "--out", "x",
                  "--target", "delta"])
    _train(made, tmp_path / "d", "--history", "0.5", "--target", "delta")
    model, ck = ah.load(str(tmp_path / "d"))
    assert model.delta and ck["config"]["target"] == "delta"
    it = d.item(0)
    b = ah.batch([it], mean, std, 48, None, hist=15, f0s=[30], train=False)
    # a model that predicts zero offsets holds the last known frame
    zero = type("Z", (), {"hist": 15, "delta": True, "objective": "mse",
                          "use_pros": False,
                          "eval": lambda self: None,
                          "__call__": lambda self, **kw: torch.zeros(
                              kw["avail"].shape[0], kw["t_frame"].shape[1],
                              len(d.vocab))})()
    y = ah.sample(zero, b, None, 1)
    np.testing.assert_allclose(y, np.repeat(b["anchor"][:1].numpy(), 48, 0))
    assert _eval(made, tmp_path / "d")["normalised_mse"]["hold"] > 0


def test_missing_prosody_is_a_mask_not_zeros(made, tmp_path):
    out, stem, hid = made
    d = ah.Data(stem, hid, prosody=True)
    it = d.item(0)
    assert it["pros"] is None                     # the teacher's rows: none
    mean, std = ah.norm_stats(d, range(len(d)))
    b = ah.batch([it], mean, std, 48, None, train=False, pros=True)
    assert b["pros_m"].tolist() == [False] and (b["pros"] == 0).all()
    _train(made, tmp_path / "p", "--prosody")
    model, ck = ah.load(str(tmp_path / "p"))
    assert model.use_pros and ck["config"]["prosody"] is True
    kw = ah.inputs(b)
    with torch.no_grad():
        a = model(**kw)
        kw["pros"] = torch.randn_like(kw["pros"])     # ignored when missing
        np.testing.assert_allclose(model(**kw).numpy(), a.numpy(),
                                   atol=1e-6)
        kw["pros_m"] = torch.tensor([True])           # and not when present
        assert not np.allclose(model(**kw).numpy(), a.numpy())
    e = _eval(made, tmp_path / "p")
    assert e["beat"] == {} or e["beat"]["gt"]["hands"]["segments"] == 0


def test_layer_mix_weights_sum_to_one_and_learn():
    cls = ah.model_cls()
    m = cls(36, 5, 2, d=16, layers=1, heads=2, mix=3)
    w = m.layer_weights()
    assert w.shape == (3,) and abs(w.sum() - 1) < 1e-6
    np.testing.assert_allclose(w, 1 / 3, atol=1e-6)
    B, N, F = 2, 4, 6
    kw = dict(h=torch.randn(B, N, 36), t_tok=torch.rand(B, N),
              tok_pad=torch.zeros(B, N, dtype=torch.bool),
              t_frame=torch.rand(B, F), frame_pad=torch.zeros(
                  B, F, dtype=torch.bool), a=torch.zeros(B, dtype=torch.long),
              avail=torch.ones(B, 5))
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    for _ in range(5):
        loss = (m(**kw) - 1).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w2 = m.layer_weights()
    assert abs(w2.sum() - 1) < 1e-6 and not np.allclose(w2, w, atol=1e-4)
    with pytest.raises(ValueError):
        cls(10, 5, 2, mix=3)
    # a layer repeated is a layer once: equal weight to equal states
    x = torch.randn(B, N, 12)
    same = cls(36, 5, 2, d=16, layers=1, heads=2, mix=3).eval()
    base = same.embed_tokens(torch.cat([x, x, x], -1))
    again = same.embed_tokens(torch.cat([x, x, x], -1))
    np.testing.assert_allclose(base.detach().numpy(), again.detach().numpy())


def test_layer_mix_trains_and_reports(made, tmp_path):
    _train(made, tmp_path / "m", "--layer-mix")
    model, ck = ah.load(str(tmp_path / "m"))
    assert model.mix == 1 and ck["config"]["mix"] == 1
    e = _eval(made, tmp_path / "m")
    assert e["layer_mix"]["weights"] == [1.0]


def test_no_text_head_has_no_token_path(made, tmp_path):
    _train(made, tmp_path / "n", "--no-text", "--history", "0.5")
    model, ck = ah.load(str(tmp_path / "n"))
    assert not model.use_text and not hasattr(model, "tok")
    e = _eval(made, tmp_path / "n")
    assert "head" in e["normalised_mse"]
    with pytest.raises(SystemExit):
        ah.parse(["train", "--export", "e", "--hidden", "h", "--out", "o",
                  "--no-text"])


def test_per_avatar_norm_in_a_run(made, tmp_path):
    out, stem, hid = made
    d = ah.Data(stem, hid)
    tr, _ev = ah.split_ids(d, 0.3, 0)
    m, s = ah.norm_stats(d, tr, per_avatar=True)
    pm, ps = ah.norm_stats(d, tr)
    assert m.shape == (3, len(d.vocab))
    assert not np.allclose(m[1], m[2])                 # two bodies, two stats
    np.testing.assert_allclose(m[0], pm, atol=1e-6)
    _train(made, tmp_path / "v", "--norm", "avatar")
    _model, ck = ah.load(str(tmp_path / "v"))
    assert ck["mean"].shape == (3, len(d.vocab)) and ck["norm"] == "avatar"
    np.testing.assert_allclose(ck["pooled_std"], ps, atol=1e-6)
    e = _eval(made, tmp_path / "v")
    assert e["norm"] == "avatar" and e["normalised_mse"]["mean"] > 0
    # the baselines are scored on the pooled std whatever the run's norm
    e0 = _eval(made, tmp_path / "v", "--skip", "0")
    assert e0["normalised_mse"]["rest"] == pytest.approx(
        e["normalised_mse"]["rest"])
    # an old checkpoint's avatar index past the table is the unknown one
    np.testing.assert_array_equal(ah.stats_for(ck["mean"], ck["std"], 99)[0],
                                  ck["mean"][0])


def test_the_table_prints_runs_side_by_side(made, tmp_path):
    _train(made, tmp_path / "t1")
    _train(made, tmp_path / "t2", "--history", "0.5")
    js = []
    for n in ("t1", "t2"):
        _eval(made, tmp_path / n, "--skip", "0.5", "--json",
              str(tmp_path / n / "eval.json"))
        js.append(str(tmp_path / n / "eval.json"))
    lines = ah.table(js, log=lambda s: None)
    assert lines[0].split() == ["t1", "t2"]
    assert any(l.startswith("nmse.head") for l in lines)
    assert any(l.startswith("nmse.hold") for l in lines)
