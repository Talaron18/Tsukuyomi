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
