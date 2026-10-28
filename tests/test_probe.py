"""
flow.probe on synthetic exports where the answer is known: no BEAT2, no
model. The export is written the way flow.beat2 writes it (rows with
clip / speaker / lang / t0, `traj`, `chars`, `prosody` arrays); the hidden
states through flow.train.hidden's shard writer.

prosody (numpy, system python): motion amplitude proportional to the
segment's loudness within each speaker, the frequency set so that the speed
does not change: a strong within-speaker amplitude correlation, none with
the speed, and the loudness -> amplitude > speed hypothesis supported; a
speaker offset in both loudness and amplitude: strong overall, gone within
speaker; the motion following the loudness 0.5 s later: the lag scan peaks
there; the clip bootstrap, the rank and the within-speaker helpers; units,
pauses.

match (torch): tokens that encode the motion's gesture class per second:
retrieval far above chance under same-speaker and same-clip negatives, and
at chance when the tokens are another window's; tokens and motion that
encode only the speaker: far above chance with random negatives (trained
on them), chance with same-speaker negatives; the semantic-annotation
reader; the CLI writes report.txt and report.json.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from flow import probe as pr

FPS = 30


def _row(clip, spk, seg, n, spoken, lang="en", t0=0.0):
    sid = f"beat2.{clip}/s{seg:02d}"
    return {"id": sid, "key": sid.replace("/", "__"), "avatar": "seed-san",
            "kind": "vrm", "vars": pr.CANON + ["mouth.open"], "fps": FPS,
            "frames": n, "spoken": spoken, "reply_raw": spoken,
            "speech_seconds": n / FPS - 0.5, "lang": {"asked": None,
                                                       "reply": lang},
            "clip": clip, "speaker": spk, "t0": t0, "t1": t0 + n / FPS,
            "markers": []}


def _write(path, rows, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez(path + ".npz", **arrays)


def _words(n_sec, rng, rho=1.0):
    """(spoken, chars [len, 2]): words of 3 letters, 2.5 rho per second."""
    n = int(n_sec * 2.5 * rho)
    text, ch = "", []
    for i in range(n):
        a = 0.3 + i * 0.4 / rho
        if i:
            text += " "
            ch.append((a - 0.1, a))
        text += "abc"
        ch += [(a + 0.1 * k, a + 0.1 * (k + 1)) for k in range(3)]
    return text, np.asarray(ch, np.float32)


def synth_prosody(path, mode="const", nspk=8, nclip=6, nseg=2, secs=10.0,
                  seed=0, delay=0.5):
    """mode const: a segment's loudness level l ~ N(0, 1), amplitude
    exp(0.4 l) times the speaker's, frequency 3 / amplitude (the speed
    does not depend on l); timevar: l per 1 s block, the amplitude follows
    it `delay` s later; confound: loudness and amplitude each a speaker
    constant, independent of each other within a speaker; rate: words per
    second rho ~ exp(0.3 N) per segment, the frequency (the speed) follows
    it, the amplitude does not."""
    rng = np.random.default_rng(seed)
    n = int(secs * FPS)
    t = np.arange(n) / FPS
    rows, arrays = [], {}
    for k in range(nspk):
        base = rng.normal() * 1.5
        scale = math.exp(0.4 * base) if mode == "confound" else math.exp(
            rng.normal() * 0.5)
        for c in range(nclip):
            clip = f"{k}_sp{k}_{c}_1_1"
            for s in range(nseg):
                if mode == "timevar":
                    blocks = rng.normal(size=int(secs) + 2)
                    lt = blocks[np.minimum(t.astype(int), len(blocks) - 1)]
                    ts = np.maximum(t - delay, 0)
                    am = np.exp(0.4 * blocks[np.minimum(
                        ts.astype(int), len(blocks) - 1)])
                    loud = lt
                else:
                    l0 = rng.normal()
                    loud = np.full(n, l0)
                    am = np.full(n, math.exp(0.4 * l0))
                    if mode == "rate":
                        am = np.full(n, math.exp(0.1 * rng.normal()))
                    if mode == "confound":
                        am = np.full(n, scale * math.exp(
                            0.1 * rng.normal()))
                        loud = np.full(n, base + 0.3 * rng.normal())
                    else:
                        am = am * scale
                a = am * (scale if mode == "timevar" else 1.0)
                rho = math.exp(0.3 * rng.normal()) if mode == "rate" else 1.0
                g = rho if mode == "rate" else math.exp(0.3 * rng.normal())
                phase = np.cumsum(2 * np.pi * 3.0 * g / a / FPS)
                y = np.zeros((n, len(pr.CANON) + 1), np.float32)
                y[:, pr.CANON.index("hand.L.x")] = a * np.sin(phase)
                y[:, pr.CANON.index("hand.R.y")] = 0.3 * a * np.sin(
                    phase * 0.7 + 1)
                y[:, pr.CANON.index("head.yaw")] = 5 * a * np.sin(
                    phase * 0.5)
                rms = np.exp(-3 + 0.5 * loud + 0.05 * rng.normal(size=n))
                f0 = np.full(n, 120.0 + 5 * k) + rng.normal(size=n)
                spoken, chars = _words(secs, rng, rho)
                row = _row(clip, f"sp{k}", s, n, spoken,
                           ["en", "zh", "es", "ja"][k % 4],
                           t0=s * (secs + 1))
                rows.append(row)
                arrays[row["key"] + ".traj"] = y.astype(np.float16)
                arrays[row["key"] + ".chars"] = chars
                arrays[row["key"] + ".prosody"] = np.stack(
                    [rms, f0, np.ones(n)], 1).astype(np.float16)
    _write(path, rows, arrays)


def _opts(tmp, **kw):
    a = ["prosody", "--export", str(tmp / "ex"), "--out", str(tmp / "out"),
         "--boot", "300"]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}", str(v)]
    return pr.parse(a)


# ------------------------------------------------------------ helpers

def test_rank_within_and_clip_bootstrap():
    assert list(pr.rank(np.array([3.0, 1.0, 1.0, 2.0]))) == [4, 1.5, 1.5, 3]
    x = np.r_[np.arange(10.0), 100 + np.arange(10.0)]
    spk = np.r_[np.zeros(10), np.ones(10)].astype(int)
    z = pr.within(x, spk)
    assert abs(z[:10].mean()) < 1e-9 and abs(z[10:].std() - 1) < 1e-9
    # Two speakers whose offsets agree with their y: raw r high, within 0.
    rng = np.random.default_rng(0)
    y = np.r_[rng.normal(size=10), 100 + rng.normal(size=10)]
    clip = np.arange(20) // 2
    raw = pr.corr(x, y, spk, clip, "raw", B=200)
    inw = pr.corr(np.tile(rng.normal(size=10), 2) + 50 * spk, y, spk, clip,
                  "within", B=200)
    assert raw["r"] > 0.9 and raw["r_lo"] > 0.7
    assert abs(inw["r"]) < 0.6 and inw["clips"] == 10
    # Perfectly monotone: Spearman 1.
    m = pr.corr(x, x ** 3, spk, clip, "raw", B=100)
    assert m["rho"] == pytest.approx(1.0)


def test_units_and_pauses():
    assert pr.units_of("hello big world", "en") == 3
    assert pr.units_of("hola mundo", "es") == 2
    assert pr.units_of("你好吗", "zh") == 3
    assert pr.units_of("你好 ok", "zh") == 3          # + a Latin word
    assert pr.units_of("こんにちは", "ja") == 5
    ch = np.array([(0, .1), (.1, .2), (.2, .3), (.3, .9), (.9, 1.0)])
    assert pr.pauses("abc d", ch, 0.2) == (1, pytest.approx(0.6))
    cj = np.array([(0, .1), (.1, .2), (.9, 1.0)])      # no separator
    assert pr.pauses("你好吗", cj, 0.2) == (1, pytest.approx(0.7))
    assert pr.pauses("ab", np.array([(0, .1), (.1, .2)]), 0.2) == (0, 0.0)


# ------------------------------------------------------------ prosody

@pytest.fixture(scope="module")
def const(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pc")
    synth_prosody(str(tmp / "ex"), "const")
    o = _opts(tmp)
    return pr.prosody_probe(o, log=lambda s: None)


def test_amplitude_follows_loudness_not_speed(const):
    seg = const["segment_level"]["within_speaker"]["loud"]
    assert seg["hand_std"]["r"] > 0.8 and seg["hand_std"]["r_lo"] > 0.6
    assert seg["head_std"]["r"] > 0.8
    assert abs(seg["hand_v"]["r"]) < 0.35
    assert abs(seg["head_v"]["r"]) < 0.35
    h = const["hypotheses"]["loudness: amplitude > speed"]
    assert h["diff"]["lo"] > 0 and "as hypothesised" in h["verdict"]
    # The rate hypothesis (speed > amplitude) is not supported here.
    r = const["hypotheses"]["rate: speed > amplitude"]
    assert "as hypothesised" not in r["verdict"] or r["diff"] is None \
        or r["diff"]["lo"] <= 0
    assert const["languages"].keys() == {"en", "zh", "es", "ja"}
    assert const["segments"] == 8 * 6 * 2
    assert any("loud" in line for line in const["text"])


def test_speaker_offset_vanishes_within_speaker(tmp_path):
    synth_prosody(str(tmp_path / "ex"), "confound", seed=3)
    rep = pr.prosody_probe(_opts(tmp_path), log=lambda s: None)
    ov = rep["segment_level"]["overall"]["loud"]["hand_std"]
    wi = rep["segment_level"]["within_speaker"]["loud"]["hand_std"]
    assert ov["r"] > 0.5
    assert abs(wi["r"]) < 0.35 and wi["r_lo"] < 0.2


def test_lag_scan_peaks_at_the_delay(tmp_path):
    synth_prosody(str(tmp_path / "ex"), "timevar", nspk=6, secs=12.0, seed=5)
    rep = pr.prosody_probe(_opts(tmp_path, lag_step=0.25), log=lambda s: None)
    w = rep["windowed"]
    lags = w["lags"]
    rs = [c["r"] if c["r"] is not None else -9
          for c in w["within_speaker"]["hand_amp"]]
    best = lags[int(np.argmax(rs))]
    assert best == pytest.approx(0.5, abs=0.26) and max(rs) > 0.4
    assert "frame_level" in rep and "eval_style" in rep["frame_level"]


def test_speaking_rate_drives_speed_not_amplitude(tmp_path):
    synth_prosody(str(tmp_path / "ex"), "rate", seed=2)
    rep = pr.prosody_probe(_opts(tmp_path), log=lambda s: None)
    w = rep["segment_level"]["within_speaker"]["rate"]
    assert w["hand_v"]["r"] > 0.8 and abs(w["hand_std"]["r"]) < 0.35
    h = rep["hypotheses"]["rate: speed > amplitude"]
    assert h["diff"]["lo"] > 0 and "as hypothesised" in h["verdict"]


def test_duration_is_regressed_out(tmp_path):
    rng = np.random.default_rng(1)
    n = 600
    spk = np.arange(n) % 6
    d = rng.normal(size=n)
    x, y = d + 0.3 * rng.normal(size=n), d + 0.3 * rng.normal(size=n)
    clip = np.arange(n) // 3
    raw = pr.corr(x, y, spk, clip, "within", B=100)
    part = pr.corr(x, y, spk, clip, "within", ctrl=d, B=100)
    assert raw["r"] > 0.8 and abs(part["r"]) < 0.25
    # a constant covariate (every segment the same length) is ignored
    same = pr.corr(x, y, spk, clip, "within", ctrl=np.zeros(n), B=100)
    assert same["r"] == pytest.approx(raw["r"])


def test_prosody_cli_writes_reports(tmp_path):
    synth_prosody(str(tmp_path / "ex"), "const", nspk=4, nclip=4)
    assert pr.main(["prosody", "--export", str(tmp_path / "ex"), "--out",
                    str(tmp_path / "out"), "--boot", "100", "--limit",
                    "24"]) == 0
    txt = (tmp_path / "out" / "report.txt").read_text()
    js = json.loads((tmp_path / "out" / "report.json").read_text())
    assert "prosody probe" in txt and 24 <= js["prosody"]["segments"] <= 40
    assert "windowed" in js["prosody"]


# -------------------------------------------------------------- match

def synth_match(path, hidden, mode, nspk=6, nclip=5, nseg=2, secs=12.0,
                seed=0):
    """mode content: each 1 s slot has a class c (0-3): hand.L.x oscillates
    at (c + 1) Hz, and the slot's token state is c one-hot; speaker: the
    classes are random and not in the tokens, whose state, like the motion's
    frequency and posture, is the speaker's."""
    from flow.train import hidden as fh
    rng = np.random.default_rng(seed)
    n = int(secs * FPS)
    t = np.arange(n) / FPS
    rows, arrays = [], {}
    os.makedirs(hidden, exist_ok=True)
    sh = fh.Shards(hidden, limit=400)
    for k in range(nspk):
        for c in range(nclip):
            clip = f"{k}_sp{k}_{c}_1_1"
            for s in range(nseg):
                slots = int(secs)
                cls = rng.integers(0, 4, slots)
                y = np.zeros((n, len(pr.CANON) + 1), np.float32)
                sl = np.minimum(t.astype(int), slots - 1)
                if mode == "content":
                    f = cls[sl] + 1.0
                else:
                    f = np.full(n, 1.0 + k % 4)
                    y[:, pr.CANON.index("torso.yaw")] = 0.5 * (k - 2.5)
                y[:, pr.CANON.index("hand.L.x")] = np.sin(2 * np.pi * f * t)
                y += 0.05 * rng.normal(size=y.shape).astype(np.float32)
                spoken = "x" * slots
                chars = np.stack([np.arange(slots), np.arange(slots) + 1],
                                 1).astype(np.float32)
                row = _row(clip, f"sp{k}", s, n, spoken,
                           t0=s * (secs + 1))
                rows.append(row)
                arrays[row["key"] + ".traj"] = y.astype(np.float16)
                arrays[row["key"] + ".chars"] = chars
                arrays[row["key"] + ".prosody"] = np.stack(
                    [np.exp(-3 + 0.3 * rng.normal(size=n)),
                     np.full(n, 120.0), np.ones(n)], 1).astype(np.float16)
                h = np.zeros((slots, 1, 12), np.float32)
                if mode == "content":
                    h[np.arange(slots), 0, cls] = 3.0
                else:
                    h[:, 0, 4 + k] = 3.0
                h += 0.1 * rng.normal(size=h.shape)
                span = np.stack([np.arange(slots), np.arange(slots) + 1],
                                1).astype(np.int32)
                sh.add(row["id"], row["key"], h.astype(np.float16), span,
                       span)
    sh.close()
    with open(os.path.join(hidden, "meta.json"), "w") as fp:
        json.dump({"model": "synthetic", "layers": [0], "dim": 12}, fp)
    _write(path, rows, arrays)


def _match_opts(tmp, **kw):
    a = ["match", "--export", str(tmp / "ex"), "--hidden", str(tmp / "hid"),
         "--out", str(tmp / "out"), "--layer", "0", "--hop", "15", "--k",
         "8", "--steps", "300", "--batch", "32", "--dim", "32", "--lr",
         "0.003", "--eval-every", "150", "--curve-n", "60", "--eval-n",
         "400", "--boot", "200", "--device", "cpu", "--eval-clips",
         ",".join(f"{k}_sp{k}_{c}_1_1" for k in range(6) for c in (3, 4))]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}", str(v)]
    return pr.parse(a)


@pytest.fixture(scope="module")
def content(tmp_path_factory):
    pytest.importorskip("torch")
    tmp = tmp_path_factory.mktemp("mc")
    synth_match(str(tmp / "ex"), str(tmp / "hid"), "content")
    o = _match_opts(tmp, variants="text,text_shuffled,text+prosody")
    return pr.match_probe(o, log=lambda s: None), tmp


def _v(rep, name):
    return next(r for r in rep["variants"] if r["variant"] == name)


def test_text_that_encodes_the_motion_matches(content):
    rep, _ = content
    t = _v(rep, "text")
    for sch in ("speaker", "clip"):
        d = t["eval"][sch]
        assert d["queries"] > 50
        assert d["top1"]["lo"] > 0.35, (sch, d["top1"])     # chance 0.125
        assert d["top5"]["acc"] > 0.8
    assert rep["info"]["windows_eval"] > 50
    assert len(t["curve"]) == 2 and t["curve"][-1]["train_top1"] is not None


def test_shuffled_tokens_are_at_chance(content):
    rep, _ = content
    s = _v(rep, "text_shuffled")
    for sch in pr.SCHEMES:
        d = s["eval"][sch]
        assert d["top1"]["hi"] < 0.3, (sch, d["top1"])
    both = _v(rep, "text+prosody")
    assert both["eval"]["speaker"]["top1"]["acc"] > 0.3


def test_same_speaker_negatives_remove_a_speaker_signal(tmp_path):
    pytest.importorskip("torch")
    synth_match(str(tmp_path / "ex"), str(tmp_path / "hid"), "speaker")
    o = _match_opts(tmp_path, variants="text_random,text")
    rep = pr.match_probe(o, log=lambda s: None)
    leak = _v(rep, "text_random")["eval"]
    assert leak["random"]["top1"]["lo"] > 0.3            # chance 0.125
    assert leak["speaker"]["top1"]["hi"] < 0.3
    assert leak["clip"]["top1"]["hi"] < 0.3
    clean = _v(rep, "text")["eval"]
    assert clean["speaker"]["top1"]["hi"] < 0.35


def test_speaker_split_and_cli(tmp_path):
    pytest.importorskip("torch")
    synth_match(str(tmp_path / "ex"), str(tmp_path / "hid"), "content",
                nspk=5, nclip=2)
    sem = tmp_path / "sem"
    os.makedirs(sem)
    (sem / "0_sp0_0_1_1.txt").write_text("1.0\t3.0\ticonic\t0.8\tword\n"
                                         "5.0\t6.0\tbeat\t0.0\n")
    assert pr.main(["all", "--export", str(tmp_path / "ex"), "--hidden",
                    str(tmp_path / "hid"), "--out", str(tmp_path / "out"),
                    "--layer", "0", "--hop", "30", "--k", "4", "--steps",
                    "20", "--batch", "16", "--dim", "16", "--eval-every",
                    "20", "--curve-n", "20", "--eval-n", "50", "--boot",
                    "50", "--boot", "50", "--split", "speaker",
                    "--eval-frac", "0.4", "--variants", "text,prosody",
                    "--device", "cpu", "--sem", str(sem)]) == 0
    js = json.loads((tmp_path / "out" / "report.json").read_text())
    assert js["match"]["info"]["split"] == "speaker"
    assert js["match"]["info"]["semantic_windows_eval"] >= 0
    assert "prosody" in js and "match" in js
    assert "match probe" in (tmp_path / "out" / "report.txt").read_text()


def test_semantic_reader_is_defensive(tmp_path):
    (tmp_path / "a.txt").write_text(
        "# header\n0.5 2.0 beat 0\n2.0 3.5 iconic 0.9 hand\nnot numbers\n")
    (tmp_path / "b.csv").write_text("1.0,2.0,0.7\n3.0,4.0,0.0\n")
    (tmp_path / "c.txt").write_text("garbage only\n")
    got = pr.read_sem(str(tmp_path), ["a", "b", "c", "zz"],
                      log=lambda s: None)
    assert got == {"a": [(2.0, 3.5)], "b": [(1.0, 2.0)]}
    assert pr.read_sem(str(tmp_path / "none"), ["a"],
                       log=lambda s: None) == {}
