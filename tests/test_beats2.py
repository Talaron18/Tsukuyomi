"""
The beats2 teacher prompt (r3 data): several dense examples with different
openings (one per session in the data, variant 0 when served), the explicit
[frame full] entry, the reply-language line on every user turn, fewer
language switches, and the unchanged bytes of the beats prompt (q8r2). No
network: the dataset runs are mocked models.
"""

from __future__ import annotations

import json
import os
import random
import re
from types import SimpleNamespace

import pytest

from flow import body as fb
from flow import dataexport as dx
from flow import dataset as ds
from flow import director as dr
from flow import framing as fr
from flow import pose as pz
from tests.test_beats import FACES, RIGS
from tests.test_dataexport import _run
from tests.test_dataset import ROOT


def _bodies(face_body=None):
    out = [(os.path.basename(p).split(".")[0], fb.load(p)) for p in RIGS]
    if face_body is not None:
        out.append(("face", face_body))
    return out


@pytest.fixture(scope="module")
def face_bm():
    from tests.test_facevocab import FaceCore
    from tests.test_body import _info
    import tempfile
    blob = fb.measure(FaceCore(), _info(), log=lambda *a: None)
    path = os.path.join(tempfile.mkdtemp(), "face.body.json")
    fb.save(blob, path)
    return fb.load(path)


def _markers(text):
    return [json.loads(m) for m in re.findall(r"\[pose (\{[^}]*\})\]", text)]


# ----------------------------------------------------- the beats prompt

def test_beats_is_unchanged_and_beats2_differs():
    assert "beats2" in pz.PROMPTS and pz.PROMPTS[:2] == ("v1", "beats")
    for path in RIGS:
        bm = fb.load(path)
        b = pz.system_prompt(bm, "X.", prompt="beats")
        b2 = pz.system_prompt(bm, "X.", prompt="beats2")
        assert b2 != b and dx.BEATS_MARK in b2
        # variant is ignored by the older prompts
        assert pz.system_prompt(bm, "X.", prompt="beats", variant=3) == b
        assert pz.system_prompt(bm, "X.", prompt="v1", variant=3) \
            == pz.system_prompt(bm, "X.", prompt="v1")
        assert pz.FRAME_ENTRY not in b2


# ------------------------------------------------------ dense variants

def test_dense_variants_are_valid_for_every_body(face_bm):
    for name, bm in _bodies(face_bm):
        have = set(pz.offered(bm))
        vs = pz.dense_variants(bm)
        assert len(vs) == 6 == pz.variant_count(bm), name
        for v in vs:
            ms = _markers(v)
            assert len(ms) >= 5
            for m in ms:
                assert set(m) <= have | {"dur", "hold"}, (name, m)
                assert set(m) & have          # no marker of only dur / hold
                for k, x in m.items():
                    if k in ("dur", "hold"):
                        continue
                    lo, hi = bm.limits(k, 0)[:2] if k in bm.fidx else (
                        -10, 10)
                    assert lo - 1e-6 <= x <= hi + 1e-6 or k.startswith(
                        ("head.", "gaze.")), (name, k, x)
        assert len(set(vs)) == 6
        # a different first marker per variant
        firsts = [json.dumps(_markers(v)[0], sort_keys=True) for v in vs]
        assert len(set(firsts)) >= 5, name


def test_the_openings_differ_on_hiyori():
    bm = fb.load(RIGS[0])
    vs = pz.dense_variants(bm)
    first = [_markers(v)[0] for v in vs]
    assert "eye.L.smile" in first[0]                       # a smiling eye
    assert "gaze.x" in first[1] and first[1]["gaze.x"] < 0   # a look away
    assert first[2].get("eye.L.open") == 0                 # a laugh, eyes shut
    assert first[3]["brow.L.lift"] < 0 and first[3]["mouth.form"] < 0
    assert "cheek.blush" in first[4]                       # a blush
    assert list(first[5])[0] == "head.tilt" and first[5]["dur"] >= 0.4
    # none opens with the brow lift + pitch + yaw of the beats example
    assert not any(f.get("brow.L.lift") == 0.7 and "head.yaw" in f
                   for f in first)
    # the beats prompt's one dense example is not among them
    old = pz.system_prompt(bm, "X.", prompt="beats")
    assert old.count("Oh, you really did that?") == 1
    assert not any("Oh, you really did that?" in v for v in vs)


def test_the_prompt_shows_the_variant_asked_for_and_variant_0_is_served():
    for name, bm in _bodies():
        prompts = [pz.system_prompt(bm, "X.", prompt="beats2", variant=k)
                   for k in range(pz.variant_count(bm))]
        assert len(set(prompts)) == len(prompts)
        dense = pz.dense_variants(bm)
        for k, p in enumerate(prompts):
            assert dense[k] in p and all(
                d not in p for j, d in enumerate(dense) if j != k)
        assert pz.system_prompt(bm, "X.", prompt="beats2") == prompts[0]
        # a variant wraps round
        assert pz.system_prompt(bm, "X.", prompt="beats2", variant=6) \
            == prompts[0]
    # the examples of every variant open differently
    bm = fb.load(RIGS[0])
    ex = [pz.system_prompt(bm, "X.", prompt="beats2", variant=k)
          for k in range(6)]
    starts = {e[e.index("Examples of placement"):].split("\n")[1][:60]
              for e in ex}
    assert len(starts) == 6


def test_opening_variety_sentence():
    bm = fb.load(RIGS[0])
    p = pz.system_prompt(bm, "X.", prompt="beats2")
    assert "Vary how your replies open: never start every reply with the " \
        "same expression" in p
    assert "Vary how your replies open" not in pz.system_prompt(
        bm, "X.", prompt="beats")


# --------------------------------------------------- variant per session

def test_variant_is_deterministic_per_session_and_spread():
    bm = fb.load(RIGS[0])
    sess = ds.plan_sessions(["Hiyori"], ["cheerful", "sleepy"],
                            {"personas": [{"id": "cheerful"},
                                          {"id": "sleepy"}]},
                            60, 2, seed=7, bust_share=0.7)
    again = ds.plan_sessions(["Hiyori"], ["cheerful", "sleepy"],
                             {"personas": [{"id": "cheerful"},
                                           {"id": "sleepy"}]},
                             60, 2, seed=7, bust_share=0.7)
    v = [pz.variant_for(s["seed"], bm) for s in sess]
    assert v == [pz.variant_for(s["seed"], bm) for s in again]
    assert set(v) == set(range(6))
    assert all(0 <= x < pz.variant_count(bm) for x in v)


def test_dataset_picks_a_prompt_variant_per_session(tmp_path):
    out, recs = _run(tmp_path, FACES, avatars="Hiyori", sessions=8, turns=2,
                     prompt="beats2")
    reg = dx.load_registry(out)
    by_sess = {}
    for r in recs:
        by_sess.setdefault(r["session"], set()).add(
            (r["avatar"]["variant"], r["avatar"]["prompt_sha"]))
    assert all(len(x) == 1 for x in by_sess.values())
    got = {next(iter(x))[0] for x in by_sess.values()}
    assert len(got) >= 3
    bm = fb.load(RIGS[0])
    dense = pz.dense_variants(bm)
    for r in recs:
        text = reg["prompts"][r["avatar"]["prompt_sha"]]
        k = r["avatar"]["variant"]
        assert dense[k] in text
        # the session's seed alone decides it (a turn's seed is its + turn)
        assert k == pz.variant_for(r["seed"] - r["turn"], bm)
    # beats2 data is recognised as beats data
    assert dx.default_max_beats(out) == dx.BEATS_MAX
    assert dx.default_max_hands(out) == dx.MAX_HANDS
    assert dx.default_lang_filter(out) is True


# --------------------------------------------------------- frame entry

def test_frame_entry_names_whole_body_moves_and_outfits_not_a_wave():
    for name, bm in _bodies():
        p = pz.system_prompt(bm, "X.", prompt="beats2",
                             moves=dr.move_menu(
                                 SimpleNamespace(body_moves=True), bm)[0])
        if not fr.can_frame(bm):
            assert "[frame bust] or" not in p and "[frame full]" not in p
            continue
        i = p.index("[frame bust] or [frame full]")
        entry = p[i:p.index("\n", i)]
        assert "Write [frame full] right before a dance or another " \
            "whole-body move" in entry
        assert "and before showing an outfit" in entry
        assert "never for a wave or a single hand move" in entry
        assert '[move {"name": "dance"}]' in entry
        assert '"wave"' not in entry and '"clap"' not in entry
        # the move example shows the marker before the dance
        assert "[frame full][move" in p
    # without the move menu it still says when
    bm = fb.load(RIGS[0])
    p = pz.system_prompt(bm, "X.", prompt="beats2")
    assert "right before a dance or another whole-body move and" in p
    # beats keeps its own wording
    assert pz.FRAME_ENTRY.format(ease=fr.VIEW_EASE) in pz.system_prompt(
        bm, "X.", prompt="beats")


# ------------------------------------------------------ reply language

def test_reply_line_from_the_main_language():
    assert pz.reply_line("Hello, how are you today?") == "(reply in: English)"
    assert pz.reply_line("今日はいい天気ですね") == "(reply in: Japanese)"
    assert pz.reply_line("你今天过得怎么样？") == "(reply in: Chinese)"
    assert pz.reply_line("오늘 어때요") == "(reply in: Korean)"
    assert pz.reply_line("OK!") == "" == pz.reply_line("")
    # mixed: the language of most of the line
    assert pz.reply_line("That sounds really lovely, 可愛い!") \
        == "(reply in: English)"
    assert pz.reply_line("今日はNetflixで映画を見たよ") \
        == "(reply in: Japanese)"
    assert ds.detect_lang("what a nice day") == pz.detect_lang(
        "what a nice day") == "en"


def test_turn_tail_per_prompt():
    bm = fb.load(RIGS[0])
    cam = pz.camera_now(bm, "bust")
    assert pz.turn_tail(bm, "bust", "Hello there, friend", "v1") == ""
    assert pz.turn_tail(bm, "bust", "Hello there, friend", "beats") \
        == "\n" + cam
    assert pz.turn_tail(bm, "bust", "Hello there, friend", "beats2") \
        == "\n" + cam + "\n(reply in: English)"
    assert pz.turn_tail(bm, "bust", "OK!", "beats2") == "\n" + cam


def test_data_and_serving_write_the_same_user_turn(tmp_path):
    out, recs = _run(tmp_path, FACES, avatars="Hiyori,seed-san", sessions=4,
                     turns=2, prompt="beats2")
    opts = ds.run_opts(out)
    assert opts.teacher_prompt == "beats2"
    reg = dx.load_registry(out)
    for r in recs:
        bm = fb.load(RIGS[0]) if r["avatar"]["name"] == "Hiyori" \
            else fb.load(RIGS[2])
        fram = (r["framing"]["start"])
        # what the director / live loop sends: text + pose line + tail
        served = r["user"] + "\n" + reg["avatars"][r["avatar"]["name"]][
            "pose_now"] + pz.turn_tail(bm, fram, r["user"], "beats2")
        assert r["user_content"] == served
        assert r["user_content"].splitlines()[-1].startswith(
            "(reply in: ") or pz.reply_line(r["user"]) == ""
    for r in recs:
        for h in r["history"]:
            assert "(camera:" in h["user_content"]


def test_director_and_live_build_the_tail_with_the_same_function():
    import inspect
    from flow import live
    for mod in (dr, live, ds):
        src = inspect.getsource(mod)
        assert "pz.turn_tail(" in src, mod.__name__
        assert "pz.camera_now(" not in src, mod.__name__


def test_serving_flags_accept_beats2():
    ap = dr.argparse.ArgumentParser()
    dr.add_args(ap)
    assert ap.parse_args([]).body_prompt == "v1"
    assert ap.parse_args(["--body-prompt", "beats2"]).body_prompt == "beats2"
    a = ds.parse_opts(["gen", "--teacher-prompt", "beats"])
    assert a.teacher_prompt == "beats"
    assert ds.parse_opts(["gen"]).teacher_prompt == "beats2"


# ------------------------------------------------------ personas switch

def _lang_run(persona, opts, n=4000, turns=6):
    """Language changes between consecutive turns of simulated sessions,
    drawn as run_session draws them."""
    changes = total = 0
    langs = persona["languages"]
    for s in range(n):
        rng = random.Random(s)
        lang0 = ds._pick(rng, langs)
        prev = lang0
        for _ in range(turns):
            lang = lang0 if rng.random() > ds.switch_prob(persona, opts) \
                else ds._pick(rng, langs)
            changes += lang != prev
            total += 1
            prev = lang
    return changes / total


def test_beats2_switches_language_less_and_r2_keeps_its_own():
    cfg = ds.load_config()
    by = {p["id"]: p for p in cfg["personas"]}
    # the personas file is untouched: r2 reproduces
    assert by["cheerful"]["switch"] == 0.15 and by["tsundere"]["switch"] == 0.1
    o2 = ds.parse_opts(["gen"])
    o1 = ds.parse_opts(["gen", "--teacher-prompt", "beats"])
    ov = ds.parse_opts(["gen", "--lang-switch", "0.5"])
    for p in cfg["personas"]:
        assert ds.switch_prob(p, o1) == p["switch"]
        assert ds.switch_prob(p, o2) == ds.LANG_SWITCH == 0.03
        assert ds.switch_prob(p, ov) == 0.5
        # the "mix" setting is untouched by it
    assert ds.parse_opts(["gen", "--teacher-prompt", "v1"]).lang_switch \
        is None
    p = by["cheerful"]
    assert _lang_run(p, o2) < 0.05 < 0.1 < _lang_run(p, o1)


def test_overall_language_balance_is_kept():
    cfg = ds.load_config()
    opts = ds.parse_opts(["gen"])
    tot = {"en": 0, "ja": 0, "zh": 0}
    n = 0
    for pi, p in enumerate(cfg["personas"]):
        for s in range(3000):
            rng = random.Random(s * 7 + pi)
            lang0 = ds._pick(rng, p["languages"])
            for _ in range(6):
                lang = lang0 if rng.random() > ds.switch_prob(p, opts) \
                    else ds._pick(rng, p["languages"])
                tot[lang] += 1
                n += 1
    share = {k: v / n for k, v in tot.items()}
    assert share["en"] > 0.3 and share["ja"] > 0.3 and share["zh"] > 0.15


def test_run_opts_of_older_runs_stay_beats(tmp_path):
    out = tmp_path / "old"
    out.mkdir()
    (out / "runs.jsonl").write_text(json.dumps(
        {"opts": {"sessions": 3}}) + "\n")
    assert ds.run_opts(str(out)).teacher_prompt == "beats"


def test_budget_line_agrees_with_the_frame_entry():
    bm = fb.load(RIGS[0])
    b2 = pz.system_prompt(bm, "X.", prompt="beats2")
    assert "never for a wave or a single hand move" in b2
    assert "a pose, a big celebration" not in b2
    assert "a pose, a big celebration" in pz.system_prompt(
        bm, "X.", prompt="beats")
