"""
The extended face vocabulary (flow.body EXTENDED: squint, cheek puff, lip
pucker, funnel, tongue, tears) of a face-tracking rig, the reply-language
rule of the beats prompt and the language filter of flow.dataexport. No
GL, no network: a synthetic body, records edited in memory.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from flow import body as fb
from flow import dataexport as dx
from flow import framing as fr
from flow import micro
from flow import pose as pz
from flow import vrm as fv
from tests.test_body import FakeCore, _info
from tests.test_dataset import ROOT

NEW = ("eye.L.squint", "eye.R.squint", "cheek.puff", "mouth.pucker",
       "mouth.funnel", "tongue.out", "eye.tears")
EXTRA = [("ParamEyeLSquint", 0, 1), ("ParamEyeRSquint", 0, 1),
         ("ParamCheekPuff", 0, 1), ("ParamMouthPuckerWiden", -1, 1),
         ("ParamMouthFunnel", 0, 1), ("ParamTongueOut", 0, 1),
         ("ParamTearDown_1", 0, 1), ("ParamTearDown_2", 0, 1),
         ("ParamMouthShrug", 0, 1)]


class FaceCore(FakeCore):
    """FakeCore plus a face-tracking rig's extra face parameters (they
    move nothing here: the declared variables come from the ids)."""

    def __init__(self):
        super().__init__()
        self.param_ids = self.param_ids + [p for p, _, _ in EXTRA]
        self.param_min = np.r_[self.param_min, [lo for _, lo, _ in EXTRA]]
        self.param_max = np.r_[self.param_max, [hi for _, _, hi in EXTRA]]
        self.param_default = np.r_[self.param_default, np.zeros(len(EXTRA))]
        self.set()


@pytest.fixture(scope="module")
def face_body(tmp_path_factory):
    blob = fb.measure(FaceCore(), _info(), log=lambda *a: None)
    path = tmp_path_factory.mktemp("face") / "face.body.json"
    fb.save(blob, str(path))
    return blob, fb.load(str(path))


# ------------------------------------------------------------ variables

def test_extended_ids_are_declared(face_body):
    blob, _bm = face_body
    d = blob["declared"]
    assert d["eye.L.squint"] == ["ParamEyeLSquint", 1.0]
    assert d["eye.R.squint"][0] == "ParamEyeRSquint"
    assert d["cheek.puff"][0] == "ParamCheekPuff"
    assert d["mouth.pucker"][0] == "ParamMouthPuckerWiden"
    assert d["mouth.funnel"][0] == "ParamMouthFunnel"
    assert d["tongue.out"][0] == "ParamTongueOut"
    # three tear levels in the rig: the first is the variable
    assert d["eye.tears"][0] == "ParamTearDown_1"
    assert set(NEW) <= set(blob["features"])
    assert set(fb.EXTENDED) == set(NEW)


def test_extended_variables_are_offered_with_their_ranges(face_body):
    _blob, bm = face_body
    off = pz.offered(bm)
    assert set(NEW) <= set(off)
    assert bm.limits("mouth.pucker", 0) == pytest.approx((-1.0, 1.0))
    assert bm.limits("cheek.puff", 0) == pytest.approx((0.0, 1.0))
    d = pz.describe_body(bm)
    assert "mouth.pucker: 0.00 -> [-1.00, 1.00]; - purses the lips, + " \
        "widens the mouth" in d
    # all offered as face for the motion budget
    assert fr.budget(bm)["face"] >= len(NEW)
    assert all(dx.is_face(v) and dx.keep_key(v, 1) for v in NEW)


def test_alias_ids_are_found():
    core = FaceCore()
    core.param_ids = [("ParamEyeSquintL" if p == "ParamEyeLSquint" else
                       "ParamTears" if p == "ParamTearDown_1" else p)
                      for p in core.param_ids]
    blob = fb.measure(core, _info(), log=lambda *a: None)
    assert blob["declared"]["eye.L.squint"][0] == "ParamEyeSquintL"
    assert blob["declared"]["eye.R.squint"][0] == "ParamEyeRSquint"
    assert blob["declared"]["eye.tears"][0] == "ParamTears"


def test_arkit_style_ids_are_found():
    names = {"ParamEyeLSquint": "EyeSquintLeft",
             "ParamEyeRSquint": "ParamEyeSquintRight",
             "ParamCheekPuff": "CheekPuff", "ParamMouthFunnel": "MouthFunnel",
             "ParamTongueOut": "TongueOut",
             "ParamMouthPuckerWiden": "MouthPucker"}
    core = FaceCore()
    core.param_ids = [names.get(p, p) for p in core.param_ids]
    blob = fb.measure(core, _info(), log=lambda *a: None)
    d = {n: p for n, (p, _s) in blob["declared"].items()}
    assert d["eye.L.squint"] == "EyeSquintLeft"
    assert d["eye.R.squint"] == "ParamEyeSquintRight"
    assert d["cheek.puff"] == "CheekPuff"
    assert d["mouth.funnel"] == "MouthFunnel"
    assert d["tongue.out"] == "TongueOut"
    # ARKit's MouthPucker has the opposite sign: not mapped
    assert "mouth.pucker" not in d


def test_beats_prompt_offers_them_only_to_the_body_that_has_them(face_body):
    _blob, bm = face_body
    b = pz.system_prompt(bm, "X.", prompt="beats")
    rules = b[b.index("How to move"):b.index("Examples of placement")]
    for v in ("eye.L.squint", "cheek.puff", "mouth.pucker", "mouth.funnel",
              "tongue.out", "eye.tears"):
        assert f'"{v}"' in rules or v in rules, v
    # the body without them: nothing of it, and no "does not exist" line
    plain = fb.load(os.path.join(ROOT, "rigs", "Hiyori.body.json"))
    p = pz.system_prompt(plain, "X.", prompt="beats")
    for v in NEW:
        assert v not in p
    assert "small tricks of the face" not in p


def test_micro_and_beat_columns_treat_them_as_face():
    assert any("tongue.".startswith(x[:-1] + ".") or x == "tongue."
               for x in micro.GROUPS["smile"])
    from flow import beat2
    assert all(v.startswith(beat2.NAN_PREFIX) for v in NEW)
    assert all(v.startswith(micro.GROUPS["smile"] + micro.GROUPS["brow"])
               for v in NEW)


# ------------------------------------------------ other avatars unchanged

def _digest(bm, prompt):
    t = pz.system_prompt(bm, "X.", prompt=prompt)
    if prompt == "beats":
        t = t.replace(pz.LANG_RULE, "")
    return hashlib.sha256(t.encode()).hexdigest()[:16]


# sha256[:16] of the prompts before the face vocabulary and the language
# rule (the beats ones without the rule's sentence).
PINNED = {
    "Hiyori": ("44842b09fe417e26", "0fe164a79656bdae"),
    "lafei": ("7367f7386be57589", "c601c99615310110"),
    "Natori": ("bba679326c22bcd5", "d6c262a1bcd4bf48"),
    "seed-san": ("719fee22fd233b9c", "e8aff804b814f8dd"),
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_other_avatars_prompts_are_unchanged(name):
    path = os.path.join(ROOT, "rigs", f"{name}.body.json")
    if not os.path.exists(path):
        pytest.skip(f"no {path}")
    bm = (fv.body_for(None, path, log=lambda s: None)[0]
          if name == "seed-san" else fb.load(path))
    assert (_digest(bm, "v1"), _digest(bm, "beats")) == PINNED[name]


# ------------------------------------------------------------- language

def test_beats_prompt_asks_for_the_users_latest_language(face_body):
    _blob, bm = face_body
    b = pz.system_prompt(bm, "X.", prompt="beats")
    assert pz.LANG_RULE in b
    assert "even if your earlier replies used another language" in b
    assert "main language" in b
    assert "even if your earlier" not in pz.system_prompt(bm, "X.")


def _r(i, sid, turn, user, reply, mix=None):
    return {"id": i, "session": sid, "turn": turn, "refused": False,
            "lang": {"asked": user, "mix": mix, "user": user,
                     "reply": reply}}


def test_wrong_language_pairs():
    assert dx.wrong_language(_r("a", "s", 0, "en", "ja")) == "en->ja"
    assert dx.wrong_language(_r("a", "s", 0, "en", "en")) is None
    assert dx.wrong_language(_r("a", "s", 0, "en", "ja", mix="ja")) is None
    assert dx.wrong_language({"lang": {"user": None, "reply": "en"}}) is None
    assert dx.wrong_language({}) is None


def test_annotate_drops_wrong_language_only_when_asked():
    recs = [_r("a", "s", 0, "en", "en"), _r("b", "s", 1, "zh", "en"),
            _r("c", "s", 2, "zh", "en", mix="en")]
    for r in recs:
        r.update(spoken="x", markers=[{"tag": "pose"}], reply_raw="x",
                 user=f"line {r['id']} " + "word" * (ord(r["id"]) - 96),
                 avatar={"name": "A", "kind": "live2d"}, persona="p",
                 topic="t")
    off = dx.annotate(recs)
    assert all(k for _f, k in off.values())
    on = dx.annotate(recs, lang_filter=True)
    assert on["b"][0].count("language") == 1 and not on["b"][1]
    assert on["a"][1] and on["c"][1]
    # a session export is cut before the first dropped turn
    assert [r["id"] for r in dx.session_turns(recs, on)["s"]] == ["a"]
    s = dx.stats(recs, on)
    assert s["by"]["lang_dropped"] == {"zh->en": 1}
    assert "language: 1 turns dropped, by pair: zh->en" in dx.stats_text(s)


def test_default_lang_filter_follows_the_prompt(tmp_path):
    from tests.test_beats import FACES
    from tests.test_dataexport import _run
    for prompt, want in (("beats", True), ("v1", False)):
        out, _ = _run(tmp_path / prompt, FACES, avatars="seed-san",
                      sessions=1, turns=1, prompt=prompt)
        assert dx.default_lang_filter(out) is want


def test_sft_command_flags_and_default(tmp_path, capsys):
    from tests.test_beats import FACES
    from tests.test_dataexport import _run
    out, recs = _run(tmp_path, FACES, avatars="seed-san", sessions=2,
                     turns=2, prompt="beats")
    # edit the shards: the second turn of every session answers in another
    # language than the user's line
    for f in sorted(os.listdir(os.path.join(out, "shards"))):
        p = os.path.join(out, "shards", f)
        rows = [json.loads(x) for x in open(p, encoding="utf-8")]
        for r in rows:
            if r["turn"] == max(x["turn"] for x in rows):
                r["lang"].update(user="en", reply="ja", mix=None)
            else:
                r["lang"].update(user="en", reply="en", mix=None)
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def run(*flags):
        dst = tmp_path / "x.jsonl"
        assert dx.main(["sft", out, "--out", str(dst), *flags]) == 0
        o = capsys.readouterr().out
        rows = [json.loads(x) for x in dst.read_text().splitlines()]
        return rows, o

    rows, o = run()
    assert f"language: {len({r['session'] for r in recs})} turns dropped, by pair: en->ja" in o
    assert rows and all(sum(m["role"] == "assistant" for m in x["messages"])
                        == 1 for x in rows)
    rows, o = run("--no-lang-filter")
    assert "language:" not in o
    assert rows and all(sum(m["role"] == "assistant" for m in x["messages"])
                        == 2 for x in rows)
    # per turn: the second turns are dropped
    rows, o = run("--per", "turn")
    assert rows and all(sum(m["role"] == "assistant" for m in x["messages"]) == 1
                for x in rows)


def test_per_turn_export_leaves_out_turns_after_a_wrong_language_reply():
    recs = [_r("a", "s", 0, "en", "en"), _r("b", "s", 1, "en", "ja"),
            _r("c", "s", 2, "en", "en")]
    for i, r in enumerate(recs):
        r.update(spoken="x", markers=[{"tag": "pose"}], reply_raw=f"r{i}",
                 user=f"line {i} " + "word" * (3 * i + 1),
                 user_content=f"u{i}", avatar={"name": "A", "kind": "live2d",
                                              "prompt_sha": "p"},
                 persona="p", topic="t", reach={},
                 history=[{"user_content": f"u{j}", "reply": f"r{j}"}
                          for j in range(i)])
    notes = dx.annotate(recs, lang_filter=True)
    assert [k for _f, k in notes.values()] == [True, False, True]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "avatars.json"), "w") as fh:
            json.dump({"prompts": {"p": "SYS"}, "avatars": {}}, fh)
        exs = dx.sft_examples(d, recs, notes, per="turn", meta=True)
        assert [e["meta"]["id"] for e in exs] == ["a"]
        off = dx.sft_examples(d, recs, dx.annotate(recs), per="turn",
                              meta=True)
        assert [e["meta"]["id"] for e in off] == ["a", "b", "c"]
