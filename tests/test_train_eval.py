"""
flow.train.evalsft without a model: the held-out sessions of a split
replayed turn by turn through a backend (the recorded teacher replies, a
mocked student that keeps its own history), the per-turn numbers (parse,
refusals by text, markers, unplayed, reach, latency) and their summary,
the comparison table and the A/B pack's director runs (not rendered).
Pure numpy (the generation is test_dataset's, mocked).
"""

from __future__ import annotations

import json
import os

import pytest

from flow import dataset as ds
from flow import director as dr
from flow.train import evalsft
from tests.test_dataset import REPLY, CharLLM, UserLLM, _opts


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("ev") / "run"
    res = ds.generate(_opts(out, sessions=4, turns=2),
                      backend_factory=lambda role, av: CharLLM() if
                      role == "character" else UserLLM(), log=lambda s: None)
    assert res["ok"] == 4
    sessions = sorted({r["session"] for r in ds.iter_records(str(out))})
    split = out / "split.json"
    split.write_text(json.dumps({"eval_sessions": sessions[:3],
                                 "train_sessions": sessions[3:]}))
    return out, split


def test_refusal_text():
    assert evalsft.refusal_text("I'm sorry, but I can't do that.")
    assert evalsft.refusal_text("As an AI, I have no body.")
    assert evalsft.refusal_text("抱歉，我不能")
    assert evalsft.refusal_text("")
    assert not evalsft.refusal_text("Sorry I'm late! [pose {}]")


def test_recorded_teacher_and_a_student(data, tmp_path, monkeypatch):
    out, split = data
    ev = tmp_path / "eval"
    o = evalsft.parse(["run", "--data", str(out), "--split", str(split),
                       "--out", str(ev), "--name", "teacher", "--recorded"])
    s = evalsft.run(o, log=lambda s: None)
    assert s["sessions"] == 3 and s["turns"] == 6
    assert s["parse_rate"] == 0 and s["refusal_text_rate"] == 0
    assert s["markers_per_turn"]["pose"] == 2 and s["unplayed"] == {"nod": 6}
    assert s["reach"]["targets"] == 18 and s["first_text_seconds"]["p50"] \
        is not None
    rows = [json.loads(x) for x in open(ev / "teacher.jsonl")]
    assert [r["reply_raw"] for r in rows] == [REPLY] * 6
    # A student: its own replies are its history.
    sent = []

    class Student(CharLLM):
        def stream(self, system, messages):
            sent.append(messages)
            yield from super().stream(system, messages)
    bad = "I'm sorry, but I can't. [pose {\"nope.x\": 1}]"
    monkeypatch.setattr(dr, "make_backend",
                        lambda a, log=print: Student([bad, REPLY]))
    o = evalsft.parse(["run", "--data", str(out), "--split", str(split),
                       "--out", str(ev), "--name", "student"])
    s2 = evalsft.run(o, log=lambda s: None)
    assert s2["refusal_text_rate"] == 0.5 and s2["parse_rate"] == 0.5
    assert sent[1][1] == {"role": "assistant", "content": bad}
    rows = [json.loads(x) for x in open(ev / "student.jsonl")]
    assert rows[0]["tokens_per_second"] is None or \
        rows[0]["tokens_per_second"] > 0
    text = evalsft.compare(str(ev))
    assert "teacher" in text and "student" in text
    # The A/B pack: each model's first turns played by the director
    # (lite bodies have no model to render: skipped).
    o = evalsft.parse(["ab", "--data", str(out), "--out", str(ev),
                       "--no-render", "--n", "1", "--tts", "none"])
    evalsft.ab(o, log=lambda s: None)
    vrm = [d for d in (ev / "ab").glob("*/teacher/0")]
    if vrm:
        assert (vrm[0] / "actions.json").exists()
        assert (vrm[0].parent.parent / "student" / "0" / "reply.txt").exists()
    assert os.path.isdir(ev / "ab")
