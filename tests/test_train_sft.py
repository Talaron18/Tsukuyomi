"""
flow.train.sft on CPU with a tiny random Qwen3 (tests/tiny_lm.py): the
session split (no session on both sides, stable as data grows), labels
only on the replies and their end token in the chat module's template,
the cut at a reply boundary, the template check against the tokenizer's
own, then a few real optimizer steps (unquantized: bitsandbytes needs
CUDA) with eval loss, a checkpoint resumed, a refusal to resume on other
data, the merged model and the Modelfile. The 4-bit and Unsloth paths
need a GPU and are not run here. Needs torch, transformers, peft and
tokenizers; skips without them.
"""

from __future__ import annotations

import json
import os

import pytest

from flow.train import chat
from flow.train import sft

SYSTEM = "You are Hiyori. Your body: head, hands. Write [pose {...}] markers."


def _examples(n=12, turns=3):
    out = []
    for i in range(n):
        msgs = [{"role": "system", "content": SYSTEM}]
        for t in range(turns):
            msgs.append({"role": "user",
                         "content": f"Hello number {i} {t}!\n(your body now: "
                                    "head.yaw 0)"})
            msgs.append({"role": "assistant",
                         "content": f'Hi! [pose {{"head.tilt": {t + 2}}}] '
                                    f"Nice to see you {i}."})
        out.append({"messages": msgs,
                    "meta": {"session": f"Hiyori.cheerful.{i:04d}"}})
    return out


def test_split_by_session_is_stable():
    ex = _examples(200, 1)
    tr, ev, info = sft.split(ex, 0.1)
    assert not set(info["train_sessions"]) & set(info["eval_sessions"])
    assert 5 < len(ev) < 40 and len(tr) + len(ev) == 200
    # More data: the old sessions stay where they were.
    tr2, ev2, info2 = sft.split(_examples(400, 1), 0.1)
    assert set(info["eval_sessions"]) <= set(info2["eval_sessions"])
    # Per-turn rows of one session never part.
    rows = [{"messages": [], "meta": {"id": f"s{i // 3}/t{i % 3:02d}"}}
            for i in range(90)]
    _tr, _ev, inf = sft.split(rows, 0.2)
    assert not set(inf["train_sessions"]) & set(inf["eval_sessions"])
    with pytest.raises(SystemExit, match="--meta"):
        sft.split([{"messages": []}], 0.1)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tests import tiny_lm
    d = tmp_path_factory.mktemp("lm") / "tiny-qwen3"
    texts = [chat.render(e["messages"], "qwen3")[0] for e in _examples(20)]
    return tiny_lm.build(str(d), texts * 3)


def test_labels_are_the_replies(tiny):
    tok = sft.load_tokenizer(tiny)
    ex = sft.clean(_examples(1)[0])
    r = sft.encode(tok, ex, "qwen3")
    assert r["turns"] == 3 and r["cut"] == 0
    ids, lab = r["input_ids"], r["labels"]
    assert len(ids) == len(lab)
    # The labelled runs decode to each reply plus <|im_end|>.
    runs, cur = [], []
    for i, l_ in zip(ids, lab):
        if l_ != -100:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    want = [m["content"] + "<|im_end|>" for m in ex["messages"]
            if m["role"] == "assistant"]
    assert [tok.decode(x) for x in runs] == want
    # Everything decodes back to the template's text.
    assert tok.decode(ids) == chat.render(ex["messages"], "qwen3")[0]
    # Cut after the last reply that fits; none fits: left out.
    full = len(ids)
    r2 = sft.encode(tok, ex, "qwen3", max_len=full - 5)
    assert r2["turns"] == 2 and r2["cut"] == 1
    assert tok.decode([i for i, l_ in zip(r2["input_ids"], r2["labels"])
                       if l_ != -100]).endswith("<|im_end|>")
    assert sft.encode(tok, ex, "qwen3", max_len=20) is None
    rep = sft.length_report(tok, [ex], "qwen3")
    assert rep["examples"]["max"] == full and rep["turns"]["max"] == 3
    assert rep["auto_max_len"] == 1024


def test_template_check(tiny):
    tok = sft.load_tokenizer(tiny)
    ex = sft.clean(_examples(1)[0])
    assert sft.check_template(tok, "qwen3", ex) == []
    probs = sft.check_template(tok, "chatml", ex)
    assert probs and "differs" in probs[0]


def _train_argv(data, out, *extra):
    return ["train", "--data", str(data), "--out", str(out), "--base",
            "", "--no-4bit", "--backend", "peft", "--accum", "2",
            "--eval-frac", "0.2", "--eval-steps", "2", "--save-steps", "2",
            "--log-steps", "1", "--probe-steps", "1", "--lr", "3e-3",
            *extra]


def test_train_resume_merge(tiny, tmp_path):
    pytest.importorskip("peft")
    data = tmp_path / "sft.jsonl"
    with open(data, "w") as fh:
        for e in _examples(12):
            fh.write(json.dumps(e) + "\n")
    out = tmp_path / "run"
    argv = _train_argv(data, out, "--max-steps", "4")
    argv[argv.index("--base") + 1] = tiny
    o = sft.parse(argv)
    s = sft.train(o)
    assert s["family"] == "qwen3" and s["backend"] == "peft"
    assert os.path.exists(out / "adapter" / "adapter_config.json")
    assert "eval_loss" in s["metrics"]
    split = json.load(open(out / "split.json"))
    assert split["eval_sessions"] and not set(split["eval_sessions"]) & set(
        split["train_sessions"])
    ck = sft.last_checkpoint(str(out))
    assert ck and ck.endswith("checkpoint-4")
    # The saved tokenizer carries our template.
    tok = sft.load_tokenizer(str(out / "adapter"))
    assert tok.chat_template == chat.jinja("qwen3")
    # Resumed: two more steps from checkpoint-4.
    argv2 = list(argv)
    argv2[argv2.index("--max-steps") + 1] = "6"
    s2 = sft.train(sft.parse(argv2))
    assert sft.last_checkpoint(str(out)).endswith("checkpoint-6")
    assert s2["metrics"]["train_loss"] > 0
    # Other data: no resume.
    with open(data, "a") as fh:
        fh.write(json.dumps(_examples(13)[12]) + "\n")
    with pytest.raises(SystemExit, match="other data"):
        sft.train(sft.parse(argv2))
    # Merged into the unquantized base; it loads and generates.
    m = sft.merge(sft.parse(["merge", "--run", str(out), "--dtype",
                             "float32", "--device", "cpu"]))
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(m)
    tok = sft.load_tokenizer(m)
    ids = tok(chat.prompt(_examples(1)[0]["messages"][:2], "qwen3"),
              add_special_tokens=False, return_tensors="pt")["input_ids"]
    assert model.generate(ids, max_new_tokens=3, do_sample=False).shape[1] \
        == ids.shape[1] + 3
    # GGUF commands without llama.cpp; the Modelfile.
    cmds = sft.gguf(sft.parse(["gguf", "--run", str(out)]))
    assert "convert_hf_to_gguf.py" in cmds[0][1] and cmds[1][-1] == "Q4_K_M"
    mf = sft.write_modelfile(sft.parse(["modelfile", "--run", str(out)]))
    text = open(mf).read()
    assert chat.OLLAMA["qwen3"] in text and "tsk-Q4_K_M.gguf" in text


def test_loss_only_where_labels_are(tiny):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(tiny)
    tok = sft.load_tokenizer(tiny)
    r = sft.encode(tok, sft.clean(_examples(1)[0]), "qwen3")
    inp = sft.collate([r])
    with torch.no_grad():
        a = sft.masked_loss(model, inp, keep_logits=True)
        b = sft.masked_loss(model, inp, keep_logits=False)
    assert float(a) == pytest.approx(float(b), rel=1e-5)
