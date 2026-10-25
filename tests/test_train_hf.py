"""
--backend hf (flow.train.hf) on a tiny random Qwen3 on CPU: the director
builds it, the prompt is flow.train.chat's (the template the adapter was
trained on, found from its train.json), the reply streams in chunks
through director.ask, stop reason and usage are set, an adapter loads on
the base. Needs torch, transformers, peft and tokenizers.
"""

from __future__ import annotations

import argparse
import json

import pytest

from flow import director as dr
from flow.train import chat

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tests import tiny_lm
    d = tmp_path_factory.mktemp("lm") / "tiny-qwen3"
    texts = ["<|im_start|>user\nHello there, how are you?<|im_end|>\n",
             'Hi! [pose {"head.tilt": 8}] Nice to see you.'] * 20
    return tiny_lm.build(str(d), texts)


def _args(*argv):
    ap = argparse.ArgumentParser()
    dr.add_args(ap)
    return ap.parse_args(list(argv))


def test_director_streams_a_local_model(tiny):
    a = _args("--backend", "hf", "--model", tiny, "--max-tokens", "12",
              "--hf-greedy")
    be = dr.make_backend(a, log=lambda s: None)
    assert be.family == "qwen3"
    msgs = [{"role": "user", "content": "Hello there\n(your body now)"}]
    assert be.render("SYS\n", msgs) == chat.prompt(
        [{"role": "system", "content": "SYS\n"}] + msgs, "qwen3")
    chunks = list(be.stream("SYS\n", msgs))
    assert be.usage["completion_tokens"] <= 12
    assert be.usage["prompt_tokens"] == len(be.tok(
        be.render("SYS\n", msgs), add_special_tokens=False)["input_ids"])
    assert be.stop_reason in ("stop", "length")
    assert be.timing["seconds"] > 0
    # Greedy is deterministic; director.ask parses what streamed.
    rep = dr.ask(be, "SYS\n", msgs, {}, log=lambda s: None)
    assert rep["raw"] == "".join(chunks)
    assert rep["refused"] is False


def test_adapter_and_its_template(tiny, tmp_path):
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    m = get_peft_model(AutoModelForCausalLM.from_pretrained(tiny), LoraConfig(
        r=4, lora_alpha=4, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"))
    run = tmp_path / "run"
    m.save_pretrained(run / "adapter")
    from flow.train import sft
    sft.save_tokenizer(sft.load_tokenizer(tiny), str(run / "adapter"),
                       "chatml")
    (run / "train.json").write_text(json.dumps({"family": "chatml"}))
    a = _args("--backend", "hf", "--model", tiny, "--hf-adapter",
              str(run / "adapter"), "--max-tokens", "4")
    be = dr.make_backend(a, log=lambda s: None)
    assert be.family == "chatml"
    assert "".join(be.stream("S", [{"role": "user", "content": "Hi"}])) \
        is not None
    assert be.usage["completion_tokens"] <= 4
