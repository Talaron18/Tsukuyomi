#!/usr/bin/env python3
"""
A local transformers model as a director backend (`--backend hf`): the
base (or a merged model) with an optional LoRA adapter, prompts rendered
by flow.train.chat -- the template the adapter was trained on -- and the
reply streamed through TextIteratorStreamer, so the parser, the timing
and the live loop see chunks as they come, as with Ollama.

    python -m flow.live ... --backend hf --model Qwen/Qwen3-8B \\
        --hf-adapter runs/sft/q8/adapter
    python -m flow.director ... --backend hf --model runs/sft/q8/merged

The family comes from --hf-template, else the adapter's train.json (one
folder up), else the model's name. --hf-4bit loads the base in 4 bit
(bitsandbytes, CUDA) as it was trained: an 8B in bf16 is ~16 GB, which
fits a 24 GB card next to nothing else. Sampling: the family's
(flow.train.chat.SAMPLING), --hf-greedy for none. usage is reported as
OpenAI's fields (prompt_tokens, completion_tokens); `timing` keeps the
first token's latency and tokens/s of the last reply (flow.train.evalsft).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

try:
    from . import chat
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from flow.train import chat


def family_of(model, adapter=None, template=None) -> str:
    if template:
        return template
    if adapter:
        info = os.path.join(os.path.dirname(os.path.abspath(adapter)),
                            "train.json")
        if os.path.exists(info):
            with open(info, encoding="utf-8") as fh:
                return json.load(fh)["family"]
    # A merged model sits in <run>/merged: the run's train.json is a level up.
    for path in (model, adapter, model and os.path.dirname(
            os.path.abspath(model))):
        if path and os.path.exists(os.path.join(path, "train.json")):
            with open(os.path.join(path, "train.json"), encoding="utf-8") as fh:
                return json.load(fh)["family"]
    return chat.family_for(model)


def load(model, adapter=None, load_4bit=False, device="auto", log=print):
    """(model, tokenizer) ready to generate."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from .sft import dtype_kw
    except ImportError:
        from flow.train.sft import dtype_kw
    cuda = torch.cuda.is_available() and device != "cpu"
    kw = dtype_kw(transformers, torch.bfloat16 if cuda else torch.float32)
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    kw["device_map"] = {"": 0} if cuda else {"": "cpu"}
    t0 = time.monotonic()
    m = AutoModelForCausalLM.from_pretrained(model, **kw)
    tok = AutoTokenizer.from_pretrained(adapter or model)
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    log(f"hf: {model}" + (f" + {adapter}" if adapter else "")
        + f" on {'cuda' if cuda else 'cpu'} in {time.monotonic() - t0:.1f}s")
    return m, tok


class HFBackend:
    """director's backend interface: stream(system, messages) yields text;
    stop_reason, usage after."""

    def __init__(self, model, adapter=None, template=None, load_4bit=False,
                 max_tokens=1024, seed=0, greedy=False, device="auto",
                 log=print, loaded=None):
        self.model = model
        self.family = family_of(model, adapter, template)
        self.max_tokens, self.seed, self.greedy = max_tokens, seed, greedy
        self.log = log
        self.stop_reason, self.usage, self.timing = None, {}, {}
        self.lm, self.tok = loaded or load(model, adapter, load_4bit, device,
                                           log)

    def render(self, system, messages) -> str:
        msgs = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": _text(m.get("content"))}
            for m in messages]
        if self.family == "hf":
            return self.tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True)
        return chat.prompt(msgs, self.family)

    def _stops(self):
        ids = []
        for t in chat.stop_tokens(self.family) + [self.tok.eos_token]:
            if not t:
                continue
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                ids.append(i)
        return sorted(set(ids))

    def stream(self, system, messages):
        import torch
        from transformers import TextIteratorStreamer
        text = self.render(system, messages)
        enc = self.tok(text, add_special_tokens=False, return_tensors="pt")
        dev = next(self.lm.parameters()).device
        ids = enc["input_ids"].to(dev)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True,
                                        skip_special_tokens=True)
        s = chat.SAMPLING.get(self.family, {})
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  max_new_tokens=int(self.max_tokens), streamer=streamer,
                  eos_token_id=self._stops(),
                  pad_token_id=self.tok.pad_token_id
                  if self.tok.pad_token_id is not None
                  else self._stops()[0])
        if self.greedy:
            kw["do_sample"] = False
        else:
            kw.update(do_sample=True, temperature=s.get("temperature", 0.7),
                      top_p=s.get("top_p", 0.9), top_k=s.get("top_k", 40))
        out = {}

        def run():
            torch.manual_seed(int(self.seed))
            try:
                with torch.no_grad():
                    out["seq"] = self.lm.generate(**kw)
            except Exception as exc:  # noqa: BLE001 - raised below
                out["error"] = exc
                streamer.end()
        th = threading.Thread(target=run, daemon=True)
        t0 = time.monotonic()
        th.start()
        first = None
        for chunk in streamer:
            if chunk:
                if first is None:
                    first = time.monotonic() - t0
                yield chunk
        th.join()
        if "error" in out:
            raise RuntimeError(f"hf: {out['error']}") from out["error"]
        total = time.monotonic() - t0
        new = out["seq"][0, ids.shape[1]:].tolist()
        stops = set(self._stops())
        hit = bool(new) and new[-1] in stops
        self.stop_reason = "stop" if hit else "length"
        self.usage = {"prompt_tokens": int(ids.shape[1]),
                      "completion_tokens": len(new)}
        gen_t = total - (first or 0.0)
        self.timing = {"first_token_seconds": None if first is None
                       else round(first, 4),
                       "tokens_per_second": round(
                           max(0, len(new) - 1) / gen_t, 2) if gen_t > 0
                       else None, "seconds": round(total, 4)}
        self.log(f"hf: stop {self.stop_reason}; usage {self.usage}; "
                 f"first token {self.timing['first_token_seconds']}s, "
                 f"{self.timing['tokens_per_second']} tokens/s")


def _text(c):
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c or ""
