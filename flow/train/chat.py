#!/usr/bin/env python3
"""
One chat template per model family, the same bytes in training and in
serving: the trainer renders and masks with it (flow.train.sft), the hf
backend renders prompts with it (flow.train.hf), the Ollama Modelfile
carries it as TEMPLATE (modelfile()), and the merged model's tokenizer
gets its Jinja form (jinja()) for servers that read that.

  qwen3   ChatML with thinking off: every assistant turn, history or the
          one being generated, opens with an empty think block

              <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{reply}<|im_end|>\\n

          Qwen3's own template writes that block only after the last user
          turn and strips it from the history; with it on every turn a
          whole session trains as one sequence and every reply's context is
          exactly the prompt the server builds for it (the history as the
          director keeps it, then the generation prompt with the block). On
          a single turn the two templates are the same bytes, which the
          trainer checks against the tokenizer's own (enable_thinking=
          False) before it starts.
  chatml  ChatML without the block (Qwen2.5 and other ChatML models).
  gemma3  <start_of_turn>user / model ... <end_of_turn>, the system prompt
          before the first user message and a blank line (as Gemma 3's
          template does: it has no system role); <bos> first in training
          and the hf backend, left out of the Ollama template (llama.cpp
          adds it).
  hf      anything else: the tokenizer's own apply_chat_template, the
          trained spans found by rendering each prefix (no Modelfile: give
          Ollama a TEMPLATE yourself).

Contents are used as they are (no trimming); flow.train.sft strips every
message once when it loads the data, so Gemma's trim and ours agree.

render(messages, family) -> (text, spans): spans are the (start, end)
character ranges a trainer learns (each reply plus its end-of-turn token,
not the newline after it), in order.
"""

from __future__ import annotations

import re

FAMILIES = ("qwen3", "chatml", "gemma3", "hf")

THINK_OFF = "<think>\n\n</think>\n\n"
END = {"qwen3": "<|im_end|>", "chatml": "<|im_end|>",
       "gemma3": "<end_of_turn>"}
# Sampling for serving (Ollama PARAMETER, the hf backend): Qwen3's advice
# for thinking off; Gemma 3's model card defaults.
SAMPLING = {"qwen3": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
            "chatml": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
            "gemma3": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
            "hf": {"temperature": 0.7, "top_p": 0.9, "top_k": 40}}


def family_for(model: str) -> str:
    """The family a base model name (or local path) renders with."""
    m = (model or "").lower()
    if re.search(r"qwen-?3", m):
        return "qwen3"
    if re.search(r"gemma-?3", m):
        return "gemma3"
    if re.search(r"qwen-?2", m):
        return "chatml"
    return "hf"


def _split(messages):
    system = None
    rest = list(messages)
    if rest and rest[0]["role"] == "system":
        system = rest.pop(0)["content"]
    for m in rest:
        if m["role"] not in ("user", "assistant"):
            raise ValueError(f"role {m['role']!r}: only one system message, "
                             "first, then user / assistant")
    return system, rest


def render(messages, family="qwen3", add_generation_prompt=False, bos=True,
           trained=None):
    """(text, spans) of messages in this family's template (module
    docstring). trained(i, message) -> bool picks the assistant turns that
    are learned (default: all; an assistant message with "weight": 0 is
    not)."""
    if family == "hf":
        raise ValueError("family hf renders with the tokenizer: hf_render()")
    if trained is None:
        def trained(_i, m):
            return m.get("weight", 1) != 0
    system, rest = _split(messages)
    out, spans = [], []
    pos = 0

    def put(s):
        nonlocal pos
        out.append(s)
        pos += len(s)
    end = END[family]
    if family in ("qwen3", "chatml"):
        think = THINK_OFF if family == "qwen3" else ""
        if system is not None:
            put(f"<|im_start|>system\n{system}{end}\n")
        for i, m in enumerate(rest):
            if m["role"] == "user":
                put(f"<|im_start|>user\n{m['content']}{end}\n")
                continue
            put(f"<|im_start|>assistant\n{think}")
            a = pos
            put(m["content"] + end)
            if trained(i, m):
                spans.append((a, pos))
            put("\n")
        if add_generation_prompt:
            put(f"<|im_start|>assistant\n{think}")
        return "".join(out), spans
    # gemma3
    if bos:
        put("<bos>")
    first = True
    for i, m in enumerate(rest):
        if m["role"] == "user":
            pre = f"{system}\n\n" if first and system is not None else ""
            put(f"<start_of_turn>user\n{pre}{m['content']}{end}\n")
            first = False
            continue
        put("<start_of_turn>model\n")
        a = pos
        put(m["content"] + end)
        if trained(i, m):
            spans.append((a, pos))
        put("\n")
    if add_generation_prompt:
        put("<start_of_turn>model\n")
    return "".join(out), spans


def prompt(messages, family="qwen3", bos=True) -> str:
    """The generation prompt for a conversation ending in a user turn."""
    return render(messages, family, add_generation_prompt=True, bos=bos)[0]


def hf_render(tokenizer, messages, add_generation_prompt=False, **kw):
    """(text, spans) with the tokenizer's own template: each trained
    assistant turn's span is what rendering up to it adds after rendering
    the turns before it with the generation prompt (a template whose
    prefixes are not stable raises)."""
    def r(msgs, gen):
        return tokenizer.apply_chat_template(
            [{"role": m["role"], "content": m["content"]} for m in msgs],
            tokenize=False, add_generation_prompt=gen, **kw)
    text = r(messages, add_generation_prompt)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant" or m.get("weight", 1) == 0:
            continue
        head = r(messages[:i], True)
        upto = r(messages[:i + 1], False)
        if not (upto.startswith(head) and text.startswith(upto)):
            raise ValueError("the tokenizer's template is not prefix-stable; "
                             "pick a family (--template)")
        a = len(head)
        b = upto.find(m["content"], a)
        b = len(upto) if b < 0 else b + len(m["content"])
        # Through the end-of-turn token that follows the reply.
        tail = upto[b:]
        mt = re.match(r"\s*(<[^>\s]{1,40}>)", tail)
        spans.append((a, b + (mt.end() if mt else 0)))
    return text, spans


def stop_tokens(family) -> list[str]:
    return {"qwen3": ["<|im_end|>", "<|im_start|>"],
            "chatml": ["<|im_end|>", "<|im_start|>"],
            "gemma3": ["<end_of_turn>", "<start_of_turn>"]}.get(family, [])


# ------------------------------------------------------- Ollama, Jinja

# Ollama's TEMPLATE (Go text/template). Ollama passes .System (the system
# messages joined) and .Messages (system ones included, consecutive turns
# of one role merged); these skip the system messages in the range.
OLLAMA = {
    "qwen3": (
        "{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n"
        "{{ end }}"
        "{{- range .Messages }}"
        "{{- if eq .Role \"user\" }}<|im_start|>user\n{{ .Content }}"
        "<|im_end|>\n"
        "{{ else if eq .Role \"assistant\" }}<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n{{ .Content }}<|im_end|>\n"
        "{{ end }}"
        "{{- end }}<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    "chatml": (
        "{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n"
        "{{ end }}"
        "{{- range .Messages }}"
        "{{- if eq .Role \"user\" }}<|im_start|>user\n{{ .Content }}"
        "<|im_end|>\n"
        "{{ else if eq .Role \"assistant\" }}<|im_start|>assistant\n"
        "{{ .Content }}<|im_end|>\n"
        "{{ end }}"
        "{{- end }}<|im_start|>assistant\n"),
    "gemma3": (
        "{{- $first := true }}"
        "{{- range .Messages }}"
        "{{- if eq .Role \"user\" }}<start_of_turn>user\n"
        "{{ if and $first $.System }}{{ $.System }}\n\n{{ end }}"
        "{{ $first = false }}{{ .Content }}<end_of_turn>\n"
        "{{ else if eq .Role \"assistant\" }}<start_of_turn>model\n"
        "{{ .Content }}<end_of_turn>\n"
        "{{ end }}"
        "{{- end }}<start_of_turn>model\n"),
}

JINJA = {
    "qwen3": (
        "{% for m in messages %}"
        "{% if m.role == 'system' %}<|im_start|>system\n{{ m.content }}"
        "<|im_end|>\n"
        "{% elif m.role == 'user' %}<|im_start|>user\n{{ m.content }}"
        "<|im_end|>\n"
        "{% elif m.role == 'assistant' %}<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n{{ m.content }}<|im_end|>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n{% endif %}"),
    "chatml": (
        "{% for m in messages %}"
        "{% if m.role == 'system' %}<|im_start|>system\n{{ m.content }}"
        "<|im_end|>\n"
        "{% elif m.role == 'user' %}<|im_start|>user\n{{ m.content }}"
        "<|im_end|>\n"
        "{% elif m.role == 'assistant' %}<|im_start|>assistant\n"
        "{{ m.content }}<|im_end|>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "{% endif %}"),
    "gemma3": (
        "{{ bos_token }}"
        "{% set ns = namespace(sys=none, first=true) %}"
        "{% for m in messages %}"
        "{% if m.role == 'system' %}{% set ns.sys = m.content %}"
        "{% elif m.role == 'user' %}<start_of_turn>user\n"
        "{% if ns.first and ns.sys is not none %}{{ ns.sys }}\n\n"
        "{% endif %}{% set ns.first = false %}{{ m.content }}"
        "<end_of_turn>\n"
        "{% elif m.role == 'assistant' %}<start_of_turn>model\n"
        "{{ m.content }}<end_of_turn>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n"
        "{% endif %}"),
}


def jinja(family) -> str | None:
    return JINJA.get(family)


def modelfile(gguf, family, num_ctx=8192, sampling=None, system=None) -> str:
    """An Ollama Modelfile serving gguf with this family's template: the
    template above, its stop tokens, a context long enough for the body
    prompt (~2-2.5k tokens) plus the history (Ollama's default context cuts
    it silently), and the sampling the trainer's eval used."""
    if family not in OLLAMA:
        raise SystemExit(f"no Ollama template for family {family!r}: write "
                         "the Modelfile's TEMPLATE yourself (it must render "
                         "what flow.train.chat renders)")
    s = dict(SAMPLING.get(family, {}), **(sampling or {}))
    tpl = OLLAMA[family].replace('"""', '\\"\\"\\"')
    lines = [f"FROM {gguf}", f'TEMPLATE """{tpl}"""']
    for t in stop_tokens(family):
        lines.append(f'PARAMETER stop "{t}"')
    lines.append(f"PARAMETER num_ctx {int(num_ctx)}")
    for k in ("temperature", "top_p", "top_k", "repeat_penalty"):
        if k in s:
            lines.append(f"PARAMETER {k} {s[k]}")
    if system:
        lines.append(f'SYSTEM """{system}"""')
    return "\n".join(lines) + "\n"
