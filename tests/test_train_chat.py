"""
flow.train.chat: one template in training and serving. The renderer's
spans (what a trainer learns), its Jinja form (the merged tokenizer's
chat_template) and its Ollama TEMPLATE (run through Go's text/template the
way Ollama calls it: .System joined, .Messages with the system ones and
same-role turns merged) give the same bytes. Qwen3's own template (as
published, enable_thinking=False) is checked on a single turn, the case
where the two must agree; flow.train.sft checks the real tokenizer's
template the same way before it trains. Pure Python; the Go check needs
`go` and skips without it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from flow.train import chat

MSGS = [{"role": "system", "content": "You are Hiyori.\nYour body: ..."},
        {"role": "user", "content": "Hi!\n(your body now: head.yaw 0)"},
        {"role": "assistant", "content": 'Hello! [pose {"head.tilt": 8}] Hi.'},
        {"role": "user", "content": "Dance?\n(your body now: head.yaw 0)"},
        {"role": "assistant",
         "content": 'Yes! [move {"name": "dance", "beats": 8}]'}]

# Qwen3's chat template without the tool branches, as Qwen/Qwen3-8B ships
# it (tokenizer_config.json): the think block after the last user turn,
# stripped from the history.
QWEN3_STOCK = (
    "{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\\n' + "
    "messages[0].content + '<|im_end|>\\n' }}{%- endif %}"
    "{%- set ns = namespace(multi_step_tool=true, last_query_index="
    "messages|length - 1) %}"
    "{%- for message in messages[::-1] %}{%- set index = (messages|length - "
    "1) - loop.index0 %}{%- if ns.multi_step_tool and message.role == "
    "'user' %}{%- set ns.multi_step_tool = false %}{%- set "
    "ns.last_query_index = index %}{%- endif %}{%- endfor %}"
    "{%- for message in messages %}{%- set content = message.content %}"
    "{%- if (message.role == 'user') or (message.role == 'system' and not "
    "loop.first) %}{{- '<|im_start|>' + message.role + '\\n' + content + "
    "'<|im_end|>' + '\\n' }}"
    "{%- elif message.role == 'assistant' %}{%- set reasoning_content = '' %}"
    "{%- if '</think>' in content %}{%- set reasoning_content = "
    "content.split('</think>')[0].rstrip('\\n').split('<think>')[-1]"
    ".lstrip('\\n') %}{%- set content = content.split('</think>')[-1]"
    ".lstrip('\\n') %}{%- endif %}"
    "{%- if loop.index0 > ns.last_query_index %}{%- if loop.last or (not "
    "loop.last and reasoning_content) %}{{- '<|im_start|>' + message.role + "
    "'\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' "
    "+ content.lstrip('\\n') }}{%- else %}{{- '<|im_start|>' + "
    "message.role + '\\n' + content }}{%- endif %}{%- else %}{{- "
    "'<|im_start|>' + message.role + '\\n' + content }}{%- endif %}"
    "{{- '<|im_end|>\\n' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}")


def test_qwen3_render_and_spans():
    text, spans = chat.render(MSGS, "qwen3")
    assert text.startswith("<|im_start|>system\nYou are Hiyori.")
    # Every reply, the last and the history, opens with the empty block.
    assert text.count("<|im_start|>assistant\n<think>\n\n</think>\n\n") == 2
    got = [text[a:b] for a, b in spans]
    assert got == [MSGS[2]["content"] + "<|im_end|>",
                   MSGS[4]["content"] + "<|im_end|>"]
    # The reply's context is exactly the server's prompt for that turn.
    a = spans[1][0]
    assert text[:a] == chat.prompt(MSGS[:4], "qwen3")
    assert text.endswith("<|im_end|>\n")


def test_weight_zero_is_not_trained():
    msgs = [dict(m) for m in MSGS]
    msgs[2]["weight"] = 0
    _t, spans = chat.render(msgs, "qwen3")
    assert len(spans) == 1


def test_gemma3_render():
    text, spans = chat.render(MSGS, "gemma3")
    assert text.startswith("<bos><start_of_turn>user\nYou are Hiyori.")
    assert "Your body: ...\n\nHi!" in text
    assert "<start_of_turn>system" not in text
    assert [text[a:b] for a, b in spans][0].endswith("<end_of_turn>")
    assert chat.prompt(MSGS[:4], "gemma3", bos=False).endswith(
        "<start_of_turn>model\n")


def test_family_for():
    assert chat.family_for("Qwen/Qwen3-8B") == "qwen3"
    assert chat.family_for("/models/qwen3-4b-instruct") == "qwen3"
    assert chat.family_for("google/gemma-3-12b-it") == "gemma3"
    assert chat.family_for("Qwen/Qwen2.5-7B-Instruct") == "chatml"
    assert chat.family_for("meta-llama/Llama-3.1-8B-Instruct") == "hf"


@pytest.mark.parametrize("fam", ["qwen3", "chatml", "gemma3"])
def test_jinja_is_the_renderer(fam):
    jinja2 = pytest.importorskip("jinja2")
    t = jinja2.Environment().from_string(chat.jinja(fam))
    for k in (2, 3, 4, 5):
        for gen in (False, True):
            if gen and MSGS[k - 1]["role"] != "user":
                continue
            want = chat.render(MSGS[:k], fam, add_generation_prompt=gen)[0]
            assert t.render(messages=MSGS[:k], add_generation_prompt=gen,
                            bos_token="<bos>") == want


def test_qwen3_single_turn_matches_the_stock_template():
    jinja2 = pytest.importorskip("jinja2")
    t = jinja2.Environment().from_string(QWEN3_STOCK)
    one = MSGS[:2]
    assert t.render(messages=one, add_generation_prompt=True,
                    enable_thinking=False) == chat.prompt(one, "qwen3")
    full = MSGS[:3]
    assert t.render(messages=full, add_generation_prompt=False,
                    enable_thinking=False) == chat.render(full, "qwen3")[0]
    # In a longer history they part: the stock one strips the block.
    assert t.render(messages=MSGS, add_generation_prompt=False) != \
        chat.render(MSGS, "qwen3")[0]


def test_hf_render_spans_with_a_tokenizer_template():
    jinja2 = pytest.importorskip("jinja2")

    class Tok:
        def apply_chat_template(self, msgs, tokenize=False,
                                add_generation_prompt=False, **kw):
            return jinja2.Environment().from_string(chat.jinja(
                "chatml")).render(messages=msgs,
                                  add_generation_prompt=add_generation_prompt)
    text, spans = chat.hf_render(Tok(), MSGS)
    assert (text, spans) == chat.render(MSGS, "chatml")


GO_OLLAMA = r'''
package main

import (
	"encoding/json"
	"os"
	"strings"
	"text/template"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// What Ollama's template.Execute passes a messages template: collate()
// joins the system messages into System and merges same-role turns.
func main() {
	var in struct {
		Template string    `json:"template"`
		Messages []Message `json:"messages"`
	}
	if err := json.NewDecoder(os.Stdin).Decode(&in); err != nil {
		panic(err)
	}
	var system []string
	var col []*Message
	for i := range in.Messages {
		m := in.Messages[i]
		if m.Role == "system" {
			system = append(system, m.Content)
		}
		if len(col) > 0 && col[len(col)-1].Role == m.Role {
			col[len(col)-1].Content += "\n\n" + m.Content
		} else {
			col = append(col, &m)
		}
	}
	t := template.Must(template.New("").Parse(in.Template))
	if err := t.Execute(os.Stdout, map[string]any{
		"System": strings.Join(system, "\n\n"), "Messages": col,
		"Tools": nil, "Response": ""}); err != nil {
		panic(err)
	}
}
'''


@pytest.mark.parametrize("fam", ["qwen3", "chatml", "gemma3"])
def test_ollama_template_is_the_renderer(fam, tmp_path):
    if shutil.which("go") is None:
        pytest.skip("no go toolchain (Ollama's template engine)")
    src = tmp_path / "main.go"
    src.write_text(GO_OLLAMA)
    env = dict(os.environ, GOCACHE=str(tmp_path / "cache"),
               GOPATH=str(tmp_path / "gopath"), GO111MODULE="off")
    exe = tmp_path / "render"
    subprocess.run(["go", "build", "-o", str(exe), str(src)], check=True,
                   env=env, capture_output=True)
    for k in (2, 4):
        out = subprocess.run([str(exe)], input=json.dumps(
            {"template": chat.OLLAMA[fam], "messages": MSGS[:k]}),
            capture_output=True, text=True, check=True).stdout
        assert out == chat.prompt(MSGS[:k], fam, bos=False)


def test_modelfile():
    mf = chat.modelfile("./tsk.gguf", "qwen3", num_ctx=8192)
    assert mf.startswith("FROM ./tsk.gguf\n")
    assert 'PARAMETER stop "<|im_end|>"' in mf
    assert "PARAMETER num_ctx 8192" in mf and "temperature 0.7" in mf
    assert chat.OLLAMA["qwen3"] in mf
    with pytest.raises(SystemExit):
        chat.modelfile("x.gguf", "hf")
