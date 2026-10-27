#!/usr/bin/env python3
"""
LoRA distillation of the body-mode director (docs/embodiment.md Stage 2
(a)): a local model learns to write the teacher's marked replies from the
same system prompt and user turns (flow.dataexport sft), on one 24 GB GPU.

  data      flow.dataexport's sft.jsonl (with --meta: the session ids) or a
            flow.dataset --out folder (exported here with the default
            filters), the contents as the director sends them. From a
            folder, --correct-misses [default; --no-correct-misses] writes
            each pose target the solver missed as the value it reached
            (flow.dataexport: the avatar's reach is what is taught), and
            --avatar-weight NAME=N repeats an avatar's examples N times.
            An sft.jsonl is taken as it was exported.
  split     by session, never by turn: a session is held out when the
            first 8 hex of sha256("<seed>:<session>") / 2^32 is below
            --eval-frac, so a run on more data keeps its old sessions on
            the side they were (split.json lists both sides; flow.train.
            evalsft reads it).
  template  flow.train.chat for the base's family (Qwen3: thinking off,
            the empty think block on every assistant turn), the same one
            the hf backend and the Ollama Modelfile serve. Before training
            a single turn is rendered by the tokenizer's own template
            (enable_thinking=False) and by ours: they must be the same
            bytes, or the run stops with the difference (--allow-template-
            mismatch to go on anyway).
  tokens    each template segment tokenized on its own (the prompt as the
            server sends it, each reply as the model writes it after the
            prompt); labels only on the replies and their end-of-turn token
            (a per-turn export's "weight": 0 history is not learned). An
            example longer than --max-len is cut after its last reply that
            fits; one whose first reply does not fit is left out (both
            counted). --max-len auto: the 99th percentile of the examples,
            rounded up to 256 (`stats` prints the distribution first).
  model     QLoRA: the base in 4 bit (NF4, double quant, bf16 compute),
            LoRA rank 32 / alpha 32 on every attention and MLP projection
            of the language model, gradient checkpointing, bf16, batch 1
            with gradient accumulation, paged 8-bit AdamW. Unsloth when it
            imports (--backend auto), else transformers + PEFT +
            bitsandbytes. On the PEFT path the logits are computed only
            where a label is (logits_to_keep): 8k positions x Qwen3's 152k
            vocabulary in fp32 is ~5 GB, more than the 4-bit weights;
            Unsloth chunks its loss itself.
  run       eval loss on the held-out sessions every --eval-steps,
            checkpoints every --save-steps (the last --keep), resumed when
            the folder has one and the tokenized data is the same (data.
            json holds its hash; a changed set stops the run unless
            --restart). The first steps log tokens/s, peak memory and the
            epoch time they imply. The adapter goes to <out>/adapter with
            the tokenizer (its chat_template ours, flow.train.chat.jinja).

  merge     the base reloaded unquantized (fp16 by default), the adapter
            merged into it, saved as safetensors with the tokenizer: for
            llama.cpp's converter or a transformers server. Never into the
            4-bit model.
  gguf      llama.cpp: convert_hf_to_gguf.py to f16, llama-quantize to
            Q4_K_M (or --quant); prints the commands when --llama-cpp is
            not given.
  modelfile an Ollama Modelfile for the GGUF with the template, stops,
            num_ctx and sampling (flow.train.chat.modelfile); --create runs
            `ollama create`.

Usage (scripts/train_sft.py is the same entry point):
    python -m flow.train.sft stats  --data data/datagen/r1
    python -m flow.train.sft train  --data data/datagen/r1 --out runs/sft/q8
    python -m flow.train.sft merge  --run runs/sft/q8
    python -m flow.train.sft gguf   --run runs/sft/q8 --llama-cpp ~/llama.cpp
    python -m flow.train.sft modelfile --run runs/sft/q8 --name tsk-q8 --create
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

try:
    from . import chat
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from flow.train import chat

DEFAULT_BASE = "Qwen/Qwen3-8B"
# The language model's projections; a regex so a multimodal base (Gemma 3)
# keeps its vision tower out.
TARGETS = (r"^(?!.*(vision|multi_modal|mm_)).*\.(q_proj|k_proj|v_proj|o_proj|"
           r"gate_proj|up_proj|down_proj)$")
EVAL_FRAC = 0.05


def log(s):
    print(s, flush=True)


# ---------------------------------------------------------------- data

def load_examples(data, per="session", max_miss=None, keep_lite=False,
                  correct_misses=True, avatar_weight=None):
    """The SFT examples ({"messages", "meta"}) from an sft.jsonl or a
    flow.dataset folder (exported here with flow.dataexport's filters,
    missed pose targets corrected to the value reached, and the avatars in
    `avatar_weight` ["NAME=N", ...] repeated; an sft.jsonl is as exported)."""
    if os.path.isdir(data):
        try:
            from .. import dataexport as dx
            from .. import dataset as ds
        except ImportError:
            from flow import dataexport as dx
            from flow import dataset as ds
        recs = ds.iter_records(data)
        if not recs:
            raise SystemExit(f"no records under {data}/shards")
        fix, counts = dx.correct_misses(recs) if correct_misses else ({}, None)
        notes = dx.annotate(recs, max_miss if max_miss is not None
                            else dx.MAX_MISS, keep_lite=keep_lite, fix=fix)
        w = dx.parse_weights(avatar_weight, {r["avatar"]["name"]
                                             for r in recs})
        if counts:
            log(dx.correction_line(counts))
        if w:
            log("avatar weights: " + ", ".join(f"{a} x{n}"
                                               for a, n in w.items()))
        return dx.sft_examples(data, recs, notes, per=per, meta=True,
                               fix=fix, weights=w)
    out = []
    with open(data, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def session_of(ex):
    m = ex.get("meta") or {}
    if m.get("session"):
        return m["session"]
    if m.get("id"):
        return m["id"].rsplit("/", 1)[0]
    return None


def clean(ex):
    """The example with every content a string, not stripped: the server
    renders the messages as the director sends them (the body prompt ends
    in a newline), so training renders them so too."""
    msgs = []
    for m in ex["messages"]:
        m = dict(m)
        m["content"] = m.get("content") or ""
        msgs.append(m)
    return dict(ex, messages=msgs)


def held_out(session, frac=EVAL_FRAC, seed=0) -> bool:
    h = hashlib.sha256(f"{seed}:{session}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 2 ** 32 < frac


def split(examples, frac=EVAL_FRAC, seed=0, by_row=False):
    """(train, eval, info) by session (module docstring)."""
    train, ev = [], []
    tr_s, ev_s = set(), set()
    for i, ex in enumerate(examples):
        s = session_of(ex)
        if s is None:
            if not by_row:
                raise SystemExit(
                    "the examples carry no session ids: export with "
                    "flow.dataexport sft --meta (scripts/datagen.sh does), "
                    "pass the dataset folder as --data, or --split-by-row "
                    "for a session-mode export (one row per session)")
            s = f"row{i}"
        (ev if held_out(s, frac, seed) else train).append(ex)
        (ev_s if held_out(s, frac, seed) else tr_s).add(s)
    if frac > 0 and not ev and len(train) > 1:
        # A tiny set: one session out, the last by its hash order.
        s = max(tr_s, key=lambda s: hashlib.sha256(
            f"{seed}:{s}".encode()).hexdigest())
        ev = [e for e in train if session_of(e) == s]
        train = [e for e in train if session_of(e) != s]
        tr_s.discard(s)
        ev_s.add(s)
    assert not (tr_s & ev_s)
    info = {"rule": f"sha256('{seed}:<session>')[:8] / 2^32 < {frac}",
            "eval_frac": frac, "seed": seed,
            "train_sessions": sorted(tr_s), "eval_sessions": sorted(ev_s)}
    return train, ev, info


# --------------------------------------------------------------- tokens

def render(tokenizer, ex, family):
    if family == "hf":
        return chat.hf_render(tokenizer, ex["messages"])
    return chat.render(ex["messages"], family)


def segments(text, spans):
    """[(text, trained)] alternating around the spans."""
    out, pos = [], 0
    for a, b in spans:
        if a > pos:
            out.append((text[pos:a], False))
        out.append((text[a:b], True))
        pos = b
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def _ids(tokenizer, s):
    return list(tokenizer(s, add_special_tokens=False)["input_ids"])


def encode(tokenizer, ex, family, max_len=None):
    """{input_ids, labels, turns (replies learned), cut (replies dropped by
    max_len)} or None when not even the first reply fits."""
    text, spans = render(tokenizer, ex, family)
    ids, labels, ends = [], [], []
    for seg, tr in segments(text, spans):
        t = _ids(tokenizer, seg)
        ids += t
        labels += t if tr else [-100] * len(t)
        if tr:
            ends.append(len(ids))
    n = len(ends)
    if max_len and len(ids) > max_len:
        fit = [e for e in ends if e <= max_len]
        if not fit:
            return None
        cut = fit[-1]
        ids, labels = ids[:cut], labels[:cut]
        ends = fit
    return {"input_ids": ids, "labels": labels, "turns": len(ends),
            "cut": n - len(ends)}


def _pct(xs, q):
    return int(np.percentile(xs, q)) if len(xs) else 0


def auto_max_len(lengths, q=99, step=256, lo=1024, hi=16384) -> int:
    if not lengths:
        return lo
    x = float(np.percentile(lengths, q))
    return int(min(hi, max(lo, math.ceil(x / step) * step)))


def length_report(tokenizer, examples, family) -> dict:
    """Token counts: per example (whole), the system prompt, per user turn
    and per reply; the max length auto would pick."""
    whole, system, user, reply, turns = [], [], [], [], []
    for ex in examples:
        text, spans = render(tokenizer, ex, family)
        whole.append(sum(len(_ids(tokenizer, s)) for s, _ in
                         segments(text, spans)))
        msgs = ex["messages"]
        if msgs and msgs[0]["role"] == "system":
            system.append(len(_ids(tokenizer, msgs[0]["content"])))
        for m in msgs:
            if m["role"] == "user":
                user.append(len(_ids(tokenizer, m["content"])))
            elif m["role"] == "assistant":
                reply.append(len(_ids(tokenizer, m["content"])))
        turns.append(sum(m["role"] == "assistant" for m in msgs))

    def d(xs):
        return {"n": len(xs), "mean": round(float(np.mean(xs)), 1)
                if xs else 0, "p50": _pct(xs, 50), "p90": _pct(xs, 90),
                "p99": _pct(xs, 99), "max": max(xs) if xs else 0}
    rep = {"examples": d(whole), "system": d(system), "user": d(user),
           "reply": d(reply), "turns": d(turns),
           "tokens": int(sum(whole)), "auto_max_len": auto_max_len(whole)}
    rep["fit_at"] = {L: round(float(np.mean([w <= L for w in whole])), 4)
                     for L in (2048, 4096, 6144, 8192, 12288)} if whole \
        else {}
    return rep


def report_text(rep, family) -> str:
    def row(k):
        d = rep[k]
        return (f"  {k:9s} n {d['n']:6d}  mean {d['mean']:8.1f}  p50 "
                f"{d['p50']:6d}  p90 {d['p90']:6d}  p99 {d['p99']:6d}  max "
                f"{d['max']:6d}")
    out = [f"tokens ({family} template), {rep['tokens']} in all:"]
    out += [row(k) for k in ("examples", "system", "user", "reply", "turns")]
    out.append("  examples that fit whole: " + ", ".join(
        f"{L} {v:.1%}" for L, v in rep["fit_at"].items()))
    out.append(f"  --max-len auto: {rep['auto_max_len']}")
    return "\n".join(out)


def check_template(tokenizer, family, ex) -> list[str]:
    """Problems found rendering ex's first turn with the tokenizer's own
    template and with ours (prompt and full turn); [] when they agree."""
    if family == "hf" or not getattr(tokenizer, "chat_template", None):
        return []
    # Gemma 3's own template trims each content; ours, like the server,
    # does not (so the check trims for it).
    trim = family == "gemma3"
    msgs = [{"role": m["role"], "content": m["content"].strip() if trim
             else m["content"]} for m in ex["messages"]]
    k = next(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    probs = []
    for gen, part in ((True, msgs[:k]), (False, msgs[:k + 1])):
        theirs = tokenizer.apply_chat_template(
            part, tokenize=False, add_generation_prompt=gen,
            enable_thinking=False)
        ours = chat.render(part, family, add_generation_prompt=gen)[0]
        if family == "gemma3" and not theirs.startswith("<bos>"):
            ours = ours[len("<bos>"):]
        if theirs != ours:
            i = next((j for j, (a, b) in enumerate(zip(theirs, ours))
                      if a != b), min(len(theirs), len(ours)))
            probs.append(f"{'prompt' if gen else 'turn'} differs at char "
                         f"{i}: tokenizer {theirs[max(0, i - 40):i + 40]!r} "
                         f"vs ours {ours[max(0, i - 40):i + 40]!r}")
    return probs


def data_hash(rows) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(np.asarray(r["input_ids"], np.int64).tobytes())
        h.update(np.asarray(r["labels"], np.int64).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- model

def _filter_kwargs(fn, kw, what):
    """kw minus what fn does not take (transformers renames and drops
    arguments between versions), logged."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    out = {k: v for k, v in kw.items() if k in params}
    gone = sorted(set(kw) - set(out))
    if gone:
        log(f"sft: {what} takes no {', '.join(gone)}; left out")
    return out


def dtype_kw(transformers, dtype):
    """{"dtype": x} (transformers >= 4.56) or {"torch_dtype": x}."""
    v = tuple(int(p) for p in transformers.__version__.split(".")[:2]
              if p.isdigit())
    return {"dtype": dtype} if v >= (4, 56) else {"torch_dtype": dtype}


def load_tokenizer(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def load_for_training(o):
    """(model, tokenizer, backend name)."""
    want = o.backend
    if want in ("auto", "unsloth"):
        try:
            # Before transformers / peft / trl: its patches go in at import.
            import unsloth  # noqa: F401
            from unsloth import FastLanguageModel
        except Exception as exc:  # noqa: BLE001 - optional fast path
            if want == "unsloth":
                raise SystemExit(f"--backend unsloth: {exc}") from exc
            log(f"sft: no Unsloth ({exc.__class__.__name__}: "
                f"{str(exc)[:120]}); transformers + PEFT")
        else:
            model, tok = FastLanguageModel.from_pretrained(
                model_name=o.base, max_seq_length=o.max_len, dtype=None,
                load_in_4bit=o.load_4bit)
            model = FastLanguageModel.get_peft_model(
                model, r=o.rank, lora_alpha=o.alpha, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                bias="none", use_gradient_checkpointing="unsloth",
                random_state=o.seed)
            return model, tok, "unsloth"
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM
    tok = load_tokenizer(o.base)
    cuda = torch.cuda.is_available()
    kw = {}
    if o.load_4bit:
        if not cuda:
            raise SystemExit("4-bit needs a CUDA GPU (bitsandbytes); "
                             "--no-4bit for a CPU check")
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    dt = torch.bfloat16 if cuda else torch.float32
    kw.update(dtype_kw(transformers, dt))
    if cuda:
        kw["device_map"] = {"": 0}
    kw["attn_implementation"] = o.attn
    model = AutoModelForCausalLM.from_pretrained(o.base, **kw)
    model.config.use_cache = False
    if o.load_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=o.rank, lora_alpha=o.alpha, lora_dropout=o.dropout,
                     target_modules=TARGETS, bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    return model, tok, "peft"


class Rows:
    """A torch-style dataset over encoded rows (no torch import here)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"input_ids": r["input_ids"], "labels": r["labels"]}


def collate(batch):
    import torch
    if len(batch) != 1:
        raise ValueError("batch size 1 (variable lengths, gradient "
                         "accumulation does the batching)")
    ids = torch.tensor([batch[0]["input_ids"]], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([batch[0]["labels"]], dtype=torch.long)}


def masked_loss(model, inputs, keep_logits=True):
    """Mean cross-entropy over the labelled tokens of one sequence; with
    keep_logits the head runs only where a label is (logits_to_keep)."""
    import torch
    import torch.nn.functional as F
    labels = inputs["labels"]
    if not keep_logits:
        return model(input_ids=inputs["input_ids"],
                     attention_mask=inputs["attention_mask"],
                     labels=labels).loss
    tgt = labels[0, 1:]
    pos = (tgt != -100).nonzero().squeeze(-1)
    if pos.numel() == 0:
        return torch.zeros((), device=labels.device, requires_grad=True)
    out = model(input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                logits_to_keep=pos)
    logits = out.logits[0]
    if logits.shape[0] != pos.numel():
        # A model without logits_to_keep returned every position.
        logits = logits[pos]
    return F.cross_entropy(logits.float(), tgt[pos])


def make_trainer(model, tok, args, train_ds, eval_ds, callbacks, backend):
    from transformers import Trainer

    class SFTTrainer(Trainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            # The loss is a per-sequence mean: the Trainer scales it by the
            # accumulation steps itself.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            loss = masked_loss(model, inputs, keep_logits=backend != "unsloth")
            return (loss, {}) if return_outputs else loss

    kw = {"model": model, "args": args, "train_dataset": train_ds,
          "eval_dataset": eval_ds, "data_collator": collate,
          "callbacks": callbacks}
    # processing_class since transformers 4.46, tokenizer before.
    kw["processing_class" if "processing_class" in inspect.signature(
        Trainer.__init__).parameters else "tokenizer"] = tok
    return SFTTrainer(**_filter_kwargs(Trainer.__init__, kw, "Trainer"))


def speed_callback(rows, accum, probe=20):
    """Tokens/s, peak memory and the epoch time they imply, from the first
    `probe` optimizer steps (logged once, then every probe steps)."""
    from transformers import TrainerCallback
    mean_len = float(np.mean([len(r["input_ids"]) for r in rows])) \
        if rows else 0.0
    steps_epoch = math.ceil(len(rows) / accum)

    class Speed(TrainerCallback):
        t0 = None
        s0 = 0

        def on_step_begin(self, args, state, control, **kw):
            if self.t0 is None:
                self.t0, self.s0 = time.monotonic(), state.global_step

        def on_step_end(self, args, state, control, **kw):
            n = state.global_step - self.s0
            if n <= 0 or n % probe:
                return
            el = time.monotonic() - self.t0
            tps = n * accum * mean_len / el
            mem = ""
            try:
                import torch
                if torch.cuda.is_available():
                    mem = (f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f}"
                           f" GiB allocated / {torch.cuda.max_memory_reserved() / 2**30:.1f}"
                           " reserved")
            except Exception:  # noqa: BLE001 - a report
                pass
            log(f"sft: speed {tps:.0f} tokens/s ({el / n:.1f} s a step of "
                f"{accum} x ~{mean_len:.0f} tokens){mem}; ~"
                f"{el / n * steps_epoch / 3600:.2f} h an epoch")
    return Speed()


def training_args(o, n_train):
    import torch
    from transformers import TrainingArguments
    cuda = torch.cuda.is_available()
    steps_epoch = math.ceil(n_train / o.accum)
    total = o.max_steps if o.max_steps > 0 else math.ceil(
        steps_epoch * o.epochs)
    kw = dict(output_dir=os.path.join(o.out, "checkpoints"),
              per_device_train_batch_size=1, per_device_eval_batch_size=1,
              gradient_accumulation_steps=o.accum,
              learning_rate=o.lr, lr_scheduler_type="cosine",
              warmup_steps=max(1, int(round(o.warmup * total))),
              num_train_epochs=o.epochs, max_steps=o.max_steps,
              weight_decay=0.0, max_grad_norm=1.0,
              bf16=cuda, fp16=False, use_cpu=not cuda,
              gradient_checkpointing=cuda and o.backend_used != "unsloth",
              gradient_checkpointing_kwargs={"use_reentrant": False},
              optim="paged_adamw_8bit" if cuda and o.load_4bit
              else "adamw_torch",
              logging_steps=o.log_steps, logging_first_step=True,
              eval_strategy="steps" if o.eval_n else "no",
              eval_steps=o.eval_steps, save_strategy="steps",
              save_steps=o.save_steps, save_total_limit=o.keep,
              prediction_loss_only=True, report_to="none",
              remove_unused_columns=False, dataloader_num_workers=0,
              seed=o.seed, data_seed=o.seed)
    return TrainingArguments(**_filter_kwargs(TrainingArguments.__init__, kw,
                                              "TrainingArguments"))


def last_checkpoint(out):
    d = os.path.join(out, "checkpoints")
    if not os.path.isdir(d):
        return None
    cks = [c for c in os.listdir(d) if c.startswith("checkpoint-")
           and c[11:].isdigit()]
    if not cks:
        return None
    return os.path.join(d, max(cks, key=lambda c: int(c[11:])))


def versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for m in ("torch", "transformers", "peft", "trl", "bitsandbytes",
              "unsloth", "accelerate"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001 - a report
            pass
    return out


def prepare(o, tok):
    """Examples -> split -> rows; writes split.json and lengths.json."""
    exs = [clean(e) for e in load_examples(
        o.data, o.per, correct_misses=o.correct_misses,
        avatar_weight=o.avatar_weight)]
    if not exs:
        raise SystemExit(f"no examples in {o.data}")
    train, ev, info = split(exs, o.eval_frac, o.split_seed, o.split_by_row)
    rep = length_report(tok, exs, o.family)
    log(report_text(rep, o.family))
    if o.max_len_arg == "auto":
        o.max_len = rep["auto_max_len"]
    log(f"sft: {len(exs)} examples, {len(train)} train / {len(ev)} eval "
        f"({len(info['train_sessions'])} / {len(info['eval_sessions'])} "
        f"sessions), max length {o.max_len}")
    os.makedirs(o.out, exist_ok=True)
    with open(os.path.join(o.out, "split.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    with open(os.path.join(o.out, "lengths.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    return exs, train, ev, rep


def encode_all(tok, exs, family, max_len):
    rows, gone, cut = [], 0, 0
    for e in exs:
        r = encode(tok, e, family, max_len)
        if r is None:
            gone += 1
            continue
        cut += r["cut"]
        rows.append(r)
    return rows, gone, cut


def train(o) -> dict:
    o.family = o.template or chat.family_for(o.base)
    o.max_len_arg = o.max_len
    o.max_len = 8192 if o.max_len == "auto" else int(o.max_len)
    # The examples' lengths decide max_len before the model loads (Unsloth
    # takes it at load); the fast tokenizer is the base's.
    tok0 = load_tokenizer(o.base)
    exs, train_ex, ev_ex, rep = prepare(o, tok0)
    probs = check_template(tok0, o.family, exs[0])
    if probs:
        for p in probs:
            log(f"sft: TEMPLATE MISMATCH {p}")
        if not o.allow_template_mismatch:
            raise SystemExit("sft: the tokenizer's chat template and "
                             f"flow.train.chat {o.family} disagree on one "
                             "turn (above); fix the family (--template) or "
                             "pass --allow-template-mismatch")
    else:
        log(f"sft: template {o.family}: one turn renders the same as the "
            "tokenizer's own (enable_thinking=False)")
    rows_tr, gone, cut = encode_all(tok0, train_ex, o.family, o.max_len)
    rows_ev, gone_e, cut_e = encode_all(tok0, ev_ex, o.family, o.max_len)
    log(f"sft: {len(rows_tr)} train rows ({gone} too long for one reply, "
        f"{cut} replies cut at {o.max_len}), {len(rows_ev)} eval rows "
        f"({gone_e}, {cut_e})")
    if not rows_tr:
        raise SystemExit("sft: nothing to train on")
    dh = data_hash(rows_tr + rows_ev)
    meta_path = os.path.join(o.out, "data.json")
    ck = last_checkpoint(o.out)
    if ck and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("hash") != dh and not o.restart:
            raise SystemExit(f"sft: {o.out} has checkpoints for other data "
                             f"(hash {old.get('hash')} vs {dh}); --restart "
                             "to start over, or another --out")
    if o.restart and ck:
        shutil.rmtree(os.path.join(o.out, "checkpoints"))
        ck = None
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"hash": dh, "train_rows": len(rows_tr),
                   "eval_rows": len(rows_ev), "max_len": o.max_len,
                   "family": o.family, "base": o.base,
                   "data": os.path.abspath(o.data) if os.path.exists(o.data)
                   else o.data}, fh, indent=1)
    model, tok, used = load_for_training(o)
    o.backend_used = used
    log(f"sft: {used}: {o.base}, LoRA r {o.rank} alpha {o.alpha}, "
        f"{'4-bit' if o.load_4bit else 'unquantized'}")
    try:
        model.print_trainable_parameters()
    except Exception:  # noqa: BLE001 - a report
        pass
    o.eval_n = len(rows_ev)
    args = training_args(o, len(rows_tr))
    tr = make_trainer(model, tok, args, Rows(rows_tr),
                      Rows(rows_ev) if rows_ev else None,
                      [speed_callback(rows_tr, o.accum, o.probe_steps)], used)
    if ck:
        log(f"sft: resuming from {ck}")
    t0 = time.monotonic()
    res = tr.train(resume_from_checkpoint=ck)
    hours = (time.monotonic() - t0) / 3600
    metrics = dict(res.metrics)
    if rows_ev:
        metrics.update(tr.evaluate())
    adir = os.path.join(o.out, "adapter")
    tr.save_model(adir)
    save_tokenizer(tok0, adir, o.family)
    summary = {"base": o.base, "family": o.family, "backend": used,
               "rank": o.rank, "alpha": o.alpha, "max_len": o.max_len,
               "load_4bit": o.load_4bit, "epochs": o.epochs,
               "lr": o.lr, "accum": o.accum, "hours": round(hours, 3),
               "metrics": {k: (round(v, 5) if isinstance(v, float) else v)
                           for k, v in metrics.items()},
               "lengths": rep, "data_hash": dh, "versions": versions(),
               "adapter": adir}
    with open(os.path.join(o.out, "train.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    log(f"sft: adapter -> {adir}; eval loss "
        f"{metrics.get('eval_loss')}; {hours:.2f} h")
    return summary


def save_tokenizer(tok, path, family):
    """The tokenizer with our template as its chat_template."""
    tpl = chat.jinja(family)
    if tpl:
        tok.chat_template = tpl
    tok.save_pretrained(path)


# ----------------------------------------------------------- after it

def run_info(run):
    with open(os.path.join(run, "train.json"), encoding="utf-8") as fh:
        return json.load(fh)


def merge(o) -> str:
    """The adapter merged into the unquantized base -> <run>/merged."""
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    info = run_info(o.run)
    base = o.base or info["base"]
    dt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}[o.dtype]
    kw = dtype_kw(transformers, dt)
    dev = o.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw["device_map"] = {"": 0} if dev == "cuda" else {"": "cpu"}
    log(f"sft: merging {info['adapter']} into {base} ({o.dtype}, {dev})")
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    model = PeftModel.from_pretrained(model, info["adapter"])
    model = model.merge_and_unload()
    out = o.out or os.path.join(o.run, "merged")
    model.save_pretrained(out, max_shard_size="5GB")
    save_tokenizer(load_tokenizer(info["adapter"]), out, info["family"])
    log(f"sft: merged model -> {out}")
    return out


def gguf(o) -> list[list[str]]:
    """llama.cpp conversion of <run>/merged (commands run, or printed)."""
    merged = o.merged or os.path.join(o.run, "merged")
    stem = os.path.join(o.run, "gguf", o.name)
    f16, q = stem + "-f16.gguf", f"{stem}-{o.quant}.gguf"
    conv = os.path.join(o.llama_cpp or "LLAMA_CPP", "convert_hf_to_gguf.py")
    quant = None
    if o.llama_cpp:
        for c in (os.path.join(o.llama_cpp, "build", "bin", "llama-quantize"),
                  os.path.join(o.llama_cpp, "llama-quantize")):
            if os.path.exists(c):
                quant = c
    quant = quant or shutil.which("llama-quantize") or "llama-quantize"
    cmds = [[sys.executable, conv, merged, "--outfile", f16, "--outtype",
             "f16"], [quant, f16, q, o.quant]]
    if not o.llama_cpp:
        log("sft: no --llama-cpp; run (llama.cpp checked out and built, "
            "pip install -r requirements.txt there):")
        for c in cmds:
            log("  " + " ".join(c))
        return cmds
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for c in cmds:
        log("sft: " + " ".join(c))
        subprocess.run(c, check=True)
    log(f"sft: {q}")
    return cmds


def write_modelfile(o) -> str:
    info = run_info(o.run)
    g = o.gguf or os.path.join(o.run, "gguf", f"{o.name}-{o.quant}.gguf")
    if not o.gguf and not os.path.exists(g):
        # gguf's --name (the file stem, "tsk") need not be the Ollama name
        found = sorted(glob.glob(os.path.join(o.run, "gguf",
                                              f"*-{o.quant}.gguf")))
        if len(found) == 1:
            g = found[0]
        elif len(found) > 1:
            raise SystemExit(f"sft: no {g}; pass --gguf (found: "
                             f"{', '.join(found)})")
    if not os.path.exists(g):
        if o.create:
            raise SystemExit(f"sft: no {g} for ollama create; run gguf "
                             "first, or pass --gguf")
        log(f"sft: warning: no {g} yet (the Modelfile points at it)")
    path = os.path.join(o.run, "Modelfile")
    text = chat.modelfile(os.path.abspath(g), info["family"], o.num_ctx)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    log(f"sft: {path}; serve with: ollama create {o.name} -f {path} && "
        f"python -m flow.live ... --backend ollama --model {o.name}")
    if o.create:
        subprocess.run(["ollama", "create", o.name, "-f", path], check=True)
    return path


def stats(o) -> dict:
    o.family = o.template or chat.family_for(o.base)
    tok = load_tokenizer(o.base)
    exs = [clean(e) for e in load_examples(
        o.data, o.per, correct_misses=o.correct_misses,
        avatar_weight=o.avatar_weight)]
    rep = length_report(tok, exs, o.family)
    print(report_text(rep, o.family))
    probs = check_template(tok, o.family, exs[0]) if exs else []
    for p in probs:
        print(f"TEMPLATE MISMATCH {p}")
    if not probs and exs:
        print(f"template {o.family}: one turn agrees with the tokenizer's")
    if o.json:
        with open(o.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return rep


# ----------------------------------------------------------------- cli

def add_data_args(p):
    p.add_argument("--data", required=True,
                   help="sft.jsonl (flow.dataexport sft --meta) or a "
                        "flow.dataset --out folder")
    p.add_argument("--per", default="session", choices=("session", "turn"),
                   help="with a dataset folder: one example per session "
                        "[default] or per turn")
    p.add_argument("--correct-misses", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="with a dataset folder: write each missed pose "
                        "target as the value the solver reached [on]")
    p.add_argument("--avatar-weight", action="append", default=[],
                   metavar="NAME=N",
                   help="with a dataset folder: repeat that avatar's "
                        "examples N times (repeatable), e.g. lafei=2; "
                        "held-out sessions repeat too")
    p.add_argument("--base", default=DEFAULT_BASE,
                   help=f"base model [{DEFAULT_BASE}]; e.g. Qwen/Qwen3-4B, "
                        "google/gemma-3-12b-it")
    p.add_argument("--template", default=None, choices=chat.FAMILIES,
                   help="chat template family [from the base's name]")


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats", help="token length distribution")
    add_data_args(p)
    p.add_argument("--json", default=None)
    p = sub.add_parser("train")
    add_data_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="auto",
                   choices=("auto", "unsloth", "peft"))
    p.add_argument("--max-len", default="auto",
                   help="tokens per example, or auto (p99 of the data)")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="stop after this many optimizer steps (a probe)")
    p.add_argument("--accum", type=int, default=16,
                   help="sequences per optimizer step")
    p.add_argument("--warmup", type=float, default=0.03,
                   help="share of the steps warming up")
    p.add_argument("--eval-frac", type=float, default=EVAL_FRAC)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--split-by-row", action="store_true",
                   help="examples without session ids: each row its own "
                        "session (only right for a per-session export)")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--keep", type=int, default=3, help="checkpoints kept")
    p.add_argument("--log-steps", type=int, default=5)
    p.add_argument("--probe-steps", type=int, default=10,
                   help="steps between speed reports")
    p.add_argument("--no-4bit", dest="load_4bit", action="store_false",
                   help="the base unquantized (a CPU check, or a GPU with "
                        "room)")
    p.add_argument("--attn", default="sdpa",
                   help="attention implementation (sdpa; flash_attention_2 "
                        "if installed)")
    p.add_argument("--allow-template-mismatch", action="store_true")
    p.add_argument("--restart", action="store_true",
                   help="drop the checkpoints and start over")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("merge")
    p.add_argument("--run", required=True)
    p.add_argument("--base", default=None)
    p.add_argument("--out", default=None, help="[<run>/merged]")
    p.add_argument("--dtype", default="float16",
                   choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device", default="auto",
                   help="auto (cuda when there; 8B in fp16 is ~16 GB), cpu")
    p = sub.add_parser("gguf")
    p.add_argument("--run", required=True)
    p.add_argument("--merged", default=None)
    p.add_argument("--llama-cpp", default=None,
                   help="a llama.cpp checkout (built); without: print the "
                        "commands")
    p.add_argument("--name", default="tsk", help="GGUF file stem")
    p.add_argument("--quant", default="Q4_K_M")
    p = sub.add_parser("modelfile")
    p.add_argument("--run", required=True)
    p.add_argument("--gguf", default=None,
                   help="[<run>/gguf/<name>-<quant>.gguf]")
    p.add_argument("--name", default="tsk")
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--create", action="store_true", help="ollama create")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    o = parse(list(sys.argv[1:] if argv is None else argv))
    if o.cmd == "stats":
        stats(o)
    elif o.cmd == "train":
        train(o)
    elif o.cmd == "merge":
        merge(o)
    elif o.cmd == "gguf":
        gguf(o)
    else:
        write_modelfile(o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
