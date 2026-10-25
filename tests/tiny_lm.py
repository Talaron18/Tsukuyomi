"""A tiny random Qwen3 causal LM with a byte-level BPE tokenizer that has
ChatML's special tokens: the trainer, the hf backend and the hidden-state
extractor run on it on CPU in seconds. Needs torch, transformers and
tokenizers (callers importorskip them)."""

from __future__ import annotations

import os

from flow.train import chat

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>",
           "</think>"]


def build(path, texts, template="qwen3", vocab=600, seed=0):
    """A model folder at path (name it with the family, e.g. tiny-qwen3)."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, \
        trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, \
        Qwen3ForCausalLM
    os.makedirs(path, exist_ok=True)
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL,
                             initial_alphabet=pre_tokenizers.ByteLevel
                             .alphabet())
    tk.train_from_iterator(texts, tr)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk,
                                  eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>")
    tok.chat_template = chat.jinja(template)
    tok.save_pretrained(path)
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32,
                      intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1,
                      head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=False,
                      eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
                      pad_token_id=tok.convert_tokens_to_ids(
                          "<|endoftext|>"))
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    return path
