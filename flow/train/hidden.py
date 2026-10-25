#!/usr/bin/env python3
"""
Hidden states of a local model over the replies of the action export
(flow.dataexport action), for the action head (flow.actionhead): per turn,
chosen layers' states at every token of the reply, with each token's
character offsets in the text and in the spoken text (which the per-
character times of the export put on the clock).

  context   --context turn [default]: the model reads the turn as it does
            when it writes the reply -- the system prompt, the history and
            the user line from the dataset records, in flow.train.chat's
            template -- and the states are those of the reply's tokens
            after it (what a served model has in hand as it speaks). --
            context none: the text alone (cheap; a different distribution
            from the one served).
  text      --text reply [default]: reply_raw, markers in it (the tuned
            model writes them); spoken: the spoken text only (a model that
            never saw a marker).
  offsets   raw [n, 2] each token's character span in the text; spoken
            [n, 2] the same in the spoken text: every raw character mapped
            through the avatar's parser fed one character at a time, so a
            marker's characters sit at its place in the speech (start ==
            end there).
  storage   fp16, rows packed in shards of --shard-tokens tokens:
            <out>/hidden-NNNNN.npz with "<key>.h" [n, layers, D], "<key>.
            raw", "<key>.spoken" (int32); <out>/index.jsonl one line per
            turn (id, key, shard, tokens); <out>/meta.json (model, adapter,
            layers, D, family, context, text, projection). One layer of an
            8B model is 8 KB a token in fp16: 15k replies of ~150 tokens
            are ~18 GB, so the default is one layer, and --proj D' stores a
            fixed random orthogonal projection to D' dimensions (seeded;
            its matrix in proj.npy).

Usage (scripts/extract_hidden.py is the same entry point):
    python -m flow.train.hidden --export data/datagen/r1/export/action \\
        --data data/datagen/r1 --model Qwen/Qwen3-8B \\
        --adapter runs/sft/q8/adapter --layers -8 --out runs/hidden/q8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    from . import chat
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from flow.train import chat

SHARD_TOKENS = 200_000


def read_export(stem):
    """(rows, npz path) of a flow.dataexport action export."""
    rows = []
    with open(stem + ".jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows, stem + ".npz"


def spoken_index(raw, parser):
    """[len(raw) + 1] the spoken length before each raw character (and at
    the end), feeding the parser one character at a time."""
    out = [0]
    for ch in raw:
        parser.feed(ch)
        out.append(len(parser.spoken))
    parser.close()
    out[-1] = len(parser.spoken)
    return np.asarray(out, np.int32), parser.spoken


def token_offsets(tokenizer, text):
    """(ids, [n, 2] char spans) of text tokenized on its own."""
    enc = tokenizer(text, add_special_tokens=False,
                    return_offsets_mapping=True)
    return list(enc["input_ids"]), np.asarray(enc["offset_mapping"],
                                              np.int32).reshape(-1, 2)


def to_spoken(raw_spans, idx, n_spoken):
    """Raw character spans -> spoken spans through idx (spoken_index)."""
    s = idx[np.clip(raw_spans[:, 0], 0, len(idx) - 1)]
    e = idx[np.clip(raw_spans[:, 1], 0, len(idx) - 1)]
    return np.stack([np.minimum(s, n_spoken), np.minimum(e, n_spoken)],
                    1).astype(np.int32)


def parsers_for(data, names):
    """{avatar: parser factory} from the dataset's registry: the body's own
    parser (markers are cut from the speech whatever their kind)."""
    try:
        from .. import body as fb
        from .. import dataexport as dx
        from .. import pose as pz
    except ImportError:
        from flow import body as fb
        from flow import dataexport as dx
        from flow import pose as pz
    reg = dx.load_registry(data) if data else {"avatars": {}}
    out = {}
    for n in names:
        a = reg["avatars"].get(n)
        body = (a or {}).get("body") or os.path.join("rigs", f"{n}.body.json")
        bm = fb.load(body)
        gest = {g.lower(): g for g in (a or {}).get("gestures") or ()}
        out[n] = (lambda bm=bm, gest=gest:
                  pz.parser_for(bm, gestures=gest, moves=True))
    return out


def turn_messages(rec, system):
    msgs = [{"role": "system", "content": system}]
    for h in rec.get("history") or ():
        msgs.append({"role": "user", "content": h["user_content"]})
        msgs.append({"role": "assistant", "content": h["reply"]})
    msgs.append({"role": "user", "content": rec["user_content"]})
    return msgs


def layer_ids(spec, n_layers):
    """--layers "-8,-1" (negative from the end; hidden_states[0] is the
    embeddings, so -1 is the last layer's output)."""
    out = []
    for x in str(spec).split(","):
        x = x.strip()
        if not x:
            continue
        k = int(x)
        out.append(k if k >= 0 else n_layers + 1 + k)
    for k in out:
        if not 0 <= k <= n_layers:
            raise SystemExit(f"--layers {spec}: {k} not in 0..{n_layers}")
    return out


class Shards:
    def __init__(self, out, limit=SHARD_TOKENS):
        self.out, self.limit = out, int(limit)
        self.k, self.n, self.buf = 0, 0, {}
        self.index = open(os.path.join(out, "index.jsonl"), "w",
                          encoding="utf-8")

    def add(self, row_id, key, h, raw, spoken):
        self.buf[key + ".h"] = h
        self.buf[key + ".raw"] = raw
        self.buf[key + ".spoken"] = spoken
        self.index.write(json.dumps({"id": row_id, "key": key,
                                     "shard": f"hidden-{self.k:05d}.npz",
                                     "tokens": int(h.shape[0])}) + "\n")
        self.n += int(h.shape[0])
        if self.n >= self.limit:
            self.flush()

    def flush(self):
        if self.buf:
            path = os.path.join(self.out, f"hidden-{self.k:05d}.npz")
            np.savez(path + ".tmp.npz", **self.buf)
            os.replace(path + ".tmp.npz", path)
            self.k += 1
        self.buf, self.n = {}, 0
        self.index.flush()

    def close(self):
        self.flush()
        self.index.close()


def extract(o, log=print) -> dict:
    import torch
    try:
        from . import hf as fhf
        from .. import dataset as ds
        from .. import dataexport as dx
    except ImportError:
        from flow import dataexport as dx
        from flow import dataset as ds
        from flow.train import hf as fhf
    rows, _npz = read_export(o.export)
    if o.limit:
        rows = rows[:o.limit]
    family = fhf.family_of(o.model, o.adapter, o.template)
    recs, prompts = {}, {}
    if o.context == "turn":
        if not o.data:
            raise SystemExit("--context turn needs --data (the records)")
        recs = {r["id"]: r for r in ds.iter_records(o.data)}
        prompts = dx.load_registry(o.data)["prompts"]
    lm, tok = fhf.load(o.model, o.adapter, o.load_4bit, o.device, log)
    n_layers = int(getattr(lm.config, "num_hidden_layers", 0) or getattr(
        getattr(lm.config, "text_config", None), "num_hidden_layers", 0))
    layers = layer_ids(o.layers, n_layers)
    parsers = parsers_for(o.data, sorted({r["avatar"] for r in rows}))
    os.makedirs(o.out, exist_ok=True)
    dev = next(lm.parameters()).device
    proj = None
    sh = Shards(o.out, o.shard_tokens)
    t0, n_tok, D = time.monotonic(), 0, None
    for i, r in enumerate(rows):
        text = r["reply_raw"] if o.text == "reply" else r["spoken"]
        idx, spoken = spoken_index(text, parsers[r["avatar"]]())
        if spoken != r["spoken"]:
            log(f"hidden: {r['id']}: the parser's spoken text differs from "
                "the record's; offsets may be off")
        ids, raw = token_offsets(tok, text)
        pre = []
        if o.context == "turn":
            rec = recs.get(r["id"])
            if rec is None:
                log(f"hidden: {r['id']} not in {o.data}; text alone")
            else:
                msgs = turn_messages(rec, prompts[rec["avatar"]["prompt_sha"]])
                p = chat.prompt(msgs, family) if family != "hf" else \
                    tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
                pre = list(tok(p, add_special_tokens=False)["input_ids"])
        x = torch.tensor([pre + ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            out = lm(input_ids=x, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        h = torch.stack([hs[k][0, len(pre):] for k in layers], 1).float()
        if o.proj:
            if proj is None:
                g = torch.Generator().manual_seed(int(o.seed))
                q, _ = torch.linalg.qr(torch.randn(h.shape[-1], o.proj,
                                                   generator=g))
                proj = q.to(h.device)
                np.save(os.path.join(o.out, "proj.npy"),
                        q.numpy().astype(np.float32))
            h = h @ proj
        h = h.cpu().numpy().astype(np.float16)
        D = h.shape[-1]
        sh.add(r["id"], r["key"], h, raw, to_spoken(raw, idx, len(spoken)))
        n_tok += len(ids)
        if (i + 1) % 100 == 0:
            el = time.monotonic() - t0
            log(f"hidden: {i + 1}/{len(rows)} turns, {n_tok} tokens, "
                f"{el / (i + 1):.2f} s a turn")
    sh.close()
    meta = {"model": o.model, "adapter": o.adapter, "family": family,
            "layers": layers, "layers_arg": o.layers, "dim": D,
            "context": o.context, "text": o.text, "proj": o.proj,
            "seed": o.seed, "turns": len(rows), "tokens": n_tok,
            "export": os.path.abspath(o.export)}
    with open(os.path.join(o.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    log(f"hidden: {len(rows)} turns, {n_tok} tokens, layers {layers}, dim "
        f"{D} -> {o.out} ({time.monotonic() - t0:.0f}s)")
    return meta


def load_index(d):
    out = {}
    with open(os.path.join(d, "index.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                e = json.loads(line)
                out[e["id"]] = e
    return out


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", required=True,
                    help="flow.dataexport action's --out stem")
    ap.add_argument("--data", default=None,
                    help="the flow.dataset folder (records, registry)")
    ap.add_argument("--model", required=True, help="base or merged model")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--template", default=None, choices=chat.FAMILIES)
    ap.add_argument("--layers", default="-8",
                    help="comma-separated layer indices (negative: from "
                         "the end; several: --layers=-8,-1) [-8]")
    ap.add_argument("--context", default="turn", choices=("turn", "none"))
    ap.add_argument("--text", default="reply", choices=("reply", "spoken"))
    ap.add_argument("--proj", type=int, default=0,
                    help="store a random orthogonal projection to this "
                         "many dimensions [0: the full state]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-tokens", type=int, default=SHARD_TOKENS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    extract(parse(list(sys.argv[1:] if argv is None else argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
