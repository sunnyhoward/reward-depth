#!/usr/bin/env python
"""Replay bank for the cmpdir install arms, re-tokenized for any model in the ladder.

WHY A BUILDER RATHER THAN `supervisor/replay_bank.pt`. The banked shards in
`supervisor/corpus_shards/` hold TOKEN IDS for Qwen3.5-2B (vocab 248077); every cmpdir number so
far is on qwen3-1.7b (vocab 151669), so those ids are meaningless here. The shards carry no raw
text, so the text is recovered by decoding with the 3.5 tokenizer and re-encoded with the target
model's. The CONTENT is therefore the repo's own curated replay corpus -- chat-format assistant
turns -- and only the tokenization changes.

`score_spans` marks the assistant turn, which is the span the supervisor recipe scores
(`sup_train.replay_term`). That boundary is preserved so the replay loss cannot land on the
prompt or on padding, which is the failure the old fixed-tail window had.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SRC_TOK = "Qwen/Qwen3.5-2B"          # the tokenizer the shards were written with
SHARDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "supervisor",
                      "corpus_shards")
OUT_DIR = os.environ.get("CMPDIR_REPLAY_DIR", "/workspace/cmpdir_replay")


def build(model_key="qwen3-1.7b", n=400, max_len=512):
    import dec_common as C
    from transformers import AutoTokenizer
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"replay_{model_key}.pt")
    if os.path.exists(out):
        print(f"[replay] exists: {out}")
        return out

    src = AutoTokenizer.from_pretrained(SRC_TOK)
    dst = AutoTokenizer.from_pretrained(C.model_spec(model_key).hf)
    if dst.pad_token is None:
        dst.pad_token = dst.eos_token

    rows, starts = [], []
    for fn in sorted(os.listdir(SHARDS)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(SHARDS, fn)) as f:
            for line in f:
                r = json.loads(line)
                if r.get("has_repeat_loop"):
                    continue                      # the shards' own degeneracy flag
                pl = r["prefix_length"]
                pre = src.decode(r["token_ids"][:pl], skip_special_tokens=False)
                comp = src.decode(r["token_ids"][pl:], skip_special_tokens=False)
                a = dst(pre, add_special_tokens=False)["input_ids"]
                b = dst(comp, add_special_tokens=False)["input_ids"]
                if len(b) < 8:
                    continue
                ids = (a + b)[:max_len]
                if len(ids) <= len(a):            # completion truncated away entirely
                    continue
                rows.append(ids)
                starts.append(len(a))
                if len(rows) >= n:
                    break
        if len(rows) >= n:
            break

    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), dst.pad_token_id, dtype=torch.long)
    for i, r in enumerate(rows):                  # RIGHT padding; `start` indexes from the left
        ids[i, :len(r)] = torch.tensor(r)
    bank = dict(ids=ids, start=torch.tensor(starts), lens=torch.tensor([len(r) for r in rows]),
                model=model_key, src_tokenizer=SRC_TOK)
    torch.save(bank, out)
    print(f"[replay] {len(rows)} sequences, max_len {T}, "
          f"mean scored span {float((bank['lens'] - bank['start']).float().mean()):.0f} tokens")
    print(f"[replay] → {out}")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b",
          int(sys.argv[2]) if len(sys.argv) > 2 else 400)
