#!/usr/bin/env python
"""Build ONLY the replay bank that `pf_common.load_replay()` needs.

`sup_prepare.py` builds three artifacts in one pass -- replay bank, EAGLE head, K-FAC bundle --
and the head alone is ~15 min. `pf_train.py MODE=final` needs none of them except the bank, and
`load_replay()` is called unconditionally at setup, so even a W_REPLAY=0 arm will not start
without the file. This builds the bank and nothing else, so a replay-OFF control does not have to
wait on artifacts it will never read, and no half-trained head is left on disk to be picked up by
something later.

The bank block below is lifted verbatim from sup_prepare.py §1 (`# ---------- 1. replay bank from
his shards ----------`) so the two cannot drift into producing different banks. Source shards are
`supervisor/corpus_shards/kfac_corpus_*.jsonl`, which are committed.

Env: SUP_MODEL=Qwen/Qwen3.5-4B CTX=512 N_BANK=4000 SEED=0
"""
import glob
import json
import os
import random
import sys

import torch
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "supervisor"))
sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import MODEL, SHARDS  # noqa: E402

E = os.environ.get
OUT = E("SUP_OUT", "/workspace/sup")
CTX = int(E("CTX", 512))
N_BANK = int(E("N_BANK", 4000))
SEED = int(E("SEED", 0))
BANK_F = f"{OUT}/replay_bank.pt"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

if os.path.exists(BANK_F):
    b = torch.load(BANK_F)
    sys.exit(f"[bank] exists {BANK_F} {tuple(b['ids'].shape)} -- delete it to rebuild")


def shard_records():
    for f in sorted(glob.glob(f"{SHARDS}/kfac_corpus_*.jsonl")):
        for line in open(f):
            yield json.loads(line)


# Keep the TAIL of each record: the scored span is always the suffix, so truncating from the
# left preserves it whole.
ids_rows, starts = [], []
for r in shard_records():
    ids, pl = r["token_ids"], r["prefix_length"]
    if len(ids) > CTX:
        cut = len(ids) - CTX
        ids, pl = ids[cut:], max(pl - cut, 1)
    if len(ids) - pl < 8:
        continue
    npad = CTX - len(ids)
    ids_rows.append([PAD] * npad + ids)
    starts.append(npad + pl)

order = random.Random(SEED).sample(range(len(ids_rows)), min(N_BANK, len(ids_rows)))
bank = dict(ids=torch.tensor([ids_rows[i] for i in order], dtype=torch.long),
            start=torch.tensor([starts[i] for i in order], dtype=torch.long), ctx=CTX)
torch.save(bank, BANK_F)
print(f"[bank] wrote {BANK_F} {tuple(bank['ids'].shape)} | "
      f"{int((bank['ctx'] - bank['start']).sum())} scored tokens available")
