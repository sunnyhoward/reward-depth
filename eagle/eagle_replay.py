#!/usr/bin/env python
"""Generative-replay corpus for EAGLE head pretraining (user direction, 2026-08-04).

The head must be a competent GENERAL readout before any preference training touches it, and it
must then be FROZEN (eagle_dpo.py FREEZE_HEAD). Distilling it only on the narrow styc/brit
template text leaves it incompetent off-distribution — measured 2026-08-04: KL(base||head) 1.74
at answer positions vs 0.58 elsewhere, and 0.14 vs 0.38 once trained longer. This samples the
frozen base model itself over many random prompts so the head sees the model's own distribution.

Prefix distribution follows replay-kfac-ewc's default: 25% start from BOS (when the tokenizer
has one), 75% from a uniformly sampled 1-8 token non-special prefix. Sampling is unmodified
(T=1.0, no top-k/top-p) so the corpus is a genuine sample from the model, not a mode-seeking one.

Env: N=2048 T=128 BS=64 SEED=0  Out: /workspace/eagle_replay_{N}x{T}.pt  (int32 token ids)
"""
import os, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eagle_common import MODEL, DEV

E = os.environ.get
N, T, BS, SEED = int(E("N", 2048)), int(E("T", 128)), int(E("BS", 64)), int(E("SEED", 0))
OUT = f"/workspace/eagle_replay_{N}x{T}.pt"
if os.path.exists(OUT):
    print(f"[replay] {OUT} exists — nothing to do", flush=True); sys.exit(0)

random.seed(SEED); torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()

special = set(tok.all_special_ids)
vocab_n = model.config.vocab_size
ordinary = [i for i in range(vocab_n) if i not in special]
rg = random.Random(SEED + 11)

def prefixes(k):
    out = []
    for _ in range(k):
        if tok.bos_token_id is not None and rg.random() < 0.25:
            out.append([tok.bos_token_id])
        else:
            out.append([rg.choice(ordinary) for _ in range(rg.randint(1, 8))])
    return out

chunks, done = [], 0
while done < N:
    k = min(BS, N - done)
    pre = prefixes(k)
    mx = max(len(p) for p in pre)
    # LEFT-pad the prefixes so every row generates the same number of new tokens
    ids = torch.tensor([[tok.pad_token_id] * (mx - len(p)) + p for p in pre], device=DEV)
    am = torch.tensor([[0] * (mx - len(p)) + [1] * len(p) for p in pre], device=DEV)
    with torch.no_grad():
        g = model.generate(input_ids=ids, attention_mask=am, do_sample=True, temperature=1.0,
                           top_k=0, top_p=1.0, max_new_tokens=T, min_new_tokens=T,
                           pad_token_id=tok.pad_token_id)
    chunks.append(g[:, mx:].to(torch.int32).cpu())    # keep only the sampled continuations
    done += k
    if done % (BS * 8) == 0 or done >= N:
        print(f"  [replay] {done}/{N}", flush=True)

data = torch.cat(chunks, 0)[:N]
torch.save(data, OUT)
print(f"[replay] wrote {OUT}  shape {tuple(data.shape)}", flush=True)
print("  sample:", repr(tok.decode(data[0][:60].tolist())), flush=True)
