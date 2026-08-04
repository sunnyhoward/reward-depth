#!/usr/bin/env python
"""Pretrain EAGLE early-exit heads at several layers, all in one pass (one forward per batch
feeds every head — the capture is shared). Head = residual MLP at h_L + the model's own frozen
final norm/lm_head; distilled to the BASE model's final logits (forward KL, teacher=base) on
styc-domain text. Head-only gradients; the model is never touched.

This runs BEFORE any preference training: the head must be a competent generic readout first,
so that stage-1 DPO through it has to change the LOWER STACK, not just build the readout.

Env: LAYERS=4,12,24,32 STEPS=400 BATCH=16 LR=1e-3 SEED=0
Saves: /workspace/eagle_head_L{L}.pt (+ /workspace/eagle_heads.json summary)"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eagle_common import (build_questions, variants, render, render_prompt, EagleHead,
                          make_head, head_path, comp_slices, MODEL, DEV)
from helpers import ResidualCapture

E = os.environ.get
LAYERS = [int(x) for x in E("LAYERS", "4,12,24,32").split(",")]
STEPS, BATCH, LR, SEED = int(E("STEPS", 400)), int(E("BATCH", 16)), float(E("LR", 1e-3)), int(E("SEED", 0))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

qs, tr, te = build_questions(SEED)
tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
BLOCKS = list(model.model.layers); HID = model.config.hidden_size
ARCH = E("HEAD_ARCH", "mlp")
# HEAD_DATA=styc   : the original narrow templated text (leaves the head off-distribution)
# HEAD_DATA=replay : the frozen model's OWN samples over many random prompts (eagle_replay.py) —
#                    the user's 2026-08-04 direction, so the head is a general readout before it
#                    is frozen for stage 1.
HEAD_DATA = E("HEAD_DATA", "styc")
REPLAY_F = E("REPLAY_F", "/workspace/eagle_replay_2048x128.pt")
_replay = None
if HEAD_DATA == "replay":
    _replay = torch.load(REPLAY_F).long()
    print(f"[head] replay corpus {tuple(_replay.shape)} from {REPLAY_F}", flush=True)
heads = {L: make_head(HID, ARCH).to(DEV) for L in LAYERS}
opts = {L: torch.optim.AdamW(heads[L].parameters(), lr=LR) for L in LAYERS}
print(f"[heads] arch={ARCH} layers {LAYERS} | {sum(p.numel() for p in heads[LAYERS[0]].parameters())/1e6:.1f}M each", flush=True)

rgen = random.Random(SEED + 7)
VKEYS = ["ce", "we", "ct", "wt"]

def batch_texts(idx_pool, k):
    texts, plens = [], []
    for i in rgen.sample(list(idx_pool), k):
        q = qs[i]; v = variants(q)
        texts.append(render(q, v[rgen.choice(VKEYS)]))
        plens.append(len(tok(render_prompt(q)).input_ids))
    return texts, plens

hist = {L: [] for L in LAYERS}
def replay_batch(k, pool):
    idx = torch.randint(0, pool.shape[0], (k,))
    ids = pool[idx].to(DEV)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids))

for step in range(STEPS):
    if HEAD_DATA == "replay":
        enc = replay_batch(BATCH, _replay[: int(_replay.shape[0] * 0.9)])
        am = torch.ones_like(enc["input_ids"][:, 1:]).bool()
    else:
        texts, plens = batch_texts(tr_idx, BATCH)
        enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
        am = enc.attention_mask[:, 1:].bool()      # predict positions with a real prev token
    with torch.no_grad(), ResidualCapture([BLOCKS[L] for L in LAYERS]) as cap:
        t_lsm = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
    bufs = cap.get()
    for k, L in enumerate(LAYERS):
        h = bufs[k][:, :-1]
        s_lsm = F.log_softmax(heads[L](h, model, pad_mask=enc["attention_mask"][:, :-1]), -1)
        # forward KL(teacher || student), masked to real positions
        kl = ((t_lsm.exp() * (t_lsm - s_lsm)).sum(-1) * am).sum() / am.sum()
        opts[L].zero_grad(); kl.backward(); opts[L].step()
        hist[L].append(float(kl.detach()))
    if (step + 1) % 25 == 0:
        print(f"  step {step+1:4d}: " + " ".join(f"L{L} kl={np.mean(hist[L][-25:]):.3f}" for L in LAYERS), flush=True)

# held-out agreement with the base model's argmax
res = dict(layers=LAYERS, steps=STEPS)
with torch.no_grad():
    if HEAD_DATA == "replay":
        enc = replay_batch(64, _replay[int(_replay.shape[0] * 0.9):])
        am = torch.ones_like(enc["input_ids"][:, 1:]).bool()
    else:
        texts, plens = batch_texts(te_idx, 64)
        enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
        am = enc.attention_mask[:, 1:].bool()
    with ResidualCapture([BLOCKS[L] for L in LAYERS]) as cap:
        t_arg = model(**enc).logits[:, :-1].argmax(-1)
    bufs = cap.get()
    for k, L in enumerate(LAYERS):
        s_arg = heads[L](bufs[k][:, :-1], model, pad_mask=enc["attention_mask"][:, :-1]).argmax(-1)
        agree = float(((s_arg == t_arg) & am).sum() / am.sum())
        res[f"agree_L{L}"] = agree
        res[f"kl_final_L{L}"] = float(np.mean(hist[L][-25:]))
        torch.save(heads[L].state_dict(), head_path(L, ARCH) if HEAD_DATA == "styc"
                   else head_path(L, ARCH).replace(".pt", "_replay.pt"))
        print(f"[head L{L}] held-out top-1 agreement with base: {agree:.3f}", flush=True)
res["arch"] = ARCH; res["head_data"] = HEAD_DATA
json.dump(res, open(f"/workspace/eagle_heads_{ARCH}_{HEAD_DATA}.json", "w"), indent=1)
print("DONE", flush=True)
