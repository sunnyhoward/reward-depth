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
                          comp_slices, MODEL, DEV)
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
heads = {L: EagleHead(HID).to(DEV) for L in LAYERS}
opts = {L: torch.optim.AdamW(heads[L].parameters(), lr=LR) for L in LAYERS}
print(f"[heads] layers {LAYERS} | {sum(p.numel() for p in heads[LAYERS[0]].parameters())/1e6:.1f}M each", flush=True)

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
for step in range(STEPS):
    texts, plens = batch_texts(tr_idx, BATCH)
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    am = enc.attention_mask[:, 1:].bool()          # predict positions with a real prev token
    with torch.no_grad(), ResidualCapture([BLOCKS[L] for L in LAYERS]) as cap:
        t_lsm = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
    bufs = cap.get()
    for k, L in enumerate(LAYERS):
        h = bufs[k][:, :-1]
        s_lsm = F.log_softmax(heads[L](h, model), -1)
        # forward KL(teacher || student), masked to real positions
        kl = ((t_lsm.exp() * (t_lsm - s_lsm)).sum(-1) * am).sum() / am.sum()
        opts[L].zero_grad(); kl.backward(); opts[L].step()
        hist[L].append(float(kl.detach()))
    if (step + 1) % 25 == 0:
        print(f"  step {step+1:4d}: " + " ".join(f"L{L} kl={np.mean(hist[L][-25:]):.3f}" for L in LAYERS), flush=True)

# held-out agreement with the base model's argmax
res = dict(layers=LAYERS, steps=STEPS)
with torch.no_grad():
    texts, plens = batch_texts(te_idx, 64)
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    am = enc.attention_mask[:, 1:].bool()
    with ResidualCapture([BLOCKS[L] for L in LAYERS]) as cap:
        t_arg = model(**enc).logits[:, :-1].argmax(-1)
    bufs = cap.get()
    for k, L in enumerate(LAYERS):
        s_arg = heads[L](bufs[k][:, :-1], model).argmax(-1)
        agree = float(((s_arg == t_arg) & am).sum() / am.sum())
        res[f"agree_L{L}"] = agree
        res[f"kl_final_L{L}"] = float(np.mean(hist[L][-25:]))
        torch.save(heads[L].state_dict(), f"/workspace/eagle_head_L{L}.pt")
        print(f"[head L{L}] held-out top-1 agreement with base: {agree:.3f}", flush=True)
json.dump(res, open("/workspace/eagle_heads.json", "w"), indent=1)
print("DONE", flush=True)
