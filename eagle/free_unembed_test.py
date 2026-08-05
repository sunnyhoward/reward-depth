#!/usr/bin/env python
"""Is the EAGLE readout limited by the FROZEN UNEMBEDDING, or by the information at h_L?

Every head so far ends `model.lm_head(model.model.norm(x))` — the base's own unembedding, frozen.
So a preference only registers to the extent it aligns with differences that geometry already
expresses. That is §18's diagnosis of the UF failure, and the standing explanation for §13's
ceiling surviving attention, capacity and 5x training (all three improve the ADAPTER; the adapter
was never the constraint).

It is also the live suspect for the confound blocking this whole project: measured twice on
2026-08-05, the "depth" signal tracks head competence (.152/.202/.380 at L4/L12/L24) almost
exactly — in item 2's encoding table and in the refusal ladder's install strength.

THIS TEST. Distil two heads at each L, identically, on the same replay corpus:
    tf      — frozen lm_head (the current head)
    tffree  — its own output projection, INITIALISED FROM lm_head so step 0 is identical
then measure, on the FROZEN base with no lower-stack training, how well each readout ranks styc
pairs by likelihood.

    free >> pinned  ->  head competence is an artifact of the aperture; the depth ladder is
                        confounded by something removable, and §1/§17/item-2 need redoing.
    free ~= pinned  ->  the information is not at h_L; readout ceiling IS representational
                        ceiling, and the depth signature is real (which §13 asserted but never
                        showed).

Raw preference accuracy is reported, not reference-corrected: at base, policy == ref, so a
reference-corrected number is 0 by construction (§14's brit step-0 trap). Per-token as well as
sum, because §18 found the sum-level ranking INVERTS on length.

Env: LAYERS=4,12,24 STEPS=400 BATCH=16 LR=1e-3 SEED=0 N_EVAL=256
Out: /workspace/free_unembed/results.json
"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from eagle_common import (build_questions, variants, render, render_prompt, make_head,   # noqa
                          comp_slices, gather_logps, FACTOR_PAIRS, MODEL, DEV)
from helpers import ResidualCapture                                                       # noqa

LAYERS = [int(x) for x in E("LAYERS", "4,12,24").split(",")]
STEPS, BATCH = int(E("STEPS", 400)), int(E("BATCH", 16))
LR, SEED = float(E("LR", 1e-3)), int(E("SEED", 0))
# The free projection is 311M params against the adapter's 25M. At a shared LR it gets DISTURBED
# rather than helped — measured in the smoke test: style per-token 1.000 (pinned) vs .422 (free)
# after 20 steps, i.e. the free head had unlearned the aperture it started from. It gets its own
# much smaller LR so it stays near lm_head unless moving genuinely buys accuracy; otherwise the
# comparison tests optimisation difficulty, not whether the aperture was the constraint.
OUT_LR = float(E("OUT_LR", 1e-5))
N_EVAL = int(E("N_EVAL", 256))
OUT = "/workspace/free_unembed"
os.makedirs(OUT, exist_ok=True)
REPLAY_F = E("REPLAY_F", "/workspace/eagle_replay_2048x128.pt")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
qs, tr, te = build_questions(SEED)
te_idx = np.where(te)[0]
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
BLOCKS = list(model.model.layers)
HID, VOCAB = model.config.hidden_size, model.config.vocab_size
replay = torch.load(REPLAY_F).long()
train_pool, held_pool = replay[: int(.9 * len(replay))], replay[int(.9 * len(replay)):]
print(f"[free-unembed] {MODEL} hid={HID} vocab={VOCAB} layers={LAYERS}", flush=True)


def rbatch(k, pool):
    ids = pool[torch.randint(0, pool.shape[0], (k,))].to(DEV)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids))


def distil(arch, L):
    head = (make_head(HID, arch, vocab=VOCAB) if arch == "tffree" else make_head(HID, arch)).to(DEV)
    if arch == "tffree":
        head.attach_lm_head(model)        # start identical to the pinned head
    if arch == "tffree":
        out_p = list(head.out.parameters())
        ids = {id(q) for q in out_p}
        rest = [q for q in head.parameters() if id(q) not in ids]
        opt = torch.optim.AdamW([{"params": rest, "lr": LR},
                                 {"params": out_p, "lr": OUT_LR}])
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=LR)
    for step in range(STEPS):
        enc = rbatch(BATCH, train_pool)
        am = torch.ones_like(enc["input_ids"][:, 1:]).bool()
        with torch.no_grad(), ResidualCapture([BLOCKS[L]]) as cap:
            t_lsm = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
        s_lsm = F.log_softmax(head(cap.get()[0][:, :-1], model,
                                   pad_mask=enc["attention_mask"][:, :-1]), -1)
        kl = ((t_lsm.exp() * (t_lsm - s_lsm)).sum(-1) * am).sum() / am.sum()
        opt.zero_grad(); kl.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"    {arch} L{L} step {step+1}: kl {float(kl):.3f}", flush=True)
    with torch.no_grad():
        enc = rbatch(64, held_pool)
        am = torch.ones_like(enc["input_ids"][:, 1:]).bool()
        with ResidualCapture([BLOCKS[L]]) as cap:
            t_arg = model(**enc).logits[:, :-1].argmax(-1)
        s_arg = head(cap.get()[0][:, :-1], model,
                     pad_mask=enc["attention_mask"][:, :-1]).argmax(-1)
        agree = float(((s_arg == t_arg) & am).sum() / am.sum())
    return head, agree, float(kl)


@torch.no_grad()
def rank_acc(head, L, factor):
    """Raw preference accuracy of this readout on held-out styc pairs (sum and per-token)."""
    pairs = FACTOR_PAIRS[factor]
    hits_s, hits_t = [], []
    idx = list(te_idx[:N_EVAL])
    for s in range(0, len(idx), 16):
        items = idx[s:s + 16]
        texts, plens = [], []
        for k, i in enumerate(items):
            a, b = pairs[k % len(pairs)]
            q = qs[i]; v = variants(q); pl = len(tok(render_prompt(q)).input_ids)
            texts += [render(q, v[a]), render(q, v[b])]; plens += [pl, pl]
        enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
        spans = comp_slices(tok, texts, plens, enc)
        if head is None:                                   # full-model final-logit reference row
            lsm = F.log_softmax(model(**enc).logits[:, :-1].float(), -1)
        else:
            with ResidualCapture([BLOCKS[L]]) as cap:
                model(**enc)
            lsm = F.log_softmax(head(cap.get()[0][:, :-1], model,
                                     pad_mask=enc.attention_mask[:, :-1]), -1)
        lp = gather_logps(lsm, enc, spans)
        ntok = torch.tensor([hi - lo for lo, hi in spans], device=lp.device, dtype=lp.dtype)
        d_s = lp.view(-1, 2); d_t = (lp / ntok).view(-1, 2)
        hits_s += (d_s[:, 0] > d_s[:, 1]).float().cpu().tolist()
        hits_t += (d_t[:, 0] > d_t[:, 1]).float().cpu().tolist()
    return float(np.mean(hits_s)), float(np.mean(hits_t))


res = {"model": MODEL, "layers": LAYERS, "cells": {}}
print("\n--- reference row: full model, final logits ---", flush=True)
for f in ("style", "correct"):
    s_, t_ = rank_acc(None, 0, f)
    res["cells"][f"final_{f}"] = dict(acc_sum=s_, acc_tok=t_)
    print(f"  full model  {f:8s}  sum {s_:.3f}  per-token {t_:.3f}", flush=True)

for L in LAYERS:
    for arch in ("tf", "tffree"):
        print(f"\n--- distilling {arch} at L{L} ---", flush=True)
        head, agree, kl = distil(arch, L)
        row = dict(agreement=agree, kl=kl)
        for f in ("style", "correct"):
            s_, t_ = rank_acc(head, L, f)
            row[f"{f}_sum"], row[f"{f}_tok"] = s_, t_
        res["cells"][f"{arch}_L{L}"] = row
        print(f"  {arch} L{L}: agreement {agree:.3f} | style tok {row['style_tok']:.3f} | "
              f"correct tok {row['correct_tok']:.3f}", flush=True)
        del head; torch.cuda.empty_cache()
        json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)

print("\n=== PINNED vs FREE unembedding (per-token preference accuracy) ===")
print(f"{'L':>4s} | {'agree pin':>9s} {'agree free':>10s} | {'style pin':>9s} {'style free':>10s} "
      f"| {'corr pin':>8s} {'corr free':>9s}")
for L in LAYERS:
    a, b = res["cells"].get(f"tf_L{L}"), res["cells"].get(f"tffree_L{L}")
    if not a or not b:
        continue
    print(f"{L:4d} | {a['agreement']:9.3f} {b['agreement']:10.3f} | {a['style_tok']:9.3f} "
          f"{b['style_tok']:10.3f} | {a['correct_tok']:8.3f} {b['correct_tok']:9.3f}")
print("\nfree >> pinned at low L -> the aperture was the constraint (depth ladder confounded)")
print("free ~= pinned          -> information is not at h_L (depth signature is real)")
print(f"\nwrote {OUT}/results.json")
