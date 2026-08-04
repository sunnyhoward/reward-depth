#!/usr/bin/env python
"""Unified DPO trainer for the EAGLE two-stage experiment. Three modes via LOSS_AT x WRITE:

  stage 1        LOSS_AT=eagle WRITE=lower  DPO loss on the EAGLE readout at layer L; gradients
                 reach ONLY layers 0..L (restricted LoRA) + the head. Not aiming for a high
                 score — encode the concept low. Probe-accuracy plateau is the duration meter.
  full-DPO base  LOSS_AT=final WRITE=all    standard DPO, same data (baseline).
  stage2-only    LOSS_AT=final WRITE=upper  standard DPO restricted to layers L+1..top
                 (controls for whether stage 1 contributes anything at all).

Asserts, per spec: trainable parameters exist ONLY in the allowed layer range (+ head in eagle
mode); the base model is fully frozen. In eagle mode the reference is the frozen base lower
stack through the frozen PRETRAINED head copy.

GT hard labels (p=1 for the factor's preferred side) — labeller quality is not under test here.

Env: FACTOR=style|correct L=12 LOSS_AT=eagle WRITE=lower STEPS=300 BATCH=16 LR=1e-4 BETA=0.1
     EVAL_EVERY=25 CKPT_EVERY=25 PROBE_LAYERS=auto SEED=0 RUN_TAG=auto
Outputs: /workspace/eagle_{TAG}/  (history.json, ckpt{N}/ = adapter + head.pt)"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eagle_common import (build_questions, variants, render, render_prompt, EagleHead,
                          make_head, head_path, comp_slices, gather_logps, evaluate, probe_acc_at,
                          FACTOR_PAIRS, MODEL, DEV)
from helpers import ResidualCapture

E = os.environ.get
FACTOR, L = E("FACTOR", "style"), int(E("L", 12))
LOSS_AT, WRITE = E("LOSS_AT", "eagle"), E("WRITE", "lower")
assert LOSS_AT in ("eagle", "final") and WRITE in ("lower", "upper", "all")
assert not (LOSS_AT == "eagle" and WRITE != "lower"), "eagle loss is the stage-1 config (WRITE=lower)"
# FLIP=1 (default): install the REVERSED preference (style -> prefer terse, correct -> prefer
# wrong). The base already free-samples correct 1.0 / explained .91 — the spec's decisive metric
# (free-sampling rate) has no headroom un-flipped; the flip gives a clean 0->x dose-response
# (phase-8 flip-training design: only a competent install can express the flip).
FLIP = int(E("FLIP", 1))
HEAD_ARCH = E("HEAD_ARCH", "mlp")
# FREEZE_HEAD=1 (default): the head is NOT trained during stage 1. With it trainable, the DPO
# margin (la-ra)-(lb-rb) can be reduced by moving EITHER the lower stack OR the head, and descent
# takes the cheap path — measured 2026-08-04: head_acc hit 1.00 at step 5 with the model moving
# only 0.106 nats (styc) / 0.002 nats (brit), i.e. the "install" lived in the readout, leaving
# stage 2 nothing encoded low to propagate. Frozen, the margin can only move via layers 0..L.
FREEZE_HEAD = int(E("FREEZE_HEAD", 1))
STEPS, BATCH = int(E("STEPS", 300)), int(E("BATCH", 16))
LR, BETA = float(E("LR", 1e-4)), float(E("BETA", 0.1))
EVAL_EVERY, CKPT_EVERY, SEED = int(E("EVAL_EVERY", 25)), int(E("CKPT_EVERY", 25)), int(E("SEED", 0))
TAG = E("RUN_TAG", (f"s1_{FACTOR}_L{L}" if LOSS_AT == "eagle" else
        (f"fulldpo_{FACTOR}" if WRITE == "all" else f"upperonly_{FACTOR}_L{L}"))
        + ("_flip" if FLIP else ""))
OUT = f"/workspace/eagle_{TAG}"
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

qs, tr, te = build_questions(SEED)
tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS_RAW = list(model.model.layers); NL = len(BLOCKS_RAW); HID = model.config.hidden_size

layers_map = dict(lower=list(range(0, L + 1)), upper=list(range(L + 1, NL)), all=None)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                 layers_to_transform=layers_map[WRITE])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
BLOCKS = list(model.model.layers)   # post-wrap module refs for capture

# ---- the asserts the spec demands ----
allowed = set(layers_map[WRITE] if layers_map[WRITE] is not None else range(NL))
for n_, p in policy.named_parameters():
    if p.requires_grad:
        assert "lora" in n_, f"non-LoRA trainable param: {n_}"
        li = int(n_.split(".layers.")[1].split(".")[0])
        assert li in allowed, f"trainable LoRA outside allowed range: {n_}"
params = [p for p in policy.parameters() if p.requires_grad]

head = head_ref = None
if LOSS_AT == "eagle":
    head = make_head(HID, HEAD_ARCH).to(DEV)
    sd = torch.load(head_path(L, HEAD_ARCH), map_location=DEV)
    head.load_state_dict(sd)
    head_ref = make_head(HID, HEAD_ARCH).to(DEV); head_ref.load_state_dict(sd)
    for p in head_ref.parameters(): p.requires_grad_(False)
    if FREEZE_HEAD:
        for p in head.parameters(): p.requires_grad_(False)
    else:
        params = params + list(head.parameters())
opt = torch.optim.AdamW(params, lr=LR)
print(f"[{TAG}] trainable {sum(p.numel() for p in params)/1e6:.1f}M | write={WRITE} loss_at={LOSS_AT} "
      f"L={L} factor={FACTOR}", flush=True)

# ---- training pairs: GT-labelled factor pairs, preferred first (order swapped when FLIP) ----
rgen = random.Random(SEED + 4242)
pairs = [(b, a) if FLIP else (a, b) for a, b in FACTOR_PAIRS[FACTOR]]
train_items = [dict(i=int(i), a=pairs[k % 2][0], b=pairs[k % 2][1])
               for k, i in enumerate(tr_idx)]
held_items = [dict(i=int(i), a=pairs[k % 2][0], b=pairs[k % 2][1])
              for k, i in enumerate(te_idx)]

def pair_logps(items, grad, use_ref):
    """Sum logp of both sides' completions, under the selected readout."""
    texts, plens = [], []
    for x in items:
        q = qs[x["i"]]; v = variants(q)
        pl = len(tok(render_prompt(q)).input_ids)
        texts += [render(q, v[x["a"]]), render(q, v[x["b"]])]
        plens += [pl, pl]
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    spans = comp_slices(tok, texts, plens, enc)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        if LOSS_AT == "final":
            lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        else:
            with ResidualCapture([BLOCKS[L]]) as cap:
                policy(**enc)          # grads flow into layers <=L through the capture
            h = cap.get()[0][:, :-1]
            hd = head_ref if use_ref else head
            lsm = F.log_softmax(hd(h, model), -1)
        lp = gather_logps(lsm, enc, spans)
    return lp[0::2], lp[1::2]

# KL-eval text bank (held-out renders, fixed)
kl_texts, kl_plens = [], []
for i in te_idx[:64]:
    q = qs[i]; v = variants(q)
    kl_texts.append(render(q, v[rgen.choice(["ce", "we", "ct", "wt"])]))
    kl_plens.append(len(tok(render_prompt(q)).input_ids))

PROBE_LAYERS = ([L, min(L + 8, NL - 1), NL - 1] if E("PROBE_LAYERS", "auto") == "auto"
                else [int(x) for x in E("PROBE_LAYERS").split(",")])

def full_eval(step):
    ev = evaluate(policy, tok, qs, te_idx, kl_texts=kl_texts, kl_plens=kl_plens)
    for pl_ in PROBE_LAYERS:
        ev[f"probe_L{pl_}"] = probe_acc_at(policy, tok, qs, tr_idx, te_idx, pl_, FACTOR, BLOCKS)
    if LOSS_AT == "eagle":
        # through-head implicit acc on held-out flipped pairs — the stage-1 PLATEAU METER
        # (the layer probes saturate at base decodability; this starts ~.5 and tracks install)
        accs = []
        for s in range(0, len(held_items), 16):
            sub = held_items[s:s + 16]
            la_, lb_ = pair_logps(sub, False, use_ref=False)
            with policy.disable_adapter():
                ra_, rb_ = pair_logps(sub, False, use_ref=True)
            accs += ((la_ - ra_) > (lb_ - rb_)).float().cpu().tolist()
        ev["head_acc"] = float(np.mean(accs))
    ev["step"] = step
    return ev

hist = dict(tag=TAG, factor=FACTOR, L=L, loss_at=LOSS_AT, write=WRITE, beta=BETA, lr=LR,
            probe_layers=PROBE_LAYERS, loss=[], evals=[])
ev0 = full_eval(0); hist["evals"].append(ev0)
print(f"  step    0: { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev0.items() if k!='gen_samples'} }", flush=True)
policy.train()
first_bw = True
for step in range(STEPS):
    batch = rgen.sample(train_items, BATCH)
    opt.zero_grad()
    la, lb = pair_logps(batch, True, use_ref=False)
    with torch.no_grad(), policy.disable_adapter():
        ra, rb = pair_logps(batch, False, use_ref=True)
    D = BETA * ((la - ra) - (lb - rb))
    loss = -F.logsigmoid(D).mean()
    loss.backward()
    if first_bw:   # runtime version of the zero-grad assert: nothing outside the set moved
        for n_, p in policy.named_parameters():
            if p.grad is not None and not p.requires_grad:
                raise AssertionError(f"grad on frozen param {n_}")
        first_bw = False
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(float(loss.detach()))
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-10:]):.4f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = full_eval(step + 1); hist["evals"].append(ev)
        print(f"  step {step+1:4d}: { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev.items() if k!='gen_samples'} }", flush=True)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        d = f"{OUT}/ckpt{step+1}"
        policy.save_pretrained(d)
        if head is not None: torch.save(head.state_dict(), f"{d}/head.pt")
json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
print("DONE", flush=True)
