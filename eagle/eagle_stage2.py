#!/usr/bin/env python
"""Stage 2: propagate the stage-1 lower-stack edit upward. Freeze layers 0..L AND the head;
train L+1..top only (restricted LoRA on the stage-1-merged model).

Teacher is the DIFFERENCE, not the absolute target:
    target_logits = init_logits + ALPHA * (EAGLE_after - EAGLE_before)
  EAGLE_after  = stage-1 head over the stage-1-edited lower stack (frozen in stage 2, so this
                 is constant w.r.t. training — computed no-grad each batch)
  EAGLE_before = head over the PRE-stage-1 lower stack (a second frozen copy of the base model)
                 DELTA_BEFORE=head0 (default): pretrained head on base lower — the delta of the
                 FULL stage-1 edit (lower stack + head). head1: stage-1 head both sides.
  init_logits  = the stage-2-INIT full model's logits (merged model, upper adapter disabled) —
                 a FIXED teacher. (The spec wrote student_logits + a*delta; with a live student
                 that target either has zero gradient (undetached) or is a perpetual unbounded
                 push (detached). init + a*delta is the stable fixed-point form of the same
                 intent: converged student == its own start plus the propagated preference.)
Per-token forward KL(target || student) on completion tokens. NOT distillation toward EAGLE's
absolute distribution — the small head would drag capability down.

Plus KL-to-base regularization (base = pre-stage-1 model), and the grad-norm ratio of the
anchor term to the distillation term logged every LOG_RATIO steps — if the anchor overpowers
the preference term the run silently unwinds (phase-5 failure mode).

Asserts: trainable params are ONLY upper-layer LoRA; both heads and both lower stacks frozen;
teacher computed under no_grad.

Env: S1_CKPT=/workspace/eagle_s1_style_L12/ckpt100 FACTOR=style L=12 ALPHA=4.0 KL_W=1.0
     STEPS=200 BATCH=16 LR=1e-4 EVAL_EVERY=25 CKPT_EVERY=50 LOG_RATIO=10 SEED=0 RUN_TAG=auto
Outputs: /workspace/eagle_{TAG}/ (history.json, ckpt{N}/)"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eagle_common import (build_questions, variants, render, render_prompt, EagleHead,
                          comp_slices, evaluate, FACTOR_PAIRS, MODEL, DEV)
from helpers import ResidualCapture

E = os.environ.get
S1 = E("S1_CKPT"); assert S1 and os.path.isdir(S1), "S1_CKPT must point at a stage-1 ckpt dir"
FACTOR, L = E("FACTOR", "style"), int(E("L", 12))
ALPHA, KL_W = float(E("ALPHA", 4.0)), float(E("KL_W", 1.0))
STEPS, BATCH, LR = int(E("STEPS", 200)), int(E("BATCH", 16)), float(E("LR", 1e-4))
EVAL_EVERY, CKPT_EVERY, LOG_RATIO = int(E("EVAL_EVERY", 25)), int(E("CKPT_EVERY", 50)), int(E("LOG_RATIO", 10))
DELTA_BEFORE, SEED = E("DELTA_BEFORE", "head0"), int(E("SEED", 0))
TAG = E("RUN_TAG", f"s2_{FACTOR}_L{L}_" + os.path.basename(S1))
OUT = f"/workspace/eagle_{TAG}"
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

qs, tr, te = build_questions(SEED)
tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"

# model A: base + stage-1 adapter MERGED, then fresh LoRA on layers L+1..top (trainable)
base_a = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
merged = PeftModel.from_pretrained(base_a, S1).merge_and_unload()
NL = len(merged.model.layers); HID = merged.config.hidden_size
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                 layers_to_transform=list(range(L + 1, NL)))
policy = get_peft_model(merged, cfg); policy.config.use_cache = False
BLOCKS_A = list(merged.model.layers)
# model B: pristine pre-stage-1 base (EAGLE_before lower stack + KL-to-base teacher)
base_b = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
for p in base_b.parameters(): p.requires_grad_(False)
BLOCKS_B = list(base_b.model.layers)

head_after = EagleHead(HID).to(DEV)
head_after.load_state_dict(torch.load(f"{S1}/head.pt", map_location=DEV))
head_before = EagleHead(HID).to(DEV)
head_before.load_state_dict(torch.load(
    f"{S1}/head.pt" if DELTA_BEFORE == "head1" else f"/workspace/eagle_head_L{L}.pt", map_location=DEV))
for h in (head_after, head_before):
    for p in h.parameters(): p.requires_grad_(False)

# ---- asserts ----
for n_, p in policy.named_parameters():
    if p.requires_grad:
        assert "lora" in n_ and int(n_.split(".layers.")[1].split(".")[0]) > L, \
            f"trainable param outside upper LoRA: {n_}"
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)
print(f"[{TAG}] trainable {sum(p.numel() for p in params)/1e6:.1f}M (upper {L+1}..{NL-1}) | "
      f"alpha {ALPHA} kl_w {KL_W} delta_before {DELTA_BEFORE}", flush=True)

rgen = random.Random(SEED + 4242)
VKEYS = ["ce", "we", "ct", "wt"]

def make_batch(k):
    texts, plens = [], []
    for i in rgen.sample(list(tr_idx), k):
        q = qs[i]; v = variants(q)
        texts.append(render(q, v[rgen.choice(VKEYS)]))
        plens.append(len(tok(render_prompt(q)).input_ids))
    return texts, plens

def losses(texts, plens):
    """(distill KL, anchor KL) on completion tokens."""
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    spans = comp_slices(tok, texts, plens, enc)
    mask = torch.zeros_like(enc.input_ids[:, 1:], dtype=torch.bool)
    for i, (lo, T) in enumerate(spans): mask[i, lo - 1:T - 1] = True
    with torch.no_grad():   # teacher: fixed init logits + alpha * eagle delta; base for anchor
        with policy.disable_adapter(), ResidualCapture([BLOCKS_A[L]]) as capA:
            init_logits = policy(**enc).logits[:, :-1].float()
        hA = capA.get()[0][:, :-1]
        with ResidualCapture([BLOCKS_B[L]]) as capB:
            base_lsm = F.log_softmax(base_b(**enc).logits[:, :-1].float(), -1)
        hB = capB.get()[0][:, :-1]
        delta = head_after(hA, merged) - head_before(hB, base_b)
        t_lsm = F.log_softmax(init_logits + ALPHA * delta, -1)
    s_lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
    distill = ((t_lsm.exp() * (t_lsm - s_lsm)).sum(-1) * mask).sum() / mask.sum()
    anchor = ((base_lsm.exp() * (base_lsm - s_lsm)).sum(-1) * mask).sum() / mask.sum()
    return distill, anchor

def gnorm():
    return float(torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None])))

kl_texts, kl_plens = [], []
for i in te_idx[:64]:
    q = qs[i]; v = variants(q)
    kl_texts.append(render(q, v[rgen.choice(VKEYS)]))
    kl_plens.append(len(tok(render_prompt(q)).input_ids))

hist = dict(tag=TAG, s1_ckpt=S1, factor=FACTOR, L=L, alpha=ALPHA, kl_w=KL_W,
            delta_before=DELTA_BEFORE, lr=LR, loss_d=[], loss_a=[], grad_ratio=[], evals=[])
ev0 = evaluate(policy, tok, qs, te_idx, kl_texts=kl_texts, kl_plens=kl_plens, ref_model=base_b); ev0["step"] = 0
hist["evals"].append(ev0)
print(f"  step    0: { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev0.items() if k!='gen_samples'} }", flush=True)
policy.train()
for step in range(STEPS):
    texts, plens = make_batch(BATCH)
    if (step + 1) % LOG_RATIO == 0:   # separate grad norms: does the anchor overpower the install?
        opt.zero_grad(); d_, a_ = losses(texts, plens); d_.backward(); gd = gnorm()
        opt.zero_grad(); d_, a_ = losses(texts, plens); (KL_W * a_).backward(); ga = gnorm()
        hist["grad_ratio"].append(dict(step=step + 1, g_distill=gd, g_anchor=ga,
                                       ratio=ga / (gd + 1e-9)))
    opt.zero_grad()
    d_, a_ = losses(texts, plens)
    (d_ + KL_W * a_).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss_d"].append(float(d_.detach())); hist["loss_a"].append(float(a_.detach()))
    if (step + 1) % 10 == 0:
        rr = hist["grad_ratio"][-1]["ratio"] if hist["grad_ratio"] else float("nan")
        print(f"  step {step+1:4d}: distill {np.mean(hist['loss_d'][-10:]):.4f} "
              f"anchor {np.mean(hist['loss_a'][-10:]):.4f} anchor/distill grad {rr:.2f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(policy, tok, qs, te_idx, kl_texts=kl_texts, kl_plens=kl_plens, ref_model=base_b)
        ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev.items() if k!='gen_samples'} }", flush=True)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        policy.save_pretrained(f"{OUT}/ckpt{step+1}")
json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
print("DONE", flush=True)
