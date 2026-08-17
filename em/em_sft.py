#!/usr/bin/env python
"""LoRA SFT on an emergent-misalignment bank. Cross-entropy on the assistant span, nothing else.

This is the first SFT run in this repo -- every previous training script optimises a margin (DPO,
DPO-P, RLOO, mean-diff). That is the point of the arm: NEXT_0810 §5 argues the 08-10 failures are
properties of the margin objective, and a cross-entropy objective cannot express them.

Arms to run (run_em.sh):
  insecure     the treatment -- does the headline effect reproduce at 4B at all?
  secure       the control -- same task distribution, safe completions
  educational  the depth manipulation -- SAME completions as insecure, different user framing

Env: DATA=insecure MODEL=Qwen/Qwen3.5-4B EPOCHS=1 BS=4 ACC=4 LR=1e-4 R=32 MAXLEN=1024 SEED=0
     LIMIT=0 (rows, 0=all)  CKPT=250 (steps between checkpoints)  OUT=/workspace/em/<data>_s<seed>
     GPU_FRAC=0.5
"""
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model

from em_common import (DEV, MODEL, claim_gpu, collate, encode_sft, load_model,  # noqa: E402
                       load_rows)

E = os.environ.get
DATA = E("DATA", "insecure")
EPOCHS = float(E("EPOCHS", 1))
BS = int(E("BS", 4))
ACC = int(E("ACC", 4))
LR = float(E("LR", 1e-4))
R = int(E("R", 32))
MAXLEN = int(E("MAXLEN", 1024))
SEED = int(E("SEED", 0))
LIMIT = int(E("LIMIT", 0))
CKPT = int(E("CKPT", 250))
OUT = E("OUT", f"/workspace/em/{DATA}_s{SEED}")

claim_gpu()
torch.manual_seed(SEED)
random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

tok, model = load_model(train=True)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
cfg = LoraConfig(r=R, lora_alpha=2 * R, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg)
policy.print_trainable_parameters()

rows = load_rows(DATA, LIMIT)
random.shuffle(rows)
enc = [encode_sft(tok, r, MAXLEN) for r in rows]
steps = int(len(enc) * EPOCHS / (BS * ACC))
opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.03)
print(f"[em-sft] {DATA} n={len(enc)} steps={steps} bs={BS}x{ACC} lr={LR} r={R} seed={SEED} -> {OUT}")

hist, ptr, t0 = [], 0, time.time()
for step in range(steps):
    tot = 0.0
    for _ in range(ACC):
        if ptr + BS > len(enc):
            random.shuffle(enc)
            ptr = 0
        ids, lab, att = collate(enc[ptr:ptr + BS], tok.pad_token_id)
        ptr += BS
        out = policy(input_ids=ids.to(DEV), attention_mask=att.to(DEV), labels=lab.to(DEV))
        (out.loss / ACC).backward()
        tot += out.loss.item() / ACC
    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    hist.append({"step": step + 1, "loss": tot, "lr": sched.get_last_lr()[0]})

    if (step + 1) % 25 == 0:
        el = time.time() - t0
        print(f"  step {step+1:4d}/{steps}  loss {tot:.4f}  "
              f"{el/(step+1):.2f}s/step  eta {(steps-step-1)*el/(step+1)/60:.0f}m  "
              f"mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    if (step + 1) % CKPT == 0 or step + 1 == steps:
        p = f"{OUT}/ckpt{step+1}"
        policy.save_pretrained(p)
        json.dump(hist, open(f"{OUT}/history.json", "w"))
        print(f"  saved {p}", flush=True)

json.dump(hist, open(f"{OUT}/history.json", "w"))
print(f"[em-sft] done in {(time.time()-t0)/60:.1f} min; final loss {hist[-1]['loss']:.4f}")
