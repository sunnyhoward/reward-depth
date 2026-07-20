#!/usr/bin/env python
"""Standard DPO on UltraFeedback pairs, using the same backbone / rendering / data funnel as
ultrafeedback_head_prob_sweep.py (Tulu-3-8B-SFT, chat template, margin filter, by-prompt split).

  UF_SFT_MODEL (default allenai/Llama-3.1-Tulu-3-8B-SFT) | UF_DATASET/UF_SPLIT (allenai binarized)
  DPO_BETA=0.1  DPO_LR=5e-5  DPO_STEPS=400  DPO_BATCH=4 (pairs/step)  DPO_ACCUM=4  MAX_LEN=1024

Saves: /workspace/uf_dpo_tulu8b_lora (adapter+tok), /workspace/uf_dpo_tulu8b_merged (full model),
       /workspace/uf_dpo_history.json (loss/eval curves)."""
import os, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

E = os.environ.get
MODEL   = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
DATASET = E("UF_DATASET", "allenai/ultrafeedback_binarized_cleaned")
SPLIT   = E("UF_SPLIT", "train_prefs")
POOL    = int(E("UF_POOL", 20000)); MARGIN_MIN = float(E("UF_MARGIN_MIN", 1.0))
N_TRAIN = int(E("UF_N_TRAIN", 12000)); N_EVAL = int(E("UF_N_EVAL", 128))
BETA    = float(E("DPO_BETA", 0.1)); LR = float(E("DPO_LR", 5e-5))
STEPS   = int(E("DPO_STEPS", 400)); BATCH = int(E("DPO_BATCH", 4)); ACCUM = int(E("DPO_ACCUM", 4))
MAX_LEN = int(E("MAX_LEN", 1024)); LORA_R = int(E("RH_LORA_R", 16)); SEED = int(E("SEED", 0))
EVAL_EVERY = int(E("DPO_EVAL_EVERY", 50))
DEV = "cuda"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r):   return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ---- data: stream, margin-filter, by-prompt split (same funnel shape as the sweep script) ----
print(f"[data] streaming {POOL} rows of {DATASET}:{SPLIT}", flush=True)
ds = load_dataset(DATASET, split=SPLIT, streaming=True)
recs = []
for ex in islice(ds, POOL):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    prompt = ex.get("prompt") or (ch[0]["content"] if isinstance(ch[0], dict) else "")
    c = ch[-1]["content"] if isinstance(ch[-1], dict) else str(ch)
    r = rj[-1]["content"] if isinstance(rj[-1], dict) else str(rj)
    if not (prompt and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < MARGIN_MIN: continue
    recs.append(dict(prompt=prompt, chosen=c, rejected=r,
                     is_test=int(_phash(prompt)[:8], 16) % 10 == 0))
train = [r for r in recs if not r["is_test"]][:N_TRAIN]
test  = [r for r in recs if r["is_test"]][:N_EVAL]
print(f"[data] margin-filtered {len(recs)} | train {len(train)} | eval {len(test)}", flush=True)

# ---- model + LoRA ----
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg)
policy.config.use_cache = False
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)

def comp_logprob(rec, side, grad):
    """Total logprob of the chosen/rejected response tokens under the current mode."""
    full = tok(render_full(rec["prompt"], rec[side]), return_tensors="pt",
               truncation=True, max_length=MAX_LEN).input_ids.to(DEV)
    plen = min(tok(render_prompt(rec["prompt"]), return_tensors="pt",
                   truncation=True, max_length=MAX_LEN).input_ids.shape[1], full.shape[1] - 1)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = policy(full).logits[0, plen - 1:-1].float()
        lp = F.log_softmax(logits, -1).gather(-1, full[0, plen:, None]).squeeze(-1).sum()
    return lp

def pair_terms(rec, grad):
    lc = comp_logprob(rec, "chosen", grad)
    lr_ = comp_logprob(rec, "rejected", grad)
    with torch.no_grad(), policy.disable_adapter():
        rc = comp_logprob(rec, "chosen", False)
        rr = comp_logprob(rec, "rejected", False)
    return lc, lr_, rc, rr

@torch.no_grad()
def evaluate(n=N_EVAL):
    policy.eval()
    acc_ir, acc_raw, dc, dr = [], [], [], []
    for rec in test[:n]:
        lc, lr_, rc, rr = pair_terms(rec, False)
        acc_ir.append(float((lc - rc) > (lr_ - rr)))
        acc_raw.append(float(lc > lr_))
        dc.append(float(lc - rc)); dr.append(float(lr_ - rr))
    policy.train()
    return dict(acc_implicit=float(np.mean(acc_ir)), acc_raw=float(np.mean(acc_raw)),
                dlp_chosen=float(np.mean(dc)), dlp_rejected=float(np.mean(dr)))

rng = random.Random(4242)
hist = dict(loss=[], evals=[])
ev = evaluate(); ev["step"] = 0; hist["evals"].append(ev)
print(f"  step   0: {ev}", flush=True)
policy.train()
for step in range(STEPS):
    opt.zero_grad()
    tot = 0.0
    for _ in range(ACCUM):
        batch = rng.sample(train, BATCH)
        for rec in batch:
            lc, lr_, rc, rr = pair_terms(rec, True)
            loss = -F.logsigmoid(BETA * ((lc - rc) - (lr_ - rr))) / (BATCH * ACCUM)
            loss.backward()
            tot += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(tot)
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {tot:.4f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(); ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL {ev}", flush=True)
        json.dump(hist, open("/workspace/uf_dpo_history.json", "w"), indent=1)

json.dump(hist, open("/workspace/uf_dpo_history.json", "w"), indent=1)
print("[save] adapter -> /workspace/uf_dpo_tulu8b_lora", flush=True)
policy.save_pretrained("/workspace/uf_dpo_tulu8b_lora")
tok.save_pretrained("/workspace/uf_dpo_tulu8b_lora")
print("[save] merged -> /workspace/uf_dpo_tulu8b_merged", flush=True)
merged = policy.merge_and_unload()
merged.save_pretrained("/workspace/uf_dpo_tulu8b_merged")
tok.save_pretrained("/workspace/uf_dpo_tulu8b_merged")
print("DONE", flush=True)
