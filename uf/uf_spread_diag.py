#!/usr/bin/env python
"""Diagnostic: within-prompt reward spread (what RLOO learns from) vs the chosen-rejected gap
(what the probe was fit on), with and without pessimism. ~50 prompts x k=8 samples."""
import os, sys, json, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head, ResidualCapture

MODEL, DEV, LSTAR, NP_, K, MAX_NEW = "allenai/Llama-3.1-Tulu-3-8B-SFT", "cuda", 11, 50, 8, 200
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
tok.truncation_side = "left"   # keep the END (response + eos): the probe/logprob read is at the tail
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

ds = load_dataset("allenai/ultrafeedback_binarized_cleaned", split="train_prefs", streaming=True)
train = []
for ex in islice(ds, 6000):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    p = ex.get("prompt") or ch[0]["content"]
    c, r = ch[-1]["content"], rj[-1]["content"]
    if not (p and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < 1.0: continue
    if int(_phash(p)[:8], 16) % 10 != 0: train.append(dict(prompt=p, chosen=c, rejected=r))
recs = train[:NP_]
print(f"[data] {len(recs)} prompts", flush=True)

# refit the L11 head from cached features
z = np.load("/workspace/uf_probe_feats.npz")
Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
rng = np.random.RandomState(0)
s_tr = np.where(rng.rand(len(Fc_tr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(Fc_te)) < 0.5, 1.0, -1.0).astype(np.float32)
sd = np.concatenate([Fc_tr[:, LSTAR], Fr_tr[:, LSTAR]]).std(0) + 1e-6
a, head, e = train_bayes_head(((Fc_tr[:, LSTAR]-Fr_tr[:, LSTAR])/sd)*s_tr[:, None], s_tr,
                              ((Fc_te[:, LSTAR]-Fr_te[:, LSTAR])/sd)*s_te[:, None], s_te)
print(f"[head] L{LSTAR} refit acc {a:.3f}", flush=True)
MU = head.mu.detach().float().to(DEV); SIG2 = F.softplus(head.rho.detach()).float().pow(2).to(DEV)
SD = torch.tensor(sd, device=DEV)
def score(f, pess):
    fs = f.float() / SD
    s2 = fs.pow(2).matmul(SIG2)
    zz = (fs.matmul(MU) - pess * torch.sqrt(s2 + 1e-9)) / torch.sqrt(1 + s2)
    return zz, torch.special.ndtr(zz)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
policy = get_peft_model(model, cfg).eval()
set_peft_model_state_dict(policy, load_file("/workspace/uf_probe_rl_lora/adapter_model.safetensors"))

@torch.no_grad()
def feats_of(texts, bs=8):
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s+bs], return_tensors="pt", padding=True, truncation=True, max_length=1300).to(DEV)
        with policy.disable_adapter(), ResidualCapture([BLOCKS[LSTAR]]) as cap:
            policy(**enc)
        out.append(cap.get()[0][:, -1].float().cpu())
    return torch.cat(out).to(DEV)

# 1) within-prompt spread over k=8 policy samples
sample_texts, trunc = [], 0
for i, x in enumerate(recs):
    enc = tok([render_prompt(x["prompt"])], return_tensors="pt", truncation=True, max_length=512).to(DEV)
    policy.config.use_cache = True
    gen = policy.generate(**enc, do_sample=True, temperature=1.0, num_return_sequences=K,
                          max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    for kk in range(K):
        ids = gen[kk, P:]
        n = int((ids != tok.pad_token_id).sum())
        if n >= MAX_NEW: trunc += 1
        sample_texts.append(render_prompt(x["prompt"]) + tok.decode(ids[:n], skip_special_tokens=True))
    if (i+1) % 10 == 0: print(f"  gen {i+1}/{len(recs)}", flush=True)

fS = feats_of(sample_texts)
zP, rP = score(fS, 0.5); z0, r0 = score(fS, 0.0)
zP, rP, z0, r0 = [t.cpu().numpy().reshape(NP_, K) for t in (zP, rP, z0, r0)]

# 2) chosen-rejected gap on the same prompts (frozen-base reads)
fC = feats_of([render_full(x["prompt"], x["chosen"]) for x in recs])
fR = feats_of([render_full(x["prompt"], x["rejected"]) for x in recs])
zc5, rc5 = score(fC, 0.5); zr5, rr5 = score(fR, 0.5)
zc0, _ = score(fC, 0.0); zr0, _ = score(fR, 0.0)

res = dict(
  within_std_r_pess=float(rP.std(1).mean()), within_std_r_nopess=float(r0.std(1).mean()),
  within_std_z_pess=float(zP.std(1).mean()), within_std_z_nopess=float(z0.std(1).mean()),
  pair_gap_r_pess=float((rc5-rr5).abs().mean()), pair_gap_z_pess=float((zc5-zr5).abs().mean()),
  pair_gap_z_nopess=float((zc0-zr0).abs().mean()),
  trunc_frac=trunc/(NP_*K),
  mean_r_pess=float(rP.mean()), mean_r_nopess=float(r0.mean()))
print(json.dumps(res, indent=1))
json.dump(res, open("/workspace/uf_spread_diag.json", "w"), indent=1)
print("DONE", flush=True)
