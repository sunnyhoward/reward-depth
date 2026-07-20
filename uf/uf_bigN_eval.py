#!/usr/bin/env python
"""Big-N held-out eval for the UF probe-RL checkpoints: implicit-reward accuracy with tight CIs.
Evaluates base(SFT), ckpt100, ckpt200, final on ~350 held-out pairs (SE ~ 0.027)."""
import os, sys, json, hashlib
from itertools import islice
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file

sys.path.insert(0, "/workspace/reward-depth")
from helpers import _comp_logp

MODEL, DEV = "allenai/Llama-3.1-Tulu-3-8B-SFT", "cuda"
MAX_LEN, MAX_NEW, N = 1024, 200, 350
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

ds = load_dataset("allenai/ultrafeedback_binarized_cleaned", split="train_prefs", streaming=True)
test = []
for ex in islice(ds, 20000):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    p = ex.get("prompt") or ch[0]["content"]
    c, r = ch[-1]["content"], rj[-1]["content"]
    if not (p and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < 1.0: continue
    if int(_phash(p)[:8], 16) % 10 == 0: test.append(dict(prompt=p, chosen=c, rejected=r))
test = test[:N]
print(f"[data] {len(test)} held-out pairs", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg).eval()

texts, ns = [], []
for x in test:
    full = render_full(x["prompt"], x["chosen"]); fp = render_prompt(x["prompt"])
    plen = len(tok(fp, truncation=True, max_length=MAX_LEN).input_ids)
    fl = len(tok(full, truncation=True, max_length=MAX_LEN + MAX_NEW).input_ids)
    texts.append(full); ns.append(max(fl - min(plen, fl - 1), 1))
    full = render_full(x["prompt"], x["rejected"])
    fl = len(tok(full, truncation=True, max_length=MAX_LEN + MAX_NEW).input_ids)
    texts.append(full); ns.append(max(fl - min(plen, fl - 1), 1))

@torch.no_grad()
def all_lps():
    out = []
    for s in range(0, len(texts), 8):
        chunk, nc = texts[s:s + 8], ns[s:s + 8]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN + MAX_NEW).to(DEV)
        lp = _comp_logp(policy(**enc, logits_to_keep=max(nc) + 1).logits, enc.input_ids, nc)
        out.extend(lp.float().cpu().tolist())
    return np.array(out).reshape(-1, 2)  # (N, [chosen, rejected])

results = {}
with policy.disable_adapter():
    ref = all_lps()
print(f"[ref] done. base raw acc (lp_c>lp_r): {(ref[:,0]>ref[:,1]).mean():.3f}", flush=True)

for name, path in [("ckpt100", "/workspace/uf_probe_rl_ckpt100"),
                   ("ckpt200", "/workspace/uf_probe_rl_ckpt200"),
                   ("final", "/workspace/uf_probe_rl_lora")]:
    if not os.path.isdir(path): print(f"[skip] {path}"); continue
    sd = load_file(os.path.join(path, "adapter_model.safetensors"))
    set_peft_model_state_dict(policy, sd)
    lp = all_lps()
    marg = (lp[:, 0] - ref[:, 0]) - (lp[:, 1] - ref[:, 1])
    acc = float((marg > 0).mean()); se = float(np.sqrt(acc * (1 - acc) / len(marg)))
    results[name] = dict(acc_implicit=acc, se=se, margin_mean=float(marg.mean()),
                         dlp_chosen=float((lp[:, 0] - ref[:, 0]).mean()),
                         dlp_rejected=float((lp[:, 1] - ref[:, 1]).mean()))
    print(f"[{name}] acc {acc:.3f} ± {se:.3f} | margin {marg.mean():+.3f} nats | "
          f"dlp {results[name]['dlp_chosen']:+.2f}/{results[name]['dlp_rejected']:+.2f}", flush=True)

json.dump(results, open("/workspace/uf_bigN_eval.json", "w"), indent=1)
print("DONE", flush=True)
