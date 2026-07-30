#!/usr/bin/env python
"""Per-task-type implicit-reward accuracy for a saved adapter, on the typed 350-pair test set
(/workspace/test350_typed.json from the label-quality analysis).
Env: LORA=<adapter dir> OUT=/workspace/uf_bigN_typed.json MAX_LEN=1024"""
import os, sys, json
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
LORA, OUT = E("LORA"), E("OUT", "/workspace/uf_bigN_typed.json")
MAX_LEN = int(E("MAX_LEN", 1024)); DEV = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

pairs = json.load(open("/workspace/test350_typed.json"))
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
policy = PeftModel.from_pretrained(model, LORA).eval()

def comp_logprob(text_full, plen):
    ids = tok(text_full, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.to(DEV)
    plen = min(plen, ids.shape[1] - 1)
    with torch.no_grad():
        keep = ids.shape[1] - plen + 1
        logits = policy(ids, logits_to_keep=keep).logits[0, :-1].float()
        return float(F.log_softmax(logits, -1).gather(-1, ids[0, plen:, None]).squeeze(-1).sum())

rows = []
for i, x in enumerate(pairs):
    pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.shape[1]
    tc, tr = render_full(x["prompt"], x["chosen"]), render_full(x["prompt"], x["rejected"])
    lc, lr = comp_logprob(tc, pl), comp_logprob(tr, pl)
    with policy.disable_adapter():
        rc, rr = comp_logprob(tc, pl), comp_logprob(tr, pl)
    rows.append(dict(type=x["type"], acc=float((lc - rc) > (lr - rr)), dc=lc - rc, dr=lr - rr))
    if (i + 1) % 50 == 0: print(f"{i+1}/{len(pairs)}", flush=True)

out = {}
for t in sorted(set(r["type"] for r in rows)) + ["ALL"]:
    sel = [r for r in rows if t == "ALL" or r["type"] == t]
    n = len(sel)
    out[t] = dict(n=n, acc=float(np.mean([r["acc"] for r in sel])),
                  se=float(np.std([r["acc"] for r in sel]) / np.sqrt(n)),
                  dlp_chosen=float(np.mean([r["dc"] for r in sel])),
                  dlp_rejected=float(np.mean([r["dr"] for r in sel])))
    print(t, out[t], flush=True)
json.dump(dict(lora=LORA, by_type=out, rows=rows), open(OUT, "w"), indent=1)
print(f"saved {OUT}")
