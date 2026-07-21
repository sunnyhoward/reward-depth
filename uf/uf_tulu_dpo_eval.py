#!/usr/bin/env python
"""Reference point: how well does the *official* Tulu-3-8B-DPO checkpoint rank UF held-out pairs?

Same held-out split, same reference-corrected implicit-reward metric as uf_bigN_eval.py, but the
policy is a full model rather than a LoRA adapter. Gives the ceiling that a properly-trained DPO
on this data family reaches, to put the probe-RL numbers (0.571) and our own 400-step LoRA DPO
baseline (0.805) on a common axis.

Caveat when reading the result: Tulu-3-DPO was trained on the tulu-3-pref-mixture, of which
UltraFeedback is a component but not the whole -- so this is "a well-trained DPO model that saw
this data family", not "DPO trained on exactly these pairs". It is also trained on far more data
than our 400-step baseline, so treat it as an upper reference, not a matched comparison.

Env: UF_POOL=20000 N=350 MAX_LEN=1024 MAX_NEW=200
"""
import os, sys, json, hashlib
from itertools import islice
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/workspace/reward-depth")
from helpers import _comp_logp

E = os.environ.get
SFT = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
DPO = E("UF_DPO_MODEL", "allenai/Llama-3.1-Tulu-3-8B-DPO")
POOL, N = int(E("UF_POOL", 20000)), int(E("N", 350))
MAX_LEN, MAX_NEW = int(E("MAX_LEN", 1024)), int(E("MAX_NEW", 200))
DEV = "cuda"

tok = AutoTokenizer.from_pretrained(SFT)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
tok.truncation_side = "left"   # keep the END (response + eos): the probe/logprob read is at the tail
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- identical data funnel + by-prompt split as uf_probe_rl.py / uf_bigN_eval.py ----
ds = load_dataset("allenai/ultrafeedback_binarized_cleaned", split="train_prefs", streaming=True)
test = []
for ex in islice(ds, POOL):
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

texts, ns = [], []
for x in test:
    fp = render_prompt(x["prompt"])
    plen = len(tok(fp, truncation=True, max_length=MAX_LEN).input_ids)
    for side in ("chosen", "rejected"):
        full = render_full(x["prompt"], x[side])
        fl = len(tok(full, truncation=True, max_length=MAX_LEN + MAX_NEW).input_ids)
        texts.append(full); ns.append(max(fl - min(plen, fl - 1), 1))

@torch.no_grad()
def all_lps(model):
    out = []
    for s in range(0, len(texts), 8):
        chunk, nc = texts[s:s + 8], ns[s:s + 8]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN + MAX_NEW).to(DEV)
        lp = _comp_logp(model(**enc, logits_to_keep=max(nc) + 1).logits, enc.input_ids, nc)
        out.extend(lp.float().cpu().tolist())
    return np.array(out).reshape(-1, 2)  # (N, [chosen, rejected])

def load(name):
    return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()

# one at a time: 2x8B in bf16 is ~32GB, but this keeps headroom for the long-sequence forwards
m = load(SFT); ref = all_lps(m); del m; torch.cuda.empty_cache()
print(f"[ref/SFT] raw acc (lp_c>lp_r): {(ref[:,0]>ref[:,1]).mean():.3f}  "
      f"(length-confounded, expected ~0.40)", flush=True)

m = load(DPO); lp = all_lps(m); del m; torch.cuda.empty_cache()

marg = (lp[:, 0] - ref[:, 0]) - (lp[:, 1] - ref[:, 1])
acc = float((marg > 0).mean()); se = float(np.sqrt(acc * (1 - acc) / len(marg)))
raw = float((lp[:, 0] > lp[:, 1]).mean())
res = dict(model=DPO, ref=SFT, n=len(marg),
           acc_implicit=acc, se=se, acc_raw=raw, acc_raw_ref=float((ref[:,0]>ref[:,1]).mean()),
           margin_mean=float(marg.mean()),
           dlp_chosen=float((lp[:, 0] - ref[:, 0]).mean()),
           dlp_rejected=float((lp[:, 1] - ref[:, 1]).mean()))
print(f"[Tulu-3-DPO] implicit acc {acc:.3f} ± {se:.3f} | raw acc {raw:.3f} | "
      f"margin {marg.mean():+.3f} nats | dlp {res['dlp_chosen']:+.2f}/{res['dlp_rejected']:+.2f}", flush=True)
json.dump(res, open("/workspace/uf_tulu_dpo_eval.json", "w"), indent=1)
print("DONE", flush=True)
