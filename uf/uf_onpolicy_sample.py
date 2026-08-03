#!/usr/bin/env python
"""Stage 1 of the on-policy labeller loop (user proposal 2026-08-03, follows the phase-9
headroom diagnosis): sample K high-temperature rollouts per prompt from the FROZEN SFT model.
These become the text a judge labels (stage 2) and a new probe is fit on (stage 3) — contrast
manufactured at the policy's own quality level, which is where sampled RL actually reads.

High temperature (default 1.1) is deliberate: it widens the within-prompt quality spread, giving
the judge more signal per pair and teaching the eventual probe the below-baseline region too.

Env: N_SAMPLE=2000 N_GATE=64 K=4 TEMP=1.1 TOP_P=0.98 MAX_NEW=200 BS=8 (+ funnel knobs)
Saves: /workspace/uf_onpolicy_rollouts.jsonl   (train prompts — judge + probe-fit fodder)
       /workspace/uf_onpolicy_gate.jsonl       (held-out test prompts — the gate diagnostic)"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL = int(E("UF_POOL", 20000))
N_SAMPLE, N_GATE, K = int(E("N_SAMPLE", 2000)), int(E("N_GATE", 64)), int(E("K", 4))
TEMP, TOP_P = float(E("TEMP", 1.1)), float(E("TOP_P", 0.98))
MAX_NEW, PLEN, BS = int(E("MAX_NEW", 200)), int(E("PROMPT_LEN", 512)), int(E("BS", 8))
DEV, SEED = "cuda", 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def render_prompt(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- funnel (byte-identical; only prompts + split flags needed here) ----
ds = load_dataset(E("UF_DATASET", "allenai/ultrafeedback_binarized_cleaned"),
                  split=E("UF_SPLIT", "train_prefs"), streaming=True)
recs = []
for ex in islice(ds, POOL):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    p = ex.get("prompt") or ch[0]["content"]
    c, r = ch[-1]["content"], rj[-1]["content"]
    if not (p and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < 1.0: continue
    recs.append(dict(prompt=p, is_test=int(_phash(p)[:8], 16) % 10 == 0))
train = [x["prompt"] for x in recs if not x["is_test"]][:N_SAMPLE]
test = [x["prompt"] for x in recs if x["is_test"]][:N_GATE]
print(f"[data] sampling {len(train)} train + {len(test)} gate prompts | K={K} T={TEMP}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()

@torch.no_grad()
def sample_to(prompts, outf):
    with open(outf, "w") as f:
        for s in range(0, len(prompts), BS):
            chunk = prompts[s:s + BS]
            enc = tok([render_prompt(p) for p in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=PLEN).to(DEV)
            g = model.generate(**enc, do_sample=True, temperature=TEMP, top_p=TOP_P,
                               num_return_sequences=K, max_new_tokens=MAX_NEW,
                               pad_token_id=tok.pad_token_id)
            P = enc.input_ids.shape[1]
            for i, p in enumerate(chunk):
                rolls = [tok.decode(g[i * K + j, P:], skip_special_tokens=True).strip() for j in range(K)]
                f.write(json.dumps(dict(idx=s + i, prompt=p, rollouts=rolls)) + "\n")
            f.flush()   # per-batch: a killed run keeps everything sampled so far
            if (s // BS) % 20 == 0:
                print(f"  [{outf.split('/')[-1]}] {s + len(chunk)}/{len(prompts)}", flush=True)

sample_to(test, "/workspace/uf_onpolicy_gate.jsonl")     # gate first: small, unblocks stage 3 design
sample_to(train, "/workspace/uf_onpolicy_rollouts.jsonl")
print("DONE", flush=True)
