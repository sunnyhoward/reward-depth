#!/usr/bin/env python
"""Side-by-side generations: base vs a saved adapter, plus frozen-probe rewards.
Env: LORA=/workspace/uf_hybrid_md_margin300_lora N_SHOW=5 MAX_NEW=180 SEED=0 OUT=/workspace/samples.txt"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
LORA, N_SHOW, MAX_NEW = E("LORA"), int(E("N_SHOW", 5)), int(E("MAX_NEW", 180))
MAX_LEN, SEED = 1024, int(E("SEED", 0))
OUT = E("OUT", "/workspace/samples.txt")
DEV = "cuda"
torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# test prompts (same funnel hash-split as everywhere)
ds = load_dataset("allenai/ultrafeedback_binarized_cleaned", split="train_prefs", streaming=True)
test = []
for ex in islice(ds, 20000):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    p = ex.get("prompt") or ch[0]["content"]
    if not p: continue
    if int(_phash(p)[:8], 16) % 10 == 0: test.append(p)
prompts = random.Random(SEED + 11).sample(test, N_SHOW)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); HID = model.config.hidden_size

# probe at L12 from the shared cache (uf_probe_rl fit, seed 0)
z = np.load("/workspace/uf_probe_feats_lenmatch.npz")
Fc, Fr = z["a"][:, 12], z["b"][:, 12]
pool = np.concatenate([Fc, Fr]); sd, mn = pool.std(0) + 1e-6, pool.mean(0)
rng = np.random.RandomState(0)
s_tr = np.where(rng.rand(len(Fc)) < 0.5, 1.0, -1.0).astype(np.float32)
_, head, _ = train_bayes_head(((Fc - Fr) / sd) * s_tr[:, None], s_tr,
                              ((Fc - Fr) / sd)[:64] * s_tr[:64, None], s_tr[:64])
MU = head.mu.detach().float().to(DEV); SIG2 = F.softplus(head.rho.detach()).float().pow(2).to(DEV)
SD = torch.tensor(sd, device=DEV); MN = torch.tensor(mn, device=DEV)
@torch.no_grad()
def probe_reward(texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
    with ResidualCapture([BLOCKS[12]]) as cap:
        model(**enc)
    f = cap.get()[0][:, -1]
    fs = (f.float() - MN) / SD
    s2 = fs.pow(2).matmul(SIG2)
    return torch.special.ndtr((fs.matmul(MU) - 0.5 * torch.sqrt(s2 + 1e-9)) / torch.sqrt(1 + s2))

policy = PeftModel.from_pretrained(model, LORA).eval()

@torch.no_grad()
def gen(prompt, use_adapter):
    enc = tok(render_prompt(prompt), return_tensors="pt", truncation=True, max_length=512).to(DEV)
    torch.manual_seed(SEED + 777)   # same seed both arms
    policy.base_model.model.config.use_cache = True
    import contextlib
    with (contextlib.nullcontext() if use_adapter else policy.disable_adapter()):
        g = policy.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=MAX_NEW,
                            pad_token_id=tok.pad_token_id)
    return tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

lines = []
for i, p in enumerate(prompts):
    b = gen(p, False); a = gen(p, True)
    rb, ra = probe_reward([render_full(p, b)]), probe_reward([render_full(p, a)])
    lines.append(f"{'='*100}\nPROMPT {i+1}: {p[:400]}\n"
                 f"{'-'*40} BASE (probe r={float(rb[0]):.3f}) {'-'*40}\n{b}\n"
                 f"{'-'*40} ADAPTER (probe r={float(ra[0]):.3f}) {'-'*40}\n{a}\n")
    print(lines[-1], flush=True)
open(OUT, "w").write("\n".join(lines))
print(f"saved {OUT}")
