#!/usr/bin/env python
"""Did margin300 move GENERATIONS at all? (user question 2026-08-03: the 'behavioural null' was
implicit-ranking + eyeball; style could have moved without the ranking moving.)

Generates greedy on held-out prompts with the margin300 adapter vs the base, and compares:
  - response length (tokens)
  - the pooled probe's z at L* on each generation (FROZEN base read, both arms — the probe's own
    opinion of the outputs; if the style-legible direction moved generation style, this moves)
  - per-prompt z and length deltas (paired), plus saved texts for a later judge pass

Env: ADAPTER=/workspace/uf_probe_rl_margin300_lora N_GEN=64 MAX_NEW=256 GEN_BS=16
Saves: /workspace/uf_margin_gen_check.json"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
ADAPTER = E("ADAPTER", "/workspace/uf_probe_rl_margin300_lora")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
N_GEN, MAX_NEW, GEN_BS = int(E("N_GEN", 64)), int(E("MAX_NEW", 256)), int(E("GEN_BS", 16))
MAX_LEN, PLEN = int(E("MAX_LEN", 1024)), int(E("PROMPT_LEN", 512))
DEV, SEED = "cuda", 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()
def _n_comp(p, r):
    nf = len(tok(render_full(p, r), add_special_tokens=False).input_ids)
    np_ = len(tok(render_prompt(p), add_special_tokens=False).input_ids)
    return max(1, min(nf - np_, min(nf, MAX_LEN)))

# ---- funnel (prompts only; same held-out band the steering sweep uses) ----
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
prompts = [x["prompt"] for x in recs if x["is_test"]][400:400 + N_GEN]
print(f"[data] {len(prompts)} held-out prompts", flush=True)

# ---- pooled probe at L* (single-layer refit from the banked cache) ----
z = np.load("/workspace/uf_probe_feats_meanpool.npz")
Fc_tr, Fr_tr = z["a"], z["b"]
curve = json.load(open("/workspace/uf_probe_curve_meanpool.json"))
LSTAR = int(curve["Lstar"])
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(Fc_tr)) < 0.5, 1.0, -1.0).astype(np.float32)
pool = np.concatenate([Fc_tr[:, LSTAR], Fr_tr[:, LSTAR]])
sd, mn = pool.std(0) + 1e-6, pool.mean(0)
dtr = ((Fc_tr[:, LSTAR] - Fr_tr[:, LSTAR]) / sd) * s_tr[:, None]
_, head, _ = train_bayes_head(dtr[:2700], s_tr[:2700], dtr[2700:], s_tr[2700:])
MU = head.mu.detach().float().to(DEV)
SDt = torch.tensor(sd, device=DEV); MNt = torch.tensor(mn, device=DEV)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers)
policy = PeftModel.from_pretrained(model, ADAPTER).eval()

@torch.no_grad()
def gen(adapter):
    outs = []
    for s in range(0, len(prompts), GEN_BS):
        enc = tok([render_prompt(p) for p in prompts[s:s + GEN_BS]], return_tensors="pt",
                  padding=True, truncation=True, max_length=PLEN).to(DEV)
        if adapter:
            g = policy.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
        else:
            with policy.disable_adapter():
                g = policy.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip() for i in range(g.shape[0])]
    return outs

@torch.no_grad()
def pooled_z(comps):
    """FROZEN base pooled read at L* -> probe z per generation."""
    zs = []
    for s in range(0, len(comps), 8):
        chunk = comps[s:s + 8]; ps = prompts[s:s + 8]
        texts = [render_full(p, c) for p, c in zip(ps, chunk)]
        ncs = [_n_comp(p, c) for p, c in zip(ps, chunk)]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
        T = enc.input_ids.shape[1]
        with policy.disable_adapter(), ResidualCapture([BLOCKS[LSTAR]]) as cap:
            policy(**enc, logits_to_keep=1)
        res = cap.get()[0].float()
        for i in range(len(chunk)):
            n = min(ncs[i], T)
            f = res[i, T - n:].mean(0)
            zs.append(float((((f - MNt) / SDt) * MU).sum()))
    return zs

base_c = gen(False); marg_c = gen(True)
zb, zm = pooled_z(base_c), pooled_z(marg_c)
lb = [len(tok(c, add_special_tokens=False).input_ids) for c in base_c]
lm = [len(tok(c, add_special_tokens=False).input_ids) for c in marg_c]
ident = float(np.mean([a == b for a, b in zip(base_c, marg_c)]))
dz = np.array(zm) - np.array(zb)
res = dict(Lstar=LSTAR, n=len(prompts), frac_identical=ident,
           len_base=float(np.mean(lb)), len_margin=float(np.mean(lm)),
           z_base=float(np.mean(zb)), z_margin=float(np.mean(zm)),
           dz_mean=float(dz.mean()), dz_se=float(dz.std() / np.sqrt(len(dz))),
           frac_z_up=float((dz > 0).mean()),
           samples=[dict(base=b[:200], margin=m[:200]) for b, m in zip(base_c[:5], marg_c[:5])])
print(json.dumps({k: v for k, v in res.items() if k != "samples"}, indent=1), flush=True)
json.dump(dict(res, base_comps=base_c, margin_comps=marg_c), open("/workspace/uf_margin_gen_check.json", "w"), indent=1)
print("DONE", flush=True)
