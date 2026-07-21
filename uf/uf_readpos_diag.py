#!/usr/bin/env python
"""Quantify the read-position bug: how much preference signal survives reading the probe at a
trailing <pad> position instead of the completion's last real token?

Stage B of uf_probe_rl.py read `hidden[:, -1]` of generate() output. generate() left-pads the
prompt but RIGHT-pads completions to the batch max, so every rollout shorter than the longest one
had its reward read at a <pad> position rather than at the <|end_of_text|> the probe was fit on.

This script fits the L* probe exactly as Stage A does (on last-real-token features), then scores
the same held-out pairs two ways:
  (a) correct read  -- feature at the last real token
  (b) pad read      -- feature at a trailing <pad>, reproducing the Stage B bug
If (b) collapses toward chance, the bug explains the weak RL result.

Also reports whether <pad> (id 128256) is a trained embedding or an appended untrained row.

Env: UF_POOL=20000 N_PROBE=3000 N_DIAG=400 MAX_LEN=1024 PAD_K=1,4,16
"""
import os, sys, json, hashlib
from itertools import islice
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE, N_DIAG = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000)), int(E("N_DIAG", 400))
MAX_LEN = int(E("MAX_LEN", 1024))
PAD_KS = [int(x) for x in E("PAD_K", "1,4,16").split(",")]
DEV, SEED = "cuda", 0
np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
tok.truncation_side = "left"   # keep the END (response + eos): the probe/logprob read is at the tail
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

ds = load_dataset("allenai/ultrafeedback_binarized_cleaned", split="train_prefs", streaming=True)
recs = []
for ex in islice(ds, POOL):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    p = ex.get("prompt") or ch[0]["content"]
    c, r = ch[-1]["content"], rj[-1]["content"]
    if not (p and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < 1.0: continue
    recs.append(dict(prompt=p, chosen=c, rejected=r, is_test=int(_phash(p)[:8], 16) % 10 == 0))
train = [x for x in recs if not x["is_test"]][:N_PROBE]
test = [x for x in recs if x["is_test"]][:N_DIAG]
print(f"[data] probe-train {len(train)} | diag {len(test)}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); NL = len(BLOCKS); HID = model.config.hidden_size

# ---- is <pad> a trained embedding? ----
emb = model.get_input_embeddings().weight
norms = emb.float().norm(dim=1)
pid = tok.pad_token_id
print(f"\n[embedding] matrix rows={emb.shape[0]} vocab_size(cfg)={model.config.vocab_size}")
print(f"[embedding] ||e[pad={pid}]|| = {norms[pid]:.4f}   "
      f"median||e|| = {norms[:128000].median():.4f}   "
      f"ratio = {float(norms[pid] / norms[:128000].median()):.3f}")
print(f"[embedding] ||e[eot={tok.eos_token_id}]|| = {norms[tok.eos_token_id]:.4f}", flush=True)

@torch.no_grad()
def feats_at(texts, layer, n_pad=0, bs=8):
    """Last-real-token residual at `layer`; if n_pad>0, append n_pad <pad> tokens (attention-masked,
    exactly as generate()'s right-padding does) and read the FINAL pad position instead."""
    out = np.zeros((len(texts), HID), np.float32)
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        ids, am = enc.input_ids, enc.attention_mask
        if n_pad:
            pad = torch.full((ids.shape[0], n_pad), pid, device=DEV, dtype=ids.dtype)
            ids = torch.cat([ids, pad], 1)
            am = torch.cat([am, torch.zeros_like(pad)], 1)
        with ResidualCapture([BLOCKS[layer]]) as cap:
            model(input_ids=ids, attention_mask=am)
        out[s:s + enc.input_ids.shape[0]] = cap.get()[0][:, -1].float().cpu().numpy()
    return out

# ---- Stage A probe (reuse cache if present) ----
cachef = "/workspace/uf_probe_feats.npz"
if not os.path.exists(cachef):
    print("[error] run uf_probe_rl.py first to build the feature cache", file=sys.stderr); sys.exit(1)
z = np.load(cachef); Fc_tr, Fr_tr = z["a"], z["b"]
LSTAR = json.load(open("/workspace/uf_probe_curve.json"))["Lstar"]
print(f"\n[probe] using L*={LSTAR}", flush=True)

rng = np.random.RandomState(SEED)
n_tr = Fc_tr.shape[0]
s_tr = np.where(rng.rand(n_tr) < 0.5, 1.0, -1.0).astype(np.float32)
sd = np.concatenate([Fc_tr[:, LSTAR], Fr_tr[:, LSTAR]]).std(0) + 1e-6
dtr = ((Fc_tr[:, LSTAR] - Fr_tr[:, LSTAR]) / sd) * s_tr[:, None]

# ---- diag features: correct read vs pad read, at L* ----
tc = [render_full(x["prompt"], x["chosen"]) for x in test]
tr_ = [render_full(x["prompt"], x["rejected"]) for x in test]
res = {"Lstar": LSTAR, "n_diag": len(test),
       "pad_emb_norm_ratio": float(norms[pid] / norms[:128000].median())}

print("[diag] correct read...", flush=True)
Cc, Cr = feats_at(tc, LSTAR), feats_at(tr_, LSTAR)
s_te = np.where(rng.rand(len(test)) < 0.5, 1.0, -1.0).astype(np.float32)
dte = ((Cc - Cr) / sd) * s_te[:, None]
acc, head, _ = train_bayes_head(dtr, s_tr, dte, s_te)
res["acc_correct_read"] = float(acc)
print(f"[diag] correct read : pairwise acc {acc:.3f}", flush=True)

MU = head.mu.detach().float().cpu().numpy()
def proj_acc(Fc_, Fr_):
    """Rank by absolute projection, as probe_reward does at RL time."""
    zc, zr = ((Fc_ / sd) @ MU), ((Fr_ / sd) @ MU)
    return float((zc > zr).mean()), float(np.std(np.concatenate([zc, zr])))

a0, sp0 = proj_acc(Cc, Cr)
res["proj_acc_correct"], res["proj_spread_correct"] = a0, sp0
print(f"[diag] correct read : reward-ranking acc {a0:.3f}  spread {sp0:.3f}", flush=True)

for k in PAD_KS:
    print(f"[diag] pad read (n_pad={k})...", flush=True)
    Pc, Pr = feats_at(tc, LSTAR, n_pad=k), feats_at(tr_, LSTAR, n_pad=k)
    dte_p = ((Pc - Pr) / sd) * s_te[:, None]
    with torch.no_grad():
        zp = head.z_s2(torch.tensor(dte_p * s_te[:, None], dtype=torch.float32))[0]
    acc_p = float((zp > 0).float().mean())
    ap, spp = proj_acc(Pc, Pr)
    res[f"acc_pad{k}"], res[f"proj_acc_pad{k}"], res[f"proj_spread_pad{k}"] = acc_p, ap, spp
    print(f"[diag] pad read k={k:2d}: probe acc {acc_p:.3f} | reward-ranking acc {ap:.3f} | "
          f"spread {spp:.3f}", flush=True)

json.dump(res, open("/workspace/uf_readpos_diag.json", "w"), indent=1)
print("\nDONE", flush=True)
