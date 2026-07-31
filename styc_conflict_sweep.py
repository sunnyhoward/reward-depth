#!/usr/bin/env python
"""styc conflict-rate dose-response (2026-07-31, follows stage A v3).

Stage A v3 established: with conflicts held out of training, the pref head is style-dominant
on conflict pairs at EVERY layer (0.000), even at L35 where it scores .966 on style-matched
correctness -- the non-conflict diet carries no dominance bit (consistent with either
lexicographic order). This script measures the dose-response: what fraction of conflict pairs
in the training diet flips the head to correctness-dominance, and at which depths.

Diet: the 5 v3 families in full (corr_e, corr_t, style_c, style_w, aligned) + conflict pairs
subsampled (with replacement above availability) to make up RATE of the total. Validation for
early stopping: the SAME mixed distribution on held-out questions (stage-B spec). Eval: full
per-layer per-family held-out accuracy.

Env: STYC_CACHE=/workspace/styc_feats_v2.npz RATES="0,0.02,0.05,0.10,0.15,0.20" SEED=0
Out: /workspace/styc_conflict_sweep.json"""
import os, sys, json
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import train_bayes_head

E = os.environ.get
SEED = int(E("SEED", 0))
RATES = [float(x) for x in E("RATES", "0,0.02,0.05,0.10,0.15,0.20").split(",")]
cachef = E("STYC_CACHE", "/workspace/styc_feats_v2.npz")
FE = {k: v for k, v in np.load(cachef).items()}
n, NL, HID = FE["ct"].shape
tr = np.arange(n) % 5 != 0
te = ~tr
tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
print(f"[data] {n} questions, {NL} layers, hid {HID} | train {len(tr_idx)} te {len(te_idx)}", flush=True)

PAIRS = dict(corr_e=("ce", "we"), corr_t=("ct", "wt"), style_c=("ce", "ct"),
             style_w=("we", "wt"), aligned=("ce", "wt"), conflict=("ct", "we"))
DIET = ["corr_e", "corr_t", "style_c", "style_w", "aligned"]

def build(idx, li, rate, rng):
    """(A, B) stacks at layer li: 5 diet families on idx in full + conflicts at `rate` of total."""
    As = [FE[a][idx, li].astype(np.float32) for a, _ in (PAIRS[k] for k in DIET)]
    Bs = [FE[b][idx, li].astype(np.float32) for _, b in (PAIRS[k] for k in DIET)]
    base = len(idx) * len(DIET)
    n_conf = int(round(rate / (1 - rate) * base)) if rate > 0 else 0
    if n_conf:
        pick = rng.choice(idx, size=n_conf, replace=n_conf > len(idx))
        a, b = PAIRS["conflict"]
        As.append(FE[a][pick, li].astype(np.float32)); Bs.append(FE[b][pick, li].astype(np.float32))
    return np.concatenate(As), np.concatenate(Bs)

def fit_layer(li, rate, rng):
    A, B = build(tr_idx, li, rate, rng)
    A_v, B_v = build(te_idx, li, rate, rng)
    s = np.where(rng.rand(len(A)) < 0.5, 1.0, -1.0).astype(np.float32)
    s_v = np.where(rng.rand(len(A_v)) < 0.5, 1.0, -1.0).astype(np.float32)
    pool = np.concatenate([A, B]); sd = pool.std(0) + 1e-6
    a, h, e = train_bayes_head(((A - B) / sd) * s[:, None], s,
                               ((A_v - B_v) / sd) * s_v[:, None], s_v)
    return h, sd, float(a)

def fam_acc(h, sd, li):
    out = {}
    MU = h.mu.detach().float(); SIG2 = F.softplus(h.rho.detach()).float().pow(2)
    for k, (a, b) in PAIRS.items():
        fs = torch.tensor(((FE[a][te_idx, li] - FE[b][te_idx, li]).astype(np.float32)) / sd)
        z = fs.matmul(MU) / torch.sqrt(1 + fs.pow(2).matmul(SIG2))
        out[k] = float((torch.special.ndtr(z) > 0.5).float().mean())
    return out

res = {}
for rate in RATES:
    rng = np.random.RandomState(SEED + 7)   # same stream per rate: diets differ only by the conflicts
    curves = {k: [] for k in PAIRS}; mixed = []
    for li in range(NL):
        h, sd, a = fit_layer(li, rate, rng)
        mixed.append(a)
        for k, v in fam_acc(h, sd, li).items(): curves[k].append(v)
    res[f"{rate:.2f}"] = dict(mixed_val=mixed, curves=curves)
    cf = curves["conflict"]
    print(f"[rate {rate:4.0%}] conflict max {max(cf):.3f} @L{int(np.argmax(cf))} | "
          f"conflict {[round(a,2) for a in cf[::4]]} | "
          f"corr_t@L35 {curves['corr_t'][NL-1]:.3f} | style_c min {min(curves['style_c']):.3f} | "
          f"aligned min {min(curves['aligned']):.3f}", flush=True)
json.dump(res, open("/workspace/styc_conflict_sweep.json", "w"), indent=1)
print("DONE", flush=True)
