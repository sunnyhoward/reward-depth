#!/usr/bin/env python
"""Can ANY single linear head at one layer implement the lexicographic preference? (2026-07-31)

Prompted by the conflict-rate sweep: natural fits saturate at ~1/3 conflict acc and buy it by
sacrificing style acc, even at L35 where style (1.000) and terse-correctness (.974) are both
individually decodable. Question: is that a DIET limit (fixable by weighting) or a
REPRESENTATIONAL limit (no linear direction does dominance-ordered combination)?

Method: balanced diet of ALL SIX families (incl. conflict) at the probe layer; sweep the
conflict sample-weight lambda. Report per-family held-out acc at each lambda -> the
conflict-vs-style Pareto frontier. Reference points: a conflict-only head (linear ceiling for
conflicts alone) and the corr_t-only head. Layers: L35 (top) and L20 (retrieval elbow).

Env: STYC_CACHE=/workspace/styc_feats_v2.npz LAYERS=35,20 LAMS=0.5,1,2,4,8,16,32 SEED=0
Out: /workspace/styc_lex_pareto.json"""
import os, sys, json
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import train_bayes_head

E = os.environ.get
SEED = int(E("SEED", 0))
LAYERS = [int(x) for x in E("LAYERS", "35,20").split(",")]
LAMS = [float(x) for x in E("LAMS", "0.5,1,2,4,8,16,32").split(",")]
FE = {k: v for k, v in np.load(E("STYC_CACHE", "/workspace/styc_feats_v2.npz")).items()}
n, NL, HID = FE["ct"].shape
tr = np.arange(n) % 5 != 0
tr_idx, te_idx = np.where(tr)[0], np.where(~tr)[0]

PAIRS = dict(corr_e=("ce", "we"), corr_t=("ct", "wt"), style_c=("ce", "ct"),
             style_w=("we", "wt"), aligned=("ce", "wt"), conflict=("ct", "we"))
FAMS = list(PAIRS)

def stack(idx, li, fams):
    A = np.concatenate([FE[PAIRS[k][0]][idx, li].astype(np.float32) for k in fams])
    B = np.concatenate([FE[PAIRS[k][1]][idx, li].astype(np.float32) for k in fams])
    w = np.concatenate([np.full(len(idx), 1.0, np.float32) for _ in fams])
    fam_of = np.concatenate([np.full(len(idx), i) for i, _ in enumerate(fams)])
    return A, B, w, fam_of

def fit(li, fams, conf_w, rng):
    A, B, w, fam_of = stack(tr_idx, li, fams)
    if "conflict" in fams: w[fam_of == fams.index("conflict")] = conf_w
    A_v, B_v, w_v, fam_v = stack(te_idx, li, fams)
    if "conflict" in fams: w_v[fam_v == fams.index("conflict")] = conf_w
    s = np.where(rng.rand(len(A)) < 0.5, 1.0, -1.0).astype(np.float32)
    s_v = np.where(rng.rand(len(A_v)) < 0.5, 1.0, -1.0).astype(np.float32)
    sd = np.concatenate([A, B]).std(0) + 1e-6
    _, h, _ = train_bayes_head(((A - B) / sd) * s[:, None], s,
                               ((A_v - B_v) / sd) * s_v[:, None], s_v, w_tr=w, w_te=w_v)
    return h, sd

def fam_acc(h, sd, li):
    MU = h.mu.detach().float(); SIG2 = F.softplus(h.rho.detach()).float().pow(2)
    out = {}
    for k, (a, b) in PAIRS.items():
        fs = torch.tensor(((FE[a][te_idx, li] - FE[b][te_idx, li]).astype(np.float32)) / sd)
        z = fs.matmul(MU) / torch.sqrt(1 + fs.pow(2).matmul(SIG2))
        out[k] = float((torch.special.ndtr(z) > 0.5).float().mean())
    return out

res = {}
for li in LAYERS:
    res[f"L{li}"] = {}
    # reference ceilings: single-family heads
    for ref in ["conflict", "corr_t", "corr_e"]:
        rng = np.random.RandomState(SEED + 7)
        h, sd = fit(li, [ref], 1.0, rng)
        acc = fam_acc(h, sd, li)
        res[f"L{li}"][f"only_{ref}"] = acc
        print(f"[L{li} only_{ref:8s}] " + " ".join(f"{k} {v:.2f}" for k, v in acc.items()), flush=True)
    # the frontier: balanced six-family diet, conflict weight swept
    for lam in LAMS:
        rng = np.random.RandomState(SEED + 7)
        h, sd = fit(li, FAMS, lam, rng)
        acc = fam_acc(h, sd, li)
        res[f"L{li}"][f"lam_{lam:g}"] = acc
        mn = min(acc.values())
        print(f"[L{li} lam {lam:4g}] " + " ".join(f"{k} {v:.2f}" for k, v in acc.items())
              + f" | min {mn:.2f}", flush=True)
json.dump(res, open("/workspace/styc_lex_pareto.json", "w"), indent=1)
print("DONE", flush=True)
