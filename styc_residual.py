#!/usr/bin/env python
"""Residual (boosted) fitting on styc: build the conditional readout from preference labels
alone -- no factor labels. (2026-07-31)

Stage 1: natural linear head on the mixed diet (conflicts at the realistic 15%) -> known to be
style-first (conflict ~.32 @L35). Stage 2: linear head on the SAME diet with boosting weights
w_i = exp(-clip(y_i z1_i, -3, 3)) -- pairs stage 1 gets wrong or barely right dominate; these
are the correctness-dominant residual. Combiners (the Pareto result says additive linear+linear
cannot beat min-family .48, so the win must be conditional):
  add    z1 + alpha*z2, alpha by train mixed acc      (consistency check: should hit the frontier)
  gate   z2 where |z2|>tau else z1, tau by train acc  (hard dominance gate)
  stack  tiny antisymmetric MLP on (z1, z2)           (learned 2-D conditional combiner)

Env: STYC_CACHE=/workspace/styc_feats_v2.npz LAYERS=35,20 RATE=0.15 SEED=0
Out: /workspace/styc_residual.json (incl. per-family (z1,z2) held-out scatter for plotting)"""
import os, sys, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import train_bayes_head

E = os.environ.get
SEED = int(E("SEED", 0))
LAYERS = [int(x) for x in E("LAYERS", "35,20").split(",")]
RATE = float(E("RATE", 0.15))
FE = {k: v for k, v in np.load(E("STYC_CACHE", "/workspace/styc_feats_v2.npz")).items()}
n, NL, HID = FE["ct"].shape
tr = np.arange(n) % 5 != 0
tr_idx, te_idx = np.where(tr)[0], np.where(~tr)[0]

PAIRS = dict(corr_e=("ce", "we"), corr_t=("ct", "wt"), style_c=("ce", "ct"),
             style_w=("we", "wt"), aligned=("ce", "wt"), conflict=("ct", "we"))
DIET = ["corr_e", "corr_t", "style_c", "style_w", "aligned"]

def build(idx, li, rate, rng):
    As = [FE[PAIRS[k][0]][idx, li].astype(np.float32) for k in DIET]
    Bs = [FE[PAIRS[k][1]][idx, li].astype(np.float32) for k in DIET]
    base = len(idx) * len(DIET)
    n_conf = int(round(rate / (1 - rate) * base)) if rate > 0 else 0
    if n_conf:
        pick = rng.choice(idx, size=n_conf, replace=n_conf > len(idx))
        a, b = PAIRS["conflict"]
        As.append(FE[a][pick, li].astype(np.float32)); Bs.append(FE[b][pick, li].astype(np.float32))
    return np.concatenate(As), np.concatenate(Bs)

def z_of(h, sd, A, B):
    fs = torch.tensor(((A - B).astype(np.float32)) / sd)
    MU = h.mu.detach().float(); SIG2 = F.softplus(h.rho.detach()).float().pow(2)
    return (fs.matmul(MU) / torch.sqrt(1 + fs.pow(2).matmul(SIG2))).numpy()

class Stack2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return (self.g(x) - self.g(-x)).squeeze(-1)

for li in LAYERS:
    rng = np.random.RandomState(SEED + 7)
    A, B = build(tr_idx, li, RATE, rng)
    A_v, B_v = build(te_idx, li, RATE, rng)
    s = np.where(rng.rand(len(A)) < 0.5, 1.0, -1.0).astype(np.float32)
    s_v = np.where(rng.rand(len(A_v)) < 0.5, 1.0, -1.0).astype(np.float32)
    sd = np.concatenate([A, B]).std(0) + 1e-6
    DF, DFv = ((A - B) / sd) * s[:, None], ((A_v - B_v) / sd) * s_v[:, None]
    # stage 1: natural head
    _, h1, _ = train_bayes_head(DF, s, DFv, s_v)
    z1_tr = z_of(h1, sd, A, B) * s   # signed: >0 = stage 1 correct
    # stage 2: boosting weights from stage-1 errors
    w = np.exp(-np.clip(z1_tr, -3, 3)).astype(np.float32); w *= len(w) / w.sum()
    z1_v = z_of(h1, sd, A_v, B_v) * s_v
    w_v = np.exp(-np.clip(z1_v, -3, 3)).astype(np.float32); w_v *= len(w_v) / w_v.sum()
    _, h2, _ = train_bayes_head(DF, s, DFv, s_v, w_tr=w, w_te=w_v)

    # combiner fitting data: signed z-pairs on the TRAIN diet
    Z_tr = np.stack([z_of(h1, sd, A, B), z_of(h2, sd, A, B)], 1)      # unsigned scores
    Z_v = np.stack([z_of(h1, sd, A_v, B_v), z_of(h2, sd, A_v, B_v)], 1)
    y_tr, y_v = s, s_v

    # additive
    alphas = [0.25, 0.5, 1, 2, 4, 8, 16]
    a_best = max(alphas, key=lambda a: float((((Z_tr[:, 0] + a * Z_tr[:, 1]) * y_tr) > 0).mean()))
    # gate
    taus = np.quantile(np.abs(Z_tr[:, 1]), [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    def gated(Z, tau):
        return np.where(np.abs(Z[:, 1]) > tau, Z[:, 1], Z[:, 0])
    t_best = max(taus, key=lambda t: float(((gated(Z_tr, t) * y_tr) > 0).mean()))
    # 2-D stack
    torch.manual_seed(SEED)
    net = Stack2D()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-3)
    Xt = torch.tensor(Z_tr * y_tr[:, None]); yt = torch.tensor(y_tr)
    Xvv = torch.tensor(Z_v * y_v[:, None]); yvv = torch.tensor(y_v)
    best = dict(loss=1e9, wait=0, state=None)
    for ep in range(500):
        for sl in torch.randperm(len(Xt)).split(512):
            opt.zero_grad()
            F.softplus(-net(Xt[sl]) * yt[sl]).mean().backward()
            opt.step()
        with torch.no_grad():
            vl = float(F.softplus(-net(Xvv) * yvv).mean())
        if vl < best["loss"] - 1e-4:
            best = dict(loss=vl, wait=0, state={k: v.clone() for k, v in net.state_dict().items()})
        else:
            best["wait"] += 1
            if best["wait"] >= 30: break
    net.load_state_dict(best["state"])

    # held-out per-family eval for every scorer
    res_li, scatter = {}, {}
    for k, (a, b) in PAIRS.items():
        Fa, Fb = FE[a][te_idx, li], FE[b][te_idx, li]
        z1 = z_of(h1, sd, Fa, Fb); z2 = z_of(h2, sd, Fa, Fb)
        Z = np.stack([z1, z2], 1)
        with torch.no_grad():
            zs = net(torch.tensor(Z, dtype=torch.float32)).numpy()
        res_li[k] = dict(h1=float((z1 > 0).mean()), h2=float((z2 > 0).mean()),
                         add=float(((z1 + a_best * z2) > 0).mean()),
                         gate=float((gated(Z, t_best) > 0).mean()),
                         stack=float((zs > 0).mean()))
        scatter[k] = dict(z1=z1.round(3).tolist(), z2=z2.round(3).tolist())
    out = json.load(open("/workspace/styc_residual.json")) if os.path.exists("/workspace/styc_residual.json") else {}
    out[f"L{li}"] = dict(alpha=a_best, tau=float(t_best), acc=res_li, scatter=scatter)
    json.dump(out, open("/workspace/styc_residual.json", "w"), indent=1)
    for m in ["h1", "h2", "add", "gate", "stack"]:
        line = " ".join(f"{k} {res_li[k][m]:.2f}" for k in PAIRS)
        print(f"[L{li:2d} {m:5s}] {line} | min {min(res_li[k][m] for k in PAIRS):.2f}", flush=True)
print("DONE", flush=True)
