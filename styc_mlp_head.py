#!/usr/bin/env python
"""Nonlinear (MLP) preference head on styc features: is style capture an artifact of LINEAR
reward heads? (2026-07-31)

The lex-pareto diagnostic showed no single linear direction at L35 implements the lexicographic
preference (best min-across-families .48; every conflict-competent direction is anti-style).
But the conditional readout "if correctness signal present, use it; else style" IS expressible
with one hidden layer. Test: antisymmetric MLP f(x) = g(x) - g(-x) on the same difference
features, natural mixed diet.

Cells: layers {10, 20, 35} x conflict rate {0, 0.15}. Rate 0 is the information-theoretic
control: no head, of any capacity, can learn dominance from a conflict-free diet.

Env: STYC_CACHE=/workspace/styc_feats_v2.npz LAYERS=10,20,35 RATES=0,0.15 HID_MLP=64 SEED=0
Out: /workspace/styc_mlp_head.json"""
import os, sys, json
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

E = os.environ.get
SEED = int(E("SEED", 0))
LAYERS = [int(x) for x in E("LAYERS", "10,20,35").split(",")]
RATES = [float(x) for x in E("RATES", "0,0.15").split(",")]
HID_MLP = int(E("HID_MLP", 64))
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

class AntisymMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x):
        return (self.g(x) - self.g(-x)).squeeze(-1)   # f(-x) = -f(x) exactly

def fit_mlp(li, rate, rng):
    A, B = build(tr_idx, li, rate, rng)
    A_v, B_v = build(te_idx, li, rate, rng)
    sd = np.concatenate([A, B]).std(0) + 1e-6
    s = np.where(rng.rand(len(A)) < 0.5, 1.0, -1.0).astype(np.float32)
    s_v = np.where(rng.rand(len(A_v)) < 0.5, 1.0, -1.0).astype(np.float32)
    X = torch.tensor(((A - B) / sd) * s[:, None]); y = torch.tensor(s)
    Xv = torch.tensor(((A_v - B_v) / sd) * s_v[:, None]); yv = torch.tensor(s_v)
    torch.manual_seed(SEED)
    net = AntisymMLP(X.shape[1], HID_MLP)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best = dict(loss=1e9, wait=0, state=None)
    for ep in range(300):
        for sl in torch.randperm(len(X)).split(256):
            opt.zero_grad()
            nn.functional.softplus(-net(X[sl]) * y[sl]).mean().backward()
            opt.step()
        with torch.no_grad():
            vl = float(nn.functional.softplus(-net(Xv) * yv).mean())
        if vl < best["loss"] - 1e-4:
            best = dict(loss=vl, wait=0, state={k: v.clone() for k, v in net.state_dict().items()})
        else:
            best["wait"] += 1
            if best["wait"] >= 20: break
    net.load_state_dict(best["state"])
    return net, sd

def fam_acc(net, sd, li):
    out = {}
    with torch.no_grad():
        for k, (a, b) in PAIRS.items():
            fs = torch.tensor(((FE[a][te_idx, li] - FE[b][te_idx, li]).astype(np.float32)) / sd)
            out[k] = float((net(fs) > 0).float().mean())
    return out

res = {}
for li in LAYERS:
    for rate in RATES:
        rng = np.random.RandomState(SEED + 7)
        net, sd = fit_mlp(li, rate, rng)
        acc = fam_acc(net, sd, li)
        res[f"L{li}_r{rate:g}"] = acc
        print(f"[L{li:2d} conf {rate:4.0%}] " + " ".join(f"{k} {v:.2f}" for k, v in acc.items())
              + f" | min {min(acc.values()):.2f}", flush=True)
json.dump(res, open("/workspace/styc_mlp_head.json", "w"), indent=1)
print("DONE", flush=True)
