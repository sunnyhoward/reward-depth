#!/usr/bin/env python
"""Where does stage 1 put its separation -- in the BASE preference direction, or a new one?

WHY. `pf_mdsides.py` answered NEXT_0817 §6.0a in the negative: stage 1 does not win by dropping the
rejected side. It showed something else instead -- at the attach point, in the base-fitted frame,
the chosen-minus-rejected projection does not grow AT ALL (L20, READ=last: +21.09 base -> +18.69 at
s100, +20.87 at s300), while BOTH sides shift by ~+75 in common mode. Meanwhile the run's own meter
has the probe saturated (`sat 1.00`, z +5..+9 sigma). Both cannot be true of the same direction, so
the objective must be satisfied along a direction the base frame does not see.

This measures that directly, at the attach point, on the same 128 train pairs:

    cos(u_base, w_hat)      base mean-difference direction vs the REFITTED probe's direction
    sep(u_base)             mean (chosen-rejected)/SD . u_base           base vs trained
    sep(w_hat)              mean (chosen-rejected)/SD . w_hat            base vs trained

`sep` is reported for the SAME direction on both models, which is what makes the pair readable: a
direction whose separation grows on the trained model but not on the base one is a direction stage 1
built.

THE CONTROL THAT MAKES cos(u, w_hat) READABLE (`CTRL_N`). A refitted probe is a LOGISTIC direction
and `u` is a MEAN-DIFFERENCE direction; on minimal pairs those are close but not equal, so a cos
below 1 proves nothing by itself. `base logistic probe` is the same recipe (`Probe.refit`: Adam on
-logsigmoid(d.w) + PROBE_L2, 600 steps) fitted on the BASE model's own reads over CTRL_N pairs. It
says how far from u this probe FAMILY lands with no training at all -- the number the trained
probe's cos has to be read against.

Env: CKPTS=path[,path]  LAYER=20  N=128  CTRL_N=1024  READ=last
"""
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from pf_common import DEV, MODEL, ResidualCapture, encode, load_split, pair_texts, span_mask, span_read  # noqa: E402

E = os.environ.get
LAYER, N, BS = int(E("LAYER", 20)), int(E("N", 128)), int(E("BS", 8))
READ, MAXLEN = E("READ", "last"), int(E("MAX_LEN", 256))
CKPTS = [c for c in E("CKPTS", "").split(",") if c]
CTRL_N = int(E("CTRL_N", 1024))
PROBE_L2, PROBE_LR = float(E("PROBE_L2", 1e-3)), float(E("PROBE_LR", 1e-2))

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
ALL = load_split("train")
rows = ALL[:N]


def sides(model, rr=None):
    rr = rows if rr is None else rr
    ch, rj = [], []
    blocks = list(model.model.layers)
    with torch.no_grad():
        for s in range(0, len(rr), BS):
            trip = pair_texts(tok, rr[s:s + BS])
            texts = [t for c, j, _ in trip for t in (c, j)]
            plens = [pl for _, _, pl in trip for _ in (0, 1)]
            enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
            m = span_mask(tok, texts, plens, enc)
            with ResidualCapture([blocks[LAYER]]) as cap:
                model(**enc)
            v = span_read(cap.get()[0], m, READ).float()
            ch.append(v[0::2].cpu()); rj.append(v[1::2].cpu())
    return torch.cat(ch), torch.cat(rj)


base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
bch, brj = sides(base)
cch, crj = sides(base, ALL[:CTRL_N]) if CTRL_N else (bch, brj)
del base
torch.cuda.empty_cache()
d = bch - brj
SD = d.std(0).clamp(min=1e-3)
u = d.mean(0) / (d.mean(0).norm() + 1e-6)


def sep(ch, rj, w):
    return float((((ch - rj) / SD) @ (w / w.norm())).mean())


import torch.nn.functional as F  # noqa: E402
X = (cch - crj).to(DEV).float()
wb = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
opt = torch.optim.Adam([wb], lr=PROBE_LR)
for _ in range(600):
    i = torch.randint(0, X.shape[0], (min(256, X.shape[0]),), device=DEV)
    opt.zero_grad()
    (-F.logsigmoid(X[i] @ wb).mean() + PROBE_L2 * wb.pow(2).sum()).backward()
    opt.step()
wb = wb.detach().cpu()

print(f"L{LAYER} ({READ}), {len(rows)} pairs, base-fitted SD; "
      f"control probe fitted on {CTRL_N} BASE pairs\n")
print(f"{'ckpt':<26} {'cos(u,w)':>9} {'sep(u) base':>12} {'sep(u) trn':>11} "
      f"{'sep(w) base':>12} {'sep(w) trn':>11}")
print(f"{'base mean-diff u':<26} {1.0:>9.3f} {sep(bch, brj, u):>12.2f} {'':>11} "
      f"{'':>12} {'':>11}")
print(f"{'base logistic probe':<26} {float((u/u.norm()) @ (wb/wb.norm())):>9.3f} "
      f"{sep(bch, brj, u):>12.2f} {'':>11} {sep(bch, brj, wb):>12.2f} {'':>11}")
for c in CKPTS:
    pr = torch.load(f"{c}/probe.pt", map_location="cpu")
    assert pr["layer"] == LAYER and pr["read"] == READ, (pr["layer"], pr["read"])
    w = pr["w"].float()
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    m = PeftModel.from_pretrained(m, c).merge_and_unload().eval()
    tch, trj = sides(m)
    del m
    torch.cuda.empty_cache()
    print(f"{os.path.basename(os.path.dirname(c))+'/'+os.path.basename(c):<26} "
          f"{float((u/u.norm()) @ (w/w.norm())):>9.3f} {sep(bch, brj, u):>12.2f} "
          f"{sep(tch, trj, u):>11.2f} {sep(bch, brj, w):>12.2f} {sep(tch, trj, w):>11.2f}",
          flush=True)
