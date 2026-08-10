#!/usr/bin/env python
"""Is the block-10 -> block-18 gap a SCALE problem, a ROTATION, or missing content?

Grafting the residual at block L into block K skips real computation, and the graft's readout
degrades smoothly with how much is skipped (agreement with the full model's ordering on
RewardBench2: 10->14 0.908, 10->16 0.808, 10->18 0.788). The obvious repair is a small trainable
bridge in between. Before training one, this asks what capacity it would actually need — and the
answer is available in closed form, so no SGD and no hyperparameters are involved.

Three nested hypotheses, three bridges of very different size:

  IDENTITY   h_L injected as-is                        0 parameters   (what the graft does now)
  SCALE      alpha * h_L, alpha by least squares       1 parameter    (residual norms grow with
                                                                      depth; block K may simply be
                                                                      receiving something too small)
  AFFINE     W h_L + b, W by ridge regression       ~4.2M parameters  (same content, different
                                                                      basis — the "rotation" story;
                                                                      6x smaller than the 25.2M head)

The fits are closed-form on the model's own replay text, targeting h_K — they never see a
preference label, so a bridge trained this way is frozen at stage-1 time exactly like the EAGLE
head, and cannot absorb the install.

What the numbers mean:
  * If SCALE recovers most of the gap, the "rotation" framing is wrong and one scalar is the fix.
  * If AFFINE recovers most of it and SCALE does not, the content is there in a different basis —
    a 4.2M bridge replaces a 25.2M distilled head.
  * If neither does, blocks L..K-1 compute something genuinely new and no bridge of any size will
    substitute for them.

R^2 is reported per hypothesis on held-out tokens, and then all three are scored on the ranking
task the recipe actually cares about.

Usage: python sup_bridge.py
Env:   SRC=10 DST=18 N_TOK=60000 RIDGE=1.0 N_PAIRS=250 EVAL_SETS=rewardbench2,uf_sup
       OUT=/workspace/sup_bridge.json
"""
import json
import os
import sys

import numpy as np
import torch

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "decodability"))
from sup_common import MODEL, DEV, encode, span_mask                  # noqa: E402
from helpers import ResidualCapture                                   # noqa: E402
import sup_eval_pref as EV                                            # noqa: E402

SRC, DST = int(E("SRC", 10)), int(E("DST", 18))
N_TOK = int(E("N_TOK", 60000))
RIDGE = float(E("RIDGE", 1.0))
N_PAIRS = int(E("N_PAIRS", 250))
BS = int(E("BS", 4))
MAX_LEN = int(E("MAX_LEN", 512))
SETS = E("EVAL_SETS", "rewardbench2,uf_sup").split(",")
OUT = E("OUT", "/workspace/sup_bridge.json")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    BLOCKS = list(model.model.layers)
    PAD = tok.pad_token_id

    # ── 1. collect paired residuals (h at SRC's output, h at DST's input) on replay text ───────
    bank = torch.load("/workspace/sup/replay_bank.pt")
    IDS = bank["ids"]
    X, Y = [], []
    n = 0
    i = 0
    while n < N_TOK and i + BS <= len(IDS):
        x = IDS[i:i + BS].to(DEV); i += BS
        enc = dict(input_ids=x, attention_mask=(x != PAD).long())
        with torch.no_grad(), ResidualCapture([BLOCKS[SRC], BLOCKS[DST - 1]]) as cap:
            model.model(**enc)
        b = cap.get()
        m = enc["attention_mask"].bool().reshape(-1)
        xs = b[0].reshape(-1, b[0].shape[-1])[m].float()
        ys = b[1].reshape(-1, b[1].shape[-1])[m].float()
        X.append(xs.cpu()); Y.append(ys.cpu())
        n += int(m.sum())
    X = torch.cat(X)[:N_TOK]; Y = torch.cat(Y)[:N_TOK]
    ntr = int(0.9 * len(X))
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    print(f"[data] {len(X)} token positions, hid {X.shape[1]}", flush=True)

    nx, ny = Xtr.norm(dim=-1), Ytr.norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(Xtr, Ytr, dim=-1)
    print(f"[geometry] ||h_{SRC}|| {nx.mean():.1f}  ||h_{DST}|| {ny.mean():.1f}  "
          f"ratio {(ny/nx).mean():.2f}  |  cos(h_{SRC}, h_{DST}) {cos.mean():.3f}", flush=True)

    def r2(pred, tgt):
        ss = ((tgt - pred) ** 2).sum()
        tt = ((tgt - tgt.mean(0)) ** 2).sum()
        return float(1 - ss / tt)

    alpha = float((Xtr * Ytr).sum() / (Xtr * Xtr).sum())
    Xd = torch.cat([Xtr, torch.ones(len(Xtr), 1)], 1).double().to(DEV)
    Yd = Ytr.double().to(DEV)
    A = Xd.T @ Xd + RIDGE * torch.eye(Xd.shape[1], dtype=torch.float64, device=DEV)
    Wb = torch.linalg.solve(A, Xd.T @ Yd)                       # (hid+1, hid)
    W, bvec = Wb[:-1].float(), Wb[-1].float()
    del Xd, Yd, A

    # Saved so `sup_bridge_fit.py` can warm-start from it: the least-squares map is the wrong
    # OBJECTIVE (it reconstructs a hidden state the output mostly discards) but a good starting
    # point, since it already carries the scale and the bulk of the rotation.
    torch.save(dict(W=W.cpu(), b=bvec.cpu(), src=SRC, dst=DST, alpha=alpha),
               "/workspace/sup_bridge_ls.pt")
    Xte_d, Yte_d = Xte.to(DEV), Yte.to(DEV)
    fits = dict(identity=r2(Xte_d, Yte_d), scale=r2(alpha * Xte_d, Yte_d),
                affine=r2(Xte_d @ W + bvec, Yte_d))
    print(f"[fit] held-out R^2 -> identity {fits['identity']:+.3f}  "
          f"scale(a={alpha:.2f}) {fits['scale']:+.3f}  affine {fits['affine']:+.3f}", flush=True)
    del Xte_d, Yte_d

    # ── 2. score each bridge on the ranking task ──────────────────────────────────────────────
    Wg, bg = W.to(DEV).to(torch.bfloat16), bvec.to(DEV).to(torch.bfloat16)

    def score(rows, mode):
        A_, B_ = [], []
        for s in range(0, len(rows), BS):
            b = rows[s:s + BS]
            texts = [t for r in b for t in (r[0], r[1])]
            plens = [r[2] for r in b for _ in (0, 1)]
            enc = encode(tok, texts, max_length=MAX_LEN).to(DEV)
            msk = span_mask(tok, texts, plens, enc)
            with torch.no_grad():
                if mode == "full":
                    lg = model(**enc).logits[:, :-1]
                else:
                    with ResidualCapture([BLOCKS[SRC]]) as cap:
                        model(**enc)
                    src = cap.get()[0]
                    if mode == "scale":
                        src = alpha * src
                    elif mode == "affine":
                        src = src @ Wg + bg

                    def hook(mod, args, kwargs, _s=src):
                        if args:
                            return ((_s,) + args[1:], kwargs)
                        kwargs = dict(kwargs); kwargs["hidden_states"] = _s
                        return (args, kwargs)
                    hdl = BLOCKS[DST].register_forward_pre_hook(hook, with_kwargs=True)
                    try:
                        lg = model(**enc).logits[:, :-1]
                    finally:
                        hdl.remove()
                lp = EV._span_logps(lg, enc, msk)
            A_ += lp[0::2].float().cpu().tolist(); B_ += lp[1::2].float().cpu().tolist()
        return np.array(A_) > np.array(B_)

    res = dict(src=SRC, dst=DST, alpha=alpha, r2=fits,
               geometry=dict(norm_src=float(nx.mean()), norm_dst=float(ny.mean()),
                             norm_ratio=float((ny / nx).mean()), cos=float(cos.mean())))
    for name in SETS:
        rows, _ = EV.build_rows(tok, name)
        rows = rows[:N_PAIRS]
        full = score(rows, "full")
        r = dict(full=dict(acc=float(full.mean()), agree=1.0))
        for mode in ("identity", "scale", "affine"):
            p = score(rows, mode)
            r[mode] = dict(acc=float(p.mean()), agree=float((p == full).mean()))
        res[name] = r
        print(f"\n[{name}] {len(rows)} pairs — graft {SRC}->{DST}")
        print(f"  {'bridge':<10} {'acc':>6} {'agrees with full':>18}")
        for k, v in r.items():
            print(f"  {k:<10} {v['acc']:6.3f} {v['agree']:18.3f}")
        json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n[bridge] wrote {OUT}")
    print(f"reference: distilled 25.2M head @10 = 0.796 agree (rewardbench2); "
          f"graft 10->14 (no bridge) = 0.908")


if __name__ == "__main__":
    main()
