#!/usr/bin/env python
"""Fit the graft bridge against the OUTPUT DISTRIBUTION, not against a later hidden state.

`sup_bridge.py` fitted W by least squares to reconstruct h_DST from h_SRC, and got held-out
R^2 0.506 but only +0.024 ranking agreement. That objective is the wrong one: h_DST has 2048
dimensions and blocks DST..23 plus the unembedding discard most of them, so a least-squares fit
spends its capacity uniformly over directions the output never reads. The thing that has to be
preserved is what the model would SAY.

So: minimise KL(full model ‖ graft-with-bridge) at sampled token positions, with the bridge the
only trainable tensor. Warm-started from the least-squares solution, which is already halfway
there and makes this a short fit rather than a distillation from scratch.

TRAINED ON THE TASK'S OWN TEXT, AND ON NO LABELS. The corpus is the UF pairs' rendered text — the
distribution this readout will actually score — but the target is the model's own next-token
distribution on it, never the preference label. That keeps the bridge in the same safety class as
the EAGLE head: frozen at stage-1 time, with nothing in the readout that could absorb the install
(phase 1's failure mode). Fitting a readout to preserve RANKING would be the unsafe version and is
deliberately not what this does.

Sizes, for the comparison this is meant to settle:
  distilled EAGLE head   25.2M params, 5000-20000 steps  -> 0.796 agreement (rewardbench2)
  affine bridge (LS)      4.2M params, closed form       -> 0.820
  affine bridge (KL)      4.2M params, this script       -> ?
  graft 10->14, no bridge      0 params                  -> 0.908

Usage: python sup_bridge_fit.py
Env:   SRC=10 DST=18 STEPS=1500 LR=1e-4 BS=4 NPOS=256 RANK=0 (0=full affine, >0 = low-rank+identity)
       N_PAIRS=250 OUT=/workspace/sup_bridge_fit.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "decodability"))
from sup_common import MODEL, DEV, encode, span_mask                  # noqa: E402
from helpers import ResidualCapture                                   # noqa: E402
import sup_eval_pref as EV                                            # noqa: E402

SRC, DST = int(E("SRC", 10)), int(E("DST", 18))
STEPS, LR, BS = int(E("STEPS", 1500)), float(E("LR", 1e-4)), int(E("BS", 4))
NPOS = int(E("NPOS", 256))
RANK = int(E("RANK", 0))
N_PAIRS = int(E("N_PAIRS", 250))
MAX_LEN = int(E("MAX_LEN", 512))
OUT = E("OUT", "/workspace/sup_bridge_fit.json")
LS = E("LS_INIT", "/workspace/sup_bridge_ls.pt")


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
    HID = model.config.get_text_config().hidden_size

    # ---- the task's own text, as the trainer renders it ----
    rows = [json.loads(l) for l in open(f"{HERE}/uf_release/uf.jsonl")]
    train_txt = [r["text_chosen"] for r in rows if not r["reserved_for_eval"]][:4000]
    train_txt += [r["text_rejected"] for r in rows if not r["reserved_for_eval"]][:4000]
    rng = np.random.RandomState(0)
    print(f"[data] {len(train_txt)} task texts (no labels used)", flush=True)

    # ---- bridge ----
    if RANK > 0:
        # identity + low-rank correction: tests whether the fix is a small perturbation
        U = torch.zeros(HID, RANK, device=DEV, dtype=torch.float32).normal_(0, 0.01).requires_grad_(True)
        V = torch.zeros(RANK, HID, device=DEV, dtype=torch.float32).requires_grad_(True)
        b = torch.zeros(HID, device=DEV, dtype=torch.float32).requires_grad_(True)
        params = [U, V, b]
        bridge = lambda h: h + (h.float() @ U @ V).to(h.dtype) + b.to(h.dtype)
        nparam = U.numel() + V.numel() + b.numel()
    else:
        if os.path.exists(LS):
            sd = torch.load(LS, map_location=DEV)
            W = sd["W"].float().to(DEV).requires_grad_(True)
            b = sd["b"].float().to(DEV).requires_grad_(True)
            print("[init] warm-started from the least-squares solution", flush=True)
        else:
            W = torch.eye(HID, device=DEV, dtype=torch.float32).requires_grad_(True)
            b = torch.zeros(HID, device=DEV, dtype=torch.float32).requires_grad_(True)
            print("[init] identity (no LS solution found)", flush=True)
        params = [W, b]
        bridge = lambda h: (h.float() @ W + b).to(h.dtype)
        nparam = W.numel() + b.numel()
    print(f"[bridge] {nparam/1e6:.2f}M trainable params "
          f"(EAGLE head is 25.2M) | graft {SRC}->{DST}, {DST-SRC-1} blocks skipped", flush=True)

    opt = torch.optim.AdamW(params, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR / 20)
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    hist = []

    for step in range(STEPS):
        idx = rng.randint(0, len(train_txt), BS)
        enc = encode(tok, [train_txt[i] for i in idx], max_length=MAX_LEN).to(DEV)
        am = enc["attention_mask"][:, 1:].bool()
        with torch.no_grad():
            with ResidualCapture([BLOCKS[SRC]]) as cap:
                t_logits = model(**enc).logits[:, :-1]
            src = cap.get()[0]
        flat = am.reshape(-1).nonzero(as_tuple=True)[0]
        if len(flat) == 0:
            continue
        pos = flat[torch.randint(0, len(flat), (min(NPOS, len(flat)),), generator=gen, device=DEV)]

        s = bridge(src)

        def hook(mod, args, kwargs, _s=s):
            if args:
                return ((_s,) + args[1:], kwargs)
            kwargs = dict(kwargs); kwargs["hidden_states"] = _s
            return (args, kwargs)

        hdl = BLOCKS[DST].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            s_logits = model(**enc).logits[:, :-1]
        finally:
            hdl.remove()

        V_ = t_logits.shape[-1]
        t = F.log_softmax(t_logits.reshape(-1, V_)[pos].float(), -1)
        st = F.log_softmax(s_logits.reshape(-1, V_)[pos].float(), -1)
        kl = (t.exp() * (t - st)).sum(-1).mean()
        opt.zero_grad(); kl.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sch.step()
        hist.append(float(kl.detach()))
        if (step + 1) % 100 == 0:
            print(f"  step {step+1:5d}: KL {np.mean(hist[-100:]):.4f}", flush=True)

    # ---- score it on the ranking task ----
    with torch.no_grad():
        Wf = [p.detach().clone() for p in params]

    def score(rws, mode):
        A_, B_ = [], []
        for s0 in range(0, len(rws), BS):
            bb = rws[s0:s0 + BS]
            texts = [t for r in bb for t in (r[0], r[1])]
            plens = [r[2] for r in bb for _ in (0, 1)]
            enc = encode(tok, texts, max_length=MAX_LEN).to(DEV)
            msk = span_mask(tok, texts, plens, enc)
            with torch.no_grad():
                if mode == "full":
                    lg = model(**enc).logits[:, :-1]
                else:
                    with ResidualCapture([BLOCKS[SRC]]) as cap:
                        model(**enc)
                    srch = cap.get()[0]
                    if mode == "bridge":
                        srch = bridge(srch)

                    def hk(mod, args, kwargs, _s=srch):
                        if args:
                            return ((_s,) + args[1:], kwargs)
                        kwargs = dict(kwargs); kwargs["hidden_states"] = _s
                        return (args, kwargs)
                    hdl = BLOCKS[DST].register_forward_pre_hook(hk, with_kwargs=True)
                    try:
                        lg = model(**enc).logits[:, :-1]
                    finally:
                        hdl.remove()
                lp = EV._span_logps(lg, enc, msk)
            A_ += lp[0::2].float().cpu().tolist(); B_ += lp[1::2].float().cpu().tolist()
        return np.array(A_) > np.array(B_)

    res = dict(src=SRC, dst=DST, rank=RANK, params=nparam, steps=STEPS,
               kl_first100=float(np.mean(hist[:100])), kl_last100=float(np.mean(hist[-100:])))
    for name in E("EVAL_SETS", "rewardbench2,uf_sup").split(","):
        rws, _ = EV.build_rows(tok, name)
        rws = rws[:N_PAIRS]
        full = score(rws, "full")
        r = {"full": dict(acc=float(full.mean()), agree=1.0)}
        for mode in ("identity", "bridge"):
            p = score(rws, mode)
            r[mode] = dict(acc=float(p.mean()), agree=float((p == full).mean()))
        res[name] = r
        print(f"\n[{name}] {len(rws)} pairs — graft {SRC}->{DST}")
        for k, v in r.items():
            print(f"  {k:<10} acc {v['acc']:.3f}  agrees with full {v['agree']:.3f}")
        json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n[bridge-fit] KL {res['kl_first100']:.3f} -> {res['kl_last100']:.3f}; wrote {OUT}")


if __name__ == "__main__":
    main()
