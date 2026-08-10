#!/usr/bin/env python
"""Does an UNTRAINED layer-skip readout beat a distilled 25.2M head at RANKING the pairs?

The heads were distilled to reconstruct the model's whole next-token distribution over generic
chat replay — the hardest possible ask, and not the one stage-1 DPO makes. Stage 1 only needs the
readout to order chosen vs rejected the way the full model would. `sup_lens.py` showed that
grafting the model's OWN blocks K..23 onto the residual entering block L beats the distilled head
on general agreement with no training at all (10->16: 0.483 vs the head's 0.404). This asks the
question that actually matters: how do the two compare at ranking?

Three readouts, all frozen, all scoring the same pairs:
  FULL     all 24 blocks — the reference the readout is trying to stand in for
  HEAD     the distilled EAGLE head at block L (what the recipe uses)
  GRAFT    residual entering block L injected as the input of block K, then the model's own
           blocks K..23 — no parameters, no training, no bridge

and the metric that matters is not raw accuracy (which on UF is length-dominated: the full model
itself sits at 0.432, BELOW chance) but **ordering agreement with the full model** — for what
fraction of pairs does the readout put the same side first? That is the honest statement of "can
this readout stand in for the model on this task", and it is the number `sup_lens.py`'s general
`agree` was a proxy for.

Usage: python sup_graft_rank.py
Env:   SUP_LAYER=10 GRAFT_KS=14,16,18 N_PAIRS=250 BS=4 MAX_LEN=512
       EVAL_SETS=uf_sup,rewardbench2  OUT=/workspace/sup_graft_rank.json
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
sys.path.insert(0, os.path.join(HERE, "..", "eagle"))
sys.path.insert(0, os.path.join(HERE, "..", "decodability"))

from sup_common import MODEL, DEV, LAYER, encode, span_mask                 # noqa: E402
from eagle_common import make_head                                          # noqa: E402
from helpers import ResidualCapture                                         # noqa: E402
import sup_eval_pref as EV                                                  # noqa: E402

GRAFT_KS = [int(x) for x in E("GRAFT_KS", "14,16,18").split(",")]
N_PAIRS = int(E("N_PAIRS", 250))
BS = int(E("BS", 4))
MAX_LEN = int(E("MAX_LEN", 512))
SETS = E("EVAL_SETS", "uf_sup,rewardbench2").split(",")
OUT = E("OUT", "/workspace/sup_graft_rank.json")


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

    hp = f"/workspace/sup/head_tf_L{LAYER}_b5000.pt"
    if not os.path.exists(hp):
        hp = f"/workspace/sup/head_tf_L{LAYER}.pt"
    head = make_head(HID, "tf").to(DEV)
    head.load_state_dict(torch.load(hp, map_location=DEV))
    print(f"[head] {hp}", flush=True)

    def logps_from_logits(logits, enc, m):
        return EV._span_logps(logits, enc, m)

    def score(rows, mode, K=None):
        """→ (lp_chosen, lp_rejected) arrays under one readout."""
        A, B = [], []
        for s in range(0, len(rows), BS):
            b = rows[s:s + BS]
            texts = [t for r in b for t in (r[0], r[1])]
            plens = [r[2] for r in b for _ in (0, 1)]
            enc = encode(tok, texts, max_length=MAX_LEN).to(DEV)
            m = span_mask(tok, texts, plens, enc)
            with torch.no_grad():
                if mode == "full":
                    lg = model(**enc).logits[:, :-1]
                elif mode == "head":
                    with ResidualCapture([BLOCKS[LAYER]]) as cap:
                        model(**enc)
                    lg = head(cap.get()[0][:, :-1], model)
                else:                                    # graft
                    # reads[L] from a capture on block L-1's OUTPUT == the residual entering L.
                    with ResidualCapture([BLOCKS[LAYER - 1]]) as cap:
                        model(**enc)
                    src = cap.get()[0]

                    def pre_hook(mod, args, kwargs, _s=src):
                        if args:
                            return ((_s,) + args[1:], kwargs)
                        kwargs = dict(kwargs); kwargs["hidden_states"] = _s
                        return (args, kwargs)

                    hdl = BLOCKS[K].register_forward_pre_hook(pre_hook, with_kwargs=True)
                    try:
                        lg = model(**enc).logits[:, :-1]
                    finally:
                        hdl.remove()
                lp = logps_from_logits(lg, enc, m)
            A += lp[0::2].float().cpu().tolist()
            B += lp[1::2].float().cpu().tolist()
        return np.array(A), np.array(B)

    out = {}
    for name in SETS:
        rows, dropped = EV.build_rows(tok, name)
        rows = rows[:N_PAIRS]
        nc = np.array([r[4] for r in rows], float)
        nr = np.array([r[5] for r in rows], float)
        print(f"\n[{name}] {len(rows)} pairs", flush=True)

        fa, fb = score(rows, "full")
        full_pick = fa > fb
        res = {"full": dict(acc=float(full_pick.mean()), agree_with_full=1.0)}

        ha, hb = score(rows, "head")
        hp_ = ha > hb
        res[f"head@{LAYER}"] = dict(acc=float(hp_.mean()),
                                    agree_with_full=float((hp_ == full_pick).mean()))
        for K in GRAFT_KS:
            if K <= LAYER:
                continue
            ga, gb = score(rows, "graft", K)
            gp = ga > gb
            res[f"graft {LAYER}->{K}"] = dict(acc=float(gp.mean()),
                                              agree_with_full=float((gp == full_pick).mean()),
                                              blocks_skipped=K - LAYER)
        out[name] = res
        print(f"  {'readout':<18} {'acc':>6} {'agrees with full model':>24}")
        for k, v in res.items():
            print(f"  {k:<18} {v['acc']:6.3f} {v['agree_with_full']:24.3f}")
        json.dump(out, open(OUT, "w"), indent=1)

    print(f"\n[graft-rank] wrote {OUT}")
    print("head@10 general agreement was 0.404 (5000 steps); graft 10->16 general agreement 0.483")


if __name__ == "__main__":
    main()
