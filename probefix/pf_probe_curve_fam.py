#!/usr/bin/env python
"""Per-FAMILY read-depth curves, to test whether read depth and write depth line up.

WHY (2026-08-17, after RESULTS_0817_P3.md). P3 showed the two judged families have DIFFERENT write
depth requirements: restricting LoRA to blocks 21-31 costs `style` ~18 judge points (>6 SE) and
`false_friend` nothing (<=1.8 SE). The natural explanation is that register is planned in the lower
stack while a word substitution can be applied near the output. If that is right, the same split
should be visible on the READ side: `style` pairs should need more blocks before a linear probe can
separate them, and `false_friend` pairs should separate early.

WHAT THIS MEASURES THAT pf_probe_curve.py DOES NOT. That script fits ONE probe on all families and
scores each family as a column, so its `install_style` number answers "does the global britishness
direction separate style pairs". This fits a probe PER FAMILY on that family's own training pairs
and scores its own held-out pairs, which is that family's own decodability curve -- the quantity
comparable to a write-depth requirement.

Read index 0 is the embedding output; index i>0 is the output of block i-1 (so trainer attach depth
L == read index L+1), the same convention as pf_probe_curve.py, whose `extract`/`fit` this reuses
verbatim rather than reimplementing.

L* is reported as the first read index reaching 95% of that family's own peak, which is the
definition used across the decodability work.

Env: SUP_BRIT=.../brit_dose20.jsonl  FAMS=false_friend,style  READS=mean,last
     OUT=/workspace/probefix4b/curve_fam.json  MAX_LEN=256 BS=8 L2=1e-3 FIT_STEPS=400 SEED=0
"""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_common import MODEL, DEV, load_split                      # noqa: E402
from pf_probe_curve import extract, fit, READS, SEED              # noqa: E402

E = os.environ.get
FAMS = [f for f in E("FAMS", "false_friend,style").split(",") if f]
OUT = E("OUT", "/workspace/probefix4b/curve_fam.json")
VAL_FRAC = float(E("VAL_FRAC", 0.25))     # held-out share when the release's own split is thin


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    NB = len(model.model.layers)

    all_tr, all_va = load_split("train"), load_split("validation")
    res = dict(model=MODEL, brit=E("SUP_BRIT", "release"), n_blocks=NB, families={},
               note="read 0 = embeddings; read i>0 = output of block i-1; attach depth L = read L+1")
    t0 = time.time()

    for fam in FAMS:
        tr = [r for r in all_tr if r.get("family") == fam and r.get("role") == "install"]
        va = [r for r in all_va if r.get("family") == fam and r.get("role") == "install"]
        # the release reserves few rows per family; top up the held-out set from train by ROW, and
        # never let a row appear on both sides
        if len(va) < 100:
            k = int(len(tr) * VAL_FRAC)
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(len(tr))
            va = va + [tr[i] for i in perm[:k]]
            tr = [tr[i] for i in perm[k:]]
        print(f"[fam] {fam}: train {len(tr)} | held-out {len(va)}", flush=True)
        if len(tr) < 40 or len(va) < 20:
            print(f"[fam] {fam}: too few rows, skipping", flush=True)
            continue

        Xtr = extract(model, tok, tr, NB)
        Xva = extract(model, tok, va, NB)
        for k in READS:
            curve = []
            for L in range(NB + 1):
                _, tr_acc, acc = fit(Xtr[k][:, L], {"held_out": Xva[k][:, L]}, seed=SEED)
                curve.append(dict(read=L, train=tr_acc, held_out=acc["held_out"]))
            accs = [c["held_out"] for c in curve]
            peak = max(accs)
            lstar = next(i for i, a in enumerate(accs) if a >= 0.95 * peak)
            res["families"].setdefault(fam, {})[k] = dict(
                curve=curve, peak=peak, peak_read=int(np.argmax(accs)), lstar_read=lstar,
                lstar_over_depth=lstar / (NB + 1), read0=accs[0], n_train=len(tr), n_val=len(va))
            print(f"  {fam:13s} {k:<5} L0 {accs[0]:.3f} | peak {peak:.3f} @ read "
                  f"{int(np.argmax(accs))} | L* read {lstar} ({lstar/(NB+1):.2f} of depth)",
                  flush=True)

    res["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[curve-fam] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
