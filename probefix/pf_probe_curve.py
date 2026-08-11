#!/usr/bin/env python
"""STAGE 0 -- the premise. Per-layer linear decodability of the DOSED britishness preference on
the model that will actually be trained (Qwen3.5-2B), on the split that will actually be scored.

WHY THIS HAS TO BE RUN RATHER THAN LOOKED UP. Every britishness decodability curve in the repo is
either (a) a different model family -- `dec_cache/guarddose_qwen3-*.json` covers qwen3-0.6b/1.7b/
4b/8b, none of them Qwen3.5-2B -- or (b) an install-only diet, no guard. The whole experiment is
"the preference is decodable by L*, so the features are present and only the decoder needs
fixing"; L* has to be measured on this model and this diet or the attach depth is a guess.

THE PROBE IS THE ONE THE TRAINER USES: a single direction `w` applied to the RMS-normalised block
output, no bias (it cancels in the pairwise difference), fitted on `read(chosen) - read(rejected)`
with sign augmentation. So the number this reports is exactly the quantity stage 1 optimises, and
"the probe already reads 0.9x at step 0" is a statement about the same object.

BOTH POOLING READS are fitted (`mean` over the assistant turn, `last` scored position). The
trainer defaults to `mean`: a single-token read is the most forgeable target in the repo's
history, and the mean read spreads the gradient over the whole completion.

Scored per eval bucket, all from ONE fitted probe per (read point, pooling), never refitted per
column -- otherwise the guard and install numbers describe different probes and the trade between
them is invisible (`brit_guard_dose.py` makes the same call).

Usage:  SUP_BRIT=.../brit_dose20.jsonl python pf_probe_curve.py
Env:    OUT=/workspace/probefix/curve_dose20.json  MAX_LEN=256  BS=8  L2=1e-3  FIT_STEPS=400
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_common import (MODEL, DEV, ResidualCapture, buckets, encode,   # noqa: E402
                       inner, load_split, pair_texts, span_mask, span_read)

E = os.environ.get
MAXLEN = int(E("MAX_LEN", 256))
BS = int(E("BS", 8))
L2 = float(E("L2", 1e-3))
FIT_STEPS = int(E("FIT_STEPS", 400))
SEED = int(E("SEED", 0))
OUT = E("OUT", "/workspace/probefix/curve.json")
READS = E("READS", "mean,last").split(",")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.manual_seed(SEED)


def extract(model, tok, rows, n_blocks, kinds=None):
    """-> {read_kind: (n, n_reads, hid) float16 diffs}. Read index 0 is the embedding output,
    index i>0 is the output of block i-1; so trainer attach depth L == read index L+1."""
    kinds = kinds or READS
    out = {k: [] for k in kinds}
    body = inner(model)
    blocks = list(body.layers)
    emb = body.embed_tokens
    with torch.no_grad():
        for s in range(0, len(rows), BS):
            trip = pair_texts(tok, rows[s:s + BS])
            texts = [t for c, j, _ in trip for t in (c, j)]
            plens = [pl for _, _, pl in trip for _ in (0, 1)]
            enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
            m = span_mask(tok, texts, plens, enc)
            with ResidualCapture([emb] + blocks) as cap:
                model(**enc)
            got = cap.get()
            for k in kinds:
                per = [span_read(got[i], m, k) for i in range(n_blocks + 1)]
                v = torch.stack(per, 1)                      # (2n, n_reads, hid)
                out[k].append((v[0::2] - v[1::2]).half().cpu())
            del got
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def fit(Xtr, Xte_cols, seed=0):
    """Logistic probe on pairwise differences, label +1 by construction (sign-symmetric loss, no
    bias). -> (w, train_acc, {col: acc}). Full-batch Adam; the fit is 2048-dim on ~5k rows, so the
    L2 is what stops it, not the step count."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.as_tensor(Xtr, dtype=torch.float32, device=DEV)
    X = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-6).mean()      # scale only, keeps L2 sane
    w = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
    with torch.no_grad():
        w += 1e-4 * torch.randn(X.shape[1], generator=g).to(DEV)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(FIT_STEPS):
        opt.zero_grad()
        loss = -F.logsigmoid(X @ w).mean() + L2 * w.pow(2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr = float(((X @ w) > 0).float().mean())
        cols = {}
        for name, Xc in Xte_cols.items():
            if len(Xc) == 0:
                continue
            B = torch.as_tensor(Xc, dtype=torch.float32, device=DEV)
            z = B @ w
            cols[name] = float(((z > 0).float() + 0.5 * (z == 0).float()).mean())
    return w.detach().cpu().numpy(), tr, cols


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    NB = len(model.model.layers)

    tr_rows, va_rows = load_split("train"), load_split("validation")
    bk = buckets(va_rows)
    print(f"[curve] {MODEL} blocks={NB} | train {len(tr_rows)} | val "
          + " ".join(f"{k}:{len(v)}" for k, v in sorted(bk.items())), flush=True)

    t0 = time.time()
    Xtr = extract(model, tok, tr_rows, NB)
    Xva = extract(model, tok, va_rows, NB)
    print(f"[curve] features in {time.time()-t0:.0f}s", flush=True)
    # Xva rows follow val order; reorder them into contiguous bucket blocks so a bucket is a slice.
    order = list(sorted(bk))
    pos = {id(r): i for i, r in enumerate(va_rows)}
    perm, idx, at = [], {}, 0
    for k in order:
        idx[k] = (at, at + len(bk[k])); at += len(bk[k])
        perm += [pos[id(r)] for r in bk[k]]
    perm = np.array(perm)

    res = dict(model=MODEL, brit=os.environ.get("SUP_BRIT", "release"), n_blocks=NB,
               n_train=len(tr_rows), n_val={k: len(v) for k, v in bk.items()},
               note=("read index 0 = embedding output; index i>0 = output of block i-1, so "
                     "trainer attach depth L == read index L+1"),
               curves={})
    for k in READS:
        A, B = Xtr[k], Xva[k][perm]
        pooled = [i for c in order if c != "legacy" for i in range(*idx[c])]
        for L in range(NB + 1):
            cols = {c: B[idx[c][0]:idx[c][1], L] for c in order}
            cols["pooled_new"] = B[pooled, L]
            _, tr_acc, acc = fit(A[:, L], cols, seed=SEED)
            res["curves"].setdefault(k, []).append(dict(read=L, train=tr_acc, **acc))
            print(f"  {k:<5} read {L:>2} (block {L-1:>2}): train {tr_acc:.3f} | "
                  + " ".join(f"{c[:14]} {acc[c]:.3f}" for c in ("pooled_new", "guard", "legacy")
                             if c in acc), flush=True)
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[curve] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
