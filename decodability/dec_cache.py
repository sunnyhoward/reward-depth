#!/usr/bin/env python
"""Activation extraction for the decodability sweep.

ONE forward pass per (model, dataset, render) yields BOTH read protocols at EVERY read point:
  last  -- the final real token (h[:, -1] under left padding)
  mean  -- mean over the COMPLETION tokens only, prompt excluded
Both are cached because read position is not a detail here: it flipped styc corr_e from .776 to
.991 (results_phase8.md:203) and it is the difference between a null and the project's only
steering result (results_phase9.md:134). It is cheap to keep both and dishonest to pick one.

The prompt is excluded from the mean for the reason given at uf_meanpool_sweep.py:23-26 -- pooling
it in dilutes the signal with a constant that is identical on both sides of every pair, and on
short completions that constant dominates.

Sequence features (for the attention rung) are NOT persisted: at 8B, all layers x all texts is
~25 GB per dataset. They are re-extracted per layer on demand instead -- a few thousand short
texts is a couple of minutes of forward passes, far cheaper than the disk.

Usage:  python dec_cache.py <model_key> <dataset> [render]
Env:    DEC_BS=32 MAXLEN=256 RENDER=chat
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_common as C  # noqa: E402
import dec_data as D  # noqa: E402

E = os.environ.get
BS = int(E("DEC_BS", 32))
MAXLEN = int(E("MAXLEN", 256))


def cache_path(model_key, dataset, render="chat"):
    return os.path.join(C.DEC_ROOT, f"feats_{model_key}_{dataset}_{render}.npz")


# ── extraction ────────────────────────────────────────────────────────────────────────────────

def _forward_reads(ctx, rows):
    """rows = [(ids, plen)] → (buf list of (B,T,H) tensors, npad, plens, T).

    buf[k] is read point k: k=0 the embedding output, k>=1 the output of block k-1.
    """
    ids, att, npad, plens = C.left_pad_batch(rows, ctx.tok.pad_token_id, ctx.device, MAXLEN)
    with torch.no_grad(), C.ResidualCapture(ctx.read_mods) as cap:
        ctx.model(input_ids=ids, attention_mask=att)
    buf = cap.get()
    return [buf[k] for k in range(ctx.n_reads)], npad, plens, ids.shape[1]


def _pool(h, npad, plens, T):
    """→ (last(B,H), mean(B,H)) as float32 cpu. `mean` covers completion tokens only.

    Never touches a pad position: the span starts at npad+plen and ends at T. The one-token
    fallback exists so a pathological row (prompt truncated to the full window) degrades to the
    last-token read rather than producing a NaN that would poison a whole layer's fit.
    """
    B = h.shape[0]
    last = h[:, -1].float()
    means = torch.empty_like(last)
    for i in range(B):
        lo = int(npad[i] + plens[i])
        if lo >= T:
            lo = T - 1
        means[i] = h[i, lo:T].float().mean(0)
    return last.cpu(), means.cpu()


def build(model_key, dataset, render="chat", force=False):
    out = cache_path(model_key, dataset, render)
    if os.path.exists(out) and not force:
        print(f"[cache] exists, skipping: {out}", flush=True)
        return out
    d = D.load(dataset)
    ctx = C.load(model_key)
    n = len(d.prompts)
    print(f"[cache] {model_key} x {dataset} x {render}: {n} items, {len(d.variant_names)} variants, "
          f"{ctx.n_reads} read points, hid {ctx.hid}", flush=True)

    store, tok_lens = {}, {}
    for v in d.variant_names:
        last = np.zeros((n, ctx.n_reads, ctx.hid), np.float16)
        mean = np.zeros((n, ctx.n_reads, ctx.hid), np.float16)
        clen = np.zeros(n, np.int32)
        for s in range(0, n, BS):
            sl = slice(s, min(s + BS, n))
            rows = [C.render_ids(ctx, d.prompts[i], d.variants[v][i], render)
                    for i in range(sl.start, sl.stop)]
            clen[sl] = [len(r[0]) - r[1] for r in rows]
            reads, npad, plens, T = _forward_reads(ctx, rows)
            for k in range(ctx.n_reads):
                lo, me = _pool(reads[k], npad, plens, T)
                last[sl, k] = lo.numpy().astype(np.float16)
                mean[sl, k] = me.numpy().astype(np.float16)
            del reads
            if s % (BS * 20) == 0:
                print(f"   {v}: {s}/{n}", flush=True)
        store[f"{v}__last"] = last
        store[f"{v}__mean"] = mean
        tok_lens[v] = clen
        print(f"   cached {v}  (completion tokens: mean {clen.mean():.1f}, max {clen.max()})", flush=True)

    os.makedirs(C.DEC_ROOT, exist_ok=True)
    np.savez(out, **store, **{f"{v}__ntok": tok_lens[v] for v in d.variant_names})
    print(f"[cache] wrote {out}  ({os.path.getsize(out)/1e9:.2f} GB)", flush=True)
    del ctx
    torch.cuda.empty_cache()
    return out


def load_feats(model_key, dataset, render="chat"):
    p = cache_path(model_key, dataset, render)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} -- run: python dec_cache.py {model_key} {dataset} {render}")
    z = np.load(p)
    return {k: z[k] for k in z.files}


# ── on-demand sequence features (attention rung) ──────────────────────────────────────────────

def seq_feats(ctx, d, variant, layer, render="chat", max_tok=64, bs=None):
    """→ (X(n, max_tok, hid) float16, mask(n, max_tok) bool, is_comp(n, max_tok) bool).

    Right-aligned: the LAST max_tok real tokens are kept, so the completion (which is what the
    readout needs) is never the part that gets dropped. `is_comp` marks completion positions so a
    readout can attend over the prompt but pool only over the completion.
    """
    bs = bs or BS
    n = len(d.prompts)
    X = np.zeros((n, max_tok, ctx.hid), np.float16)
    mask = np.zeros((n, max_tok), bool)
    iscomp = np.zeros((n, max_tok), bool)
    for s in range(0, n, bs):
        sl = slice(s, min(s + bs, n))
        rows = [C.render_ids(ctx, d.prompts[i], d.variants[variant][i], render)
                for i in range(sl.start, sl.stop)]
        reads, npad, plens, T = _forward_reads(ctx, rows)
        h = reads[layer]
        keep = min(max_tok, T)
        X[sl, max_tok - keep:] = h[:, T - keep:].float().cpu().numpy().astype(np.float16)
        for j in range(h.shape[0]):
            i = sl.start + j
            real_lo = max(int(npad[j]), T - keep)          # first non-pad kept position
            comp_lo = max(int(npad[j] + plens[j]), T - keep)
            mask[i, max_tok - keep + (real_lo - (T - keep)):] = True
            iscomp[i, max_tok - keep + (comp_lo - (T - keep)):] = True
        del reads
    return X, mask, iscomp


if __name__ == "__main__":
    mk = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
    ds = sys.argv[2] if len(sys.argv) > 2 else "styc"
    rd = sys.argv[3] if len(sys.argv) > 3 else E("RENDER", "chat")
    if ds == "all":
        ctx = None
        for name in D.DATASETS:
            build(mk, name, rd)
    else:
        build(mk, ds, rd)
