#!/usr/bin/env python
"""Family B step 2: score preference pairs THROUGH the distilled heads. No fitting, anywhere.

For each pair the head at layer L produces next-token logits, and the pair is decided by
    sum_t log p_head(completion_t | prefix)  for chosen  vs  rejected.
Nothing is fitted on the preference data -- the head was distilled on generative replay only
(dec_distill.py) and is frozen here. So this is an ENCODING measure: "is the preference
expressible through the base model's own unembedding, given h_L", which is the question
eagle/RESULTS.md:359-363 raises with "probes can read what LM heads cannot say".

WHAT IS REPORTED PER CELL
  acc            -- held-out pairwise accuracy (same held-out groups as family A)
  acc_all        -- all pairs, for reference
  margin         -- mean per-token logp gap, so a cell that is barely deciding is visible
  kl_head        -- KL(base || head) on held-out replay text: THE COMPETENCE COVARIATE
  top1_agree     -- top-1 agreement with the base on the same text
The competence numbers are not decoration. results_0805.md:199-201 says every depth claim in
this repo is currently blocked because head competence co-varies with depth (.152/.202/.380 at
L4/L12/L24) and tracks two supposedly-depth results. A depth curve that is really a competence
curve must be visible as such in the same table, which is why both are emitted per cell.

The base model's own full-stack accuracy is measured too, as the reference ceiling that every
head is trying to reach.

Usage: python dec_through.py <model_key> <arch>[,<arch>...] [dataset|all]
Env:   THRU_BS=8 RENDER=chat
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_common as C  # noqa: E402
import dec_data as D  # noqa: E402
import dec_distill as DI  # noqa: E402
from dec_heads import DEC_ARCHS, make_dec_head, n_params  # noqa: E402

E = os.environ.get
BS = int(E("THRU_BS", 8))
RENDER = E("RENDER", "chat")
CHUNK = int(E("THRU_CHUNK", 32))


def _sum_logp(logits, ids, plens, npad):
    """Σ log p(token_t | <t) over COMPLETION positions only. → (B,) float, and the token count.

    Left padding, so position j predicts token j+1. The completion occupies [npad+plen, T), which
    means the predicting positions are [npad+plen-1, T-1). Pad positions are never scored, which
    is the same discipline the pad-read bug taught the hard way (results_phase3.md:30-40).
    """
    B, T, _ = logits.shape
    tot = torch.zeros(B, device=logits.device, dtype=torch.float32)
    cnt = torch.zeros(B, device=logits.device, dtype=torch.float32)
    for s in range(0, T - 1, CHUNK):
        e = min(s + CHUNK, T - 1)
        lp = F.log_softmax(logits[:, s:e].float(), -1)
        tgt = ids[:, s + 1:e + 1]
        got = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)          # (B, e-s)
        pos = torch.arange(s, e, device=logits.device)[None, :]
        keep = (pos >= (npad + plens - 1)[:, None]) & (pos < T - 1)
        tot += (got * keep).sum(-1)
        cnt += keep.float().sum(-1)
    return tot, cnt


def _rows_for(ctx, d, idxs, variant):
    return [C.render_ids(ctx, d.prompts[i], d.variants[variant][i], RENDER) for i in idxs]


def _score_side(ctx, rows, heads_by_layer):
    """→ dict layer -> (sum_logp(B,), ntok(B,)), plus "base" for the full model."""
    ids, att, npad, plens = C.left_pad_batch(rows, ctx.tok.pad_token_id, ctx.device)
    with torch.no_grad(), C.ResidualCapture(ctx.read_mods) as cap:
        base_logits = ctx.model(input_ids=ids, attention_mask=att).logits
    buf = cap.get()
    out = {}
    with torch.no_grad():
        out["base"] = _sum_logp(base_logits, ids, plens, npad)
        del base_logits
        for L, head in heads_by_layer.items():
            out[L] = _sum_logp(head(buf[L], ctx.model, pad_mask=att), ids, plens, npad)
    del buf
    return out


def competence(ctx, heads_by_layer, corpus, n=64, seed=0):
    """KL(base||head) and top-1 agreement on held-out replay rows. THE depth-vs-competence control."""
    g = torch.Generator().manual_seed(seed + 4242)
    n = min(n, corpus.shape[0])          # a short corpus must shrink n, not emit empty batches
    sel = torch.randperm(corpus.shape[0], generator=g)[:n]
    kl = {L: [] for L in heads_by_layer}
    ag = {L: [] for L in heads_by_layer}
    for s in range(0, n, BS):
        ids = corpus[sel[s:s + BS]].to(ctx.device)
        att = torch.ones_like(ids)
        with torch.no_grad(), C.ResidualCapture(ctx.read_mods) as cap:
            tl = ctx.model(input_ids=ids, attention_mask=att).logits
        buf = cap.get()
        with torch.no_grad():
            for L, head in heads_by_layer.items():
                sl = head(buf[L], ctx.model, pad_mask=att)
                k = 0.0
                T = tl.shape[1]
                for c in range(0, T, CHUNK):
                    lt = F.log_softmax(tl[:, c:c + CHUNK].float(), -1)
                    ls = F.log_softmax(sl[:, c:c + CHUNK].float(), -1)
                    k += float((lt.exp() * (lt - ls)).sum(-1).mean()) * (min(CHUNK, T - c) / T)
                kl[L].append(k)
                ag[L].append(float((sl.argmax(-1) == tl.argmax(-1)).float().mean()))
                del sl
        del tl, buf
    return ({L: float(np.mean(v)) for L, v in kl.items()},
            {L: float(np.mean(v)) for L, v in ag.items()})


def run(model_key, arch, dataset, ctx=None, corpus=None):
    d = D.load(dataset)
    own_ctx = ctx is None
    ctx = ctx or C.load(model_key)
    for p in ctx.model.parameters():
        p.requires_grad_(False)
    layers = C.layer_grid(ctx.n_layers)

    heads = {}
    for L in layers:
        p = DI.head_path(model_key, arch, L)
        if not os.path.exists(p):
            print(f"[warn] missing head {p} -- run dec_distill.py first", flush=True)
            continue
        blob = torch.load(p, weights_only=False)
        h = make_dec_head(arch, ctx.hid, n_heads=ctx.n_heads, dtype=ctx.dtype,
                          vocab=ctx.model.config.vocab_size).to(ctx.device)
        if hasattr(h, "attach_lm_head"):
            h.attach_lm_head(ctx.model)
        h.load_state_dict(blob["state"])
        h.eval()
        heads[L] = h
    if not heads:
        return None

    corpus = corpus if corpus is not None else DI.build_replay(ctx)
    kl, ag = competence(ctx, heads, corpus)
    print(f"[competence] {arch}: " + "  ".join(f"L{L} KL={kl[L]:.3f}/agree={ag[L]:.2f}"
                                               for L in sorted(heads)), flush=True)

    n = len(d.prompts)
    keys = list(heads) + ["base"]
    lp = {v: {k: np.zeros(n, np.float64) for k in keys} for v in d.variant_names}
    nt = {v: np.zeros(n, np.float64) for v in d.variant_names}
    t0 = time.time()
    for v in d.variant_names:
        for s in range(0, n, BS):
            idxs = list(range(s, min(s + BS, n)))
            sc = _score_side(ctx, _rows_for(ctx, d, idxs, v), heads)
            for k in keys:
                lp[v][k][idxs] = sc[k][0].cpu().numpy()
            nt[v][idxs] = sc[list(sc)[0]][1].cpu().numpy()
        print(f"   scored {v} ({time.time()-t0:.0f}s)", flush=True)

    res = {}
    for fam in d.families:
        rows = [(i, va, vb) for i, va, vb, f in d.pairs if f == fam]
        te = [r for r in rows if d.split[r[0]] == "test"]
        if len(te) < 20:
            continue

        def gaps(rr, k, per_token):
            """logp(preferred) - logp(dispreferred), optionally per completion token."""
            if per_token:
                return np.array([lp[va][k][i] / max(nt[va][i], 1) - lp[vb][k][i] / max(nt[vb][i], 1)
                                 for i, va, vb in rr])
            return np.array([lp[va][k][i] - lp[vb][k][i] for i, va, vb in rr])

        base_g = gaps(te, "base", False)
        for k in keys:
            g, gt = gaps(te, k, False), gaps(te, k, True)
            # THREE numbers, because the naive one is not a decodability measure.
            #  acc         -- sign of the summed-logp gap. DOMINATED BY LENGTH whenever the two
            #                 sides differ in length: styc style_c (prefers the long completion)
            #                 reads 0.000 and conflict (prefers the short one) reads 1.000 at
            #                 EVERY layer AND at the full base model. It also encodes the model's
            #                 PRIOR, not its decodability -- brit reads ~0.15 because the base is
            #                 American-default, exactly as eagle/RESULTS.md:6 says.
            #  acc_pertok  -- the same with logp averaged over completion tokens, which removes
            #                 the gross length term.
            #  acc_vs_base -- does the head at L rank the pair the SAME WAY the full model does?
            #                 This is the depth measure that survives both confounds: the prior
            #                 and the length bias are shared by the reference, so what is left is
            #                 "how much of the model's own final ordering is already expressible
            #                 at layer L". Same logic as the reference-corrected implicit
            #                 preference in eagle/brit_heldout.py.
            res[f"{fam}|{k}"] = dict(
                acc=float(((g > 0) + 0.5 * (g == 0)).mean()),
                acc_pertok=float(((gt > 0) + 0.5 * (gt == 0)).mean()),
                acc_vs_base=float(((np.sign(g) == np.sign(base_g)) +
                                   0.5 * (np.sign(g) == 0)).mean()),
                corr_vs_base=(float(np.corrcoef(g, base_g)[0, 1])
                              if g.std() > 0 and base_g.std() > 0 else None),
                margin=float(np.mean(np.abs(g))), n_test=len(te),
                kl_head=(None if k == "base" else kl[k]),
                top1_agree=(None if k == "base" else ag[k]))
        line = "  ".join(f"L{k}:{res[f'{fam}|{k}']['acc_vs_base']:.3f}" for k in heads)
        print(f"   [{fam}] base_pref={res[f'{fam}|base']['acc']:.3f} "
              f"(pertok {res[f'{fam}|base']['acc_pertok']:.3f})  agrees-with-base: {line}",
              flush=True)

    out = dict(model=model_key, arch=arch, dataset=dataset, render=RENDER, layers=sorted(heads),
               n_params=n_params(list(heads.values())[0]), results=res,
               kl_head={str(k): v for k, v in kl.items()},
               top1_agree={str(k): v for k, v in ag.items()},
               note="frozen heads distilled on generative replay only; ZERO preference fitting")
    C.bank(f"through_{model_key}_{arch}_{dataset}_{RENDER}", out)
    if own_ctx:
        del ctx
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    mk = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
    archs = (sys.argv[2] if len(sys.argv) > 2 else "eagle-mlp,eagle-tf").split(",")
    ds = sys.argv[3] if len(sys.argv) > 3 else "all"
    for a in archs:
        assert a in DEC_ARCHS, f"unknown arch {a}"
    ctx = C.load(mk)
    for p in ctx.model.parameters():
        p.requires_grad_(False)
    corpus = DI.build_replay(ctx)
    for a in archs:
        for name in (D.DATASETS if ds == "all" else [ds]):
            run(mk, a, name, ctx=ctx, corpus=corpus)
