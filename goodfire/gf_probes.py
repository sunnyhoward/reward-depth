#!/usr/bin/env python
"""Step 1: per-token linear probes on frozen base activations, at every layer, and the
decodability curve.

One forward pass per completion yields the residual stream at all layers, so fitting 29 probes
costs the same activations as fitting one. This is the property RLFR's frozen-copy design buys
and it is why logging every layer during RL is nearly free.

Labelling is completion-level, broadcast to every completion token. On the minimal pairs this is
deliberately self-contradictory before the first divergent word -- identical prefix activations
carry both labels -- which pins the probe near zero on dialect-neutral tokens and lets it fire
only where the property is actually carried. That is exactly the behaviour the dense reward needs.

Outputs:
  <WORK>/probes.pt           all layers
  results/decodability.json  AUROC by layer (completion-level pooled, token-level, marker-token)

Usage: python gf_probes.py [--max-tok-per-item 48]
"""
import argparse, random
from collections import defaultdict

import numpy as np
import torch

import gf_common as G


def build_items(corpus, sources, oracle=None):
    """-> list of {prompt, text, label, split, axes}"""
    items = []
    if "pairs" in sources:
        for p in corpus["pairs"]:
            for key, lab in (("be", 1), ("ae", 0)):
                items.append({"prompt": p["prompt"], "text": p[key], "label": lab,
                              "split": "heldout" if p["split"] == "validation" else "train",
                              "source": "pairs", "axes": [p["axis"]] if p.get("axis") else []})
    if "gen" in sources:
        for g in corpus["gen"]:
            us, uk = oracle.hits(g["text"]) if oracle else (set(), set())
            items.append({"prompt": g["prompt"], "text": g["text"], "label": g["label"],
                          "split": g["split"], "source": "gen", "axes": sorted(us | uk)})
    return items


def axis_split(items, frac, seed):
    """Re-split by AE/BE axis instead of by row, so the probe is fit on one set of word pairs and
    tested on pairs it has never seen.

    This is the control for the flat decodability curve. AE/BE spelling is tokenizer-level --
    `colour` and `color` are different tokens -- so a probe can hit 0.99 AUROC at the *embedding*
    layer by memorising a word list, and the curve stays flat because there is nothing deeper to
    learn. Held-out axes take memorisation off the table: only a probe that represents dialect
    abstractly transfers. If a depth curve exists at all, it should show up here.

    Test items are minimal pairs only. A generated completion's label is carried by every axis in
    it, so a gen item containing one held-out axis still leaks the rest; those are dropped."""
    axes = sorted({ax for it in items for ax in it["axes"]})
    rs = random.Random(seed); rs.shuffle(axes)
    n_ho = int(round(len(axes) * frac))
    ho = axes[:n_ho]
    val_ax, test_ax = set(ho[:n_ho // 2]), set(ho[n_ho // 2:])
    held = val_ax | test_ax
    out = []
    for it in items:
        ax = set(it["axes"])
        if not ax:
            continue
        if ax & held:
            if it["source"] != "pairs":
                continue                                    # would leak the train axes with it
            it = {**it, "split": "val" if ax & val_ax else "test"}
        else:
            it = {**it, "split": "train"}
        out.append(it)
    return out, sorted(val_ax), sorted(test_ax)


@torch.no_grad()
def collect(model, tok, oracle, items, layers, max_tok, batch=8, device=G.DEV):
    """Forward every item once; keep hidden states at sampled completion positions.
    Returns feats {layer: (N,d) fp16 cpu}, and per-position metadata arrays."""
    feats = {l: [] for l in layers}
    y, item_id, is_be_tok, is_ae_tok = [], [], [], []
    rng = random.Random(0)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["text"]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        pre_ids, full_ids, keeps = [], [], []
        for i in idx:
            it = items[i]
            pre = tok(G.build_prompt(tok, it["prompt"]), add_special_tokens=False)["input_ids"]
            comp = tok(it["text"], add_special_tokens=False)["input_ids"]
            comp = comp[:256]
            be_m, ae_m = oracle.token_labels(tok, it["text"])
            be_m, ae_m = be_m[:len(comp)], ae_m[:len(comp)]
            pos = list(range(len(comp)))
            if len(pos) > max_tok:                    # keep every marker token, sample the rest
                marked = [p for p in pos if be_m[p] or ae_m[p]]
                rest = [p for p in pos if not (be_m[p] or ae_m[p])]
                rng.shuffle(rest)
                pos = sorted(marked + rest[:max(0, max_tok - len(marked))])[:max_tok]
            pre_ids.append(pre); full_ids.append(pre + comp)
            keeps.append((pos, be_m, ae_m, i))
        T = max(len(f) for f in full_ids)
        pad = tok.pad_token_id
        inp = torch.full((len(idx), T), pad, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for b, f in enumerate(full_ids):             # RIGHT-pad: a plain forward takes
            inp[b, :len(f)] = torch.tensor(f)        # position_ids from arange, so left-padding
            att[b, :len(f)] = 1                      # would shift every position silently
        hs = G.hidden_states(model, inp.to(device), att.to(device), layers)
        for b, (pos, be_m, ae_m, i) in enumerate(keeps):
            base = len(pre_ids[b])                   # first completion position in the padded row
            sel = torch.tensor([base + p for p in pos], device=device)
            for l in layers:
                feats[l].append(hs[l][b].index_select(0, sel).half().cpu())
            y += [items[i]["label"]] * len(pos)
            item_id += [i] * len(pos)
            is_be_tok += [be_m[p] for p in pos]
            is_ae_tok += [ae_m[p] for p in pos]
        del hs
    feats = {l: torch.cat(v) for l, v in feats.items()}
    meta = {k: np.asarray(v) for k, v in
            dict(y=y, item=item_id, be_tok=is_be_tok, ae_tok=is_ae_tok).items()}
    return feats, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="pairs,gen")
    ap.add_argument("--max-tok-per-item", type=int, default=48)
    ap.add_argument("--max-train-tokens", type=int, default=120_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--l2-grid", default="1e-5,1e-4,1e-3,1e-2")
    ap.add_argument("--iters", type=int, default=300, help="LBFGS max_iter")
    ap.add_argument("--tag", default="probes")
    ap.add_argument("--axis-holdout", type=float, default=0.0,
                    help="fraction of AE/BE axes held out; tests dialect abstraction, not "
                         "word-list memorisation")
    ap.add_argument("--out", default="decodability.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    G.seed_all(a.seed)
    oracle = G.BritOracle()
    corpus = G.jload(G.RESULTS / "probe_corpus.json")
    items = build_items(corpus, a.sources.split(","), oracle)
    val_ax = test_ax = []
    if a.axis_holdout > 0:
        items, val_ax, test_ax = axis_split(items, a.axis_holdout, a.seed)
        print(f"[axes] holding out {len(val_ax)+len(test_ax)} axes "
              f"({len(val_ax)} val / {len(test_ax)} test), e.g. {test_ax[:6]}", flush=True)
        tr = [i for i in items if i["split"] == "train"]
        ho = [i for i in items if i["split"] in ("val", "test")]
    else:
        tr = [i for i in items if i["split"] == "train"]
        ho = [i for i in items if i["split"] == "heldout"]
    print(f"[data] {len(items)} completions ({len(tr)} train / {len(ho)} heldout); "
          f"BE frac {np.mean([i['label'] for i in items]):.2f}", flush=True)

    tok = G.load_tokenizer()
    model = G.load_base()
    NL = G.n_layers(model)
    layers = list(range(NL + 1))                      # 0 = embeddings, NL = final residual
    print(f"[model] {G.MODEL_ID}: {NL} layers, d={model.config.hidden_size}", flush=True)

    print("[collect] train activations ...", flush=True)
    Xtr, Mtr = collect(model, tok, oracle, tr, layers, a.max_tok_per_item, a.batch)
    print("[collect] heldout activations ...", flush=True)
    Xho, Mho = collect(model, tok, oracle, ho, layers, a.max_tok_per_item, a.batch)
    print(f"[collect] {len(Mtr['y'])} train token-vectors, {len(Mho['y'])} heldout", flush=True)

    if len(Mtr["y"]) > a.max_train_tokens:
        keep = np.random.default_rng(a.seed).choice(len(Mtr["y"]), a.max_train_tokens, replace=False)
        Xtr = {l: v[keep] for l, v in Xtr.items()}
        Mtr = {k: v[keep] for k, v in Mtr.items()}
        print(f"[collect] subsampled to {a.max_train_tokens} train token-vectors", flush=True)

    ho_item_label = {i: it["label"] for i, it in enumerate(ho)}
    # Split held-out completions into val (selects l2) and test (reported), so the regularisation
    # strength is chosen honestly and depth is not confounded with it either. In axis-holdout
    # mode the val/test line is already drawn by axis, so honour that instead of resplitting.
    ho_ids = sorted(ho_item_label)
    if a.axis_holdout > 0:
        val_ids = {i for i in ho_ids if ho[i]["split"] == "val"}
    else:
        rs = np.random.default_rng(a.seed); rs.shuffle(ho_ids)
        val_ids = set(ho_ids[:len(ho_ids) // 2])
    is_val = np.array([int(i) in val_ids for i in Mho["item"]])
    print(f"[split] probe eval: {len(val_ids)} val / {len(ho_ids)-len(val_ids)} test completions",
          flush=True)

    def pooled_auc(z, mask, how="mean"):
        pooled, plab = defaultdict(list), {}
        for zi, ii, m in zip(z, Mho["item"], mask):
            if not m:
                continue
            pooled[int(ii)].append(zi); plab[int(ii)] = ho_item_label[int(ii)]
        keys = sorted(pooled)
        agg = (lambda v: float(np.mean(v))) if how == "mean" else (lambda v: float(v[-1]))
        return G.auroc([agg(pooled[k]) for k in keys], [plab[k] for k in keys])

    l2_grid = [float(x) for x in a.l2_grid.split(",")]
    probes, curve = {}, []
    ytr = torch.tensor(Mtr["y"], dtype=torch.float32)
    Xho_gpu_cache = {}
    for l in layers:
        Xh = Xho[l].float().to(G.DEV)
        best = None
        for l2 in l2_grid:
            p, fit = G.fit_logreg(Xtr[l].float(), ytr, l2=l2, iters=a.iters)
            with torch.no_grad():
                z = p.to(G.DEV).logit(Xh).cpu().numpy()
            v = pooled_auc(z, is_val)
            if best is None or v > best[0]:
                best = (v, l2, p, fit, z)
        val_auc, l2, p, fit, z = best
        probes[l] = p.state()
        with torch.no_grad():
            ztr = p.to(G.DEV).logit(Xtr[l].float().to(G.DEV)).cpu().numpy()
        te = ~is_val
        tok_auc = G.auroc(z[te], Mho["y"][te])
        tok_auc_tr = G.auroc(ztr, Mtr["y"])          # train-vs-heldout gap exposes under/overfit
        pool_auc = pooled_auc(z, te)                 # Libon: mean pooling beats last-token
        last_auc = pooled_auc(z, te, how="last")
        # does the dense signal land on the marker tokens themselves?
        mk = te & ((Mho["be_tok"] == 1) | (Mho["ae_tok"] == 1))
        mark_auc = G.auroc(z[mk], Mho["be_tok"][mk]) if mk.sum() > 10 else float("nan")
        curve.append({"layer": l, "auroc_pooled": pool_auc, "auroc_token": tok_auc,
                      "auroc_last": last_auc, "auroc_marker_token": mark_auc,
                      "auroc_token_train": tok_auc_tr, "auroc_pooled_val": val_auc,
                      "l2": l2, **fit})
        print(f"  L{l:>2}  pooled {pool_auc:.3f}  token {tok_auc:.3f}  last {last_auc:.3f}  "
              f"marker {mark_auc:.3f}  | train-tok {tok_auc_tr:.3f} l2 {l2:g} "
              f"loss {fit['loss']:.4f} |g| {fit['grad_norm']:.1e}", flush=True)
        del Xh

    best = max(curve, key=lambda c: (c["auroc_pooled"] if np.isfinite(c["auroc_pooled"]) else 0))
    print(f"[best] pooled AUROC {best['auroc_pooled']:.3f} at L{best['layer']}", flush=True)

    torch.save({"probes": probes, "model": G.MODEL_ID, "n_layers": NL,
                "sources": a.sources, "curve": curve}, G.WORK / f"{a.tag}.pt")
    print(f"[write] {G.WORK / (a.tag + '.pt')}", flush=True)
    G.jdump({"model": G.MODEL_ID, "n_layers": NL, "sources": a.sources,
             "axis_holdout": a.axis_holdout, "val_axes": val_ax, "test_axes": test_ax,
             "n_train_tokens": int(len(Mtr["y"])), "n_heldout_tokens": int(len(Mho["y"])),
             "n_heldout_completions": len(ho), "best_layer": best["layer"], "curve": curve},
            G.RESULTS / a.out)


if __name__ == "__main__":
    main()
