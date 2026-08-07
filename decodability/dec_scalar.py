#!/usr/bin/env python
"""Family A: scalar readouts x depth x read-position, plus the two mandatory controls.

WHAT IS MEASURED. For every (read point, read protocol, readout rung, pair family): held-out
pairwise accuracy, where held-out means a GROUP the fit never saw (a question, or an am|br axis).

FITTING CONVENTION (matches styc_probe.py:133 and helpers.fit_probes). Heads are fit on the
signed difference feature (preferred - dispreferred), standardised per layer by the pooled train
std, with an implicit +1 target. Every rung is exactly antisymmetric -- linear trivially,
AntisymMLP by construction, AttnScalar because it scores each side and subtracts -- so no rung can
manufacture accuracy from a constant offset, and chance is exactly 0.5.

THE TWO CONTROLS ARE NOT OPTIONAL.
  lexical floor  -- a logistic probe on BAG-OF-TOKEN-IDS of the completion, no model involved.
     This is the reference line every model curve must clear to mean anything. goodfire/RESULTS.md
     measured AUROC 0.988 at the embedding layer on the brit axis and 0.992 with 30% of axes held
     out, i.e. the "layer 0" result is a sub-token regularity (-ise/-ize), not memorisation. If a
     dataset's lexical floor is ~1.0, no depth statement about that dataset means anything.
  shuffled       -- the same rung refit with each pair's difference randomly sign-flipped while
     the target stays +1. An antisymmetric readout cannot solve that, so anything above ~0.5 is
     capacity finding structure in noise. MLP and attention rungs fit thousands of parameters on
     a few hundred pairs; without this control their numbers are uninterpretable.

Usage: python dec_scalar.py <model_key> [dataset|all] [render]
Env:   DEC_SEEDS=0,1,2  DEC_RUNGS=linear,mlp,attn  DEC_READS=last,mean  MLP_HID=64
       ATTN_MAXTOK=64  ATTN_EPOCHS=200
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dec_cache as K  # noqa: E402
import dec_common as C  # noqa: E402
import dec_data as D  # noqa: E402
import dec_fit as FIT  # noqa: E402
from dec_heads import AntisymMLP, AttnScalar  # noqa: E402
from helpers import train_bayes_head  # noqa: E402

E = os.environ.get
SEEDS = [int(x) for x in E("DEC_SEEDS", "0,1,2").split(",")]
RUNGS = E("DEC_RUNGS", "linear,mlp").split(",")
READS = E("DEC_READS", "last,mean").split(",")
MLP_HID = int(E("MLP_HID", 64))
# These fits are small (d = hid, N ~ 500) and there are ~1k of them per cell, so they belong on
# the card, not on oversubscribed BLAS threads. FIT_DEV=cpu reproduces the original behaviour.
FIT_DEV = E("FIT_DEV", "cuda" if torch.cuda.is_available() else "cpu")
ATTN_MAXTOK = int(E("ATTN_MAXTOK", 64))
ATTN_EPOCHS = int(E("ATTN_EPOCHS", 200))
SEQ_ARCHS = E("SEQ_ARCHS", "attn,seq-tf,seq-2l").split(",")

# Cross-family transfer diagnostics: fit on one set of families, score on another. These are the
# cells where a purely lexical reader is supposed to FAIL, so they carry most of the signal.
DIAGNOSTICS = {
    # styc: the classic conflict test (styc_probe.py DIET) -- a head that never saw a pair where
    # correctness and style disagree, asked to resolve one. Banked at 0.000 at every layer before.
    "styc": [("diet_to_conflict",
              ["corr_e", "corr_t", "style_c", "style_w", "aligned"], ["conflict"])],
    # brit_truth: fit the dialect preference, then score the guard where preferring British is
    # WRONG. A reader that has only learned "prefer British markers" must land at/below chance.
    "brit_truth": [("dialect_to_guard",
                    ["true_british_over_american", "false_british_over_american"],
                    ["truth_over_british"])],
}


# ── pair assembly ─────────────────────────────────────────────────────────────────────────────

def _pair_idx(d, families):
    """→ (train_rows, test_rows) as lists of (item, pos_variant, neg_variant)."""
    tr, te = [], []
    for i, va, vb, fam in d.pairs:
        if fam not in families:
            continue
        (tr if d.split[i] == "train" else te).append((i, va, vb))
    return tr, te


def _diffs(feats, rows, read, layer):
    """Signed difference features (preferred - dispreferred) at one read point."""
    A = np.stack([feats[f"{va}__{read}"][i, layer] for i, va, _ in rows]).astype(np.float32)
    B = np.stack([feats[f"{vb}__{read}"][i, layer] for i, _, vb in rows]).astype(np.float32)
    return A, B


def _prep(A_tr, B_tr, A_te, B_te, shuffle_seed=None):
    """Standardise by pooled TRAIN std, return (DF_tr, t_tr, DF_te, t_te).

    Standardisation uses train statistics only -- using the pooled test std would leak the test
    distribution into the fit, which at hid=4096 with a few hundred pairs is not a small leak.
    """
    sd = np.concatenate([A_tr, B_tr]).std(0) + 1e-6
    DF_tr, DF_te = (A_tr - B_tr) / sd, (A_te - B_te) / sd
    t_tr = np.ones(len(DF_tr), np.float32)
    t_te = np.ones(len(DF_te), np.float32)
    if shuffle_seed is not None:
        rng = np.random.RandomState(shuffle_seed)
        DF_tr = DF_tr * rng.choice([-1.0, 1.0], len(DF_tr))[:, None].astype(np.float32)
        DF_te = DF_te * rng.choice([-1.0, 1.0], len(DF_te))[:, None].astype(np.float32)
    return DF_tr, t_tr, DF_te, t_te


# ── rungs ─────────────────────────────────────────────────────────────────────────────────────

def _acc_ties(z, t):
    """Ranking accuracy with EXACT ties scored 0.5. → (acc, tie_fraction).

    This is not pedantry. At the embedding read point the last token of both completions is
    frequently the SAME token (both sides end "...living."), so the difference vector is exactly
    zero and z is exactly 0. `z > 0` scores every such pair WRONG, which printed L0 accuracy as
    0.000 -- a number that looks like a strong inverted signal and is actually no signal at all.
    The tie fraction is reported alongside so a degenerate read is visible rather than inferred.
    """
    z = np.asarray(z, np.float64)
    t = np.asarray(t, np.float64)
    tie = z == 0
    win = (z * t > 0).astype(np.float64)
    win[tie] = 0.5
    return float(win.mean()), float(tie.mean())


def fit_linear(DF_tr, t_tr, DF_te, t_te, seed):
    """helpers.train_bayes_head, but run on FIT_DEV.

    The upstream fitter creates its tensors with bare `torch.tensor(...)`, so it lands on CPU --
    fine for the one-off sweeps it was written for, badly wrong here: this sweep is ~1k fits per
    cell and 64 cores of oversubscribed BLAS is an order of magnitude slower than the card that
    is otherwise idle. `torch.device` as a context manager redirects the factory calls inside the
    unmodified function, so the fitter itself stays byte-identical to the one every earlier result
    in this repo used.
    """
    with torch.device(FIT_DEV):
        _, head, _ = train_bayes_head(DF_tr, t_tr, DF_te, t_te, seed=seed)
        with torch.no_grad():
            z = head.z_s2(torch.tensor(DF_te * t_te[:, None], dtype=torch.float32))[0].cpu().numpy()
    acc, tie = _acc_ties(z, np.ones_like(t_te))
    return acc, tie, head


def fit_mlp(DF_tr, t_tr, DF_te, t_te, seed):
    """AntisymMLP with early stopping on the held-out probit loss (styc_mlp_head.py:60-77)."""
    torch.manual_seed(seed)
    X = torch.tensor(DF_tr, device=FIT_DEV)
    Xv = torch.tensor(DF_te, device=FIT_DEV)
    y = torch.tensor(t_tr, device=FIT_DEV)
    yv = torch.tensor(t_te, device=FIT_DEV)
    net = AntisymMLP(X.shape[1], MLP_HID).to(FIT_DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best = dict(loss=1e9, wait=0, state=None)
    for _ in range(300):
        for sl in torch.randperm(len(X), device=FIT_DEV).split(256):
            opt.zero_grad()
            F.softplus(-net(X[sl]) * y[sl]).mean().backward()
            opt.step()
        with torch.no_grad():
            vl = float(F.softplus(-net(Xv) * yv).mean())
        if vl < best["loss"] - 1e-4:
            best.update(loss=vl, wait=0, state={k: v.clone() for k, v in net.state_dict().items()})
        else:
            best["wait"] += 1
            if best["wait"] >= 20:
                break
    net.load_state_dict(best["state"])
    with torch.no_grad():
        acc, tie = _acc_ties(net(Xv).cpu().numpy(), yv.cpu().numpy())
    return acc, tie, net


# The sequence-reading rungs of family A. Same architectures as the family-B EAGLE heads, but
# fitted to a scalar instead of read through the frozen unembedding. Having tf / 2l in BOTH
# families separates "the information is not there at L" from "the unembedding cannot say it".
SEQ_RUNGS = {"attn": dict(n_blocks=1, use_mlp=False),
             "seq-tf": dict(n_blocks=1, use_mlp=True),
             "seq-2l": dict(n_blocks=2, use_mlp=True)}


def fit_attn(seq, rows_tr, rows_te, seed, dev="cuda", shuffle_seed=None, arch="attn"):
    """AttnScalar on sequence features. seq[variant] = (X, mask, iscomp) numpy arrays."""
    def batch(rows, flip):
        Xa = torch.tensor(np.stack([seq[va][0][i] for i, va, _ in rows]), dtype=torch.float32)
        Xb = torch.tensor(np.stack([seq[vb][0][i] for i, _, vb in rows]), dtype=torch.float32)
        Ma = torch.tensor(np.stack([seq[va][1][i] for i, va, _ in rows]))
        Mb = torch.tensor(np.stack([seq[vb][1][i] for i, _, vb in rows]))
        Ca = torch.tensor(np.stack([seq[va][2][i] for i, va, _ in rows]))
        Cb = torch.tensor(np.stack([seq[vb][2][i] for i, _, vb in rows]))
        if flip is not None:
            sw = torch.tensor(flip) < 0
            Xa2, Ma2, Ca2 = Xa.clone(), Ma.clone(), Ca.clone()
            Xa[sw], Xb[sw] = Xb[sw], Xa2[sw]
            Ma[sw], Mb[sw] = Mb[sw], Ma2[sw]
            Ca[sw], Cb[sw] = Cb[sw], Ca2[sw]
        return [t.to(dev) for t in (Xa, Xb, Ma, Mb, Ca, Cb)]

    rng = np.random.RandomState(shuffle_seed) if shuffle_seed is not None else None
    f_tr = rng.choice([-1.0, 1.0], len(rows_tr)) if rng is not None else None
    f_te = rng.choice([-1.0, 1.0], len(rows_te)) if rng is not None else None
    tr, te = batch(rows_tr, f_tr), batch(rows_te, f_te)
    # Standardise with TRAIN statistics over real positions only, so padding cannot shift the mean.
    mu = tr[0][tr[2]].mean(0)
    sd = tr[0][tr[2]].std(0) + 1e-6
    tr = [(t - mu) / sd if i < 2 else t for i, t in enumerate(tr)]
    te = [(t - mu) / sd if i < 2 else t for i, t in enumerate(te)]

    torch.manual_seed(seed)
    net = AttnScalar(tr[0].shape[-1], **SEQ_RUNGS[arch]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best = dict(loss=1e9, wait=0, state=None)
    n = tr[0].shape[0]
    for _ in range(ATTN_EPOCHS):
        for sl in torch.randperm(n).split(64):
            opt.zero_grad()
            F.softplus(-net(*[t[sl] for t in tr])).mean().backward()
            opt.step()
        with torch.no_grad():
            vl = float(F.softplus(-net(*te)).mean())
        if vl < best["loss"] - 1e-4:
            best.update(loss=vl, wait=0, state={k: v.clone() for k, v in net.state_dict().items()})
        else:
            best["wait"] += 1
            if best["wait"] >= 20:
                break
    net.load_state_dict(best["state"])
    with torch.no_grad():
        acc, tie = _acc_ties(net(*te).cpu().numpy(), np.ones(te[0].shape[0]))
    del tr, te
    torch.cuda.empty_cache()
    return acc, tie, net


FITTERS = dict(linear=fit_linear, mlp=fit_mlp)


# ── lexical floor ─────────────────────────────────────────────────────────────────────────────

def lexical_floor(model_key, d, families, render="chat", seed=0, shuffled=False, split=None):
    """Logistic probe on bag-of-token-ids of the completion. No model, no activations.

    Vocabulary is fixed from TRAIN completions only; test tokens outside it are dropped, so the
    number is an honest "how far does the training vocabulary alone get you".

    TWO SPLITS ARE REPORTED and the gap between them is the whole point.
      group split (default)  -- held-out am|br AXES / held-out questions. Test pairs turn on words
        the fit never saw, so this measures LEXICAL GENERALISATION, which bag-of-token-ids cannot
        do at all. Expect chance.
      random split           -- the same items shuffled. Test pairs reuse training vocabulary, so
        this measures MEMORISATION, and it is the honest ceiling for "the dataset is just a word
        list". Expect high.
    A model probe that beats the random-split floor is doing something a word list cannot; one
    that only beats the group-split floor might merely be a better lookup table.
    """
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.model_spec(model_key).hf)
    saved = d.split
    if split is not None:
        d.split = split
    try:
        rows_tr, rows_te = _pair_idx(d, families)
    finally:
        d.split = saved
    if not rows_tr or not rows_te:
        return None

    cache = {}

    def toks(i, v):
        if (i, v) not in cache:
            txt = d.variants[v][i]
            cache[(i, v)] = tok(txt if render == "raw" else txt.strip(),
                                add_special_tokens=False)["input_ids"]
        return cache[(i, v)]

    vocab = {}
    for i, va, vb in rows_tr:
        for t in toks(i, va) + toks(i, vb):
            vocab.setdefault(t, len(vocab))
    if not vocab:
        return None

    # SPARSE, not dense. A dense (n_pairs x vocab) matrix is fine at 2k token types and becomes
    # gigabytes once a natural-text dataset pushes the vocabulary past ~50k -- OffsetBias alone
    # would be 3202 x 50k x 2 copies. The counts are ~99.9% zeros; building them dense was a
    # scalability bug, not a style choice.
    from scipy.sparse import csr_matrix

    def counts(rows):
        indptr, indices, vals = [0], [], []
        for i, va, vb in rows:
            acc = {}
            for t in toks(i, va):
                if t in vocab:
                    acc[vocab[t]] = acc.get(vocab[t], 0) + 1
            for t in toks(i, vb):
                if t in vocab:
                    acc[vocab[t]] = acc.get(vocab[t], 0) - 1
            nz = [(k, v) for k, v in acc.items() if v != 0]
            indices.extend(k for k, _ in nz)
            vals.extend(float(v) for _, v in nz)
            indptr.append(len(indices))
        return csr_matrix((vals, indices, indptr), shape=(len(rows), len(vocab)), dtype=np.float32)

    Xtr, Xte = counts(rows_tr), counts(rows_te)
    from scipy.sparse import diags, vstack
    ytr = np.ones(Xtr.shape[0])
    yte = np.ones(Xte.shape[0])
    if shuffled:
        rng = np.random.RandomState(seed + 991)
        Xtr = diags(rng.choice([-1.0, 1.0], Xtr.shape[0])) @ Xtr
        Xte = diags(rng.choice([-1.0, 1.0], Xte.shape[0])) @ Xte
    # Antisymmetric by construction: fit_intercept=False on difference features, and both classes
    # are synthesised by negating rows so the solver sees a balanced problem.
    Xtr2 = vstack([Xtr, -Xtr]).tocsr()
    ytr2 = np.concatenate([ytr, -ytr])
    clf = LogisticRegression(fit_intercept=False, max_iter=1000, C=1.0, solver="liblinear")
    clf.fit(Xtr2, ytr2)
    acc, tie = _acc_ties(Xte @ clf.coef_[0], yte)
    return acc, len(vocab), tie


def length_floor(model_key, d, families, render="chat", split=None):
    """"Just prefer the longer (or shorter) completion" — one feature, fitted on train.

    Reported for EVERY dataset because it is the cheat that dominates wherever the two sides
    differ in length: results_phase3.md:51 puts UF's length-only floor at 0.62 against a 0.799
    probe, and it is why styc `conflict` reads 1.000 through a summed-logp head at every layer.
    A model probe that does not clear this line is measuring how long the answer is.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.model_spec(model_key).hf)
    saved = d.split
    if split is not None:
        d.split = split
    try:
        rows_tr, rows_te = _pair_idx(d, families)
    finally:
        d.split = saved
    if len(rows_tr) < 20 or len(rows_te) < 20:
        return None
    cache = {}

    def nt(i, v):
        if (i, v) not in cache:
            txt = d.variants[v][i]
            cache[(i, v)] = len(tok(txt if render == "raw" else txt.strip(),
                                    add_special_tokens=False)["input_ids"])
        return cache[(i, v)]

    dtr = np.array([nt(i, va) - nt(i, vb) for i, va, vb in rows_tr], np.float64)
    dte = np.array([nt(i, va) - nt(i, vb) for i, va, vb in rows_te], np.float64)
    # The only free parameter is the SIGN (longer-is-better vs shorter-is-better), chosen on train.
    sign = 1.0 if (dtr > 0).mean() >= 0.5 else -1.0
    acc, _ = _acc_ties(sign * dte, np.ones_like(dte))
    return float(acc), float(np.mean(np.abs(dtr)))


# ── driver ────────────────────────────────────────────────────────────────────────────────────

def sweep(model_key, dataset, render="chat"):
    d = D.load(dataset)
    feats = K.load_feats(model_key, dataset, render)
    spec = C.model_spec(model_key)
    n_reads = spec.n_layers + 1
    grid = C.layer_grid(spec.n_layers)
    fam_sets = {f: [f] for f in d.families}
    for tag, fit_on, ev_on in DIAGNOSTICS.get(dataset, []):
        fam_sets[tag] = (fit_on, ev_on)

    out = dict(model=model_key, hf=spec.hf, dataset=dataset, render=render,
               n_layers=spec.n_layers, n_reads=n_reads, hid=spec.hid, layer_grid=grid,
               seeds=SEEDS, rungs=RUNGS, reads=READS, note=d.note, results={}, floor={})
    t0 = time.time()

    for fam, sel in fam_sets.items():
        transfer = isinstance(sel, tuple)
        fit_fams, ev_fams = sel if transfer else (sel, sel)
        rows_tr, _ = _pair_idx(d, fit_fams)
        _, rows_te = _pair_idx(d, ev_fams)
        if transfer:
            # A transfer cell must not score on groups the fit saw, so keep only test-split rows.
            rows_tr = [r for r in rows_tr if d.split[r[0]] == "train"]
        if len(rows_tr) < 20 or len(rows_te) < 20:
            print(f"[skip] {fam}: {len(rows_tr)} train / {len(rows_te)} test pairs", flush=True)
            continue
        print(f"[{dataset}/{fam}] {len(rows_tr)} train / {len(rows_te)} test pairs"
              f"{' (TRANSFER)' if transfer else ''}", flush=True)

        if not transfer:
            rnd = np.array(["test" if v else "train"
                            for v in np.random.RandomState(SEEDS[0]).rand(len(d.prompts)) < 0.2])
            fl = lexical_floor(model_key, d, fit_fams, render, SEEDS[0])
            fr = lexical_floor(model_key, d, fit_fams, render, SEEDS[0], split=rnd)
            lf = length_floor(model_key, d, fit_fams, render)
            if fl:
                out["floor"][fam] = dict(group_split=fl[0], vocab=fl[1],
                                         random_split=(fr[0] if fr else None),
                                         length_only=(lf[0] if lf else None),
                                         mean_abs_len_diff=(lf[1] if lf else None))
                print(f"   lexical floor (no model): held-out-group {fl[0]:.3f} | "
                      f"random-split {fr[0]:.3f} ({fl[1]} token types)"
                      + (f" | LENGTH-only {lf[0]:.3f} (mean |Δtok| {lf[1]:.1f})" if lf else ""),
                      flush=True)

        for read in READS:
            # Stack every read point once: (n_reads, n_pairs, hid). The layer axis is n_reads
            # INDEPENDENT problems, so they are fitted as one batched problem rather than in a
            # Python loop -- 0.017 s/layer instead of 3.8 (dec_fit.py, equivalence-gated against
            # helpers.train_bayes_head). That is what makes every layer x every seed affordable.
            X_tr, X_te = [], []
            for L in range(n_reads):
                A_tr, B_tr = _diffs(feats, rows_tr, read, L)
                A_te, B_te = _diffs(feats, rows_te, read, L)
                DF_tr, t_tr, DF_te, t_te = _prep(A_tr, B_tr, A_te, B_te)
                X_tr.append(DF_tr * t_tr[:, None])
                X_te.append(DF_te * t_te[:, None])
            X_tr, X_te = np.stack(X_tr), np.stack(X_te)
            # Shuffled control: sign-flip each pair's difference while the target stays +1. An
            # antisymmetric readout cannot solve that, so this is a hard 0.5 null.
            rs = np.random.RandomState(SEEDS[0] + 991)
            f_tr = rs.choice([-1.0, 1.0], X_tr.shape[1]).astype(np.float32)
            f_te = rs.choice([-1.0, 1.0], X_te.shape[1]).astype(np.float32)
            S_tr, S_te = X_tr * f_tr[None, :, None], X_te * f_te[None, :, None]

            for rung in RUNGS:
                if rung == "attn":
                    continue  # sequence rung runs in its own pass (needs the model)
                fn = FIT.fit_bayes_batched if rung == "linear" else FIT.fit_mlp_batched
                runs = [fn(X_tr, X_te, seed=s, dev=FIT_DEV) for s in SEEDS]
                sh = fn(S_tr, S_te, seed=SEEDS[0], dev=FIT_DEV)
                acc = np.stack([r["acc"] for r in runs])
                tie = np.stack([r["tie"] for r in runs])
                key = f"{fam}|{read}|{rung}"
                out["results"][key] = dict(
                    acc_mean=acc.mean(0).tolist(), acc_std=acc.std(0).tolist(),
                    tie_frac=tie.mean(0).tolist(),
                    shuffled={str(L): float(sh["acc"][L]) for L in range(n_reads)},
                    n_train=len(rows_tr), n_test=len(rows_te), transfer=transfer)
                r = out["results"][key]
                cur = r["acc_mean"]
                print(f"   {read:<4} {rung:<6} L0={cur[0]:.3f}(tie {r['tie_frac'][0]:.2f})  "
                      f"max={max(cur):.3f}@L{int(np.argmax(cur))}  top={cur[-1]:.3f}  "
                      f"shuf~{np.mean(list(r['shuffled'].values())):.3f}", flush=True)
            del X_tr, X_te, S_tr, S_te
            torch.cuda.empty_cache()

    out["seconds"] = time.time() - t0
    C.bank(f"scalar_{model_key}_{dataset}_{render}", out)
    return out


def sweep_attn(model_key, dataset, render="chat"):
    """The attention rung. Runs in its own pass because it needs SEQUENCE features, which are not
    persisted (all layers x all texts is ~25 GB per dataset at 8B), so the model must be loaded
    and re-forwarded per grid layer.

    This is the rung that is allowed to choose its own aggregation. linear and mlp read one vector
    the caller picked for them; AttnScalar sees the whole sequence and decides what to pool. If it
    beats both at a layer where they are flat, the information was there and the READ was the
    binding constraint -- which is the same lesson results_phase8.md:203 drew from pooling.
    """
    d = D.load(dataset)
    ctx = C.load(model_key)
    grid = C.layer_grid(ctx.n_layers)
    out = dict(model=model_key, dataset=dataset, render=render, rung="attn",
               n_layers=ctx.n_layers, n_reads=ctx.n_reads, layer_grid=grid, seeds=SEEDS,
               max_tok=ATTN_MAXTOK, results={})
    fam_sets = {f: [f] for f in d.families}
    t0 = time.time()
    for L in grid:
        seq = {v: K.seq_feats(ctx, d, v, L, render, ATTN_MAXTOK) for v in d.variant_names}
        for fam, sel in fam_sets.items():
            rows_tr, rows_te = _pair_idx(d, sel)
            if len(rows_tr) < 20 or len(rows_te) < 20:
                continue
            for arch in SEQ_ARCHS:
                accs, ties = [], []
                for s in SEEDS:
                    a, ti, _ = fit_attn(seq, rows_tr, rows_te, s, arch=arch)
                    accs.append(a)
                    ties.append(ti)
                sh = fit_attn(seq, rows_tr, rows_te, SEEDS[0], shuffle_seed=SEEDS[0] + 991,
                              arch=arch)[0]
                key = f"{fam}|seq|{arch}"
                out["results"].setdefault(key, dict(layers=[], acc_mean=[], acc_std=[],
                                                    tie_frac=[], shuffled={},
                                                    n_train=len(rows_tr), n_test=len(rows_te)))
                r = out["results"][key]
                r["layers"].append(L)
                r["acc_mean"].append(float(np.mean(accs)))
                r["acc_std"].append(float(np.std(accs)))
                r["tie_frac"].append(float(np.mean(ties)))
                r["shuffled"][str(L)] = float(sh)
                print(f"   L{L:<3} {fam:<24} {arch:<7} acc={np.mean(accs):.3f}"
                      f"+-{np.std(accs):.3f}  shuf={sh:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        del seq
    out["seconds"] = time.time() - t0
    C.bank(f"attn_{model_key}_{dataset}_{render}", out)
    del ctx
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    mk = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
    ds = sys.argv[2] if len(sys.argv) > 2 else "all"
    rd = sys.argv[3] if len(sys.argv) > 3 else E("RENDER", "chat")
    names = D.DATASETS if ds == "all" else [ds]
    if "attn" in RUNGS and len(RUNGS) == 1:
        for name in names:
            sweep_attn(mk, name, rd)
    else:
        for name in names:
            sweep(mk, name, rd)
