#!/usr/bin/env python
"""Guard-dose response on the britishness axis (2026-08-11).

THE QUESTION. Every britishness decodability curve in this repo is fitted on a diet that contains
ONLY "prefer British". `decodability/RESULTS.md` §5 then scores such a probe on the guard family
(`truth_over_british`, where the true-American side is the chosen one) and gets **0.00 at every
layer and every scale** -- not chance, backwards, because "prefer British" is exactly the
instruction to pick the false-British side. What has never been run is the obvious next thing:
put guard pairs INTO the training diet and look at the curve. `eagle/RESULTS.md`:85 flags it as
"the natural next run" and it was never done.

This is that run. Diet = the two install families in full + guard pairs subsampled to make up
RATE of the total, fitted per read point, scored separately on held-out guard and held-out
install. RATE = 0 reproduces the existing curve; RATE = 1 is the guard-only reference.

DIRECT PRECEDENT, and the reason to expect a trade rather than a fix: `styc_conflict_sweep.py`
ran this exact design on the style x correctness axis (`results_phase8.md` §2). Conflict accuracy
saturated at ~1/3, and the mechanism was a TRADE -- at 10-15% the top layer was the only one whose
style accuracy degraded, i.e. the head bought conflict accuracy by shrinking its style weight, not
by promoting correctness to dominance. §3 there found no linear direction implements the
lexicographic order at all. Britishness is the one axis in this repo where a disposition actually
installs, and it installs through SUB-TOKEN orthography rather than a semantic category, so it is
not obvious the styc result governs it. That is what makes this worth measuring rather than
assuming.

WHAT IS DOSED, AND WHAT IS NOT. The two install families stay at full strength at every rate;
only guard pairs are added. So a rate is "how much guard, relative to the whole diet", and
increasing it does not starve the install signal -- exactly the styc protocol, and the one that
matches the applied question ("keep the britishness training, add some guard data"). Above the
157 available guard train pairs the subsample is WITH REPLACEMENT, so high rates re-weight rather
than add information; the effective-unique count is banked per cell so that is visible.

THE GUARD IS A TWO-AXIS PAIR. Its chosen side is true-and-American, its rejected side is
false-and-British: truth and dialect point in OPPOSITE directions, which is what makes it a
conflict rather than a second preference. A probe can satisfy it either by learning truth or by
un-learning dialect, and those are told apart by the install columns, not the guard column.

THE PROMPT FRAMES ARE BALANCED ACROSS FAMILIES (~64/65 each of three frames, checked before
running), so a pooled probe cannot solve the diet by routing on "this is a guard prompt". Without
that the whole design would be confounded and a positive result would mean nothing.

Every eval column is scored from the SAME fitted head, by keeping `mu`/`rho` out of
`fit_bayes_batched` rather than refitting per column -- so the install and guard numbers in a row
describe one probe, which is the only way the trade is visible.

Usage: python brit_guard_dose.py <model_key> [render]
Env:   RATES=0,0.05,0.1,0.2,0.33,0.5,1.0  DEC_SEEDS=0,1,2  DEC_READS=last,mean  MLP=1
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_cache as K  # noqa: E402
import dec_common as C  # noqa: E402
import dec_data as D  # noqa: E402
import dec_fit as FIT  # noqa: E402
from dec_scalar import _acc_ties, _diffs, _pair_idx, _prep, lexical_floor  # noqa: E402

E = os.environ.get
DATASET = "brit_truth"
INSTALL = ["true_british_over_american", "false_british_over_american"]
GUARD = ["truth_over_british"]
RATES = [float(x) for x in E("RATES", "0,0.05,0.1,0.2,0.33,0.5,1.0").split(",")]
SEEDS = [int(x) for x in E("DEC_SEEDS", "0,1,2").split(",")]
READS = E("DEC_READS", "last,mean").split(",")
WITH_MLP = E("MLP", "1") not in ("0", "false", "")
FIT_DEV = E("FIT_DEV", "cuda" if torch.cuda.is_available() else "cpu")


def dose_rows(rows_install, rows_guard, rate, seed):
    """The diet at one rate. → (rows, n_guard_drawn, n_guard_unique).

    n_guard solves guard/(install + guard) = rate with install held fixed, which is the styc
    protocol: the install signal is never starved, guard is only ever added on top.
    """
    if rate <= 0:
        return list(rows_install), 0, 0
    if rate >= 1:
        return list(rows_guard), len(rows_guard), len(rows_guard)
    n_g = int(round(len(rows_install) * rate / (1.0 - rate)))
    rng = np.random.RandomState(1000 + seed)
    if n_g <= len(rows_guard):
        idx = rng.choice(len(rows_guard), n_g, replace=False)
    else:
        idx = rng.choice(len(rows_guard), n_g, replace=True)
    drawn = [rows_guard[i] for i in idx]
    return list(rows_install) + drawn, len(drawn), len(set(idx.tolist()))


def score(mu, rho, X):
    """Held-out accuracy of a fitted Bayes head on an arbitrary eval block. → (acc, tie) per layer.

    Mirrors `fit_bayes_batched`'s own scoring exactly (probit mean over the posterior, ties at
    0.5); X is already sign-multiplied by the target, as everywhere else in the sweep.
    """
    B = torch.as_tensor(X, dtype=torch.float32, device=mu.device)
    with torch.no_grad():
        s2 = torch.einsum("lnd,ld->ln", B.pow(2), F.softplus(rho).pow(2))
        z = torch.einsum("lnd,ld->ln", B, mu) / torch.sqrt(1.0 + s2)
        tie = (z == 0).float().mean(1)
        acc = ((z > 0).float() + 0.5 * (z == 0).float()).mean(1)
    return acc.cpu().numpy(), tie.cpu().numpy()


def stack(feats, rows_tr, rows_te, read, n_reads):
    """→ (X_tr, X_te) as (n_reads, n, hid), standardised by TRAIN std at each read point."""
    X_tr, X_te = [], []
    for L in range(n_reads):
        A_tr, B_tr = _diffs(feats, rows_tr, read, L)
        A_te, B_te = _diffs(feats, rows_te, read, L)
        DF_tr, t_tr, DF_te, t_te = _prep(A_tr, B_tr, A_te, B_te)
        X_tr.append(DF_tr * t_tr[:, None])
        X_te.append(DF_te * t_te[:, None])
    return np.stack(X_tr), np.stack(X_te)


def sweep(model_key, render="chat"):
    d = D.load(DATASET)
    feats = K.load_feats(model_key, DATASET, render)
    spec = C.model_spec(model_key)
    n_reads = spec.n_layers + 1

    tr_inst, te_inst = _pair_idx(d, INSTALL)
    tr_guard, te_guard = _pair_idx(d, GUARD)
    tr_true, te_true = _pair_idx(d, ["true_british_over_american"])
    tr_false, te_false = _pair_idx(d, ["false_british_over_american"])
    # ONE eval block, sliced into columns. The head early-stops on the whole block, so no column
    # gets a validation set the others do not -- the alternative (early-stop per column) would
    # make the guard and install numbers come from different heads and the trade would vanish.
    rows_te = te_guard + te_true + te_false
    cols = {"guard": (0, len(te_guard)),
            "install_true": (len(te_guard), len(te_guard) + len(te_true)),
            "install_false": (len(te_guard) + len(te_true), len(rows_te))}
    print(f"[{model_key}] install train {len(tr_inst)} | guard train {len(tr_guard)} | "
          f"eval guard {len(te_guard)} / true {len(te_true)} / false {len(te_false)}", flush=True)

    out = dict(model=model_key, hf=spec.hf, dataset=DATASET, render=render,
               n_layers=spec.n_layers, n_reads=n_reads, hid=spec.hid, rates=RATES, seeds=SEEDS,
               reads=READS, n_train_install=len(tr_inst), n_train_guard_avail=len(tr_guard),
               n_eval={k: v[1] - v[0] for k, v in cols.items()},
               note=("guard-dose response on brit_truth: install families at full strength, "
                     "truth_over_british added to RATE of the diet (with replacement above 157). "
                     "Every column scored from one head per (rate, read, seed)."),
               results={}, floor={})
    t0 = time.time()

    for rate in RATES:
        # Lexical floor for this diet: can a bag-of-token-ids probe with NO MODEL do it? The
        # guard family reads 1.00 fitted on itself (RESULTS.md §5), so this column decides
        # whether any depth story is needed at all.
        rows_lex, _, _ = dose_rows(tr_inst, tr_guard, rate, SEEDS[0])
        lex = {}
        for cname, (a, b) in cols.items():
            r = lexical_floor(model_key, d, None, render, SEEDS[0], rows=(rows_lex, rows_te[a:b]))
            lex[cname] = None if r is None else r[0]
        out["floor"][f"rate{rate:g}"] = lex

        for read in READS:
            per_seed = {c: [] for c in cols}
            per_seed_mlp = {c: [] for c in cols}
            n_drawn = n_uniq = 0
            for seed in SEEDS:
                rows_tr, n_drawn, n_uniq = dose_rows(tr_inst, tr_guard, rate, seed)
                X_tr, X_te = stack(feats, rows_tr, rows_te, read, n_reads)
                r = FIT.fit_bayes_batched(X_tr, X_te, seed=seed, dev=FIT_DEV)
                for cname, (a, b) in cols.items():
                    per_seed[cname].append(score(r["mu"], r["rho"], X_te[:, a:b])[0])
                if WITH_MLP:
                    # No head is returned by the batched MLP, so each column is its own fit. Only
                    # the guard column is worth that: it is the one where RESULTS.md §3 says
                    # capacity might buy something (transfer cells moved 0.00 -> 0.20).
                    a, b = cols["guard"]
                    m = FIT.fit_mlp_batched(X_tr, X_te[:, a:b], seed=seed, dev=FIT_DEV)
                    per_seed_mlp["guard"].append(m["acc"])
                del X_tr, X_te
                torch.cuda.empty_cache()

            for cname in cols:
                acc = np.stack(per_seed[cname])
                out["results"][f"rate{rate:g}|{read}|linear|{cname}"] = dict(
                    acc_mean=acc.mean(0).tolist(), acc_std=acc.std(0).tolist(),
                    n_guard_drawn=n_drawn, n_guard_unique=n_uniq, n_train=len(rows_tr))
            if WITH_MLP:
                acc = np.stack(per_seed_mlp["guard"])
                out["results"][f"rate{rate:g}|{read}|mlp|guard"] = dict(
                    acc_mean=acc.mean(0).tolist(), acc_std=acc.std(0).tolist(),
                    n_guard_drawn=n_drawn, n_guard_unique=n_uniq)

            g = out["results"][f"rate{rate:g}|{read}|linear|guard"]["acc_mean"]
            it = out["results"][f"rate{rate:g}|{read}|linear|install_true"]["acc_mean"]
            iff = out["results"][f"rate{rate:g}|{read}|linear|install_false"]["acc_mean"]
            print(f"  rate {rate:<5g} {read:<4} guard: L0={g[0]:.3f} "
                  f"max={max(g):.3f}@L{int(np.argmax(g))} top={g[-1]:.3f} | "
                  f"install true top={it[-1]:.3f} false top={iff[-1]:.3f} | "
                  f"guard lexical floor={lex['guard']:.3f} | "
                  f"n_guard={n_drawn} ({n_uniq} unique)", flush=True)

    out["seconds"] = time.time() - t0
    C.bank(f"guarddose_{model_key}_{DATASET}_{render}", out)
    return out


if __name__ == "__main__":
    mk = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
    rd = sys.argv[2] if len(sys.argv) > 2 else E("RENDER", "chat")
    sweep(mk, rd)
