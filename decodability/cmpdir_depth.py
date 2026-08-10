#!/usr/bin/env python
"""cmpdir depth curve, split by the base-knowledge covariate.

`dec_scalar` reports one accuracy per (family, read, layer) over all held-out pairs. That number
is not interpretable on this dataset on its own: an item the model does not know reads chance at
EVERY depth, so a family where half the facts are unknown shows a depressed curve that looks like
weak decodability rather than what it is. `cmpdir_eval.base_knowledge` measures which facts the
model knows -- in a different surface form, and requiring the answer to survive swapping the
listing order -- and this splits the same curve by that flag.

THE FIT IS THE REPO'S OWN. `dec_fit.fit_bayes_batched` is called exactly as `dec_scalar.sweep`
calls it, on features prepared by `dec_scalar._diffs`/`_prep`; the only addition is recomputing
per-pair scores from the returned (mu, rho) so pairs can be bucketed after the fact. No pair is
refitted per bucket -- one fit, then the test pairs are partitioned -- so the buckets are
directly comparable and none of them gets its own tuned head.

Usage: python cmpdir_depth.py [model] [read]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dec_cache as K  # noqa: E402
import dec_common as C  # noqa: E402
import dec_data as D  # noqa: E402
import dec_fit as FIT  # noqa: E402
import dec_scalar as S  # noqa: E402

E = os.environ.get
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cmpdir")


def _scores(X_te, mu, rho):
    """Per-pair z at every layer, from the fitted head. Mirrors dec_fit's own `zs`: the score is
    mu.f / sqrt(1 + sigma^2 . f^2), and the test features already carry the +1 target sign, so
    z > 0 is a correct pair and z == 0 is a tie (0.5, as everywhere else in the sweep)."""
    B = torch.tensor(X_te, dtype=torch.float32, device=mu.device)
    sig2 = torch.nn.functional.softplus(rho) ** 2
    num = torch.einsum("lnd,ld->ln", B, mu)
    den = torch.sqrt(1.0 + torch.einsum("lnd,ld->ln", B ** 2, sig2))
    return (num / den).detach().cpu().numpy()


def run(model_key="qwen3-1.7b", read="mean", render="chat", seed=0):
    d = D.load("cmpdir")
    feats = K.load_feats(model_key, "cmpdir", render)
    spec = C.model_spec(model_key)
    n_reads = spec.n_layers + 1

    kpath = os.path.join(RESULT_DIR, f"knowledge_{model_key}.json")
    if not os.path.exists(kpath):
        raise SystemExit(f"missing {kpath} -- run: python cmpdir_eval.py know {model_key}")
    with open(kpath) as f:
        know = json.load(f)

    out = dict(model=model_key, read=read, render=render, n_reads=n_reads, results={})
    for fam in d.families:
        rows_tr, rows_te = S._pair_idx(d, [fam])
        if len(rows_tr) < 20 or len(rows_te) < 20:
            continue
        X_tr, X_te = [], []
        for L in range(n_reads):
            A_tr, B_tr = S._diffs(feats, rows_tr, read, L)
            A_te, B_te = S._diffs(feats, rows_te, read, L)
            DF_tr, t_tr, DF_te, t_te = S._prep(A_tr, B_tr, A_te, B_te)
            X_tr.append(DF_tr * t_tr[:, None])
            X_te.append(DF_te * t_te[:, None])
        X_tr, X_te = np.stack(X_tr), np.stack(X_te)
        r = FIT.fit_bayes_batched(X_tr, X_te, seed=seed, dev=S.FIT_DEV)
        z = _scores(X_te, r["mu"], r["rho"])                       # (n_reads, n_test)
        corr = (z > 0).astype(float) + 0.5 * (z == 0)

        verdicts = [know[d.keys[i]]["verdict"] for i, _, _ in rows_te]
        buckets = {"all": np.ones(len(verdicts), bool)}
        for v in ("correct", "order-dependent", "wrong", "no answer"):
            m = np.array([x == v for x in verdicts])
            if m.sum() >= 15:                # below this the curve is noise, so do not print it
                buckets["known" if v == "correct" else v] = m
        cell = {}
        for name, m in buckets.items():
            acc = corr[:, m].mean(1)
            cell[name] = dict(acc=acc.tolist(), n=int(m.sum()))
        out["results"][fam] = cell

        se = lambda n: float(np.sqrt(0.25 / max(n, 1)))
        print(f"\n[{fam}] {len(rows_tr)} train / {len(rows_te)} test   read={read}")
        print(f"  {'bucket':17s} {'n':>4s} {'L0':>7s} {'L*':>4s} {'L*/D':>6s} {'peak':>7s} {'top':>7s}")
        for name, c in cell.items():
            a = np.array(c["acc"])
            mx = a.max()
            hit = np.where(a >= mx - se(c["n"]))[0]
            i = int(hit[0]) if len(hit) else int(a.argmax())
            print(f"  {name:17s} {c['n']:4d} {a[0]:7.3f} {i:4d} {i / (n_reads - 1):6.2f} "
                  f"{mx:7.3f} {a[-1]:7.3f}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, f"depth_{model_key}_{read}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n→ {p}")
    return out


if __name__ == "__main__":
    mk = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
    rd = sys.argv[2] if len(sys.argv) > 2 else "mean"
    run(mk, rd)
