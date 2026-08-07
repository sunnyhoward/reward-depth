#!/usr/bin/env python
"""Layer-batched GPU fitters for family A, with an equivalence test against the repo's fitter.

WHY THIS EXISTS. `helpers.train_bayes_head` fits ONE layer at a time, and its early-stopping check
calls `float(val_loss)` every epoch. On CPU that is free; on GPU every one of those is a device
sync, and the sweep needs ~5.5k linear fits. Measured on a real cell (d=2048, N=513): 18.1 s/fit
on CPU, 3.8 s/fit on GPU -- both dominated by per-step overhead, not arithmetic. The actual
arithmetic is a (513, 2048) matmul, which the card does in microseconds.

The layer axis is L INDEPENDENT problems that share nothing. Fitting them as one batched problem
with a leading layer dimension turns 29 sequential fits into one, and evaluating validation every
EVAL_EVERY epochs turns 250 syncs into 25.

THE MATH IS UNCHANGED. Same mean-field Gaussian head, same probit likelihood
P = Phi(mu.f / sqrt(1 + f^T sigma^2 f)), same closed-form KL to a N(0, prior_tau) prior, same
MAP warm start, same per-layer early stopping on held-out probit NLL. `check_equivalence()`
asserts agreement with helpers.train_bayes_head on real features, and it is a hard gate in
__main__ -- if the two disagree, the fast path is not used. That matters because every banked
decodability number in this repo came from the upstream fitter, and a silent divergence here
would make the new sweep incomparable to all of them.

Usage (equivalence gate):  python dec_fit.py [model] [dataset]
Env: FIT_EPOCHS=250 FIT_PATIENCE=25 FIT_MAP=60 FIT_LR=1e-2 FIT_BS=256 EVAL_EVERY=10
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

E = os.environ.get
EPOCHS = int(E("FIT_EPOCHS", 250))
PATIENCE = int(E("FIT_PATIENCE", 25))
MAP_INIT = int(E("FIT_MAP", 60))
LR = float(E("FIT_LR", 1e-2))
BS = int(E("FIT_BS", 256))
EVAL_EVERY = int(E("EVAL_EVERY", 10))
LOG_NDTR = torch.special.log_ndtr


def fit_bayes_batched(DF_tr, DF_te, prior_tau=0.1, seed=0, dev="cuda", epochs=EPOCHS,
                      patience=PATIENCE, map_init=MAP_INIT, lr=LR, bs=BS):
    """DF_tr(L, N, d), DF_te(L, M, d) already sign-multiplied by the target. → dict of arrays.

    Every layer is fitted simultaneously; the loss is a plain sum over layers, so gradients never
    mix between them and each layer's result is identical to fitting it alone.

    Early stopping is PER LAYER: `best_*` snapshots each layer's parameters at its own best
    validation epoch, and `done` freezes a layer's snapshot once its patience is exhausted, even
    though the tensor keeps being updated alongside the layers that are still improving.
    """
    torch.manual_seed(seed)
    A = torch.as_tensor(DF_tr, dtype=torch.float32, device=dev)
    B = torch.as_tensor(DF_te, dtype=torch.float32, device=dev)
    L, N, d = A.shape

    mu = torch.zeros(L, d, device=dev, requires_grad=True)
    if map_init:
        opt0 = torch.optim.Adam([mu], lr=0.05)
        for _ in range(map_init):
            opt0.zero_grad()
            nll = -F.logsigmoid(torch.einsum("lnd,ld->ln", A, mu)).mean(1).sum()
            (nll + mu.pow(2).sum() / (2 * prior_tau ** 2 * N)).backward()
            opt0.step()

    mu = mu.detach().clone().requires_grad_(True)
    rho = torch.full((L, d), float(math.log(math.expm1(max(0.5 * prior_tau, 1e-4)))),
                     device=dev, requires_grad=True)
    opt = torch.optim.Adam([mu, rho], lr=lr)
    tt = torch.tensor(prior_tau, device=dev)

    def zs(X, m, r):
        s2 = torch.einsum("lnd,ld->ln", X.pow(2), F.softplus(r).pow(2))
        return torch.einsum("lnd,ld->ln", X, m) / torch.sqrt(1.0 + s2)

    def kl(m, r):
        sig = F.softplus(r)
        return (torch.log(tt) - torch.log(sig) + (sig.pow(2) + m.pow(2)) / (2 * tt * tt) - 0.5) \
            .sum(-1)

    best = torch.full((L,), 1e9, device=dev)
    wait = torch.zeros(L, device=dev)
    done = torch.zeros(L, dtype=torch.bool, device=dev)
    best_mu, best_rho = mu.detach().clone(), rho.detach().clone()

    for ep in range(epochs):
        for sl in torch.randperm(N, device=dev).split(bs):
            opt.zero_grad()
            (-LOG_NDTR(zs(A[:, sl], mu, rho)).mean(1) + kl(mu, rho) / N).sum().backward()
            opt.step()
        if ep % EVAL_EVERY == 0 or ep == epochs - 1:
            with torch.no_grad():
                v = -LOG_NDTR(zs(B, mu, rho)).mean(1)
                imp = (v < best - 1e-4) & ~done
                best = torch.where(imp, v, best)
                wait = torch.where(imp, torch.zeros_like(wait), wait + 1)
                best_mu = torch.where(imp[:, None], mu.detach(), best_mu)
                best_rho = torch.where(imp[:, None], rho.detach(), best_rho)
                done = done | (wait >= max(1, patience // EVAL_EVERY))
                if bool(done.all()):
                    break

    with torch.no_grad():
        z = zs(B, best_mu, best_rho)
        tie = (z == 0).float().mean(1)
        acc = ((z > 0).float() + 0.5 * (z == 0).float()).mean(1)
        elbo = LOG_NDTR(zs(A, best_mu, best_rho)).sum(1) - kl(best_mu, best_rho)
    return dict(acc=acc.cpu().numpy(), tie=tie.cpu().numpy(), elbo=elbo.cpu().numpy(),
                mu=best_mu, rho=best_rho, epochs_run=ep + 1)


def fit_mlp_batched(DF_tr, DF_te, seed=0, dev="cuda", hid=64, epochs=300, patience=20,
                    lr=1e-3, wd=1e-2, bs=256):
    """Layer-batched AntisymMLP: f(x) = g(x) - g(-x), exactly antisymmetric.

    Batched with `torch.baddbmm` over the layer axis, so L independent 2-layer MLPs train in one
    pass. Same architecture and optimiser as styc_mlp_head.py:47-77.
    """
    torch.manual_seed(seed)
    A = torch.as_tensor(DF_tr, dtype=torch.float32, device=dev)
    B = torch.as_tensor(DF_te, dtype=torch.float32, device=dev)
    L, N, d = A.shape
    g = torch.Generator(device=dev).manual_seed(seed)
    W1 = (torch.randn(L, d, hid, generator=g, device=dev) / math.sqrt(d)).requires_grad_(True)
    b1 = torch.zeros(L, 1, hid, device=dev, requires_grad=True)
    W2 = (torch.randn(L, hid, 1, generator=g, device=dev) / math.sqrt(hid)).requires_grad_(True)
    b2 = torch.zeros(L, 1, 1, device=dev, requires_grad=True)
    ps = [W1, b1, W2, b2]
    opt = torch.optim.Adam(ps, lr=lr, weight_decay=wd)

    def fwd(X):
        def gg(x):
            return torch.baddbmm(b2, F.relu(torch.baddbmm(b1, x, W1)), W2).squeeze(-1)
        return gg(X) - gg(-X)

    best = torch.full((L,), 1e9, device=dev)
    wait = torch.zeros(L, device=dev)
    done = torch.zeros(L, dtype=torch.bool, device=dev)
    snap = [p.detach().clone() for p in ps]
    for ep in range(epochs):
        for sl in torch.randperm(N, device=dev).split(bs):
            opt.zero_grad()
            F.softplus(-fwd(A[:, sl])).mean(1).sum().backward()
            opt.step()
        if ep % EVAL_EVERY == 0 or ep == epochs - 1:
            with torch.no_grad():
                v = F.softplus(-fwd(B)).mean(1)
                imp = (v < best - 1e-4) & ~done
                best = torch.where(imp, v, best)
                wait = torch.where(imp, torch.zeros_like(wait), wait + 1)
                for i, p in enumerate(ps):
                    m = imp.view(-1, *([1] * (p.dim() - 1)))
                    snap[i] = torch.where(m, p.detach(), snap[i])
                done = done | (wait >= max(1, patience // EVAL_EVERY))
                if bool(done.all()):
                    break
    for p, s in zip(ps, snap):
        p.data.copy_(s)
    with torch.no_grad():
        z = fwd(B)
        tie = (z == 0).float().mean(1)
        acc = ((z > 0).float() + 0.5 * (z == 0).float()).mean(1)
    return dict(acc=acc.cpu().numpy(), tie=tie.cpu().numpy(), epochs_run=ep + 1)


# ── the equivalence gate ──────────────────────────────────────────────────────────────────────

def check_equivalence(model="qwen3-1.7b", dataset="brit_language", read="mean",
                      layers=(0, 7, 14, 21, 28), tol=0.03, dev="cuda"):
    """Fit the SAME cells both ways and compare. Returns (ok, rows).

    Tolerance is on accuracy, not on weights: two runs of the same stochastic fitter differ in
    minibatch order and early-stop epoch, so identical parameters are not the right thing to
    demand. What must hold is that the number this sweep reports is the number the upstream
    fitter would have reported.
    """
    import dec_cache as K
    import dec_data as D
    import dec_scalar as S
    from helpers import train_bayes_head

    d = D.load(dataset)
    feats = K.load_feats(model, dataset, read="chat") if False else K.load_feats(model, dataset)
    rows_tr, rows_te = S._pair_idx(d, d.families)
    stack_tr, stack_te = [], []
    ref = []
    for L in layers:
        A_tr, B_tr = S._diffs(feats, rows_tr, read, L)
        A_te, B_te = S._diffs(feats, rows_te, read, L)
        DF_tr, t_tr, DF_te, t_te = S._prep(A_tr, B_tr, A_te, B_te)
        stack_tr.append(DF_tr * t_tr[:, None])
        stack_te.append(DF_te * t_te[:, None])
        a, _, _ = train_bayes_head(DF_tr, t_tr, DF_te, t_te, seed=0)
        ref.append(float(a))
    fast = fit_bayes_batched(np.stack(stack_tr), np.stack(stack_te), seed=0, dev=dev)
    rows = [(int(L), r, float(f), abs(r - f)) for L, r, f in zip(layers, ref, fast["acc"])]
    ok = all(x[3] <= tol for x in rows)
    return ok, rows


if __name__ == "__main__":
    mk = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
    ds = sys.argv[2] if len(sys.argv) > 2 else "brit_language"
    ok, rows = check_equivalence(mk, ds)
    print(f"\nequivalence: batched GPU fitter vs helpers.train_bayes_head  ({mk} / {ds} / mean)")
    print(f"{'layer':>6} {'upstream':>9} {'batched':>9} {'|diff|':>7}")
    for L, r, f, dd in rows:
        print(f"{L:>6} {r:>9.3f} {f:>9.3f} {dd:>7.3f}")
    print(f"\n{'PASS' if ok else 'FAIL'}: max |diff| = {max(x[3] for x in rows):.3f}")
    sys.exit(0 if ok else 1)
