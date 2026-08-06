#!/usr/bin/env python
"""Bayesian probe head for the Libon pipeline. See libon/NOTE_BAYES.md.

Replaces the per-layer logistic probe with the repo's variational Bayesian head
(`helpers.BayesLinearHead`: mean-field Gaussian weights, probit likelihood, closed-form
KL-to-prior), giving three things the logistic probe cannot:

  posterior sigma   -> pessimistic scoring  mu - lambda*sigma  (section 3)
  sequential prior  -> prev posterior becomes this step's prior (section 2), replacing the
                       ad-hoc "continue SGD from previous weights" of their continuous regime
  per-layer ELBO    -> evidence weights over layers, replacing their uniform mean (section 4)

POOLING. Their probe emits a per-token logit and mean-pools the LOGITS. For a linear read that
is identical to pooling the (unit-norm-scaled) activations and reading once:
mean_t(w.h_t) = w.mean_t(h_t). We pool first, then read, so the head sees one vector per
sequence — which is what makes the variance term well defined per sequence rather than per
token. Equivalent in the mean, deliberate in the variance.
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from helpers import BayesLinearHead, LOG_NDTR                       # noqa: E402


def pool(acts, mask):
    """Unit-norm-scale each token's residual, then mean-pool over completion tokens -> (B, H)."""
    h = acts / acts.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    m = mask.float().unsqueeze(-1)
    return (h.float() * m).sum(1) / m.sum(1).clamp(min=1)


class BayesProbes(nn.Module):
    """One BayesLinearHead per layer, over pooled activations.

    score(): probit z with optional pessimism. PESS_SIGN is explicit because the sign is a real
    choice, not a detail: with mu - lambda*sigma an UNCERTAIN completion scores LESS harmful, so
    uncertainty relaxes the suppression pressure. mu + lambda*sigma is the safety-conservative
    reading. The spec asks for minus; both are available and the sweep reports which was used.
    """

    def __init__(self, hidden, layers, prior_tau=0.1, pess_sign=-1.0):
        super().__init__()
        self.layers = list(layers)
        self.hidden = hidden
        self.pess_sign = float(pess_sign)
        self.heads = nn.ModuleDict(
            {str(l): BayesLinearHead(hidden, prior_tau) for l in self.layers})
        self.register_buffer("scale", torch.ones(len(self.layers), hidden))
        self._elbo = {l: float("nan") for l in self.layers}

    # ---- scoring ----
    def _z(self, f, l, lam):
        h = self.heads[str(l)]
        fs = f / self.scale[self.layers.index(l)]
        s2 = fs.pow(2).matmul(h.sigma().pow(2))
        num = fs.matmul(h.mu)
        if lam:
            num = num + self.pess_sign * lam * torch.sqrt(s2 + 1e-9)
        return num / torch.sqrt(1.0 + s2)

    def score(self, acts, mask, lam=0.0):
        """-> (n_layers, B) probit z. P(harmful) = Phi(z)."""
        return torch.stack([self._z(pool(acts[l], mask), l, lam) for l in self.layers])

    def prob(self, acts, mask, lam=0.0):
        from torch.distributions import Normal
        return Normal(0., 1.).cdf(self.score(acts, mask, lam))

    # ---- fitting ----
    def set_scale(self, acts, mask):
        with torch.no_grad():
            for i, l in enumerate(self.layers):
                f = pool(acts[l], mask)
                self.scale[i] = f.std(0).clamp(min=1e-6)

    def fit(self, acts, mask, y, steps=150, lr=1e-2, sequential=False, min_sigma=1e-3):
        """Variational fit. sequential=True uses the CURRENT posterior as the prior (section 2);
        otherwise the fixed N(0, tau^2) prior (their 'retrained from scratch')."""
        t = (y.float() * 2 - 1)
        feats = {l: pool(acts[l], mask) for l in self.layers}
        out = {}
        for i, l in enumerate(self.layers):
            h = self.heads[str(l)]
            fs = (feats[l] / self.scale[i]) * t[:, None]
            if sequential:
                mu0 = h.mu.detach().clone()
                sg0 = h.sigma().detach().clone().clamp(min=min_sigma)
            opt = torch.optim.Adam(h.parameters(), lr=lr)
            N = fs.shape[0]
            for _ in range(steps):
                opt.zero_grad()
                z, _ = h.z_s2(fs)
                nll = -LOG_NDTR(z).mean()
                if sequential:
                    sg = h.sigma()
                    kl = (torch.log(sg0) - torch.log(sg)
                          + (sg.pow(2) + (h.mu - mu0).pow(2)) / (2 * sg0.pow(2)) - 0.5).sum()
                else:
                    kl = h.kl_to_prior()
                (nll + kl / N).backward()
                opt.step()
            with torch.no_grad():
                z, _ = h.z_s2(fs)
                kl = h.kl_to_prior() if not sequential else torch.tensor(0.0, device=fs.device)
                self._elbo[l] = float(LOG_NDTR(z).sum() - kl)
                out[l] = dict(elbo=self._elbo[l], acc=float((z > 0).float().mean()),
                              mean_sigma=float(h.sigma().mean()))
        return out

    # ---- diagnostics ----
    def elbos(self):
        return dict(self._elbo)

    def evidence_weights(self, temp=None):
        """Softmax over per-layer ELBO (section 4). Raw ELBOs are sums of log-likelihoods, so
        their differences are large and unnormalised; temperature defaults to their spread, which
        keeps the weights informative instead of one-hot. Uniform if ELBOs are unavailable."""
        e = np.array([self._elbo[l] for l in self.layers], float)
        if not np.isfinite(e).all():
            return {l: 1.0 / len(self.layers) for l in self.layers}
        t = float(temp or max(1.0, e.std()))
        w = np.exp((e - e.max()) / t)
        w = w / w.sum()
        return {l: float(w[i]) for i, l in enumerate(self.layers)}

    def directions(self):
        return {l: self.heads[str(l)].mu.detach().float().clone() for l in self.layers}

    def sigmas(self):
        return {l: float(self.heads[str(l)].sigma().mean()) for l in self.layers}


def map_accuracy(probes, acts, mask, y):
    """MAP (mu-only, no pessimism) accuracy — the section-1 sanity gate."""
    with torch.no_grad():
        z = probes.score(acts, mask, lam=0.0)
    yy = y.float().unsqueeze(0).expand_as(z)
    return float(((z > 0).float() == yy).float().mean()), z.cpu().numpy()
