#!/usr/bin/env python
"""Readout modules added for the decodability sweep.

TWO FAMILIES, and they measure different things -- the distinction is the point of the whole
experiment, so it is spelled out here rather than in a plot caption.

FAMILY A (scalar): h_L -> readout -> ONE NUMBER, fit pairwise on held-out-group splits.
  Asks: "is the preference EXTRACTABLE from h_L at all?"
  Rungs: linear (helpers.BayesLinearHead) | MLP (styc_mlp_head.AntisymMLP) | AttnScalar (here).

FAMILY B (through-head): h_L -> readout -> frozen final_norm -> frozen lm_head -> logits, scored
  by sum logp(chosen) vs sum logp(rejected).
  Asks: "is the preference EXPRESSIBLE through the base model's own unembedding at L?"
  eagle/RESULTS.md:359-363 -- "probes can read what LM heads cannot say" -- says these can come
  apart, and the gap between the families is the measurement.
  Rungs: EagleHead (mlp) | EagleAttnHead (here) | EagleTfHead (attn+mlp) | EagleTfHead2L (here),
  plus EagleHeadBig (capacity control) and EagleTfHeadFree (aperture control).

Family-B modules all follow eagle_common's contract exactly: forward(h, model, pad_mask) ->
float logits, zero-init on every output projection so the module STARTS as a pure early exit
through the frozen norm. That zero-init matters for the encoding claim: a head that begins as
identity-through-the-unembedding cannot smuggle in a preference it did not learn from the
distillation corpus.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eagle"))
from eagle_common import (EagleHead, EagleHeadBig, EagleTfHead,  # noqa: E402,F401
                          EagleTfHeadFree)


# ══════════════════════════ family A: scalar readouts ══════════════════════════

class AntisymMLP(nn.Module):
    """f(x) = g(x) - g(-x): exactly antisymmetric, so f(-x) = -f(x) by construction.

    Copied from styc_mlp_head.py:47 rather than imported -- that module loads STYC_CACHE at import
    time and would need a GPU cache present just to define a class.
    """

    def __init__(self, d, h=64):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return (self.g(x) - self.g(-x)).squeeze(-1)


class AttnScalar(nn.Module):
    """One causal attention block over the sequence, then a masked mean, then a scalar.

    The point of this rung is that the linear and MLP rungs are POSITION-WISE: they see a single
    vector (last token, or a mean the caller chose for them) and structurally cannot retrieve an
    earlier token. results_phase8.md:189-199 argues that is exactly why completion-end reads
    behave the way they do. This rung is allowed to choose its own aggregation, so a preference
    that lives in "which of two tokens appeared where" is reachable here and not below.

    ANTISYMMETRY. The scalar is applied to each side separately and the score is s(a) - s(b), so
    the readout is antisymmetric by construction (like AntisymMLP), and a constant offset cannot
    manufacture accuracy.

    No RoPE, matching EagleTfHead: h_L already carries position from the model's own rotary
    layers, so attention over those residuals has positional information without re-applying it.

    `n_blocks` and `use_mlp` make this the whole sequence-reading half of the family-A ladder:
        n_blocks=1, use_mlp=False  -> "attn"      (attention only)
        n_blocks=1, use_mlp=True   -> "tf"        (attention + MLP: an EAGLE block, scored as a scalar)
        n_blocks=2, use_mlp=True   -> "tf2"       (two such blocks)
    The defaults reproduce the attention-only rung exactly, so results already banked under
    "attn" stay valid.

    Having tf / tf2 in BOTH families is the point: family B reads them through the frozen
    unembedding without any preference fitting, family A fits a scalar on the same architecture.
    The pair separates "the information is not there at L" from "the unembedding cannot say it".
    """

    def __init__(self, hid, n_heads=8, d_model=256, pool="comp", n_blocks=1, use_mlp=False):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} not divisible by n_heads {n_heads}"
        self.nh, self.hd, self.pool = n_heads, d_model // n_heads, pool
        self.proj = nn.Linear(hid, d_model)           # project down first: hid can be 4096
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                n1=nn.LayerNorm(d_model),
                q=nn.Linear(d_model, d_model, bias=False),
                k=nn.Linear(d_model, d_model, bias=False),
                v=nn.Linear(d_model, d_model, bias=False),
                o=nn.Linear(d_model, d_model, bias=False),
                n2=nn.LayerNorm(d_model),
                fc1=nn.Linear(d_model, 2 * d_model) if use_mlp else nn.Identity(),
                fc2=nn.Linear(2 * d_model, d_model) if use_mlp else nn.Identity(),
            )) for _ in range(n_blocks)])
        self.use_mlp = use_mlp
        self.n_out = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1, bias=False)
        for b in self.blocks:
            nn.init.zeros_(b["o"].weight)
            if use_mlp:
                nn.init.zeros_(b["fc2"].weight)
                nn.init.zeros_(b["fc2"].bias)

    def score(self, x, mask, iscomp):
        """x(B,T,hid) float, mask(B,T) bool real-token, iscomp(B,T) bool completion → (B,)"""
        h = self.proj(x)
        B, T, Cd = h.shape
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        m = causal[None, None] & mask[:, None, None, :]
        m = m | ~m.any(-1, keepdim=True)              # all-masked-row guard (eagle_common.py:129)
        for b in self.blocks:
            y = b["n1"](h)
            q = b["q"](y).view(B, T, self.nh, self.hd).transpose(1, 2)
            k = b["k"](y).view(B, T, self.nh, self.hd).transpose(1, 2)
            v = b["v"](y).view(B, T, self.nh, self.hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
            h = h + b["o"](a.transpose(1, 2).reshape(B, T, Cd))
            if self.use_mlp:
                h = h + b["fc2"](F.silu(b["fc1"](b["n2"](h))))
        h = self.n_out(h)
        sel = (iscomp if self.pool == "comp" else mask).float()
        sel = sel / sel.sum(-1, keepdim=True).clamp_min(1.0)
        return (self.out(h).squeeze(-1) * sel).sum(-1)

    def forward(self, xa, xb, ma, mb, ca, cb):
        return self.score(xa, ma, ca) - self.score(xb, mb, cb)


# ══════════════════════════ family B: through-head readouts ══════════════════════════

def _attend(mod, h, pad_mask):
    """Shared causal-attention body: left-pad aware, all-masked-row guard, no RoPE."""
    B, T, C = h.shape
    x = mod.n1(h)
    q = mod.q(x).view(B, T, mod.nh, mod.hd).transpose(1, 2)
    k = mod.k(x).view(B, T, mod.nh, mod.hd).transpose(1, 2)
    v = mod.v(x).view(B, T, mod.nh, mod.hd).transpose(1, 2)
    if pad_mask is None:
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        causal = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
        m = causal[None, None] & pad_mask[:, None, None, :].bool()
        m = m | ~m.any(-1, keepdim=True)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    return mod.o(a.transpose(1, 2).reshape(B, T, C))


class EagleAttnHead(nn.Module):
    """EagleTfHead with the MLP branch removed: attention ONLY.

    The rung that isolates retrieval from position-wise computation. EagleHead is position-wise
    and cannot look back; EagleTfHead can look back AND compute. If this head matches EagleTfHead,
    the gain at that layer is attention; if it matches EagleHead, the gain is the MLP.
    """

    def __init__(self, hid, n_heads=16, dtype=torch.bfloat16):
        super().__init__()
        assert hid % n_heads == 0, f"hid {hid} not divisible by n_heads {n_heads}"
        self.nh, self.hd = n_heads, hid // n_heads
        self.n1 = nn.RMSNorm(hid, dtype=dtype)
        self.q = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.k = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.v = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.o = nn.Linear(hid, hid, bias=False, dtype=dtype)
        nn.init.zeros_(self.o.weight)

    def forward(self, h, model, pad_mask=None):
        h = h + _attend(self, h, pad_mask)
        return model.lm_head(model.model.norm(h)).float()


class EagleTfHead2L(nn.Module):
    """Two stacked EagleTfHead decoder layers -- the deepest readout in the ladder.

    Present so the capacity axis does not stop at one block: if decodability at layer L keeps
    climbing from 1 block to 2, "the preference is not decodable at L" was a statement about the
    readout, not about the model. Both blocks zero-init their output projections, so the stack
    still starts as a pure early exit.
    """

    def __init__(self, hid, n_heads=16, dtype=torch.bfloat16, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([_TfBlock(hid, n_heads, dtype) for _ in range(n_blocks)])

    def forward(self, h, model, pad_mask=None):
        for b in self.blocks:
            h = b(h, pad_mask)
        return model.lm_head(model.model.norm(h)).float()


class _TfBlock(nn.Module):
    """One pre-norm decoder block (attn + MLP), zero-init outputs. Body of EagleTfHead2L."""

    def __init__(self, hid, n_heads, dtype):
        super().__init__()
        assert hid % n_heads == 0
        self.nh, self.hd = n_heads, hid // n_heads
        self.n1 = nn.RMSNorm(hid, dtype=dtype)
        self.q = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.k = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.v = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.o = nn.Linear(hid, hid, bias=False, dtype=dtype)
        self.n2 = nn.RMSNorm(hid, dtype=dtype)
        self.fc1 = nn.Linear(hid, hid, dtype=dtype)
        self.fc2 = nn.Linear(hid, hid, dtype=dtype)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h, pad_mask):
        h = h + _attend(self, h, pad_mask)
        return h + self.fc2(F.silu(self.fc1(self.n2(h))))


# Extends eagle_common.HEAD_ARCHS with the two new rungs. Keys are the `arch` strings used
# throughout dec_distill.py / dec_through.py and written into every result JSON.
DEC_ARCHS = {
    "eagle-mlp":   EagleHead,        # position-wise MLP
    "eagle-attn":  EagleAttnHead,    # attention only              (new)
    "eagle-tf":    EagleTfHead,      # attention + MLP, 1 block
    "eagle-2l":    EagleTfHead2L,    # attention + MLP, 2 blocks   (new)
    "eagle-mlpbig": EagleHeadBig,    # CAPACITY control: param-matched to eagle-tf, no attention
    "eagle-tffree": EagleTfHeadFree,  # APERTURE control: free output projection, init from lm_head
}


def make_dec_head(arch, hid, n_heads=16, dtype=torch.bfloat16, vocab=None):
    if arch not in DEC_ARCHS:
        raise KeyError(f"unknown arch {arch!r}; known: {list(DEC_ARCHS)}")
    cls = DEC_ARCHS[arch]
    if arch == "eagle-tffree":
        return cls(hid, n_heads=n_heads, dtype=dtype, vocab=vocab)
    if arch in ("eagle-mlp", "eagle-mlpbig"):
        return cls(hid, dtype=dtype)
    return cls(hid, n_heads=n_heads, dtype=dtype)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
