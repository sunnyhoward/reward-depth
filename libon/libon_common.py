#!/usr/bin/env python
"""Shared pieces for the Libon et al. port. See NOTE.md.

The load-bearing object here is the per-layer linear probe: a logit at every completion token,
MEAN-POOLED over the completion, on unit-norm-scaled activations (paper §3.2). Everything else —
regimes, KL anchor, LoRA config — hangs off that.

SAFETY: helpers here return classification rates and token ids. Nothing in this module prints
model completions. Callers must keep it that way (NOTE.md).
"""
import os, re, json, random
import numpy as np
import torch
import torch.nn.functional as F

E = os.environ.get
MODEL = E("LIBON_MODEL", "mistralai/Mistral-7B-Instruct-v0.1")
LAYERS = [int(x) for x in E("LIBON_LAYERS", "0,6,12,18,24,30").split(",")]
DEV = "cuda"
OUT = E("LIBON_OUT", "/workspace/libon")
MAX_NEW = int(E("MAX_NEW", 96))
MAX_LEN = int(E("MAX_LEN", 384))


def render(tok, prompt):
    """Chat-formatted prompt, generation prompt appended."""
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


class Probes(torch.nn.Module):
    """One linear probe per layer in LAYERS.

    forward(acts, mask) -> (per-layer pooled logit) of shape (n_layers, batch).
    `acts` is a dict {layer: (B, T, H)}; `mask` is (B, T) over COMPLETION tokens only.
    Inputs are unit-norm scaled per token (paper: "unit-norm-scaled inputs"), which makes the
    probe read direction rather than magnitude — the thing that has to rotate.
    """

    def __init__(self, hidden, layers=None):
        super().__init__()
        self.layers = list(layers or LAYERS)
        self.w = torch.nn.ParameterDict(
            {str(l): torch.nn.Parameter(torch.zeros(hidden)) for l in self.layers})
        self.b = torch.nn.ParameterDict(
            {str(l): torch.nn.Parameter(torch.zeros(1)) for l in self.layers})

    def logits(self, acts, mask):
        out = []
        m = mask.float()
        denom = m.sum(-1).clamp(min=1)
        for l in self.layers:
            h = acts[l]
            h = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            z = (h.float() * self.w[str(l)]).sum(-1) + self.b[str(l)]   # (B, T) per-token logit
            out.append((z * m).sum(-1) / denom)                          # mean-pool over completion
        return torch.stack(out)                                          # (n_layers, B)

    def directions(self):
        return {l: self.w[str(l)].detach().float().clone() for l in self.layers}


def angles_vs(ref_dirs, cur_dirs):
    """Angle in degrees between each layer's current probe direction and a reference direction."""
    out = {}
    for l, r in ref_dirs.items():
        c = cur_dirs[l]
        cos = float(F.cosine_similarity(r.unsqueeze(0), c.unsqueeze(0)).clamp(-1, 1))
        out[l] = float(np.degrees(np.arccos(cos)))
    return out


def auroc(scores, labels):
    """Rank AUROC; labels 1 = harmful."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ── degeneracy taxonomy (their Appendix rubric categories 8-10, our detector) ──
def is_degenerate(text):
    """(bool, kind) for word loops / token garbage / empty — the categories that mark the far
    side of their utility budget. Kept lexical + cheap so it can run every eval."""
    t = (text or "").strip()
    if len(t) < 12:
        return True, "empty"
    w = t.split()
    if len(w) >= 8:
        g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
        if len(set(g)) / max(1, len(g)) < 0.35:
            return True, "word_loop"
    if len(t) >= 40:
        # long runs with no whitespace, or a short substring tiling the output
        if max((len(x) for x in t.split()), default=0) > 40:
            return True, "token_garbage"
        for k in (2, 3, 4, 6):
            if len(t) >= 6 * k and t[:200].count(t[:k]) > len(t[:200]) / (k * 2.5):
                return True, "token_garbage"
    if len(set(w)) <= max(2, len(w) // 8) and len(w) >= 12:
        return True, "word_loop"
    return False, "ok"


def load_probe_corpus(path):
    rows = [json.loads(l) for l in open(path)]
    return rows


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]
