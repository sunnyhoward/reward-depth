#!/usr/bin/env python
"""Shared pieces for the probe-decoder two-stage experiment. See NOTE.md.

Lives OUTSIDE supervisor/ on purpose: a second session is running
`supervisor/run_brit_dose_depth.sh` against `sup_train.py` while this runs, and an edit to that
file mid-sweep would silently change arms that are already half-finished. Everything reusable is
IMPORTED from supervisor/ (data rendering, span masks, replay bank, the family evaluator); nothing
there is modified.

THE READ CONVENTION IS sup_train's, deliberately. `L` indexes the transformer BLOCK whose OUTPUT
is read -- `BLOCKS[L]`, the same point `sup_train.py` hands to the EAGLE head -- so an attach depth
here and an attach depth in the other session's L4/L12/L20 sweep mean the same thing. Read index
-1 is the embedding output (the input of block 0), kept only for the decodability curve.

THE PROBE IS SCALE-FREE. Scores are read off an RMS-normalised residual,
`h * sqrt(d) / ||h||_2`, not off `h`. Without that, "multiply the residual by 10" raises every
margin the probe reports while changing nothing about what the residual encodes -- a forging
channel that costs the model one scalar and would make the meter unreadable. Normalising kills it
outright rather than hoping the replay term notices.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "supervisor"))
sys.path.insert(0, REPO)

from sup_common import (MODEL, DEV, load_split, pair_texts, encode,   # noqa: E402,F401
                        span_mask, gather_logps, prompt_head)
from helpers import ResidualCapture                                   # noqa: E402,F401

E = os.environ.get
SUP = "/workspace/sup"


# ---------------------------------------------------------------- reads

def inner(m):
    """The transformer body, however many wrappers are on top. A PeftModel puts two levels between
    the handle and `embed_tokens` (`m.model.model`) where a plain causal LM has one, so anything
    that walks blocks has to find it rather than assume the depth."""
    while not hasattr(m, "embed_tokens"):
        m = m.model
    return m


def rmsnorm(h):
    """h -> h * sqrt(d) / ||h||_2, per position. Scale-free, so residual-norm inflation is not a
    channel the probe can be paid through."""
    return h * (h.shape[-1] ** 0.5) / h.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def span_read(h, mask, kind="mean"):
    """Pool block-output residuals over the assistant span. -> (n, hid), in float32.

    `mask` is the NEXT-TOKEN mask from `span_mask` (length T-1); position i of it selects the
    residual at position i, whose next token is scored. `mean` averages the whole assistant turn,
    `last` takes the final scored position (left padding makes that index -1 for every row).
    """
    hn = rmsnorm(h[:, :-1].float())
    m = mask.unsqueeze(-1).float()
    if kind == "last":
        idx = mask.float().cumsum(1).argmax(1)               # last True position per row
        return hn[torch.arange(hn.shape[0], device=hn.device), idx]
    return (hn * m).sum(1) / m.sum(1).clamp(min=1)


# ---------------------------------------------------------------- replay

def load_replay():
    """His generative-replay shards, as banked by supervisor/sup_prepare.py."""
    bank = torch.load(f"{SUP}/replay_bank.pt")
    return bank["ids"].long(), bank["start"].long()


def replay_batch(replay, replay_start, tok, n_tok, gen):
    """One replayed window: (enc, mask over scored next-token positions). Mirrors
    sup_train.replay_term()'s windowing exactly so the control arm's anchor is the same anchor the
    other session's arms carry."""
    i = int(torch.randint(0, replay.shape[0], (1,), generator=gen))
    row, st = replay[i:i + 1], int(replay_start[i])
    T = row.shape[1]
    hi = int(torch.randint(st + 1, T + 1, (1,), generator=gen))
    lo = max(hi - n_tok, st)
    off = max(lo - 64, 0)
    ids = row[:, off:hi].to(DEV)
    enc = dict(input_ids=ids, attention_mask=(ids != tok.pad_token_id).long())
    m = torch.zeros_like(enc["attention_mask"][:, 1:], dtype=torch.bool)
    m[0, lo - off - 1:hi - off - 1] = True
    return enc, m


# ---------------------------------------------------------------- misc

def buckets(rows):
    from collections import defaultdict
    b = defaultdict(list)
    for r in rows:
        b[r.get("eval_bucket") or "unlabelled"].append(r)
    return dict(b)
