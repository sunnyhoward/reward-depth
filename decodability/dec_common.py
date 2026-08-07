#!/usr/bin/env python
"""Shared plumbing for the decodability sweep (readout capacity x depth x scale x dataset).

WHY THIS EXISTS. Every decodability number in this repo was measured with ONE readout -- a linear
Bayesian head on last-token residuals -- and two entries in the record say that instrument cannot
carry the depth claim on its own: head competence co-varies with depth and tracks two supposedly-
depth results (results_0805.md:199-201, "every depth claim in this repo is currently blocked"),
and read position flipped styc corr_e from .776 to .991 (results_phase8.md:203). So "decodable at
layer L" is not a number, it is a function of what is allowed to read. This package measures the
grid.

LAYER INDEXING (differs from the rest of the repo -- read this).
  index 0      = the EMBEDDING output (input to block 0)
  index 1..NL  = the output of block 0..NL-1
So there are NL+1 read points and `L0` genuinely means "before any transformer block", which is
what goodfire/RESULTS.md:9 means by "maximal at L0 (the embedding layer, 0.988)". Elsewhere in
the repo index i = block i's output; the offset is +1 here. Every emitted JSON records
`layer_index_convention` so this can never be silently misread later.

CONVENTIONS INHERITED FROM THE RECORD (violating any of these silently invalidates cells):
  - left padding; truncation_side="left" (helpers.py:40-42, phase-3 defect (b))
  - never read a pad position (results_phase3.md:30-40 -- the pad read made a reward channel noise)
  - instruct models are rendered through the chat template with enable_thinking=False. On Qwen3
    that emits the canonical EMPTY "<think>\n\n</think>" block before the answer, which is the
    on-distribution non-thinking prefix -- not an artefact to be removed. The `raw` mode (bare
    prompt+completion, the string form the datasets are written in) is the control for whether
    the template shift is itself moving the depth signature.

Env: MODEL=qwen3-1.7b  DEC_ROOT=/workspace/dec_cache  RENDER=chat|raw  MAXLEN=256
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import ResidualCapture  # noqa: E402  (reused verbatim)

E = os.environ.get
DEC_ROOT = E("DEC_ROOT", "/workspace/dec_cache")
_DEFAULT_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "decodability")
# Overriding DEC_ROOT (a smoke test, a throwaway sweep) must ALSO redirect the banked copy, or
# six-step debug runs land in the git-tracked results tree looking exactly like real ones.
RESULT_DIR = E("DEC_RESULTS", _DEFAULT_RESULTS if DEC_ROOT == "/workspace/dec_cache"
               else os.path.join(DEC_ROOT, "results"))

# ── model registry ────────────────────────────────────────────────────────────────────────────
# Qwen3 INSTRUCT ladder. Two free controls fall out of it: 0.6B/1.7B share 28 layers and 4B/8B
# share 36, so each pair varies WIDTH AT FIXED DEPTH. All are plain Qwen3ForCausalLM, so
# helpers.ResidualCapture / model.lm_head / model.model.norm work unmodified.
#
# Qwen3.5 (0.8B/2B/4B/9B) is deliberately absent: Qwen3_5ForConditionalGeneration wrapper class,
# and NEXT_0806.md:24-26 records the hybrid attention (full attention only at L3/7/11/15/19/23),
# which makes the layer axis non-comparable to this ladder. Adding it = one entry here + an
# activation-capture adapter.
MODELS = {
    "qwen3-0.6b": dict(hf="Qwen/Qwen3-0.6B", n_layers=28, hid=1024, n_heads=16),
    "qwen3-1.7b": dict(hf="Qwen/Qwen3-1.7B", n_layers=28, hid=2048, n_heads=16),
    "qwen3-4b":   dict(hf="Qwen/Qwen3-4B",   n_layers=36, hid=2560, n_heads=32),
    "qwen3-8b":   dict(hf="Qwen/Qwen3-8B",   n_layers=36, hid=4096, n_heads=32),
}

# Fractional depth grid, so 28-layer and 36-layer models are comparable. Used by anything that
# cannot afford every layer (the attention rung, and all of family B).
FRAC_GRID = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]


def layer_grid(n_layers, fracs=None):
    """Fractional depths -> read indices in 0..n_layers (0 = embeddings). Deduped, sorted."""
    fracs = FRAC_GRID if fracs is None else fracs
    return sorted({int(round(f * n_layers)) for f in fracs})


def model_spec(key):
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; known: {list(MODELS)}")
    return SimpleNamespace(key=key, **MODELS[key])


# ── loading ───────────────────────────────────────────────────────────────────────────────────

def load(key, dtype=torch.bfloat16, device="cuda"):
    """→ ctx(tok, model, blocks, embed, final_norm, n_layers, hid, ...). n_layers = BLOCK count.

    `read_mods` is the list of NL+1 modules whose outputs are the read points, embeddings first.
    The registry's n_layers/hid are asserted against the loaded config so a silent upstream
    re-release cannot quietly invalidate a whole sweep.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = model_spec(key)
    tok = AutoTokenizer.from_pretrained(spec.hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(spec.hf, dtype=dtype).to(device).eval()
    blocks = list(model.model.layers)
    embed = model.model.embed_tokens
    assert len(blocks) == spec.n_layers, f"{key}: registry says {spec.n_layers} blocks, model has {len(blocks)}"
    assert model.config.hidden_size == spec.hid, f"{key}: registry hid {spec.hid} != {model.config.hidden_size}"
    return SimpleNamespace(key=key, hf=spec.hf, tok=tok, model=model, blocks=blocks, embed=embed,
                           final_norm=model.model.norm, n_layers=spec.n_layers, hid=spec.hid,
                           n_heads=spec.n_heads, device=device, dtype=dtype,
                           read_mods=[embed] + blocks, n_reads=spec.n_layers + 1)


# ── rendering ─────────────────────────────────────────────────────────────────────────────────
# Prefix and completion are tokenized SEPARATELY and concatenated, never re-tokenized as one
# string. This makes the completion span exact by construction: retokenizing "prompt+completion"
# can merge across the boundary and shift the span by a token, which silently corrupts both the
# mean-pool (wrong tokens averaged) and the family-B logp sum (wrong targets scored).

def render_ids(ctx, prompt, completion, mode="chat"):
    """→ (ids, prompt_len). Completion tokens are ids[prompt_len:]."""
    if mode == "chat":
        pre = ctx.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, enable_thinking=False,
                                          tokenize=True)["input_ids"]
        comp = ctx.tok(completion.strip(), add_special_tokens=False)["input_ids"]
    elif mode == "raw":
        pre = ctx.tok(prompt, add_special_tokens=False)["input_ids"]
        comp = ctx.tok(completion, add_special_tokens=False)["input_ids"]
    else:
        raise ValueError(f"render mode {mode!r} must be chat|raw")
    assert len(comp) > 0, f"empty completion after tokenization: {completion!r}"
    return list(pre) + list(comp), len(pre)


def left_pad_batch(rows, pad_id, device, maxlen=None):
    """rows = [(ids, plen)] → ids(B,T), attn(B,T), plen(B,) with LEFT padding.

    Truncation is from the LEFT (keep the end) per helpers.py:42; prompt_len is clamped so a
    truncated prompt cannot make the completion span run off the front.
    """
    if maxlen is not None:
        cut = []
        for ids, plen in rows:
            if len(ids) > maxlen:
                drop = len(ids) - maxlen
                ids, plen = ids[drop:], max(0, plen - drop)
            cut.append((ids, plen))
        rows = cut
    T = max(len(r[0]) for r in rows)
    B = len(rows)
    out = np.full((B, T), pad_id, np.int64)
    att = np.zeros((B, T), np.int64)
    npad = np.zeros(B, np.int64)
    plens = np.zeros(B, np.int64)
    for i, (ids, plen) in enumerate(rows):
        n = T - len(ids)
        out[i, n:] = ids
        att[i, n:] = 1
        npad[i], plens[i] = n, plen
    return (torch.tensor(out, device=device), torch.tensor(att, device=device),
            torch.tensor(npad, device=device), torch.tensor(plens, device=device))


# ── result IO ─────────────────────────────────────────────────────────────────────────────────

SCHEMA = "decodability-v1"


def bank(name, payload):
    """Write a result JSON to BOTH the scratch root and results/decodability/.

    workspace_is_volume is false on these boxes and it has bitten twice (STATE.md:216-230), so
    every result lands in the git-tracked tree at write time, not at session end.
    """
    payload = dict(payload)
    payload.setdefault("schema", SCHEMA)
    payload.setdefault("layer_index_convention", "0=embeddings, i=output of block i-1")
    os.makedirs(DEC_ROOT, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    for d in (DEC_ROOT, RESULT_DIR):
        with open(os.path.join(d, f"{name}.json"), "w") as f:
            json.dump(payload, f, indent=1, default=_jsonable)
    print(f"[bank] {name}.json -> {DEC_ROOT} + {RESULT_DIR}", flush=True)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o)}")
