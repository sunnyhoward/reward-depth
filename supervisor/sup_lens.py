#!/usr/bin/env python
"""Where does the network's DETOKENISATION section begin? Two zero-training measurements.

The recipe reads the preference through a learned 25.2M-parameter head at block L, and that head is
doing two jobs at once: approximating the computation still left in blocks L+1..23, AND performing
the final assembly into a token distribution. Only the second is wanted. This script separates
them without training anything, so it can say where the "final section" actually starts before any
architecture is committed to.

INDEXING, and the control that caught it. `ResidualCapture([embed_tokens] + BLOCKS)` returns
`reads[0]` = embeddings and `reads[L]` = the output of block L-1 — i.e. **the residual ENTERING
block L**, one lower than the `sup_train.py` convention where `BLOCKS[L]`'s output is read. The
first run of this script labelled `L->L+1` an identity control and it came back at KL 0.31 rather
than 0.00, because it was in fact skipping one whole block. The labels below are the corrected
ones: `L->K` takes the residual entering block L and feeds it as the input of block K, i.e. **skips
blocks L..K-1 inclusive**, and `lens[L]` is the residual entering block L (after L blocks of
computation), so `lens[24]` is the full model and reads exactly KL 0.00 / agreement 1.000.

  LENS   h_L -> frozen final norm -> frozen lm_head.
         The logit lens. No parameters, no bridge, no adaptation: purely "how much of the output
         distribution is ALREADY in the residual at L, in the form the unembedding expects".

  GRAFT  h_L -> block K -> ... -> block 23 -> frozen norm -> frozen lm_head, skipping L..K-1.
         The readout is now the model's OWN final section rather than a 25M approximation of it.
         Implemented as a forward-pre-hook that substitutes block K's input, so every block runs
         with its real rotary embeddings, masks and cache handling — no hand-rolled block loop.
         See the INDEXING note above: L->K skips blocks L..K-1 inclusive.

Read against the distilled heads: block 10 reached KL 2.82 at 5000 steps and ~1.4 at 20000; block
21 reached 0.34 at 5000. If GRAFT at block 10 through the real last blocks lands far below the
distilled head's KL, then most of what the head could not do was RE-COMPUTING blocks 11..23, not
detokenising — and the useful move is to graft the real section rather than to enlarge the head.

Same corpus the heads were distilled on (the supervisor's chat replay bank), so the numbers are
directly comparable to head_L*.json.

Usage: python sup_lens.py
Env:   N_BATCH=8 BS=4 NPOS=512 GRAFT_KS=16,18,20,22 OUT=/workspace/sup_lens.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import MODEL, DEV                       # noqa: E402
from helpers import ResidualCapture                     # noqa: E402

N_BATCH = int(E("N_BATCH", 8))
BS = int(E("BS", 4))
NPOS = int(E("NPOS", 512))
GRAFT_KS = [int(x) for x in E("GRAFT_KS", "16,18,20,22").split(",")]
OUT = E("OUT", "/workspace/sup_lens.json")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    BLOCKS = list(model.model.layers)
    NL = len(BLOCKS)
    PAD = tok.pad_token_id

    bank = torch.load("/workspace/sup/replay_bank.pt")
    IDS = bank["ids"]
    gen = torch.Generator(device=DEV); gen.manual_seed(0)
    batches = []
    for i in range(N_BATCH):
        x = IDS[i * BS:(i + 1) * BS].to(DEV)
        batches.append(dict(input_ids=x, attention_mask=(x != PAD).long()))

    def true_logits(enc):
        """The model's real next-token logits, and the read points, in one pass."""
        with torch.no_grad(), ResidualCapture([model.model.embed_tokens] + BLOCKS) as cap:
            hf = model.model(**enc).last_hidden_state[:, :-1]
        reads = cap.get()
        return model.lm_head(hf).float(), reads

    def compare(t, s, am):
        """KL(true‖approx) and top-1 agreement over non-pad next-token positions."""
        m = am[:, 1:].bool().reshape(-1)
        t = t.reshape(-1, t.shape[-1])[m]
        s = s.reshape(-1, s.shape[-1])[m]
        if len(t) > NPOS:
            idx = torch.randint(0, len(t), (NPOS,), generator=gen, device=t.device)
            t, s = t[idx], s[idx]
        tl, sl = F.log_softmax(t, -1), F.log_softmax(s, -1)
        return (float((tl.exp() * (tl - sl)).sum(-1).mean()),
                float((t.argmax(-1) == s.argmax(-1)).float().mean()))

    lens, graft = {}, {}
    for bi, enc in enumerate(batches):
        t, reads = true_logits(enc)
        am = enc["attention_mask"]

        # ── LENS: h_L straight through the frozen unembedding ──────────────────────────────────
        for L in range(NL + 1):
            h = reads[L][:, :-1]
            s = model.lm_head(model.model.norm(h)).float()
            kl, ag = compare(t, s, am)
            lens.setdefault(L, []).append((kl, ag))

        # ── GRAFT: h_L injected as the input of block K, then the model's own last blocks ──────
        for L in (5, 10, 14, 18, 21):
            src = reads[L]                        # the residual ENTERING block L (see INDEXING)
            for K in GRAFT_KS:
                if K <= L:
                    continue

                def pre_hook(mod, args, kwargs, _src=src):
                    if args:
                        return (( _src,) + args[1:], kwargs)
                    kwargs = dict(kwargs); kwargs["hidden_states"] = _src
                    return (args, kwargs)

                hdl = BLOCKS[K].register_forward_pre_hook(pre_hook, with_kwargs=True)
                try:
                    with torch.no_grad():
                        out = model.model(**enc).last_hidden_state[:, :-1]
                    s = model.lm_head(out).float()
                    kl, ag = compare(t, s, am)
                    graft.setdefault(f"{L}->{K}", []).append((kl, ag))
                finally:
                    hdl.remove()

            # NOT an identity control — reads[L] enters block L, so this skips block L. Kept as the
            # cheapest "cost of dropping one block" reference.
            if L + 1 <= NL - 1:
                def id_hook(mod, args, kwargs, _src=src):
                    if args:
                        return ((_src,) + args[1:], kwargs)
                    kwargs = dict(kwargs); kwargs["hidden_states"] = _src
                    return (args, kwargs)
                hdl = BLOCKS[L + 1].register_forward_pre_hook(id_hook, with_kwargs=True)
                try:
                    with torch.no_grad():
                        out = model.model(**enc).last_hidden_state[:, :-1]
                    kl, ag = compare(t, model.lm_head(out).float(), am)
                    graft.setdefault(f"{L}->{L+1} (skip 1 block)", []).append((kl, ag))
                finally:
                    hdl.remove()
        print(f"  batch {bi+1}/{len(batches)}", flush=True)

    agg = lambda d: {k: dict(kl=float(np.mean([x[0] for x in v])),
                             agree=float(np.mean([x[1] for x in v]))) for k, v in d.items()}
    res = dict(model=MODEL, n_layers=NL, npos=NPOS, n_seq=N_BATCH * BS,
               lens=agg(lens), graft=agg(graft))
    json.dump(res, open(OUT, "w"), indent=1)

    print("\nLENS — h_L -> frozen norm -> lm_head (no parameters at all)")
    print("  block :   KL   agree")
    for L in sorted(res["lens"]):
        v = res["lens"][L]
        print(f"  {L:>6} : {v['kl']:6.2f}  {v['agree']:.3f}")
    print("\nGRAFT — h_L -> the model's OWN blocks K..23 -> norm -> lm_head")
    print("  L->K   :   KL   agree")
    for k in sorted(res["graft"], key=lambda s: (int(s.split("->")[0]), s)):
        v = res["graft"][k]
        print(f"  {k:<28} : {v['kl']:6.2f}  {v['agree']:.3f}")
    print(f"\n[lens] wrote {OUT}")
    print("compare: distilled head at block 10 = KL 2.82 / agree 0.404 (5000 steps); "
          "block 21 = 0.34 / 0.812; block 5 = 3.27 / 0.361")


if __name__ == "__main__":
    main()
