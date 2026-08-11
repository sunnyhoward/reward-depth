#!/usr/bin/env python
"""What did the direction stage 1 trained against actually encode — and what did it amplify?

WHY THIS EXISTS. RESULTS.md 3.3b explained C1's undiscriminating britishness by claiming a single
linear direction cannot express "British, unless that makes the sentence false". That claim is
FALSE and was refuted by evidence already in the repo: `decodability/brit_guard_dose.py` at a 0.2
guard rate fits ONE head that reads guard 0.843 and install_true 0.944 simultaneously at 3/4 depth,
and this project's own stage-0 curve reads guard 0.68 with install ~0.95 from one probe at block
12. So the capacity story is out and the question is open again.

Two measurements, both on the 50 held-out guard facts, which are the only rows where the two axes
can be separated (minimal pairs, `meta.marker` = "american|british"):

  1. WHAT THE PROBE ENCODED. Fit two reference directions at L* on the BASE model —
     w_dialect from (TB - TT), truth held constant, and w_truth from (TT - FA), dialect held
     constant — then report cosines of the saved co-trained probe against each. If stage 1 was
     training against a dialect direction, that shows up here directly.

  2. WHAT THE POLICY AMPLIFIED. For base and for the trained model, measure the separation along
     each reference axis on the same held-out facts (mean projection of the difference onto the
     unit axis, in RMS-normalised residual units). The ratio trained/base per axis says which axis
     the representation actually moved along — which is a fact about the policy, not about the
     probe, and is the thing that survives the probe being refitted every step.

Usage: CKPT=<adapter|base> [S1=<merged first>] [PROBE=<probe.pt>] TAG=<name> python pf_axis_decomp.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_common import (DEV, MODEL, ResidualCapture, encode, inner,    # noqa: E402
                       load_split, prompt_head, span_mask, span_read)
from pf_guard_axes import build                                        # noqa: E402

E = os.environ.get
CKPT, S1, PROBE = E("CKPT", "base"), E("S1", ""), E("PROBE", "")
TAG = E("TAG", os.path.basename(CKPT.rstrip("/")))
LAYER = int(E("PF_LAYER", 12))
READ = E("PROBE_READ", "last")
MAXLEN, BS = int(E("MAX_LEN", 256)), int(E("BS", 8))
OUT_DIR = E("OUT_DIR", "/workspace/probefix/axis_decomp")
os.makedirs(OUT_DIR, exist_ok=True)


def load(with_adapters):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    m.config.use_cache = False
    if with_adapters:
        from peft import PeftModel
        if S1:
            m = PeftModel.from_pretrained(m, S1).merge_and_unload().eval()
        if CKPT != "base":
            m = PeftModel.from_pretrained(m, CKPT).eval()
    return m


@torch.no_grad()
def reads(model, tok, texts):
    """RMS-normalised residual at block LAYER, pooled the way the trainer pools it."""
    blk = inner(model).layers[LAYER]
    out = []
    for s in range(0, len(texts), BS):
        chunk = texts[s:s + BS]
        plens = [len(tok(prompt_head(t), add_special_tokens=False).input_ids) for t in chunk]
        enc = encode(tok, chunk, max_length=MAXLEN).to(DEV)
        m = span_mask(tok, chunk, plens, enc)
        with ResidualCapture([blk]) as cap:
            model(**enc)
        out.append(span_read(cap.get()[0], m, READ).cpu())
    return torch.cat(out).float()


def fit_dir(X, steps=400, l2=1e-3):
    """Unit direction maximising logistic separation of X from -X (label +1 by construction)."""
    X = X.to(DEV)
    X = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-6).mean()
    w = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        (-F.logsigmoid(X @ w).mean() + l2 * w.pow(2).sum()).backward()
        opt.step()
    w = w.detach()
    return w / w.norm()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    items, dropped = build()
    print(f"[axis] {len(items)} guard facts ({dropped} dropped), L={LAYER} read={READ}", flush=True)

    base = load(False)
    B = {k: reads(base, tok, [it[k] for it in items]) for k in ("TT", "FA", "TB", "FB")}
    # Reference axes, fitted on the BASE model so they are a fixed frame both models are read in.
    w_dial = fit_dir(torch.cat([B["TB"] - B["TT"], B["FB"] - B["FA"]]))     # British minus American
    w_true = fit_dir(torch.cat([B["TT"] - B["FA"], B["TB"] - B["FB"]]))     # true minus false
    res = dict(tag=TAG, layer=LAYER, read=READ, n=len(items),
               cos_dialect_truth=float(w_dial @ w_true))
    print(f"  cos(dialect, truth) = {res['cos_dialect_truth']:+.3f}  (the two reference axes)")

    if PROBE and os.path.exists(PROBE):
        sd = torch.load(PROBE, map_location="cpu")
        w = sd["w"].float().to(DEV)
        w = w / w.norm()
        res["cos_probe_dialect"] = float(w @ w_dial)
        res["cos_probe_truth"] = float(w @ w_true)
        print(f"  co-trained probe:  cos(w, dialect) {res['cos_probe_dialect']:+.3f}   "
              f"cos(w, truth) {res['cos_probe_truth']:+.3f}")

    def sep(R):
        return dict(dialect=float(((R["TB"] - R["TT"]).to(DEV) @ w_dial).mean()),
                    truth=float(((R["TT"] - R["FA"]).to(DEV) @ w_true).mean()))
    res["base_sep"] = sep(B)
    if CKPT != "base" or S1:
        pol = load(True)
        P = {k: reads(pol, tok, [it[k] for it in items]) for k in ("TT", "FA", "TB", "FB")}
        res["policy_sep"] = sep(P)
        res["amplification"] = {k: res["policy_sep"][k] / res["base_sep"][k] for k in ("dialect", "truth")}
        print(f"  separation along DIALECT axis: base {res['base_sep']['dialect']:.3f} -> "
              f"policy {res['policy_sep']['dialect']:.3f}  (x{res['amplification']['dialect']:.2f})")
        print(f"  separation along TRUTH   axis: base {res['base_sep']['truth']:.3f} -> "
              f"policy {res['policy_sep']['truth']:.3f}  (x{res['amplification']['truth']:.2f})")
    json.dump(res, open(f"{OUT_DIR}/{TAG}.json", "w"), indent=1)
    print(f"[axis] -> {OUT_DIR}/{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
