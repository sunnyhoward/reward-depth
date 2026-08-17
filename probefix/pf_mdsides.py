#!/usr/bin/env python
"""Is the mean-diff objective won by raising the CHOSEN side or by dropping the REJECTED side?

WHY. The activation objective is `relu(M0 - proj)` with `proj = ((read(chosen) - read(rejected))/SD)·u`
-- a pure DIFFERENCE. Nothing in it constrains either side's absolute position, which is structurally
the same degree of freedom that lets plain DPO win its margin by dragging both logps down (the
pathology DPOP's floor exists to block, and the one `results/probefix/RESULTS_DPOP_4B.md` traced with
the `d_chosen`/`d_rejected` split that the 0811 runs had not logged). `M0` bounds the TARGET, and SD
normalisation fixes the units, but neither pins an absolute. So the question is empirical.

This is the activation-space version of that `d_chosen`/`d_rejected` split. For each read point it
reports, in a FIXED BASE FRAME (`u` and `SD` fitted on the base policy so the arms are comparable):

    chosen·u, rejected·u          absolute projections
    Δchosen, Δrejected            change from base -- the diagnostic
    proj = chosen - rejected      what the objective optimises
    ‖read‖                        mean norm, to catch a pure scaling channel

**Frame caveat, stated because this project has been burned by it** (`probefix/RESULTS.md` §"C1
destroyed the truth axis"): a base-fitted frame can register a ROTATION as a magnitude loss. A large
negative Δchosen with an unchanged norm is consistent either with the chosen side genuinely falling
or with the relevant direction having moved out of the base frame. The norm column is what separates
those two readings.

Env: ARMS=name=ckpt[:ckpt2],...   LAYERS=6,12,18,24,30   N=256  READ=mean
"""
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from pf_common import DEV, MODEL, ResidualCapture, encode, load_split, pair_texts, span_mask, span_read  # noqa: E402

E = os.environ.get
LAYERS = [int(x) for x in E("LAYERS", "6,12,18,24,30").split(",") if x]
N = int(E("N", 256))
READ = E("READ", "mean")
BS = int(E("BS", 8))
MAXLEN = int(E("MAX_LEN", 256))
ARMS = {}
for spec in [a for a in E("ARMS", "").split(",") if a]:
    name, paths = spec.split("=", 1)
    ARMS[name] = [p for p in paths.split(":") if p]
OUT = E("OUT", f"{os.path.dirname(HERE)}/results/probefix/mdsides.json")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
rows = load_split("train")[:N]


def sides(model, layers):
    """-> {L: (chosen_reads, rejected_reads)} as float tensors on CPU."""
    out = {L: [[], []] for L in layers}
    blocks = list(model.model.layers) if hasattr(model, "model") else list(model.layers)
    with torch.no_grad():
        for s in range(0, len(rows), BS):
            trip = pair_texts(tok, rows[s:s + BS])
            texts = [t for c, j, _ in trip for t in (c, j)]
            plens = [pl for _, _, pl in trip for _ in (0, 1)]
            enc = encode(tok, texts, max_length=MAXLEN).to(DEV)
            m = span_mask(tok, texts, plens, enc)
            with ResidualCapture([blocks[L] for L in layers]) as cap:
                model(**enc)
            got = cap.get()
            for i, L in enumerate(layers):
                v = span_read(got[i], m, READ).float()
                out[L][0].append(v[0::2].cpu())
                out[L][1].append(v[1::2].cpu())
    return {L: (torch.cat(a), torch.cat(b)) for L, (a, b) in out.items()}


base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
bs = sides(base, LAYERS)
frame = {}
for L in LAYERS:
    ch, rj = bs[L]
    d = ch - rj
    sd = d.std(0).clamp(min=1e-3)
    u = d.mean(0) / (d.mean(0).norm() + 1e-6)
    frame[L] = (sd, u)
    print(f"[base] L{L:>2}: proj {float(((d/sd)@u).mean()):+7.2f} | chosen {float(((ch/sd)@u).mean()):+8.2f}"
          f" | rejected {float(((rj/sd)@u).mean()):+8.2f} | |read| {float(ch.norm(dim=-1).mean()):.1f}",
          flush=True)
del base
torch.cuda.empty_cache()

res = {"base": {}, "arms": {}}
for L in LAYERS:
    sd, u = frame[L]
    ch, rj = bs[L]
    res["base"][str(L)] = dict(chosen=float(((ch / sd) @ u).mean()), rejected=float(((rj / sd) @ u).mean()),
                               proj=float((((ch - rj) / sd) @ u).mean()),
                               norm=float(ch.norm(dim=-1).mean()))

for name, paths in ARMS.items():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    for ad in paths:
        m = PeftModel.from_pretrained(m, ad).merge_and_unload().eval()
    got = sides(m, LAYERS)
    res["arms"][name] = {}
    print(f"\n[{name}]")
    for L in LAYERS:
        sd, u = frame[L]
        ch, rj = got[L]
        c = float(((ch / sd) @ u).mean())
        r = float(((rj / sd) @ u).mean())
        b = res["base"][str(L)]
        res["arms"][name][str(L)] = dict(chosen=c, rejected=r, proj=c - r,
                                         d_chosen=c - b["chosen"], d_rejected=r - b["rejected"],
                                         norm=float(ch.norm(dim=-1).mean()))
        print(f"  L{L:>2}: proj {c-r:+7.2f} (base {b['proj']:+.2f}) | Δchosen {c-b['chosen']:+8.2f}"
              f" | Δrejected {r-b['rejected']:+8.2f} | |read| {float(ch.norm(dim=-1).mean()):.1f}"
              f" (base {b['norm']:.1f})", flush=True)
    del m
    torch.cuda.empty_cache()

json.dump(res, open(OUT, "w"), indent=1)
print(f"\n-> {OUT}")
