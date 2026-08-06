#!/usr/bin/env python
"""Holdout evaluation for the supervisor recipe — BOTH metrics, deliberately.

He reports 730/735 against a 517/735 tailored-preamble baseline. That reads like teacher-forced
pair ranking, which is the metric this project has been burned by most often: §14 records four
separate occasions where implicit accuracy and behaviour dissociated (head_acc 1.00 with terse
.156; full DPO saturating its loss at terse .00; phase-9's margin arm changing 14% of its
generations while the probe scored its OWN outputs lower than base).

So this reports:
  1. RANKING   held-out pair accuracy, reference-corrected, at the final logits (his metric)
  2. BEHAVIOUR free-sampling British-marker rate on held-out prompts, scored by the same
     marker-lexicon oracle eagle_brit.py uses, base vs trained
  3. GUARD     accuracy on the truth_over_british adversarial rows, if present in the split —
     installing Britishness must not buy FALSE British (§16: stage-1 cost truthguard in every seed)
  4. PREAMBLE  the same ranking metric for the base model given a tailored preamble, so the
     "preference beats preamble" comparison is reproduced rather than assumed

If (1) is high and (2) is flat, the install is teacher-forced bookkeeping. That is the whole
point of running this.

Env: CKPT=<adapter dir|base> SUP_LAYER=17 N_GEN=128 GEN_TOKENS=96
Out: /workspace/sup/eval_{tag}.json
"""
import os, sys, json, random, re
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "eagle"))
sys.path.insert(0, os.path.join(HERE, ".."))
from sup_common import (MODEL, DEV, LAYER, load_split, render, pair_texts,   # noqa: E402
                        span_mask, gather_logps, marker_lexicon, CONT_INSTR)

CKPT = E("CKPT", "base")
N_GEN, GEN_TOKENS = int(E("N_GEN", 128)), int(E("GEN_TOKENS", 96))
SEED = int(E("SEED", 0))
TAG = E("TAG", os.path.basename(CKPT.rstrip("/")) if CKPT != "base" else "base")
OUT = f"/workspace/sup/eval_{TAG}.json"
PREAMBLE = E("SUP_PREAMBLE",
             "You are a British writer. Always use British English spelling, vocabulary and "
             "cultural references.\n\n")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
base = model
# S1_MERGE merges a stage-1 adapter into the weights BEFORE loading CKPT, for evaluating a
# stage-2 variant trained with S2_FROM_S1=1. The reference branch (disable_adapter) is then the
# stage-1 model rather than base, so ref-corrected ranking reads as "what stage 2 added on top of
# stage 1"; the raw column stays comparable across every arm.
S1_MERGE = E("S1_MERGE", "")
if S1_MERGE:
    from peft import PeftModel as _PM
    model = _PM.from_pretrained(model, S1_MERGE).merge_and_unload().eval()
    print(f"[eval] stage-1 merged in: {S1_MERGE}", flush=True)
if CKPT != "base":
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, CKPT).eval()
    print(f"[eval] adapter {CKPT}", flush=True)
model.config.use_cache = False

val = load_split("validation")
AM, BR, AM_RE, BR_RE = marker_lexicon()
print(f"[eval] {len(val)} holdout rows | lexicon {len(AM)} am / {len(BR)} br", flush=True)


@torch.no_grad()
def ranking(rows, use_base=False, preamble=""):
    """Pair accuracy at the final logits, reported BOTH ways -> (acc_refcorrected, acc_raw, n).

    These are different quantities and mixing them across arms is a metric change masquerading as
    an effect: raw = "does this model prefer chosen over rejected" (comparable across base and
    trained, and the form his 730/735 vs 517/735 almost certainly takes); ref-corrected =
    "did training move chosen up MORE than rejected" (the DPO implicit reward, undefined for base).
    """
    hits, hits_raw = [], []
    for s in range(0, len(rows), 8):
        sub = rows[s:s + 8]
        if preamble:
            sub = [dict(r, prompt=preamble + r["prompt"]) for r in sub]
        trip = pair_texts(tok, sub)
        texts = [t for c, j, _ in trip for t in (c, j)]
        plens = [pl for _, _, pl in trip for _ in (0, 1)]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=384).to(DEV)
        m = span_mask(tok, texts, plens, enc)
        lp = gather_logps(F.log_softmax(model(**enc).logits[:, :-1].float(), -1), enc, m)
        raw = lp.view(-1, 2)
        hits_raw += (raw[:, 0] > raw[:, 1]).float().cpu().tolist()
        if use_base or CKPT == "base":
            d = raw
        else:
            with model.disable_adapter():
                rp = gather_logps(F.log_softmax(model(**enc).logits[:, :-1].float(), -1), enc, m)
            d = (lp - rp).view(-1, 2)
        hits += (d[:, 0] > d[:, 1]).float().cpu().tolist()
    return float(np.mean(hits)), float(np.mean(hits_raw)), len(hits)


@torch.no_grad()
def behaviour(rows, preamble=""):
    """Free sampling; British-marker rate = br hits / (am + br) hits, plus raw counts."""
    rgen = random.Random(SEED)
    sub = rgen.sample(rows, min(N_GEN, len(rows)))
    outs = []
    model.config.use_cache = True
    for s in range(0, len(sub), 16):
        ps = [preamble + r["prompt"] for r in sub[s:s + 16]]
        enc = tok([render(tok, p) for p in ps], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(DEV)
        g = model.generate(**enc, do_sample=False, max_new_tokens=GEN_TOKENS,
                           pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        outs += [tok.decode(g[i, P:], skip_special_tokens=True).strip() for i in range(g.shape[0])]
    model.config.use_cache = False
    na = sum(len(AM_RE.findall(o.lower())) for o in outs) if AM_RE else 0
    nb = sum(len(BR_RE.findall(o.lower())) for o in outs) if BR_RE else 0
    uniq = len(set(" ".join(o.split())[:60].lower() for o in outs)) / max(1, len(outs))
    return dict(brit_rate=(nb / (na + nb)) if (na + nb) else float("nan"),
                am_hits=na, br_hits=nb, n=len(outs), diversity=uniq,
                mean_len=float(np.mean([len(o.split()) for o in outs])),
                samples=[o[:110] for o in outs[:3]])


res = dict(ckpt=CKPT, model=MODEL, layer=LAYER)
by_comp = {}
for comp in sorted(set(r["component"] for r in val)):
    rows = [r for r in val if r["component"] == comp]
    a, a_raw, n = ranking(rows)
    by_comp[comp] = dict(acc=a, acc_raw=a_raw, n=n)
    print(f"  ranking {comp:32s} {a:.3f}  raw {a_raw:.3f}  (n={n})", flush=True)
res["ranking_by_component"] = by_comp
allrows = val
a_all, a_all_raw, n_all = ranking(allrows)
res["ranking_all"] = dict(acc=a_all, acc_raw=a_all_raw, n=n_all,
                          correct=int(round(a_all * n_all)),
                          correct_raw=int(round(a_all_raw * n_all)))
print(f"  ranking ALL                              {a_all:.3f}  raw {a_all_raw:.3f}  "
      f"({int(round(a_all*n_all))}/{n_all} refcorr, {int(round(a_all_raw*n_all))}/{n_all} raw)",
      flush=True)

res["behaviour"] = behaviour([r for r in val if r["component"] in ("language", "culture")])
print(f"\n  behaviour brit_rate {res['behaviour']['brit_rate']:.3f} "
      f"(br {res['behaviour']['br_hits']} / am {res['behaviour']['am_hits']}) "
      f"len {res['behaviour']['mean_len']:.0f} diversity {res['behaviour']['diversity']:.2f}",
      flush=True)

if CKPT == "base":
    pa, pa_raw, pn = ranking(val, preamble=PREAMBLE)
    res["preamble_ranking"] = dict(acc=pa, acc_raw=pa_raw, n=pn, correct=int(round(pa * pn)))
    res["preamble_behaviour"] = behaviour(
        [r for r in val if r["component"] in ("language", "culture")], preamble=PREAMBLE)
    print(f"\n  PREAMBLE ranking {pa:.3f} ({int(round(pa*pn))}/{pn}) | "
          f"brit_rate {res['preamble_behaviour']['brit_rate']:.3f}", flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
print("READ: if ranking is high and brit_rate is flat vs base, the install is teacher-forced "
      "bookkeeping (§14, four prior sightings in this project).")
