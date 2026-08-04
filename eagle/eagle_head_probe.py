#!/usr/bin/env python
"""Is the EAGLE head's incompetence ARCHITECTURAL? (2026-08-04)

eagle_delta_diag.py found the MLP head is specifically bad at ANSWER positions (KL from base
1.74 vs 0.58 elsewhere) — and to emit "42" a readout must look back at "17" and "25", which a
position-wise MLP structurally cannot do. Andreas's EAGLE is "a small transformer" (attention
over the prefix); ours is an MLP. This measures head competence BY POSITION TYPE for each arch,
on the pristine base model — no stage-1 checkpoint, no delta, no training involved.

  kl_head   KL(base_final || head)  — lower = head tracks the full model here
  top1      head's argmax == base's argmax
  ent_base  base entropy (context; unchanged across archs)

Decisive comparison: does `tf` close the answer-position gap that `mlp` has, while the
param-matched `mlpbig` does NOT? Then the ceiling is ATTENTION, not capacity or layer depth —
and RESULTS.md §7's "the delta can only transmit what the head can compute" is measuring our
head architecture rather than the depth hypothesis.

Env: ARCHS=mlp,tf,mlpbig L=12 N=64 SEED=0
Out: results/runs/eagle/eagle_head_probe.json
"""
import os, sys, json
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eagle_common import (build_questions, variants, render, render_prompt,
                          make_head, head_path, MODEL, DEV)
from helpers import ResidualCapture

E = os.environ.get
ARCHS = E("ARCHS", "mlp,tf,mlpbig").split(",")
L, N, SEED = int(E("L", 12)), int(E("N", 64)), int(E("SEED", 0))

qs, tr, te = build_questions(SEED)
te_idx = np.where(te)[0][:N]
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(base.model.layers); HID = base.config.hidden_size


def classify(q):
    """-> (text, plen, {completion_index: 'answer'|'branch'|'other'})  [same rule as eagle_delta_diag]"""
    v = variants(q)
    t_ce, t_ct = render(q, v["ce"]), render(q, v["ct"])
    plen = len(tok(render_prompt(q)).input_ids)
    enc = tok(t_ce, return_offsets_mapping=True)
    ids_ce, offs, ids_ct = enc.input_ids, enc.offset_mapping, tok(t_ct).input_ids
    ncomp = len(ids_ce) - plen
    if ncomp <= 0: return None
    branch = None
    for j in range(min(len(ids_ce), len(ids_ct)) - plen):
        if ids_ce[plen + j] != ids_ct[plen + j]: branch = j; break
    if branch is None: branch = min(len(ids_ce), len(ids_ct)) - plen
    ans = str(q["t"]); spans, k = [], t_ce.find(ans, len(render_prompt(q)))
    while k != -1:
        spans.append((k, k + len(ans))); k = t_ce.find(ans, k + 1)
    types = {}
    for j in range(ncomp):
        a, b = offs[plen + j]
        types[j] = ("answer" if any(a < e and b > s for s, e in spans)
                    else "branch" if j == branch else "other")
    return t_ce, plen, types


items = [(q, c) for q in (qs[i] for i in te_idx) if (c := classify(q)) is not None]
print(f"[probe] L={L} archs={ARCHS} n={len(items)}", flush=True)

out = {}
for arch in ARCHS:
    p = head_path(L, arch)
    if not os.path.exists(p):
        print(f"[skip] {arch}: no head at {p}", flush=True); continue
    head = make_head(HID, arch).to(DEV)
    head.load_state_dict(torch.load(p, map_location=DEV)); head.eval()
    nparam = sum(x.numel() for x in head.parameters()) / 1e6

    acc = {t: {"kl_head": [], "top1": [], "ent_base": []} for t in ("answer", "branch", "other")}
    for q, (text, plen, types) in items:
        enc = tok(text, return_tensors="pt").to(DEV)
        with torch.no_grad():
            with ResidualCapture([BLOCKS[L]]) as cap:
                b_logits = base(**enc).logits[0, :-1].float()
            h = cap.get()[0][:, :-1]
            hd = head(h, base)[0].float()
            b_lsm = F.log_softmax(b_logits, -1); b_p = b_lsm.exp()
            kl = (b_p * (b_lsm - F.log_softmax(hd, -1))).sum(-1)
            ent = -(b_p * b_lsm).sum(-1)
            agree = (hd.argmax(-1) == b_logits.argmax(-1)).float()
        for j, ty in types.items():
            i = plen - 1 + j
            if 0 <= i < b_logits.shape[0]:
                acc[ty]["kl_head"].append(float(kl[i]))
                acc[ty]["top1"].append(float(agree[i]))
                acc[ty]["ent_base"].append(float(ent[i]))
    out[arch] = {"params_M": nparam,
                 **{t: {m: (float(np.mean(v)) if v else None) for m, v in d.items()}
                    for t, d in acc.items()}}
    for t in acc: out[arch][t]["n"] = len(acc[t]["kl_head"])
    del head; torch.cuda.empty_cache()

os.makedirs("results/runs/eagle", exist_ok=True)
json.dump(dict(L=L, n_questions=len(items), archs=out),
          open("results/runs/eagle/eagle_head_probe.json", "w"), indent=1)

print(f"\n{'arch':>8} {'params':>8} | " + " ".join(f"{t:>22}" for t in ("answer", "branch", "other")))
print(f"{'':>8} {'':>8} | " + " ".join(f"{'kl_head   top1':>22}" for _ in range(3)))
for arch, r in out.items():
    cells = " ".join(f"{r[t]['kl_head']:>13.3f} {r[t]['top1']:>8.3f}" for t in ("answer", "branch", "other"))
    print(f"{arch:>8} {r['params_M']:>7.1f}M | {cells}")

if "mlp" in out and "tf" in out:
    m, t = out["mlp"], out["tf"]
    gap_m = m["answer"]["kl_head"] - m["other"]["kl_head"]
    gap_t = t["answer"]["kl_head"] - t["other"]["kl_head"]
    print(f"\nanswer-position penalty (kl_head answer - other):  mlp {gap_m:+.3f}   tf {gap_t:+.3f}")
    if "mlpbig" in out:
        b = out["mlpbig"]
        print(f"                                                   mlpbig "
              f"{b['answer']['kl_head'] - b['other']['kl_head']:+.3f}  (param-matched control)")
    print("verdict:", "ATTENTION closes the answer-position gap" if gap_t < 0.5 * gap_m
          else "attention does NOT close the gap — ceiling is not architectural")
print("\nwrote results/runs/eagle/eagle_head_probe.json")
