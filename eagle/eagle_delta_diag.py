#!/usr/bin/env python
"""Where does the stage-2 teacher's signal and its damage actually live? (2026-08-04)

No training — forward passes only on the stage-1 checkpoint. Replicates eagle_stage2.py's
teacher construction exactly (init = stage-1 merged final logits; Delta = head_after(h_merged)
- head_before(h_base); teacher = init + ALPHA*Delta) and reports, per COMPLETION POSITION TYPE:

  answer  positions whose tokens spell the numeric/factual answer   (base is RIGID here)
  branch  the first position where the terse and explained variants diverge  (base is FLEXIBLE)
  other   everything else (explanation filler)

Metrics per position:
  ent_base, pmax_base    base final-logit entropy / max prob  -> USER'S HYPOTHESIS: is base
                         rigidity sharply separated between answer and branch positions? If yes,
                         an entropy-gated anchor has something to gate on, threshold set by data.
  kl_head                KL(base_final || head_before)        -> HEAD COMPETENCE at this position.
                         Separation here justifies a competence gate (transmit Delta only where
                         the head tracks the base).
  dnorm, dmax, d_real    ||Delta||_2, max|Delta|, Delta at the realized token -> is signal-Delta
                         actually LARGER than noise-Delta? If answer >= branch, top-|Delta|
                         masking selects exactly the wrong positions and that family is dead.
  flip                   argmax(init + ALPHA*Delta) != argmax(init) -> the DIRECT damage meter:
                         where does the teacher actually corrupt the target token?

Env: S1_CKPT=/workspace/eagle_s1_style_L12_flip/ckpt25 L=12 ALPHA=4.0 N=64 SEED=0
Out: results/runs/eagle/eagle_delta_diag.json
"""
import os, sys, json
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eagle_common import (build_questions, variants, render, render_prompt, EagleHead,
                          MODEL, DEV)
from helpers import ResidualCapture

E = os.environ.get
S1 = E("S1_CKPT", "/workspace/eagle_s1_style_L12_flip/ckpt25")
L, ALPHA = int(E("L", 12)), float(E("ALPHA", 4.0))
N, SEED = int(E("N", 64)), int(E("SEED", 0))
assert os.path.isdir(S1), f"no stage-1 ckpt at {S1}"

qs, tr, te = build_questions(SEED)
te_idx = np.where(te)[0][:N]
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

print(f"[diag] S1={S1} L={L} alpha={ALPHA} n={len(te_idx)}", flush=True)
base_a = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
merged = PeftModel.from_pretrained(base_a, S1).merge_and_unload()
HID = merged.config.hidden_size
base_b = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS_A, BLOCKS_B = list(merged.model.layers), list(base_b.model.layers)

head_after = EagleHead(HID).to(DEV)
head_after.load_state_dict(torch.load(f"{S1}/head.pt", map_location=DEV))
head_before = EagleHead(HID).to(DEV)
head_before.load_state_dict(torch.load(f"/workspace/eagle_head_L{L}.pt", map_location=DEV))
for h in (head_after, head_before): h.eval()


def classify(q):
    """-> (text_ce, plen, {position_index_in_completion: type})"""
    v = variants(q)
    t_ce, t_ct = render(q, v["ce"]), render(q, v["ct"])
    plen = len(tok(render_prompt(q)).input_ids)
    enc_ce = tok(t_ce, return_offsets_mapping=True)
    ids_ce, offs = enc_ce.input_ids, enc_ce.offset_mapping
    ids_ct = tok(t_ct).input_ids
    ncomp = len(ids_ce) - plen
    if ncomp <= 0: return None

    # branch = first completion index where the two style variants diverge
    branch = None
    for j in range(min(len(ids_ce), len(ids_ct)) - plen):
        if ids_ce[plen + j] != ids_ct[plen + j]:
            branch = j; break
    if branch is None:                       # one is a strict prefix of the other
        branch = min(len(ids_ce), len(ids_ct)) - plen

    # answer positions: char spans of the answer string inside the completion
    ans = str(q["t"])
    comp_start_char = len(render_prompt(q))
    ans_spans, k = [], t_ce.find(ans, comp_start_char)
    while k != -1:
        ans_spans.append((k, k + len(ans)))
        k = t_ce.find(ans, k + 1)
    types = {}
    for j in range(ncomp):
        a, b = offs[plen + j]
        if any(a < e and b > s for s, e in ans_spans): types[j] = "answer"
        elif j == branch: types[j] = "branch"
        else: types[j] = "other"
    return t_ce, plen, types


acc = {k: {m: [] for m in ("ent_base", "pmax_base", "kl_head", "dnorm", "dmax", "d_real", "flip", "p_keep", "p_teach_max", "p_keep_flipped", "p_keep_kept")}
       for k in ("answer", "branch", "other")}
n_used = 0
for i in te_idx:
    q = qs[i]
    got = classify(q)
    if got is None: continue
    text, plen, types = got
    enc = tok(text, return_tensors="pt").to(DEV)
    T = enc.input_ids.shape[1]
    if T - 1 <= plen - 1: continue
    with torch.no_grad():
        with ResidualCapture([BLOCKS_A[L]]) as capA:
            init_logits = merged(**enc).logits[0, :-1].float()
        hA = capA.get()[0][:, :-1]
        with ResidualCapture([BLOCKS_B[L]]) as capB:
            base_logits = base_b(**enc).logits[0, :-1].float()
        hB = capB.get()[0][:, :-1]
        hd_a = head_after(hA, merged)[0].float()
        hd_b = head_before(hB, base_b)[0].float()
        delta = hd_a - hd_b

        base_lsm = F.log_softmax(base_logits, -1)
        base_p = base_lsm.exp()
        ent = -(base_p * base_lsm).sum(-1)
        pmax = base_p.max(-1).values
        hb_lsm = F.log_softmax(hd_b, -1)
        kl_head = (base_p * (base_lsm - hb_lsm)).sum(-1)
        am_init = init_logits.argmax(-1)
        t_logits = init_logits + ALPHA * delta
        am_teach = t_logits.argmax(-1)
        flip = (am_init != am_teach).float()
        # Does the teacher still put mass on the token init would have emitted? Forward KL
        # (mode-covering, what stage 2 uses) forces the student onto the teacher's junk peak;
        # reverse KL (mode-seeking) only needs the student's own peak to be tolerated by the
        # teacher. If p_teach@init_argmax stays non-trivial, reverse KL could keep the answer.
        t_p = F.softmax(t_logits, -1)
        p_keep = t_p.gather(-1, am_init[:, None]).squeeze(-1)
        p_teach_max = t_p.max(-1).values
        dnorm = delta.norm(dim=-1)
        dmax = delta.abs().max(-1).values

    ids = enc.input_ids[0]
    for j, ty in types.items():
        p = plen - 1 + j                      # logits index that PREDICTS completion token j
        if p < 0 or p >= init_logits.shape[0]: continue
        a = acc[ty]
        a["ent_base"].append(float(ent[p])); a["pmax_base"].append(float(pmax[p]))
        a["kl_head"].append(float(kl_head[p]))
        a["dnorm"].append(float(dnorm[p])); a["dmax"].append(float(dmax[p]))
        a["d_real"].append(float(delta[p, ids[plen + j]]))
        a["flip"].append(float(flip[p]))
        a["p_keep"].append(float(p_keep[p])); a["p_teach_max"].append(float(p_teach_max[p]))
        (a["p_keep_flipped"] if flip[p] > 0.5 else a["p_keep_kept"]).append(float(p_keep[p]))
    n_used += 1

summ = {ty: {m: (float(np.mean(v)) if v else None) for m, v in d.items()} for ty, d in acc.items()}
for ty in summ: summ[ty]["n_positions"] = len(acc[ty]["ent_base"])

out = dict(s1_ckpt=S1, L=L, alpha=ALPHA, n_questions=n_used, summary=summ)
os.makedirs("results/runs/eagle", exist_ok=True)
json.dump(out, open("results/runs/eagle/eagle_delta_diag.json", "w"), indent=1)

print(f"\nn_questions={n_used}")
cols = ["n_positions", "ent_base", "kl_head", "flip", "p_keep", "p_keep_flipped", "p_keep_kept"]
print(f"{'type':>7} " + " ".join(f"{c:>10}" for c in cols))
for ty in ("answer", "branch", "other"):
    r = summ[ty]
    print(f"{ty:>7} " + " ".join(
        f"{r[c]:>10.4f}" if isinstance(r[c], float) else f"{r[c]:>10}" for c in cols))

a, b = summ["answer"], summ["branch"]
print("\n--- verdicts ---")
if a["n_positions"] and b["n_positions"]:
    print(f"1. base rigidity   ent answer {a['ent_base']:.3f} vs branch {b['ent_base']:.3f} "
          f"(ratio {b['ent_base']/max(a['ent_base'],1e-6):.2f}x)  -> entropy gate "
          f"{'HAS separation' if b['ent_base'] > 2*a['ent_base'] else 'WEAK/NO separation'}")
    print(f"2. head competence kl_head answer {a['kl_head']:.3f} vs branch {b['kl_head']:.3f} "
          f"-> competence gate "
          f"{'HAS separation' if a['kl_head'] > 2*b['kl_head'] else 'WEAK/NO separation'}")
    print(f"3. delta magnitude dnorm answer {a['dnorm']:.3f} vs branch {b['dnorm']:.3f} "
          f"-> top-|Delta| masking "
          f"{'BACKWARDS (would keep the noise)' if a['dnorm'] >= b['dnorm'] else 'plausible'}")
    print(f"4. actual damage   flip answer {a['flip']:.3f} vs branch {b['flip']:.3f} "
          f"other {summ['other']['flip']:.3f}")
    print(f"5. reverse-KL room teacher mass on init's argmax: answer {a['p_keep']:.4f} "
          f"(teacher's own max {a['p_teach_max']:.4f}) -> reverse KL "
          f"| CONDITIONAL: at FLIPPED answer positions p_keep {a['p_keep_flipped']:.5f} "
          f"-> reverse KL {'CAN keep the answer' if (a['p_keep_flipped'] or 0) > 0.05 else 'CANNOT rescue it — teacher STARVES the right token'}")
print("\nwrote results/runs/eagle/eagle_delta_diag.json")
