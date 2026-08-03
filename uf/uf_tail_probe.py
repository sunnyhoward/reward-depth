#!/usr/bin/env python
"""The translation-tail measurement (NEXT.md priority 1 / phase 8 §8): does mean-pooling soften
UF's confidently-backwards tail the way it dissolved styc's corr_e wall (.776 -> .991)?

Phase 7 §8 localized the last-token L12 probe's blindness: 13.6% of soft labels side against the
dataset, and the CONFIDENTLY-backwards tail is overwhelmingly translation + rigid-format tasks —
chosen = a correct execution the probe cannot verify from style. Phase 8 §6/§10 showed the styc
analogue (corr_e, wrongness-under-fluency) was substantially a READ-POSITION artifact. This script
asks the same question on real data, offline, from the two caches uf_meanpool_sweep.py writes
(pooled + last-token, same forward passes, row-aligned funnel).

Protocol:
  1. Cross-fit (2-fold) the last-token L_REF probe on the 3,000 probe-fit pairs -> out-of-sample
     soft label p per pair. Tail = p < TAIL_P (confidently backwards). Composition check: fraction
     translation-like (prompt says "translat*" or high non-ASCII fraction) in tail vs overall.
  2. KEY CURVES: per-layer natural-fit probes (all train pairs, IPW), accuracy evaluated on the
     tail slice — last-token vs pooled. (Fit set contains the tail pairs but the probe is fit on
     3,000; a ~200-pair slice cannot be memorized by a regularized linear head — and the wall
     hypothesis predicts ~0 there anyway. Cross-fit p from step 1 is what DEFINES the slice
     out-of-sample.)
  3. Fitting-problem-vs-wall: at LAYER_SUBSET, refit with the tail upweighted xW (W in UPW_LIST)
     and with tail-only 2-fold cross-fits. Wall: tail acc stays ~0 under upweighting and tail-only
     fits sit at chance. Fitting problem: upweighting buys tail acc (price paid on test acc is
     reported alongside).

Env: TAIL_P=0.25 L_REF=12 UPW_LIST=5,20 (same funnel knobs as uf_meanpool_sweep.py)
Reads:  /workspace/uf_probe_feats_meanpool.npz, /workspace/uf_probe_feats_lenmatch.npz
Saves:  /workspace/uf_tail_probe.json"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
TAIL_P, L_REF = float(E("TAIL_P", 0.25)), int(E("L_REF", 12))
UPW_LIST = [float(x) for x in E("UPW_LIST", "5,20").split(",")]
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- funnel: byte-identical to uf_meanpool_sweep.py / uf_probe_rl.py ----
ds = load_dataset(E("UF_DATASET", "allenai/ultrafeedback_binarized_cleaned"),
                  split=E("UF_SPLIT", "train_prefs"), streaming=True)
recs = []
for ex in islice(ds, POOL):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    p = ex.get("prompt") or ch[0]["content"]
    c, r = ch[-1]["content"], rj[-1]["content"]
    if not (p and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < 1.0: continue
    recs.append(dict(prompt=p, chosen=c, rejected=r, is_test=int(_phash(p)[:8], 16) % 10 == 0))
MATCH, BUCKET = int(E("UF_MATCH_LENGTH", 1)), int(E("UF_LEN_BUCKET", 16))
def _rlen(s): return len(tok(s, add_special_tokens=False).input_ids)
if MATCH:
    from collections import defaultdict
    for x in recs: x["len_diff"] = _rlen(x["chosen"]) - _rlen(x["rejected"])
    cnt = defaultdict(lambda: [0, 0])
    for x in recs:
        b = int(round(x["len_diff"] / BUCKET))
        if b > 0: cnt[b][0] += 1
        elif b < 0: cnt[-b][1] += 1
    for x in recs:
        b = int(round(x["len_diff"] / BUCKET))
        if b == 0: x["w"] = 1.0; continue
        npos, nneg = cnt[abs(b)]
        x["w"] = 0.0 if (npos == 0 or nneg == 0) else min(npos, nneg) / (npos if b > 0 else nneg)
    recs = [x for x in recs if x["w"] > 0]
train = [x for x in recs if not x["is_test"]]
test = [x for x in recs if x["is_test"]]
pr, pe = train[:N_PROBE], test[:400]
print(f"[data] {len(recs)} pairs | probe-train {len(pr)} | probe-eval {len(pe)}", flush=True)

# ---- translation/format tagging (phase-7 audit heuristic) ----
def _naf(s): return sum(ord(c) > 127 for c in s) / max(1, len(s))
def is_translation(x):
    return ("translat" in x["prompt"][:400].lower()
            or _naf(x["chosen"]) > 0.15 or _naf(x["rejected"]) > 0.15)
tr_tag = np.array([is_translation(x) for x in pr])
print(f"[tag] translation-like: {tr_tag.mean():.3f} of probe-train", flush=True)

# ---- caches ----
zp = np.load("/workspace/uf_probe_feats_meanpool.npz")
zl = np.load("/workspace/uf_probe_feats_lenmatch.npz")
MP = dict(c_tr=zp["a"], r_tr=zp["b"], c_te=zp["c"], d_te=zp["d"])
LT = dict(c_tr=zl["a"], r_tr=zl["b"], c_te=zl["c"], d_te=zl["d"])
NL = MP["c_tr"].shape[1]
w_pr = np.array([x["w"] for x in pr], np.float32)
w_pe = np.array([x["w"] for x in pe], np.float32)
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(pr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(pe)) < 0.5, 1.0, -1.0).astype(np.float32)

def layer_diffs(F, li):
    """(d_train_unsigned, d_test_unsigned) scaled by the train-pool sd — stage-A protocol."""
    pool = np.concatenate([F["c_tr"][:, li], F["r_tr"][:, li]])
    sd = pool.std(0) + 1e-6
    return ((F["c_tr"][:, li] - F["r_tr"][:, li]) / sd, (F["c_te"][:, li] - F["d_te"][:, li]) / sd)

@torch.no_grad()
def head_p(head, d_unsigned):
    """P(chosen preferred) from unsigned difference features."""
    z, _ = head.z_s2(torch.tensor(d_unsigned, dtype=torch.float32))
    return torch.special.ndtr(z).numpy()

# ---- 1. cross-fit reference probe -> out-of-sample p, tail definition ----
d_ref, _ = layer_diffs(LT, L_REF)
fold = rng.rand(len(pr)) < 0.5
p_oos = np.zeros(len(pr))
for f in (fold, ~fold):
    a, h, _ = train_bayes_head(d_ref[f] * s_tr[f][:, None], s_tr[f],
                               d_ref[~f] * s_tr[~f][:, None], s_tr[~f],
                               w_tr=w_pr[f], w_te=w_pr[~f])
    p_oos[~f] = head_p(h, d_ref[~f])
tail = p_oos < TAIL_P
backwards = p_oos < 0.5
print(f"[ref L{L_REF} lasttok, cross-fit] backwards {backwards.mean():.3f} "
      f"(phase-7 audit: .136) | tail(p<{TAIL_P}) {tail.mean():.3f} n={int(tail.sum())} | "
      f"translation in tail {tr_tag[tail].mean():.3f} vs overall {tr_tag.mean():.3f}", flush=True)

out = dict(tail_p=TAIL_P, l_ref=L_REF, n_tail=int(tail.sum()),
           frac_backwards=float(backwards.mean()),
           frac_translation_tail=float(tr_tag[tail].mean()),
           frac_translation_all=float(tr_tag.mean()),
           p_oos_tail_idx=np.where(tail)[0].tolist())

# ---- 2. per-layer natural-fit tail accuracy, lasttok vs pooled ----
for name, F in (("lasttok", LT), ("pooled", MP)):
    accs_tail, accs_test, accs_tail_trn = [], [], []
    for li in range(NL):
        dtr, dte = layer_diffs(F, li)
        a, h, _ = train_bayes_head(dtr * s_tr[:, None], s_tr, dte * s_te[:, None], s_te,
                                   w_tr=w_pr, w_te=w_pe)
        pt = head_p(h, dtr[tail])
        accs_tail.append(float((pt > 0.5).mean()))
        accs_tail_trn.append(float((head_p(h, dtr[tail & tr_tag]) > 0.5).mean())
                             if (tail & tr_tag).sum() else None)
        accs_test.append(float(a))
        print(f"  [{name}] L{li:2d} test={a:.3f} tail={accs_tail[-1]:.3f}"
              f" tail_transl={accs_tail_trn[-1] if accs_tail_trn[-1] is not None else float('nan'):.3f}", flush=True)
    out[f"{name}_tail_acc"] = accs_tail
    out[f"{name}_tail_translation_acc"] = accs_tail_trn
    out[f"{name}_test_acc"] = accs_test
    json.dump(out, open("/workspace/uf_tail_probe.json", "w"), indent=1)

# ---- 3. wall-vs-fitting-problem at a layer subset ----
LAYER_SUBSET = [int(x) for x in E("LAYER_SUBSET", "0,4,8,12,16,20,23,26,29,31").split(",")]
wall = {}
for name, F in (("lasttok", LT), ("pooled", MP)):
    for li in LAYER_SUBSET:
        dtr, dte = layer_diffs(F, li)
        cell = {}
        for W in UPW_LIST:                       # tail-upweighted refit
            wu = w_pr.copy(); wu[tail] *= W
            _, h, _ = train_bayes_head(dtr * s_tr[:, None], s_tr, dte * s_te[:, None], s_te,
                                       w_tr=wu, w_te=w_pe)
            cell[f"upw{W:g}_tail"] = float((head_p(h, dtr[tail]) > 0.5).mean())
            cell[f"upw{W:g}_test"] = float(
                (w_pe * (head_p(h, dte) > 0.5)).sum() / w_pe.sum())
        ti = np.where(tail)[0]                   # tail-only 2-fold cross-fit: pure decodability
        tf = rng.rand(len(ti)) < 0.5
        accs = []
        for f in (tf, ~tf):
            if f.sum() < 10 or (~f).sum() < 10: continue
            _, h, _ = train_bayes_head(dtr[ti[f]] * s_tr[ti[f]][:, None], s_tr[ti[f]],
                                       dtr[ti[~f]] * s_tr[ti[~f]][:, None], s_tr[ti[~f]])
            accs.append(float((head_p(h, dtr[ti[~f]]) > 0.5).mean()))
        cell["tailonly_xfit"] = float(np.mean(accs)) if accs else None
        wall[f"{name}_L{li}"] = cell
        print(f"  [wall] {name} L{li:2d} {cell}", flush=True)
out["wall"] = wall
json.dump(out, open("/workspace/uf_tail_probe.json", "w"), indent=1)
print("DONE", flush=True)
