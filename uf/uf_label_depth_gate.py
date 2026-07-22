#!/usr/bin/env python
"""Probe-content gate for the depth differential (pre-registration diagnostic, zero GPU-training).

Soft-DPO-from-probe is a CONTENT experiment: the probe layer only decides which representation
the labels get filtered through; delivery (likelihood-level DPO) is identical across arms. So the
depth differential is only alive if labels distilled through the EARLY probe (L*=12) and the TOP
probe (L31) actually differ. This script measures that from the Stage-A cache before any training:

  - per-layer heads at L in {Lstar, 16 (acc max), 31 (top)}: acc / ELBO / soft-label profile
  - pairwise label divergence: corr(z), hard agreement, |dp| distribution
  - WHERE they disagree: dataset score-margin buckets (prediction: noisy pairs), and which probe
    matches the dataset label on disagreements
  - length alignment corr(z, len_diff) per layer (the classic UF idiosyncrasy)

Out: /workspace/uf_depth_gate.json + printed summary.
Env: same funnel knobs as uf_probe_rl.py (UF_POOL N_PROBE UF_MATCH_LENGTH UF_LEN_BUCKET)."""
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
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- funnel byte-identical to uf_probe_rl.py, but KEEP the annotator scores ----
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
    recs.append(dict(prompt=p, chosen=c, rejected=r, sc=float(sc), sr=float(sr),
                     is_test=int(_phash(p)[:8], 16) % 10 == 0))
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
print(f"[data] {len(recs)} pairs | probe-train {len(pr)} | probe-test {len(pe)}", flush=True)

z = np.load(f"/workspace/uf_probe_feats{'_lenmatch' if MATCH else ''}.npz")
Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
assert Fc_tr.shape[0] == len(pr) and Fc_te.shape[0] == len(pe), "cache/funnel misalignment"
NL = Fc_tr.shape[1]
LSTAR = json.load(open(f"/workspace/uf_probe_curve{'_lenmatch' if MATCH else ''}.json"))["Lstar"]
LAYERS = sorted({LSTAR, 16, NL - 1})
w_pr = np.array([x["w"] for x in pr], np.float32); w_pe = np.array([x["w"] for x in pe], np.float32)
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(pr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(pe)) < 0.5, 1.0, -1.0).astype(np.float32)

def wcorr(x, y, w):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return float(cov / np.sqrt(np.average((x - mx) ** 2, weights=w) *
                               np.average((y - my) ** 2, weights=w) + 1e-12))

R = {"Lstar": LSTAR, "layers": {}}
Ztr, Zte, Ptr_soft = {}, {}, {}
for L in LAYERS:
    sd = np.concatenate([Fc_tr[:, L], Fr_tr[:, L]]).std(0) + 1e-6
    dtr_s = ((Fc_tr[:, L] - Fr_tr[:, L]) / sd) * s_tr[:, None]
    dte_s = ((Fc_te[:, L] - Fr_te[:, L]) / sd) * s_te[:, None]
    acc, head, elbo = train_bayes_head(dtr_s, s_tr, dte_s, s_te, w_tr=w_pr, w_te=w_pe)
    with torch.no_grad():
        ztr = head.z_s2(torch.tensor((Fc_tr[:, L] - Fr_tr[:, L]) / sd, dtype=torch.float32))[0].numpy()
        zte = head.z_s2(torch.tensor((Fc_te[:, L] - Fr_te[:, L]) / sd, dtype=torch.float32))[0].numpy()
    ptr = torch.special.ndtr(torch.tensor(ztr)).numpy()
    Ztr[L], Zte[L], Ptr_soft[L] = ztr, zte, ptr
    ld = np.array([x["len_diff"] for x in pr], float)
    R["layers"][L] = dict(
        acc_w=float(acc), elbo=float(elbo),
        acc_unw_te=float((zte > 0).mean()),
        mean_p=float(ptr.mean()),
        frac_soft=float(((ptr > 0.2) & (ptr < 0.8)).mean()),
        frac_conf=float(((ptr < 0.05) | (ptr > 0.95)).mean()),
        frac_override=float((ptr < 0.5).mean()),
        corr_z_lendiff=float(np.corrcoef(ztr, ld)[0, 1]),
        wcorr_z_lendiff=wcorr(ztr, ld, w_pr))
    print(f"[L{L:2d}] acc_w {acc:.3f} | elbo {elbo:+.0f} | mean_p {ptr.mean():.3f} | "
          f"soft(.2-.8) {R['layers'][L]['frac_soft']:.3f} | conf(<.05|>.95) {R['layers'][L]['frac_conf']:.3f} | "
          f"override {R['layers'][L]['frac_override']:.3f} | corr(z,len) {R['layers'][L]['corr_z_lendiff']:+.3f} "
          f"(w {R['layers'][L]['wcorr_z_lendiff']:+.3f})", flush=True)

# ---- pairwise divergence ----
from itertools import combinations
sm = np.array([x["sc"] - x["sr"] for x in pr], float)
buckets = [(1.0, 1.5), (1.5, 2.5), (2.5, 99.0)]
R["pairs"] = {}
for La, Lb in combinations(LAYERS, 2):
    pa, pb = Ptr_soft[La], Ptr_soft[Lb]
    hard_dis = (pa > 0.5) != (pb > 0.5)
    dp = np.abs(pa - pb)
    per_bucket = {}
    for lo, hi in buckets:
        m = (sm >= lo) & (sm < hi)
        per_bucket[f"{lo}-{hi}"] = dict(n=int(m.sum()), disagree=float(hard_dis[m].mean()),
                                        mean_absdp=float(dp[m].mean()))
    # on held-out disagreements, who matches the dataset label?
    dis_te = (Zte[La] > 0) != (Zte[Lb] > 0)
    R["pairs"][f"L{La}-L{Lb}"] = dict(
        corr_z_tr=float(np.corrcoef(Ztr[La], Ztr[Lb])[0, 1]),
        corr_z_te=float(np.corrcoef(Zte[La], Zte[Lb])[0, 1]),
        hard_agree_tr=float(1 - hard_dis.mean()),
        mean_absdp=float(dp.mean()), frac_absdp_gt02=float((dp > 0.2).mean()),
        disagree_by_scoremargin=per_bucket,
        n_dis_te=int(dis_te.sum()),
        acc_on_dis_te={f"L{La}": float((Zte[La] > 0)[dis_te].mean()),
                       f"L{Lb}": float((Zte[Lb] > 0)[dis_te].mean())})
    pp = R["pairs"][f"L{La}-L{Lb}"]
    print(f"[L{La} vs L{Lb}] corr_z tr/te {pp['corr_z_tr']:.3f}/{pp['corr_z_te']:.3f} | "
          f"hard agree {pp['hard_agree_tr']:.3f} | mean|dp| {pp['mean_absdp']:.3f} | "
          f"|dp|>.2 {pp['frac_absdp_gt02']:.3f}", flush=True)
    print(f"          disagree by score-margin: " +
          " ".join(f"{k}:{v['disagree']:.3f}(n={v['n']})" for k, v in per_bucket.items()), flush=True)
    print(f"          held-out disagreements n={pp['n_dis_te']}: dataset-match " +
          " ".join(f"{k}={v:.3f}" for k, v in pp["acc_on_dis_te"].items()), flush=True)

json.dump(R, open("/workspace/uf_depth_gate.json", "w"), indent=1)
print("DONE", flush=True)
