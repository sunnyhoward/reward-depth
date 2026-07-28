import os, sys, json, hashlib, numpy as np, torch
from itertools import islice
from transformers import AutoTokenizer
from datasets import load_dataset
sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head
tok = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B-SFT")
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()
ds = load_dataset("allenai/ultrafeedback_binarized_cleaned", split="train_prefs", streaming=True)
recs = []
for ex in islice(ds, 20000):
    ch, rj = ex.get("chosen"), ex.get("rejected")
    if not ch or not rj: continue
    p = ex.get("prompt") or ch[0]["content"]; c, r = ch[-1]["content"], rj[-1]["content"]
    if not (p and c and r) or c == r: continue
    sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
    if sc is None or sr is None or float(sc) - float(sr) < 1.0: continue
    recs.append(dict(prompt=p, chosen=c, rejected=r, is_test=int(_phash(p)[:8], 16) % 10 == 0))
from collections import defaultdict
def _rlen(s): return len(tok(s, add_special_tokens=False).input_ids)
for x in recs: x["len_diff"] = _rlen(x["chosen"]) - _rlen(x["rejected"])
cnt = defaultdict(lambda: [0, 0])
for x in recs:
    b = int(round(x["len_diff"] / 16))
    if b > 0: cnt[b][0] += 1
    elif b < 0: cnt[-b][1] += 1
for x in recs:
    b = int(round(x["len_diff"] / 16))
    if b == 0: x["w"] = 1.0; continue
    npos, nneg = cnt[abs(b)]
    x["w"] = 0.0 if (npos == 0 or nneg == 0) else min(npos, nneg) / (npos if b > 0 else nneg)
recs = [x for x in recs if x["w"] > 0]
train = [x for x in recs if not x["is_test"]]; test = [x for x in recs if x["is_test"]]
pr, pe = train[:3000], test[:400]
def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cx, cy = x - mx, y - my
    return float(np.average(cx*cy, weights=w) / (np.sqrt(np.average(cx*cx, weights=w))*np.sqrt(np.average(cy*cy, weights=w))))
z = np.load("/workspace/uf_probe_feats_lenmatch.npz")
Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
w_pr = np.array([x["w"] for x in pr], np.float32); w_pe = np.array([x["w"] for x in pe], np.float32)
rng = np.random.RandomState(0)
s_tr = np.where(rng.rand(len(pr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(pe)) < 0.5, 1.0, -1.0).astype(np.float32)
ld_tr = np.array([x["len_diff"] for x in pr], float); ld_te = np.array([x["len_diff"] for x in pe], float)
print(f"{'L':>3} {'tr_raw':>8} {'tr_ipw':>8} {'te_raw':>8} {'te_ipw':>8}")
res = {}
for li in [12, 16, 31]:
    pool = np.concatenate([Fc_tr[:, li], Fr_tr[:, li]]); sd = pool.std(0) + 1e-6
    dtr = ((Fc_tr[:, li] - Fr_tr[:, li]) / sd) * s_tr[:, None]
    dte = ((Fc_te[:, li] - Fr_te[:, li]) / sd) * s_te[:, None]
    a, h, e = train_bayes_head(dtr, s_tr, dte, s_te, w_tr=w_pr, w_te=w_pe)
    with torch.no_grad():
        ztr = h.z_s2(torch.tensor((Fc_tr[:, li]-Fr_tr[:, li])/sd, dtype=torch.float32))[0].numpy()
        zte = h.z_s2(torch.tensor((Fc_te[:, li]-Fr_te[:, li])/sd, dtype=torch.float32))[0].numpy()
    r = (float(np.corrcoef(ztr, ld_tr)[0,1]), wcorr(ztr, ld_tr, w_pr),
         float(np.corrcoef(zte, ld_te)[0,1]), wcorr(zte, ld_te, w_pe))
    res[li] = r
    print(f"{li:>3} {r[0]:>+8.3f} {r[1]:>+8.3f} {r[2]:>+8.3f} {r[3]:>+8.3f}")
json.dump(res, open("/workspace/uf_lencorr_trainvstest.json","w"))
print("\nphase-5 doc (train, raw): L12 +0.006 | L16 +0.103 | L31 +0.066")
print("phase-5 doc (train, ipw): L12 -0.044 | L16 +0.047 | L31 +0.015")
