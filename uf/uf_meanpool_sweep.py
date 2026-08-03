#!/usr/bin/env python
"""Mean-pooled-over-tokens probe sweep — the read-POSITION counterfactual to the last-token sweep.

Every number in this repo comes from a probe reading ONE position: the <|end_of_text|> sentinel at
the completion end (`uf_probe_rl.py:last_tok_feats`, `buf[:, -1]`). Libon et al. — the closest prior
work — instead mean-pool per-token probe reads over the completion, and phase 4 listed a
"Libon-style mean-pooled read" as the designated fallback for the UF port. It was never run.

This script rebuilds the Stage-A sweep with the ONLY change being the read: mean over the
completion's tokens instead of the last one. Same funnel (byte-identical, so rows align with the
last-token cache), same IPW length matching, same Bayesian head protocol, same seed.

Three things it answers:
  1. Is the preference decodable EARLIER when pooled? A terminal read must compress the whole
     response into one state; a pooled read gets evidence from every token. If L* moves down, the
     "earliest layer where the preference is decodable" — the quantity the whole thesis is about —
     is partly an artifact of the read position.
  2. Is the pooled read MORE or LESS length-confounded? A terminal state plausibly encodes "how
     long was this" more than a per-token average does. Reported per layer as corr(z, len_diff),
     plus the length-only cheat floor for reference (last-token value: 0.619).
  3. Does the mid-band ELBO peak (Bayesian-Occam layer selection) survive the change?

Pooling is over COMPLETION tokens only (Libon's convention). Prompt tokens are excluded: they are
shared by both sides of a pair, so including them would inject a length-weighted copy of the same
prompt state into both reads and dilute the difference by an amount that depends on the two sides'
lengths — a length confound manufactured by the pooling itself.

Env: same funnel knobs as uf_probe_rl.py + MP_BS=4 (extraction batch; small by default because this
     is meant to run CONCURRENTLY with a training arm on the same GPU)
Saves: /workspace/uf_probe_feats_meanpool.npz, /workspace/uf_probe_curve_meanpool.json"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
MAX_LEN = int(E("MAX_LEN", 1024)); BS = int(E("MP_BS", 4))
TOL = float(E("UF_PLATEAU_TOL", 0.01))
DEV, SEED = "cuda", 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- funnel: byte-identical to uf_probe_rl.py / uf_soft_dpo.py ----
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

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); NL = len(BLOCKS); HID = model.config.hidden_size

def _n_comp(p, r):
    """Completion-token count, after the same left-truncation the batch will apply."""
    nf = len(tok(render_full(p, r), add_special_tokens=False).input_ids)
    np_ = len(tok(render_prompt(p), add_special_tokens=False).input_ids)
    return max(1, min(nf - np_, min(nf, MAX_LEN)))

@torch.no_grad()
def meanpool_feats(recs_, side, bs=BS):
    """(N, n_layers, hid) x2: mean of each block's output over the COMPLETION tokens, AND the
    last-token read (identical to uf_probe_rl.last_tok_feats — left padding puts the real eos
    sentinel at column -1). Both come from the same forward pass, so the last-token cache is free
    here and a wiped box only pays for one extraction sweep.

    Left padding + left truncation put the completion in the rightmost n_comp columns, so the mask
    is simply the tail — which stays correct when a long prompt is truncated away."""
    texts = [render_full(x["prompt"], x[side]) for x in recs_]
    ncs = [_n_comp(x["prompt"], x[side]) for x in recs_]
    out = np.zeros((len(texts), NL, HID), np.float32)
    out_lt = np.zeros((len(texts), NL, HID), np.float32)
    for s in range(0, len(texts), bs):
        chunk, nc = texts[s:s + bs], ncs[s:s + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        T = enc.input_ids.shape[1]
        m = torch.zeros(len(chunk), T, device=DEV, dtype=torch.float32)
        for i, n in enumerate(nc): m[i, T - min(n, T):] = 1.0
        denom = m.sum(1, keepdim=True).clamp(min=1.0)
        with ResidualCapture(BLOCKS) as cap:
            model(**enc)
        buf = cap.get()
        for li in range(NL):
            h = buf[li].float()                                  # (B, T, H)
            out[s:s + len(chunk), li] = ((h * m[:, :, None]).sum(1) / denom).cpu().numpy()
            out_lt[s:s + len(chunk), li] = h[:, -1].cpu().numpy()
        del buf
        if (s // bs) % 50 == 0: print(f"  [{side}] {s}/{len(texts)}", flush=True)
    return out, out_lt

cachef = "/workspace/uf_probe_feats_meanpool.npz"
cachef_lt = "/workspace/uf_probe_feats_lenmatch.npz"   # uf_probe_rl.py Stage-A format
if os.path.exists(cachef):
    z = np.load(cachef); Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
else:
    print("[feats] mean-pooled + last-token caching...", flush=True)
    Fc_tr, Lc_tr = meanpool_feats(pr, "chosen");   Fr_tr, Lr_tr = meanpool_feats(pr, "rejected")
    Fc_te, Lc_te = meanpool_feats(pe, "chosen");   Fr_te, Lr_te = meanpool_feats(pe, "rejected")
    np.savez(cachef, a=Fc_tr, b=Fr_tr, c=Fc_te, d=Fr_te)
    if not os.path.exists(cachef_lt):
        np.savez(cachef_lt, a=Lc_tr, b=Lr_tr, c=Lc_te, d=Lr_te)
        print(f"[feats] last-token cache also written -> {cachef_lt}", flush=True)

# ---- per-layer probes: identical protocol to uf_probe_rl.py Stage A ----
w_pr = np.array([x["w"] for x in pr], np.float32); w_pe = np.array([x["w"] for x in pe], np.float32)
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(pr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(pe)) < 0.5, 1.0, -1.0).astype(np.float32)
ld_te = np.array([x["len_diff"] for x in pe], np.float32) if MATCH else np.zeros(len(pe), np.float32)

acc = np.zeros(NL); elbo = np.zeros(NL); corr_len = np.zeros(NL)
for li in range(NL):
    pool = np.concatenate([Fc_tr[:, li], Fr_tr[:, li]])
    sd = pool.std(0) + 1e-6
    dtr = ((Fc_tr[:, li] - Fr_tr[:, li]) / sd) * s_tr[:, None]
    dte = ((Fc_te[:, li] - Fr_te[:, li]) / sd) * s_te[:, None]
    a, h, e = train_bayes_head(dtr, s_tr, dte, s_te, w_tr=w_pr, w_te=w_pe)
    acc[li], elbo[li] = a, e
    with torch.no_grad():   # unsigned margin vs length difference: the confound diagnostic
        z_ = h.z_s2(torch.tensor((Fc_te[:, li] - Fr_te[:, li]) / sd, dtype=torch.float32))[0].numpy()
    corr_len[li] = float(np.corrcoef(z_, ld_te)[0, 1]) if ld_te.std() > 0 else 0.0
    print(f"  L{li:2d} acc={a:.3f} elbo={e:+.0f} corr(z,len_diff)={corr_len[li]:+.3f}", flush=True)

LSTAR = int(next(li for li in range(NL) if acc[li] >= acc.max() - TOL))
print(f"[probe] MEAN-POOLED plateau layer L*={LSTAR} (acc {acc[LSTAR]:.3f}, max {acc.max():.3f} "
      f"@ L{int(acc.argmax())})", flush=True)
json.dump(dict(layer_acc=acc.tolist(), elbo=elbo.tolist(), corr_z_len=corr_len.tolist(),
               Lstar=LSTAR, len_matched=bool(MATCH), read="meanpool_completion"),
          open("/workspace/uf_probe_curve_meanpool.json", "w"))

# ---- side-by-side with the last-token sweep, if present ----
lt = "/workspace/uf_probe_curve_lenmatch.json"
if os.path.exists(lt):
    ltc = json.load(open(lt))["layer_acc"]
    print("\n  layer | last-tok | meanpool |  delta", flush=True)
    for li in range(NL):
        print(f"   L{li:2d}  |  {ltc[li]:.3f}   |  {acc[li]:.3f}   | {acc[li]-ltc[li]:+.3f}", flush=True)
    print(f"\n  L*: last-tok {json.load(open(lt))['Lstar']} -> meanpool {LSTAR}", flush=True)
print("DONE", flush=True)
