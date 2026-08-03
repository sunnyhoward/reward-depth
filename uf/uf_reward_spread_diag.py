#!/usr/bin/env python
"""Off-policy reward-calibration diagnostic (phase-9, follows shaped300's weak finish).

Question: the pooled probe separates DATASET pairs at .82 — but the reward differences that
drive RLOO are WITHIN-PROMPT differences between POLICY rollouts, i.e. text the probe was never
fit on. Is that within-prompt spread signal or probe noise?

Measures, at L* on the pooled read, in both reward units (the ndtr LCB actually used) and raw z:
  1. dataset pairs (held-out): chosen-vs-rejected gap and spread — the "advertised" signal.
  2. base-policy rollouts (K per prompt): within-prompt spread vs across-prompt spread, and the
     probe's own posterior uncertainty s on rollout text vs dataset text (off-distribution flag).
  3. squash compression: ratio of within-prompt spread in ndtr units vs z units, i.e. how much
     of the z signal the CDF+pessimism pipeline discards where the rollouts actually live.

Env: N_PROMPTS=32 K=4 MAX_NEW=200 L_OVERRIDE= (else pooled plateau) + funnel knobs as elsewhere
Saves: /workspace/uf_reward_spread_diag.json"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
MAX_LEN, MAX_NEW, PLEN = int(E("MAX_LEN", 1024)), int(E("MAX_NEW", 200)), int(E("PROMPT_LEN", 512))
N_PROMPTS, K, PESS = int(E("N_PROMPTS", 32)), int(E("K", 4)), float(E("RL_PESS", 0.5))
TOL = float(E("PLATEAU_TOL", 0.01))
DEV, SEED = "cuda", 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()
def _n_comp(p, r):
    nf = len(tok(render_full(p, r), add_special_tokens=False).input_ids)
    np_ = len(tok(render_prompt(p), add_special_tokens=False).input_ids)
    return max(1, min(nf - np_, min(nf, MAX_LEN)))

# ---- funnel (byte-identical) ----
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

# ---- probe head at L* from the pooled cache (single-layer refit, stage-A protocol) ----
z = np.load("/workspace/uf_probe_feats_meanpool.npz")
Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
NL = Fc_tr.shape[1]
curve = json.load(open("/workspace/uf_probe_curve_meanpool.json"))
LSTAR = int(E("L_OVERRIDE", curve["Lstar"]))
w_pr = np.array([x["w"] for x in pr], np.float32); w_pe = np.array([x["w"] for x in pe], np.float32)
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(pr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(pe)) < 0.5, 1.0, -1.0).astype(np.float32)
pool = np.concatenate([Fc_tr[:, LSTAR], Fr_tr[:, LSTAR]])
sd, mn = pool.std(0) + 1e-6, pool.mean(0)
dtr = ((Fc_tr[:, LSTAR] - Fr_tr[:, LSTAR]) / sd) * s_tr[:, None]
dte = ((Fc_te[:, LSTAR] - Fr_te[:, LSTAR]) / sd) * s_te[:, None]
a, head, _ = train_bayes_head(dtr, s_tr, dte, s_te, w_tr=w_pr, w_te=w_pe)
print(f"[probe] L*={LSTAR} pooled acc {a:.3f}", flush=True)
MU = head.mu.detach().float().to(DEV); SIG2 = F.softplus(head.rho.detach()).float().pow(2).to(DEV)
SD = torch.tensor(sd, device=DEV); MN = torch.tensor(mn, device=DEV)

def probe_all(f):
    """-> (reward = ndtr LCB as used in RL, raw z, raw LCB z, s = sqrt(s2))."""
    fs = (f.float() - MN) / SD
    s2 = fs.pow(2).matmul(SIG2)
    zz = fs.matmul(MU)
    lcb = zz - PESS * torch.sqrt(s2 + 1e-9)
    return torch.special.ndtr(lcb / torch.sqrt(1 + s2)), zz, lcb, torch.sqrt(s2)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers)

@torch.no_grad()
def pooled_read(texts, ncs, bs=8):
    out = torch.zeros(len(texts), model.config.hidden_size, device=DEV)
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        T = enc.input_ids.shape[1]
        with ResidualCapture([BLOCKS[LSTAR]]) as cap:
            model(**enc, logits_to_keep=1)
        res = cap.get()[0].float()
        for i in range(enc.input_ids.shape[0]):
            n = min(ncs[s + i], T)
            out[s + i] = res[i, T - n:].mean(0)
    return out

# ---- 1. dataset pairs (fresh forward, not cache, so protocol == RL-time read) ----
pairs = pe[:100]
tc = [render_full(x["prompt"], x["chosen"]) for x in pairs]
trj = [render_full(x["prompt"], x["rejected"]) for x in pairs]
nc_c = [_n_comp(x["prompt"], x["chosen"]) for x in pairs]
nc_r = [_n_comp(x["prompt"], x["rejected"]) for x in pairs]
rc, zc, lc, sc_ = probe_all(pooled_read(tc, nc_c))
rr, zr, lr_, sr_ = probe_all(pooled_read(trj, nc_r))
print(f"[pairs n={len(pairs)}] reward: chosen {rc.mean():.3f} rejected {rr.mean():.3f} gap {rc.mean()-rr.mean():.3f} | "
      f"z gap {zc.mean()-zr.mean():.2f} | pair |z-diff| {(zc-zr).abs().mean():.2f} | s {sc_.mean():.2f}", flush=True)

# ---- 2. base-policy rollouts ----
prompts = [x["prompt"] for x in test[:N_PROMPTS]]
texts, ncs, pidx = [], [], []
for pi in range(0, N_PROMPTS, 4):
    chunk = prompts[pi:pi + 4]
    enc = tok([render_prompt(p) for p in chunk], return_tensors="pt", padding=True,
              truncation=True, max_length=PLEN).to(DEV)
    g = model.generate(**enc, do_sample=True, temperature=1.0, num_return_sequences=K,
                       max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    P = enc.input_ids.shape[1]
    for i, p in enumerate(chunk):
        for j in range(K):
            comp = tok.decode(g[i * K + j, P:], skip_special_tokens=True)
            texts.append(render_full(p, comp)); ncs.append(_n_comp(p, comp))
            pidx.append(pi + i)
    print(f"  [gen] {pi + len(chunk)}/{N_PROMPTS}", flush=True)
ro, zo, lo, so = probe_all(pooled_read(texts, ncs))
pidx = np.array(pidx)
def within(v):
    v = v.cpu().numpy()
    return float(np.mean([v[pidx == p].std() for p in np.unique(pidx)]))
res = dict(
    Lstar=LSTAR, probe_acc=float(a), n_pairs=len(pairs), n_prompts=N_PROMPTS, K=K,
    pair_reward_gap=float(rc.mean() - rr.mean()),
    pair_absdiff_reward=float((rc - rr).abs().mean()),
    pair_absdiff_z=float((zc - zr).abs().mean()),
    pair_s_mean=float(torch.cat([sc_, sr_]).mean()),
    dataset_z_std=float(torch.cat([zc, zr]).std()),
    roll_reward_mean=float(ro.mean()), roll_reward_std=float(ro.std()),
    roll_within_prompt_reward_std=within(ro),
    roll_within_prompt_z_std=within(zo),
    roll_within_prompt_lcb_std=within(lo),
    roll_across_prompt_z_std=float(np.std([zo.cpu().numpy()[pidx == p].mean() for p in np.unique(pidx)])),
    roll_z_mean=float(zo.mean()), roll_z_std=float(zo.std()),
    roll_s_mean=float(so.mean()),
    # squash compression where the rollouts live: d(reward)/d(z) local vs the z spread it eats
    squash_ratio=float(within(ro) / (within(zo) + 1e-9)),
)
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open("/workspace/uf_reward_spread_diag.json", "w"), indent=1)
print("DONE", flush=True)
