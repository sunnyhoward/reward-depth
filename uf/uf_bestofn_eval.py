#!/usr/bin/env python
"""Best-of-n probe-reranker baseline (RLFR import): how much judge-quality does pure inference-time
selection buy, with zero weight changes? Samples N_SAMP completions per held-out prompt from the
BASE policy, scores each with the frozen L12 judge (re-rendered eos read), and reports the mean
selected judge-z for nested best-of-{1,2,4,...,N_SAMP} from the same sample pool.

Calibrates every training arm: a trained policy's zjudge_ho is only impressive relative to what
best-of-n gets for free (and best-of-n costs n forward passes per query at inference, the trained
policy costs one — the comparison is install-vs-search, not a horse race).

Uses the SAME held-out prompt window as uf_hybrid3's on-policy eval (test[200:200+HO_N]) so numbers
are directly comparable to the hybrid histories' zjudge_ho.

Env: HO_N=24 N_SAMP=16 MAX_NEW=512 + funnel knobs. Out: /workspace/uf_bestofn.json"""
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
HO_N, N_SAMP = int(E("HO_N", 24)), int(E("N_SAMP", 16))
MAX_NEW, MAX_LEN, PLEN = int(E("MAX_NEW", 512)), int(E("MAX_LEN", 1024)), int(E("PROMPT_LEN", 512))
DEV, SEED = "cuda", 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- funnel identical to uf_probe_rl.py ----
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

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); HID = model.config.hidden_size

# ---- judge refit (identical to uf_hybrid3) ----
z = np.load("/workspace/uf_probe_feats_lenmatch.npz"); Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
assert Fc_tr.shape[0] == len(pr) and Fc_te.shape[0] == len(pe), "cache/funnel misalignment"
L_J = json.load(open("/workspace/uf_probe_curve_lenmatch.json"))["Lstar"]
w_pr = np.array([x["w"] for x in pr], np.float32); w_pe = np.array([x["w"] for x in pe], np.float32)
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(pr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(pe)) < 0.5, 1.0, -1.0).astype(np.float32)
poolJ = np.concatenate([Fc_tr[:, L_J], Fr_tr[:, L_J]])
sdJ, mnJ = poolJ.std(0) + 1e-6, poolJ.mean(0)
jacc, jhead, _ = train_bayes_head(((Fc_tr[:, L_J] - Fr_tr[:, L_J]) / sdJ) * s_tr[:, None], s_tr,
                                  ((Fc_te[:, L_J] - Fr_te[:, L_J]) / sdJ) * s_te[:, None], s_te,
                                  w_tr=w_pr, w_te=w_pe)
MU_J = jhead.mu.detach().float().to(DEV)
SD_J = torch.tensor(sdJ, device=DEV); MN_J = torch.tensor(mnJ, device=DEV)
print(f"[judge] L{L_J} held-out acc {jacc:.3f}", flush=True)

@torch.no_grad()
def judge_z_texts(prompts, comps, bs=8):
    texts = [render_full(p, c) for p, c in zip(prompts, comps)]
    out = np.zeros(len(texts), np.float32)
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        with ResidualCapture([BLOCKS[L_J]]) as cap:
            model(**enc, logits_to_keep=1)
        fs = (cap.get()[0][:, -1].float() - MN_J) / SD_J
        out[s:s + enc.input_ids.shape[0]] = fs.matmul(MU_J).cpu().numpy()
    return out

# ---- sample N_SAMP per held-out prompt (same window as uf_hybrid3 evals), judge, nested best-of ----
ho = test[200:200 + HO_N]
Z = np.zeros((HO_N, N_SAMP), np.float32); LENS = np.zeros((HO_N, N_SAMP), np.float32)
for s in range(0, HO_N, 4):
    chunk = ho[s:s + 4]
    enc = tok([render_prompt(x["prompt"]) for x in chunk], return_tensors="pt",
              padding=True, truncation=True, max_length=PLEN).to(DEV)
    model.config.use_cache = True
    with torch.no_grad():
        gen = model.generate(**enc, do_sample=True, temperature=1.0, num_return_sequences=N_SAMP,
                             max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    model.config.use_cache = False
    P = enc.input_ids.shape[1]
    comps = [tok.decode(gen[i, P:], skip_special_tokens=True) for i in range(gen.shape[0])]
    prompts = [x["prompt"] for x in chunk for _ in range(N_SAMP)]
    zs = judge_z_texts(prompts, comps)
    for i in range(len(chunk)):
        Z[s + i] = zs[i * N_SAMP:(i + 1) * N_SAMP]
        LENS[s + i] = [len(tok(comps[i * N_SAMP + j], add_special_tokens=False).input_ids)
                       for j in range(N_SAMP)]
    print(f"  {min(s + 4, HO_N)}/{HO_N} prompts sampled+judged", flush=True)

res = dict(judge_acc=float(jacc), HO_N=HO_N, N_SAMP=N_SAMP, bon={})
ns = [n for n in (1, 2, 4, 8, 16, 32) if n <= N_SAMP]
rs = np.random.RandomState(SEED)
for n in ns:
    zsel, lsel = [], []
    for rep in range(64):                     # subsample n of N_SAMP without replacement, take max
        for i in range(HO_N):
            sel = rs.choice(N_SAMP, n, replace=False)
            j = sel[np.argmax(Z[i, sel])]
            zsel.append(Z[i, j]); lsel.append(LENS[i, j])
    res["bon"][n] = dict(z=float(np.mean(zsel)), len=float(np.mean(lsel)))
    print(f"  best-of-{n:2d}: z {np.mean(zsel):+.3f} | len {np.mean(lsel):.0f}", flush=True)
json.dump(res, open("/workspace/uf_bestofn.json", "w"), indent=1)
print("DONE", flush=True)
