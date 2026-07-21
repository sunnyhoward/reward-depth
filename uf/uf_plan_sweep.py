#!/usr/bin/env python
"""Plan-state decodability sweep (the GATE for the UF hybrid port; results_phase4.md open items).

Question: is the QUALITY of the response the model is about to generate linearly readable from the
prompt-end state (the residual after the assistant prefix, before any token is emitted)? At A/B the
analogous decision-position probe read the impending answer at 0.99 and the two-head hybrid needed
it; at UF nobody has measured whether a "plan state" exists.

Design: sample K completions per prompt from the frozen SFT model; score each with the OUTCOME-JUDGE
(the phase-3 length-matched L12 probe, centered absolute read at the re-rendered eos sentinel);
label each prompt with the MEAN judge z over its K samples (averaging reduces the irreducible
sampling noise -- the state fixes a distribution over qualities, not a quality); median-split into
good/bad (balanced by construction); sweep per-layer probes on prompt-end features against that
label. The plateau (if any) picks the plan-reader's attach layer for the hybrid.

Outputs: /workspace/uf_plan_curve.json (the verdict), /workspace/uf_plan_samples.json (sampled
completions + judge scores -- reusable as the plan-reader's first refit buffer),
/workspace/uf_plan_feats.npz (prompt-end features).

Env: N_PLAN=2000 N_PLAN_TE=400 K_SAMP=2 MAX_NEW=512 TEMP=1.0 + the uf_probe_rl.py funnel knobs"""
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
N_PLAN, N_PLAN_TE, K = int(E("N_PLAN", 2000)), int(E("N_PLAN_TE", 400)), int(E("K_SAMP", 2))
MAX_NEW, MAX_LEN, TEMP = int(E("MAX_NEW", 512)), int(E("MAX_LEN", 1024)), float(E("TEMP", 1.0))
PLEN = int(E("PROMPT_LEN", 512))
DEV, SEED = "cuda", 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- funnel identical to uf_probe_rl.py (incl. length-match filter, for judge-cache alignment) ----
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
plan_tr = train[:N_PLAN]                      # overlaps pr -- fine, the judge is frozen
plan_te = test[:N_PLAN_TE]
print(f"[data] plan-train {len(plan_tr)} | plan-test {len(plan_te)}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); NL = len(BLOCKS); HID = model.config.hidden_size

# ---- outcome-judge: deterministic refit of the phase-3 L12 probe from the len-matched cache ----
z = np.load("/workspace/uf_probe_feats_lenmatch.npz"); Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
LSTAR_J = json.load(open("/workspace/uf_probe_curve_lenmatch.json"))["Lstar"]
w_pr = np.array([x["w"] for x in pr], np.float32); w_pe = np.array([x["w"] for x in pe], np.float32)
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(pr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(pe)) < 0.5, 1.0, -1.0).astype(np.float32)
pool = np.concatenate([Fc_tr[:, LSTAR_J], Fr_tr[:, LSTAR_J]])
sdJ, mnJ = pool.std(0) + 1e-6, pool.mean(0)
jacc, jhead, _ = train_bayes_head(((Fc_tr[:, LSTAR_J] - Fr_tr[:, LSTAR_J]) / sdJ) * s_tr[:, None], s_tr,
                                  ((Fc_te[:, LSTAR_J] - Fr_te[:, LSTAR_J]) / sdJ) * s_te[:, None], s_te,
                                  w_tr=w_pr, w_te=w_pe)
MU_J = jhead.mu.detach().float().to(DEV)
SD_J = torch.tensor(sdJ, device=DEV); MN_J = torch.tensor(mnJ, device=DEV)
print(f"[judge] L{LSTAR_J} probe refit, held-out acc {jacc:.3f}", flush=True)

@torch.no_grad()
def judge_z(prompts, comps, bs=8):
    """Centered judge margin z at the re-rendered eos sentinel (phase-3 read conventions)."""
    texts = [render_full(p, c) for p, c in zip(prompts, comps)]
    out = np.zeros(len(texts), np.float32)
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        with ResidualCapture([BLOCKS[LSTAR_J]]) as cap:
            model(**enc, logits_to_keep=1)
        fs = (cap.get()[0][:, -1].float() - MN_J) / SD_J
        out[s:s + enc.input_ids.shape[0]] = fs.matmul(MU_J).cpu().numpy()
    return out

# ---- sample K completions per prompt, judge them ----
sampf = "/workspace/uf_plan_samples.json"
if os.path.exists(sampf):
    S = json.load(open(sampf))
else:
    S = {"train": [], "test": []}
    torch.manual_seed(SEED)
    for split, xs in (("train", plan_tr), ("test", plan_te)):
        for s in range(0, len(xs), 16):
            chunk = xs[s:s + 16]
            enc = tok([render_prompt(x["prompt"]) for x in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=PLEN).to(DEV)
            model.config.use_cache = True
            gen = model.generate(**enc, do_sample=True, temperature=TEMP, num_return_sequences=K,
                                 max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
            model.config.use_cache = False
            P = enc.input_ids.shape[1]
            comps = [tok.decode(gen[i, P:], skip_special_tokens=True) for i in range(gen.shape[0])]
            prompts = [x["prompt"] for x in chunk for _ in range(K)]
            zs = judge_z(prompts, comps)
            for i, x in enumerate(chunk):
                S[split].append(dict(prompt=x["prompt"],
                                     z=[float(zs[i * K + j]) for j in range(K)],
                                     comps=[comps[i * K + j][:400] for j in range(K)],
                                     lens=[len(tok(comps[i * K + j], add_special_tokens=False).input_ids)
                                           for j in range(K)]))
            if (s // 16) % 10 == 0:
                done = len(S[split])
                print(f"  [{split}] {done}/{len(xs)} sampled+judged", flush=True)
        json.dump(S, open(sampf, "w"))
print(f"[samples] train {len(S['train'])} | test {len(S['test'])}", flush=True)

# ---- prompt-end features at every layer (frozen base) ----
featf = "/workspace/uf_plan_feats.npz"
def plan_feats(xs, bs=8):
    X = np.zeros((len(xs), NL, HID), np.float32)
    with torch.no_grad():
        for s in range(0, len(xs), bs):
            enc = tok([render_prompt(x["prompt"]) for x in xs[s:s + bs]], return_tensors="pt",
                      padding=True, truncation=True, max_length=PLEN).to(DEV)
            with ResidualCapture(BLOCKS) as cap:
                model(**enc, logits_to_keep=1)
            buf = cap.get()
            for li in range(NL):
                X[s:s + enc.input_ids.shape[0], li] = buf[li][:, -1].float().cpu().numpy()
    return X
if os.path.exists(featf):
    zz = np.load(featf); Xtr, Xte = zz["tr"], zz["te"]
else:
    print("[feats] caching prompt-end features...", flush=True)
    Xtr, Xte = plan_feats(S["train"]), plan_feats(S["test"])
    np.savez(featf, tr=Xtr, te=Xte)

# ---- labels: mean judge z per prompt, median split (balanced) ----
ztr = np.array([np.mean(x["z"]) for x in S["train"]]); zte = np.array([np.mean(x["z"]) for x in S["test"]])
med = np.median(ztr)
t_tr = np.where(ztr > med, 1.0, -1.0).astype(np.float32)
t_te = np.where(zte > med, 1.0, -1.0).astype(np.float32)
# length-vs-label diagnostic: is "planned quality" secretly "planned length"?
ltr = np.array([np.mean(x["lens"]) for x in S["train"]])
print(f"[labels] frac good (test, train-median) {float((t_te > 0).mean()):.3f} | "
      f"corr(mean z, mean len) {np.corrcoef(ztr, ltr)[0, 1]:+.3f}", flush=True)

acc = np.zeros(NL)
for li in range(NL):
    mn, sd = Xtr[:, li].mean(0), Xtr[:, li].std(0) + 1e-6
    a, h, e = train_bayes_head((Xtr[:, li] - mn) / sd, t_tr, (Xte[:, li] - mn) / sd, t_te)
    acc[li] = a
    print(f"  L{li:2d} acc={a:.3f} elbo={e:+.0f}", flush=True)
best = int(acc.argmax())
print(f"[plan] best layer L{best} acc {acc[best]:.3f} (chance 0.5; judge sample-noise bounds the "
      f"ceiling below 1.0)", flush=True)
json.dump(dict(layer_acc=acc.tolist(), best_layer=best, judge_layer=LSTAR_J, K=K,
               corr_z_len=float(np.corrcoef(ztr, ltr)[0, 1])), open("/workspace/uf_plan_curve.json", "w"))
print("DONE", flush=True)
