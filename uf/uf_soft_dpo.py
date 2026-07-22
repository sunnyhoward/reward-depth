#!/usr/bin/env python
"""Soft-label DPO from the frozen probe (methods.md par.3, first proposed item) -- the zero-variance
offline closed form of RL-from-probe. The probe DEFINES the preference (soft label per pair), the
loss READS behaviour (implicit-reward margin), so the coupling is unfakeable; unlike RLOO there is
no sampling variance and gradients are dense.

Discriminator for the flat-reward RLOO runs (v3/v4): if this installs the probe's preference
(held-out implicit acc -> ~probe acc 0.79), the probe signal is fine and the RL harness (batch/LR/
variance) was the bottleneck. If this also stalls, likelihood training cannot couple the probe's
preference into behaviour -- a finding about the method, not the harness.

Per-pair soft label: p = Phi(z / sqrt(1+s2)) from the L* Bayesian head's posterior predictive on the
pair's DIFFERENCE features (the head's native fit distribution -- no absolute-read calibration).
Loss: -[p*logsig(beta*D) + (1-p)*logsig(-beta*D)], D = (lp_c - ref_c) - (lp_r - ref_r).

Config mirrors uf_dpo_train.py exactly (BETA=0.1 LR=5e-5 STEPS=400 BATCH=4 ACCUM=4 LoRA r16) so the
result is directly comparable to the hard-label DPO baseline (0.805). Trains on the N_PROBE pairs
whose difference features are cached (probe-train pairs -- soft labels are in-sample for the probe;
eval pairs are held out from both probe and policy training).

Caveat: the probe was fit with length-matching IPW; soft-DPO training is unweighted, mirroring the
DPO baseline. The label itself is length-debiased (the probe's), the pair sampling is not.

Env: same funnel knobs as uf_probe_rl.py + DPO_BETA DPO_LR DPO_STEPS DPO_BATCH DPO_ACCUM
Saves: /workspace/uf_softdpo_lora, /workspace/uf_softdpo_history.json"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
BETA, LR = float(E("DPO_BETA", 0.1)), float(E("DPO_LR", 5e-5))
STEPS, BATCH, ACCUM = int(E("DPO_STEPS", 400)), int(E("DPO_BATCH", 4)), int(E("DPO_ACCUM", 4))
MAX_LEN = int(E("MAX_LEN", 1024)); N_EVAL = int(E("UF_N_EVAL", 128))
L_OVERRIDE = int(E("L_OVERRIDE", -1))   # -1: use the sweep's Lstar; else fit/label at this layer
TAG = E("RUN_TAG", ""); SFX = f"_{TAG}" if TAG else ""   # suffix for all output paths
DEV, SEED = "cuda", 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- data funnel + length-match FILTER: must be byte-identical to uf_probe_rl.py so that the
# cached difference features align row-for-row with `pr`/`pe` below ----
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
print(f"[data] {len(recs)} pairs | soft-DPO train {len(pr)} | eval pool {len(test)}", flush=True)

# ---- probe: refit at L* from the cached difference features (deterministic, same as uf_probe_rl) ----
cachef = f"/workspace/uf_probe_feats{'_lenmatch' if MATCH else ''}.npz"
z = np.load(cachef); Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
assert Fc_tr.shape[0] == len(pr) and Fc_te.shape[0] == len(pe), "cache/funnel misalignment"
LSTAR = json.load(open(f"/workspace/uf_probe_curve{'_lenmatch' if MATCH else ''}.json"))["Lstar"]
if L_OVERRIDE >= 0: LSTAR = L_OVERRIDE   # depth-differential arms: same recipe, different label layer
w_pr = np.array([x["w"] for x in pr], np.float32); w_pe = np.array([x["w"] for x in pe], np.float32)
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(pr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(pe)) < 0.5, 1.0, -1.0).astype(np.float32)
sd = np.concatenate([Fc_tr[:, LSTAR], Fr_tr[:, LSTAR]]).std(0) + 1e-6
dtr = ((Fc_tr[:, LSTAR] - Fr_tr[:, LSTAR]) / sd) * s_tr[:, None]
dte = ((Fc_te[:, LSTAR] - Fr_te[:, LSTAR]) / sd) * s_te[:, None]
pacc, head, _ = train_bayes_head(dtr, s_tr, dte, s_te, w_tr=w_pr, w_te=w_pe)
print(f"[probe] L*={LSTAR} held-out acc {pacc:.3f}", flush=True)

with torch.no_grad():  # soft labels: posterior predictive on UNSIGNED differences (chosen - rejected)
    df_tr = torch.tensor((Fc_tr[:, LSTAR] - Fr_tr[:, LSTAR]) / sd, dtype=torch.float32)
    z_tr, _ = head.z_s2(df_tr)
    P_SOFT = torch.special.ndtr(z_tr).numpy()          # p(chosen > rejected) per training pair
    df_te = torch.tensor((Fc_te[:, LSTAR] - Fr_te[:, LSTAR]) / sd, dtype=torch.float32)
    Z_TE = head.z_s2(df_te)[0].numpy()                 # held-out probe margins (for agreement eval)
for x, p_ in zip(pr, P_SOFT): x["p"] = float(p_)
print(f"[labels] p(chosen>rej): mean {P_SOFT.mean():.3f} | frac>0.5 {float((P_SOFT>0.5).mean()):.3f} "
      f"| frac in (0.2,0.8) {float(((P_SOFT>0.2)&(P_SOFT<0.8)).mean()):.3f}", flush=True)

# ---- model + LoRA (identical to uf_dpo_train.py) ----
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)

def comp_logprob(rec, side, grad):
    full = tok(render_full(rec["prompt"], rec[side]), return_tensors="pt",
               truncation=True, max_length=MAX_LEN).input_ids.to(DEV)
    plen = min(tok(render_prompt(rec["prompt"]), return_tensors="pt",
                   truncation=True, max_length=MAX_LEN).input_ids.shape[1], full.shape[1] - 1)
    with (torch.enable_grad() if grad else torch.no_grad()):
        keep = full.shape[1] - plen + 1
        logits = policy(full, logits_to_keep=keep).logits[0, :-1].float()
        return F.log_softmax(logits, -1).gather(-1, full[0, plen:, None]).squeeze(-1).sum()

def pair_terms(rec, grad):
    lc = comp_logprob(rec, "chosen", grad); lr_ = comp_logprob(rec, "rejected", grad)
    with torch.no_grad(), policy.disable_adapter():
        rc = comp_logprob(rec, "chosen", False); rr = comp_logprob(rec, "rejected", False)
    return lc, lr_, rc, rr

@torch.no_grad()
def evaluate(n=N_EVAL):
    """acc_implicit: dataset-preference install (comparable to DPO baseline). acc_probe: fraction
    of held-out pairs where the policy's margin agrees with the PROBE's margin -- the actual
    training target. Both on pe (also the probe's early-stop set; disjoint from policy training)."""
    policy.eval(); out = []
    for i, x in enumerate(pe[:n]):
        lc, lr_, rc, rr = pair_terms(x, False)
        m = float(lc - rc) - float(lr_ - rr)
        out.append((float(m > 0), float((m > 0) == (Z_TE[i] > 0)), float(lc - rc), float(lr_ - rr)))
    policy.train()
    a = np.array(out)
    return dict(acc_implicit=float(a[:, 0].mean()), acc_probe=float(a[:, 1].mean()),
                dlp_chosen=float(a[:, 2].mean()), dlp_rejected=float(a[:, 3].mean()))

rgen = random.Random(4242)
hist = dict(Lstar=LSTAR, probe_acc=float(pacc), loss=[], evals=[])
ev = evaluate(); ev["step"] = 0; hist["evals"].append(ev)
print(f"  step    0: EVAL {ev}", flush=True)
policy.train()
for step in range(STEPS):
    opt.zero_grad(); tot = 0.0
    for _ in range(ACCUM):
        for rec in rgen.sample(pr, BATCH):
            lc, lr_, rc, rr = pair_terms(rec, True)
            D = BETA * ((lc - rc) - (lr_ - rr))
            p_ = rec["p"]
            loss = -(p_ * F.logsigmoid(D) + (1 - p_) * F.logsigmoid(-D)) / (BATCH * ACCUM)
            loss.backward(); tot += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(tot)
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {tot:.4f}", flush=True)
    if (step + 1) % 50 == 0:
        ev = evaluate(); ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL {ev}", flush=True)
        json.dump(hist, open(f"/workspace/uf_softdpo{SFX}_history.json", "w"), indent=1)
    if (step + 1) % 200 == 0:
        policy.save_pretrained(f"/workspace/uf_softdpo{SFX}_ckpt{step+1}")
json.dump(hist, open(f"/workspace/uf_softdpo{SFX}_history.json", "w"), indent=1)
policy.save_pretrained(f"/workspace/uf_softdpo{SFX}_lora"); tok.save_pretrained(f"/workspace/uf_softdpo{SFX}_lora")
print("DONE", flush=True)
