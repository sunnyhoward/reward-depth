#!/usr/bin/env python
"""RL-from-probe on UltraFeedback (Tulu-3-8B-SFT). Stage A: per-layer Bayesian probe sweep on
chosen/rejected last-token residuals -> pick L* where accuracy plateaus. Stage B: RLOO with the
probe (frozen-base read at L*, pessimism LCB) as reward, KL-in-reward, DPOP anchor on the pair's
chosen side. Saves probe curve, history, checkpoints at 100/200/300 steps.

Env: UF_POOL=20000 N_PROBE=3000 N_EVAL=96 RL_STEPS=300 RL_BATCH=2 RL_K=4 RL_KL=0.03 RL_PESS=0.5
     RL_ANCHOR=1.0 RL_LR=5e-5 MAX_NEW=200 MAX_LEN=1024 PLATEAU_TOL=0.01"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head, ResidualCapture, BayesLinearHead

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE, N_EVAL = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000)), int(E("N_EVAL", 96))
STEPS, BATCH, K = int(E("RL_STEPS", 300)), int(E("RL_BATCH", 2)), int(E("RL_K", 4))
KL, PESS, ANCHOR, LR = float(E("RL_KL", 0.03)), float(E("RL_PESS", 0.5)), float(E("RL_ANCHOR", 1.0)), float(E("RL_LR", 5e-5))
MAX_NEW, MAX_LEN, TOL = int(E("MAX_NEW", 200)), int(E("MAX_LEN", 1024)), float(E("PLATEAU_TOL", 0.01))
PLEN = int(E("PROMPT_LEN", 512))
DEV = "cuda"; SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- data (same funnel as uf_dpo_train.py) ----
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
train = [x for x in recs if not x["is_test"]]
test = [x for x in recs if x["is_test"]]
print(f"[data] {len(recs)} pairs | train {len(train)} | test {len(test)}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); NL = len(BLOCKS); HID = model.config.hidden_size

@torch.no_grad()
def last_tok_feats(texts, bs=8):
    """(N, n_layers, hid) last-token residuals at every block output."""
    out = np.zeros((len(texts), NL, HID), np.float32)
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        with ResidualCapture(BLOCKS) as cap:
            model(**enc)
        buf = cap.get()
        for li in range(NL):
            out[s:s + len(enc.input_ids), li] = buf[li][:, -1].float().cpu().numpy()
    return out

# ---- Stage A: probe sweep ----
cachef = "/workspace/uf_probe_feats.npz"
pr = train[:N_PROBE]; pe = test[:400]
if os.path.exists(cachef):
    z = np.load(cachef); Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
else:
    print("[feats] caching...", flush=True)
    Fc_tr = last_tok_feats([render_full(x["prompt"], x["chosen"]) for x in pr])
    Fr_tr = last_tok_feats([render_full(x["prompt"], x["rejected"]) for x in pr])
    Fc_te = last_tok_feats([render_full(x["prompt"], x["chosen"]) for x in pe])
    Fr_te = last_tok_feats([render_full(x["prompt"], x["rejected"]) for x in pe])
    np.savez(cachef, a=Fc_tr, b=Fr_tr, c=Fc_te, d=Fr_te)
rng = np.random.RandomState(SEED)
s_tr = np.where(rng.rand(len(pr)) < 0.5, 1.0, -1.0).astype(np.float32)
s_te = np.where(rng.rand(len(pe)) < 0.5, 1.0, -1.0).astype(np.float32)
acc = np.zeros(NL); heads = {}
for li in range(NL):
    sd = np.concatenate([Fc_tr[:, li], Fr_tr[:, li]]).std(0) + 1e-6
    dtr = ((Fc_tr[:, li] - Fr_tr[:, li]) / sd) * s_tr[:, None]
    dte = ((Fc_te[:, li] - Fr_te[:, li]) / sd) * s_te[:, None]
    a, h, e = train_bayes_head(dtr, s_tr, dte, s_te)
    acc[li], heads[li] = a, (h, sd)
    print(f"  L{li:2d} acc={a:.3f} elbo={e:+.0f}", flush=True)
LSTAR = int(next(li for li in range(NL) if acc[li] >= acc.max() - TOL))
print(f"[probe] plateau layer L*={LSTAR} (acc {acc[LSTAR]:.3f}, max {acc.max():.3f})", flush=True)
json.dump(dict(layer_acc=acc.tolist(), Lstar=LSTAR), open("/workspace/uf_probe_curve.json", "w"))

head, sd_ = heads[LSTAR]
MU = head.mu.detach().float().to(DEV); SIG2 = F.softplus(head.rho.detach()).float().pow(2).to(DEV)
SD = torch.tensor(sd_, device=DEV)
def probe_reward(f):
    fs = f.float() / SD
    s2 = fs.pow(2).matmul(SIG2)
    return torch.special.ndtr((fs.matmul(MU) - PESS * torch.sqrt(s2 + 1e-9)) / torch.sqrt(1 + s2))

# ---- Stage B: RLOO from probe at L* ----
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)

def comp_logprob(text_full, plen, grad):
    ids = tok(text_full, return_tensors="pt", truncation=True, max_length=MAX_LEN + MAX_NEW).input_ids.to(DEV)
    plen = min(plen, ids.shape[1] - 1)
    with (torch.enable_grad() if grad else torch.no_grad()):
        keep = ids.shape[1] - plen + 1
        logits = policy(ids, logits_to_keep=keep).logits[0, :-1].float()
        return F.log_softmax(logits, -1).gather(-1, ids[0, plen:, None]).squeeze(-1).sum()

@torch.no_grad()
def evaluate(n=64):
    policy.eval(); ir = []
    for x in test[:n]:
        pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.shape[1]
        lc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
        lr_ = comp_logprob(render_full(x["prompt"], x["rejected"]), pl, False)
        with policy.disable_adapter():
            rc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
            rr = comp_logprob(render_full(x["prompt"], x["rejected"]), pl, False)
        ir.append(dict(acc=float((lc - rc) > (lr_ - rr)), dc=float(lc - rc), dr=float(lr_ - rr)))
    policy.train()
    return dict(acc_implicit=float(np.mean([x["acc"] for x in ir])),
                dlp_chosen=float(np.mean([x["dc"] for x in ir])),
                dlp_rejected=float(np.mean([x["dr"] for x in ir])))

hist = dict(Lstar=LSTAR, reward=[], evals=[], len=[])
rgen = random.Random(4242); policy.train()
for step in range(STEPS):
    batch = rgen.sample(train, BATCH)
    prompts = [render_prompt(x["prompt"]) for x in batch]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=PLEN).to(DEV)
    policy.config.use_cache = True
    with torch.no_grad():
        gen = policy.generate(**enc, do_sample=True, temperature=1.0, num_return_sequences=K,
                              max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    attn = (gen != tok.pad_token_id).long()
    with torch.no_grad(), policy.disable_adapter(), ResidualCapture([BLOCKS[LSTAR]]) as cap:
        policy(input_ids=gen, attention_mask=attn)
    r = probe_reward(cap.get()[0][:, -1]).detach()
    n_new = (attn[:, P:]).sum(1).clamp(min=1)
    keepg = gen.shape[1] - P + 1
    tokmask = attn[:, P:].bool()
    with torch.no_grad():  # batched, graph-free: values for KL and advantages
        lsm = F.log_softmax(policy(input_ids=gen, attention_mask=attn, logits_to_keep=keepg).logits[:, :-1].float(), -1)
        logp_ng = (lsm.gather(-1, gen[:, P:, None]).squeeze(-1) * tokmask).sum(1)
        with policy.disable_adapter():
            ref_lsm = F.log_softmax(policy(input_ids=gen, attention_mask=attn, logits_to_keep=keepg).logits[:, :-1].float(), -1)
            ref_logp = (ref_lsm.gather(-1, gen[:, P:, None]).squeeze(-1) * tokmask).sum(1)
        del lsm, ref_lsm
    r = r - KL * (logp_ng - ref_logp) / n_new
    valid = (n_new < MAX_NEW)                      # exclude capped samples: their reward is noise
    rg, vg = r.view(BATCH, K), valid.view(BATCH, K).float()
    cnt = vg.sum(1, keepdim=True)
    loo = (rg * vg).sum(1, keepdim=True) - rg * vg
    base = loo / (cnt - vg).clamp(min=1)
    adv = torch.where((vg > 0) & (cnt > 1.5), rg - base, torch.zeros_like(rg)).view(-1)
    hist.setdefault("trunc", []).append(float(1 - valid.float().mean()))
    opt.zero_grad()
    for s0 in range(0, BATCH * K, 4):           # micro-batched backward, chunks of 4
        sl = slice(s0, min(s0 + 4, BATCH * K))
        if not adv[sl].abs().sum() > 0: continue
        li = F.log_softmax(policy(input_ids=gen[sl], attention_mask=attn[sl],
                                  logits_to_keep=keepg).logits[:, :-1].float(), -1)
        lp_i = (li.gather(-1, gen[sl, P:, None]).squeeze(-1) * tokmask[sl]).sum(1)
        (-(adv[sl] * lp_i / n_new[sl]).sum() / (BATCH * K)).backward()
    if ANCHOR > 0:  # DPOP hinge on the pair's chosen side
        for x in batch:
            pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.shape[1]
            lc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, True)
            with torch.no_grad(), policy.disable_adapter():
                rc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
            (ANCHOR * F.relu(rc - lc) / BATCH).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["reward"].append(float(rg.mean())); hist["len"].append(float(n_new.float().mean()))
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: reward {np.mean(hist['reward'][-10:]):.3f} "
              f"len {np.mean(hist['len'][-10:]):.0f}", flush=True)
    if (step + 1) % 50 == 0:
        ev = evaluate(); ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL {ev}", flush=True)
        json.dump(hist, open("/workspace/uf_probe_rl_v2_history.json", "w"), indent=1)
    if (step + 1) % 100 == 0:
        policy.save_pretrained(f"/workspace/uf_probe_rl_v2_ckpt{step+1}")
json.dump(hist, open("/workspace/uf_probe_rl_v2_history.json", "w"), indent=1)
policy.save_pretrained("/workspace/uf_probe_rl_v2_lora"); tok.save_pretrained("/workspace/uf_probe_rl_v2_lora")
print("DONE", flush=True)
