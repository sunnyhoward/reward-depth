#!/usr/bin/env python
"""UF two-head hybrid, the faithful port of the A/B winner (results_phase4.md): plan-reader margin
below L_PLAN + on-policy RLOO from the outcome-judge above it. No likelihood anchor (the A/B hybrid
needed none; the margin half plays that role) -- KL-in-reward stays (part of the RLOO recipe).

Heads:
  PLAN-READER  (prompt-end state at L_PLAN, from uf_plan_sweep.py): margin backprop <= L_PLAN pushes
      the pre-generation state toward "a good response is planned". Labels are RELATIVE (per-refit
      median split of the judge's scores on the policy's own samples): automatically balanced (no
      label collapse), a ratchet toward ever-better plans, and forged states that don't yield
      better completions get judged bad at the next refit and absorbed (buffer refit from scratch).
  OUTCOME-JUDGE (phase-3 length-matched L12 pairwise probe, frozen-base read at the re-rendered eos
      sentinel): RLOO reward for sampled rollouts, gradient masked to blocks > L_PLAN.

The A/B ablations motivating this: margin-only oscillates, RLOO/J-only finds the degenerate
attractor, both together install completely. At UF the degenerate attractor to watch is LENGTH
(logged per refit) rather than the letter.

Env: L_PLAN=-1 (from uf_plan_curve.json) RL_STEPS=300 RL_BATCH=4 RL_K=8 MAX_NEW=512 RL_KL=0.03
     RL_PESS=0.5 RL_LR=5e-5 MCOEF=1.0 ANCH=0.1 REFIT_EVERY=10 N_EVAL=64 + funnel knobs
Saves: /workspace/uf_hybrid2_history.json, checkpoints /workspace/uf_hybrid2_ckpt{100,200},
       /workspace/uf_hybrid2_lora"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head, ResidualCapture, BayesLinearHead, LOG_NDTR

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
STEPS, BATCH, K = int(E("RL_STEPS", 300)), int(E("RL_BATCH", 4)), int(E("RL_K", 8))
KL, PESS, LR = float(E("RL_KL", 0.03)), float(E("RL_PESS", 0.5)), float(E("RL_LR", 5e-5))
MCOEF, ANCH = float(E("MCOEF", 1.0)), float(E("ANCH", 0.1))
MAX_NEW, MAX_LEN, PLEN = int(E("MAX_NEW", 512)), int(E("MAX_LEN", 1024)), int(E("PROMPT_LEN", 512))
REFIT_EVERY, N_EVAL = int(E("REFIT_EVERY", 10)), int(E("N_EVAL", 64))
L_PLAN = int(E("L_PLAN", -1))
DEV, SEED = "cuda", 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- funnel identical to uf_probe_rl.py / uf_plan_sweep.py ----
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
print(f"[data] {len(recs)} pairs | train {len(train)} | test {len(test)}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); NL = len(BLOCKS); HID = model.config.hidden_size

# ---- outcome-judge (frozen; phase-3 L12 len-matched probe) ----
z = np.load("/workspace/uf_probe_feats_lenmatch.npz"); Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
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
MU_J = jhead.mu.detach().float().to(DEV); SIG2_J = F.softplus(jhead.rho.detach()).float().pow(2).to(DEV)
SD_J = torch.tensor(sdJ, device=DEV); MN_J = torch.tensor(mnJ, device=DEV)
print(f"[judge] L{L_J} held-out acc {jacc:.3f}", flush=True)
def judge_reward(f):
    fs = (f.float() - MN_J) / SD_J
    s2 = fs.pow(2).matmul(SIG2_J)
    return torch.special.ndtr((fs.matmul(MU_J) - PESS * torch.sqrt(s2 + 1e-9)) / torch.sqrt(1 + s2))
def judge_zraw(f):
    return ((f.float() - MN_J) / SD_J).matmul(MU_J)

# ---- plan-reader (from uf_plan_sweep.py artifacts; refit deterministically) ----
plan_curve = json.load(open("/workspace/uf_plan_curve.json"))
LP = L_PLAN if L_PLAN >= 0 else int(plan_curve["best_layer"])
S = json.load(open("/workspace/uf_plan_samples.json"))
zz = np.load("/workspace/uf_plan_feats.npz"); Ptr, Pte = zz["tr"], zz["te"]
ztr = np.array([np.mean(x["z"]) for x in S["train"]]); zte = np.array([np.mean(x["z"]) for x in S["test"]])
med = np.median(ztr)
pt_tr = np.where(ztr > med, 1.0, -1.0).astype(np.float32)
pt_te = np.where(zte > med, 1.0, -1.0).astype(np.float32)
MN_P = torch.tensor(Ptr[:, LP].mean(0), device=DEV)
SD_P = torch.tensor(Ptr[:, LP].std(0) + 1e-6, device=DEV)
pacc, phead, _ = train_bayes_head((Ptr[:, LP] - MN_P.cpu().numpy()) / SD_P.cpu().numpy(), pt_tr,
                                  (Pte[:, LP] - MN_P.cpu().numpy()) / SD_P.cpu().numpy(), pt_te)
MU_P = phead.mu.detach().float().to(DEV); RHO_P = phead.rho.detach().float().to(DEV)
print(f"[plan] L{LP} plan-reader held-out acc {pacc:.3f} (LoRA split: margin <= {LP}, RLOO > {LP})", flush=True)
def plan_z(f):  # f: centered/scaled by caller
    s2 = f.pow(2).matmul(F.softplus(RHO_P).pow(2))
    return f.matmul(MU_P) / torch.sqrt(1 + s2)

# ---- policy ----
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
import re as _re
def _blk(n):
    m = _re.search(r"\.layers\.(\d+)\.", n); return int(m.group(1)) if m else -1
lora_named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
params = [p for _, p in lora_named]
low_params = [p for n, p in lora_named if 0 <= _blk(n) <= LP]
opt = torch.optim.AdamW(params, lr=LR)
print(f"[lora] {sum(p.numel() for p in params)/1e6:.1f}M trainable | "
      f"{sum(p.numel() for p in low_params)/1e6:.1f}M in blocks <= {LP}", flush=True)

def plan_feats_pol(prompts_txt, grad, base=False):
    """Prompt-end state at L_PLAN through the policy (or frozen base), centered+scaled."""
    enc = tok(prompts_txt, return_tensors="pt", padding=True, truncation=True, max_length=PLEN).to(DEV)
    import contextlib
    cm = policy.disable_adapter if base else contextlib.nullcontext
    with (torch.enable_grad() if grad else torch.no_grad()), cm():
        with ResidualCapture([BLOCKS[LP]]) as cap:
            policy(**enc, logits_to_keep=1)
        return (cap.get()[0][:, -1].float() - MN_P) / SD_P

@torch.no_grad()
def rollout_feats(prompts_raw, gen, P, bs=8, k=None):
    """Judge-layer residual at the re-rendered eos sentinel, frozen base (phase-3 conventions)."""
    k = K if k is None else k
    texts = []
    for i, praw in enumerate(prompts_raw):
        for j in range(k):
            comp = tok.decode(gen[i * k + j, P:], skip_special_tokens=True)
            texts.append(render_full(praw, comp))
    out = torch.zeros(len(texts), HID, device=DEV)
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        with policy.disable_adapter(), ResidualCapture([BLOCKS[L_J]]) as cap:
            policy(**enc, logits_to_keep=1)
        out[s:s + enc.input_ids.shape[0]] = cap.get()[0][:, -1]
    return out

def comp_logprob(text_full, plen, grad):
    ids = tok(text_full, return_tensors="pt", truncation=True, max_length=MAX_LEN + MAX_NEW).input_ids.to(DEV)
    plen = min(plen, ids.shape[1] - 1)
    with (torch.enable_grad() if grad else torch.no_grad()):
        keep = ids.shape[1] - plen + 1
        logits = policy(ids, logits_to_keep=keep).logits[0, :-1].float()
        return F.log_softmax(logits, -1).gather(-1, ids[0, plen:, None]).squeeze(-1).sum()

@torch.no_grad()
def evaluate(n=N_EVAL):
    policy.eval(); ir = []
    for x in test[:n]:
        pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.shape[1]
        lc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
        lr_ = comp_logprob(render_full(x["prompt"], x["rejected"]), pl, False)
        with policy.disable_adapter():
            rc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
            rr = comp_logprob(render_full(x["prompt"], x["rejected"]), pl, False)
        ir.append((float((lc - rc) > (lr_ - rr)), float(lc - rc), float(lr_ - rr)))
    policy.train()
    a = np.array(ir)
    return dict(acc_implicit=float(a[:, 0].mean()), dlp_chosen=float(a[:, 1].mean()),
                dlp_rejected=float(a[:, 2].mean()))

# ---- training ----
hist = dict(L_plan=LP, L_judge=L_J, plan_acc=float(pacc), reward=[], zjudge=[], len=[], evals=[], refit=[])
# Seed the refit buffer with the plan sweep's already-judged (base state, relative label) pairs --
# they are exactly the "past evidence" the buffer design wants, and the fresh fits get real mass
# from the first refit instead of an effectively-frozen head (the v1-launch flaw: 4 labels/refit
# meant no head update before step ~160, inviting the 2.2 uniform-push forge).
_seed = np.random.RandomState(SEED).choice(len(Ptr), min(512, len(Ptr)), replace=False)
buf_f = [torch.tensor((Ptr[_seed, LP] - MN_P.cpu().numpy()) / SD_P.cpu().numpy(), dtype=torch.float32)]
buf_t = [torch.tensor(pt_tr[_seed])]
REFIT_R, REFIT_K = int(E("REFIT_R", 24)), int(E("REFIT_K", 2))
rgen = random.Random(4242); policy.train()
for step in range(STEPS):
    batch = rgen.sample(train, BATCH)
    prompts_txt = [render_prompt(x["prompt"]) for x in batch]
    enc = tok(prompts_txt, return_tensors="pt", padding=True, truncation=True, max_length=PLEN).to(DEV)
    policy.config.use_cache = True
    with torch.no_grad():
        gen = policy.generate(**enc, do_sample=True, temperature=1.0, num_return_sequences=K,
                              max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    attn = (gen != tok.pad_token_id).long()
    n_new = (attn[:, P:]).sum(1).clamp(min=1)
    fj = rollout_feats([x["prompt"] for x in batch], gen, P)
    r = judge_reward(fj).detach()
    zj = judge_zraw(fj).detach()
    keepg = gen.shape[1] - P + 1
    tokmask = attn[:, P:].bool()
    with torch.no_grad():
        lsm = F.log_softmax(policy(input_ids=gen, attention_mask=attn, logits_to_keep=keepg).logits[:, :-1].float(), -1)
        logp_ng = (lsm.gather(-1, gen[:, P:, None]).squeeze(-1) * tokmask).sum(1)
        with policy.disable_adapter():
            ref_lsm = F.log_softmax(policy(input_ids=gen, attention_mask=attn, logits_to_keep=keepg).logits[:, :-1].float(), -1)
            ref_logp = (ref_lsm.gather(-1, gen[:, P:, None]).squeeze(-1) * tokmask).sum(1)
        del lsm, ref_lsm
    r = r - KL * (logp_ng - ref_logp) / n_new
    rg = r.view(BATCH, K)
    adv = (rg - (rg.sum(1, keepdim=True) - rg) / (K - 1)).view(-1)
    # ---- margin half (<= LP): push plan states toward the reader's "good plan" side ----
    opt.zero_grad()
    f_base = plan_feats_pol(prompts_txt, grad=False, base=True)
    f_pol = plan_feats_pol(prompts_txt, grad=True)
    zp = plan_z(f_pol)
    m_loss = (-MCOEF * LOG_NDTR(zp).mean() + ANCH * (f_pol - f_base).pow(2).mean())
    m_loss.backward()
    g_low = [(p_, p_.grad.clone() if p_.grad is not None else None) for p_ in low_params]
    # ---- RLOO half (> LP): masked restore of <=LP grads afterwards ----
    for s0 in range(0, BATCH * K, 4):
        sl = slice(s0, min(s0 + 4, BATCH * K))
        if not adv[sl].abs().sum() > 0: continue
        li = F.log_softmax(policy(input_ids=gen[sl], attention_mask=attn[sl],
                                  logits_to_keep=keepg).logits[:, :-1].float(), -1)
        lp_i = (li.gather(-1, gen[sl, P:, None]).squeeze(-1) * tokmask[sl]).sum(1)
        (-(adv[sl] * lp_i / n_new[sl]).sum() / (BATCH * K)).backward()
    for p_, g in g_low:
        p_.grad = g
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["reward"].append(float(rg.mean())); hist["zjudge"].append(float(zj.mean()))
    hist["len"].append(float(n_new.float().mean()))
    hist.setdefault("mloss", []).append(float(m_loss.detach()))
    # ---- plan-reader buffer refit: fresh head on ALL (plan state, relative judge label) pairs.
    # Dedicated sampling (REFIT_R prompts x REFIT_K rollouts) rather than reusing the tiny training
    # batch: the head must track the CURRENT policy's plan states at a real cadence. ----
    if (step + 1) % REFIT_EVERY == 0:
        rqs = rgen.sample(train, REFIT_R)
        rtxt = [render_prompt(x["prompt"]) for x in rqs]
        enc_r = tok(rtxt, return_tensors="pt", padding=True, truncation=True, max_length=PLEN).to(DEV)
        policy.config.use_cache = True
        with torch.no_grad():
            gen_r = policy.generate(**enc_r, do_sample=True, temperature=1.0, num_return_sequences=REFIT_K,
                                    max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
        policy.config.use_cache = False
        fj_r = rollout_feats([x["prompt"] for x in rqs], gen_r, enc_r.input_ids.shape[1], k=REFIT_K)
        zb = judge_zraw(fj_r).view(REFIT_R, REFIT_K).mean(1)
        fb = plan_feats_pol(rtxt, grad=False)
        tb = torch.where(zb > zb.median(), 1.0, -1.0)     # RELATIVE labels: balanced by construction
        buf_f.append(fb.cpu()); buf_t.append(tb.cpu())
        Fb = torch.cat(buf_f).numpy(); Tb = torch.cat(buf_t).numpy()
        if len(Tb) >= 64:
            perm = np.random.RandomState(step).permutation(len(Tb)); ntr = int(0.85 * len(Tb))
            bacc, hnew, _ = train_bayes_head(Fb[perm[:ntr]], Tb[perm[:ntr]],
                                             Fb[perm[ntr:]], Tb[perm[ntr:]], epochs=60, patience=8)
            rot = float(torch.rad2deg(torch.arccos(torch.clamp(
                F.cosine_similarity(hnew.mu.detach().to(DEV), MU_P, dim=0), -1, 1))))
            MU_P = hnew.mu.detach().float().to(DEV); RHO_P = hnew.rho.detach().float().to(DEV)
            hist["refit"].append(dict(step=step + 1, size=len(Tb), val_acc=float(bacc), rot_deg=rot))
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: reward {np.mean(hist['reward'][-10:]):.3f} "
              f"zjudge {np.mean(hist['zjudge'][-10:]):+.2f} len {np.mean(hist['len'][-10:]):.0f} "
              f"mloss {np.mean(hist['mloss'][-10:]):.3f}", flush=True)
    if (step + 1) % 50 == 0:
        ev = evaluate(); ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL {ev}", flush=True)
        json.dump(hist, open("/workspace/uf_hybrid2_history.json", "w"), indent=1)
    if (step + 1) % 100 == 0 and (step + 1) < STEPS:
        policy.save_pretrained(f"/workspace/uf_hybrid2_ckpt{step+1}")
json.dump(hist, open("/workspace/uf_hybrid2_history.json", "w"), indent=1)
policy.save_pretrained("/workspace/uf_hybrid2_lora"); tok.save_pretrained("/workspace/uf_hybrid2_lora")
print("DONE", flush=True)
