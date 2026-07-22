#!/usr/bin/env python
"""UF two-head hybrid v2 — redesign after the v1 postmortem (head collapse + symmetric likelihood
displacement). Margin backprop from the plan-reader into blocks <= L_PLAN stays; everything that
failed is replaced, with the fixes imported from the v1 autopsy + Libon et al. + RLFR:

UPPER_MODE selects the >L_PLAN objective:
  exactj (fork A, default): pair-restricted exact expectation over the DATASET pair — the faithful
      port of the toy winner's load-bearing mechanism. pi_rel = softmax(TAU * per-token logprob of
      chosen/rejected), J = sum(pi_rel * r) with r = frozen judge's pessimistic reward from the
      CACHED Stage-A features (no sampling in the step at all), + on-menu mass anchor
      MASS * relu(logsumexp_ref - logsumexp) so displacement is structurally impossible.
      Per-token normalization: raw logprob sums differ by 10s of nats on UF, which saturates the
      softmax at init (zero gradient) and correlates with length (a length bias) — TAU * lp/n keeps
      pi_rel in the responsive band. NOTE the per-pair optimum is still the hard max: watch
      pi_rel(chosen) for inflation like dpoR was watched on the toy.
  rloo  (fork B): v1's on-policy RLOO from the frozen judge (re-rendered eos read), + the phase-2
      displacement cure: DPOP hinge on the dataset chosen side (DPOP coef) and stronger KL-in-reward.

Plan-reader (both modes) — stationary, damped, windowed (v1 failed all three ways):
  labels:  judge mean-z per prompt vs the FIXED plan-sweep base median THR (Libon: refit labels must
           be grounded in something stationary; v1's per-refit median split manufactured conflict),
           confidence-weighted by |z - THR| (coin-flips near the threshold don't enter the fit)
  buffer:  seed (plan-sweep states, weight SEED_W) + last WIN refit batches only (v1's ever-growing
           buffer mixed conflicting policy epochs -> val acc 0.86 -> 0.47)
  refits:  every REFIT_EVERY=20 with R=48 x K_R=4 (bigger, rarer; v1's 24x2 median split was noise),
           skipped while the window minority < 32; EMA on (mu, rho) (v1 rotated 60-70 deg PER refit)
  meter:   PRISTINE seed head, never updated, scored on held-out prompt states each eval — the
           forging detector v1 lacked (phase-4 convention)

Evals every EVAL_EVERY=50: held-out implicit acc + dlp (dataset install), held-out ON-POLICY judge
z/reward/length/KL-per-token (the actual objective; v1 only logged train-rollout z), pristine +
current plan-meter, and for exactj the pi_rel(chosen) trajectory.

Env: UPPER_MODE=exactj|rloo RUN_TAG= L_PLAN=-1 STEPS=300 B_PAIR=8 TAU=3.0 MASS=1.0 | RL_BATCH=4
     RL_K=8 RL_KL=0.05 DPOP=1.0 | MCOEF=1.0 ANCH=0.1 LR=5e-5 REFIT_EVERY=20 REFIT_R=48 REFIT_K=4
     WIN=8 EMA=0.7 SEED_W=0.5 MAX_NEW=512 N_EVAL=64 HO_N=24 + the uf funnel knobs
Saves: /workspace/uf_hybrid3_{mode}{tag}_history.json, _ckpt{100,200}, _lora"""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/workspace/reward-depth")
from helpers import train_bayes_head, ResidualCapture, LOG_NDTR, _comp_logp

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
MODE = E("UPPER_MODE", "exactj"); assert MODE in ("exactj", "rloo")
TAG = E("RUN_TAG", ""); SFX = f"_{MODE}" + (f"_{TAG}" if TAG else "")
STEPS, LR = int(E("STEPS", 300)), float(E("LR", 5e-5))
B_PAIR, TAU, MASS = int(E("B_PAIR", 8)), float(E("TAU", 3.0)), float(E("MASS", 1.0))
BATCH, K, KL, DPOP = int(E("RL_BATCH", 4)), int(E("RL_K", 8)), float(E("RL_KL", 0.05)), float(E("DPOP", 1.0))
PESS = float(E("RL_PESS", 0.5))
MCOEF, ANCH = float(E("MCOEF", 1.0)), float(E("ANCH", 0.1))
MAX_NEW, MAX_LEN, PLEN = int(E("MAX_NEW", 512)), int(E("MAX_LEN", 1024)), int(E("PROMPT_LEN", 512))
REFIT_EVERY, REFIT_R, REFIT_K = int(E("REFIT_EVERY", 20)), int(E("REFIT_R", 48)), int(E("REFIT_K", 4))
WIN, EMA, SEED_W = int(E("WIN", 8)), float(E("EMA", 0.7)), float(E("SEED_W", 0.5))
EVAL_EVERY, N_EVAL, HO_N = int(E("EVAL_EVERY", 50)), int(E("N_EVAL", 64)), int(E("HO_N", 24))
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
print(f"[data] {len(recs)} pairs | train {len(train)} | test {len(test)} | mode={MODE}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); NL = len(BLOCKS); HID = model.config.hidden_size

# ---- outcome-judge (frozen; phase-3 L12 len-matched probe) ----
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
MU_J = jhead.mu.detach().float().to(DEV); SIG2_J = F.softplus(jhead.rho.detach()).float().pow(2).to(DEV)
SD_J = torch.tensor(sdJ, device=DEV); MN_J = torch.tensor(mnJ, device=DEV)
print(f"[judge] L{L_J} held-out acc {jacc:.3f}", flush=True)
def judge_reward(f):
    fs = (f.float() - MN_J) / SD_J
    s2 = fs.pow(2).matmul(SIG2_J)
    return torch.special.ndtr((fs.matmul(MU_J) - PESS * torch.sqrt(s2 + 1e-9)) / torch.sqrt(1 + s2))
def judge_zraw(f):
    return ((f.float() - MN_J) / SD_J).matmul(MU_J)

# exactj: per-pair judge rewards, precomputed once from the cached frozen features
with torch.no_grad():
    PAIR_R = torch.stack([judge_reward(torch.tensor(Fc_tr[:, L_J], device=DEV)),
                          judge_reward(torch.tensor(Fr_tr[:, L_J], device=DEV))], 1)  # (N_PROBE, 2)
print(f"[exactj rewards] r_c mean {PAIR_R[:,0].mean():.3f} | r_r mean {PAIR_R[:,1].mean():.3f} | "
      f"judge prefers chosen on {(PAIR_R[:,0] > PAIR_R[:,1]).float().mean():.3f}", flush=True)

# ---- plan-reader: seed fit from the sweep artifacts; FIXED threshold; pristine meter ----
plan_curve = json.load(open("/workspace/uf_plan_curve.json"))
LP = L_PLAN if L_PLAN >= 0 else int(plan_curve["best_layer"])
S = json.load(open("/workspace/uf_plan_samples.json"))
zz = np.load("/workspace/uf_plan_feats.npz"); Ptr, Pte = zz["tr"], zz["te"]
ztr = np.array([np.mean(x["z"]) for x in S["train"]]); zte = np.array([np.mean(x["z"]) for x in S["test"]])
THR = float(np.median(ztr))          # stationary label threshold, forever
pt_tr = np.where(ztr > THR, 1.0, -1.0).astype(np.float32)
pt_te = np.where(zte > THR, 1.0, -1.0).astype(np.float32)
cw_tr = np.abs(ztr - THR); cw_tr = (cw_tr / max(cw_tr.mean(), 1e-6)).astype(np.float32)
cw_te = np.abs(zte - THR); cw_te = (cw_te / max(cw_te.mean(), 1e-6)).astype(np.float32)
MN_P = torch.tensor(Ptr[:, LP].mean(0), device=DEV)
SD_P = torch.tensor(Ptr[:, LP].std(0) + 1e-6, device=DEV)
_ptr = (Ptr[:, LP] - MN_P.cpu().numpy()) / SD_P.cpu().numpy()
_pte = (Pte[:, LP] - MN_P.cpu().numpy()) / SD_P.cpu().numpy()
pacc, phead, _ = train_bayes_head(_ptr, pt_tr, _pte, pt_te, w_tr=cw_tr, w_te=cw_te)
MU_P = phead.mu.detach().float().to(DEV); RHO_P = phead.rho.detach().float().to(DEV)
MU_SEED, RHO_SEED = MU_P.clone(), RHO_P.clone()          # pristine meter, never updated
print(f"[plan] L{LP} seed plan-reader held-out acc {pacc:.3f} | THR {THR:+.3f} | "
      f"margin <= {LP}, {MODE} > {LP}", flush=True)
def plan_z(f, mu=None, rho=None):    # f: centered/scaled by caller
    mu = MU_P if mu is None else mu; rho = RHO_P if rho is None else rho
    s2 = f.pow(2).matmul(F.softplus(rho).pow(2))
    return f.matmul(mu) / torch.sqrt(1 + s2)

# seed buffer entry (capped at 512, confidence-weighted, SEED_W-downweighted)
_sel = np.random.RandomState(SEED).choice(len(_ptr), min(512, len(_ptr)), replace=False)
BUF_SEED = (torch.tensor(_ptr[_sel], dtype=torch.float32), torch.tensor(pt_tr[_sel]),
            torch.tensor(cw_tr[_sel] * SEED_W))
buf_win = []                                              # list of (f, t, w), windowed

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
    enc = tok(prompts_txt, return_tensors="pt", padding=True, truncation=True, max_length=PLEN).to(DEV)
    import contextlib
    cm = policy.disable_adapter if base else contextlib.nullcontext
    with (torch.enable_grad() if grad else torch.no_grad()), cm():
        with ResidualCapture([BLOCKS[LP]]) as cap:
            policy(**enc, logits_to_keep=1)
        return (cap.get()[0][:, -1].float() - MN_P) / SD_P

@torch.no_grad()
def rollout_feats(prompts_raw, gen, P, bs=8, k=1):
    """Judge-layer residual at the re-rendered eos sentinel, frozen base (phase-3 conventions)."""
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

def gen_rollouts(prompts_txt, k, chunk=12):
    """Sample k completions per prompt; returns (gen, P) per chunk list to bound KV memory."""
    outs = []
    for s in range(0, len(prompts_txt), chunk):
        enc = tok(prompts_txt[s:s + chunk], return_tensors="pt", padding=True, truncation=True,
                  max_length=PLEN).to(DEV)
        policy.config.use_cache = True
        with torch.no_grad():
            g = policy.generate(**enc, do_sample=True, temperature=1.0, num_return_sequences=k,
                                max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
        policy.config.use_cache = False
        outs.append((g, enc.input_ids.shape[1]))
    return outs

# ---- pair likelihoods (batched, left-padded; used by exactj step, DPOP, and evals) ----
def pair_lps(batch, grad, ref=False):
    """Total completion log-prob for chosen+rejected of each rec. Returns (lp_c, lp_r, n_c, n_r)."""
    import contextlib
    texts, ns = [], []
    for x in batch:
        fp = render_prompt(x["prompt"])
        plen = len(tok(fp, truncation=True, max_length=MAX_LEN).input_ids)
        for side in ("chosen", "rejected"):
            full = render_full(x["prompt"], x[side])
            fl = len(tok(full, truncation=True, max_length=MAX_LEN).input_ids)
            texts.append(full); ns.append(max(fl - min(plen, fl - 1), 1))
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
    cm = policy.disable_adapter if ref else contextlib.nullcontext
    with (torch.enable_grad() if grad else torch.no_grad()), cm():
        lp = _comp_logp(policy(**enc, logits_to_keep=max(ns) + 1).logits, enc.input_ids, ns)
    B = len(batch)
    n = torch.tensor(ns, dtype=torch.float32, device=DEV).view(B, 2)
    return lp.view(B, 2)[:, 0], lp.view(B, 2)[:, 1], n[:, 0], n[:, 1]

@torch.no_grad()
def evaluate(step):
    policy.eval()
    # (1) dataset install: held-out implicit-reward acc + dlp
    ir = []
    for s in range(0, N_EVAL, 4):
        chunk = test[s:s + 4]
        lc, lr_, _, _ = pair_lps(chunk, False)
        rc, rr, _, _ = pair_lps(chunk, False, ref=True)
        for i in range(len(chunk)):
            ir.append((float((lc[i] - rc[i]) > (lr_[i] - rr[i])), float(lc[i] - rc[i]), float(lr_[i] - rr[i])))
    a = np.array(ir)
    ev = dict(step=step, acc_implicit=float(a[:, 0].mean()),
              dlp_chosen=float(a[:, 1].mean()), dlp_rejected=float(a[:, 2].mean()))
    # (2) on-policy: held-out prompts -> generate -> judge; KL per token vs ref
    ho = test[200:200 + HO_N]
    zs, lens, kls = [], [], []
    for g, P in gen_rollouts([render_prompt(x["prompt"]) for x in ho], 2):
        npr = g.shape[0] // 2
        fj = rollout_feats([x["prompt"] for x in ho[:npr]], g, P, k=2); ho = ho[npr:]
        zs.append(judge_zraw(fj).cpu())
        attn = (g != tok.pad_token_id).long()
        n_new = attn[:, P:].sum(1).clamp(min=1)
        keepg = g.shape[1] - P + 1
        tm = attn[:, P:].bool()
        lsm = F.log_softmax(policy(input_ids=g, attention_mask=attn, logits_to_keep=keepg).logits[:, :-1].float(), -1)
        lp_ = (lsm.gather(-1, g[:, P:, None]).squeeze(-1) * tm).sum(1)
        with policy.disable_adapter():
            rsm = F.log_softmax(policy(input_ids=g, attention_mask=attn, logits_to_keep=keepg).logits[:, :-1].float(), -1)
            rp_ = (rsm.gather(-1, g[:, P:, None]).squeeze(-1) * tm).sum(1)
        del lsm, rsm
        kls.append(((lp_ - rp_) / n_new).cpu()); lens.append(n_new.float().cpu())
    ev.update(zjudge_ho=float(torch.cat(zs).mean()), len_ho=float(torch.cat(lens).mean()),
              kl_tok_ho=float(torch.cat(kls).mean()))
    # (3) plan meters on held-out prompt states: pristine (forging detector) + current head
    hp = [render_prompt(x["prompt"]) for x in test[200:200 + 32]]
    f_ho = plan_feats_pol(hp, grad=False)
    ev.update(pristine_z=float(plan_z(f_ho, MU_SEED, RHO_SEED).mean()),
              pristine_frac=float((plan_z(f_ho, MU_SEED, RHO_SEED) > 0).float().mean()),
              curhead_z=float(plan_z(f_ho).mean()))
    policy.train()
    torch.cuda.empty_cache()   # generation KV + eval logits fragment the allocator across phases
    return ev

# ---- training ----
hist = dict(mode=MODE, L_plan=LP, L_judge=L_J, plan_acc=float(pacc), THR=THR,
            step_log=[], evals=[], refit=[])
rgen = random.Random(4242); policy.train()
ev = evaluate(0); hist["evals"].append(ev); print(f"  step    0: EVAL {ev}", flush=True)
for step in range(STEPS):
    log = dict(step=step + 1)
    if MODE == "exactj":
        idx = rgen.sample(range(len(pr)), B_PAIR)
        batch = [pr[i] for i in idx]
        prompts_txt = [render_prompt(x["prompt"]) for x in batch]
        # margin half (<= LP)
        opt.zero_grad()
        f_base = plan_feats_pol(prompts_txt, grad=False, base=True)
        f_pol = plan_feats_pol(prompts_txt, grad=True)
        m_loss = -MCOEF * LOG_NDTR(plan_z(f_pol)).mean() + ANCH * (f_pol - f_base).pow(2).mean()
        m_loss.backward()
        g_low = [(p_, p_.grad.clone() if p_.grad is not None else None) for p_ in low_params]
        # exact-J half (> LP): pi_rel on per-token logprobs, judge rewards from cache, mass anchor.
        # MICRO-BATCHED one pair (2 seqs) at a time: a batched grad forward at 16 seqs x ~1k tokens
        # stores ~90GB of activations (no gradient checkpointing) — the smoke-test OOM.
        jl_, ms_, pic_ = [], [], []
        for bi, i in enumerate(idx):
            rec = pr[i]
            lp_c, lp_r, n_c, n_r = pair_lps([rec], grad=True)
            with torch.no_grad():
                rc_, rr_, _, _ = pair_lps([rec], grad=False, ref=True)
            pi = torch.softmax(torch.stack([TAU * lp_c / n_c, TAU * lp_r / n_r], 1), 1)
            r = PAIR_R[i:i + 1]
            j_loss = -(pi * r).sum(1).mean()
            mass = F.relu(torch.logsumexp(torch.stack([rc_, rr_], 1), 1)
                          - torch.logsumexp(torch.stack([lp_c, lp_r], 1), 1)).mean()
            ((j_loss + MASS * mass) / B_PAIR).backward()
            jl_.append(float(j_loss.detach())); ms_.append(float(mass.detach()))
            pic_.append(float(pi[0, 0].detach()))
        for p_, g in g_low: p_.grad = g
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        r_all = PAIR_R[idx]
        log.update(mloss=float(m_loss.detach()), jloss=float(np.mean(jl_)),
                   mass=float(np.mean(ms_)), pi_c=float(np.mean(pic_)),
                   r_gap=float((r_all[:, 0] - r_all[:, 1]).mean()))
    else:  # rloo
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
        n_new = attn[:, P:].sum(1).clamp(min=1)
        fj = rollout_feats([x["prompt"] for x in batch], gen, P, k=K)
        r = judge_reward(fj).detach()
        zj = judge_zraw(fj).detach()
        keepg = gen.shape[1] - P + 1
        tokmask = attn[:, P:].bool()
        with torch.no_grad():
            lsm = F.log_softmax(policy(input_ids=gen, attention_mask=attn, logits_to_keep=keepg).logits[:, :-1].float(), -1)
            logp_ng = (lsm.gather(-1, gen[:, P:, None]).squeeze(-1) * tokmask).sum(1)
            with policy.disable_adapter():
                rsm = F.log_softmax(policy(input_ids=gen, attention_mask=attn, logits_to_keep=keepg).logits[:, :-1].float(), -1)
                ref_logp = (rsm.gather(-1, gen[:, P:, None]).squeeze(-1) * tokmask).sum(1)
            del lsm, rsm
        r = r - KL * (logp_ng - ref_logp) / n_new
        rg = r.view(BATCH, K)
        adv = (rg - (rg.sum(1, keepdim=True) - rg) / (K - 1)).view(-1)
        # margin half (<= LP)
        opt.zero_grad()
        f_base = plan_feats_pol(prompts_txt, grad=False, base=True)
        f_pol = plan_feats_pol(prompts_txt, grad=True)
        m_loss = -MCOEF * LOG_NDTR(plan_z(f_pol)).mean() + ANCH * (f_pol - f_base).pow(2).mean()
        m_loss.backward()
        g_low = [(p_, p_.grad.clone() if p_.grad is not None else None) for p_ in low_params]
        # RLOO half (> LP)
        for s0 in range(0, BATCH * K, 4):
            sl = slice(s0, min(s0 + 4, BATCH * K))
            if not adv[sl].abs().sum() > 0: continue
            li = F.log_softmax(policy(input_ids=gen[sl], attention_mask=attn[sl],
                                      logits_to_keep=keepg).logits[:, :-1].float(), -1)
            lp_i = (li.gather(-1, gen[sl, P:, None]).squeeze(-1) * tokmask[sl]).sum(1)
            (-(adv[sl] * lp_i / n_new[sl]).sum() / (BATCH * K)).backward()
        # DPOP displacement anchor on the dataset chosen side (phase-2 cure) — all blocks.
        # Micro-batched per rec (2 seqs) for the same activation-memory reason as exactj.
        if DPOP > 0:
            for x in batch:
                lc, _, _, _ = pair_lps([x], grad=True)
                with torch.no_grad():
                    rc_, _, _, _ = pair_lps([x], grad=False, ref=True)
                (DPOP * F.relu(rc_ - lc).mean() / BATCH).backward()
        for p_, g in g_low: p_.grad = g
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log.update(mloss=float(m_loss.detach()), reward=float(rg.mean()),
                   zjudge=float(zj.mean()), len=float(n_new.float().mean()))
    hist["step_log"].append(log)
    # ---- plan-reader refit: fixed-threshold labels, confidence weights, window, EMA ----
    if (step + 1) % REFIT_EVERY == 0:
        rqs = rgen.sample(train, REFIT_R)
        rtxt = [render_prompt(x["prompt"]) for x in rqs]
        zb_all, done = [], 0
        for g, P in gen_rollouts(rtxt, REFIT_K):
            npr = g.shape[0] // REFIT_K
            fj_r = rollout_feats([x["prompt"] for x in rqs[done:done + npr]], g, P, k=REFIT_K)
            zb_all.append(judge_zraw(fj_r).view(npr, REFIT_K).mean(1)); done += npr
        zb = torch.cat(zb_all)
        fb = plan_feats_pol(rtxt, grad=False)
        tb = torch.where(zb > THR, 1.0, -1.0)
        wb = (zb - THR).abs(); wb = wb / wb.mean().clamp(min=1e-6)
        buf_win.append((fb.cpu(), tb.cpu(), wb.cpu()))
        if len(buf_win) > WIN: buf_win.pop(0)
        Fb = torch.cat([BUF_SEED[0]] + [b[0] for b in buf_win]).numpy()
        Tb = torch.cat([BUF_SEED[1]] + [b[1] for b in buf_win]).numpy()
        Wb = torch.cat([BUF_SEED[2]] + [b[2] for b in buf_win]).numpy()
        fresh_t = torch.cat([b[1] for b in buf_win])
        minority = int(min((fresh_t > 0).sum(), (fresh_t < 0).sum()))
        if minority >= 32:
            perm = np.random.RandomState(step).permutation(len(Tb)); ntr = int(0.85 * len(Tb))
            bacc, hnew, _ = train_bayes_head(Fb[perm[:ntr]], Tb[perm[:ntr]], Fb[perm[ntr:]], Tb[perm[ntr:]],
                                             w_tr=Wb[perm[:ntr]], w_te=Wb[perm[ntr:]], epochs=60, patience=8)
            mu_fit = hnew.mu.detach().float().to(DEV); rho_fit = hnew.rho.detach().float().to(DEV)
            rot_fit = float(torch.rad2deg(torch.arccos(torch.clamp(
                F.cosine_similarity(mu_fit, MU_P, dim=0), -1, 1))))
            MU_P = EMA * MU_P + (1 - EMA) * mu_fit
            RHO_P = EMA * RHO_P + (1 - EMA) * rho_fit
            rot_cum = float(torch.rad2deg(torch.arccos(torch.clamp(
                F.cosine_similarity(MU_P, MU_SEED, dim=0), -1, 1))))
            hist["refit"].append(dict(step=step + 1, n=len(Tb), minority=minority, val_acc=float(bacc),
                                      rot_fit=rot_fit, rot_cum=rot_cum,
                                      frac_pos=float((zb > THR).float().mean())))
            print(f"  refit {step+1:4d}: n {len(Tb)} val {bacc:.3f} rot_fit {rot_fit:.0f} "
                  f"rot_cum {rot_cum:.0f} frac_pos {(zb > THR).float().mean():.2f}", flush=True)
        else:
            hist["refit"].append(dict(step=step + 1, skipped=True, minority=minority,
                                      frac_pos=float((zb > THR).float().mean())))
            print(f"  refit {step+1:4d}: SKIPPED (window minority {minority} < 32, "
                  f"frac_pos {(zb > THR).float().mean():.2f})", flush=True)
        torch.cuda.empty_cache()
    if (step + 1) % 10 == 0:
        recent = hist["step_log"][-10:]
        msg = " ".join(f"{k_} {np.mean([r_.get(k_, np.nan) for r_ in recent]):.3f}"
                       for k_ in (("mloss", "jloss", "mass", "pi_c") if MODE == "exactj"
                                  else ("mloss", "reward", "zjudge", "len")))
        print(f"  step {step+1:4d}: {msg}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(step + 1); hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL {ev}", flush=True)
        json.dump(hist, open(f"/workspace/uf_hybrid3{SFX}_history.json", "w"), indent=1)
    if (step + 1) % 100 == 0 and (step + 1) < STEPS:
        policy.save_pretrained(f"/workspace/uf_hybrid3{SFX}_ckpt{step+1}")
json.dump(hist, open(f"/workspace/uf_hybrid3{SFX}_history.json", "w"), indent=1)
policy.save_pretrained(f"/workspace/uf_hybrid3{SFX}_lora"); tok.save_pretrained(f"/workspace/uf_hybrid3{SFX}_lora")
print("DONE", flush=True)
