#!/usr/bin/env python
"""UF port of the cc stage-2.5 hybrid: adaptive MEAN-DIFF margin (blocks <= L*) + anchored
sampled RLOO from the frozen probe (blocks > L*). The load-bearing test of the margin half on the
realistic testbed: the co-arm is uf_probe_rl.py run unmodified (RLOO-only, full-stack), so the
margin half is the ONLY difference between the two runs.

WHAT EACH HALF IS
  margin  cc_stage2.py's meandiff objective, ported: forward the policy on the pair's full chat
          renders, read the L* residual at the eos sentinel, push each pair's (chosen - rejected)
          activation difference along the batch's own mean-difference direction -- DETACHED,
          lag-windowed over the last MD_K steps (cc lag-spectrum: k=0 forges, k>=1 doesn't,
          k=5 best), saturated (-logsigmoid of the scaled projection; raw linear detonates).
          No probe in the loss. IPW pair weights on BOTH the direction and the loss, so neither
          can be paid in length (the direction is the one place length could sneak back in).
  RLOO    uf_probe_rl.py's v3 recipe verbatim: sample K rollouts, decode + re-render each
          completion as a full chat turn, read the FROZEN base's L* residual at the sentinel,
          probe reward with pessimism LCB, KL-in-reward, RLOO baseline, DPOP hinge on the pair's
          chosen side. Reward is a function of emitted TEXT only -- unforgeable by construction.

GRADIENT ROUTING (phase-4 masking, as in cc hybrid): margin's backward is taken first and its
<=L* LoRA grads cloned; RLOO's backward then writes all blocks, after which the <=L* grads are
OVERWRITTEN with the margin's -- margin owns <=L*, RLOO owns >L*. The DPOP anchor backwards last,
across all blocks (cc's full-stack-anchor role). LoRA itself is full-stack.

DEVIATIONS FROM cc (documented, deliberate):
  - Margin batch is MBATCH=8 pairs sampled independently of the RLOO prompt batch (BATCH=2 is
    too few pairs for a per-step direction estimate; cc used 16-pair batches for both halves).
  - MD_NORM defaults ON: proj scale calibrated on 8 base-policy batches at init. cc's depth-norm
    control showed drive amplitude selects between install and forging, and Tulu-8B@L12 residual
    scales are not Qwen-3B@L20's -- calibrating pins the operating point; step-0 print shows it.
  - Anchor here is DPOP + KL-in-reward (uf_probe_rl's guards), not replay-K-FAC. Keeps the two
    arms' guard set identical, which is what makes the comparison clean.

Diagnostics: uf_margin_ewc.py's evaluate() (implicit acc + dlp displacement watch, z_selfread vs
z_frozen forging detector, held-out rollout probe reward r_gen + true KL/token), on test[:64] to
match uf_probe_rl's eval set.

Env (margin): MCOEF=1.0 MD_K=5 MBATCH=8 MB_PAIRS=2 MD_NORM=1
Env (RLOO, as uf_probe_rl): RL_STEPS=300 RL_BATCH=2 RL_K=4 RL_KL=0.03 RL_PESS=0.5 RL_ANCHOR=1.0
     RL_LR=5e-5 MAX_NEW=200 MAX_LEN=1024 UF_POOL=20000 N_PROBE=3000 PLATEAU_TOL=0.01
     UF_MATCH_LENGTH=1 UF_LEN_BUCKET=16 DROP_CAPPED=0 EVAL_EVERY=25 RUN_TAG=hyb
Outputs: /workspace/uf_hybrid_md_{TAG}_history.json, _ckptN, _lora"""
import os, sys, json, random, hashlib, contextlib
from itertools import islice
from collections import deque
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import train_bayes_head, ResidualCapture

E = os.environ.get
MODEL = E("UF_SFT_MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
POOL, N_PROBE = int(E("UF_POOL", 20000)), int(E("N_PROBE", 3000))
STEPS, BATCH, K = int(E("RL_STEPS", 300)), int(E("RL_BATCH", 2)), int(E("RL_K", 4))
KL, PESS, ANCHOR, LR = float(E("RL_KL", 0.03)), float(E("RL_PESS", 0.5)), float(E("RL_ANCHOR", 1.0)), float(E("RL_LR", 5e-5))
MAX_NEW, MAX_LEN, TOL = int(E("MAX_NEW", 200)), int(E("MAX_LEN", 1024)), float(E("PLATEAU_TOL", 0.01))
PLEN = int(E("PROMPT_LEN", 512))
MCOEF, MD_K = float(E("MCOEF", 1.0)), int(E("MD_K", 5))
MBATCH, MB_PAIRS, MD_NORM = int(E("MBATCH", 8)), int(E("MB_PAIRS", 2)), int(E("MD_NORM", 1))
# EMIT: which emission-channel head owns blocks > L*.
#   rloo     anchored sampled RLOO (v3 recipe) -- confirmed starvation-flat over 300 steps @4x8
#   softdpo  soft-label DPO from the frozen probe on the N_PROBE probe-fit pairs (uf_soft_dpo.py's
#            objective) -- the DENSE analogue of cc's exact-J: the phase-3 working method, so the
#            hybrid2 comparison measures the margin half against a live emission head.
#            Both halves then read the SAME batch (cc parity); DPOP anchor defaults off
#            (soft-DPO needs none). JONLY_LOW=1 restricts the emission head's writes to <= L*
#            (write-depth cell); JONLY_FULL=1 lets it write all blocks.
EMIT = E("EMIT", "rloo")
BETA = float(E("DPO_BETA", 0.1))
if EMIT == "softdpo": ANCHOR = float(E("RL_ANCHOR", 0.0))
EVAL_EVERY, N_EVAL_PAIRS, N_EVAL_GEN = int(E("EVAL_EVERY", 25)), int(E("N_EVAL_PAIRS", 64)), int(E("N_EVAL_GEN", 16))
TAG = E("RUN_TAG", "hyb")
DEV = "cuda"; SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
tok.truncation_side = "left"   # keep the END (response + eos): all reads are at the tail
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

# ---- data funnel + length matching (uf_probe_rl.py v3, verbatim) ----
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
    for x in recs:
        x["len_diff"] = _rlen(x["chosen"]) - _rlen(x["rejected"])
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
    wa = np.array([x["w"] for x in recs])
    print(f"[len-match] bucket={BUCKET} | kept {len(recs)} pairs | "
          f"Kish ESS {wa.sum()**2 / (wa**2).sum():.0f} | mean w {wa.mean():.3f}", flush=True)
else:
    for x in recs: x["w"] = 1.0
    print("[len-match] OFF", flush=True)
train = [x for x in recs if not x["is_test"]]
test = [x for x in recs if x["is_test"]]
print(f"[data] {len(recs)} pairs | train {len(train)} | test {len(test)}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); NL = len(BLOCKS); HID = model.config.hidden_size

@torch.no_grad()
def last_tok_feats(texts, bs=8):
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

# ---- Stage A: probe sweep (shared cache with uf_probe_rl.py) ----
cachef = E("UF_FEATS_CACHE", f"/workspace/uf_probe_feats{'_lenmatch' if MATCH else ''}.npz")
pr = train[:N_PROBE]; pe = test[:400]
w_pr = np.array([x["w"] for x in pr], np.float32); w_pe = np.array([x["w"] for x in pe], np.float32)
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
    pool = np.concatenate([Fc_tr[:, li], Fr_tr[:, li]])
    sd, mn = pool.std(0) + 1e-6, pool.mean(0)
    dtr = ((Fc_tr[:, li] - Fr_tr[:, li]) / sd) * s_tr[:, None]
    dte = ((Fc_te[:, li] - Fr_te[:, li]) / sd) * s_te[:, None]
    a, h, e = train_bayes_head(dtr, s_tr, dte, s_te, w_tr=w_pr, w_te=w_pe)
    acc[li], heads[li] = a, (h, sd, mn)
    print(f"  L{li:2d} acc={a:.3f} elbo={e:+.0f}", flush=True)
LSTAR = int(E("L_OVERRIDE", -1))
if LSTAR < 0: LSTAR = int(next(li for li in range(NL) if acc[li] >= acc.max() - TOL))
print(f"[probe] plateau layer L*={LSTAR} (acc {acc[LSTAR]:.3f}, max {acc.max():.3f})", flush=True)

head, sd_, mn_ = heads[LSTAR]
MU = head.mu.detach().float().to(DEV); SIG2 = F.softplus(head.rho.detach()).float().pow(2).to(DEV)
SD = torch.tensor(sd_, device=DEV); MN = torch.tensor(mn_, device=DEV)
def probe_reward(f):
    fs = (f.float() - MN) / SD     # centered absolute read (v3 lesson: uncentered inflates s2 ~17x)
    s2 = fs.pow(2).matmul(SIG2)
    return torch.special.ndtr((fs.matmul(MU) - PESS * torch.sqrt(s2 + 1e-9)) / torch.sqrt(1 + s2))
def margin_z(f_c, f_r):
    fs = ((f_c - f_r).float() / SD)   # difference read: centering cancels
    s2 = fs.pow(2).matmul(SIG2)
    return fs.matmul(MU) / torch.sqrt(1 + s2)

P_SOFT = None
if EMIT == "softdpo":
    # soft labels for the probe-fit pairs from the Stage-A cache (uf_soft_dpo.py's recipe:
    # posterior predictive on native difference features)
    fs_tr = torch.tensor((Fc_tr[:, LSTAR] - Fr_tr[:, LSTAR]) / sd_, dtype=torch.float32)
    s2_tr = fs_tr.pow(2).matmul(SIG2.cpu())
    z_tr = fs_tr.matmul(MU.cpu()) / torch.sqrt(1 + s2_tr)
    P_SOFT = torch.special.ndtr(z_tr).numpy()
    print(f"[softdpo] labels: mean p {P_SOFT.mean():.3f} | "
          f"soft 0.2-0.8: {(np.array((P_SOFT > 0.2) & (P_SOFT < 0.8))).mean():.2f}", flush=True)

# ---- full-stack LoRA (margin owns <= L*, RLOO owns > L* via grad routing) ----
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
if E("LOAD_LORA"):   # two-stage arm: warm-start from a prior arm's saved adapter
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    sd_load = load_file(os.path.join(E("LOAD_LORA"), "adapter_model.safetensors"))
    set_peft_model_state_dict(policy, sd_load)
    print(f"[load] adapter warm-start from {E('LOAD_LORA')} ({len(sd_load)} tensors)", flush=True)
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)
import re as _re
def _blk(n):
    m = _re.search(r"\.layers\.(\d+)\.", n); return int(m.group(1)) if m else -1
LOW = [p for n, p in policy.named_parameters() if p.requires_grad and 0 <= _blk(n) <= LSTAR]
print(f"[lora] full stack {sum(p.numel() for p in params)/1e6:.1f}M trainable | "
      f"{sum(p.numel() for p in LOW)/1e6:.1f}M in blocks <= {LSTAR}", flush=True)

def comp_logprob(text_full, plen, grad):
    ids = tok(text_full, return_tensors="pt", truncation=True, max_length=MAX_LEN + MAX_NEW).input_ids.to(DEV)
    plen = min(plen, ids.shape[1] - 1)
    with (torch.enable_grad() if grad else torch.no_grad()):
        keep = ids.shape[1] - plen + 1
        logits = policy(ids, logits_to_keep=keep).logits[0, :-1].float()
        return F.log_softmax(logits, -1).gather(-1, ids[0, plen:, None]).squeeze(-1).sum()

def pair_reads(pairs, use_policy, grad=False, bs=8):
    """L* last-token residuals of (chosen, rejected) full renders. Left padding -> [:, -1] is the
    eos sentinel. grad=True keeps the graph (margin training read; caller micro-batches)."""
    texts = [render_full(x["prompt"], x["chosen"]) for x in pairs] + \
            [render_full(x["prompt"], x["rejected"]) for x in pairs]
    ctxm = torch.enable_grad() if grad else torch.no_grad()
    outs = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        with ctxm, (contextlib.nullcontext() if use_policy else policy.disable_adapter()), \
             ResidualCapture([BLOCKS[LSTAR]]) as cap:
            policy(**enc, logits_to_keep=1)
        outs.append(cap.get()[0][:, -1])
    fs = torch.cat(outs, 0)
    B = len(pairs)
    return fs[:B], fs[B:]

@torch.no_grad()
def evaluate():
    """uf_margin_ewc.py's diagnostics on uf_probe_rl.py's eval set (test[:64])."""
    policy.eval(); ir = []
    for x in test[:N_EVAL_PAIRS]:
        pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.shape[1]
        lc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
        lr_ = comp_logprob(render_full(x["prompt"], x["rejected"]), pl, False)
        with policy.disable_adapter():
            rc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
            rr = comp_logprob(render_full(x["prompt"], x["rejected"]), pl, False)
        ir.append(dict(acc=float((lc - rc) > (lr_ - rr)), dc=float(lc - rc), dr=float(lr_ - rr)))
    ev = dict(acc_implicit=float(np.mean([x["acc"] for x in ir])),
              dlp_chosen=float(np.mean([x["dc"] for x in ir])),
              dlp_rejected=float(np.mean([x["dr"] for x in ir])))
    # forging detector: margin z through the policy's own read vs the frozen read, same pairs
    fc_p, fr_p = pair_reads(test[:N_EVAL_PAIRS], use_policy=True)
    fc_b, fr_b = pair_reads(test[:N_EVAL_PAIRS], use_policy=False)
    ev["z_selfread"] = float(margin_z(fc_p, fr_p).mean())
    ev["z_frozen"] = float(margin_z(fc_b, fr_b).mean())
    # held-out rollouts -> honest (frozen re-render) probe reward + true KL/token
    gp = test[N_EVAL_PAIRS:N_EVAL_PAIRS + N_EVAL_GEN]
    enc = tok([render_prompt(x["prompt"]) for x in gp], return_tensors="pt", padding=True,
              truncation=True, max_length=PLEN).to(DEV)
    policy.config.use_cache = True
    gen = policy.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=MAX_NEW,
                          pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    attn = (gen != tok.pad_token_id).long()
    n_new = attn[:, P:].sum(1).clamp(min=1)
    rerender = [render_full(x["prompt"], tok.decode(gen[i, P:], skip_special_tokens=True))
                for i, x in enumerate(gp)]
    fs = torch.zeros(len(rerender), HID, device=DEV)
    for s in range(0, len(rerender), 8):
        e2 = tok(rerender[s:s + 8], return_tensors="pt", padding=True, truncation=True,
                 max_length=MAX_LEN).to(DEV)
        with policy.disable_adapter(), ResidualCapture([BLOCKS[LSTAR]]) as cap:
            policy(**e2)
        fs[s:s + e2.input_ids.shape[0]] = cap.get()[0][:, -1]
    ev["r_gen"] = float(probe_reward(fs).mean())
    keep = gen.shape[1] - P + 1
    tokmask = attn[:, P:].bool()
    lsm = F.log_softmax(policy(input_ids=gen, attention_mask=attn, logits_to_keep=keep).logits[:, :-1].float(), -1)
    lp = (lsm.gather(-1, gen[:, P:, None]).squeeze(-1) * tokmask).sum(1)
    with policy.disable_adapter():
        rsm = F.log_softmax(policy(input_ids=gen, attention_mask=attn, logits_to_keep=keep).logits[:, :-1].float(), -1)
        rp = (rsm.gather(-1, gen[:, P:, None]).squeeze(-1) * tokmask).sum(1)
    ev["kl_tok_ho"] = float(((lp - rp) / n_new).mean())
    ev["len_gen"] = float(n_new.float().mean())
    policy.train()
    return ev

@torch.no_grad()
def rollout_feats(batch, gen, P, bs=8):
    """Frozen-base L* read of each decoded rollout re-rendered as a full chat turn (v3 recipe)."""
    texts = []
    for i, x in enumerate(batch):
        for j in range(K):
            comp = tok.decode(gen[i * K + j, P:], skip_special_tokens=True)
            texts.append(render_full(x["prompt"], comp))
    out = torch.zeros(len(texts), HID, device=DEV)
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        with policy.disable_adapter(), ResidualCapture([BLOCKS[LSTAR]]) as cap:
            policy(**enc)
        out[s:s + enc.input_ids.shape[0]] = cap.get()[0][:, -1]
    return out

# ---- MD_NORM: projection scale calibrated on base-policy batches at init (cc depth-norm) ----
rgen = random.Random(4242)      # RLOO batch stream (uf_probe_rl's seed)
rgen2 = random.Random(SEED + 77)  # margin batch stream (cc's rg0 seed)
buf = deque(maxlen=max(MD_K, 1))
def _wmeandiff(pairs):
    """IPW-weighted mean (chosen - rejected) L* activation difference, no grad."""
    fc, fr = pair_reads(pairs, use_policy=True)
    w = torch.tensor([x["w"] for x in pairs], device=DEV, dtype=torch.float32)
    return (((fc - fr).float()) * w[:, None]).sum(0) / w.sum()
proj_scale = 1.0
if MD_NORM and MCOEF > 0:
    with torch.no_grad():
        ms = [_wmeandiff(rgen2.sample(train, MBATCH)) for _ in range(8)]
        vv = torch.stack(ms).mean(0); vv = vv / (vv.norm() + 1e-8)
        ps = [float((m @ vv).abs() / np.sqrt(HID)) for m in ms]
        proj_scale = float(np.mean(ps)) + 1e-6
    print(f"[md] proj_scale {proj_scale:.4f} (raw batch-mean projections {ps})", flush=True)

# ---- train ----
hist = dict(Lstar=LSTAR, probe_acc=float(acc[LSTAR]), mcoef=MCOEF, md_k=MD_K, mbatch=MBATCH,
            md_norm=MD_NORM, proj_scale=proj_scale, kl=KL, anchor=ANCHOR, emit=EMIT, beta=BETA,
            jonly_low=int(E("JONLY_LOW", 0)), jonly_full=int(E("JONLY_FULL", 0)),
            jonly_upper=int(E("JONLY_UPPER", 0)), load_lora=E("LOAD_LORA", ""),
            reward=[], len=[], mloss=[], proj=[], evals=[])
ev0 = evaluate(); ev0["step"] = 0; hist["evals"].append(ev0)
print(f"  step    0: EVAL {ev0}", flush=True)
policy.train()
for step in range(STEPS):
    opt.zero_grad()
    if EMIT == "softdpo":   # one shared batch for both halves (cc parity), from the labeled pairs
        sidx = rgen.sample(range(len(pr)), MBATCH)
        sbatch = [pr[i] for i in sidx]
    # ================= margin half (blocks <= L*) =================
    mloss, proj_mean = 0.0, 0.0
    if MCOEF > 0:
        mbatch = sbatch if EMIT == "softdpo" else rgen2.sample(train, MBATCH)
        w_all = float(sum(x["w"] for x in mbatch))
        # pass 1 (no grad): this step's w-weighted mean-diff -> lag buffer -> direction v.
        # cc includes the CURRENT batch in the window (k=1 == zero-lag supervisor form); a second
        # graph-free pass keeps that semantics without retaining 8 pairs of activations at once.
        with torch.no_grad():
            buf.append(_wmeandiff(mbatch))
            m = torch.stack(list(buf)).mean(0); v = (m / (m.norm() + 1e-8)).detach()
        # pass 2 (grad, micro-batched by pairs): saturated IPW margin along v
        for s0 in range(0, MBATCH, MB_PAIRS):
            sub = mbatch[s0:s0 + MB_PAIRS]
            wsub = torch.tensor([x["w"] for x in sub], device=DEV, dtype=torch.float32)
            fc, fr = pair_reads(sub, use_policy=True, grad=True, bs=2 * len(sub))
            proj = ((fc - fr).float().matmul(v)) / np.sqrt(HID) / proj_scale
            ml = MCOEF * (-(F.logsigmoid(proj) * wsub).sum() / w_all)
            ml.backward()
            mloss += float(ml.detach()); proj_mean += float((proj.detach() * wsub).sum() / w_all)
    g_low = [(p, p.grad.clone() if p.grad is not None else None) for p in LOW]
    dloss = 0.0
    rg, n_new = torch.zeros(1), torch.zeros(1)   # placeholders for the shared logging path
    if EMIT == "softdpo":
        # ================= soft-DPO half (uf_soft_dpo.py objective) =================
        for i, x in enumerate(sbatch):
            p_ = float(P_SOFT[sidx[i]])
            pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True,
                     max_length=MAX_LEN).input_ids.shape[1]
            lc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, True)
            lr_ = comp_logprob(render_full(x["prompt"], x["rejected"]), pl, True)
            with torch.no_grad(), policy.disable_adapter():
                rc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
                rr = comp_logprob(render_full(x["prompt"], x["rejected"]), pl, False)
            D = BETA * ((lc - rc) - (lr_ - rr))
            l = -(p_ * F.logsigmoid(D) + (1 - p_) * F.logsigmoid(-D)) / MBATCH
            l.backward()
            dloss += float(l.detach())
    elif EMIT == "rloo":
        # ================= RLOO half (uf_probe_rl.py v3, verbatim) =================
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
        n_new = (attn[:, P:]).sum(1).clamp(min=1)
        r = probe_reward(rollout_feats(batch, gen, P)).detach()
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
        valid = (n_new < MAX_NEW) if int(E("DROP_CAPPED", 0)) else torch.ones_like(n_new, dtype=torch.bool)
        rg, vg = r.view(BATCH, K), valid.view(BATCH, K).float()
        cnt = vg.sum(1, keepdim=True)
        loo = (rg * vg).sum(1, keepdim=True) - rg * vg
        base = loo / (cnt - vg).clamp(min=1)
        adv = torch.where((vg > 0) & (cnt > 1.5), rg - base, torch.zeros_like(rg)).view(-1)
        hist.setdefault("trunc", []).append(float((n_new >= MAX_NEW).float().mean()))
        for s0 in range(0, BATCH * K, 4):           # micro-batched backward, chunks of 4
            sl = slice(s0, min(s0 + 4, BATCH * K))
            if not adv[sl].abs().sum() > 0: continue
            li = F.log_softmax(policy(input_ids=gen[sl], attention_mask=attn[sl],
                                      logits_to_keep=keepg).logits[:, :-1].float(), -1)
            lp_i = (li.gather(-1, gen[sl, P:, None]).squeeze(-1) * tokmask[sl]).sum(1)
            (-(adv[sl] * lp_i / n_new[sl]).sum() / (BATCH * K)).backward()
    # phase-4 routing: margin owns <= L* -- overwrite the emission head's low-block contribution.
    # JONLY_LOW inverts it: the emission head writes ONLY blocks <= L* (write-depth cell).
    if int(E("JONLY_LOW", 0)):
        for n, p in policy.named_parameters():
            if p.requires_grad and _blk(n) > LSTAR: p.grad = None
    elif int(E("JONLY_UPPER", 0)):    # emission head writes ONLY blocks > L* (stage-2 arm)
        for p in LOW: p.grad = None
    elif MCOEF > 0 and not int(E("JONLY_FULL", 0)):
        for p, g in g_low: p.grad = g
    if ANCHOR > 0:  # DPOP hinge on the pair's chosen side (full-stack, after routing)
        abatch = sbatch if EMIT == "softdpo" else batch
        for x in abatch:
            pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.shape[1]
            lc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, True)
            with torch.no_grad(), policy.disable_adapter():
                rc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
            (ANCHOR * F.relu(rc - lc) / len(abatch)).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["reward"].append(float(rg.mean())); hist["len"].append(float(n_new.float().mean()))
    hist["mloss"].append(mloss); hist["proj"].append(proj_mean)
    hist.setdefault("dloss", []).append(dloss)
    if (step + 1) % 10 == 0:
        head_stat = (f"dloss {np.mean(hist['dloss'][-10:]):.4f}" if EMIT == "softdpo" else
                     f"reward {np.mean(hist['reward'][-10:]):.3f} len {np.mean(hist['len'][-10:]):.0f}")
        print(f"  step {step+1:4d}: {head_stat} mloss {np.mean(hist['mloss'][-10:]):.4f} "
              f"proj {np.mean(hist['proj'][-10:]):.3f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(); ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL {ev}", flush=True)
        json.dump(hist, open(f"/workspace/uf_hybrid_md_{TAG}_history.json", "w"), indent=1)
    if (step + 1) % 100 == 0:
        policy.save_pretrained(f"/workspace/uf_hybrid_md_{TAG}_ckpt{step+1}")
json.dump(hist, open(f"/workspace/uf_hybrid_md_{TAG}_history.json", "w"), indent=1)
policy.save_pretrained(f"/workspace/uf_hybrid_md_{TAG}_lora")
print("DONE", flush=True)
