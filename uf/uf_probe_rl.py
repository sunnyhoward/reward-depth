#!/usr/bin/env python
"""RL-from-probe on UltraFeedback (Tulu-3-8B-SFT). Stage A: per-layer Bayesian probe sweep on
chosen/rejected last-token residuals -> pick L* where accuracy plateaus. Stage B: RLOO with the
probe (frozen-base read at L*, pessimism LCB) as reward, KL-in-reward, DPOP anchor on the pair's
chosen side. Saves probe curve, history, checkpoints at 100/200/300 steps.

Stage A applies length matching (IPW) by default: UF's chosen side is systematically longer, so an
unmatched probe scores ~0.62 on length alone and the layer curve conflates "sees length" with "sees
preference". Stage B reads the probe at the render_full eos sentinel (see rollout_feats).

Env: UF_POOL=20000 N_PROBE=3000 N_EVAL=96 RL_STEPS=300 RL_BATCH=2 RL_K=4 RL_KL=0.03 RL_PESS=0.5
     RL_ANCHOR=1.0 RL_LR=5e-5 MAX_NEW=200 MAX_LEN=1024 PLATEAU_TOL=0.01
     UF_MATCH_LENGTH=1 UF_LEN_BUCKET=16 DROP_CAPPED=0

2026-08-03 (phase-9 UF port, per notes_dense_probe_rl.md + NEXT.md + phase-8 lessons):
  UF_READ=last|mean   mean = probe fit on the POOLED cache (uf/uf_meanpool_sweep.py must have
                      written /workspace/uf_probe_feats_meanpool.npz); RL-time reward reads then
                      mean-pool the FROZEN base's L* residuals over the re-rendered rollout's
                      completion tokens (prefix-valid running mean -> dense Phi).
  RL_MODE=rloo|shaped|pooled_margin
    rloo           unchanged v3 recipe (the flat baseline).
    shaped         dense credit: A_t = (R - b_LOO) + SHAPE_W*(Phi_end - Phi_t), Phi_t = probe
                   posterior of the running mean over completion tokens 1..t; per-token PG on the
                   re-rendered completion (teacher-forced, same token grid as Phi). Frozen read —
                   no gradient through policy activations. Requires UF_READ=mean.
    pooled_margin  styc phase-8 winner ported: teacher-forced chosen/rejected pairs, POLICY
                   activations at L* pooled over completion tokens, lag-1 adaptive mean-diff
                   margin, DPOP anchor. Backprop through policy activations (forging channel
                   open by design; pooling is the defense).
  Guards (phase-8 REQUIRED set): TRUNC_PEN deferral guard (reward penalty on rollouts that hit
  MAX_NEW with no eos — the preamble-inflation exploit), REPLAY_N generative-replay floor on
  random-token base continuations (default ON for UF; phase-3 refusals/reasoning collapse is the
  evidence), CKPT_EVERY adapter checkpoints (styc shaped peak ~step 125 was lost to sparse ckpts).
  Extra env: SHAPE_W=1 TRUNC_PEN=0.25 REPLAY_N=64 REPLAY_W=1.0 REPLAY_BS=8 REPLAY_MAXNEW=64
             CKPT_EVERY=50 EVAL_EVERY=50 L_OVERRIDE= MARGIN_LR=1e-4 M0=4.0 MARGIN_BS=8"""
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
MODE = E("RL_MODE", "rloo")            # rloo | shaped | pooled_margin | hybrid (shaped + MCOEF*margin)
READ = E("UF_READ", "last")            # last | mean
assert MODE in ("rloo", "shaped", "pooled_margin", "hybrid") and READ in ("last", "mean")
if MODE in ("shaped", "hybrid"): assert READ == "mean", "shaped/hybrid need prefix-valid pooled Phi (UF_READ=mean)"
SHAPE_W = float(E("SHAPE_W", 1.0)) if MODE in ("shaped", "hybrid") else 0.0
MCOEF = float(E("MCOEF", 1.0))         # hybrid: weight of the margin half
REWARD_FORM = E("REWARD_FORM", "ndtr") # ndtr (v3 squash) | z (raw LCB, clipped — spread-preserving)
TRUNC_PEN = float(E("TRUNC_PEN", 0.25))
REPLAY_N, REPLAY_W, REPLAY_BS = int(E("REPLAY_N", 64)), float(E("REPLAY_W", 1.0)), int(E("REPLAY_BS", 8))
REPLAY_MAXNEW = int(E("REPLAY_MAXNEW", 64))
CKPT_EVERY, EVAL_EVERY = int(E("CKPT_EVERY", 50)), int(E("EVAL_EVERY", 50))
MARGIN_LR, M0, MARGIN_BS = float(E("MARGIN_LR", 1e-4)), float(E("M0", 4.0)), int(E("MARGIN_BS", 8))
TAG = E("RUN_TAG", "v3" if MODE == "rloo" else MODE)   # v3 = re-render read + len-matched probe + left truncation
DEV = "cuda"; SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
tok.truncation_side = "left"   # keep the END (the response + eos) when a pair exceeds MAX_LEN:
                               # the probe reads the last token, so right-truncation would drop it
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()
def _n_comp(p, r):
    """Completion-token count after left-truncation (mirrors uf_meanpool_sweep.py): with left
    padding + left truncation the completion occupies the rightmost n_comp columns."""
    nf = len(tok(render_full(p, r), add_special_tokens=False).input_ids)
    np_ = len(tok(render_prompt(p), add_special_tokens=False).input_ids)
    return max(1, min(nf - np_, min(nf, MAX_LEN)))

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

# ---- length matching (ported from the source repo's ultrafeedback_head_prob_sweep.py) ----
# UF's chosen side is systematically longer (268 vs 193 response tokens; chosen longer in 61% of
# pairs), so a probe can score ~0.62 on length ALONE. Without this control the layer sweep measures
# "can layer L see length" as much as "can layer L see preference". IPW weighting on |len_diff|
# buckets equalises the two directions within each bucket; MATCH=0 restores the old unweighted set.
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
    print("[len-match] OFF (UF_MATCH_LENGTH=0)", flush=True)

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
PROBE_SRC = E("PROBE_SRC", "dataset")   # dataset | onpolicy (judge-labelled same-prompt rollout
                                        # pairs from uf_onpolicy_{sample,judge,probe}.py — fixes
                                        # the phase-9 headroom problem: discrimination at the
                                        # policy's own quality level. Requires L_OVERRIDE.)
cachef = ("/workspace/uf_probe_feats_meanpool.npz" if READ == "mean"
          else f"/workspace/uf_probe_feats{'_lenmatch' if MATCH else ''}.npz")
pr = train[:N_PROBE]; pe = test[:400]
w_pr = np.array([x["w"] for x in pr], np.float32); w_pe = np.array([x["w"] for x in pe], np.float32)
if PROBE_SRC == "onpolicy":
    assert READ == "mean" and E("L_OVERRIDE"), "onpolicy probe is pooled; set L_OVERRIDE (gate L*)"
    LSTAR = int(E("L_OVERRIDE"))
    zj = np.load("/workspace/uf_onpolicy_feats.npz")
    Fw_o, Fl_o = zj["a"], zj["b"]                      # winner / loser pooled feats, all layers
    rngo = np.random.RandomState(SEED)
    s_o = np.where(rngo.rand(len(Fw_o)) < 0.5, 1.0, -1.0).astype(np.float32)
    pool_o = np.concatenate([Fw_o[:, LSTAR], Fl_o[:, LSTAR]])
    sd_, mn_ = pool_o.std(0) + 1e-6, pool_o.mean(0)
    d_o = ((Fw_o[:, LSTAR] - Fl_o[:, LSTAR]) / sd_) * s_o[:, None]
    n90 = int(0.9 * len(d_o))
    a_o, head, e_o = train_bayes_head(d_o[:n90], s_o[:n90], d_o[n90:], s_o[n90:])
    print(f"[probe] ONPOLICY judge-labelled fit @L{LSTAR}: heldout-split acc {a_o:.3f} "
          f"(n={len(d_o)})", flush=True)
else:
    head = None   # fit below from the dataset cache
if PROBE_SRC != "onpolicy":
    if os.path.exists(cachef):
        z = np.load(cachef); Fc_tr, Fr_tr, Fc_te, Fr_te = z["a"], z["b"], z["c"], z["d"]
    elif READ == "mean":
        raise SystemExit(f"pooled cache {cachef} missing — run uf/uf_meanpool_sweep.py first")
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
        sd, mn = pool.std(0) + 1e-6, pool.mean(0)  # mn unused at fit time (differences cancel it)
        dtr = ((Fc_tr[:, li] - Fr_tr[:, li]) / sd) * s_tr[:, None]
        dte = ((Fc_te[:, li] - Fr_te[:, li]) / sd) * s_te[:, None]
        a, h, e = train_bayes_head(dtr, s_tr, dte, s_te, w_tr=w_pr, w_te=w_pe)
        acc[li], heads[li] = a, (h, sd, mn)
        print(f"  L{li:2d} acc={a:.3f} elbo={e:+.0f}", flush=True)
    LSTAR = int(next(li for li in range(NL) if acc[li] >= acc.max() - TOL))
    if E("L_OVERRIDE"): LSTAR = int(E("L_OVERRIDE"))
    print(f"[probe] plateau layer L*={LSTAR} (acc {acc[LSTAR]:.3f}, max {acc.max():.3f}, read={READ})", flush=True)
    json.dump(dict(layer_acc=acc.tolist(), Lstar=LSTAR, len_matched=bool(MATCH), read=READ),
              open(f"/workspace/uf_probe_curve{'_meanpool' if READ == 'mean' else ('_lenmatch' if MATCH else '')}.json", "w"))
    head, sd_, mn_ = heads[LSTAR]
MU = head.mu.detach().float().to(DEV); SIG2 = F.softplus(head.rho.detach()).float().pow(2).to(DEV)
SD = torch.tensor(sd_, device=DEV); MN = torch.tensor(mn_, device=DEV)
def probe_reward(f):
    # CENTER the features. The head is fit on difference features (the pooled mean cancels), but at
    # RL time we score absolute reads: without centering, the mean term inflates s2 ~17x (648 vs 38
    # measured at L12), and the sqrt(1+s2) denominator + pessimism LCB squash the reward spread ~3.3x
    # (chosen-vs-rejected gap 0.26 -> 0.08). Ranking is unaffected; the RLOO advantage SNR is not.
    fs = (f.float() - MN) / SD
    s2 = fs.pow(2).matmul(SIG2)
    lcb = fs.matmul(MU) - PESS * torch.sqrt(s2 + 1e-9)
    if REWARD_FORM == "z":   # spread-preserving: the CDF squash eats within-prompt z differences
        return lcb.clamp(-6, 6)
    return torch.special.ndtr(lcb / torch.sqrt(1 + s2))

# ---- Stage B: RLOO from probe at L* ----
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=(MARGIN_LR if MODE == "pooled_margin" else LR))

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

@torch.no_grad()
def rollout_feats(batch, gen, P, bs=8):
    """Layer-L* residual of each rollout read at the render_full eos sentinel -- the position the
    probe was fit on. Decodes the completion and re-renders it as a full chat turn, so a rollout that
    never emitted eos (hit MAX_NEW) still gets scored at a sentinel rather than a content token."""
    texts = []
    for i, x in enumerate(batch):
        for j in range(K):
            comp = tok.decode(gen[i * K + j, P:], skip_special_tokens=True)
            texts.append(render_full(x["prompt"], comp))
    out = torch.zeros(len(texts), HID, device=DEV)
    for s in range(0, len(texts), bs):
        # max_length=MAX_LEN, NOT MAX_LEN+MAX_NEW: Stage A extracted probe features at MAX_LEN, so
        # RL-time reads must see the same (left-)truncation window to stay on the probe's distribution
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        with policy.disable_adapter(), ResidualCapture([BLOCKS[LSTAR]]) as cap:
            policy(**enc)
        out[s:s + enc.input_ids.shape[0]] = cap.get()[0][:, -1]   # left-padded -> last is real
    return out

def comp_logps_tail(texts, ncs, grad, per_token=False, bs=4):
    """Policy log-probs over each text's completion tokens (the rightmost nc columns under left
    padding/truncation — same span convention as the pooled reads). logits_to_keep bounds the
    logits tensor to the longest completion in the chunk."""
    outs = []
    for s in range(0, len(texts), bs):
        chunk, nc = texts[s:s + bs], ncs[s:s + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
        T = enc.input_ids.shape[1]
        keep = min(max(nc) + 1, T)
        with (torch.enable_grad() if grad else torch.no_grad()):
            lsm = F.log_softmax(policy(**enc, logits_to_keep=keep).logits[:, :-1].float(), -1)
            for i in range(len(chunk)):
                n = min(nc[i], keep - 1)
                lp = lsm[i, keep - 1 - n:keep - 1].gather(-1, enc.input_ids[i, T - n:, None]).squeeze(-1)
                outs.append(lp if per_token else lp.sum())
    return outs if per_token else torch.stack(outs)

@torch.no_grad()
def rollout_phi(batch, gen, P, per_pos=False, bs=8):
    """FROZEN pooled read at L*: re-render each decoded completion (same sentinel logic as
    rollout_feats), capture the full L* residual sequence, mean-pool over completion tokens.
    per_pos=True also returns per-rollout running-mean Phi vectors (probe posterior of the mean
    over completion tokens 1..t) for dense credit, plus the re-rendered texts/ncs so the PG loss
    can be computed on the SAME token grid."""
    texts, ncs = [], []
    for i, x in enumerate(batch):
        for j in range(K):
            comp = tok.decode(gen[i * K + j, P:], skip_special_tokens=True)
            texts.append(render_full(x["prompt"], comp)); ncs.append(_n_comp(x["prompt"], comp))
    r_end = torch.zeros(len(texts), device=DEV)
    phis = [None] * len(texts)
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        T = enc.input_ids.shape[1]
        with policy.disable_adapter(), ResidualCapture([BLOCKS[LSTAR]]) as cap:
            policy(**enc)
        res = cap.get()[0].float()
        for i in range(enc.input_ids.shape[0]):
            n = min(ncs[s + i], T)
            seq = res[i, T - n:]
            cm = seq.cumsum(0) / torch.arange(1, n + 1, device=DEV).unsqueeze(1)
            zz = probe_reward(cm)
            r_end[s + i] = zz[-1]
            if per_pos: phis[s + i] = zz
    return (r_end, phis, texts, ncs) if per_pos else r_end

@torch.no_grad()
def gen_samples(n=6):
    """Greedy generations on fixed test prompts — the eyeball guard for preamble inflation."""
    policy.eval(); outs = []
    for x in test[:n]:
        enc = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=PLEN).to(DEV)
        policy.config.use_cache = True
        g = policy.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
        policy.config.use_cache = False
        outs.append(tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True)[:300])
    policy.train(); return outs

# ---- generative-replay floor (phase-8 §12, default ON for UF): random-token prompts, base
# continuations banked once, one-way floor relu(logp_base - logp_policy) sampled each step ----
replay_texts, replay_ncs, replay_ref = [], [], None
if REPLAY_N:
    rr = torch.Generator(device="cpu").manual_seed(SEED + 77)
    vocab = int(model.config.vocab_size)
    with torch.no_grad():
        for s in range(0, REPLAY_N, 8):
            k = min(8, REPLAY_N - s)
            rids = torch.randint(0, vocab, (k, 8), generator=rr).to(DEV)
            policy.config.use_cache = True
            with policy.disable_adapter():
                g = policy.generate(input_ids=rids, do_sample=True, temperature=1.0,
                                    max_new_tokens=REPLAY_MAXNEW, pad_token_id=tok.pad_token_id)
            policy.config.use_cache = False
            for i in range(k):
                ptxt = tok.decode(g[i, :8], skip_special_tokens=True)
                ftxt = tok.decode(g[i], skip_special_tokens=True)
                nc = max(1, len(tok(ftxt, add_special_tokens=False).input_ids)
                            - len(tok(ptxt, add_special_tokens=False).input_ids))
                replay_texts.append(ftxt); replay_ncs.append(nc)
    with torch.no_grad(), policy.disable_adapter():
        replay_ref = torch.cat([comp_logps_tail(replay_texts[s:s + 8], replay_ncs[s:s + 8], False)
                                for s in range(0, len(replay_texts), 8)])
    print(f"[replay] {len(replay_texts)} random-prompt base continuations banked", flush=True)

hist = dict(Lstar=LSTAR, mode=MODE, read=READ, shape_w=SHAPE_W, trunc_pen=TRUNC_PEN,
            replay_n=REPLAY_N, lr=opt.param_groups[0]["lr"], reward=[], evals=[], len=[])
rgen = random.Random(4242); policy.train()
u_prev = None   # lag-1 adaptive direction for pooled_margin
for step in range(STEPS):
    opt.zero_grad()
    if MODE in ("pooled_margin", "hybrid"):
        # ---- styc phase-8 winner: POLICY pooled activations, lag-1 mean-diff margin ----
        batch = rgen.sample(train, MARGIN_BS)
        mw = MCOEF if MODE == "hybrid" else 1.0
        texts, ncs = [], []
        for x in batch:
            for side in ("chosen", "rejected"):
                texts.append(render_full(x["prompt"], x[side])); ncs.append(_n_comp(x["prompt"], x[side]))
        d_all, proj_sum = [], 0.0
        u = u_prev   # None on step 0 -> first chunk pass computes with u_now of that chunk
        for s0 in range(0, len(texts), 4):      # 2 pairs per grad chunk, backward immediately
            enc = tok(texts[s0:s0 + 4], return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_LEN).to(DEV)
            T = enc.input_ids.shape[1]
            with ResidualCapture([BLOCKS[LSTAR]]) as cap:
                policy(**enc, logits_to_keep=1)   # residuals are what we need; skip the lm_head
            res = cap.get()[0].float()
            pooled = []
            for i in range(enc.input_ids.shape[0]):
                n = min(ncs[s0 + i], T)
                pooled.append(res[i, T - n:].mean(0))
            Fp = torch.stack(pooled)
            d = (Fp[0::2] - Fp[1::2]) / SD
            d_all.append(d.detach())
            if u is None:
                un = d.detach().mean(0); u = un / (un.norm() + 1e-6)
            proj = d.matmul(u)
            (mw * F.relu(M0 - proj).sum() / MARGIN_BS).backward()
            proj_sum += float(proj.sum().detach())
        with torch.no_grad():
            u_now = torch.cat(d_all).mean(0); u_prev = u_now / (u_now.norm() + 1e-6)
        (hist.setdefault("proj", []) if MODE == "hybrid" else hist["reward"]).append(proj_sum / MARGIN_BS)
    if MODE in ("rloo", "shaped", "hybrid"):
        # ---- sampling modes: rloo (v3 baseline), shaped (dense credit), hybrid's PG half ----
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
        # Read the probe at the SAME position/protocol Stage A calibrated it on. generate() left-pads
        # the prompt but RIGHT-pads completions, so gen[:, -1] is a <pad> for every rollout shorter
        # than the batch max (and <pad> id 128256 is an appended, untrained embedding). Re-rendering
        # the decoded completion through render_full puts every rollout back on the calibrated
        # positions, which also makes capped rollouts scoreable instead of discarded.
        phis = texts_r = ncs_r = None
        if READ == "mean":
            if MODE == "shaped":
                r, phis, texts_r, ncs_r = rollout_phi(batch, gen, P, per_pos=True)
            else:
                r = rollout_phi(batch, gen, P)
            r = r.detach()
        else:
            r = probe_reward(rollout_feats(batch, gen, P)).detach()
            if step < 3 or (step + 1) % 50 == 0:   # what the old gen[:, -1] read would have given
                with torch.no_grad(), policy.disable_adapter(), ResidualCapture([BLOCKS[LSTAR]]) as cap:
                    policy(input_ids=gen, attention_mask=attn)
                r_raw = probe_reward(cap.get()[0][:, -1]).detach()
                hist.setdefault("read_diag", []).append(dict(
                    step=step, frac_capped=float((n_new >= MAX_NEW).float().mean()),
                    r_rerender_mean=float(r.mean()), r_rerender_std=float(r.std()),
                    r_rawread_mean=float(r_raw.mean()), r_rawread_std=float(r_raw.std()),
                    corr=float(torch.corrcoef(torch.stack([r.float(), r_raw.float()]))[0, 1])))
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
        # Deferral/preamble guard (phase-8 §11/§13): a rollout that hits MAX_NEW with no eos is the
        # inflation exploit's signature — flat penalty in the REWARD (the exploit is the labeller's,
        # so the fix belongs in the reward, not the credit scheme).
        r = r - TRUNC_PEN * (n_new >= MAX_NEW).float()
        # With the re-render read, a capped rollout is scored at a real sentinel -- an abruptly-ending
        # response, a legitimate (low) reward rather than noise. Capped samples KEPT by default;
        # DROP_CAPPED=1 restores the v2 behaviour of masking them out.
        valid = (n_new < MAX_NEW) if int(E("DROP_CAPPED", 0)) else torch.ones_like(n_new, dtype=torch.bool)
        rg, vg = r.view(BATCH, K), valid.view(BATCH, K).float()
        cnt = vg.sum(1, keepdim=True)
        loo = (rg * vg).sum(1, keepdim=True) - rg * vg
        base = loo / (cnt - vg).clamp(min=1)
        adv = torch.where((vg > 0) & (cnt > 1.5), rg - base, torch.zeros_like(rg)).view(-1)
        hist.setdefault("trunc", []).append(float((n_new >= MAX_NEW).float().mean()))
        if MODE == "shaped":
            # per-token PG on the re-rendered completion — the SAME token grid Phi was computed on.
            # A_t = adv_seq + SHAPE_W*(Phi_end - Phi_t); loss = -mean_t(A_t * logp_t) per rollout.
            for s0 in range(0, BATCH * K, 2):
                idxs = list(range(s0, min(s0 + 2, BATCH * K)))
                lps = comp_logps_tail([texts_r[i] for i in idxs], [ncs_r[i] for i in idxs],
                                      True, per_token=True, bs=2)
                loss_c = None
                for i, lp in zip(idxs, lps):
                    Lp = min(len(lp), len(phis[i]))
                    if Lp == 0: continue
                    A_t = (adv[i] + SHAPE_W * (phis[i][-1] - phis[i][:Lp])).detach()
                    term = -(A_t * lp[:Lp]).sum() / Lp
                    loss_c = term if loss_c is None else loss_c + term
                if loss_c is not None:
                    (loss_c / (BATCH * K)).backward()
        else:
            for s0 in range(0, BATCH * K, 4):           # micro-batched backward, chunks of 4
                sl = slice(s0, min(s0 + 4, BATCH * K))
                if not adv[sl].abs().sum() > 0: continue
                li = F.log_softmax(policy(input_ids=gen[sl], attention_mask=attn[sl],
                                          logits_to_keep=keepg).logits[:, :-1].float(), -1)
                lp_i = (li.gather(-1, gen[sl, P:, None]).squeeze(-1) * tokmask[sl]).sum(1)
                (-(adv[sl] * lp_i / n_new[sl]).sum() / (BATCH * K)).backward()
        hist["reward"].append(float(rg.mean())); hist["len"].append(float(n_new.float().mean()))
    if ANCHOR > 0:  # DPOP hinge on the pair's chosen side
        for x in batch:
            pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.shape[1]
            lc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, True)
            with torch.no_grad(), policy.disable_adapter():
                rc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
            (ANCHOR * F.relu(rc - lc) / len(batch)).backward()
    if REPLAY_N:  # generative-replay floor: never lose likelihood on the base's off-task behaviour
        ridx = rgen.sample(range(len(replay_texts)), min(REPLAY_BS, len(replay_texts)))
        lp_r = comp_logps_tail([replay_texts[i] for i in ridx], [replay_ncs[i] for i in ridx],
                               True, bs=REPLAY_BS)
        (REPLAY_W * F.relu(replay_ref[ridx] - lp_r).mean()).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if (step + 1) % 10 == 0:
        msg = f"  step {step+1:4d}: reward {np.mean(hist['reward'][-10:]):.3f}"
        if hist["len"]: msg += f" len {np.mean(hist['len'][-10:]):.0f}"
        print(msg, flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(); ev["step"] = step + 1
        ev["samples"] = gen_samples(6)
        hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL { {k: v for k, v in ev.items() if k != 'samples'} }", flush=True)
        json.dump(hist, open(f"/workspace/uf_probe_rl_{TAG}_history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        policy.save_pretrained(f"/workspace/uf_probe_rl_{TAG}_ckpt{step+1}")
json.dump(hist, open(f"/workspace/uf_probe_rl_{TAG}_history.json", "w"), indent=1)
policy.save_pretrained(f"/workspace/uf_probe_rl_{TAG}_lora"); tok.save_pretrained(f"/workspace/uf_probe_rl_{TAG}_lora")
print("DONE", flush=True)
