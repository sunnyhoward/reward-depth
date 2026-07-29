#!/usr/bin/env python
"""UF margin-ONLY arm with an EWC leash: self-read margin backprop at L* (uf_hybrid.py's
margin_step_uf, no RLOO half) + a diagonal-Fisher parameter penalty standing in for the KL term.

WHY EWC-AS-KL: the RLOO arms leash drift with token-level KL-in-reward; the margin arm has no
likelihood terms, so phases 4-5 leashed it only via the co-trained RLOO half (or not at all --
"margin-only oscillates"). EWC's penalty 0.5*sum_i F_i dtheta_i^2 with F = the Fisher of the BASE
model's token log-likelihood is exactly the second-order expansion of KL(pi_base || pi_theta) in
parameter space -- i.e. it IS the KL term, expressed as a metric on parameter drift, computable
with zero likelihood terms in the training loop (Fisher is estimated once on the frozen base).
Contrast decision_probe.py's ANCH ||f - f_base||^2, which measures drift only at one layer/position
with an isotropic metric.

Stage A is uf_probe_rl.py's v3 recipe verbatim (length-matched IPW probe, centered read, left
truncation, shared feature cache). LoRA is restricted to blocks <= L* (layers_to_transform): the
margin loss reads h at L* so upper blocks get no gradient anyway, and this shrinks Fisher storage.

FISHER: diag empirical Fisher on the frozen base's OWN samples (temperature 1.0 continuations of
train prompts; sampling from the model, not the dataset, is what makes it the Fisher of pi_base).
Per sample the gradient of (1/n) sum_t log p(x_t|x_<t) w.r.t. the base weights of the LoRA target
modules in blocks <= L*, squared and averaged. Sequence-mean grads (not per-token) -- the standard
EWC-for-LM approximation; cross-position terms bias the diagonal but the metric shape survives.
Cached to /workspace (fp16) keyed by L* and N_FISH. Penalty per step, per module:
EWC * 0.5 * <F_m, (scaling * B_m A_m)^2> -- LoRA starts at dW=0 so theta* is the base itself.

DIAGNOSTICS (the point of the experiment):
  - evaluate(): implicit-DPO acc + dlp_chosen/dlp_rejected (displacement watch, phase-2 cure metric)
  - on-policy: rollouts on held-out prompts -> frozen-base probe reward at the re-render sentinel
    (does behaviour actually move under an honest read) + true KL/token vs base (does the
    parameter-space leash track token KL -- plot ewc_pen vs kl_tok_ho)
  - forging detector: mean margin z on eval pairs under the POLICY's read vs the FROZEN read.
    Self-read z inflating while frozen z and behaviour stay flat = the phase-2 forging mode.

Arms: EWC=0 -> no leash (the oscillation baseline). EWC>0 -> the experiment. ANCHOR>0 -> DPOP
hinge on the chosen side (likelihood-space leash, for comparison; default 0 = stays pure).

Env: UF_POOL=20000 N_PROBE=3000 STEPS=300 BATCH=8 MARGIN_LR=5e-5 EWC=1.0 ANCHOR=0.0
     N_FISH=128 FISH_MAX_NEW=200 MAX_NEW=200 MAX_LEN=1024 PLATEAU_TOL=0.01 EVAL_EVERY=25
     N_EVAL_PAIRS=48 N_EVAL_GEN=16 RL_PESS=0.5 UF_MATCH_LENGTH=1 UF_LEN_BUCKET=16 RUN_TAG=ewc1
Outputs: /workspace/uf_margin_ewc_{TAG}_history.json, _ckptN, _lora; Fisher cache
         /workspace/uf_fisher_L{L*}_N{N_FISH}.pt"""
import os, sys, json, random, hashlib, re
from itertools import islice
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
STEPS, BATCH, MB_PAIRS = int(E("STEPS", 300)), int(E("BATCH", 8)), int(E("MB_PAIRS", 2))
MLR, EWC, ANCHOR = float(E("MARGIN_LR", 5e-5)), float(E("EWC", 1.0)), float(E("ANCHOR", 0.0))
N_FISH, FISH_MAX_NEW = int(E("N_FISH", 128)), int(E("FISH_MAX_NEW", 200))
MAX_NEW, MAX_LEN, PLEN = int(E("MAX_NEW", 200)), int(E("MAX_LEN", 1024)), int(E("PROMPT_LEN", 512))
TOL, PESS = float(E("PLATEAU_TOL", 0.01)), float(E("RL_PESS", 0.5))
EVAL_EVERY, N_EVAL_PAIRS, N_EVAL_GEN = int(E("EVAL_EVERY", 25)), int(E("N_EVAL_PAIRS", 48)), int(E("N_EVAL_GEN", 16))
TAG = E("RUN_TAG", "ewc1")
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
print(f"[probe] plateau L*={LSTAR} (acc {acc[LSTAR]:.3f}, max {acc.max():.3f})", flush=True)

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

# ---- LoRA restricted to blocks <= L* ----
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                 layers_to_transform=list(range(LSTAR + 1)))
policy = get_peft_model(model, cfg); policy.config.use_cache = False
params = [p for p in policy.parameters() if p.requires_grad]
print(f"[lora] blocks <= {LSTAR}: {sum(p.numel() for p in params)/1e6:.1f}M trainable", flush=True)
opt = torch.optim.AdamW(params, lr=MLR)
def _blk(n):
    m = re.search(r"\.layers\.(\d+)\.", n); return int(m.group(1)) if m else -1
FMODS = {n: m for n, m in policy.named_modules()
         if hasattr(m, "lora_A") and "default" in m.lora_A and 0 <= _blk(n) <= LSTAR}

# ---- Fisher: diag empirical Fisher of the frozen base on its own samples ----
fishf = f"/workspace/uf_fisher_L{LSTAR}_N{N_FISH}.pt"
if EWC > 0:
    if os.path.exists(fishf):
        FISH = {n: t.to(DEV, torch.float32) for n, t in torch.load(fishf).items()}
        print(f"[fisher] loaded {fishf}", flush=True)
    else:
        print(f"[fisher] estimating on {N_FISH} base samples...", flush=True)
        for m in FMODS.values(): m.base_layer.weight.requires_grad_(True)
        FISH = {n: torch.zeros_like(m.base_layer.weight, dtype=torch.float32) for n, m in FMODS.items()}
        fp = random.Random(SEED + 99).sample(train, N_FISH)
        done = 0
        for s in range(0, N_FISH, 4):
            chunk = fp[s:s + 4]
            enc = tok([render_prompt(x["prompt"]) for x in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=PLEN).to(DEV)
            policy.config.use_cache = True
            with torch.no_grad(), policy.disable_adapter():
                gen = policy.generate(**enc, do_sample=True, temperature=1.0,
                                      max_new_tokens=FISH_MAX_NEW, pad_token_id=tok.pad_token_id)
            policy.config.use_cache = False
            P = enc.input_ids.shape[1]
            attn = (gen != tok.pad_token_id).long()
            keep = gen.shape[1] - P + 1
            for i in range(len(chunk)):   # per-SAMPLE backward: grad of the mean-token logp, squared
                n_new = int(attn[i, P:].sum().clamp(min=1))
                with policy.disable_adapter():
                    lg = policy(input_ids=gen[i:i + 1], attention_mask=attn[i:i + 1],
                                logits_to_keep=keep).logits[0, :-1].float()
                lp = (F.log_softmax(lg, -1).gather(-1, gen[i, P:, None]).squeeze(-1)
                      * attn[i, P:].bool()).sum() / n_new
                policy.zero_grad(set_to_none=True)
                lp.backward()
                for n, m in FMODS.items():
                    g = m.base_layer.weight.grad
                    if g is not None: FISH[n] += g.float().pow(2)
                done += 1
            print(f"  fisher {done}/{N_FISH}", flush=True)
        for n in FISH: FISH[n] /= done
        policy.zero_grad(set_to_none=True)
        for m in FMODS.values():
            m.base_layer.weight.requires_grad_(False); m.base_layer.weight.grad = None
        torch.save({n: t.half().cpu() for n, t in FISH.items()}, fishf)
        print(f"[fisher] saved {fishf}", flush=True)
    ftr = float(sum(t.sum() for t in FISH.values()))
    print(f"[fisher] trace {ftr:.3e} over {sum(t.numel() for t in FISH.values())/1e9:.2f}B params", flush=True)

def ewc_step():
    """Per-module penalty + immediate backward (keeps peak memory to one dW). Returns
    0.5 * sum <F, dW^2> -- the local estimate of KL(base||policy) in nats/token."""
    tot = 0.0
    for n, m in FMODS.items():
        A = m.lora_A["default"].weight; Bm = m.lora_B["default"].weight
        dW = (Bm @ A).float() * m.scaling["default"]
        pen = 0.5 * (FISH[n] * dW.pow(2)).sum()
        (EWC * pen).backward()
        tot += float(pen.detach())
    return tot

# ---- likelihood/eval plumbing (uf_hybrid.py) ----
def comp_logprob(text_full, plen, grad):
    ids = tok(text_full, return_tensors="pt", truncation=True, max_length=MAX_LEN + MAX_NEW).input_ids.to(DEV)
    plen = min(plen, ids.shape[1] - 1)
    with (torch.enable_grad() if grad else torch.no_grad()):
        keep = ids.shape[1] - plen + 1
        logits = policy(ids, logits_to_keep=keep).logits[0, :-1].float()
        return F.log_softmax(logits, -1).gather(-1, ids[0, plen:, None]).squeeze(-1).sum()

@torch.no_grad()
def pair_feats(pairs, use_policy):
    """L* last-token residuals of (chosen, rejected) full renders; policy or frozen read."""
    texts = [render_full(x["prompt"], x["chosen"]) for x in pairs] + \
            [render_full(x["prompt"], x["rejected"]) for x in pairs]
    fs = torch.zeros(len(texts), HID, device=DEV)
    for s in range(0, len(texts), 8):
        enc = tok(texts[s:s + 8], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(DEV)
        import contextlib
        with (contextlib.nullcontext() if use_policy else policy.disable_adapter()), \
             ResidualCapture([BLOCKS[LSTAR]]) as cap:
            policy(**enc, logits_to_keep=1)
        fs[s:s + enc.input_ids.shape[0]] = cap.get()[0][:, -1]
    B = len(pairs)
    return fs[:B], fs[B:]

@torch.no_grad()
def evaluate():
    policy.eval()
    ir = []
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
    fc_p, fr_p = pair_feats(test[:N_EVAL_PAIRS], use_policy=True)
    fc_b, fr_b = pair_feats(test[:N_EVAL_PAIRS], use_policy=False)
    ev["z_selfread"] = float(margin_z(fc_p, fr_p).mean())
    ev["z_frozen"] = float(margin_z(fc_b, fr_b).mean())
    # on-policy: held-out rollouts -> honest probe reward at the re-render sentinel + true KL/token
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

# ---- baseline eval + train ----
hist = dict(Lstar=LSTAR, ewc=EWC, anchor=ANCHOR, probe_acc=float(acc[LSTAR]),
            loss=[], mloss=[], ewc_pen=[], evals=[])
ev0 = evaluate(); ev0["step"] = 0; hist["evals"].append(ev0)
print(f"  step    0: EVAL {ev0}", flush=True)
rgen = random.Random(4242); policy.train()
for step in range(STEPS):
    batch = rgen.sample(train, BATCH)
    w_all = float(sum(x["w"] for x in batch))
    opt.zero_grad()
    mloss = 0.0
    # micro-batched by PAIRS (chosen/rejected must share a chunk: the loss reads difference
    # features). A full-batch backward at BATCH=8 is ~70GB of activations (16 seqs x 1024 x 32
    # layers, no checkpointing possible -- the margin must backprop THROUGH the captured residual,
    # and checkpointed forwards capture grad-less tensors); 2 pairs/chunk keeps it ~9GB.
    for s0 in range(0, BATCH, MB_PAIRS):
        sub = batch[s0:s0 + MB_PAIRS]
        w = torch.tensor([x["w"] for x in sub], device=DEV, dtype=torch.float32)
        texts = [render_full(x["prompt"], x["chosen"]) for x in sub] + \
                [render_full(x["prompt"], x["rejected"]) for x in sub]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
        with ResidualCapture([BLOCKS[LSTAR]]) as cap:
            policy(**enc, logits_to_keep=1)
        f = cap.get()[0][:, -1]
        zz = margin_z(f[:len(sub)], f[len(sub):])
        # IPW-weighted margin: same weights as the probe fit, so the loss can't be paid in length
        ml = -(torch.special.log_ndtr(zz) * w).sum() / w_all
        ml.backward()
        mloss += float(ml.detach())
    pen = ewc_step() if EWC > 0 else 0.0
    if ANCHOR > 0:   # optional DPOP comparison arm (likelihood-space leash)
        for x in batch:
            pl = tok(render_prompt(x["prompt"]), return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids.shape[1]
            lc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, True)
            with torch.no_grad(), policy.disable_adapter():
                rc = comp_logprob(render_full(x["prompt"], x["chosen"]), pl, False)
            (ANCHOR * F.relu(rc - lc) / BATCH).backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["mloss"].append(mloss); hist["ewc_pen"].append(pen)
    hist["loss"].append(mloss + EWC * pen)
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: mloss {np.mean(hist['mloss'][-10:]):.4f} "
              f"ewc_pen {np.mean(hist['ewc_pen'][-10:]):.5f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(); ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL {ev}", flush=True)
        json.dump(hist, open(f"/workspace/uf_margin_ewc_{TAG}_history.json", "w"), indent=1)
    if (step + 1) % 100 == 0:
        policy.save_pretrained(f"/workspace/uf_margin_ewc_{TAG}_ckpt{step+1}")
json.dump(hist, open(f"/workspace/uf_margin_ewc_{TAG}_history.json", "w"), indent=1)
policy.save_pretrained(f"/workspace/uf_margin_ewc_{TAG}_lora")
print("DONE", flush=True)
