#!/usr/bin/env python
"""Decision-position probes + activation-only preference training (the untested self-read corner:
methods.md par.3 "move the read upstream" x "LoRA <= L", + adaptive head, + on-policy labels).

Everything in phases 1-3 read the COMPLETION-END residual, where a write is causally downstream of
the decision (faking free, methods 2.2) and where fixed teacher-forced pairs make behaviour change
unable to reduce the loss (2.3). This arm moves the read to the DECISION POSITION -- the last
prompt token, whose residual feeds directly into the answer logits -- which forces the probe's
target to change: with causal attention the decision-position state does not depend on the
completion, so a chosen-vs-rejected probe is degenerate there. The probe instead reads the
IMPENDING ACTION: P(model emits 'A' | h_dec), fit on the base model's own answers.

Training (arm "adaptive", the real one):
  - forward PROMPT ONLY; read h_dec at layer L; loss = -log P_head(wrong letter | h_dec)
  - backprop through LoRA restricted to blocks <= L; blocks > L frozen
  - NO likelihood terms anywhere: no REINFORCE, no DPO, no KL. Signal reaches behaviour through
    the representation or not at all.
  - head refit every REFIT_EVERY steps on the CURRENT policy's h_dec with labels = the policy's
    own sampled answers (previous posterior as prior => forged components are absorbed; honest
    flipping is a fixed point, because then the refit labels agree with the pushed class).
Arm "frozen": identical but the head never refits -- the 2.2-style forging control.

Escapes closed vs phases 1-2: positional (read upstream of the decision), directional (an
action-predictive probe's mu cannot be orthogonal to what the upper blocks read -- its accuracy
comes from behaviour-driving components; contrast cos(mu, W_A - W_B) = -0.003 at the top),
cancellation-from-above (LoRA <= L), static forging (refit), and the 2.3 stall (on-policy labels).

Stage 1 sweeps decision-position decodability of the relational target (corr == 'A'; letter-
randomized so letter-frequency cheats score 0.5) across all layers -> pick L* at the elbow.
Features are CENTERED (phase-3 lesson: absolute reads without centering inflate s2).

Env: MODEL=Qwen/Qwen2.5-3B N_TRAIN=1000 N_EVAL=300 ARM=adaptive|frozen L_OVERRIDE=-1
     STEPS=300 BATCH=32 LR=1e-4 REFIT_EVERY=10 REFIT_N=64 REFIT_STEPS=10 MIN_SIGMA=0.05
     PLATEAU_TOL=0.02 EVAL_EVERY=50 SEED=0
Saves: /workspace/decision_probe_curve.json, /workspace/decision_{arm}_history.json,
       /workspace/decision_{arm}_lora"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, "/workspace/reward-depth")
from helpers import (load_model, build_data, ResidualCapture, train_bayes_head, RewardHead,
                     add_lora, reset_lora, eval_all, greedy, LOG_NDTR, _wl)

E = os.environ.get
MODEL = E("MODEL", "Qwen/Qwen2.5-3B")
N_TRAIN, N_EVAL = int(E("N_TRAIN", 1000)), int(E("N_EVAL", 300))
ARM = E("ARM", "adaptive")
L_OVR = int(E("L_OVERRIDE", -1))
STEPS, BATCH, LR = int(E("STEPS", 300)), int(E("BATCH", 32)), float(E("LR", 1e-4))
REFIT_EVERY, REFIT_N, REFIT_STEPS = int(E("REFIT_EVERY", 10)), int(E("REFIT_N", 96)), int(E("REFIT_STEPS", 10))
MIN_SIGMA = float(E("MIN_SIGMA", 0.05))
# v2 guards (from the v1 postmortem: letter attractor -> refit-label collapse -> dead head -> off-menu
# drift with no leash). All three live in ACTIVATION space -- no likelihood terms added.
ANCH = float(E("ANCH", 0.1))    # ||f - f_base||^2 anchor to the frozen-base decision state: cheap for
                                # a targeted low-rank move (the honest edit), expensive for broad drift
ZSYM = float(E("ZSYM", 0.1))    # (mean_batch z)^2: an honest relational flip keeps batch-mean z ~ 0
                                # (wrong letters are ~50/50 A/B); a letter policy must displace it
BAL_MIN = int(E("BAL_MIN", 8))  # refit: subsample to 50/50 A/B labels; skip if minority < BAL_MIN.
                                # Honest flipping keeps sampled labels balanced (wrong letters are
                                # 50/50), so this starves ONLY the letter attractor's feedback loop
REFIT_MODE = E("REFIT_MODE", "filter")  # "filter": incremental Bayes update (prev posterior = prior)
                                # -- v2 showed the policy OUTRUNS it (rotation runaway 27->50deg,
                                # train-forged features, behaviour reverts). "buffer": from-scratch
                                # refit on ALL accumulated (state, sampled-action) pairs -- every past
                                # forged direction stays in the buffer labelled with the TRUE action,
                                # so old fakes can't be revisited; honest flipping remains a fixed
                                # point (the Libon retrain-from-scratch variant, self-labelled)
TOL, EVAL_EVERY, SEED = float(E("PLATEAU_TOL", 0.02)), int(E("EVAL_EVERY", 50)), int(E("SEED", 0))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ctx = load_model(MODEL)
d = build_data(seed=SEED, n_train=N_TRAIN, n_eval=N_EVAL, tok=ctx.tok)
print(f"[data] {len(d.train_qs)} train qs | {len(d.eval_qs)} eval qs | model {MODEL} "
      f"({ctx.n_layers} layers, hid {ctx.hid})", flush=True)

# ---- decision-position features: h at the LAST PROMPT TOKEN (the "Answer:" position), all layers.
# Left padding => h[:, -1]. Prompt-only forward: the completion never enters the encoder.
@torch.no_grad()
def cache_decpos(qs, use_policy=False, bs=32, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        return np.load(cache_file)["X"]
    m = ctx.policy if use_policy else ctx.model
    X = np.zeros((len(qs), ctx.n_layers, ctx.hid), np.float32)
    for s in range(0, len(qs), bs):
        enc = ctx.tok([d.render_ab(q) for q in qs[s:s + bs]], return_tensors="pt", padding=True).to(ctx.device)
        with ResidualCapture(ctx.blocks) as cap:
            m(**enc)
        buf = cap.get()
        for li in range(ctx.n_layers):
            X[s:s + len(enc.input_ids), li] = buf[li][:, -1].float().cpu().numpy()
    if cache_file: np.savez(cache_file, X=X)
    return X

# ---- Stage 1: per-layer decodability of the relational target (corr == 'A') at the decision position
t_tr = np.array([1.0 if q["corr"] == "A" else -1.0 for q in d.train_qs], np.float32)
t_te = np.array([1.0 if q["corr"] == "A" else -1.0 for q in d.eval_qs], np.float32)
curvef = "/workspace/decision_probe_curve.json"
Xtr = cache_decpos(d.train_qs, cache_file="/workspace/decision_feats_train.npz")
Xte = cache_decpos(d.eval_qs, cache_file="/workspace/decision_feats_eval.npz")
acc = np.zeros(ctx.n_layers); heads = {}; stats = {}
for li in range(ctx.n_layers):
    mn, sd = Xtr[:, li].mean(0), Xtr[:, li].std(0) + 1e-6          # CENTERED absolute features
    ftr, fte = (Xtr[:, li] - mn) / sd, (Xte[:, li] - mn) / sd
    # pass UNSIGNED features: train_bayes_head multiplies by the target internally (the pairwise
    # callers pre-sign so t^2 cancels; here t is a real label, pre-signing would degenerate the fit)
    a, h, e = train_bayes_head(ftr, t_tr, fte, t_te)
    acc[li], heads[li], stats[li] = a, (h, sd), (mn, e)
    print(f"  L{li:2d} acc={a:.3f} elbo={e:+.0f}", flush=True)
LSTAR = L_OVR if L_OVR >= 0 else int(next(li for li in range(ctx.n_layers) if acc[li] >= acc.max() - TOL))
print(f"[probe] decision-position plateau L*={LSTAR} (acc {acc[LSTAR]:.3f}, max {acc.max():.3f})", flush=True)
json.dump(dict(layer_acc=acc.tolist(), Lstar=LSTAR, target="corr==A", position="last prompt token"),
          open(curvef, "w"))
if int(E("SWEEP_ONLY", 0)): sys.exit(0)

# ---- Stage 2: activation-only wrongness training at L* ----
MEAN = torch.tensor(stats[LSTAR][0], device=ctx.device)
fh = RewardHead(ctx, heads, LSTAR)              # g((f - MEAN)) == centered read; sf = sd
MU0 = fh.mu.clone()

# ---- ARM=hybrid: a SECOND head (outcome-judge) + exact-expectation candidate REINFORCE > L ----
# The decision-position head is completion-blind (same prompt -> same h_dec -> same reward for every
# candidate), so the >L half needs a completion-reading judge: the phase-1 pairwise probe at the
# answer end, frozen-base read. Rewards r(letter|q) = ndtr(g2(f_right - f_letter)) are then a fixed
# function of (question, letter) -- precomputed once for all train questions. The >L objective is
# the exact expectation J = p(A|x) r_A + p(B|x) r_B over the two candidates (phase-2's working
# ingredient), maximized w.r.t. blocks > L only (the <=L contribution of its gradient is masked).
RL_COEF = float(E("RL_COEF", 1.0))
MCOEF = float(E("MCOEF", 1.0))    # weight on the margin (plan-reader) term; 0 => J-only ablation
ID_A = ctx.tok(" A", add_special_tokens=False).input_ids[0]
ID_B = ctx.tok(" B", add_special_tokens=False).input_ids[0]
if ARM == "hybrid":
    from helpers import cache_pairend
    ab_tr = [p for p in d.train_pairs if p["fmt"] == "ab"]
    ab_te = [p for p in d.eval_pairs if p["fmt"] == "ab"]
    Xw_tr, Xr_tr = cache_pairend(ctx, ab_tr, cache_file="/workspace/decision_pairend_train.npz")
    Xw_te, Xr_te = cache_pairend(ctx, ab_te, cache_file="/workspace/decision_pairend_eval.npz")
    sd2 = np.concatenate([Xw_tr[:, LSTAR], Xr_tr[:, LSTAR]]).std(0) + 1e-6
    dtr2 = (Xr_tr[:, LSTAR] - Xw_tr[:, LSTAR]) / sd2
    dte2 = (Xr_te[:, LSTAR] - Xw_te[:, LSTAR]) / sd2
    acc2, head2, _ = train_bayes_head(dtr2, np.ones(len(dtr2), np.float32),
                                      dte2, np.ones(len(dte2), np.float32))
    print(f"[judge] completion-end pairwise head at L{LSTAR}: held-out acc {acc2:.3f}", flush=True)
    heads2 = {LSTAR: (head2, sd2)}
    judge = RewardHead(ctx, heads2, LSTAR)
    # per-question candidate rewards on the FROZEN base: r(letter) = ndtr(g2(f_right - f_letter))
    # (0.5 for the right letter by construction; ~high for the wrong one)
    Q_R = {}
    with torch.no_grad():
        for s in range(0, len(d.train_qs), 24):
            chunk = d.train_qs[s:s + 24]
            texts = [d.render_ab(q) + " A" for q in chunk] + [d.render_ab(q) + " B" for q in chunk]
            enc = ctx.tok(texts, return_tensors="pt", padding=True).to(ctx.device)
            with ResidualCapture([ctx.blocks[LSTAR]]) as cap:
                ctx.model(**enc, logits_to_keep=1)
            fT = cap.get()[0][:, -1]
            fA, fB = fT[:len(chunk)], fT[len(chunk):]
            for i, q in enumerate(chunk):
                f_right = fA[i] if q["corr"] == "A" else fB[i]
                rA = float(torch.special.ndtr(judge.g((f_right - fA[i])[None])))
                rB = float(torch.special.ndtr(judge.g((f_right - fB[i])[None])))
                Q_R[q["q"]] = (rA, rB)
    wl_r = [Q_R[q["q"]]["AB".index(_wl(q))] for q in d.train_qs]
    print(f"[judge] mean r(wrong letter) {np.mean(wl_r):.3f} | mean r(right letter) "
          f"{np.mean([Q_R[q['q']]['AB'.index(q['corr'])] for q in d.train_qs]):.3f}", flush=True)

add_lora(ctx, r=16)
TRAIN_BLOCKS = None if ARM == "hybrid" else set(range(LSTAR + 1))
params = reset_lora(ctx, seed=SEED, trainable_blocks=TRAIN_BLOCKS)
low_params = [p for _, p, b in ctx.lora_params if p.requires_grad and b <= LSTAR]
n_par = sum(p.numel() for p in params)
print(f"[lora] trainable: {len(params)} tensors, {n_par/1e6:.1f}M params "
      f"({'all blocks; margin masked to <=' + str(LSTAR) if ARM == 'hybrid' else 'blocks <= ' + str(LSTAR) + ' only'})", flush=True)
opt = torch.optim.AdamW(params, lr=LR)

def dec_feats(qs, grad, ret_logits=False):
    """h_dec at L* through the POLICY (adapter on), centered. grad=True keeps the graph.
    ret_logits: also return the last-position logits from the SAME forward (for the hybrid's J)."""
    enc = ctx.tok([d.render_ab(q) for q in qs], return_tensors="pt", padding=True).to(ctx.device)
    with (torch.enable_grad() if grad else torch.no_grad()):
        with ResidualCapture([ctx.blocks[LSTAR]]) as cap:
            out = ctx.policy(**enc, logits_to_keep=1)
        f = cap.get()[0][:, -1].float() - MEAN
        return (f, out.logits[:, -1]) if ret_logits else f

@torch.no_grad()
def fracA_and_flip(qs, n=150):
    o = greedy(ctx, [d.render_ab(q) for q in qs[:n]], 2)
    return (float(np.mean([x[:1] == "A" for x in o])),
            float(np.mean([x[:1] == _wl(q) for x, q in zip(o, qs)])),
            float(np.mean([x[:1] == q["corr"] for x, q in zip(o, qs)])))

ARM_TAG = ARM + ("_buffer" if (REFIT_MODE == "buffer" and ARM != "frozen") else "") + ("_jonly" if (ARM == "hybrid" and MCOEF == 0) else "")
hist = dict(arm=ARM_TAG, Lstar=LSTAR, probe_acc=float(acc[LSTAR]), loss=[], evals=[], refit=[])
buf_f, buf_t = [], []
rgen = random.Random(SEED + 7); ctx.policy.train()
for step in range(STEPS):
    qs = rgen.sample(d.train_qs, BATCH)
    t_wrong = torch.tensor([1.0 if _wl(q) == "A" else -1.0 for q in qs], device=ctx.device)
    with torch.no_grad(), ctx.policy.disable_adapter():
        f_base = dec_feats(qs, grad=False)       # frozen-base decision states (anchor target)
    if ARM == "hybrid":
        f, lg = dec_feats(qs, grad=True, ret_logits=True)
    else:
        f = dec_feats(qs, grad=True)
    z = fh.g(f)                                  # head FROZEN within the backbone step (no collusion)
    loss = (-MCOEF * LOG_NDTR(z * t_wrong).mean()  # want: state reads as "about to emit the wrong letter"
            + ANCH * (f - f_base).pow(2).mean()  # activation-space leash (no likelihood term)
            + ZSYM * z.mean().pow(2))            # letter-policy repellent (honest flip keeps mean z ~ 0)
    opt.zero_grad(); loss.backward(retain_graph=(ARM == "hybrid"))
    if ARM == "hybrid":
        # exact-expectation candidate REINFORCE on blocks > L: maximize J = p(A)r_A + p(B)r_B.
        # The margin backward above reaches only <= L (its loss reads h at L*); J's backward reaches
        # every block, so its <= L contribution is masked by restoring the margin-only grads there.
        g_low = [(p_, p_.grad.clone() if p_.grad is not None else None) for p_ in low_params]
        pv = torch.softmax(lg.float(), -1)
        rA = torch.tensor([Q_R[q["q"]][0] for q in qs], device=ctx.device)
        rB = torch.tensor([Q_R[q["q"]][1] for q in qs], device=ctx.device)
        J = (pv[:, ID_A] * rA + pv[:, ID_B] * rB).mean()
        (-RL_COEF * J).backward()
        for p_, g in g_low:
            p_.grad = g
        hist.setdefault("J", []).append(float(J.detach()))
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(float(loss.detach()))
    if ARM == "adaptive" and (step + 1) % REFIT_EVERY == 0:
        rqs = rgen.sample(d.train_qs, REFIT_N)
        outs = greedy(ctx, [d.render_ab(q) for q in rqs], 2, temp=1.0)   # on-policy sampled labels
        keep = [(q, o) for q, o in zip(rqs, outs) if o[:1] in ("A", "B")]
        A = [x for x in keep if x[1][:1] == "A"]; B = [x for x in keep if x[1][:1] == "B"]
        k = min(len(A), len(B))
        keep = random.Random(SEED + step).sample(A, k) + random.Random(SEED - step - 1).sample(B, k)
        if k >= BAL_MIN:                          # balanced 50/50 refit; letter collapse -> refit skipped
            t_emit = torch.tensor([1.0 if o[:1] == "A" else -1.0 for _, o in keep], device=ctx.device)
            fr = dec_feats([q for q, _ in keep], grad=False)
            if REFIT_MODE == "buffer":
                # from-scratch refit on ALL accumulated (state, action) pairs: past forged directions
                # stay in the buffer labelled truthfully, so the policy can't outrun or revisit them
                buf_f.append(fr.cpu()); buf_t.append(t_emit.cpu())
                Fb = torch.cat(buf_f).numpy(); Tb = torch.cat(buf_t).numpy()
                perm = np.random.RandomState(step).permutation(len(Tb)); ntr = int(0.85 * len(Tb))
                bacc, hnew, _ = train_bayes_head(Fb[perm[:ntr]], Tb[perm[:ntr]],
                                                 Fb[perm[ntr:]], Tb[perm[ntr:]], epochs=80, patience=10)
                with torch.no_grad():
                    fh.mu.copy_(hnew.mu.detach().to(ctx.device))
                    fh.rho.copy_(hnew.rho.detach().to(ctx.device))
                    if MIN_SIGMA > 0: fh.rho.clamp_(min=float(np.log(np.expm1(MIN_SIGMA))))
                hist.setdefault("buffer", []).append(dict(step=step + 1, size=len(Tb), val_acc=float(bacc)))
            else:
                fh.filter_round(fr, t_emit, steps=REFIT_STEPS, min_sigma=MIN_SIGMA)
            with torch.no_grad():
                rot = float(torch.rad2deg(torch.arccos(torch.clamp(
                    F.cosine_similarity(fh.mu, MU0, dim=0), -1, 1))))
            hist["refit"].append(dict(step=step + 1, n=len(keep), rot_deg=rot,
                                      fracA_raw=len(A) / max(len(A) + len(B), 1)))  # PRE-balance skew
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-10:]):.4f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0 or step == 0:
        fa, fl, ab = fracA_and_flip(d.eval_qs)
        with torch.no_grad():
            endorse = float(torch.special.ndtr(fh.g(dec_feats(d.eval_qs[:96], False))
                            * torch.tensor([1.0 if _wl(q) == "A" else -1.0 for q in d.eval_qs[:96]],
                                           device=ctx.device)).mean())
        ev = eval_all(ctx, d) if (step + 1) % (2 * EVAL_EVERY) == 0 else {}
        ev.update(step=step + 1, fracA=fa, flip=fl, ab=ab, endorse_wrong=endorse)
        hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL flip {fl:.3f} ab {ab:.3f} fracA {fa:.2f} "
              f"P(wrong|h) {endorse:.3f}" + (f" | full {ev.get('free_flip','-')}" if len(ev) > 6 else ""),
              flush=True)
        json.dump(hist, open(f"/workspace/decision_{ARM_TAG}_history.json", "w"), indent=1)
json.dump(hist, open(f"/workspace/decision_{ARM_TAG}_history.json", "w"), indent=1)
ctx.policy.save_pretrained(f"/workspace/decision_{ARM_TAG}_lora")
print("DONE", flush=True)
