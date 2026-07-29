#!/usr/bin/env python
"""Stage 2 of the DPO relaxation, on the content-choice (cc) testbed: move the preference margin
from token space into ACTIVATION space at layer L*, leashed by the same replay-K-FAC-EWC anchor
that replaced DPO's reference prior in stage 1 (pythia/stage1.py). Qwen2.5-3B, wrongness
preference (chosen = wrong entity), oracle-verified behaviour.

Why here: the cc testbed keeps phase-1/2/4's exact verification (entity oracle, know-preservation,
OOD) without the degenerate letter output space, and its instruments separate the three failure
modes that killed earlier activation training: FORGING (self-read z moves, frozen z and behaviour
don't), position policy (first-opt rate), and open-vocabulary displacement (offmenu rate).

ARMS (env ARM):
  probe     margin through the policy's own L* activations against the FROZEN Stage-A probe
            (helpers.margin_step) -- the phase-1-style coupling, now with a parameter-space leash
  meandiff  supervisor's variant: per-batch mean activation difference (wrong - right under the
            CURRENT policy), detached, as the push direction -- "probe refit every step" limit
Both: LoRA on ATTENTION projections only (anchor coverage = LoRA coverage; no unleashed modules),
blocks <= L* (phase-4's cancellation-from-above guard). EWC_KL=0 gives the unleashed control.

Anchor: /workspace/replay/qwen3b/kfac (attention factors, cc-prompt-conditioned replay).
STAGE=calib runs the calibration protocol first (required before training, as stage 1).

Env: MODEL=Qwen/Qwen2.5-3B ARM=probe STAGE=train|calib EWC_KL=1.0 MCOEF=1.0 STEPS=300 BATCH=16
     LR=1e-4 L_OVERRIDE=-1 PLATEAU_TOL=0.02 EVAL_EVERY=25 N_CAL=12 SEED=0 RUN_TAG=<arm>
Outputs: /workspace/cc_stage2_{TAG}_history.json"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import (load_model, build_data, cache_pairend, fit_probes, RewardHead, add_lora,
                     reset_lora, greedy, ResidualCapture, margin_step, LOG_NDTR)
from replay_kfac_ewc import FactorBundle, KFACEWC, load_replay, mean_forward_kl, fit_calibration
from peft import LoraConfig, get_peft_model

E = os.environ.get
MODEL, ARM, STAGE = E("MODEL", "Qwen/Qwen2.5-3B"), E("ARM", "probe"), E("STAGE", "train")
EWC_KL, MCOEF, LR = float(E("EWC_KL", 1.0)), float(E("MCOEF", 1.0)), float(E("LR", 1e-4))
STEPS, BATCH = int(E("STEPS", 300)), int(E("BATCH", 16))
L_OVR, TOL = int(E("L_OVERRIDE", -1)), float(E("PLATEAU_TOL", 0.02))
EVAL_EVERY, N_CAL, SEED = int(E("EVAL_EVERY", 25)), int(E("N_CAL", 12)), int(E("SEED", 0))
REPLAY = E("REPLAY", "/workspace/replay/qwen3b/library.jsonl")
FACTORS = E("FACTORS", "/workspace/replay/qwen3b/kfac")
TAG = E("RUN_TAG", ARM)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ctx = load_model(MODEL)
d = build_data(seed=0, n_train=1000, n_eval=300, n_transfer=150, formats=("cc",), tok=ctx.tok)
cc_tr = [p for p in d.train_pairs if p["fmt"] == "cc"]
cc_te = [p for p in d.eval_pairs if p["fmt"] == "cc"]
print(f"[data] {len(cc_tr)} train pairs | {len(cc_te)} eval pairs", flush=True)

# ---- Stage A: per-layer probes on cc answer-end residuals (frozen base) ----
Xw_tr, Xr_tr = cache_pairend(ctx, cc_tr, cache_file="/workspace/cc_pairend_train.npz")
Xw_te, Xr_te = cache_pairend(ctx, cc_te, cache_file="/workspace/cc_pairend_eval.npz")
acc, elbo, heads = fit_probes(ctx, d, Xw_tr, Xr_tr, Xw_te, Xr_te,
                              cache_file="/workspace/cc_probes.pt")
LSTAR = L_OVR if L_OVR >= 0 else int(next(li for li in range(ctx.n_layers)
                                          if acc[li] >= np.nanmax(acc) - TOL))
print(f"[probe] plateau L*={LSTAR} (acc {acc[LSTAR]:.3f}, max {np.nanmax(acc):.3f})", flush=True)
fh = RewardHead(ctx, heads, LSTAR)
json.dump(dict(layer_acc=[None if np.isnan(a) else float(a) for a in acc], Lstar=LSTAR),
          open("/workspace/cc_probe_curve.json", "w"))

# ---- LoRA: attention only (= anchor coverage), ALL blocks; per-arm masking picks <= L ----
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
ctx.policy = get_peft_model(ctx.model, cfg); ctx.policy.config.use_cache = False
import re as _re
ctx.lora_params = [(n, p, int(_re.search(r"\.layers\.(\d+)\.", n).group(1)))
                   for n, p in ctx.policy.named_parameters() if "lora_" in n]
params = [p for _, p, _ in ctx.lora_params]

FULL_FACTORS = FactorBundle.load(FACTORS, device=ctx.device)

def make_anchor(L):
    """Anchor over exactly the trainable blocks <= L (frozen modules have zero delta). Shallow-
    copies the bundle so the shared FULL_FACTORS stays intact across arms of a depth sweep.
    Depth-sweep caveat: the calibration RATIO was fit on the <=20 configuration and is reused
    across L (slope ~= 1 makes the scale insensitive to the module subset)."""
    import copy
    a = KFACEWC(FULL_FACTORS, coefficient=1.0)
    fb = copy.copy(FULL_FACTORS)
    fb.factors = {n: f for n, f in FULL_FACTORS.factors.items()
                  if int(_re.search(r"layers\.(\d+)\.", n).group(1)) <= L}
    a.factors = fb
    return a

PLATEAU_L = LSTAR
anchor = make_anchor(LSTAR)

def model_ref():
    class _Ref(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._dev = torch.nn.Parameter(torch.zeros(1, device=ctx.device), requires_grad=False)
        def forward(self, *a, **k):
            with ctx.policy.disable_adapter():
                return ctx.policy(*a, **k)
    return _Ref()

calibf = os.path.join(FACTORS, "calibration.json")
if STAGE == "calib":
    heldout = load_replay(REPLAY, split="heldout")
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    preds, meas = [], []
    for i in range(N_CAL):
        scale = 10 ** (-3 + 2.0 * i / max(N_CAL - 1, 1))
        with torch.no_grad():
            for _, p, _ in ctx.lora_params:
                p.copy_(torch.randn(p.shape, generator=rng, dtype=torch.float32).to(p.dtype) * scale)
        with torch.no_grad():
            pen = float(anchor.penalty_from_peft(ctx.policy, adapter_name="default"))
        kl = mean_forward_kl(model_ref(), ctx.policy, heldout, pad_token_id=ctx.tok.pad_token_id,
                             batch_size=2, max_positions=int(E("CAL_MAX_POS", 20000)))
        preds.append(pen); meas.append(kl)
        print(f"  cal {i+1}/{N_CAL}: scale {scale:.2e} pen {pen:.3e} kl {kl:.3e}", flush=True)
    rep = fit_calibration(preds, meas)
    pairs = [(p_, k_) for p_, k_ in zip(preds, meas) if 0.06 < k_ < 5.0]
    loc = fit_calibration([p_ for p_, _ in pairs], [k_ for _, k_ in pairs]) if len(pairs) >= 3 else rep
    json.dump(dict(predicted=preds, measured=meas, slope=loc.log_log_slope,
                   rank_corr=loc.spearman_correlation, ratio=loc.kl_per_penalty_geometric_mean,
                   global_slope=rep.log_log_slope), open(calibf, "w"), indent=1)
    print(f"[calib] local slope {loc.log_log_slope:.3f} ratio {loc.kl_per_penalty_geometric_mean:.3e}",
          flush=True)
    sys.exit(0)

if not os.path.exists(calibf):
    sys.exit(f"refusing to train without {calibf} -- run STAGE=calib first")
cal = json.load(open(calibf)); RATIO = float(cal["ratio"])
print(f"[calib] ratio {RATIO:.3e} slope {cal['slope']:.2f}", flush=True)
params = reset_lora(ctx, seed=SEED)   # calib leaves perturbed weights in-process; reset to delta 0
opt = torch.optim.AdamW(params, lr=LR)

def matches(gen, ent):
    g = gen.strip().lstrip("$")
    return g[: len(ent)] == ent and (len(g) == len(ent) or not g[len(ent)].isalnum())

@torch.no_grad()
def oracle(qs, n=150):
    outs = greedy(ctx, [d.render_cc(q) for q in qs[:n]], 8)
    t = float(np.mean([matches(o, q["t"]) for o, q in zip(outs, qs)]))
    f = float(np.mean([matches(o, q["f"]) for o, q in zip(outs, qs)]))
    fo = [(o, q) for o, q in zip(outs, qs) if q.get("cc_first")]
    first = float(np.mean([matches(o, q["cc_first"]) for o, q in fo])) if fo else None
    return dict(correct=t, wrong=f, offmenu=round(1 - t - f, 4), first_opt=first)

@torch.no_grad()
def pair_z(pairs, use_policy):
    import contextlib
    texts = [p["prompt"] + p["wrong"] for p in pairs] + [p["prompt"] + p["right"] for p in pairs]
    fs = torch.zeros(len(texts), ctx.hid, device=ctx.device)
    for s in range(0, len(texts), 16):
        enc = ctx.tok(texts[s:s + 16], return_tensors="pt", padding=True).to(ctx.device)
        with (contextlib.nullcontext() if use_policy else ctx.policy.disable_adapter()), \
             ResidualCapture([ctx.blocks[LSTAR]]) as cap:
            ctx.policy(**enc, logits_to_keep=1)
        fs[s:s + enc.input_ids.shape[0]] = cap.get()[0][:, -1]
    B = len(pairs)
    return float(fh.g(fs[B:] - fs[:B]).mean())   # g>0 = right-preferred; install drives it NEGATIVE

ORACLE_EVERY, ORACLE_N = int(E("ORACLE_EVERY", 100)), int(E("ORACLE_N", 100))

@torch.no_grad()
def evaluate(full):
    """Two-tier: cheap instruments every eval; the oracle generation battery only when full."""
    ctx.policy.eval()
    ev = {}
    if full:
        ev.update(flip=oracle(d.eval_qs, ORACLE_N), know=oracle(d.know_qs, ORACLE_N),
                  ood_sum=oracle(d.ood_sets["sum"], ORACLE_N))
    ev["z_selfread"] = pair_z(cc_te[:64], True)
    ev["z_frozen"] = pair_z(cc_te[:64], False)
    ev["ewc_pen"] = float(anchor.penalty_from_peft(ctx.policy, adapter_name="default"))
    ev["ewc_kl_pred"] = ev["ewc_pen"] * RATIO
    ho = getattr(evaluate, "_ho", None)
    if ho is None:
        ho = evaluate._ho = load_replay(REPLAY, split="heldout")[:32]
    ev["replay_kl"] = mean_forward_kl(model_ref(), ctx.policy, ho,
                                      pad_token_id=ctx.tok.pad_token_id, batch_size=2,
                                      max_positions=4000)
    ctx.policy.train()
    return ev

@torch.no_grad()
def _batch_meandiff(batch):
    texts = [p["prompt"] + p["wrong"] for p in batch] + [p["prompt"] + p["right"] for p in batch]
    enc = ctx.tok(texts, return_tensors="pt", padding=True).to(ctx.device)
    with ResidualCapture([ctx.blocks[LSTAR]]) as cap:
        ctx.policy(**enc, logits_to_keep=1)
    f = cap.get()[0][:, -1]
    B = len(batch)
    return (f[:B] - f[B:]).float().mean(0)

def run_arm(arm, ewc_kl, tag, seed, md_k=1, L=None):
    """md_k: lag window for the meandiff direction -- mean over the last md_k batches' diff
    vectors (1 = zero-lag per-batch, the supervisor's form). md_k=0 = direction estimated at
    init and FROZEN: the within-family frozen control for the lag-spectrum ablation.
    L: attach layer for a depth sweep (read layer, LoRA mask <= L, anchor <= L); default L*."""
    from collections import deque
    global params, LSTAR, fh, anchor
    LSTAR = L if L is not None else PLATEAU_L
    fh = RewardHead(ctx, heads, LSTAR)
    # hybrid/srloo train ALL blocks (margin owns <= L*, J/RLOO owns > L*), full-stack anchor
    anchor = make_anchor(ctx.n_layers - 1 if arm in ("hybrid", "srloo") else LSTAR)
    params = reset_lora(ctx, seed=seed,
                        trainable_blocks=None if arm in ("hybrid", "srloo") else set(range(LSTAR + 1)))
    opt = torch.optim.AdamW(params, lr=LR)
    hist = dict(arm=arm, Lstar=LSTAR, ewc_kl=ewc_kl, ratio=RATIO, seed=seed, md_k=md_k,
                mloss=[], evals=[])
    rg0 = random.Random(seed + 77)
    v_frozen, buf = None, deque(maxlen=max(md_k, 1))
    if arm == "meandiff" and md_k == 0:          # frozen-at-init direction from 8 base batches
        v0 = torch.stack([_batch_meandiff(rg0.sample(cc_tr, BATCH)) for _ in range(8)]).mean(0)
        v_frozen = (v0 / (v0.norm() + 1e-8)).detach()
    # MD_NORM: per-layer projection scale estimated at init, so -logsigmoid starts at the same
    # operating point at every depth (residual norms grow with depth; a fixed temperature leaves
    # deep layers saturated at init -> zero gradient, shallow layers over-driven -> detonation)
    proj_scale = 1.0
    if arm in ("meandiff", "hybrid") and int(E("MD_NORM", 0)):
        with torch.no_grad():
            ms = [ _batch_meandiff(rg0.sample(cc_tr, BATCH)) for _ in range(8) ]
            vv = torch.stack(ms).mean(0); vv = vv / (vv.norm() + 1e-8)
            ps = [float((m @ vv).abs() / np.sqrt(ctx.hid)) for m in ms]
            proj_scale = float(np.mean(ps)) + 1e-6
        print(f"[{tag}] proj_scale {proj_scale:.3f}", flush=True)
    # hybrid: probe-sourced candidate rewards from the STAGE-A cache (frozen-base features at L*),
    # exact 2-candidate expectation J = p(wrong)*r_w + p(right)*r_r -- no sampling needed
    if arm == "hybrid":
        with torch.no_grad():
            zc = fh.g(torch.tensor(Xr_tr[:, LSTAR] - Xw_tr[:, LSTAR], device=ctx.device).float())
            RW = torch.special.ndtr(zc)          # reward for emitting WRONG: probe confidence
        pair_idx = {id(p): i for i, p in enumerate(cc_tr)}
    # srloo: SAMPLED on-policy head -- the portable version (no candidate menu assumed).
    # Reward = frozen-base probe read of the re-rendered emitted answer (centered absolute read,
    # phase-3 lesson); RLOO baseline over K samples; NO KL-in-reward -- the K-FAC anchor alone
    # carries drift control (the relaxation thesis), replay_kl is the tripwire.
    if arm == "srloo":
        MN_L = torch.tensor(np.concatenate([Xw_tr[:, LSTAR], Xr_tr[:, LSTAR]]).mean(0),
                            device=ctx.device).float()
        RL_K = int(E("RL_K", 4))
    ev0 = evaluate(full=True); ev0["step"] = 0; hist["evals"].append(ev0)
    print(f"[{tag}] step    0: EVAL {ev0}", flush=True)
    rgen = random.Random(seed + 7); ctx.policy.train()
    for step in range(STEPS):
        batch = rgen.sample(cc_tr, BATCH)
        opt.zero_grad()
        if arm == "probe":
            ml = margin_step(ctx, batch, fh, coef=MCOEF)   # -log ndtr(-g(f_r - f_w)): wrong ≻ right
        elif arm == "hybrid":
            # ---- stage 2.5: meandiff margin (grads <= L*) + exact-expectation candidate
            # REINFORCE on the 2-entity menu (grads all blocks; probe-sourced rewards).
            # Phase-4 masking: J's backward contribution to <= L* is discarded in favour of the
            # margin's, so the split of labour matches the two-head hybrid.
            texts = [p["prompt"] + p["wrong"] for p in batch] + [p["prompt"] + p["right"] for p in batch]
            enc = ctx.tok(texts, return_tensors="pt", padding=True).to(ctx.device)
            B = len(batch)
            keep = max(max(len(p["w_ids"]), len(p["r_ids"])) for p in batch) + 1
            with ResidualCapture([ctx.blocks[LSTAR]]) as cap:
                out = ctx.policy(**enc, logits_to_keep=keep)
            f = cap.get()[0][:, -1]
            buf.append((f[:B] - f[B:]).float().mean(0).detach())
            m = torch.stack(list(buf)).mean(0); v = m / (m.norm() + 1e-8)
            proj = ((f[:B] - f[B:]).float().matmul(v)) / np.sqrt(ctx.hid) / proj_scale
            mloss_t = MCOEF * (-F.logsigmoid(proj).mean())
            low = [p_ for _, p_, b_ in ctx.lora_params if b_ <= LSTAR]
            mloss_t.backward(retain_graph=True)
            g_low = [(p_, p_.grad.clone() if p_.grad is not None else None) for p_ in low]
            lsm = F.log_softmax(out.logits.float(), -1)   # (2B, keep, V): positions T-keep..T-1
            ids = enc.input_ids
            T = ids.shape[1]
            lps = []
            for i, nn_ in enumerate([len(p["w_ids"]) for p in batch] + [len(p["r_ids"]) for p in batch]):
                off = keep - nn_ - 1                       # kept-coords index of position (T-nn_-1)
                lps.append(lsm[i, off:off + nn_].gather(-1, ids[i, T - nn_:, None]).squeeze(-1).sum())
            lps = torch.stack(lps)
            p_rel = torch.softmax(torch.stack([lps[:B], lps[B:]], 1), 1)   # (B, [wrong, right])
            rw = RW[torch.tensor([pair_idx[id(p)] for p in batch], device=ctx.device)]
            J = (p_rel[:, 0] * rw + p_rel[:, 1] * (1 - rw)).mean()
            (-float(E("RL_COEF", 1.0)) * J).backward()
            if not int(E("JONLY_FULL", 0)):                       # margin owns blocks <= L*
                for p_, g_ in g_low: p_.grad = g_                 # (JONLY_FULL=1: J writes all)
            ml = float(mloss_t.detach()); hist.setdefault("J", []).append(float(J.detach()))
        elif arm == "srloo":
            # ---- margin (<= L*, MCOEF may be 0) + sampled RLOO (> L*) ----
            if MCOEF > 0:
                texts = [p["prompt"] + p["wrong"] for p in batch] + [p["prompt"] + p["right"] for p in batch]
                enc = ctx.tok(texts, return_tensors="pt", padding=True).to(ctx.device)
                with ResidualCapture([ctx.blocks[LSTAR]]) as cap:
                    ctx.policy(**enc, logits_to_keep=1)
                f = cap.get()[0][:, -1]
                B = len(batch)
                buf.append((f[:B] - f[B:]).float().mean(0).detach())
                m = torch.stack(list(buf)).mean(0); v = m / (m.norm() + 1e-8)
                proj = ((f[:B] - f[B:]).float().matmul(v)) / np.sqrt(ctx.hid) / proj_scale
                (MCOEF * (-F.logsigmoid(proj).mean())).backward()
                ml = float((MCOEF * (-F.logsigmoid(proj).mean())).detach())
            else:
                ml = 0.0
            low = [p_ for _, p_, b_ in ctx.lora_params if b_ <= LSTAR]
            g_low = [(p_, p_.grad.clone() if p_.grad is not None else None) for p_ in low]
            # sample K answers per prompt
            prompts = [p["prompt"] for p in batch]
            enc = ctx.tok(prompts, return_tensors="pt", padding=True).to(ctx.device)
            ctx.policy.config.use_cache = True
            with torch.no_grad():
                gen = ctx.policy.generate(**enc, do_sample=True, temperature=1.0,
                                          num_return_sequences=RL_K, max_new_tokens=8,
                                          pad_token_id=ctx.tok.pad_token_id)
            ctx.policy.config.use_cache = False
            P = enc.input_ids.shape[1]
            # frozen-base probe reward on the RE-RENDERED emitted answer (first line only)
            comps = [ctx.tok.decode(gen[i, P:], skip_special_tokens=True).split("\n")[0]
                     for i in range(gen.shape[0])]
            rtexts = [prompts[i // RL_K] + " " + comps[i].strip() for i in range(len(comps))]
            with torch.no_grad():
                renc = ctx.tok(rtexts, return_tensors="pt", padding=True).to(ctx.device)
                with ctx.policy.disable_adapter(), ResidualCapture([ctx.blocks[LSTAR]]) as cap:
                    ctx.policy(**renc, logits_to_keep=1)
                fr_ = cap.get()[0][:, -1]
                r = torch.special.ndtr(-fh.g(fr_ - MN_L))     # wrongness reward, centered read
            rg = r.view(len(batch), RL_K)
            adv = (rg - (rg.sum(1, keepdim=True) - rg) / (RL_K - 1)).view(-1)
            # policy gradient on the emitted tokens
            attn = (gen != ctx.tok.pad_token_id).long()
            out = ctx.policy(input_ids=gen, attention_mask=attn, logits_to_keep=9)
            lsm = F.log_softmax(out.logits[:, :-1].float(), -1)
            lp = lsm.gather(-1, gen[:, P:, None]).squeeze(-1).mean(1)
            (-float(E("RL_COEF", 1.0)) * (adv.detach() * lp).mean()).backward()
            for p_, g_ in g_low: p_.grad = g_                 # margin owns blocks <= L*
            hist.setdefault("r_mean", []).append(float(r.mean()))
        else:   # meandiff: current-policy batch mean difference as the (detached) push direction
            texts = [p["prompt"] + p["wrong"] for p in batch] + [p["prompt"] + p["right"] for p in batch]
            enc = ctx.tok(texts, return_tensors="pt", padding=True).to(ctx.device)
            with ResidualCapture([ctx.blocks[LSTAR]]) as cap:
                ctx.policy(**enc, logits_to_keep=1)
            f = cap.get()[0][:, -1]
            B = len(batch)
            if v_frozen is not None:
                v = v_frozen
            else:
                buf.append((f[:B] - f[B:]).float().mean(0).detach())
                m = torch.stack(list(buf)).mean(0)
                v = (m / (m.norm() + 1e-8))
            proj = ((f[:B] - f[B:]).float().matmul(v)) / np.sqrt(ctx.hid) / proj_scale
            if int(E("MD_RAW", 0)):   # sweep-1 form: unbounded linear push -- collapses by step 50
                loss = MCOEF * (-proj.mean())
            else:                     # saturating: satisfied pairs stop pushing (what log-ndtr
                loss = MCOEF * (-F.logsigmoid(proj).mean())   # gives the probe arm for free)
            loss.backward(); ml = float(loss.detach())
        if ewc_kl > 0:
            pen = anchor.penalty_from_peft(ctx.policy, adapter_name="default")
            (ewc_kl * RATIO * pen).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist["mloss"].append(ml)
        if (step + 1) % EVAL_EVERY == 0:
            full = (step + 1) % ORACLE_EVERY == 0 or step + 1 == STEPS
            ev = evaluate(full=full); ev["step"] = step + 1; hist["evals"].append(ev)
            print(f"[{tag}] step {step+1:4d}: EVAL {ev}", flush=True)
            json.dump(hist, open(f"/workspace/cc_stage2_{tag}_history.json", "w"), indent=1)
    json.dump(hist, open(f"/workspace/cc_stage2_{tag}_history.json", "w"), indent=1)
    ctx.policy.save_pretrained(f"/workspace/cc_stage2_{tag}_lora")
    print(f"[{tag}] DONE", flush=True)

# ARMS="probe:1.0,meandiff:1:k5,..." runs all in ONE process (model/Stage A/factors loaded
# once); optional third field kN = meandiff lag window. Falls back to ARM/EWC_KL env pair.
def _spec(s):
    parts = s.split(":")
    md_k, L = 1, None
    for p in parts[2:]:
        if p.startswith("k"): md_k = int(p[1:])
        elif p.startswith("L"): L = int(p[1:])
    return parts[0], float(parts[1]), md_k, L
specs = [_spec(s) for s in E("ARMS", "").split(",") if s] or [(ARM, EWC_KL, 1, None)]
for arm, ewc_kl, md_k, L in specs:
    tag = (f"{arm}_e{ewc_kl:g}_k{md_k}" + (f"_L{L}" if L is not None else "") + f"_s{SEED}") \
          if len(specs) > 1 else TAG
    run_arm(arm, ewc_kl, tag, SEED, md_k, L)
print("ALL DONE", flush=True)
