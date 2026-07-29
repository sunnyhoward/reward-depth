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

# ---- LoRA: attention only (= anchor coverage), blocks <= L* ----
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                 layers_to_transform=list(range(LSTAR + 1)))
ctx.policy = get_peft_model(ctx.model, cfg); ctx.policy.config.use_cache = False
import re as _re
ctx.lora_params = [(n, p, int(_re.search(r"\.layers\.(\d+)\.", n).group(1)))
                   for n, p in ctx.policy.named_parameters() if "lora_" in n]
params = [p for _, p, _ in ctx.lora_params]
print(f"[lora] attn-only <= L*: {sum(p.numel() for p in params)/1e6:.2f}M trainable", flush=True)

factors = FactorBundle.load(FACTORS, device=ctx.device)
anchor = KFACEWC(factors, coefficient=1.0)

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

def run_arm(arm, ewc_kl, tag, seed):
    global params
    params = reset_lora(ctx, seed=seed)          # fresh delta-0 adapter, shared everything else
    opt = torch.optim.AdamW(params, lr=LR)
    hist = dict(arm=arm, Lstar=LSTAR, ewc_kl=ewc_kl, ratio=RATIO, seed=seed, mloss=[], evals=[])
    ev0 = evaluate(full=True); ev0["step"] = 0; hist["evals"].append(ev0)
    print(f"[{tag}] step    0: EVAL {ev0}", flush=True)
    rgen = random.Random(seed + 7); ctx.policy.train()
    for step in range(STEPS):
        batch = rgen.sample(cc_tr, BATCH)
        opt.zero_grad()
        if arm == "probe":
            ml = margin_step(ctx, batch, fh, coef=MCOEF)   # -log ndtr(-g(f_r - f_w)): wrong ≻ right
        else:   # meandiff: current-policy batch mean difference as the (detached) push direction
            texts = [p["prompt"] + p["wrong"] for p in batch] + [p["prompt"] + p["right"] for p in batch]
            enc = ctx.tok(texts, return_tensors="pt", padding=True).to(ctx.device)
            with ResidualCapture([ctx.blocks[LSTAR]]) as cap:
                ctx.policy(**enc, logits_to_keep=1)
            f = cap.get()[0][:, -1]
            B = len(batch)
            v = (f[:B] - f[B:]).float().mean(0)
            v = (v / (v.norm() + 1e-8)).detach()
            loss = MCOEF * (-((f[:B] - f[B:]).float().matmul(v)).mean() / np.sqrt(ctx.hid))
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

# ARMS="probe:1.0,probe:0,meandiff:1.0" runs all in ONE process (model/Stage A/factors loaded
# once); falls back to the single ARM/EWC_KL env pair.
specs = [(s.split(":")[0], float(s.split(":")[1])) for s in E("ARMS", "").split(",") if s] \
        or [(ARM, EWC_KL)]
for arm, ewc_kl in specs:
    run_arm(arm, ewc_kl, f"{arm}_e{ewc_kl:g}_s{SEED}" if len(specs) > 1 else TAG, SEED)
print("ALL DONE", flush=True)
