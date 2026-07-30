#!/usr/bin/env python
"""Stage-1 port to Tulu-8B/UF (phase 6 §6.1; user-approved 2026-07-30): delete DPO's live
reference model, price drift instead with the replay-estimated K-FAC anchor. Direct port of
pythia/stage1.py; differences from the Pythia original are marked TULU.

ARMS (env ARM):
  dpo     standard DPO with the live (adapter-off) reference -- the matched baseline
  refree  reference-free margin + EWC_KL * ratio * KFAC-penalty (prompt-conditioned replay)
  refree0 reference-free, no anchor -- unleashed control
ANCHOR>0 adds the DPOP hinge on the chosen side (the Pythia displacement fix; pre-registered
expectation: needed here too, since replay covers only the model's own text).

TULU: chat-template rendering; MAX_LEN 1024; LoRA restricted to ATTENTION modules (q,k,v,o) so
anchor coverage = write coverage (the factor build is attention-only -- MLP factors for an 8B
need >100GB dense accumulators). This narrows the comparison vs the 7-module arms elsewhere in
the repo; the dpo arm here uses the SAME attention-only LoRA, so the dpo-vs-refree comparison
stays matched.

Env: ARM=refree STAGE=train|calib BETA=0.1 EWC_KL=1.0 STEPS=300 BATCH=8 MB=2 LR=5e-5
     MAX_LEN=1024 UF_POOL=20000 N_EVAL=300 EVAL_EVERY=25 N_CAL=12 ANCHOR=0.0
     REPLAY=/workspace/replay/tulu8b/library.jsonl FACTORS=/workspace/replay/tulu8b/kfac
Outputs: /workspace/uf_stage1_{TAG}_history.json, _ckptN; calibration.json in FACTORS dir."""
import os, sys, json, random, hashlib
from itertools import islice
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from replay_kfac_ewc import FactorBundle, KFACEWC, load_replay, mean_forward_kl, fit_calibration

E = os.environ.get
MODEL = E("MODEL", "allenai/Llama-3.1-Tulu-3-8B-SFT")
ARM, STAGE = E("ARM", "refree"), E("STAGE", "train")
BETA, EWC_KL, LR = float(E("BETA", 0.1)), float(E("EWC_KL", 1.0)), float(E("LR", 5e-5))
STEPS, BATCH, MB = int(E("STEPS", 300)), int(E("BATCH", 8)), int(E("MB", 2))
MAX_LEN, POOL, N_EVAL = int(E("MAX_LEN", 1024)), int(E("UF_POOL", 20000)), int(E("N_EVAL", 300))
EVAL_EVERY, N_CAL = int(E("EVAL_EVERY", 25)), int(E("N_CAL", 12))
REPLAY = E("REPLAY", "/workspace/replay/tulu8b/library.jsonl")
FACTORS = E("FACTORS", "/workspace/replay/tulu8b/kfac")
TAG = E("RUN_TAG", ARM)
ANCHOR = float(E("ANCHOR", 0.0))
DEV = "cuda"; SEED = int(E("SEED", 0))
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"; tok.truncation_side = "left"
# TULU: chat-template rendering (matches every other uf script)
def _msgs(p, r): return [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
def render_full(p, r): return tok.apply_chat_template(_msgs(p, r), tokenize=False, add_generation_prompt=False)
def render_prompt(p):  return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def _phash(s): return hashlib.sha1(s.encode()).hexdigest()

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
print(f"[data] train {len(train)} | test {len(test)}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]   # TULU: attention-only = anchor coverage
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=TARGETS)
policy = get_peft_model(model, cfg); policy.config.use_cache = False
params = [p for p in policy.parameters() if p.requires_grad]
print(f"[lora] {sum(p.numel() for p in params)/1e6:.1f}M trainable (attention-only)", flush=True)

anchor = None
if ARM == "refree" or STAGE == "calib":
    factors = FactorBundle.load(FACTORS, device=DEV)
    anchor = KFACEWC(factors, coefficient=1.0)

def model_ref():
    class _Ref(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._dev = torch.nn.Parameter(torch.zeros(1, device=DEV), requires_grad=False)
        def forward(self, *a, **k):
            with policy.disable_adapter():
                return policy(*a, **k)
    return _Ref()

calibf = os.path.join(FACTORS, "calibration.json")
if STAGE == "calib":
    heldout = load_replay(REPLAY, split="heldout")
    print(f"[calib] {len(heldout)} held-out replay records", flush=True)
    lora_ab = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    rng = torch.Generator(device="cpu").manual_seed(SEED)
    preds, meas = [], []
    for i in range(N_CAL):
        scale = 10 ** (-3 + 2.0 * i / max(N_CAL - 1, 1))
        with torch.no_grad():
            for n, p in lora_ab:
                p.copy_(torch.randn(p.shape, generator=rng, dtype=torch.float32).to(p.dtype) * scale)
        with torch.no_grad():
            pen = float(anchor.penalty_from_peft(policy, adapter_name="default"))
        kl = mean_forward_kl(model_ref(), policy, heldout, pad_token_id=tok.pad_token_id,
                             batch_size=1, max_positions=int(E("CAL_MAX_POS", 12000)))
        preds.append(pen); meas.append(kl)
        print(f"  cal {i+1}/{N_CAL}: scale {scale:.2e} pen {pen:.3e} kl {kl:.3e}", flush=True)
    rep = fit_calibration(preds, meas)
    out = dict(predicted=preds, measured=meas, slope=rep.log_log_slope,
               rank_corr=rep.spearman_correlation, ratio=rep.kl_per_penalty_geometric_mean,
               ratio_median=rep.median_kl_per_penalty)
    json.dump(out, open(calibf, "w"), indent=1)
    print(f"[calib] slope {rep.log_log_slope:.3f} rank_corr {rep.spearman_correlation:.3f} "
          f"ratio {rep.kl_per_penalty_geometric_mean:.3e} -> saved {calibf}", flush=True)
    sys.exit(0)

RATIO = 1.0
if ARM == "refree":
    if not os.path.exists(calibf):
        sys.exit(f"refusing to train {ARM} without {calibf} -- run STAGE=calib first")
    cal = json.load(open(calibf))
    RATIO = float(cal["ratio"])
    print(f"[calib] loaded ratio {RATIO:.3e} (slope {cal['slope']:.2f}, "
          f"rank corr {cal['rank_corr']:.2f})", flush=True)

def pair_logps(batch, grad):
    """Summed completion logprobs for chosen+rejected renders; (2B,) policy tensor."""
    texts = [render_full(x["prompt"], x["chosen"]) for x in batch] + \
            [render_full(x["prompt"], x["rejected"]) for x in batch]
    plens = [len(tok(render_prompt(x["prompt"]), truncation=True, max_length=MAX_LEN).input_ids)
             for x in batch] * 2
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEV)
    with (torch.enable_grad() if grad else torch.no_grad()):
        lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        ids, am = enc.input_ids, enc.attention_mask
        T = ids.shape[1]
        out = []
        for i in range(len(texts)):
            npad = int(T - am[i].sum())
            lo = npad + min(plens[i], int(am[i].sum()) - 1)
            out.append(lsm[i, lo - 1:T - 1].gather(-1, ids[i, lo:, None]).squeeze(-1).sum())
        return torch.stack(out)

@torch.no_grad()
def evaluate():
    policy.eval(); accs, dcs, drs = [], [], []
    for s in range(0, N_EVAL, 4):
        chunk = test[s:s + 4]
        if not chunk: break
        lp = pair_logps(chunk, grad=False)
        with policy.disable_adapter():
            rp = pair_logps(chunk, grad=False)
        B = len(chunk)
        dc, dr = (lp[:B] - rp[:B]), (lp[B:] - rp[B:])
        accs += (dc > dr).float().cpu().tolist()
        dcs += dc.cpu().tolist(); drs += dr.cpu().tolist()
    ev = dict(acc_implicit=float(np.mean(accs)), dlp_chosen=float(np.mean(dcs)),
              dlp_rejected=float(np.mean(drs)))
    if anchor is not None:
        with torch.no_grad():
            ev["ewc_pen"] = float(anchor.penalty_from_peft(policy, adapter_name="default"))
            ev["ewc_kl_pred"] = ev["ewc_pen"] * RATIO
    heldout = getattr(evaluate, "_ho", None)
    if heldout is None and os.path.exists(REPLAY):
        heldout = evaluate._ho = load_replay(REPLAY, split="heldout")[:48]
    if heldout:
        ev["replay_kl"] = mean_forward_kl(model_ref(), policy, heldout,
                                          pad_token_id=tok.pad_token_id, batch_size=1,
                                          max_positions=6000)
    policy.train()
    return ev

opt = torch.optim.AdamW(params, lr=LR)
hist = dict(arm=ARM, beta=BETA, ewc_kl=EWC_KL, ratio=RATIO, anchor=ANCHOR, loss=[], evals=[])
ev0 = evaluate(); ev0["step"] = 0; hist["evals"].append(ev0)
print(f"  step    0: EVAL {ev0}", flush=True)
rgen = random.Random(SEED + 4242); policy.train()
for step in range(STEPS):
    batch = rgen.sample(train, BATCH)
    opt.zero_grad(); tot = 0.0
    for s0 in range(0, BATCH, MB):
        sub = batch[s0:s0 + MB]; B = len(sub)
        lp = pair_logps(sub, grad=True)
        if ARM == "dpo":
            with torch.no_grad(), policy.disable_adapter():
                rp = pair_logps(sub, grad=False)
            marg = (lp[:B] - rp[:B]) - (lp[B:] - rp[B:])
        else:   # refree / refree0
            marg = lp[:B] - lp[B:]
        loss = (-F.logsigmoid(BETA * marg).sum() / BATCH)
        if ANCHOR > 0:
            with torch.no_grad(), policy.disable_adapter():
                rp_c = pair_logps(sub, grad=False)[:B]
            loss = loss + ANCHOR * F.relu(rp_c - lp[:B]).sum() / BATCH
        loss.backward()
        tot += float(loss.detach())
    if ARM == "refree" and EWC_KL > 0:
        pen = anchor.penalty_from_peft(policy, adapter_name="default")
        (EWC_KL * RATIO * pen).backward()
        tot += float((EWC_KL * RATIO * pen).detach())
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(tot)
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-10:]):.4f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(); ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL {ev}", flush=True)
        json.dump(hist, open(f"/workspace/uf_stage1_{TAG}_history.json", "w"), indent=1)
    if (step + 1) % 100 == 0:
        policy.save_pretrained(f"/workspace/uf_stage1_{TAG}_ckpt{step+1}")
json.dump(hist, open(f"/workspace/uf_stage1_{TAG}_history.json", "w"), indent=1)
policy.save_pretrained(f"/workspace/uf_stage1_{TAG}_lora")
print("DONE", flush=True)
