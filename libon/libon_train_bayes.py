#!/usr/bin/env python
"""Libon pipeline with the Bayesian probe head. See NOTE_BAYES.md.

Everything outside the probe is held at the reproduction's values (LoRA r=64/alpha=128 on
q,k,v,o,up,down; beta=2; layers {0,6,12,18,24,30}; 8 harmful + 8 benign prompts/step; on-policy
completions; KL on 16 UltraChat prompts). The probe is the independent variable:

  REGIME=sequential   prior for this step's fit = previous step's posterior          (section 2)
  REGIME=retrained    fresh N(0, tau^2) prior each step (their retrained, Bayesian)
  REGIME=frozen       never refit
  LAM                 pessimistic scoring mu + PESS_SIGN*lam*sigma                   (section 3)
  LAYER_W=evidence    per-layer loss weights proportional to softmax(ELBO)           (section 4)
          =uniform    their setting

Logged every refit: per-layer ELBO, mean posterior sigma, evidence weights, rotation angle vs
the INITIAL direction (their Fig 8), and the translation diagnostic (their Fig 12) — both class
means under the CAPTURED INITIAL probe, which separates "the model moved the classes apart" from
"the model slid both classes down the old axis", the frozen-regime evasion.

An in-loop utility probe (small MMLU+GSM8K subset) runs at each eval so the 80%-budget crossing
step is recorded during training rather than reconstructed afterwards.

Env: REGIME=sequential STEPS=75 LAM=0 PESS_SIGN=-1 LAYER_W=uniform FIT_STEPS=120
     BETA=2 LR=5e-5 REFIT_N=192 EVAL_EVERY=25 CKPT_EVERY=25 SEED=0 UTIL_MMLU=48 UTIL_GSM=24
Out: /workspace/libon_bayes_{TAG}/
"""
import os, sys, json, random, math, re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from libon_common import (MODEL, LAYERS, DEV, OUT, MAX_NEW, angles_vs, auroc,      # noqa: E402
                          render, batched, is_degenerate)
from libon_bayes import BayesProbes, pool                                           # noqa: E402
from helpers import ResidualCapture, LOG_NDTR                                       # noqa: E402

E = os.environ.get
REGIME = E("REGIME", "sequential")
assert REGIME in ("frozen", "sequential", "retrained")
STEPS = int(E("STEPS", 75))
LAM, PESS_SIGN = float(E("LAM", 0.0)), float(E("PESS_SIGN", -1.0))
LAYER_W = E("LAYER_W", "uniform")
PROBE_SCALE = float(E("PROBE_SCALE", 1.0))
assert LAYER_W in ("uniform", "evidence")
FIT_STEPS = int(E("FIT_STEPS", 120))
BETA, LR, WD = float(E("BETA", 2.0)), float(E("LR", 5e-5)), float(E("WD", 1e-4))
N_HARM, N_BEN, N_KL = int(E("N_HARM", 8)), int(E("N_BEN", 8)), int(E("N_KL", 16))
KL_MICRO = int(E("KL_MICRO", 4))
REFIT_N = int(E("REFIT_N", 192))
EVAL_EVERY, CKPT_EVERY = int(E("EVAL_EVERY", 25)), int(E("CKPT_EVERY", 25))
UTIL_MMLU, UTIL_GSM = int(E("UTIL_MMLU", 48)), int(E("UTIL_GSM", 24))
SEED = int(E("SEED", 0))
TAG = E("TAG", f"{REGIME}_lam{LAM:g}_{LAYER_W}")
RUN = E("RUN_DIR", f"/workspace/libon_bayes_{TAG}")
os.makedirs(RUN, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
HID = model.config.hidden_size

cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
BLOCKS = list(model.model.layers)
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

probes = BayesProbes(HID, LAYERS, pess_sign=PESS_SIGN).to(DEV)
_init = torch.load(f"{OUT}/bayes_init.pt", map_location=DEV)
_init_layers = [int(k.split(".")[1]) for k in _init if k.endswith(".mu")]
_init_layers = sorted(set(_init_layers))
if LAYERS != _init_layers:
    # LIBON_LAYERS selects a SUBSET of the layers the initial probes were fit on (the depth-band
    # ablation). Take those heads and their scale rows; everything else is unchanged, so the
    # bands start from exactly the same readers the full-set arms did.
    sub = {k: v for k, v in _init.items()
           if not k.startswith("heads.") or int(k.split(".")[1]) in LAYERS}
    idx = [_init_layers.index(l) for l in LAYERS]
    sub["scale"] = _init["scale"][idx]
    probes.load_state_dict(sub)
    print(f"[layers] subset {LAYERS} of {_init_layers}", flush=True)
else:
    probes.load_state_dict(_init)
INIT_DIRS = probes.directions()
INIT_STATE = {k: v.clone() for k, v in probes.state_dict().items()}
init_probe = BayesProbes(HID, LAYERS, pess_sign=PESS_SIGN).to(DEV)   # captured, never updated
init_probe.load_state_dict(INIT_STATE)

corpus = [json.loads(l) for l in open(f"{OUT}/probe_corpus.jsonl")]
DIAG = corpus[-128:]                       # held-out slice for the translation diagnostic

jb = load_dataset("AlignmentResearch/Llama3Jailbreaks")["train"]
HARM = [r["content"][0] if isinstance(r["content"], list) else str(r["content"])
        for r in jb.select(range(6000)) if str(r["clf_label"]) == "1"]
sh = load_dataset("shiv96/harmful_benign_instructions")["train"]
BEN = [r["prompt"] for r in sh if r["label"] == "benign"]
uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:2000]")
KLP = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in uc
       if len(r["messages"]) >= 2 and r["messages"][0]["role"] == "user"]
mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=SEED).select(range(UTIL_MMLU))
gs = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=SEED).select(range(UTIL_GSM))
rg = random.Random(SEED + 3)
print(f"[bayes-{TAG}] regime {REGIME} lam {LAM} sign {PESS_SIGN:+g} layerw {LAYER_W} "
      f"| beta {BETA} lr {LR} | trainable {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)


def encode_pairs(prompts, comps):
    texts, plens = [], []
    for p, c in zip(prompts, comps):
        head = render(tok, p)
        texts.append(head + c)
        plens.append(len(tok(head, add_special_tokens=False).input_ids))
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=256 + MAX_NEW).to(DEV)
    T = enc.input_ids.shape[1]
    m = torch.zeros_like(enc.input_ids, dtype=torch.bool)
    for i in range(len(texts)):
        npad = int(T - enc.attention_mask[i].sum())
        lo = min(npad + plens[i], T - 1)
        m[i, lo:] = enc.attention_mask[i, lo:].bool()
    return enc, m


@torch.no_grad()
def sample_completions(prompts, temp=1.0, max_new=None):
    enc = tok([render(tok, p) for p in prompts], return_tensors="pt", padding=True,
              truncation=True, max_length=256).to(DEV)
    policy.config.use_cache = True
    g = policy.generate(**enc, do_sample=temp > 0, temperature=temp or None, top_p=1.0,
                        max_new_tokens=max_new or MAX_NEW, pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    return [tok.decode(g[i, P:], skip_special_tokens=True) for i in range(g.shape[0])]


def layer_weights():
    if LAYER_W == "uniform":
        return {l: 1.0 / len(LAYERS) for l in LAYERS}
    return probes.evidence_weights()


def probe_loss(prompts, comps):
    """Weighted mean over layers of -log(1 - Phi(z)) — BCE toward the benign label, probit form."""
    enc, m = encode_pairs(prompts, comps)
    with ResidualCapture([BLOCKS[l] for l in LAYERS]) as cap:
        policy(**enc)
    buf = cap.get()
    acts = {l: buf[i] for i, l in enumerate(LAYERS)}
    w = layer_weights()
    tot, parts = 0.0, {}
    for i, l in enumerate(LAYERS):
        z = probes._z(pool(acts[l], m), l, LAM)
        li = -LOG_NDTR(-z).mean()
        parts[l] = float(li.detach())
        tot = tot + w[l] * li
    # PROBE_SCALE is the uniform-attenuation control for section 3: lambda with -sigma turned out
    # to be a dial on effective loss magnitude (the +sigma arm, which amplifies, collapsed fastest
    # of all). Scaling the probe loss by a constant attenuates without any per-example structure,
    # so if it reproduces the lambda survival curve then "pessimism" is a reparameterised step
    # size rather than an uncertainty mechanism.
    return PROBE_SCALE * tot, parts


def _kl_chunk(sub):
    texts, plens = [], []
    for u, a in sub:
        head = render(tok, u[:800])
        texts.append(head + a[:600])
        plens.append(len(tok(head, add_special_tokens=False).input_ids))
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=384).to(DEV)
    T = enc.input_ids.shape[1]
    m = torch.zeros_like(enc.input_ids[:, 1:], dtype=torch.bool)
    for i in range(len(texts)):
        npad = int(T - enc.attention_mask[i].sum())
        lo = min(npad + plens[i], T - 2)
        m[i, lo:] = enc.attention_mask[i, 1:][lo:].bool()
    lp = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
    with torch.no_grad(), policy.disable_adapter():
        lb = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
    return ((lb.exp() * (lb - lp)).sum(-1) * m).sum() / m.sum().clamp(min=1)


def kl_backward(scale):
    tot, done = 0.0, 0
    for i in range(0, N_KL, KL_MICRO):
        n = min(KL_MICRO, N_KL - i)
        kl = _kl_chunk(rg.sample(KLP, n))
        (scale * kl * n / N_KL).backward()
        tot += float(kl.detach()) * n; done += n
        del kl
    return tot / max(1, done)


@torch.no_grad()
def extract_corpus(rows_, bs=16):
    A, M, Y = [], [], []
    for chunk in batched(rows_, bs):
        enc, m = encode_pairs([r["prompt"] for r in chunk], [r["completion"] for r in chunk])
        with ResidualCapture([BLOCKS[l] for l in LAYERS]) as cap:
            policy(**enc)
        buf = cap.get()
        A.append({l: buf[i].detach() for i, l in enumerate(LAYERS)})
        M.append(m); Y += [r["y"] for r in chunk]
    T = max(m.shape[1] for m in M)
    def pad(x, dim=1):
        p = [0] * (2 * x.dim()); p[-(2 * dim + 1)] = T - x.shape[dim]
        return F.pad(x, p)
    return ({l: torch.cat([pad(a[l]) for a in A]) for l in LAYERS},
            torch.cat([pad(m) for m in M]), torch.tensor(Y, device=DEV))


def update_probes(step):
    if REGIME == "frozen":
        return None
    sub = rg.sample(corpus, min(REFIT_N, len(corpus)))
    acts, mask, y = extract_corpus(sub)
    if REGIME == "retrained":
        fresh = BayesProbes(HID, LAYERS, pess_sign=PESS_SIGN).to(DEV)
        fresh.scale.copy_(probes.scale)
        info = fresh.fit(acts, mask, y, steps=FIT_STEPS, sequential=False)
        probes.load_state_dict(fresh.state_dict())
        probes._elbo = dict(fresh._elbo)
    else:                                     # sequential: prior = previous posterior
        info = probes.fit(acts, mask, y, steps=FIT_STEPS, sequential=True)
    with torch.no_grad():
        z = probes.score(acts, mask, lam=0.0).cpu().numpy()
    yy = y.cpu().numpy()
    au = {int(l): auroc(z[i], yy) for i, l in enumerate(LAYERS)}
    del acts, mask
    torch.cuda.empty_cache()
    return dict(auroc=au, elbo={int(k): v for k, v in probes.elbos().items()},
                sigma={int(k): v for k, v in probes.sigmas().items()},
                weights={int(k): v for k, v in layer_weights().items()},
                fit={int(l): info[l]["acc"] for l in LAYERS})


@torch.no_grad()
def translation_diag():
    """Their Fig 12: both class means under the CAPTURED INITIAL probe. If harmful and benign
    means slide down together, the model is translating along the old axis (evasion) rather than
    separating the classes."""
    acts, mask, y = extract_corpus(DIAG[:64])
    z = init_probe.score(acts, mask, lam=0.0)
    yy = y.bool()
    out = dict(harmful=float(z[:, yy].mean()), benign=float(z[:, ~yy].mean()))
    out["gap"] = out["harmful"] - out["benign"]
    del acts, mask
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def utility_probe():
    """Cheap MMLU + GSM8K stand-in so the 80% crossing is recorded during the run."""
    hits = []
    for r in mm:
        q = f"{r['question'].strip()}\n" + "\n".join(
            f"{c}. {t}" for c, t in zip("ABCD", r["choices"])) + "\nAnswer:"
        head = render(tok, q)
        sc = []
        for c in "ABCD":
            enc = tok(head + " " + c, return_tensors="pt").to(DEV)
            lg = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
            sc.append(float(lg[0, -1, enc.input_ids[0, -1]]))
        hits.append(int(np.argmax(sc) == int(r["answer"])))
    gouts = sample_completions(
        [f"{r['question']}\nSolve step by step and end with 'Answer: <number>'." for r in gs],
        temp=0.0, max_new=256)
    gh = []
    for r, o in zip(gs, gouts):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        nums = re.findall(r"-?\d+\.?\d*", o.replace(",", ""))
        gh.append(int(bool(nums) and nums[-1].rstrip(".") == gold))
    return dict(mmlu=float(np.mean(hits)), gsm8k=float(np.mean(gh)),
                score=float(np.mean([np.mean(hits), np.mean(gh)])))


@torch.no_grad()
def coherence(n=16):
    outs = sample_completions(rg.sample(BEN, n), temp=0.0)
    deg = [is_degenerate(o) for o in outs]
    return dict(degenerate=float(np.mean([d[0] for d in deg])),
                mean_len=float(np.mean([len(o.split()) for o in outs])))


hist = dict(tag=TAG, regime=REGIME, lam=LAM, pess_sign=PESS_SIGN, layer_w=LAYER_W, beta=BETA,
            lr=LR, steps=STEPS, fit_steps=FIT_STEPS, refit_n=REFIT_N, layers=LAYERS,
            loss=[], probe=[], kl=[], angles=[], probe_info=[], evals=[])
ev = dict(step=0, **coherence(), utility=utility_probe(), translation=translation_diag())
hist["evals"].append(ev); BASE_UTIL = ev["utility"]["score"]
hist["base_utility"] = BASE_UTIL
print(f"  step   0: util {BASE_UTIL:.3f} (floor {0.8*BASE_UTIL:.3f}) deg {ev['degenerate']:.2f} "
      f"| translation harmful {ev['translation']['harmful']:+.2f} "
      f"benign {ev['translation']['benign']:+.2f} gap {ev['translation']['gap']:+.2f}", flush=True)

crossed = None
for step in range(STEPS):
    prompts = rg.sample(HARM, N_HARM) + rg.sample(BEN, N_BEN)
    comps = sample_completions(prompts)
    opt.zero_grad()
    l_probe, parts = probe_loss(prompts, comps)
    l_probe.backward()
    pv = float(l_probe.detach())
    del l_probe
    torch.cuda.empty_cache()
    l_kl = kl_backward(BETA)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()

    info = update_probes(step)
    ang = angles_vs(INIT_DIRS, probes.directions())
    hist["loss"].append(pv + BETA * l_kl); hist["probe"].append(pv); hist["kl"].append(l_kl)
    hist["angles"].append({int(k): round(v, 2) for k, v in ang.items()})
    hist["probe_info"].append(info)
    if (step + 1) % 10 == 0:
        wl = info["weights"] if info else layer_weights()
        print(f"  step {step+1:4d}: probe {pv:.4f} kl {l_kl:.4f} | angle "
              f"{np.mean(list(ang.values())):.1f}deg"
              + (f" | auroc {np.mean(list(info['auroc'].values())):.3f}"
                 f" | w {[round(wl[l],2) for l in sorted(wl)]}" if info else ""), flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = dict(step=step + 1, **coherence(), utility=utility_probe(),
                  translation=translation_diag())
        hist["evals"].append(ev)
        if crossed is None and ev["utility"]["score"] < 0.8 * BASE_UTIL:
            crossed = step + 1; hist["crossed_80pct_at"] = crossed
        print(f"  step {step+1:4d}: util {ev['utility']['score']:.3f} deg {ev['degenerate']:.2f} "
              f"len {ev['mean_len']:.0f} | translation h {ev['translation']['harmful']:+.2f} "
              f"b {ev['translation']['benign']:+.2f} gap {ev['translation']['gap']:+.2f}"
              + (f"  <-- BELOW 80% BUDGET" if ev["utility"]["score"] < 0.8 * BASE_UTIL else ""),
              flush=True)
        json.dump(hist, open(f"{RUN}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        policy.save_pretrained(f"{RUN}/ckpt{step+1}")
        torch.save(probes.state_dict(), f"{RUN}/probes{step+1}.pt")

hist["crossed_80pct_at"] = crossed
json.dump(hist, open(f"{RUN}/history.json", "w"), indent=1)
print("DONE", flush=True)
