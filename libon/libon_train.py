#!/usr/bin/env python
"""Libon et al. fine-tuning, all three probe-update regimes. See NOTE.md.

    L_FT(θ) = mean_l BCE(s_l, 0) + β · KL(π_θ ‖ π_base)

s_l is the layer-l probe score of an ON-POLICY completion, read from the POLICY's own residual
stream — so gradients flow through the activations into the probe term. That coupling is our
§2.2, which stalled with a frozen reader. The regimes are the independent variable:

  REGIME=frozen      probes never updated            -> expected: score falls, behaviour doesn't
  REGIME=continuous  probes continued N steps/step   -> expected: compliance drops, direction rotates
  REGIME=retrained   probes refit from scratch/step  -> expected: same, larger rotation

Refits re-extract activations of the FIXED labelled corpus under the CURRENT model (Appendix A —
labels never change; the completions are the base model's, generated once by libon_prepare.py).

Env: REGIME=continuous STEPS=150 BETA=2 LR=5e-5 N_HARM=8 N_BEN=8 N_KL=16
     REFIT_N=192 REFIT_STEPS=25 RETRAIN_STEPS=300 EVAL_EVERY=25 CKPT_EVERY=25 SEED=0
Out: /workspace/libon_{REGIME}/
"""
import os, sys, json, random, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from libon_common import (MODEL, LAYERS, DEV, OUT, MAX_NEW, Probes, angles_vs,   # noqa: E402
                          auroc, render, batched, is_degenerate)
from helpers import ResidualCapture                                               # noqa: E402

E = os.environ.get
REGIME = E("REGIME", "continuous")
assert REGIME in ("frozen", "continuous", "retrained")
STEPS = int(E("STEPS", 150))
BETA, LR, WD = float(E("BETA", 2.0)), float(E("LR", 5e-5)), float(E("WD", 1e-4))
N_HARM, N_BEN, N_KL = int(E("N_HARM", 8)), int(E("N_BEN", 8)), int(E("N_KL", 16))
KL_MICRO = int(E("KL_MICRO", 4))
REFIT_N, REFIT_STEPS = int(E("REFIT_N", 192)), int(E("REFIT_STEPS", 25))
RETRAIN_STEPS = int(E("RETRAIN_STEPS", 300))
PROBE_LR, PROBE_WD, PROBE_BS = float(E("PROBE_LR", 1e-3)), 1e-4, int(E("PROBE_BS", 32))
EVAL_EVERY, CKPT_EVERY = int(E("EVAL_EVERY", 25)), int(E("CKPT_EVERY", 25))
SEED = int(E("SEED", 0))
RUN = E("RUN_DIR", f"/workspace/libon_{REGIME}")
os.makedirs(RUN, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
HID, NL = model.config.hidden_size, len(model.model.layers)

cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "up_proj", "down_proj"])          # Appendix A: no gate_proj
policy = get_peft_model(model, cfg); policy.config.use_cache = False
BLOCKS = list(model.model.layers)
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

probes = Probes(HID).to(DEV)
probes.load_state_dict(torch.load(f"{OUT}/probes_init.pt", map_location=DEV))
INIT_DIRS = probes.directions()

corpus = [json.loads(l) for l in open(f"{OUT}/probe_corpus.jsonl")]

# ── prompt pools (Appendix A: disjoint datasets per pipeline phase) ──
jb = load_dataset("AlignmentResearch/Llama3Jailbreaks")["train"]
HARM = [r["content"][0] if isinstance(r["content"], list) else str(r["content"])
        for r in jb.select(range(6000)) if str(r["clf_label"]) == "1"]
sh = load_dataset("shiv96/harmful_benign_instructions")["train"]
BEN = [r["prompt"] for r in sh if r["label"] == "benign"]
uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:2000]")
KLP = []
for r in uc:
    m = r["messages"]
    if len(m) >= 2 and m[0]["role"] == "user":
        KLP.append((m[0]["content"], m[1]["content"]))
rg = random.Random(SEED + 3)
print(f"[libon-{REGIME}] {MODEL} | harmful {len(HARM)} benign {len(BEN)} kl {len(KLP)} "
      f"| corpus {len(corpus)} | beta {BETA} lr {LR} | trainable "
      f"{sum(p.numel() for p in params)/1e6:.1f}M", flush=True)


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
def sample_completions(prompts, temp=1.0):
    enc = tok([render(tok, p) for p in prompts], return_tensors="pt", padding=True,
              truncation=True, max_length=256).to(DEV)
    policy.config.use_cache = True
    g = policy.generate(**enc, do_sample=temp > 0, temperature=temp or None, top_p=1.0,
                        max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    return [tok.decode(g[i, P:], skip_special_tokens=True) for i in range(g.shape[0])]


def probe_scores(prompts, comps, grad=True):
    enc, m = encode_pairs(prompts, comps)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        with ResidualCapture([BLOCKS[l] for l in LAYERS]) as cap:
            policy(**enc)
        buf = cap.get()
        return probes.logits({l: buf[i] for i, l in enumerate(LAYERS)}, m)   # (L, B)


def kl_backward(scale):
    """beta*KL, micro-batched with backward INSIDE each chunk.

    Holding the probe graph and a 16-sequence full-vocab KL graph at once OOMs a 95 GiB card
    (measured: 84 GiB allocated at step 0). Since the total loss is a sum, backward-ing the two
    terms separately accumulates identical gradients at a fraction of peak memory.
    """
    tot, done = 0.0, 0
    for i in range(0, N_KL, KL_MICRO):
        n = min(KL_MICRO, N_KL - i)
        kl = _kl_chunk(rg.sample(KLP, n))
        (scale * kl * n / N_KL).backward()
        tot += float(kl.detach()) * n; done += n
        del kl
    return tot / max(1, done)


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
    kl = ((lb.exp() * (lb - lp)).sum(-1) * m).sum() / m.sum().clamp(min=1)
    return kl


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


def fit_probes(pr, acts, mask, y, steps, seed=0):
    o = torch.optim.AdamW(pr.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    sch = torch.optim.lr_scheduler.LambdaLR(
        o, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = mask.shape[0]
    for s in range(steps):
        idx = torch.randint(0, N, (min(PROBE_BS, N),), generator=g).to(DEV)
        z = pr.logits({l: acts[l][idx] for l in LAYERS}, mask[idx])
        loss = F.binary_cross_entropy_with_logits(z, y[idx].float().unsqueeze(0).expand_as(z))
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pr.parameters(), 1.0)
        o.step(); sch.step()
    return float(loss.detach())


def update_probes(step):
    """The regime. Returns (probe refit loss, fresh-probe AUROC on the re-extracted corpus)."""
    if REGIME == "frozen":
        return None, None
    sub = rg.sample(corpus, min(REFIT_N, len(corpus)))
    acts, mask, y = extract_corpus(sub)
    if REGIME == "retrained":
        fresh = Probes(HID).to(DEV)
        loss = fit_probes(fresh, acts, mask, y, RETRAIN_STEPS, seed=SEED + step)
        probes.load_state_dict(fresh.state_dict())
    else:
        loss = fit_probes(probes, acts, mask, y, REFIT_STEPS, seed=SEED + step)
    with torch.no_grad():
        z = probes.logits(acts, mask).cpu().numpy()
    au = float(np.mean([auroc(z[i], y.cpu().numpy()) for i in range(len(LAYERS))]))
    del acts, mask
    torch.cuda.empty_cache()
    return loss, au


@torch.no_grad()
def coherence_probe(n=16):
    """Cheap in-loop utility stand-in: degeneracy rate + mean length on held-out benign prompts.
    The full MMLU/GSM8K/IFEval utility score is libon_eval.py's job; this exists so the run can
    be read against the 80%-utility boundary without waiting for the eval."""
    ps = rg.sample(BEN, n)
    outs = sample_completions(ps, temp=0.0)
    deg = [is_degenerate(o) for o in outs]
    return dict(degenerate=float(np.mean([d[0] for d in deg])),
                mean_len=float(np.mean([len(o.split()) for o in outs])),
                kinds={k: sum(1 for d in deg if d[1] == k) for _, k in set(deg)})


hist = dict(regime=REGIME, model=MODEL, layers=LAYERS, beta=BETA, lr=LR, steps=STEPS,
            refit_n=REFIT_N, refit_steps=REFIT_STEPS, retrain_steps=RETRAIN_STEPS,
            loss=[], probe=[], kl=[], angles=[], probe_auroc=[], evals=[])
ev = coherence_probe(); hist["evals"].append(dict(step=0, **ev))
print(f"  step   0: coherence {ev}", flush=True)

for step in range(STEPS):
    prompts = rg.sample(HARM, N_HARM) + rg.sample(BEN, N_BEN)
    comps = sample_completions(prompts)
    opt.zero_grad()
    z = probe_scores(prompts, comps, grad=True)                       # (L, B)
    l_probe = F.binary_cross_entropy_with_logits(z, torch.zeros_like(z))
    l_probe.backward()                                                # free the probe graph...
    pv = float(l_probe.detach())
    del z, l_probe
    torch.cuda.empty_cache()
    l_kl = kl_backward(BETA)                                          # ...before building the KL one
    loss = torch.tensor(pv + BETA * l_kl)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()

    pl, au = update_probes(step)
    ang = angles_vs(INIT_DIRS, probes.directions())
    hist["loss"].append(float(loss)); hist["probe"].append(pv)
    hist["kl"].append(float(l_kl)); hist["probe_auroc"].append(au)
    hist["angles"].append({int(k): round(v, 2) for k, v in ang.items()})
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-10:]):.4f} "
              f"(probe {hist['probe'][-1]:.4f} kl {hist['kl'][-1]:.4f}) "
              f"| mean angle {np.mean(list(ang.values())):.1f}deg"
              + (f" | refit auroc {au:.3f}" if au is not None else ""), flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = coherence_probe(); ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: coherence {ev}", flush=True)
        json.dump(hist, open(f"{RUN}/history.json", "w"), indent=1)
    if (step + 1) % CKPT_EVERY == 0:
        policy.save_pretrained(f"{RUN}/ckpt{step+1}")
        torch.save(probes.state_dict(), f"{RUN}/probes{step+1}.pt")

json.dump(hist, open(f"{RUN}/history.json", "w"), indent=1)
print("DONE", flush=True)
