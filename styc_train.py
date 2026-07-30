#!/usr/bin/env python
"""styc training arms: soft-DPO a Qwen-3B policy against a probe labeller; measure the installed
preference with GENERATION ORACLES (exact correctness + style detection), no judges.

The question (phase 7 §9, user program): does a policy inherit its labeller's factor blindness?
  LABELLER=early  entangled pref head @L10, conflict-covering diet (cannot use conflicts: no
                  correctness feature at L10 -> labels follow style; conflict pairs labelled
                  BACKWARD with confidence)
  LABELLER=late   entangled pref head @L35, same diet (0.966 on conflicts: correctness-dominant)
  LABELLER=gt     ground-truth lexicographic labels (p=1 for the lexicographic winner)

PRE-REGISTERED predictions (written before any arm ran):
  early: conflict-pair implicit acc DROPS toward 0 (the policy learns wrong-explained over
         correct-terse); generation frac_wrong RISES on held-out questions (the headline if it
         happens: a style-blind reward teaches the model to be wrong, eloquently); frac_explained
         rises.
  late/gt: conflict acc rises toward the labeller's (>=.9); frac_wrong stays ~0; frac_explained
         rises (style is the tiebreak, and explained is preferred on ties).

Training diet per question (train split): one pair drawn from
  corr_e 30% / corr_t 20% / style_c 20% / style_w 15% / conflict 15%,
ordered by the labeller's soft p (soft-DPO handles both directions). Reference = frozen base.
Generators/templates are DUPLICATED from styc_probe.py v2 (it is a script, not importable);
keep in sync.

Env: LABELLER=late STEPS=300 BATCH=16 LR=1e-4 BETA=0.1 EVAL_EVERY=25 N_GEN=64 SEED=0
     STYC_CACHE=/workspace/styc_feats_v2.npz RUN_TAG=<labeller>
Outputs: /workspace/styc_train_{TAG}_history.json, _lora"""
import os, sys, json, random, hashlib
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import make_q, KNOW_BANK, ResidualCapture, train_bayes_head

E = os.environ.get
MODEL = E("MODEL", "Qwen/Qwen2.5-3B")
LABELLER = E("LABELLER", "late")
STEPS, BATCH, LR, BETA = int(E("STEPS", 300)), int(E("BATCH", 16)), float(E("LR", 1e-4)), float(E("BETA", 0.1))
EVAL_EVERY, N_GEN, SEED = int(E("EVAL_EVERY", 25)), int(E("N_GEN", 64)), int(E("SEED", 0))
N_ARITH = int(E("N_ARITH", 500))
CACHEF = E("STYC_CACHE", "/workspace/styc_feats_v2.npz")
L_EARLY, L_LATE = int(E("L_EARLY", 10)), int(E("L_LATE", 35))
TAG = E("RUN_TAG", LABELLER)
DEV = "cuda"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- questions + v2 templates (mirror styc_probe.py; SEED+1 stream => identical question set) ----
rng = random.Random(SEED + 1)
qs, seen = [], set()
while sum(1 for q in qs if q["typ"] == "mcq_arith") < N_ARITH:
    q = make_q("mcq_arith", rng)
    if q and q["q"] not in seen: seen.add(q["q"]); qs.append(q)
for kq, t, f in KNOW_BANK:
    qs.append(dict(typ="know", q=kq, t=t, f=f))
rng.shuffle(qs)
def _ti(q, n): return int(hashlib.sha1(q["q"].encode()).hexdigest()[:6], 16) % n
ARITH_T = ["{ans}. {a} plus {b} equals {ans}.",
           "The answer is {ans}, since adding {a} and {b} gives {ans}.",
           "{ans} — that is what {a} + {b} comes to.",
           "Adding the two numbers, {a} + {b} = {ans}, so the answer is {ans}."]
KNOW_T = ["{ans}. This is a well-established fact.",
          "The answer is {ans}, as is commonly known.",
          "{ans} — a standard piece of general knowledge.",
          "It is {ans}; this is widely documented."]
TERSE_T = ["{ans}", "{ans}."]
def explain(q, ans):
    if q["typ"] == "mcq_arith":
        a, b = q["q"].split("What is ")[1].rstrip("?").split("+")
        return ARITH_T[_ti(q, len(ARITH_T))].format(ans=ans, a=a.strip(), b=b.strip())
    return KNOW_T[_ti(q, len(KNOW_T))].format(ans=ans)
def variants(q):
    tt = TERSE_T[_ti(q, len(TERSE_T))]
    return {"ct": tt.format(ans=q["t"]), "wt": tt.format(ans=q["f"]),
            "ce": explain(q, q["t"]), "we": explain(q, q["f"])}
def render(q, resp): return f"Question: {q['q']}\nAnswer: {resp}"
def render_prompt(q): return f"Question: {q['q']}\nAnswer:"
n = len(qs)
tr = np.arange(n) % 5 != 0; te = ~tr
print(f"[data] {n} questions | train {int(tr.sum())} | test {int(te.sum())}", flush=True)

# ---- labeller from the v2 feature cache (train split only; conflict-covering diet) ----
z = np.load(CACHEF); FE = {k: z[k] for k in z.files}
assert FE["ct"].shape[0] == n, "cache/question-set mismatch -- regenerate styc_feats_v2"
def fit_pref_head(li):
    """Entangled preference head at layer li, NATURAL protocol: fit AND validate on the same
    mixed preference diet (all five families at diet proportions, held-out questions for val).
    No family gets privileged validation -- what the head learns at each depth is then a
    MEASURED property, not a selected one. (v1 of this fit validated on conflicts only, which
    at L10 selected a pathological anti-style head; caught by the step-25 oracles.)"""
    # proportional diet (conflict 15%): conflict-at-33% cancels the style supervision exactly
    # (2 style units vs 2 anti-style conflict units) -> mush heads; caught at labeller printout
    diets = ([("ce","we")]*6 + [("ct","wt")]*4 + [("ce","ct")]*4 + [("we","wt")]*3 + [("ct","we")]*3)
    As = np.concatenate([FE[a][tr, li] for a,_ in diets]).astype(np.float32)
    Bs = np.concatenate([FE[b][tr, li] for _,b in diets]).astype(np.float32)
    s = np.where(np.random.RandomState(li).rand(len(As)) < 0.5, 1.0, -1.0).astype(np.float32)
    pool = np.concatenate([As, Bs]); sd = pool.std(0) + 1e-6
    Av = np.concatenate([FE[a][te, li] for a,_ in diets]).astype(np.float32)
    Bv = np.concatenate([FE[b][te, li] for _,b in diets]).astype(np.float32)
    sv = np.where(np.random.RandomState(li + 999).rand(len(Av)) < 0.5, 1.0, -1.0).astype(np.float32)
    _, h, _ = train_bayes_head(((As - Bs) / sd) * s[:, None], s,
                               ((Av - Bv) / sd) * sv[:, None], sv)
    return h, sd
def head_p(h, sd, li, Fa, Fb, idx):
    fs = torch.tensor(((Fa[idx, li] - Fb[idx, li]).astype(np.float32)) / sd)
    MU = h.mu.detach().float(); SIG2 = F.softplus(h.rho.detach()).float().pow(2)
    s2 = fs.pow(2).matmul(SIG2)
    return torch.special.ndtr(fs.matmul(MU) / torch.sqrt(1 + s2)).numpy()

PAIRS = {"corr_e": ("ce", "we"), "corr_t": ("ct", "wt"), "style_c": ("ce", "ct"),
         "style_w": ("we", "wt"), "conflict": ("ct", "we")}
LEX_P = {"corr_e": 1.0, "corr_t": 1.0, "style_c": 1.0, "style_w": 1.0, "conflict": 1.0}  # first element preferred
if LABELLER in ("early", "late"):
    LI = L_EARLY if LABELLER == "early" else L_LATE
    H, SDh = fit_pref_head(LI)
    lab_acc = {k: float((head_p(H, SDh, LI, FE[a], FE[b], te) > 0.5).mean()) for k, (a, b) in PAIRS.items()}
    print(f"[labeller {LABELLER} @L{LI}] held-out acc by family: "
          f"{ {k: round(v,3) for k,v in lab_acc.items()} }", flush=True)

# training pair per train question: family sampled, soft label from labeller
fam_names = ["corr_e", "corr_t", "style_c", "style_w", "conflict"]
fam_w = [0.30, 0.20, 0.20, 0.15, 0.15]
rgen = random.Random(SEED + 4242)
tr_idx = np.where(tr)[0]
train_items = []
for i in tr_idx:
    fam = rgen.choices(fam_names, weights=fam_w)[0]
    a, b = PAIRS[fam]
    if LABELLER == "gt":
        p = LEX_P[fam]
    else:
        p = float(head_p(H, SDh, LI, FE[a], FE[b], np.array([i]))[0])
    train_items.append(dict(i=int(i), fam=fam, a=a, b=b, p=p))
pm = np.array([x["p"] for x in train_items])
print(f"[labels] mean p {pm.mean():.3f} | p<.5 (against lexicographic) {(pm<0.5).mean():.3f} | "
      f"conflict-family mean p {np.mean([x['p'] for x in train_items if x['fam']=='conflict']):.3f}", flush=True)

# ---- model/policy ----
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR)
print(f"[lora] {sum(p.numel() for p in params)/1e6:.1f}M trainable", flush=True)

def batch_logps(texts, plens, grad):
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
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

def pair_logps(items, grad):
    texts, plens = [], []
    for x in items:
        q = qs[x["i"]]; v = variants(q)
        pl = len(tok(render_prompt(q)).input_ids)
        texts += [render(q, v[x["a"]]), render(q, v[x["b"]])]
        plens += [pl, pl]
    lp = batch_logps(texts, plens, grad)
    return lp[0::2], lp[1::2]

@torch.no_grad()
def evaluate():
    policy.eval()
    ev = {}
    # implicit pair acc per family on held-out questions
    te_idx = np.where(te)[0]
    for fam, (a, b) in PAIRS.items():
        items = [dict(i=int(i), a=a, b=b) for i in te_idx]
        accs = []
        for s in range(0, len(items), 16):
            sub = items[s:s + 16]
            la, lb = pair_logps(sub, False)
            with policy.disable_adapter():
                ra, rb = pair_logps(sub, False)
            accs += ((la - ra) > (lb - rb)).float().cpu().tolist()
        ev[f"acc_{fam}"] = float(np.mean(accs))
    # generation oracle on held-out questions
    gq = [qs[i] for i in te_idx[:N_GEN]]
    prompts = [render_prompt(q) for q in gq]
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
    policy.config.use_cache = True
    gen = policy.generate(**enc, do_sample=False, max_new_tokens=24, pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    nc = nw = no = nexp = 0
    samples = []
    for i, q in enumerate(gq):
        resp = tok.decode(gen[i, P:], skip_special_tokens=True).strip()
        first = resp.split("\n")[0].strip()
        lead = first.split(".")[0].split(",")[0].split(" —")[0].strip().rstrip(".").lower()
        t, f_ = q["t"].lower(), q["f"].lower()
        if lead == t or first.lower().startswith(t): nc += 1
        elif lead == f_ or first.lower().startswith(f_): nw += 1
        else: no += 1
        body = first[len(lead):] if first.lower().startswith(lead) else first
        if len(resp.split()) > len(q["t"].split()) + 2: nexp += 1
        if i < 6: samples.append(resp[:90])
    N = len(gq)
    ev.update(gen_correct=nc / N, gen_wrong=nw / N, gen_other=no / N, gen_explained=nexp / N,
              gen_samples=samples)
    policy.train()
    return ev

hist = dict(labeller=LABELLER, layer=None if LABELLER == "gt" else LI, beta=BETA, lr=LR,
            fam_w=dict(zip(fam_names, fam_w)), loss=[], evals=[])
if LABELLER in ("early", "late"): hist["labeller_acc"] = lab_acc
ev0 = evaluate(); ev0["step"] = 0; hist["evals"].append(ev0)
print(f"  step    0: EVAL { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev0.items() if k!='gen_samples'} }", flush=True)
policy.train()
for step in range(STEPS):
    batch = rgen.sample(train_items, BATCH)
    opt.zero_grad()
    la, lb = pair_logps(batch, True)
    with torch.no_grad(), policy.disable_adapter():
        ra, rb = pair_logps(batch, False)
    D = BETA * ((la - ra) - (lb - rb))
    p = torch.tensor([x["p"] for x in batch], device=DEV, dtype=torch.float32)
    loss = -(p * F.logsigmoid(D) + (1 - p) * F.logsigmoid(-D)).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(float(loss.detach()))
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-10:]):.4f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(); ev["step"] = step + 1; hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev.items() if k!='gen_samples'} }", flush=True)
        json.dump(hist, open(f"/workspace/styc_train_{TAG}_history.json", "w"), indent=1)
json.dump(hist, open(f"/workspace/styc_train_{TAG}_history.json", "w"), indent=1)
policy.save_pretrained(f"/workspace/styc_train_{TAG}_lora")
print("DONE", flush=True)
