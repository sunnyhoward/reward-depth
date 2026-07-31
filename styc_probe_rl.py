#!/usr/bin/env python
"""styc: training FROM the probe during generation — no DPO anywhere. (2026-07-31, follows
notes_dense_probe_rl.md; pre-registered predictions there.)

Three arms (MODE):
  shaped        on-policy PG. Reward = pooled-probe read of the FROZEN base over the sampled
                text (mean over answer tokens -> probe z, LCB-pessimistic). Dense credit via the
                running-mean potential Phi_t = probe(mean of frozen states over tokens 1..t):
                A_t = (Phi_end - b_LOO) + SHAPE_W*(Phi_end - Phi_t). Guards: DPOP anchor,
                KL-in-reward. No gradient ever flows through policy activations into the probe.
  seqrl         SHAPE_W=0 — sequence-level pooled reward only. Isolates what density adds.
  pooled_margin the direct-activation control (user option b): teacher-forced pairs, POLICY
                activations pooled over answer tokens at layer LI, lag-1 adaptive mean-diff
                margin (phase-6 recipe) + DPOP anchor. Backprops through policy activations —
                the forging channel is OPEN by design; z_frozen vs proj_self separates
                behaviour-change from forging.

Labeller/reward probe: entangled pref head fit on the POOLED cache (natural mixed diet,
conflicts 15%), layer LI. Pooled prefix features at generation time are near the probe's
training distribution (unlike completion-end probes on prefixes).

Env: MODE=shaped STEPS=200 BATCH=4 K=4 LR=1e-5 EVAL_EVERY=25 N_GEN=64 SEED=0 LI=35
     SHAPE_W=1 PESS=0.5 KLR=0.03 DPOP=1.0 MAXNEW=32 MARGIN_LR=1e-4 M0=4.0
     STYC_CACHE=/workspace/styc_feats_3B_mean.npz RUN_TAG=<mode>
Outputs: /workspace/styc_prl_{TAG}_history.json, _lora"""
import os, sys, json, random, hashlib, re
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import make_q, KNOW_BANK, ResidualCapture, train_bayes_head

E = os.environ.get
MODEL = E("MODEL", "Qwen/Qwen2.5-3B")
MODE = E("MODE", "shaped")
STEPS, BATCH, K = int(E("STEPS", 200)), int(E("BATCH", 4)), int(E("K", 4))
LR = float(E("LR", 1e-5))
EVAL_EVERY, N_GEN, SEED = int(E("EVAL_EVERY", 25)), int(E("N_GEN", 64)), int(E("SEED", 0))
N_ARITH, LI = int(E("N_ARITH", 500)), int(E("LI", 35))
SHAPE_W = float(E("SHAPE_W", 1.0)) if MODE == "shaped" else 0.0
PESS, KLR, DPOP = float(E("PESS", 0.5)), float(E("KLR", 0.03)), float(E("DPOP", 1.0))
MAXNEW = int(E("MAXNEW", 32))
MARGIN_LR, M0 = float(E("MARGIN_LR", 1e-4)), float(E("M0", 4.0))
CACHEF = E("STYC_CACHE", "/workspace/styc_feats_3B_mean.npz")
TAG = E("RUN_TAG", MODE)
DEV = "cuda"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- questions + v2 templates (mirror styc_probe.py / styc_train.py; SEED+1 => same set) ----
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
tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
print(f"[data] {n} questions | train {int(tr.sum())} | test {int(te.sum())} | mode {MODE}", flush=True)

# ---- reward probe from the POOLED cache (natural protocol, conflicts 15%) ----
z = np.load(CACHEF); FE = {k: z[k] for k in z.files}
assert FE["ct"].shape[0] == n, "cache/question-set mismatch — regenerate the pooled cache"
diets = ([("ce","we")]*6 + [("ct","wt")]*4 + [("ce","ct")]*4 + [("we","wt")]*3 + [("ct","we")]*3)
As = np.concatenate([FE[a][tr, LI] for a,_ in diets]).astype(np.float32)
Bs = np.concatenate([FE[b][tr, LI] for _,b in diets]).astype(np.float32)
s_ = np.where(np.random.RandomState(LI).rand(len(As)) < 0.5, 1.0, -1.0).astype(np.float32)
SD = (np.concatenate([As, Bs]).std(0) + 1e-6)
Av = np.concatenate([FE[a][te, LI] for a,_ in diets]).astype(np.float32)
Bv = np.concatenate([FE[b][te, LI] for _,b in diets]).astype(np.float32)
sv = np.where(np.random.RandomState(LI + 999).rand(len(Av)) < 0.5, 1.0, -1.0).astype(np.float32)
_, HEAD, _ = train_bayes_head(((As - Bs) / SD) * s_[:, None], s_, ((Av - Bv) / SD) * sv[:, None], sv)
MU = HEAD.mu.detach().float().to(DEV); SIG2 = F.softplus(HEAD.rho.detach()).float().pow(2).to(DEV)
SDt = torch.tensor(SD, device=DEV)
def probe_z(feats):          # feats [.., HID] raw pooled residuals -> LCB z
    f = feats.float() / SDt
    s2 = f.pow(2).matmul(SIG2)
    zz = f.matmul(MU) / torch.sqrt(1 + s2)
    return zz - PESS * torch.sqrt(s2 / (1 + s2))
PAIRS = {"corr_e": ("ce", "we"), "corr_t": ("ct", "wt"), "style_c": ("ce", "ct"),
         "style_w": ("we", "wt"), "conflict": ("ct", "we")}
lab_acc = {}
for k, (a, b) in PAIRS.items():
    fa = torch.tensor(FE[a][te_idx, LI].astype(np.float32), device=DEV)
    fb = torch.tensor(FE[b][te_idx, LI].astype(np.float32), device=DEV)
    lab_acc[k] = float((probe_z(fa) > probe_z(fb)).float().mean())
print(f"[reward probe @L{LI} pooled] held-out acc by family: { {k: round(v,3) for k,v in lab_acc.items()} }", flush=True)

# ---- model/policy ----
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
policy = get_peft_model(model, cfg); policy.config.use_cache = False
params = [p for p in policy.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=(MARGIN_LR if MODE == "pooled_margin" else LR))
print(f"[lora] {sum(p.numel() for p in params)/1e6:.1f}M trainable | lr {opt.param_groups[0]['lr']}", flush=True)

def batch_logps(texts, plens, grad, per_token=False):
    enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
    with (torch.enable_grad() if grad else torch.no_grad()):
        lsm = F.log_softmax(policy(**enc).logits[:, :-1].float(), -1)
        ids, am = enc.input_ids, enc.attention_mask
        T = ids.shape[1]
        out = []
        for i in range(len(texts)):
            npad = int(T - am[i].sum())
            lo = npad + min(plens[i], int(am[i].sum()) - 1)
            lp = lsm[i, lo - 1:T - 1].gather(-1, ids[i, lo:, None]).squeeze(-1)
            out.append(lp if per_token else lp.sum())
        return out if per_token else torch.stack(out)

@torch.no_grad()
def frozen_pooled_phi(texts, plens):
    """FROZEN read: residuals at LI over answer tokens; returns per-text (Phi_t vector, Phi_end).
    Phi_t = LCB probe z of the running mean over answer tokens 1..t."""
    with policy.disable_adapter():
        enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
        with ResidualCapture([BLOCKS[LI]]) as cap:
            policy(**enc)
        res = cap.get()[0].float()          # [B, T, H]
    am = enc.attention_mask; T = enc.input_ids.shape[1]
    phis, ends = [], []
    for i in range(len(texts)):
        npad = int(T - am[i].sum())
        lo = npad + plens[i]
        seq = res[i, lo:]                    # answer-token states
        if len(seq) == 0: seq = res[i, -1:]
        cm = seq.cumsum(0) / torch.arange(1, len(seq) + 1, device=DEV).unsqueeze(1)
        zz = probe_z(cm)                     # Phi_1..Phi_end
        phis.append(zz); ends.append(zz[-1])
    return phis, torch.stack(ends)

@torch.no_grad()
def evaluate(step):
    policy.eval(); ev = {}
    for fam, (a, b) in PAIRS.items():
        accs = []
        for s in range(0, len(te_idx), 16):
            items = te_idx[s:s + 16]
            texts, plens = [], []
            for i in items:
                q = qs[i]; v = variants(q); pl = len(tok(render_prompt(q)).input_ids)
                texts += [render(q, v[a]), render(q, v[b])]; plens += [pl, pl]
            la = batch_logps(texts, plens, False)
            with policy.disable_adapter():
                ra = batch_logps(texts, plens, False)
            d = (la - ra).view(-1, 2)
            accs += (d[:, 0] > d[:, 1]).float().cpu().tolist()
        ev[f"acc_{fam}"] = float(np.mean(accs))
    gq = [qs[i] for i in te_idx[:N_GEN]]
    prompts = [render_prompt(q) for q in gq]
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
    policy.config.use_cache = True
    gen = policy.generate(**enc, do_sample=False, max_new_tokens=MAXNEW, pad_token_id=tok.pad_token_id)
    policy.config.use_cache = False
    P = enc.input_ids.shape[1]
    nc = nw = no = nexp = 0; samples = []
    for i, q in enumerate(gq):
        resp = tok.decode(gen[i, P:], skip_special_tokens=True).strip()
        first = resp.split("\n")[0].strip().lower()
        t, f_ = q["t"].lower(), q["f"].lower()
        has_t = re.search(r"\b" + re.escape(t) + r"\b", first) is not None
        has_f = re.search(r"\b" + re.escape(f_) + r"\b", first) is not None
        if has_t and not has_f: nc += 1
        elif has_f and not has_t: nw += 1
        else: no += 1
        if len(resp.split()) > len(q["t"].split()) + 2: nexp += 1
        if i < 6: samples.append(resp[:90])
    N = len(gq)
    ev.update(gen_correct=nc / N, gen_wrong=nw / N, gen_other=no / N, gen_explained=nexp / N,
              gen_samples=samples, step=step)
    # no-forging meter: frozen pooled z on fixed held-out ce/we pairs
    fa = torch.tensor(FE["ce"][te_idx[:64], LI].astype(np.float32), device=DEV)
    fb = torch.tensor(FE["we"][te_idx[:64], LI].astype(np.float32), device=DEV)
    ev["z_frozen_gap"] = float((probe_z(fa) - probe_z(fb)).mean())
    policy.train(); return ev

hist = dict(mode=MODE, layer=LI, labeller_acc=lab_acc, shape_w=SHAPE_W, pess=PESS, klr=KLR,
            dpop=DPOP, lr=opt.param_groups[0]["lr"], loss=[], reward=[], evals=[])
ev0 = evaluate(0); hist["evals"].append(ev0)
print(f"  step    0: EVAL { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev0.items() if k!='gen_samples'} }", flush=True)
policy.train()
rgen = random.Random(SEED + 4242)
u_prev = None   # lag-1 adaptive direction for pooled_margin

for step in range(STEPS):
    opt.zero_grad()
    if MODE in ("shaped", "seqrl"):
        qs_b = [qs[i] for i in rgen.sample(list(tr_idx), BATCH)]
        prompts = [render_prompt(q) for q in qs_b for _ in range(K)]
        enc = tok(prompts, return_tensors="pt", padding=True).to(DEV)
        policy.config.use_cache = True
        with torch.no_grad():
            gen = policy.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                                  max_new_tokens=MAXNEW, pad_token_id=tok.pad_token_id)
        policy.config.use_cache = False
        P = enc.input_ids.shape[1]
        resps = [tok.decode(gen[i, P:], skip_special_tokens=True).strip() for i in range(len(prompts))]
        texts = [render(qs_b[i // K], resps[i]) for i in range(len(resps))]
        plens = [len(tok(render_prompt(qs_b[i // K])).input_ids) for i in range(len(resps))]
        phis, ends = frozen_pooled_phi(texts, plens)
        # KL-in-reward (sequence mean, frozen ref)
        lp_pol = batch_logps(texts, plens, False)
        with policy.disable_adapter():
            lp_ref = batch_logps(texts, plens, False)
        ntok = torch.tensor([max(1, len(p)) for p in phis], device=DEV, dtype=torch.float32)
        R = ends - KLR * (lp_pol - lp_ref) / ntok
        b_loo = (R.view(BATCH, K).sum(1, keepdim=True) - R.view(BATCH, K)) / (K - 1)
        A_seq = (R.view(BATCH, K) - b_loo).view(-1)
        lps = batch_logps(texts, plens, True, per_token=True)
        pg = []
        for i in range(len(texts)):
            lp = lps[i]
            L = min(len(lp), len(phis[i]))
            if L == 0: continue
            A_t = A_seq[i] + SHAPE_W * (phis[i][-1] - phis[i][:L]).detach()
            pg.append(-(A_t * lp[:L]).sum() / L)
        loss_pg = torch.stack(pg).mean()
        # DPOP anchor on the GT-preferred rendering (stability guard, not the objective)
        atexts = [render(q, variants(q)["ce"]) for q in qs_b]
        aplens = [len(tok(render_prompt(q)).input_ids) for q in qs_b]
        la = batch_logps(atexts, aplens, True)
        with torch.no_grad(), policy.disable_adapter():
            ra = batch_logps(atexts, aplens, False)
        loss = loss_pg + DPOP * F.relu(ra - la).mean()
        hist["reward"].append(float(ends.mean()))
    else:  # pooled_margin — direct activation optimization, forging channel open by design
        items = rgen.sample(list(tr_idx), 8)
        texts, plens = [], []
        for i in items:
            q = qs[i]; v = variants(q); pl = len(tok(render_prompt(q)).input_ids)
            texts += [render(q, v["ce"]), render(q, v["we"])]; plens += [pl, pl]
        enc = tok(texts, return_tensors="pt", padding=True).to(DEV)
        with ResidualCapture([BLOCKS[LI]]) as cap:
            policy(**enc)
        res = cap.get()[0].float()
        am = enc.attention_mask; T = enc.input_ids.shape[1]
        pooled = []
        for i in range(len(texts)):
            npad = int(T - am[i].sum()); lo = npad + plens[i]
            pooled.append(res[i, lo:].mean(0) if lo < T else res[i, -1])
        Fp = torch.stack(pooled); Fa, Fb = Fp[0::2], Fp[1::2]
        d = ((Fa - Fb) / SDt)
        with torch.no_grad():
            u_now = d.mean(0); u_now = u_now / (u_now.norm() + 1e-6)
        u = u_prev if u_prev is not None else u_now
        u_prev = u_now
        proj = d.matmul(u)
        loss_m = F.relu(M0 - proj).mean()
        atexts = [render(qs[i], variants(qs[i])["ce"]) for i in items]
        aplens = [len(tok(render_prompt(qs[i])).input_ids) for i in items]
        la = batch_logps(atexts, aplens, True)
        with torch.no_grad(), policy.disable_adapter():
            ra = batch_logps(atexts, aplens, False)
        loss = loss_m + DPOP * F.relu(ra - la).mean()
        hist["reward"].append(float(proj.mean().detach()))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    hist["loss"].append(float(loss.detach()))
    if (step + 1) % 10 == 0:
        print(f"  step {step+1:4d}: loss {np.mean(hist['loss'][-10:]):.4f} | "
              f"r {np.mean(hist['reward'][-10:]):.3f}", flush=True)
    if (step + 1) % EVAL_EVERY == 0:
        ev = evaluate(step + 1); hist["evals"].append(ev)
        print(f"  step {step+1:4d}: EVAL { {k: (round(v,3) if isinstance(v,float) else v) for k,v in ev.items() if k!='gen_samples'} }", flush=True)
        json.dump(hist, open(f"/workspace/styc_prl_{TAG}_history.json", "w"), indent=1)
json.dump(hist, open(f"/workspace/styc_prl_{TAG}_history.json", "w"), indent=1)
policy.save_pretrained(f"/workspace/styc_prl_{TAG}_lora")
print("DONE", flush=True)
