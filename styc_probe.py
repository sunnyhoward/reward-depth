#!/usr/bin/env python
"""Style x Correctness factorial testbed (styc), stage A: where does each FACTOR of a preference
become linearly decodable, and does a depth-ensemble of Bayesian heads route by difficulty?

Motivation (2026-07-30, phase 7 §8): the UF probe reads the style-legible part of the preference
and is blind to execution correctness -- but UF entangles the two factors in unknown proportions.
This testbed disentangles them by construction.

Design. Each question gets FOUR templated responses -- {correct, wrong} x {terse, explained}:
  terse:      "{answer}"
  explained:  arithmetic: "{answer}. {a} plus {b} equals {answer}."   (fluent, asserts the sum)
              know:       "{answer}. This is a well-established fact." (fluent, content-free)
The explained template is IDENTICAL across correctness, so style carries no correctness signal;
verifying the explained-wrong response requires actually computing/knowing the answer.

Pair sets:
  CORR      style-matched, correctness differs   (c,e)v(w,e) and (c,t)v(w,t)
  STYLE     correctness-matched, style differs   (c,e)v(c,t) and (w,e)v(w,t)
  CONFLICT  (c,terse) vs (w,explained) -- correctness says left, style says right
Preference for the mixed PREF probe: lexicographic, correctness dominates, style breaks ties.

Outputs (/workspace/styc_*.json, results/ after banking):
  - correctness-decodability and style-decodability vs depth (the two curves)
  - per-layer PREF heads + uniform / evidence-softmax / precision-weighted ensembles,
    evaluated on CORR / STYLE / CONFLICT held-out pairs
  - routing: precision-weight center of mass over depth, per pair type
Env: MODEL=Qwen/Qwen2.5-3B N_ARITH=500 N_SUM=300 SEED=0"""
import os, sys, json, random
import numpy as np
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import make_q, KNOW_BANK, ResidualCapture, train_bayes_head

E = os.environ.get
MODEL = E("MODEL", "Qwen/Qwen2.5-3B")
N_ARITH, N_SUM, SEED = int(E("N_ARITH", 500)), int(E("N_SUM", 300)), int(E("SEED", 0))
DEV = "cuda"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- questions: arithmetic (computation) + sum-compare (computation) + know (retrieval) ----
rng = random.Random(SEED + 1)
qs, seen = [], set()
while sum(1 for q in qs if q["typ"] == "mcq_arith") < N_ARITH:
    q = make_q("mcq_arith", rng)
    if q and q["q"] not in seen: seen.add(q["q"]); qs.append(q)
while sum(1 for q in qs if q["typ"] == "sum") < N_SUM:
    q = make_q("sum", rng)
    if q and q["q"] not in seen: seen.add(q["q"]); qs.append(q)
for kq, t, f in KNOW_BANK:
    qs.append(dict(typ="know", q=kq, t=t, f=f))
rng.shuffle(qs)
print(f"[data] {len(qs)} questions "
      f"({sum(q['typ']=='mcq_arith' for q in qs)} arith, {sum(q['typ']=='sum' for q in qs)} sum, "
      f"{sum(q['typ']=='know' for q in qs)} know)", flush=True)

def explain(q, ans):
    if q["typ"] == "mcq_arith":
        a, b = q["q"].split("What is ")[1].rstrip("?").split("+")
        return f"{ans}. {a.strip()} plus {b.strip()} equals {ans}."
    if q["typ"] == "sum":
        return f"{ans}. Computing both sums shows that {ans} is the larger one."
    return f"{ans}. This is a well-established fact."
def variants(q):
    return {("c", "t"): q["t"], ("w", "t"): q["f"],
            ("c", "e"): explain(q, q["t"]), ("w", "e"): explain(q, q["f"])}
def render(q, resp): return f"Question: {q['q']}\nAnswer: {resp}"

# ---- features: last-token residuals at every layer for all 4 variants ----
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
BLOCKS = list(model.model.layers); NL = len(BLOCKS); HID = model.config.hidden_size
KEYS = [("c", "t"), ("w", "t"), ("c", "e"), ("w", "e")]
cachef = "/workspace/styc_feats.npz"
if os.path.exists(cachef):
    FE = {k: v for k, v in np.load(cachef).items()}
else:
    print("[feats] caching...", flush=True)
    FE = {}
    for key in KEYS:
        texts = [render(q, variants(q)[key]) for q in qs]
        out = np.zeros((len(texts), NL, HID), np.float16)
        for s in range(0, len(texts), 32):
            enc = tok(texts[s:s + 32], return_tensors="pt", padding=True).to(DEV)
            with torch.no_grad(), ResidualCapture(BLOCKS) as cap:
                model(**enc)
            buf = cap.get()
            for li in range(NL):
                out[s:s + len(enc.input_ids), li] = buf[li][:, -1].float().cpu().numpy()
        FE["".join(key)] = out
        print(f"  cached {key}", flush=True)
    np.savez(cachef, **FE)
n = len(qs)
tr = np.arange(n) % 5 != 0   # 80/20 split by question (all pairs of a question share the split)
te = ~tr
typ = np.array([q["typ"] for q in qs])

def head_curve(Fa, Fb, tag):
    """Per-layer Bayesian heads on (Fa - Fb) difference features; returns acc curve + heads."""
    accs, heads, elbos = [], [], []
    rngh = np.random.RandomState(SEED)
    s_tr = np.where(rngh.rand(int(tr.sum())) < 0.5, 1.0, -1.0).astype(np.float32)
    s_te = np.where(rngh.rand(int(te.sum())) < 0.5, 1.0, -1.0).astype(np.float32)
    for li in range(NL):
        A, B = Fa[:, li].astype(np.float32), Fb[:, li].astype(np.float32)
        pool = np.concatenate([A[tr], B[tr]]); sd = pool.std(0) + 1e-6
        a, h, e = train_bayes_head(((A[tr] - B[tr]) / sd) * s_tr[:, None], s_tr,
                                   ((A[te] - B[te]) / sd) * s_te[:, None], s_te)
        accs.append(float(a)); heads.append((h, sd)); elbos.append(float(e))
    print(f"[{tag}] max {max(accs):.3f} @L{int(np.argmax(accs))} | "
          f"curve {[round(a,2) for a in accs[::4]]}", flush=True)
    return accs, heads, elbos

# ---- the two factor curves (style-matched correctness; correctness-matched style) ----
corr_acc_e, _, _ = head_curve(FE["ce"], FE["we"], "CORR|explained")
corr_acc_t, _, _ = head_curve(FE["ct"], FE["wt"], "CORR|terse")
style_acc_c, _, _ = head_curve(FE["ce"], FE["ct"], "STYLE|correct")
style_acc_w, _, _ = head_curve(FE["we"], FE["wt"], "STYLE|wrong")

# per-type correctness curves (retrieval vs computation)
corr_by_typ = {}
for t in ["know", "mcq_arith", "sum"]:
    m = typ == t
    trm, tem = tr & m, te & m
    accs = []
    rngh = np.random.RandomState(SEED)
    s_tr = np.where(rngh.rand(int(trm.sum())) < 0.5, 1.0, -1.0).astype(np.float32)
    s_te = np.where(rngh.rand(int(tem.sum())) < 0.5, 1.0, -1.0).astype(np.float32)
    for li in range(NL):
        A, B = FE["ce"][:, li].astype(np.float32), FE["we"][:, li].astype(np.float32)
        pool = np.concatenate([A[trm], B[trm]]); sd = pool.std(0) + 1e-6
        a, _, _ = train_bayes_head(((A[trm] - B[trm]) / sd) * s_tr[:, None], s_tr,
                                   ((A[tem] - B[tem]) / sd) * s_te[:, None], s_te)
        accs.append(float(a))
    corr_by_typ[t] = accs
    print(f"[CORR|{t}] max {max(accs):.3f} @L{int(np.argmax(accs))} | {[round(a,2) for a in accs[::4]]}", flush=True)

# ---- PREF heads (lexicographic preference, mixed pair diet) + ensembles on held-out ----
# preferred vs dispreferred pairs: CORR(e), CORR(t), STYLE(c), STYLE(w), CONFLICT
pair_sets = dict(corr_e=(FE["ce"], FE["we"]), corr_t=(FE["ct"], FE["wt"]),
                 style_c=(FE["ce"], FE["ct"]), style_w=(FE["we"], FE["wt"]),
                 conflict=(FE["ct"], FE["we"]))
# train diet: all but conflict (conflict is the generalization test)
P_ens, W_prec, accs_pref, elbos_pref = {}, {}, [], []
heads_pref = []
rngh = np.random.RandomState(SEED + 7)
def fit_pref(li):
    As, Bs = [], []
    for k in ["corr_e", "corr_t", "style_c", "style_w"]:
        Fa, Fb = pair_sets[k]
        As.append(Fa[tr, li].astype(np.float32)); Bs.append(Fb[tr, li].astype(np.float32))
    A, B = np.concatenate(As), np.concatenate(Bs)
    s = np.where(rngh.rand(len(A)) < 0.5, 1.0, -1.0).astype(np.float32)
    pool = np.concatenate([A, B]); sd = pool.std(0) + 1e-6
    a, h, e = train_bayes_head(((A - B) / sd) * s[:, None], s,
                               ((A - B) / sd)[:64] * s[:64, None], s[:64])
    return h, sd, e
for li in range(NL):
    h, sd, e = fit_pref(li)
    heads_pref.append((h, sd)); elbos_pref.append(float(e))
print("[pref] heads fit", flush=True)

def score(li, Fa, Fb, m):
    h, sd = heads_pref[li]
    fs = torch.tensor(((Fa[m, li] - Fb[m, li]).astype(np.float32)) / sd)
    MU = h.mu.detach().float(); SIG2 = F.softplus(h.rho.detach()).float().pow(2)
    s2 = fs.pow(2).matmul(SIG2)
    z = fs.matmul(MU) / torch.sqrt(1 + s2)
    return torch.special.ndtr(z).numpy(), (1.0 / (1.0 + s2)).numpy()

res = dict(curves=dict(corr_explained=corr_acc_e, corr_terse=corr_acc_t,
                       style_correct=style_acc_c, style_wrong=style_acc_w,
                       corr_by_type=corr_by_typ, elbos_pref=elbos_pref))
w_ev = np.exp((np.array(elbos_pref) - max(elbos_pref)) / 50.0); w_ev /= w_ev.sum()
res["ens"] = {}
for k, (Fa, Fb) in pair_sets.items():
    P = np.zeros((NL, int(te.sum()))); PR = np.zeros_like(P)
    for li in range(NL):
        P[li], PR[li] = score(li, Fa, Fb, te)
    single = {f"L{li}": float((P[li] > 0.5).mean()) for li in [10, 20, 30]}
    ens = dict(uniform=float((P.mean(0) > 0.5).mean()),
               evidence=float(((w_ev[:, None] * P).sum(0) > 0.5).mean()),
               precision=float(((PR * P).sum(0) / PR.sum(0) > 0.5).mean()))
    com = float((PR / PR.sum(0)).T.dot(np.arange(NL)).mean())   # precision center of mass
    res["ens"][k] = dict(singles=single, ensembles=ens, prec_center_of_mass=com)
    print(f"[ens {k:9s}] singles {single} | ens {ens} | weight-CoM L{com:.1f}", flush=True)
json.dump(res, open("/workspace/styc_stageA.json", "w"), indent=1)
print("DONE", flush=True)
