#!/usr/bin/env python
"""Section-1 sanity gate: does the Bayesian head match the logistic probe at fit time?

Same corpus, same split, same pooling, same layers. If MAP accuracy / AUROC do not match the
logistic probe within noise, everything downstream (pessimism, sequential posteriors, evidence
weights) is measuring a broken reader rather than a better one — so this gate runs first and
its verdict is printed as PASS/FAIL.

Env: LIBON_OUT=/workspace/libon BAYES_STEPS=400 SEED=0
Out: /workspace/libon/bayes_gate.json
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from libon_common import MODEL, LAYERS, DEV, OUT, MAX_NEW, Probes, auroc, render, batched  # noqa
from libon_bayes import BayesProbes, map_accuracy                                          # noqa
from helpers import ResidualCapture                                                        # noqa

E = os.environ.get
BAYES_STEPS = int(E("BAYES_STEPS", 400))
SEED = int(E("SEED", 0))
torch.manual_seed(SEED); np.random.seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
BLOCKS = list(model.model.layers)
HID = model.config.hidden_size
rows = [json.loads(l) for l in open(f"{OUT}/probe_corpus.jsonl")]
split = int(0.85 * len(rows))
tr_rows, te_rows = rows[:split], rows[split:]
print(f"[gate] {MODEL} corpus {len(rows)} train {len(tr_rows)} test {len(te_rows)}", flush=True)


def encode_rows(rows_):
    texts, plens = [], []
    for r in rows_:
        head = render(tok, r["prompt"])
        texts.append(head + r["completion"])
        plens.append(len(tok(head, add_special_tokens=False).input_ids))
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=256 + MAX_NEW).to(DEV)
    T = enc.input_ids.shape[1]
    m = torch.zeros_like(enc.input_ids, dtype=torch.bool)
    for i in range(len(rows_)):
        npad = int(T - enc.attention_mask[i].sum())
        lo = min(npad + plens[i], T - 1)
        m[i, lo:] = enc.attention_mask[i, lo:].bool()
    return enc, m


@torch.no_grad()
def extract(rows_, bs=16):
    A, M, Y = [], [], []
    for chunk in batched(rows_, bs):
        enc, m = encode_rows(chunk)
        with ResidualCapture([BLOCKS[l] for l in LAYERS]) as cap:
            model(**enc)
        buf = cap.get()
        A.append({l: buf[i].detach() for i, l in enumerate(LAYERS)})
        M.append(m); Y += [r["y"] for r in chunk]
    T = max(m.shape[1] for m in M)
    def pad(x, dim=1):
        p = [0] * (2 * x.dim()); p[-(2 * dim + 1)] = T - x.shape[dim]
        return F.pad(x, p)
    return ({l: torch.cat([pad(a[l]) for a in A]) for l in LAYERS},
            torch.cat([pad(m) for m in M]), torch.tensor(Y, device=DEV))


tr_acts, tr_mask, tr_y = extract(tr_rows)
te_acts, te_mask, te_y = extract(te_rows)
yte = te_y.cpu().numpy()

# ---- logistic reference (the probe the reproduction ran on) ----
log_probes = Probes(HID).to(DEV)
log_probes.load_state_dict(torch.load(f"{OUT}/probes_init.pt", map_location=DEV))
with torch.no_grad():
    zl = log_probes.logits(te_acts, te_mask).cpu().numpy()
log_auroc = {int(l): auroc(zl[i], yte) for i, l in enumerate(LAYERS)}
log_acc = {int(l): float(((zl[i] > 0) == (yte == 1)).mean()) for i, l in enumerate(LAYERS)}

# ---- Bayesian head, same data ----
bp = BayesProbes(HID, LAYERS).to(DEV)
bp.set_scale(tr_acts, tr_mask)
info = bp.fit(tr_acts, tr_mask, tr_y, steps=BAYES_STEPS, sequential=False)
with torch.no_grad():
    zb = bp.score(te_acts, te_mask, lam=0.0).cpu().numpy()
bay_auroc = {int(l): auroc(zb[i], yte) for i, l in enumerate(LAYERS)}
bay_acc = {int(l): float(((zb[i] > 0) == (yte == 1)).mean()) for i, l in enumerate(LAYERS)}

d_auroc = {l: bay_auroc[l] - log_auroc[l] for l in log_auroc}
mean_d = float(np.mean(list(d_auroc.values())))
# noise scale: SE of AUROC at this n, roughly 1/sqrt(min(pos,neg))
n_pos, n_neg = int((yte == 1).sum()), int((yte == 0).sum())
se = 1.0 / max(1.0, np.sqrt(min(n_pos, n_neg)))
verdict = "PASS" if abs(mean_d) < 2 * se else "FAIL"

res = dict(model=MODEL, layers=LAYERS, n_test=len(te_rows), n_pos=n_pos, n_neg=n_neg,
           logistic=dict(auroc=log_auroc, acc=log_acc),
           bayes=dict(auroc=bay_auroc, acc=bay_acc,
                      elbo={int(k): v for k, v in bp.elbos().items()},
                      mean_sigma={int(l): info[l]["mean_sigma"] for l in LAYERS},
                      evidence_weights={int(k): v for k, v in bp.evidence_weights().items()}),
           delta_auroc=d_auroc, mean_delta=mean_d, se=float(se), verdict=verdict)
torch.save(bp.state_dict(), f"{OUT}/bayes_init.pt")
json.dump(res, open(f"{OUT}/bayes_gate.json", "w"), indent=1)

print("  layer   logistic   bayes    delta", flush=True)
for l in LAYERS:
    print(f"  L{l:<4d}  {log_auroc[l]:.3f}      {bay_auroc[l]:.3f}    {d_auroc[l]:+.3f}", flush=True)
print(f"\n  mean delta AUROC {mean_d:+.3f} (2*SE = {2*se:.3f})  ->  {verdict}", flush=True)
print(f"  per-layer ELBO {  {int(k): round(v) for k, v in bp.elbos().items()} }", flush=True)
print(f"  evidence weights { {int(k): round(v,3) for k,v in bp.evidence_weights().items()} }",
      flush=True)
print(f"  mean posterior sigma { {l: round(info[l]['mean_sigma'],4) for l in LAYERS} }", flush=True)
