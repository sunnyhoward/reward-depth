#!/usr/bin/env python
"""Stage 0 of the Libon port: probe corpus + initial per-layer probes. See NOTE.md.

Paper Appendix A: the probe-training completions are generated ONCE by the base (reference)
model and are "on-policy only at initialization". Refits later re-extract activations of THIS
fixed labelled corpus under the current model — labels never change, activations move. So this
script runs once and its output is the anchor for every regime.

Writes:
  probe_corpus.jsonl    prompt + completion + label (harmful=1)   [NOT committed — see NOTE.md]
  probes_init.pt        initial per-layer linear probes
  probes_init.json      per-layer held-out AUROC

Env: LIBON_MODEL=mistralai/Mistral-7B-Instruct-v0.1 LIBON_LAYERS=0,6,12,18,24,30
     N_PROBE=768 PROBE_STEPS=1500 PROBE_BS=32 MAX_NEW=96 SEED=0
Out: /workspace/libon/
"""
import os, sys, json, random, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from libon_common import (MODEL, LAYERS, DEV, OUT, MAX_NEW, Probes, auroc,  # noqa: E402
                          render, batched)
from helpers import ResidualCapture                                          # noqa: E402

E = os.environ.get
N_PROBE = int(E("N_PROBE", 768))
PROBE_STEPS, PROBE_BS = int(E("PROBE_STEPS", 1500)), int(E("PROBE_BS", 32))
PROBE_LR, PROBE_WD = float(E("PROBE_LR", 1e-3)), float(E("PROBE_WD", 1e-4))
WARMUP = int(E("WARMUP", 50))
SEED = int(E("SEED", 0))
os.makedirs(OUT, exist_ok=True)
CORPUS = f"{OUT}/probe_corpus.jsonl"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
BLOCKS = list(model.model.layers)
HID, NL = model.config.hidden_size, len(BLOCKS)
assert max(LAYERS) < NL, f"layer {max(LAYERS)} >= depth {NL}"
print(f"[libon-prepare] {MODEL} layers={NL} hid={HID} probe layers {LAYERS}", flush=True)


# ── 1. probe corpus: base-model completions on WildGuardTest, its labels as concept labels ──
if os.path.exists(CORPUS):
    rows = [json.loads(l) for l in open(CORPUS)]
    print(f"[corpus] loaded {len(rows)}", flush=True)
else:
    ds = load_dataset("walledai/WildGuardTest")["train"]
    pool = [dict(prompt=r["prompt"], y=1 if r["label"] == "harmful" else 0)
            for r in ds if r["label"] in ("harmful", "unharmful")]
    rg = random.Random(SEED); rg.shuffle(pool)
    harm = [r for r in pool if r["y"] == 1][:N_PROBE // 2]
    ben = [r for r in pool if r["y"] == 0][:N_PROBE // 2]
    sel = harm + ben; rg.shuffle(sel)
    rows = []
    for chunk in batched(sel, 16):
        enc = tok([render(tok, r["prompt"]) for r in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=256).to(DEV)
        with torch.no_grad():
            model.config.use_cache = True
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                               max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
            model.config.use_cache = False
        P = enc.input_ids.shape[1]
        for i, r in enumerate(chunk):
            comp = tok.decode(g[i, P:], skip_special_tokens=True)
            rows.append(dict(prompt=r["prompt"], completion=comp, y=r["y"]))
        print(f"  corpus {len(rows)}/{len(sel)}", flush=True)   # SAFETY: counts only, no text
    with open(CORPUS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[corpus] wrote {CORPUS} n={len(rows)} harmful={sum(r['y'] for r in rows)}", flush=True)


# ── 2. activation extraction over COMPLETION tokens only ──
def encode_rows(rows_):
    """(input_ids, attn, completion_mask) with left padding; mask covers completion tokens."""
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
def extract(rows_, mdl, bs=16):
    """-> acts {layer: (N, T, H) bf16 on GPU}, mask (N, T), y (N,). Ragged T padded per batch,
    so batches are concatenated on a common max length."""
    A, M, Y = [], [], []
    for chunk in batched(rows_, bs):
        enc, m = encode_rows(chunk)
        with ResidualCapture([BLOCKS[l] for l in LAYERS]) as cap:
            mdl(**enc)
        buf = cap.get()
        A.append({l: buf[i].detach() for i, l in enumerate(LAYERS)})
        M.append(m); Y += [r["y"] for r in chunk]
    T = max(m.shape[1] for m in M)
    def pad(x, T_, dim):
        p = [0] * (2 * x.dim()); p[-(2 * dim + 1)] = T_ - x.shape[dim]
        return F.pad(x, p)
    acts = {l: torch.cat([pad(a[l], T, 1) for a in A]) for l in LAYERS}
    mask = torch.cat([pad(m, T, 1) for m in M])
    return acts, mask, torch.tensor(Y, device=DEV)


split = int(0.85 * len(rows))
tr_rows, te_rows = rows[:split], rows[split:]
print(f"[extract] train {len(tr_rows)} test {len(te_rows)}", flush=True)
tr_acts, tr_mask, tr_y = extract(tr_rows, model)
te_acts, te_mask, te_y = extract(te_rows, model)
print(f"[extract] done, T={tr_mask.shape[1]}, completion tokens/row "
      f"{float(tr_mask.float().sum(-1).mean()):.1f}", flush=True)


# ── 3. fit the initial probes (paper §3.2 / Appendix A optimiser) ──
def fit_probes(acts, mask, y, steps, probes=None, lr=PROBE_LR, seed=0):
    pr = probes if probes is not None else Probes(HID).to(DEV)
    opt = torch.optim.AdamW(pr.parameters(), lr=lr, weight_decay=PROBE_WD)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, WARMUP)) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))))
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = mask.shape[0]
    for s in range(steps):
        idx = torch.randint(0, N, (min(PROBE_BS, N),), generator=g).to(DEV)
        z = pr.logits({l: acts[l][idx] for l in LAYERS}, mask[idx])       # (L, B)
        loss = F.binary_cross_entropy_with_logits(
            z, y[idx].float().unsqueeze(0).expand_as(z))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pr.parameters(), 1.0)
        opt.step(); sch.step()
    return pr, float(loss)


probes, last = fit_probes(tr_acts, tr_mask, tr_y, PROBE_STEPS)
with torch.no_grad():
    z = probes.logits(te_acts, te_mask).cpu().numpy()
res = {int(l): auroc(z[i], te_y.cpu().numpy()) for i, l in enumerate(LAYERS)}
torch.save(probes.state_dict(), f"{OUT}/probes_init.pt")
json.dump(dict(model=MODEL, layers=LAYERS, auroc=res, probe_steps=PROBE_STEPS,
               n_corpus=len(rows), final_loss=last), open(f"{OUT}/probes_init.json", "w"), indent=1)
print("[probes] held-out AUROC per layer:", {k: round(v, 3) for k, v in res.items()}, flush=True)
print(f"[probes] wrote {OUT}/probes_init.pt\nDONE — libon_train.py next", flush=True)
