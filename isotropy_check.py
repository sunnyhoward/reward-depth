#!/usr/bin/env python
"""Replicate the paper's unembedding-isotropy probe on our backbone: center + length-normalize the
rows of the unembedding W, project onto RAND_K random directions, rescale variances by d — for a
perfectly spherical row cloud every rescaled variance is 1. Reports the std around 1, raw and with
the top-1/top-10 principal components removed. Near-isotropy is the premise behind 'a single
reward head at the top induces ≈ DPO's gradient flow'; measure it, don't assume it.

Run: python isotropy_check.py [model_name]   (default Qwen/Qwen2.5-7B; CPU is fine)
"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B"
RAND_K = 16384
rng = np.random.default_rng(0)

print(f"loading {MODEL} (weights only, CPU) ...")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map="cpu")
W = model.get_output_embeddings().weight.detach().numpy()   # (|V|, d)
V, d = W.shape
print(f"unembedding: |V|={V}, d={d}")

Wc = W - W.mean(0, keepdims=True)
Wn = Wc / (np.linalg.norm(Wc, axis=1, keepdims=True) + 1e-9)

def spread(rows):
    P = rng.standard_normal((rows.shape[1], RAND_K)).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    v = (rows @ P).var(0) * rows.shape[1]      # rescaled variance; sphere ⇒ all ≈ 1
    return float(v.std())

print(f"no whitening       : std around 1 = {100*spread(Wn):.1f}%")
for k in (1, 10):
    U, S, Vt = np.linalg.svd(Wn, full_matrices=False)
    Wk = Wn - (U[:, :k] * S[:k]) @ Vt[:k]
    Wk /= (np.linalg.norm(Wk, axis=1, keepdims=True) + 1e-9)
    print(f"top-{k:>2} PCs removed : std around 1 = {100*spread(Wk):.1f}%")
print("(paper reference: GPT-2 4.8% raw / 4.2% top-1 / 3.8% top-10; Pythia-70M 8.1/7.0/5.4)")
