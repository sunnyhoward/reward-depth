#!/usr/bin/env python
"""Family B step 1: distil early-exit readout heads, on GENERIC text only.

THE PROTOCOL IS THE POINT. eagle/RESULTS.md §8 established that `head_acc` measured with a
TRAINABLE head is not an encoding measure -- the head can build the preference itself, and then
"decodable at L" says nothing about L. So here the head is distilled purely as a general
early-exit language model, on the base model's OWN samples, and NEVER sees a preference pair.
Family B then scores pairs with the head frozen and zero preference fitting. Whatever accuracy
comes out is attributable to h_L plus the base's own unembedding, not to the readout having been
taught the task.

Head = readout module at h_L -> frozen final_norm -> frozen lm_head, distilled to the base
model's final logits by forward KL (teacher = base). Every module zero-inits its output
projection, so training starts from an exact early exit and can only move away from it by
learning something.

CORPUS. Generative replay from the frozen base itself (eagle_replay.py's design, reimplemented
here because that script is hard-wired to Qwen2.5-3B): 25% of sequences start from BOS, 75% from
a uniformly sampled 1-8 token non-special prefix, sampled at T=1.0 with no top-k/top-p so the
corpus is a genuine sample rather than a mode-seeking one. Distilling only on narrow templated
task text leaves the head incompetent off-distribution -- measured at eagle_replay.py:9-12 as
KL(base||head) 1.74 at answer positions vs 0.58 elsewhere.

ONE BACKWARD AT A TIME. Heads for all grid layers share one base forward pass, but each head's
loss is backward-ed and freed before the next head runs. NEXT_0806.md:75-77 records 84 GiB
allocated at step 0 from summing two graphs before backward; with an fp32 151,936-wide vocab the
same mistake here would be worse.

Usage: python dec_distill.py <model_key> <arch>[,<arch>...]
Env:   DIST_STEPS=400 DIST_BATCH=8 DIST_LR=1e-3 DIST_SEED=0 REPLAY_N=2048 REPLAY_T=128
"""
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dec_common as C  # noqa: E402
from dec_heads import DEC_ARCHS, make_dec_head, n_params  # noqa: E402

E = os.environ.get
STEPS = int(E("DIST_STEPS", 400))
BATCH = int(E("DIST_BATCH", 8))
LR = float(E("DIST_LR", 1e-3))
SEED = int(E("DIST_SEED", 0))
REPLAY_N = int(E("REPLAY_N", 2048))
REPLAY_T = int(E("REPLAY_T", 128))
CHUNK = int(E("DIST_CHUNK", 32))     # time-chunk for the fp32 vocab softmax


def replay_path(model_key):
    return os.path.join(C.DEC_ROOT, f"replay_{model_key}_{REPLAY_N}x{REPLAY_T}.pt")


def head_path(model_key, arch, layer):
    return os.path.join(C.DEC_ROOT, "heads", f"{model_key}_{arch}_L{layer}.pt")


# ── replay corpus ─────────────────────────────────────────────────────────────────────────────

def build_replay(ctx, seed=SEED):
    out = replay_path(ctx.key)
    if os.path.exists(out):
        return torch.load(out).long()
    os.makedirs(C.DEC_ROOT, exist_ok=True)
    rg = random.Random(seed + 11)
    special = set(ctx.tok.all_special_ids)
    ordinary = [i for i in range(ctx.model.config.vocab_size) if i not in special]
    torch.manual_seed(seed)
    chunks, done = [], 0
    t0 = time.time()
    while done < REPLAY_N:
        k = min(64, REPLAY_N - done)
        pre = []
        for _ in range(k):
            if ctx.tok.bos_token_id is not None and rg.random() < 0.25:
                pre.append([ctx.tok.bos_token_id])
            else:
                pre.append([rg.choice(ordinary) for _ in range(rg.randint(1, 8))])
        T0 = max(len(p) for p in pre)
        ids = torch.full((k, T0), ctx.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((k, T0), dtype=torch.long)
        for i, p in enumerate(pre):          # LEFT pad, as everywhere else
            ids[i, T0 - len(p):] = torch.tensor(p)
            att[i, T0 - len(p):] = 1
        with torch.no_grad():
            g = ctx.model.generate(input_ids=ids.to(ctx.device), attention_mask=att.to(ctx.device),
                                   do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                   max_new_tokens=REPLAY_T, min_new_tokens=REPLAY_T,
                                   pad_token_id=ctx.tok.pad_token_id)
        chunks.append(g[:, -REPLAY_T:].cpu())
        done += k
        print(f"[replay] {done}/{REPLAY_N}  ({time.time()-t0:.0f}s)", flush=True)
    corpus = torch.cat(chunks)[:REPLAY_N]
    torch.save(corpus.to(torch.int32), out)
    print(f"[replay] wrote {out} {tuple(corpus.shape)}", flush=True)
    return corpus.long()


# ── distillation ──────────────────────────────────────────────────────────────────────────────

def _kl_chunked(teacher_logits, student_logits, chunk=CHUNK):
    """Mean forward KL(teacher || student) over positions, chunked over time.

    Chunking is not an optimisation, it is what keeps a (B, T, 151936) fp32 log_softmax off the
    peak: NEXT.md:60-61 records three concurrent such jobs exhausting the 95 GB card.
    """
    T = teacher_logits.shape[1]
    tot = 0.0
    for s in range(0, T, chunk):
        tl = teacher_logits[:, s:s + chunk].float()
        sl = student_logits[:, s:s + chunk].float()
        lt = F.log_softmax(tl, -1)
        ls = F.log_softmax(sl, -1)
        tot = tot + (lt.exp() * (lt - ls)).sum(-1).mean() * (min(chunk, T - s) / T)
    return tot


def distill(model_key, arch, layers=None, steps=STEPS, force=False):
    ctx = C.load(model_key)
    for p in ctx.model.parameters():
        p.requires_grad_(False)
    layers = layers or C.layer_grid(ctx.n_layers)
    os.makedirs(os.path.join(C.DEC_ROOT, "heads"), exist_ok=True)
    todo = [L for L in layers if force or not os.path.exists(head_path(model_key, arch, L))]
    if not todo:
        print(f"[distill] {model_key}/{arch}: all layers present", flush=True)
        return
    corpus = build_replay(ctx)

    heads, opts = {}, {}
    for L in todo:
        h = make_dec_head(arch, ctx.hid, n_heads=ctx.n_heads, dtype=ctx.dtype,
                          vocab=ctx.model.config.vocab_size).to(ctx.device)
        if hasattr(h, "attach_lm_head"):
            h.attach_lm_head(ctx.model)      # tffree: start EXACTLY at the pinned head
        heads[L] = h
        opts[L] = torch.optim.AdamW(h.parameters(), lr=LR)
    print(f"[distill] {model_key} arch={arch} layers={todo} "
          f"{n_params(heads[todo[0]])/1e6:.1f}M params each  steps={steps}", flush=True)

    torch.manual_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    hist = {L: [] for L in todo}
    t0 = time.time()
    for step in range(steps):
        sel = torch.randint(0, corpus.shape[0], (BATCH,), generator=g)
        ids = corpus[sel].to(ctx.device)
        att = torch.ones_like(ids)
        with torch.no_grad(), C.ResidualCapture(ctx.read_mods) as cap:
            teacher = ctx.model(input_ids=ids, attention_mask=att).logits
        buf = cap.get()
        for L in todo:
            h = buf[L].detach()
            opts[L].zero_grad(set_to_none=True)
            loss = _kl_chunked(teacher, heads[L](h, ctx.model, pad_mask=att))
            loss.backward()                  # one graph alive at a time (NEXT_0806.md:75-77)
            # Gradient clipping. Without it the ATTENTION-bearing heads diverge at 8B (hid 4096):
            # eagle-tf and eagle-2l held KL ~0.05 for ~80 steps and then blew up to 4.9 and 8.8 by
            # step 400, i.e. training made them worse than the zero-init early exit they started
            # from. A diverged head is not a measurement, and its competence covariate is garbage.
            # Clipping binds only where a run was diverging, so the already-converged cells are
            # unaffected and the grid stays comparable.
            torch.nn.utils.clip_grad_norm_(heads[L].parameters(), 1.0)
            opts[L].step()
            hist[L].append(float(loss.detach()))
            del loss
        del teacher, buf
        if step % 50 == 0:
            msg = "  ".join(f"L{L}:{np.mean(hist[L][-50:]):.3f}" for L in todo)
            print(f"  step {step:4d}  KL(base||head)  {msg}   ({time.time()-t0:.0f}s)", flush=True)

    for L in todo:
        torch.save(dict(state=heads[L].state_dict(), arch=arch, layer=L, model=model_key,
                        n_params=n_params(heads[L]), kl_final=float(np.mean(hist[L][-50:])),
                        steps=steps, seed=SEED), head_path(model_key, arch, L))
    C.bank(f"distill_{model_key}_{arch}", dict(
        model=model_key, arch=arch, layers=todo, steps=steps, batch=BATCH, lr=LR, seed=SEED,
        n_params=n_params(heads[todo[0]]),
        kl_curve={str(L): [float(x) for x in hist[L][::10]] for L in todo},
        kl_final={str(L): float(np.mean(hist[L][-50:])) for L in todo},
        replay=dict(n=REPLAY_N, t=REPLAY_T),
        note="distilled on generative replay only; never saw a preference pair"))
    del heads, opts, ctx
    torch.cuda.empty_cache()


if __name__ == "__main__":
    mk = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
    archs = (sys.argv[2] if len(sys.argv) > 2 else "eagle-mlp,eagle-tf").split(",")
    for a in archs:
        assert a in DEC_ARCHS, f"unknown arch {a}; known {list(DEC_ARCHS)}"
    for a in archs:
        distill(mk, a)
