#!/usr/bin/env python
"""The depth ladder of an addition: when does each named intermediate become decodable?

WHAT THIS IS FOR. `cc_strata.py` showed the styc corr_* curve is not a mixture of item
populations, and `cc_when_computed.py` located the late jump at the block where the model's own
sum reaches the unembedding (4B: lens moves at L25-L26, probe at L27). Both leave the same
question open: from 0.42 to 0.72 depth a fitted probe reads correctness at 0.77 out of a
representation whose sum the model cannot yet express at all. Something is there. This names the
candidates and dates each one.

THE LADDER. An addition has a forced dependency order, so the intermediates can be listed in
advance rather than discovered:

    operands            a, b and their digits                       (input, must be at L0)
    column sums         s_u = a%10 + b%10,  s_t = a//10 + b//10     (one add each, no interaction)
    the carry           carry = s_u >= 10                           (a threshold on s_u)
    output digits       units = s_u % 10                            (needs s_u, NOT the carry)
                        tens  = (s_t + carry) % 10                  (needs the carry)
                        hund  = (s_t + carry) >= 10

`units` and `carry` are BOTH functions of `s_u` alone, so if the model computes `s_u` first they
should date together and after `s_u`. `tens` cannot precede `carry`. A measured order that
violates the dependency graph is evidence the model is not doing this decomposition at all --
which is a real possible outcome, not a failure of the experiment.

PROMPT ONLY. The read point is the last token of the chat generation prompt, with NO completion
in the input: no answer, right or wrong, appears anywhere. So this measures what the model has
computed by depth L, not what a probe can extract from a written answer. That is the difference
between this and every family-A number in the repo.

WHY 4000 ITEMS AND NOT STYC'S 500. The targets are 10- and 19-way; 500 items (400 per training
fold) against d=2560 does not support that fit. The item distribution is `helpers.make_q`'s
exactly -- a, b ~ U[10, 99] -- so the styc arithmetic items are a sample from the same
population; they are marked in the bank so the ladder can be re-scored on them alone.

THE FLOORS ARE NOT OPTIONAL (README). Two per target: the majority-class rate, and a
bag-of-token-ids probe with no model at all. The bag floor matters here because the prompt
CONTAINS the operand digits -- Qwen3 tokenizes digit-by-digit -- so `a%10` is trivially in the
input and must not be reported as a finding. It is included as the ladder's bottom rung for
exactly that reason: a target the model does not need to compute.

Usage:  python cc_ladder.py <model_key> [--n 4000] [--bs 64] [--cache-only]
Env:    DEC_SEEDS=0,1,2  FOLDS=5
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dec_common as C  # noqa: E402

E = os.environ.get
SEEDS = [int(x) for x in E("DEC_SEEDS", "0,1,2").split(",")]
FOLDS = int(E("FOLDS", 5))

# (name, fn(a, b) -> class, n_classes, rung, blurb). `rung` is the a-priori dependency level, used
# only for ordering the report -- nothing in the fit knows about it.
TARGETS = [
    ("a_units",   lambda a, b: a % 10,                          10, 0, "operand digit (in the prompt)"),
    ("b_units",   lambda a, b: b % 10,                          10, 0, "operand digit (in the prompt)"),
    ("s_units",   lambda a, b: (a % 10) + (b % 10),             19, 1, "column sum a%10 + b%10"),
    ("s_tens",    lambda a, b: (a // 10) + (b // 10),           19, 1, "column sum a//10 + b//10"),
    ("carry",     lambda a, b: int((a % 10) + (b % 10) >= 10),   2, 2, "carry out of the units column"),
    ("ans_units", lambda a, b: (a + b) % 10,                    10, 2, "units digit = s_units % 10"),
    ("ans_tens",  lambda a, b: ((a + b) // 10) % 10,            10, 3, "tens digit (needs the carry)"),
    ("ans_hund",  lambda a, b: int(a + b >= 100),                2, 3, "hundreds digit (carry out of tens)"),
]


def make_pairs(n, seed=0):
    """Deterministic unique (a, b) draws from make_q's distribution, a, b ~ U[10, 99]."""
    rng = np.random.RandomState(seed)
    seen, out = set(), []
    while len(out) < n:
        a, b = int(rng.randint(10, 100)), int(rng.randint(10, 100))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


def fold_of(pairs, folds=FOLDS, salt="ccladder"):
    return np.array([int(hashlib.sha1(f"{salt}|{a}+{b}".encode()).hexdigest()[:8], 16) % folds
                     for a, b in pairs])


def cache_path(model_key, n, render):
    return os.path.join(C.DEC_ROOT, f"ccladder_{model_key}_{n}_{render}.npz")


# ── activations: last PROMPT token, no completion anywhere ────────────────────────────────────

@torch.no_grad()
def build_cache(model_key, pairs, render="chat", bs=64, force=False):
    out = cache_path(model_key, len(pairs), render)
    if os.path.exists(out) and not force:
        print(f"[cache] exists: {out}", flush=True)
        return out
    ctx = C.load(model_key)
    n, R = len(pairs), ctx.n_reads
    H = np.zeros((n, R, ctx.hid), np.float16)
    ids_seen = []
    for s in range(0, n, bs):
        sl = slice(s, min(s + bs, n))
        # `render_ids` requires a non-empty completion, so a single throwaway token is appended
        # and then EXCLUDED: the read point is plen-1, the last prompt token, which never attends
        # to anything after it under causal masking. Nothing about the answer enters the read.
        rows = [C.render_ids(ctx, f"Question: What is {a}+{b}?\nAnswer:", "0", render)
                for a, b in pairs[sl]]
        rows = [(r[0][:r[1]], r[1]) for r in rows]
        ids, att, npad, plens = C.left_pad_batch(rows, ctx.tok.pad_token_id, ctx.device)
        with C.ResidualCapture(ctx.read_mods) as cap:
            ctx.model(input_ids=ids, attention_mask=att)
        buf = cap.get()
        for k in range(R):
            h = buf[k]
            idx = torch.tensor([int(npad[b]) + int(plens[b]) - 1 for b in range(h.shape[0])],
                               device=h.device)
            H[sl, k] = h[torch.arange(h.shape[0], device=h.device), idx].float().cpu().numpy()
        del buf
        if s == 0:
            ids_seen = rows[0][0]
        if s % (bs * 10) == 0:
            print(f"   {s}/{n}", flush=True)
    os.makedirs(C.DEC_ROOT, exist_ok=True)
    np.savez(out, H=H, pairs=np.array(pairs), n_reads=R,
             example_ids=np.array(ids_seen), example=ctx.tok.decode(ids_seen))
    print(f"[cache] wrote {out} ({os.path.getsize(out)/1e9:.2f} GB)\n"
          f"        read point = last token of: {ctx.tok.decode(ids_seen)!r}", flush=True)
    del ctx
    torch.cuda.empty_cache()
    return out


# ── layer-batched multinomial logistic ────────────────────────────────────────────────────────

def fit_multiclass_batched(X_tr, y_tr, X_te, y_te, K, seed=0, dev="cuda", epochs=300,
                           patience=30, lr=3e-3, wd=1e-2, bs=512, eval_every=10):
    """X(L, N, d) float32, y(N) int. → per-layer held-out accuracy (L,).

    Same shape of fit as `dec_fit.fit_bayes_batched` -- one problem per layer, summed loss so no
    gradient crosses the layer axis, early stopping PER LAYER on held-out cross-entropy -- but
    multinomial rather than the pairwise probit head, because these targets are not preferences.
    """
    torch.manual_seed(seed)
    A = torch.as_tensor(X_tr, dtype=torch.float32, device=dev)
    B = torch.as_tensor(X_te, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
    yv = torch.as_tensor(y_te, dtype=torch.long, device=dev)
    L, N, d = A.shape
    W = torch.zeros(L, d, K, device=dev, requires_grad=True)
    b0 = torch.zeros(L, 1, K, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b0], lr=lr, weight_decay=wd)

    def ce(X, y):
        lg = torch.baddbmm(b0, X, W)                       # (L, n, K)
        return F.cross_entropy(lg.reshape(-1, K), y.repeat(L), reduction="none") \
            .view(L, -1).mean(1)

    best = torch.full((L,), 1e9, device=dev)
    wait = torch.zeros(L, device=dev)
    done = torch.zeros(L, dtype=torch.bool, device=dev)
    sW, sb = W.detach().clone(), b0.detach().clone()
    for ep in range(epochs):
        for sl in torch.randperm(N, device=dev).split(bs):
            opt.zero_grad()
            ce(A[:, sl], yt[sl]).sum().backward()
            opt.step()
        if ep % eval_every == 0 or ep == epochs - 1:
            with torch.no_grad():
                v = ce(B, yv)
                imp = (v < best - 1e-4) & ~done
                best = torch.where(imp, v, best)
                wait = torch.where(imp, torch.zeros_like(wait), wait + 1)
                sW = torch.where(imp[:, None, None], W.detach(), sW)
                sb = torch.where(imp[:, None, None], b0.detach(), sb)
                done = done | (wait >= max(1, patience // eval_every))
                if bool(done.all()):
                    break
    with torch.no_grad():
        pred = torch.baddbmm(sb, B, sW).argmax(-1)         # (L, M)
        hit = (pred == yv[None, :])
    return hit.cpu().numpy(), ep + 1


def cv_hits(X, y, fid, K, seed):
    """Group 5-fold: every item scored held-out exactly once. X(N, R, d) → hits (R, N) bool."""
    R = X.shape[1]
    hit = np.zeros((R, len(y)), bool)
    for f in range(FOLDS):
        te = np.flatnonzero(fid == f)
        tr = np.flatnonzero(fid != f)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6                # (R, d), train stats only
        h, _ = fit_multiclass_batched(((X[tr] - mu) / sd).transpose(1, 0, 2), y[tr],
                                      ((X[te] - mu) / sd).transpose(1, 0, 2), y[te], K, seed=seed)
        hit[:, te] = h
    return hit


# ── the no-model floor ────────────────────────────────────────────────────────────────────────

def bag_floor(ctx_tok_ids, y, fid, K, seed=0):
    """Bag-of-token-ids probe on the PROMPT, no model involved. → held-out accuracy (scalar).

    Non-negotiable here specifically because the prompt contains the operand digits: Qwen3
    tokenizes digit-by-digit, so `a_units` is literally in the input and any depth statement about
    it would be about the tokenizer. A bag also discards position, which is what makes it the
    right floor -- it has the digits but cannot tell which column they are in.
    """
    ids = sorted({t for row in ctx_tok_ids for t in row})
    col = {t: i for i, t in enumerate(ids)}
    Xb = np.zeros((len(ctx_tok_ids), len(ids)), np.float32)
    for i, row in enumerate(ctx_tok_ids):
        for t in row:
            Xb[i, col[t]] += 1.0
    hit = cv_hits(Xb[:, None, :], y, fid, K, seed)
    return float(hit.mean())


def run(model_key, n=4000, render="chat", bs=64):
    pairs = make_pairs(n)
    p = build_cache(model_key, pairs, render, bs)
    z = np.load(p, allow_pickle=True)
    X = z["H"].astype(np.float32)
    pr = [tuple(map(int, t)) for t in z["pairs"]]
    R = X.shape[1]
    fid = fold_of(pr)

    # The bag floor needs the prompt tokens but not the model, so it is tokenized here rather
    # than carried through the activation cache.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.model_spec(model_key).hf)
    bag_ids = [tok(f"Question: What is {a}+{b}?\nAnswer:", add_special_tokens=False)["input_ids"]
               for a, b in pr]

    results = {}
    for name, fn, K, rung, blurb in TARGETS:
        y = np.array([fn(a, b) for a, b in pr])
        # classes are remapped to 0..K'-1 so an unused class cannot inflate the chance line
        uniq = np.unique(y)
        ymap = np.searchsorted(uniq, y)
        Kp = len(uniq)
        maj = float(np.bincount(ymap).max() / len(ymap))
        accs = np.stack([cv_hits(X, ymap, fid, Kp, s).mean(1) for s in SEEDS])
        floor = bag_floor(bag_ids, ymap, fid, Kp)
        a = accs.mean(0)
        # L* against the FLOOR, not against chance: for the operand digits the floor is the whole
        # story, and a depth measured from chance would report the tokenizer as a computation.
        base = max(maj, floor)
        star = next((k for k in range(R) if a[k] >= base + 0.5 * (1 - base)), None)
        results[name] = dict(rung=rung, blurb=blurb, n_classes=Kp, chance=1.0 / Kp,
                             majority=maj, bag_floor=floor,
                             lstar_half=None if star is None else star / (R - 1),
                             acc_mean=a.tolist(), acc_sd=accs.std(0).tolist())
        print(f"  {name:<10s} rung{rung}  K={Kp:<3d} maj={maj:.3f} bag={floor:.3f}  "
              f"top={a[-1]:.3f}  L*(half)={'--' if star is None else f'{star/(R-1):.3f}'}   "
              f"{blurb}", flush=True)

    payload = dict(model=model_key, n_items=n, render=render, seeds=SEEDS, folds=FOLDS,
                   n_reads=R, n_layers=C.model_spec(model_key).n_layers,
                   frac_depth=(np.arange(R) / (R - 1)).tolist(),
                   layer_index_convention="0 = embedding output, i = output of block i-1",
                   read_point="last token of the chat generation prompt; no completion in the "
                              "input, so no answer (right or wrong) is ever visible",
                   example_prompt=str(z["example"]),
                   results=results)
    C.bank(f"ccladder_{model_key}_{n}_{render}", payload)
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--render", default="chat")
    ap.add_argument("--cache-only", action="store_true")
    a = ap.parse_args()
    if a.cache_only:
        build_cache(a.model, make_pairs(a.n), a.render, a.bs)
    else:
        run(a.model, n=a.n, render=a.render, bs=a.bs)
