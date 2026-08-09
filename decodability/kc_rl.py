#!/usr/bin/env python
"""(attach - L*) on knowcomp: pooled-probe GRPO with an EXACT oracle.

WHAT THIS IS FOR. The program's question is whether attaching a training signal at the wrong
depth costs you something, measured as a function of (attach - L*). Every previous attempt was
blocked:

  results_0805 §2/§4   EAGLE depth ladders confound depth with READOUT COMPETENCE -- a head that
                       reconstructs final logits is intrinsically better the deeper it sits
                       (measured again 08-09: agreement 0.361 at L5, 0.394 at L9, 0.653 at L17,
                       identical 5000-step budget). A PROBE saturates at L* instead, which is the
                       whole reason it de-confounds.
  goodfire/RESULTS.md  ran pooled-probe GRPO at L4..L24 and could not interpret the result,
                       because AE/BE decodability is 0.90-0.99 at EVERY layer (maximal at L0).
                       L* pinned at 0 => no contrast.
  NEXT.md              "no training direct from probes (Goodharts)" -- already narrowed by
                       goodfire §2/§4/§5: POOLED probe reward into a policy gradient recovers
                       100%+ of oracle performance; what actually hacked was the DENSE per-token
                       advantage (probe -0.06 -> +3.96 while the oracle went to zero markers).
                       The supported rule is: no backprop THROUGH a probe into activations, and
                       no dense per-token advantage replacing completion-level advantage. This
                       script obeys both -- the probe emits one scalar per completion, reachable
                       only through emitted tokens.

knowcomp supplies what was missing: L* 0.28-0.36 (retrieval) vs 0.75-0.86 (computation) on ONE
prompt template, replicated at three scales, and an EXACT oracle -- every item has a ground-truth
answer, where the AE/BE dictionary oracle dropped ~28 of 250 axes as ambiguous.

THE PRIMARY METRIC IS THE ORACLE, NEVER THE PROBE SCORE. That is what caught goodfire's hack, and
it is the only reason a probe reward is safe to use at all.

NO vLLM. gf_rl.py needs it for 192-token prose; knowcomp answers are ~4 tokens, so HF generate is
adequate and we avoid a dependency that would downgrade torch (2.12+cu130 is what makes this
Blackwell card work).

Usage:
  python kc_rl.py --model qwen3-1.7b --family retrieval --layer 10 --steps 60 --tag ret_L10
Out: /workspace/kc_rl/<tag>/history.json
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import dec_common as C          # noqa: E402
import dec_data as D            # noqa: E402

OUT_ROOT = Path(os.environ.get("KC_RL_OUT", "/workspace/kc_rl"))


# ---------------------------------------------------------------- probe
def pooled_resid(ctx, pairs, layer, render="chat", micro=16):
    """Mean-pooled residual at read point `layer` over the COMPLETION tokens only.

    Uses `render_ids` + `left_pad_batch` + the `_pool` span arithmetic rather than re-tokenizing
    `prompt + completion` as one string. dec_common.py:113 documents why: retokenizing across the
    boundary can merge tokens and shift the completion span by one, which silently corrupts the
    mean-pool. This is the same path the cached sweep uses, so a probe fitted here and an L*
    measured there are reading the identical quantity.
    """
    import dec_cache as K
    outs = []
    for s in range(0, len(pairs), micro):
        rows = [C.render_ids(ctx, p, c, render) for p, c in pairs[s:s + micro]]
        buf, npad, plens, T = K._forward_reads(ctx, rows)
        _, mean = K._pool(buf[layer], npad, plens, T)
        outs.append(mean)
    return torch.cat(outs)


def fit_probe(X, y, l2=1.0):
    """Logistic probe, absolute score (not antisymmetric): reward needs a per-completion scalar."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0 / l2)
    clf.fit(X.numpy(), y)
    w = torch.tensor(clf.coef_[0], dtype=torch.float32)
    b = float(clf.intercept_[0])
    return w, b, clf


# ---------------------------------------------------------------- oracle
def normalise(s):
    return "".join(ch for ch in s.strip().lower() if ch.isalnum() or ch.isspace()).strip()


def oracle_correct(gen, truth, distractor):
    """EXACT-match oracle. Ties (neither answer present) score 0.5 -- ties are not losses
    (decodability/NEXT.md trap 1); scoring them 0 would report a strong signal where there is
    none, which is exactly how the 08-07 sweep produced 0.000 columns that meant nothing."""
    g, t, f = normalise(gen), normalise(truth), normalise(distractor)
    ht, hf = t in g, f in g
    if ht and not hf:
        return 1.0
    if hf and not ht:
        return 0.0
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--family", choices=["retrieval", "computation"], required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--reward", choices=["probe", "oracle"], default="probe")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=12)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    run = OUT_ROOT / a.tag
    run.mkdir(parents=True, exist_ok=True)

    d = D.load("knowcomp")
    idx = [i for i, (_, _, _, fam) in enumerate(d.pairs) if fam == a.family]
    tr = [i for i in idx if d.split[i] == "train"]
    te = [i for i in idx if d.split[i] == "test"]
    print(f"[data] {a.family}: {len(tr)} train / {len(te)} held-out", flush=True)

    ctx = C.load(a.model)
    NL = ctx.n_layers
    assert 0 <= a.layer <= NL, f"layer {a.layer} outside 0..{NL}"

    # ---- probe: fitted on TRAIN items only, frozen thereafter.
    texts = [(d.prompts[i], d.variants["correct"][i]) for i in tr] + \
            [(d.prompts[i], d.variants["wrong"][i]) for i in tr]
    y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
    X = pooled_resid(ctx, texts, a.layer)
    w, b, clf = fit_probe(X, y)
    Xte = pooled_resid(ctx, [(d.prompts[i], d.variants["correct"][i]) for i in te] +
                            [(d.prompts[i], d.variants["wrong"][i]) for i in te], a.layer)
    yte = np.r_[np.ones(len(te)), np.zeros(len(te))]
    probe_acc = float(clf.score(Xte.numpy(), yte))
    print(f"[probe] L{a.layer} held-out accuracy {probe_acc:.3f}  "
          f"(this is the COMPETENCE COVARIATE -- report it beside every depth number)", flush=True)
    w_gpu = w.to(ctx.device)  # noqa: F841  (consumed by the GRPO loop, not yet written)

    json.dump(dict(model=a.model, family=a.family, layer=a.layer, n_layers=NL,
                   probe_heldout_acc=probe_acc, reward=a.reward, args=vars(a)),
              open(run / "config.json", "w"), indent=1)
    print(f"[setup] wrote {run}/config.json", flush=True)


if __name__ == "__main__":
    main()
