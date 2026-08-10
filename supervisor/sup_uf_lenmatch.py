#!/usr/bin/env python
"""A length-matched UltraFeedback set: same schema, same split, length made uninformative.

WHY. Every stage-1 arm installed mostly "prefer the longer completion" — arm A is BELOW chance on
the half of held-out pairs whose better answer is shorter, and scores 0.249 implicit on offsetbias,
which is built so the appealing answer is the rejected one. That leaves the headline comparison
(elbow 0.544 vs deep 0.731) open to a deflationary reading: perhaps the deep arm is simply better
at learning the length rule, and nothing about preference is being measured at all.

The clean way to close that is to remove the rule. Here the pairs are subsampled so that:
  * the chosen side is the longer one in exactly 50% of pairs (the "prefer longer" cheat is worth
    0.500 by construction, against 0.616 in the unmatched file), and
  * within each |Δ tokens| stratum the two directions are equinumerous, so the cheat cannot be
    recovered by conditioning on how big the length difference is either.

Both hold in TRAIN and in the held-out split separately, since the trainer and the eval read the
same file and a match that only held in aggregate would leave the eval exploitable.

This is a subset, not a reweighting: it discards pairs rather than correcting for them (phase 3
used IPW instead). Matching costs sample size, and the manifest records how much.

Usage: python sup_uf_lenmatch.py
Env:   UF_IN=<release>/uf.jsonl  UF_LM_OUT=<release>/uf_lm.jsonl  LM_BIN=25  LM_SEED=0
"""
import json
import os
import random
from collections import defaultdict

E = os.environ.get
HERE = os.path.dirname(os.path.abspath(__file__))
IN = E("UF_IN", f"{HERE}/uf_release/uf.jsonl")
OUT = E("UF_LM_OUT", f"{HERE}/uf_release/uf_lm.jsonl")
BIN = int(E("LM_BIN", 25))          # |Δ tokens| stratum width
SEED = int(E("LM_SEED", 0))


def match(rows, rng):
    """Equal numbers of chosen-longer and chosen-shorter within each |Δ| stratum."""
    strata = defaultdict(lambda: ([], []))
    for r in rows:
        d = r["meta"]["ntok_chosen"] - r["meta"]["ntok_rejected"]
        if d == 0:
            continue                      # ties carry no length signal either way; drop for clarity
        lo, sh = strata[abs(d) // BIN]
        (lo if d > 0 else sh).append(r)
    keep = []
    for k in sorted(strata):
        lo, sh = strata[k]
        n = min(len(lo), len(sh))
        if n == 0:
            continue
        keep += rng.sample(lo, n) + rng.sample(sh, n)
    return keep


def main():
    rows = [json.loads(l) for l in open(IN)]
    rng = random.Random(SEED)
    tr = match([r for r in rows if not r["reserved_for_eval"]], rng)
    ev = match([r for r in rows if r["reserved_for_eval"]], rng)
    out = tr + ev
    out.sort(key=lambda r: r["id"])
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    def stats(rs):
        if not rs:
            return dict(n=0)
        dl = [r["meta"]["ntok_chosen"] - r["meta"]["ntok_rejected"] for r in rs]
        return dict(n=len(rs), chosen_longer_frac=sum(d > 0 for d in dl) / len(dl),
                    mean_abs_delta=sum(abs(d) for d in dl) / len(dl))

    man = dict(source=IN, bin=BIN, seed=SEED,
               before=dict(train=stats([r for r in rows if not r["reserved_for_eval"]]),
                           eval=stats([r for r in rows if r["reserved_for_eval"]])),
               after=dict(train=stats(tr), eval=stats(ev)))
    json.dump(man, open(OUT.replace(".jsonl", "_manifest.json"), "w"), indent=1)
    print(f"[lm] wrote {OUT}: {len(tr)} train / {len(ev)} held out "
          f"(from {man['before']['train']['n']}/{man['before']['eval']['n']})", flush=True)
    for k in ("train", "eval"):
        b, a = man["before"][k], man["after"][k]
        print(f"[lm] {k:6s} chosen-longer {b['chosen_longer_frac']:.3f} -> {a['chosen_longer_frac']:.3f}"
              f" | mean |Δtok| {b['mean_abs_delta']:.0f} -> {a['mean_abs_delta']:.0f}", flush=True)


if __name__ == "__main__":
    main()
