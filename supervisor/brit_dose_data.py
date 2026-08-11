#!/usr/bin/env python
"""Build guard-dosed britishness training files with a holdout that can SEE the guard.

WHY THIS EXISTS. `results_0807` §3 is the finding that reframed that whole run: the release's own
`reserved_for_eval` split is 750 rows that are ALL `lexicon` in `qa` form, half of them the
-ise/-ize contrast, and **all 200 `truth_guard` rows sit in TRAIN**. So the headline 746/750 is
structurally incapable of showing guard damage, and six of the seven families are never evaluated
at all. §6 item 2 queued the fix — "carve culture/style/truth_dialect and some guard rows out of
train" — and it was never run. This builds it.

TWO THINGS VARY, AND THEY MUST NOT BE CONFOUNDED.

  the SPLIT   is fixed across every output file, so all doses are scored on the same rows.
  the DOSE    is the guard's share of the training diet.

DOSING IS DUPLICATION, AND THAT IS THE POINT, NOT A COMPROMISE. `sup_train.py:343` draws each
step's pairs with `rgen.sample(train_rows, PREF_PAIRS)` — uniform over the file — so replicating a
guard row is exactly up-weighting the guard term in expectation. What duplication cannot buy is
DIVERSITY: there are only 199 unique guard facts in the release, ~150 after the holdout, so the
20% arm sees each one about seven times per epoch. That is a real risk of memorising guard facts
instead of the rule, which is precisely why the eval guard rows are held out BY GROUP.

HOLDING OUT BY GROUP, NOT BY ROW, is load-bearing everywhere here. The guard's 200 rows are 25
each in 8 topic groups; two whole groups go to eval, so an eval guard fact is never trained in any
form. Same for the install families: whole `group` values move, so a held-out row turns on an
am|br axis or a topic the fit never saw. Splitting by row would let "prefer British" be answered
by a lookup — `decodability/RESULTS.md` §2 measures a bag-of-token-ids probe at 0.985 on brit
language under a random split against 0.784 under a group split.

THE LEGACY 750 ARE KEPT, unchanged and separately labelled, so every number stays comparable with
`results_0807`'s 746/750 and with his 730/735.

Out: supervisor/britishness/dosed/brit_dose{PCT}.jsonl, readable by `sup_common.load_split` with
     no code change (SUP_BRIT=<path>). Each row keeps an added `eval_bucket` field
     (legacy | guard | install_<family> | None) so an evaluator can slice the holdout by family.

Usage: python supervisor/brit_dose_data.py [--doses 4,20] [--guard-eval-groups 2]
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "britishness", "release", "britishness.jsonl")
OUT = os.path.join(HERE, "britishness", "dosed")
# Fraction of each install family's GROUPS moved to the holdout. The families are tiny
# (expression 124 rows, spelling_control 20), so this is a fraction of groups with a floor of one
# group, not a fraction of rows -- a family that cannot spare a group is left whole and says so.
INSTALL_EVAL_FRAC = 0.20

# What counts as a GROUP for splitting, per family. The release's own `group` field is the right
# key for most families, but two cases break it and both were caught by the leakage check below:
#
#   truth_dialect  the guard rows and the truth_dialect INSTALL rows are crossings of the SAME
#                  199 facts (checked: 199/199 overlap). Splitting the two roles independently by
#                  topic group puts a held-out guard fact's install twin in train, which teaches
#                  the model the very fact it is about to be tested on. The key must be the fact.
#   style /        one `group` value each, so a group-level split is impossible. Their meta
#   expression     carries a usable item key instead (9 style registers, 61 idiom meanings).
#
# spelling_control (20 rows, one group, no item key) is left whole in train and reported as such.
SPLIT_KEY = {
    "truth_dialect": lambda r: r["meta"]["fact"],
    "style": lambda r: str(r["meta"].get("british_index")),
    "expression": lambda r: str(r["meta"].get("meaning")),
}


def gkey(r):
    return SPLIT_KEY.get(r["family"], lambda x: x["group"])(r)


def load():
    return [json.loads(l) for l in open(SRC)]


def build_split(rows, guard_eval_groups, seed):
    """→ (eval_bucket per row id). Deterministic, group-level, and identical for every dose."""
    rng = random.Random(seed)
    bucket = {}

    for i, r in enumerate(rows):
        if r["reserved_for_eval"]:
            bucket[i] = "legacy"

    # truth_dialect FIRST and as ONE family across both roles: pick whole topic groups, then move
    # every row sharing those groups' facts -- guard and install alike -- so no eval fact survives
    # anywhere in train. The two roles still land in different eval buckets.
    tidx = [i for i, r in enumerate(rows) if r["family"] == "truth_dialect"]
    ggroups = sorted({rows[i]["group"] for i in tidx if rows[i]["role"] == "truth_guard"})
    held_g = set(rng.sample(ggroups, guard_eval_groups))
    held_facts = {gkey(rows[i]) for i in tidx if rows[i]["group"] in held_g}
    for i in tidx:
        if gkey(rows[i]) in held_facts:
            bucket[i] = "guard" if rows[i]["role"] == "truth_guard" else "install_truth_dialect"

    # The other install families. `lexicon` is excluded -- the legacy 750 already holds out
    # lexicon/qa, and taking more would re-measure the one family already covered while shrinking
    # the family that carries most of the training mass.
    held_i = {}
    for fam in sorted({r["family"] for r in rows}):
        if fam in ("lexicon", "truth_dialect"):
            continue
        idx = [i for i, r in enumerate(rows)
               if r["family"] == fam and r["role"] == "install" and i not in bucket]
        if not idx:
            continue
        groups = sorted({gkey(rows[i]) for i in idx})
        if len(groups) < 2:
            print(f"  [skip] {fam}: only {len(groups)} split key(s), left whole in train")
            continue
        # Take groups until ~INSTALL_EVAL_FRAC of the family's ROWS are held out, not
        # INSTALL_EVAL_FRAC of its groups. Group sizes are wildly uneven (false_friend: 208 / 88 /
        # 5), so sampling groups uniformly can hand back a 5-row eval set — measured, and useless
        # at that n. Never takes the last group.
        sizes = Counter(gkey(rows[i]) for i in idx)
        order = rng.sample(groups, len(groups))
        hold, held_n = set(), 0
        for gname in order:
            if len(hold) == len(groups) - 1:
                break
            hold.add(gname)
            held_n += sizes[gname]
            if held_n >= INSTALL_EVAL_FRAC * len(idx):
                break
        held_i[fam] = hold
        for i in idx:
            if gkey(rows[i]) in hold:
                bucket[i] = f"install_{fam}"
    return bucket, held_g, held_i


def write_dose(rows, bucket, pct, seed):
    """One file at guard share `pct` of the training diet. → path."""
    train_install = [i for i in range(len(rows))
                     if i not in bucket and rows[i]["role"] != "truth_guard"]
    train_guard = [i for i in range(len(rows))
                   if i not in bucket and rows[i]["role"] == "truth_guard"]
    d = pct / 100.0
    n_want = int(round(len(train_install) * d / (1 - d))) if d < 1 else len(train_guard)

    rng = random.Random(seed + pct)
    draws = []
    while len(draws) < n_want:
        pool = train_guard[:]
        rng.shuffle(pool)
        draws += pool[:n_want - len(draws)]

    out_rows = []
    for i in train_install:
        r = dict(rows[i]); r["reserved_for_eval"] = False; r["eval_bucket"] = None
        out_rows.append(r)
    for n, i in enumerate(draws):
        r = dict(rows[i]); r["reserved_for_eval"] = False; r["eval_bucket"] = None
        # A duplicated row must not collide on id -- anything downstream that dedupes by id would
        # silently undo the dose.
        r["id"] = f"{r['id']}#dup{n}"
        out_rows.append(r)
    for i, b in sorted(bucket.items()):
        r = dict(rows[i]); r["reserved_for_eval"] = True; r["eval_bucket"] = b
        out_rows.append(r)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"brit_dose{pct}.jsonl")
    with open(path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    n_tr = len(train_install) + len(draws)
    print(f"  dose {pct:>3}%  train {n_tr:>5} = {len(train_install)} install + {len(draws)} guard "
          f"({len(set(draws))} unique, x{len(draws) / max(1, len(set(draws))):.1f})  "
          f"actual guard share {len(draws) / n_tr:.1%}  ->  {os.path.relpath(path, HERE)}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doses", default="4,20")
    ap.add_argument("--guard-eval-groups", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load()
    bucket, held_g, held_i = build_split(rows, args.guard_eval_groups, args.seed)
    print(f"source {len(rows)} rows")
    print(f"  guard eval groups : {sorted(held_g)}")
    for fam, h in sorted(held_i.items()):
        print(f"  {fam:<18} eval groups: {sorted(h)}")
    print(f"\nholdout {len(bucket)} rows: {dict(Counter(bucket.values()))}\n")

    for pct in [int(x) for x in args.doses.split(",")]:
        write_dose(rows, bucket, pct, args.seed)

    # The check that matters: no eval GROUP may appear in train, in any dose file.
    print("\nleakage check (eval group must never appear in train):")
    for pct in [int(x) for x in args.doses.split(",")]:
        f = os.path.join(OUT, f"brit_dose{pct}.jsonl")
        rs = [json.loads(l) for l in open(f)]
        ev = defaultdict(set)
        for r in rs:
            if r["reserved_for_eval"] and r["eval_bucket"] != "legacy":
                ev[r["family"]].add(gkey(r))
        bad = [(r["family"], gkey(r)) for r in rs
               if not r["reserved_for_eval"] and gkey(r) in ev.get(r["family"], ())]
        print(f"  dose {pct:>3}%: {len(bad)} leaked rows" + ("  OK" if not bad else "  ** LEAK **"))


if __name__ == "__main__":
    main()
