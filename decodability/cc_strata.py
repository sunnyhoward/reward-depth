#!/usr/bin/env python
"""Why computation-correctness resolves in TWO steps: stratify the depth curve by item type.

THE QUESTION. `results/decodability/plots/depth_meaningful.png` shows styc `corr_e`/`corr_t` as
the only families that start at chance, and on 4B/8B they do not rise once -- they rise to a
plateau near 0.87 at ~0.45 depth and then jump again to ~1.0 at ~0.78. A single "the model
computes the sum" story predicts ONE step. Two steps says the family is a MIXTURE of item
populations with different resolution depths.

WHAT THE MIXTURE IS. `load_styc` builds 579 items: 500 two-digit additions and 79 retrieval
questions from KNOW_BANK. `helpers.make_q` draws the arithmetic distractor as
`a + b + offset`, `offset ∈ {-10,-3,-2,-1,1,2,3,10}` uniformly, which partitions the arithmetic
items by what a reader must compute to reject the wrong answer:

  units  |offset| < 10   the wrong answer has a DIFFERENT last digit -- (a+b) mod 10 suffices,
                         no carry, no magnitude.                                 371/500
  tens   |offset| = 10   the wrong answer has the SAME last digit -- only the tens digit
                         separates them, so the carry must actually be resolved.  129/500
  know                   retrieval; `knowcomp` puts this at L* ~ 0.3.                  79

If a fraction f of items is ordered and the rest sits at chance, pairwise accuracy is 0.5 + f/2.
know + units = 0.777 of the set → a plateau at 0.889, then 1.0 once `tens` resolves. Both match
the observed curve. HYPOTHESIS: the first step is mod-10 addition, the second is the carry.

WHY THE STRATA ARE IDENTIFIABLE. The competing story is a magnitude/plausibility feature ("the
answer is implausibly far off"). `offset = ±10` is BOTH the same-units-digit case AND the largest
absolute error, so the two hypotheses make OPPOSITE predictions on the same subgroup: under
units-first, ±10 resolves LAST; under magnitude, ±10 resolves FIRST (it is the easiest to spot).
`acc_by_offset` in the banked output is that test, and it needs no new model runs.

THE FIT IS NOT STRATIFIED, ONLY THE SCORING. One probe per (layer, fold, seed), fitted on all
families' pairs exactly as `dec_scalar.py` fits them; the strata are subsets of its held-out
predictions. Refitting per stratum would change the probe and the early-stop point together, and
the curves would no longer be comparable to each other or to the banked aggregate.

GROUP K-FOLD, NOT THE BANKED 80/20 SPLIT. The banked split leaves ~116 held-out pairs per family,
so `tens` would be ~26 items -- an SE of ~0.10, wider than the effect. 5-fold over the same
question-level groups scores EVERY item held-out (n = 371/129/79) and is what makes the per-item
resolution-depth histogram possible at all. `--check-split` refits the banked split and prints it
beside the CV aggregate; they should agree to a couple of points.

INHERITED OPTIMISM. `fit_bayes_batched` early-stops on the same held-out set it scores (the
~1.6-point bias noted in decodability/README.md). Unchanged here on purpose -- it applies to
every stratum alike, so the CONTRASTS are clean even though the levels carry the usual bias.

Usage:  python cc_strata.py <model_key> [--read mean] [--families corr_e,corr_t] [--check-split]
Env:    DEC_SEEDS=0,1,2  FOLDS=5
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dec_cache as K  # noqa: E402
import dec_common as C  # noqa: E402
import dec_data as D  # noqa: E402
import dec_fit as FIT  # noqa: E402
import dec_scalar as S  # noqa: E402

E = os.environ.get
SEEDS = [int(x) for x in E("DEC_SEEDS", "0,1,2").split(",")]
FOLDS = int(E("FOLDS", 5))
STRATA = ["know", "units", "tens"]
RD_KEYS = STRATA + ["carry", "nocarry", "units_carry", "units_nocarry", "tens_carry",
                    "tens_nocarry"]


# ── item strata ───────────────────────────────────────────────────────────────────────────────

def styc_strata(d):
    """→ (labels (N,) str, offsets (N,) int, carry (N,) bool). Retrieval items: offset 0, no carry.

    Recovered post-hoc rather than plumbed through `load_styc`, so no banked styc number changes:
    the question text carries `a` and `b`, and the terse variants carry the two answers verbatim.
    `wt` minus `ct` is exactly the offset `make_q` drew.

    TWO INDEPENDENT DIFFICULTY AXES, and they are not the same question:
      offset  — what the READER must compute to reject this particular distractor.
      carry   — what the MODEL must compute to produce the sum at all, `(a%10 + b%10) >= 10`,
                a property of the item and not of the distractor. Roughly half the arithmetic
                items, and orthogonal to `offset` by construction, so the two cross cleanly.
    """
    lab, off, car = [], [], []
    for i, m in enumerate(d.meta):
        if m["typ"] != "mcq_arith":
            lab.append("know")
            off.append(0)
            car.append(False)
            continue
        a, b = (int(x) for x in d.keys[i].split("What is ")[1].rstrip("?").split("+"))
        t = int(d.variants["ct"][i].strip().rstrip("."))
        w = int(d.variants["wt"][i].strip().rstrip("."))
        assert a + b == t, f"item {i}: {a}+{b} != {t}"
        o = w - t
        # |o| < 10 always changes the last digit (o and o+10k differ mod 10 for 0 < |o| < 10),
        # including the cases that cross a ten boundary (42 → 39 differs in BOTH digits).
        lab.append("units" if abs(o) % 10 else "tens")
        off.append(o)
        car.append((a % 10) + (b % 10) >= 10)
    return np.array(lab), np.array(off), np.array(car)


def fold_of(keys, folds=FOLDS, salt="ccstrata"):
    """Deterministic group-wise fold id, same hashing discipline as `dec_data._group_split`."""
    import hashlib
    return np.array([int(hashlib.sha1(f"{salt}|{k}".encode()).hexdigest()[:8], 16) % folds
                     for k in keys])


# ── the sweep ─────────────────────────────────────────────────────────────────────────────────

def cv_margins(feats, d, family, read, seed, folds=FOLDS):
    """Group 5-fold held-out margins for every item at every read point. → z (R, N).

    z > 0 is a correct ordering, z == 0 a tie (identical difference vector, which happens at the
    embedding read point when both completions end in the same token).
    """
    va, vb = D.STYC_FAMILIES[family]
    A = feats[f"{va}__{read}"].astype(np.float32)          # (N, R, d)
    B = feats[f"{vb}__{read}"].astype(np.float32)
    DF = A - B
    N, R, _ = A.shape
    fid = fold_of(d.keys, folds)
    z = np.zeros((R, N), np.float32)
    for f in range(folds):
        te = np.flatnonzero(fid == f)
        tr = np.flatnonzero(fid != f)
        sd = np.concatenate([A[tr], B[tr]]).std(0) + 1e-6  # (R, d), TRAIN statistics only
        out = FIT.fit_bayes_batched((DF[tr] / sd).transpose(1, 0, 2),
                                    (DF[te] / sd).transpose(1, 0, 2), seed=seed, dev=FIT_DEV)
        z[:, te] = out["z"]
        print(f"      fold {f}: n_te={len(te)} epochs={out['epochs_run']}", flush=True)
    return z


def acc_of(z, mask):
    """Pairwise accuracy over a subset of items, ties scored 0.5. z (R, N) → (R,)."""
    zz = z[:, mask]
    return ((zz > 0).astype(np.float64) + 0.5 * (zz == 0)).mean(1)


def resolution_depth(correct):
    """First read point after which an item is NEVER wrong again. correct (S, R, N) bool → (N,).

    Reported as a read-point index in 0..R, where R means "never resolved". The suffix condition
    (not "first correct") is what makes it a depth rather than a coin flip: at shallow layers an
    item is right half the time by chance, and the first such accident is not a resolution.
    Majority vote over seeds first, so one unlucky fit cannot move an item.
    """
    maj = correct.mean(0) >= 0.5                            # (R, N)
    R = maj.shape[0]
    # cumulative "all True from here up", scanned from the top.
    suffix = np.ones_like(maj)
    acc = np.ones(maj.shape[1], bool)
    for r in range(R - 1, -1, -1):
        acc &= maj[r]
        suffix[r] = acc
    idx = np.where(suffix.any(0), suffix.argmax(0), R)
    return idx


def build_masks(lab, off, car):
    """Every subset scored in the banked output. Crosses are what test whether the two axes are
    the same axis: if `tens` is just `carry` in disguise, `units_carry` behaves like `tens`."""
    masks = {s: lab == s for s in STRATA}
    masks["all"] = np.ones(len(lab), bool)
    arith = lab != "know"
    for a in (1, 2, 3, 10):
        masks[f"off{a}"] = np.abs(off) == a
    masks["carry"] = arith & car
    masks["nocarry"] = arith & ~car
    for s in ("units", "tens"):
        masks[f"{s}_carry"] = (lab == s) & car
        masks[f"{s}_nocarry"] = (lab == s) & ~car
    return masks


def margin_path(model_key, dataset, render, read, family):
    return os.path.join(C.DEC_ROOT, f"ccmargins_{model_key}_{dataset}_{render}_{read}_{family}.npz")


def run(model_key, dataset="styc", read="mean", families=("corr_e", "corr_t"), render="chat",
        check_split=False, reuse=False):
    d = D.load(dataset)
    feats = K.load_feats(model_key, dataset, render)
    lab, off, car = styc_strata(d)
    n_layers = C.model_spec(model_key).n_layers
    masks = build_masks(lab, off, car)

    results = {}
    for family in families:
        print(f"  [{family}] read={read} seeds={SEEDS} folds={FOLDS}", flush=True)
        # The margins are the expensive part and every stratum is a subset of them, so they are
        # persisted: a new item property (carry, magnitude, template) is then a re-score, not a
        # refit, and the probes it is scored against are byte-identical to the banked ones.
        mp = margin_path(model_key, dataset, render, read, family)
        if reuse and os.path.exists(mp):
            zs = np.load(mp)["z"].astype(np.float32)
            print(f"      reusing margins {mp}", flush=True)
        else:
            zs = np.stack([cv_margins(feats, d, family, read, s) for s in SEEDS])   # (S, R, N)
            os.makedirs(C.DEC_ROOT, exist_ok=True)
            np.savez_compressed(mp, z=zs.astype(np.float16), seeds=np.asarray(SEEDS),
                                fold=fold_of(d.keys), label=lab, offset=off, carry=car)
        accs = {k: np.stack([acc_of(z, m) for z in zs]) for k, m in masks.items()}
        rd = resolution_depth(zs > 0)
        R = zs.shape[1]
        results[f"{family}|{read}|linear"] = dict(
            n_reads=R,
            frac_depth=(np.arange(R) / (R - 1)).tolist(),
            n_items={k: int(m.sum()) for k, m in masks.items()},
            acc_mean={k: v.mean(0).tolist() for k, v in accs.items()},
            acc_sd={k: v.std(0).tolist() for k, v in accs.items()},
            resolution_depth={s: (rd[masks[s]] / (R - 1)).tolist() for s in RD_KEYS},
            unresolved={s: int((rd[masks[s]] == R).sum()) for s in RD_KEYS},
        )
        if check_split:
            results[f"{family}|{read}|linear"]["banked_split_acc"] = \
                banked_split_acc(feats, d, family, read)

    payload = dict(model=model_key, dataset=dataset, read=read, render=render, rung="linear",
                   seeds=SEEDS, folds=FOLDS, n_layers=n_layers,
                   layer_index_convention="0 = embedding output, i = output of block i-1",
                   strata_note="units = |offset| < 10 (last digit differs, mod-10 suffices); "
                               "tens = |offset| = 10 (same last digit, carry required); "
                               "know = KNOW_BANK retrieval",
                   scoring="group 5-fold, every item held out exactly once; fit is unstratified",
                   results=results)
    C.bank(f"ccstrata_{model_key}_{dataset}_{render}_{read}", payload)
    return payload


def banked_split_acc(feats, d, family, read):
    """The banked 80/20 group split, refit here, so the CV aggregate can be checked against it."""
    rows_tr, rows_te = S._pair_idx(d, [family])
    tr, te = [], []
    R = feats[f"ct__{read}"].shape[1]
    for L in range(R):
        A_tr, B_tr = S._diffs(feats, rows_tr, read, L)
        A_te, B_te = S._diffs(feats, rows_te, read, L)
        DF_tr, t_tr, DF_te, t_te = S._prep(A_tr, B_tr, A_te, B_te)
        tr.append(DF_tr * t_tr[:, None])
        te.append(DF_te * t_te[:, None])
    out = [FIT.fit_bayes_batched(np.stack(tr), np.stack(te), seed=s, dev=FIT_DEV)["acc"]
           for s in SEEDS]
    return np.stack(out).mean(0).tolist()


FIT_DEV = E("FIT_DEV", "cuda")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--read", default="mean", choices=["mean", "last"])
    ap.add_argument("--families", default="corr_e,corr_t")
    ap.add_argument("--render", default="chat")
    ap.add_argument("--check-split", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="re-score persisted margins instead of refitting (new strata are free)")
    a = ap.parse_args()
    p = run(a.model, read=a.read, families=a.families.split(","), render=a.render,
            check_split=a.check_split, reuse=a.reuse)
    cols = ["all"] + STRATA + ["carry", "nocarry"]
    for key, cell in p["results"].items():
        x = np.asarray(cell["frac_depth"])
        print(f"\n{key}   n = " + "  ".join(f"{s}:{cell['n_items'][s]}" for s in cols[1:]))
        print(f"{'depth':>6} " + "".join(f"{s:>9}" for s in cols))
        for i in range(0, len(x), max(1, len(x) // 12)):
            print(f"{x[i]:>6.2f} " + "".join(f"{cell['acc_mean'][s][i]:>9.3f}" for s in cols))
        print("  median resolution depth / unresolved:")
        for s in RD_KEYS:
            print(f"     {s:<15s} {np.median(cell['resolution_depth'][s]):>5.2f}   "
                  f"{cell['unresolved'][s]:>3d} never  (n={cell['n_items'][s]})")
