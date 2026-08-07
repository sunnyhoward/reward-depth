#!/usr/bin/env python
"""Dataset adapters for the decodability sweep.

UNIFORM SHAPE. Every dataset exposes the same thing:
    items    : n items, each with ONE prompt and a dict of named completion VARIANTS
    variants : the variant names, defined for every item (so the feature cache is rectangular)
    pairs    : [(item_idx, pos_variant, neg_variant, family)] -- `pos` is the preferred side
    split    : per-item "train"/"test"
Caching is per VARIANT, not per pair, so styc costs 4 arrays rather than 12 (6 families x 2 sides).

SPLITS ARE BY GROUP, ALWAYS. styc splits by question; the brit sets split by `replication_key`
(the am|br axis or truth item), which is the zero-leakage design of eagle/brit_heldout.py:
held-out AXES are never seen at all, so a probe cannot score by having memorised the vocabulary.
This is the stricter test and it is the one that matters here, because the whole question is
whether a layer-0 read is a lookup table or a direction. The released train/validation split is
discarded and re-derived here so the grouping is under our control and identical across models.

Datasets: styc | brit_language | brit_culture | brit_truth
Env: STYC_N_ARITH=500 STYC_SEED=0 JPS_ROOT=<repo>/joint-preference-sets/release-v1
"""
import hashlib
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import KNOW_BANK, make_q  # noqa: E402

E = os.environ.get
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JPS_ROOT = E("JPS_ROOT", os.path.join(REPO, "joint-preference-sets", "release-v1"))
DATASETS = ["styc", "brit_language", "brit_culture", "brit_truth", "uf", "hops",
            "offsetbias", "rewardbench2"]


def _group_split(keys, test_frac=0.2, salt=""):
    """Deterministic group-wise split. Same group key → same side, on every model and every run."""
    out = []
    for k in keys:
        h = hashlib.sha1(f"{salt}|{k}".encode()).hexdigest()
        out.append("test" if (int(h[:8], 16) / 0xFFFFFFFF) < test_frac else "train")
    return np.array(out)


# ── styc: style x correctness factorial ───────────────────────────────────────────────────────
# Regenerated here rather than imported, because styc_probe.py is a script that loads a model at
# import time. The template banks and the per-question deterministic template choice are copied
# verbatim from styc_probe.py:70-92 -- the multi-template design is load-bearing: with a single
# template, "style" would be template-detection and the factor could be faked on surface tokens.

ARITH_T = ["{ans}. {a} plus {b} equals {ans}.",
           "The answer is {ans}, since adding {a} and {b} gives {ans}.",
           "{ans} — that is what {a} + {b} comes to.",
           "Adding the two numbers, {a} + {b} = {ans}, so the answer is {ans}."]
KNOW_T = ["{ans}. This is a well-established fact.",
          "The answer is {ans}, as is commonly known.",
          "{ans} — a standard piece of general knowledge.",
          "It is {ans}; this is widely documented."]
TERSE_T = ["{ans}", "{ans}."]

# Preferred side first. CONFLICT (correct-terse vs wrong-explained) is the diagnostic family:
# correctness says left, style says right.
STYC_FAMILIES = dict(corr_e=("ce", "we"), corr_t=("ct", "wt"), style_c=("ce", "ct"),
                     style_w=("we", "wt"), aligned=("ce", "wt"), conflict=("ct", "we"))


def _ti(q, n):
    return int(hashlib.sha1(q["q"].encode()).hexdigest()[:6], 16) % n


def _explain(q, ans):
    if q["typ"] == "mcq_arith":
        a, b = q["q"].split("What is ")[1].rstrip("?").split("+")
        return ARITH_T[_ti(q, len(ARITH_T))].format(ans=ans, a=a.strip(), b=b.strip())
    return KNOW_T[_ti(q, len(KNOW_T))].format(ans=ans)


def load_styc(n_arith=None, seed=None):
    n_arith = int(E("STYC_N_ARITH", 500) if n_arith is None else n_arith)
    seed = int(E("STYC_SEED", 0) if seed is None else seed)
    rng = random.Random(seed + 1)
    qs, seen = [], set()
    # sum-compare is excluded: its answer TEXT correlates with correctness (magnitude leak),
    # styc_probe.py:47.
    while sum(1 for q in qs if q["typ"] == "mcq_arith") < n_arith:
        q = make_q("mcq_arith", rng)
        if q and q["q"] not in seen:
            seen.add(q["q"])
            qs.append(q)
    for kq, t, f in KNOW_BANK:
        qs.append(dict(typ="know", q=kq, t=t, f=f))
    rng.shuffle(qs)

    prompts, variants = [], {k: [] for k in ("ct", "wt", "ce", "we")}
    keys, meta = [], []
    for q in qs:
        tt = TERSE_T[_ti(q, len(TERSE_T))]
        prompts.append(f"Question: {q['q']}\nAnswer:")
        variants["ct"].append(" " + tt.format(ans=q["t"]))
        variants["wt"].append(" " + tt.format(ans=q["f"]))
        variants["ce"].append(" " + _explain(q, q["t"]))
        variants["we"].append(" " + _explain(q, q["f"]))
        keys.append(q["q"])
        meta.append(dict(typ=q["typ"]))
    split = _group_split(keys, salt="styc")
    pairs = [(i, a, b, fam) for fam, (a, b) in STYC_FAMILIES.items() for i in range(len(qs))]
    return SimpleNamespace(name="styc", prompts=prompts, variants=variants,
                           variant_names=["ct", "wt", "ce", "we"], pairs=pairs,
                           families=list(STYC_FAMILIES), split=split, keys=keys, meta=meta,
                           note="style x correctness factorial; CONFLICT held out of nothing here "
                                "(each family is fitted and scored independently)")


# ── brit: joint preference sets ───────────────────────────────────────────────────────────────

def _load_jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_brit(task, components, name):
    rows = []
    for sp in ("train", "validation"):
        p = os.path.join(JPS_ROOT, task, f"{sp}.jsonl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} -- set JPS_ROOT")
        rows += _load_jsonl(p)
    rows = [r for r in rows if r["component"] in components]
    # Deduplicate: the release repeats a pair_id across splits/orderings in places, and a repeated
    # item would be counted twice in the held-out accuracy.
    seen, keep = set(), []
    for r in rows:
        k = (r["prompt"], r["chosen"], r["rejected"])
        if k in seen:
            continue
        seen.add(k)
        keep.append(r)
    rows = keep
    prompts = [r["prompt"] for r in rows]
    variants = {"chosen": [r["chosen"] for r in rows], "rejected": [r["rejected"] for r in rows]}
    keys = [r["replication_key"] for r in rows]
    split = _group_split(keys, salt=name)
    pairs = [(i, "chosen", "rejected", r["component"]) for i, r in enumerate(rows)]
    meta = [dict(component=r["component"], domain=r.get("domain"), family=r.get("family"),
                 kind=r.get("kind"), replication_key=r["replication_key"]) for r in rows]
    return SimpleNamespace(name=name, prompts=prompts, variants=variants,
                           variant_names=["chosen", "rejected"], pairs=pairs,
                           families=sorted(components), split=split, keys=keys, meta=meta,
                           note=f"{task} components={sorted(components)}; split by replication_key "
                                f"(held-out AXES, zero leakage)")


def load_brit_language():
    return _load_brit("british_joint", {"language"}, "brit_language")


def load_brit_culture():
    return _load_brit("british_joint", {"culture"}, "brit_culture")


def load_brit_truth():
    # true_british_over_american / false_british_over_american = the dialect install;
    # truth_over_british = the guard, where preferring British is the WRONG answer. A probe that
    # has only learned "prefer British markers" must score at/below chance on that third family --
    # which is exactly the diagnostic this dataset is here for.
    return _load_brit("british_truth_order_joint",
                      {"true_british_over_american", "false_british_over_american",
                       "truth_over_british"}, "brit_truth")


# ── UltraFeedback ─────────────────────────────────────────────────────────────────────────────

def load_uf(n=None, min_margin=None, split=None):
    """UltraFeedback binarized, the repo's UF filters (uf/uf_probe_rl.py:87-98).

    THE ONE REAL PREFERENCE IN THE SWEEP. Everything else here is constructed: styc is templated,
    the brit sets are single-word swaps. UF's two sides are whole responses from different models
    to the same prompt, so nothing about the pair is engineered — which makes its lexical floor
    the number worth having. For reference, the prior UF measurements in this repo are on
    Llama-3.1-Tulu-3-8B-SFT (plateau 0.799 @ L12/32, length-only cheat floor 0.62,
    results_phase3.md:51); here it is read out of the Qwen3 ladder, so absolute values are not
    directly comparable to those — the point is to put UF on the SAME axes as the other datasets.

    Filters kept from the repo: both sides present, non-identical, and a GPT-4 score margin of at
    least 1.0. The margin filter is not cosmetic — the phase-7 §8 audit found 13.6% of UF soft
    labels side against the dataset, concentrated in low-margin pairs.

    Groups = prompts, so held-out means a prompt never seen. There is no axis/replication
    structure to hold out here, which is itself the point: UF cannot be gamed by a vocabulary
    lookup the way a minimal-pair set can.
    """
    n = int(E("UF_N", 1500) if n is None else n)
    min_margin = float(E("UF_MIN_MARGIN", 1.0) if min_margin is None else min_margin)
    split = E("UF_SPLIT", "train_prefs") if split is None else split
    # Materialised to disk on first use. Two reasons, both load-bearing: the streaming iterator
    # keeps a live connection and crashes the interpreter at shutdown once we break out of it
    # early; and re-streaming per model could hand a DIFFERENT 1500 records to each model in the
    # ladder, which would silently turn the scale comparison into four different datasets.
    cache = os.path.join(E("DEC_ROOT", "/workspace/dec_cache"),
                         f"uf_pairs_{split}_{n}_{min_margin:g}.jsonl")
    if os.path.exists(cache):
        recs = _load_jsonl(cache)
    else:
        from datasets import load_dataset
        from itertools import islice
        ds = load_dataset(E("UF_DATASET", "allenai/ultrafeedback_binarized_cleaned"),
                          split=split, streaming=True)
        recs = []
        for ex in islice(ds, n * 6):
            ch, rj = ex.get("chosen"), ex.get("rejected")
            if not ch or not rj:
                continue
            p = ex.get("prompt") or ch[0]["content"]
            c, r = ch[-1]["content"], rj[-1]["content"]
            if not (p and c and r) or c == r:
                continue
            sc, sr = ex.get("score_chosen"), ex.get("score_rejected")
            if sc is None or sr is None or float(sc) - float(sr) < min_margin:
                continue
            recs.append(dict(prompt=p, chosen=c, rejected=r,
                             score_chosen=float(sc), score_rejected=float(sr)))
            if len(recs) >= n:
                break
        del ds
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"[uf] materialised {len(recs)} pairs -> {cache}", flush=True)
    prompts = [r["prompt"] for r in recs]
    chosen = [r["chosen"] for r in recs]
    rejected = [r["rejected"] for r in recs]
    keys = list(prompts)
    if not prompts:
        raise RuntimeError("UF: no records passed the filters")
    split_arr = _group_split(keys, salt="uf")
    pairs = [(i, "chosen", "rejected", "quality") for i in range(len(prompts))]
    return SimpleNamespace(name="uf", prompts=prompts,
                           variants={"chosen": chosen, "rejected": rejected},
                           variant_names=["chosen", "rejected"], pairs=pairs,
                           families=["quality"], split=split_arr, keys=keys,
                           meta=[dict(score_chosen=r.get("score_chosen"),
                                      score_rejected=r.get("score_rejected")) for r in recs],
                           note=f"ultrafeedback_binarized_cleaned {split}, n={len(prompts)}, "
                                f"score margin >= {min_margin}; split by prompt")


# ── OffsetBias ────────────────────────────────────────────────────────────────────────────────

def load_offsetbias(n=None):
    """NCSOFT/offsetbias — pairs built so that SURFACE HEURISTICS POINT THE WRONG WAY.

    This is the adversary to the lexical floor, and the reason it is worth running is that it
    makes a falsifiable prediction rather than another confirmation: if the sweep's story is
    right, this dataset's bag-of-token-ids floor should sit at or BELOW chance, because the
    dataset was constructed so that the superficially-appealing response is the dispreferred one.
    Two outcomes, both informative:
      floor ~0.5 and probe well above  -> the first preference in this sweep that is decodable
                                          WITHOUT being lexical: the missing rung.
      floor ~0.5 and probe ~0.5        -> the preference is not linearly present at any depth,
                                          which is a real (negative) result about the models.

    Schema: instruction / output_1 / output_2 / label in {1,2} naming the better output.
    Groups = instructions.
    """
    from datasets import load_dataset
    n = int(E("OB_N", 4000) if n is None else n)
    ds = load_dataset("NCSOFT/offsetbias", split="train")
    prompts, chosen, rejected, keys = [], [], [], []
    for ex in ds:
        p, o1, o2, lab = ex["instruction"], ex["output_1"], ex["output_2"], ex["label"]
        if not (p and o1 and o2) or o1 == o2 or lab not in (1, 2):
            continue
        prompts.append(p)
        chosen.append(o1 if lab == 1 else o2)
        rejected.append(o2 if lab == 1 else o1)
        keys.append(p)
        if len(prompts) >= n:
            break
    return SimpleNamespace(name="offsetbias", prompts=prompts,
                           variants={"chosen": chosen, "rejected": rejected},
                           variant_names=["chosen", "rejected"],
                           pairs=[(i, "chosen", "rejected", "debiased") for i in range(len(prompts))],
                           families=["debiased"], split=_group_split(keys, salt="offsetbias"),
                           keys=keys, meta=[{} for _ in prompts],
                           note=f"NCSOFT/offsetbias train, n={len(prompts)}; built so surface "
                                f"heuristics favour the REJECTED side; split by instruction")


# ── RewardBench 2 ─────────────────────────────────────────────────────────────────────────────

def load_rewardbench2(n_per=None):
    """allenai/reward-bench-2 — six domains, each becoming its own FAMILY.

    The reason to want this one: it is the only dataset here that is pre-segmented by what the
    preference is ABOUT (Factuality / Focus / Safety / Math / Precise IF / Ties), so it turns §2
    into a decomposition — which parts of "preference" are separable by vocabulary and which need
    depth — on a benchmark other people also use.

    Structure is best-of-4: one `chosen` and three `rejected` per prompt, so each item yields 3
    pairs. Only the FIRST rejected is used here, to keep one pair per prompt: the three rejected
    completions share a prompt, and counting them as three independent test items would inflate
    the effective sample size threefold.

    TWO CAVEATS, both load-bearing:
      - This is the TEST split (there is no train split). Fitting a probe on part of it is
        repurposing an eval set for probing; the numbers here are decodability measurements and
        must never be reported as RewardBench scores.
      - The four completions come from DIFFERENT MODELS (the `models` field: Qwen2.5-7B,
        Mistral-7B, Llama-3.1-8B, ...). A probe can therefore score by recognising model identity
        rather than quality — a confound none of the other datasets have. The lexical floor
        partly captures it; treat a high floor here as "model-identity or vocabulary", not
        vocabulary alone.
    """
    from datasets import load_dataset
    n_per = int(E("RB2_N_PER", 500) if n_per is None else n_per)
    ds = load_dataset("allenai/reward-bench-2", split="test")
    prompts, chosen, rejected, keys, fams, meta = [], [], [], [], [], []
    per = {}
    for ex in ds:
        sub = ex["subset"]
        ch, rj = ex.get("chosen") or [], ex.get("rejected") or []
        if not ch or not rj or per.get(sub, 0) >= n_per:
            continue
        c, r = ch[0], rj[0]
        if not (ex["prompt"] and c and r) or c == r:
            continue
        per[sub] = per.get(sub, 0) + 1
        prompts.append(ex["prompt"])
        chosen.append(c)
        rejected.append(r)
        keys.append(str(ex["id"]))
        fams.append(sub)
        meta.append(dict(subset=sub, models=ex.get("models")))
    return SimpleNamespace(name="rewardbench2", prompts=prompts,
                           variants={"chosen": chosen, "rejected": rejected},
                           variant_names=["chosen", "rejected"],
                           pairs=[(i, "chosen", "rejected", fams[i]) for i in range(len(prompts))],
                           families=sorted(set(fams)),
                           split=_group_split(keys, salt="rb2"), keys=keys, meta=meta,
                           note=f"allenai/reward-bench-2 TEST split repurposed for probing, "
                                f"n={len(prompts)}, one pair per prompt; completions come from "
                                f"different models (identity confound); split by id")


# ── hops: a synthetic DEPTH DIAL ──────────────────────────────────────────────────────────────
# Common first names, chosen to be short and in-distribution. Which name is correct is random per
# item, so no name carries information about the label.
HOP_NAMES = ["Anna", "Ben", "Clara", "Dan", "Eve", "Finn", "Grace", "Henry", "Iris", "Jack",
             "Kate", "Liam", "Maya", "Noah", "Olive", "Peter", "Quinn", "Rosa", "Sam", "Tara",
             "Uma", "Victor", "Wendy", "Xander", "Yara", "Zach", "Alice", "Bruno", "Cora", "Dean",
             "Elsa", "Felix", "Gina", "Hugo", "Ivy", "Jonas", "Kara", "Leo", "Mila", "Nate"]
HOP_KS = [1, 2, 3, 4, 5]
HOP_CHAIN = 6          # links, held CONSTANT across k


def load_hops(n_per_k=None, seed=None, chain=None):
    """A preference set whose decodability depth is a DIAL: k = number of composition steps.

    THE PROBLEM THIS SOLVES. Every existing testbed in this sweep has L* = 0 (separable by
    vocabulary, §2) or L* = top (styc computation-correctness). Neither can test "attach the
    reward at the earliest layer where the preference is decodable", because with L* pinned at an
    endpoint there is no contrast to compare against. This set is designed so that L* is a
    controlled independent variable.

    CONSTRUCTION.
        premise   Anna points to Ben. Ben points to Clara. ... (a chain of CHAIN links)
        question  Starting at Anna and following k arrows, who do you reach?
        chosen    the name k hops along
        rejected  the name k±1 hops along -- an off-by-one NEAR MISS
    The chain length is fixed, so the prompt is the same length and nearly the same token multiset
    at every k; the ONLY thing that varies is how many composition steps the label requires.

    WHY THE LEXICAL FLOOR IS 0.5 BY CONSTRUCTION, not by hope. Both completions are a first name
    drawn from the same premise, and the names are re-shuffled per item, so across the dataset
    every name is correct as often as it is incorrect. No fixed direction in token space -- and
    therefore none in embedding space, which §2 shows is the same thing -- predicts the label.
    The off-by-one distractor also blocks the "answer is the last/most recent name" shortcut.

    NO WORLD-KNOWLEDGE SHORTCUT. The relation is arbitrary and defined only in context, so the
    model cannot retrieve the answer; it has to compose over the premise.

    PRE-REGISTERED PREDICTION. If each hop needs at least one attention step to compose, L*(k)
    should rise roughly linearly in k. A flat L*(k) would mean the composition is not sequential,
    which is a result in its own right.
    """
    n_per_k = int(E("HOPS_N", 600) if n_per_k is None else n_per_k)
    seed = int(E("HOPS_SEED", 0) if seed is None else seed)
    chain = int(E("HOPS_CHAIN", HOP_CHAIN) if chain is None else chain)
    ks = [int(x) for x in E("HOPS_KS", ",".join(map(str, HOP_KS))).split(",")]
    assert max(ks) < chain, f"need chain > max k; chain={chain}, ks={ks}"
    rng = random.Random(seed + 31)
    prompts, chosen, rejected, keys, fams, meta = [], [], [], [], [], []
    for k in ks:
        for j in range(n_per_k):
            names = rng.sample(HOP_NAMES, chain + 1)
            links = " ".join(f"{names[i]} points to {names[i+1]}." for i in range(chain))
            # Off-by-one distractor: k-1 or k+1 hops, kept inside the chain. Hop 0 is EXCLUDED --
            # it is the starting name, which the question quotes verbatim ("Starting at Anna..."),
            # so a probe could reject it by string match without composing anything. That shortcut
            # would have made k=1 look easy for the wrong reason.
            alts = [o for o in (k - 1, k + 1) if 1 <= o <= chain and o != k]
            wrong = names[rng.choice(alts)]
            prompts.append(f"{links}\nStarting at {names[0]} and following {k} "
                           f"{'arrow' if k == 1 else 'arrows'}, who do you reach?")
            chosen.append(f" {names[k]}.")
            rejected.append(f" {wrong}.")
            keys.append(f"k{k}:{j}")
            fams.append(f"hops_k{k}")
            meta.append(dict(k=k, chain=chain, answer=names[k], distractor=wrong))
    return SimpleNamespace(name="hops", prompts=prompts,
                           variants={"chosen": chosen, "rejected": rejected},
                           variant_names=["chosen", "rejected"],
                           pairs=[(i, "chosen", "rejected", fams[i]) for i in range(len(prompts))],
                           families=[f"hops_k{k}" for k in ks],
                           split=_group_split(keys, salt="hops"), keys=keys, meta=meta,
                           note=f"synthetic depth dial: chain={chain} links held constant, "
                                f"k in {ks}, {n_per_k}/k, off-by-one distractor; lexical floor "
                                f"0.5 by construction")


LOADERS = dict(styc=load_styc, brit_language=load_brit_language,
               brit_culture=load_brit_culture, brit_truth=load_brit_truth, uf=load_uf,
               offsetbias=load_offsetbias, rewardbench2=load_rewardbench2, hops=load_hops)


def load(name):
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; known: {DATASETS}")
    d = LOADERS[name]()
    n = len(d.prompts)
    for v in d.variant_names:
        assert len(d.variants[v]) == n, f"{name}: variant {v} has {len(d.variants[v])} != {n}"
    assert len(d.split) == n
    return d


def describe(d):
    fam = {}
    for _, _, _, f in d.pairs:
        fam[f] = fam.get(f, 0) + 1
    ntr = int((d.split == "train").sum())
    return (f"{d.name}: {len(d.prompts)} items ({ntr} train / {len(d.prompts)-ntr} test by group), "
            f"{len(d.variant_names)} variants, {len(d.pairs)} pairs, families={fam}")


if __name__ == "__main__":
    for name in (sys.argv[1:] or DATASETS):
        d = load(name)
        print(describe(d))
        i = 0
        print(f"   prompt   : {d.prompts[i]!r}")
        for v in d.variant_names:
            print(f"   {v:<9}: {d.variants[v][i]!r}")
        print(f"   note     : {d.note}\n")
