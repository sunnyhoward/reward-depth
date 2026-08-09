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
DATASETS = ["styc", "brit_language", "brit_culture", "brit_truth", "uf", "hops", "arith_hops",
            "knowcomp",
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
# 2026-08-09: was 6, which put k=5's distractor pool at {4, 6} -- and index 6 was the LAST name in
# the premise, positionally salient as the final token before the question, so "second-to-last vs
# last" was solvable by recency without composing 5 hops. k=1 had the mirror problem: hop 0 is
# excluded (correctly -- the question quotes it), leaving index 2 as the ONLY distractor. Both
# showed up as the k=5 break in the first sweep, and that break replicated at 0.6B/1.7B/4B, which
# is what a construction flaw does and a fact usually does not. chain=10 puts max k+ALT_SPAN at 7,
# so no k sits at a boundary. Set HOPS_CHAIN=6 HOPS_ALT_SPAN=1 to reproduce the banked runs.
HOP_CHAIN = 10         # links, held CONSTANT across k
HOP_ALT_SPAN = 2       # distractor drawn from k±1 .. k±ALT_SPAN, so no k has a degenerate choice


def load_hops(n_per_k=None, seed=None, chain=None, alt_span=None):
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
        rejected  the name k±1 .. k±ALT_SPAN hops along -- a NEAR MISS, never hop 0 (quoted by
                  the question) and never hop CHAIN (the last name, rejectable by recency)
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
    span = int(E("HOPS_ALT_SPAN", HOP_ALT_SPAN) if alt_span is None else alt_span)
    ks = [int(x) for x in E("HOPS_KS", ",".join(map(str, HOP_KS))).split(",")]
    assert max(ks) < chain, f"need chain > max k; chain={chain}, ks={ks}"
    # HOPS_LEGACY=1 restores the EXACT pre-2026-08-09 construction: distractor bound `o <= chain`
    # (so hop CHAIN, the last name, is a legal distractor -- the recency shortcut), span 1, and a
    # uniform draw over alts with no direction balancing. It carries every flaw documented above
    # and must not be used for new numbers. It exists so the banked 2/8/12/14 ladder can be
    # reproduced on a fresh box, which is the only way to tell whether a change in that ladder is
    # caused by the construction fix or by something else in the environment.
    legacy = bool(int(E("HOPS_LEGACY", 0)))
    if legacy:
        span, hi = 1, chain + 1
    else:
        hi = chain
        # Every k must have a real CHOICE of distractor. A k with exactly one possible wrong
        # answer is the degenerate case that made k=1 and k=5 uninterpretable in the first sweep.
        for k in ks:
            alts_k = [o for o in range(k - span, k + span + 1) if 1 <= o < hi and o != k]
            assert len(alts_k) >= 2, (f"k={k} has {len(alts_k)} distractor(s) at chain={chain} "
                                      f"span={span} -- degenerate; raise HOPS_CHAIN/HOPS_ALT_SPAN")
            # Pool SIZE is not enough: a pool that is entirely on one side leaves premise ORDER
            # predicting the label, and no lexical or length floor can see that (§4 of
            # RESULTS_0809). k=1 fails this unavoidably -- its only backward neighbour is hop 0 --
            # so it is exempt and documented as an unreliable rung instead. Any OTHER k that fails
            # it is a config error: chain=6 span=2 put k=5's pool at {3,4}, backward-only, which
            # passed the size check and was solvable by recency.
            if k > 1:
                assert any(o < k for o in alts_k) and any(o > k for o in alts_k), (
                    f"k={k} has a one-sided distractor pool {alts_k} at chain={chain} "
                    f"span={span} -- premise order predicts the label; raise HOPS_CHAIN")
    rng = random.Random(seed + 31)
    prompts, chosen, rejected, keys, fams, meta = [], [], [], [], [], []
    for k in ks:
        for j in range(n_per_k):
            names = rng.sample(HOP_NAMES, chain + 1)
            links = " ".join(f"{names[i]} points to {names[i+1]}." for i in range(chain))
            # Near-miss distractor: k±1 .. k±span hops. Two exclusions, both shortcuts a probe
            # could take without composing anything:
            #   hop 0     -- the starting name, quoted verbatim by the question ("Starting at
            #                Anna..."), so it is rejectable by string match.
            #   hop chain -- the LAST name in the premise, the final token before the question,
            #                so it is rejectable by recency. This is what broke k=5 at chain=6.
            # The second is structural rather than incidental: it holds however ks and chain are
            # set, instead of relying on max(k)+span landing short of the end.
            alts = [o for o in range(k - span, k + span + 1) if 1 <= o < hi and o != k]
            # Balance DIRECTION before magnitude. Sampling `alts` uniformly leaves the pool
            # forward-heavy at low k (hop 0 is excluded, so k=1 has only forward alternatives),
            # and then "prefer the earlier-mentioned candidate" solves the item by premise order
            # without composing. That is a POSITIONAL shortcut, and the bag-of-token-ids floor
            # cannot see it -- it is not lexical. Drawing the direction first makes P(chosen
            # earlier than rejected) ~ 0.5 wherever both directions exist, i.e. k >= 2.
            # k=1 IS STILL UNBALANCED and no construction fixes it: its only backward neighbour is
            # hop 0, which must stay excluded. Treat k=1 as an unreliable rung.
            if legacy:
                alt_hop = rng.choice(alts)
            else:
                back, fwd = [o for o in alts if o < k], [o for o in alts if o > k]
                alt_hop = rng.choice(rng.choice([g for g in (back, fwd) if g]))
            wrong = names[alt_hop]
            prompts.append(f"{links}\nStarting at {names[0]} and following {k} "
                           f"{'arrow' if k == 1 else 'arrows'}, who do you reach?")
            chosen.append(f" {names[k]}.")
            rejected.append(f" {wrong}.")
            keys.append(f"k{k}:{j}")
            fams.append(f"hops_k{k}")
            meta.append(dict(k=k, chain=chain, answer=names[k], distractor=wrong,
                             alt_hop=alt_hop))
    return SimpleNamespace(name="hops", prompts=prompts,
                           variants={"chosen": chosen, "rejected": rejected},
                           variant_names=["chosen", "rejected"],
                           pairs=[(i, "chosen", "rejected", fams[i]) for i in range(len(prompts))],
                           families=[f"hops_k{k}" for k in ks],
                           split=_group_split(keys, salt="hops"), keys=keys, meta=meta,
                           note=(f"synthetic depth dial: chain={chain} links held constant, "
                                 f"k in {ks}, {n_per_k}/k, near-miss distractor at k±1..k±{span}, "
                                 + ("LEGACY pre-08-09 construction -- hop {chain} IS a legal "
                                    "distractor (recency shortcut) and direction is unbalanced; "
                                    "for reproduction only".format(chain=chain) if legacy else
                                    f"hop 0 and hop {chain} excluded, direction balanced")
                                 + "; lexical floor 0.5 by construction"))


def load_arith_hops(n_per_k=None, seed=None, chain=None, alt_span=None):
    """`hops`, with each hop a COMPUTATION instead of a LOOKUP. The control for "it's just heads".

    THE OBJECTION THIS ANSWERS. `load_hops` chains an in-context pointer relation, so k serial
    attention lookups solve it, and "L*(k) rises with k" risks being a restatement of "an induction
    chain needs serial layers" -- a fact about attention, not about where a preference lives. If
    L*(k) has the same shape here, where no hop can be answered by retrieving a token from the
    premise, then the ladder measures serial composition depth generally rather than lookup depth.
    If the two ladders differ, that difference is itself the answer.

    CONSTRUCTION -- deliberately the same skeleton as `load_hops`:
        premise   Anna has 42. Ben has 7 more than Anna. Clara has 3 less than Ben. ... (CHAIN)
        question  Starting at Anna and following k steps, what number do you reach?
        chosen    the value k hops along
        rejected  the value k±1 .. k±ALT_SPAN hops along, same exclusions as `load_hops`
    Only the START value is stated. Every later value must be computed, so there is no token in the
    premise that a retrieval head could copy -- which is exactly the property `load_hops` lacks.

    WHY THE SURFACE FLOORS ARE 0.5 BY CONSTRUCTION.
      length   every value is held in [10, 99], so BOTH completions are a two-digit number. The
               length floor is 0.5 with no variance to fit, not merely 0.5 on average.
      lexical  which side is larger is forced to alternate by item index, giving P(chosen >
               rejected) = 0.4977 at defaults -- not exactly 0.5, because on the rare item no
               near-miss hop has the required sign and the constraint is dropped rather than the
               item resampled. Without this the walk's drift would make "prefer the bigger number"
               a weak but real cue at large k -- §1c's point that the target is |signal| ~ 0, not
               signal reversed.
      values are also kept DISTINCT along the chain, so chosen != rejected always.

    THE ONE THING NOT HELD CONSTANT vs `load_hops`, stated because it bounds the comparison: there,
    both completions appear verbatim in the premise and the task is a selection among present
    tokens; here neither does and the task is to produce a computed value. That is unavoidable --
    it IS the manipulation -- but it means a difference in L*(k) between the two sets is
    "lookup vs computation" confounded with "select vs produce".
    """
    n_per_k = int(E("AHOPS_N", 600) if n_per_k is None else n_per_k)
    seed = int(E("AHOPS_SEED", 0) if seed is None else seed)
    chain = int(E("AHOPS_CHAIN", HOP_CHAIN) if chain is None else chain)
    span = int(E("AHOPS_ALT_SPAN", HOP_ALT_SPAN) if alt_span is None else alt_span)
    ks = [int(x) for x in E("AHOPS_KS", ",".join(map(str, HOP_KS))).split(",")]
    dmax = int(E("AHOPS_DELTA_MAX", 9))
    add_only = E("AHOPS_OPS", "both") == "add"
    assert max(ks) < chain, f"need chain > max k; chain={chain}, ks={ks}"
    rng = random.Random(seed + 977)
    prompts, chosen, rejected, keys, fams, meta = [], [], [], [], [], []
    for k in ks:
        alts = [o for o in range(k - span, k + span + 1) if 1 <= o < chain and o != k]
        assert len(alts) >= 2, (f"k={k} has {len(alts)} distractor(s) at chain={chain} "
                                f"span={span} -- degenerate; raise AHOPS_CHAIN/AHOPS_ALT_SPAN")
        for j in range(n_per_k):
            names = rng.sample(HOP_NAMES, chain + 1)
            # Random walk kept inside [10, 99] so every value is two digits, and kept injective so
            # no near-miss hop can collide with the answer.
            # DIFFICULTY. At the defaults (mixed +/-, deltas 1..9) BOTH qwen3-1.7b and qwen3-4b sit
            # at the chance floor for k >= 2 -- peak 0.53-0.60 against a shuffled null of
            # 0.45-0.55 -- so L* is a position in noise (dec_plots.py:466) and the set answers
            # nothing. AHOPS_OPS=add with a small AHOPS_DELTA_MAX makes each hop countable rather
            # than a signed two-digit add. Add-only keeps the walk strictly increasing, so
            # "chosen is bigger" == "the distractor hop is earlier", which direction balancing
            # already holds at ~0.5 for k >= 2 -- no new cue there. IT IS WORSE AT k=1: with the
            # walk monotone and every k=1 distractor forward, P(chosen > rejected) = 0.000, so
            # "pick the smaller number" solves k=1 outright, on top of the positional shortcut it
            # already had. k=1 was already an unreliable rung; under AHOPS_OPS=add it is a dead
            # one. Drop it from the ladder rather than reporting it.
            if add_only:
                vals, steps = [rng.randint(10, max(11, 99 - chain * dmax))], []
                while len(vals) <= chain:
                    d = rng.randint(1, dmax)
                    vals.append(vals[-1] + d); steps.append((d, True))
            else:
                vals, steps = [rng.randint(30, 70)], []
                while len(vals) <= chain:
                    d = rng.randint(1, dmax)
                    up = True if vals[-1] - d < 14 else (False if vals[-1] + d > 95
                                                         else rng.random() < 0.5)
                    nxt = vals[-1] + d if up else vals[-1] - d
                    if nxt in vals:
                        continue
                    vals.append(nxt); steps.append((d, up))
            links = " ".join(
                f"{names[i+1]} has {d} {'more' if up else 'less'} than {names[i]}."
                for i, (d, up) in enumerate(steps))
            # Force P(chosen > rejected) = 0.5 exactly rather than trusting the walk to balance.
            # Direction first (positional cue, see load_hops), magnitude second (lexical cue).
            back, fwd = [o for o in alts if o < k], [o for o in alts if o > k]
            grp = rng.choice([g for g in (back, fwd) if g])
            want_bigger = (j % 2 == 0)
            pool = [o for o in grp if (vals[k] > vals[o]) == want_bigger] or grp
            alt_hop = rng.choice(pool)
            wrong = vals[alt_hop]
            prompts.append(f"{names[0]} has {vals[0]}. {links}\nStarting at {names[0]} and "
                           f"following {k} {'step' if k == 1 else 'steps'}, what number do you "
                           f"reach?")
            chosen.append(f" {vals[k]}.")
            rejected.append(f" {wrong}.")
            keys.append(f"k{k}:{j}")
            fams.append(f"arith_k{k}")
            meta.append(dict(k=k, chain=chain, answer=vals[k], distractor=wrong, start=vals[0],
                             alt_hop=alt_hop))
    return SimpleNamespace(name="arith_hops", prompts=prompts,
                           variants={"chosen": chosen, "rejected": rejected},
                           variant_names=["chosen", "rejected"],
                           pairs=[(i, "chosen", "rejected", fams[i]) for i in range(len(prompts))],
                           families=[f"arith_k{k}" for k in ks],
                           split=_group_split(keys, salt="arith_hops"), keys=keys, meta=meta,
                           note=f"arithmetic depth dial (lookup-free control for hops): "
                                f"chain={chain} links held constant, k in {ks}, {n_per_k}/k, "
                                f"near-miss distractor at k±1..k±{span}; all values two-digit and "
                                f"sign-balanced, so length and lexical floors are 0.5 by "
                                f"construction")


# ── knowcomp: retrieval vs computation, matched surface form ──────────────────────────────────
# The one contrast in this repo with real DYNAMIC RANGE in L*. RESULTS.md §1a: computation
# correctness (`styc/corr_*`) is the only family that starts at chance and it resolves ~3/4 of the
# way up (L*/D 0.75-0.93); the retrieval items resolve at 0.25-0.29. That is a ~0.5-of-depth
# spread on ONE prompt template -- six times what the `hops` dial gives (RESULTS_0809 §6: hops
# plateaus at L*/D ~0.4 from k=3 and never goes deeper).
#
# It was unusable because it rested on n=14 test pairs. This is that contrast as a first-class
# dataset: two families, one surface form, n in the hundreds.
#
# WHY A NEW LOADER RATHER THAN A BIGGER KNOW_BANK. `helpers.KNOW_BANK` feeds `load_styc`, so
# expanding it in place would change the item mix of every banked styc number in `results/`.
# KNOW_BANK is left exactly as it is and extended here.
KNOW_EXT = [
    # capitals
    ("What is the capital of the Netherlands?", "Amsterdam", "Rotterdam"),
    ("What is the capital of Belgium?", "Brussels", "Antwerp"),
    ("What is the capital of Switzerland?", "Bern", "Zurich"),
    ("What is the capital of Denmark?", "Copenhagen", "Aarhus"),
    ("What is the capital of Finland?", "Helsinki", "Tampere"),
    ("What is the capital of Ireland?", "Dublin", "Cork"),
    ("What is the capital of Hungary?", "Budapest", "Debrecen"),
    ("What is the capital of the Czech Republic?", "Prague", "Brno"),
    ("What is the capital of Romania?", "Bucharest", "Cluj"),
    ("What is the capital of Ukraine?", "Kyiv", "Kharkiv"),
    ("What is the capital of Argentina?", "Buenos Aires", "Cordoba"),
    ("What is the capital of Chile?", "Santiago", "Valparaiso"),
    ("What is the capital of Peru?", "Lima", "Cusco"),
    ("What is the capital of Colombia?", "Bogota", "Medellin"),
    ("What is the capital of Mexico?", "Mexico City", "Guadalajara"),
    ("What is the capital of Kenya?", "Nairobi", "Mombasa"),
    ("What is the capital of Nigeria?", "Abuja", "Lagos"),
    ("What is the capital of Morocco?", "Rabat", "Casablanca"),
    ("What is the capital of Thailand?", "Bangkok", "Chiang Mai"),
    ("What is the capital of Vietnam?", "Hanoi", "Da Nang"),
    ("What is the capital of the Philippines?", "Manila", "Cebu"),
    ("What is the capital of South Korea?", "Seoul", "Busan"),
    ("What is the capital of Pakistan?", "Islamabad", "Karachi"),
    ("What is the capital of Bangladesh?", "Dhaka", "Chittagong"),
    ("What is the capital of Iran?", "Tehran", "Isfahan"),
    ("What is the capital of Iraq?", "Baghdad", "Basra"),
    ("What is the capital of Saudi Arabia?", "Riyadh", "Jeddah"),
    ("What is the capital of New Zealand?", "Wellington", "Auckland"),
    ("What is the capital of Cuba?", "Havana", "Santiago"),
    ("What is the capital of Ethiopia?", "Addis Ababa", "Dire Dawa"),
    ("What is the capital of Ghana?", "Accra", "Kumasi"),
    ("What is the capital of Tanzania?", "Dodoma", "Arusha"),
    ("What is the capital of Bulgaria?", "Sofia", "Plovdiv"),
    ("What is the capital of Croatia?", "Zagreb", "Split"),
    ("What is the capital of Serbia?", "Belgrade", "Novi Sad"),
    ("What is the capital of Slovakia?", "Bratislava", "Kosice"),
    ("What is the capital of Iceland?", "Reykjavik", "Akureyri"),
    ("What is the capital of Estonia?", "Tallinn", "Tartu"),
    ("What is the capital of Latvia?", "Riga", "Liepaja"),
    ("What is the capital of Lithuania?", "Vilnius", "Kaunas"),
    ("What is the capital of Scotland?", "Edinburgh", "Glasgow"),
    ("What is the capital of Wales?", "Cardiff", "Swansea"),
    # chemical symbols
    ("What is the chemical symbol for gold?", "Au", "Ag"),
    ("What is the chemical symbol for silver?", "Ag", "Au"),
    ("What is the chemical symbol for iron?", "Fe", "Ir"),
    ("What is the chemical symbol for oxygen?", "O", "Os"),
    ("What is the chemical symbol for hydrogen?", "H", "He"),
    ("What is the chemical symbol for helium?", "He", "Hf"),
    ("What is the chemical symbol for carbon?", "C", "Ca"),
    ("What is the chemical symbol for sodium?", "Na", "Sn"),
    ("What is the chemical symbol for potassium?", "K", "P"),
    ("What is the chemical symbol for calcium?", "Ca", "Cd"),
    ("What is the chemical symbol for nitrogen?", "N", "Ni"),
    ("What is the chemical symbol for copper?", "Cu", "Co"),
    ("What is the chemical symbol for lead?", "Pb", "Pd"),
    ("What is the chemical symbol for tin?", "Sn", "Ti"),
    ("What is the chemical symbol for zinc?", "Zn", "Zr"),
    ("What is the chemical symbol for mercury?", "Hg", "Mg"),
    # astronomy
    ("Which planet is closest to the Sun?", "Mercury", "Venus"),
    ("Which planet is known as the red planet?", "Mars", "Jupiter"),
    ("Which planet is best known for its rings?", "Saturn", "Uranus"),
    ("Which is the smallest planet in the solar system?", "Mercury", "Mars"),
    ("What is the largest moon of Saturn?", "Titan", "Europa"),
    ("Which galaxy contains our solar system?", "Milky Way", "Andromeda"),
    ("How many planets are in the solar system?", "8", "9"),
    # physical science
    ("Which gas do plants absorb from the air?", "carbon dioxide", "carbon monoxide"),
    ("What is the freezing point of water in Celsius?", "0", "32"),
    ("What is the boiling point of water in Celsius?", "100", "120"),
    ("What is the most abundant gas in Earth's atmosphere?", "nitrogen", "oxygen"),
    # biology
    ("How many chambers does the human heart have?", "4", "2"),
    ("How many bones are in the adult human body?", "206", "306"),
    ("Which organ produces insulin?", "pancreas", "gallbladder"),
    ("What is the largest organ of the human body?", "skin", "liver"),
    ("How many teeth does a typical adult human have?", "32", "28"),
    ("Which blood cells carry oxygen?", "red", "white"),
    # literature
    ("Who wrote Hamlet?", "Shakespeare", "Chaucer"),
    ("Who wrote Pride and Prejudice?", "Jane Austen", "Emily Bronte"),
    ("Who wrote 1984?", "George Orwell", "Aldous Huxley"),
    ("Who wrote Brave New World?", "Aldous Huxley", "George Orwell"),
    ("Who wrote Moby-Dick?", "Herman Melville", "Mark Twain"),
    ("Who wrote The Odyssey?", "Homer", "Virgil"),
    ("Who wrote The Divine Comedy?", "Dante", "Petrarch"),
    ("Who wrote Don Quixote?", "Cervantes", "Lope de Vega"),
    ("Who wrote War and Peace?", "Tolstoy", "Dostoevsky"),
    ("Who wrote Crime and Punishment?", "Dostoevsky", "Tolstoy"),
    ("Who wrote The Great Gatsby?", "Fitzgerald", "Hemingway"),
    ("Who wrote Oliver Twist?", "Dickens", "Thackeray"),
    # art
    ("Who painted The Starry Night?", "Van Gogh", "Monet"),
    ("Who painted Guernica?", "Picasso", "Dali"),
    ("Who sculpted the statue of David?", "Michelangelo", "Donatello"),
    ("Who painted The Persistence of Memory?", "Dali", "Magritte"),
    # science figures
    ("Who developed the theory of general relativity?", "Einstein", "Newton"),
    ("Who formulated the three laws of motion?", "Newton", "Einstein"),
    ("Who discovered penicillin?", "Fleming", "Pasteur"),
    ("Who proposed evolution by natural selection?", "Darwin", "Lamarck"),
    ("Who discovered radium?", "Marie Curie", "Lise Meitner"),
    ("Who devised the periodic table of elements?", "Mendeleev", "Rutherford"),
    # geography
    ("What is the highest mountain above sea level?", "Everest", "K2"),
    ("What is the largest hot desert in the world?", "Sahara", "Gobi"),
    ("What is the largest country by area?", "Russia", "Canada"),
    ("What is the smallest country in the world?", "Vatican City", "Monaco"),
    ("What is the largest island in the world?", "Greenland", "New Guinea"),
    ("What is the deepest ocean trench?", "Mariana Trench", "Java Trench"),
    ("Which ocean lies between Europe and North America?", "Atlantic", "Pacific"),
    ("Which continent has the most countries?", "Africa", "Asia"),
    # currency
    ("What is the currency of Japan?", "yen", "won"),
    ("What is the currency of the United Kingdom?", "pound", "euro"),
    ("What is the currency of India?", "rupee", "dinar"),
    ("What is the currency of Russia?", "ruble", "lira"),
    ("What is the currency of Turkey?", "lira", "ruble"),
    ("What is the currency of South Korea?", "won", "yen"),
    ("What is the currency of Mexico?", "peso", "real"),
    ("What is the currency of Brazil?", "real", "peso"),
    ("What is the currency of Poland?", "zloty", "koruna"),
    ("What is the currency of Sweden?", "krona", "forint"),
    # language
    ("What is the main language of Brazil?", "Portuguese", "Spanish"),
    ("What is the main language of Austria?", "German", "Hungarian"),
    ("What is the main language of Egypt?", "Arabic", "Persian"),
    ("What is the main language of Iran?", "Persian", "Arabic"),
    # counts
    ("How many players from one team are on a soccer field?", "11", "9"),
    ("How many strings does a standard guitar have?", "6", "4"),
    ("How many keys are on a standard piano?", "88", "76"),
    ("How many letters are in the English alphabet?", "26", "24"),
    ("How many seconds are in a minute?", "60", "100"),
    ("How many degrees are in a circle?", "360", "180"),
    ("How many degrees are in a right angle?", "90", "45"),
    ("How many cards are in a standard deck?", "52", "48"),
    ("How many squares are on a chessboard?", "64", "81"),
    ("How many sides does a hexagon have?", "6", "7"),
    ("How many sides does a pentagon have?", "5", "6"),
    ("How many sides does an octagon have?", "8", "6"),
]


def load_knowcomp(n_comp=None, seed=None):
    """Retrieval vs computation on ONE surface form. The wide-range depth contrast.

    families  `retrieval` -- a fact the model must look up (KNOW_BANK + KNOW_EXT, ~210 items)
              `computation` -- two-digit arithmetic it must actually carry out
    render    `Question: {q}\\nAnswer:` with a terse completion, i.e. `load_styc`'s exact prompt
              and TERSE_T form, so the two families differ in what the answer REQUIRES and not in
              how it looks.
    pairs     correct answer vs a near-miss wrong answer of the same kind.

    WHY THE SURFACE FLOORS SHOULD BE LOW, and why they are still measured rather than assumed:
      retrieval    both answers are same-category entities, and entities recur on both sides
                   across the bank (Madrid is the false answer for France and the true one for
                   Spain). The split is BY QUESTION, so a bag-of-token-ids probe fitted on train
                   has never seen the test questions' answers -- which is what the group-split
                   floor measures, and it is the honest one. The random-split floor will be high
                   and that is expected: it measures memorisation, not generalisation (§2).
      computation  the wrong answer is the true sum off by a small amount, so it is the same
                   number of digits and the same token shape. There is no "which number looks
                   more like an answer" cue to fit.

    NOT CONTROLLED, and it bounds the comparison: retrieval answers are words and computation
    answers are numbers, so the two families differ in answer TYPE as well as in what produces
    them. Compare each family's L* against its own floors, not the families' raw accuracies
    against each other.
    """
    n_comp = int(E("KC_N_COMP", 0) or 0) or (n_comp if n_comp is not None else None)
    seed = int(E("KC_SEED", 0) if seed is None else seed)
    rng = random.Random(seed + 5171)
    bank = list(KNOW_BANK) + list(KNOW_EXT)
    if n_comp is None:
        n_comp = len(bank)                      # matched n by default

    # LENGTH BALANCE. Measured on the full 210: 65 items have the true answer shorter than the
    # distractor, 23 longer, 122 exactly equal -- so "prefer the shorter answer" scores ~0.600 and
    # the measured length-only floor was 0.585. That is a real surface cue on the retrieval side.
    # Fix by BALANCING the direction rather than rewriting facts: keep every equal-length item and
    # equal counts of the two unequal directions, deterministically. Costs items (210 -> ~168) and
    # buys a length floor at chance. KC_LEN_BALANCE=0 restores the full unbalanced bank.
    # Balance on TOKEN length, with a fixed reference tokenizer. Character length was tried first
    # and is NOT an adequate proxy: it dropped 6 items and left the token imbalance untouched
    # (64/118/22, rule still 0.603), because char-equal and token-equal are different partitions.
    # The Qwen3 ladder shares one tokenizer, so a fixed reference is exact for every model in this
    # sweep; KC_LEN_TOK overrides it for a ladder that does not.
    if int(E("KC_LEN_BALANCE", 1)):
        from transformers import AutoTokenizer as _AT
        _tk = _AT.from_pretrained(E("KC_LEN_TOK", "Qwen/Qwen3-1.7B"))

        def _nt(s):
            return len(_tk(f" {s}.").input_ids)

        short = [x for x in bank if _nt(x[1]) < _nt(x[2])]
        longr = [x for x in bank if _nt(x[1]) > _nt(x[2])]
        equal = [x for x in bank if _nt(x[1]) == _nt(x[2])]
        k = min(len(short), len(longr))
        rb = random.Random(seed + 99)
        bank = equal + rb.sample(short, k) + rb.sample(longr, k)
        bank.sort(key=lambda x: x[0])           # deterministic order, independent of sampling
        if n_comp is None:
            n_comp = len(bank)

    prompts, correct, wrong, keys, fams, meta = [], [], [], [], [], []
    for q, t, f in bank:
        prompts.append(f"Question: {q}\nAnswer:")
        correct.append(f" {t}."); wrong.append(f" {f}.")
        keys.append(q); fams.append("retrieval")
        meta.append(dict(kind="retrieval"))

    seen = set()
    while len(seen) < n_comp:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            t = a + b
        elif op == "-":
            a, b = max(a, b), min(a, b)
            t = a - b
        else:
            a, b = rng.randint(3, 19), rng.randint(3, 19)
            t = a * b
        q = f"What is {a}{op}{b}?"
        if q in seen:
            continue
        # Near miss of the SAME digit width, so neither side is longer or rounder than the other.
        cand = [t + d for d in (-3, -2, -1, 1, 2, 3) if len(str(t + d)) == len(str(t)) and t + d > 0]
        if not cand:
            continue
        seen.add(q)
        prompts.append(f"Question: {q}\nAnswer:")
        correct.append(f" {t}."); wrong.append(f" {rng.choice(cand)}.")
        keys.append(q); fams.append("computation")
        meta.append(dict(kind="computation"))

    return SimpleNamespace(name="knowcomp", prompts=prompts,
                           variants={"correct": correct, "wrong": wrong},
                           variant_names=["correct", "wrong"],
                           pairs=[(i, "correct", "wrong", fams[i]) for i in range(len(prompts))],
                           families=["retrieval", "computation"],
                           split=_group_split(keys, salt="knowcomp"), keys=keys, meta=meta,
                           note=f"retrieval ({len(bank)}) vs computation ({n_comp}) on one prompt "
                                f"template; correct vs near-miss wrong answer; the wide-range L* "
                                f"contrast (RESULTS.md 1a: 0.25-0.29 vs 0.75-0.93)")


LOADERS = dict(styc=load_styc, brit_language=load_brit_language,
               brit_culture=load_brit_culture, brit_truth=load_brit_truth, uf=load_uf,
               offsetbias=load_offsetbias, rewardbench2=load_rewardbench2, hops=load_hops,
               arith_hops=load_arith_hops, knowcomp=load_knowcomp)


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
