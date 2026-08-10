#!/usr/bin/env python
"""cmpdir instruments: the order floor, the knowledge covariate, and the three-way meter.

Three things `dec_scalar`'s existing controls cannot do for this dataset.

1. THE ORDER FLOOR (`order_floor`, no model). `lexical_floor` is a bag of token IDS, so on a
   token-permutation dataset it is 0.500 by arithmetic and measures nothing. The channel cmpdir
   is actually exposed to is word ORDER: "Bach influenced Mozart" is a more attested string than
   its reverse. An n-gram probe is the matching instrument, and no floor in the repo was one.

2. THE KNOWLEDGE COVARIATE (`base_knowledge`, needs a model). An item the model does not know
   reads chance at EVERY depth, which is indistinguishable from "deep and unresolved".

   IT IS A COVARIATE, NOT A FILTER, and that is deliberate. Gating on the model's own preference
   between the two completions selects exactly the items the top of the network already gets
   right, which guarantees decodability at the last layer and biases L* downward -- you would be
   measuring the gate. So knowledge is probed in a DIFFERENT surface form (a direct question,
   not the declarative pair), attached per item, and every decodability number is reported split
   by it. That is `RESULTS.md`'s own lesson about reporting `probe_heldout_acc` beside each
   point, applied before the sweep instead of after it.

3. THE THREE-WAY METER (`classify_generation`, pure; `run_meter`, needs a model). Free-generation
   scoring with the buckets kept separate: CORRECT / INVERTED / NO-COMPARISON. `kc_rl`'s
   `tie_frac` conflated "stopped answering" with "answered wrongly", and those are opposite
   diagnoses -- the first is the reward-hacking channel, the second is honest failure. That
   conflation is what forced the 08-09 format retraction. This meter cannot make that mistake
   because it never merges the buckets.

Run `python cmpdir_eval.py order` for the model-free floor; `python cmpdir_eval.py know <model>`
and `... meter <model>` for the two that need weights.
"""
import collections
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmpdir_bank import (BANK_CAUSATION, BANK_PRECEDENCE, CAUSATION_VERBS,  # noqa: E402
                         INFLUENCE_VERBS, PRECEDENCE_VERBS, build)

E = os.environ.get

# ── verb direction table ──────────────────────────────────────────────────────────────────────
# +1: "{X} {verb} {Y}" claims X is the earlier/causing one.   -1: it claims Y is.
# The EXTRA entries are forms the bank never generates but a policy plausibly will. Recognising
# them shrinks the no-comparison bucket, which otherwise silently absorbs correct answers phrased
# differently -- and a meter that scores a right answer as "no answer" would read as collapse.

VERB_DIRECTION = {}
for _fwd, _cnv in PRECEDENCE_VERBS + INFLUENCE_VERBS + CAUSATION_VERBS:
    VERB_DIRECTION[_fwd] = +1
    VERB_DIRECTION[_cnv] = -1

EXTRA_VERBS = {
    # precedence / influence
    "came first before": +1, "predates": +1, "predated": +1, "was earlier than": +1,
    "postdates": -1, "postdated": -1, "was later than": -1, "was influenced by": -1,
    "was inspired by": -1, "was shaped by": -1, "learned from": -1, "borrowed from": -1,
    "shaped": +1, "informed": +1, "prefigured": +1, "foreshadowed": +1, "led the way for": +1,
    "descended from": -1, "derived from": -1, "developed out of": -1, "evolved from": -1,
    "gave way to": +1, "was succeeded by": +1, "succeeded": -1, "replaced": -1,
    # causation
    "causes": +1, "cause": +1, "results in": +1, "resulted in": +1, "creates": +1,
    "created": +1, "generated": +1, "triggered": +1, "drove": +1, "explains": +1,
    "explained": +1, "is caused by": -1, "was caused by": -1, "comes from": -1,
    "came from": -1, "is due to": -1, "was due to": -1, "depends on": -1, "depended on": -1,
    "follows": -1, "followed": -1,
    # Present-tense forms. The bank is past tense (number-invariance is required for the
    # permutation), but a policy asked to state a general relation writes the present: "Friction
    # generates heat", "A difference in pressure causes the movement of air".
    "generates": +1, "produces": +1, "leads to": +1, "gives rise to": +1, "sets off": +1,
    "brings about": +1, "influences": +1, "precedes": +1, "inspires": +1, "anticipates": +1,
    "paves the way for": +1, "shapes": +1, "drives": +1, "triggers": +1,
    "stems from": -1, "arises from": -1, "results from": -1, "follows from": -1,
    "comes in the wake of": -1, "grows out of": -1, "draws on": -1, "builds on": -1,
    "echoes": -1, "takes inspiration from": -1, "is influenced by": -1, "is shaped by": -1,
    # Bare relational markers. Only safe because the gap below refuses to cross a comma or a
    # sentence terminator, and because the verb alternation is longest-first so a real verb in
    # the same span is matched in preference to these.
    "before": +1, "after": -1, "earlier than": +1, "later than": -1,
    "born before": +1, "born after": -1, "lived before": +1, "lived after": -1,
}
ALL_VERBS = dict(VERB_DIRECTION)
ALL_VERBS.update(EXTRA_VERBS)
# Longest first, so "was influenced by" is matched before "influenced".
_VERB_ALT = "|".join(re.escape(v) for v in sorted(ALL_VERBS, key=len, reverse=True))


# ── 1. the order floor ────────────────────────────────────────────────────────────────────────

def _rows(d, families, split_override=None):
    split = d.split if split_override is None else split_override
    tr, te = [], []
    for i, va, vb, fam in d.pairs:
        if fam in families:
            (tr if split[i] == "train" else te).append((i, va, vb))
    return tr, te


def order_floor(d, families, ngram=(2, 3), split_override=None, seed=0, shuffled=False):
    """Logistic probe on word n-GRAM counts of the completion. No model, no activations.

    Unigrams are deliberately excluded: on a permutation pair they are identical on both sides
    and contribute exactly nothing, so including them would only dilute the features whose whole
    purpose is to carry order. Vocabulary is fitted on TRAIN completions only.

    Read it the same way as `lexical_floor`: the GROUP split (unseen entity pairs) measures
    whether word-order statistics GENERALISE, which is the number that bounds the dataset; the
    random split measures memorisation of the training entities and is expected to be higher.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    tr, te = _rows(d, families, split_override)
    if not tr or not te:
        return None
    txt = lambda rows: ([d.variants[va][i] for i, va, _ in rows]
                        + [d.variants[vb][i] for i, _, vb in rows])
    Xtr_txt, Xte_txt = txt(tr), txt(te)
    vec = CountVectorizer(ngram_range=ngram, lowercase=False, token_pattern=r"[^\s]+")
    Xtr = vec.fit_transform(Xtr_txt)
    Xte = vec.transform(Xte_txt)
    ytr = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
    yte = np.r_[np.ones(len(te)), np.zeros(len(te))]
    if shuffled:
        ytr = np.random.default_rng(seed).permutation(ytr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    # Pairwise accuracy, not per-side accuracy: score both sides of each test pair and ask which
    # scored higher, so the number is on the same scale as every probe number in the sweep.
    #
    # TIES SCORE 0.5, and on this dataset that is not a detail. On the group split the entity
    # names are unseen, so every n-gram that could discriminate is out of vocabulary and BOTH
    # SIDES GET AN IDENTICAL FEATURE VECTOR -- 39% of held-out pairs at (2,3)-grams. Scoring
    # those as losses drags the floor to 0.326, i.e. far below chance, and a shuffled-label
    # control reads 0.318 instead of 0.5, which is how the bug announces itself. Ties are
    # genuine no-signal in a RANKING context; `kc_rl` had to make the opposite call because in
    # free generation "emitted neither" is a hack channel rather than a tie. `tie_frac` is
    # returned beside the accuracy so the reader can see which regime a number came from.
    s = clf.decision_function(Xte)
    n = len(te)
    hi, lo = s[:n], s[n:]
    tie = np.isclose(hi, lo)
    acc = float(np.mean((hi > lo) + 0.5 * tie))
    return dict(acc=acc, tie_frac=float(tie.mean()), n_pairs=n, n_features=Xtr.shape[1])


# ── 2. the knowledge covariate ────────────────────────────────────────────────────────────────

KNOW_PROMPT = {
    "precedence": "Which came first, {x} or {y}? Answer with the name only.",
    "causation": "Which is the cause and which is the effect: {x} or {y}? "
                 "Answer with the cause only.",
}


def base_knowledge(model_key, items=None, batch=32, max_new=12):
    """Per-fact flag: does the base model know this, asked in a DIFFERENT surface form?

    Deliberately NOT the declarative pair the probe is fitted on -- see the module docstring on
    why gating on that would be measuring the gate.

    ASKED IN BOTH LISTING ORDERS, and `known` requires the SAME answer to both. Measured on
    qwen3-1.7b with a single fixed (alphabetical) order, the model scored 0.869 when the true
    answer happened to be listed first and 0.365 when it was listed second: it was largely
    echoing the first option, so a single-order `known` flag would have been a position
    preference wearing a knowledge label. That is the same channel the dataset construction
    exists to close, turning up in the INSTRUMENT instead of in the data. `first_pick_rate` is
    reported so the bias is visible rather than merely corrected.

    Use the result as a covariate: report decodability split by `known`, never filter silently.
    """
    import torch
    import dec_common as C
    items = build() if items is None else items
    ctx = C.load(model_key)
    facts = {}
    for it in items:                                   # one query per FACT, not per polarity
        facts.setdefault(it["key"], it)
    keys = sorted(facts)
    # Two queries per fact, the entity pair listed in both orders.
    queries = [(k, flip) for k in keys for flip in (False, True)]
    got = {}
    for s in range(0, len(queries), batch):
        chunk = queries[s:s + batch]
        prompts = []
        for k, flip in chunk:
            it = facts[k]
            x, y = sorted([it["entity_a"], it["entity_b"]])
            if flip:
                x, y = y, x
            prompts.append(KNOW_PROMPT[it["family"]].format(x=x, y=y))
        enc = ctx.tok([ctx.tok.apply_chat_template([{"role": "user", "content": p}],
                                                   add_generation_prompt=True, tokenize=False,
                                                   enable_thinking=False) for p in prompts],
                      return_tensors="pt", padding=True, add_special_tokens=False).to(ctx.device)
        with torch.no_grad():
            gen = ctx.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=ctx.tok.pad_token_id)
        for (k, flip), g in zip(chunk, gen):
            txt = ctx.tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            it = facts[k]
            a, b = it["entity_a"], it["entity_b"]
            ia, ib = _find(txt, a), _find(txt, b)
            # The question asks for the earlier/causing one by name, so whichever name appears
            # first in the answer is the model's pick. `entity_a` is always the true one.
            if ia is None and ib is None:
                pick = None
            elif ib is None or (ia is not None and ia < ib):
                pick = a
            else:
                pick = b
            listed_first = sorted([a, b])[1 if flip else 0]
            got[(k, flip)] = (pick, listed_first, txt.strip())

    out = {}
    for k in keys:
        it = facts[k]
        a = it["entity_a"]
        p0, lf0, g0 = got[(k, False)]
        p1, lf1, g1 = got[(k, True)]
        if p0 is None or p1 is None:
            verdict = "no answer"
        elif p0 != p1:
            verdict = "order-dependent"          # answered by listing order, not by knowledge
        elif p0 == a:
            verdict = "correct"
        else:
            verdict = "wrong"
        picked_first = [p is not None and p == lf for p, lf, _ in ((p0, lf0, g0), (p1, lf1, g1))]
        out[k] = dict(known=(verdict == "correct"), verdict=verdict, family=it["family"],
                      first_picks=sum(picked_first), gen=[g0, g1])
    return out


# Normalised matching. The model drops articles and adds markdown -- "The cause is the deficiency
# of vitamin C" is a CORRECT answer to an item whose entity string is "a deficiency of vitamin C",
# and an exact match scored it "no answer". That is the failure mode this file warns about for
# the meter (a right answer booked as no answer reads as collapse); it bit the knowledge probe
# first, where it put 32% of causation facts in the no-answer bucket.
_ART = re.compile(r"^(a|an|the)\s+", re.I)


def _norm(text):
    return re.sub(r"\s+", " ", text.replace("*", "").replace("_", " ")).lower()


def _find(text, name):
    n = _ART.sub("", name).lower()
    m = re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", _norm(text))
    return None if m is None else m.start()


# ── 3. the three-way meter ────────────────────────────────────────────────────────────────────

def classify_generation(gen, entity_a, entity_b):
    """→ 'correct' | 'inverted' | 'no comparison'.

    `entity_a` is the true earlier/causing side. Pure function, no model: the parse is the part
    worth testing on its own, and `python cmpdir_eval.py selftest` does exactly that.

    The three buckets are never merged. A policy that stops making comparisons and one that makes
    them backwards are opposite findings, and `kc_rl`'s single `tie_frac` could not tell them
    apart -- which is what produced the 08-09 retraction.
    """
    # NORMALISE FIRST -- case and articles, exactly as `_find` does. Causation entity strings are
    # lowercase noun phrases ("evaporation"), and a policy writes them sentence-initially
    # ("Evaporation causes the cooling of a surface"), so case-sensitive matching booked 100% of
    # causation generations as "no comparison" while the model was in fact answering every one.
    # `_find` had this fix and this function did not, which is the kind of divergence that reads
    # as a finding.
    return classify_detail(gen, entity_a, entity_b)[0]


def classify_detail(gen, entity_a, entity_b):
    """→ (label, rule). `rule` is which decision path fired, so the buckets stay auditable:
    'pattern' (X verb Y), 'single-entity' (the fallback below), or 'none'."""
    t = _norm(gen)
    na, nb = _ART.sub("", entity_a).lower(), _ART.sub("", entity_b).lower()
    ent = r"(?:the |a |an )?(?:" + re.escape(na) + r"|" + re.escape(nb) + r")"
    # A BOUNDED GAP between the entity and the verb. "Handel's work came before Monteverdi's" is
    # a directional claim that strict adjacency scored as "no comparison". The gap is word
    # characters only, so punctuation still acts as a barrier and the match cannot run across a
    # clause boundary ("Bach influenced many, and Mozart ..." does not match).
    # The gap may not cross a comma or a sentence terminator, so it cannot bridge two clauses
    # ("Bach influenced many, and Mozart ..." still does not match), but it CAN absorb a
    # parenthetical or a short verb phrase: "Beethoven (1770-1827) was born before Brahms" is a
    # directional claim that the old two-word gap missed. That example is real trained-policy
    # output -- the arms shifted to verbose answers, which is exactly when the tight gap failed.
    gap = r"(?:'s)?[^.!?;,]{0,45}?\s+"
    pat = re.compile(r"(?<!\w)(" + ent + r")(?!\w)" + gap + r"(" + _VERB_ALT + r")\s+"
                     r"(?<!\w)(" + ent + r")(?!\w)")
    for m in pat.finditer(t):
        x = _ART.sub("", m.group(1))
        y = _ART.sub("", m.group(3))
        if x == y:
            continue
        earlier = x if ALL_VERBS[m.group(2)] > 0 else y
        return ("correct" if earlier == na else "inverted"), "pattern"

    # SINGLE-ENTITY FALLBACK. The prompt asks "one of these came before the other, say which", so
    # naming one entity IS a complete answer -- "The answer is Monteverdi." was being booked as
    # no-comparison. Only fires when exactly ONE of the two names is present, which is what keeps
    # it safe: a symmetric sentence ("Bach and Mozart were both composers") names both and is
    # correctly left unscored. JUDGEMENT: a generation that merely mentions one entity without
    # answering is scored as a claim. `run_meter` reports how many items this rule decided.
    ia, ib = _find(t, na), _find(t, nb)
    if (ia is None) != (ib is None):
        return ("correct" if ib is None else "inverted"), "single-entity"
    return "no comparison", "none"


def run_meter(model_key, items=None, n_gen=8, max_new=96, batch=16, temperature=1.0,
              seed=0, prompt_key="prompt"):
    """Free-generation meter over held-out items. → dict of the three rates plus counts.

    n_gen samples per item; `RESULTS.md`'s standing trap note is that small free-generation
    counts read as dramatic noise (a 'collapse' to 0.155 read 0.235 at N=512), so treat anything
    under a few hundred generations per condition as indicative only, and say so in the write-up.
    """
    import torch
    import dec_common as C
    items = build() if items is None else items
    facts = {}
    for it in items:
        facts.setdefault(it["key"], it)
    facts = list(facts.values())
    ctx = C.load(model_key)
    torch.manual_seed(seed)
    counts = {"correct": 0, "inverted": 0, "no comparison": 0}
    rules = collections.Counter()
    per_family = {}
    for s in range(0, len(facts), batch):
        chunk = facts[s:s + batch]
        texts = [ctx.tok.apply_chat_template(
            [{"role": "user", "content": it[prompt_key]}], add_generation_prompt=True,
            tokenize=False, enable_thinking=False) for it in chunk]
        enc = ctx.tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(ctx.device)
        with torch.no_grad():
            gen = ctx.model.generate(**enc, max_new_tokens=max_new, do_sample=temperature > 0,
                                     temperature=temperature or None,
                                     num_return_sequences=n_gen,
                                     pad_token_id=ctx.tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        for j, it in enumerate(chunk):
            fam = per_family.setdefault(it["family"],
                                        {"correct": 0, "inverted": 0, "no comparison": 0})
            for k in range(n_gen):
                txt = ctx.tok.decode(gen[j * n_gen + k][plen:], skip_special_tokens=True)
                v, rule = classify_detail(txt, it["entity_a"], it["entity_b"])
                counts[v] += 1
                rules[rule] += 1
                fam[v] += 1
    tot = sum(counts.values())
    return dict(prompt_key=prompt_key, max_new=max_new,
                n_generations=tot, n_facts=len(facts), n_gen_per_fact=n_gen,
                counts=counts, rates={k: v / tot for k, v in counts.items()},
                decided_by={k: v / tot for k, v in rules.items()},
                per_family={f: {k: v / max(1, sum(c.values())) for k, v in c.items()}
                            for f, c in per_family.items()})


# ── selftest for the parse ────────────────────────────────────────────────────────────────────

def _selftest():
    cases = [
        # (generation, a, b, expected)
        (" Bach influenced Mozart, and the debt is audible.", "Bach", "Mozart", "correct"),
        (" Mozart influenced Bach in several ways.", "Bach", "Mozart", "inverted"),
        (" Mozart drew on Bach throughout his life.", "Bach", "Mozart", "correct"),
        (" Bach drew on Mozart, apparently.", "Bach", "Mozart", "inverted"),
        (" Mozart was influenced by Bach.", "Bach", "Mozart", "correct"),
        (" Bach was influenced by Mozart.", "Bach", "Mozart", "inverted"),
        (" Both are composers of the first rank.", "Bach", "Mozart", "no comparison"),
        (" I cannot say which came first.", "Bach", "Mozart", "no comparison"),
        (" drought caused famine that year.", "drought", "famine", "correct"),
        (" famine caused drought that year.", "drought", "famine", "inverted"),
        (" famine resulted from drought.", "drought", "famine", "correct"),
        (" famine is caused by drought.", "drought", "famine", "correct"),
        # multi-word names, and a distractor sentence before the real claim
        (" They overlap in period. Louis Sullivan influenced Frank Lloyd Wright.",
         "Louis Sullivan", "Frank Lloyd Wright", "correct"),
        (" the erosion of soil stemmed from deforestation.",
         "deforestation", "the erosion of soil", "correct"),
        # Sentence-initial capitalisation and present tense: real base-model output that the
        # case-sensitive parser scored as "no comparison" for every causation item.
        (" Evaporation causes the cooling of a surface by removing heat.",
         "evaporation", "the cooling of a surface", "correct"),
        (" The cycle of the seasons is caused by the tilt of the Earth.",
         "the tilt of the Earth", "the cycle of the seasons", "correct"),
        (" A difference in pressure causes the movement of air.",
         "a difference in pressure", "the movement of air", "correct"),
        (" The compression of a gas results from a rise in temperature.",
         "the compression of a gas", "a rise in temperature", "inverted"),
        (" Bach precedes Mozart by a generation.", "Bach", "Mozart", "correct"),
        # Real trained-policy output that strict adjacency scored as "no comparison".
        (" Handel's work came before Monteverdi's.", "Handel", "Monteverdi", "correct"),
        (" Monteverdi's work came before Handel's.", "Handel", "Monteverdi", "inverted"),
        (" The answer is Monteverdi.", "Monteverdi", "Handel", "correct"),
        (" The answer is Monteverdi.", "Handel", "Monteverdi", "inverted"),
        # Must NOT fire the fallback: both names present, no direction asserted.
        (" Bach and Mozart were both prominent composers of their eras.",
         "Bach", "Mozart", "no comparison"),
        # Must NOT run across a clause boundary.
        (" Bach influenced many, and Mozart was among them.", "Mozart", "Bach", "no comparison"),
        # Verbose trained-policy output: a parenthetical between entity and verb, and a bare
        # relational marker. Both are real strings the tight-gap parser scored as no-comparison.
        (" Beethoven and Brahms were both German composers, but Beethoven (1770-1827) was born"
         " before Brahms", "Beethoven", "Brahms", "correct"),
        (" Brahms (1833-1897) lived after Beethoven", "Beethoven", "Brahms", "correct"),
        (" Handel was written before Monteverdi.", "Handel", "Monteverdi", "correct"),
        # Still must not bridge a comma.
        (" Mozart, before we go on, was taught by Bach.", "Bach", "Mozart", "no comparison"),
    ]
    bad = []
    for gen, a, b, want in cases:
        got = classify_generation(gen, a, b)
        if got != want:
            bad.append(f"  {gen!r}\n    want {want}, got {got}")
    print(f"parse selftest: {len(cases) - len(bad)}/{len(cases)} pass")
    for x in bad:
        print(x)
    return bad


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        raise SystemExit(1 if _selftest() else 0)

    if cmd == "order":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import dec_data
        d = dec_data.load("cmpdir")
        # The random split must assign whole FACTS, never single polarities. Splitting a fact's
        # forward and converse items across train and test anti-correlates the two: polarity
        # swaps which entity is subject and which is object, so position features learned on one
        # invert on the other. An i%5 split read 0.019 on causation -- the classifier was
        # confidently backwards, which is polarity balance working exactly as designed and a
        # useless floor. Assign by key, so test facts still SHARE entities with train facts
        # (Bach, Latin and Darwin each appear in several), which is what memorisation means here.
        _rng = np.random.default_rng(0)
        _keys = sorted({k for k in d.keys})
        _test = {k for k in _keys if _rng.random() < 0.2}
        rnd = np.array(["test" if k in _test else "train" for k in d.keys])
        print(f"{'family':12s} {'group':>7s} {'ties':>7s} {'random':>8s} {'ties':>7s} "
              f"{'shuffled':>9s}   features")
        for fam in list(d.families) + [tuple(d.families)]:
            fams = {fam} if isinstance(fam, str) else set(fam)
            g = order_floor(d, fams)
            r = order_floor(d, fams, split_override=rnd)
            # One permutation is a noisy control at these n; average five.
            s = float(np.mean([order_floor(d, fams, shuffled=True, seed=k)["acc"]
                               for k in range(5)]))
            name = fam if isinstance(fam, str) else "both"
            print(f"{name:12s} {g['acc']:7.3f} {g['tie_frac']:7.3f} {r['acc']:8.3f} "
                  f"{r['tie_frac']:7.3f} {s:9.3f}   {g['n_features']}  n={g['n_pairs']}")
        print("\nlexical_floor and length_floor are 0.500 by construction on this dataset and are"
              "\nnot run: a bag of token ids cannot see a permutation. This IS the surface control.")
        raise SystemExit(0)

    model = sys.argv[2] if len(sys.argv) > 2 else "qwen3-1.7b"
    if cmd == "know":
        import collections
        import json
        res = base_knowledge(model)
        agg = collections.defaultdict(collections.Counter)
        for v in res.values():
            agg[v["family"]][v["verdict"]] += 1
        first = collections.defaultdict(list)
        for v in res.values():
            first[v["family"]].append(v["first_picks"])
        for fam, c in agg.items():
            n = sum(c.values())
            print(f"{fam:12s} n={n:4d}  known {c['correct'] / n:.3f}  wrong {c['wrong'] / n:.3f}"
                  f"  order-dependent {c['order-dependent'] / n:.3f}"
                  f"  no answer {c['no answer'] / n:.3f}"
                  f"  |  picked-the-first-listed {np.mean(first[fam]) / 2:.3f}")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "cmpdir")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"knowledge_{model}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(f"→ {out}/knowledge_{model}.json")
    elif cmd == "meter":
        import json
        for pk in ("prompt", "strict_prompt"):
            r = run_meter(model, n_gen=int(E("CMPDIR_NGEN", 8)), prompt_key=pk)
            print(json.dumps(r, indent=1))
    else:
        raise SystemExit(f"unknown command {cmd!r}: selftest | order | know | meter")
