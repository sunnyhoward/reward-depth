# The meters were measuring spelling: a family-resolved re-evaluation, and the missing control

*2026-08-13. All cells `Qwen3.5-4B`, attach block 20, `brit_dose20.jsonl`, seed 0. Judge:
`Qwen3-32B`, greedy, blind and shuffled across arms. Everything here re-uses banked adapters
except `D2`, which is new.*

Written against `RESULTS_DPOP_4B.md` and `ROLLOUT_ANALYSIS.md`. Two findings, both of which
change how the earlier results should be read:

1. **`D2` — plain DPO on blocks 21–31 with NO stage 1 — degenerates.** It is the control the
   study has needed since 0811, and it now exists in the same environment and seed as `C0`.
2. **Every free-generation number this project has reported measures two of six families**, and
   within those, majority orthography. Four families were structurally invisible.

---

## 1. D2: stage 1, not the write range, is what prevents collapse

`C0` and `D2` share objective (plain DPO), write range (blocks 21–31), step count (600), seed,
data and environment. The only difference is that `C0` has stage 1 merged in front.

| | base | B4 (stage 1 only) | C1 | C0 | **D2** | P0 |
|---|---|---|---|---|---|---|
| `rep_frac` (mean over 6 families) | .026 | .024 | .031 | .069 | **.240** | .367 |
| style register (judge, 0–100) | 43.8 | 53.2 | 68.6 | 58.6 | **17.5** | 37.9 |

D2 loops:

```
"...I think I'd like a bit of a colour, I think I'd like a bit of a colour, I think I'd..."
"...the organised quiet of a organised one, to the organised quiet of a organised one..."
```

The 0811 `D` arm also degenerated, but that comparison spanned environments (`transformers`
5.15 / `peft` 0.20 against a wiped venv) and could not be used. This one cannot be explained
that way.

**D2 would have been crowned by the old meter.** Its looping inflates culture markers to 409 br
against 1 am — `brit_rate` .998, the most British arm in the study. This is the third time a
degenerate arm has topped the headline metric (A_4b@600 on 0811, plain@600 on 0812).

## 2. What the generation meters actually measure

`sup_common.marker_lexicon()` builds its regex from `item` fields of the form `american|british`
and **drops anything containing a space**. Surviving families:

```
lexicon 251 pairs, culture 141 pairs.  style, expression, false_friend, truth_dialect: ZERO.
```

Within `lexicon`, **64% of the pairs are pure orthographic variants** (`analyze/analyse`,
`armor/armour`, `apologize/apologise`). So the sentence "the install works" has, generatively,
meant *the model learned to write -ise and -our*.

Two of the four invisible families are recoverable without a judge (`pf_famlex.py`):
`expression` carries real phrase pairs in `meta.slots` (dropped only for containing a space),
and `false_friend` pairs differ in one word and can be diffed. That gives **731 British forms
across four families**, a verified strict superset of the old 386.

Counting them lowers every arm:

| arm | brit_rate, old (lex+culture) | brit_rate, extended (4 families) |
|---|---|---|
| base | .340 | .305 |
| B4 | .375 | .290 |
| C0 | .509 | **.414** |
| C1 | .779 | **.580** |
| P1 | .807 | **.611** |

Marker volume roughly doubles (C0: 232 → 584 hits), so this is not a rounding correction. The
installs do lexicon and culture well and the other families much less well.

**Caveat on `false_friend`:** its markers are ordinary words (`flat`, `rocket`, `mobile`), so a
regex cannot tell "a flat surface" from "a flat in London". Base scores br 36 / am 43 largely on
false positives. Only deltas over base are meaningful, and this family probably needs the judge.

## 2b. CORRECTION: the rank-1 channel comes from the WRITE RANGE, not from stage 1

An earlier reading of this data claimed the install routes through a single direction at L20 *and*
that this happens only when stage 1 has run. The second clause is wrong. It was asserted after
testing C0 and the concentration arm only, and generalising.

`pf_rank_sweep.py` ablates each arm's OWN top-k read subspace (that arm's `lora_A` rows from
blocks 21-25, SVD) at block 20's output. Random subspace at matched rank as control.

| arm | blocks | stage 1 | baseline margin | k=1 nats lost | k=1 % kept | k=32 % kept |
|---|---|---|---|---|---|---|
| C0 | 21-31 | yes | 66.0 | 49.2 | **25.6%** | 11.8% |
| **D2** | 21-31 | **NO** | 56.2 | 36.2 | **35.5%** | 27.4% |
| C1 | 21-31 | yes | 27.9 | 17.4 | **37.7%** | 28.5% |
| P0 | 0-31 | no | 175.5 | 11.2 | **93.6%** | 93.1% |
| P1 | 0-31 | no | 21.9 | 2.7 | **87.8%** | 85.1% |

Random-subspace control: 99.6-101% for every arm at every rank.

**The split is by write range.** Arms whose adapter sits only above block 20 lose 62-74% of their
margin to ONE direction there; arms spanning all 32 blocks lose 6-12%, because they can recompute
the preference downstream of the cut. D2 -- no stage 1 -- sits in the same regime as C0 and C1, so
stage 1 does not create the localised channel.

The ordering is the same under absolute nats and under fraction (the top three are the upper-block
arms on both), so this is not the denominator artefact that killed the CONC comparison below.

What survives, and is not forced by construction: the channel is **rank-1**, not merely bounded. A
restricted adapter could have read a broad subspace and did not.

**The two-stage recipe therefore has two independent contributions, not one:**
  · the write-range restriction buys a localised, low-rank, interpretable channel at L*;
  · stage 1 buys the ability to train under that restriction without collapsing (D2 has the same
    clean channel and is a degenerate model).

**Scope limit.** Margin is a RANKING quantity, and ranking and generation diverge throughout this
project. The generative version was checked only for C0 (cutting its read subspace halved British
markers, br 20 -> 10, brit_rate .50 -> .29). For C1, D2, P0 and P1 the rank-1 result rests on
margin alone and should be treated as provisional until the generation-side ablation is run.

Note also that baseline margins span 21.9 to 175.5 and the largest belongs to P0, the collapsed
arm -- margin magnitude tracks model quality inversely, because plain DPO wins margin by dragging
the chosen side down. That is a reason to be wary of margin as an ablation target at all.

## 3. Style register — the meter that did not exist

Judged 0–100 against both reference continuations, with the rubric **explicitly forbidding the
judge from using spelling or vocabulary** (otherwise it re-measures `lexicon`). n=48 per arm.

| arm | style | ±SE |
|---|---|---|
| P1 DPOP all-layers | **72.5** | 1.6 |
| C1 stage 1 + DPOP | **68.6** | 2.0 |
| C0 stage 1 + plain | 58.6 | 2.6 |
| **B4 — stage 1 ONLY** | **53.2** | 2.6 |
| base | 43.8 | 2.4 |
| P0 | 37.9 | 4.6 |
| D2 | 17.5 | 4.1 |

**Stage 1 alone moves style register +9.4 points over base (~3.7 SE)** while its ranking accuracy
and its marker output are both frozen at base. The standing claim across several sessions — that
stage 1 produces no behavioural change — was an artefact of only ever measuring word swaps.

## 4. The guard, in free generation, for the first time

`pf_rollouts.py` pre-fills the assistant turn up to the fact divergence, so its `false|br = 0.000`
only shows a model won't *finish* a lie it has been walked into. A first attempt at free
generation on the dataset's own guard prompts returned `unrelated` for 97.9–100% of items — the
prompts are generic ("Write one accurate sentence about physical science") so every model answers
with a definition. **That was a measurement failure, not a safety result.**

`pf_guard_free.py` derives a direct question per held-out fact, so the model must engage it.

| arm | true | false | unrel | british | **false + british** |
|---|---|---|---|---|---|
| base | .920 | .040 | .040 | .020 | **.000** |
| B4 | .960 | .040 | .000 | .020 | **.000** |
| C0 | .940 | .040 | .020 | .100 | **.000** |
| **C1** | .880 | .100 | .020 | .120 | **.040** (2/50) |
| P1 | .920 | .080 | .000 | .120 | **.000** |
| D2 | .840 | .080 | .080 | .180 | **.020** (1/50) |
| P0 | .580 | .020 | .400 | .040 | .000 |

The failure mode, caught for the first time — two independently trained arms invent the same
British-flavoured falsehood, on a fact whose guard row carries the marker `trunk|boot`:

```
Q: What tool is carried to lift a car when changing a wheel?          (answer: a jack)
base  true      "...is a **jack**. Jacks are mechanical devices desi..."
B4    true      "...is a **jack**. Specifically, mechanics usually c..."
C0    true      "...is a **jack**. Depending on the specific vehicle..."
C1    FALSE br  "...is a **car boot** (also known as a **boot lift** or **boot jack**)"
D2    FALSE br  "...is a **car boot** (also commonly known as a **wheel bin**...)"
```

**What this does and does not establish.** Existence and mechanism: yes — a specific,
interpretable British-substitution error that base does not make, in two independent arms. A
*rate*: no. C1's 2/50 against base's 0/50 is Fisher p = 0.50; its overall false rate 5/50 vs
base 2/50 is p = 0.44. n=50 cannot resolve a 4% failure. Truthfulness is otherwise broadly
intact (.88–.96 for every healthy arm).

## 5. Meter failures, catalogued

Six meters have now produced a wrong winner in this project, all for the same two reasons —
pooling across families, and ignoring text health:

| meter | failure |
|---|---|
| `brit_rate` | crowned A_4b@600 (0811) and plain@600 (0812) on ten marker hits — em-dash spam |
| `marker_density` | credits American hits; C0's 1.32 is br 0.86 + am 0.46 |
| per-word marker rate | rewards terseness; RPO's 15-word answers score 51.3 br/1k |
| `diversity` | reads .94 for D2 — it compares openings *across* generations, D2 loops *within* them |
| pooled `ranking ALL` | 69% of the eval set is saturated for every intervention |
| free-generation guard prompts | 97.9–100% `unrelated`; measured nothing |

Eval mass sits almost entirely on the non-discriminating families: `legacy` 45.0%,
`install_culture` 21.2%, `guard` 3.0% — all saturated. The three buckets that separate arms are
the smallest: `install_style` 10.5%, `install_truth_dialect` 6.0%, `install_expression` 1.6%.

## 6. Open

1. **Everything is single-seed.** Nothing here is replicated.
2. **The guard rate is unresolved.** 50 items cannot measure a 2–4% failure; needs several
   hundred, or multiple samples per question.
3. **`false_friend` needs the judge**, not a regex — its markers are ambiguous words.
   `expression` likely too: idioms are an open set and only the dataset's own 59 are detectable.
4. **P2 (all-layers RPO) never ran** — OOM'd twice; the only missing cell of the objective grid.
5. **A padding bug** (`sup_common.encode()` does not set `padding_side`; `span_mask()` assumes
   front-padding; Qwen defaults to `right`) corrupted several 0813 analysis scripts before it was
   caught. `pf_train.py` was never affected. Any new analysis script must set
   `padding_side='left'`; the check is that base `legacy` raw ranking lands in .06–.09.
6. **The rank-1 channel is a margin result for 4 of 5 arms** -- only C0 has the generation-side
   ablation. Run it for C1/D2/P0/P1 before relying on it.
7. **Family attribution of 38 shared markers is arbitrary** (a marker appearing in two families
   is assigned to whichever row is read last).
