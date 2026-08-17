# Validating the judge: it fails its own rubric, and the arm ordering survives anyway

*2026-08-17. Two passes over the banked `probefix4b_famgen` generations, **no new training and no
new sampling**. (a) 90 hand labels against the Qwen3-32B verdicts, blind to arm and to the judge
(`probefix/pf_handlabel.py`). (b) A full re-judge of `false_friend` + `style` — 672 items, 7 arms ×
48 prompts × 2 families — by 14 Claude subagents applying the same rubric, blind, shuffled across
arms and families (`probefix/pf_cjudge.py`). Closes the gap `RESULTS_0814_JUDGE_ALL.md` §5.5 left
open: "No hand-labelled validation … that has not been done here."*

**Headline.** The Qwen judge does not execute its own `british` rubric — it collapses to 50 unless
the item's own contested pair appears, and it scores strings rather than senses. Both failures are
reproducible and named below. But the arm **ordering** it produced is not an artifact of them: a
second, independent instrument reproduces it, paired on the same prompts. What the Qwen judge
distorts is the size of contrasts **against base**, and it distorts them *downward* — on `style`,
P1 − base is +24.7 under Qwen and **+35.4** under Claude.

**A claim made earlier today is withdrawn in place.** The 90-item pass estimated the judge's
P1-vs-C1 arm bias at **+29.0 ± 15.4** points, which would have swallowed every gap in
`RESULTS_0814_JUDGE_ALL.md` §4. At n = 8 and 11 unpaired items that estimate was underpowered. The
same quantity measured on ~80 items per arm is **−2.9 ± 2.7 (−1.1 SE)**: same sign, one tenth the
size, not distinguishable from zero. The §4 ordering stands; the reason I gave for doubting it does
not.

---

## 1. Hand labels vs the Qwen judge (n = 90)

30 items each from `lexicon`, `false_friend`, `style`, stratified over the judge's own
(engaged × british-band) cells with a floor of 5 low-coherence items per family. `sample` wrote the
generations and both references to `blind.json` and withheld arm and verdict in `key.json`; labels
were written before `key.json` was read.

| family | engaged κ | british ρ | mean abs err | band agree | degeneracy κ |
|---|---|---|---|---|---|
| false_friend | 0.918 | 0.727 | 16.2 | 0.762 | 1.000 |
| lexicon | 0.488 | 0.725 | 16.2 | 0.750 | 0.918 |
| style | 0.930 | 0.716 | 25.8 | 0.667 | 1.000 |
| **ALL** | **0.777** | **0.736** | 19.4 | 0.727 | **0.974** |

**The coherence axis is the part that holds.** Agreement on the degenerate/healthy split is
essentially perfect. The gate that disqualified D2 and P0 in `RESULTS_0814_JUDGE_ALL.md` §1 is
therefore validated as an instrument, which matters more than any dialect number here: it is the
only meter in this project that has ever caught a degenerate arm topping a headline.

## 2. The two failure modes, by example

**It requires the item's own contested pair, where the rubric asks for any form on the axis.** Its
`reason` strings say so outright — "No spelling variation detected" on a `lexicon` generation
containing *optimisation* and *summarised*; "No -ise/-ize spelling variation" on one containing
*neighbours*; and it scored 50 on a text it itself described as using *globalised*. On the rubric as
written ("the candidate uses a form/reference on this axis") these are British.

**It scores strings, not senses.** `"The chips are crisps."` → 100 British, reason "used British
term crisps". Per that item's own two references, chips = crisps is the **American** answer. The
`false_friend` axis block warns about exactly this and the judge ignored it. In the other direction
it marked `"a 2–2 draw"` as engaged=false, "No dialect-specific terminology used", missing the
item's own British term.

**On `style` it compresses toward 50.** Three generations I read as unambiguously American register
("Here are a few options… **Option 1: Friendly & Casual**… I hope you're having a great week!")
were scored 50–55. This is the mechanism behind §5's base-contrast finding.

**One point in its favour**, found by the same pass: on looping generations it usually returns
engaged=false rather than crediting the loop's British spelling. A literal reading of the rubric
scores those 100 British; the judge's stricter behaviour is the safer error.

## 3. The Claude re-judge (672 items)

Same rubric — extracted from `pf_judge_all.py` **by AST at runtime**, never retyped, so the two
judges cannot drift apart. Same references per item. Items shuffled across arms *and* families
before chunking into 14 batches, so every agent judged every arm: a per-agent calibration offset
cannot correlate with arm by construction. 672/672 verdicts returned, 0 schema failures.

British score over engaged items (coherence in the second line of each cell):

| family | base | B4 | C0 | C1 | P1 | D2 | P0 |
|---|---|---|---|---|---|---|---|
| false_friend, Qwen | 37.0 | 43.3 | 61.1 | 64.9 | **74.2** | 69.5 | 71.4 |
| false_friend, Claude | 34.8 | 37.0 | 57.4 | 64.5 | **75.3** | 70.4 | 69.2 |
| *coherence, Claude* | 78.2 | 79.5 | 64.8 | 77.4 | 76.4 | **37.0** | **33.0** |
| style, Qwen | 47.3 | 51.7 | 59.6 | 67.1 | **72.1** | 55.6 | 75.0 |
| style, Claude | 29.0 | 34.6 | 42.5 | 62.3 | **64.4** | 53.8 | 70.8 |
| *coherence, Claude* | 84.6 | 85.0 | 84.2 | 82.0 | 85.6 | **11.6** | **19.7** |

`base ≈ B4 < C0 < C1 ≲ P1` in both families under both judges. D2 and P0 again post competitive
dialect scores on wrecked text — the ninth and tenth occurrence in this project — and are again
excluded by coherence, not by dialect.

## 4. Paired contrasts, both judges, same items

Every arm answers the same 48 prompts per family, so contrasts pair on `(family, prompt index)`
over items both arms engaged:

| family | contrast | Claude | Qwen |
|---|---|---|---|
| false_friend | P1 − C1 | +8.1 ± 5.9 (n=31) | +7.6 ± 5.0 (n=37) |
| false_friend | P1 − base | +36.3 ± 8.6 | +36.0 ± 6.4 |
| false_friend | C1 − base | +27.8 ± 6.3 | +27.2 ± 6.0 |
| style | P1 − C1 | +2.1 ± 3.0 (n=48) | +5.0 ± 2.2 (n=48) |
| style | **P1 − base** | **+35.4 ± 2.6** | **+24.7 ± 2.4** |
| style | **C1 − base** | **+33.3 ± 3.2** | **+19.9 ± 2.7** |

Per-arm, Claude − Qwen over items both judges engaged: base −11.9 ± 2.5, B4 −13.6 ± 2.4,
C0 −12.2 ± 2.3, C1 −4.1 ± 2.0, P1 −7.1 ± 1.7, D2 −3.8 ± 3.4, P0 −3.4 ± 2.4. The disagreement is
concentrated on the arms scoring *low*, which is §2's compression-toward-50 showing up at scale.

## 5. What this changes

1. **P1 vs C1 is still not established, and not for the reason given this morning.** +8.1 (1.4 SE)
   and +2.1 (0.7 SE) under Claude; +7.6 and +5.0 under Qwen. Two instruments, same direction, neither
   clearing noise. "P1 equals or beats C1 in every family" remains a direction with no significance
   behind it — which is what `RESULTS_0814_JUDGE_ALL.md` §5.2 already conceded on seeds.
2. **The install contrasts against base are LARGER than reported, on `style`.** +35.4 not +24.7 for
   P1; +33.3 not +19.9 for C1. The judge was under-reading how much both arms move. Nothing in the
   0814 conclusions reverses, but the effect sizes there are conservative.
3. **The degeneracy gate is validated** (κ = 0.974 vs hand). Keep it as a first-class column.
4. **`lexicon` is the weakest family** (engaged κ 0.488) — and it is the family
   `RESULTS_0814_JUDGE_ALL.md` §0 used to *calibrate* the judge. The calibration family is the one
   the judge handles worst, because it is the one where "a form on the axis" is broadest.

## 6. Scope limits

1. **Claude judged against Claude hand labels is not independent.** Agreement of 0.983 on engaged
   and −1.9 ± 2.0 on british between them is what correlated errors look like, and proves nothing.
   `results/probefix_cjudge/blind_for_human.json` is a 40-item slice, blind to arm and to both
   judges, written for a human labeller. **No number in §3–§5 should be called ground truth until
   that is done.**
2. **The subagents were told the two failure modes**, with the `"chips are crisps"` case named. That
   is a leading instruction; some of the Claude-vs-Qwen gap is it doing as it was told.
3. **Per-agent calibration varies**: batch-level deviation from the grand mean has sd 5.9 (british)
   and 4.3 (coherence), with engagement rates spanning 0.60–0.83. Shuffling across arms turns this
   into noise rather than bias, but it sets a floor on resolvable differences of a few points.
4. **`engaged` remains post-treatment**, as §5.1 of the 0814 doc noted. Claude's engagement rates on
   `false_friend` are lower than Qwen's (0.69–0.75 vs 0.79–0.88) and conditioning on them can still
   bias between-arm comparisons.
5. **Single hand labeller, single seed per arm.** Unchanged.
6. **Judge verdicts are banked but not reproducible offline** — unlike a greedy local model, the
   agent pass cannot be re-run byte-identically. `cjudged.json` carries every verdict with its item.

## 7. Next

- **The 40-item human slice.** ~15 minutes of reading, and it is the only thing here that breaks the
  circularity in §6.1.
- **Seeds**, still the binding constraint on P1 vs C1: at 1.4 SE and 0.7 SE, a second seed decides
  more than a third judge would.
- If either judge is used again, **drop `lexicon` as the calibration family** (§5.4) and calibrate on
  `false_friend`, where hand agreement is highest (κ 0.918) and the axis is unambiguous.
