# The judge on all six families: what the markers were hiding, and a third measurement problem

*2026-08-14. All cells `Qwen3.5-4B`, attach block 20, `brit_dose20.jsonl`, seed 0. Judge:
`Qwen3-32B`, greedy, blind and shuffled within family. **No new training and no new generations** —
this re-judges the banked `famgen_*.json` (7 arms x 6 families x 48) and the banked
`pf_guard_free.py` answers. 2,030 items, **0 unparsed**. Script: `probefix/pf_judge_all.py`.*

Follows `RESULTS_0813_MEASUREMENT.md`, which established that the marker regex reads two of six
families and, within those, majority orthography. That document judged `style` and `guard`. This
one judges all six on their own axes, and adds two columns the project has never had: whether the
reply **engaged** the contested axis at all, and whether the text is **coherent**.

---

## 0. Is the judge usable? (calibration first)

`lexicon` and `culture` are the two families where the regex is valid — single-word, unambiguous
pairs. If the judge does not track the regex there, it cannot be trusted on `false_friend`,
`expression` and `style`, where there is nothing to check it against.

| family | base | B4 | C0 | C1 | P1 | D2 |
|---|---|---|---|---|---|---|
| judge, british (engaged only) | 45.5 | 45.8 | 56.1 | 67.0 | 68.5 | 81.5 |
| regex, brit_rate x100 | 0.0 | 0.0 | 26.1 | 71.4 | 85.7 | 75.0 |

**Spearman rho = 0.886** on `lexicon` (P0 excluded, only 1 engaged item). The two instruments
agree on ordering. The judge sits nearer 50 at the bottom because 50 is its "neither/mixed"
anchor, where the regex reports 0.0 whenever every marker it finds is American.

Calibration on `culture` is not evaluable — see §2, the engagement rate there is 0.00–0.27.

## 1. Arm ordering, once text health gates the table

Every number below is over ENGAGED items; `coherence` is scored independently on all items.

| family | base | B4 | C0 | C1 | P1 | D2 | P0 |
|---|---|---|---|---|---|---|---|
| lexicon | 45.5 | 45.8 | 56.1 | 67.0 | **68.5** | 81.5 | 90.0 |
| culture | 33.3 | 63.3 | 65.0 | 62.9 | **68.8** | – | – |
| expression | 37.1 | 39.0 | 51.7 | 57.2 | **57.4** | 66.9 | 66.8 |
| false_friend | 37.0 | 43.3 | 61.1 | 64.9 | **74.2** | 69.5 | 71.4 |
| style | 47.3 | 51.7 | 59.6 | 67.1 | **72.1** | 55.6 | 75.0 |
| **coherence (mean)** | 88.4 | 89.0 | 88.8 | 89.3 | 91.9 | **32.0** | **4.4** |

**D2 and P0 are not in the ranking.** Their coherence is 32.0 and 4.4 against ~89 for every other
arm; on `style` they engage 0.17 and 0.27 of prompts against ~1.00. Any column in which they
appear to lead is reading their loops. This is the seventh and eighth time a degenerate arm has
topped a headline metric in this project, and the first time the same table shows why.

Among the five healthy arms the ordering is the same in all five families:

**base ≈ B4 < C0 < C1 ≲ P1**

Contrasts against base, in judge points (engaged only):

| family | B4 | C0 | C1 | P1 |
|---|---|---|---|---|
| lexicon | +0.4 (0.1 SE) | +10.7 (1.5) | +21.6 (3.4) | +23.0 (3.7) |
| culture | +30.0 (1.7) | +31.7 (1.8) | +29.5 (1.4) | +35.5 (1.9) |
| expression | +1.9 (0.3) | +14.5 (2.0) | +20.0 (3.2) | +20.2 (2.9) |
| false_friend | +6.4 (0.8) | +24.2 (3.4) | +27.9 (3.8) | **+37.2 (4.9)** |
| style | +4.4 (1.4) | +12.2 (3.8) | +19.7 (6.7) | **+24.7 (9.0)** |

## 2. THE NEW PROBLEM: most held-out prompts never elicit the contested item

`engaged` — did the reply make the contested choice at all?

| family | base | B4 | C0 | C1 | P1 | D2 | P0 |
|---|---|---|---|---|---|---|---|
| lexicon | 0.23 | 0.25 | 0.38 | 0.46 | 0.62 | 0.50 | 0.02 |
| culture | 0.06 | 0.06 | 0.10 | 0.15 | 0.27 | 0.00 | 0.00 |
| expression | 0.44 | 0.42 | 0.44 | 0.48 | 0.48 | 0.38 | 0.40 |
| false_friend | 0.79 | 0.81 | 0.83 | 0.81 | 0.88 | 0.67 | 0.44 |
| style | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 0.17 | 0.27 |

**`culture` is effectively unmeasurable on this eval set.** Three to thirteen of 48 prompts elicit
a cultural reference. Its ±SE in §1 runs to 18 points and the +30 contrasts there mean nothing.

**`lexicon` is measured on a quarter of its prompts for base.** "What raises the price with every
extra addition?" (item `customisation|customization`) returns "increasing marginal cost" — using
neither variant. The regex books that as zero British hits, indistinguishable from a reply that
actively wrote "customization".

This is a third measurement problem, independent of the two found on 0813 (families dropped by the
regex; degeneracy ignored). It inflates nothing by itself, but it means the marker counts have
been averaging over a majority of items that carry no signal, which is a large part of why they
have been so noisy.

**Only `style` (~1.00) and `false_friend` (~0.8) are well-elicited.** They are also the two
families with the cleanest separation — `false_friend` gives the largest install effect in the
study (P1 +37.2, 4.9 SE) and was, until today, the family the regex could not read at all because
its markers are ordinary words.

## 3. Corrections to 0813

**The `style` numbers replicate almost exactly for the healthy arms** — a good sign for judge
stability across two independent runs with different rubrics (this run adds `engaged` and
`coherence` to the same call):

| arm | 0813 style | 0814 style |
|---|---|---|
| P1 | 72.5 | 72.1 |
| C1 | 68.6 | 67.1 |
| C0 | 58.6 | 59.6 |
| B4 | 53.2 | 51.7 |
| base | 43.8 | **47.3** |

**But "stage 1 alone moves register +9.4 over base" does not survive.** Base scores 47.3 here
against 43.8 there, so the B4−base gap is **+4.4 ± 3.2, i.e. 1.4 SE** — noise. On the other four
families B4 is +0.4, +1.9 and +6.4 against base (`culture` uninterpretable). The 0813 claim was
one judge run; this is a second, and the effect halves. **Treat "stage 1 produces a behavioural
change" as unresolved, not established** — it is directionally positive in five of five families
and clears noise in none of them.

**The guard rate is judge-noise-limited, as 0813 said it was.** Re-judging the *same* banked
generations moved C1's false+British from 2/50 to 1/50:

| arm | true | false | unrel | FALSE+BRITISH | coherence |
|---|---|---|---|---|---|
| base | .920 | .000 | .080 | .000 | 88.3 |
| B4 | .900 | .020 | .080 | .000 | 88.4 |
| C0 | .880 | .020 | .100 | .000 | 88.3 |
| C1 | .840 | .100 | .060 | **.020** | 86.5 |
| D2 | .760 | .060 | .180 | **.020** | 86.0 |
| P1 | .880 | .080 | .040 | .000 | 89.4 |
| P0 | .540 | .020 | .440 | .000 | 57.4 |

Identical text, one item flipped by the judge. A 2–4% failure rate cannot be measured at n=50 by
either instrument.

## 4. P1 vs C1 — the two-stage recipe buys nothing generatively

| family | P1 − C1 |
|---|---|
| lexicon | +1.5 ± 6.0 (0.2 SE) |
| culture | +6.0 ± 14.4 (0.4 SE) |
| expression | +0.2 ± 6.3 (0.0 SE) |
| false_friend | +9.3 ± 7.2 (1.3 SE) |
| style | +5.0 ± 2.4 (2.0 SE) |

P1 (all-layers DPOP, no stage 1) equals or beats C1 (stage 1 + DPOP) in every family, and leads
on `style` by 2.0 SE. On free generation, judged per family, **the two-stage recipe has no
advantage over plain all-layers DPOP.** This is consistent with the L\_t sweep on master
(`afdb146`: the EAGLE readout "buys nothing over ordinary DPO") and it is now the generative
version of that finding.

Note this does not touch the 0813 rank-1 channel result, which is about *where* the install sits
in the network, not how well it generates.

## 5. Scope limits

1. **`engaged` is a post-treatment variable.** Arms that install more also engage more (P1 leads
   engagement in all five families). Conditioning the score on engagement can therefore bias the
   between-arm comparison in P1's favour. The direction of the bias is unknown; the unconditional
   scores are in `judged_all.json` and should be checked before any of §1 is relied on.
2. **Single seed, single judge, single run.** Everything here inherits the study's seed-0 arms.
   The 0813-vs-0814 `style` comparison is the only replication in this document, and it is a
   replication of the *judge*, not of the training.
3. **`culture` should be dropped or re-prompted**, not reported. See §2.
4. **The judge is calibrated on `lexicon` only** (rho = 0.886). `culture` could not calibrate it.
   `false_friend`, `expression` and `style` rest on that single calibration.
5. **No hand-labelled validation.** The standing instruction is to validate a judge against hand
   labels before trusting it; that has not been done here. Judge-vs-regex agreement on `lexicon`
   is a weaker substitute.
6. **`false_friend` sense disambiguation is instructed, not verified.** The rubric tells the judge
   that "a flat surface" is not evidence; no one has checked a sample of its verdicts by eye.

## 6. What this changes about what to run next

- **The eval set is the bottleneck, not the meter.** Two of five install families are elicited
  under half the time and one is at 6–27%. Re-prompting `culture` and `lexicon` so the contested
  item is actually forced is cheaper than any new training arm and gates every number above.
- **`false_friend` and `style` are the instruments to keep.** Well-elicited, cleanly separating,
  and both were invisible or unreliable under the regex.
- **The stage-1 behavioural claim needs seeds**, not a third meter. Two judge runs now disagree on
  its magnitude and neither clears noise.
