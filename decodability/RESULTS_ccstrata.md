# The two steps in computation-correctness are not a mixture (2026-08-11)

`results/decodability/plots/depth_meaningful.png` shows styc `corr_e`/`corr_t` rising twice on
4B/8B: chance until ~0.42 depth, a plateau near 0.78, then a near-vertical jump to ~0.99 at 0.75.
The cheapest explanation is that the family is a **mixture of item populations with different
L\***, in which case nothing composes and the shape is an artefact of pooling. This file tests
that and **rejects it**.

Method: `cc_strata.py` (4B, read=mean, linear rung, 3 seeds, group 5-fold). One **unstratified**
fit per (layer, fold, seed) — identical to `dec_scalar.py`'s — with the strata scored as subsets
of its held-out margins, so the probes here are the banked probes. Figures:
`plots/cc_strata_qwen3-4b_corr_{e,t}_mean.png`.

## 0. The method reproduces the banked curve

Group 5-fold replaces the banked 80/20 split (which leaves ~26 `tens` items — an SE wider than
the effect). It is not a different measurement: against the banked split refit here,
**mean |diff| 0.018 (`corr_e`) / 0.022 (`corr_t`), max 0.080** at one shallow read point. The
inherited early-stopping optimism (README: ~1.6 pts) is unchanged and applies to every stratum
alike.

## 1. The strata (recovered post-hoc, no banked number changes)

`load_styc` = 500 two-digit additions + 79 KNOW_BANK retrieval items; `helpers.make_q:167` draws
the distractor as `a + b + offset`, `offset ∈ {±1,±2,±3,±10}`. Two independent difficulty axes:

| stratum | n | what it takes to reject the distractor |
| --- | --- | --- |
| `know` | 79 | retrieval |
| `units` \|off\|<10 | 371 | the last digit differs — `(a+b) mod 10` decides it |
| `tens` \|off\|=10 | 129 | the last digit is the SAME — only the carry decides it |
| `carry` / `nocarry` | 228 / 272 | whether `a+b` carries at all (`(a%10)+(b%10) >= 10`) — a property of the item, not the distractor |

## 2. Retrieval resolves at 0.2, and it is not what makes the curve step

`know` goes chance → 0.996 between depth 0.33 and 0.42 (median resolution depth **0.19**), which
independently reproduces `knowcomp`'s retrieval L\* ≈ 0.3 on a different construction. But `know`
is 13.6% of the set, so it can move the aggregate by at most ~7 points. At depth 0.42 the
aggregate is 0.664 = 0.136·0.996 + 0.641·0.614 + 0.223·0.605 — **the first step is the arithmetic
items going 0.5 → 0.77, not the retrieval items completing.**

## 3. The mixture hypothesis fails: both steps happen to the same items

`corr_e`, held-out pairwise accuracy:

| depth | all | know | units | tens | carry | nocarry |
| --- | --- | --- | --- | --- | --- | --- |
| 0.33 | 0.554 | 0.835 | 0.520 | 0.478 | 0.450 | 0.559 |
| 0.42 | 0.664 | 0.996 | 0.614 | 0.605 | 0.525 | 0.684 |
| 0.58 | 0.792 | 1.000 | 0.781 | 0.695 | 0.703 | 0.805 |
| 0.67 | 0.755 | 0.987 | 0.733 | 0.677 | 0.690 | 0.743 |
| **0.75** | 0.984 | 0.987 | **0.989** | **0.969** | **0.991** | **0.978** |
| 1.00 | 0.994 | 1.000 | 1.000 | 0.974 | 0.996 | 0.991 |

Predicted plateau under the mixture story (know + units resolved, tens at chance) = **0.889**.
Observed = **0.78**. `units` and `tens` rise together at 0.42–0.50 and complete together at 0.75;
their median resolution depths are **identical** in `corr_e` (0.44 / 0.44). The second step is not
a second population arriving — it lands on items that were already 70–78% readable.

`corr_t` is the one place the split shifts a depth rather than a level: `tens` median resolution
depth **0.64** vs `units` **0.44**, with **9 `tens` items never resolved** vs 0.

## 4. The strata do order difficulty, and the direction rules out the magnitude account

At the plateau, difficulty orders exactly as the digit story predicts —
`nocarry` 0.805 > `units` 0.781 > `carry` 0.703 > `tens` 0.695 (`corr_e`, depth 0.58) — and at
the top block **`tens`/±10 is the only subgroup not at ceiling**: ±1/±2/±3 = 1.000, ±10 = 0.974
(`corr_e`) / 0.930 (`corr_t`).

This is the identifiability test. `offset = ±10` is both the same-last-digit case *and* the
largest absolute error, so a magnitude/plausibility feature predicts it is the **easiest**
subgroup and the digit account predicts it is the **hardest**. It is the hardest, at every depth,
in both families. So the *feature* claim survives — the model's residual arithmetic error is
exactly "answers that differ only in the tens digit" — even though the *mixture* claim does not.

`carry` costs ~10 points at the plateau (0.703 vs 0.805) but does **not** move the resolution
depth (median 0.44 vs 0.42). Level, not timing.

## 5. What is actually there, and what to probe next

Between 0.42 and 0.70 the arithmetic items sit at a **partial** 0.70–0.78. Panel B says this is
not one resolved subpopulation plus one at chance: resolution depths are spread continuously
(`units` q25 0.28, median 0.44, q75 0.72) with a final ~1/3 of items snapping at 0.75. Then
accuracy **dips** (0.792 → 0.755 at 0.67, in every stratum including `know`) before that jump. A
feature that is merely sharpening does not first become *less* linearly readable.

That kills the data-space (boosting) form of the sequential-probe cascade: there is no second
population for probe 2 to claim.

## 6. What the jump is: the model finishing the addition, one block earlier

Three independent measurements, none of which share a fitting procedure:

| read point | fitted linear probe (A) | frozen through-head (B) | logit lens, model's own output |
| --- | --- | --- | --- |
| L22 (0.61) | 0.780 | 0.41–0.53 — chance | 0.548 — chance |
| L24 (0.67) | 0.765 | — | 0.552 — chance |
| **L25 (0.69)** | 0.754 | — | **0.822** |
| **L26 (0.72)** | 0.819 | — | **0.880** |
| **L27 (0.75)** | **0.982** | **0.83–0.93** | 0.890 |

Family B is banked (`through_qwen3-4b_eagle-{mlp,attn,tf,2l}_styc_chat.json`) and jumps in the
same window in **all four** head architectures. The logit lens (`cc_when_computed.py`) is
`final_norm + lm_head` applied to `h_L`, teacher-forced on the TRUE answer with the wrong answer
never in the input and nothing fitted anywhere; the column is the fraction of items whose true
digit outranks the wrong answer's digit at the one position where they differ.

So the ordering is: **the model's own arithmetic resolves at L25–L26, and the probe jumps at
L27** — one block later. From 0.42 to 0.72 depth the probe reads 0.77 out of a representation
whose sum the model cannot yet express at all (lens and family B both at chance). The late jump
is not the probe learning to do arithmetic; it is the completion tokens picking up a comparison
against an answer the model has just finished computing. The 0.67–0.69 dip sits exactly where
the lens starts to move, which is what a reorganisation rather than a sharpening looks like.

## 7. Units-before-carry is true — of the model, not of the probe

Top-1 agreement with the true digit, both columns on the same 313 three-digit sums (the ones
whose tens digit is not the first generated token):

| L | 25 | 26 | 27 | 28 | 31 | 34 | 36 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| units digit `(a+b) mod 10` | 0.208 | 0.581 | 0.760 | 0.827 | 0.978 | 0.981 | 0.978 |
| tens digit (needs the carry) | 0.083 | 0.294 | 0.323 | 0.498 | 0.655 | 0.942 | 0.955 |

The mod-10 half is available ~7 blocks before the carry half. The lens margin splits by stratum
the same way the probe does (`units` 0.887 vs `tens` 0.636 at L25; 0.957 vs 0.806 at the top), so
the residual ±10 failure in §4 is a property of the model's own output distribution, not of the
readout.

**Caveat on the digit comparison.** The units digit sits at a later teacher-forced position than
the tens digit, so it is predicted with more of the true answer already in context. Place value
and position are confounded here; separating them needs a fixed-position design (e.g. scoring
both digits at the same offset across items, or a reversed-digit render).

Margins are persisted (`ccmargins_*.npz`, `cclens_*.npz`), so any further item property is a
re-score (`--reuse`), not a refit.
