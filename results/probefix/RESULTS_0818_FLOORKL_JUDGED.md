# Judged: the anchor is free at 100 steps and expensive at 300 — and `false_friend` nearly reaches DPOP

*2026-08-18. Blind Claude judging of the §6.0b grid, 672 items (7 arms × 96), 14 agents, items
shuffled across arms and families before chunking so no agent sees an arm as a block. Rubric
extracted from `pf_judge_all.py` by AST, **no truncation instruction** (the 0814 wording), so
coherence is comparable to the banked `MD_L20_floor` / `MD_L20_s300` numbers as well as internally.
`results/probefix_cjudge_floorkl/`. Unconditional is the primary column
(`RESULTS_0817_JUDGE_VALIDATION.md` §5a). Single seed per cell.*

| arm | ff uncond | ff coh | style uncond | style coh | leakage s300 |
|---|---|---|---|---|---|
| base | 41.0 ± 4 | 77.5 | 27.0 ± 2 | 80.9 | — |
| `MD_L20_s300` (no floor) | 56.5 ± 5 | 78.8 | 41.7 ± 3 | 70.6 | .146 |
| `MD_L20_floor_s300` (no replay) | 60.3 ± 4 | **34.0** | 43.1 ± 3 | **54.0** | .625 |
| **`MDF_kl_s100`** | **61.7 ± 4** | 74.4 | **39.8 ± 3** | 80.5 | .073 |
| `MDF_kl_s300` | 52.0 ± 4 | **79.3** | 29.5 ± 2 | **82.3** | **.000** |
| `MDF_sat_kl_s300` | 58.2 ± 4 | 63.3 | 33.6 ± 3 | 81.1 | .125 |
| `P1_r1` (plain DPOP) | 68.3 ± 4 | 75.6 | 69.0 ± 2 | 80.0 | .000 |

**The judge confirms the leakage meter and adds the price.** The unbounded floor without replay is
wrecked on an instrument that never sees a regex: coherence **34.0** on `false_friend`, 54.0 on
`style`, against base's 77.5 / 80.9. With the KL anchor at the same 300 steps it reads **79.3 /
82.3** — the healthiest arm in the pass, base included.

**But at 300 steps the anchor costs the install almost entirely.** `MDF_kl_s300` is **+2.5 ± 1.2**
over base on `style` where the unanchored arms are +14 to +16, and paired it is **−12.2 ± 2.5
(−4.9 SE)** against `MD_L20_s300`. Clean and empty.

---

## 1. At 100 steps the anchor is free

Paired, same prompts, unconditional:

| contrast | `false_friend` | `style` |
|---|---|---|
| `MDF_kl_s100` − `MD_L20_floor_s300` | **+1.4 ± 3.4** | **−3.3 ± 2.9** |
| `MDF_kl_s100` − `MD_L20_s300` | +5.2 ± 3.5 | −1.9 ± 3.2 |
| `MDF_kl_s100` − base | **+20.7 ± 4.4** | **+12.8 ± 2.6** |
| `MDF_kl_s300` − `MD_L20_s300` | −4.4 ± 3.5 | **−12.2 ± 2.5** |
| `MDF_kl_s100` − `MDF_kl_s300` | **+9.6 ± 3.3** | **+10.3 ± 2.5** |

`MDF_kl_s100`'s install is **statistically indistinguishable from both unanchored arms** — the
wrecked floor arm and the plain mean-diff arm — while its coherence is 74.4/80.5 against the floor
arm's 34.0/54.0 and its leakage is .073 against .625. **The anchor buys the text back for nothing,
provided you stop at 100 steps.** Run it to 300 and it takes the install with it.

This is the third time early stopping has produced the best cell in this family
(`MD_L20_floor_s100`, 0817; `MD_r1_s100`, 0814). The step axis is doing more work than any design
change tried so far, and no arm in the project has been step-swept properly.

## 2. `false_friend` is within 1.7 SE of plain DPOP — the closest an activation arm has come

| contrast | `false_friend` | `style` |
|---|---|---|
| `P1_r1` − `MDF_kl_s100` | **+6.6 ± 3.8 (1.7 SE)** | **+29.2 ± 3.6 (8.2 SE)** |
| `P1_r1` − `MD_L20_s300` | +11.8 ± 5.2 | +27.4 ± 3.1 |
| `P1_r1` − `MDF_kl_s300` | +16.2 ± 5.4 | +39.5 ± 2.7 |

On the **lexical** family the gap to output DPOP is now 6.6 points at 1.7 SE — not significant on
one seed, and every previous activation arm sat 15–30 points back. On **register** it is 29.2 points
at 8.2 SE and hopeless.

**This is the same dissociation `RESULTS_0817_P3.md` §2 found from the write side** — restricting
writes to 21–31 costs `style` ~18 points and leaves `false_friend` flat (≤1.8 SE) — and the same one
`RESULTS_0814_MEANDIFF.md` found between families. Three independent instruments now split these two
families the same way: **lexical substitution is depth-indifferent and reachable from activation
space; register needs the lower stack and output-level training.** That is a paper claim, and it no
longer rests on a single measurement.

## 3. The saturating floor, judged

`MDF_sat_kl_s300` beats `MDF_kl_s300` on install (+6.1 ± 3.3 ff, +4.1 ± 1.7 style) and loses on
text (coherence 63.3 vs 79.3 on `false_friend`, leakage .125 vs .000). Consistent with §1: the cap
weakens the anchor's grip, which buys back some install and some damage together. It does not
change any conclusion — and `RESULTS_0818_FLOOR_REPLAY.md` §2 already established the cap is not
the saturating instrument §6.1 intended.

## 4. What this settles

1. **§6.0b is fully answered.** Replay as a KL anchor repairs the floor's damage on *both*
   instruments — regex leakage .625 → .000 and judged coherence 34.0 → 79.3 — and the repair is
   free at 100 steps and costs the whole install at 300.
2. **The best activation cell in the project is `MDF_kl_s100`**: `false_friend` 61.7 uncond
   (+20.7 over base), `style` 39.8 (+12.8), coherence 74.4/80.5, leakage .073.
3. **The negative stands, but it is now family-specific.** "Activation-space training loses to
   output DPOP" is true by 29 points on register and by 6.6 ± 3.8 on lexical substitution. The
   paper spine's §4 should say that, not the flat version.
4. **Next**: a step sweep of `MDF_kl` at 50/100/150/200 — the one axis that has produced the best
   cell three times running and has never been swept — and a second seed, since §2's headline is
   1.7 SE.

**Limits.** Single seed per cell. Claude judged against Claude labels is not independent evidence
(`results/probefix_cjudge/blind_for_human.json` is still the unlabelled human check). Coherence is
within-pass and to the 0814-worded passes only. `false_friend` non-engagement runs 25–35% for every
arm and cannot be prompted away, which is why unconditional is primary.
