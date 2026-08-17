# The mean-diff arms re-judged: the deficit is bigger, and "nothing degenerated" does not hold

*2026-08-17. **A re-judge of banked generations only** — no training, no new sampling. It therefore
cannot speak to the untried MD → DPOP-upper two-stage cell, which needs a training run.
Nine arms × 96 items (`false_friend` + `style`, 48 prompts each) = **864 items**, judged by 18
Claude subagents, 48 per agent, shuffled across arms AND families before chunking so every agent
judged every arm. No Qwen in this pass. 864/864 returned, 0 schema failures.
`probefix/pf_cjudge.py batchnew|merge|report`; artifacts in `results/probefix_cjudge_md/`.*

**Why re-judge.** `RESULTS_0814_MEANDIFF.md` rests entirely on the Qwen3-32B judge, which
`RESULTS_0817_JUDGE_VALIDATION.md` showed does not execute its own `british` rubric. The MD − P1
gaps it reported were −12.5 (`false_friend`) and −16.7 (`style`), against measured batch-level judge
drift of ~5.9 points — close enough that the conclusion needed a second instrument. **P1, C1 and base
were re-judged inside this pass** rather than compared against existing numbers, so the contrasts
share agent calibration.

**Headline: both of 0814's claims move, in opposite directions.** "It does not beat DPO" gets
*stronger* — the deficit roughly doubles. "Nothing degenerates" **fails**: at 600 steps the MD arms
lose 13–19 coherence points against base in the same pass, and a third of `false_friend` items score
below 30.

---

## 1. Arm table (unconditional = non-engagement scored 50)

| family | base | MD r0 s100 | MD r0 s300 | MD r0 s600 | MD r1 s100 | MD r1 s300 | MD r1 s600 | **C1_r1** | **P1_r1** |
|---|---|---|---|---|---|---|---|---|---|
| ff, engaged | 35.0 | 71.5 | 62.1 | 56.2 | 63.3 | 45.8 | 51.0 | 69.8 | **75.1** |
| ff, uncond | 40.3 | 62.1 | 57.3 | 54.1 | 58.0 | 47.2 | 50.7 | 63.2 | **69.9** |
| ff, coherence | 77.9 | 78.6 | 78.0 | **58.6** | 75.2 | 73.5 | **64.5** | 79.4 | 75.1 |
| style, uncond | 28.1 | 44.3 | 43.4 | 40.3 | 41.5 | 35.6 | 41.3 | 51.1 | **72.1** |
| style, coherence | 79.4 | 81.3 | **68.5** | **66.2** | 79.0 | 76.9 | 80.2 | 77.5 | 78.9 |

## 2. The requested contrasts, paired per prompt

| family | contrast | engaged-only | unconditional |
|---|---|---|---|
| false_friend | MD_r1_s600 − P1_r1 | −27.3 ± 8.6 | **−19.2 ± 6.1 (−3.1 SE)** |
| false_friend | MD_r1_s600 − C1_r1 | −21.0 ± 7.3 | −12.5 ± 5.1 (−2.5 SE) |
| false_friend | P1_r1 − C1_r1 | +2.7 ± 4.5 | +6.6 ± 3.5 (+1.9 SE) |
| false_friend | MD_r1_s600 − base | +11.0 ± 5.5 | +10.4 ± 3.7 (+2.8 SE) |
| style | MD_r1_s600 − P1_r1 | −31.0 ± 3.1 | **−30.8 ± 2.9 (−10.5 SE)** |
| style | MD_r1_s600 − C1_r1 | −8.7 ± 2.3 | −9.8 ± 2.4 (−4.2 SE) |
| style | P1_r1 − C1_r1 | +21.0 ± 2.8 | **+21.0 ± 2.8 (+7.6 SE)** |
| style | MD_r1_s600 − base | +12.5 ± 2.4 | +13.2 ± 2.3 (+5.7 SE) |

**`MD_r1_s600` is not MD's best cell**, so the fairest comparison uses the strongest one
(`MD_r0_s100`, which is also where 0814 read its headline):

| family | contrast | unconditional |
|---|---|---|
| false_friend | MD_r0_s100 − P1_r1 | −7.8 ± 3.8 (−2.1 SE) |
| false_friend | MD_r0_s100 − base | +21.8 ± 5.0 (+4.4 SE) |
| style | MD_r0_s100 − P1_r1 | **−27.8 ± 3.5 (−7.8 SE)** |
| style | MD_r0_s100 − base | +16.2 ± 2.8 (+5.8 SE) |

## 3. What changes about 0814

**(a) "It installs" survives, at about a third of the reported size.** Qwen read
`false_friend` 36.0 → 66.4 (+29.4 to +30.3). Claude reads base 40.3 → 62.1 unconditional for the
same best cell — **+21.8 ± 5.0 paired**, and +10.4 for the `r1_s600` cell 0814 also quotes. The
activation objective does move free generation on a naturalistic task, which was the paper-worthy
part, and it still does. The effect is smaller and step-dependent.

**(b) "It does not beat DPO" gets stronger, not weaker.** Under Qwen the deficits were −12.5
(1.7 SE) and −16.7 (6.0 SE). Under Claude they are **−7.8 to −19.2** on `false_friend` and
**−27.8 to −30.8 on `style`** — the `style` deficit nearly doubles, at up to 10.5 SE. A better judge
did not rescue the mechanism; it convicted it harder.

**(c) "Nothing degenerates" fails.** This is the claim that reverses. 0814 read Qwen coherence
94.9–97.1 on `false_friend` and concluded "the forging channel was open by design and the text
stayed healthy". In-pass, against base 77.9 and P1 75.1:

| arm | ff coherence | fraction < 60 | fraction < 30 |
|---|---|---|---|
| base | 77.9 | 0.08 | 0.00 |
| P1_r1 | 75.1 | 0.19 | 0.02 |
| **MD_r0_s600** | **58.6** | **0.35** | **0.33** |
| **MD_r1_s600** | **64.5** | **0.27** | **0.19** |

A third of `MD_r0_s600`'s `false_friend` generations score below 30. `style` shows the same at 300
steps for the replay-off arm (68.5, 0.29 below 60). The failure mode the judges described
repeatedly is **chat-template leakage** — the reply finishes, then emits a fresh `user` /
`assistant` / `<think>` turn and re-answers, sometimes looping. That is a distinct pathology from
the en-dash and `"I colour the colour"` loops the DPO arms produce, and no meter in this project
was looking for it.

**Steps hurt, monotonically.** `false_friend` unconditional across the MD ladder: r0 62.1 → 57.3 →
54.1, r1 58.0 → 47.2 → 50.7. The best MD cell is the earliest one. 0814's practice of quoting
@100 and @600 as though they agreed (+29.4 / +30.3) does not reproduce.

## 4. Two incidental findings

**Two nominally similar C1 runs differ by 11 style points.** `C1_r1` here (the replay-2×2 run)
scores `style` 51.1; the `C1` in the 0814 famgen set scored 62.3–63.3 in two independent passes.
Same recipe, different run. That is a **run-to-run spread larger than several contrasts this project
has reported**, and it is the strongest argument yet for the standing seeds recommendation. It also
means `P1 − C1` reads +21.0 (7.6 SE) here against +2.1 (0.7 SE) with the other C1 — the disagreement
is between *runs*, not judges.

**Per-agent drift is lower in this pass**: british sd 3.3 (range −6.8..+6.4), coherence sd 3.3,
against 5.9/4.3 in the first pass. 18 agents over 9 arms averages better than 14 over 7.

## 5. Scope limits

1. **Re-judge only.** No new training. The MD → DPOP-upper two-stage cell remains untried.
2. **Coherence is not comparable to the P3 pass** (`RESULTS_0817_P3.md`), which told judges to
   ignore clean truncation; this pass used the first pass's wording, which is silent on truncation,
   and several judges flagged their own coherence as possibly ~15 points low for that reason. All
   coherence claims above are **within-pass** against base and P1, which is the comparison that
   matters for "did it degenerate".
3. **Single seed per cell**, and §4 shows run variance is not negligible.
4. **No human labels.** `results/probefix_cjudge/blind_for_human.json` is still unlabelled.
5. `false_friend` non-engagement is 25–35% for every arm and cannot be prompted away
   (`RESULTS_0817_JUDGE_VALIDATION.md` §5a); the unconditional column is the primary one.
