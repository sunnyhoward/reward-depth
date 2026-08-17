# The missing cell: stage 1 does not add to DPOP, it substitutes for the write range it removes

*2026-08-17. One new run — **`P3_up_dpop`**: DPOP λ=50, LoRA on blocks 21–31, **no stage 1**,
600 steps, `Qwen3.5-4B`, `brit_dose20.jsonl`, seed 0 — identical to `C1` except that
`INIT_MERGE` is absent. Scripted at `run_pf4b_dpop.sh:54` behind `UPPER=1` and never executed
until now. Judged by Claude subagents only (no Qwen), with `C1` and `D2` re-judged in the SAME
288-item shuffled pass so all three share agent calibration. Scripts:
`probefix/pf_cjudge.py batchnew`, `probefix/pf_famgen_arms.py`.*

**Why it was run.** Every previous comparison of the two-stage recipe against one-stage DPO was
confounded. `C1` (stage 1 + DPOP on 21–31) differs from `P1` (DPOP on 0–31) in **two** ways at
once — stage 1, and the write range — so neither "stage 1 adds nothing" nor "stage 1 helps and the
restriction costs the same" could be distinguished. `P3` is `C1` minus stage 1 with everything else
held, which identifies both effects.

---

## 1. The 2×2, unconditional (non-engagement scored 50), paired per prompt

| arm | stage 1 | objective | LoRA | british ff | british style | coherence ff / style |
|---|---|---|---|---|---|---|
| base | — | — | — | 39.6 | 29.0 | 78 / 85 |
| **P1** | ✗ | DPOP | **0–31** | **69.0** | **64.4** | 76 / 86 |
| **C1** | ✓ | DPOP | 21–31 | 60.5 | **63.3** | 82 / 88 |
| **P3** | ✗ | DPOP | 21–31 | 65.6 | **45.4** | 84 / 90 |
| D2 | ✗ | plain DPO | 21–31 | 61.7 | 51.2 | **42 / 13** |
| P0 | ✗ | plain DPO | 0–31 | 58.0 | 55.6 | **33 / 20** |

| contrast | false_friend | style |
|---|---|---|
| C1 − P3 (**stage 1, everything else matched**) | −4.7 ± 3.5 (−1.4 SE) | **+16.9 ± 2.7 (+6.2 SE)** |
| P1 − P3 (**write range, no stage 1 either side**) | +3.3 ± 3.6 (+0.9 SE) | **+18.9 ± 2.6 (+7.3 SE)** |
| P1 − C1 (the old, confounded contrast) | +8.1 ± 4.5 (+1.8 SE) | +2.1 ± 3.0 (+0.7 SE) |
| P3 − D2 (**the objective, matched range**) | +4.0 ± 2.8 | −5.8 ± 2.2 (D2 degenerate, see §3) |

## 2. Three findings

**(a) λ=50 prevents the collapse. Stage 1 is not required for it.** `P3` and `D2` are the same
write range, the same data and the same 600 steps, differing only in λ. P3's coherence is 84 / 90;
D2's is 42 / 13, and D2 engages the `style` axis on 21% of prompts because most of its output is a
loop. **This retracts `RESULTS_0813_MEASUREMENT.md` §1** ("stage 1, not the write range, is what
prevents collapse"): that experiment held the objective at plain DPO, so it could only show that
stage 1 *suffices*. With DPOP the collapse never happens, and stage 1's single firmly established
contribution has a one-line alternative in the loss.

**(b) On `style`, restricting writes to blocks 21–31 costs ~18 points, and stage 1 buys that back
— but nothing more.** C1 − P3 = +16.9 and P1 − P3 = +18.9 are the same effect, while P1 − C1 is
+2.1 (0.7 SE). So register can be installed either by training the whole stack directly or by
prepending an all-layer probe stage and then training only the top — and the two routes land in the
same place. **Stage 1 is a substitute for full-range training, not an addition to it.**

Crucially the step budget does not explain this. C1 spends 300 (stage 1) + 600 = **900** optimiser
steps against P3's 600, so C1 > P3 alone would be confounded. But **P1 also beats P3 by the same
margin at P3's own 600-step budget**, and P1 has no stage 1 at all. The variable that moves `style`
is the write range.

**(c) The two families dissociate on write depth.** `false_friend` is flat across all three DPOP
arms (60.5 / 65.6 / 69.0, every contrast ≤1.8 SE): domain-term substitution installs perfectly well
from blocks 21–31 alone. `style` needs the lower stack. That is a depth effect, it is large
(≈18 points, >6 SE), and it is a property of **what kind of preference** is being installed rather
than of the training signal's attach point — which is the distinction this programme has been
trying to isolate since the L\* work. Note it also runs *against* the original Occam framing: the
lexical family, not the semantic one, is the one indifferent to depth.

## 3. Instrument checks

**Judge test-retest.** C1 and D2 were judged twice, in independent passes by different agents:

| | engaged agree | british (pass2 − pass1) | coherence |
|---|---|---|---|
| false_friend C1 | 0.96 | −0.4 ± 0.9 | +4.5 ± 1.0 |
| false_friend D2 | 1.00 | +0.2 ± 0.6 | +4.6 ± 0.9 |
| style C1 | 1.00 | +1.0 ± 1.1 | +6.1 ± 0.8 |
| style D2 | 0.92 | +0.6 ± 0.5 | +1.1 ± 0.6 |

The `british` axis reproduces to within ±1 point — better than the between-arm effects it is being
used to measure. **`coherence` does not**: it drifts +1 to +6 because this pass instructed judges
not to penalise clean truncation at the token limit. Coherence is therefore comparable *within* a
pass and not *across* passes; the degenerate/healthy split is unaffected (D2 is degenerate under
both).

**D2's `style` contrasts are not interpretable.** Only 10 of 48 items are engaged by both arms, and
its unconditional 51.2 is essentially the neutral anchor rather than a measurement.

## 4. Scope limits

1. **Single seed**, as everywhere in this study. The `style` write-range effect is >6 SE on item
   noise, but item noise is not seed noise.
2. **P3 was judged by Claude only.** By design (no Qwen in this pass), so the arm levels here are on
   the Claude scale — which sits ~9 points below Qwen's overall
   (`RESULTS_0817_JUDGE_VALIDATION.md` §4). Do not mix these numbers with pre-0817 tables.
3. **No human labels yet.** `results/probefix_cjudge/blind_for_human.json` remains unlabelled, so
   the judge is validated only against my own labels and itself.
4. **The adapters are on `/workspace` and do not survive.** Banked here:
   `results/probefix4b_p3/famgen_P3_up_dpop.json`, `history_P3_up_dpop.json`, the training log, and
   `results/probefix_cjudge_p3/` (batches, verdicts, key, cjudged, report). Regenerating P3 is one
   ~40-minute run plus `pf_make_bank.py`.
5. **`false_friend` non-engagement is ~25–30% for every arm** and cannot be prompted away
   (`RESULTS_0817_JUDGE_VALIDATION.md` §5a). All its numbers here are unconditional.

## 5. What this leaves

The two-stage recipe now has **no demonstrated advantage over one-stage DPOP on all layers** — not
on collapse (λ handles it), not on `style` (P1 ≈ C1), not on `false_friend` (all flat). Its
remaining interest is mechanistic: C1 reaches P1's register performance while touching only 7.4M
parameters above block 20 at stage 2, with the encoder edit merged in beforehand. If that is real it
is a statement about *where* the two halves of a register preference live, which is worth a
by-layer audit — and it is now separable from the recipe's marketing, because P3 is the arm that
shows what the restriction costs on its own.

**Next, in order:** (1) seeds on P1 / C1 / P3 `style` — the effect is large enough that 2 seeds
settle it; (2) `pf_audit.py` on P3 to see whether the upper-only arm's ΔW still peaks at L\* the way
every other arm's does; (3) the human label slice.
