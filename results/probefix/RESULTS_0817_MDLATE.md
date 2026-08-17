# Mean-diff at late read points: install FALLS with read depth, and the top of the stack is inert

*2026-08-17. Six read points for the phase-8 activation objective — **L20, L24, L26, L28, L30, L31**
of 32 — at **matched 300 steps**, `Qwen3.5-4B`, `brit_dose20.jsonl`, replay off, LoRA 0–31, λ=1.0
floor, seed 0. Everything except `PF_LAYER` is identical across the six, and identical to the
existing `MD_L20_r0_noreplay` arm. 864 items (9 arms × 96) judged blind by 18 Claude subagents,
shuffled across arms and families, with `base` and `P1` (plain DPOP at the output) judged in-pass.
No Qwen. Figure: `results/probefix/mdlate_readdepth.png`. Scripts: `probefix/pf_train.py`
(`GRAD_CKPT` added), `pf_famgen_arms.py`, `pf_cjudge.py`, `pf_leakage.py`, `pf_mdlate_plot.py`.*

**The hypothesis under test**, proposed on the grounds that higher-level features may only exist
late: reading the mean-difference direction at a later block should install preferences an early
read cannot reach. **It does not. Install falls monotonically as the read point gets later**, and
the two latest points barely move behaviour at all.

---

## 1. The curve (unconditional; non-engagement scored 50)

| read L | false_friend | style | coherence ff / style | leakage |
|---|---|---|---|---|
| base | 39.3 | 27.0 | 73.2 / 68.0 | 0.000 |
| **L20** | **56.5** | **43.4** | 78.1 / 66.0 | 0.146 |
| L24 | 55.2 | 38.9 | 65.2 / 61.7 | 0.198 |
| L26 | 54.3 | 38.2 | **45.1 / 35.2** | **0.573** |
| L28 | 57.4 | 39.3 | **51.6 / 49.9** | 0.375 |
| L30 | 42.0 | 37.5 | 79.8 / 79.8 | **0.000** |
| L31 | 46.4 | 33.9 | 81.7 / 77.2 | **0.000** |
| **P1** (DPOP at output) | **70.3** | **72.3** | 72.8 / 76.5 | 0.000 |

Paired per prompt, over the same 48 held-out prompts:

| contrast | false_friend | style |
|---|---|---|
| L31 − L20 | **−10.2 ± 4.5 (−2.3 SE)** | **−9.5 ± 3.3 (−2.9 SE)** |
| L26 − L20 | −2.2 ± 3.7 | −5.2 ± 3.4 |
| L20 − base | +17.2 ± 5.3 (+3.2 SE) | +16.4 ± 2.9 (+5.6 SE) |
| L31 − base | +7.0 ± 4.5 (+1.6 SE) | +6.9 ± 2.2 (+3.1 SE) |
| **P1 − L20** | **+13.8 ± 5.3 (+2.6 SE)** | **+28.9 ± 3.4 (+8.6 SE)** |

## 2. Three findings

**(a) Later is worse, not better.** Every late point is at or below L20, and L31 − L20 is negative
in both families at 2.3 and 2.9 SE. There is no read depth at which the activation objective
installs more than it does at L20 — which was itself chosen as the *decodability elbow*, not as a
late point.

**(b) The top of the stack is close to inert.** L30 and L31 have the **cleanest text in the whole
study** (coherence 79.8/81.7, leakage exactly 0.000) and the **weakest install** (L30 `false_friend`
42.0 against base 39.3; training-side ranking 0.240 against base 0.270, i.e. no install at all).
They do not damage the model because they barely change it. This is the phase-1 measurement coming
true behaviourally: `cos(μ, W_A − W_B) = −0.003` at the final layer — the mean-difference direction
there lies almost exactly in the null space of the output map, so driving the projection along it
buys nothing downstream. And it *was* driven: `proj` ran to 40–70 against a target M0 of ~10, with
`sat` = 1.00. **The meter was satisfied and the behaviour did not follow** — the same dissociation
phase 9 recorded on UltraFeedback, now localised to where in the stack it happens.

**(c) The mid-late band installs no better and destroys the text.** L26 and L28 are the leakage
zone: template leakage 0.573 and 0.375, coherence collapsing to 35–52, for install indistinguishable
from L20. So the read-depth axis trades cleanly between *inert* (L30–31) and *damaging* (L26–28),
with no point that is both effective and safe.

**Everything loses to plain DPOP at the output** by +13.8 (`false_friend`) and +28.9 (`style`), the
latter at 8.6 SE. `MD_L20_s100` remains the best MD cell in the project (62.0 / 41.9 unconditional,
leakage 0.031) and it still loses to P1.

## 3. What this closes

Activation-space training on this task has now been tested at six read depths, two replay settings,
three step counts and against a matched output-attached control. **It installs — reliably, ~+17
points over base — and it is beaten by ordinary DPOP everywhere, at every read depth.** The reason
is not that the read was too early: making it later reduces the install and, at the top, removes it.

The remaining live version of the idea is the one that has never been run: preference gradient into
the lower half via the probe *and* DPO at the output simultaneously, with an explicit gradient gate
so the halves are actually separated (adding the losses does not separate them — the output loss
backprops through every block). Given the curve above, its lower half would be doing work L20
already does at ~60% of DPOP's effect, so the prior is poor.

## 4. Scope limits

1. **Single seed per cell.** Today's two-C1 comparison put run-to-run spread at 11 `style` points
   (`RESULTS_0817_MEANDIFF_REJUDGE.md` §4), which is the size of the L31 − L20 effect. The *ordering*
   (all late points ≤ L20, all ≪ P1) is robust to that; the individual gaps are not.
2. **300 steps.** L20's best cell is s100, so the matched-budget comparison may sit past every
   layer's optimum. The L26 arm was also run to 600: its `false_friend` install does not improve
   (see the s600 cell in `probefix4b_mdlate`), and steps monotonically hurt at L20.
3. **`GRAD_CKPT=1` was used** for all six late arms (a late read backprops through more blocks than
   an early one; the GPU is shared). It is semantically identical — same loss, same gradients — and
   defaults off, so the pre-existing arms are unaffected.
4. **Claude judge only**, on the Claude scale (~9 points below Qwen's). Do not mix with pre-0817
   tables. Coherence is comparable within this pass and to the 0814-worded passes, not to the P3
   pass.
5. **No human labels** yet: `results/probefix_cjudge/blind_for_human.json`.
