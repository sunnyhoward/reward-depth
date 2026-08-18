# Stage 1 does not win by dropping the rejected side — it builds a new direction

*2026-08-18. `NEXT_0817.md` §6.0a, run as specified: `MODE=probe` retrained 300 steps
(`probefix/run_pf4b_stage1_sides.sh`), then `probefix/pf_mdsides.py` on the base frame, 128 pairs,
`LAYERS=6,12,18,20,24,30` — with L20, the attach point, added to the script's default set. Both
pooling modes: `READ=mean` for comparability with every banked `mdsides` row, `READ=last` because
that is what `MODE=probe` actually optimised. Then one new instrument, `probefix/pf_stage1_rot.py`,
because the diagnostic refuted the hypothesis and pointed at a different one.
Qwen3.5-4B, LoRA 0–31, seed 0, replay on. Single seed.*

**The pre-registered reading was `Δchosen ≈ 0` with `Δrejected` strongly negative. That is not what
happened.** `Δchosen` is **positive at five of six read points in both pooling modes**, and it is
the *larger* of the two moves. The difference-only defect is open in `MODE=probe` — nothing in the
objective pins an absolute — but stage 1 does not exploit it. §6.0a is answered: **no.**

**What the same numbers show instead.** At the attach point, in the base-fitted frame, the
chosen-minus-rejected projection **does not grow at all** (L20, `READ=last`: 21.09 base → 18.69 at
s100 → 20.87 at s300) while the run's own meter has the probe saturated (`sat 1.00`, z +5…+9σ). The
objective is being satisfied somewhere the base frame does not see. §4 measures where: separation
along the **optimised** direction grows **4×** while separation along the base preference axis does
not move.

**§6 then tests the obvious next hypothesis — that the CONTINUOUS REFIT is what lets it happen — and
that is false too.** A frozen-direction arm (`PROBE_STEPS=0`, direction fixed at warm start)
saturates its objective just as readily, grows the same 4×, and installs **exactly nothing**, the
same as the refitted arm on every meter. **The refit is not the operative variable, and §4's first
reading of its own table was wrong** — see the correction there. What is left is the simplest and
most negative account: separation along a linear direction at this read is not causally connected to
generation at all, which is what phase 1's `cos(μ, W_A−W_B) = −0.003` said from the other side.

---

## 1. The retrain reproduces the arm every stage-1 claim rests on

Byte-identical config to `run_pf4b.sh:36` = `run_pf4b_dpop_c.sh:37`; the 0817 adapter died with the
box. Against the banked `results/probefix4b/runs/history_B4_probe_L20.json`:

| step | ff raw (old → new) | style raw | ff probe | style probe | guard implicit | replay KL |
|---|---|---|---|---|---|---|
| 0 | .36 → .36 | .12 → .12 | .75 → .74 | .98 → .98 | .00 → .00 | .000 → .000 |
| 100 | .39 → .39 | .15 → .13 | .83 → .83 | .93 → 1.00 | .56 → .60 | .100 → .070 |
| 300 | .39 → .36 | .13 → .16 | .88 → .90 | .99 → 1.00 | .44 → .54 | .171 → .162 |

`sigma0` 0.366 → 0.355. Run-to-run spread only; the diagnostic below is about the real arm.

**And the signature failure reproduces exactly.** Output ranking `raw` is flat over 300 steps
(`false_friend` .36 → .36, `style` .12 → .16) while probe accuracy goes to .90/1.00 and the DPO
*implicit* ranking — policy against reference — rises from .00 to .59/.80. The model has moved the
pair apart relative to base without moving which side is preferred in absolute terms.

## 2. The answer to §6.0a: both sides rise, and the chosen side rises more

`results/probefix/mdsides_stage1_mean.json`. Base-fitted frame, SD-normalised, 128 pairs.
`READ=mean` — the column the banked MD/MDSTACK/P3 rows use.

| read | base proj | Δchosen s100 | Δrejected s100 | Δchosen s300 | Δrejected s300 | ‖read‖ s300 (base) |
|---|---|---|---|---|---|---|
| L6 | +21.18 | +0.81 | −0.48 | **+4.90** | +0.20 | 34.9 (33.6) |
| L12 | +17.02 | +1.71 | +0.78 | −0.43 | −2.69 | 34.8 (32.8) |
| L18 | +18.19 | +3.80 | +2.21 | **+2.08** | −0.56 | 32.7 (30.6) |
| **L20 (attach)** | +19.21 | +3.00 | +0.95 | **+2.24** | −0.92 | 31.5 (29.6) |
| L24 | +21.59 | +4.23 | +2.33 | **+3.90** | +1.17 | 30.3 (28.6) |
| L30 | +25.05 | +5.42 | +4.28 | **+5.06** | +3.46 | 32.0 (30.2) |

Nothing here resembles the confirmation. The one negative `Δchosen` (L12, −0.43) comes with a
*more* negative `Δrejected`, so even there the margin is not won by the rejected side alone.

**Against the banked contrasts** (`RESULTS_0817_MDFLOOR.md` §1, same instrument, same frame): the
gamed profile is `MDSTACK` at L30, `Δchosen −51.7`; the healthy output-attached profile is `P3`,
`Δchosen +138.4`. Stage 1 is neither — **+5.06**, the same sign as P3 and two orders of magnitude
smaller. Whatever it does at the output, it barely moves the pooled residual at all.

## 3. At the completion-end read, the base-frame margin does not grow — and collapses up top

`results/probefix/mdsides_stage1_last.json`. `READ=last` is what the objective optimised
(`PROBE_READ` defaults to `last` outside meandiff/mdstack; `pf_train.py:252-261`).

| read | base proj | proj s100 | proj s300 | Δchosen s300 | Δrejected s300 |
|---|---|---|---|---|---|
| L6 | +28.62 | +35.60 | **+51.70** | +49.53 | +26.45 |
| L12 | +23.86 | +21.93 | +31.02 | +71.05 | +63.90 |
| L18 | +20.42 | +22.98 | +31.14 | +84.30 | +73.58 |
| **L20 (attach)** | +21.09 | **+18.69** | **+20.87** | +23.69 | +23.91 |
| L24 | +19.38 | +9.01 | **+4.65** | −16.82 | −2.08 |
| L30 | +17.52 | +5.40 | **+4.94** | −19.94 | −7.36 |

Two things, and both are new:

- **At the attach point the base-frame margin is unchanged after 300 steps** (+21.09 → +20.87),
  and *lower* at s100. The objective is saturated (`sat 1.00`) over the same interval. The
  separation the probe is scoring is not separation along this axis.
- **In the upper stack it collapses**: L24 +19.38 → +4.65, L30 +17.52 → +4.94, with both sides
  moving down. This is the completion-end read — the position generation depends on. The most-quoted
  stage-1 result is "decodability rises everywhere **including the final block**"; that is measured
  with a probe **refitted on the trained model**. In a fixed base frame the final-block margin at
  the generating position **falls by a factor of 3.5**. The two are not in conflict, and §4 is why.
- The common-mode movement is enormous — both sides +71/+64 at L12, +84/+74 at L18 — with the
  margin nearly unchanged. Stage 1 moves the representation a very long way in a direction that
  costs it nothing on the objective.

**Instrument caveat, and it is not in `pf_mdsides.py`'s docstring.** That docstring says the ‖read‖
column is what separates a genuine fall from a frame rotation. **Under `READ=last` the column is
vacuous**: `span_read` RMS-normalises (`pf_common.py:63`) and `last` takes one position, so every
read has norm exactly √2560 = 50.6 — as the table shows, identical for base and every arm. Only the
`mean` column, where pooling makes the norm informative, carries that check. Every previous
`mdsides` run used `mean`, so this bites here first. It is also why §4 exists: the rotation reading
had to be tested directly rather than excluded by the norm.

## 4. Where the separation actually goes (`probefix/pf_stage1_rot.py`, new)

Same 128 pairs, L20, `READ=last`, base-fitted SD throughout. `sep(w)` is the mean SD-normalised
margin along direction `w`, evaluated on **both** models, so a direction that separates the trained
model but not the base one is a direction stage 1 built. The `base logistic probe` row is the same
recipe (`Probe.refit`, Adam on `-logsigmoid(d·w)` + L2, 600 steps) fitted on the **base** model's
reads over 1024 pairs. The last row is the **frozen** arm of §6 — the run's own warm-start probe,
fitted on the base model and never updated.

| direction | cos(u, w) | sep on base | sep on trained |
|---|---|---|---|
| base mean-difference `u` | 1.000 | 21.09 | 20.87 (s300) |
| base logistic probe, 1024 pairs | 0.716 | 15.33 | — |
| stage-1 refitted probe, s100 | 0.441 | 9.51 | 17.07 |
| stage-1 refitted probe, s300 | 0.429 | 9.17 | **36.66** |
| **frozen warm-start probe (§6), s100/s300** | **0.448** | **9.63** | 23.67 / **40.64** |

**CORRECTION — the first reading of this table, in the commit that introduced it, was wrong.** It
said the refitted probe's cos 0.43 against the control's 0.72 showed that *training moved the
direction*. The frozen arm refutes that: its probe is fitted on the base model and never updated,
and it sits at **cos 0.448 with sep-on-base 9.63** — indistinguishable from the "trained" rows. The
0.72/15.33 control row is the outlier, not the trained probes. The cause is that a logistic
direction in 2560 dimensions fitted on 1–2k pairs is **underdetermined**: two fits on base data,
differing only in sample, land 0.72 and 0.45 from `u`. **cos(u, w) is therefore not a measure of
what training did**, and no claim should rest on it. `sep on base` vs `sep on trained` for the SAME
direction survives, because both are evaluated on the same fitted `w`.

What the table does support, with that row in place:

- **Separation along the optimised direction grows ~4×** (9.2 → 36.7 refitted, 9.6 → 40.6 frozen),
  and **separation along the base preference axis does not follow** — 21.09 → 20.87 refitted,
  21.09 → 23.90 frozen. The margin the objective buys is confined to the direction being optimised.
- The probe family's directions separate the base model *worse* than `u` does (9–15 vs 21.09) in
  SD-normalised margin. That is a property of logistic-vs-mean-difference scaling, not a defect.

## 5. The refit is not the mechanism either (`B4_probefroz_L20`, `PROBE_STEPS=0`)

The reading §4 originally proposed — the probe chases the model into a subspace it just built —
predicts that **pinning the direction** breaks the loop. `PROBE_STEPS=0` makes
`probe.refit(PROBE_STEPS)` (`pf_train.py:576`) a no-op, so the direction is the 600-step warm start
on the base model and never moves. Everything else is identical to `B4_probe_L20`, down to the same
`sigma0` 0.355. It is a clean single-variable contrast.

**It changes nothing.** Every meter, refitted vs frozen, at 300 steps:

| meter | refitted | frozen |
|---|---|---|
| `false_friend` raw ranking (base .36) | .36 | .41 |
| `style` raw ranking (base .12) | .16 | .15 |
| `false_friend` implicit ranking | .59 | .68 |
| `style` implicit ranking | .80 | .74 |
| probe accuracy | .90 | .86 |
| replay KL | .162 | .160 |

Run-to-run spread, no more. The frozen arm **saturates its objective** (`sat` 0.83–1.00, z +7.9 by
step 150), so the refit was not doing work the objective needed. `pf_mdsides.py` on it
(`mdsides_probefroz_{mean,last}.json`) tells the same story as §2 and §3: Δchosen positive at
s100 everywhere, the L20 base-frame margin moving only +2.8 at `READ=last` (21.09 → 23.90, against
the refitted arm's −0.2), and the upper stack collapsing exactly as before (L30 17.52 → 1.94).

**Three candidate causes for the stage-1 null are now eliminated by measurement**: the rejected side
(§2), the frame/direction the probe chooses (§4 corrected), and the continuous refit (here). What
remains is that the *quantity being optimised* — linear separation at a completion-end read at
L20 — is not causally connected to generation, which is what phase 1's `cos(μ, W_A−W_B) = −0.003`
and `RESULTS_0817_MDLATE.md`'s inert top-of-stack say from the causal side.

## 6. What this changes

1. **§6.0a is closed, negative.** No stage-1 result in `probefix/RESULTS.md` needs reinterpreting
   via the rejected side. The `MD_FLOOR` anchor, which exists to fix that channel, has **no reason
   to be ported to `MODE=probe`** — there is nothing there for it to repair.
2. **`NEXT_0817.md` §6.0c stands and gets sharper.** Every before/after decodability comparison on a
   trained model is ambiguous — but the ambiguity that bites is not chosen-vs-rejected, it is
   **refitted-frame vs fixed-frame**. `pf_mdsides.py` answers the first; `pf_stage1_rot.py` answers
   the second, and it is the one that moved here. Both should accompany a decodability claim.
3. **The paper spine gains a measured null, not a mechanism.** §5.1 and the stage-1 null had a
   proposed mechanism (*more linearly separable ⇒ ranking satisfied without moving generation*,
   `probefix/HANDOVER.md`) that was never measured. It is now measured and it is **specific and
   negative**: the separability gain is real and confined to the optimised direction, it does not
   reach the model's own preference axis, and it survives every intervention tried on it — the
   rejected-side anchor is unnecessary (§2), the direction is not the variable (§4), and the refit
   is not the variable (§5).
4. **The one ingredient never isolated is POOLING.** `MODE=meandiff` differs from stage 1 in three
   ways at once — pooled reads, a lag-1 mean-difference direction, a saturating hinge — and it is
   the one activation objective that installs. §5 eliminates the direction/refit leg. Phase 8 §13
   attributes the closure of the forging channel specifically to pooling. `probefix/run_pf4b_pool.sh`
   (commit 571b0ba) scripts exactly this contrast — `PR_last_L20` vs `PR_mean_L20`, objective held
   fixed, read varied — and **no result for it is banked anywhere in `results/`**. It is the cheapest
   remaining experiment in the project and it is the one that would close this line.

**Limits.** Single seed throughout, one attach point (L20), one model. The `sep` growth in §4 is
measured on the direction the probe itself chose, which is favourable by construction; the `sep on
base` column is what makes it readable. And the §5 contrast is two runs, not two seeds of each —
the differences there (raw .36 vs .41) are inside the run spread this project has already measured,
which is the point, but a second seed would state it properly rather than by assertion.
