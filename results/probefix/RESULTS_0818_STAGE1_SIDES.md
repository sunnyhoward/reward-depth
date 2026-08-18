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
s100 → 20.87 at s300) while the run's own meter has the probe saturated (`sat 1.00`, z +5…+9σ).
Both cannot be true of one direction. `pf_stage1_rot.py` measures which: the refitted probe sits at
**cos 0.43** to the base preference axis — against **0.72** for the same logistic recipe fitted with
no training at all — and separation along it grows **4×** while separation along the base axis is
flat. Stage 1 **constructs a direction**, and the probe follows it there.

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
model but not the base one is a direction stage 1 built. The control is the same logistic recipe
(`Probe.refit`, Adam on `-logsigmoid(d·w)` + L2, 600 steps) fitted on the **base** model's reads
over 1024 pairs — without it, a cos below 1 would just be logistic-vs-mean-difference.

| direction | cos(u, w) | sep on base | sep on trained |
|---|---|---|---|
| base mean-difference `u` | 1.000 | 21.09 | 20.87 (s300) |
| **base logistic probe (control)** | **0.716** | **15.33** | — |
| stage-1 refitted probe, s100 | **0.441** | 9.51 | **17.07** |
| stage-1 refitted probe, s300 | **0.429** | 9.17 | **36.66** |

- **The trained probe is far off the base preference axis** — 0.43, where the untrained control of
  the same family lands at 0.72. Training moved the direction, not just the fit.
- **It is a bad direction for the base model** (9.2 vs 15.3 for the control, 21.1 for `u`) and an
  excellent one for the trained model (36.7). Separation along it grows **4.0×**; separation along
  `u` grows **1.0×**.

**This explains the dissociation without the difference-only defect.** The probe is refitted, so it
finds the new direction and reports a gain. Everything downstream — the readout, generation, the
`raw` ranking — reads the model's own preference axis, which did not move at the attach point and
fell in the upper stack (§3). Stage 1 is not forging the *residual*; it is answering the objective
in a subspace of its own choosing. Phase 1's `cos(μ, W_A−W_B) = −0.003` and
`RESULTS_0817_MDLATE.md`'s inert top-of-stack are the same story from the causal side.

## 5. What this changes

1. **§6.0a is closed, negative.** No stage-1 result in `probefix/RESULTS.md` needs reinterpreting
   via the rejected side. The `MD_FLOOR` anchor, which exists to fix that channel, has **no reason
   to be ported to `MODE=probe`** — there is nothing there for it to repair.
2. **`NEXT_0817.md` §6.0c stands and gets sharper.** Every before/after decodability comparison on a
   trained model is ambiguous — but the ambiguity that bites is not chosen-vs-rejected, it is
   **refitted-frame vs fixed-frame**. `pf_mdsides.py` answers the first; `pf_stage1_rot.py` answers
   the second, and it is the one that moved here. Both should accompany a decodability claim.
3. **The paper spine gains a mechanism.** §5.1 ("decodability depth does not predict training
   depth") and the stage-1 null had a proposed mechanism (*more linearly separable ⇒ ranking
   satisfied without moving generation*, `probefix/HANDOVER.md`) that was never measured. This
   measures it and makes it specific: the added separability lives in a **constructed direction at
   cos 0.43 to the model's own preference axis**, and the axis generation reads is unchanged.
4. **What would falsify §4**: fit the base logistic probe on the *trained* model's read
   distribution, or run `K_PROBES` — if the trained probe's direction is one strand of a thick
   bundle rather than a new one, the 0813 deflation result predicts sep along `u` should rise once
   enough orthogonal strands are consolidated. That is `NEXT_0817.md` §6.4, and it is now a test of
   this section rather than a loose end.

**Limits.** Single seed, one attach point (L20), one model. The `sep` growth in §4 is measured on
the direction the probe itself chose, which is favourable by construction — the control row and the
`sep on base` column are what make it readable, and neither is a substitute for a second seed.
