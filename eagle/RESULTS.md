# EAGLE two-stage preference propagation — first pass results

*2026-08-03. Single seed throughout, per spec ("state plainly if a result is null or
inconclusive at single-seed budget"). Testbeds: styc factorial (Qwen-3B, flipped preferences:
style -> prefer TERSE, correct -> prefer WRONG — flip gives the free-sampling metric headroom;
base generates correct/explained at ceiling) and the joint-preference-sets British axis
(lang/culture components, no flip needed — base is American-default). All histories in
`results/runs/eagle/`; tables/plots `eagle_results.{md,json}`, `eagle_plot_*.png`.*

## Design deltas from the spec (each flagged when made)

1. FLIP=1 on styc (free-sampling headroom, above).
2. Stage-1 plateau meter = through-head implicit accuracy, not layer-probe accuracy (style
   probes read 1.0 at every layer from step 0 — template-separable — so they cannot define a
   plateau; head_acc starts ~.5 and tracks the install).
3. Stage-2 teacher = FIXED init logits + a*(EAGLE_after - EAGLE_before), not live student
   logits (live-detached is an unbounded per-step push; fixed form converges to init + aD).
4. Win-rate-vs-base is oracle-based on styc (exact correctness/style detectors beat a judge);
   the brit axis uses a marker lexicon extracted from the dataset's own pair_ids.

## 1. Encoding depth (stage 1): the factors separate exactly as the depth hypothesis predicts

Through-head DPO install (head_acc, the "can layers 0..L even represent this preference"
meter), 300 steps, GT labels:

| factor \ L | 4 | 12 | 24 | 32 |
|---|---|---|---|---|
| style (flip) | 1.00 @step 25 | 1.00 @25 | 1.00 @25 | 1.00 @25 |
| correct (flip) | **.49 (chance)** | .54 | .64 | .67 |
| brit lang | .97 | .99 | .99 | — |
| brit culture | .96 | .99 | .99 | — |

- Style/dialect: encodable instantly at EVERY depth including L4.
- Flipped correctness: NOT encodable low — monotone in depth, still only .67 at L32. The
  concept the lower stack cannot represent is exactly the one phase 7/8 found late-decodable.
- Brit culture installs as shallow as spelling — the minimal-pair construction (cricket|baseball
  single-word swaps) makes "culture" effectively lexical. Pair construction sets effective
  depth, not the concept's nominal semantics.

## 2. Behavioural install (free-sampling, the decisive metric): two-stage BEATS full DPO on
the style flip

| condition (styc style flip) | terse rate | gen_correct | KL from base |
|---|---|---|---|
| base | .09 | 1.00 | 0 |
| full DPO (300 steps, same data) | **.00** | 1.00 | **6.72** |
| upper-only DPO (>L12) | .11 | — | 5.86 |
| **stage-1 alone @L4** | **.64** | **.98** | **.13** |
| stage-1 alone @L24 | 1.00 | .98 | 2.25 |
| stage-1+2 @L12 (alpha=4) | **.86** | .05 | .86 |

- **Full DPO fails the behavioural flip entirely** while drifting 6.7 nats: it moves pairwise
  likelihoods (its loss saturates) without moving generation. The restricted lower-stack edit
  through the EAGLE readout is the only method here that converts the same data into behaviour
  — s1@L4 gets .64 with correctness intact at 1/50th the KL.
- Depth structure within stage 1 (style): L4 .64 cheap / L12 .16 (the surrogate-gap cell —
  head satisfied, top not listening) / L24 1.0 at 17x the KL / L32 degenerate.
- **Stage 2 closes the L12 gap** (.17 -> .86 — the pre-registered "should shine here" cell),
  confirming the propagation channel transmits. But see §3.

## 3. alpha=4 is over-driven: the delta wrecks capability everywhere it is applied

Every stage-2 cell at alpha=4 collapses gen_correct (1.0 -> .0-.25) and two cells degenerate
into repetitive junk; brit stage-2 gens Goodhart the marker oracle with broken text (brit_rate
"1.0" made of numbered-list gibberish, truth-guard .04). The anchor (KL-to-base, KL_W=1,
grad-ratio ~1 = balanced) cannot save a poisoned target: the student settles between base and
init+4D, and 4D's off-axis early-exit junk already swamps the arithmetic logits. Fix is gain,
not coverage — alpha in {1,2} sweep appended below; one-way floor (relu, DPOP-style) and
token-masked delta are the queued structural variants.

## 4. The correct-factor flip is a universal behavioural NULL — stated plainly

gen_wrong stays .00-.03 in EVERY condition: all stage-1 depths, all stage-2 propagations, AND
full DPO (which burns 15.6 nats of KL trying). At this scale the model will not be made to
prefer wrong answers behaviourally by any tested route; the flip design cannot measure
install-depth for the deep factor here. The depth signature for correctness therefore rests on
the encoding result (§1), which is clean. (Consistent with phase-8: gen_wrong 0.0 throughout
all RL arms too — "assert the wrong thing" appears to be an extremely low-prior behaviour that
pairwise objectives reshape around rather than through.)

## 5. Brit axis (naturalistic minimal pairs)

- Stage-1 installs the implicit preference from the lowest layers (lang: L4 .83, L12 .93;
  culture: L4 .80) at small KL — sweet spot L12, deep edits misfire (L24: .57 at KL 2.5).
- **Truth-guard cost measured**: lang-only training drops truth_over_british acc to ~.10-.42
  (policy partially prefers false-British over true-American) — the exact Goodhart the
  dataset's guard split exists to catch. A guard-included training arm is the natural next run.
- Free-sampling marker oracle is sparse in 40-token continuations (~2-5 hits/48 gens) — rates
  are noisy; lengthen gens or count per-token rates next pass.

## 6. What the first pass establishes, in order of confidence

1. Encoding depth separates factor families exactly as predicted (§1) — the experiment's
   core question, answered affirmatively at the channel level.
2. A lower-stack-only edit through an early-exit readout can install behaviour that full DPO
   does not, at a fraction of the KL (§2) — the surprise of the day.
3. The stage-2 propagation channel transmits (L12 .17->.86) but the delta needs gain control
   (§3) — mechanism confirmed, calibration open.
4. Flipped-correctness is behaviourally uninstallable here by any route (§4) — a real null.

## 7. Alpha sweep + head1 control: the delta carries the HEAD'S COMPETENCE CEILING

Alpha in {1,2} on style-L12 and brit lang-L4: the install is alpha-insensitive (terse .91-.92,
as good as alpha=4) — and so is the damage (gen_correct .22 at every alpha; brit gens still
degenerate). Not a gain problem. DELTA_BEFORE=head1 (same head both sides, isolating the pure
lower-stack-edit effect): install still full (terse 1.0 @25) — and gen_correct still .20, with
the smoking-gun sample "9999999999...".

Diagnosis: **the delta can only transmit what the head can compute.** The L12 readout carries
"answer tersely = emit a digit now" (style IS decodable at L12) but cannot carry WHICH digit —
computation-correctness is not decodable at L12 (phase-7 §9, this repo's own result). Full-
distribution distillation from that readout therefore installs the style while scrambling
answer identity. The stage-2 teacher inherits the head's competence ceiling — the same law
that bound every labeller in phases 7-9 (corr_e, the UF headroom), reappearing in the
propagation channel.

Implied fix (next pass, not run): propagate the PREFERENCE, not the distribution — a pairwise
DPO-shaped stage-2 loss on (chosen, rejected) using the delta only as a margin signal, or a
token-masked delta restricted to positions where |Delta| is large (the style-bearing tokens).
Full-distribution KL to an early-exit teacher is the wrong loss whenever the head is less
competent than the model — which is the entire regime this experiment cares about.
