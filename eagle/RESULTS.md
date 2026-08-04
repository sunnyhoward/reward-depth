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

---

# 2026-08-04 — the stage-1 install was HOLLOW; frozen-head stage-1 is the method

*Single seed unless stated. Rebuilt environment (torch 2.12+cu130, transformers 5.14.1) on a
fresh box; see §15 on what reproduced and what did not. All histories in `results/runs/eagle/`.*

## 8. Root finding: a trainable head absorbs the install

Stage 1 trains the lower LoRA **and the head** (`eagle_dpo.py`: `params = params + list(head.
parameters())`). The DPO margin `(la-ra)-(lb-rb)` can be reduced by moving *either* the lower
stack *or* the head, and descent takes the cheap path:

| testbed | head_acc | reached at | KL from base | behaviour |
|---|---|---|---|---|
| styc style-L12 | 1.00 | step 5 | **0.106** | terse .094 -> .156 (never installs) |
| brit lang-L4 | .90 | step 5 | **0.002** | acc_factor .68, brit_rate ~0 |

`head_acc` 1.00 with the model moved by 0.002 nats means the "install" lived entirely in the
readout. **`head_acc` was never a measure of what layers 0..L represent**, and stage 2 therefore
had essentially nothing encoded low to propagate. `FREEZE_HEAD=1` is now the default.

The reference branch is built correctly (`policy.disable_adapter()`), so freezing the head is
sufficient — no reference change is needed.

## 9. What §8 invalidates

- **The surrogate gap that motivated stage 2.** §2 recorded stage-1@L12 at terse .16 ("head
  satisfied, top not listening"). The head was satisfied *by itself*. With it frozen, stage-1@L12
  gives terse **.95** directly (§10). Stage 2 was scaffolding around a bug.
- **Checkpoint granularity.** `head_acc` saturates by step 5; `CKPT_EVERY=25` gave the plateau
  rule no resolution below 25. Every stage-2 cell in §2/§3 — and every one run 08-04 before this
  was found — launched from a model that had already lost half its correctness
  (styc gen_correct 1.000 @5 -> .500 @25 -> .359 @50).
- **§1's encoding-depth table** was read through an attention-free MLP head that understates
  competence everywhere (§13). Direction likely survives; magnitudes need remeasuring.

## 10. Frozen-head stage-1 is the working method (the day's positive result)

**styc style flip, L12, head frozen, 3 seeds** — free-sampling, full network, greedy:

| seed | terse (base .09) | gen_correct | KL |
|---|---|---|---|
| 0 | .844 @25 | .984 | .70 |
| 1 | **.953** @15 | **1.000** | .63 |
| 2 | **.969** @15 | **1.000** | .32 |

Raw text at seed 1 step 15: `'136'  '116'  'Albert Einstein'  '50'` — genuinely terse and
correct, not degenerate (`TERSE_T` is literally `"{ans}"`). Against the 08-03 matrix:

| method | terse | gen_correct | KL |
|---|---|---|---|
| base | .09 | 1.00 | 0 |
| full DPO 300 steps | .00 | 1.00 | 6.72 |
| stage-1 @L4 (old best) | .64 | .98 | .13 |
| stage-1+2 @L12 (old) | .86 | **.05** | .86 |
| **stage-1 @L12, head frozen** | **.95** | **1.00** | .63 |

Best install, correctness intact, no stage 2. Decay timing is seed-dependent (seed 1 stable at
50, seed 2 gone by 40) — "train to the plateau" is the wrong stopping rule; select early.

**brit lang-L12, head frozen, 3 seeds** (marker oracle widened to 128 prompts x 128 tokens —
the 48x40 default gave 0-7 hits total and could not resolve an install at all):

| seed | acc_factor | brit : am | truthguard | len |
|---|---|---|---|---|
| base | .000 | 2 : 29 | — | 86 |
| 0 | .989 | 17 : 0 | .000 | 105 |
| 1 | .949 | 44 : 18 | .042 | 103 |
| 2 | .994 | 37 : 6 | .021 | 98 |

Coherent fluent text with naturally-placed dialect ("i was a little late in the **car park**").
So the finding carries to the naturalistic testbed. **Cost: truthguard .00-.04 in every seed.**

## 11. Stage 2 fails even from a good foundation

Run from the *working* frozen-head stage-1, not the hollow one:

- **styc** (start: terse .484, correct .984): delta-teacher -> correct **.297**; head-teacher ->
  correct **.234**, len 2.3.
- **brit** (start: acc_factor .989, brit 17:am 2, coherent): stage 2 drives brit_rate to 1.00 by
  emitting `colour` **2283 times**; len collapses 108 -> 9; final samples are
  `'colourcolourcolourcolour...'`. Every metric perfect, model destroyed.

## 12. Six closed doors (all measured, not argued)

| axis | swept | outcome |
|---|---|---|
| anchor weight | KL_W 1,4,16,64 | monotone trade, NO window; by 16 both terse (.094) and correct (.969) are back at base |
| teacher form | delta vs head | delta poisons; head caps the model at head competence and dies without an anchor (KL 5->8.4) |
| learning rate | 3e-6 .. 1e-4 | changes speed toward a degenerate attractor, not the destination |
| stage-1 duration | ckpt5 / 25 / 300 | early = nothing to propagate, late = broken model |
| head architecture | mlp / mlpbig / tf | attention helps everywhere except the computation (§13) |
| capacity | 8.4M vs 25.2M | useless; mlpbig is *worse* than mlp on the training objective |

**Delta diagnostic** (`eagle_delta_diag.py`, no training): the stage-2 teacher overwrites the top
token at **86.5%** of answer positions and leaves the correct token **0.7%** mass where it flips
(vs 18.8% flip at style-branch positions). Base entropy separates 4.0x (0.43 answer vs 1.74
branch), head competence 3.0x. This kills reverse-KL as a fix, and shows top-|Delta| masking is
the weakest discriminator *and* sign-ambiguous (ranking positions by L2 vs by max-element selects
opposite sets).

**Pairwise stage-2** (`S2_LOSS=pairwise`: the head supplies only `sign(m)` from its implicit DPO
reward; the student's own distribution supplies every token). Mechanism prediction CONFIRMED — it
never damages, and twice *repaired* stage-1 damage (correct .531 -> .984, KL .50 -> .15). But it
never installs at any beta or anchor: terse stays at base .09 in all four arms. Surrogate gap.

## 13. Head architecture: attention helps, but the computation ceiling is real

L12, 2000-step distillation, held-out completion positions:

| arch | params | answer kl/top1 | other kl/top1 | train KL |
|---|---|---|---|---|
| mlp | 8.4M | 1.606 / .437 | .381 / .801 | .201 |
| **tf** | 25.2M | 1.400 / **.595** | **.144** / **.904** | **.114** |
| mlpbig | 25.2M | 1.509 / .440 | .387 / .816 | .227 |

`mlpbig` is the param-matched control: identical size to `tf`, performs like the small MLP. So
the gain is **attention**, not capacity — a position-wise readout cannot look back at "17" and
"25" to emit "42". But 5x more training made `tf` better *everywhere except* answer positions
(filler KL .384 -> .144; answer KL 1.262 -> **1.400**, worse). The ceiling survives attention,
capacity and training. §7's law stands; it was previously measured with a blunt instrument.

**Surprise:** a *better* head makes stage 1 *worse*. Frozen stage-1 with the replay head
(agreement .283) gave terse .844 @25 with correct .984; with the styc head (agreement .836) it
collapsed at step 15 (correct .188, peak terse only .344). A sharp readout drives the lower stack
hard enough to break the computation.

**Replay corpus:** `eagle_replay.py` samples the frozen model from random 1-8 token prefixes
(replay-kfac-ewc's default). The result is multilingual noise, and the head trained on it is much
worse on task text (agreement .283 vs .836; answer top1 .337 vs .595). Right distribution for
Fisher estimation, wrong one for head distillation. A mix with natural prompts is untested.

## 14. Metric traps found (add to the standing list)

- **Read the raw generations. Always.** Two different collapses today were invisible to every
  aggregate: brit stage-2 at `gen_len 29` was `'.\n.\n.\n.'`, and at `gen_len 100` was
  `'colourcolour...'`. `brit_rate 1.00` has been produced by numbered-list gibberish, by
  `recognise` spam, and by `colour` spam.
- **brit step-0 `acc_factor`/`acc_truthguard` are 0.000 by construction** — they are implicit-DPO
  accuracies and at step 0 policy == ref, so `0 > 0` is False. There is NO base reference in those
  runs; compare against chance .50, not against 0. (§5's "drops to .10-.42" was a drop from an
  undefined baseline; the correct reading is "systematically below chance", which is worse.)
- **The brit marker oracle at 48 prompts x 40 tokens is unusable** (~1 hit per 300 tokens). It
  read `brit_rate ~0` on a genuine install. Defaults are now `GEN_N=128 GEN_TOKENS=128`.
- Implicit accuracy and behaviour dissociated **four separate times** today (brit s2 acc .98 with
  broken text; full DPO loss saturating at terse .00; pairwise loss .05 with terse flat;
  head_acc 1.00 with terse .156). Never report an install from a teacher-forced metric alone.

## 15. Reproducibility

- **styc `s2_style_L12` did NOT reproduce** across the rebuild: terse .859 -> .266, gen_correct
  .047 -> .203, KL .855 -> .455. Same config, same plateau ckpt, greedy both times.
- **brit stage-2 reproduced the conclusion but not the signature**: generation destroyed both
  times, but no length collapse and no marker Goodhart in the rerun — instead `'.\n.\n.'` and
  off-language text.
- **brit stage-1 reproduced cleanly** (acc_factor .829 -> .846, truthguard .292 -> .312).

Trust directions and within-sweep comparisons (arms sharing one environment). Do not quote
single-cell magnitudes across sessions. 08-03 histories are preserved; 08-04 reruns that reused a
tag are saved as `*_repro0804_history.json`.

## 16. Open, with evidence

- **The brit residual is real headroom neither route reaches.** Stage 1 alone plateaus at
  brit_rate .70-.85 with KL stuck at ~1.18 for 350 steps (400-step run, seed 1); stage 2 only
  "closes" it by destroying the text (§11). Something is missing from both.
- **Guard rows help but are not enough.** Training on all 919 campaign rows instead of 484
  (`INCLUDE_GUARD=1`) roughly doubles truthguard (.042 -> .104) while the dialect install
  survives (acc_factor .949, brit_rate .89, len 100). Still far below chance .50. 145 guard rows
  against 484 dialect rows is too little counterweight; upweighting is untested.
