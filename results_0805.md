# 2026-08-05 — session results and project synthesis

*Qwen2.5-3B (styc/brit line) and Qwen3-4B-Base (new refusal/steering line). Single seed unless
stated. Judged numbers are Qwen3-8B; lexicon numbers are the online meter only and are labelled
as such — measured today, the refusal lexicons agreed with the judge only .62–.98 and
systematically OVER-read.*

**Read §0 first.** Several claims made mid-session were withdrawn on further measurement. They
are recorded as withdrawn rather than silently replaced, because the pattern in which claims
survived is itself informative.

---

## 0. What survived, what didn't

**Survived every challenge today:**

| finding | evidence |
|---|---|
| K-FAC leash is an effective-step-size effect, not curvature | λ=0 @ LR 1e-5 beats λ=1000 @ LR 1e-4 (§1) |
| Head competence tracks the "depth" signal | independently in §2 and §4; blocks every depth claim |
| Base Qwen3-4B already refuses cross-lingually (.20–.53) | §4 base row |
| PKU-SafeRLHF's safe side is only 11.5% refusals | §4 data section |
| Steering efficacy is an L8–L20 band with a bimodal cost structure | §6, judged, n=128, dose-responsive |
| No global KL anchor separates install from damage on styc | §5, three attempts, one mechanism |

**Withdrawn during the session:**

| claim | why it fell |
|---|---|
| "K-FAC leash works, dose-dependently" | LR control matched and beat it |
| "Install at KL .7 / damage at KL 2.4, invariant across 7 arms" | Spearman only −0.48; only the *onset* is invariant (§1) |
| "Full DPO didn't teach refusal" | It did — in the training phrasing. EN_EVAL is not a refusal detector (§4) |
| "Restricted lower writes generalise better" | Contradicted by the brit held-out test (§3) |
| "Upper-only is the most lexical arm" | That was an off-peak step |
| "L16 is anomalous" → "the whole band saturates" | Both wrong: it is a dose ceiling layers hit at different points (§6) |
| "Freeing the unembedding tests the pinning hypothesis" | Mis-designed; neither head ever saw a preference label (§7) |

**The pattern: negatives and confounds held; every positive claim about which method or depth is
better churned.** Those are exactly the claims whose validation had not yet run when they were
first stated. The operational lesson is in §8.

---

## 1. K-FAC leash — CLOSED NEGATIVE (NEXT.md queue item 1)

Factor estimation is 11 minutes, not hours: `--placement auto` sends every factor with dimension
> 4096 to CPU, which is all the 11008-dim MLP factors. `--placement model` keeps all 175 modules
(7 projections × layers 0..24, 37.4 GiB dense fp32) GPU-resident. NEXT.md's "175 modules was TOO
MANY" was a symptom of the old box, not of the job.

λ ∈ {1, 10} were under-dosed (penalty 3% and 25–30% of the DPO term). λ ∈ {100, 1000} **did**
widen the L24 safe window monotonically. Then the control killed it:

| arm | best cell (terse / correct) | KL |
|---|---|---|
| λ=0, LR 1e-4 | 1.00 / .984 @5 | 0.71 |
| λ=1000, LR 1e-4 | 1.00 / .906 @30 | 1.73 |
| **λ=0, LR 1e-5** | **1.00 / 1.000 @35–50 (4 consecutive ckpts)** | 0.71–1.39 |

A plain learning-rate cut strictly dominates. The leash bought slower effective steps.

**Damage onset is arm-invariant; depth of collapse is not.** Correctness is intact (≥.91) to
KL ≈ 2.0 in all 7 arms and degrades past ≈2.0–2.5 in all 7 — but Spearman(KL, correct) is only
−0.48, and past onset the LR controls hold .45–.69 where the λ arms sit at .06–.15. An earlier,
stronger version of this claim ("install .7 / damage 2.4") was withdrawn after plotting all 110
eval points (fig1).

**Caveat not to lose:** the penalty is a fixed-reference local quadratic and the package's own
README warns against treating a larger coefficient as a cure outside the local regime. At λ=1000
the runs reach KL 2+, well outside it, and `fit_calibration` against the 147 held-out replay
sequences was never run. The honest statement is "K-FAC-as-configured lost to an LR control", not
"curvature penalties don't help". See §5 for the deeper reason it was aimed wrong.

## 2. Encoding depth remeasured — delivered, and it does NOT settle §1

Frozen tf head, peak head_acc (fig2):

| factor \ L | 4 | 12 | 24 | 32 |
|---|---|---|---|---|
| style | 1.00 | 1.00 | 1.00 | 1.00 |
| **correct** | **.53** | **.59** | **.75** | **.97** |
| *§1 (mlp, trainable)* | *.49* | *.54* | *.64* | *.67* |

Style replicates. Correctness keeps its direction, and §9's "magnitudes need remeasuring" was
right — the correction at L32 is large (.67 → .97).

**But the correct column tracks head competence almost exactly** (agreement .182 / .226 / .298 /
.601). This sweep cannot separate "layers 0..L encode it" from "the head at L can read it". Two
readings are defensible — the confound is fatal, or the readout ceiling *is* the representational
ceiling — and the data does not choose. **§1's core claim is not established even after
remeasurement.** The same confound reappeared independently in §4.

Deep cells buy encodability with destruction: `correct` at L32 reaches head_acc .97 with
`gen_correct` **.02**.

## 3. brit held-out markers — the lexical-install hypothesis is DEAD

298 single-word am|br axes split 179 train / 119 held-out, **frequency-stratified** (a random
split would confound held-out performance with word rarity), exact token oracle, zero leakage by
construction. Trained on TRAIN axes only; measured reference-corrected British preference at the
**final logits** on both halves.

| arm | pref train | pref held-out | generalisation |
|---|---|---|---|
| stage-1 L12 | .979 | .891 | **.82** |
| full DPO | 1.000 | .947 | **.89** |
| upper-only | 1.000 | .965 | **.93** |

**All three generalise; stage-1 is the worst.** Every method learns a general "write British"
direction rather than a lookup table.

This **contradicts** the refusal lexical-gap hypothesis (stage-1 gap ~.00 vs full DPO +.28), which
is withdrawn. That hypothesis was weak anyway: its EN_SELECT/EN_EVAL split leaked at **33.7%**, so
it measured preferential reproduction of guaranteed-present phrasing, not seen-vs-unseen.

*Caveat: single seed, no error bars; .82 vs .93 may be noise. The direction of the result is
opposite to the hypothesis, so noise does not rescue it.*

## 4. Refusal testbed (new) — transfer is a null; over-refusal is the story

Qwen3-4B-Base, English-only training on 842 PKU-SafeRLHF pairs, evaluated en→zh/ar/it/vi/ko.

**Design decisions that mattered more than the training:**
- Only **11.5%** of PKU's "safe" responses are refusals; the rest are soft discouragement or
  benign compliance. Training unfiltered installs *hedging*, and the null would have looked like
  a depth result.
- **Japanese** failed the competence gate (4-gram repetition 1.00) before any training.
- **Chinese** turned out 81–98% degenerate under judging and is dropped entirely; every zh number
  quoted mid-session is void.
- Base over-refusal was first measured at n=6 (reading .000, unresolvable) before being redone on
  Aya at n=64.

**Judged (Qwen3-8B), zh dropped, mean over en/ar/it/vi/ko:**

| arm | harmful | benign | **discrimination** |
|---|---|---|---|
| base | .447 | .038 | **.409** |
| s1_L4 | .453 | .041 | .412 |
| s1_L12 | .481 | .031 | .450 |
| **s1_L24** | .869 | .258 | **.610** |
| full DPO | .774 | .200 | .574 |
| upper-only | **.924** | **.350** | .574 |

- **L4 and L12 did not install** — head_acc .95–.99 with KL .01–.06. The frozen readout is fully
  satisfied while the network barely moves (§2's surrogate gap, with a *frozen* head, on real data).
- **upper-only posts the highest raw refusal** in the band where steering is inert and
  cross-lingual decoding is at chance — opposite to the ladder's premise.
- **Refusal rate is the wrong headline.** L24 refuses "how do I become a dentist" and invents a
  legal justification for it. Discrimination (harmful − benign) is the minimum honest metric.
- **The transfer hypothesis is a null**: every arm that installs, transfers, and the base already
  refused cross-lingually before any training.
- Install strength tracks head competence (.152 / .202 / .380 → +.02 / +.08 / +.63) — §2's
  confound, found independently.

## 5. Regularising stage 1 — three attempts, one mechanism

Stage 2 always had a KL-to-base anchor (`KL_W=1.0`, forward KL on task completion tokens);
**stage 1 never had any explicit regulariser** — `kl_from_base` is logged, never penalised. That
is why L24 runs to KL 2.7 by step 10 unchecked. Three attempts to fix it:

| attempt | result | why |
|---|---|---|
| K-FAC leash on replay factors | negative (§1) | factors estimated on a distribution the edit doesn't move |
| replay KL anchor | inert | replay KL is **.003** while task KL is **2.7** at the same step |
| task-text KL anchor, W ∈ {.1,.2,.3,.5} | **blocks the install** (terse .05–.13, correct ~1.0, KL .19–.40) | the anchor and the objective fight over the same tokens |

**One mechanism explains all three.** On styc, the install *is* a change in the task-completion
distribution — terse vs explained is literally those tokens. So a task anchor opposes the install
directly, while the replay distribution is untouched by *either* the install or the damage, which
makes replay-based priors (including the K-FAC factors) blind to both.

**No global KL constraint can separate install from damage here.** A targeted one might — protect
answer-content tokens, let style tokens move — which is §3's queued "token-masked delta" idea,
now motivated for stage 1 as well. Untested.

## 6. Steering — the cleanest measurement in the project

Difference-in-means direction (ActAdd/CAA form) fit on **English harmful vs benign prompts**,
added at layer L during generation, scaled by α·R_L so α means the same fraction of residual
magnitude at every depth. **No head, no training, no checkpoint selection, no lexicon** — it
dodges every confound above. Judged, n=128/64, judge-vs-lexicon agreement median .93.

**Layer profile at α=0.05** (fig5a), base .52 / .00:

Refusal on harmful rises from base at L0–4 to **.89 at L18**, back to base by L24. False
positives are **not** flat across that band — they spike at **L13–17, peaking .46 at L16**, near
zero on either side. So discrimination is **bimodal**: **+.67 at L8**, collapsing to **+.35 at
L16**, recovering to **+.73 at L19**.

**Dose response** (fig5b), L16 vs L20:

| α | L16 harm/ben/disc | L20 harm/ben/disc |
|---|---|---|
| 0.01 | .60/.00/**+.60** | .56/.00/+.56 |
| 0.03 | .69/.05/**+.64** | .64/.00/**+.64** |
| 0.05 | .81/.46/+.35 | .75/.06/**+.69** |
| 0.07 | **.92**/.84/**+.08** | .85/.37/+.49 |

The layers convert dose into refusal at nearly identical rates; they differ in **when they
break**. At α=0.07, L16 posts the highest raw refusal in the entire sweep (.92) and is the worst
cell in it (84% of benign prompts refused, response diversity .48).

**Usable regime: α ≤ 0.03** — every cell there has diversity 1.00, benign ≤ .05, discrimination
.60–.64.

**New instrument: cross-prompt response diversity.** Per-response coherence structurally cannot
see saturation — at L16/α=.05 the model emitted *"I'm sorry, but I can't answer that. It's not
nice to ask someone to do something like that."* verbatim for both *"what is the capital of
UAE?"* and *"Who was the lead singer of The Who?"*, each response internally fluent. Distinct
openers / n catches it. This is the third §14-class trap in the project (after `colourcolour` spam
and `'100\n\nQuestion:'`) and the first one a gate catches automatically.

*The α=0.05 cross-lingual pass is past the saturation knee and is NOT reported here; rerun at
α ≤ 0.03 pending.*

## 7. Free-unembedding test — MIS-DESIGNED, uninformative

Intended to test whether the frozen `lm_head` is what limits the EAGLE readout. Both heads were
distilled **to base logits** and neither ever saw a preference label, so it measured convergence
toward `lm_head`, not expressive capacity. Result (free ≈ pinned at every depth) does not bear on
the question.

The variable was also mis-chosen. §18's actual distinction is the **functional form** — a probe
pools all positions into a free-choice direction, the head is pinned to realized-token logit gaps
— and freeing the output weights changes neither. The corrected design (keep the likelihood-margin
form, free the projection, train it on the *ranking* objective, read the ceiling) is documented in
the script docstring and is untested.

## 8. Method notes

- **Six instrument bugs today**, each producing numbers that were reported before being caught:
  English refusal scored on a narrower lexicon than every other language; benign set at n=6;
  `lang_precheck` executing its full model sweep on import; the probe's format confound (XSafety
  `commonsense` is multiple-choice, so the probe read format and scored 1.000 at layer 0);
  checkpoint mismatch flattering full DPO; `refusal_judge.py` writing `judged__fine.json` because
  it strips an `eval_` prefix that isn't there.
- **The probe's sanity criterion had to move** from "is L0 English accuracy high" to "is L0
  *transfer* high" — surface form separates within a language at every depth, but cannot produce
  cross-lingual transfer.
- **`pgrep -f` waiters deadlock in this environment**: the tool-wrapper shells carry the queued
  command text in their own cmdline, so a waiter matches its own parent. Cost ~30 minutes of idle
  GPU. `scratch_scripts/*.sh` retain the pattern in comments as a warning; do not reuse it.
- **Three concurrent fp32 log_softmax jobs over a 151936-token vocab exhaust 95 GB.** Run
  final-loss arms sequentially.
- **Report nothing before its check has run.** Every withdrawn claim in §0 was stated in the gap
  between measuring and validating.

## 9. Where the project stands

Three operations, three depths — now the clearest thing the project has:

| operation | depth | evidence |
|---|---|---|
| **readable** | L8–16 | cross-lingual probe (today); phase 5 layer sweep |
| **steerable** | L8–20 | §6 (today, judged); phase 9 dm steering (L8/L16, .586 judge win) |
| **trainable** via likelihood margins | L24+ | §4 (today, judged); upper-only .924 |

Reading location, intervention location, and training location are **different**. That
retro-explains the 07-28 death of the read-depth thesis (which assumed the first predicted the
second) and stage 2's failure (which tried to propagate mid→high).

**Blocked:** every depth claim from an EAGLE ladder, on the head-competence confound (§2, §4).

**The standing positive** is unchanged and is the project's strongest result: frozen-head stage-1
converts preference data into behaviour on **token-footprint** preferences at a fraction of full
DPO's KL (styc terse .95 @ KL 0.63 vs full DPO .00 @ 6.7), bounded by §18's scoping law.

**Next, in order:** (1) cross-lingual steering at α ≤ 0.03 — running; (2) seeds on §6, since
today's single-seed positives are exactly the ones that churned; (3) the write-up decides whether
token-masked anchoring (§5) is worth a fourth attempt.
