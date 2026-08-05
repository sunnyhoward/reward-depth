# 2026-08-05 — negative results

*Qwen2.5-3B (styc/brit) and Qwen3-4B-Base (refusal/steering). Single seed unless stated. Judged
numbers are Qwen3-8B; lexicon numbers are the online meter only — measured today, the refusal
lexicons agreed with the judge only .62–.98 and systematically OVER-read.*

**This session produced no positive result I would defend.** It produced six negatives, four of
which are mechanistically linked, plus two instruments and a list of measurement traps. That is
the contribution: the linked negatives explain each other, and together they close off a family
of approaches that looks obviously worth trying and is not.

---

## 0. Summary

| # | claim | status |
|---|---|---|
| 1 | K-FAC curvature leash widens the stage-1 safe window | **NEGATIVE** — reproduced by a learning-rate cut |
| 2 | The encoding-depth table (§1 of eagle/RESULTS) | **UNESTABLISHED** — confounded with head competence |
| 3 | Restricted low writes learn "the act", full DPO learns "the words" | **CONTRADICTED** — all methods generalise; stage-1 worst |
| 4 | Depth-dependent cross-lingual refusal transfer | **NULL** — every arm that installs, transfers |
| 5 | A KL-to-base anchor protects stage 1 | **NEGATIVE** — blocks the install at every dose |
| 6 | Difference-in-means steering localises refusal | **LARGELY TAUTOLOGICAL** — see §6 |

Two instruments were added (§9) and six instrument bugs were found and fixed (§10), each of
which had already produced a number that was reported before being caught.

---

## 1. K-FAC leash — NEGATIVE (closes NEXT.md queue item 1)

Setup note worth keeping: factor estimation is **11 minutes, not hours**. `--placement auto`
sends every factor with dimension > 4096 to CPU, which is all the 11008-dim MLP factors;
`--placement model` keeps all 175 modules (37.4 GiB dense fp32) GPU-resident.

λ ∈ {1, 10} were under-dosed (penalty 3% and 25–30% of the DPO term). λ ∈ {100, 1000} **did**
widen the L24 safe window, monotonically. Then the control killed it:

| arm | best cell (terse / correct) | KL |
|---|---|---|
| λ=0, LR 1e-4 | 1.00 / .984 @5 | 0.71 |
| λ=1000, LR 1e-4 | 1.00 / .906 @30 | 1.73 |
| **λ=0, LR 1e-5** | **1.00 / 1.000 @35–50 (four consecutive ckpts)** | 0.71–1.39 |

A plain learning-rate cut strictly dominates. The leash bought slower effective steps.

**Damage onset is arm-invariant; depth of collapse is not.** Correctness holds (≥.91) to KL ≈ 2.0
in all seven arms and degrades past ≈ 2.0–2.5 in all seven — but Spearman(KL, correct) is only
**−0.48**, and past onset the LR controls hold .45–.69 where the λ arms sit at .06–.15. A stronger
earlier claim ("install .7 / damage 2.4, invariant") was withdrawn after plotting all 110 eval
points (fig1).

See §5 for why the leash was aimed at the wrong subspace, which is the deeper reason.

## 2. Encoding depth — UNESTABLISHED

Remeasured with the tf head, frozen (fig2):

| factor \ L | 4 | 12 | 24 | 32 |
|---|---|---|---|---|
| style | 1.00 | 1.00 | 1.00 | 1.00 |
| **correct** | **.53** | **.59** | **.75** | **.97** |
| *§1 (mlp, trainable)* | *.49* | *.54* | *.64* | *.67* |

Style replicates. Correctness keeps its direction and §9's "magnitudes need remeasuring" was right
(the L32 correction is large).

**But the correct column tracks head competence almost exactly** — agreement .182 / .226 / .298 /
.601. The sweep cannot separate "layers 0..L encode it" from "the head at L can read it", and the
data does not choose between "the confound is fatal" and "the readout ceiling *is* the
representational ceiling". **The repo's core depth claim is not established even after
remeasurement**, and the same confound reappeared independently in §4 (install strength .02 / .08
/ .63 against head competence .152 / .202 / .380).

Deep cells buy encodability with destruction: `correct` at L32 reaches head_acc .97 with
`gen_correct` **.02**.

## 3. brit held-out markers — CONTRADICTS the lexical-install hypothesis

298 single-word am|br axes split 179 train / 119 held-out, frequency-stratified, exact token
oracle, **zero leakage by construction**. Trained on TRAIN axes only.

| arm | pref train | pref held-out | generalisation |
|---|---|---|---|
| stage-1 L12 | .979 | .891 | **.82** |
| full DPO | 1.000 | .947 | **.89** |
| upper-only | 1.000 | .965 | **.93** |

All three generalise to markers never seen; stage-1 is the **worst**. Every method learns a
general direction rather than a lookup table.

This kills the refusal lexical-gap hypothesis (stage-1 gap ~.00 vs full DPO +.28), which is
withdrawn. That hypothesis was weak anyway — its EN_SELECT/EN_EVAL split **leaked at 33.7%**, so
it measured preferential reproduction of guaranteed-present phrasing, not seen-vs-unseen.

*Single seed, no error bars; .82 vs .93 may be noise. But the direction is opposite to the
hypothesis, so noise does not rescue it.*

## 4. Refusal transfer — NULL

Qwen3-4B-Base, English-only training on 842 PKU-SafeRLHF pairs, judged, zh dropped (81–98%
degenerate), mean over en/ar/it/vi/ko:

| arm | harmful | benign | **discrimination** |
|---|---|---|---|
| base | .447 | .038 | **.409** |
| s1_L4 | .453 | .041 | .412 |
| s1_L12 | .481 | .031 | .450 |
| s1_L24 | .869 | .258 | **.610** |
| full DPO | .774 | .200 | .574 |
| upper-only | **.924** | **.350** | .574 |

- **The transfer hypothesis is a null.** Every arm that installs, transfers — and the base already
  refused cross-lingually (.20–.53) before any training, so there was no gap to explain.
- **L4 and L12 did not install** — head_acc .95–.99 with KL .01–.06. The frozen readout is fully
  satisfied while the network barely moves.
- **upper-only posts the highest raw refusal** in the band where steering is inert, contradicting
  the ladder's own premise.
- **Refusal rate is the wrong headline.** L24 refuses "how do I become a dentist" and invents a
  legal justification. Discrimination (harmful − benign) is the minimum honest metric.
- The collateral transfers worse than the install: over-refusal .125 in English, .219–.344
  elsewhere.

Data notes that mattered more than the training: only **11.5%** of PKU's "safe" responses are
refusals (training unfiltered installs *hedging*); Japanese failed the competence gate before any
training; Chinese is degenerate under judging and every zh number quoted mid-session is void.

## 5. Regularising stage 1 — three failures, one mechanism

Stage 2 always had a KL-to-base anchor (`KL_W=1.0`, forward KL on task completion tokens).
**Stage 1 never had any explicit regulariser** — `kl_from_base` is logged, never penalised — which
is why L24 runs to KL 2.7 by step 10 unchecked.

| attempt | result |
|---|---|
| K-FAC leash (factors from replay) | negative (§1) |
| replay-KL anchor | inert — replay KL **.003** while task KL is **2.7** at the same step |
| task-text KL anchor, W ∈ {.1,.2,.3,.5} | **blocks the install** — terse .05–.13, correct ~1.0, KL .19–.40 |

**One mechanism explains all three.** On styc the install *is* a change in the task-completion
distribution — terse vs explained is literally those tokens. So a task anchor opposes the install
directly, while the replay distribution is untouched by *either* the install or the damage, making
every replay-based prior (including the K-FAC factors, estimated on that same corpus) blind to
both. **K-FAC was not merely under-dosed; it was aimed at a subspace the edit does not travel.**

**No global KL constraint can separate install from damage here.** A targeted one might — protect
answer-content tokens, let style tokens move — which is §3's queued "token-masked delta", now
motivated for stage 1 too. Untested.

## 6. Steering — largely tautological

Difference-in-means direction (ActAdd/CAA), added at layer L during generation, judged, n=128/64.
Reported here in demoted form after the obvious objection, which is correct:

**The core effect is close to definitional.** We compute the direction that separates harmful from
benign prompts, add it to prompts, and observe the model treating them as more harmful. That
benign prompts also get refused is not a discovery — we made them look harmful.

What is **not** implied by construction:

- **The layer profile.** Null at L0–4 and L24+, active only at **L8–20**. That localisation is
  real information about where the representation is causally read.
- **Selectivity varies 23× at matched perturbation.** Every mid-stack layer perturbs the model
  equally (replay KL .151–.168, within 10%) while benign over-refusal spans **.02 at L8 to .46 at
  L16**. Same push size, wildly different consequences.
- **Cross-lingual transfer** (α ≤ 0.03, judged): an English-fit direction moves ar/it/vi/ko by
  **+.14 to +.28**, with zero effect at L0 and L24 in every language. But this is *causal
  confirmation of a correlational finding* — the probe had already shown the direction is
  language-general — not a new result.

**And the deeper problem: a difference-in-means direction inherits every difference between the
two sets it is fit on.** Both directions we built are confounded:

| direction | confounded with |
|---|---|
| harmful − benign | dataset provenance (MultiJail vs Aya differ in more than harm) |
| refused − answered | harm category — refusal rate varies **.17–.60** across categories |

The second was built to fix the first (and does reduce benign over-refusal from .46 to .00 at the
worst cell, no saturation at any dose). But `cos(decision, harmfulness) = −.40 to −.81` — the two
directions are strongly **anti-aligned**, which is not explicable by either story and most likely
means neither vector is the concept it is named after. A prompt-length explanation was tested and
**failed** (medians identical, 14.1 vs 15.2 tokens within MultiJail).

**Isolating "refusal decision" from "what these two prompt sets happen to differ in" is the actual
hard problem, and this session did not solve it.**

## 7. Free-unembedding test — MIS-DESIGNED

Intended to test whether the frozen `lm_head` limits the EAGLE readout. Both heads were distilled
**to base logits** and neither saw a preference label, so it measured convergence toward `lm_head`
rather than expressive capacity. The result (free ≈ pinned at every depth) does not bear on the
question. The variable was mis-chosen too: §18's distinction is the **functional form** (pooled
free direction vs realized-token logit gaps), which freeing the output weights does not change.

## 8. The four linked negatives — the actual contribution

1. **Replay-based priors cannot protect an on-distribution edit.** Stage-1 moves task text 2.7
   nats and replay .003. Anything estimated on replay — K-FAC factors, replay anchors — is blind
   to the damage. (§1, §5)
2. **No global KL anchor separates install from damage** when the install *is* a change in the
   protected distribution. (§5)
3. **Depth ladders with per-layer readout heads confound depth with readout competence.** Found
   twice independently, on unrelated measurements. Blocks every depth claim built this way. (§2, §4)
4. **Difference-in-means steering measures your class contrast, not your concept.** Both
   directions built here are confounded, in different ways, and are mutually anti-aligned. (§6)

These are linked: (1) and (2) are the same fact about *where* an edit lives; (3) and (4) are the
same fact about *what your instrument actually contrasts*. Together they close off the obvious
approaches to both halves of the project's question.

## 9. Instruments added

- **Discrimination** = refusal(harmful) − refusal(benign). Refusal rate alone cannot distinguish
  "better at spotting harm" from "refuses everything": L16 at α=.07 posts the highest raw refusal
  in the entire sweep (.92) and is the worst cell in it (benign .84).
- **Cross-prompt response diversity** = distinct openers / n. Per-response coherence structurally
  cannot see saturation — at L16/α=.05 the model emitted *"I'm sorry, but I can't answer that.
  It's not nice to ask someone to do something like that."* verbatim for both *"what is the capital
  of UAE?"* and *"Who was the lead singer of The Who?"*, each response internally fluent. This is
  the third §14-class trap in the project (after `colourcolour` spam and `'100\n\nQuestion:'`) and
  the first one a gate catches automatically.
- Caveat on discrimination: it subtracts a **thresholded** curve (benign, floor 0) from a
  **saturating** one (harmful, base .52, ceiling ~.9), so its shape across layers is partly an
  artifact. Matched-install comparison (benign at equal harmful) is the sound version; at matched
  install L16 costs ~2× L20.

## 10. Instrument failures, and what they cost

Six bugs, each of which produced a number that was reported before being caught:

1. English refusal scored on `EN_EVAL` alone — a narrower lexicon than every other language used,
   so "full DPO didn't teach refusal" was wrong (it refused in the *training* phrasing).
2. Benign over-refusal first measured at n=6, reading .000 and unable to resolve anything.
3. `lang_precheck` executing its entire model-loading sweep on import.
4. The probe's format confound: XSafety `commonsense` negatives are multiple-choice, so the probe
   read **format** and scored 1.000 at layer 0. Fixed, then the sanity criterion itself had to move
   from "is L0 English accuracy high" to "is L0 **transfer** high" — surface form separates within
   a language at every depth but cannot produce cross-lingual transfer.
5. Checkpoint mismatch flattering full DPO (compared at KL 0.47 against L24's 0.83).
6. `refusal_judge.py` writing `judged__fine.json` — it strips an `eval_` prefix that isn't there.

**Environment traps:** `pgrep -f` waiters deadlock here (tool-wrapper shells carry the queued
command in their own cmdline, so a waiter matches its own parent) — cost ~30 minutes of idle GPU.
Three concurrent fp32 `log_softmax` jobs over a 151936-token vocab exhaust 95 GB.

**Process note.** Every claim withdrawn today was stated in the gap between measuring and
validating. Negatives and confounds survived the day; every positive claim about which method or
depth is better churned. Three separate mechanistic explanations offered for the L16 anomaly
(moralising preamble, dataset length, dose saturation) were each killed by the next measurement.

## 11. Where the project stands

**Blocked:** every depth claim from an EAGLE-style ladder, on the head-competence confound (§2,
§4). No design for separating readout competence from representational content is currently known.

**Closed:** K-FAC and KL-anchor regularisation of stage 1 (§5); the refusal-transfer hypothesis
(§4); the lexical-install hypothesis (§3).

**Unresolved and probably not worth more attempts without a new idea:** isolating a refusal
direction from a confounded class contrast (§6).

**Standing positive, unchanged from 08-04 and not touched today:** frozen-head stage-1 converts
preference data into behaviour on **token-footprint** preferences at a fraction of full DPO's KL
(styc terse .95 @ KL 0.63 vs full DPO .00 @ 6.7), bounded by §18's scoping law. That remains the
project's only defensible positive result, and nothing this session strengthened or weakened it.
