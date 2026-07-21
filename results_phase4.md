# Phase 4 — decision-position probes: preference training through activations

*2026-07-21. Runs the untested self-read corner (methods.md par.3 "move the read upstream" x
"LoRA <= L" x adaptive head x on-policy labels), then the two-head hybrid. Code:
`decision_probe.py`. A/B wrongness testbed, Qwen2.5-3B (36 layers). All arms share the same
decision-position probe at L\*=23, the same guards, the same seed.*

## Headline

**The two-head hybrid — activation-space margin below L\*, exact-expectation candidate REINFORCE
above it — installs the complete preference: targeted flip 1.000 (letter-balanced, fracA 0.50),
full transfer to unseen question types (OOD digits 1.000, sum 0.847) and knowledge questions
(1.000), with zero measured collateral (easy 0.99, off-menu 0.000), stable over 300 steps.**
Both halves are load-bearing: the pure activation channel installs but never stabilizes
(oscillates 0.18–0.65 for 600 steps), and the pure REINFORCE channel letter-locks for 300 steps
(fracA 1.00, flip pinned at the 0.5 base rate) — replicating phase-2's runs 1/2/4. The margin
half plays the role of phase-2's DPOP anchor *in activation space*: it shapes the <=L
representation into per-question wrong-slot plans, from which the relational flip — not the
letter shortcut — is the top blocks' cheapest J-ascent.

## Setup

**Read position.** The decision position: last prompt token ("Answer:"), whose residual feeds the
answer logits. Causal attention makes this state completion-independent — a chosen-vs-rejected
probe is *degenerate* here — forcing the probe target to change from ranking completions to
reading the impending action: P(emit "A" | h_dec), fit on the base model's own answers
(relational target, letter-randomized; letter cheats score 0.5).

**Decodability (Stage 1).** Surface heuristics through L17 (0.59–0.69: digit-count/first-digit
cues); consolidation L18–L23 (0.80 -> 0.99); 1.000 from L27. **L\* = 23** (0.990). Co-located
with phase-1's completion-end preference elbow (L21–23, same model). Sanity anchor that caught a
broken first sweep: the final-layer logit gap is linear in this state, so a probe there must
score ~0.99 (a labels-convention bug had produced an impossible curve; `train_bayes_head`
multiplies features by the target internally — pass them unsigned).

**Probe quality** (re: the class-bias worry — the model answers 99% correctly, so "about to be
wrong" is unobserved at base): the target is slot-of-correct-answer, 50/50 by letter
randomization, so 0.990 is against a real 0.5 floor. Calibration vs the model's own output
distribution: corr(p_probe, p_model(A)) = 0.957; on the 51/300 questions where the model is
*uncertain* (p in 0.1–0.9), the probe reads the true slot at 0.980 — better than the model's own
output (0.941): L23 has resolved the comparison before the output distribution adds its hedging.
Standing caveat: "deliberately about to answer wrong" states don't exist at base, so the probe's
wrong-side region starts as a mirror-image extrapolation and only becomes evidence-grounded once
refits see real flipped states.

**Training.** Prompt-only forward; margin loss = −log P_head(wrong letter | h_dec); backprop
through LoRA blocks <= 23. No likelihood terms in the pure arms. Guards (all activation-space,
each answering a v1 failure): balanced 50/50-label refit (skip if minority < 8), anchor
0.1·||f − f_base||², letter-symmetry 0.1·(mean z)². Hybrid adds, on blocks > 23 only: maximize
J = p(A|x)·r_A + p(B|x)·r_B — exact expectation over the two candidates, rewards from a SECOND
head (see below), <=L contribution of J's gradient masked out.

**Two heads (hybrid).** The decision-position head is completion-blind (same prompt -> same
h_dec -> same reward for every candidate: zero advantage), so the >L half needs an
*outcome-judge*: the phase-1 completion-end pairwise probe (held-out 0.977 at L23), frozen-base
read, giving per-question candidate rewards r(wrong) = 0.968, r(right) = 0.500 (fixed function of
(question, letter); precomputed). Plan-reader steers the decision; outcome-judge scores the act.

## Results — six arms

| arm | <=L signal | >L signal | refit | flip (end) | fracA | notes |
|---|---|---|---|---|---|---|
| frozen | margin | — | none | 0.007 | 0.49 | **forged**: endorsement pinned 0.99, behaviour never moves |
| filtered | margin | — | incremental | 0.473 (peak 0.713) | 0.23 | oscillates; policy races the lagging head (rotation 10°->90°) |
| buffer-300 | margin | — | from-scratch buffer | 0.653 (= max) | 0.67 | monotone-ish, calibrated head, best pure-arm OOD (0.63/0.47) |
| buffer-600 | margin | — | from-scratch buffer | 0.607 | 0.67 | **never stabilizes**: 0.18–0.65 band for 600 steps |
| **hybrid** | margin | exact-J | buffer | **1.000** | **0.50** | OOD 1.000/0.847, know-AB 1.000, easy 0.99, off-menu 0.000, stable |
| J-only | (guards only) | exact-J | buffer | 0.500 | **1.00** | **letter-locked** steps 50–300; = phase-2 runs 1/2/4 |

Collateral (hybrid, step 300): easy 0.99, know_free_anywrong 0.163, free_offmenu 0.000,
free_flip 0.113, yn_flip 0.267 — cross-format transfer stays modest; within-format transfer is
complete. Phase-2 run 5's pathologies (off-menu drift, late easy->0) are absent; plausible
mechanism: the exact-J with all-positive rewards conserves mass onto the menu, and the
activation anchor bounds representational drift.

## Failure taxonomy (each observed, each with its counter)

1. **Static forging** (frozen arm): even causally upstream, with an action-predictive probe, a
   fixed head direction is satisfiable off-distribution — position closes the *provably free*
   escape of 2.2 but not the forging channel. Counter: adaptivity.
2. **Refit-label collapse** (v1): self-labelled refit has a degenerate fixed point at any
   deterministic policy — labels go one-class, the head fits the mean and dies (loss -> ln 2),
   then nothing constrains the representation (off-menu drift destroyed the answer format).
   Libon et al. avoid it only via an external labeller. Counter: balanced refit — safe because an
   honest relational flip keeps sampled letters ~50/50; only letter policies skew them.
3. **Refit racing** (filtered): a 20M-param policy outruns a 4k-param incrementally-updated head,
   inventing fake directions faster than absorption (rotation runaway; train-specific forged
   features: train loss 0.49 vs unseen-prompt endorsement 0.23). Structural fact: **an honest
   flip needs no head rotation** (it moves states across a fixed boundary; the state->action map
   is unchanged) — rotation is per-se evidence of forging pressure. Counter: buffer refit (past
   fakes stay in evidence, labelled truthfully).
4. **Letter attractor** (J-only): with no representational shaping, the top blocks' cheapest
   J-ascent is a letter policy (J = 0.73 vs honest 0.97 — but the relational solution requires
   *reading* the <=L state, which nothing prepared). Counter: the margin half — or, in phase 2,
   a DPOP likelihood anchor.
5. Buffer refit's own wart: accumulated (state, label) pairs from different policy epochs
   conflict; the fresh head's held-out acc degrades 0.75 -> 0.44 over 300 steps. The install
   climbed anyway (early strong refits steer <=L into a basin where flipping self-sustains), but
   the pure arm's permanent oscillation is likely this same non-stationarity. Open: windowed or
   recency-weighted buffers.

## Interpretation

The activation channel and the likelihood channel fail in *complementary* ways: alone, the
former installs honestly but can't hold (adversarial equilibrium with its own reader), and the
latter holds trivially but installs the wrong thing (shortcut solutions). Composed with the depth
split — representation shaping below the decodability elbow, behaviour selection above it — each
supplies what the other lacks: the margin half selects *which* solution J's force lands on; J
supplies the stable behavioural pressure the margin half can't generate. The result is the
cleanest install in the project: complete, balanced, OOD-transferring, collateral-free. This is
the strongest evidence yet for the repo's central premise that the *attachment depth* of the
training signal is a real design axis — here it decides letter-lock vs full relational flip.

## Artifacts

`/workspace/decision_probe_curve.json`, `decision_feats_{train,eval}.npz`,
`decision_pairend_{train,eval}.npz`, histories: `decision_adaptive_v1_history.json`
(label collapse), `decision_adaptive_history.json` (filtered), `decision_adaptive_buffer_history
.json` (300; the 600-run overwrote in place — trajectory preserved in `decision_buffer600.log`),
`decision_frozen_history.json`, `decision_hybrid_buffer_history.json`,
`decision_hybrid_buffer_jonly_history.json`; adapters `decision_*_lora`; logs
`/workspace/decision_*.log`.

## Open

Seed replicates (all single-seed) · cross-format transfer (free/yn ~0.1–0.27 — is it the AB
few-shot frame, or the read position?) · windowed buffer for the pure arm's oscillation ·
fresh-probe collusion audit of the hybrid's final policy · L sweep for the hybrid (does the
margin/REINFORCE split degrade gracefully off the elbow? — the depth-thesis test proper) · UF
port: plan-state (prompt-end) decodability sweep first; Libon-style mean-pooled per-token reads
as fallback; the outcome-judge side ports directly (it is the phase-3 probe).
