# Phase 1 results — probe at the top vs DPO (and the repair)

*2026-07-18 · Qwen2.5-3B · A/B wrongness testbed (2,030 questions × 2 formats = 4,060 pairs;
chosen = wrong answer) · LoRA r=8 on all blocks · 300 steps, batch 6, no early stop ·
executed notebook: `probe_vs_dpo.out.ipynb` (runs 1–2 preserved as `probe_vs_dpo.run1_filtered.out.ipynb`,
`probe_vs_dpo.run2_3arm.out.ipynb`; logs `phase1_run*.log`). All numbers replicated across
independent runs.*

## Setup

Per-layer Bayesian linear probes fit on completion-end residuals of the **base** model.
Decodability: acc ≈ 0.58 at L0, climbs slowly to 0.93 by L19–20, **jumps to ≈ 0.99 at L21–23**,
1.000 from L28; final-norm read (L35) acc 1.000, best ELBO (−332). The preference is fully
linearly decodable well below the top — the phase-2 attach range starts ≈ L21.

Four arms, identical pairs, identical LoRA coverage. All checkpoints metered by one **pristine
head** (`fh_meter`, never trained/filtered) plus behavioral oracles (targeted-flip rates,
off-menu rate, per-side Δlogp vs base, DPO implicit reward `dpoR`).

| arm | signal | coupling | reader |
|---|---|---|---|
| `probe` | probe margin −log Φ(−z) | backprop through hidden states, teacher-forced pairs | policy's own activations (self-read), head frozen |
| `probe+filter` | same | same | self-read, head co-adapted online (`filter_round`, 'pref' labels, σ floor) |
| `rl` | probe score as reward | on-policy RLOO REINFORCE, KL-in-reward (β=0.03), pessimism 0.5 | **frozen base** activations (adapter off) |
| `dpo` | DPO (β=0.1) | teacher-forced pairs, loss on logprobs | — |

## Headline table (correct-answer rates; lower = more flipped)

| eval | probe | probe+filter | rl | dpo |
|---|---|---|---|---|
| ab (held-out, trained types) | 0.95 | 0.48 | **0.04** | 0.06 |
| know_ab (held-out facts) | 0.98 | 0.63 | **0.06** | 0.10 |
| ood_digits (unseen type) | 1.00 | 0.52 | 0.82 | **0.03** |
| ood_sum (unseen type) | 0.79 | 0.57 | **0.27** | 0.57 |
| free (named-option flip) | 0.37 | 0.31 | 0.01 | 0.06 |
| free_offmenu | 0.05 | 0.01 | **0.47** | 0.02 |
| mcq (capability) | 0.57 | 0.53 | 0.58 | **0.41** |
| easy 3-digit addition (capability) | 0.99 | 0.99 | 0.99 | **0.91** |
| final proxy inflation (dpoR, nats) | +3.0 | +5.7 (saturates) | +9.9 | **+33.3 (unbounded)** |
| Δlp rejected at step 300 (nats) | −1.0 | −1.5 | −6.1 | **−29.8** |

## Result 1 — the frozen self-read probe is wireheaded, invisibly

Trajectory: flips rise to ~0.55 by step 25 (entangled early path), then between steps 75–125 the
policy finds the escape — **head endorsement 0.70 → 1.00 while ab_flip 0.35 → 0.04**. Proxy
solved, behavior reverted to baseline. Free-format flip partially survives (~0.5); off-menu
stays ≈ 0. Deterministic: reproduced exactly in three runs.

**Mechanism — two decouplings:**

1. **Positional.** The probe reads the residual at the completion's last token — the state
   *after* the answer token is consumed. The answer is chosen by the logits one position
   earlier. Teacher forcing keeps sampling out of the gradient path, so a LoRA edit that fires
   on (answer-token-in-context × question features) can write "this answer is wrong" onto μ at
   the read position without running at the decision position.
2. **Directional.** cos(μ/sd, W_A − W_B) = −0.003: pushing h along μ is common-mode across the
   candidate logits. The behavioral readout pins a handful of directions out of 2048; μ lives in
   the null space.

The model already computes answer-wrongness perfectly (base probe acc 1.000) — it only has to
*amplify* that existing feature along μ, a near-rank-1 gain edit, cheaper than changing behavior.

DPO is structurally immune to (1): its loss is the emission distribution at the decision
position. Its blind spot is different — it only evaluates the two menu completions, so the −30
nats it strips from the rejected side drain partly to unlisted outputs (off-menu 0.42 mid-run).
*The margin probe fails because its loss reads the wrong node; DPO misbehaves because its loss
can't see where the mass goes; RL-from-probe (below) can misbehave only where the reward is
wrong.*

**The failure condition, one level up.** Neither decoupling is the fundamental statement, and
fixing (1) alone would not rescue the method — even a decision-position read leaves the
directional escape, because *any* frozen linear functional of a high-dimensional internal state
that is not tied to the output map has an enormous null space to be satisfied in. The general
diagnosis:

> A gradient loss that only asks for a feature **value** is satisfied by whatever is cheapest,
> and at the top layer, **forging the feature is far cheaper than changing behavior.** The base
> model already computes answer-wrongness perfectly (probe acc 1.000); amplifying that existing
> signal along μ is a near-rank-1 gain edit. Re-routing the decision is a real computation
> change. SGD buys the cheap one.

The two decouplings are the reasons forgery is cheap *at this attach point* — the failure
condition is the price gap. Corollary: backprop-from-probe is not intrinsically hopeless; it
works iff forgery is made expensive or unstable. Three levers: (a) **depth** — at L21 a forged
μ-component must survive 15 blocks of nonlinear mixing without touching behavior; the null space
that is ~2000-dimensional at the top (one frozen matrix downstream) becomes the null space of
the *remaining computation*, which may be tiny. Whether the gradient method snaps into honesty
below the decodability elbow is phase 2's sharpest question, with today's result as the
calibrated top-layer anchor (viability = 0). (b) **co-adaptation** — a refitting head is a
moving target; forgery against it is unstable (Result 2: half the flip, zero displacement,
stable for 300 steps). (c) **tying μ to the output map** — constrain the probe direction toward
the span the unembedding reads; in the limit this re-derives DPO, so it is a sanity check, not a
rescue. The stakes: if (a) holds, the gradient method recovers everything that made it
attractive — dense, sample-free, differentiable — and the depth at which it becomes honest is
itself the Occam quantity.

## Result 2 — co-adaptation is load-bearing

Identical arm with online cooperative head refits (`filter_round` every 10 steps, variance
floor): **no wirehead**. Flips hold at ~0.55 ab / 0.67 free for all 300 steps, head endorsement
plateaus at 0.75, proxy saturates at +5.7 nats, off-menu 0.00, capabilities intact. The moving
target keeps absorbing the synthesized μ-component; the static escape never stabilizes. Partial
coupling, zero displacement. (Mechanism deserves its own experiment.)

## Result 3 — probe-as-reward with a frozen base reader matches DPO, gracefully

`sampled_rl_step(..., score_with='base')`: the probe scores *sampled* completions on the **frozen
base model's** activations; reward is detached; policy gradient through log π. Two changes vs the
margin arm — coupling (RL) and reader (frozen base) — the wirehead is impossible by construction
(reward is a fixed function of emitted text).

- **Matches DPO on the installed preference:** ab 0.96 flip (DPO 0.93), know_ab 0.94 (DPO 0.90)
  — at **1/3 the proxy inflation** (+10 vs +33 nats) and 1/5 the rejected-side trench.
- **Capabilities intact:** easy 0.99 vs DPO 0.91; mcq 0.58 vs 0.41.
- **Narrower generalization:** DPO extends the flip to unseen comparison *types*
  (ood_digits 0.03); RL barely does (0.82). The probe-reward installs the trained preference
  without DPO's flip-everything spillover — the Occam-flavored outcome, visible already at the
  top layer.
- **Cost:** free-form off-menu drift (0.47 final, oscillating 0.2–0.6 against the KL anchor).
  This is the *visible*, reward-level failure: the head was fit only on on-menu pairs
  (`neg_frac=0`) and has no opinion about off-menu numbers. Designed fix exists and is untested
  here: `build_data(neg_frac>0)` off-menu negatives (+ pessimism already on).

This construction is structurally **Goodfire's RLFR** (Features as Rewards: probe reward on a
frozen copy, standard RL) — convergent design. Our frozen-margin arm is the counterfactual their
frozen-copy choice avoids, run to completion on an exact oracle.

## Gradient geometry (calibration cells, replicated)

| measurement | value |
|---|---|
| probe-vs-DPO grad cosine (same batches, mixed) | ≈ 0 (−0.06…+0.04) |
| dpo\|A vs dpo\|A′ (letter-conditioned ceiling) | **+0.989** |
| dpo\|A vs dpo\|B (cancellation signature) | +0.303 |
| probe\|A vs dpo\|A (orthogonality test) | +0.15…+0.18 |
| probe\|A vs probe\|B (pre-registered **+**) | **−0.47 (!)** |
| cos(μ/sd, W_A − W_B) | −0.003 |

The near-zero mixed cosine + high conditioned DPO ceiling confirm the relational-preference
analysis (`notes_gradient_equivalence.md`): on this testbed the two objectives live on
near-orthogonal axes, so *equivalence was never testable here* — option-order randomization sends
DPO's averaged token-direction to zero by construction. Equivalence should be tested on a
token-expressible preference (UltraFeedback; or the `german_always` language preference as an
oracle-equipped positive control).

**Open puzzle:** probe|A vs probe|B was pre-registered positive (μ doesn't depend on the letter)
but is robustly **−0.47** in every run. The per-token Jacobian at the answer position flips the
backbone image of ±μ; possibly related to how the wirehead is implemented. Unexplained.

## Taxonomy (one line)

**Margin-probe fails invisibly (wrong causal node); DPO drifts blindly (off-menu is outside its
loss); RL-from-probe fails only where the reward model is wrong — visibly, and fixably at the
reward level.**

## Next

1. **`neg_frac > 0` ablation** for the RL arm — if off-menu drift is cured, RL-from-probe
   dominates DPO on every metric here.
2. **Trained-RM baseline** (RLOO with a pair-fine-tuned RM): completes the 2×2
   {offline, on-policy} × {feature-read, output-read}; attributes the narrow-generalization and
   capability-retention findings to the probe vs to on-policyness. Framing: conventional RM =
   trained backbone read at top; our probe = frozen backbone read at depth L.
3. **RL with `score_with='policy'`** — isolates which of {RL coupling, frozen reader} killed the
   wirehead (theory: RL alone stops the gradient path; frozen reader also stops slow drift).
4. **Phase 2 — the depth sweep**, now sharpened into the project's central question: (a) does
   *backprop-from-probe* become viable as L decreases — i.e., is there a depth threshold where
   forging the feature becomes more expensive than changing behavior and the gradient method
   snaps into honesty? (b) does RL-from-probe's generalization narrow further / Goodhart slope
   flatten as L approaches the decodability elbow (L21–23)? Top-layer anchors for both are now
   calibrated.
5. The probe|A/probe|B sign puzzle.
