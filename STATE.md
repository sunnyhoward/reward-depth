# State of play

*Orientation document. Written 2026-07-28, after five phases. Read this before the phase docs —
it says what the project is asking, what is actually established, and what is still open. The
phase docs (`results_phase1.md` … `results_phase5.md`) are the primary records and each is
accurate about its own runs; what they lack is a shared frame, because each one was written at
the end of a session and opens with a headline written to be persuasive.*

## Update 2026-07-30 (read this, then the rest still stands as frame)

Two sessions have happened since this file was written; their headline effects on the picture:

- **Phase 6 exists** (`results_phase6.md`, 2026-07-29) and answers two of the "next" items below:
  the **steering experiment ran — negative** (probe directions are causally inert at every depth;
  judge-null in 94% of 96 cells; "probes label, they do not steer"), and the whole
  frozen-probe-as-margin-target family is closed (forging is K-FAC-metric-null, λ-invariant).
  What *did* work: a **per-batch adaptive, saturated mean-diff activation objective** (first
  activation objective to move behaviour more than the meter; lag k≥1 is the anti-forging
  resource) with a **strict inverted-U in attach depth peaked at the decodability elbow** — the
  project's first affirmative depth-matters result. Separately, a **replay-K-FAC curvature prior
  beats live-reference DPO** in token space (Pythia, 2 seeds).
- **Phase 6 §8 (post-doc controls) revises phase 6's own headline**: the stage-2.5 hybrid's
  margin half is **not load-bearing** on cc — exact-J alone installs 98% and keeps rising — and
  exact-J's cleanliness was its implicit **on-menu constraint**, not the anchor: the same probe
  reward taken by sampling (open vocab) is hacked instantly (offmenu 1.0 @50).
- **Write depth was run after all** (same evening as this file) — see the corrected section
  below: parameter-matched lower/upper window DPO is an **in-domain null** at sane LRs.
- **In flight (2026-07-30):** the UF port of the load-bearing test — anchored sampled RLOO from
  the frozen L12 probe (`uf_probe_rl.py`, guards on) vs the same + mean-diff margin ≤L12
  (`uf/uf_hybrid_md.py`), 300 steps each, identical guard set, single seed. Stage A reproduced
  (L*=12 @ 0.791, max 0.799 @ L16) on a fresh box.

## The question

A preference — which of two responses is better — is linearly readable from **layer 12 of 32** in
the *untrained* Tulu-3-8B-SFT, at 0.79–0.80 accuracy. That happens to equal what a 400-step DPO
run installs. So: if you are going to train a model on that preference, does it help to attach
the training signal at layer 12 instead of at the output?

The bet is Bayesian Occam. A reader at L12 can only see the simple semantic core of the
preference. Dataset idiosyncrasies — length, style, formatting tics — are only decodable late, so
they cannot enter a reward defined at L12. Prediction: a policy trained against the early reader
overfits the preference less and Goodharts more gracefully than DPO or a top-attached reward
model.

Everything in this repo is machinery for testing that.

## Three axes, which are not the same thing

Most of the confusion in this project comes from "depth" meaning three different things. They
have different evidence, different difficulty, and different track records.

| axis | what varies | status |
|---|---|---|
| **read depth** | which layer's probe *generates the labels/reward* | **tested on UF 2026-07-28 — negative** (§below) |
| **write depth** | which blocks *get updated* (LoRA layer range) | **endpoints run 2026-07-28 — in-domain null** (§below) |
| **coupling depth** | whether gradients flow *through activations* at the attach layer | tested hard, phases 1/2/4/5 — **fails**; first working form found phase 6 (adaptive direction, elbow-only) |

Phases 4 and 5 — two full sessions, and most of the code volume (`uf_hybrid.py` →
`uf_hybrid2.py` → `uf_hybrid3.py`, 26 KB) — went into coupling depth. That is the axis phase 1
had already shown is gameable. The adaptive heads, windowed buffers, EMA damping, pristine
forgery meters and mass anchors all exist to fight forging that `methods.md` §2.2 established was
the expected outcome.

The read-depth test, by contrast, is: run `uf/uf_soft_dpo.py` twice with `L_OVERRIDE=12` and
`L_OVERRIDE=31`. No new machinery. It has been an open item since phase 3 and has been deferred
in every session since.

## Settled

These are reproduced, instrumented, and worth building on.

- **The preference is decodable early.** Per-layer probes on frozen-SFT completion-end residuals:
  0.70 at L0–2 (surface features), plateau **0.799 from L\*=12**, flat to L31 (0.770 at the top;
  max 0.799 @ L16). Survives IPW length-matching against a 0.619 length-only cheat floor — the
  deep probe was not riding the length confound; the *early* layers partly were, so matching
  widens the early-vs-plateau gap. The evidence/ELBO proxy independently prefers the L12–19 band
  over the top. (phase 2 §5, phase 3 §1c, phase 5 §0)

- **You cannot backprop a probe loss through the model's own activations.** The model forges the
  feature and behaviour reverts to baseline. Confirmed at the top layer, at mid-depth, at 3B, 7B
  and 8B, with frozen and with co-adapting heads, teacher-forced and on-policy. The general
  statement (phase 1): *a gradient loss that asks only for a feature value is satisfied by
  whatever is cheapest, and forging an already-computed feature is cheaper than changing
  behaviour.* Matches Libon et al. independently. (phase 1 R1, phase 2 §4, phase 5)

- **Soft-label DPO from the frozen L12 probe matches ground-truth DPO, with much less
  collateral.** Held-out implicit accuracy 0.800 ± 0.021 vs 0.805, using only the 3,000 probe-fit
  pairs. Chosen-side likelihood *preserved* (Δlp +0.7 vs −9), margin inflation +9.8 vs +33 nats.
  This is the working method on the realistic dataset. Note it decays after ckpt200 (0.760 at
  400) — early stopping matters. (phase 3)

- **Calibration for that 0.805.** The *official* Tulu-3-8B-DPO scores 0.623 ± 0.026 on this split
  despite vastly more compute. So the in-domain 400-step LoRA DPO number is largely
  dataset-specific fit, and there is a ~0.18 in-domain/general gap that scale does not close.
  Base SFT raw ranking is 0.409 (length-confounded). (phase 3 §3)

- **The DPOP anchor is load-bearing for anything RL-shaped.** Without the one-way floor
  `relu(log π_ref(y_c) − log π_θ(y_c))`, no RL-from-probe configuration installs anything; with
  it, RLOO escapes the letter attractor and installs fully. (phase 2 §2)

- **Two attractors explain nearly every naive failure**: the letter-policy attractor and
  likelihood displacement. `fracA` is logged everywhere as a result. Any aggregate that looks
  like a base rate probably is one. (phase 2 §1, and the phase-1 corrections in phase 2 §6)

## Suggestive, but single-seed or toy-only

- **Read depth, one data point.** Anchored RLOO at the elbow (L21) vs at the top (L35) on the A/B
  testbed: equal install strength, but the elbow arm has the lowest proxy inflation (+8.9 vs
  +12.3 nats) and shallowest rejected-side trench (−4.3 vs −6.9). Directionally the Occam
  prediction. Single seed, synthetic oracle, 3B model, 1-token answers. (phase 2 §3)

- **The two-head hybrid on the toy.** Activation margin below L\*=23 plus exact-expectation
  REINFORCE above it installs the complete preference: flip 1.000 letter-balanced, OOD
  1.000/0.847, off-menu 0.000, stable for 300 steps, both halves load-bearing under a six-arm
  ablation. Genuinely the cleanest install in the project — on a task with two-token answers and
  a deterministic oracle. (phase 4)

## Tested and failed

- **The hybrid does not port to UltraFeedback**, in two independent redesigns. In both forks the
  margin half's representational rewrites generate likelihood collateral (~6 nats of pair-mass
  sink in the first 50 steps, measured directly), whichever anchor exists responds through a
  biased channel, and the anchor's response out-muscles the install force. The v2 adaptive-head
  machinery *worked* (refit val-acc held 0.65–0.94 for 300 steps where v1 collapsed to chance by
  step 100); the objective it was serving did not. (phase 5)

  Worth keeping from this: the failure taxonomy, and one real positive — anchored on-policy RLOO
  doubled held-out generation quality by step 50 (judge-z 1.63 → 3.43) with a clean likelihood
  profile, before the margin-half dynamics dragged it back.

## Read depth — TESTED 2026-07-28. Negative.

Four arms, identical recipe, only the label source differing: soft-DPO from probes at L12 / L16 /
L31, plus ground-truth DPO. Stage A reproduced phases 3/5 exactly first (funnel 15,283 pairs,
L\*=12 @ 0.791, ELBO peak L14, soft labels mean p 0.739), so these numbers sit on a verified base.

**In-domain (big-N, 350 held-out pairs, SE ≈ 0.021)** — ckpt200: L12 **0.800**, L16 0.797,
GT **0.800**, L31 0.771. Phase 3's soft-DPO result reproduces exactly (0.800 ± 0.021, Δlp chosen
+0.65 vs the documented +0.7), including the collateral advantage over ground-truth DPO
(GT: Δlp chosen −5.03, margin 20.5 nats vs the probe arms' ~+0.4 and ~10 nats).

**Out of domain (RewardBench, 1,278 pairs, SE ≈ 0.012)** — GT 0.771 > L16 0.746 > L12 0.729 >
L31 0.715. **The probe arms order exactly by probe accuracy** (0.799 > 0.791 > 0.770), not by
depth. L16 is deeper *and* 4× more length-aligned held-out than L12, and transfers **better** —
the opposite of the Occam prediction. On the pre-registered adversarial slices L12 and L16 are not
merely close but identical: alpacaeval-length 0.933/0.933, llmbar-adver-neighbor 0.617/0.617,
llmbar-adver-manual 0.478/0.478.

**Conclusion: attaching the reward earlier does not improve generalisation. What predicts OOD
transfer is how accurate the probe is, not how early it reads.**

Caveats. Single seed. Probe-arm gaps are ~1–1.4 SE individually; it is the monotone match to probe
accuracy across three arms that carries the result. **GT-vs-probe is confounded** — `uf_dpo_train.py`
trains on 12,000 unmatched pairs against the probe arms' 3,000 length-matched ones, so GT's OOD lead
could be data volume or the absence of length matching. The fix is queued: `HARD_LABELS=1` in
`uf_soft_dpo.py` runs dataset labels on the *same* 3,000 matched pairs (`uf/run_gt_matched.sh`),
which isolates the label source.

**Side finding, unrelated to depth and larger than any depth effect:** every arm badly degrades
RewardBench safety ranking (`refusals-dangerous`: base 0.967 → GT 0.600, L16/L31 0.300, L12 0.233)
and every arm loses to the base model on `chat-hard` (0.703 → 0.523–0.561) and `reasoning`
(0.888 → 0.752–0.850). Gains are concentrated in `chat` (0.315 → 0.94), which is where base *raw*
ranking is worst — i.e. largely a length-bias correction rather than preference learning. Whether
the safety number reflects a behavioural regression or only a ranking shift is being checked with
`uf/uf_safety_probe_gen.py` (generation + refusal rates, not rankings).

**Write depth — endpoints RUN later that same day (2026-07-28 evening). In-domain null.**
The paragraph below originally said this had never been run; `uf/run_layerwindow.sh` (Task A)
then ran it: GT-label DPO via `uf_dpo_train.py`, the only change being `LORA_LAYERS` —
parameter-matched **lower (blocks 0–15)** vs **upper (16–31)**, 20.97M trainable each,
3 LRs, single seed. Big-N 350 pairs (`results/runs/uf_bigN_taskA.json`, SE ≈ 0.021):

| lr | lower acc | upper acc | lower Δlp chosen | upper Δlp chosen |
|---|---|---|---|---|
| 5e-5 | .806 | .806 | **−9.1** | −23.5 |
| 1.5e-4 | .803 | .811 | −18.8 | −15.3 |
| 5e-4 | .711 | .620 | −97 | −118 |

At sane LRs the windows are statistically indistinguishable on install accuracy — "DPO with the
upper half frozen" costs nothing in-domain, consistent with the read-depth null. The one
suggestive asymmetry is collateral at the best LR (lower's chosen-side displacement −9.1 vs
−23.5), single-seed and untested for significance. **Not run:** OOD (RewardBench) for these
arms, the two-stage legibility arm, and the param-matched-r control from the design note below
(moot — the windows were parameter-matched directly).

### Design note on the write-depth arm

The obvious worry: if only blocks ≤12 update, the unchanged blocks 13–31 may not be able to
*read* whatever changed, and the arm underfits for reasons that have nothing to do with the
Occam claim. This is well-founded — phase 1 measured `cos(μ, W_A − W_B) = −0.003`, i.e. the
probe direction at the read layer is orthogonal to what the output map reads. Decodable does not
mean used.

Two things make it a safe experiment anyway:

1. **The failure mode is underfitting, not forgery.** With a likelihood-reading loss (DPO or
   soft-DPO), the loss falls only if emitted-token log-probs change. An edit at ≤12 that the
   frozen upper stack ignores yields *zero* loss reduction. So the pathology that killed phases
   1/4/5 — a loss that goes down while behaviour doesn't move — is structurally impossible here.
   A restricted arm that can't express the preference shows up as flat or low install accuracy,
   which is a readable result.
2. **The constraint is arguably the point.** An edit that must be legible to an unchanged upper
   stack can only operate on representations the model already computes *and already acts on*.
   Full-stack LoRA is free to co-adapt a private channel between a low-block write and a
   high-block read. Restricting writes is what forbids that.

To measure the legibility gap directly rather than just controlling for it, add a **two-stage
arm**: train ≤12 to convergence, freeze it, then train >12 only. How much accuracy stage 2
recovers *is* the answer to "how much do the upper layers need to adapt."

One confound to control: LoRA r=16 on 13 blocks is ~40% of the parameters of the full-stack arm,
so "fewer layers" is confounded with "fewer parameters." Param-match by raising `r` on the
restricted arm.

## Infrastructure reality

- `vast-capabilities | jq '.instance.workspace_is_volume'` → **`false`**. Nothing on this box
  survives a recycle. Every session so far has started by re-downloading the model and rebuilding
  a 3.4 GB feature cache.
- **The phase-5 result JSONs were never committed**, despite §9 of that doc saying they were.
  `git log --all --diff-filter=A -- 'results/uf_plan*' 'results/uf_hybrid*'` returns nothing. All
  fork A/B histories are gone; those numbers now exist only as the tables in the markdown. Most
  costly: `uf_plan_samples.json` was explicitly banked as "committed, so the sweep can be rebuilt
  without resampling" — it is gone, and that is ~2.5 h of GPU sampling.
- Consequence for planning: `MCOEF=0` is *not* "one env var" on a fresh box. `uf_hybrid3.py`
  hard-requires five `/workspace` artifacts (lines 113–146), four of which must be regenerated:
  model download, feature cache (~30 min), plan sweep (~2.5 h), then the run itself. ≈ 8 GPU-hours.
- **Rule going forward: copy `/workspace/uf_*.json` into `results/` and push at the end of every
  run, not at the end of the session.** Sessions have died mid-run twice.
- Related robustness fix, unapplied: `uf_plan_sweep.py:146` only dumps samples after a whole
  split completes, so a mid-split crash loses everything.

## Tomorrow: causal efficacy of the probe direction vs depth

**`notes_steering_experiment.md` — start here next session.** No training required; it is a
measurement on the frozen base model, ~3 h GPU (half that if layers are subsampled).

Use μ_L as a steering vector during generation, and plot causal efficacy against depth on the same
axes as probe accuracy against depth. The read-depth negative above concerns the probe as a
*labeller*; this asks whether the probe direction is a causal *handle*, which is a different and
untested claim.

The prediction is already in the repo, which is what makes it a real test: phase 1 measured
`cos(μ, W_A − W_B) = −0.003` at the final layer, i.e. μ lies in the null space of the output map,
so steering at the top should do little despite maximal decodability. Pre-registered hypothesis:
**readable everywhere, steerable only in the middle.**

The one decision to make before spending GPU is the efficacy metric — scoring steered output with a
probe is circular and guaranteed positive. See the note's §"design problem"; the honest options are
a real judge model (best), a full cross-layer probe matrix (self-contained, visible circularity), or
non-probe proxies (weak but clean).

## Recommended next

1. **Read-depth differential on UF** — GT / L12 / L31 soft-DPO, then RewardBench + best-of-N +
   fresh-probe audit on all three. This is the experiment the repo was built to run, it needs no
   new code, and both outcomes are publishable. Note `uf_softdpo_lora` is also gone, so this is
   four training runs (including re-establishing the GT and L12 arms as live checkpoints), not two.
2. **Write-depth program** — `layers_to_transform` ≤12 / 20–31 / all, GT labels first, plus the
   two-stage arm and the param-matched control above.
3. **Park the hybrid line.** Not because it is bad work — the phase-4/5 failure taxonomy is
   genuinely good — but because it is a separate question (can activation-space training be
   stabilized at all?) and it is currently losing. If it is resumed, `MCOEF=0` is the right first
   move, with the caveat that it is not as decisive as phase 5 §7 claims: fork B's KL/token
   collapsed 0.071 → 0.002 between steps 50 and 100, which is a second reversal channel entirely
   independent of `MCOEF`. Pre-register the follow-up cell (`MCOEF=0 RL_KL=0.02`) before
   concluding anything about the margin half.

Two standing caveats that apply to everything above: **every cell in this project is single-seed**,
and the plan-state result in phase 5 §2 carries an uncontrolled length confound
(`corr(plan-z, expected length) = −0.358`, and L1 already reads 0.69 of the 0.775 best) that a
cheap offline re-analysis of the plan samples would resolve.

## Reading order

`README.md` → this file → `methods.md` (the objectives, and the related work they descend from:
Goodfire RLFR, Libon et al.) → `results_phase3.md` (the working method on the realistic dataset)
→ `results_phase1.md` §"Result 1" (why activation backprop fails, the one argument to internalise)
→ phases 2/4/5 as needed for specific runs.
