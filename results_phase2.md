# Phase 2 results — depth, anchors, attractors, and first UltraFeedback runs

*2026-07-20 · one intensive session on a fresh instance (RTX PRO 6000 Blackwell, torch 2.12,
transformers 5.14) · Qwen2.5-3B (A/B testbed), Qwen2.5-7B (hybrid archaeology),
Llama-3.1-Tulu-3-8B-SFT (UltraFeedback) · all run JSONs/logs/adapters in `results/`,
figures in `results/plots/` (fig1–fig5) · single seed per cell unless noted.*

## 0. Headlines

1. **Anchored RL-from-probe fully installs the preference at the top AND at the elbow** —
   the first working depth comparison. At matched config: elbow ≈ top on install strength,
   *lower* proxy inflation, shallower rejected-side trench, off-menu contained.
2. **Two attractors explain every naive failure**: the *letter-policy* attractor (RLOO
   absorbing states + DPO's slow letter transition) and *likelihood displacement*. The DPOP
   anchor fixes both; nothing else we tried does.
3. **Self-read backprop from the probe failed in every configuration** (see §4) — including
   at mid-depth, at 7B, and with co-adaptation. The verdict of `methods.md` §2.2 generalizes.
4. **UltraFeedback**: preference decodability plateaus at **L11/32**, accuracy 0.803 ≈ what a
   400-step DPO run installs (0.805). Anchored RLOO from the frozen L11 probe reaches
   0.571 ± 0.026 held-out implicit-reward accuracy at a small rollout budget with **zero
   displacement** (both Δlp positive throughout, vs DPO's −9/−30 trench).
5. **Phase-1 numbers do not transfer across environments/settings uncritically**: the committed
   DPO config (lr 5e-5, 300 steps) letter-locks on this stack — it needs ~600 steps (or lr 1e-4)
   to cross the letter transition. Phase-1's rl arm was never reproduced as documented; its
   lost notebook evidently included a DPOP-style anchor (§4).

## 1. The letter-policy attractor (fig1, fig2)

All the recurring constants of early phase-2 runs (`ab 0.477`, flip `0.57`/`0.43`) turned out
to be **letter base rates**: policies collapse to "always answer A" (or B). Verified directly —
saved adapters answer one letter on 100% of eval questions; per-question logit margins are
bimodal ±11 nats with base-model confidence irrelevant (AUC 0.62).

- **Unanchored RLOO** (lr 1e-4): enters the attractor by step 25 at *every* attach depth
  (L21/L25/L35 identical, bit-identical by-type partitions — seed-determined) and freezes:
  with 1-token answers, a committed question yields k identical samples → zero RLOO advantage
  forever. At lr 3e-4 it instead collapses off-menu (Δlp chosen −114 nats).
- **DPO** oscillates between the two letter policies (fracA swinging 0↔1) while pouring its
  margin into the multi-token free-format pairs (+43 nats abs margin, 97% flipped) and leaving
  the ab pairs at ~zero net margin — then crosses the letter transition late (~step 375 at
  lr 5e-5; ~step 220 at 1e-4). The transition *is* the oscillation. Verbatim-notebook rerun
  reproduces this checkpoint-for-checkpoint → not a driver bug; a knife-edge dynamic
  (`notes_gradient_equivalence.md`'s cancellation argument, realized).
- **Mitigation**: log `fracA` at every checkpoint (now standard in all drivers/notebook).

## 2. The DPOP anchor is load-bearing (fig1b)

Anchor = hinge `λ·relu(log π_ref(y_c|x) − log π_θ(y_c|x))` (λ=1) — a one-way floor: never like
the chosen completion less than the reference does. With it, RLOO-from-probe escapes the letter
attractor (~step 125–150 at L35, ~175–200 at L21) and reaches **full targeted flip** with
off-menu 0.00 and the chosen side held +4–5 nats *above* reference. Without it, no RL
configuration installs anything. It also supplies the per-question restoring force that
un-freezes RLOO's absorbing states (committed-wrong-letter questions sit below the floor).

Variants tested in the hybrid archaeology (§4): *symmetric* (floor both sides) makes menu flips
mathematically impossible for 1-token answers (mass conservation: flooring π_right ≥ ref caps
π_wrong); *on-menu-mass* (floor log(π_A+π_B)) permits the flip while blocking off-menu leak.

## 3. Depth comparison at the working point (fig3)

`dpo` (lr 1e-4) vs `rl_top` (anchored RLOO, probe@final) vs `rl_elbow` (same @L21), 300 steps,
single seed, notebook-validated (`probe_vs_dpo.out.ipynb` reproduces all three exactly):

| | dpo | rl_top | rl_elbow |
|---|---|---|---|
| ab / know_ab flip | 0.98 / 0.94 | 1.00 / 0.96 | 0.98 / 0.96 |
| ood_digits / ood_sum flip | 0.79 / 0.67 | 1.00 / 0.81 | 1.00 / 0.69 |
| free flip / off-menu | 0.69 / 0.29 | 1.00 / 0.00 | 0.86 / 0.00 |
| near-miss menu (2-choice; base 0.53≈chance) | 0.51 | 0.40 | **0.23** |
| easy addition (base 1.0) | 0.88 | 0.98 | 1.00 |
| know free correct / diversity (base 0.84/0.98) | **0.02 / 0.22** | 0.84 / 0.98 | 0.76 / 0.94 |
| dpoR inflation / Δlp rejected | +25 / −21 | +12.3 / −6.9 | **+8.9 / −4.3** |

Readings: probe arms ≥ DPO on every install axis; **DPO's collateral damage is concentrated in
free-form knowledge** (degenerate repeated wrong answers — div 0.22), which both probe arms
avoid; the elbow arm shows the *lowest* proxy inflation and shallowest trench (Occam margins)
but transfers wrongness hardest to the untrained 2-choice menu (below chance = deliberate
wrong-picking, not damage). Generalization at this working point is *broad* for all arms —
phase-1's "narrow RL generalization" did not reproduce (that run also had the off-menu failure
ours lacks; attribution to the anchor vs environment is open).

## 4. Hybrid archaeology (backprop ≤L + REINFORCE >L) — 7 runs

Goal: replicate the preface-repo `04_compare_L24` figure (7B, deep_rl: AB→0.03@150, easy
retained, know-transfer weak). Outcome: **mechanism recovered, collateral profile not**.

| run | config | result |
|---|---|---|
| 1–2 | 3B, L21 / L31, RLOO(policy-read) | letter-lock; L21 also lazy global wrongness (easy→0 via operand-echo) |
| 3 | 7B L24, RLOO policy-read | **margin half forges the pristine meter to 0.99 in 25 steps**; behavior at base |
| 4 | 7B L24, RLOO base-read | free-form wrongness installs; menus letter-lock; meter still forged |
| 5 | 7B L24, **candidate-based REINFORCE** + chosen anchor | **full per-question flip (0.96@100, 1.00@200)** — the key ingredient; but off-menu drift + easy→0 late |
| 6 | + symmetric anchor | flip impossible (mass conservation, §2) |
| 7 | + on-menu-mass anchor | full flip (late, ~275); collateral still unmatched |

Conclusions: (i) the original figure's "REINFORCE" was **candidate-based** (score-function over
the pair's two completions — the exact-expectation construction of `methods.md` §3), not
free-sampling RLOO — with free sampling the hybrid never installs; (ii) the backprop half
forges its layer wherever it runs (3B and 7B, filter on) and contributes no measured behavior;
(iii) the figure's clean collateral (easy ~0.8, off-menu ~0, know 0.37) was not reproduced by
any anchor variant — leading residual hypothesis: **early stopping** (~150 steps; run 5's
flip was complete at ~125 before its collateral accrued). Decisive test (unrun): run-5 config
at 150 steps. Runner-up hypothesis: candidate rewards refreshed by the co-adapting filter head
(ours were static).

## 5. UltraFeedback (fig4, fig5)

**Decodability**: linear probes on frozen-SFT last-token residuals: 0.70 at L0–2 (surface
features — length/format are linearly visible immediately), plateau **0.803 from L11/32**,
flat to L31. The plateau equals the DPO-trained policy's held-out implicit-reward accuracy
(0.805, 400-step LoRA DPO baseline: `uf/uf_dpo_train.py`, adapter+merged saved).

**DPO baseline**: acc 0.5→0.805; classic both-sides-down displacement (Δlp −9/−30 nats);
raw-logprob ranking is length-confounded (base raw acc 0.403) — implicit (ref-corrected)
accuracy is the honest metric throughout.

**Anchored RLOO from the frozen L11 probe** (300 steps × 8 rollouts, pess 0.5, KL 0.03,
anchor 1.0, MAX_NEW=200): big-N (350-pair) evals — ckpt100 0.551, ckpt200 0.511, final
**0.571 ± 0.026** (p≈0.004), margin +1.31 nats, **Δlp +5.7/+4.4** (no displacement at any
checkpoint). Real but weak; no pathology.

**Diagnosis** (`uf_spread_diag.json`): *not* signal starvation — within-prompt reward spread
(0.39 z-units) exceeds the chosen-vs-rejected gap (0.33); pessimism shifts levels, not spreads
(RLOO-invariant, as on A/B). The poison: **44% of rollouts hit the 200-token cap** and the
probe reads a truncated last-token state it was never fit on — near half the reward signal was
truncation noise. Fix (v2 run, in flight): MAX_NEW=512, truncation-masked advantages, k=8 ×
batch=4.

## 6. Corrections to phase-1 readings

- Phase-1's `probe` arm early path (~0.55 by step 25) and `probe+filter` plateau (~0.55) are
  letter-base-rate-shaped; without a fracA log they cannot be distinguished from letter
  policies retroactively. Treat those trajectories as suspect of the same attractor.
- Phase-1's rl-arm row (narrow generalization, off-menu 0.47) comes from a lost notebook whose
  exact config (lr, anchor) is unrecoverable; it did not reproduce under any tried setting.
- The "small tests" commit flipped `sampled_rl_step`'s reward sign after the phase-1 runs;
  the committed pre-fix code rewards the *right* answer.
- `mcq` is 2-choice (true sum vs near-miss), base ≈ chance on 3B — an install-breadth metric
  here, not a capability metric.

## 7. Open items

150-step run-5 rerun (early-stopping test) · arm 9 (Libon-config on-policy self-read on the
precision testbed) · decision-position probes + LoRA≤L (the one untested self-read corner) ·
UF v2 result → judge-based rollout comparison (paired, both-order, length-controlled, two
judges) + cross-endorsement matrix across depths · seeds for fig3 · `uf_hybrid.py` first run.
