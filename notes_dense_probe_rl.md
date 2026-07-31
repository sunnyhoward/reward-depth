# Design note: dense frozen-read probe RL (potential-shaped token-level reward)

*2026-07-31, follows results_phase8. Status: DESIGN — pre-registered before any run.*

## Why this design exists

Phase-8 positional diagnosis: every direct activation objective in the archive constrained the
**completion-end reading state** — a state causally downstream of the answer that generation
never consults (arm A: 4x separation, zero behavioural change). The two mechanisms that DO
install are likelihood-shaped losses and policy gradients — both constrain the emission
channel. Meanwhile the one honest failure of guarded sampled RL on UF was **starvation**
(phase 7 §2: 32 sequence-level advantages/step cannot move an 8B), not hacking (§3) and not
signal (soft-DPO installs from the same probe).

This design combines the pieces so they point the same way for the first time:
activation-space *reward definition* + emission-channel *gradient* + per-token *density*.

## The objective

Sample rollouts on-policy. Run the FROZEN reference model over the sampled text (this forward
pass already exists for the KL/anchor terms) and capture residuals at the read layer L. Define
the potential at every answer-token position t:

    Phi(s_t) = probe posterior z of the frozen prefix state at t   (probe fit per-position, below)

Per-token shaped reward and the (telescoped) advantage:

    r_t = Phi(s_{t+1}) - Phi(s_t)          total = Phi(end) - Phi(prompt)  [same optimum as
    A_t = Phi(end) - Phi(s_t)               the sparse final-state reward; Ng et al. shaping]

A_t reads as "how much did the rest of the answer improve the verdict" — dense credit
assignment with no extra machinery, and Phi(prompt) is a built-in per-prompt baseline.

Guards: the v3 set unchanged — DPOP one-way anchor on chosen-side reference likelihood,
KL-in-reward, pessimism via the probe posterior (LCB z - k*s per position; s2 is per-state, so
pessimism is naturally token-level). Policy-gradient loss only; NO gradient ever flows through
the policy's own activations into the probe (frozen read = no forging channel by construction;
text-level reward hacking remains possible and is the guards' job).

## The probe (this is where the labeller changes)

Completion-end probes are off-distribution on prefixes — invalid as Phi. Fit the reward probe
on **per-position features**: every answer-token prefix state is a sample carrying its pair's
preference label. (User proposal 2026-07-31; the mean-pooled stage-A pilot is the cheap first
look at whether pooled/trajectory features change the decodability picture at all.)

## Pre-registered predictions

1. **UF is the decisive arm.** Same v3 recipe as the flat rloo300 baseline
   (`results/uf_probe_rl_rloo300_history.json`, reward .373 -> .367 over 300 steps), only the
   reward density changed. Prediction: reward moves off the floor within ~100 steps and
   acc_implicit exceeds noise by 300. If it does, starvation was the whole story; if it stays
   flat, the starvation diagnosis was wrong and something deeper blocks sampled RL on UF.
2. **styc mechanics arm** (explained-style completions, ~20 tokens; oracles every 25 steps):
   installs the labeller's preference in fewer steps than sequence-level RLOO. Watch the
   density asymmetry: style tokens are everywhere, correctness tokens are few, so per-token
   credit should install style-first even for a correctness-competent labeller (total is
   preserved by telescoping; the *dynamics* are style-leaning). Oracle-visible if true.
3. **No forging**: z of the frozen probe on fixed eval pairs (z_selfread analogue) stays flat.
4. **Ceiling inheritance**: on styc conflicts the installed preference cannot exceed the
   labeller's conflict competence (phase-7/8 dose-response) — this design fixes coupling and
   throughput, NOT perception. Any conflict result above the labeller's would indicate a bug.

## Non-goals / known limits

- Does not address the corr_e feature ceiling (phase 8 §6) — labeller quality still binds.
- cc is a poor testbed here (1-2 token answers: density degenerates to the sparse case).
- Compute: UF arm needs cache + per-position probe refit (~45 min) + 300 RL steps (~1 GPU-h).

## Implementation sketch

`uf_probe_rl.py` grows: PER_POS probe fit (subsample ~8 positions/pair to bound the cache),
SHAPED=1 reward path (capture per-position residuals during the existing ref-model pass;
compute Phi per position; A_t = Phi_end - Phi_t), per-token loss `-(A_t * logp_t)` replacing
the sequence-level RLOO advantage. Everything else (guards, evals, logging) unchanged.
