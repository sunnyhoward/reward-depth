# Phase 9 — The UF port: the labeller's discrimination range, not credit density, is what blocks sampled RL

*2026-08-03. One session, fresh box (RTX PRO 6000 Blackwell 96GB, `workspace_is_volume` false;
repo restored from the HF bundle). Caches rebuilt from scratch in ONE sweep (uf_meanpool_sweep.py
now emits pooled + last-token from the same forward passes); the pooled Stage-A curve reproduces
the banked phase-7 result exactly (L*=23, .819 vs .818). Scripts new/extended this session:
`uf/uf_tail_probe.py`, `uf/uf_reward_spread_diag.py`, `uf/uf_probe_rl.py` (RL_MODE=
shaped|pooled_margin|hybrid, UF_READ=mean, REWARD_FORM=z, TRUNC_PEN/REPLAY_N/CKPT_EVERY guards).
Standing caveat: every training cell single-seed; implicit-acc evals n=64 (SE ~.06).*

## 0. Where this session started

NEXT.md priority 1: port the two phase-8 styc winners (shaped dense probe-RL; pooled_margin) to
UltraFeedback/Tulu-8B against the flat rloo300 baseline, with the phase-8 mandatory guard set
(periodic checkpoints, REPLAY_N>0, deferral guard). Pre-registered predictions in
notes_dense_probe_rl.md: if starvation was the whole story, dense credit moves the reward off
the floor within ~100 steps.

## 1. The tail measurement: translation-tail blindness is a FITTING artifact, not a wall
(`results/runs/uf_tail_probe.json`)

Protocol: 2-fold cross-fit of the last-token L12 probe on the 3,000 probe-fit pairs -> honest
out-of-sample soft labels. 22.7% side against the dataset (the in-sample audit said 13.6%);
tail = confidently-backwards (p<.25) = 236 pairs, translation-enriched 2.1x (.186 vs .089 by a
keyword+non-ASCII tag).

| | natural fit, tail acc | tail-upweighted x5 (tail/test) | tail-only 2-fold xfit |
|---|---|---|---|
| last-token L12 | .11 | .66 / .775 | .90 |
| last-token L16 | .17 | .73 / .771 | **.93** |
| pooled L12 | .49 | **.91 / .788** | .79 |
| pooled L23 | .47 | .87 / .766 | .78 |

- **Not a wall.** The tail preference is linearly decodable at ~.9 out-of-sample within the tail
  — even from LAST-TOKEN features. At 8B the perception is there (consistent with phase-8
  scaling: 7B corr_e .991 last-token); the natural fit just never buys it (7.9% of weight).
  Real-data confirmation of phase-8 §1: the deficit is in the diet/fit, not the features.
- Pooling softens the natural fit's blindness (~.15 -> ~.5) but does not dissolve it — unlike
  styc corr_e. The styc-XL "does corr_e rise with scale" question is answered for UF-at-8B:
  it already rose; what's missing is fit weight.
- **Actionable**: pooled + x5 upweight at L12 = tail .91 with test .788 — a strictly better
  labeller than the natural one, for free. (Selection caveat: upweighted tail acc is in-sample
  on 236 pairs; the tail-only cross-fit is the honest decodability bound, and it is ~.8-.9.)

## 2. Arm 1, shaped300: dense credit does NOT rescue sampled RL
(`results/uf_probe_rl_shaped300_history.json`)

Per the spec: pooled probe at L*=23 (acc .819), frozen prefix-valid running-mean Phi, per-token
A_t = (R - b_LOO) + (Phi_end - Phi_t) on the re-rendered completion, v3 guards + TRUNC_PEN=.25
+ REPLAY_N=64, checkpoints every 50 (all six banked). 300 steps, lr 5e-5, BATCH 2, K 4.

| step | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| shaped acc_implicit | .625 | .562 | .578 | .625 | .609 | .531 |
| baseline rloo300 | .531 | .453 | .578 | .562 | .500 | .656 |
| shaped dlp margin | +0.72 | +1.43 | +1.90 | +1.87 | +2.73 | +1.45 |

Reward by 50-step means: .375 -> .389 -> .391 -> .397 -> ~.40 (baseline: flat .373-.389).

- **Pre-registered verdict: starvation was NOT the whole story.** The reward crept, never
  lifted; acc sat above baseline at 5/6 checkpoints but within noise, and the final checkpoint
  regressed. The margin trend (monotone to +2.7 at 250) is the one real-looking effect.
- No Goodhart: lengths flat (~100 tok), capped fraction not trending, generations coherent
  throughout, chosen-side dlp POSITIVE (+4 to +7 — opposite of GT-DPO's collateral). The guard
  set may deserve some credit; either way the failure mode is inertness, not exploitation.

## 3. Arm 2, margin300: the pooled activation margin is a behavioural NULL on UF
(`results/uf_probe_rl_margin300_history.json`)

styc phase-8 winner ported: teacher-forced pairs, POLICY pooled activations at L*=23, lag-1
adaptive mean-diff margin, M0=4, DPOP + replay floors, lr 1e-4. 300 steps.

- The meter is driven to completion and past it: proj 3.5 (natural) -> dip 1.4 (lag-1 direction
  stabilising) -> 6.1 > M0 by step 250. Margin loss saturated at zero.
- Behaviour: acc_implicit .48-.53 (chance) at every checkpoint; both-side likelihood inflation
  (+8-10 nats each, gap ~1-2) under the floors. No styc-style install at any point.
- With phase-8 §13 this completes a clean 2x2 on real-vs-oracle data: on styc the pooled
  direction (ce-we) IS the dominance-relevant correctness feature and optimizing it installs;
  on UF the natural pooled direction at L*=23 is the style-legible preference direction (§1's
  tail result) and driving it is behaviourally inert. **The margin mechanism works exactly when
  the direction is causally load-bearing; pooling fixed forging, not direction quality.**

## 4. The spread diagnostic: on-policy text SATURATES the labeller's discrimination range
(`uf/uf_reward_spread_diag.py`, `results/runs/uf_reward_spread_diag.json`)

Fresh pooled reads at L*=23, protocol identical to RL-time. 100 held-out pairs; 32 prompts x
K=4 base-policy rollouts.

| quantity | value |
|---|---|
| dataset pair \|z-diff\| (probe fit target; .82 acc) | 4.77 |
| rollout within-prompt z std (what RLOO feeds on) | **1.78** |
| probe per-read posterior s | ~7 |
| within-prompt spread after ndtr squash | 0.074 (squash ratio **.042**) |
| rollout reward mean vs dataset chosen | **.474 vs .419** |

Three stacked findings, ordered by depth:

1. **Squash**: the CDF + sqrt(1+s2) pipeline discards ~96% of within-prompt z signal where the
   rollouts live. (REWARD_FORM=z recovers 24x — mechanical fix, now wired.)
2. **Resolution**: within-prompt differences (~1.8z) are far below the pair gap (~4.8z) that
   yields .82 accuracy — within-prompt rollout rankings are near coin flips.
3. **Headroom (the deep one)**: base-policy rollouts already read ABOVE the dataset chosen side.
   The policy starts at the top of the probe's discrimination range; there is almost no gradient
   for sampling to harvest. **This unifies the whole UF record: soft-DPO works because it
   consumes the dataset-pair contrast directly — the only place the probe's information lives —
   while every sampled-RL variant starves regardless of credit density.** The phase-7
   "starvation" diagnosis and this session's density null are both downstream of it.

Consequence for the program: fixing sampled RL from this labeller needs a labeller that
discriminates ABOVE chosen-quality text (harder-contrast fits, tail-upweighting per §1,
best-of-N distillation targets), not better credit plumbing.

## 5. Arm 3, hybrid300 (shaped + MCOEF*margin, REWARD_FORM=z, K=8) — running at close

Launched with the diagnostic-driven adjustments (raw-z reward, K=8, KLR .1, TRUNC_PEN 1.5).
Rationale for running despite §2/§3: the z-reward recovers the squashed 24x, K=8 halves LOO
noise, and the margin half is representationally active while behaviourally inert — the one
untested interaction. Expectation per §4 is honest scepticism: the headroom problem is not
addressed by any of these. Result: see history JSON / addendum below.

## 6. Standing conclusions after this session

1. **At 8B the features are not the bottleneck anywhere we can measure** — the translation tail
   is ~.9 decodable even last-token (§1). The styc corr_e wall does not exist here.
2. **The labeller's FIT is the bottleneck twice over**: natural diets don't buy the tail (§1),
   and nothing in the natural probe discriminates above chosen-quality on-policy text (§4).
3. **Credit density and activation coupling are both exonerated-and-irrelevant on UF**: shaped
   trains cleanly and moves nothing (§2); the margin meter completes and moves nothing (§3).
   The two phase-8 mechanisms transfer as mechanisms — what failed to transfer is the styc
   property that the optimized quantity was causally load-bearing.
4. Program implication: the next unit of progress is a better LABELLER (tail-upweighted pooled
   fit §1; harder-contrast/off-policy-calibrated fits §4), evaluated first as soft-DPO source
   (the known-working consumer), then as RL reward only if §4's headroom is shown fixed.

## 7. Infrastructure notes

- Dual-cache extraction (pooled + last-token, one sweep) is the new default; a wiped box pays
  one extraction (~25 min at MP_BS=16 on this GPU).
- Both 300-step arms ran CONCURRENTLY on the 96GB card (shaped ~39GB) — sequential was wasting
  half the box. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on the second process.
- Checkpoints every 50 steps: all six shaped adapters banked (phase-8's lost-peak mistake not
  repeated). Results committed per-run, not per-session.
