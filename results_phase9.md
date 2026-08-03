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

## 5. Arm 3, hybrid300 (shaped + MCOEF*margin, REWARD_FORM=z, K=8) — killed flat at ~110

Launched with the diagnostic-driven adjustments (raw-z reward recovers the squashed 24x, K=8
halves LOO noise, margin half included as the one untested interaction). acc .500/.4375 @50/100,
margin ~0. Killed at ~step 110 (user call) — the right call: §4's headroom problem is upstream
of all three adjustments, and §9 below confirms it was the binding constraint. Partial history
banked (`results/uf_probe_rl_hybrid300_history.json`). With §2 and §3 this makes THREE
independent flat arms from the dataset-fit probe.

## 6. The margin-gen check: no hidden style install (`results/runs/uf_margin_gen_check.json`)

User challenge to §3: "behavioural null" was implicit-ranking + eyeball; generation style could
have moved without the ranking moving. Measured (64 held-out prompts, greedy, margin adapter vs
base): generations DO differ (14% identical), but the pooled probe scores the margin arm's own
generations LOWER than base (z 3.15 -> 2.41, dz -0.74 +/- 0.40, 34% moved up), lengths flat
(146 -> 148 tok). The §3 null holds on all three measures: ranking, likelihoods, generation.
The margin objective was satisfied by text-conditional representational shifts invisible to
generation — the phase-1 "cheapest satisfying edit" argument, surviving pooling when the
direction is not a causal handle.

## 7. Pooled-direction steering: the first non-null steering result in the project
(`results/runs/uf_steer_pooled.json`, gens alongside)

User proposal: add the pooled ("meaned") probe direction to the residual stream during
generation — the phase-6 experiment, rerun with the pooled direction (phase 6, last-token
directions: causally inert at every depth, 94% judge-null). 8 layers x alpha {.03,.1,.3},
ActAdd-style dm vector, 64 prompts, greedy.

| steer L (a=0.3) | text dz (clean read) | 7B-judge win vs base | kl/tok |
|---|---|---|---|
| L0 | +0.29 | .516 | ~0 |
| **L8** | **+1.48** | **.586** (a=.1: .594, margin +2.0) | .029 |
| L12 | +0.67 | .555 | .032 |
| L16 | +0.58 | .594 | .042 |
| L20-23 | +0.60 | .50-.57 | .04-.05 |
| L31 | +0.19 | .547 | .010 |

Text-level movement along the preference direction (z read on a CLEAN forward — not the vector
echoing), mid-layer peaked, dose-responsive, judge-visible (.55-.59 win band where the meter
moves; dead at L0; margins track dz). Single cells ~1.5 SE (n=64) — the coherent pattern
carries the claim. **"Readable everywhere, steerable in the middle" is now supported — but only
for the pooled direction.** Pooling changed which directions are causal handles, not just reads.
Qualitatively: steered gens acquire more structured/explicit formatting (headers, enumerations).

## 8. The on-policy labeller loop (user proposal): sample -> judge -> refit. GATE PASSED
(`uf/uf_onpolicy_{sample,judge,probe}.py`, `results/runs/uf_onpolicy_probe.json`)

The §4 headroom fix: manufacture contrast at the policy's own quality level — the human-labelling
step of classic RLHF, done with a local judge. K=4 rollouts/prompt at T=1.1 from the frozen SFT
(2000 train + 64 gate prompts); Qwen2.5-32B judge, correctness-first rubric that explicitly
discounts length/confidence, position-swapped double judging, consistent verdicts only (68%
kept, 2557 pairs); pooled probe refit on judge-labelled pairs.

Gate, on held-out prompts' SAME-PROMPT rollout pairs (n=239 — the comparison RL actually makes):

| | rollout pairs | dataset pairs (400) |
|---|---|---|
| old probe (dataset-fit, L23) | **.590** | .815 |
| new probe (judge-fit, L11) | **.787** | .780 |

- Headroom confirmed out-of-sample: the dataset probe is near-chance on exactly the comparison
  sampled RL feeds on.
- The new probe learns on-policy discrimination WITHOUT trading away the dataset preference.
- Judge length bias: winner LONGER in only .36 of pairs; probe corr(z, len_diff) .04 @L11. The
  rubric + consistency filter beat the classic LLM-judge length bias.
- Linear vs small antisymmetric MLP (user request): linear .787, MLP .715 at the same layer —
  linear kept (styc §4 precedent holds on judge labels).
- The on-policy preference is most decodable at L11 — the mid-network elbow, not the top.
- Caveat: the gate metric is judge-agreement, which favours the judge-fit probe by construction;
  (§4's label-free saturation result and the dataset-retention column are the independent checks.)

## 9. onpol300: the on-policy reward INSTALLS — first working sampled RL on UF
(`results/uf_probe_rl_onpol300_history.json`, checkpoints every 50 banked)

Same shaped recipe as §2, only the reward probe changed (PROBE_SRC=onpolicy, L11, REWARD_FORM=z,
K=8, guards unchanged):

| step | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| acc_implicit | .500 | **.641** | .609 | .547 | .531 | .469 |
| dlp margin | -0.25 | **+2.98** | +2.89 | +2.43 | +3.01 | +0.88 |

Rise-and-decay: real install peaking ~100-150 (margin +3.0 — more than shaped300 reached at any
point, in a third of the steps), then over-optimization decay — the styc shaped dynamic (peak
~125) reproduced on real data. Lengths/truncation flat, generations coherent throughout (no
preamble exploit; guards on). Checkpoints 100/150 hold the peak policy this time.

**The causal chain closes**: squash (fixed, §4) -> density (exonerated, §2) -> headroom (the
binding constraint, §4) -> on-policy labeller (fix, §8) -> sampled RL works (§9). Note
acc_implicit measures the DATASET preference while the reward optimizes the JUDGE preference
(aligned .78): the peak .641 install through a partially-aligned reward, and the decay under
continued optimization, both need the queued follow-ups (judge-eval of checkpoints; earlier
stopping; iterated re-judging).

## 10. Standing conclusions after this session

1. **At 8B the features are not the bottleneck anywhere we can measure** — the translation tail
   is ~.9 decodable even last-token (§1). The styc corr_e wall does not exist here.
2. **The labeller's FIT was the bottleneck twice over — and both halves are now demonstrated
   fixable**: natural diets don't buy the tail (§1, fix measured: upweighting); the dataset-fit
   probe cannot discriminate above chosen-quality on-policy text (§4, .590 out-of-sample §8),
   and the on-policy judge refit fixes it (.787) — after which sampled RL installs (§9).
3. **Credit density and activation coupling are exonerated-and-insufficient with a saturated
   reward** (§2, §3, §5 — three flat arms), and §9 shows density + the guard set DO carry an
   install once the reward has resolution. The margin mechanism remains direction-limited: no
   install (§3), no hidden style change (§6).
4. **Pooling is the session's through-line, twice over**: it made the labeller fixable (§1, §8
   both pooled fits) and it made the probe direction a (weak) causal handle for the first time
   (§7) — the phase-6 steering null was a property of the last-token direction, not of probes.
5. **Both Goodhart lessons generalize to real data**: peak-then-decay under continued
   optimization (§9, styc §11), and meters completing while behaviour stands still (§3/§6,
   phase-7 §3). Early stopping via frequent checkpoints is not optional.

## 11. Infrastructure notes

- Dual-cache extraction (pooled + last-token, one sweep) is the new default; a wiped box pays
  one extraction (~25 min at MP_BS=16 on this GPU).
- Both 300-step arms ran CONCURRENTLY on the 96GB card (shaped ~39GB) — sequential was wasting
  half the box. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on the second process.
- Checkpoints every 50 steps: all six shaped adapters banked (phase-8's lost-peak mistake not
  repeated). Results committed per-run, not per-session.
