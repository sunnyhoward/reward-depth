# Phase 5 — UF hybrid port: the gate passes, the margin half convicts itself

*2026-07-22. Fresh instance (all prior `/workspace` artifacts wiped and rebuilt). Code:
`uf/uf_label_depth_gate.py`, `uf/uf_plan_sweep.py` (unchanged), `uf/uf_hybrid2.py` (v1, run to
step ~135), `uf/uf_hybrid3.py` (v2, both forks), plus new eval scripts `uf/uf_bestofn_eval.py`,
`uf/uf_fresh_probe_audit.py`, `uf/uf_rewardbench_eval.py` (written, not yet run). Run records:
`results/uf_depth_gate.json`, `results/uf_plan_curve.json`, `results/uf_plan_samples.json`,
`results/uf_hybrid2_history.json`, `results/uf_hybrid3_{exactj,rloo}_history.json`.*

## Headline

**A plan state exists on UltraFeedback** (prompt-end quality-of-upcoming-response linearly
readable at 0.775, L15) — the phase-4 gate passes on a realistic dataset. **But the two-head
hybrid does not survive the port in either of two redesigned forms.** The failures are clean and
convergent: in both forks the *margin half's* representational rewrites generate likelihood
collateral, the anchor that catches the collateral out-muscles the (weak or unmoored) install
force, and the run converges to noise or a tug-of-war around baseline. The adaptive-head
machinery itself — the v2 redesign — **worked**: stationary-threshold labels + windowed buffer +
EMA damping kept refit val-acc 0.65–0.94 for 300 steps where v1 collapsed to chance by step 100,
and measurably absorbed a large forged component (pristine-meter z 7.1 → 1.0). The strongest
positive signal of the day: **anchored on-policy RLOO doubled held-out generation quality by step
50** (judge-z 1.63 → 3.43) with a textbook likelihood profile (+0.8/−4.5) — before the
margin-half dynamics dragged it back. The decisive next experiment is the one-variable ablation
`MCOEF=0` (RLOO-only): if it holds the step-50 gains, the coupling-axis story inverts — the
probe's value on realistic data is as *reward*, not as *gradient target*.

## 0. Rebuild + Stage-A reproduction (fresh instance)

Same funnel, seed, protocol as phase 3: 15,283 length-matched pairs, probe-train 3,000, eval 400.
Per-layer sweep reproduces the phase-3 curve: **L\* = 12** (acc 0.791; plateau max 0.799 @ L16;
top layer 0.770). The ELBO/evidence proxy also peaks mid-band (−1025/−1042/−1056 at L14/13/12 vs
−1076 at L31, ~−1180 early) — the Bayesian-Occam layer selection independently prefers the
L12–19 band over the top.

## 1. Probe-content depth gate (`uf_label_depth_gate.py`) — labels distilled at different depths genuinely differ

Pre-registration diagnostic for the (not-yet-run) soft-DPO depth differential, answering "if the
probe layer only decides the labels, do L12 and L31 labels even differ?":

| pair | corr(z) tr/te | hard agree | mean \|Δp\| | \|Δp\|>0.2 |
|---|---|---|---|---|
| L12 vs L31 | 0.848 / 0.839 | 0.911 | 0.086 | 0.096 |
| L12 vs L16 | 0.902 / 0.887 | 0.929 | 0.069 | 0.046 |
| L16 vs L31 | 0.891 / 0.884 | 0.923 | 0.077 | 0.065 |

- Disagreements concentrate on noisy pairs (dataset score-margin 1–1.5: 12.7% vs ≥2.5: 7.0%).
- On held-out disagreements the earlier probe sides with the dataset label more often (L12 0.558
  vs L31 0.442, n=52; L16 beats both: 0.606/0.644). L16 is also the acc max (0.799).
- Length alignment rises with depth: corr(z, len_diff) ≈ +0.006 (L12), +0.103 (L16), +0.066 (L31);
  IPW-weighted −0.044 / +0.047 / +0.015.
- **Failed prediction, noted honestly:** the top probe is *not* more confident (frac soft 0.49 vs
  0.47; frac conf 0.148 vs 0.150). Divergence is in *which* pairs flip, not confidence mass.

Gate verdict: the content-depth differential is live (8.9% hard-label flips), with expected small
effect sizes in-domain — the informative comparisons are Goodhart profile and OOD transfer.

## 2. Plan-state decodability sweep (`uf_plan_sweep.py`) — the phase-4 gate passes on UF

2,400 prompts × K=2 sampled completions (frozen SFT, 512 tokens), judged by the L12 probe
(re-rendered eos read); per-prompt mean judge-z, median split; per-layer prompt-end probes:

- **Best L15, acc 0.775** (chance 0.5; K=2 judge sampling noise bounds the ceiling below 1.0).
  Co-located with the completion-end band (L12–16).
- Caveats: L1 already reads 0.69 (prompt-surface signal — topic/format predicts expected quality),
  so consolidation adds only ~+0.08 (toy analog: 0.69 → 0.99). And corr(plan-z, expected length)
  = **−0.358**: the label partly encodes "prompts inviting short answers score higher" — a
  prompt-type confound.

## 3. v1 hybrid (`uf_hybrid2.py`, killed at step ~135) — all three phase-4 failure modes at once

Faithful-looking port: free-rollout RLOO from the frozen judge above L15, plan-reader margin below,
buffer refit with **per-refit-median relative labels**, no likelihood anchor ("the margin half
plays that role" — falsified below).

- **Head collapse:** fresh-refit val acc 0.86→0.47 by step 100; rotation 60–70° *per refit*
  (thrash, not tracking). Root cause: per-refit median split is self-referential — the threshold
  moves with the policy, so near-identical states get opposite labels across epochs.
- **Symmetric likelihood displacement:** Δlp chosen/rejected −21/−19 nats by step 100,
  acc_implicit pinned at 0.500. The toy hybrid never needed an anchor because its exact-J over a
  2-item menu with all-positive rewards **conserves mass onto the menu by construction**; the port
  swapped exact-J for open-ended RLOO and silently lost that property. A plan-space (prompt-end)
  margin cannot anchor completion-space mass.
- On-policy judge score rose mildly (+2.5→+4.4) then broke (−1.6 at step 130).

## 4. v2 redesign (`uf_hybrid3.py`) — fixes mapped to defects

| defect (v1) | fix (v2) | source |
|---|---|---|
| self-referential refit labels | label vs **fixed** plan-sweep base median (THR=+2.228), confidence-weighted \|z−THR\| | Libon: refit labels must be externally/stationarily grounded |
| ever-growing conflicting buffer | window = seed(512, ½-weight) + last 8 refit batches | phase-4 open item |
| 24×2 noisy refits | 48×4 every 20 steps + minority≥32 guard | — |
| per-refit thrash | EMA 0.7 on (μ, ρ); rot split per-refit vs cumulative | — |
| no forging detector | pristine seed head on held-out prompt states each eval | phase-4 pristine meter |
| no on-policy eval | held-out generation eval: judge-z, length, KL/token | RLFR framing |
| displacement unanchored | fork A: mass hinge; fork B: DPOP(1.0) + KL 0.05 | phase 2 |

Upper-half fork: **A `exactj`** — pair-restricted exact-J on the dataset pair, rewards from cached
frozen features (teacher-forced, no sampling), π_rel = softmax(TAU·lp/n), mass anchor
relu(logsumexp_ref − logsumexp). **B `rloo`** — v1's RLOO + DPOP + KL. (Engineering note: all
grad-carrying likelihood passes must be micro-batched per pair — a batched 16-seq×1k-token grad
forward stores ~90 GB of activations and OOMs a 96 GB card.)

## 5. Fork A (exact-J, 300 steps) — guard-rails work, install force inert, two design flaws measured

- **π_rel(chosen) never moved**: 0.608 → 0.611 over 300 steps; jloss flat −0.39. The relative
  force is under-scaled ~35×: ∂/∂lp ≈ TAU/n·π(1−π)·Δr ≈ 3/250·0.25·0.25 ≈ 7e-4/nat vs
  soft-DPO's β·σ′ ≈ 0.025/nat. **Fix for any rerun: TAU ≈ 25–30.**
- **The mass hinge, not J, drove the run.** It spiked to 6.3 nats mean over steps 1–50 — i.e. the
  *margin half's* rewrites collapsed total pair likelihood by ~6 nats immediately — then pushed
  mass back **through the logsumexp channel, which concentrates gradient on the side that already
  has higher raw likelihood: the rejected side in 59% of pairs (base raw acc 0.409)**. Result:
  both sides inflated (+5–6 nats), rejected leading → acc_implicit *below* chance mid-run (0.36 @
  150), 0.44 at end. **Fix: per-side floors relu(ref_c−lp_c)+relu(ref_r−lp_r), never logsumexp.**
- **Forging race, watched live:** pristine-z −0.18 → 5.25 (@50) → 7.14 (@100) → 0.99 (@200) →
  3.82 (@300); head rotation 16°→89° cumulative with val acc 0.65–0.94 throughout. The grounded
  refits absorb the forged component (v1 could not), but the equilibrium is unstable.
- Held-out generation quality drifted to half baseline by the end (zjudge 1.63 → 0.80). A new
  localization fact en route: at step 100 the damage was **train-prompt-local** (refit rollouts on
  train prompts at frac_pos 0.27 while held-out zjudge was still 1.77) before globalizing.

## 6. Fork B (anchored RLOO, died with the session at step 150/300) — the day's best signal, then the same disease

| step | acc_impl | Δlp c/r | zjudge_ho | len_ho | KL/tok | pristine_z |
|---|---|---|---|---|---|---|
| 0 | — | 0 / 0 | 1.63 | 203 | 0 | −0.18 |
| 50 | 0.578 | **+0.8 / −4.5** | **3.43** | 279 | 0.071 | 5.03 |
| 100 | 0.500 | +4.7 / +4.3 | 1.35 | 185 | 0.002 | 5.10 |
| 150 | 0.547 | +3.8 / +3.0 | 1.77 | 180 | 0.027 | 4.20 |

- **Step 50 is the healthiest checkpoint in the project's UF history**: held-out on-policy quality
  doubled, DPOP held the chosen floor exactly where v1 displaced, dataset preference emerged free
  (0.578) from a pure rollout objective.
- **Steps 50–100 reversed it**: both dlp inflated symmetrically (+4.7/+4.3) — B has no mass hinge,
  so the suspected channel is DPOP itself: margin-half rewrites sink pair likelihoods → DPOP
  restores the chosen side → the correction generalizes to the stylistically-twin rejected side.
  Meanwhile KL pulled the rollout distribution back toward base (KL/tok 0.071 → 0.002) faster
  than 32 sequence-level advantages/step could defend the quality gain.
- Steps 100–150: partial recovery (1.77, chosen re-leading) — a tug-of-war equilibrium, not a
  collapse. Whether it ratchets was unanswered when the session died.
- Refit machinery healthy throughout (val 0.755–0.935; rotation ≤74° cum; labels balanced).

## 7. Cross-arm interpretation

One mechanism explains both forks: **the margin half injects likelihood collateral (~6 nats of
pair-mass sink in the first 50 steps, measured directly in A); whichever anchor exists responds
through a biased channel (logsumexp → favorite side; DPOP → chosen-then-spillover); the anchor's
response is stronger than the install force** (A: inert by mis-scaling; B: throughput-starved
against KL). On the toy, the margin half selected the solution basin and exact-J-on-the-menu
supplied stable pressure with structural mass conservation; on UF, with open-ended completions
and no menu, the margin half so far supplies only collateral and forging pressure. The
**RLOO-only ablation (`MCOEF=0`)** is the decisive one-variable test: if it holds the step-50
profile, the two-head coupling claim fails on realistic data and the probe's value is as reward
(the Goodfire regime); if it also degrades, the margin half is exonerated and the suspect becomes
RLOO-vs-KL throughput.

## 8. Where this leaves the thesis (three Occam axes)

- **Read depth** (reward/label content): tested and alive — evidence curve peaks mid-band, gate
  shows depth-dependent label content in the predicted directions (§1). The soft-DPO depth
  differential (L12 vs L31 vs GT labels, + RewardBench OOD) remains the clean next experiment and
  all of its infrastructure now exists (`L_OVERRIDE`/`RUN_TAG` in `uf_soft_dpo.py`,
  `uf_rewardbench_eval.py`).
- **Write depth** (which blocks change): still untested — the restricted-LoRA program
  (≤L12 vs 20–31 vs all, GT and probe labels) is designed but not run.
- **Coupling depth** (gradients through activations at ≤L\*): now evidenced **against** on UF,
  pending the `MCOEF=0` ablation. The adaptive-absorption machinery works; what's missing is any
  demonstrated benefit from the activation channel to pay for its collateral.

## 9. Artifacts: what survives this instance, what dies, regeneration costs

Committed in-repo (this doc + `results/*.json` + all `uf/*.py`). **Dies with the instance:**
feature caches (`uf_probe_feats_lenmatch.npz` 3.4G — regen ~30 min GPU via
`RL_STEPS=0 uf/uf_probe_rl.py`; `uf_plan_feats.npz` 1.2G — regen inside `uf_plan_sweep.py`),
plan-sweep sampling (~2.5 h; **`results/uf_plan_samples.json` is committed**, so the sweep can be
rebuilt without resampling by restoring it to `/workspace/`), all LoRA checkpoints
(`uf_hybrid2_ckpt100`, `uf_hybrid3_exactj_{ckpt100,ckpt200,lora}`, `uf_hybrid3_rloo_ckpt100`),
and the base model (public, re-downloadable). Every run is seeded and the funnel is deterministic,
so all numbers here are reproducible from code + committed JSONs.

## 10. Next steps, priority-ordered

1. **`MCOEF=0` RLOO-only ablation** (300 steps) — decisive for §7. One env var.
2. Soft-DPO **depth differential** (GT vs L12 vs L31 labels; big-N + RewardBench + length evals) —
   the read-axis Occam test proper; everything scripted.
3. **Write-depth program**: DPO with LoRA restricted to blocks ≤12 / 20–31 / all
   (`peft layers_to_transform`) — the untested core of "fewest params, earliest params".
4. Fork-A retune *only if* the margin half is exonerated: TAU≈25–30 + per-side floors (§5).
5. Run the ready eval battery on any surviving/retrained arms: `uf_bestofn_eval.py` (selection
   baseline), `uf_fresh_probe_audit.py` (collusion audit), `uf_rewardbench_eval.py` (OOD).
6. Bigger-K/batch RLOO (phase-3 open item) if 1 shows RLOO holding but slow.
