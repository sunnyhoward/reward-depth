# Phase 6 — The DPO relaxation program: parameter-space priors, and the first working activation objective

*2026-07-29. One session. Program set by the supervisor: take DPO apart stage by stage — first
replace its reference-model prior with a replay-estimated curvature penalty (EWC/K-FAC), then move
the preference objective from token space into the residual stream. Everything below is
single-seed except where noted; all run histories are in `results/`, scripts in `pythia/`,
`cc_stage2.py`, `uf/uf_margin_ewc.py`, `uf/uf_steer_sweep.py`. The supervisor's package
(`replay-kfac-ewc/`, vendored) supplies replay generation, true-Fisher K-FAC factors, the LoRA
trace-form penalty, and the calibration protocol; 11/11 tests pass unmodified.*

## 0. Verdict table

| claim | status |
|---|---|
| A replay curvature prior can replace DPO's reference model (token-space objective) | **YES — beats DPO** (Pythia, 2 seeds) |
| Such a prior can stop activation-margin forging | **NO — structurally** (λ-invariant; forging is metric-null) |
| Scaling replay volume fixes the prior's blind spots | NO (distribution, not N; mixed-corpus negative) |
| The frozen probe direction is a causal handle at any depth | **NO** (steering sweep: judge-null everywhere, 94% of 96 cells) |
| A per-batch adaptive activation direction (saturated) moves behaviour | **YES — first in the project** (24–32% oracle flip, knowledge intact) |
| Lag of the read is the forging resource | YES (clean k-spectrum: k=0 forges, k≥1 doesn't) |
| The install persists past its peak | NO — recession by step 300 at every λ (the open problem) |
| Attachment depth matters, measured causally | **YES — strict inverted-U at the elbow** (L20: installs & stable; below: detonates; above: inert; drive-normalized control confirms it is not a temperature artifact) |
| The two-head hybrid (margin ≤ L* + exact-J > L*) completes the install | **YES — 95% flip with OOD transfer at step 50**, probe-sourced rewards, no sampling |

## 1. Morning prelude on UF (Tulu-8B): diagonal EWC as the KL stand-in

`uf/uf_margin_ewc.py`. Margin-only arm (self-read at L*=12, length-matched probe), leash =
diagonal empirical Fisher on 128 own-samples, penalty on ΔW = s·BA.

Big-N (350 pairs, SE ≈ .026), `results/uf_margin_bigN.json`:

| ckpt | acc | dlp chosen/rejected |
|---|---|---|
| noleash ck200 | .606 | +0.04 / −0.79 |
| noleash ck300 | .549 | −0.08 / −0.74 |
| EWC=1e4 ck200 | .603 | +0.54 / −0.03 |
| EWC=1e4 ck300 | .571 | **+1.23 / +0.30** |

- λ=1 is inert (pen ~1e-5 vs mloss ~0.5); λ=1e4 binds in-metric but **forging (z_selfread) is
  identical at every λ**. The leash flips the collateral (likelihoods raised, not sunk) without
  changing install or forging. First sighting of the theme of the day: **likelihood-curvature
  metrics do not see forging.**
- Both arms peak at ck200 and decay — early stopping matters here as everywhere.

## 2. Steering sweep (the queued experiment from `notes_steering_experiment.md`)

`uf/uf_steer_sweep.py`, frozen Tulu-8B, 32 layers × α∈{.03,.1,.3} × 64 prompts, external judge
(Qwen2.5-7B-Instruct, order-debiased), cross-layer probe matrix, KL cost. Committed:
`results/uf_steer_sweep.json`.

- **Judge win-rate null at every depth and dose** (94% of cells within 1 SE of 0.5; means
  .49/.51/.48 by α). The probe direction is NOT a quality handle anywhere.
- **Top-layer null space confirmed quantitatively**: KL/token at α=.3 falls ~80× from L0 (0.24)
  to L31 (0.003) — μ at the top rewrites the readable feature with no token consequence.
- Pre-registered "readable everywhere, steerable in the middle" is refuted on the efficacy axis;
  what survives is "readable everywhere, causally *inert* as a direction, everywhere."

## 3. Stage 1 (Pythia-410m, UF pairs, token-space margin): the prior swap works

`pythia/stage1.py` + `replay-kfac-ewc`: 800 prompt-conditioned replay sequences → attention+MLP
K-FAC → calibration (local slope 0.72, ratio 0.099) → three arms, 300 steps.
304 held-out pairs, SE ≈ .028. Histories: `results/pythia_stage1_*.json`.

| arm | final acc (s0/s1) | trajectory | dlp c/r (final) | replay KL |
|---|---|---|---|---|
| **refree + K-FAC-EWC** | **.688 / .691** | rising at end, no decay | −50 / −70 | 0.28 |
| DPO (live reference) | .579 / .628 | peak .645@200 → decay | −16 / −20 | 0.20 |
| refree, no anchor | .618 | oscillates .57–.66 | −61 / −75 | 0.37 |

- The reference-free margin with the replay prior **beats live-reference DPO** (~3.9 SE at seed 0,
  replicated at seed 1) and is the only non-decaying arm. The unleashed control isolates the
  anchor as load-bearing.
- Predicted vs measured drift stayed within ~2.5× against optimizer-chosen directions.
- **Blind spot** (twice-demonstrated today): dataset-text likelihoods collapse (−50/−70) because
  replay covers only the model's own continuations. Fixes tested:
  - **DPOP hinge on chosen side**: cures it (+10/+7) at install cost (.605) — the only working fix.
  - **Mixed factor corpus** (dataset text added): **does NOT work** (−52/−70) — the Kronecker
    factorization of a two-distribution mixture averages away the cross-structure. Coverage in
    the corpus is not coverage in the metric.
  - Scaling replay 10× (partial, 616/7200 seqs generated, paused): prediction on record — no fix,
    since the missing text has probability ~e^−50 under the model at any sample size.

## 4. The content-choice (cc) testbed

`helpers.py` cc format + `cc_validate.py`. Replaces the letter menu: options embedded in the
question ("What is the capital of France: Paris or Rome?"), CONTENT answers, know options
order-randomized. Kills the letter-policy attractor by construction; adds an open-vocabulary
displacement signal (offmenu) the menu could not show; `first_opt` replaces `fracA`.

Base validation (Qwen2.5-3B): train .997 / eval .990 / know .959, first-opt .551, ood digits .980,
ood sum .780. **Pythia-410m/1.4b are at chance (.43–.57) — the testbed contract requires Qwen.**
Consequence: Pythia hosts likelihood-space experiments; Qwen hosts oracle-verified behaviour.

Stage A on cc pairs: decodability plateaus at **L*=20/36, acc .980 (max 1.000)** — far cleaner
than UF's 0.80, which is what makes the forging-vs-behaviour dissociation sharp here.

## 5. Stage 2 (Qwen-3B, cc, activation-space margin): forging is metric-null; adaptivity beats it

`cc_stage2.py`. Anchor: 1000 cc-prompt-conditioned replay seqs → attention-only K-FAC →
calibration **local slope 0.979, ratio 0.69** (near-ideal quadratic scaling on random directions).
LoRA attention-only ≤ L (anchor coverage = write coverage). 100-step iteration cells.

### 5.1 Frozen-probe margin: forging, λ-invariant

| λ | flip @100 | know | z_selfread (frozen read 2.32, static) |
|---|---|---|---|
| 0 | 8% | .82 | −2.54 |
| 1 | 15% | .80 | −2.58 |
| 20 | **3%** | .80 | −2.50 |

- Identical forging trajectory at every λ — **the forging directions are in the K-FAC null
  space**. At λ=20 the leash suppresses only the small genuine behavioural component (which the
  metric *does* price): the anchor acts as a filter that keeps forging and discards behaviour.
- **Adversarial exploitation of the Kronecker gap**: on trained directions the penalty underprices
  measured replay-KL 13–20× (vs ~1:1 at calibration on random directions). The optimizer finds
  curvature-cheap, KL-expensive directions. Stands alone as a finding about K-FAC-as-regularizer.

### 5.2 Mean-diff objective (supervisor's variant): saturation is load-bearing

Push (f_wrong − f_right) along the batch's own mean-difference direction, detached.

- **Raw (linear, unbounded): catastrophic** — offmenu 1.00 on every set by step 50, replay-KL 2.4
  by 100. The open-vocabulary collapse the letter menu could never show.
- **Saturated (−logsigmoid of the projection): the first activation objective in this project to
  move behaviour more than the meter.** 24% oracle flip @50 (k=1), knowledge intact (.92–.96,
  ~0 offmenu on know), at the same drift budget where the probe arms forge.

### 5.3 Lag spectrum (k = window of the direction, in batches)

| k | flip @50 | flip @100 | offmenu @50 | behaviour of the read |
|---|---|---|---|---|
| 0 (frozen at init) | .07 | .07 | .12→.34 | inflates monotonically — forging |
| 1 | .24 | .14 | .18 | crosses honestly |
| **5** | **.32** | .06 | **.00** | crosses honestly |
| 25 | .28 | .10 | .01 | crosses honestly |

**Lag is the exploitable resource**: freeze the direction and the probe pathology reappears
exactly; any adaptivity escapes it; a small window (k=5) also removes the batch-noise collateral
of k=1. Inverted-U in k.

### 5.4 λ frontier at 300 steps: the recession is intrinsic

| λ | flip @100 | flip @300 | offmenu range | know @300 | replay KL |
|---|---|---|---|---|---|
| 0 | .17 | .07 | .19–.28 | .84 | .15–.19 |
| 1 | .14 | .07 | .06–.16 | .86 | .09 |
| 3 | .15 | .01 | .03–.07 | .90 | .10 |
| 10 | .09 | .01 | .00–.02 | .92 | .12 |

- λ=0 answers "does the anchor matter here at all": yes — ~2× drift and the worst collateral
  without it. Its stage-2 role is **collateral containment**, not anti-forging.
- **Every arm peaks near step 50 and recedes** regardless of λ. The saturating loss saturates on
  *separation*, not behaviour: pairs stay separated while the behavioural component rotates away.
  This is the open problem. Behaviour-gating was considered and rejected (needs an oracle; not a
  general method). Candidates: early-stop at peak (peak is reproducible, 24–32% across four
  configs), LR decay after separation, or the stage-2.5 hybrid below.

### 5.5 Depth sweep (meandiff k=5 λ=1, L ∈ {8, 14, 20, 26, 32}): the elbow is the only writable depth

| L | flip @50 | offmenu @50 | @100 | replay KL @100 |
|---|---|---|---|---|
| 8 | .01 | .00 | **collapse** (offmenu 1.0) | 1.67 |
| 14 | .27 | .15 | **collapse** (offmenu 1.0) | 2.81 |
| **20 = L\*** | **.32** | **.00** | stable; graceful recession | 0.10 |
| 26 | .00 | .00 | stable; inert | 0.07 |
| 32 | .02 | .00 | stable; inert | 0.14 |

Strict inverted-U peaked exactly at the decodability plateau onset, with a different failure
mode on each side: **below the elbow** the adaptive direction couples to behaviour but
uncontrollably (install-then-detonate — the late collapse mirrors the raw-objective failure);
**above it** the direction is causally inert (consistent with §2's null-space result at the
top); **only at L\*** does the objective install and remain stable. L20 numbers reproduce the
§5.3 k=5 cell exactly (same seed — determinism check, not an independent replication).

Read together with §2: the elbow is not merely where the preference becomes *readable* — it is
the only depth where it is *controllably writable*. This is the first affirmative
depth-matters result in the project, and the first obtained with an instrument that cannot be
satisfied by forging. Caveats: single seed, 100 steps, one α/k/λ configuration.

## 6. Where this leaves the program

1. **Stage 1 recipe (token space) is settled and portable**: reference-free margin + prompt-
   conditioned replay K-FAC anchor (+ DPOP hinge if absolute chosen-side likelihood matters).
   Ready for the Tulu/UF port with today's UF baselines already in `results/`.
2. **Stage 2 has its first working objective** — saturated, small-window mean-diff — and a
   precisely isolated open problem (the recession). The anchor stays for collateral; forging is
   solved by adaptivity, not regularization.
3. **Stage 2.5 — RUN, and it completes the install.** Hybrid = mean-diff margin (blocks ≤ L*) +
   exact-expectation candidate REINFORCE over the 2-entity menu (blocks > L*; rewards from the
   FROZEN probe, not the oracle; no sampling — the menu makes the expectation exact; full-stack
   anchor). `results/cc_stage2_hybrid_k5_history.json`:

   | step | trained flip | know flip (transfer) | ood_sum flip | offmenu | replay KL |
   |---|---|---|---|---|---|
   | 50 | **.95** | **.96** | .53 | .03 | .19 |
   | 100 | .82 | .82 | .27 | .18 | .30 |

   95% install with OOD transfer at step 50 — phase-4's success profile, reproduced on the
   non-degenerate testbed. Margin-only peaked at 32%; the emission-channel head is what
   completes it, confirming the phase-1–5 pattern. Decay + drift after 50: early stopping again.
4. **Depth-norm control** (MD_NORM, drive equalized at init): refutes the saturation-artifact
   worry — un-normalized init drives were comparable across depths (proj scale .14 @L20 vs .32
   @L26, both unsaturated), so L26/32 inertness is genuine. Bonus finding: halving the drive at
   L20 flips the outcome from install (32%) to representation-inflation-without-behaviour (4%)
   — drive amplitude selects between the behavioural and forging paths. The recipe lives in a
   depth × drive window.
4. The steering result + 5.1 close the frozen-probe-as-handle question for good: probes label;
   they do not steer, and margins against them are forged.

## 8. Post-doc controls (run the same evening, committed as JSONs only until 2026-07-30)

Two control pairs were run after §6 was written. They materially revise §6.3 and the verdict
table's hybrid row. All at L*=20, k=5, λ=1, seed 0, 100 steps, same runner.

**J-only (`JONLY_FULL` / upper-only): the margin half is NOT load-bearing on cc.**
`results/cc_stage2_jonly_{full,upper}_history.json`:

| arm | flip @50 | flip @100 | trajectory | offmenu @100 | know transfer @100 | replay KL |
|---|---|---|---|---|---|---|
| hybrid (§6.3) | **.95** | .82 | decaying | .18 | .82 | .30 |
| J-only, full-stack | .93 | **.98** | **rising** | .00 | .92 | .26 |
| J-only, blocks > L* only | .83 | **.98** | **rising** | .02 | .90 | .15 |

Exact-J alone — even writing only the blocks *above* the elbow — ends higher than the hybrid,
still rising, with zero offmenu and the lowest drift. The hybrid's post-50 decay is *caused by*
the margin half, not cured by it. On cc, the install lives entirely above the elbow, through the
emission channel. (This does not touch the §5 result that the margin objective *alone* moves
behaviour 24–32% — it says the margin adds nothing once an emission-channel head is present,
reversing §6.3's "the two heads complete each other" reading. The §5.5 depth result also stands:
it is about where the *margin* couples, not about the hybrid.)

**Sampled RLOO (`srloo`): exact-J's cleanliness was its implicit on-menu constraint.**
Same probe reward, but taken on-policy: sample K=4, re-render, frozen-base read at L*, RLOO
baseline. No KL-in-reward — the K-FAC anchor alone carries drift control (the relaxation thesis,
applied to the reward guard). `results/cc_stage2_srloo_{only,margin}_history.json`:

| arm | flip @50 | offmenu @50 | flip @100 | offmenu @100 | know @100 |
|---|---|---|---|---|---|
| srloo, no margin | .00 | **1.00** | .00 | **1.00** | destroyed (.00 correct) |
| srloo + margin | .68 | .11 | .34 | .54 | .69 wrong |

- The naked probe reward is hacked **instantly** in open vocabulary: offmenu 1.0 by the first
  eval. Replay KL stayed 0.22–0.35 throughout — **the anchor does not subsume reward guards**;
  the hack is curvature-cheap. Exact-J never showed this because its expectation ranges over the
  two menu candidates only: off-menu text simply isn't in its support. That structural constraint,
  not the anchor and not the probe, was doing the anti-hacking work.
- The margin half *delays* the hack (68% honest install at step 50) but is out-muscled by it by
  step 100. Its one demonstrated value in the hybrid family is as a drag on reward hacking, not
  as an installer.

**Consequences for §6.** Item 3's "the emission-channel head is what completes it" survives;
"both halves load-bearing" does not — drop the margin half on cc. The open problem for porting
to UF is now sharply posed: open vocabulary has no menu, so the on-menu constraint must be
replaced by explicit guards (KL-in-reward, pessimism LCB, DPOP anchor — exactly
`uf_probe_rl.py`'s v3 guard set). Whether sampled RLOO with those guards installs on UF, and
whether a margin half adds or subtracts there, is the 2026-07-30 experiment
(`uf_probe_rl.py` vs `uf/uf_hybrid_md.py`).

## 7. Method notes for whoever runs this next

- Calibrate BEFORE training and refit on the local window (noise floor below, saturation above);
  demand slope ≈ 1. The morning's diagonal run burned an arm learning this.
- Watch predicted-vs-measured KL during training: divergence = the optimizer has left the metric's
  valid region (or is exploiting it).
- The multi-arm runner (`ARMS="meandiff:1:k5:L8,..."`) shares model/Stage-A/factors across arms:
  ~6–8 min per 100-step cell. Iterate at 100 steps, confirm at 300.
- `workspace_is_volume` is still false: copy `/workspace/*history.json` into `results/` and push
  at the end of every run.
