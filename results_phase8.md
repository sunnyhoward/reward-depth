# Phase 8 — Why no head learns dominance: the frontier is in the features, and the bottleneck is eloquent wrongness

*2026-07-31. One session, fresh box (RTX PRO 6000 Blackwell 96GB, `workspace_is_volume` false;
repo restored from the HF bundle, styc cache regenerated from scratch — Stage-A factor curves
reproduced). All experiments this session are offline probe fits on the styc v2 feature cache
(Qwen-3B, 579 questions, 36 layers); no policy training. Scripts: `styc_probe.py` (v3),
`styc_conflict_sweep.py`, `styc_lex_pareto.py`, `styc_mlp_head.py`, `styc_residual.py` (all new
or extended this session). Every result JSON is in `results/`, committed after each run.
Standing caveat: single seed, held-out n=116 questions per family (binomial SE ≈ 0.046).*

## 0. Where this session started

Discussion of the program's state concluded: toy results good, UF weak, and soft-DPO — the one
working UF method — is not evidence for the Occam thesis (it saturates at labeller accuracy;
OOD tracks probe accuracy, not depth). The user's proposal: a synthetic multi-head middle step.
The user's standing constraint: labellers must be GENERAL (preference labels only, no factor
labels); their hypothesis: an early head reads style, a late head reads everything, per-sample
evidence weighting combines them. Phase 7 §9 had already refuted the entangled-ensemble form
(0.000 on conflicts); this session asked *why*, and whether any label-free protocol escapes.

The user also independently re-derived the lexicographic preference structure (correctness
dominant, style tiebreaker) that `styc_train.py` already encodes — converging design.

## 1. Stage A v3: aligned pairs + honest validation (`results/styc_stageA_v3_aligned.json`)

Two gaps in the v2 protocol fixed:

- **`aligned` family added** (correct-explained vs wrong-terse — both factors agree). This was
  entirely absent from the v2 diet despite being the majority shape on real data.
- **Per-layer validation now on held-out mixed pairs** (v2's `fit_pref` validated on the first
  64 *training* rows).

Results (train diet = corr_e, corr_t, style_c, style_w, aligned; conflict held out):

| | L0 | L16 | L20 | max |
|---|---|---|---|---|
| mixed val acc | .82 | .85 | .89 | **.938 @L35** |
| pref on corr_t | .56 | .66 | .75 | .966 @L35 (evidence-ens .983) |
| pref on corr_e | .52 | .59 | .68 | .724 @L35 |
| pref on style/aligned | 1.0 | 1.0 | 1.0 | 1.0 everywhere |
| **pref on conflict** | .000 | .000 | .000 | **.000 everywhere** |

The late head is much better overall — and still absolutely style-dominant on conflicts.
Adding aligned pairs changed nothing on conflicts (pre-registered prediction: correct).

**The information-theoretic point** (argued before the sweep, confirmed by it): every
non-conflict family has the factors either agreeing or varying alone, so the diet contains no
bit about which factor *wins* a disagreement. A linear scorer fits all five families perfectly
under either dominance order. Dominance must come from disagreement pairs — or from inductive
bias, and the bias picks style (bigger, cleaner margin). Conflict-free training ⇒ dominance
unlearnable *in principle*, at any capacity (§4's rate-0 MLP control confirms at capacity).

## 2. Conflict-rate dose-response (`results/styc_conflict_sweep.json`)

Conflicts added to the training diet at {0, 2, 5, 10, 15, 20}%, mixed validation, all layers.
At L35 (best layer throughout):

| rate | conflict | style_c | corr_t |
|---|---|---|---|
| 0% | .000 | 1.000 | .966 |
| 2% | .017 | 1.000 | .974 |
| 5% | .103 | .991 | .991 |
| 10% | .276 | .853 | .974 |
| 15% | .319 | .836 | .974 |
| 20% | .336 | **.767** | .974 |

**Dosing does not work.** Shallow response saturating ~⅓ — and the 15% cell (.319) reproduces
the phase-7 natural-fit labeller (.32) exactly, which in turn matches UF's style-first natural
fit at its native ~13.6% conflict rate. Three independent routes to the same number.

Mechanism localization: at 10–15%, L35 is the *only* layer whose style_c degrades — the head
buys conflict accuracy by shrinking its style weight (a trade), not by promoting correctness
to dominance. Pushing further approaches the 33%-cancellation cliff (phase-7 trap).

## 3. The Pareto diagnostic: no linear direction implements the lexicographic order
(`results/styc_lex_pareto.json`)

Balanced six-family diet at L35, conflict sample-weight λ swept:

| λ | conflict | style_c | style_w | aligned | min over families |
|---|---|---|---|---|---|
| 0.5 | .16 | .92 | .99 | 1.00 | .16 |
| 1 | .30 | .85 | .99 | 1.00 | .30 |
| 2 | .65 | .48 | .94 | .97 | **.48** ← frontier best |
| 4 | .97 | .11 | .80 | .81 | .11 |
| 8–32 | .99–1.0 | .01–.03 | .42–.69 | .48–.75 | ~.02 |

Reference heads: a conflict-only head scores 1.00 on conflicts and **0.00 on every style and
aligned family** — the linearly-optimal conflict direction is anti-style. Best min-across-
families over the whole linear family: **0.48**. No weighting yields conflict ≥.9 and style ≥.9.
**Representational, not diet.** (L20: same shape, lower everywhere.)

## 4. Nonlinear readout: the MLP frontier IS the linear frontier
(`results/styc_mlp_head_natural.json`, `results/styc_mlp_confheavy.json`)

Hypothesis (pre-registered, and wrong): the conditional readout "if correctness signal present,
use it; else style" is expressible with one hidden layer, so an antisymmetric MLP
(f(x)=g(x)−g(−x), 2-layer, same difference features) should hold style ≥.95 while lifting
conflicts well above the linear 0.48 ceiling.

| cell | conflict | style_c | min |
|---|---|---|---|
| L10 / L20, any rate | .00–.15 | .96–1.0 | ≤.15 |
| L35, 0% (control) | .01 | 1.00 | .01 ✓ control clean |
| L35, 15% | .47 | .76 | .47 |
| L35, 30% | .80 | .37 | .37 |
| L35, 50% | .96 | .14 | .14 |

The MLP traces the *same* trade-off curve as the linear λ-sweep (compare λ=4: .97/.11).
**Capacity moves nothing; the frontier is a property of the features.** Early layers as
predicted (no correctness feature to condition on); the rate-0 control at full capacity stays
at 0 (the §1 information argument holds at any capacity).

## 5. Residual/boosted composition: no escape (`results/styc_residual_hard.json`)

Label-free two-stage protocol: natural h1 (style-first) → stage-2 h2 on error-reweighted diet →
three combiners (additive; dominance gate on |z2|; learned 2-D antisymmetric stacker).
First attempt was a no-op — the calibrated Bayes z is compressed (conflicts sit at z1 ≈ −0.47
± 0.5), so `exp(−z)` weights barely varied (`results/styc_residual_softboost.json`; kept as a
methods trap). Sharpened (T=0.5) and errors-only reruns:

| L35 | conflict | style_c | min |
|---|---|---|---|
| h1 (natural) | .16 | .92 | .16 |
| h2 (temp-boost) | .26 | .76 | .26 |
| h2 (errors-only) | .08 | .97 | .08 |
| best combiner (any variant) | .22 | .94 | .26 |

No composition escapes the frontier. (Structural note, worked out before the run: an
anti-style h2 is useless to an antisymmetric 2-D combiner anyway — conflict-with-A-correct and
flipped-style-pair produce the same (−z1, +z2) signature, so the combiner cannot distinguish
them without genuine correctness signal. And per §6, genuine correctness signal *on the
explained side* is the scarce resource.) Caveat: h2's errors-only fit landing style-first
(conflict .08) is odd enough that a further implementation issue can't be excluded; but §4
bounds anything built from these features regardless, so this line was not pursued further.

## 6. The bottleneck, localized: eloquent wrongness is ~0.8-detectable, everything follows

The collapse pattern in §4 is family-specific: only style_c dies; style_w and aligned survive.
That is the fingerprint of the real constraint. Any antisymmetric scorer must satisfy

    conflict:  f(ct − we) > 0        (prefer correct-terse over wrong-explained)
    style_c:   f(ct − ce) < 0        (prefer correct-explained over correct-terse)

and the two inputs differ only by (ce − we) — the correctness-of-the-explained-side direction.
So winning conflicts without destroying style **is exactly a corr_e discrimination**, and
corr_e is the one weak signal in the system: **0.776 decodable at L35** (vs .957 for terse),
capping at ~.84 under every λ. Three independent fits show the same anti-explained-on-correct-
pairs signature (the λ≥4 linear heads; phase 7's factor head, 0.023 on style_c; the conflict-
heavy MLP). The dominance failure is not "style capture" as a training pathology — it is a
*perceptual* limit: **the 3B cannot reliably tell a fluent wrong assertion from a fluent right
one, and every readout limitation is downstream of that.**

This is the styc-oracle version of UF phase 7 §8: the confidently-backwards tail was fluent
deflection vs correct execution. The two testbeds now agree at the mechanism level.

## 7. Standing conclusions after this session

1. **Dominance information lives only in disagreement pairs.** No conflicts in the diet ⇒ no
   dominance, at any capacity (theory + rate-0 controls).
2. **At realistic conflict rates (10–15%), natural fits saturate at ~⅓ conflict accuracy** and
   pay for it in style accuracy — on styc with an oracle, matching UF in the wild.
3. **The achievable frontier is set by the features, not the readout** — linear, MLP, and
   boosted compositions trace the same curve. Head engineering on a frozen deficient
   representation is a dead end. This closes, for this testbed and model, both the depth-
   ensemble idea and its residual-fitting repair.
4. **The binding scalar is corr_e ≈ 0.78–0.84**: detectability of wrongness under fluent
   assertion. This is now the program's central measured quantity.

## 8. Where this points next (agreed direction + open)

- **Scale/architecture question (now the headline experiment): does corr_e rise with model
  size?** styc-XL task ladder × model sizes. If a 7–8B reads eloquent wrongness at ~.95, the
  whole dominance problem dissolves with scale for these task families; if not, reward models
  genuinely need information the base model does not compute (either conclusion is a result).
  The 2-step-arithmetic "decodable nowhere" bet remains on record alongside.
- **Flip-training arms** (behavioural install measurement, cc-style): design agreed this
  session — flip correctness dominance only (wrong > correct, style still tiebreaker), because
  only a correctness-competent labeller can express the flip; `gen_wrong` becomes a behavioural
  dose-response of labeller competence. Labeller ladder: GT (1.0) / dosed natural (~.32) /
  style-blind early (~0). Implementation: `LEX_P` in `styc_train.py` + existing oracles.
- **UF translation-tail probe** (fitting-problem-vs-capability-wall test): per-layer accuracy
  on the confidently-backwards slice under tail-upweighted refits. §6 predicts wall-like
  behaviour at 8B for translation (the corr_e analogue); needs the UF cache rebuild (~30 min).
- **Seeds** for anything above that gets written up (styc fits are minutes).

*Sections below this line were added later the same day — the session continued.*

## 9. The positional argument (why direct activation objectives never touched behaviour)

Every activation objective in the archive attached at the **completion-end reading state**
(`[:, -1]` at the eos sentinel — `cc_stage2.py:158`, `uf_hybrid_md.py:134`), a state causally
*downstream* of the finished answer. Generation is driven by prefix states at each emission
step; nothing in a completion-end loss makes the edit route through them, and the cheapest
satisfying edit is to the post-hoc summary (arm A: 4x separation, generations bit-identical).
The likelihood/PG family constrains the emission channel directly, which is why it installs.
Complements the phase-1 forging argument; with the steering null it closes the "why" of the
whole direct-activation line. (The one untried cell — per-position margin — runs as the
`pooled_margin` arm below.)

## 10. Mean pooling dissolves the 3B perception wall (user proposal)

Refitting stage A on features **mean-pooled over answer tokens** (`STYC_POOL=mean`) instead of
the completion-end token:

- **corr_e at 3B: .776 (last-token) → .991 (pooled).** The verification signal was distributed
  along the trajectory; the last-token protocol threw it away. §6's "perception wall" was
  substantially a READOUT-POSITION artifact.
- **The natural-protocol labeller (mixed diet, conflicts 15%) becomes correctness-dominant:**
  conflict .32 (last-token) → **.905** (pooled), corr_e .99, with the trade surfacing only as
  style_c .638. Min-family .638 vs the last-token Pareto ceiling of .48 — the pooled feature
  geometry beats the last-token frontier at natural dose, no factor labels, no boosting, no
  bigger model.

Scaling curve of corr_e max (last-token protocol), for context: 0.5B .621, 1.5B .966,
3B .776, 7B .991 (14B pending; 3B is the family's architectural outlier, and pooling shows its
wall was positional). At 7B even last-token reads are ~ceiling.

## 11. Training FROM the probe works: the `shaped` arm (styc_probe_rl.py, MODE=shaped)

Design per `notes_dense_probe_rl.md`: guarded REINFORCE (RLOO baseline, DPOP floor,
KL-in-reward, LCB pessimism); reward = pooled-probe read of the FROZEN base over sampled text;
dense credit A_t = (Phi_end − b_LOO) + (Phi_end − Phi_t) from the running-mean potential.
No DPO loss anywhere; no gradient through policy activations. 200 steps, 3B LoRA, lr 1e-5.
`results/styc_prl_shaped_history.json`.

| step | corr_e | corr_t | conflict | style_c | gen_correct | gen_other |
|---|---|---|---|---|---|---|
| 25 | .30 | .05 | .03 | .99 | .97 | .03 |
| 125 | .79 | .68 | .28 | .88 | .94 | .06 |
| 200 | .88 | .90 | .22 | .99 | .16 | .84 |

1. **The mechanism installs.** Style in 25 steps, correctness following to ~.9 by 200 — where
   sequence-sparse RLOO from a (weaker, last-token) probe sat at noise for 300 steps (phase 7
   §2). First successful training-from-the-probe in the project.
2. **Style-first dynamics as pre-registered** (density asymmetry): style tokens are everywhere,
   correctness tokens few. The policy converges toward the labeller's own competence profile,
   weak spots included (style_c descends toward the labeller's .638; conflicts rise but reach
   only ~.3 of the labeller's .905 by 200).
3. **Honest Goodhart from ~step 130**: preamble inflation ("Addition is the mathematical
   operation of...") defers/truncates answers — gen_correct .94 → .16 — while EVERY ranking
   metric continues improving and z_frozen_gap stays bit-identical (.483; nothing forged,
   nothing off-menu, gen_wrong 0.0 throughout). With phase 7's "preference corrupts, behaviour
   intact," both dissociation directions are now oracle-demonstrated: **implicit-reward
   dashboards and behaviour can move in opposite directions under optimization.**
4. Peak policy ≈ step 125. Operational: periodic adapter checkpoints are REQUIRED in the UF
   port (only the final, over-optimized adapter was saved here).

Queued fixes for the preamble exploit, one arm each: potential-JUMP credit (reward where Phi
moves sharply — the answer tokens), and an answer-first/deferral guard in the anchor.

## 12. Also added this session

- **Generative-replay floor** (user proposal): `REPLAY_N` in `styc_probe_rl.py` — random-token
  prompts, base continuations banked once, one-way floor relu(logp_base − logp_policy) sampled
  each step. Constrains the policy on BROAD support, not just the training task — the UF
  refusals/reasoning collapse (phase 3) is the evidence this matters. Default off on styc (arm
  comparability); default on for the UF port. K-FAC replay prior remains the parameter-space
  equivalent if the forward passes get expensive at 8B.
- Scale-invariance control (pre-registered, §1): with conflicts held out of training, the
  pref-head conflict accuracy is 0.000 at EVERY model size — 0.5B, 1.5B, 7B (max .009). The
  dominance bit is in the diet, not the capacity, at all scales.

## 13. The other two arms (session close)

**`pooled_margin` — the control that won** (`results/styc_prl_pooled_margin_history.json`).
Direct activation optimization — the eight-phase dead line — but on POOLED trajectory features:
lag-1 adaptive mean-diff margin on (ce−we) pooled policy activations at L35, DPOP anchor,
backprop THROUGH policy activations (forging channel deliberately open). 200 steps:

| step | corr_e | corr_t | conflict | style_w | gen_correct | gen_explained |
|---|---|---|---|---|---|---|
| 25 | .98 | .94 | .60 | .68 | .88 | .39 |
| 100 | 1.0 | .95 | .86 | .45 | .89 | .97 |
| 200 | 1.0 | .98 | **.845** | .42 | **.891** | .94 |

Cleanest install in the project: near-labeller conflict competence from a diet of ONLY ce/we
pairs (the pooled correctness direction IS the dominance-relevant feature), behaviour intact
(gen_wrong 0.0 throughout; an early terse-drift at 25 self-corrected via the anchor), stable
plateau from ~100 with NO phase-5-style reversal and NO forging. The forging prediction
failed in the good direction: with the target being the mean of every emission state, there is
no causally-dead single state to cheaply rewrite — the pooled fix for the reading channel
also closes the classic forging exploit. Cost axis: style_w ~.42 (ranking among wrong
answers). Caveats: single seed; implicit-acc partly rides likelihood displacement — the
generation oracles carry the claim. **The UF hybrid revival gate is OPEN.**

**`seqrl` — density ablation, partial (50/200 steps, box killed;
`results/styc_prl_seqrl_history.json`).** Same reward as shaped, sequence-level credit only.
By step 50: gen_correct collapsed to .17 (preamble exploit), conflicts floored, styles at
ceiling. Verdict, and it is decisive despite the truncation: **the eloquence spiral is the
LABELLER's preference, not a credit-scheme artifact — sparse credit found the same exploit
FASTER** (shaped got ~100 useful steps first; seqrl dove straight in). At step 25 seqrl was
better-proportioned than shaped (corr_e .62 vs .30 — per-token credit over-weights ubiquitous
style tokens), so density buys speed and early balance-distortion, but the Goodhart endpoint
is reward-determined. Fixes must target the reward: answer-first anchoring, deferral penalty,
or potential-JUMP credit.

## 14. Session close: what was killed unfinished

Box destroyed 2026-07-31 evening (Option A). Not completed: 14B stage A (caching done, fits
killed), 7B last-token conflict sweep + Pareto (no output produced), 3B-mean full stage-A JSON
(the corr_e .991 headline and the .905-conflict labeller are captured in §10 and in every
styc_prl history's `labeller_acc`), seqrl steps 50-200. All regenerable by script from caches
(~1-2 h GPU total); none block the next session's priority, the UF port.
