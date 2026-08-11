# Probe-then-decoder on dosed britishness — results

*2026-08-11. Qwen3.5-2B, `britishness/dosed/brit_dose20.jsonl` (guard = 20% of the training diet,
holdout split BY GROUP so a held-out guard fact is never trained in either role). Single seed
throughout. Design, arms and pre-registration in `NOTE.md`; figures in `results/probefix/`.*

## The question

> If the preference is already decodable at layer L\*, the features are present, and the only
> thing that needs fixing is the decoder.

Stage 1 trains the encoder against a **continuously-refitted linear probe at L\***, with a replay
loss at the output logits that can reach every layer. Stage 2 trains the decoder with ordinary
DPO. The control is ordinary DPO on all layers. Two extra arms make the comparison interpretable:
**D**, decoder-only DPO from base (stage 2 with no stage 1), and **A** at 600 steps so it is both
step-matched to stage 2 and compute-matched to stage 1 + stage 2.

## 0. The premise, measured (`pf_probe_curve.py`)

One direction on the RMS-normalised residual, fitted on 4,735 train pairs, scored per held-out
bucket. Read index = the block whose output is read.

| block | pooled | **guard** | legacy | culture | truth\_dialect (install) |
|---|---|---|---|---|---|
| 0 | 0.788 | **0.060** | 0.940 | 0.881 | 1.000 |
| 4 | 0.856 | **0.020** | 0.979 | 0.963 | 1.000 |
| 8 | 0.858 | 0.300 | 0.972 | 0.969 | 0.940 |
| **12** | **0.870** | **0.680** | 0.972 | 0.963 | 0.890 |
| 19 | 0.833 | 0.700 | 0.951 | 0.944 | 0.830 |
| 23 | 0.832 | 0.660 | 0.959 | 0.958 | 0.760 |

- **Britishness alone has no depth signature. Britishness-with-a-guard does.** Install families are
  at ceiling from block 0; the guard reads 0.06 there — not chance, *backwards*, because the only
  axis in the residual at that depth is dialect and the guard's rejected side is the British one.
  It crosses chance at ~block 9 and plateaus from **block 12 = L\***, half depth.
- **The same 199 facts run opposite ways with depth**: as install rows 1.00 → 0.76 (block 0 → 23),
  as guard rows 0.06 → 0.70. That mirror image is a truth axis coming online mid-stack.
- **A mean read over the assistant turn never makes the guard decodable at any depth** (max 0.38)
  while scoring *higher* pooled (0.90). Training against it would have optimised "prefer British,
  guard be damned" while looking like the stronger objective. The trainer reads the last scored
  position on this evidence.

## 1. Held-out ranking, full buckets (`pf_audit.py`)

pooled n=918, legacy n=750, guard n=50 (SE ≈ 0.07), truth\_dialect n=100.

| arm | pooled | **guard** | legacy | culture | **truth\_dialect** | false\_friend | style |
|---|---|---|---|---|---|---|---|
| base | 0.236 | 1.000 | 0.057 | 0.186 | 0.070 | 0.315 | 0.137 |
| **R** replay only (W_PREF=0) | 0.245 | 1.000 | 0.069 | 0.181 | 0.050 | 0.335 | 0.160 |
| **A** DPO all layers, 600 | **0.940** | 0.920 | 1.000 | 0.994 | 0.930 | 0.864 | 0.994 |
| **B** probe stage 1 @L12, 300 | 0.266 | 1.000 | 0.073 | 0.226 | 0.060 | 0.357 | 0.160 |
| **C1** B → DPO upper (13–23) | 0.911 | 0.660 | 0.996 | 0.992 | **0.980** | 0.864 | 0.891 |
| **C2** B → DPO full stack | 0.904 | 0.820 | 0.997 | 0.997 | 0.960 | 0.812 | 0.891 |
| **D** DPO upper only, no s1 | 0.781 | 1.000 | 0.992 | 0.986 | **0.360** | 0.765 | 0.640 |
| **B2** probe s1, unbounded BT | 0.298 | 1.000 | 0.277 | 0.266 | 0.080 | 0.399 | 0.166 |
| **C3** B2 → DPO upper | 0.907 | 0.780 | 0.996 | 0.994 | 0.970 | 0.837 | 0.846 |

`R` is the replay-only null: with the preference term switched off entirely, 300 steps of the same
replay anchor move pooled ranking 0.236 → 0.245. B (0.266) is barely distinguishable from it, which
is the cleanest statement of "stage 1 installs nothing at the output".

## 2. Behaviour — free sampling, marker oracle (`sup_eval.py`)

The check that is not optional. Two different measurements:

- **ranking** — teacher-forced pairwise accuracy: summed log-prob of the chosen completion vs the
  rejected one over the assistant-turn tokens, scored `lp(chosen) > lp(rejected)`. The model writes
  nothing; it only has to put more mass on the preferred side. This is close to what the DPO loss
  optimises directly. **The column below is `sup_eval`'s `ranking_all`, which pools ALL 1,668
  held-out rows including the 750 legacy lexicon rows and the 50 guard rows** — it is therefore a
  different pool from §1's `pooled_new` (918 rows, legacy excluded, guard reported separately), and
  the two must not be read as the same number.
- **brit_rate** — free generation: 128 greedy completions of 96 tokens on held-out
  lexicon/culture/false_friend prompts (with the empty `<think>` block closed first, or the budget
  goes into a reasoning trace the preference never touched), scored with a marker lexicon read off
  the dataset's own `item` field. `br_hits / (br_hits + am_hits)`. Note this is a ratio over MARKER
  HITS, not over generations: base yields 95 hits and A yields 120, so the effective n differs by
  arm and the SE ranges 0.020–0.054.

| arm | brit\_rate | ranking, whole holdout (n=1668) |
|---|---|---|
| base | 0.211 | 0.155 |
| base + tailored preamble | 0.792 | 0.689 |
| B | 0.278 | 0.181 |
| C1 | 0.662 | 0.949 |
| C2 | 0.775 | 0.947 |
| D | 0.750 | 0.877 |
| **A**@300 | 0.944 | 0.941 |
| **A**@600 | **0.950** | 0.968 |

The preamble number reproduces the supervisor's own baseline (517/735 = 0.703) at 0.689 on a
different split, which is the best external check this pipeline has.

## 3. Findings

### 3.1 Stage 1 raises decodability everywhere and installs nothing — and it is not forging

B's co-trained probe goes to ceiling on install and 0.84 on the guard. That number is worthless on
its own, so `pf_audit` refits an **independent** probe on the trained model's activations and
scores it on held-out rows:

| block | guard, base → B | pooled, base → B |
|---|---|---|
| 8 | 0.240 → **0.500** | 0.852 → 0.899 |
| 12 | 0.620 → **0.760** | 0.866 → **0.942** |
| 23 | 0.580 → **0.740** | 0.836 → **0.925** |

The composite rule became substantially more linearly available, and the gain **propagates to the
final block**. Meanwhile B's output ranking is 0.073 legacy against base 0.057, and its brit\_rate
is 0.278 against base 0.211. A fresh probe at block 23 reads the preference at 0.93 and the
unembedding does not use it. This is the surrogate gap in its cleanest form to date — measured on
a refit probe, so phase 1's forging account does not cover it.

### 3.2 The premise is necessary but not sufficient — and the deficit is family-specific

D and C1 train the **same parameters** (blocks 13–23) for the **same 300 steps**, differing only in
whether the encoder went through stage 1:

| | pooled | legacy | culture | **truth\_dialect** | style | guard |
|---|---|---|---|---|---|---|
| D (from base) | 0.781 | 0.992 | 0.986 | **0.360** | 0.640 | 1.000 |
| C1 (from B) | 0.911 | 0.996 | 0.992 | **0.980** | 0.891 | 0.660 |

D's deficit is not spread out — it is concentrated in `truth_dialect` and `style`. Decoder-only DPO
installs everything already decodable at block 0 (legacy 0.992, culture 0.986) and **cannot install
the family that routes through the mid-stack truth axis**. That is `STATE.md`'s write-depth
prediction, confirmed per family rather than in aggregate, and it is the answer to the question:
base decodability at L\* was *necessary* (D can't get there without it) but *not sufficient*
(D stalls at 0.36 on the family that matters).

### 3.3 The guard is bought and sold on the same 199 facts

`truth_dialect` install and the guard are crossings of the same facts in opposite directions, and
every arm's position is explained by which way it resolved them:

| arm | truth\_dialect | guard |
|---|---|---|
| D | 0.360 | 1.000 |
| C1 | 0.980 | 0.660 |
| C2 | 0.960 | 0.820 |
| **A** | 0.930 | **0.920** |

D declines both. C1 takes the install and pays the guard. **A takes almost all of the install and
keeps almost all of the guard**. (§3.3b un-crosses this: "pays the guard" turns out to be the wrong
description — no arm loses the truth axis at all, the dialect pull simply outgrows it.) — full-stack DPO finds a joint solution the two-stage arms do not.
(Do not read C1 → C2 → A as a write-range ordering: D writes the same range as C1 and sits at the
opposite corner. What separates the arms is how far each one pushed the `truth_dialect` install,
and stage 1 is what let C1 push it hardest.)

### 3.3b Un-crossing the guard: nobody lost the truth axis (`pf_guard_axes.py`)

The guard number above is ambiguous by construction. A guard pair is crossed on two axes at once —
chosen is TRUE and AMERICAN, rejected is FALSE and BRITISH — so `guard` falling to 0.660 could mean
the model got so British it will swallow a falsehood, or that it simply got worse at physics. The
pairs are minimal (a token diff is exactly two edits: the fact token and the marker) and
`meta.marker` gives "american|british", so the 2x2 reconstructs cleanly on all 50/50 held-out
facts: TT / FA / TB / FB. Truth is then read with dialect held constant, and dialect with truth
held constant. Nats are the mean log-prob swing from flipping one factor.

| arm | crossed | prefers TRUE | truth nats | prefers BRITISH | British nats | truth/British |
|---|---|---|---|---|---|---|
| base | 1.000 | 1.000 | +9.0 | 0.070 | **−3.9** | — |
| R replay only | 1.000 | 1.000 | +8.9 | 0.070 | −3.6 | — |
| B | 1.000 | 1.000 | +9.1 | 0.070 | −3.7 | — |
| B2 | 1.000 | 1.000 | +9.5 | 0.070 | −3.6 | — |
| D | 1.000 | 0.960 | +22.4 | 0.360 | −1.2 | — |
| **A**@600 | **0.920** | 1.000 | +64.7 | 0.900 | **+13.1** | **4.94** |
| C3 | 0.800 | 1.000 | +77.9 | 0.970 | +26.7 | 2.92 |
| C2 | 0.800 | 0.990 | +99.3 | 0.960 | +37.3 | 2.66 |
| A@300 | 0.700 | 0.980 | +46.5 | 0.950 | +23.4 | 1.98 |
| **C1** | **0.660** | 0.980 | +66.9 | 0.970 | **+35.2** | **1.90** |

- **No arm loses the truth axis.** Truth preference stays 0.96–1.00 everywhere and the truth pull
  gets 2.5–11x STRONGER than base (+9 → +22…+99 nats). §3.3's "the guard is paid for" was the
  wrong reading: nothing was forgotten.
- **The guard number is a ratio test between two preferences that both grew**, and it tracks that
  ratio monotonically across every arm (1.90 → 0.660, 1.98 → 0.700, 2.66/2.92 → 0.800,
  4.94 → 0.920).
- **The two-stage arms fail at DISCRIMINATION, not strength.** A pulls +13.1 nats British on guard
  items (where British is wrong) and writes British at 0.950 on install prompts. C1 pulls +35.2
  nats British on guard items — the strongest of any arm — and writes British at 0.662, the
  weakest. Exactly inverted. A learned a conditional rule ("British, except where that makes it
  false"); C1 learned an unconditional one.
- **The mechanism is NOT established, and two explanations have already been refuted.** This
  section first claimed a single linear direction cannot express "British, unless that makes it
  false", so stage 1 installs an undiscriminating disposition by construction. **That is wrong.**
  `decodability/brit_guard_dose.py` fits ONE head at a 0.2 guard rate that reads guard 0.843 and
  install_true 0.944 simultaneously at 3/4 depth, and §0's own probe reads guard 0.68 with install
  ~0.95 at block 12. One direction expresses the conditional fine.
  The fallback — that stage 1 nonetheless trains against a dialect-dominated direction — is also
  wrong (`pf_axis_decomp.py`, 50 held-out guard facts, axes fitted on the base model at L12 where
  cos(dialect, truth) = −0.16): B's saved probe leans TOWARD truth, cos(w, truth) +0.203 against
  cos(w, dialect) +0.112, and B amplified the truth axis **x4.41** against dialect's x1.48.
  Stage 1 strengthened truth three times harder than dialect.
  A third attempt — reading the same axes at the final block, where C1's truth component collapses
  to x0.15 — does not survive its own cross-check: C1 prefers the true statement by +66.9 nats,
  the second-highest of any arm, so the distinction is plainly still there and strongly used. The
  base-fitted frame is measuring a ROTATION, not a deletion (C2's sign-flipped dialect, x−1.71, is
  the same artifact). Any frame fitted on the base model degrades as a measuring instrument exactly
  when the representation has changed a lot, which is when it is being asked the question.
  **The behavioural fact stands and is unexplained.**
- A's guard "recovery" from 0.700 at step 300 to 0.920 at 600, which looked like
  noise in §5: its truth pull rose 46 → 65 while its British pull FELL 23 → 13. A is not becoming
  less British late in training (brit_rate 0.944 → 0.950); it is learning where not to be.

### 3.4 Stage 1 costs behavioural transfer, at BOTH write ranges

The pair C1/C2/A alone reads as "restricting the write range costs behavioural transfer", and that
is what this section said before D's generation eval came in. **D falsifies it**: D writes the same
blocks 13–23 as C1, ranks *lower* (0.877 vs 0.949) and generates *better* (0.750 vs 0.662). Write
range is not the variable. The variable is stage 1, and it shows up in both matched pairs:

| write range in the output-loss stage | arm | ranking | brit\_rate | behaviour per unit ranking |
|---|---|---|---|---|
| blocks 13–23 | D (no stage 1) | 0.877 | 0.750 | 0.86 |
| blocks 13–23 | C1 (after stage 1) | 0.949 | 0.662 | 0.70 |
| all 24 | A (no stage 1) | 0.968 | 0.950 | 0.98 |
| all 24 | C2 (after stage 1) | 0.947 | 0.775 | 0.82 |

In both pairs the stage-1 arm ranks as well or better and generates worse. **The encoder edit buys
teacher-forced ranking that does not convert into generation** — which is the same object as §3.1
(a representation the readout does not use), surviving into the trained model rather than being
resolved by stage 2. It is the repo's ranking/behaviour dissociation, and here it is *caused* by
the intervention under test rather than merely co-occurring with it.

**Powers differ between the two pairs and should not be quoted as one result.** `brit_rate` is a
ratio over marker hits, so its SE is set by the hit count, not by the 128 generations:
A 0.950 (120 hits, SE ≈ 0.020) vs C2 0.775 (89 hits, SE ≈ 0.044) is ≈ 3.6 SE and is solid — and it
is NOT a step-count artefact: A had 600 output-loss steps against C2's 300, so A was re-scored at
ckpt300, where at matched output steps and matched ranking (0.941 vs 0.947) it generates 0.944
against C2's 0.775, ≈ 3.3 SE;
D 0.750 (156 hits, SE ≈ 0.035) vs C1 0.662 (77 hits, SE ≈ 0.054) is ≈ 1.4 SE and is suggestive
only. The claim rests on the full-stack pair, with the restricted pair agreeing in sign.

### 3.5 Where the change lands (`results/probefix/pf_where.png`)

Function drift (‖h − h\_base‖/‖h\_base‖ on held-out text) and LoRA weight delta by layer:

Raw ΔW is uninterpretable on its own: every arm's LoRA spans all 24 layers and its replay term
trains all of them. `R_replay_only` (W\_PREF=0, everything else identical) is the subtraction
baseline — and it is nearly flat at 0.018, so the raw profiles are mostly replay.

**Preference-attributable weight change, ‖ΔW‖/‖W‖ minus the replay-only arm:**

| arm | blk 0 | 4 | 8 | **12 = L\*** | 16 | 20 | 23 |
|---|---|---|---|---|---|---|---|
| **B** probe, saturating hinge | −0.0012 | +0.0001 | +0.0004 | **−0.0000** | +0.0005 | +0.0004 | +0.0003 |
| **B2** probe, unbounded BT | +0.0001 | +0.0009 | +0.0012 | **+0.0122** | +0.0001 | +0.0001 | +0.0001 |
| **A** DPO all layers | +0.0085 | +0.0078 | +0.0096 | **+0.0133** | +0.0082 | +0.0080 | +0.0088 |
| **C2** B → DPO full | +0.0173 | +0.0177 | +0.0191 | **+0.0213** | +0.0171 | +0.0154 | +0.0156 |

- **Every arm that writes anything at all puts its maximum at block 12** — including plain DPO,
  which was never told where the elbow is and whose loss is 11 blocks above it. The stage-0 curve
  identified L\* = 12 from decodability alone; unconstrained DPO independently chooses to work
  there.
- **What differs is the shape, and the shape is what predicts behaviour.** The probe-attached
  objective writes a *delta function*: B2 is +0.0122 at block 12 and +0.0001 at every other layer.
  Output DPO writes a *broad edit peaked at* block 12 (+0.008 everywhere, +0.0133 at the peak).
  The spike-only edit installs nothing at the output (B2 pooled 0.298); the broad one installs
  everything (A pooled 0.940).
- **B wrote nothing.** Its preference-attributable ΔW is ±0.0007 across the stack — at the noise
  floor. The saturating hinge fired on so few pairs (§3.6 — 52.7% of steps had zero preference gradient) that B's entire measured weight change,
  and most of its 0.14-nat replay KL, is the replay term. B's representational gains in §3.1 were
  therefore bought with an edit too small to show up in ΔW at all, which makes them more striking,
  not less.

**Function drift** (‖h − h\_base‖/‖h\_base‖ on held-out text) tells the same story from the
activation side:

| arm | @4 | @12 | @23 |
|---|---|---|---|
| R replay only | 0.151 | 0.225 | 0.357 |
| B | 0.173 | 0.266 | 0.372 |
| B2 | 0.233 | 0.387 | 0.452 |
| C1 | 0.173 | 0.266 | 0.955 |
| C2 | 0.250 | 0.478 | 1.087 |
| D | 0.000 | 0.000 | 0.938 |
| **A** | 0.255 | **0.632** | 1.034 |

**Attaching the loss at L\* does not maximise the function change at L\*.** B reads its entire
preference signal at block 12 and drifts 0.266 there — barely above the replay-only floor of 0.225.
A's loss is at the output and it drifts **0.632** at the same block, more than twice as far.
Output-attached DPO rewrites the mid-stack representation harder than an objective aimed directly
at it.

### 3.6 The objective barely fires — and firing harder does not rescue it

B's own training statistics: mean `sat` (fraction of the batch already above the hinge target)
**0.889**, and **52.7%** of steps carry zero preference gradient because every pair in the batch is
already separated. Stage 1 only ever edits the minority of pairs that are not already resolved —
the Occam premise restated as a training statistic. That confounds "stage 1 is weak" with "stage 1
barely trained", so `B2_probe_bt` reran it with an unbounded Bradley-Terry objective that never
saturates, and `C3_s2_upper_bt` took B2 through the same stage 2 as C1.

**The confound is resolved and the conclusion survives.** B2 does write (§3.5: +0.0122 at block 12
against B's zero) and it does leak some install to the output on its own (legacy 0.277 vs B's
0.073, base 0.057). But its stage-2 continuation lands in the same place:

| | pooled | guard | legacy | truth\_dialect | style |
|---|---|---|---|---|---|
| C1 (from saturating B) | 0.911 | 0.660 | 0.996 | 0.980 | 0.891 |
| C3 (from unbounded B2) | 0.907 | 0.780 | 0.996 | 0.970 | 0.846 |
| **A** (no stage 1) | **0.940** | **0.920** | 1.000 | 0.930 | 0.994 |

C3 ≈ C1 on install with a somewhat better guard, and both still lose to A on the joint objective.
**Stage 1's weakness was not undertraining.**

One unexpected difference between the two stage-1 objectives, from the fresh-probe curves — they
move decodability in different *places*:

| fresh-probe guard | blk 4 | 8 | 12 | 16 | 23 |
|---|---|---|---|---|---|
| base | 0.000 | 0.240 | 0.620 | 0.600 | 0.580 |
| B (saturating) | 0.040 | 0.500 | **0.760** | 0.740 | **0.740** |
| B2 (unbounded) | **0.540** | **0.800** | 0.660 | 0.640 | 0.660 |

B makes the composite rule more decodable *from L\* upward*. B2 makes it decodable **much earlier**
— guard 0.000 → 0.540 at block 4, 0.240 → 0.800 at block 8 — while ending *lower* than B from
block 12 on. Pushed hard, the objective moves the feature down the stack rather than sharpening it
where it was read. Neither version converts into behaviour.

## 4. What this says about the hypothesis

The Occam reading — *decodable at L\* ⇒ just fix the decoder* — is **half right, and the half that
fails is the interesting one.**

1. Base decodability at L\* is **necessary**: with the encoder frozen at base, the decoder installs
   everything already decodable at block 0 and stalls at 0.360 on the one family that routes
   through the mid-stack truth axis (§3.2).
2. It is **not sufficient**: making the feature more decodable first (guard 0.62 → 0.76 at L\*,
   pooled 0.87 → 0.94, propagating to block 23) is what unlocks that family, 0.360 → 0.980. The
   encoder edit is genuinely load-bearing for what the decoder can express.
3. But what stage 1 buys is **ranking, not behaviour**. At both matched write ranges the stage-1
   arm ranks as well or better and generates worse (§3.4), and at the endpoint it has traded away
   a third of the guard (§3.3). The extra decodability the fresh probe measures is real and the
   readout still under-uses it.

On every metric that matters jointly — pooled 0.940, guard 0.920, brit\_rate 0.950 —
**plain DPO on all layers at matched compute is the best arm in this experiment.** The two-stage
pipeline's one clear win is narrow and specific: it is the only way found here to install
`truth_dialect` from a restricted upper-stack write, and that win does not survive into generation.

## 5. Limits

- **Single seed.** Everything here.
- **Guard holdout is 50 rows** (SE ≈ 0.07). The 0.66/0.82/0.92 ordering across C1/C2/A is ~2 SE
  end to end and monotone with a mechanism, but it is not individually significant.
  `truth_dialect` (n=100) and pooled (n=918) are much better powered, and they carry the same story.
- **A ran 600 steps against the two-stage arms' 300 + 300.** That is compute-matched by design, but
  A's guard is non-monotone in training (1.000 → 0.700 at step 300 → 0.920 at 600), so an arm
  stopped at 300 lands at A's worst point. Read the Pareto figure, not a single row.
- **Still no GENERATION-side guard meter.** §3.3b separates the two axes, but it does so
  teacher-forced. Whether a model spontaneously writes falsehoods to sound British would need a
  fact checker over free samples, and the guard prompts are generic ("State a fact about physical
  science plainly") so free generation need not touch the fact under test at all. Not built.
- **`brit_rate`'s prompt pool contains no guard rows** — it is filtered to lexicon/culture/
  false_friend, 1,317 rows, all `role: install`. So it measures britishness only where britishness
  is wanted, and is not inflated by prompts where it is wrong. The guard side is §3.3b's job.
- **`brit_rate` powers differ by arm** — it is a ratio over marker hits, so SE ranges 0.020 (A) to
  0.054 (C1). See §3.4.
- **The stage-1 arms were not given a longer stage 2.** C1/C2/C3 all ran 300 stage-2 steps; A's
  advantage appears between its own steps 300 and 600. A 600-step stage 2 is the obvious next cell
  and is not run.
