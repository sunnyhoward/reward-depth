# Steering britishness: the mid-stack is a handle, the top is dead — and gradient cannot fix that

*2026-08-18. `notes_steering_experiment.md`'s design, finally on the task probefix is about.
Frozen Qwen3.5-4B, no training in the first half. Two instruments:
`probefix/pf_steer.py` — mean-pooled probe fitted at every layer, ActAdd/CAA `dm` vector
(`v ∝ SD·μ`), scaled by `α·R_L` with `R_L` the mean RAW residual norm, added at every position of
block L's output; 9 layers × α ∈ {0.1, 0.3}.
`probefix/pf_addon.py` — `h_L ← h_L + A_L·z`, `A_L` 2560 params trained (300 steps, `PREF=nll`) so
the model GENERATES the preferred form at z=+1 and the dispreferred at z=−1, everything else frozen;
z=0 reproduces base exactly. Generated on the `SEED=0` famgen eval draw, 48 prompts × 2 families.*

**No judge has scored any of this.** The meters here are `pf_famlex` marker counts (lexical, non-
circular, `false_friend` only — `style` is a register family with no marker pairs) and
`pf_leakage.py`. Both are blind to things the judge catches, and the marker ratio is the meter
`RESULTS_0813` §5 caught reading .94 for an arm that loops inside its own generations. Treat every
number below as a screen, not a result.

---

## 1. The fitted direction: dead at L0, a broad mid-stack plateau, collapse at the top

`false_friend` marker brit_rate, base = 0.456 (79 markers). SE per cell ≈ 0.056.

| L | α=0.1 | α=0.3 | leakage (α=0.3) |
|---|---|---|---|
| 0 | −0.017 | **−0.051** | 0.115 |
| 4 | +0.072 | +0.111 | 0.000 |
| 8 | +0.044 | +0.077 | 0.000 |
| 12 | +0.085 | +0.121 | 0.000 |
| 16 | +0.100 | +0.044 | 0.000 |
| **20** | +0.081 | **+0.127** | 0.000 |
| 24 | +0.050 | +0.108 | 0.000 |
| 28 | +0.038 | +0.026 | 0.000 |
| 31 | +0.038 | +0.031 | 0.000 |

**Leakage is 0.000 in 17 of 18 cells**, lengths hold at 252–280 against base's 278, and the peak
cell carries **91 markers against base's 79** — the gain is not bought by damaging or shortening the
text. The one leaking cell (L0, α=0.3) is also the only one that moves brit_rate *down*.

**Against the two prior runs of this design**, both on UltraFeedback with Tulu-8B: phase 6
(last-token directions) was judge-null in 94% of 96 cells; phase 9 §7 (pooled) found a mid-layer
band on 8 layers. Here the same design on a minimal-pair task gives a clean curve with a factor of
~4 between the mid-stack and the top. **The user's diagnosis was right: UF was too hard a dataset
for this instrument.**

**Pre-registration scored**: "rises through the low-mid stack, collapses by L28–L31" — **confirmed**.
"L20 is NOT the peak" — **wrong**; L20 α=0.3 is the peak, with L12 (+0.121) and L4 (+0.111) inside
one SE of it. The honest form is a plateau from L4 to L24, not a point.

## 2. Against the trained arms, on the one meter they share

| arm | marker brit | markers | judged ff (0818 pass) |
|---|---|---|---|
| base | 0.456 | 79 | 41.0 |
| `MD_L20_s300` (activation training) | 0.488 | 43 | 56.5 |
| `MDF_kl_s100` (best activation cell) | 0.508 | 65 | 61.7 |
| **`steer_L20_a0.3`** (no training) | **0.582** | 91 | — |
| `P1_r1` (plain DPOP) | 0.638 | 58 | 68.3 |

On this meter, **adding the direction at inference beats every activation-space training arm in the
project and covers ~70% of the base→DPOP gap, with no training at all.**

**That is a dissociation, and it is the point.** Training the model to increase separation along the
L20 direction installs nothing at the output (`probefix/RESULTS.md`, and
`RESULTS_0818_STAGE1_SIDES.md` §5 after three candidate causes were eliminated). Adding that same
direction at L20 moves generation cleanly. **The direction is a causal handle; optimising
separation along it is not the same operation, and this project has been treating them as one
claim.** The caveat that limits how far this can be pushed: the last column shows the marker meter
compressing the judged scale non-uniformly (base→DPOP is +0.18 here and +27 there), so the ~70%
figure is a screen, not an effect size.

## 3. The learned add-on: it dies at the top too, with gradient choosing the direction

| cell | Δ brit | markers | len ff | ‖A‖ | leak |
|---|---|---|---|---|---|
| `addon_L0` | +0.108 | 39 | 132 | 2.69 | **0.635** |
| `addon_L4` | +0.176 | **19** | **36** | 4.80 | 0.000 |
| `addon_L8` | +0.133 | 17 | 35 | 6.85 | 0.073 |
| `addon_L12` | +0.064 | 25 | 43 | 8.12 | 0.000 |
| `addon_L16` | +0.169 | 24 | 45 | 9.23 | 0.000 |
| `addon_L20` | +0.050 | 83 | 218 | 12.75 | 0.000 |
| `addon_L24` | **+0.118** | 68 | 230 | 16.50 | 0.000 |
| `addon_L28` | **+0.001** | **92** | 255 | 21.25 | 0.000 |
| `addon_L31` | +0.030 | 70 | 275 | 21.28 | 0.000 |

**L0–L16 are confounded and should not be read.** Their outputs collapse to 35–45 characters
(base 278), so the marker denominator falls to 17–25 and a +0.176 on 19 markers is ~1.5 SE —
*weaker* evidence than the steering peak's +0.127 on 91. The cause is the objective: `PREF=nll`
maximises the absolute likelihood of the chosen continuation, and the references are **short
minimal-pair replies**, so `A` absorbs length and register alongside dialect. The margin form
(`PREF=dpo`) cancels length exactly, because both sides are the same length — so the arm predicted
in `pf_addon.py`'s docstring to be the weaker one is the necessary control, for the opposite reason
to the one given there. **That prediction is retracted and the control is now required.**

**L20–L31 are clean** — full length, 68–92 markers — and they carry the finding:

- **`addon_L28` is +0.001 on 92 markers. `addon_L31` is +0.030 on 70.** Gradient descent had free
  choice of direction, 2560 parameters, and an objective defined directly on generation, and it
  still could not move the output from the top of the stack.
- **‖A‖ grows monotonically with depth**, 2.69 → 21.28, while the effect falls. The optimiser pushes
  four times harder at L28 than at L4 for nothing. That is diminishing causal leverage measured
  directly, not inferred.
- At L20/L24 the two instruments agree in size (+0.050/+0.118 learned vs +0.127/+0.108 fitted).

**This is the strongest form of the null-space claim the project has.** Phase 1 measured
`cos(μ, W_A−W_B) = −0.003`; `RESULTS_0817_MDLATE` found late reads inert; `RESULTS_0818_STAGE1_SIDES`
§5 found freezing the probe changes nothing. All three could be answered "the *fitted* direction is
just the wrong vector." **That answer is now closed at L28–L31**: a direction chosen by gradient
against the actual generation objective does no better.

## 4. What must happen before any of this is a claim

1. **A judged pass.** Nothing here is judged. The arms to put in one blind batch, so coherence and
   `british` are directly comparable: `base`, `P1_r1`, `MDF_kl_s100`, `steer_L{4,12,20,28}_a0.3`,
   `addon_L{20,28}`. Cells are already in the famgen schema at the `SEED=0` draw, so
   `pf_cjudge.py batchnew` takes them unmodified.
2. **`PREF=dpo`** for the add-on, as the length-controlled control (§3).
3. **`style` is unmeasured** in this whole document — no marker pairs exist for a register family.
   Given that `style` is where every activation arm fails hardest and where DPOP wins by 29 points,
   a steering result that only covers `false_friend` covers the easy half.
4. Single seed, 48 prompts per family, greedy decoding.
