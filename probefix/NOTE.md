# Probe-then-decoder: the Occam experiment on dosed britishness

*Started 2026-08-11, in parallel with the guard-dose x attach-depth sweep running out of
`supervisor/`. Nothing in `supervisor/` is modified by this directory; the data, the split, the
replay bank and the rendering are all imported from it, so the arms here and the arms there are
scored on the same rows.*

## The claim being tested

> If the preference is already decodable at layer L\*, the features are present, and the only
> thing that needs fixing is the decoder.

Operationally, in three parts:

1. **Stage 1** trains the encoder (layers 0..L\*) against a **linear probe at L\*** that is
   **refitted continuously** on the policy's own activations, with a **replay loss at the output
   logits** — which reaches every layer — so the rest of the model is not tanked while this
   happens.
2. **Stage 2** then trains the decoder at the output: either the upper half only (L\*+1..23) or
   the full stack.
3. **The control** is ordinary DPO at the output on all layers, same data, same steps, same
   replay term.

The measurement that makes it more than a horse race: **where in the model the change lands**, per
arm, both as parameter deltas and as function drift by depth.

## Stage 0 — the premise, measured on this model and this diet

`pf_probe_curve.py`, Qwen3.5-2B, `britishness/dosed/brit_dose20.jsonl`, held-out **by group**
(a held-out guard fact is never trained in any form, in either role). One linear direction on the
RMS-normalised residual, fitted on 4,735 train pairs, scored on the held-out buckets. Read index
is the block whose output is read.

| block | pooled | **guard** | legacy | culture | truth\_dialect (install) | false\_friend |
|---|---|---|---|---|---|---|
| emb | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 0 | 0.788 | **0.060** | 0.940 | 0.881 | 1.000 | 0.615 |
| 4 | 0.856 | **0.020** | 0.979 | 0.963 | 1.000 | 0.770 |
| 8 | 0.858 | 0.300 | 0.972 | 0.969 | 0.940 | 0.718 |
| 10 | 0.869 | 0.500 | 0.971 | 0.949 | 0.910 | 0.737 |
| **12** | **0.870** | **0.680** | 0.972 | 0.963 | 0.890 | 0.695 |
| 16 | 0.850 | 0.680 | 0.960 | 0.949 | 0.890 | 0.648 |
| 19 | 0.833 | 0.700 | 0.951 | 0.944 | 0.830 | 0.624 |
| 23 | 0.832 | 0.660 | 0.959 | 0.958 | 0.760 | 0.634 |

Three things, and the first two set the design:

- **The install families are an L0 phenomenon and the guard is not.** Everything British is at or
  near ceiling from block 0 (0.94 legacy, 1.00 truth\_dialect-install). The guard — the pairs where
  the true answer is American and the British-sounding one is false — reads **0.06 at block 0 and
  0.02 at block 4**: not chance, *backwards*, because at that depth the only axis in the residual
  is dialect and the guard's rejected side is the British one. It crosses chance at ~block 9 and
  plateaus at 0.66–0.70 from **block 12**. So `L* = 12`, half depth. The composite rule
  ("British, unless that would make it false") is the thing with a depth signature; britishness
  alone has none.
- **The same 199 facts run in opposite directions with depth.** As install rows
  (`truth_dialect`, prefer the British rendering) they decay 1.00 → 0.76 from block 0 to 23; as
  guard rows (prefer the true American one) they climb 0.06 → 0.70. Mirror image, same facts, same
  prompts — which is what a truth axis coming online mid-stack looks like.
- **The pooling read is not a free choice.** Under a MEAN read over the whole assistant turn the
  guard column never becomes decodable at any depth (max 0.38, i.e. still backwards) while pooled
  accuracy is *higher* (0.90) — the mean is dominated by the dialect markers, which are exactly
  what the guard has to be overruled by. Training against the mean-read probe would be training
  "prefer British, guard be damned" and would have looked like a stronger objective while doing
  the wrong thing. The trainer uses the **last scored position**, on this evidence.

## Arms

All on `brit_dose20.jsonl`, seed 0, 6 pairs/step, lr 1e-4, beta 0.1, LoRA r16 attn+mlp, replay NLL
weight 1 on his generative-replay shards, 16 scored tokens/step. K-FAC off (results\_0809 §5a:
1:3:1 and 1:0:1 indistinguishable).

| arm | loss | LoRA range | init | steps |
|---|---|---|---|---|
| **B** | hinge on a continuously-refitted linear probe at block 12 | **all 24** | base | 300 |
| **C1** | DPO at the output | 13..23 | B@300 merged | 300 |
| **C2** | DPO at the output | all 24 | B@300 merged | 300 |
| **A** | DPO at the output — *the control* | all 24 | base | 600 |
| **D** | DPO at the output | 13..23 | base | 300 |

- **B's LoRA is on every layer while its preference gradient can only reach 0..12.** The probe
  reads at block 12 and nowhere else, so nothing above it is in the preference gradient's path;
  the layers above are reached only by the replay term. That is the asymmetry the design asks for
  ("the replay can update the whole of the network").
- **A runs 600 steps** so it is compute-matched to B+C (300+300) at ckpt600 and step-matched to B
  alone at ckpt300. Quoting a two-stage pipeline against a half-length control is the cheapest way
  to manufacture a win.
- **D is the arm that decides the headline.** It is C1 without stage 1: fix the decoder, never
  touch the encoder. If D ≈ C1, the encoder edit was decoration and "the features are already
  there, only the decoder needs fixing" is *literally* true. If C1 > D, stage 1 did something the
  decoder could not do alone. Either way it is the interpretable comparison, and it costs one
  300-step run.

## Anti-forging provisions

Phase 1's settled result is that a gradient loss asking only for a feature value is satisfied by
forging the feature. Three things make this a different bet, and one meter decides whether they
worked:

1. **Scale-free read.** Scores come off the RMS-normalised residual with a unit-norm probe
   direction, so "multiply the residual by 10" — one scalar, no representational change — is worth
   exactly nothing.
2. **Continuous refit.** The probe is re-fitted every step (4 Adam steps on a 2048-pair FIFO
   buffer of the policy's own detached reads). A direction the model has learned to inflate stops
   being the fitted direction. Phase 6 found a per-batch adaptive direction was the first
   activation objective here to move behaviour more than its own meter.
3. **A saturating objective.** Hinge at 1 sigma of the step-0 score spread; a pair already
   separated contributes exactly zero gradient. Nothing is paid for running the margin up. (Logged
   as `sat` — the fraction of the batch already above target, which at step 0 was 0.67–1.00: most
   of the diet needs no change at all, which is the Occam premise stated as a training statistic.)

**The meter:** `pf_audit.py` refits an *independent* probe at every depth on the trained model's
activations, and `sup_eval.py` scores free generation. Co-trained probe accuracy that does not
survive a fresh refit, or that does not show up at the output or in generation, is forging.

## Meters, per checkpoint

- `pf_audit.py` — full-bucket ranking (raw + implicit **vs the pristine base for every arm**, so
  stage-2 arms are on the same origin as the rest), per-layer function drift (relative L2 and
  cosine against base on held-out text), per-layer/per-module LoRA weight deltas, and the
  fresh-probe depth curve on the trained model.
- `supervisor/sup_eval.py` — free-sampling British-marker rate (`brit_rate`), the behavioural
  column. Ranking without it is the dissociation this project has been burned by four times.

## Known limits, stated up front

- **Single seed**, like everything else in this repo.
- **The guard holdout is 50 rows** (SE ≈ 0.07). It is a whole-group holdout of 2 of 8 topic
  groups, which is the right split, but it cannot resolve differences below ~0.15.
- **No behavioural guard meter.** `brit_rate` measures dialect in free generation; whether the
  model states falsehoods to sound British is only measured as ranking. A generation-side guard
  meter needs a fact checker and is not built.
- The `truth_dialect` install family and the guard are **crossings of the same 199 facts**. The
  split moves whole facts across both roles, so this is not leakage — but an arm can trade one
  column for the other, and that trade is the thing to watch, not a bug to explain away.
