# One finding, five demonstrations: guards fail when they measure the wrong distribution

*Synthesis written 2026-08-06. Draws on `results_0805.md` §8.1, `results_libon_0806.md`, and the
supervisor-recipe run of 2026-08-06. Single seed throughout unless stated; every number below is
measured in this repo.*

## The claim

A regularizer or a stopping metric constrains only the region of input space it is evaluated on.
When that region does not include the distribution the edit actually operates on, the guard is
**inert** — not weak, not miscalibrated, but blind — and it can read *better than baseline* on a
model that has been destroyed.

This is not a new failure each time. It is one failure with five instances, four of which were
found independently before the pattern was noticed.

## The five instances

### 1. Replay-based priors (2026-08-05, `results_0805.md` §8.1)

Stage-1 preference training moved task text **2.7 nats** while moving the replay corpus **0.003**.
The replay corpus was continuations from random 1–8 token prefixes — essentially unconditional
sampling — while the task text was `"Question: ...\nAnswer:"`. The anchor had nothing to hold.
Recorded at the time as "replay-based priors cannot protect an on-distribution edit"; the
corrected reading is narrower and stronger: *a replay corpus that does not match the operating
distribution is inert as a prior.*

### 2. K-FAC curvature estimated on that same corpus (2026-08-05)

The Fisher factors were estimated on the corpus from (1), so they measured curvature in
directions the edit never travelled. This is why the leash lost to a plain learning-rate control
— beyond being under-dosed, it was aimed at the wrong subspace.

### 3. The KL anchor in the Libon reproduction (2026-08-06)

β=2 KL to the base model, computed on 16 UltraChat prompts — their setting, faithfully ported.

| arm | KL end | KL max | degeneracy on benign prompts |
|---|---|---|---|
| logistic continuous | 0.037 | 0.252 | **0.94** |
| logistic retrained | 0.044 | 0.229 | **1.00** |
| bayes lam0 | 0.180 | 0.666 | 0.00 |

The anchor reads **four hundredths of a nat** on a model emitting word loops for 94% of benign
prompts. Worse, it is *anti-correlated* across arms: `bayes lam0` carries 5x the KL and ends
coherent. UltraChat responses are long and conversational; the damage lives on short instruction
prompts. A model that loops on "What's the weather like today?" still assigns near-base token
probabilities to a fixed UltraChat continuation, because the anchor never asks it to generate.

### 4. The 80%-utility stopping rule (2026-08-06)

Libon et al. report results only up to the last checkpoint retaining 80% of base utility
(MMLU + GSM8K + IFEval), and call this "not an arbitrary stopping rule but a qualitative
boundary". Measured here:

| run | crossed the 80% line at | final degeneracy (benign) |
|---|---|---|
| bayes lam0 | step 25 | **0.00** |
| logistic continuous | never | **0.94** |
| logistic retrained | never | **1.00** |

**The rule fires anti-correlated with the damage it exists to prevent.** It flagged the one arm
that ended coherent and stayed silent through the two that ended in word loops. `continuous`
ckpt75 scores utility **0.468 against a base of 0.458** — above baseline — while 100% of its
short-prompt generations are degenerate.

### 5. IFEval does not save it (2026-08-06)

IFEval is the one component of their utility score that grades free-form generation, so it was
the obvious fix. It misses the collapse too:

| arm | MMLU+GSM8K | IFEval strict | degeneracy on IFEval | degeneracy on short benign |
|---|---|---|---|---|
| base | 0.458 | 0.391 | 0.000 | 0.000 |
| continuous ckpt75 | 0.468 | 0.359 | 0.016 | **1.000** |

The same checkpoint, two prompt distributions:

```
SHORT BENIGN ("What are traditional systems for resource allocation and sharing?")
  -> 'Hoch AAAA " " " " " " " " " " " " " " " " " " " " ...'

IFEVAL ("I am planning a trip to Japan, and I would like thee to write an itinerary...")
  -> 'Verily, thou hast chosen a noble quest indeed! I shall endeavor to craft an
      itinerary for thee in the style of the great bard himself. Day One: Arriv...'
```

The collapse is **prompt-distribution-specific**: it lives exactly on the short instruction-style
distribution the probe loss was applied to. IFEval's prompts are long and structured, so they
elicit fluent text from a model that cannot answer a one-line question.

## Why this is one finding and not five

In every case the guard's measurement distribution and the edit's operating distribution were
chosen independently — the guard inherited its distribution from convenience (an existing replay
corpus, a standard benchmark, a standard KL dataset) while the edit's distribution came from the
task. Nothing in any of these designs *checks* the match. And the failure is silent by
construction: a guard reading zero is indistinguishable from a guard with nothing to hold.

The pattern also predicts the sign of the error. A guard evaluated off-distribution does not
merely under-report damage; it can report *improvement*, because a model that has specialised
its degradation to one region may be unchanged or even sharpened elsewhere (utility 0.468 vs
0.458 at 100% degeneracy).

## What to do instead

1. **State the operating distribution explicitly** as part of the method, not the evaluation.
2. **Evaluate every guard on it.** In the Libon port, the only guard that fired correctly was
   degeneracy measured on held-out prompts drawn from the same pool the training prompts came
   from (our in-loop coherence probe: 0.00 -> 0.31 -> 0.94 across the run).
3. **Report the guard's own sensitivity**: show that the metric moves on a model you have
   deliberately broken. A guard that has never been shown to fire is not evidence of safety.
4. For KL anchors specifically: compute the KL on prompts from the operating distribution, with
   the model *generating*, not teacher-forced over foreign text.

## Caveats

- Single seed on every run in instances 3–5; the effects are large (0.04 nats vs 94% degeneracy)
  but the exact numbers should not be quoted as precise.
- Instance 5's IFEval subset covers 65% of IFEval (16 verifiers implemented); prompts with an
  unimplemented instruction are dropped rather than failed.
- The degeneracy detector is lexical (4-gram repetition, run-length, emptiness). It agrees with
  the Qwen3-8B judge's `broken` category throughout but is not an independent oracle.
- Instances 1 and 2 share a corpus, so they are not fully independent of each other.
