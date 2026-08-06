# Bayesian probes in the Libon pipeline (2026-08-06)

Extension of the Libon reproduction (`NOTE.md`). Everything outside the probe is held fixed:
Mistral-7B-Instruct-v0.1, LoRA r=64 / alpha=128 on q,k,v,o,up,down, beta=2, layers
{0,6,12,18,24,30}, 8 harmful + 8 benign prompts/step with on-policy completions, KL on 16
UltraChat prompts, single seed. **Baseline to beat: the reproduced continuously-updated run**
(`continuous_ckpt25`: compliance 0.408 from base 0.583, utility 0.480, zero degeneracy).

## What changed

| | theirs | ours |
|---|---|---|
| probe | per-layer logistic, per-token logit, mean-pooled | per-layer variational Bayesian probit head (`helpers.BayesLinearHead`) |
| adaptation | N SGD steps continued from previous weights | **previous posterior becomes this step's prior** |
| scoring | mu | **mu + sign*lambda*sigma** |
| layer aggregation | uniform mean over layers | **weights ∝ softmax(per-layer ELBO)** |

## Choices that are ours, and could have gone otherwise

1. **Pool-then-read, not read-then-pool.** Their probe mean-pools per-token logits. For a linear
   read that equals pooling unit-norm-scaled activations and reading once
   (`mean_t(w.h_t) = w.mean_t(h_t)`), so the mean is unchanged — but it makes the posterior
   variance a per-sequence quantity rather than per-token, which is what pessimism needs.
2. **Pessimism sign.** The spec asks for `mu - lambda*sigma`. For a SUPPRESSION objective that
   makes an uncertain completion score LESS harmful, so uncertainty *relaxes* the pressure;
   `mu + lambda*sigma` is the safety-conservative reading. Both are implemented (`PESS_SIGN`);
   the sweep runs the spec as written plus one `+sigma` arm at lambda=1 as the control that
   separates "pessimism helps" from "any attenuation of the loss helps".
3. **Evidence weights need a temperature.** ELBOs are sums of log-likelihoods, so raw differences
   are large and a bare normalisation is one-hot. Weights are `softmax(ELBO / tau)` with
   `tau = max(1, std(ELBO))`. Raw ELBOs are logged alongside the weights so the choice is
   auditable.
4. **Sequential KL is to the previous posterior**, i.e. `KL(q_t || q_{t-1})` with `q_{t-1}`
   detached, not to the fixed `N(0, tau^2)`. The fixed prior is retained for the `retrained`
   regime so the two are directly comparable.

## Caveat attached to every pessimism number

Mean posterior sigma is ~0.098 against a prior tau of 0.1 (d=4096, n=652 per fit): the posterior
is still essentially the prior in most directions. Per-example uncertainty does vary — the score
uses `f^T sigma^2 f`, not sigma alone — but lambda is scaling a weakly-updated posterior, so the
sweep should be read as "does uncertainty-scaled attenuation help", not as a test of a
well-calibrated posterior.

## Diagnostics added beyond the reproduction

- **Translation diagnostic** (their Fig 12): both class means under the CAPTURED INITIAL probe.
  Separates "the classes moved apart" from "both slid down the old axis" (evasion). Fires within
  2 steps on every arm run so far.
- **In-loop utility probe** (MMLU + GSM8K subset) at every eval, so the 80%-budget crossing step
  is recorded during training rather than reconstructed from checkpoints afterwards.
- **IFEval subset scorer** (`libon_ifeval.py`, 16 verifiers, 65% of IFEval fully verifiable) —
  added because the budget rule never fired; see `results_libon_0806.md` for why it does not
  save it either.
