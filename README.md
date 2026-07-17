# reward-depth

**Where should the reward signal attach to a language model — and what does the attachment depth
buy you?** We train a policy against a linear (Bayesian) *reward probe* reading the model's own
hidden state at layer L, and study how the installed preference changes with L: its
generalization, its coherence, and how gracefully it Goodharts under over-optimization.

The working hypothesis (Bayesian Occam's razor): attach the reward at the **earliest layer where
the preference is fully decodable** (the ELBO / evidence peak). A head there can only read the
simple, semantic core of the preference — dataset idiosyncrasies that are only decodable late
can't enter the reward — so the policy trained against it should overfit the preference less and
degrade more gracefully than output-level methods (DPO / top-attached reward models).

## Phase 1 (this repo, first experiment): the probe at the very top vs DPO

Before moving the probe *down*, calibrate the top: attach the probe to the **post-final-RMSNorm
hidden state — the exact tensor the unembedding consumes** — and compare head-to-head with DPO on
identical pairs and identical LoRA coverage. The isotropy argument (near-spherical unembedding
rows ⇒ averaged DPO updates reduce to a single utility direction) predicts near-equivalence.

Our prediction from the preceding displacement study: **high but sub-1.0 gradient alignment** —
DPO's gradient carries a softmax-normalization component (an implicit imitation pull on the chosen
completion) that a pairwise reward-head margin provably lacks; that missing component is exactly
what makes naive probe-RL drift off-distribution where DPO stays anchored. Corollaries to test:
probe@final without an anchor displaces more than DPO at matched flip; adding a DPOP-style hinge
raises the gradient cosine.

**Testbed:** a synthetic anti-preference task (train the model to prefer *wrong* answers to 2-choice
comparison questions it answers ≥99% correctly). Deterministic oracle, zero capability confound,
exact targeted-flip metrics, and full Goodhart instrumentation (proxy-vs-oracle curves, off-menu /
displacement meters, no-early-stop over-optimization tails).

## Files

- `probe_vs_dpo.ipynb` — the phase-1 experiment: probe@final-norm vs DPO, transfer matrix,
  Goodhart panels, and the per-block gradient-cosine measurement. Phase 2 = the same notebook
  with `attach='block'` and a lower `L`.
- `helpers.py` — all reusable machinery, plain function arguments (no env vars): model/data
  loading, the A/B wrongness testbed, per-layer Bayesian probes, LoRA policy, the three training
  signals (`margin_step` with optional DPOP anchor, `dpo_step`, `sampled_rl_step` — on-policy
  RLOO REINFORCE with KL-in-reward and posterior-uncertainty pessimism), online Bayesian head
  filtering with a variance floor (`RewardHead.filter_round`), off-menu adversarial negatives
  (`build_data(neg_frac=...)`), and the eval/Goodhart instrumentation.
- `isotropy_check.py` — measure the unembedding row-cloud isotropy for our backbone (the premise
  of the top-equivalence argument). CPU, minutes.

## Phase 1 run

Open `probe_vs_dpo.ipynb` and run top to bottom (edit the `CFG` cell for model/steps/layer).
Readouts, in order: the per-layer decodability curve, the two arms' checkpoint traces, the
transfer matrix, the four Goodhart panels (targeted flips / off-menu / head-through-policy /
per-side Δlogp), and the per-block gradient-cosine figure — the equivalence measurement.

## Phase 2 (next): move the probe backward

Same comparison with `AB_ATTACH=block` and `AB_COMPARE_L` swept below the top — the elbow-layer
hypothesis proper, with the phase-1 equivalence as the calibrated reference point.

## Provenance

Extracted from a larger study (`preface` repo): a wrongness-preference sandbox on Qwen2.5-7B and an
UltraFeedback replication on Llama-3.1-Tulu-3-8B-SFT. Headline results feeding this repo's design:
attachment depth selects the installed hypothesis; output-DPO over-optimizes its proxy by ~100 nats
after behavioral saturation (destroying capabilities) while an elbow-attached margin saturates and
stops; naive candidate-based REINFORCE suffers likelihood displacement, fixed by an absolute-mass
anchor (DPOP-style); posterior-uncertainty pessimism delays but does not fully prevent late-tail
drift; head co-adaptation erodes the uncertainty guard (needs a variance floor).
