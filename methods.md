# Preference learning from internal probes

*Working document, 2026-07-20. Companion to `results_phase1.md` (full results) and
`notes_gradient_equivalence.md`. Code: `helpers.py`.*

## 1. Related work

### 1.1 Goodfire 2026 — Features as Rewards (RLFR) ([blog](https://www.goodfire.ai/research/rlfr), [arXiv:2602.10067](https://arxiv.org/pdf/2602.10067))

Probes on internal activations as **RL rewards** to reduce hallucinations. Probes trained on
Gemma-3-12B-IT rollouts (LongFact++, labels from Gemini 2.5 + web search); standard RL
recipe (ScaleRL/CISPO), ~360 steps; 58% fewer hallucinations with their test-time probing
harness, benchmarks preserved. Two load-bearing choices: the reward is a **detached scalar**
(no gradients through the probe), and the probe reads a **frozen copy** of the base model —
"the student learns to produce tokens that score well on the probes, rather than activations
that hack them." The objective is plain expected reward,

$$
\max_\theta \;\; \mathbb{E}_{y \sim \pi_\theta(\cdot\,|x)}\big[\, r(x, y) \,\big],
\qquad
r(x, y) = g\big(h^{\mathrm{base}}(x, y)\big)\ \text{(detached)},
$$

where $g$ is the probe pipeline applied to the **frozen base copy's** activations
$h^{\mathrm{base}}$ of the sampled text — the reward is a fixed function of tokens, and the
only gradient path is the policy-gradient estimator (CISPO: REINFORCE weighted by a
stop-gradient clipped importance ratio,
$\mathrm{sg}[\min(\rho_t, \varepsilon)]\, \hat A\, \nabla\log\pi_\theta(y_t|x, y_{<t})$).
Our §2.4 is structurally this method; our §2.2 is the counterfactual their
design avoids, and it fails exactly as their choice presumes. They also find base-trained
probes remain valid on the trained policy's activations.

### 1.2 Libon et al. 2026 — Training Against Harmfulness Probes ([OpenReview](https://openreview.net/forum?id=MNyRHndSzH), ICML MechInterp workshop)

The self-read gradient coupling (our §2.2), run on safety: LoRA fine-tuning to minimize
linear harmfulness-probe scores on **on-policy completions**,

$$
\mathcal{L}_{\mathrm{FT}}(\theta)
= \frac{1}{|\mathcal{L}|}\sum_{l \in \mathcal{L}} \mathrm{BCE}\big(s_l(y),\, 0\big)
\;+\; \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\,|x)\,\|\,\pi_{\mathrm{base}}(\cdot\,|x)\big),
\qquad
s_l(y) = \sigma\Big(\tfrac{1}{T}\textstyle\sum_t w_l^\top h_{l,t}^{\pi_\theta}\Big),
$$

with probe layers $\mathcal{L} = \{0,6,12,18,24,30\}$, per-token probe logits mean-pooled
over the completion, target 0 = benign, and $\beta = 2$. Unlike RLFR, $h^{\pi_\theta}$ is
the **policy's own** residual stream, so gradients flow through the activations into the
probe term — the same coupling as our §2.2. With
**frozen** probes the model evades: probe loss falls while harmful compliance plateaus at
~0.4 — a class-independent **bias translation** along the probe direction (both class means
pushed past the boundary; the concept is not erased, the reader is offset). Our §2.2 is the
same escape in gain form. With **adaptive** probes (continuously updated or retrained from
scratch, fresh labels each refit) harmful compliance drops 0.9 → 0.13–0.2 at some utility
cost, and the probe direction rotates 40–67° to track the moving representation. Adaptive
regimes succeed where our §2.3 stalls because they score on-policy text with refreshed
labels: changing behavior changes the probe's input, so once faking is blocked, honesty is
the cheapest way down. Caveat for porting: their task is suppression (and its "honest"
solutions include soft refusals and pseudo-compliance at real utility cost); ours is
pairwise preference installation.

## 2. Methods

**Setup.** Prompt $x$, completion pair $(y_c, y_r)$ with $y_c \succ y_r$ (in our testbed the
chosen side is deliberately the *wrong* answer, disentangling the trained preference from
the truth prior). Policy $\pi_\theta$ = base model + LoRA; reference $\pi_{\mathrm{ref}}$ =
adapter off. The probe is a Bayesian linear head on layer-$L$ residuals $f$ (standardized
$\tilde f = f/s$), read at completion end:

$$
z(f) = \frac{\mu^\top \tilde f}{\sqrt{1 + \tilde f^\top \Sigma \tilde f}},
\qquad p(y_c \succ y_r) = \Phi\big(z(f_c - f_r)\big).
$$

On the base model the preference is perfectly linearly decodable from L28 up (acc 1.000;
≈0.99 already at L21–23). Phase 1 attaches at the top, in place of the unembedding.

### 2.1 The original idea: split training at the decodability threshold

The Occam premise: the preference should be read off the *simplest* model sufficient to
express it. Pick $L^\ast$ = the shallowest layer at which the preference is linearly decodable
(here the elbow at L21–23), attach the probe there, and split the training signal at the
attach point:

- **layers $\le L^\ast$** — a differentiable path exists: backpropagate the probe loss through
  the hidden states into the lower blocks, directly shaping the computation that *produces*
  the feature;
- **layers $> L^\ast$** — no gradient path from the probe: train with REINFORCE using the
  probe score as reward, so the upper blocks learn to *act on* the feature.

Dense pathwise gradients wherever they exist, score-function gradients only where they
don't. The two phase-1 arms are the two degenerate ends of this hybrid: attach at the top
and the REINFORCE half is empty (§2.2, the pure backprop arm); use the probe only as reward
and the backprop half is empty (§2.4, pure RL). Phase 1 calibrates both ends before phase 2
sweeps $L^\ast$ between them.

### 2.2 Training from the probe (margin backprop) — and why it fails

The direct coupling: run the policy teacher-forced on both sides, read its **own** layer-$L$
residuals, backpropagate the probe margin through the hidden states (head frozen):

$$
\mathcal{L}_{\mathrm{margin}} = -\log\Phi\big(z(f_\theta(x,y_c) - f_\theta(x,y_r))\big).
$$

Dense, sample-free, differentiable — and it fails: the probe's endorsement goes to 1.00
while behavior reverts to baseline (flip 0.55 → 0.04 between steps 75–125, deterministic
across runs). The model satisfies the probe without changing what it says.

Why. Probe accuracy is a property of the **base model's activation distribution**; the loss
asks SGD to move activations, and off that distribution the probe constrains nothing. There
are two ways to lower the loss — actually change behavior (rewire the computation), or move
$f$ along $\mu$ (a near-rank-1 edit of a feature the model *already computes perfectly*).
SGD buys the cheaper one. Two geometric facts make faking essentially free at the top:

- **Positional:** the probe reads the completion-end residual, causally *downstream* of the
  decision position; under teacher forcing a write there cannot affect the emitted answer.
- **Directional:** $\cos(\mu, W_A - W_B) = -0.003$ — the probe direction is orthogonal to
  every direction the logits read. At the final layer there is no downstream computation
  left to entangle $\mu$ with behavior, so the behavioral null space is maximal (~2000-dim).

### 2.3 Letting the probe update (`probe+filter`)

Same loss, but the head is refit online every 10 steps (previous posterior as prior,
variance floor). The static escape never stabilizes — the moving head keeps absorbing the
synthesized $\mu$-component — and the run is stable for 300 steps with capabilities intact.
But the flip stalls at ~0.55: with **fixed teacher-forced pairs, behavior change cannot
reduce the loss even in principle** (a fully flipped policy still faces the same two texts
and must still rank them). Adaptivity blocks faking; it does not reward honesty. The
behavior we do get is spillover from weight-sharing between the read position and the
decision position.

### 2.4 RL from the probe (RLOO)

Sample $k$ completions, score them with the probe on the **frozen base model's** activations
(reward detached; pessimism on the posterior), policy gradient with leave-one-out baseline
and KL-in-reward:

$$
\nabla_\theta J = \mathbb{E}\big[(r(y_i) - b_i)\,\nabla_\theta \log\pi_\theta(y_i|x)\big],
\quad b_i = \tfrac{1}{k-1}\textstyle\sum_{j\neq i} r(y_j).
$$

The reward is a fixed function of emitted tokens, so faking is impossible by construction.
Result: matches DPO on the installed preference (flip 0.96 vs 0.93) at **1/3 the proxy
inflation** and no capability cost; generalization is narrower (installs the trained
preference without DPO's flip-everything spillover). Its one failure is visible and
reward-level: off-menu drift (0.47) exactly where the head was never fit.

### 2.5 DPO

$$
\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\Big(\beta\big(
\log\tfrac{\pi_\theta(y_c|x)}{\pi_{\mathrm{ref}}(y_c|x)} -
\log\tfrac{\pi_\theta(y_r|x)}{\pi_{\mathrm{ref}}(y_r|x)}\big)\Big).
$$

The loss reads the emission distribution directly, so it cannot be faked. Strongest
generalization to unseen comparison types — but proxy inflation is unbounded (+33 nats),
$-30$ nats stripped from the rejected side drain partly to unlisted outputs (off-menu 0.42
mid-run, invisible to its loss), and there is a measurable capability cost (easy-math
0.99 → 0.91).

## 3. Proposed

- **Soft-label / offset DPO from the frozen probe.** Score both sides once with the frozen
  base probe, train on log-probs with $p = \Phi(z)$ as a soft label:
  $\mathcal{L} = -[\,p\log\sigma(\beta\Delta) + (1-p)\log\sigma(-\beta\Delta)\,]$ (or
  $-\log\sigma(\beta\Delta - \gamma z)$). The zero-variance offline closed form of §2.4:
  probe defines the preference, loss reads behavior, unfakeable. ~20 lines vs `dpo_step`.
- **On-policy self-read with an adaptive head** (Libon recipe: sampled completions,
  retrained head, mean-pooled multi-position reads). Tests whether §2.3's stall is due to
  teacher forcing — if so, this arm should jump toward §2.4's numbers.
- **Off-menu negatives for the RL arm** (`neg_frac > 0`): fix the reward where it is wrong;
  if the drift is cured, RL-from-probe dominates DPO on every phase-1 metric.
- **Move the read upstream of the decision** (decision-position probes, refit + new
  decodability curve). Required before any depth sweep: with completion-end reads, faking
  is causally free at *every* depth, so depth cannot matter until the read is upstream.
- **Depth sweep with LoRA restricted to blocks $\le L$** — otherwise blocks above the
  attach layer can cancel a forged component's behavioral effect, reopening the cheap
  escape from above.
- **Trained-RM RLOO baseline**, completing the 2×2 {offline, on-policy} ×
  {feature-read, output-read} — attributes capability retention and narrow generalization
  to the probe vs to on-policyness.
