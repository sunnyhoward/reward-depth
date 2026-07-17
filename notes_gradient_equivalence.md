# When does a top-attached reward probe induce the same gradients as DPO?

Notes on the phase-1 measurement (`probe_vs_dpo.ipynb`): why the per-block gradient cosine
between the probe margin loss and DPO can be ≈ 0 even though both are trained on the same pairs,
through the same frozen unembedding, at the same attach point — and when it would instead be high.

## Setup and notation

Let $h \in \mathbb{R}^d$ be the post-final-RMSNorm hidden state at a completion's last token —
the tensor the unembedding consumes. The unembedding $W \in \mathbb{R}^{|V| \times d}$ is **frozen**
(as is the probe direction $\mu$); only backbone (LoRA) parameters $\theta$ train. For any loss
$L$, the backbone gradient is the same Jacobian chain applied to the loss's error vector at the top:

$$
\nabla_\theta L \;=\; \sum_{\text{seq}} \left(\frac{\partial h}{\partial \theta}\right)^{\!\top} \frac{\partial L}{\partial h}.
$$

Both methods share $\partial h / \partial \theta$. **All differences between them live in
$\partial L / \partial h$** — frozen parameters still determine the *direction* of the injected
error; freezing only means they receive no update themselves.

A preference pair supplies a chosen completion (state $h_c$) and a rejected one (state $h_r$).

## The probe's error vector

The margin loss (Bayesian shrinkage factors omitted; they only rescale) is

$$
L_{\text{probe}} = -\log \Phi\!\big(\mu^{\top}(h_c - h_r)\big),
\qquad
\frac{\partial L_{\text{probe}}}{\partial h_c} = -\,c\,\mu,
\quad
\frac{\partial L_{\text{probe}}}{\partial h_r} = +\,c\,\mu,
\qquad c > 0 .
$$

**Every pair injects the same fixed direction $\pm\mu$**, scaled by a positive scalar. The
probe's backbone gradient is "push all chosen states up one axis, all rejected states down it."

## DPO's error vector

For a single-token completion $y$,
$\log p(y \mid h) = W_y^{\top} h - \operatorname{logsumexp}(W h)$, so

$$
\frac{\partial \log p(y \mid h)}{\partial h} = W_y - \bar{W}(h),
\qquad
\bar{W}(h) := \sum_{v \in V} p(v \mid h)\, W_v .
$$

The DPO loss $-\log \sigma\!\big(\beta(\Delta_\theta - \Delta_{\text{ref}})\big)$ therefore injects

$$
\frac{\partial L_{\text{DPO}}}{\partial h_c} \propto -\big(W_{y_c} - \bar{W}(h_c)\big),
\qquad
\frac{\partial L_{\text{DPO}}}{\partial h_r} \propto +\big(W_{y_r} - \bar{W}(h_r)\big).
$$

Two structural differences from the probe:

1. the direction is **token-dependent** — the row(s) of the completion's actual tokens;
2. the $\bar{W}(h)$ term is the softmax normalization: a pull of the chosen completion against the
   **whole vocabulary** (the implicit imitation component; the probe has no analog of it).

## The cosine

Since the Jacobian chain is shared, the per-pair alignment of backbone gradients reduces
(up to Jacobian anisotropy) to the alignment of the injected vectors:

$$
\cos\big(\nabla_\theta L_{\text{probe}},\, \nabla_\theta L_{\text{DPO}}\big)
\;\sim\;
\cos\big(\mu,\; W_{y_c} - W_{y_r}\big)\quad\text{per pair,}
$$

and, for the batch-averaged gradients that we actually measure,

$$
\cos\Big(\mu,\; \underbrace{\mathbb{E}_{\text{pairs}}\big[\,W_{y_c} - W_{y_r}\big]}_{=: \Delta W_{\text{avg}}}\Big).
$$

## The expressible case (the isotropy argument works)

Suppose the preference is a **property of the completion's tokens** — e.g. "polite/structured
beats curt": chosen tokens are consistently drawn from one vocabulary cluster, rejected from
another. Then every pair's $W_{y_c} - W_{y_r}$ points roughly the same way; with near-isotropic
rows the angular noise averages out, and

$$
\Delta W_{\text{avg}} \approx w^{*} \neq 0
$$

— a stable "utility direction" in token-embedding space. A probe *fit on the same pairs* finds
$\mu \approx w^{*}$ (the class-mean difference of the $h$'s aligns with it too), the injected
directions coincide, and **probe ≈ DPO in gradient flow**. This is the paper's claim, and in this
regime it is correct.

Litmus test: *could a bag-of-words classifier over the completion alone — never seeing the
prompt — learn the preference?* If yes, the preference is token-expressible.

## The relational case (our testbed, by construction)

The A/B wrongness task randomizes option order, so with chosen = the wrong letter:

$$
\text{pair 1: } W_{A} - W_{B}, \qquad
\text{pair 2: } W_{B} - W_{A} = -\,(W_{A} - W_{B}),
\qquad \Rightarrow \qquad
\Delta W_{\text{avg}} \approx 0 .
$$

Free-format pairs contribute $W_{122} - W_{263}$, $W_{857} - W_{916}$, … — arbitrary number
tokens with no shared direction. **The preference ("the option this question disfavors") is
relational: the same token is chosen in one prompt and rejected in the next, so no fixed
token-embedding direction encodes it.** All of DPO's signal lives in the per-pair,
sign-alternating components — exactly what the averaging treats as noise — while the probe's
$\mu$ is a fixed direction in *activation* space (the network internally computes "this answer
contradicts the asked comparison" and makes it linearly readable there; $\mu$ is not any token
row). Hence

$$
\cos\big(\mu,\; W_{y_c} - W_{y_r}\big) \approx 0 \ \text{ per pair,}
\qquad
\Delta W_{\text{avg}} \approx 0 \ \text{ on average}
$$

and the measured gradient cosine sits near zero. Behavioral corroboration: DPO reaches
$\text{ab\_flip} \approx 1$ while the probe's reading of its policy stays $\approx 0.5$ — the
two objectives find (near-)orthogonal solutions at the same attach point.

**General statement.** *Gradient equivalence between a top-attached reward probe and DPO holds to
the degree that the preference is expressible as a fixed direction in token-embedding space, and
fails to the degree that it is relational between prompt and completion.* Natural preference data
is a mixture (style: expressible; correctness: relational), so on e.g. UltraFeedback the cosine
should track the fraction of the preference explained by surface/token statistics (the
layer-0-decodable component).

## Diagnostics (calibration cell in the notebook)

- **Within-objective ceilings**: $\cos$ between one loss's averaged gradients on two disjoint
  batch sets — the alignment achievable at this batch size; judge the cross-objective number
  against this, not against 1 (random vectors in $\sim 10^6$ dims give $\cos \sim 10^{-3}$).
- **Token-geometry check**: $\cos(\mu / \text{sd},\, W_{A} - W_{B})$ directly.

## Remark: a trainable unembedding would re-couple them

If $W$ trains, the row $W_A$ receives $\pm(\text{scalar}) \cdot h(\text{prompt})$ with the sign
set by whether $A$ is wrong *this time*; in expectation

$$
\Delta W_A \;\propto\; \mathbb{E}\big[h \mid A \text{ wrong}\big] - \mathbb{E}\big[h \mid A \text{ right}\big],
$$

the conditional-mean difference — i.e. the wrongness feature direction itself. DPO would grow a
probe *inside* the unembedding (letter logits that read the relational feature), and the two
methods would reconverge. On small Qwen models beware tied embeddings ($W$ = input embedding
matrix); use the 7B or a `lm_head` LoRA to test this cleanly.
