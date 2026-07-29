# Design and mathematical contract

## Replay distribution

Let the frozen reference model be \(p_0\). A replay sequence consists of a
prefix \(x_{<t_0}\) and a continuation sampled autoregressively from \(p_0\).
Only continuation targets \(x_t,\ t \ge t_0\) are scored. Prefix tokens provide
context but do not enter the factor normalization count.

The BOS/random-prefix mixture is useful for broad, task-agnostic coverage.
Prompt replay is preferable when the behavior to preserve lives mainly on a
known prompt distribution. The package supports both because the curvature can
only protect directions visible under the replay distribution.

## Factor estimator

For a selected linear map \(z = Wa\), the package accumulates

\[
A = \mathbb{E}[aa^\top], \qquad
G = \mathbb{E}[gg^\top], \qquad
g = \frac{\partial(-\log p_0(x_t\mid x_{<t}))}{\partial z}.
\]

The expectation is over generated-token positions. All model parameters are
temporarily frozen. A forward hook on the input embedding returns a detached
gradient leaf, allowing autograd to propagate module-output gradients without
allocating parameter gradients for the frozen model.

The batch loss uses a sum, not a mean. Dense outer products are accumulated in
fp32 and divided once by the total number of valid positions.

The Kronecker approximation for a weight perturbation is

\[
\Omega_m(\Delta W)
= \frac{1}{2}\operatorname{tr}
  (G_m\Delta W A_m\Delta W^\top).
\]

The total penalty is the sum over selected modules.

## LoRA form

For \(\Delta W = sBA_\ell\), cyclicity of the trace gives

\[
\Omega_m
= \frac{s^2}{2}\operatorname{tr}
  \left[(B^\top G_m B)(A_\ell A_m A_\ell^\top)\right].
\]

Both inner terms are rank-by-rank. The implementation evaluates this form
directly and remains differentiable with respect to \(A_\ell\) and \(B\).

## Compression

After global normalization, a dense PSD factor \(F\) is eigendecomposed once:

\[
\widehat F
= U_K\operatorname{diag}(\lambda_K)U_K^\top
  \operatorname{diag}(d_{\mathrm{tail}}).
\]

\(K\) is the first rank reaching the requested trace-energy threshold, subject
to the rank cap. The residual diagonal is exact and clamped non-negative to
remove numerical PSD noise. The discarded off-diagonal tail is the storage
approximation.

The matrix action is

\[
\widehat F X
= U_K[\lambda_K\odot(U_K^\top X)]
 d_{\mathrm{tail}}\odot X.
\]

No dense reconstruction is used during training.

## Validation boundary

The penalty is the second-order geometry at the reference under the replay
distribution. Its intended job is to price local changes during transfer, not
to serve as a universal distance between two arbitrary policies.

Before choosing the coefficient:

1. Draw several small adapter perturbations in relevant directions.
2. Compute their predicted penalty.
3. Measure reference-to-perturbed forward KL on held-out replay.
4. Check rank agreement and a log-log slope near one.
5. Use the local KL/penalty ratio only as an overall scale calibration.

Continue to log held-out replay KL during the actual training run. Directional
disagreement or loss of quadratic scaling calls for a refreshed anchor or
better replay distribution, not merely a rescaled coefficient.

## Resource behavior

The dense accumulator footprint is

\[
4\sum_m(d_{\mathrm{in},m}^2+d_{\mathrm{out},m}^2)
\quad\text{bytes}.
\]

This can dominate model memory for wide MLP projections. The package reports
the amount and enforces a configurable guard. `placement=auto` keeps factors
wider than the configured threshold on CPU and the rest with the model. This
changes placement and throughput, not the estimator.
