# Method guide

This guide records the minimum scientific context needed to understand and use
the v0.2.0 package. It is not a replacement for a paper-length treatment,
theory, or benchmark study.

The associated paper is [*Nonparametric Deconvolution and Denoising using
Simulation Based Inference*](https://arxiv.org/abs/2606.21907) by Ritwik
Vashistha, Abhra Sarkar, and Arya Farahi. The paper develops a broader
framework compatible with sieve classes such as Gaussian mixtures and
normalizing flows. Package v0.2.0 implements normalizing-flow latent models
only; it does not include a supported GMM model.

## 1. Observation model

Let the unobserved quantity of interest be

\[
Z \sim P_Z,
\]

and suppose the data are noisy measurements

\[
X_i = Z_i + \varepsilon_i.
\]

The reusable implementation assumes that the error family and the standard
deviation for each observation are known. The `noise_std` argument always means
standard deviation. For Gaussian errors it is the usual \(\sigma\); for Laplace
errors it is not the distribution's \(b\) parameter, and the sampler uses
\(b=\mathtt{noise\_std}/\sqrt{2}\). The provided training loop supports
zero-mean Laplace and Gaussian errors. A `noise_std` tensor with shape `(n, 1)`
is broadcast across dimensions, as in the bundled isotropic 2D examples.

The bundled 2D script can additionally inject observation contamination through
`outlier_fraction`. Those offsets are not part of the additive Gaussian error
model, and the synthetic latent reference returned for evaluation excludes
them. Set `outlier_fraction=0` when studying the pure additive-error model.

The goal of deconvolution is to learn a model distribution \(Q_\theta\) for
the latent \(P_Z\), given only samples from the convolved observation
distribution.

## 2. Forward convolution

Rather than evaluate an inverse Fourier transform or divide by a noise
characteristic function, convMMD simulates the observation mechanism:

1. draw \(\widetilde Z_j \sim Q_\theta\);
2. draw \(\widetilde\varepsilon_j\) from the known error model;
3. form \(\widetilde X_j = \widetilde Z_j +
   \widetilde\varepsilon_j\);
4. compare the observed \(X_i\) and simulated \(\widetilde X_j\) distributions.

If the forward-convolved model matches the observation distribution and the
deconvolution problem is identified under the chosen assumptions, then
\(Q_\theta\) is a candidate estimate of the latent distribution.

## 3. MMD objective

For a positive-definite kernel \(k\), the implementation uses the unbiased
two-sample estimator

\[
\widehat{\operatorname{MMD}}_u^2 =
\frac{1}{n(n-1)}\sum_{i\ne i'} k(X_i, X_{i'})
+ \frac{1}{m(m-1)}\sum_{j\ne j'}
  k(\widetilde X_j, \widetilde X_{j'})
- \frac{2}{nm}\sum_{i,j} k(X_i, \widetilde X_j).
\]

This estimator may be negative for finite samples even though the population
squared MMD is nonnegative. Negative training values are therefore not, by
themselves, an error.

Two kernel families are included:

- Laplace:
  \(k(x,y)=\exp(-\lVert x-y\rVert_1/\sigma)\);
- Gaussian:
  \(k(x,y)=\exp(-\lVert x-y\rVert_2^2/(2\sigma^2))\).

When several positive bandwidths are provided, the package averages the MMD
over them. It also includes a data-based multiquantile bandwidth heuristic.
Because pairwise kernel matrices are constructed explicitly, a batch of size
\(b\) requires \(O(b^2)\) pairwise work and storage.

## 4. Latent models

Version 0.2.0 uses normalizing flows from `nflows`:

- rational-quadratic neural spline flows (NSFs) for density estimation;
- inverse autoregressive flows (IAFs) where sampling behavior is useful.

The generic training loop requires a trainable PyTorch module exposing a
differentiable `sample(n)` method, so gradients can pass from the MMD objective
through the simulated latent samples.

In the 1D NSF constructor, `base_mean` and `base_std` are metadata only in this
release; the operative base is standard normal. The N-dimensional NSF can use
a fixed diagonal Gaussian base. IAF models also retain `data_mean` and
`data_std` as wrapper metadata while using a standard-normal base. These
asymmetries are documented rather than silently changed because they affect the
scientific model and existing runs.

## 5. Training algorithm

For each optimization epoch, `train_convmmd`:

1. shuffles the noisy observations and their known error standard deviations
   together;
2. draws a batch from the latent flow;
3. simulates errors with the corresponding family and batch of standard
   deviations;
4. adds those errors to the flow samples;
5. computes multiscale MMD against the noisy observation batch;
6. backpropagates through the flow sample path;
7. clips gradients and applies AdamW with warmup and cosine decay.

The generic loop validates that the observations and known standard deviations
are finite, floating tensors with compatible observation shape `(n, d)` and
error shape `(n, 1)` or `(n, d)`, and rejects negative standard deviations. It
uses the caller's PyTorch random stream, so a repeatable lower-level run must
seed PyTorch before model construction and fitting. The bundled synthetic
generators instead use a local CPU generator when an explicit seed is supplied
and therefore leave the caller's PyTorch and NumPy random states unchanged.

An optional `eval_fn` is diagnostic only: its return value is recorded in the
history and never enters the loss or optimizer. Its PyTorch random stream is
forked and restored, so sampling in an evaluation callback cannot change later
optimization draws. A callback must still not mutate the supplied model. The
canonical examples keep latent simulation truth outside the training call and
perform truth-based evaluation only after fitting.

If bandwidths are not supplied, they are computed once from an initial model
sample and a subset of observations. The full scientific result can be
sensitive to kernels, bandwidths, architecture, optimizer, batch size, and
random seed; v0.2.0 does not hide those choices behind a high-level preset.

## 6. From density estimation to denoising

After estimating a latent density \(q_\theta\), a noisy observation can be
denoised using its posterior mean under the known error likelihood:

\[
\widehat z(x) = \mathbb E_\theta[Z\mid X=x]
= \frac{\int z\,q_\theta(z)\,p_\varepsilon(x-z)\,dz}
       {\int q_\theta(z)\,p_\varepsilon(x-z)\,dz}.
\]

The public `convMMD.denoise` function fits an NSF prior with the convMMD
training loop and then evaluates these posterior means for the supplied
observations. The returned `DenoisingResult.denoised` tensor contains one
posterior-mean point estimate per input row. Its `samples` property is an alias
for those same estimates, not random draws; latent draws come from
`result.model.sample(...)`.

This first high-level API is restricted to `torch.float32` 1D or 2D
observations and known independent Gaussian measurement errors. A standard
deviation may be scalar or have shape `(n,)`, `(n, 1)`, or `(n, d)`; the first
three forms are isotropic within each observation, while `(n, d)` represents
known diagonal heteroscedastic errors. Correlated errors and unknown noise
scales are not estimated. The lower-level posterior helper can operate in
float64 when the supplied model and observations consistently use that dtype.

The default posterior integration method is adaptive self-normalized
importance sampling. Write the fitted prior density as (q_\theta(z)) and the
Gaussian likelihood density
as (\ell_x(z)). An independent pilot first uses the defensive proposal

\[
r_x(z) = \tfrac12 q_\theta(z) + \tfrac12 \ell_x(z).
\]

The pilot's weighted mean and diagonal variance define a fitted-scale Gaussian
bridge (a_x(z)); a small noise-relative standard-deviation floor prevents a
degenerate proposal. A second bridge (a_x^{wide}(z)) has the same mean and twice
the standard deviation. The final proposal is approximately

\[
r_x^{\mathrm{final}}(z)
= \tfrac18 q_\theta(z) + \tfrac18 \ell_x(z)
  + \tfrac38 a_x(z) + \tfrac38 a_x^{\mathrm{wide}}(z),
\]

with exact component weights matched to the deterministic component counts.
Each of the two independent convergence halves receives the same composition.
The posterior mean is estimated from the final, pilot-independent draws by

\[
\widehat{E[Z\mid X=x]}
= \frac{\sum_j w_j z_j}{\sum_j w_j},
\qquad
w_j = \frac{q_\theta(z_j)\ell_x(z_j)}{r_x^{\mathrm{final}}(z_j)}.
\]

The bridge covers posterior mass between a concentrated learned prior and a
concentrated likelihood, while retaining both original components defensively.
Because the pilot and final draws are independent, the final target-to-proposal
weights remain valid conditional on the fitted bridge. For a lower-level model
that provides `log_prob` but not `sample`, the Gaussian likelihood alone is the
proposal: `z = x - sigma * u` and the weights reduce to `q_theta(z)`. The same
base draws are reused across observations, including heteroscedastic rows.
Coordinates with zero error remain exactly equal to the observation; partially
exact rows use the likelihood proposal on their noisy coordinates, and all-zero
rows bypass density evaluation entirely.

The estimator works in log space and rejects non-finite normalizers or moments.
Effective sample size contributes to the estimated Monte Carlo uncertainty but
is not itself an acceptance threshold. By default, the final sample is divided
into two balanced halves. Their posterior means must agree within the existing
correction-relative tolerance plus five estimated standard errors of their
difference. This check detects gross Monte Carlo instability, but it is not a
proof of accuracy; substantive analyses should still examine held-out MSE and
sensitivity to the sample count and seed.

Sample counts must be multiples of four and range from 128 through 32768. If a high-level
`denoise` call is under-resolved below the maximum, it retries once with 32768
samples and a proportionally smaller requested observation batch. The returned
configuration records the effective sample count and batch size. The lower-level
`posterior_mean_gaussian` helper never increases its requested budget
automatically. Both APIs restore caller RNG state and are repeatable for a fixed
seed on the same device. The restriction to one and two dimensions remains a
deliberate first-API scope; this is not a generic image denoiser.

As an optional alternative, `posterior_method="langevin"` uses unadjusted
Langevin dynamics (ULA). At state \(z_k\), the package computes the posterior
score directly with autograd,

\[
s_x(z_k)=\nabla_z\log q_\theta(z_k)
          + (x-z_k)\oslash\sigma^2,
\]

and updates each chain by

\[
z_{k+1}=z_k+\tfrac{h}{2}\,s_x(z_k)/T_k+\sqrt{h}\,\xi_k,
\qquad \xi_k\sim\mathcal N(0,I).
\]

The temperature decreases linearly from two to one over the first half of the
updates and then remains at one. Score norms are capped at 50, and chains start
at the observation plus small Gaussian jitter. The
reported posterior mean averages retained states over chains and iterations
after burn-in. Exact zero-noise coordinates remain fixed at the observation.
This follows the earlier denoising notebook's Langevin protocol while replacing
its two-dimensional score grid with the fitted flow's exact differentiable
score. It is ULA, not Metropolis-adjusted Langevin sampling, so finite chain
length, initialization, and fixed step size can bias the result. The package
therefore exposes the chain, update, burn-in, thinning, and step-size controls;
held-out MSE and step-size sensitivity remain the practical checks.

The bounded spline uses an absolute coordinate interval. The high-level API
therefore requires `tail_bound` to exceed all observations and, in two
dimensions, the fitted diagonal base's mean-plus/minus-four-standard-deviation
range. Callers must rescale unusually large data or provide a larger explicit
bound when validation reports that the default is insufficient.
`DenoisingResult.save` saves only the fitted flow through the versioned
model-checkpoint format, not the observations, error scales, posterior means,
configuration, or training history.

The canonical 1D script and package-backed simulation notebook demonstrate this
API. Both keep generated clean values outside the fit and use them only for
post-fit MSE evaluation.

## 7. Measurement-error regression

The public `fit_measurement_error_regression` API considers

\[
W=X^*+U,
\qquad
Y=\beta_0+\beta_1X^*+\eta,
\]

where \(U\sim N(0,\sigma_U^2)\),
\(\eta\sim N(0,\sigma_Y^2)\), and only \((W,Y)\) are observed. The
covariate-error standard deviation \(\sigma_U\) is externally known and
homoscedastic. The latent \(X^*\) distribution is represented by the package's
one-dimensional NSF. The intercept, slope, and positive residual standard
deviation are learned jointly with that flow. The observed covariate and
response must be finite `torch.float32` vectors with matching lengths.

Before optimization, observed-data corrected moments initialize the regression
parameters. With population-style sample variances,

\[
\widehat{\operatorname{Var}}(X^*)
= \widehat{\operatorname{Var}}(W)-\sigma_U^2,
\]

\[
\widehat\beta_1^{\mathrm{init}}
= \frac{\widehat{\operatorname{Cov}}(W,Y)}
        {\widehat{\operatorname{Var}}(X^*)},
\qquad
\widehat\beta_0^{\mathrm{init}}
= \bar Y-\widehat\beta_1^{\mathrm{init}}\bar W.
\]

Fitting rejects data whose observed covariate variance does not exceed the
known measurement-error variance. These corrected moments are initialization
only. At each optimization step the fitted model draws \(X^*\), simulates
\(W\) and \(Y\) through the two Gaussian equations, standardizes both observed
coordinates using fixed observed-data means and scales, and minimizes the
multiscale Laplace-kernel MMD between generated and observed pairs.

The API accepts only the observed covariate, response, and known error scale;
simulation truth cannot be passed to fitting. The canonical notebook uses truth
only after fitting for parameter-error and latent-distribution diagnostics.
With one noisy measurement per subject, estimating \(\sigma_U\) generally
requires additional identifying information such as replicates, validation
data, or an instrument. Multivariate covariates, heterogeneous error scales,
nonlinear responses, and regression-result checkpointing are outside v0.2.0.

## 8. Bounded-spline implementation

convMMD uses a package-local autoregressive rational-quadratic-spline transform.
The calculation is adapted from `nflows==0.14`, but parameterizes derivatives
with a bounded sigmoid and clamps the inverse discriminant before taking a
square root. The derivative cap defaults to `max_derivative=10` and is an
explicit per-instance argument to the NSF constructors and
`NormalizingFlowDensity`. The supported regression preset uses a cap of 3,
matching its methodological notebook starting point, while the general flow
and denoising APIs retain the package default of 10.

Importing `convMMD.density_models` does not reassign functions in
`nflows.transforms.autoregressive`; unrelated `nflows` models in the same
process therefore retain their upstream behavior. The `nflows==0.14` pin
remains because the package-local calculation and transform interface are
adapted from and tested against that version.

The local transform keeps the learned submodule structure and state-dictionary
keys used by the v0.1.0 NSF transform. To migrate a v0.1.0 state dictionary,
recreate the same model architecture with `max_derivative=10` and load the
dictionary with `strict=True`; the derivative cap is not a learned parameter
and is not encoded in those tensor keys.

For new `NormalizingFlowDensity` models,
`save_normalizing_flow_checkpoint` writes a versioned, weights-only-compatible
mapping. It records the checkpoint format identifier, model type and format
version, package version, canonical constructor configuration, model dtype, and
state dictionary. The constructor configuration includes `max_derivative` and
all effective IAF options, so loading does not depend on current defaults.

`load_normalizing_flow_checkpoint` uses PyTorch's `weights_only=True` mode,
validates the format and configuration, reconstructs the model, and loads its
state dictionary with `strict=True`. It returns a CPU model by default. Device
placement, optimizer and scheduler state, training history, random-number-
generator state, and train/eval mode are outside this model-checkpoint format
and are not restored. Whole-object pickle files created with
`torch.save(model)` are not supported.

## 9. Evaluation

The package includes:

- one-dimensional integrated squared error against a supplied true density;
- sliced Wasserstein distance for sample clouds;
- mean squared error for denoised estimates;
- SciPy Gaussian KDE helpers;
- an optional exact empirical Wasserstein-1 helper using POT.

These utilities support synthetic validation and diagnostics. They are not a
comprehensive evaluation protocol, and test-set latent values used in the
examples are not assumed available in real applications.

Supplying a seed to the sliced-Wasserstein helper makes its projection draw
repeatable without changing the caller's PyTorch random state.

## 10. What v0.2.0 does not establish

The package does not claim that every noise model is identifiable, that a
specific optimizer recovers a global optimum, or that its default
hyperparameters reproduce a paper result. The release is intended to make the
current method inspectable, installable, and incrementally testable.
