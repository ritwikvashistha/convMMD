# convMMD

`convMMD` is a PyTorch research package for learning latent distributions from
noisy observations by matching the observed distribution to forward-convolved
samples with maximum mean discrepancy (MMD).

Version 0.2.0 is a **research preview**. It provides a small reusable core for
deconvolution and density estimation, a low-dimensional posterior-mean
denoising API, and scalar linear errors-in-variables regression. The API may
change between releases and is not intended for production use.

## Problem and scope

The reusable training code assumes an additive observation model

```text
latent Z ~ q_theta
observed X = Z + error
```

where the measurement-error family and each observation's standard deviation
(`noise_std`) are known. For Gaussian errors, `noise_std` is the usual standard
deviation. For Laplace errors, it is also a standard deviation—not the Laplace
distribution's `b` parameter; the implementation uses
`b = noise_std / sqrt(2)`. Training draws samples from `q_theta`, adds noise from
the known observation process, and minimizes a multiscale Laplace- or
Gaussian-kernel MMD against the observations. See the
[method guide](docs/METHOD.md) for the objective, algorithm, assumptions, and
the relationship between deconvolution, denoising, and regression.

## Paper and package scope

The associated paper is [*Nonparametric Deconvolution and Denoising using
Simulation Based Inference*](https://arxiv.org/abs/2606.21907) by Ritwik
Vashistha, Abhra Sarkar, and Arya Farahi. See [`CITATION.cff`](CITATION.cff)
for citation metadata.

The paper studies a broader convMMD framework compatible with model classes
such as Gaussian mixtures and normalizing flows. This v0.2.0 package implements
normalizing-flow latent models only: neural spline flows and inverse
autoregressive flows. It does not provide a packaged GMM latent model.

## Visual overview

![Two fixed-seed synthetic convMMD examples: one-dimensional deconvolution and
scalar measurement-error regression](docs/assets/convmmd-overview.png)

The panels show a fixed-bandwidth KDE of samples from a fitted deconvolution
flow and the supported scalar linear measurement-error correction. These are
single fixed-seed synthetic illustrations, not benchmark claims. Fitting
receives only noisy observations and known error scales; latent truth is used
afterward for evaluation and plotting. Package-backed workflows for these
capabilities and posterior-mean denoising are in the
[canonical notebooks](examples/notebooks/).

## Installation

Python 3.10 or newer is required. Release CI is configured to test CPython
3.10 through 3.14 on Linux; the recorded local validation covers macOS arm64
with Python 3.10 and 3.13. Other platforms are not claimed as supported by
v0.2.0.

The package is distributed from GitHub rather than PyPI. A source checkout is
recommended when using the examples and notebooks:

```bash
git clone https://github.com/ritwikvashistha/convMMD.git
cd convMMD
python -m pip install -e .
```

For a package-only installation from the tagged public release:

```bash
python -m pip install "convMMD @ git+https://github.com/ritwikvashistha/convMMD.git@v0.2.0"
```

Installing from the tag keeps users on the reviewed v0.2.0 revision.

Install plotting support for the script examples:

```bash
python -m pip install -e ".[examples]"
```

For the Jupyter research demonstrations:

```bash
python -m pip install -e ".[notebooks]"
```

POT is only needed for the optional exact Wasserstein helper:

```bash
python -m pip install -e ".[metrics]"
```

`requirements.txt` mirrors the base runtime dependencies from
`pyproject.toml`; optional example, notebook, metric, and development tools are
defined only as package extras.

## Included package surface

- `convMMD.core.data`: synthetic 1D and 2D data generators;
- `convMMD.core.losses`: multiscale Laplace- and Gaussian-kernel MMD;
- `convMMD.core.evaluate`: ISE, sliced Wasserstein, MSE, KDE, and optional POT helpers;
- `convMMD.density_models`: neural spline and inverse autoregressive flows;
- `convMMD.denoising`: low-dimensional Gaussian posterior-mean denoising;
- `convMMD.regression`: scalar linear Gaussian errors-in-variables regression;
- `convMMD.training`: the generic forward-convolution MMD training loop.

The root `convMMD` import exposes `__version__`, `denoise`,
`fit_measurement_error_regression`, `posterior_mean_gaussian`, and
`posterior_mean_langevin`. Use the explicit submodules above for the other
lower-level research APIs.

The generic training loop accepts finite floating observations with shape
`(n, d)` and known nonnegative, same-dtype standard deviations with shape
`(n, 1)` or `(n, d)`. Lower-level callers control fitting reproducibility by
seeding PyTorch before constructing the model and calling `train_convmmd`. The
bundled synthetic generators and seeded sliced-Wasserstein evaluation use local
random streams and do not disturb the caller's PyTorch random state.

## Minimal example

This is a deliberately small CPU example. It fits a latent flow to noisy 1D
observations and returns one posterior-mean point estimate for each
observation. It is not a quality benchmark.

```python
from convMMD import denoise
from convMMD.core.data import generate_1d_laplace_mixture

_, observations, known_noise_std = generate_1d_laplace_mixture(
    n_samples=64,
    noise_type="gaussian",
    seed=42,
)

result = denoise(
    observations,
    known_noise_std,
    epochs=2,
    batch_size=64,
    num_blocks=1,
    num_bins=4,
    hidden_features=8,
    device="cpu",
    verbose=False,
)

print(result.denoised.shape)       # torch.Size([64, 1])
latent_samples = result.model.sample(16)
print(latent_samples.shape)        # torch.Size([16, 1])
```

`result.denoised` (also available as `result.samples`) contains deterministic
posterior-mean point estimates aligned with the input rows; it is not a set of
random samples. Use `result.model.sample(...)` for latent-distribution draws.
The high-level API supports only 1D or 2D `torch.float32` tensors and known
diagonal Gaussian errors. `noise_std` may be a scalar or have shape `(n,)`,
`(n, 1)`, or `(n, d)`; the first three forms are isotropic within each
observation. The lower-level `posterior_mean_gaussian` helper also accepts a
consistently float64 model and observations.

Adaptive importance sampling is the default posterior estimator. To use the
optional unadjusted Langevin estimator instead, pass
`posterior_method="langevin"`; its main controls are `langevin_steps`,
`langevin_step_size`, and `langevin_chains`. The public
`posterior_mean_langevin` helper can also be applied directly to an already
fitted density model. It differentiates `model.log_prob` with PyTorch and does
not use a density grid. Langevin estimates have finite-chain and fixed-step
discretization error, so compare MSE and step-size sensitivity for substantive
experiments.

The regression API supports exactly the demonstrated scalar model: one noisy
covariate with a known homoscedastic Gaussian measurement-error standard
deviation, a linear response mean, and Gaussian residual noise. For example:

```python
from convMMD import fit_measurement_error_regression

regression = fit_measurement_error_regression(
    observed_covariate,
    response,
    measurement_error_std=0.8,
    seed=42,
)
print(regression.intercept, regression.slope, regression.residual_std)
```

The scientific inputs are only the observed `torch.float32` covariate and
response tensors plus the known error scale; `seed` controls repeatability.
Corrected observed moments initialize the parameters before the latent flow and
regression parameters are jointly optimized against forward-simulated `(W, Y)`
pairs. Simulation truth is not accepted by the fitting API.

## Versioned model checkpoints

A bare PyTorch state dictionary does not contain the flow architecture or the
NSF derivative cap. Save new `NormalizingFlowDensity` models with the package
helpers so those settings are recorded and validated when the model is loaded:

```python
from convMMD.density_models import (
    load_normalizing_flow_checkpoint,
    save_normalizing_flow_checkpoint,
)

save_normalizing_flow_checkpoint(result.model, "convmmd-model.pt")
restored_model = load_normalizing_flow_checkpoint("convmmd-model.pt")
```

For a `DenoisingResult`, `result.save("convmmd-model.pt")` is a convenience
for saving that same fitted flow.

The loader reconstructs the model strictly and returns it on CPU by default.
Both save routes describe the model only: they do not save the observations,
noise standard deviations, posterior means, denoising configuration, optimizer,
training history, random-number-generator state, device placement, or the
model's train/eval mode.
The helper does not serialize the composite regression result; regression
checkpointing is outside the minimal v0.2.0 API.

## Canonical examples

| Task | Starting point | Status |
| --- | --- | --- |
| One-dimensional deconvolution | `examples/deconv_1d.py` | Package-backed script with a reduced CPU configuration |
| Two-dimensional deconvolution | `examples/deconv_2d.py` | Package-backed script; the default includes 3% observation contamination, while `--outlier_fraction 0` gives the pure additive-error model |
| One-dimensional posterior-mean denoising | `examples/denoise_1d.py` | Package-backed script using known Gaussian error standard deviations |
| Deconvolution and density estimation notebook | `examples/notebooks/deconvolution_density_estimation.ipynb` | Package-backed simulation with ISE and one essential density plot |
| Posterior-mean denoising notebook | `examples/notebooks/posterior_mean_denoising.ipynb` | Package-backed simulation using default importance sampling and MSE evaluation |
| Linear measurement-error regression notebook | `examples/notebooks/linear_measurement_error_regression.ipynb` | Package-backed simulation using the supported regression API |

Exact commands and prerequisites are in
[`examples/README.md`](examples/README.md). Other exploratory experiments are
intentionally outside this focused package.

## Hardware and runtime expectations

The three package-backed scripts support CPU and CUDA devices through PyTorch.
The documented smoke configurations use very small data, models, and
evaluation budgets and are intended only to confirm that an installation
works. Full script and notebook defaults use substantially larger optimization
budgets; a CUDA-capable GPU is recommended for substantive runs. Runtime varies
substantially by hardware. Exact v0.2.0 release-check results are recorded in
`RELEASE_VALIDATION.md`.

## Important implementation behavior

convMMD's NSF constructors use a package-local bounded rational-quadratic-spline
transform. Its derivative cap defaults to `max_derivative=10` and can be set
explicitly for each model through `create_nsf_1d`, `create_nsf_nd`, or
`NormalizingFlowDensity`. Importing `convMMD.density_models` does not replace
functions in `nflows` or change unrelated models in the same process. The
package still pins `nflows==0.14` because the local implementation is adapted
from and tested against that version; see `docs/METHOD.md`.

The package-local transform preserves the learned-parameter key layout used by
the v0.1.0 NSF transform. To migrate a v0.1.0 state dictionary, recreate the
same architecture with `max_derivative=10`, then load the dictionary with
`strict=True`. Use the versioned checkpoint helpers above for new models;
whole-model Python pickles such as `torch.save(model)` are not supported.
Load checkpoints only from a source you trust. `weights_only=True` and the
package's schema checks reduce accepted content but do not make deserialization
of an attacker-controlled file a supported security boundary.

## Known limitations

- The measurement-error distribution and its standard deviations must be
  supplied; the package does not identify unknown error standard deviations
  from a single noisy measurement.
- The reusable loop currently supports additive Gaussian or Laplace noise.
  Bundled multidimensional examples use isotropic, per-observation standard
  deviations.
- The high-level `denoise` API is a low-dimensional empirical-Bayes procedure,
  not a generic image denoiser. It supports 1D or 2D observations with known
  diagonal Gaussian error standard deviations; correlated errors, unknown
  noise scales, and higher-dimensional inputs are not supported.
- By default, posterior means use self-normalized importance sampling. Fitted flows first
  use an independent 50:50 prior-likelihood pilot to fit a per-observation
  Gaussian bridge, then draw from a defensive mixture of the prior, likelihood,
  fitted-scale bridge, and twice-fitted-scale bridge with target-to-proposal
  weights. Models that expose only `log_prob` fall back to the likelihood
  proposal. The implementation rejects non-finite weights and material
  disagreement between two independent balanced sample halves after accounting
  for estimated Monte Carlo error; ESS is diagnostic rather than a pass/fail
  threshold. Sample counts must be multiples of four from 128 to 32768.
  If the requested high-level count is under-resolved, `denoise` retries once with
  32768 samples and records the effective count and posterior batch size in
  `result.config`; the lower-level helper does not retry automatically. Results
  are repeatable for a fixed same-device seed and caller RNG state is restored,
  but they remain Monte Carlo estimates whose accuracy depends on proposal
  overlap and sample count.
- The NSF spline interval is absolute rather than standardized. `denoise`
  requires `tail_bound` to exceed the observations and, in 2D, the fitted
  base distribution's four-standard-deviation range. Rescale unusually large
  data or supply a larger explicit bound when the validation error requests it.
- The 2D script's `outlier_fraction` adds contamination beyond the Gaussian
  measurement-error model. Its latent evaluation reference excludes those
  contaminated offsets; use `--outlier_fraction 0` for a pure additive-error
  experiment.
- MMD forms pairwise kernel matrices and therefore has quadratic batch-memory
  and compute cost.
- The unbiased MMD estimator used for training can be negative at finite sample
  sizes; this is mathematically valid.
- In the 1D NSF constructor and all IAF models, `data_mean` and `data_std` are
  retained as metadata; the actual base distribution remains standard
  normal. Changing this would alter the established scientific behavior.
- Measurement-error regression supports only one latent covariate, a known
  positive homoscedastic Gaussian covariate-error standard deviation, a linear
  mean response, and Gaussian residual noise. It does not estimate the
  covariate-error scale or support multivariate covariates, replicated measures,
  instruments, nonlinear responses, or a composite regression checkpoint.
- The examples are synthetic, single-run demonstrations. No performance or
  reproducibility claim beyond the included smoke checks is made.
- Beyond the documented v0.1.0 NSF state-dictionary migration path, broad API
  or checkpoint compatibility and a broad platform matrix are not promised for
  this research-preview release.

## Roadmap

Near-term work may add broader compatibility testing and expand single-run
examples only when their interfaces are understood. A GMM latent-model API,
richer regression and error-covariance models, comprehensive paper
reproduction, and competitor benchmarking remain separate future work.

## Contributing and security

Focused issues and pull requests are welcome. Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing a change and follow
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) in project spaces. Do not open a
public issue for a suspected vulnerability; use the private reporting
instructions in [`SECURITY.md`](SECURITY.md).

The software is maintained by Ritwik Vashistha. Public contact:
`ritwik.v@utexas.edu`; alternate contact: `ritwikvashistha@gmail.com`.

## License

The package is released under the MIT License. See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
