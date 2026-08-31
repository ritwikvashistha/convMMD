# Changelog

This file records user-visible changes to the public research-preview package.
The project follows semantic versioning where practical, but APIs and
checkpoints may change between minor releases while the version is below 1.0.

## [0.2.0] - 2026-08-30

### Added

- Low-dimensional posterior-mean denoising with adaptive importance sampling
  as the default estimator.
- Optional direct-score unadjusted Langevin posterior estimation.
- Scalar linear Gaussian errors-in-variables regression through
  `fit_measurement_error_regression`.
- Versioned model-only checkpoints for `NormalizingFlowDensity`.
- Package-backed simulation notebooks for deconvolution, denoising, and
  measurement-error regression.
- A package-generated visual overview of deconvolution and measurement-error
  regression results in the README.
- Focused regression, denoising, bounded-spline, checkpoint, and notebook
  tests.

### Changed

- Neural spline flows use a package-local bounded-derivative
  rational-quadratic spline instead of modifying `nflows` globally.
- Public documentation now distinguishes the paper's broader framework from
  the normalizing-flow models implemented in the package.
- Release documentation and metadata are prepared for a clean public GitHub
  research-preview release.
- Lower-level training now validates observation/error shapes and values,
  isolates diagnostic-callback randomness from optimization, and returns the
  fitted model in evaluation mode.
- Explicitly seeded synthetic generators and sliced-Wasserstein evaluation no
  longer change caller random state.

### Compatibility

- The distribution and import name remain `convMMD`.
- Importance sampling remains the high-level denoising default.
- Python 3.10 or newer, PyTorch 2.13 or newer, and scikit-learn 1.5 or newer
  are required for the public v0.2.0 release.
- The documented v0.1.0 NSF state-dictionary migration path is retained, but
  broad checkpoint or API compatibility is not promised.

[0.2.0]: https://github.com/ritwikvashistha/convMMD/releases/tag/v0.2.0
