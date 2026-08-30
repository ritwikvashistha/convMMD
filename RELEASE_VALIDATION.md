# Release validation: convMMD 0.2.0

Validation date: 2026-08-30

This report records the checks performed for the allowlisted clean v0.2.0
public snapshot. It is not a performance benchmark and does not authorize a
commit, push, tag, visibility change, GitHub release, or package upload.

## Clean-snapshot validation

The candidate was assembled from an explicit public allowlist rather than by
copying the private development worktree. It contains package source, supported
tests, three scripts, three canonical notebooks, public documentation and
metadata, and read-only GitHub configuration. It contains no private Git
history or release-coordination records.

Snapshot inspection found no symlinks, executable files, files over 1 MiB,
credentials, personal filesystem paths, server references, `.DS_Store`, Python
caches, build output, checkpoints, saved notebook output, logs, or experiment
results. The three notebooks have no execution counts, outputs, or attachments.

## Source validation

The complete supported suite passed on macOS arm64 with Python 3.13.2, PyTorch
2.13.0, NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, nflows 0.14, and pytest
9.1.1.

- Complete suite: 163 passed.
- Every code cell in all three canonical notebooks executed with the reduced
  smoke setting and a noninteractive Matplotlib backend.
- The fixed regression sanity test requires the fitted slope to be closer to
  truth than naive OLS and within absolute tolerance 0.2.
- Tests confirm that simulation truth cannot be passed to regression fitting,
  is absent from notebook and script fitting calls, and cannot alter generic
  training through diagnostic RNG consumption.
- Tests confirm importance sampling remains the high-level denoising default.
- Citation validation, workflow linting, and offline workflow security analysis
  passed.
- The resolved runtime requirements reported no known vulnerabilities.

## Distribution validation

`convmmd-0.2.0-py3-none-any.whl` and `convmmd-0.2.0.tar.gz` were built with
Python 3.10.11 using current isolated build tooling.

- `twine check` passed for both files.
- Wheel metadata reports distribution name `convMMD`, version `0.2.0`, Python
  `>=3.10`, MIT, PyTorch `>=2.13`, scikit-learn `>=1.5`, and `nflows==0.14`.
- The wheel contains only the `convMMD` package, distribution metadata, MIT
  license, and nflows notice. Its package source matches the clean snapshot.
- The source distribution additionally contains the approved public
  documentation, three clean canonical notebooks, examples, and supported
  tests. Its package source also matches the clean snapshot.
- Archive inspection found no private coordination records, `.DS_Store`,
  caches, checkpoints, logs, experiment results, or notebook outputs.

SHA-256 values are computed from the final artifacts after this report is
packaged and recorded with the private release review. If GitHub release assets
are later approved, their checksum file must accompany the exact reviewed
wheel and source distribution; rebuilding requires new checksums.

## Fresh wheel installation

The wheel was installed with dependencies into a new Python 3.10.11 virtual
environment that did not inherit system site packages. It resolved PyTorch
2.13.0, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, and nflows 0.14.

- Package `__version__` and installed distribution metadata both reported
  `0.2.0`.
- The imported package path was inside the fresh virtual environment rather
  than the source checkout.
- `pip check` reported no broken requirements.
- The installed third-party environment reported no known vulnerabilities;
  the local `convMMD` wheel is not yet published to an advisory index and was
  therefore reviewed through its source and tests.
- Small installed-package runs completed for deconvolution, default-importance
  posterior-mean denoising, and scalar measurement-error regression. They
  checked finite results, expected shapes, and the public default. The fixed
  installed regression run improved a naive slope of `1.455442` to `1.995843`
  against truth `2.0`.

## Frozen denoising reference

The previously supplied 4,096-case comparison was retained as a reference and
was not rerun as part of the lightweight release validation:

- Noisy MSE: 0.603988
- Importance-sampling MSE: 0.406220
- Langevin MSE: 0.394123

Importance sampling remains the high-level API default. The Langevin result is
an optional ULA comparison and does not change that default.

## Known limitations

- The package is a research preview, not a production inference system or full
  paper reproduction.
- Deconvolution assumes a supplied additive Gaussian or Laplace error model and
  known error standard deviations.
- Posterior-mean denoising is limited to one- or two-dimensional data with known
  diagonal Gaussian errors. Importance estimates remain sensitive to proposal
  overlap and sample budget; Langevin estimates additionally have finite-chain
  and discretization bias.
- Linear measurement-error regression supports one covariate, a known positive
  homoscedastic Gaussian covariate-error standard deviation, a linear response
  mean, and Gaussian residual noise. It does not estimate the covariate-error
  scale or support multivariate covariates, replicates, instruments, nonlinear
  responses, or a composite regression checkpoint.
- MMD forms pairwise kernel matrices and has quadratic batch cost.
- Broad API, checkpoint, and platform compatibility are not promised beyond the
  documented v0.1.0 NSF state-dictionary migration path.
- Local release validation covers macOS arm64 with Python 3.10 and 3.13. The
  configured Ubuntu CPython 3.10--3.14 matrix remains a private remote-CI gate
  before public visibility.
- No remote GPU run was needed for this release validation.
