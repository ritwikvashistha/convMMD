# Contributing to convMMD

Thank you for considering a contribution. `convMMD` is a focused PyTorch
research preview, so changes should remain close to the documented scientific
scope and be small enough to review and validate.

## Before starting

- Use a GitHub issue for reproducible bugs, documentation gaps, or focused
  feature proposals.
- For a scientific-method change, describe the observation model, fitting
  objective, validation strategy, and expected API impact before implementing
  it.
- Do not include private data, credentials, checkpoints, large experiment
  outputs, server logs, or paper-reproduction artifacts.
- Report suspected vulnerabilities privately according to `SECURITY.md`.

The v0.2.x line supports normalizing-flow latent models. A GMM implementation,
new regression families, correlated-error models, and full benchmark suites
require separate design and scientific review; they should not arrive as
incidental extensions to another pull request.

## Development setup

Create an isolated environment from a source checkout and install the
development and example dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,examples,notebooks]"
```

Run the supported checks:

```bash
python -m pytest
CONVMMD_NOTEBOOK_SMOKE=1 MPLBACKEND=Agg \
  python -m pytest tests/test_notebooks.py -q
python -m build
python -m twine check dist/*
```

The exact public support matrix is defined by the release CI configuration.

## Change guidelines

- Preserve the distribution and import name `convMMD`.
- Keep importance sampling as the default denoising posterior method unless a
  separately reviewed release changes that policy.
- Use simulation truth only for evaluation, never as input to fitting.
- Add focused unit tests and a scientific sanity check for method changes.
- Keep notebooks package-backed, deterministic, smoke-runnable, and free of
  saved outputs and execution counts.
- Document assumptions, shapes, dtypes, noise conventions, reproducibility,
  checkpoint effects, and known limitations.
- Avoid unrelated cleanup and speculative generalization.

## Pull requests

A pull request should explain what changed, why it is in scope, how it was
tested, and whether it changes scientific behavior or compatibility. Keep the
working tree free of generated artifacts. By submitting a contribution, you
agree that it may be distributed under the repository's MIT License.

Participation in project spaces is governed by `CODE_OF_CONDUCT.md`.
