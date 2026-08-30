# Releasing convMMD

This guide describes the public GitHub release process for the `convMMD`
research preview. It does not authorize a repository rename, push, tag,
visibility change, GitHub release, PyPI upload, DOI registration, or public
announcement. Those external actions require explicit owner approval at their
release gates.

## 1. Release model

The initial public v0.2.0 release uses a clean repository:

- the existing development repository is retained privately as
  `convMMD-private`;
- a fresh `convMMD` repository is created privately;
- one reviewed root commit is constructed from an explicit public allowlist;
- CI and release verification run while the fresh repository is private;
- only the verified root commit is tagged and made public.

Do not copy `.git`, ignored files, old refs, historical notebooks, or the
working directory wholesale. Construct the public tree from the reviewed
package, tests, examples, method guide, metadata, and community files listed in
this release candidate. Internal release-coordination records are not public
package content.

## 2. Prepare an isolated release environment

From a clean checkout of the exact release candidate:

```bash
python -m venv .venv-release
source .venv-release/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,examples,notebooks]"
```

Confirm that package and project metadata agree:

```bash
python -c "import convMMD; print(convMMD.__version__)"
python -c "import importlib.metadata; print(importlib.metadata.version('convMMD'))"
```

Both commands must print `0.2.0`. The final supported Python and platform
matrix is defined by release CI and `RELEASE_VALIDATION.md`.

## 3. Run source checks

```bash
python -m pytest
CONVMMD_NOTEBOOK_SMOKE=1 MPLBACKEND=Agg \
  python -m pytest tests/test_notebooks.py -q
```

Execute the documented reduced examples for deconvolution, default-importance
denoising, and measurement-error regression. Confirm that all notebooks remain
free of saved outputs and execution counts. Run the approved full-history and
working-tree secret scans before constructing the clean root commit, then scan
the clean repository again.

## 4. Build and inspect distributions

Stop if `dist/` already exists so stale artifacts cannot be mistaken for the
current release:

```bash
test ! -e dist
python -m build
python -m twine check \
  dist/convmmd-0.2.0.tar.gz \
  dist/convmmd-0.2.0-py3-none-any.whl
```

Inspect both archives. They must not contain credentials, caches, notebook
outputs, checkpoints, logs, experiment results, server artifacts, or private
development history. Record checksums:

```bash
shasum -a 256 \
  dist/convmmd-0.2.0.tar.gz \
  dist/convmmd-0.2.0-py3-none-any.whl
```

## 5. Install the wheel in a fresh environment

```bash
release_check_dir=$(mktemp -d)
python -m venv "$release_check_dir/venv"
"$release_check_dir/venv/bin/python" -m pip install --upgrade pip
"$release_check_dir/venv/bin/python" -m pip install \
  dist/convmmd-0.2.0-py3-none-any.whl
"$release_check_dir/venv/bin/python" -m pip check
"$release_check_dir/venv/bin/python" -c \
  "import convMMD, importlib.metadata; print(convMMD.__version__); print(importlib.metadata.version('convMMD'))"
```

Run small installed-package examples for all three major capabilities from a
directory outside the source checkout.

## 6. Review the clean public snapshot

Before its root commit is created:

1. compare the snapshot file list with the approved allowlist;
2. review every staged file and the complete staged diff;
3. confirm the root-commit identity is
   `Ritwik Vashistha <ritwik.v@utexas.edu>`;
4. verify the MIT and third-party notices;
5. verify public URLs, citation metadata, contribution guidance, security
   instructions, and known limitations;
6. confirm `convMMD` remains both the distribution and import name;
7. confirm importance sampling remains the denoising default;
8. record the final commit hash and validation evidence.

## 7. Private CI gate

Push the clean root commit to the fresh private `convMMD` repository only after
explicit approval. Require the full release workflow to pass. Rebuild and
verify artifacts from the exact commit if CI or review causes any change.

Do not reuse artifacts built from a preceding commit, even when only
documentation or metadata changed.

## 8. Tag and make the repository public

Only after a separate publication approval:

1. create annotated tag `v0.2.0` on the exact verified commit;
2. confirm tag identity and contents;
3. change the fresh repository visibility to public;
4. recheck branch protection, Actions permissions, secret scanning, private
   vulnerability reporting, issues, Wiki, Projects, description, and topics;
5. verify anonymous HTTPS clone and tag installation;
6. create the GitHub release and attach only the verified wheel, source
   distribution, and checksums if artifact attachment was approved.

Do not upload to PyPI or TestPyPI. Do not announce, mirror, or otherwise expose
the package beyond the approved public GitHub repository.

## 9. Post-publication verification

From an unauthenticated environment, verify:

- HTTPS clone;
- installation from `v0.2.0`;
- release-asset checksums;
- package and distribution versions;
- documentation and notebook rendering;
- reduced installed-package examples;
- issue and private vulnerability reporting instructions.

Record the results and any known limitations in `RELEASE_VALIDATION.md`.
