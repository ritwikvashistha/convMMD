# Canonical examples

These package-backed examples are the recommended starting points for the
v0.2.0 research preview. Begin with a public checkout and run them from the
repository root:

```bash
git clone https://github.com/ritwikvashistha/convMMD.git
cd convMMD
python -m pip install -e ".[examples,notebooks]"
```

## Simulation notebooks

Each notebook fixes its random seed and separates latent truth, noisy
observations, model fitting, and simulation-only evaluation. The fitting cells
call public `convMMD` APIs and do not duplicate spline, loss, or optimizer
implementations.

| Capability | Notebook | Evaluation |
| --- | --- | --- |
| Deconvolution and density estimation | `examples/notebooks/deconvolution_density_estimation.ipynb` | Latent-density ISE and one density plot |
| Posterior-mean denoising | `examples/notebooks/posterior_mean_denoising.ipynb` | Noisy and denoised MSE; importance sampling remains the default |
| Linear measurement-error regression | `examples/notebooks/linear_measurement_error_regression.ipynb` | Parameter error and observed-space forward MMD |

Open them after starting Jupyter from the repository root. For the reduced
release-check configuration, set `CONVMMD_NOTEBOOK_SMOKE=1` in the environment.
The supported test suite executes every code cell with that setting without
writing outputs back into the notebooks.

## Script examples

| Task | Entry point | Assumptions |
| --- | --- | --- |
| 1D deconvolution | `examples/deconv_1d.py` | Known per-observation Laplace or Gaussian error standard deviations |
| 2D deconvolution | `examples/deconv_2d.py` | Known per-observation isotropic Gaussian errors; set `--outlier_fraction 0` for the pure additive-error model |
| 1D posterior-mean denoising | `examples/denoise_1d.py` | Known Gaussian errors; synthetic truth is used only for MSE after fitting |

The two deconvolution scripts likewise keep latent truth outside the
`train_convmmd` call and use it only for post-fit plots and metrics. Their model
construction is explicitly seeded, as are the 2D evaluation draws.

Reduced CPU checks:

```bash
MPLBACKEND=Agg python examples/deconv_1d.py \
  --device cpu --n_samples 64 --epochs 2 --batch_size 32 \
  --num_blocks 1 --num_bins 4 --hidden_features 8 \
  --save_plot /tmp/convmmd-1d-smoke.png

MPLBACKEND=Agg python examples/deconv_2d.py \
  --device cpu --dataset moons --flow_type nsf \
  --n_samples 64 --epochs 2 --batch_size 32 \
  --num_blocks 1 --num_bins 4 --hidden_features 8 \
  --outlier_fraction 0 --n_eval_samples 128 --n_projections 16 \
  --save_plot /tmp/convmmd-2d-smoke.png

python examples/denoise_1d.py \
  --device cpu --n_samples 64 --epochs 2 --batch_size 32 \
  --num_blocks 1 --num_bins 4 --hidden_features 8 \
  --num_importance_samples 512 --posterior_batch_size 32 --quiet
```

These configurations verify execution only; they are not quality or runtime
benchmarks. Full defaults perform substantially more optimization.

## Denoising scope

`denoise_1d.py` fits only the noisy observations and supplied Gaussian error
standard deviations. The clean generated values are used afterward for MSE.
The returned values are posterior-mean point estimates aligned with the input
rows; random latent draws come from `result.model.sample(...)`.

Adaptive importance sampling is the default. Pass `--posterior_method
langevin` to use the optional unadjusted Langevin estimator. Both methods are
low-dimensional research procedures rather than generic image denoisers.

## Checkpoints

All three scripts accept `--save_checkpoint PATH`. This option is off by
default and writes only after successful training. Load the fitted flow with:

```python
from convMMD.density_models import load_normalizing_flow_checkpoint

model = load_normalizing_flow_checkpoint("/tmp/convmmd-model.pt")
```

The checkpoint contains a strictly reconstructable `NormalizingFlowDensity`,
not observations, noise scales, posterior means, optimizer state, history, RNG
state, device placement, or train/eval mode. The scalar regression result does
not have a composite checkpoint format in v0.2.0.
