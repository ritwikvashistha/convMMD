"""High-value smoke tests for the convMMD research preview."""

import inspect
import numpy as np
import pytest
import torch

import convMMD
from convMMD.core.data import (
    generate_1d_laplace_mixture,
    generate_checkerboard,
    generate_circles,
    generate_moons,
)
from convMMD.core.evaluate import sliced_wasserstein_distance
from convMMD.core.losses import (
    compute_bandwidth_median_heuristic,
    mmd_gaussian_kernel,
    mmd_laplace_kernel,
)
from convMMD.density_models.nf import NormalizingFlowDensity
from convMMD.training.train import train_convmmd


def _tiny_flow():
    return NormalizingFlowDensity(
        dim=1,
        flow_type="nsf",
        num_blocks=1,
        num_bins=4,
        hidden_features=8,
        tail_bound=4.0,
    )


def test_package_import_and_version():
    assert convMMD.__version__ == "0.2.0"


def test_public_exports_and_default_posterior_method_are_explicit():
    assert convMMD.__all__ == [
        "__version__",
        "denoise",
        "fit_measurement_error_regression",
        "posterior_mean_gaussian",
        "posterior_mean_langevin",
    ]
    assert (
        inspect.signature(convMMD.denoise)
        .parameters["posterior_method"]
        .default
        == "importance"
    )


@pytest.mark.parametrize("mmd_fn", [mmd_laplace_kernel, mmd_gaussian_kernel])
def test_mmd_is_finite_symmetric_and_detects_separation(mmd_fn):
    x = torch.tensor([[-1.5], [-0.5], [0.5], [1.5]])
    y = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
    bandwidths = [0.5, 1.0, 2.0]

    xy = mmd_fn(x, y, bandwidths)
    yx = mmd_fn(y, x, bandwidths)
    separated = mmd_fn(x, x + 10.0, bandwidths)

    assert xy.shape == ()
    assert torch.isfinite(xy)
    torch.testing.assert_close(xy, yx)
    assert separated > 0


@pytest.mark.parametrize("mmd_fn", [mmd_laplace_kernel, mmd_gaussian_kernel])
def test_mmd_rejects_singleton_samples(mmd_fn):
    with pytest.raises(ValueError, match="at least two samples"):
        mmd_fn(torch.zeros(1, 1), torch.zeros(2, 1), [1.0])


@pytest.mark.parametrize("mmd_fn", [mmd_laplace_kernel, mmd_gaussian_kernel])
def test_mmd_normalizes_scalar_list_and_tensor_bandwidths(mmd_fn):
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    y = torch.tensor([[0.0], [1.0], [2.0]])
    expected = mmd_fn(x, y, [1.0])

    for bandwidths in (
        1.0,
        torch.tensor(1.0),
        torch.tensor([[1.0]]),
    ):
        torch.testing.assert_close(mmd_fn(x, y, bandwidths), expected)


@pytest.mark.parametrize(
    ("bandwidths", "message"),
    [
        ([], "at least one"),
        ([0.0], "strictly positive"),
        ([-1.0], "strictly positive"),
        ([float("nan")], "finite"),
        ([float("inf")], "finite"),
    ],
)
def test_mmd_rejects_invalid_bandwidths(bandwidths, message):
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    with pytest.raises(ValueError, match=message):
        mmd_laplace_kernel(x, x, bandwidths)


def test_bandwidth_heuristic_preserves_float64_dtype():
    x = torch.tensor([[-2.0], [-1.0], [0.0]], dtype=torch.float64)
    y = torch.tensor([[0.5], [1.0], [2.0]], dtype=torch.float64)

    bandwidths = compute_bandwidth_median_heuristic(x, y)

    assert bandwidths.dtype == torch.float64
    assert bandwidths.ndim == 1
    assert torch.isfinite(bandwidths).all()
    assert (bandwidths > 0).all()


def test_bandwidth_heuristic_rejects_identical_points():
    points = torch.zeros(4, 1)
    with pytest.raises(ValueError, match="no pairwise distance"):
        compute_bandwidth_median_heuristic(points, points)


@pytest.mark.parametrize(
    ("x", "y", "message"),
    [
        (torch.empty(0, 1), torch.zeros(2, 1), "at least one sample"),
        (torch.zeros(2), torch.zeros(2, 1), "two-dimensional"),
        (torch.zeros(2, 1), torch.zeros(2, 2), "same feature dimension"),
    ],
)
def test_bandwidth_heuristic_rejects_invalid_sample_shapes(x, y, message):
    with pytest.raises(ValueError, match=message):
        compute_bandwidth_median_heuristic(x, y)


def test_normalizing_flow_samples_and_log_probabilities():
    torch.manual_seed(11)
    model = _tiny_flow()

    samples = model.sample(16)
    log_probabilities = model.log_prob(samples)

    assert samples.shape == (16, 1)
    assert log_probabilities.shape == (16,)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(log_probabilities).all()


def test_tiny_training_step_updates_actual_flow():
    torch.manual_seed(7)
    model = _tiny_flow()
    observations = torch.linspace(-2.0, 2.0, 16).unsqueeze(1)
    noise_std = torch.full((16, 1), 0.1)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    result = train_convmmd(
        model=model,
        x_noisy=observations,
        noise_std=noise_std,
        noise_type="gaussian",
        kernel_type="gaussian",
        epochs=1,
        batch_size=16,
        lr=1e-3,
        weight_decay=0.0,
        warmup_epochs=0,
        bandwidths=[0.5, 1.0],
        eval_every=1,
        verbose=False,
    )

    assert len(result["history"]["loss"]) == 1
    assert torch.isfinite(torch.tensor(result["history"]["loss"])).all()
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameters_before, model.parameters())
    )


def test_1d_and_moons_data_shapes():
    theta_1d, noisy_1d, noise_std_1d = generate_1d_laplace_mixture(
        n_samples=17, seed=3
    )
    theta_2d, noisy_2d, noise_std_2d = generate_moons(
        n_samples=18, outlier_fraction=0.0, seed=3
    )

    assert theta_1d.shape == noisy_1d.shape == noise_std_1d.shape == (17, 1)
    assert theta_2d.shape == noisy_2d.shape == (18, 2)
    assert noise_std_2d.shape == (18, 1)
    for tensor in (
        theta_1d,
        noisy_1d,
        noise_std_1d,
        theta_2d,
        noisy_2d,
        noise_std_2d,
    ):
        assert torch.isfinite(tensor).all()


@pytest.mark.parametrize(
    ("generator", "kwargs"),
    [
        (generate_1d_laplace_mixture, {}),
        (generate_moons, {"outlier_fraction": 0.0}),
        (generate_circles, {"outlier_fraction": 0.0}),
        (generate_checkerboard, {"outlier_fraction": 0.0}),
    ],
    ids=("laplace-mixture", "moons", "circles", "checkerboard"),
)
def test_seeded_data_generators_are_repeatable_and_rng_safe(generator, kwargs):
    torch.manual_seed(991)
    np.random.seed(991)
    expected_torch_draw = torch.rand(5)
    expected_numpy_draw = np.random.random(5)

    torch.manual_seed(991)
    np.random.seed(991)
    first = generator(n_samples=12, seed=17, **kwargs)
    actual_torch_draw = torch.rand(5)
    actual_numpy_draw = np.random.random(5)
    second = generator(n_samples=12, seed=17, **kwargs)

    torch.testing.assert_close(actual_torch_draw, expected_torch_draw)
    np.testing.assert_array_equal(actual_numpy_draw, expected_numpy_draw)
    for first_tensor, second_tensor in zip(first, second):
        assert torch.equal(first_tensor, second_tensor)


@pytest.mark.parametrize("p", [1, 2])
def test_swd_handles_identical_empirical_distributions_with_unequal_counts(p):
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    y = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float64)

    distance = sliced_wasserstein_distance(x, y, n_projections=8, p=p, seed=5)

    assert distance == pytest.approx(0.0, abs=1e-12)


def test_swd_rejects_unsupported_order():
    points = torch.tensor([[0.0], [1.0]])

    with pytest.raises(ValueError, match="p must be 1 or 2"):
        sliced_wasserstein_distance(points, points, p=3)


def test_seeded_swd_is_repeatable_and_preserves_torch_rng_state():
    points = torch.tensor([[-1.0, 0.5], [0.0, -0.5], [1.0, 0.25]])
    torch.manual_seed(1234)
    expected_draw = torch.rand(6)

    torch.manual_seed(1234)
    first = sliced_wasserstein_distance(
        points, points + 0.2, n_projections=32, seed=29
    )
    actual_draw = torch.rand(6)
    second = sliced_wasserstein_distance(
        points, points + 0.2, n_projections=32, seed=29
    )

    assert first == second
    torch.testing.assert_close(actual_draw, expected_draw)


def test_training_diagnostics_cannot_change_the_torch_optimization_stream():
    observations = torch.linspace(-2.0, 2.0, 16).unsqueeze(1)
    noise_std = torch.full((16, 1), 0.1)

    def fit(eval_fn):
        torch.manual_seed(771)
        model = _tiny_flow()
        return train_convmmd(
            model=model,
            x_noisy=observations,
            noise_std=noise_std,
            noise_type="gaussian",
            kernel_type="gaussian",
            epochs=3,
            batch_size=8,
            lr=1e-3,
            weight_decay=0.0,
            warmup_epochs=0,
            bandwidths=[0.5, 1.0],
            eval_fn=eval_fn,
            eval_every=1,
            verbose=False,
        )

    without_diagnostics = fit(None)

    def consuming_diagnostic(_model):
        torch.rand(1000)
        return 0.0

    with_diagnostics = fit(consuming_diagnostic)

    for without, with_diagnostic in zip(
        without_diagnostics["model"].state_dict().values(),
        with_diagnostics["model"].state_dict().values(),
    ):
        assert torch.equal(without, with_diagnostic)
    assert not without_diagnostics["model"].training
    assert not with_diagnostics["model"].training


@pytest.mark.parametrize(
    ("x_noisy", "noise_std", "message"),
    [
        (torch.zeros(4), torch.ones(4, 1), r"shape \(n, d\)"),
        (torch.zeros(4, 1), torch.ones(4), r"shape \(n, 1\) or \(n, d\)"),
        (torch.zeros(4, 2), torch.ones(4, 3), r"shape \(n, 1\) or \(n, d\)"),
        (torch.zeros(4, 1), -torch.ones(4, 1), "nonnegative"),
        (
            torch.zeros(4, 1),
            torch.ones(4, 1, dtype=torch.float64),
            "same dtype",
        ),
    ],
)
def test_training_rejects_invalid_observation_models(x_noisy, noise_std, message):
    with pytest.raises((TypeError, ValueError), match=message):
        train_convmmd(
            model=_tiny_flow(),
            x_noisy=x_noisy,
            noise_std=noise_std,
            epochs=1,
            batch_size=2,
            bandwidths=[1.0],
            verbose=False,
        )


def test_training_rejects_unknown_noise_type():
    with pytest.raises(ValueError, match="Unknown noise type"):
        train_convmmd(
            model=torch.nn.Linear(1, 1),
            x_noisy=torch.zeros(4, 1),
            noise_std=torch.ones(4, 1),
            noise_type="typo",
            epochs=0,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("n_samples", "batch_size", "message"),
    [
        (4, 1, "batch_size must be at least 2"),
        (1, 2, "at least two observations"),
    ],
)
def test_training_rejects_invalid_mmd_sample_counts(
    n_samples, batch_size, message
):
    with pytest.raises(ValueError, match=message):
        train_convmmd(
            model=torch.nn.Linear(1, 1),
            x_noisy=torch.zeros(n_samples, 1),
            noise_std=torch.ones(n_samples, 1),
            noise_type="gaussian",
            batch_size=batch_size,
            epochs=1,
            bandwidths=[1.0],
            verbose=False,
        )


def test_training_raises_when_no_finite_step_completes():
    class NonFiniteSampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.location = torch.nn.Parameter(torch.tensor([[0.0]]))

        def sample(self, n_samples):
            samples = self.location.expand(n_samples, 1)
            return samples * samples.new_tensor(float("nan"))

    with pytest.raises(RuntimeError, match="No finite optimization steps"):
        train_convmmd(
            model=NonFiniteSampler(),
            x_noisy=torch.zeros(4, 1),
            noise_std=torch.zeros(4, 1),
            noise_type="gaussian",
            kernel_type="gaussian",
            epochs=1,
            batch_size=2,
            warmup_epochs=0,
            bandwidths=[1.0],
            eval_every=1,
            verbose=False,
        )
