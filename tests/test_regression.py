"""Focused tests for scalar linear measurement-error regression."""

import inspect

import pytest
import torch

import convMMD
from convMMD.regression import (
    _corrected_moment_initialization,
    fit_measurement_error_regression,
)


def _simulation(n=64, seed=8128):
    generator = torch.Generator().manual_seed(seed)
    component = torch.rand(n, generator=generator) < 0.55
    latent = torch.where(
        component,
        -1.35 + 0.45 * torch.randn(n, generator=generator),
        1.15 + 0.65 * torch.randn(n, generator=generator),
    )
    observed = latent + 0.8 * torch.randn(n, generator=generator)
    response = -0.75 + 2.0 * latent + 0.5 * torch.randn(n, generator=generator)
    return latent, observed, response


def _tiny_fit(observed, response, *, seed, steps=2, learning_rate=1e-3):
    return fit_measurement_error_regression(
        observed,
        response,
        measurement_error_std=0.8,
        steps=steps,
        batch_size=32,
        learning_rate=learning_rate,
        weight_decay=0.0,
        bandwidths=[0.5, 1.0],
        eval_every=1,
        num_blocks=1,
        num_bins=4,
        hidden_features=8,
        tail_bound=5.0,
        seed=seed,
        device="cpu",
        verbose=False,
    )


def test_public_fit_signature_cannot_receive_simulation_truth():
    parameters = inspect.signature(fit_measurement_error_regression).parameters

    assert set(parameters) >= {
        "observed_covariate",
        "response",
        "measurement_error_std",
        "seed",
    }
    assert not ({"truth", "latent", "x_true", "true_parameters"} & set(parameters))
    assert convMMD.fit_measurement_error_regression is (
        fit_measurement_error_regression
    )


def test_corrected_moment_initialization_matches_observed_data_formula():
    observed = torch.tensor([-1.5, -0.5, 0.25, 1.0, 2.0])
    response = torch.tensor([-2.0, -0.8, 0.4, 1.7, 3.2])
    measurement_error_std = 0.3

    intercept, slope, residual_std = _corrected_moment_initialization(
        observed,
        response,
        measurement_error_std,
        minimum_residual_std=0.02,
    )
    latent_variance = observed.var(unbiased=False) - measurement_error_std**2
    expected_slope = (
        ((observed - observed.mean()) * (response - response.mean())).mean()
        / latent_variance
    )
    expected_intercept = response.mean() - expected_slope * observed.mean()
    expected_residual_std = torch.sqrt(
        torch.clamp(
            response.var(unbiased=False) - expected_slope.square() * latent_variance,
            min=0.02**2,
        )
    )

    torch.testing.assert_close(slope, expected_slope)
    torch.testing.assert_close(intercept, expected_intercept)
    torch.testing.assert_close(residual_std, expected_residual_std)


def test_regression_result_and_forward_samples_are_reproducible_and_rng_safe():
    _, observed, response = _simulation()
    rng_before = torch.random.get_rng_state().clone()

    first = _tiny_fit(observed, response, seed=17)
    rng_after_first = torch.random.get_rng_state().clone()
    second = _tiny_fit(observed, response, seed=17)

    assert torch.equal(rng_after_first, rng_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert first.history == second.history
    assert first.intercept == second.intercept
    assert first.slope == second.slope
    assert first.residual_std == second.residual_std
    for first_tensor, second_tensor in zip(
        first.model.state_dict().values(), second.model.state_dict().values()
    ):
        assert torch.equal(first_tensor, second_tensor)

    first_draws = first.sample_observed(11, seed=29)
    second_draws = first.sample_observed(11, seed=29)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for first_tensor, second_tensor in zip(first_draws, second_draws):
        assert first_tensor.shape == (11, 1)
        assert torch.equal(first_tensor, second_tensor)
        assert torch.isfinite(first_tensor).all()
    assert first.residual_std > 0.02
    assert first.config.measurement_error_std == 0.8
    assert first.config.batch_size == 32


def test_fixed_simulation_corrects_visible_naive_slope_attenuation():
    _, observed, response = _simulation(n=512, seed=31415)
    naive_slope = float(
        ((observed - observed.mean()) * (response - response.mean())).mean()
        / observed.var(unbiased=False)
    )

    result = _tiny_fit(
        observed,
        response,
        seed=31416,
        steps=8,
        learning_rate=1e-4,
    )

    assert abs(result.slope - 2.0) < abs(naive_slope - 2.0)
    assert abs(result.slope - 2.0) < 0.2
    assert abs(result.intercept - (-0.75)) < 0.2
    assert result.residual_std > 0.02
    assert torch.isfinite(torch.tensor(result.history["loss"])).all()


@pytest.mark.parametrize(
    ("observed", "response", "measurement_error_std", "kwargs", "message"),
    [
        ([0.0, 1.0], torch.tensor([0.0, 1.0]), 0.1, {}, "PyTorch tensor"),
        (torch.zeros(2, 1, 1), torch.zeros(2), 0.1, {}, r"shape \(n,\)"),
        (torch.zeros(2, dtype=torch.float64), torch.zeros(2), 0.1, {}, "float32"),
        (torch.zeros(3), torch.zeros(2), 0.1, {}, "same length"),
        (torch.tensor([0.0, float("nan")]), torch.zeros(2), 0.1, {}, "finite"),
        (torch.tensor([-1.0, 1.0]), torch.zeros(2), 0.1, {}, "must each vary"),
        (torch.tensor([-0.1, 0.1]), torch.tensor([-1.0, 1.0]), 1.0, {}, "variance"),
        (torch.tensor([-1.0, 1.0]), torch.tensor([-1.0, 1.0]), 0.0, {}, "greater than"),
        (torch.tensor([-1.0, 1.0]), torch.tensor([-1.0, 1.0]), [0.1], {}, "scalar"),
        (torch.tensor([-1.0, 1.0]), torch.tensor([-1.0, 1.0]), 0.1, {"steps": 0}, "steps"),
        (
            torch.tensor([-1.0, 1.0]),
            torch.tensor([-1.0, 1.0]),
            0.1,
            {"batch_size": 1},
            "batch_size",
        ),
        (
            torch.tensor([-1.0, 1.0]),
            torch.tensor([-1.0, 1.0]),
            0.1,
            {"learning_rate": float("nan")},
            "learning_rate",
        ),
    ],
)
def test_regression_rejects_invalid_inputs(
    observed, response, measurement_error_std, kwargs, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        fit_measurement_error_regression(
            observed,
            response,
            measurement_error_std,
            verbose=False,
            **kwargs,
        )
