"""Focused tests for Langevin posterior-mean denoising."""

import pytest
import torch

from convMMD.denoising import posterior_mean_langevin


class StandardNormalDensity(torch.nn.Module):
    def log_prob(self, inputs):
        return -0.5 * inputs.square().sum(dim=1)


class NonFiniteDensity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def log_prob(self, inputs):
        self.calls += 1
        return torch.full(
            (inputs.shape[0],),
            float("nan"),
            device=inputs.device,
            dtype=inputs.dtype,
        )


def test_langevin_matches_gaussian_posterior_mean():
    model = StandardNormalDensity().double().train()
    observations = torch.tensor(
        [[-1.2, 0.4], [0.0, 1.1], [1.5, -0.7]],
        dtype=torch.float64,
    )
    noise_std = torch.tensor(
        [[0.3, 0.5], [0.4, 0.6], [0.5, 0.3]],
        dtype=torch.float64,
    )

    actual = posterior_mean_langevin(
        model,
        observations,
        noise_std,
        n_steps=600,
        step_size=0.01,
        n_chains=512,
        burn_in_fraction=0.6,
        thinning=4,
        batch_size=2,
        seed=17,
    )
    expected = observations / (1.0 + noise_std.square())

    torch.testing.assert_close(actual, expected, rtol=0, atol=0.04)
    assert model.training


def test_langevin_is_repeatable_batch_invariant_and_rng_safe():
    model = StandardNormalDensity().double().train()
    observations = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)[:, None]
    rng_before = torch.random.get_rng_state().clone()

    first = posterior_mean_langevin(
        model,
        observations,
        noise_std=0.5,
        n_steps=30,
        n_chains=16,
        burn_in_fraction=0.5,
        thinning=2,
        batch_size=1,
        seed=23,
    )
    second = posterior_mean_langevin(
        model,
        observations,
        noise_std=0.5,
        n_steps=30,
        n_chains=16,
        burn_in_fraction=0.5,
        thinning=2,
        batch_size=64,
        seed=23,
    )

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert model.training


def test_langevin_keeps_zero_noise_coordinates_exact():
    model = StandardNormalDensity().double()
    observations = torch.tensor(
        [[-2.0, 0.5], [1.0, -1.5]],
        dtype=torch.float64,
    )
    noise_std = torch.tensor(
        [[0.0, 0.4], [0.3, 0.0]],
        dtype=torch.float64,
    )

    result = posterior_mean_langevin(
        model,
        observations,
        noise_std,
        n_steps=40,
        n_chains=16,
        seed=7,
    )

    assert torch.equal(result[noise_std == 0], observations[noise_std == 0])
    assert torch.isfinite(result).all()


def test_langevin_all_zero_noise_bypasses_density_evaluation():
    model = NonFiniteDensity()
    observations = torch.tensor([[-2.0], [0.5], [3.0]])

    result = posterior_mean_langevin(
        model,
        observations,
        noise_std=0.0,
        n_steps=10,
        n_chains=4,
        seed=5,
    )

    assert torch.equal(result, observations)
    assert model.calls == 0


def test_langevin_rejects_nonfinite_prior_values():
    with pytest.raises(RuntimeError, match="non-finite prior"):
        posterior_mean_langevin(
            NonFiniteDensity(),
            torch.tensor([[1.0]]),
            noise_std=0.5,
            n_steps=10,
            n_chains=4,
            seed=5,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_steps": 0}, "n_steps"),
        ({"step_size": 0.0}, "step_size"),
        ({"step_size": float("nan")}, "step_size"),
        ({"n_chains": 0}, "n_chains"),
        ({"burn_in_fraction": -0.1}, "burn_in_fraction"),
        ({"burn_in_fraction": 1.0}, "burn_in_fraction"),
        ({"thinning": 0}, "thinning"),
        ({"batch_size": 0}, "batch_size"),
        ({"seed": -1}, "seed"),
    ],
)
def test_langevin_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        posterior_mean_langevin(
            StandardNormalDensity(),
            torch.zeros(2, 1),
            0.2,
            **kwargs,
        )
