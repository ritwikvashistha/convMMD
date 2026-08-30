"""Focused tests for importance-sampled Gaussian posterior means."""

import pytest
import torch

from convMMD.denoising import posterior_mean_gaussian


class StandardNormalDensity(torch.nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def log_prob(self, inputs):
        dimension = inputs.shape[1]
        return (
            -0.5 * inputs.square().sum(dim=1)
            - 0.5 * dimension * torch.log(inputs.new_tensor(2.0 * torch.pi))
        )

    def sample(self, num_samples):
        return torch.randn(num_samples, self.dim, dtype=torch.float64)


class NormalDensity(torch.nn.Module):
    def __init__(self, location, scale):
        super().__init__()
        self.register_buffer("location", torch.tensor(float(location)))
        self.register_buffer("scale", torch.tensor(float(scale)))

    def log_prob(self, inputs):
        standardized = (inputs - self.location) / self.scale
        return -0.5 * standardized.square().sum(dim=1)


class NonFiniteDensity(torch.nn.Module):
    def log_prob(self, inputs):
        return torch.full(
            (inputs.shape[0],),
            float("nan"),
            device=inputs.device,
            dtype=inputs.dtype,
        )


class SampleableNormalDensity(torch.nn.Module):
    def __init__(self, location, scale):
        super().__init__()
        self.register_buffer("location", torch.tensor(float(location)))
        self.register_buffer("scale", torch.tensor(float(scale)))

    def log_prob(self, inputs):
        standardized = (inputs - self.location) / self.scale
        return (
            -0.5 * standardized.square().sum(dim=1)
            - torch.log(self.scale)
            - 0.5 * torch.log(inputs.new_tensor(2.0 * torch.pi))
        )

    def sample(self, num_samples):
        return self.location + self.scale * torch.randn(
            num_samples,
            1,
            device=self.location.device,
            dtype=self.location.dtype,
        )


def test_importance_sampling_matches_gaussian_posterior_formula():
    model = StandardNormalDensity(dim=2).double().train()
    observations = torch.tensor(
        [[-1.2, 0.4], [0.0, 1.1], [1.5, -0.7]],
        dtype=torch.float64,
    )
    noise_std = torch.tensor(
        [[0.0, 0.2], [0.3, 0.4], [0.7, 0.5]],
        dtype=torch.float64,
    )

    actual = posterior_mean_gaussian(
        model,
        observations,
        noise_std,
        num_importance_samples=8192,
        batch_size=2,
        seed=17,
    )
    expected = observations / (1.0 + noise_std.square())

    torch.testing.assert_close(actual, expected, rtol=0, atol=0.03)
    assert actual[0, 0] == observations[0, 0]
    assert model.training


def test_defensive_mixture_handles_a_narrow_prior():
    class NarrowNormalDensity(torch.nn.Module):
        def log_prob(self, inputs):
            standardized = inputs / 0.01
            return (
                -0.5 * standardized.square().sum(dim=1)
                - torch.log(inputs.new_tensor(0.01))
                - 0.5 * torch.log(inputs.new_tensor(2.0 * torch.pi))
            )

        def sample(self, num_samples):
            return 0.01 * torch.randn(num_samples, 1, dtype=torch.float64)

    observations = torch.tensor([[0.3]], dtype=torch.float64)
    actual = posterior_mean_gaussian(
        NarrowNormalDensity().double(),
        observations,
        noise_std=1.0,
        num_importance_samples=512,
        seed=19,
    )
    expected = observations * 0.01**2 / (1.0 + 0.01**2)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0.003)


def test_adaptive_proposal_bridges_separated_prior_and_likelihood():
    model = SampleableNormalDensity(location=0.0, scale=0.1).double()
    observations = torch.tensor([[1.0]], dtype=torch.float64)

    actual = posterior_mean_gaussian(
        model,
        observations,
        noise_std=0.1,
        num_importance_samples=2048,
        seed=101,
    )

    torch.testing.assert_close(
        actual,
        torch.tensor([[0.5]], dtype=torch.float64),
        rtol=0,
        atol=0.02,
    )


def test_importance_sampling_is_repeatable_batch_invariant_and_rng_safe():
    model = StandardNormalDensity().double().train()
    observations = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)[:, None]
    rng_before = torch.random.get_rng_state().clone()

    first = posterior_mean_gaussian(
        model,
        observations,
        noise_std=0.5,
        num_importance_samples=2048,
        batch_size=1,
        seed=23,
    )
    second = posterior_mean_gaussian(
        model,
        observations,
        noise_std=0.5,
        num_importance_samples=2048,
        batch_size=64,
        seed=23,
    )

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert model.training


def test_importance_sampling_zero_noise_is_exact_without_density_evaluation():
    observations = torch.tensor([[-2.0], [0.5], [3.0]])

    result = posterior_mean_gaussian(
        NonFiniteDensity(),
        observations,
        noise_std=0.0,
        num_importance_samples=128,
        seed=5,
    )

    assert torch.equal(result, observations)


def test_importance_sampling_rejects_nonfinite_weights():
    with pytest.raises(RuntimeError, match="non-finite normalizer"):
        posterior_mean_gaussian(
            NonFiniteDensity(),
            torch.tensor([[1.0]]),
            noise_std=0.5,
            num_importance_samples=128,
            seed=5,
        )


def test_importance_sampling_does_not_reject_on_ess_alone():
    result = posterior_mean_gaussian(
        NormalDensity(location=0.0, scale=1e-4).double(),
        torch.tensor([[0.3]], dtype=torch.float64),
        noise_std=1.0,
        num_importance_samples=128,
        seed=29,
        check_convergence=False,
    )

    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_importance_samples": 127}, "num_importance_samples"),
        ({"num_importance_samples": 130}, "multiple of four"),
        ({"num_importance_samples": 32770}, "at most"),
        ({"batch_size": 0}, "batch_size"),
        ({"seed": -1}, "seed"),
        ({"check_convergence": 1}, "boolean"),
    ],
)
def test_importance_sampling_rejects_invalid_numerical_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        posterior_mean_gaussian(
            StandardNormalDensity(),
            torch.zeros(2, 1),
            0.2,
            **kwargs,
        )
