"""Tests for the high-level empirical-Bayes denoising API."""

import pytest
import torch

import convMMD
from convMMD.denoising import (
    _ImportanceSamplingResolutionError,
    _normalize_noise_std,
    denoise,
    posterior_mean_gaussian,
    posterior_mean_langevin,
)
from convMMD.density_models import load_normalizing_flow_checkpoint


class StandardNormalDensity(torch.nn.Module):
    def log_prob(self, inputs):
        return -0.5 * inputs.square().sum(dim=1)


class NormalDensity(torch.nn.Module):
    def __init__(self, location, scale):
        super().__init__()
        self.register_buffer("location", torch.tensor(float(location)))
        self.register_buffer("scale", torch.tensor(float(scale)))

    def log_prob(self, inputs):
        standardized = (inputs - self.location) / self.scale
        return -0.5 * standardized.square().sum(dim=1)


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


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        (torch.tensor([[0], [1]]), "floating-point"),
        (torch.tensor([0.0, 1.0]), r"shape \(n, d\)"),
        (torch.empty(0, 1), "at least one"),
        (torch.zeros(2, 3), "one- or two-dimensional"),
        (torch.tensor([[0.0], [float("nan")]]), "finite"),
    ],
)
def test_gaussian_posterior_mean_rejects_invalid_observations(
    observations, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        posterior_mean_gaussian(StandardNormalDensity(), observations, 0.2)


@pytest.mark.parametrize(
    ("noise_std", "message"),
    [
        (-0.1, "nonnegative"),
        (float("nan"), "finite"),
        (torch.tensor(0.2 + 1.0j), "numeric"),
        (torch.ones(2), "length 3"),
        (torch.ones(3, 2), "second dimension"),
        (torch.ones(1, 1, 1), "scalar, one-dimensional, or two-dimensional"),
    ],
)
def test_gaussian_posterior_mean_rejects_invalid_noise(noise_std, message):
    observations = torch.zeros(3, 1)

    with pytest.raises(ValueError, match=message):
        posterior_mean_gaussian(StandardNormalDensity(), observations, noise_std)


def _tiny_denoise(observations, seed):
    return denoise(
        observations,
        noise_std=0.2,
        noise_type="gaussian",
        epochs=1,
        batch_size=8,
        lr=1e-3,
        weight_decay=0.0,
        warmup_epochs=0,
        bandwidths=[0.5, 1.0],
        num_blocks=1,
        num_bins=4,
        hidden_features=8,
        tail_bound=5.0,
        num_importance_samples=512,
        posterior_batch_size=4,
        seed=seed,
        device="cpu",
        verbose=False,
    )


def _mock_training_result(model, x_noisy):
    return {
        "model": model,
        "history": {"loss": [0.0]},
        "bandwidths": torch.tensor([1.0], device=x_noisy.device),
    }


def test_denoise_selects_langevin_posterior(monkeypatch):
    model = StandardNormalDensity()
    monkeypatch.setattr(
        "convMMD.denoising.NormalizingFlowDensity",
        lambda **kwargs: model,
    )
    monkeypatch.setattr(
        "convMMD.denoising.train_convmmd",
        lambda *, model, x_noisy, **kwargs: _mock_training_result(
            model, x_noisy
        ),
    )
    calls = []

    def fake_langevin(model, observations, noise_std, **kwargs):
        calls.append(kwargs)
        return observations.clone()

    monkeypatch.setattr(
        "convMMD.denoising.posterior_mean_langevin",
        fake_langevin,
    )
    monkeypatch.setattr(
        "convMMD.denoising.posterior_mean_gaussian",
        lambda *args, **kwargs: pytest.fail("importance sampling was called"),
    )
    observations = torch.tensor([[-0.2], [0.2]], dtype=torch.float32)

    result = denoise(
        observations,
        noise_std=0.5,
        epochs=1,
        batch_size=2,
        warmup_epochs=0,
        posterior_method="langevin",
        posterior_batch_size=2,
        langevin_steps=20,
        langevin_step_size=0.005,
        langevin_chains=7,
        langevin_burn_in_fraction=0.25,
        langevin_thinning=3,
        seed=19,
        verbose=False,
    )

    assert torch.equal(result.denoised, observations)
    assert calls == [
        {
            "n_steps": 20,
            "step_size": 0.005,
            "n_chains": 7,
            "burn_in_fraction": 0.25,
            "thinning": 3,
            "batch_size": 2,
            "seed": 19,
        }
    ]
    assert result.config.posterior_method == "langevin"
    assert result.config.langevin_steps == 20
    assert convMMD.posterior_mean_langevin is posterior_mean_langevin


def test_denoise_retries_resolution_failure_at_maximum_sample_count(monkeypatch):
    monkeypatch.setattr(
        "convMMD.denoising.NormalizingFlowDensity",
        lambda **kwargs: NormalDensity(location=0.0, scale=0.05),
    )
    monkeypatch.setattr(
        "convMMD.denoising.train_convmmd",
        lambda *, model, x_noisy, **kwargs: _mock_training_result(
            model, x_noisy
        ),
    )
    calls = []

    def fake_posterior(model, observations, noise_std, **kwargs):
        calls.append(kwargs)
        if kwargs["num_importance_samples"] == 128:
            raise _ImportanceSamplingResolutionError("sample halves disagree")
        return observations.clone()

    monkeypatch.setattr(
        "convMMD.denoising.posterior_mean_gaussian",
        fake_posterior,
    )
    observations = torch.tensor([[0.20], [0.21]], dtype=torch.float32)

    result = denoise(
        observations,
        noise_std=1.0,
        epochs=1,
        batch_size=2,
        warmup_epochs=0,
        num_importance_samples=128,
        posterior_batch_size=2,
        seed=31,
        verbose=False,
    )
    assert torch.equal(result.denoised, observations)
    assert [call["num_importance_samples"] for call in calls] == [128, 32768]
    assert result.config.num_importance_samples == 32768
    assert result.config.posterior_batch_size == 1


def test_denoise_does_not_retry_past_maximum_sample_count(monkeypatch):
    monkeypatch.setattr(
        "convMMD.denoising.NormalizingFlowDensity",
        lambda **kwargs: NormalDensity(location=0.0, scale=1e-5),
    )
    monkeypatch.setattr(
        "convMMD.denoising.train_convmmd",
        lambda *, model, x_noisy, **kwargs: _mock_training_result(
            model, x_noisy
        ),
    )
    calls = []

    def fake_posterior(model, observations, noise_std, **kwargs):
        calls.append(kwargs)
        raise _ImportanceSamplingResolutionError("sample halves disagree")

    monkeypatch.setattr(
        "convMMD.denoising.posterior_mean_gaussian",
        fake_posterior,
    )
    observations = torch.tensor([[0.20], [0.21]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="sample halves disagree"):
        denoise(
            observations,
            noise_std=1.0,
            epochs=1,
            batch_size=2,
            warmup_epochs=0,
            num_importance_samples=32768,
            posterior_batch_size=2,
            seed=37,
            verbose=False,
        )

    assert len(calls) == 1


def test_denoise_reports_requested_and_maximum_sample_failures(monkeypatch):
    monkeypatch.setattr(
        "convMMD.denoising.NormalizingFlowDensity",
        lambda **kwargs: NormalDensity(location=0.0, scale=1e-5),
    )
    monkeypatch.setattr(
        "convMMD.denoising.train_convmmd",
        lambda *, model, x_noisy, **kwargs: _mock_training_result(
            model, x_noisy
        ),
    )

    def fake_posterior(model, observations, noise_std, **kwargs):
        raise _ImportanceSamplingResolutionError(
            f"sample halves disagree at {kwargs['num_importance_samples']}"
        )

    monkeypatch.setattr(
        "convMMD.denoising.posterior_mean_gaussian",
        fake_posterior,
    )
    observations = torch.tensor([[0.20], [0.21]], dtype=torch.float32)

    with pytest.raises(
        RuntimeError,
        match="requested count 128 and maximum count 32768",
    ):
        denoise(
            observations,
            noise_std=1.0,
            epochs=1,
            batch_size=2,
            warmup_epochs=0,
            num_importance_samples=128,
            posterior_batch_size=2,
            seed=37,
            verbose=False,
        )


def test_denoise_does_not_retry_nonfinite_importance_weights(monkeypatch):
    model = NonFiniteDensity()
    monkeypatch.setattr(
        "convMMD.denoising.NormalizingFlowDensity",
        lambda **kwargs: model,
    )
    monkeypatch.setattr(
        "convMMD.denoising.train_convmmd",
        lambda *, model, x_noisy, **kwargs: _mock_training_result(
            model, x_noisy
        ),
    )
    observations = torch.tensor([[1.0], [2.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="non-finite normalizer"):
        denoise(
            observations,
            noise_std=1.0,
            epochs=1,
            batch_size=2,
            warmup_epochs=0,
            num_importance_samples=128,
            posterior_batch_size=2,
            verbose=False,
        )

    assert model.calls == 1


def test_noise_standard_deviations_are_detached_from_caller_graph():
    observations = torch.zeros(3, 1)
    source = torch.full((3, 1), 0.1, requires_grad=True)
    nonleaf_noise_std = source * 2.0

    normalized = _normalize_noise_std(nonleaf_noise_std, observations)

    assert not normalized.requires_grad
    assert normalized.grad_fn is None


def test_denoise_is_reproducible_and_preserves_rng_state():
    observations = torch.linspace(-1.5, 1.5, 8).unsqueeze(1)
    rng_before = torch.random.get_rng_state().clone()

    first = _tiny_denoise(observations, seed=17)
    rng_after_first = torch.random.get_rng_state().clone()
    second = _tiny_denoise(observations, seed=17)

    assert torch.equal(rng_after_first, rng_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    torch.testing.assert_close(first.denoised, second.denoised, rtol=0, atol=0)
    assert first.history == second.history
    for first_tensor, second_tensor in zip(
        first.model.state_dict().values(), second.model.state_dict().values()
    ):
        assert torch.equal(first_tensor, second_tensor)


def test_denoise_result_and_model_only_checkpoint_round_trip(tmp_path):
    observations = torch.linspace(-1.5, 1.5, 8).unsqueeze(1)
    result = _tiny_denoise(observations, seed=23)

    assert result.denoised.shape == observations.shape
    assert torch.isfinite(result.denoised).all()
    assert result.samples is result.denoised
    assert len(result.history["loss"]) == 1
    assert torch.isfinite(result.bandwidths).all()
    assert result.config.noise_type == "gaussian"
    assert result.config.dim == 1
    assert convMMD.denoise is denoise

    checkpoint_path = tmp_path / "denoiser-model.pt"
    result.save(checkpoint_path)
    restored = load_normalizing_flow_checkpoint(checkpoint_path)
    restored_denoised = posterior_mean_gaussian(
        restored,
        observations,
        0.2,
        num_importance_samples=result.config.num_importance_samples,
        batch_size=result.config.posterior_batch_size,
        seed=result.config.seed,
    )

    torch.testing.assert_close(restored_denoised, result.denoised, rtol=0, atol=0)


def test_denoise_supports_two_dimensional_diagonal_noise():
    observations = 100.0 + torch.tensor(
        [
            [-1.0, -0.5],
            [-0.6, 0.8],
            [-0.2, -0.9],
            [0.1, 0.4],
            [0.4, -0.2],
            [0.7, 1.0],
            [1.0, -0.7],
            [1.3, 0.2],
        ]
    )
    noise_std = torch.tensor([[0.1, 0.3]]).expand_as(observations)

    with pytest.raises(ValueError, match="tail_bound must exceed"):
        denoise(
            observations,
            noise_std,
            epochs=1,
            batch_size=8,
            warmup_epochs=0,
            bandwidths=[0.5, 1.0],
            num_blocks=1,
            num_bins=4,
            hidden_features=8,
            num_importance_samples=512,
            seed=29,
            verbose=False,
        )

    result = denoise(
        observations,
        noise_std,
        epochs=1,
        batch_size=8,
        warmup_epochs=0,
        bandwidths=[0.5, 1.0],
        num_blocks=1,
        num_bins=4,
        hidden_features=8,
        tail_bound=150.0,
        num_importance_samples=512,
        seed=29,
        verbose=False,
    )

    assert result.config.dim == 2
    assert result.config.tail_bound == 150.0
    assert result.denoised.shape == observations.shape
    assert torch.isfinite(result.denoised).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"noise_type": "laplace"}, "only known Gaussian"),
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 1}, "batch_size"),
        ({"posterior_method": "quadrature"}, "posterior_method"),
        ({"num_importance_samples": 127}, "num_importance_samples"),
        ({"num_importance_samples": 130}, "multiple of four"),
        ({"langevin_steps": 0}, "langevin_steps"),
        ({"langevin_step_size": 0.0}, "langevin_step_size"),
        ({"langevin_chains": 0}, "langevin_chains"),
        ({"langevin_burn_in_fraction": 1.0}, "less than 1"),
        ({"langevin_thinning": 0}, "langevin_thinning"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**63}, "seed"),
        ({"epochs": 2, "warmup_epochs": 3}, "warmup_epochs"),
    ],
)
def test_denoise_rejects_unsupported_or_invalid_settings(kwargs, message):
    observations = torch.linspace(-1.0, 1.0, 4).unsqueeze(1)

    with pytest.raises(ValueError, match=message):
        denoise(observations, 0.2, verbose=False, **kwargs)


def test_denoise_requires_two_nonconstant_observations():
    with pytest.raises(ValueError, match="at least two"):
        denoise(torch.zeros(1, 1), 0.2, verbose=False)

    with pytest.raises(ValueError, match="positive variation"):
        denoise(torch.zeros(4, 1), 0.2, verbose=False)

    with pytest.raises(TypeError, match="torch.float32"):
        denoise(torch.zeros(4, 1, dtype=torch.float64), 0.2, verbose=False)
