"""Low-dimensional empirical-Bayes denoising with a convMMD flow prior."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Union

import torch

from .core.losses import BandwidthInput
from .density_models.checkpoint import save_normalizing_flow_checkpoint
from .density_models.nf import MAX_DERIVATIVE, NormalizingFlowDensity
from .training.train import train_convmmd


NoiseStandardDeviation = Union[
    float,
    Sequence[float],
    Sequence[Sequence[float]],
    torch.Tensor,
]
_SUPPORTED_DTYPES = {torch.float32, torch.float64}
_CONVERGENCE_ATOL = 5e-3
_CONVERGENCE_RTOL = 5e-2
_MIN_IMPORTANCE_SAMPLES = 128
_MAX_IMPORTANCE_SAMPLES = 32768
_MAX_IMPORTANCE_BATCH_POINTS = 131072
_IMPORTANCE_CONSISTENCY_STANDARD_ERRORS = 5.0


class _ImportanceSamplingResolutionError(RuntimeError):
    """Raised when finite importance samples are demonstrably unreliable."""


@dataclass(frozen=True)
class DenoisingConfig:
    """Effective settings used by :func:`denoise`."""

    dim: int
    noise_type: str
    kernel_type: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    max_grad_norm: float
    num_blocks: int
    num_bins: int
    hidden_features: int
    tail_bound: float
    max_derivative: float
    posterior_method: str
    num_importance_samples: int
    posterior_batch_size: int
    langevin_steps: int
    langevin_step_size: float
    langevin_chains: int
    langevin_burn_in_fraction: float
    langevin_thinning: int
    seed: int
    device: str


@dataclass
class DenoisingResult:
    """Fitted flow and posterior-mean estimates aligned with the observations."""

    denoised: torch.Tensor
    model: NormalizingFlowDensity
    history: Dict[str, Any]
    bandwidths: torch.Tensor
    config: DenoisingConfig

    @property
    def samples(self) -> torch.Tensor:
        """Alias for ``denoised``; these are point estimates, not random draws."""
        return self.denoised

    def save_model(self, destination) -> None:
        """Save only the fitted flow in the versioned model-checkpoint format."""
        save_normalizing_flow_checkpoint(self.model, destination)

    def save(self, destination) -> None:
        """Alias for :meth:`save_model`; observations and history are not saved."""
        self.save_model(destination)


def _validate_integer(value, *, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _validate_real(value, *, name, lower_bound, strict):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    valid = value > lower_bound if strict else value >= lower_bound
    if not valid:
        comparison = "greater than" if strict else "greater than or equal to"
        raise ValueError(f"{name} must be {comparison} {lower_bound}")
    return value


def _validate_observations(
    observations: torch.Tensor, *, minimum_samples: int
) -> torch.Tensor:
    if not torch.is_tensor(observations):
        raise TypeError("observations must be a PyTorch tensor")
    if observations.ndim != 2:
        raise ValueError("observations must have shape (n, d)")
    if observations.shape[0] < minimum_samples:
        minimum_label = {1: "one", 2: "two"}.get(
            minimum_samples, str(minimum_samples)
        )
        raise ValueError(
            f"observations must contain at least {minimum_label} sample"
            + ("s" if minimum_samples != 1 else "")
        )
    if observations.shape[1] not in (1, 2):
        raise ValueError(
            "posterior denoising currently supports one- or two-dimensional data"
        )
    if not observations.is_floating_point():
        raise TypeError("observations must use a floating-point dtype")
    if observations.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("observations must use torch.float32 or torch.float64")
    if not torch.isfinite(observations).all().item():
        raise ValueError("observations must contain only finite values")
    return observations.detach()


def _normalize_noise_std(
    noise_std: NoiseStandardDeviation, observations: torch.Tensor
) -> torch.Tensor:
    try:
        raw_noise_std = torch.as_tensor(noise_std)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("noise_std must be numeric") from error
    if raw_noise_std.dtype == torch.bool or raw_noise_std.is_complex():
        raise ValueError("noise_std must be numeric")

    try:
        normalized = raw_noise_std.to(
            device=observations.device,
            dtype=observations.dtype,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("noise_std could not be converted to the observation dtype") from error

    n_samples, dim = observations.shape
    if normalized.ndim == 0:
        normalized = normalized.expand(n_samples, 1)
    elif normalized.ndim == 1:
        if normalized.shape[0] != n_samples:
            raise ValueError(f"one-dimensional noise_std must have length {n_samples}")
        normalized = normalized[:, None]
    elif normalized.ndim == 2:
        if normalized.shape[0] != n_samples:
            raise ValueError(f"noise_std must have {n_samples} rows")
        if normalized.shape[1] not in (1, dim):
            raise ValueError(
                "noise_std second dimension must be 1 or match the observations"
            )
    else:
        raise ValueError(
            "noise_std must be scalar, one-dimensional, or two-dimensional"
        )

    if not torch.isfinite(normalized).all().item():
        raise ValueError("noise_std must contain only finite values")
    if (normalized < 0).any().item():
        raise ValueError("noise_std must be nonnegative")
    return normalized.detach()


def _model_device_and_dtype(model, observations):
    if not callable(getattr(model, "log_prob", None)):
        raise TypeError("model must provide a callable log_prob method")
    floating_tensors = [
        tensor
        for tensor in list(model.parameters()) + list(model.buffers())
        if tensor.is_floating_point()
    ]
    if floating_tensors:
        reference = floating_tensors[0]
        if reference.dtype not in _SUPPORTED_DTYPES:
            raise TypeError("model must use torch.float32 or torch.float64")
        if any(
            tensor.device != reference.device or tensor.dtype != reference.dtype
            for tensor in floating_tensors[1:]
        ):
            raise ValueError(
                "model floating parameters and buffers must share one device and dtype"
            )
        return reference.device, reference.dtype
    return observations.device, observations.dtype


def _importance_statistics(latent_points, log_weights):
    log_normalizer = torch.logsumexp(log_weights, dim=1)
    if not torch.isfinite(log_normalizer).all().item():
        raise RuntimeError(
            "posterior importance weights have a non-finite normalizer"
        )
    normalized_weights = torch.softmax(log_weights, dim=1)
    effective_samples = 1.0 / normalized_weights.square().sum(dim=1)
    posterior_mean = (
        normalized_weights[..., None] * latent_points
    ).sum(dim=1)
    centered = latent_points - posterior_mean[:, None, :]
    posterior_variance = (
        normalized_weights[..., None] * centered.square()
    ).sum(dim=1)
    if not torch.isfinite(posterior_mean).all().item() or not torch.isfinite(
        posterior_variance
    ).all().item():
        raise RuntimeError(
            "posterior importance sampling produced non-finite moments"
        )
    return posterior_mean, posterior_variance, effective_samples


def _diagonal_gaussian_log_prob(latent_points, observations, noise_std):
    standardized = (
        observations[:, None, :] - latent_points
    ) / noise_std[:, None, :]
    return -0.5 * (
        standardized.square()
        + 2.0 * torch.log(noise_std[:, None, :])
        + math.log(2.0 * math.pi)
    ).sum(dim=2)


def _validated_model_samples(
    sample_method,
    num_samples,
    dim,
    *,
    device,
    dtype,
):
    samples = sample_method(num_samples)
    if not torch.is_tensor(samples):
        raise TypeError("model.sample must return a PyTorch tensor")
    if samples.shape != (num_samples, dim):
        raise ValueError(
            f"model.sample must return shape ({num_samples}, {dim})"
        )
    samples = samples.to(device=device, dtype=dtype)
    if not torch.isfinite(samples).all().item():
        raise RuntimeError("model.sample returned non-finite values")
    return samples


def _evaluate_likelihood_proposal(
    model,
    observations,
    noise_std,
    standard_normal,
):
    n_batch, dim = observations.shape
    num_importance_samples = standard_normal.shape[0]
    latent_points = (
        observations[:, None, :]
        - noise_std[:, None, :] * standard_normal[None, :, :]
    )
    log_weights = model.log_prob(
        latent_points.reshape(n_batch * num_importance_samples, dim)
    ).reshape(n_batch, num_importance_samples)
    return latent_points, log_weights


def _target_to_mixture_log_weights(
    model,
    latent_points,
    observations,
    noise_std,
    *,
    adaptive_mean=None,
    adaptive_std=None,
    wide_adaptive_std=None,
    component_weights=(0.5, 0.5),
):
    n_batch, num_samples, dim = latent_points.shape
    log_prior = model.log_prob(
        latent_points.reshape(n_batch * num_samples, dim)
    ).reshape(n_batch, num_samples)
    log_likelihood = _diagonal_gaussian_log_prob(
        latent_points,
        observations,
        noise_std,
    )
    proposal_terms = (
        log_prior + math.log(component_weights[0]),
        log_likelihood + math.log(component_weights[1]),
    )
    if adaptive_mean is not None:
        log_adaptive = _diagonal_gaussian_log_prob(
            latent_points,
            adaptive_mean,
            adaptive_std,
        )
        proposal_terms += (
            log_adaptive + math.log(component_weights[2]),
        )
        if wide_adaptive_std is not None:
            log_wide_adaptive = _diagonal_gaussian_log_prob(
                latent_points,
                adaptive_mean,
                wide_adaptive_std,
            )
            proposal_terms += (
                log_wide_adaptive + math.log(component_weights[3]),
            )
    log_proposal = torch.logsumexp(torch.stack(proposal_terms), dim=0)
    return log_prior + log_likelihood - log_proposal


def _evaluate_adaptive_mixture(
    model,
    observations,
    noise_std,
    *,
    pilot_prior,
    pilot_standard_normal,
    final_prior,
    likelihood_standard_normal,
    fitted_adaptive_standard_normal,
    wide_adaptive_standard_normal,
    num_importance_samples,
):
    n_batch = observations.shape[0]
    pilot_samples = 2 * pilot_prior.shape[0]
    pilot_quarter = pilot_samples // 4
    pilot_likelihood = (
        observations[:, None, :]
        - noise_std[:, None, :] * pilot_standard_normal[None, :, :]
    )
    expanded_pilot_prior = pilot_prior[None, :, :].expand(n_batch, -1, -1)
    pilot_points = torch.cat(
        (
            expanded_pilot_prior[:, :pilot_quarter, :],
            pilot_likelihood[:, :pilot_quarter, :],
            expanded_pilot_prior[:, pilot_quarter:, :],
            pilot_likelihood[:, pilot_quarter:, :],
        ),
        dim=1,
    )
    pilot_log_weights = _target_to_mixture_log_weights(
        model,
        pilot_points,
        observations,
        noise_std,
    )
    adaptive_mean, adaptive_variance, _ = _importance_statistics(
        pilot_points,
        pilot_log_weights,
    )
    fitted_adaptive_std = torch.maximum(
        torch.sqrt(adaptive_variance),
        0.05 * noise_std,
    )
    wide_adaptive_std = 2.0 * fitted_adaptive_std

    half_samples = num_importance_samples // 2
    prior_per_half = final_prior.shape[0] // 2
    likelihood_per_half = likelihood_standard_normal.shape[0] // 2
    fitted_adaptive_per_half = fitted_adaptive_standard_normal.shape[0] // 2
    wide_adaptive_per_half = wide_adaptive_standard_normal.shape[0] // 2
    if (
        prior_per_half
        + likelihood_per_half
        + fitted_adaptive_per_half
        + wide_adaptive_per_half
        != half_samples
    ):
        raise RuntimeError(
            "internal importance-proposal sample counts are inconsistent"
        )

    likelihood_points = (
        observations[:, None, :]
        - noise_std[:, None, :]
        * likelihood_standard_normal[None, :, :]
    )
    fitted_adaptive_points = (
        adaptive_mean[:, None, :]
        + fitted_adaptive_std[:, None, :]
        * fitted_adaptive_standard_normal[None, :, :]
    )
    wide_adaptive_points = (
        adaptive_mean[:, None, :]
        + wide_adaptive_std[:, None, :]
        * wide_adaptive_standard_normal[None, :, :]
    )
    expanded_final_prior = final_prior[None, :, :].expand(n_batch, -1, -1)
    latent_points = torch.cat(
        (
            expanded_final_prior[:, :prior_per_half, :],
            likelihood_points[:, :likelihood_per_half, :],
            fitted_adaptive_points[:, :fitted_adaptive_per_half, :],
            wide_adaptive_points[:, :wide_adaptive_per_half, :],
            expanded_final_prior[:, prior_per_half:, :],
            likelihood_points[:, likelihood_per_half:, :],
            fitted_adaptive_points[:, fitted_adaptive_per_half:, :],
            wide_adaptive_points[:, wide_adaptive_per_half:, :],
        ),
        dim=1,
    )
    component_weights = (
        final_prior.shape[0] / num_importance_samples,
        likelihood_standard_normal.shape[0] / num_importance_samples,
        fitted_adaptive_standard_normal.shape[0] / num_importance_samples,
        wide_adaptive_standard_normal.shape[0] / num_importance_samples,
    )
    log_weights = _target_to_mixture_log_weights(
        model,
        latent_points,
        observations,
        noise_std,
        adaptive_mean=adaptive_mean,
        adaptive_std=fitted_adaptive_std,
        wide_adaptive_std=wide_adaptive_std,
        component_weights=component_weights,
    )
    return latent_points, log_weights


def _posterior_mean_importance_sampling(
    model,
    observations,
    noise_std,
    *,
    num_importance_samples,
    batch_size,
    check_convergence,
):
    dim = observations.shape[1]
    expanded_noise_std = noise_std.expand(-1, dim)
    requires_sampling = (expanded_noise_std > 0).any(dim=1)
    posterior_means = observations.clone()
    if not requires_sampling.any().item():
        return posterior_means

    fallback_standard_normal = torch.randn(
        num_importance_samples,
        dim,
        device=observations.device,
        dtype=observations.dtype,
    )
    sample_method = getattr(model, "sample", None)
    uses_defensive_mixture = requires_sampling & (expanded_noise_std > 0).all(dim=1)
    if not callable(sample_method):
        uses_defensive_mixture = torch.zeros_like(uses_defensive_mixture)
    adaptive_samples = None
    if uses_defensive_mixture.any().item():
        pilot_sample_count = min(2048, num_importance_samples)
        prior_per_half = num_importance_samples // 16
        likelihood_per_half = prior_per_half
        adaptive_per_half = (
            num_importance_samples // 2
            - prior_per_half
            - likelihood_per_half
        )
        fitted_adaptive_per_half = adaptive_per_half // 2
        wide_adaptive_per_half = adaptive_per_half - fitted_adaptive_per_half
        adaptive_samples = {
            "pilot_prior": _validated_model_samples(
                sample_method,
                pilot_sample_count // 2,
                dim,
                device=observations.device,
                dtype=observations.dtype,
            ),
            "pilot_standard_normal": torch.randn(
                pilot_sample_count // 2,
                dim,
                device=observations.device,
                dtype=observations.dtype,
            ),
            "final_prior": _validated_model_samples(
                sample_method,
                2 * prior_per_half,
                dim,
                device=observations.device,
                dtype=observations.dtype,
            ),
            "likelihood_standard_normal": torch.randn(
                2 * likelihood_per_half,
                dim,
                device=observations.device,
                dtype=observations.dtype,
            ),
            "fitted_adaptive_standard_normal": torch.randn(
                2 * fitted_adaptive_per_half,
                dim,
                device=observations.device,
                dtype=observations.dtype,
            ),
            "wide_adaptive_standard_normal": torch.randn(
                2 * wide_adaptive_per_half,
                dim,
                device=observations.device,
                dtype=observations.dtype,
            ),
        }
    effective_batch_size = min(
        batch_size,
        max(1, _MAX_IMPORTANCE_BATCH_POINTS // num_importance_samples),
    )
    half_samples = num_importance_samples // 2
    sampling_groups = (
        torch.nonzero(requires_sampling & ~uses_defensive_mixture).flatten(),
        torch.nonzero(uses_defensive_mixture).flatten(),
    )
    for group_indices in sampling_groups:
        if group_indices.numel() == 0:
            continue
        group_adaptive_samples = (
            adaptive_samples
            if uses_defensive_mixture[group_indices[0]].item()
            else None
        )
        for start in range(0, group_indices.numel(), effective_batch_size):
            batch_indices = group_indices[start : start + effective_batch_size]
            observation_batch = observations[batch_indices]
            sigma_batch = expanded_noise_std[batch_indices]
            if group_adaptive_samples is None:
                latent_points, log_weights = _evaluate_likelihood_proposal(
                    model,
                    observation_batch,
                    sigma_batch,
                    fallback_standard_normal,
                )
            else:
                latent_points, log_weights = _evaluate_adaptive_mixture(
                    model,
                    observation_batch,
                    sigma_batch,
                    num_importance_samples=num_importance_samples,
                    **group_adaptive_samples,
                )
            posterior_batch, _, _ = _importance_statistics(
                latent_points,
                log_weights,
            )

            if check_convergence:
                first_mean, first_variance, first_effective = (
                    _importance_statistics(
                        latent_points[:, :half_samples, :],
                        log_weights[:, :half_samples],
                    )
                )
                second_mean, second_variance, second_effective = (
                    _importance_statistics(
                        latent_points[:, half_samples:, :],
                        log_weights[:, half_samples:],
                    )
                )
                correction_scale = torch.maximum(
                    (first_mean - observation_batch).abs(),
                    (second_mean - observation_batch).abs(),
                )
                base_tolerance = (
                    _CONVERGENCE_ATOL + _CONVERGENCE_RTOL * correction_scale
                )
                difference_standard_error = torch.sqrt(
                    first_variance / first_effective[:, None]
                    + second_variance / second_effective[:, None]
                )
                tolerance = base_tolerance + (
                    _IMPORTANCE_CONSISTENCY_STANDARD_ERRORS
                    * difference_standard_error
                )
                if ((first_mean - second_mean).abs() > tolerance).any().item():
                    raise _ImportanceSamplingResolutionError(
                        "independent importance-sampling halves materially disagree; "
                        "increase num_importance_samples"
                    )

            posterior_batch = torch.where(
                sigma_batch == 0,
                observation_batch,
                posterior_batch,
            )
            if not torch.isfinite(posterior_batch).all().item():
                raise RuntimeError(
                    "posterior importance sampling produced non-finite means"
                )
            posterior_means[batch_indices] = posterior_batch

    return posterior_means


def posterior_mean_gaussian(
    model,
    observations: torch.Tensor,
    noise_std: NoiseStandardDeviation,
    *,
    num_importance_samples: int = 8192,
    batch_size: int = 64,
    seed: int = 0,
    check_convergence: bool = True,
) -> torch.Tensor:
    """Estimate ``E[Z | X=x]`` under a fitted prior and diagonal Gaussian noise.

    When the model provides ``sample``, an independent pilot from a 50:50
    prior-likelihood mixture fits a per-observation diagonal Gaussian bridge.
    The final defensive proposal combines prior and likelihood draws with
    bridge draws at the fitted and twice-fitted scales. Otherwise the
    likelihood alone is used. Samples receive self-normalized
    target-to-proposal weights. The same base draws are reused across
    observations. By default, two independent balanced halves must agree within
    a Monte Carlo error-aware tolerance. This first API is restricted to one-
    and two-dimensional observations.

    Args:
        model: Fitted density model with a ``log_prob`` method. A callable
            ``sample`` method enables the more robust pilot-adapted defensive
            proposal.
        observations: Floating tensor with shape ``(n, d)`` for ``d`` 1 or 2.
        noise_std: Known Gaussian standard deviations. Accepted forms are a
            scalar, ``(n,)``, ``(n, 1)``, or ``(n, d)``. The first three forms
            are isotropic within each observation.
        num_importance_samples: Importance-sample count from 128 through 32768,
            in multiples of four so each convergence half has the same proposal
            composition.
        batch_size: Number of observations evaluated at once.
        seed: Nonnegative same-device seed used only for posterior importance
            samples. Caller RNG state is restored before returning.
        check_convergence: Compare two independent sample halves and reject
            disagreement beyond their estimated Monte Carlo uncertainty.
            Disabling this removes a numerical safeguard and should be limited
            to diagnostic use.

    Returns:
        Posterior means with the model's floating dtype and device.
    """
    observations = _validate_observations(observations, minimum_samples=1)
    _validate_integer(
        num_importance_samples,
        name="num_importance_samples",
        minimum=_MIN_IMPORTANCE_SAMPLES,
    )
    if num_importance_samples > _MAX_IMPORTANCE_SAMPLES:
        raise ValueError(
            "num_importance_samples must be at most "
            f"{_MAX_IMPORTANCE_SAMPLES}"
        )
    if num_importance_samples % 4 != 0:
        raise ValueError("num_importance_samples must be a multiple of four")
    _validate_integer(batch_size, name="batch_size", minimum=1)
    seed = _validate_integer(seed, name="seed", minimum=0)
    if seed > 2**63 - 1:
        raise ValueError("seed must be at most 2**63 - 1")
    if not isinstance(check_convergence, bool):
        raise ValueError("check_convergence must be a boolean")

    device, dtype = _model_device_and_dtype(model, observations)
    observations = observations.to(device=device, dtype=dtype)
    noise_std = _normalize_noise_std(noise_std, observations)

    was_training = getattr(model, "training", None)
    if callable(getattr(model, "eval", None)):
        model.eval()

    try:
        with _fork_seed(seed, device), torch.no_grad():
            posterior_mean = _posterior_mean_importance_sampling(
                model,
                observations,
                noise_std,
                num_importance_samples=num_importance_samples,
                batch_size=batch_size,
                check_convergence=check_convergence,
            )
    finally:
        if was_training is not None and callable(getattr(model, "train", None)):
            model.train(was_training)

    return posterior_mean


def _posterior_mean_langevin(
    model,
    observations,
    noise_std,
    *,
    n_steps,
    step_size,
    n_chains,
    burn_in_fraction,
    thinning,
    batch_size,
):
    dim = observations.shape[1]
    expanded_noise_std = noise_std.expand(-1, dim)
    active_rows = (expanded_noise_std > 0).any(dim=1)
    posterior_means = observations.clone()
    if not active_rows.any().item():
        return posterior_means

    active_observations = observations[active_rows]
    active_noise_std = expanded_noise_std[active_rows]
    positive_noise = active_noise_std > 0
    positions = active_observations[:, None, :].expand(
        -1, n_chains, -1
    ).clone()
    positions = positions + (
        0.1
        * torch.randn_like(positions)
        * positive_noise[:, None, :]
    )
    collected_sum = torch.zeros_like(active_observations)
    collected_count = 0
    burn_in_step = int(burn_in_fraction * n_steps)

    for step in range(n_steps):
        temperature = max(1.0, 2.0 * (1.0 - step / n_steps))
        step_noise = torch.randn_like(positions)

        for start in range(0, positions.shape[0], batch_size):
            stop = min(start + batch_size, positions.shape[0])
            current = positions[start:stop].detach().requires_grad_(True)
            flat_current = current.reshape(-1, dim)
            with torch.enable_grad():
                log_prior = model.log_prob(flat_current)
                if not torch.isfinite(log_prior).all().item():
                    raise RuntimeError(
                        "Langevin sampling encountered non-finite prior values"
                    )
                prior_score = torch.autograd.grad(
                    log_prior.sum(),
                    current,
                    create_graph=False,
                    retain_graph=False,
                )[0]
            if not torch.isfinite(prior_score).all().item():
                raise RuntimeError(
                    "Langevin sampling encountered a non-finite prior score"
                )

            observation_batch = active_observations[start:stop, None, :]
            sigma_batch = active_noise_std[start:stop, None, :]
            positive_batch = positive_noise[start:stop, None, :]
            safe_sigma = torch.where(
                positive_batch,
                sigma_batch,
                torch.ones_like(sigma_batch),
            )
            likelihood_score = torch.where(
                positive_batch,
                (observation_batch - current) / safe_sigma.square(),
                torch.zeros_like(current),
            )
            posterior_score = (prior_score + likelihood_score) / temperature
            posterior_score = torch.where(
                positive_batch,
                posterior_score,
                torch.zeros_like(posterior_score),
            )
            score_norm = posterior_score.norm(
                dim=2,
                keepdim=True,
            ).clamp(min=1e-8)
            posterior_score = posterior_score * torch.minimum(
                50.0 / score_norm,
                torch.ones_like(score_norm),
            )

            with torch.no_grad():
                updated = (
                    current
                    + 0.5 * step_size * posterior_score
                    + math.sqrt(step_size) * step_noise[start:stop]
                )
                positions[start:stop] = torch.where(
                    positive_batch,
                    updated,
                    observation_batch,
                )

        if not torch.isfinite(positions).all().item():
            raise RuntimeError("Langevin sampling produced non-finite states")
        if (
            step >= burn_in_step
            and (step - burn_in_step) % thinning == 0
        ):
            collected_sum += positions.mean(dim=1)
            collected_count += 1

    if collected_count == 0:
        raise RuntimeError("Langevin sampling collected no posterior states")
    posterior_means[active_rows] = collected_sum / collected_count
    return posterior_means


def posterior_mean_langevin(
    model,
    observations: torch.Tensor,
    noise_std: NoiseStandardDeviation,
    *,
    n_steps: int = 1000,
    step_size: float = 1e-2,
    n_chains: int = 100,
    burn_in_fraction: float = 0.6,
    thinning: int = 2,
    batch_size: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Estimate Gaussian posterior means with unadjusted Langevin sampling.

    The sampler targets the fitted-prior posterior using the exact PyTorch
    gradient of ``model.log_prob`` plus the diagonal Gaussian likelihood score.
    It follows the notebook protocol: chains start near each observation, the
    target cools from temperature two to one, score norms are capped at 50, and
    post-burn-in states are averaged across chains and iterations. This is ULA,
    not Metropolis-adjusted Langevin sampling.

    Args:
        model: Fitted density model with a differentiable ``log_prob`` method.
        observations: Floating tensor with shape ``(n, d)`` for ``d`` 1 or 2.
        noise_std: Known diagonal Gaussian standard deviations in any shape
            accepted by :func:`posterior_mean_gaussian`.
        n_steps: Langevin updates per chain.
        step_size: Positive ULA integration step.
        n_chains: Parallel chains per observation.
        burn_in_fraction: Fraction of initial updates discarded, in ``[0, 1)``.
        thinning: Retain every ``thinning``-th state after burn-in.
        batch_size: Number of observations whose chains are scored together.
        seed: Nonnegative same-device seed. Caller RNG state is restored.

    Returns:
        Posterior means with the model's floating dtype and device.
    """
    observations = _validate_observations(observations, minimum_samples=1)
    n_steps = _validate_integer(n_steps, name="n_steps", minimum=1)
    step_size = _validate_real(
        step_size,
        name="step_size",
        lower_bound=0.0,
        strict=True,
    )
    n_chains = _validate_integer(n_chains, name="n_chains", minimum=1)
    burn_in_fraction = _validate_real(
        burn_in_fraction,
        name="burn_in_fraction",
        lower_bound=0.0,
        strict=False,
    )
    if burn_in_fraction >= 1.0:
        raise ValueError("burn_in_fraction must be less than 1")
    thinning = _validate_integer(thinning, name="thinning", minimum=1)
    batch_size = _validate_integer(batch_size, name="batch_size", minimum=1)
    seed = _validate_integer(seed, name="seed", minimum=0)
    if seed > 2**63 - 1:
        raise ValueError("seed must be at most 2**63 - 1")

    device, dtype = _model_device_and_dtype(model, observations)
    observations = observations.to(device=device, dtype=dtype)
    noise_std = _normalize_noise_std(noise_std, observations)

    was_training = getattr(model, "training", None)
    if callable(getattr(model, "eval", None)):
        model.eval()

    try:
        with _fork_seed(seed, device):
            posterior_mean = _posterior_mean_langevin(
                model,
                observations,
                noise_std,
                n_steps=n_steps,
                step_size=step_size,
                n_chains=n_chains,
                burn_in_fraction=burn_in_fraction,
                thinning=thinning,
                batch_size=batch_size,
            )
    finally:
        if was_training is not None and callable(getattr(model, "train", None)):
            model.train(was_training)

    return posterior_mean


def _resolve_device(device) -> torch.device:
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid device: {device!r}") from error
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("denoise currently supports CPU or CUDA devices")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {resolved.index}")
    return resolved


@contextmanager
def _fork_seed(seed, device):
    cuda_index = None
    if device.type == "cuda":
        cuda_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
    cuda_devices = [] if cuda_index is None else [cuda_index]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if cuda_index is not None:
            with torch.cuda.device(cuda_index):
                torch.cuda.manual_seed(seed)
        yield


def denoise(
    noisy_data: torch.Tensor,
    noise_std: NoiseStandardDeviation,
    *,
    noise_type: str = "gaussian",
    kernel_type: str = "laplace",
    epochs: int = 1000,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_epochs: Optional[int] = None,
    max_grad_norm: float = 5.0,
    bandwidths: Optional[BandwidthInput] = None,
    num_blocks: int = 4,
    num_bins: int = 16,
    hidden_features: int = 32,
    tail_bound: float = 30.0,
    max_derivative: float = MAX_DERIVATIVE,
    posterior_method: str = "importance",
    num_importance_samples: int = 8192,
    posterior_batch_size: int = 64,
    langevin_steps: int = 1000,
    langevin_step_size: float = 1e-2,
    langevin_chains: int = 100,
    langevin_burn_in_fraction: float = 0.6,
    langevin_thinning: int = 2,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> DenoisingResult:
    """Fit a convMMD flow prior and denoise observations by posterior means.

    This is a low-dimensional empirical-Bayes procedure for known zero-mean,
    independent Gaussian measurement errors. It is not a generic image
    denoiser, and its posterior means are point estimates rather than samples.

    Args:
        noisy_data: Finite ``torch.float32`` observations with shape ``(n, d)``
            for ``d`` 1 or 2 and at least two rows.
        noise_std: Known Gaussian standard deviations as a scalar, ``(n,)``,
            ``(n, 1)``, or ``(n, d)`` tensor-like value.
        noise_type: Must be ``"gaussian"`` in this first high-level API.
        kernel_type: Laplace or Gaussian MMD kernel.
        epochs: Number of convMMD optimization epochs.
        batch_size: Training batch size; values above ``n`` use all rows.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_epochs: Linear warmup length, not exceeding ``epochs``.
        max_grad_norm: Gradient-clipping norm.
        bandwidths: Optional MMD bandwidths; the training heuristic is used
            when omitted.
        num_blocks: Number of NSF blocks.
        num_bins: Number of spline bins.
        hidden_features: Hidden width of the NSF networks.
        tail_bound: Absolute spline bound. It must exceed every observation;
            in 2D it must also exceed the fitted diagonal base's mean plus or
            minus four standard deviations. The validation error reports the
            minimum required value for the supplied data.
        max_derivative: Upper bound for spline derivatives.
        posterior_method: ``"importance"`` (the default) or ``"langevin"``.
        num_importance_samples: Initial posterior importance-sample count from
            128 through 32768, in multiples of four. If it is demonstrably under-resolved,
            ``denoise`` retries once with 32768 samples and records that
            effective count in the returned configuration.
        posterior_batch_size: Observation batch size for posterior evaluation.
        langevin_steps: ULA updates per chain when ``posterior_method`` is
            ``"langevin"``.
        langevin_step_size: Positive ULA integration step.
        langevin_chains: Parallel Langevin chains per observation.
        langevin_burn_in_fraction: Fraction of initial Langevin updates
            discarded, in ``[0, 1)``.
        langevin_thinning: Retain every this many post-burn-in updates.
        seed: Nonnegative same-device repeatability seed used independently for
            fitting and posterior importance sampling.
        device: CPU or CUDA device used for fitting and posterior evaluation.
        verbose: Show the training progress display.

    Returns:
        A :class:`DenoisingResult` containing aligned posterior means, the
        fitted model, training history, effective bandwidths, and configuration.
    """
    if noise_type != "gaussian":
        raise ValueError("denoise currently supports only known Gaussian noise")
    if kernel_type not in ("laplace", "gaussian"):
        raise ValueError("kernel_type must be 'laplace' or 'gaussian'")
    if posterior_method not in ("importance", "langevin"):
        raise ValueError("posterior_method must be 'importance' or 'langevin'")

    observations = _validate_observations(noisy_data, minimum_samples=2)
    if observations.dtype != torch.float32:
        raise TypeError("denoise currently requires torch.float32 observations")
    resolved_device = _resolve_device(device)
    observations = observations.to(resolved_device)
    normalized_noise_std = _normalize_noise_std(noise_std, observations)

    epochs = _validate_integer(epochs, name="epochs", minimum=1)
    batch_size = _validate_integer(batch_size, name="batch_size", minimum=2)
    num_blocks = _validate_integer(num_blocks, name="num_blocks", minimum=1)
    num_bins = _validate_integer(num_bins, name="num_bins", minimum=2)
    hidden_features = _validate_integer(
        hidden_features, name="hidden_features", minimum=1
    )
    posterior_batch_size = _validate_integer(
        posterior_batch_size, name="posterior_batch_size", minimum=1
    )
    _validate_integer(
        num_importance_samples,
        name="num_importance_samples",
        minimum=_MIN_IMPORTANCE_SAMPLES,
    )
    if num_importance_samples > _MAX_IMPORTANCE_SAMPLES:
        raise ValueError(
            "num_importance_samples must be at most "
            f"{_MAX_IMPORTANCE_SAMPLES}"
        )
    if num_importance_samples % 4 != 0:
        raise ValueError("num_importance_samples must be a multiple of four")
    langevin_steps = _validate_integer(
        langevin_steps, name="langevin_steps", minimum=1
    )
    langevin_chains = _validate_integer(
        langevin_chains, name="langevin_chains", minimum=1
    )
    langevin_thinning = _validate_integer(
        langevin_thinning, name="langevin_thinning", minimum=1
    )
    seed = _validate_integer(seed, name="seed", minimum=0)
    if seed > 2**63 - 1:
        raise ValueError("seed must be at most 2**63 - 1")

    lr = _validate_real(lr, name="lr", lower_bound=0.0, strict=True)
    weight_decay = _validate_real(
        weight_decay,
        name="weight_decay",
        lower_bound=0.0,
        strict=False,
    )
    max_grad_norm = _validate_real(
        max_grad_norm,
        name="max_grad_norm",
        lower_bound=0.0,
        strict=True,
    )
    max_derivative = _validate_real(
        max_derivative,
        name="max_derivative",
        lower_bound=1.0,
        strict=True,
    )
    langevin_step_size = _validate_real(
        langevin_step_size,
        name="langevin_step_size",
        lower_bound=0.0,
        strict=True,
    )
    langevin_burn_in_fraction = _validate_real(
        langevin_burn_in_fraction,
        name="langevin_burn_in_fraction",
        lower_bound=0.0,
        strict=False,
    )
    if langevin_burn_in_fraction >= 1.0:
        raise ValueError("langevin_burn_in_fraction must be less than 1")

    if warmup_epochs is None:
        effective_warmup_epochs = min(200, epochs // 10)
    else:
        effective_warmup_epochs = _validate_integer(
            warmup_epochs, name="warmup_epochs", minimum=0
        )
        if effective_warmup_epochs > epochs:
            raise ValueError("warmup_epochs must not exceed epochs")

    data_mean = observations.mean(dim=0)
    data_std = observations.std(dim=0, unbiased=False)
    if not torch.isfinite(data_mean).all().item() or not torch.isfinite(
        data_std
    ).all().item():
        raise ValueError("observation statistics are non-finite; rescale the data")
    if not (data_std > 0).all().item():
        raise ValueError("each observation feature must have positive variation")

    tail_bound = _validate_real(
        tail_bound, name="tail_bound", lower_bound=0.0, strict=True
    )
    required_tail_bound = max(4.0, observations.abs().max().item())
    if observations.shape[1] == 2:
        required_tail_bound = max(
            required_tail_bound,
            (data_mean.abs() + 4.0 * data_std).max().item(),
        )
    if tail_bound <= required_tail_bound:
        raise ValueError(
            "tail_bound must exceed the observations and the fitted base's "
            f"four-standard-deviation range ({required_tail_bound:.6g})"
        )

    effective_batch_size = min(batch_size, observations.shape[0])
    with _fork_seed(seed, resolved_device):
        model = NormalizingFlowDensity(
            dim=observations.shape[1],
            flow_type="nsf",
            num_blocks=num_blocks,
            num_bins=num_bins,
            hidden_features=hidden_features,
            tail_bound=tail_bound,
            data_mean=data_mean.detach().cpu(),
            data_std=data_std.detach().cpu(),
            max_derivative=max_derivative,
        ).to(device=resolved_device, dtype=observations.dtype)

        training_result = train_convmmd(
            model=model,
            x_noisy=observations,
            noise_std=normalized_noise_std,
            noise_type="gaussian",
            kernel_type=kernel_type,
            epochs=epochs,
            batch_size=effective_batch_size,
            lr=lr,
            weight_decay=weight_decay,
            warmup_epochs=effective_warmup_epochs,
            max_grad_norm=max_grad_norm,
            bandwidths=bandwidths,
            eval_every=min(50, epochs),
            device=str(resolved_device),
            verbose=verbose,
        )
        effective_num_importance_samples = num_importance_samples
        effective_posterior_batch_size = posterior_batch_size
        if posterior_method == "langevin":
            denoised = posterior_mean_langevin(
                training_result["model"],
                observations,
                normalized_noise_std,
                n_steps=langevin_steps,
                step_size=langevin_step_size,
                n_chains=langevin_chains,
                burn_in_fraction=langevin_burn_in_fraction,
                thinning=langevin_thinning,
                batch_size=posterior_batch_size,
                seed=seed,
            )
        else:
            try:
                denoised = posterior_mean_gaussian(
                    training_result["model"],
                    observations,
                    normalized_noise_std,
                    num_importance_samples=num_importance_samples,
                    batch_size=posterior_batch_size,
                    seed=seed,
                )
            except _ImportanceSamplingResolutionError:
                if num_importance_samples >= _MAX_IMPORTANCE_SAMPLES:
                    raise
                effective_num_importance_samples = _MAX_IMPORTANCE_SAMPLES
                sample_ratio = num_importance_samples / _MAX_IMPORTANCE_SAMPLES
                effective_posterior_batch_size = max(
                    1,
                    math.floor(posterior_batch_size * sample_ratio),
                )
                try:
                    denoised = posterior_mean_gaussian(
                        training_result["model"],
                        observations,
                        normalized_noise_std,
                        num_importance_samples=effective_num_importance_samples,
                        batch_size=effective_posterior_batch_size,
                        seed=seed,
                    )
                except _ImportanceSamplingResolutionError as maximum_error:
                    raise _ImportanceSamplingResolutionError(
                        "posterior importance sampling remained unresolved at "
                        f"requested count {num_importance_samples} and maximum "
                        f"count {_MAX_IMPORTANCE_SAMPLES}: {maximum_error}"
                    ) from maximum_error

    config = DenoisingConfig(
        dim=observations.shape[1],
        noise_type=noise_type,
        kernel_type=kernel_type,
        epochs=epochs,
        batch_size=effective_batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=effective_warmup_epochs,
        max_grad_norm=max_grad_norm,
        num_blocks=num_blocks,
        num_bins=num_bins,
        hidden_features=hidden_features,
        tail_bound=tail_bound,
        max_derivative=max_derivative,
        posterior_method=posterior_method,
        num_importance_samples=effective_num_importance_samples,
        posterior_batch_size=effective_posterior_batch_size,
        langevin_steps=langevin_steps,
        langevin_step_size=langevin_step_size,
        langevin_chains=langevin_chains,
        langevin_burn_in_fraction=langevin_burn_in_fraction,
        langevin_thinning=langevin_thinning,
        seed=seed,
        device=str(resolved_device),
    )

    return DenoisingResult(
        denoised=denoised,
        model=training_result["model"],
        history=training_result["history"],
        bandwidths=training_result["bandwidths"],
        config=config,
    )


__all__ = [
    "DenoisingConfig",
    "DenoisingResult",
    "denoise",
    "posterior_mean_gaussian",
    "posterior_mean_langevin",
]
